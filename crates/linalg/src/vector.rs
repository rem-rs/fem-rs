use fem_core::Scalar;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ─── f64 SIMD-friendly inner loops ───────────────────────────────────────────

/// 8-unrolled dot product for f64 slices.
///
/// Using 8 independent accumulators breaks the loop-carried dependency chain,
/// allowing the compiler (with `-C target-feature=+avx2,+fma`) to emit 4×256-bit
/// FMA instructions per iteration — matching the theoretical throughput of
/// modern x86 and ARM cores.
///
/// This is the **serial** primitive: its association (8 parallel accumulators
/// combined at the end, in slice-index order) is a pure function of `len`.
/// [`dot_f64_parallel`] reuses it as the per-block kernel so that a parallel
/// call and a serial call agree on every block's partial sum.
#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut s4 = 0.0_f64;
    let mut s5 = 0.0_f64;
    let mut s6 = 0.0_f64;
    let mut s7 = 0.0_f64;

    let end8 = n / 8 * 8;
    let mut i = 0;
    while i < end8 {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
        i += 8;
    }
    let mut sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// 8-unrolled axpy for f64 slices: `y += α * x`.
#[inline]
fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let end8 = n / 8 * 8;
    let mut i = 0;
    while i < end8 {
        y[i]     += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        y[i + 4] += alpha * x[i + 4];
        y[i + 5] += alpha * x[i + 5];
        y[i + 6] += alpha * x[i + 6];
        y[i + 7] += alpha * x[i + 7];
        i += 8;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

/// Minimum vector length before Rayon parallelisation is worthwhile.
/// Shorter vectors have thread-spawn overhead that exceeds the compute savings.
#[cfg(feature = "parallel")]
const PAR_VEC_MIN: usize = 4_096;

/// Fixed block length of the parallel reduction in [`dot_f64_parallel`].
///
/// The block boundaries depend only on the vector length — never on the Rayon
/// thread count or on the work-stealing schedule.
#[cfg(feature = "parallel")]
const DOT_REDUCTION_BLOCK: usize = 4_096;

/// Parallel dot product with a **bitwise reproducible** result.
///
/// Rayon's `par_iter().sum()` splits the range according to the number of
/// threads and to whatever the work-stealing splitter decides at run time, and
/// combines the partial sums in that (thread-count- and schedule-dependent)
/// tree shape.  Floating-point addition is not associative, so the same input
/// produced different last bits from run to run — D754: `mfem_ex26_geom_mg`'s
/// `sol.gf` took 5 distinct sha256 values in 5 runs (6653 of 274627 entries
/// differing, max relative 2.1e-14), while `RAYON_NUM_THREADS=1` was stable.
///
/// This helper keeps the parallelism but makes the *association* a pure
/// function of `len`: the range is cut into fixed [`DOT_REDUCTION_BLOCK`]-sized
/// chunks whose boundaries depend only on `len`, each chunk is reduced by the
/// serial 8-unrolled [`dot_f64`] on whichever worker picks it up, and the chunk
/// partials are combined serially in chunk-index order.  The result is
/// therefore identical for any thread count, any schedule and any run.
///
/// The same pattern (and the same block length) is used by
/// `fem_parallel::ParVector`'s `deterministic_dot` (D746).
#[cfg(feature = "parallel")]
#[inline]
fn dot_f64_parallel(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let partials: Vec<f64> = a
        .par_chunks(DOT_REDUCTION_BLOCK)
        .zip(b.par_chunks(DOT_REDUCTION_BLOCK))
        .map(|(xa, xb)| dot_f64(xa, xb))
        .collect();
    partials.iter().sum()
}

/// A heap-allocated column vector with BLAS-like operations.
#[derive(Debug, Clone)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T: Scalar> Vector<T> {
    /// Create a vector of `n` zeros.
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![T::zero(); n] }
    }

    /// Create from an existing `Vec<T>`.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Length.
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    /// True if length is zero.
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Borrow as slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] { &self.data }

    /// Mutably borrow as slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] { &mut self.data }

    /// Consume into inner `Vec`.
    pub fn into_vec(self) -> Vec<T> { self.data }

    /// `y = α x + y`  (BLAS daxpy)
    ///
    /// Uses an 8-unrolled loop for `f64` to enable AVX2 auto-vectorisation.
    /// With the `parallel` feature and `n ≥ 4096`, Rayon parallelises the update.
    pub fn axpy(&mut self, alpha: T, x: &Self) {
        assert_eq!(self.len(), x.len(), "axpy: length mismatch");

        // Fast path: f64 with 8-unroll (+ optional Rayon).
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let y_f64 = unsafe {
                std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f64, self.data.len())
            };
            let x_f64 = unsafe {
                std::slice::from_raw_parts(x.data.as_ptr() as *const f64, x.data.len())
            };
            let alpha_f64 = unsafe { std::ptr::read(&alpha as *const T as *const f64) };

            #[cfg(feature = "parallel")]
            if self.data.len() >= PAR_VEC_MIN {
                y_f64.par_iter_mut()
                    .zip(x_f64.par_iter())
                    .for_each(|(yi, &xi)| *yi += alpha_f64 * xi);
                return;
            }

            axpy_f64(alpha_f64, x_f64, y_f64);
            return;
        }

        // Generic fallback.
        for (yi, &xi) in self.data.iter_mut().zip(x.data.iter()) {
            *yi += alpha * xi;
        }
    }

    /// `y = α x`  (in-place scale + assign from x)
    pub fn assign_scaled(&mut self, alpha: T, x: &Self) {
        assert_eq!(self.len(), x.len());
        for (yi, &xi) in self.data.iter_mut().zip(x.data.iter()) {
            *yi = alpha * xi;
        }
    }

    /// Scale in place: `x = α x`.
    ///
    /// With the `parallel` feature and `n ≥ 4096`, Rayon parallelises the scale.
    pub fn scale(&mut self, alpha: T) {
        #[cfg(feature = "parallel")]
        if self.data.len() >= PAR_VEC_MIN {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let d_f64 = unsafe {
                    std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f64, self.data.len())
                };
                let alpha_f64 = unsafe { std::ptr::read(&alpha as *const T as *const f64) };
                d_f64.par_iter_mut().for_each(|v| *v *= alpha_f64);
                return;
            }
            self.data.par_iter_mut().for_each(|v| *v *= alpha);
            return;
        }

        for v in self.data.iter_mut() { *v *= alpha; }
    }

    /// Euclidean dot product `x · y`.
    ///
    /// Uses an 8-unrolled loop for `f64` to enable AVX2 auto-vectorisation.
    /// With the `parallel` feature and `n ≥ 4096`, Rayon parallelises the
    /// reduction — through [`dot_f64_parallel`], whose association depends only
    /// on `n`, so the result is bitwise reproducible across thread counts,
    /// schedules and runs (D754/D766).
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(self.len(), other.len(), "dot: length mismatch");

        // Fast path: f64 with 8-unroll (+ optional Rayon).
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let a = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr() as *const f64, self.data.len())
            };
            let b = unsafe {
                std::slice::from_raw_parts(other.data.as_ptr() as *const f64, other.data.len())
            };

            #[cfg(feature = "parallel")]
            let result = if self.data.len() >= PAR_VEC_MIN {
                dot_f64_parallel(a, b)
            } else {
                dot_f64(a, b)
            };

            #[cfg(not(feature = "parallel"))]
            let result = dot_f64(a, b);

            // SAFETY: T is f64 (same layout).
            return unsafe { std::ptr::read(&result as *const f64 as *const T) };
        }

        // Generic fallback.
        self.data.iter().zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    /// Euclidean norm `‖x‖₂`.
    ///
    /// Bitwise reproducible under the same conditions as [`Vector::dot`]
    /// (it *is* `sqrt(dot(x, x))`).
    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }

    /// MFEM `Vector::Norml2()` (upstream `linalg/vector.cpp:968`): the
    /// LAPACK-dnrm2-style scaled 2-norm, sequential over the entries — the
    /// exact arithmetic MFEM's CPU reduce path performs.
    ///
    /// Unlike [`Vector::norm`] (the plain `sqrt(Σxᵢ²)`), the rescaling keeps
    /// every squared argument ≤ 1 so overflowing dynamic ranges stay finite,
    /// and the different summation shape is bit-exact with upstream: the two
    /// algorithms differ by 1-2 ulp on ~40% of generic inputs, which ODE
    /// solvers amplify into visible last-digit flips.  Always prefer this
    /// over [`Vector::norm`] where 1:1 parity with MFEM matters (D906
    /// follow-up, round-130; pins in this file).
    pub fn norml2(&self) -> T {
        norml2(&self.data)
    }

    /// Fill with constant value.
    ///
    /// With the `parallel` feature and `n ≥ 4096`, Rayon parallelises the fill.
    pub fn fill(&mut self, v: T) {
        #[cfg(feature = "parallel")]
        if self.data.len() >= PAR_VEC_MIN {
            self.data.par_iter_mut().for_each(|x| *x = v);
            return;
        }

        for x in self.data.iter_mut() { *x = v; }
    }

    /// Copy `src` into `self[offset .. offset + src.len()]`.
    ///
    /// # Panics
    /// Panics if `offset + src.len() > self.len()`.
    pub fn set_sub_vector(&mut self, offset: usize, src: &[T]) {
        self.data[offset..offset + src.len()].copy_from_slice(src);
    }

    /// Return a slice `self[offset .. offset + len]`.
    ///
    /// # Panics
    /// Panics if `offset + len > self.len()`.
    pub fn get_sub_vector(&self, offset: usize, len: usize) -> &[T] {
        &self.data[offset..offset + len]
    }
}

impl<T: Scalar> std::ops::Index<usize> for Vector<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.data[i] }
}

impl<T: Scalar> std::ops::IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.data[i] }
}

/// MFEM `Vector::Norml2()` (upstream `linalg/vector.cpp:968`): the scaled
/// 2-norm in the style of LAPACK's `dnrm2` / `std::hypot()`, applied
/// sequentially over `x` — the exact arithmetic MFEM's CPU reduce path
/// performs.
///
/// A running sum `first` of squared entries rescaled around the running
/// maximum `second` keeps every squared argument ≤ 1, so the result never
/// overflows to +inf the way plain `sqrt(Σxᵢ²)` does on wide dynamic
/// ranges, and the summation shape is **bit-exact with MFEM**: the two
/// algorithms disagree by 1-2 ulp on ~40% of generic inputs (probe:
/// 10/24 two-entry and 2/6 hundred-entry random vectors), which ODE
/// solvers amplify into visible last-digit flips (ex23, round-129).
pub fn norml2<T: Scalar>(x: &[T]) -> T {
    if x.is_empty() {
        return T::zero();
    }
    let mut first = T::zero();
    let mut second = T::zero();
    for &xi in x {
        let n = xi.abs();
        if n > T::zero() {
            if second <= n {
                let arg = second / n;
                first = first * (arg * arg) + T::one();
                second = n;
            } else {
                let arg = n / second;
                first += arg * arg;
            }
        }
    }
    second * first.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy() {
        let mut y = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let x = Vector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        y.axpy(2.0, &x);
        assert_eq!(y.as_slice(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn dot_norm() {
        let v = Vector::<f64>::from_vec(vec![3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-14);
    }

    #[test]
    fn sub_vector_roundtrip() {
        let mut v = Vector::<f64>::zeros(10);
        v.set_sub_vector(3, &[1.0, 2.0, 3.0]);
        assert_eq!(v.get_sub_vector(3, 3), &[1.0, 2.0, 3.0]);
        assert_eq!(v[2], 0.0);
        assert_eq!(v[6], 0.0);
    }

    // ── MFEM `Vector::Norml2` bit-parity pins (round-130, D906 follow-up) ──
    //
    // Reference data: `tmp/rr130mfem/ref/norml2_probe.{cpp,out}` — MFEM
    // 4.10 `Vector::Norml2` (mfem410_ser, g++ -O2, serial CPU reduce) on a
    // xorshift64* sample, emitted as raw f64 bit patterns.  The plain
    // `sqrt(Σxᵢ²)` column documents exactly where the algorithms diverge.

    /// `Vector::Norml2` vs naive `sqrt(Σxᵢ²)` (bits) for 24 two-entry
    /// random vectors: the scaled algorithm wins 10 bit flips.
    const MFEM_NORML2_V2: [([u64; 2], u64, u64); 24] = [
        ([0x3fc0975fbde15b00, 0xbfe232a1474ce994], 0x3fe2aa1c6c9c60ea, 0x3fe2aa1c6c9c60e9),
        ([0x3fec41e89d4a6b40, 0x3fc3df8432a8be50], 0x3fecb0dea42c8fc9, 0x3fecb0dea42c8fc9),
        ([0x3fe4244c9c5a1ec8, 0x3fd5d9a0c3a831e8], 0x3fe6e9f806b495ef, 0x3fe6e9f806b495ef),
        ([0x3fc7bcd4b21c3710, 0xbfd77d0939c41a48], 0x3fda511aa17a7301, 0x3fda511aa17a7300),
        ([0x3fd905762119fc70, 0x3fe723c109dd69c4], 0x3fea4e1e39ebd04c, 0x3fea4e1e39ebd04d),
        ([0x3fe7bfa715842214, 0xbfead9981fbad0b0], 0x3ff1ec423f0348e8, 0x3ff1ec423f0348e8),
        ([0xbfef006c86c38ef4, 0x3feb48720b2bcaec], 0x3ff4a60cd4b8ffa6, 0x3ff4a60cd4b8ffa6),
        ([0xbfd94bbfdf3ab310, 0xbfb58950cd981660], 0x3fd9dccf4dde303c, 0x3fd9dccf4dde303d),
        ([0xbfe18034ed7e5634, 0x3fad04ea2cf599c0], 0x3fe19833a7c32122, 0x3fe19833a7c32122),
        ([0x3fed9ac12737202c, 0xbfe893cdf9933a94], 0x3ff33d0bdb52a7c0, 0x3ff33d0bdb52a7c0),
        ([0xbfc64a80472ce070, 0x3feb7bf3eaf5d768], 0x3fec0b20fe811be4, 0x3fec0b20fe811be4),
        ([0x3fd633a38e61cd88, 0x3fcbaf2b613ed820], 0x3fda29d0fbd4789d, 0x3fda29d0fbd4789d),
        ([0x3feade05fe1bab64, 0xbfb331cb9ff1d680], 0x3feaf96513174d35, 0x3feaf96513174d35),
        ([0xbfd91d69a682ea90, 0xbfc65cb822d69950], 0x3fdb7dc98db61a07, 0x3fdb7dc98db61a07),
        ([0x3feccc9a813f261c, 0xbfd0042abcd14160], 0x3fede453fd60eec4, 0x3fede453fd60eec5),
        ([0x3fe48958b131a5ec, 0xbfe42eb1fc30c25c], 0x3feccb307579d47b, 0x3feccb307579d47a),
        ([0x3fc109cc782d2b70, 0x3fad28dc52e6b480], 0x3fc28843bf9e0ba5, 0x3fc28843bf9e0ba4),
        ([0xbfe545ac6c901a64, 0xbfbc82b2b2a11560], 0x3fe591918dd5cc12, 0x3fe591918dd5cc13),
        ([0xbfea209cab82a014, 0xbfeed587652021b0], 0x3ff43522b8249f1f, 0x3ff43522b8249f1e),
        ([0xbfd32a0267e01428, 0xbfe7c6481120eefc], 0x3fe9a2041c16aba0, 0x3fe9a2041c16ab9f),
        ([0xbfe96fd7f75191dc, 0xbfe5942036efc408], 0x3ff0ada99dab2c46, 0x3ff0ada99dab2c46),
        ([0x3feb9a3ffeebe58c, 0x3fe69394e5d6d144], 0x3ff1d469a5c9c85d, 0x3ff1d469a5c9c85d),
        ([0x3fda806670935bc8, 0xbfe37d5609693844], 0x3fe7914847fdfb18, 0x3fe7914847fdfb18),
        ([0x3fe4bf78d25d8530, 0xbfd4cbf43f631b90], 0x3fe735332c391c1c, 0x3fe735332c391c1c),
    ];

    /// `Vector::Norml2` bits for the six 100-entry sample vectors (inputs
    /// regenerated by [`Xorshift64Star`]; anchor-checked against
    /// [`MFEM_NORML2_V2`] inputs).
    const MFEM_NORML2_N100: [u64; 6] = [
        0x4017adfc8a64513d,
        0x4018ecaf5653db9a,
        0x40176cb4eb8ae52d,
        0x4015350d9cf68a0b,
        0x4015f01a899078f4,
        0x4016218d20b78736,
    ];

    /// The probe's xorshift64* generator (uniforms exact in f64, so the
    /// Rust regeneration reproduces the C++ probe inputs bit-for-bit).
    struct Xorshift64Star(u64);

    impl Xorshift64Star {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform in [-1, 1); every intermediate is exact in f64.
        fn uniform(&mut self) -> f64 {
            let m = (self.next() & ((1u64 << 52) - 1)) as f64;
            (m / 4503599627370496.0) * 2.0 - 1.0
        }
    }

    #[test]
    fn norml2_matches_mfem_two_entry_samples() {
        for (input, want, plain) in &MFEM_NORML2_V2 {
            let v = Vector::from_vec(vec![
                f64::from_bits(input[0]),
                f64::from_bits(input[1]),
            ]);
            assert_eq!(
                v.norml2().to_bits(),
                *want,
                "input bits {:#018x} {:#018x}",
                input[0],
                input[1]
            );
            // the pins really exercise the difference: the plain formula
            // differs from MFEM on 10 of these 24 inputs (probe STATS2).
            let naive = {
                let x = v.as_slice();
                (x[0] * x[0] + x[1] * x[1]).sqrt()
            };
            assert_eq!(naive.to_bits(), *plain);
        }
    }

    #[test]
    fn norml2_matches_mfem_hundred_entry_samples() {
        // same seed as the C++ probe; draw the 24 two-entry vectors first
        // and anchor-check them, then the six 100-entry ones.
        let mut rng = Xorshift64Star(88172645463325252);
        for (input, _, _) in &MFEM_NORML2_V2 {
            assert_eq!(rng.uniform().to_bits(), input[0]);
            assert_eq!(rng.uniform().to_bits(), input[1]);
        }
        for &want in &MFEM_NORML2_N100 {
            let v: Vector<f64> = Vector::from_vec((0..100).map(|_| rng.uniform()).collect());
            assert_eq!(v.norml2().to_bits(), want);
        }
    }

    #[test]
    fn norml2_extreme_dynamic_range_stays_finite() {
        // naive sqrt(Σxᵢ²) overflows to +inf (probe EXT lines); the scaled
        // algorithm returns the exact finite values MFEM produces.
        let v = Vector::from_vec(vec![1e300_f64, 1e-300]);
        assert_eq!(v.norml2().to_bits(), 0x7e37e43c8800759c); // = 1e300
        assert!((1e300_f64 * 1e300_f64 + 1e-300_f64 * 1e-300_f64)
            .sqrt()
            .is_infinite());

        let w = Vector::from_vec(vec![1e200_f64, 1e200]);
        assert_eq!(w.norml2().to_bits(), 0x697d8f9811335b57); // = √2·1e200
        assert!((1e200_f64 * 1e200_f64 + 1e200_f64 * 1e200_f64)
            .sqrt()
            .is_infinite());

        // 3-4-5 sanity, exact on both sides (probe EXT3).
        let z = Vector::from_vec(vec![3.0_f64, 4.0]);
        assert_eq!(z.norml2().to_bits(), 0x4014000000000000); // = 5
    }

    #[test]
    fn norml2_empty_is_zero() {
        assert_eq!(Vector::<f64>::zeros(0).norml2().to_bits(), 0);
    }

    #[test]
    fn norml2_generic_f32_smoke() {
        let v = Vector::<f32>::from_vec(vec![3.0, 4.0]);
        assert!((v.norml2() - 5.0).abs() < 1e-6);
    }
}
