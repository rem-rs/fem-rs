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
/// For the `parallel` feature and `n >= 4096`, the first half of the slice is
/// handled by Rayon and the second half serially, then the two partial sums are
/// added.  (This is simpler and avoids per-chunk task overhead for short vectors.)
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
    /// With the `parallel` feature and `n ≥ 4096`, Rayon parallelises the reduction.
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
                a.par_iter().zip(b.par_iter()).map(|(&ai, &bi)| ai * bi).sum::<f64>()
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
    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
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
}
