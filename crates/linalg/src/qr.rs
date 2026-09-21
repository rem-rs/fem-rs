//! Dense Householder QR — the minimal LAPACK face of MFEM's `NNLSSolver`
//! (D533(b)).
//!
//! MFEM's `NNLSSolver` (`linalg/solvers.cpp`, `#ifdef MFEM_USE_LAPACK`) calls
//! exactly four LAPACK/BLAS kernels:
//!
//! * `dgeqrf` — QR factorization.  The NNLS matrices are small
//!   (`min(m,n) ≤ ILAENV block size 32`), so LAPACK dispatches to the
//!   **unblocked** `dgeqr2`; that is what is ported here as
//!   [`qr_factor`].
//! * `dormqr` — apply the leading `k` Householder reflectors
//!   (`Qᵀ·C` / `Q·C`).  Again small sizes run the unblocked `dorm2r`;
//!   ported as [`apply_q_transpose`] / [`apply_q`].
//! * `dtrsm`  — upper-triangular solve `R·x = b` for a single right-hand
//!   side ([`solve_upper_triangular`], netlib `dtrsm` operation order).
//! * `dgemv`  — plain matrix-vector products, kept inline in the NNLS
//!   module (`nnls.rs`).
//!
//! Storage is **column-major** throughout (LAPACK convention, MFEM
//! `DenseMatrix::Data` layout, same choice as [`crate::dense::CholeskyFactors`]).
//! The operations replicate the netlib reference algorithms operation by
//! operation (scaled 2-norm, `dlapy2`-style hypotenuse, reflector
//! generation and application), so results agree with a netlib-LAPACK +
//! reference-BLAS build to the last ulp for well-scaled inputs.  Against
//! optimized BLAS (e.g. OpenBLAS, which the MFEM LAPACK reference build
//! links) agreement is at the rounding-noise level rather than bit-exact;
//! the NNLS discrete path is insensitive to that in practice.
//!
//! Out of scope (documented gap, D561): the *blocked* `dgeqrf`/`dormqr`
//! panel updates — they only run for `min(m,n) > 32`, a size the NURBS
//! reduced-integration rules never reach.

use fem_core::{FemError, FemResult};

// -------------------------------------------------------------------------
// BLAS/LAPACK auxiliary kernels (netlib reference semantics)
// -------------------------------------------------------------------------

/// `dnrm2` (reference BLAS, LAPACK 3.12 `dnrm2.f90`): the 3-accumulator
/// (abig/asml/amed) sum of squares.  For mid-range inputs this is the plain
/// ascending `sqrt(Σ xᵢ²)`.
fn dnrm2(x: &[f64]) -> f64 {
    // Blue's scaling constants (f64): tbig = 2^486, tsml = 2^-511,
    // ssml = 2^537, sbig = 2^-538 (exact powers of two).
    let (tbig, tsml, ssml, sbig) =
        (2f64.powi(486), 2f64.powi(-511), 2f64.powi(537), 2f64.powi(-538));
    const MAXN: f64 = f64::MAX;

    if x.is_empty() {
        return 0.0;
    }
    let mut abig = 0.0_f64;
    let mut asml = 0.0_f64;
    let mut amed = 0.0_f64;
    let mut notbig = true;
    for &xi in x {
        let ax = xi.abs();
        if ax > tbig {
            abig += (ax * sbig) * (ax * sbig);
            notbig = false;
        } else if ax < tsml {
            if notbig {
                asml += (ax * ssml) * (ax * ssml);
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > MAXN || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > MAXN || amed.is_nan() {
            amed = amed.sqrt();
            asml = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml > amed { (amed, asml) } else { (asml, amed) };
            scl = 1.0;
            sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        // All values mid-range.
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// `dlapy2`: `sqrt(x² + y²)` in the overflow-safe LAPACK form.
fn dlapy2(x: f64, y: f64) -> f64 {
    let xabs = x.abs();
    let yabs = y.abs();
    let w = xabs.max(yabs);
    let z = xabs.min(yabs);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// Fortran `SIGN(z, a)`: `|z|` with the sign of `a` (`a >= 0` → `+|z|`).
fn fortran_sign(z: f64, a: f64) -> f64 {
    if a < 0.0 {
        -z.abs()
    } else {
        z.abs()
    }
}

/// `dlarfg` (LAPACK 3.12): generate an elementary reflector
/// `H = I - tau·v·vᵀ` such that `H·(alpha, x) = (beta, 0, …, 0)`.
///
/// On entry `x` holds the `n-1` subdiagonal entries (`n = x.len() + 1`); on
/// exit `x` holds the reflector vector `v[1..n]` scaled by
/// `1/(alpha - beta)` (the implicit `v[0] == 1` is not stored) and `alpha`
/// is overwritten with `beta`.  `tau` receives the scalar factor.  An empty
/// `x` yields `H = I` (`tau = 0`, everything untouched), as does `n <= 1`
/// in LAPACK.
fn larfg(alpha: &mut f64, x: &mut [f64], tau: &mut f64) {
    let safmin = f64::MIN_POSITIVE / f64::EPSILON; // DLAMCH('S')/DLAMCH('E')
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        // H = I.
        *tau = 0.0;
        return;
    }
    let mut alpha_s = *alpha;
    let mut beta = -fortran_sign(dlapy2(alpha_s, xnorm), alpha_s);
    let mut knt = 0_u32;
    if beta.abs() < safmin {
        // XNORM, BETA may be inaccurate; scale X and recompute them.
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            for xi in x.iter_mut() {
                *xi *= rsafmn;
            }
            beta *= rsafmn;
            alpha_s *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        let xnorm = dnrm2(x);
        beta = -fortran_sign(dlapy2(alpha_s, xnorm), alpha_s);
    }
    *tau = (beta - alpha_s) / beta;
    let scal = 1.0 / (alpha_s - beta);
    for xi in x.iter_mut() {
        *xi *= scal;
    }
    // If ALPHA was subnormal it may have lost relative accuracy.
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
}

/// `dlarf` with `side = 'L'` (LAPACK 3.12): apply `H = I - tau·v·vᵀ` from
/// the left to the `m × nc` column-major matrix `c` (leading dimension
/// `ldc`), `v` the reflector with implicit leading 1.  Mirrors the
/// `dgemv('T') + dger` pair exactly, including the `dger` product grouping
/// `A(i,j) += V(i)·((-tau)·WORK(j))` and its zero-column skip.
fn larf_left(m: usize, nc: usize, v: &[f64], tau: f64, c: &mut [f64], ldc: usize) {
    if tau == 0.0 || nc == 0 {
        return;
    }
    // WORK = Cᵀ·V (dgemv 'T' with alpha = 1, beta = 0: plain ascending dots).
    let mut work = vec![0.0_f64; nc];
    for j in 0..nc {
        let mut t = 0.0_f64;
        for i in 0..m {
            t += c[i + j * ldc] * v[i];
        }
        work[j] = t;
    }
    // C := C - tau·V·WORKᵀ (dger with alpha = -tau).
    for j in 0..nc {
        if work[j] != 0.0 {
            let temp = -tau * work[j];
            for i in 0..m {
                c[i + j * ldc] += v[i] * temp;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

/// Unblocked Householder QR of a column-major `m × n` matrix, in place
/// (LAPACK `dgeqr2`, i.e. what `dgeqrf` runs for `min(m,n)` up to its
/// block-size threshold).
///
/// On exit the strict upper triangle holds `R` (including the diagonal),
/// the strict lower triangle holds the reflector vectors `v[1..]`, and
/// `tau[k]` (length ≥ `min(m,n)`) the reflector scalars.  `Q` is the
/// product of the `min(m,n)` elementary reflectors
/// `Hₖ = I - tau[k]·vₖ·vₖᵀ` with `vₖ` stored in column `k`.
pub fn qr_factor(a: &mut [f64], m: usize, n: usize, tau: &mut [f64]) -> FemResult<()> {
    if tau.len() < m.min(n) {
        return Err(FemError::DimMismatch { expected: m.min(n), actual: tau.len() });
    }
    let kmax = m.min(n);
    for k in 0..kmax {
        // dgeqr2: dlarfg over rows k..m of column k, then apply Hₖ to the
        // remaining columns k+1..n.  AII is read AFTER dlarfg (it holds
        // beta = R[k][k]) and restored after the application.
        {
            let (head, tail) = a[k + k * m..].split_at_mut(1);
            larfg(&mut head[0], &mut tail[..m - k - 1], &mut tau[k]);
        }
        if k + 1 < n {
            let aii = a[k + k * m];
            a[k + k * m] = 1.0;
            // Hₖ · C with C = rows k..m of columns k+1..n: the reflector
            // column is rows k..m of column k; the sub-matrix starts at
            // linear index k + (k+1)·m and keeps leading dim m.
            let v: Vec<f64> = a[k + k * m..k * m + m].to_vec();
            larf_left(m - k, n - k - 1, &v, tau[k], &mut a[k + (k + 1) * m..], m);
            a[k + k * m] = aii;
        }
    }
    Ok(())
}

/// Apply the leading `k` Householder reflectors produced by [`qr_factor`]
/// from the left to the `m × nc` column-major matrix `c`
/// (`dorm2r` with `side = 'L', trans = 'T'`): `C ← Qₖᵀ·C`.  Since
/// `Q = H₀·H₁···Hₖ₋₁`, the transpose product applies the reflectors in
/// generation order (ascending `i`), each on rows `i..m`.
#[allow(clippy::too_many_arguments)] // argument list mirrors LAPACK dorm2r
pub fn apply_q_transpose(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
) {
    for i in 0..k {
        // v = [1, a(i+1..m, i)] with the implicit leading one.
        let mut v = vec![0.0_f64; m - i];
        v[0] = 1.0;
        v[1..].copy_from_slice(&a[i + 1 + i * lda..i * lda + m]);
        larf_left(m - i, nc, &v, tau[i], &mut c[i..], ldc);
    }
}

/// Apply the leading `k` Householder reflectors in reverse generation order
/// (`dorm2r` with `side = 'L', trans = 'N'`): `C ← Qₖ·C`, i.e. the
/// reflectors run from the last one down to the first.
// The argument list mirrors LAPACK's dorm2r parameter order (k, A, lda,
// tau, C, m, nc, ldc).
#[allow(clippy::too_many_arguments)]
pub fn apply_q(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
) {
    for i in (0..k).rev() {
        let mut v = vec![0.0_f64; m - i];
        v[0] = 1.0;
        v[1..].copy_from_slice(&a[i + 1 + i * lda..i * lda + m]);
        larf_left(m - i, nc, &v, tau[i], &mut c[i..], ldc);
    }
}

/// Solve `R·x = b` in place for the `n × n` leading upper triangle of the
/// column-major matrix `r` (leading dimension `ldr`), single right-hand
/// side.  Netlib `dtrsm` operation order for `side = 'L', uplo = 'U',
/// transa = 'N', diag = 'N'`: sweep `k` bottom-up, divide by the diagonal,
/// then rank-1-update all preceding entries.
pub fn solve_upper_triangular(r: &[f64], ldr: usize, b: &mut [f64], n: usize) {
    for k in (0..n).rev() {
        b[k] /= r[k + k * ldr];
        let bk = b[k];
        if bk != 0.0 {
            for i in 0..k {
                b[i] -= bk * r[i + k * ldr];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG (same as the C++ LAPACK probe in tmp/d533) so the
    /// test matrices match the probe-generated ones bit for bit.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    fn random_matrix(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut g = Lcg(seed);
        (0..m * n).map(|_| g.next_f64()).collect()
    }

    #[test]
    fn qr_rectangular_reconstructs_a() {
        // A = Q·R must reproduce A; Q from applying the reflectors to the
        // identity columns.
        for (m, n) in [(6usize, 4usize), (5, 5), (4, 6), (1, 3), (7, 1)] {
            let a = random_matrix(m, n, 1000 + (m * 31 + n) as u64);
            let mut fac = a.clone();
            let mut tau = vec![0.0_f64; m.min(n)];
            qr_factor(&mut fac, m, n, &mut tau).expect("qr_factor");

            // Build Q explicitly: apply the reflectors (transposed order for
            // Q itself) to each identity column.
            let mut q = vec![0.0_f64; m * m];
            for j in 0..m {
                q[j + j * m] = 1.0;
            }
            for j in 0..m {
                let mut col = vec![0.0_f64; m];
                col.copy_from_slice(&q[j * m..j * m + m]);
                apply_q(m.min(n), &fac, m, &tau, &mut col, m, 1, m);
                q[j * m..j * m + m].copy_from_slice(&col);
            }
            // A - Q·R
            let mut err = 0.0_f64;
            let mut nrm = 0.0_f64;
            for j in 0..n {
                for i in 0..m {
                    let mut qr_ij = 0.0_f64;
                    for k in 0..m.min(n) {
                        qr_ij += q[i + k * m]
                            * if k <= j { fac[k + j * m] } else { 0.0 };
                    }
                    err = err.max((qr_ij - a[i + j * m]).abs());
                    nrm = nrm.max(a[i + j * m].abs());
                }
            }
            assert!(
                err <= 1e-13 * nrm.max(1.0),
                "({m},{n}) reconstruction error {err:.3e}"
            );

            // Qᵀ·Q = I
            let mut orth = 0.0_f64;
            for j in 0..m {
                for i in 0..m {
                    let mut d = 0.0_f64;
                    for k in 0..m {
                        d += q[k + i * m] * q[k + j * m];
                    }
                    orth = orth.max((d - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            assert!(orth <= 1e-13, "({m},{n}) orthogonality error {orth:.3e}");
        }
    }

    #[test]
    fn qr_apply_q_transpose_is_inverse_of_apply_q() {
        let m = 6;
        let n = 4;
        let a = random_matrix(m, n, 42);
        let mut fac = a.clone();
        let mut tau = vec![0.0_f64; m.min(n)];
        qr_factor(&mut fac, m, n, &mut tau).expect("qr_factor");

        // C ← Qᵀ C then C ← Q C must return the original C.
        let mut c = random_matrix(m, 3, 7);
        let c0 = c.clone();
        apply_q_transpose(m.min(n), &fac, m, &tau, &mut c, m, 3, m);
        apply_q(m.min(n), &fac, m, &tau, &mut c, m, 3, m);
        for (x, y) in c.iter().zip(&c0) {
            assert!((x - y).abs() <= 1e-14, "Q Qᵀ round-trip {x} vs {y}");
        }
    }

    #[test]
    fn solve_upper_triangular_matches_back_substitution() {
        // R upper triangular (column-major), known solution x.
        #[rustfmt::skip]
        let r = vec![
            2.0, 0.5, 0.25, // col 0
            0.0, -1.5, 0.5, // col 1
            0.0, 0.0,  3.0, // col 2
        ];
        let x = [1.0, -2.0, 0.5];
        let mut b = [0.0_f64; 3];
        for j in 0..3 {
            for i in 0..=j {
                b[i] += r[i + j * 3] * x[j];
            }
        }
        solve_upper_triangular(&r, 3, &mut b, 3);
        for (got, want) in b.iter().zip(&x) {
            assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
        }
    }

    /// Golden values dumped by the LAPACK probe `tmp/d533/d533_qr_probe.cpp`
    /// (MFEM 4.10 LAPACK build: netlib dgeqrf + OpenBLAS kernels) for the
    /// probe's seeded matrices: R diagonal and tau vectors must agree to
    /// rounding noise.
    #[test]
    fn qr_matches_lapack_probe_golden() {
        // Probe case (m=6, n=4, seed 12345): tau values and R diagonal from
        // `dgeqrf` (probe output, LAPACK 3.12.0 + OpenBLAS, 2026-09-21).
        let tau_golden: [f64; 4] = [
            1.5505122211865232,
            1.2448066214047795,
            1.1873890551513562,
            1.971849343662508,
        ];
        let rdiag_golden: [f64; 4] = [
            1.4183931945162893,
            1.7149207833232352,
            -1.053766558962163,
            1.2216953772482007,
        ];
        let a = random_matrix(6, 4, 12345);
        let mut fac = a.clone();
        let mut tau = vec![0.0_f64; 4];
        qr_factor(&mut fac, 6, 4, &mut tau).expect("qr_factor");
        for k in 0..4 {
            assert!(
                (tau[k] - tau_golden[k]).abs() <= 1e-12 * tau_golden[k].abs().max(1.0),
                "tau[{k}] {} vs golden {}",
                tau[k],
                tau_golden[k]
            );
            let rd = fac[k + k * 6];
            assert!(
                (rd - rdiag_golden[k]).abs() <= 1e-12 * rdiag_golden[k].abs(),
                "R[{k},{k}] {rd} vs golden {}",
                rdiag_golden[k]
            );
        }

        // dormqr('L','T') / dormqr('L','N') of a seeded 6x2 matrix C
        // (probe output, LAPACK 3.12.0 + reference netlib BLAS 3.12.0 via
        // `LD_LIBRARY_PATH`, 2026-09-21).  The port is compared **exactly**:
        // with the reference BLAS the whole pipeline is deterministic, and
        // the port replicates its operation order bit for bit.
        let qt_golden: [f64; 12] = [
            -0.60857618693025428,
            -0.50363620712832113,
            -0.88223798848879487,
            0.86963877978091797,
            -0.047430846374018162,
            0.69981940104482954,
            1.2584458193488615,
            -0.06495726628684928,
            -0.26961992229261111,
            0.49446031325655948,
            -0.6091217302640336,
            0.85386074106241772,
        ];
        let q_golden: [f64; 12] = [
            0.28309652046929079,
            1.1433012274646133,
            0.3942152696725994,
            -0.39609998117693285,
            0.93366845832804302,
            0.28161633243703432,
            1.0864166206454875,
            0.55565175470787331,
            -0.27478846272045959,
            -0.5920523285427759,
            -0.75990314149086291,
            0.71600369302180833,
        ];
        let mut g = Lcg(777);
        let mut c = vec![0.0_f64; 12];
        for v in c.iter_mut() {
            *v = g.next_f64();
        }
        let mut ct = c.clone();
        apply_q_transpose(4, &fac, 6, &tau, &mut ct, 6, 2, 6);
        let mut cn = c;
        apply_q(4, &fac, 6, &tau, &mut cn, 6, 2, 6);
        for i in 0..12 {
            assert_eq!(ct[i], qt_golden[i], "QᵀC[{i}] exact");
            assert_eq!(cn[i], q_golden[i], "QC[{i}] exact");
        }
    }

    #[test]
    fn larfg_handles_trivial_columns() {
        // n <= 1 and zero subdiagonal -> H = I (tau = 0, alpha untouched).
        let mut a = 0.75_f64;
        let mut x = [0.0_f64; 3];
        let mut tau = 9.0_f64;
        larfg(&mut a, &mut x, &mut tau);
        assert_eq!(tau, 0.0);
        assert_eq!(a, 0.75);
        let mut tau = 9.0_f64;
        larfg(&mut a, &mut [], &mut tau);
        assert_eq!(tau, 0.0);
        assert_eq!(a, 0.75);
    }
}
