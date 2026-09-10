//! Complex dense factorizations (D15 fix).
//!
//! 1:1 port of the no-LAPACK path of MFEM's `ComplexCholeskyFactors`
//! (`linalg/complex_densemat.cpp`): Hermitian (LLᴴ) Cholesky–Crout
//! factorization with a **real positive diagonal**, plus the triangular
//! solves `LSolve` (`X ← L⁻¹X`) and `USolve` (`X ← L⁻ᴴX`).
//!
//! Storage convention: the matrix is `n × n`, split into separate real /
//! imaginary row-major arrays (`ar[i*n + j]`, `ai[i*n + j]`).  MFEM stores
//! column-major `std::complex` data; the recurrence is identical with the
//! index roles transposed.  As in MFEM, only the **lower triangle** of the
//! input is consumed and overwritten by `L` (the strict upper triangle keeps
//! the original entries and is ignored by the solves).
//!
//! The DPG test Gram matrices (`miniapps/dpg/util/complexweakform.cpp`) are
//! Hermitian positive definite by construction (adjoint graph norm), and MFEM
//! factors them with exactly this factorization (`ComplexCholeskyFactors::
//! Factor`, VERIFY `Re(pivot) > 0`).  The pivot is forced real: the imaginary
//! part of a Hermitian diagonal must vanish, and `Factor` reads only its real
//! part (MFEM semantics, not checked).

use fem_core::{FemError, FemResult};

/// Hermitian Cholesky–Crout factorization `A = L·Lᴴ` in place (MFEM
/// `ComplexCholeskyFactors::Factor`, TOL = 0).
///
/// Returns `true` on success.  Returns `false` when a pivot is not strictly
/// positive — i.e. the matrix is not numerically Hermitian positive definite
/// (MFEM `MFEM_VERIFY` + `return false` path).  On failure the data arrays
/// are left partially factored (MFEM behaves the same, operating in place).
pub fn complex_cholesky_her_factor(ar: &mut [f64], ai: &mut [f64], n: usize) -> FemResult<bool> {
    if ar.len() != n * n || ai.len() != n * n {
        return Err(FemError::DimMismatch { expected: n * n, actual: ar.len() });
    }
    for j in 0..n {
        // d = A[j][j] − Σ_{k<j} L[j][k]·conj(L[j][k])  (real)
        let mut d = ar[j * n + j];
        for k in 0..j {
            let (lr, li) = (ar[j * n + k], ai[j * n + k]);
            d -= lr * lr + li * li;
        }
        if !(d > 0.0) {
            return Ok(false);
        }
        let ljj = d.sqrt();
        ar[j * n + j] = ljj;
        ai[j * n + j] = 0.0;
        for i in (j + 1)..n {
            // L[i][j] = (A[i][j] − Σ_{k<j} L[i][k]·conj(L[j][k])) / L[j][j]
            let mut sr = ar[i * n + j];
            let mut si = ai[i * n + j];
            for k in 0..j {
                let (lir, lil) = (ar[i * n + k], ai[i * n + k]);
                let (ljr, lji) = (ar[j * n + k], ai[j * n + k]);
                // L[i][k]·conj(L[j][k]) = (lir + i·lil)(ljr − i·lji)
                sr -= lir * ljr + lil * lji;
                si -= ljr * lil - lir * lji;
            }
            ar[i * n + j] = sr / ljj;
            ai[i * n + j] = si / ljj;
        }
    }
    Ok(true)
}

/// Forward substitution `X ← L⁻¹X` for `k` right-hand-side columns stored
/// row-major `n × k` (entry `(i, c)` at `x[i*k + c]`; the triangular
/// recurrence matches MFEM `ComplexCholeskyFactors::LSolve`).
pub fn complex_cholesky_her_lsolve(
    lr: &[f64],
    li: &[f64],
    n: usize,
    xr: &mut [f64],
    xi: &mut [f64],
    k: usize,
) {
    for i in 0..n {
        for c in 0..k {
            let mut sr = xr[i * k + c];
            let mut si = xi[i * k + c];
            for j in 0..i {
                sr -= lr[i * n + j] * xr[j * k + c] - li[i * n + j] * xi[j * k + c];
                si -= lr[i * n + j] * xi[j * k + c] + li[i * n + j] * xr[j * k + c];
            }
            let inv = 1.0 / lr[i * n + i];
            xr[i * k + c] = sr * inv;
            xi[i * k + c] = si * inv;
        }
    }
}

/// Back substitution `X ← L⁻ᴴX` for `k` right-hand-side columns stored
/// row-major `n × k` (MFEM `ComplexCholeskyFactors::USolve`).
pub fn complex_cholesky_her_usolve(
    lr: &[f64],
    li: &[f64],
    n: usize,
    xr: &mut [f64],
    xi: &mut [f64],
    k: usize,
) {
    for i in (0..n).rev() {
        for c in 0..k {
            let mut sr = xr[i * k + c];
            let mut si = xi[i * k + c];
            for j in (i + 1)..n {
                // conj(L[j][i]) = (lr − i·li)
                sr -= lr[j * n + i] * xr[j * k + c] + li[j * n + i] * xi[j * k + c];
                si -= lr[j * n + i] * xi[j * k + c] - li[j * n + i] * xr[j * k + c];
            }
            let inv = 1.0 / lr[i * n + i];
            xr[i * k + c] = sr * inv;
            xi[i * k + c] = si * inv;
        }
    }
}

/// Solve `A·X = B` given a [`complex_cholesky_her_factor`] output:
/// `X ← L⁻ᴴ(L⁻¹B)` (MFEM `ComplexCholeskyFactors::Solve`).
#[allow(dead_code)] // symmetric pair of lsolve/usolve; kept for A x = b use sites
pub fn complex_cholesky_her_solve(
    lr: &[f64],
    li: &[f64],
    n: usize,
    xr: &mut [f64],
    xi: &mut [f64],
    k: usize,
) {
    complex_cholesky_her_lsolve(lr, li, n, xr, xi, k);
    complex_cholesky_her_usolve(lr, li, n, xr, xi, k);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG in [0, 1) (reproducible across platforms).
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn next_pair(&mut self) -> (f64, f64) {
            (self.next_f64() - 0.5, self.next_f64() - 0.5)
        }
    }

    /// Random Hermitian positive definite: A = Bᴴ·B + n·I for a fixed random
    /// complex B (drawn once, so `A` is reproducible from the seed).
    fn random_hpd(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Lcg(seed);
        let mut br = vec![0.0; n * n];
        let mut bi = vec![0.0; n * n];
        for v in br.iter_mut() {
            *v = rng.next_pair().0;
        }
        for v in bi.iter_mut() {
            *v = rng.next_pair().1;
        }
        let mut ar = vec![0.0; n * n];
        let mut ai = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sr = 0.0;
                let mut si = 0.0;
                for k in 0..n {
                    // conj(B[k][i])·B[k][j]
                    sr += br[k * n + i] * br[k * n + j] + bi[k * n + i] * bi[k * n + j];
                    si += br[k * n + i] * bi[k * n + j] - bi[k * n + i] * br[k * n + j];
                }
                ar[i * n + j] = sr;
                ai[i * n + j] = si;
            }
            ar[i * n + i] += n as f64;
        }
        (ar, ai)
    }

    #[test]
    fn factor_and_solve_random_hpd() {
        let n = 9;
        let (mut ar, mut ai) = random_hpd(n, 42);
        assert!(complex_cholesky_her_factor(&mut ar, &mut ai, n).unwrap());
        // Random RHS with k = 3 columns.
        let mut rng = Lcg(7);
        let k = 3;
        let mut br = vec![0.0; n * k];
        let mut bi = vec![0.0; n * k];
        for v in br.iter_mut() {
            *v = rng.next_pair().0 * 4.0;
        }
        for v in bi.iter_mut() {
            *v = rng.next_pair().1 * 4.0;
        }
        let b0r = br.clone();
        let b0i = bi.clone();
        complex_cholesky_her_solve(&ar, &ai, n, &mut br, &mut bi, k);
        // Residual ‖A x − b‖ / ‖b‖.
        let (mut rr, mut ri) = (vec![0.0; n * k], vec![0.0; n * k]);
        // `ar`/`ai` hold L after factorization; rebuild A from the generator
        // for the residual check.
        let (a2r, a2i) = random_hpd(n, 42);
        for i in 0..n {
            for j in 0..n {
                for c in 0..k {
                    rr[c + i * k] +=
                        a2r[i * n + j] * br[c + j * k] - a2i[i * n + j] * bi[c + j * k];
                    ri[c + i * k] +=
                        a2r[i * n + j] * bi[c + j * k] + a2i[i * n + j] * br[c + j * k];
                }
            }
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for c in 0..n * k {
            num += (rr[c] - b0r[c]).powi(2) + (ri[c] - b0i[c]).powi(2);
            den += b0r[c].powi(2) + b0i[c].powi(2);
        }
        assert!((num / den).sqrt() < 1e-12, "solve residual {}", (num / den).sqrt());
    }

    #[test]
    fn factor_llh_reconstructs_a() {
        let n = 6;
        let (a0r, a0i) = random_hpd(n, 1234);
        let (mut lr, mut li) = (a0r.clone(), a0i.clone());
        assert!(complex_cholesky_her_factor(&mut lr, &mut li, n).unwrap());
        // L·Lᴴ (row-major lower-triangle L).
        for i in 0..n {
            for j in 0..n {
                let mut sr = 0.0;
                let mut si = 0.0;
                for k in 0..n {
                    let lik = if k <= i { lr[i * n + k] } else { 0.0 };
                    let liki = if k <= i { li[i * n + k] } else { 0.0 };
                    let ljk = if k <= j { lr[j * n + k] } else { 0.0 };
                    let ljki = if k <= j { li[j * n + k] } else { 0.0 };
                    // L[i][k]·conj(L[j][k])
                    sr += lik * ljk + liki * ljki;
                    si += liki * ljk - lik * ljki;
                }
                assert!((sr - a0r[i * n + j]).abs() < 1e-11, "re ({i},{j}) {sr}");
                assert!((si - a0i[i * n + j]).abs() < 1e-11, "ie ({i},{j}) {si}");
            }
        }
        // Diagonal of L is real positive.
        for i in 0..n {
            assert_eq!(li[i * n + i], 0.0);
            assert!(lr[i * n + i] > 0.0);
        }
    }

    #[test]
    fn factor_rejects_indefinite_hermitian() {
        // Real symmetric indefinite: eigenvalues of [[2,3],[3,1]] are
        // (3 ± √13)/2 — one negative.
        let mut ar = vec![2.0, 3.0, 3.0, 1.0];
        let mut ai = vec![0.0; 4];
        assert!(!complex_cholesky_her_factor(&mut ar, &mut ai, 2).unwrap());
        // Genuinely complex Hermitian indefinite: [[2, 1+2i],[1−2i, 1]],
        // det = 2·1 − (1+4) = −3 < 0.
        let mut ar2 = vec![2.0, 1.0, 1.0, 1.0];
        let mut ai2 = vec![0.0, 2.0, -2.0, 0.0];
        assert!(!complex_cholesky_her_factor(&mut ar2, &mut ai2, 2).unwrap());
    }

    #[test]
    fn factor_rejects_singular() {
        // Rank-1 Hermitian: A = v·vᴴ with v = (1, i): [[1, −i],[i, 1]].
        let mut ar = vec![1.0, 1.0, 1.0, 1.0];
        let mut ai = vec![0.0, -1.0, 1.0, 0.0];
        assert!(!complex_cholesky_her_factor(&mut ar, &mut ai, 2).unwrap());
    }

    #[test]
    fn lsolve_only_matches_normal_equations_convention() {
        // The DPG weak form uses LSolve only: Y = L⁻¹B, then A_local = YᴴY.
        // This must equal BᴴA⁻¹B for A = L Lᴴ (MFEM `MultAtB` semantics).
        // Reference: Z = A⁻¹B via the full solve, contract BᴴZ.
        let n = 5;
        let (a0r, a0i) = random_hpd(n, 99);
        let (mut lr, mut li) = (a0r.clone(), a0i.clone());
        assert!(complex_cholesky_her_factor(&mut lr, &mut li, n).unwrap());
        let mut rng = Lcg(5);
        let mut br = vec![0.0; n * n];
        let mut bi = vec![0.0; n * n];
        for v in br.iter_mut() {
            *v = rng.next_pair().0 * 2.0;
        }
        for v in bi.iter_mut() {
            *v = rng.next_pair().1 * 2.0;
        }
        let mut yr = br.clone();
        let mut yi = bi.clone();
        complex_cholesky_her_lsolve(&lr, &li, n, &mut yr, &mut yi, n);
        let (mut zr, mut zi) = (br.clone(), bi.clone());
        complex_cholesky_her_solve(&lr, &li, n, &mut zr, &mut zi, n);
        // Compare YᴴY with BᴴZ entrywise.
        for i in 0..n {
            for j in 0..n {
                let mut s1r = 0.0;
                let mut s1i = 0.0;
                let mut s2r = 0.0;
                let mut s2i = 0.0;
                for k in 0..n {
                    // conj(Y[k][i])·Y[k][j]
                    s1r += yr[k * n + i] * yr[k * n + j] + yi[k * n + i] * yi[k * n + j];
                    s1i += yr[k * n + i] * yi[k * n + j] - yi[k * n + i] * yr[k * n + j];
                    // conj(B[k][i])·Z[k][j]
                    s2r += br[k * n + i] * zr[k * n + j] + bi[k * n + i] * zi[k * n + j];
                    s2i += br[k * n + i] * zi[k * n + j] - bi[k * n + i] * zr[k * n + j];
                }
                assert!((s1r - s2r).abs() < 1e-10);
                assert!((s1i - s2i).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn dim_mismatch_errors() {
        let mut ar = vec![1.0];
        let mut ai = vec![0.0];
        assert!(complex_cholesky_her_factor(&mut ar, &mut ai, 2).is_err());
    }

    #[test]
    fn solve_rectangular_rhs_layout() {
        // Non-square RHS (n=4, k=2): catches row/column-major mix-ups in the
        // `n × k` block layout (entry (i,c) at x[i*k + c]).
        let n = 4;
        let k = 2;
        let (mut ar, mut ai) = random_hpd(n, 2024);
        assert!(complex_cholesky_her_factor(&mut ar, &mut ai, n).unwrap());
        let (a2r, a2i) = random_hpd(n, 2024);
        let mut rng = Lcg(11);
        let mut br = vec![0.0; n * k];
        let mut bi = vec![0.0; n * k];
        for v in br.iter_mut() {
            *v = rng.next_pair().0 * 3.0;
        }
        for v in bi.iter_mut() {
            *v = rng.next_pair().1 * 3.0;
        }
        let b0r = br.clone();
        let b0i = bi.clone();
        complex_cholesky_her_solve(&ar, &ai, n, &mut br, &mut bi, k);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..n {
            for c in 0..k {
                let mut sr = 0.0;
                let mut si = 0.0;
                for j in 0..n {
                    sr += a2r[i * n + j] * br[j * k + c] - a2i[i * n + j] * bi[j * k + c];
                    si += a2r[i * n + j] * bi[j * k + c] + a2i[i * n + j] * br[j * k + c];
                }
                num += (sr - b0r[i * k + c]).powi(2) + (si - b0i[i * k + c]).powi(2);
                den += b0r[i * k + c].powi(2) + b0i[i * k + c].powi(2);
            }
        }
        assert!((num / den).sqrt() < 1e-12, "rect solve residual {}", (num / den).sqrt());
    }
}
