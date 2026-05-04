//! Block-GMRES — simultaneous GMRES for multiple right-hand sides.
//!
//! # Overview
//!
//! When solving `A xᵢ = bᵢ` for several right-hand sides that share the same
//! coefficient matrix `A`, this module solves all of them through a single
//! function call.  The implementation uses standard restarted GMRES (GMRES(m))
//! per column, while sharing the matrix object, tolerance parameters, and
//! restart configuration.
//!
//! The data layout is **column-major**: `b[j * n + i]` is row `i` of RHS column
//! `j`.  This matches LAPACK/BLAS conventions and allows each column to be
//! passed as a plain `&[f64]` slice.
//!
//! # Algorithm — restarted GMRES (Saad & Schultz 1986)
//!
//! For each RHS column `j`:
//! 1. Compute residual `r = b_j − A x_j`; set `β = ‖r‖`, `v₁ = r / β`.
//! 2. Run the Arnoldi process for up to `restart` steps:
//!    - `w = A vᵢ`
//!    - Orthogonalise `w` against `v₁,…,vᵢ` (modified Gram-Schmidt)
//!    - Apply accumulated Givens rotations to maintain upper-triangular form
//! 3. Solve the small upper-triangular LS system for `y` (back-substitution).
//! 4. Update `x_j += Vₘ y`.
//! 5. Restart until converged or `max_iter` is reached.
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::block_gmres::{solve_block_gmres, BlockGmresConfig};
//! use fem_solver::SolverConfig;
//!
//! let cfg = BlockGmresConfig { restart: 30, base: SolverConfig { rtol: 1e-8, ..Default::default() } };
//! let res = solve_block_gmres(&a, &b, &mut x, &cfg)?;
//! println!("converged in {} total iterations", res.iterations);
//! ```

use fem_linalg::CsrMatrix;
use crate::{SolverConfig, SolveResult, SolverError};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`solve_block_gmres`].
#[derive(Debug, Clone)]
pub struct BlockGmresConfig {
    /// Convergence / stopping parameters (tolerance, max iterations, verbosity).
    pub base: SolverConfig,
    /// Inner Krylov dimension before restart.  Larger values need more memory
    /// but may require fewer restarts.  Typical range: 20–100.
    pub restart: usize,
}

impl Default for BlockGmresConfig {
    fn default() -> Self {
        BlockGmresConfig {
            base: SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 500, ..Default::default() },
            restart: 30,
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Solve `A X = B` for `s` right-hand sides simultaneously using restarted GMRES.
///
/// # Data layout
/// Both `b` and `x` are **column-major** with `s` columns:
/// `b[j * n + i]` = row `i`, column `j`.
///
/// # Arguments
/// * `a`   — square coefficient matrix (n × n)
/// * `b`   — right-hand sides; length `n × n_rhs`
/// * `x`   — initial guess (same layout as `b`); overwritten with solution
/// * `cfg` — solver parameters
///
/// # Returns
/// [`SolveResult`] with `iterations` = sum of per-column GMRES iterations and
/// `final_residual` = maximum relative residual across all columns.
pub fn solve_block_gmres(
    a:   &CsrMatrix<f64>,
    b:   &[f64],
    x:   &mut [f64],
    cfg: &BlockGmresConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows;
    let n_rhs = if n == 0 { 0 } else { b.len() / n };
    assert_eq!(b.len(), n * n_rhs, "b length must equal n_rows × n_rhs");
    assert_eq!(x.len(), n * n_rhs, "x length must equal n_rows × n_rhs");

    let mut total_iters = 0usize;
    let mut max_rel_res = 0.0f64;
    let mut any_failed  = false;

    for j in 0..n_rhs {
        let bj = &b[j * n .. (j + 1) * n];
        let xj = &mut x[j * n .. (j + 1) * n];
        match gmres_single(a, bj, xj, &cfg.base, cfg.restart) {
            Ok(res) => {
                total_iters += res.iterations;
                if res.final_residual > max_rel_res { max_rel_res = res.final_residual; }
            }
            Err(SolverError::ConvergenceFailed { residual, .. }) => {
                if residual > max_rel_res { max_rel_res = residual; }
                any_failed = true;
            }
            Err(e) => return Err(e),
        }
    }

    if any_failed {
        Err(SolverError::ConvergenceFailed { max_iter: cfg.base.max_iter, residual: max_rel_res })
    } else {
        Ok(SolveResult { converged: true, iterations: total_iters, final_residual: max_rel_res })
    }
}

// ─── Internal: restarted GMRES for one column ─────────────────────────────────

fn gmres_single(
    a:       &CsrMatrix<f64>,
    b:       &[f64],
    x:       &mut [f64],
    cfg:     &SolverConfig,
    restart: usize,
) -> Result<SolveResult, SolverError> {
    let n      = a.nrows;
    let rtol   = cfg.rtol;
    let atol   = cfg.atol;
    let max_it = cfg.max_iter;
    let m      = restart.max(1);

    let b_norm = vec_norm(b);
    let tol    = rtol * b_norm + atol;

    let mut total = 0usize;

    loop {
        // ── Residual ──────────────────────────────────────────────────────
        let mut r = vec![0.0f64; n];
        spmv_neg(a, x, b, &mut r);
        let beta = vec_norm(&r);

        if beta <= tol {
            return Ok(SolveResult {
                converged:      true,
                iterations:     total,
                final_residual: beta / b_norm.max(1.0),
            });
        }
        if total >= max_it { break; }

        // ── Arnoldi ───────────────────────────────────────────────────────
        // V: up to (m+1) vectors of length n
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(r.iter().map(|x| x / beta).collect());

        // H: row-major (m+1) × m upper Hessenberg
        let mut h  = vec![0.0f64; (m + 1) * m];
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut g  = vec![0.0f64; m + 1];
        g[0] = beta;

        let mut inner            = 0usize;
        let mut inner_converged  = false;

        for j in 0..m {
            if total + inner >= max_it { break; }

            // w = A v[j]
            let mut w = vec![0.0f64; n];
            spmv(a, &v[j], &mut w);

            // Modified Gram-Schmidt
            for k in 0..=j {
                let dot = vec_dot(&v[k], &w);
                h[k * m + j] = dot;
                let vk = v[k].clone();
                for i in 0..n { w[i] -= dot * vk[i]; }
            }
            let nw = vec_norm(&w);
            h[(j + 1) * m + j] = nw;

            // Push normalised v[j+1]
            if nw > 1e-14 {
                let inv = 1.0 / nw;
                v.push(w.iter().map(|xi| xi * inv).collect());
            } else {
                v.push(vec![0.0f64; n]);
            }

            // Apply previous Givens rotations to column j of H
            for k in 0..j {
                let tmp        =  cs[k] * h[k * m + j] + sn[k] * h[(k + 1) * m + j];
                h[(k+1)*m + j] = -sn[k] * h[k * m + j] + cs[k] * h[(k+1)*m + j];
                h[k * m + j]   = tmp;
            }

            // Compute new Givens rotation
            let (c, s, r_val) = givens(h[j * m + j], h[(j + 1) * m + j]);
            cs[j] = c; sn[j] = s;
            h[j * m + j]       = r_val;
            h[(j + 1) * m + j] = 0.0;

            // Update LS RHS
            let tmp  =  c * g[j] + s * g[j + 1];
            g[j + 1] = -s * g[j] + c * g[j + 1];
            g[j]     = tmp;

            inner += 1;

            if g[j + 1].abs() <= tol {
                inner_converged = true;
                break;
            }
        }

        // ── Solve Ry = g via back-substitution ────────────────────────────
        let y = back_solve_upper(&h, &g, m, inner);

        // ── Update x += V y ───────────────────────────────────────────────
        for k in 0..inner {
            let vk = &v[k];
            let yk = y[k];
            for i in 0..n { x[i] += yk * vk[i]; }
        }

        total += inner;

        if inner_converged {
            // True residual check
            let mut r2 = vec![0.0f64; n];
            spmv_neg(a, x, b, &mut r2);
            let res = vec_norm(&r2);
            return Ok(SolveResult {
                converged:      true,
                iterations:     total,
                final_residual: res / b_norm.max(1.0),
            });
        }

        if total >= max_it { break; }
    }

    // Final check after exhausting budget
    let mut r_f = vec![0.0f64; n];
    spmv_neg(a, x, b, &mut r_f);
    let res = vec_norm(&r_f);
    if res <= tol {
        Ok(SolveResult { converged: true, iterations: total, final_residual: res / b_norm.max(1.0) })
    } else {
        Err(SolverError::ConvergenceFailed { max_iter: max_it, residual: res / b_norm.max(1.0) })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Solve upper-triangular `R y = g[0..m]`.
/// `h` is flat row-major `(m+1) × m_full`; `m` = number of Arnoldi steps taken.
fn back_solve_upper(h: &[f64], g: &[f64], m_full: usize, m: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    for i in (0..m).rev() {
        let mut acc = g[i];
        for j in (i + 1)..m { acc -= h[i * m_full + j] * y[j]; }
        let diag = h[i * m_full + i];
        y[i] = if diag.abs() > 1e-15 { acc / diag } else { 0.0 };
    }
    y
}

fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-15 {
        (1.0, 0.0, a)
    } else if a.abs() < 1e-15 {
        (0.0, b.signum(), b.abs())
    } else {
        let r = a.hypot(b);
        (a / r, b / r, r)
    }
}

fn spmv(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    y.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..a.nrows {
        let mut s = 0.0;
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            s += a.values[idx] * x[a.col_idx[idx] as usize];
        }
        y[i] = s;
    }
}

fn spmv_neg(a: &CsrMatrix<f64>, x: &[f64], b: &[f64], r: &mut [f64]) {
    for i in 0..a.nrows {
        let mut s = 0.0;
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            s += a.values[idx] * x[a.col_idx[idx] as usize];
        }
        r[i] = b[i] - s;
    }
}

fn vec_norm(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }
fn vec_dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn tridiag(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i + 1 < n { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    fn nonsym_tridiag(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i + 1 < n { coo.add(i, i + 1, -2.0); }
        }
        coo.into_csr()
    }

    fn res_norm(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.nrows;
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
                s += a.values[idx] * x[a.col_idx[idx] as usize];
            }
            r[i] = b[i] - s;
        }
        r.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn cfg() -> BlockGmresConfig {
        BlockGmresConfig {
            base: SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, ..Default::default() },
            restart: 40,
        }
    }

    // ── single RHS ───────────────────────────────────────────────────────────

    #[test]
    fn single_rhs_spd_converges() {
        let n = 16; let a = tridiag(n);
        let b = vec![1.0f64; n]; let mut x = vec![0.0f64; n];
        let r = solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap();
        assert!(r.converged);
        assert!(res_norm(&a, &x, &b) < 1e-8);
    }

    #[test]
    fn single_rhs_nonsym_converges() {
        let n = 20; let a = nonsym_tridiag(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x = vec![0.0f64; n];
        let r = solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap();
        assert!(r.converged);
        assert!(res_norm(&a, &x, &b) < 1e-7);
    }

    // ── multiple RHS ─────────────────────────────────────────────────────────

    #[test]
    fn two_rhs_spd_both_converge() {
        let n = 16; let a = tridiag(n);
        let mut b = vec![0.0f64; n * 2];
        for i in 0..n { b[i] = 1.0; b[n + i] = (i % 3) as f64 + 0.5; }
        let mut x = vec![0.0f64; n * 2];
        let r = solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap();
        assert!(r.converged);
        for j in 0..2 {
            assert!(res_norm(&a, &x[j*n..(j+1)*n], &b[j*n..(j+1)*n]) < 1e-8);
        }
    }

    #[test]
    fn four_rhs_spd_all_converge() {
        let n = 24; let a = tridiag(n); let n_rhs = 4;
        let mut b = vec![0.0f64; n * n_rhs];
        for j in 0..n_rhs { for i in 0..n { b[j*n+i] = ((i+j) as f64).sin() + 1.0; } }
        let mut x = vec![0.0f64; n * n_rhs];
        let r = solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap();
        assert!(r.converged, "max_res = {:.3e}", r.final_residual);
        for j in 0..n_rhs {
            assert!(res_norm(&a, &x[j*n..(j+1)*n], &b[j*n..(j+1)*n]) < 1e-7);
        }
    }

    #[test]
    fn two_rhs_nonsym_both_converge() {
        let n = 20; let a = nonsym_tridiag(n);
        let mut b = vec![0.0f64; n * 2];
        for i in 0..n { b[i] = 1.0; b[n+i] = (i as f64 + 1.0).sqrt(); }
        let mut x = vec![0.0f64; n * 2];
        assert!(solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap().converged);
        for j in 0..2 {
            assert!(res_norm(&a, &x[j*n..(j+1)*n], &b[j*n..(j+1)*n]) < 1e-6);
        }
    }

    // ── solution quality ─────────────────────────────────────────────────────

    #[test]
    fn block_gmres_matches_exact_solution() {
        let n = 12; let a = tridiag(n);
        let mut b = vec![0.0f64; n]; b[0] = 1.0;
        let mut x = vec![0.0f64; n];
        solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap();
        assert!(res_norm(&a, &x, &b) < 1e-10);
    }

    #[test]
    fn zero_rhs_gives_zero_solution() {
        let n = 10; let a = tridiag(n);
        let b = vec![0.0f64; n]; let mut x = vec![0.0f64; n];
        let r = solve_block_gmres(&a, &b, &mut x, &cfg()).unwrap();
        assert!(r.converged);
        assert!(res_norm(&a, &x, &b) < 1e-14);
    }

    #[test]
    fn solution_scales_linearly_with_rhs() {
        let n = 16; let a = tridiag(n);
        let b1: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b2: Vec<f64> = b1.iter().map(|v| 2.0 * v).collect();
        let mut x1 = vec![0.0f64; n]; let mut x2 = vec![0.0f64; n];
        solve_block_gmres(&a, &b1, &mut x1, &cfg()).unwrap();
        solve_block_gmres(&a, &b2, &mut x2, &cfg()).unwrap();
        for i in 0..n {
            assert!((x2[i] - 2.0 * x1[i]).abs() < 1e-9, "node {i}");
        }
    }

    // ── configuration ────────────────────────────────────────────────────────

    #[test]
    fn tight_tolerance_still_converges() {
        let n = 16; let a = tridiag(n);
        let b = vec![1.0f64; n]; let mut x = vec![0.0f64; n];
        let c = BlockGmresConfig {
            base: SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..Default::default() },
            restart: 60,
        };
        assert!(solve_block_gmres(&a, &b, &mut x, &c).unwrap().converged);
        assert!(res_norm(&a, &x, &b) < 1e-11);
    }

    #[test]
    fn insufficient_iterations_returns_error() {
        let n = 64; let a = tridiag(n);
        let b = vec![1.0f64; n]; let mut x = vec![0.0f64; n];
        let c = BlockGmresConfig {
            base: SolverConfig { rtol: 1e-12, max_iter: 1, ..Default::default() },
            restart: 1,
        };
        assert!(solve_block_gmres(&a, &b, &mut x, &c).is_err());
    }

    // ── block matches individual calls ───────────────────────────────────────

    #[test]
    fn block_result_matches_individual_calls() {
        let n = 20; let a = tridiag(n); let n_rhs = 3;
        let mut b = vec![0.0f64; n * n_rhs];
        for j in 0..n_rhs {
            for i in 0..n { b[j*n+i] = (i as f64 * (j+1) as f64).cos() + 2.0; }
        }
        let c = cfg();
        let mut x_block = vec![0.0f64; n * n_rhs];
        solve_block_gmres(&a, &b, &mut x_block, &c).unwrap();
        let mut x_indiv = vec![0.0f64; n * n_rhs];
        for j in 0..n_rhs {
            solve_block_gmres(&a, &b[j*n..(j+1)*n], &mut x_indiv[j*n..(j+1)*n], &c).unwrap();
        }
        for i in 0..n * n_rhs {
            assert!((x_block[i] - x_indiv[i]).abs() < 1e-12, "index {i}");
        }
    }
}
