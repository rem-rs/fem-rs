//! Stationary Linear Iteration (SLI) solvers.
//!
//! These are simple iterative methods that apply a fixed splitting:
//! - **Jacobi** (a.k.a. damped Richardson with diagonal scaling)
//! - **Gauss-Seidel** (forward sweep, uses updated values immediately)
//!
//! Convergence requires the matrix to be diagonally dominant or SPD (for ω < 2/ρ(D⁻¹A)).
//! Mainly useful as smoothers inside AMG or as simple standalone solvers
//! for diagonally dominant systems.

use fem_linalg::CsrMatrix;

use crate::{SolveResult, SolverConfig, PrintLevel};

/// Damped Jacobi stationary iteration.
///
/// Each step:  `x^{k+1} = x^k + ω D⁻¹ (b − A x^k)`
///
/// # Arguments
/// * `a`     — system matrix (CSR).
/// * `b`     — right-hand side.
/// * `x`     — initial guess on entry, solution on exit.
/// * `omega` — damping parameter (typically 2/3 for use as smoother).
/// * `cfg`   — convergence parameters.
pub fn solve_jacobi_sli(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    omega: f64,
    cfg: &SolverConfig,
) -> SolveResult {
    let n = a.nrows;
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);

    // Extract diagonal
    let mut diag = vec![1.0; n];
    for i in 0..n {
        let d = a.get(i, i);
        if d.abs() > 1e-30 { diag[i] = d; }
    }

    let mut r = vec![0.0; n];
    let b_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-30);

    let mut iter = 0;
    let mut residual = f64::MAX;

    for k in 0..cfg.max_iter {
        iter = k + 1;

        // r = b - A*x
        a.spmv(x, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }

        residual = r.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if cfg.verbose {
            eprintln!("  SLI Jacobi iter {iter}: ||r|| = {residual:.6e}");
        }
        if residual < cfg.atol || residual / b_norm < cfg.rtol {
            if matches!(cfg.effective_print_level(), PrintLevel::Summary | PrintLevel::Iterations | PrintLevel::Debug) {
                eprintln!("  SLI Jacobi converged in {iter} iterations (||r|| = {residual:.3e})");
            }
            return SolveResult { converged: true, iterations: iter, final_residual: residual };
        }

        // x += omega * D^{-1} * r
        for i in 0..n {
            x[i] += omega * r[i] / diag[i];
        }
    }

    SolveResult { converged: false, iterations: iter, final_residual: residual }
}

/// Forward Gauss-Seidel stationary iteration.
///
/// Each step sweeps rows 0..n in order, updating `x[i]` immediately:
///   `x_i = (b_i − Σ_{j≠i} a_ij x_j) / a_ii`
///
/// Convergence is guaranteed for SPD matrices (SOR with ω=1).
///
/// # Arguments
/// * `a`   — system matrix (CSR).
/// * `b`   — right-hand side.
/// * `x`   — initial guess on entry, solution on exit.
/// * `cfg` — convergence parameters.
pub fn solve_gs_sli(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> SolveResult {
    let n = a.nrows;
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);

    let b_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-30);
    let mut r = vec![0.0; n];
    let mut iter = 0;
    let mut residual = f64::MAX;

    for k in 0..cfg.max_iter {
        iter = k + 1;

        // Forward GS sweep
        for i in 0..n {
            let start = a.row_ptr[i];
            let end = a.row_ptr[i + 1];
            let mut s = b[i];
            let mut diag = 1.0;
            for p in start..end {
                let j = a.col_idx[p] as usize;
                if j == i {
                    diag = a.values[p];
                } else {
                    s -= a.values[p] * x[j];
                }
            }
            if diag.abs() > 1e-30 {
                x[i] = s / diag;
            }
        }

        // Check residual
        a.spmv(x, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }
        residual = r.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if cfg.verbose {
            eprintln!("  SLI GS iter {iter}: ||r|| = {residual:.6e}");
        }
        if residual < cfg.atol || residual / b_norm < cfg.rtol {
            if matches!(cfg.effective_print_level(), PrintLevel::Summary | PrintLevel::Iterations | PrintLevel::Debug) {
                eprintln!("  SLI GS converged in {iter} iterations (||r|| = {residual:.3e})");
            }
            return SolveResult { converged: true, iterations: iter, final_residual: residual };
        }
    }

    SolveResult { converged: false, iterations: iter, final_residual: residual }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::{CooMatrix, CsrMatrix};

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    fn diag_dominant(n: usize) -> CsrMatrix<f64> {
        // Strongly diagonally dominant for fast SLI convergence
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 4.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn jacobi_sli_converges() {
        let n = 20;
        let a = diag_dominant(n);
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let res = solve_jacobi_sli(&a, &b, &mut x, 2.0 / 3.0, &cfg);
        assert!(res.converged, "Jacobi SLI failed: iter={}, res={:.3e}", res.iterations, res.final_residual);
        // Verify Ax ≈ b
        let mut ax = vec![0.0; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "Jacobi residual too large: {err}");
    }

    #[test]
    fn gs_sli_converges() {
        let n = 20;
        let a = diag_dominant(n);
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let res = solve_gs_sli(&a, &b, &mut x, &cfg);
        assert!(res.converged, "GS SLI failed: iter={}, res={:.3e}", res.iterations, res.final_residual);
        let mut ax = vec![0.0; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "GS residual too large: {err}");
    }

    #[test]
    fn gs_faster_than_jacobi() {
        // GS should converge in fewer iterations than Jacobi for the same problem
        let n = 20;
        let a = diag_dominant(n);
        let b = vec![1.0; n];
        let cfg = SolverConfig { rtol: 1e-6, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };

        let mut x_j = vec![0.0; n];
        let res_j = solve_jacobi_sli(&a, &b, &mut x_j, 2.0 / 3.0, &cfg);

        let mut x_g = vec![0.0; n];
        let res_g = solve_gs_sli(&a, &b, &mut x_g, &cfg);

        assert!(res_j.converged, "Jacobi didn't converge");
        assert!(res_g.converged, "GS didn't converge");
        assert!(
            res_g.iterations <= res_j.iterations,
            "GS ({}) should converge in ≤ iterations than Jacobi ({})",
            res_g.iterations, res_j.iterations
        );
    }
}
