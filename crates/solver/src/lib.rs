//! # fem-solver
//!
//! Iterative linear solvers backed by [`linger`] — a pure-Rust FEA solver library.
//!
//! ## Solvers
//! - [`solve_cg`]          — Conjugate Gradient (SPD systems)
//! - [`solve_pcg_jacobi`]  — PCG with Jacobi preconditioner
//! - [`solve_pcg_ilu0`]    — PCG with ILU(0) preconditioner
//! - [`solve_gmres`]       — GMRES (non-symmetric systems)
//! - [`solve_bicgstab`]    — BiCGSTAB
//!
//! All solvers operate on [`fem_linalg::CsrMatrix<T>`] and return a
//! [`SolveResult`] with iteration count and final residual.

use fem_linalg::CsrMatrix as FemCsr;
use linger::{
    core::scalar::Scalar as LingerScalar,
    iterative::{BiCgStab, ConjugateGradient, Gmres},
    sparse::CsrMatrix as LingerCsr,
    DenseVec, Ilu0Precond, JacobiPrecond, KrylovSolver, SolverParams, VerboseLevel,
};
use thiserror::Error;

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("solver did not converge in {max_iter} iterations (residual = {residual:.3e})")]
    ConvergenceFailed { max_iter: usize, residual: f64 },
    #[error("dimension mismatch: matrix is {rows}×{cols}, rhs has length {rhs}")]
    DimensionMismatch { rows: usize, cols: usize, rhs: usize },
    #[error("linger error: {0}")]
    Linger(String),
}

impl From<linger::SolverError> for SolverError {
    fn from(e: linger::SolverError) -> Self {
        match e {
            linger::SolverError::ConvergenceFailed { max_iter, residual } => {
                SolverError::ConvergenceFailed { max_iter, residual }
            }
            other => SolverError::Linger(other.to_string()),
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Outcome returned by every solver in this crate.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub converged:      bool,
    pub iterations:     usize,
    pub final_residual: f64,
}

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Convergence parameters passed to every solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Relative tolerance (converge when ‖r‖/‖b‖ < rtol).
    pub rtol: f64,
    /// Absolute tolerance.
    pub atol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Print residual each iteration when `true`.
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 1_000, verbose: false }
    }
}

impl SolverConfig {
    pub fn to_linger(&self) -> SolverParams {
        SolverParams {
            rtol:           self.rtol,
            atol:           self.atol,
            max_iter:       self.max_iter,
            verbose: if self.verbose { VerboseLevel::Iterations } else { VerboseLevel::Silent },
            check_interval: 10,
        }
    }
}

// ─── Type conversion ─────────────────────────────────────────────────────────

/// Convert a `fem_linalg::CsrMatrix<T>` to a `linger::sparse::CsrMatrix<T>`.
///
/// The only structural difference is that fem-rs stores column indices as
/// `u32` while linger uses `usize`.
pub fn fem_to_linger_csr<T: LingerScalar>(a: &FemCsr<T>) -> LingerCsr<T> {
    LingerCsr::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

// ─── Solvers ─────────────────────────────────────────────────────────────────

/// Conjugate Gradient — for symmetric positive definite systems.
///
/// # Arguments
/// * `a`   — system matrix (fem-rs CSR)
/// * `b`   — right-hand side
/// * `x`   — initial guess on entry, solution on exit
/// * `cfg` — convergence parameters
pub fn solve_cg<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = ConjugateGradient::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Preconditioned CG with a Jacobi (diagonal scaling) preconditioner.
pub fn solve_pcg_jacobi<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = JacobiPrecond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Preconditioned CG with an ILU(0) preconditioner.
///
/// Requires the matrix to have a factorisation-compatible sparsity pattern.
pub fn solve_pcg_ilu0<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = Ilu0Precond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES — for general (possibly non-symmetric) systems.
///
/// `restart` controls the Krylov subspace dimension before restart (default 30).
pub fn solve_gmres<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// BiCGSTAB — for non-symmetric systems; often faster than GMRES per iteration.
pub fn solve_bicgstab<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = BiCgStab::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn check_dims<T>(a: &FemCsr<T>, b: &[T], x: &[T]) -> Result<(), SolverError> {
    if a.nrows != b.len() || a.ncols != x.len() {
        return Err(SolverError::DimensionMismatch {
            rows: a.nrows,
            cols: a.ncols,
            rhs:  b.len(),
        });
    }
    Ok(())
}

pub fn into_result(r: linger::SolverResult) -> SolveResult {
    SolveResult {
        converged:      r.converged,
        iterations:     r.iterations,
        final_residual: r.final_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::{CooMatrix, CsrMatrix};

    /// 1-D Laplacian: tridiagonal [-1, 2, -1] of size n.
    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn cg_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_cg(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "CG failed to converge");
        // verify Ax ≈ b
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "residual too large: {err}");
    }

    #[test]
    fn pcg_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_jacobi(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged);
        assert!(res.iterations < 60, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn gmres_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }
}
