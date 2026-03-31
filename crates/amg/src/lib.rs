//! # fem-amg
//!
//! Algebraic Multigrid backed by [`linger`].
//!
//! Supports both classical Ruge–Stüben (RS) and smoothed aggregation (SA)
//! coarsening strategies, together with a full preconditioner menu.
//!
//! ## Usage
//! ```ignore
//! use fem_amg::{AmgConfig, CoarsenStrategy, solve_amg_cg};
//! use fem_linalg::CooMatrix;
//!
//! let mut coo = CooMatrix::<f64>::new(n, n);
//! // … fill …
//! let a = coo.into_csr();
//! let b = vec![1.0_f64; n];
//! let mut x = vec![0.0_f64; n];
//!
//! let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &Default::default()).unwrap();
//! assert!(res.converged);
//! ```

use fem_linalg::CsrMatrix as FemCsr;
use fem_solver::{fem_to_linger_csr, into_result, SolveResult, SolverConfig, SolverError};
use linger::{
    core::scalar::Scalar as LingerScalar,
    iterative::ConjugateGradient,
    DenseVec, KrylovSolver,
};

// Re-export linger AMG config types so callers don't need to depend on linger directly.
pub use linger::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType, SmootherType};

// ─── solve_amg_cg ────────────────────────────────────────────────────────────

/// Solve `A x = b` using AMG-preconditioned Conjugate Gradient.
///
/// Builds the AMG hierarchy once, wraps it as a preconditioner, and calls
/// `linger`'s PCG.
///
/// # Arguments
/// * `a`      — system matrix
/// * `b`      — right-hand side
/// * `x`      — initial guess on entry, solution on exit
/// * `amg`    — AMG hierarchy configuration
/// * `solver` — Krylov solver convergence parameters
pub fn solve_amg_cg<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    amg: &AmgConfig,
    solver: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let hier    = AmgHierarchy::build(la.clone(), amg.clone());
    let precond = AmgPrecond::new(hier);
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&precond), &lb, &mut lx, &solver.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── AmgSolver ───────────────────────────────────────────────────────────────

/// A reusable AMG hierarchy for repeated solves with the same matrix.
///
/// Factorisation cost is paid once via [`AmgSolver::setup`]; subsequent
/// [`AmgSolver::solve`] calls only run PCG cycles.
pub struct AmgSolver<T: LingerScalar> {
    hierarchy: AmgHierarchy<T>,
    cycle:     CycleType,
}

impl<T: LingerScalar> AmgSolver<T> {
    /// Build the AMG hierarchy for matrix `a`.
    pub fn setup(a: &FemCsr<T>, config: AmgConfig) -> Self {
        let la = fem_to_linger_csr(a);
        let hierarchy = AmgHierarchy::build(la, config);
        AmgSolver { hierarchy, cycle: CycleType::V }
    }

    /// Switch to W-cycle (more expensive but sometimes faster convergence).
    pub fn with_cycle(mut self, cycle: CycleType) -> Self {
        self.cycle = cycle; self
    }

    /// Number of levels in the AMG hierarchy.
    pub fn n_levels(&self) -> usize {
        self.hierarchy.n_levels()
    }

    /// Solve `A x = b` using the pre-built hierarchy.
    ///
    /// The hierarchy is cloned into the preconditioner wrapper so the
    /// `AmgSolver` can be reused across multiple right-hand sides.
    pub fn solve(
        &self,
        a: &FemCsr<T>,
        b: &[T],
        x: &mut [T],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let la = fem_to_linger_csr(a);
        let lb = DenseVec::from_vec(b.to_vec());
        let mut lx = DenseVec::from_vec(x.to_vec());
        let precond = AmgPrecond::new(self.hierarchy.clone()).with_cycle(self.cycle);
        let res = ConjugateGradient::<T>::default()
            .solve(&la, Some(&precond), &lb, &mut lx, &cfg.to_linger())
            .map_err(SolverError::from)?;
        x.copy_from_slice(lx.as_slice());
        Ok(into_result(res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn laplacian_1d(n: usize) -> FemCsr<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn amg_cg_laplacian() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_amg_cg(
            &a, &b, &mut x,
            &AmgConfig::default(),
            &SolverConfig::default(),
        ).unwrap();
        assert!(res.converged, "AMG-CG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 30, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn amg_solver_reuse() {
        let n = 80;
        let a = laplacian_1d(n);
        let solver = AmgSolver::setup(&a, AmgConfig::default());
        assert!(solver.n_levels() >= 2);
        for _ in 0..3 {
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let res = solver.solve(&a, &b, &mut x, &SolverConfig::default()).unwrap();
            assert!(res.converged);
        }
    }
}
