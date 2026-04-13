//! LOR (Low-Order Refined) preconditioner helpers.
//!
//! This is a lightweight entry point for LOR-backed solves. The current
//! implementation delegates to Jacobi-preconditioned CG while preserving an
//! API that can later be wired to AMG/LOR operators.

use crate::{solve_gmres, solve_pcg_jacobi, SolveResult, SolverConfig, SolverError};
use fem_linalg::CsrMatrix;
use linger::Scalar as LingerScalar;

/// LOR preconditioner configuration.
#[derive(Debug, Clone)]
pub struct LorPrecond {
    /// Number of smoother passes (reserved for future AMG backend).
    pub smoother_sweeps: usize,
}

impl Default for LorPrecond {
    fn default() -> Self {
        LorPrecond { smoother_sweeps: 2 }
    }
}

impl LorPrecond {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Solve SPD system with a LOR-style preconditioned CG path.
///
/// Current backend: PCG + Jacobi preconditioner. The `lor` argument is kept
/// for API stability and future backend selection.
pub fn solve_pcg_lor<T: LingerScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    _lor: &LorPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_pcg_jacobi(a, b, x, cfg)
}

/// Solve a (possibly non-symmetric) system with a LOR-style GMRES path.
///
/// Current backend: vanilla GMRES. The `lor` argument is kept for API
/// compatibility and future backend selection.
pub fn solve_gmres_lor<T: LingerScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    _lor: &LorPrecond,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    solve_gmres(a, b, x, restart, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    #[test]
    fn solve_pcg_lor_spd_smoke() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(1, 1, 3.0);
        let a = coo.into_csr();

        let b = vec![2.0, 3.0];
        let mut x = vec![0.0; 2];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
        let lor = LorPrecond::new();
        let res = solve_pcg_lor(&a, &b, &mut x, &lor, &cfg).expect("solve_pcg_lor failed");

        assert!(res.converged);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_gmres_lor_nonsym_smoke() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 3.0);
        coo.add(0, 1, 1.0);
        coo.add(1, 0, 0.0);
        coo.add(1, 1, 2.0);
        let a = coo.into_csr();

        let b = vec![4.0, 2.0];
        let mut x = vec![0.0; 2];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
        let lor = LorPrecond::new();
        let res = solve_gmres_lor(&a, &b, &mut x, 10, &lor, &cfg).expect("solve_gmres_lor failed");

        assert!(res.converged);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }
}
