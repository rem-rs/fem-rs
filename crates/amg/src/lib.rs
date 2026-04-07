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

// ─── Chebyshev smoother ──────────────────────────────────────────────────────

/// Chebyshev polynomial smoother for SPD matrices.
///
/// Applies `n_iter` iterations of the Chebyshev-accelerated iteration
/// as a smoother for a system `A x = b`.  The eigenvalue bounds
/// `[lambda_min, lambda_max]` determine the Chebyshev polynomial interval.
///
/// The smoother minimizes the error in the A-norm over the polynomial
/// space — better than Jacobi for high-frequency error when the spectrum
/// is well-bounded.
///
/// # Arguments
/// * `a`           — SPD system matrix.
/// * `x`           — current iterate (updated in-place).
/// * `b`           — right-hand side.
/// * `lambda_min`  — lower eigenvalue bound (typically `λ_max / ratio`, ratio ≈ 30).
/// * `lambda_max`  — upper eigenvalue bound (estimate via Gershgorin or a few power iterations).
/// * `n_iter`      — number of Chebyshev iterations (typically 2–5).
pub fn chebyshev_smooth(
    a: &FemCsr<f64>,
    x: &mut [f64],
    b: &[f64],
    lambda_min: f64,
    lambda_max: f64,
    n_iter: usize,
) {
    let n = x.len();
    let alpha = (lambda_max + lambda_min) / 2.0;
    let delta = (lambda_max - lambda_min) / 2.0;

    let diag_inv: Vec<f64> = (0..n)
        .map(|i| {
            let di = a.get(i, i);
            if di.abs() > 1e-14 { 1.0 / di } else { 1.0 }
        })
        .collect();

    let mut r = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n]; // search direction

    // Standard Chebyshev iteration (see Golub & Van Loan, or Saad ch. 12):
    //   d₀ = (2/α) M⁻¹ r₀
    //   dₖ = ρₖ (2/α M⁻¹ rₖ + ρₖ₋₁ dₖ₋₁)
    // where ρ₀ = 1, ρ₁ = 1/(1 - δ²/(2α²)), ρₖ = 1/(1 - (δ/(2α))² ρₖ₋₁)

    let two_over_alpha = 2.0 / alpha;
    let quarter_delta_sq_over_alpha_sq = (delta / (2.0 * alpha)).powi(2);

    let mut rho_prev = 0.0_f64;

    for k in 0..n_iter {
        // r = b - A x
        a.spmv(x, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }

        if k == 0 {
            for i in 0..n { d[i] = two_over_alpha * diag_inv[i] * r[i]; }
            rho_prev = 1.0;
        } else {
            let rho = 1.0 / (1.0 - quarter_delta_sq_over_alpha_sq * rho_prev);
            for i in 0..n {
                d[i] = rho * (two_over_alpha * diag_inv[i] * r[i] + (rho - 1.0) * d[i]);
            }
            rho_prev = rho;
        }

        for i in 0..n { x[i] += d[i]; }
    }
}

/// Estimate the spectral radius `ρ(A)` using a few power iterations.
///
/// Returns an upper bound on the largest eigenvalue magnitude.
pub fn estimate_spectral_radius(a: &FemCsr<f64>, n_iter: usize) -> f64 {
    let n = a.nrows;
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut w = vec![0.0_f64; n];
    let mut rho = 1.0;

    for _ in 0..n_iter {
        a.spmv(&v, &mut w);
        rho = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if rho > 1e-14 {
            let inv = 1.0 / rho;
            for i in 0..n { v[i] = w[i] * inv; }
        }
    }
    rho
}

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

    #[test]
    fn chebyshev_reduces_residual() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];

        let rho = estimate_spectral_radius(&a, 20);
        let lambda_max = rho * 1.1;
        let lambda_min = lambda_max / 30.0;

        chebyshev_smooth(&a, &mut x, &b, lambda_min, lambda_max, 20);

        // Check residual is reduced.
        let mut r = vec![0.0_f64; n];
        a.spmv(&x, &mut r);
        let res_norm: f64 = r.iter().zip(b.iter())
            .map(|(ri, bi)| (bi - ri).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(res_norm / b_norm < 0.9, "Chebyshev should reduce residual, got {}", res_norm / b_norm);
    }

    #[test]
    fn spectral_radius_estimate() {
        let n = 50;
        let a = laplacian_1d(n);
        let rho = estimate_spectral_radius(&a, 30);
        // Eigenvalues of 1-D Laplacian: 2 - 2cos(kπ/(n+1)), k=1..n
        // Largest ≈ 4 for large n.
        assert!(rho > 3.5, "spectral radius should be near 4, got {rho}");
        assert!(rho < 4.1, "spectral radius too high: {rho}");
    }

    #[test]
    fn amg_cg_chebyshev_smoother() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let config = AmgConfig {
            smoother: SmootherType::Chebyshev { degree: 3, ratio: 3.0 },
            ..AmgConfig::default()
        };
        let res = solve_amg_cg(&a, &b, &mut x, &config, &SolverConfig::default()).unwrap();
        assert!(res.converged, "Chebyshev AMG-CG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 50, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn amg_cg_fcycle() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver = AmgSolver::setup(&a, AmgConfig::default()).with_cycle(CycleType::F);
        let res = solver.solve(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "F-cycle AMG-CG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 30, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn amg_cg_chebyshev_with_fcycle() {
        let n = 80;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let config = AmgConfig {
            smoother: SmootherType::Chebyshev { degree: 3, ratio: 3.0 },
            ..AmgConfig::default()
        };
        let solver = AmgSolver::setup(&a, config).with_cycle(CycleType::F);
        let res = solver.solve(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "Chebyshev+F-cycle failed: residual = {}", res.final_residual);
    }
}
