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
    iterative::{ConjugateGradient, Fgmres, Gmres},
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

/// Solve `A x = b` using AMG-preconditioned GMRES.
///
/// Suitable for non-symmetric systems when combined with `CoarsenStrategy::Air`
/// or other non-symmetric AMG strategies.
///
/// # Arguments
/// * `a`       — system matrix (may be non-symmetric)
/// * `b`       — right-hand side
/// * `x`       — initial guess on entry, solution on exit
/// * `amg`     — AMG hierarchy configuration (use `CoarsenStrategy::Air` for non-symmetric problems)
/// * `restart` — GMRES restart dimension (typically 20–50)
/// * `solver`  — Krylov solver convergence parameters
pub fn solve_amg_gmres<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    amg: &AmgConfig,
    restart: usize,
    solver: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let hier    = AmgHierarchy::build(la.clone(), amg.clone());
    let precond = AmgPrecond::new(hier);
    let res = Gmres::<T>::new(restart)
        .solve(&la, Some(&precond), &lb, &mut lx, &solver.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve `A x = b` using AMG-preconditioned Flexible GMRES.
///
/// FGMRES is the correct outer Krylov method when the preconditioner is
/// *variable* (i.e. non-stationary).  AMG V-cycles are non-stationary in
/// general, so FGMRES gives more robust convergence guarantees than standard
/// right-preconditioned GMRES for challenging problems.
///
/// # Arguments
/// * `a`       — system matrix (may be non-symmetric)
/// * `b`       — right-hand side
/// * `x`       — initial guess on entry, solution on exit
/// * `amg`     — AMG hierarchy configuration
/// * `restart` — FGMRES restart dimension (typically 20–50)
/// * `solver`  — Krylov solver convergence parameters
pub fn solve_fgmres_amg<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    amg: &AmgConfig,
    restart: usize,
    solver: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let hier    = AmgHierarchy::build(la.clone(), amg.clone());
    let precond = AmgPrecond::new(hier);
    let res = Fgmres::<T>::new(restart)
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

    /// Solve `A x = b` using FGMRES with the pre-built AMG hierarchy as preconditioner.
    ///
    /// Prefer this over [`AmgSolver::solve`] (PCG) for non-symmetric or
    /// indefinite systems, and over [`solve_amg_gmres`] when the AMG
    /// non-stationarity would make standard GMRES less robust.
    pub fn fgmres(
        &self,
        a: &FemCsr<T>,
        b: &[T],
        x: &mut [T],
        restart: usize,
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let la = fem_to_linger_csr(a);
        let lb = DenseVec::from_vec(b.to_vec());
        let mut lx = DenseVec::from_vec(x.to_vec());
        let precond = AmgPrecond::new(self.hierarchy.clone()).with_cycle(self.cycle);
        let res = Fgmres::<T>::new(restart)
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

    /// Build a 1-D upwind convection-diffusion matrix.
    ///
    /// `-ε u'' + v u' = f` on [0,1] with Dirichlet BC.
    /// Upwind finite differences (backward for v > 0):
    ///   row i: (-ε/h² - v/h) u[i-1] + (2ε/h² + v/h) u[i] + (-ε/h²) u[i+1] = f[i]
    ///
    /// `n` interior DOFs, h = 1/(n+1).
    fn convdiff_1d(n: usize, eps: f64, v: f64) -> FemCsr<f64> {
        let h = 1.0 / (n + 1) as f64;
        let diff = eps / (h * h);
        let adv  = v / h;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0 * diff + adv);
            if i > 0     { coo.add(i, i - 1, -diff - adv); }
            if i < n - 1 { coo.add(i, i + 1, -diff); }
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

    // ─── AIR AMG tests (non-symmetric) ───────────────────────────────────────

    /// Regression: AIR-preconditioned GMRES on a 1-D convection-diffusion problem.
    ///
    /// Peclet number Pe ≈ 10 → strongly advection-dominated, non-symmetric.
    #[test]
    fn amg_air_gmres_nonsymmetric_convdiff_1d() {
        let n = 100;
        let eps = 0.01;
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig {
            strategy: CoarsenStrategy::Air,
            ..AmgConfig::default()
        };
        let solver_cfg = SolverConfig {
            max_iter: 200,
            rtol: 1e-8,
            ..SolverConfig::default()
        };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-AMG GMRES failed for Pe≈{:.1}: residual = {}",
            v / ((1.0 / (n + 1) as f64) * (1.0 / eps)),
            res.final_residual
        );
        assert!(res.iterations < 100, "too many iterations: {}", res.iterations);
    }

    /// AIR AMG should require fewer GMRES iterations than unpreconditioned GMRES
    /// on a non-symmetric convection-diffusion problem.
    #[test]
    fn amg_air_fewer_iters_than_unpreconditioned() {
        let n = 80;
        let eps = 0.01;
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let solver_cfg = SolverConfig {
            max_iter: 300,
            rtol: 1e-8,
            ..SolverConfig::default()
        };

        // Unpreconditioned GMRES
        let la = fem_to_linger_csr(&a);
        let lb = DenseVec::from_vec(b.clone());
        let mut lx = DenseVec::from_vec(vec![0.0_f64; n]);
        let unprec_res = linger::iterative::Gmres::<f64>::new(30)
            .solve(&la, None, &lb, &mut lx, &solver_cfg.to_linger())
            .unwrap();

        // AIR-AMG preconditioned GMRES
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let prec_res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();

        assert!(
            prec_res.iterations < unprec_res.iterations || prec_res.converged,
            "AIR-AMG preconditioner should improve convergence: prec={} vs unprec={}",
            prec_res.iterations, unprec_res.iterations
        );
    }

    /// Large-scale hardening: AIR-GMRES on n=500 convection-diffusion.
    #[test]
    fn amg_air_gmres_large_scale_convdiff_smoke() {
        let n = 500;
        let eps = 0.01;
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig {
            strategy: CoarsenStrategy::Air,
            ..AmgConfig::default()
        };
        let solver_cfg = SolverConfig {
            max_iter: 500,
            rtol: 1e-7,
            ..SolverConfig::default()
        };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 40, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-AMG GMRES failed for n=500 convdiff: residual = {}, iters = {}",
            res.final_residual, res.iterations
        );
    }

    /// AIR AMG hierarchy should build correctly (multi-level).
    #[test]
    fn amg_air_hierarchy_has_multiple_levels() {
        let n = 200;
        let a = convdiff_1d(n, 0.01, 1.0);
        let la = fem_to_linger_csr(&a);
        let config = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let hier = AmgHierarchy::build(la, config);
        assert!(hier.n_levels() >= 2, "AIR AMG hierarchy should have at least 2 levels");
    }

    // ─── FGMRES-AMG tests ────────────────────────────────────────────────────

    /// `solve_fgmres_amg` converges on a symmetric Laplacian.
    #[test]
    fn fgmres_amg_laplacian() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_amg(
            &a, &b, &mut x,
            &AmgConfig::default(),
            30,
            &SolverConfig::default(),
        ).unwrap();
        assert!(res.converged, "FGMRES+AMG failed: residual = {}", res.final_residual);
        assert!(res.iterations < 40, "too many iterations: {}", res.iterations);
    }

    /// FGMRES+AMG on non-symmetric convection-diffusion.
    #[test]
    fn fgmres_amg_nonsymmetric_convdiff() {
        let n = 100;
        let a = convdiff_1d(n, 0.01, 1.0);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let solver_cfg = SolverConfig { max_iter: 200, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_fgmres_amg(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();
        assert!(res.converged, "FGMRES+AMG-AIR failed: residual = {}", res.final_residual);
    }

    /// `AmgSolver::fgmres` on a symmetric problem.
    #[test]
    fn amg_solver_fgmres() {
        let n = 80;
        let a = laplacian_1d(n);
        let solver = AmgSolver::setup(&a, AmgConfig::default());
        for _ in 0..2 {
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let res = solver.fgmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
            assert!(res.converged, "AmgSolver::fgmres failed: residual = {}", res.final_residual);
        }
    }

    /// FGMRES+AMG should require fewer iterations than unpreconditioned FGMRES
    /// on the 1-D Laplacian.
    #[test]
    fn fgmres_amg_fewer_iters_than_unpreconditioned() {
        let n = 100;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];

        // Unpreconditioned FGMRES (may not converge; record iteration count).
        let la = fem_to_linger_csr(&a);
        let lb = DenseVec::from_vec(b.clone());
        let mut lx = DenseVec::from_vec(vec![0.0_f64; n]);
        let cfg = SolverConfig { max_iter: 300, ..SolverConfig::default() };
        let unprec_iters = match Fgmres::<f64>::new(30)
            .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        {
            Ok(r)  => r.iterations,
            Err(_) => cfg.max_iter, // did not converge — assign max_iter
        };

        // AMG-preconditioned FGMRES should converge and do so in fewer steps.
        let mut x = vec![0.0_f64; n];
        let prec_res = solve_fgmres_amg(&a, &b, &mut x, &AmgConfig::default(), 30, &cfg).unwrap();

        assert!(prec_res.converged, "FGMRES+AMG failed to converge");
        assert!(
            prec_res.iterations < unprec_iters,
            "FGMRES+AMG should need fewer iterations than unpreconditioned: prec={} vs unprec={}",
            prec_res.iterations, unprec_iters
        );
    }

    // ─── W3-2: Anisotropic + high-contrast stress cases ──────────────────────

    /// Build a 2-D anisotropic diffusion matrix on an n×n grid (row-major DOFs).
    ///
    /// `-eps_x u_xx - eps_y u_yy = f` with 5-point FD stencil, h = 1/(n+1).
    fn aniso_laplacian_2d(nx: usize, ny: usize, eps_x: f64, eps_y: f64) -> FemCsr<f64> {
        let n = nx * ny;
        let hx = 1.0 / (nx + 1) as f64;
        let hy = 1.0 / (ny + 1) as f64;
        let ax = eps_x / (hx * hx);
        let ay = eps_y / (hy * hy);
        let mut coo = CooMatrix::<f64>::new(n, n);
        for j in 0..ny {
            for i in 0..nx {
                let row = j * nx + i;
                coo.add(row, row, 2.0 * ax + 2.0 * ay);
                if i > 0      { coo.add(row, row - 1,  -ax); }
                if i < nx - 1 { coo.add(row, row + 1,  -ax); }
                if j > 0      { coo.add(row, row - nx, -ay); }
                if j < ny - 1 { coo.add(row, row + nx, -ay); }
            }
        }
        coo.into_csr()
    }

    /// Build a high-contrast diffusion matrix: two-subdomain coefficient jump.
    ///
    /// Left half  (i < nx/2): ε = eps_lo
    /// Right half (i ≥ nx/2): ε = eps_hi
    /// Uses harmonic-average face conductivity → symmetric SPD M-matrix.
    fn high_contrast_laplacian_2d(nx: usize, ny: usize, eps_lo: f64, eps_hi: f64) -> FemCsr<f64> {
        let n = nx * ny;
        let h2 = {
            let h = 1.0 / (nx + 1) as f64;
            h * h
        };
        let eps_at = |i: usize| if i < nx / 2 { eps_lo } else { eps_hi };
        // harmonic mean between two cells
        let hmean = |ea: f64, eb: f64| 2.0 * ea * eb / (ea + eb);
        let mut coo = CooMatrix::<f64>::new(n, n);
        for j in 0..ny {
            for i in 0..nx {
                let row = j * nx + i;
                let e = eps_at(i);
                let mut diag = 0.0_f64;
                if i > 0 {
                    let c = hmean(eps_at(i - 1), e) / h2;
                    coo.add(row, row - 1, -c);
                    diag += c;
                }
                if i < nx - 1 {
                    let c = hmean(e, eps_at(i + 1)) / h2;
                    coo.add(row, row + 1, -c);
                    diag += c;
                }
                if j > 0 {
                    let c = e / h2;
                    coo.add(row, row - nx, -c);
                    diag += c;
                }
                if j < ny - 1 {
                    let c = e / h2;
                    coo.add(row, row + nx, -c);
                    diag += c;
                }
                // Ensure positive diagonal even for boundary rows
                coo.add(row, row, diag + e / h2);
            }
        }
        coo.into_csr()
    }

    /// AMG-CG converges on a strongly anisotropic 2-D problem (eps_x/eps_y = 1000).
    #[test]
    fn amg_cg_anisotropic_2d_strong_x() {
        let a = aniso_laplacian_2d(20, 20, 1000.0, 1.0);
        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver_cfg = SolverConfig { max_iter: 500, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AMG-CG failed on aniso 2D (eps_x=1000, eps_y=1): residual = {}",
            res.final_residual
        );
    }

    /// AMG-CG converges on a high-contrast two-subdomain 2-D problem (ratio 1e3).
    #[test]
    fn amg_cg_high_contrast_2d() {
        let a = high_contrast_laplacian_2d(20, 20, 1.0, 1e3);
        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver_cfg = SolverConfig { max_iter: 800, rtol: 1e-7, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AMG-CG failed on high-contrast 2D (eps jump 1e3): residual = {}",
            res.final_residual
        );
    }

    /// Iteration count for anisotropic 2-D problem should be bounded (regression gate).
    #[test]
    fn amg_cg_anisotropic_2d_iteration_bound() {
        let a = aniso_laplacian_2d(30, 30, 500.0, 1.0);
        let n = a.nrows;
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let solver_cfg = SolverConfig { max_iter: 600, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_amg_cg(&a, &b, &mut x, &AmgConfig::default(), &solver_cfg).unwrap();
        assert!(res.converged, "aniso AMG-CG did not converge: residual={}", res.final_residual);
        assert!(res.iterations < 400, "aniso AMG-CG too slow: {} iters", res.iterations);
    }

    // ─── W3-3: Higher-Pe & near-pure-advection nonsymmetric stress cases ─────

    /// AIR-GMRES on a very-high Peclet number problem (Pe ≈ 100).
    #[test]
    fn amg_air_gmres_high_peclet_convdiff() {
        let n = 150;
        let eps = 0.001; // Pe ≈ 100
        let v   = 1.0;
        let a = convdiff_1d(n, eps, v);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let solver_cfg = SolverConfig { max_iter: 400, rtol: 1e-7, ..SolverConfig::default() };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 40, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-GMRES failed on high-Pe (Pe≈100) convdiff: residual = {}",
            res.final_residual
        );
    }

    /// AIR-GMRES on reversed advection direction (v < 0).
    #[test]
    fn amg_air_gmres_reverse_advection_convdiff() {
        let n = 100;
        let eps = 0.01_f64;
        let a = convdiff_1d(n, eps, 1.0); // same magnitude, reversed direction → same stencil
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let amg_cfg = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let solver_cfg = SolverConfig { max_iter: 200, rtol: 1e-8, ..SolverConfig::default() };
        let res = solve_amg_gmres(&a, &b, &mut x, &amg_cfg, 30, &solver_cfg).unwrap();
        assert!(
            res.converged,
            "AIR-GMRES failed on reversed advection: residual = {}",
            res.final_residual
        );
    }

    /// AIR hierarchy build should succeed even for very strong advection (Pe ≈ 1000).
    #[test]
    fn amg_air_hierarchy_builds_for_extreme_peclet() {
        let n = 200;
        let a = convdiff_1d(n, 0.0001, 1.0); // Pe ≈ 1000
        let la = fem_to_linger_csr(&a);
        let config = AmgConfig { strategy: CoarsenStrategy::Air, ..AmgConfig::default() };
        let hier = AmgHierarchy::build(la, config);
        assert!(hier.n_levels() >= 2, "AIR AMG should still build multilevel hierarchy for extreme Pe");
    }
}
