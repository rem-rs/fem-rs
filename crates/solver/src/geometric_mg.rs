//! Geometric multigrid preconditioner.
//!
//! Uses nested meshes (uniformly refined) as the multigrid hierarchy.
//! The polynomial order is the same at each level.
//!
//! # Usage
//! ```ignore
//! // Build hierarchy externally (e.g. in example code), then:
//! let mg = GeometricMgPrecond::default();
//! mg.v_cycle(&hierarchy, &b, &mut x);
//! ```

use fem_linalg::CsrMatrix;
use crate::SolverConfig;
use crate::constrained_operator::RectangularConstrainedOperator;

/// A single level in the geometric multigrid hierarchy.
pub struct GeometricMgLevel {
    /// System matrix at this level.
    pub mat: CsrMatrix<f64>,
    /// Boundary DOF list for this level.
    pub bc_dofs: Vec<u32>,
}

/// Geometric multigrid hierarchy.
///
/// `levels[0]` = finest, `levels[n-1]` = coarsest.
/// `prolong[l]` maps from level l+1 (coarse) to level l (fine) with BC enforcement.
pub struct GeometricMgHierarchy {
    pub levels: Vec<GeometricMgLevel>,
    pub prolong: Vec<RectangularConstrainedOperator>,
}

impl GeometricMgHierarchy {
    pub fn new(
        levels: Vec<GeometricMgLevel>,
        prolong_mat: Vec<CsrMatrix<f64>>,
    ) -> Self {
        assert_eq!(prolong_mat.len(), levels.len() - 1,
            "GeometricMgHierarchy: need len(prolong) == len(levels) - 1");
        let mut prolong = Vec::with_capacity(prolong_mat.len());
        for l in 0..prolong_mat.len() {
            prolong.push(RectangularConstrainedOperator {
                mat: prolong_mat[l].clone(),
                ess_fine: levels[l].bc_dofs.clone(),
                ess_coarse: levels[l + 1].bc_dofs.clone(),
            });
        }
        GeometricMgHierarchy { levels, prolong }
    }
    pub fn n_levels(&self) -> usize { self.levels.len() }
    pub fn finest_matrix(&self) -> &CsrMatrix<f64> { &self.levels[0].mat }
}

/// Geometric multigrid V-cycle configuration.
#[derive(Debug, Clone)]
pub struct GeometricMgConfig {
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    pub chebyshev_order: usize,  // set to 0 to use Jacobi smoothing
    pub jacobi_omega: f64,
    pub coarse_max_iter: usize,
    pub coarse_rtol: f64,
    /// Override λ_max estimate. None = auto-estimate.
    pub max_eig_override: Option<f64>,
}

impl Default for GeometricMgConfig {
    fn default() -> Self {
        GeometricMgConfig {
            pre_sweeps: 2, post_sweeps: 2,
            chebyshev_order: 2,
            jacobi_omega: 0.8,
            coarse_max_iter: 200, coarse_rtol: 1e-12,
            max_eig_override: None,
        }
    }
}

/// Chebyshev smoother for geometric MG.
/// Stores precomputed coefficients and diagonal.
pub struct MgChebyshevSmoother {
    dinv: Vec<f64>,
    coeffs: Vec<f64>,
}

impl MgChebyshevSmoother {
    fn new(a: &CsrMatrix<f64>, bc: &[u32], order: usize, max_eig_override: Option<f64>) -> Self {
        let n = a.nrows;
        let diag = a.diagonal();
        let mut dinv = vec![0.0; n];
        for i in 0..n { dinv[i] = if diag[i].abs() > 1e-30 { 1.0 / diag[i] } else { 1.0 }; }
        for &d in bc { if (d as usize) < n { dinv[d as usize] = 1.0; } }

        // Estimate λ_max(D⁻¹A) via power iteration, or use override
        let max_eig = max_eig_override.unwrap_or_else(|| {
            estimate_max_eigenvalue_simple(a, &dinv, bc)
        });

        // MFEM OperatorChebyshevSmoother parameters
        let upper = 1.2 * max_eig;
        let lower = 0.3 * max_eig;
        let theta = 0.5 * (upper + lower);
        let delta = 0.5 * (upper - lower);
        let th2 = theta * theta;
        let d2 = delta * delta;

        let coeffs = match order - 1 {
            0 => vec![1.0 / theta],
            1 => {
                let tmp = 1.0 / (d2 - 2.0 * th2);
                vec![-4.0 * theta * tmp, 2.0 * tmp]
            }
            2 => {
                let t0 = 3.0 * d2;
                let t1 = th2;
                let t2 = 1.0 / (-4.0 * theta * th2 + theta * t0);
                vec![t2 * (t0 - 12.0 * t1), 12.0 / (t0 - 4.0 * t1), -4.0 * t2]
            }
            _ => panic!("MgChebyshevSmoother: order {order} not supported (1-3)"),
        };
        MgChebyshevSmoother { dinv, coeffs }
    }

    fn smooth(&self, a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], bc: &[u32]) {
        let n = a.nrows;
        // Residual: r = b - A*x
        let mut r = vec![0.0; n];
        a.spmv(x, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }

        // Symmetric Chebyshev smoother: Δ = D⁻¹/² * p(D¹/²*A*D¹/²) * D⁻¹/² * r
        // where p(t) = Σ c_k * t^k is the Chebyshev polynomial.
        let d_sqrt: Vec<f64> = (0..n).map(|i| (1.0 / self.dinv[i]).sqrt()).collect();
        let d_inv_sqrt: Vec<f64> = (0..n).map(|i| self.dinv[i].sqrt()).collect();

        // Scale residual: r' = D⁻¹/² * r
        let mut v = vec![0.0; n];
        for i in 0..n { v[i] = r[i] * d_inv_sqrt[i]; }

        // Apply Chebyshev polynomial p(D¹/²*A*D¹/²) to v
        // p(D¹/²*A*D¹/²) * v = Σ c_k * (D¹/²*A*D¹/²)^k * v
        let m = self.coeffs.len();
        let mut sol = vec![0.0; n];
        let mut w = vec![0.0; n];
        for k in 0..m {
            let c = self.coeffs[k];
            for i in 0..n { sol[i] += c * v[i]; }
            if k + 1 < m {
                // w = D¹/² * A * (D¹/² * v) = D¹/² * A * D¹/² * v
                a.spmv(&v, &mut w);
                for i in 0..n { w[i] *= d_sqrt[i]; }
                v.copy_from_slice(&w);
            }
        }

        // Scale back: Δ = D⁻¹/² * sol
        for i in 0..n { x[i] += sol[i] * d_inv_sqrt[i]; }
        for &d in bc { if (d as usize) < n { x[d as usize] = 0.0; } }
    }
}

fn estimate_max_eigenvalue_simple(a: &CsrMatrix<f64>, dinv: &[f64], bc: &[u32]) -> f64 {
    let n = a.nrows;
    // Power iteration on D¹/²*A*D¹/² (symmetric, eigenvalues same as D⁻¹A).
    // Use D⁻¹/² as preconditioner: v ← D¹/² * A * D¹/² * v
    let d_sqrt: Vec<f64> = (0..n).map(|i| (1.0 / dinv[i]).sqrt()).collect();
    let mut v = vec![1.0; n];
    for i in 0..n { v[i] = 1.0 + (i as f64 % 5.0) * 0.1; }
    for &d in bc { if (d as usize) < n { v[d as usize] = 0.0; } }
    let normalize = |w: &mut [f64]| {
        let nrm: f64 = w.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);
        for x in w.iter_mut() { *x /= nrm; }
    };
    normalize(&mut v);
    let mut lambda = 0.0;
    let mut w = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    for _iter in 0..20 {
        // w = D¹/² * A * D¹/² * v
        for i in 0..n { tmp[i] = v[i] * d_sqrt[i]; }
        a.spmv(&tmp, &mut w);
        for i in 0..n { w[i] *= d_sqrt[i]; }
        for &d in bc { if (d as usize) < n { w[d as usize] = 0.0; } }
        let rq: f64 = (0..n).map(|i| v[i]*w[i]).sum();
        if (rq - lambda).abs() < 1e-4 * rq.abs() && _iter > 2 { lambda = rq; break; }
        lambda = rq;
        normalize(&mut w);
        v.copy_from_slice(&w);
    }
    lambda.abs().max(0.1).min(2.0)
}

/// Geometric multigrid V-cycle preconditioner.
pub struct GeometricMgPrecond {
    pub config: GeometricMgConfig,
    /// Pre-computed Chebyshev smoothers for each level.
    pub smoothers: Vec<MgChebyshevSmoother>,
}

impl GeometricMgPrecond {
    pub fn new(config: GeometricMgConfig, h: &GeometricMgHierarchy) -> Self {
        let cfg = &config;
        let smoothers = if cfg.chebyshev_order == 0 {
            // Build Jacobi smoothers (guaranteed SPD)
            Vec::new()
        } else {
            let mut s = Vec::new();
            for level in &h.levels {
                s.push(MgChebyshevSmoother::new(
                    &level.mat, &level.bc_dofs, cfg.chebyshev_order, cfg.max_eig_override));
            }
            s
        };
        GeometricMgPrecond { config, smoothers }
    }

    /// Apply one V-cycle: `x ← V-cycle(levels, prolong, b)` starting from zero.
    pub fn v_cycle(&self, h: &GeometricMgHierarchy, b: &[f64], x: &mut [f64]) {
        // Start from zero (matching MFEM MultigridBase::ArrayMult: *Y(M-1,j) = 0.0)
        for v in x.iter_mut() { *v = 0.0; }
        self.v_cycle_level(h, 0, b, x);
    }

    fn v_cycle_level(&self, h: &GeometricMgHierarchy, lvl: usize, b: &[f64], x: &mut [f64]) {
        let level = &h.levels[lvl];
        let a = &level.mat;
        let n = a.nrows;

        if lvl + 1 == h.levels.len() {
            // Coarsest level: CG solve
            let cfg = SolverConfig {
                rtol: self.config.coarse_rtol, atol: 0.0,
                max_iter: self.config.coarse_max_iter,
                verbose: false, ..Default::default()
            };
            let _res = crate::solve_cg(a, b, x, &cfg);
            // Debug: print coarse CG convergence
            // eprintln!("  coarse CG lvl={} iters={} resid={:.6e}", lvl,
            //           res.as_ref().map_or(0, |r| r.iterations),
            //           res.as_ref().map_or(0.0, |r| r.final_residual));
            for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
            return;
        }

        // Pre-smooth
        for _ in 0..self.config.pre_sweeps {
            self.smooth_level(lvl, a, b, x, &level.bc_dofs);
        }

        // Restrict residual (with BC enforcement)
        let mut ax = vec![0.0; n];
        a.spmv(x, &mut ax);
        let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let _r_norm: f64 = r.iter().map(|v| v*v).sum::<f64>().sqrt();
        let mut r_c = Vec::new();
        h.prolong[lvl].restrict(&r, &mut r_c);
        let _r_c_norm: f64 = r_c.iter().map(|v| v*v).sum::<f64>().sqrt();
        let n_c = h.levels[lvl + 1].mat.nrows;
        let mut e_c = vec![0.0; n_c];
        self.v_cycle_level(h, lvl + 1, &r_c, &mut e_c);
        let _e_c_norm: f64 = e_c.iter().map(|v| v*v).sum::<f64>().sqrt();

        // Prolongate correction (with BC enforcement)
        let mut corr = vec![0.0; n];
        h.prolong[lvl].prolong(&e_c, &mut corr);
        let _corr_norm: f64 = corr.iter().map(|v| v*v).sum::<f64>().sqrt();
        let _x_before = x.iter().map(|v| v*v).sum::<f64>().sqrt();
        for i in 0..n { x[i] += corr[i]; }

        // Post-smooth
        self.smooth_level(lvl, a, b, x, &level.bc_dofs);
        for _ in 1..self.config.post_sweeps {
            self.smooth_level(lvl, a, b, x, &level.bc_dofs);
        }
        for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
    }

    fn smooth_level(&self, lvl: usize, a: &CsrMatrix<f64>, b: &[f64],
                    x: &mut [f64], bc: &[u32]) {
        if lvl < self.smoothers.len() {
            self.smoothers[lvl].smooth(a, b, x, bc);
        } else {
            // Jacobi smoothing fallback
            let diag = a.diagonal();
            let mut r = vec![0.0; a.nrows];
            a.spmv(x, &mut r);
            for i in 0..r.len() { r[i] = b[i] - r[i]; }
            let omega = self.config.jacobi_omega;
            for i in 0..r.len() {
                if diag[i].abs() > 1e-30 { x[i] += omega * r[i] / diag[i]; }
            }
            for &d in bc { if (d as usize) < x.len() { x[d as usize] = 0.0; } }
        }
    }
}

