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
}

impl Default for GeometricMgConfig {
    fn default() -> Self {
        GeometricMgConfig {
            pre_sweeps: 2, post_sweeps: 2,
            chebyshev_order: 2,  // degree-2 Chebyshev (matching C++ ex26)
            jacobi_omega: 0.8,
            coarse_max_iter: 200, coarse_rtol: 1e-12,
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
    fn new(a: &CsrMatrix<f64>, bc: &[u32], order: usize) -> Self {
        let n = a.nrows;
        let diag = a.diagonal();
        let mut dinv = vec![0.0; n];
        for i in 0..n { dinv[i] = if diag[i].abs() > 1e-30 { 1.0 / diag[i] } else { 1.0 }; }
        for &d in bc { if (d as usize) < n { dinv[d as usize] = 1.0; } }

        // Estimate λ_max(D⁻¹A) via power iteration
        let max_eig = estimate_max_eigenvalue_simple(a, &dinv, bc);

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
        // Compute residual: r = b - A*x
        let mut r = vec![0.0; n];
        a.spmv(x, &mut r);
        for i in 0..n { r[i] = b[i] - r[i]; }

        // Apply Chebyshev polynomial to residual: r ← p(D⁻¹A) D⁻¹ * r
        let m = self.coeffs.len();
        let mut sol = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        for k in 0..m {
            if k > 0 { a.spmv(&tmp, &mut r); }
            for i in 0..n { r[i] *= self.dinv[i]; }
            let c = self.coeffs[k];
            for i in 0..n { sol[i] += c * r[i]; }
            tmp.copy_from_slice(&r);
        }

        // x += correction
        for i in 0..n { x[i] += sol[i]; }
        for &d in bc { if (d as usize) < n { x[d as usize] = 0.0; } }
    }
}

fn estimate_max_eigenvalue_simple(a: &CsrMatrix<f64>, dinv: &[f64], bc: &[u32]) -> f64 {
    let n = a.nrows;
    // Use Gershgorin circle: λ_max ≈ max_i (Σ_j |A[i,j]| / D[i])
    // For the Jacobi-preconditioned Laplacian, this gives a good estimate.
    let mut max_gersh = 0.0_f64;
    for i in 0..n {
        if bc.contains(&(i as u32)) { continue; }
        let mut row_sum = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            row_sum += a.values[p].abs();
        }
        let val = row_sum * dinv[i];
        if val > max_gersh { max_gersh = val; }
    }
    // The Gershgorin estimate is an upper bound. λ_max is typically less.
    // For 2D Laplacian, use 0.9 * max_gersh as a rough estimate.
    let lambda = 0.9 * max_gersh;
    lambda.max(0.1)
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
                    &level.mat, &level.bc_dofs, cfg.chebyshev_order));
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
            let _ = crate::solve_cg(a, b, x, &cfg);
            for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
            return;
        }

        // Pre-smooth (skip if pre_sweeps == 0)
        for _ in 0..self.config.pre_sweeps {
            self.smooth_level(lvl, a, b, x, &level.bc_dofs);
        }

        // Restrict residual (with BC enforcement)
        let mut ax = vec![0.0; n];
        a.spmv(x, &mut ax);
        let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let mut r_c = Vec::new();
        h.prolong[lvl].restrict(&r, &mut r_c);
        let n_c = h.levels[lvl + 1].mat.nrows;
        let mut e_c = vec![0.0; n_c];
        self.v_cycle_level(h, lvl + 1, &r_c, &mut e_c);

        // Prolongate correction (with BC enforcement)
        let mut corr = vec![0.0; n];
        h.prolong[lvl].prolong(&e_c, &mut corr);
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

