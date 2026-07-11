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
/// `prolong[l]` maps from level l+1 (coarse) to level l (fine).
pub struct GeometricMgHierarchy {
    pub levels: Vec<GeometricMgLevel>,
    pub prolong: Vec<CsrMatrix<f64>>,
}

impl GeometricMgHierarchy {
    pub fn new(levels: Vec<GeometricMgLevel>, prolong: Vec<CsrMatrix<f64>>) -> Self {
        assert_eq!(prolong.len(), levels.len() - 1,
            "GeometricMgHierarchy: need len(prolong) == len(levels) - 1");
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
    pub chebyshev_order: usize,
    pub coarse_max_iter: usize,
    pub coarse_rtol: f64,
}

impl Default for GeometricMgConfig {
    fn default() -> Self {
        GeometricMgConfig {
            pre_sweeps: 2, post_sweeps: 2,
            chebyshev_order: 2,  // degree-2 Chebyshev (matching C++ ex26)
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
    let mut v = vec![1.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n { v[i] = 1.0 + (i as f64 % 7.0) / 7.0; }
    for &d in bc { if (d as usize) < n { v[d as usize] = 0.0; } }
    let mut lambda = 0.0;
    for _ in 0..30 {
        a.spmv(&v, &mut w);
        for i in 0..n { w[i] *= dinv[i]; }
        for &d in bc { if (d as usize) < n { w[d as usize] = 0.0; } }
        let vw: f64 = (0..n).map(|i| v[i]*w[i]).sum();
        let vv: f64 = (0..n).map(|i| v[i]*v[i]).sum();
        if vv < 1e-30 { break; }
        let nl = vw / vv;
        if (nl - lambda).abs() < 1e-6 * nl.abs() { lambda = nl; break; }
        lambda = nl;
        let nrm = w.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);
        for i in 0..n { v[i] = w[i] / nrm; }
        for &d in bc { if (d as usize) < n { v[d as usize] = 0.0; } }
    }
    lambda.abs().max(0.1)
}

/// Geometric multigrid V-cycle preconditioner.
pub struct GeometricMgPrecond {
    pub config: GeometricMgConfig,
    /// Pre-computed Chebyshev smoothers for each level.
    pub smoothers: Vec<MgChebyshevSmoother>,
}

impl GeometricMgPrecond {
    pub fn new(config: GeometricMgConfig, h: &GeometricMgHierarchy) -> Self {
        let mut smoothers = Vec::new();
        for level in &h.levels {
            smoothers.push(MgChebyshevSmoother::new(
                &level.mat, &level.bc_dofs, config.chebyshev_order));
        }
        GeometricMgPrecond { config, smoothers }
    }

    /// Apply one V-cycle: `x ← V-cycle(levels, prolong, b, x)`.
    pub fn v_cycle(&self, h: &GeometricMgHierarchy, b: &[f64], x: &mut [f64]) {
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

        // Pre-smooth (Chebyshev)
        self.smoothers[lvl].smooth(a, b, x, &level.bc_dofs);
        for _ in 1..self.config.pre_sweeps {
            self.smoothers[lvl].smooth(a, b, x, &level.bc_dofs);
        }

        // Restrict residual
        let mut ax = vec![0.0; n];
        a.spmv(x, &mut ax);
        let mut r = vec![0.0; n];
        for i in 0..n { r[i] = b[i] - ax[i]; }
        let r_c = spmv_transpose(&h.prolong[lvl], &r);
        let n_c = h.levels[lvl + 1].mat.nrows;
        let mut e_c = vec![0.0; n_c];
        self.v_cycle_level(h, lvl + 1, &r_c, &mut e_c);

        // Prolongate correction
        let mut corr = vec![0.0; n];
        h.prolong[lvl].spmv(&e_c, &mut corr);
        for i in 0..n { x[i] += corr[i]; }

        // Post-smooth (Chebyshev)
        self.smoothers[lvl].smooth(a, b, x, &level.bc_dofs);
        for _ in 1..self.config.post_sweeps {
            self.smoothers[lvl].smooth(a, b, x, &level.bc_dofs);
        }
        for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
    }
}

// ─── y = A^T * x ─────────────────────────────────────────────────────────────

fn spmv_transpose(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.ncols];
    for row in 0..a.nrows {
        for p in a.row_ptr[row]..a.row_ptr[row + 1] {
            y[a.col_idx[p] as usize] += a.values[p] * x[row];
        }
    }
    y
}
