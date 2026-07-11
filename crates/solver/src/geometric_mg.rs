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
    pub jacobi_omega: f64,
    pub coarse_max_iter: usize,
    pub coarse_rtol: f64,
}

impl Default for GeometricMgConfig {
    fn default() -> Self {
        GeometricMgConfig {
            pre_sweeps: 2, post_sweeps: 2,
            jacobi_omega: 0.8, coarse_max_iter: 200, coarse_rtol: 1e-12,
        }
    }
}

/// Geometric multigrid V-cycle preconditioner.
pub struct GeometricMgPrecond {
    pub config: GeometricMgConfig,
}

impl GeometricMgPrecond {
    pub fn new(config: GeometricMgConfig) -> Self { GeometricMgPrecond { config } }

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

        // Pre-smooth
        jacobi_smooth(a, b, x, self.config.jacobi_omega, self.config.pre_sweeps, &level.bc_dofs);

        // Restrict residual: r_c = P^T * (b - A*x)
        let mut ax = vec![0.0; n];
        a.spmv(x, &mut ax);
        let mut r = vec![0.0; n];
        for i in 0..n { r[i] = b[i] - ax[i]; }
        let r_c = spmv_transpose(&h.prolong[lvl], &r);
        let n_c = h.levels[lvl + 1].mat.nrows;
        let mut e_c = vec![0.0; n_c];
        self.v_cycle_level(h, lvl + 1, &r_c, &mut e_c);

        // Prolongate and add correction
        let mut corr = vec![0.0; n];
        h.prolong[lvl].spmv(&e_c, &mut corr);
        for i in 0..n { x[i] += corr[i]; }

        // Post-smooth
        jacobi_smooth(a, b, x, self.config.jacobi_omega, self.config.post_sweeps, &level.bc_dofs);
        for &d in &level.bc_dofs { if (d as usize) < n { x[d as usize] = 0.0; } }
    }
}

// ─── Jacobi smoother ─────────────────────────────────────────────────────────

fn jacobi_smooth(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], omega: f64, sweeps: usize, bc: &[u32]) {
    let diag = a.diagonal();
    for _ in 0..sweeps {
        let mut ax = vec![0.0; b.len()];
        a.spmv(x, &mut ax);
        for i in 0..b.len() {
            if diag[i].abs() > 1e-30 {
                x[i] += omega * (b[i] - ax[i]) / diag[i];
            }
        }
        for &d in bc { if (d as usize) < b.len() { x[d as usize] = 0.0; } }
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
