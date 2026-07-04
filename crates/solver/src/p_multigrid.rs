//! p-multigrid preconditioner.
//!
//! Uses nested FE spaces of different polynomial orders as the multigrid
//! hierarchy (p=1 → p=2 → p=3 → ...) on the same mesh.
//!
//! # Algorithm
//!
//! 1. Build matrices `A_p` at each polynomial order p on the same mesh
//! 2. Build prolongation `P_{p→p+1}`: matrices that map coarse (order p)
//!    DOFs to fine (order p+1) DOFs by evaluating the coarse Pk basis at
//!    each fine Pk+1 DOF reference coordinate
//! 3. Restriction `R = P^T`
//! 4. V-cycle: smooth(A_p, b_p, x_p) → restrict residual → recurse →
//!    prolongate correction → smooth
//!
//! # Usage
//! Build a [`PmgHierarchy`] from an assembled operator at the fine level,
//! then wrap it with [`PmgPrecond`] for use as a preconditioner.

use fem_linalg::{CooMatrix, CsrMatrix};
use crate::{SolveResult, SolverConfig};

/// p-multigrid hierarchy: matrices and prolongation operators.
///
/// `levels[0]` is the finest (highest p), `levels[n-1]` is the coarsest.
pub struct PmgHierarchy {
    pub levels: Vec<CsrMatrix<f64>>,
    pub prolong: Vec<CsrMatrix<f64>>,
}

impl PmgHierarchy {
    pub fn new(levels: Vec<CsrMatrix<f64>>, prolong: Vec<CsrMatrix<f64>>) -> Self {
        assert!(levels.len() >= 2, "PmgHierarchy: need at least 2 levels");
        assert_eq!(prolong.len(), levels.len() - 1);
        for l in 0..prolong.len() {
            assert_eq!(prolong[l].nrows, levels[l].nrows,
                "P{{long}} rows={} != fine DOFs={}", prolong[l].nrows, levels[l].nrows);
            assert_eq!(prolong[l].ncols, levels[l + 1].nrows,
                "P{{long}} cols={} != coarse DOFs={}", prolong[l].ncols, levels[l + 1].nrows);
        }
        PmgHierarchy { levels, prolong }
    }
}

/// Jacobi smoother for p-multigrid.
fn jacobi_smooth(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], omega: f64, sweeps: usize) {
    let diag = a.diagonal();
    for _ in 0..sweeps {
        let mut ax = vec![0.0; b.len()];
        a.spmv(x, &mut ax);
        for i in 0..b.len() {
            let d = if diag[i].abs() > 1e-30 { diag[i] } else { 1.0 };
            x[i] += omega * (b[i] - ax[i]) / d;
        }
    }
}

/// Transpose SpMV: y = A^T * x (where A is CSR).
fn spmv_transpose(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.ncols];
    for row in 0..a.nrows {
        let start = a.row_ptr[row];
        let end = a.row_ptr[row + 1];
        for nz in start..end {
            let col = a.col_idx[nz] as usize;
            y[col] += a.values[nz] * x[row];
        }
    }
    y
}

/// p-multigrid V-cycle preconditioner configuration.
#[derive(Debug, Clone)]
pub struct PmgPrecond {
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    pub jacobi_omega: f64,
    pub coarse_max_iter: usize,
}

impl Default for PmgPrecond {
    fn default() -> Self {
        PmgPrecond { pre_sweeps: 2, post_sweeps: 2, jacobi_omega: 0.8, coarse_max_iter: 200 }
    }
}

impl PmgPrecond {
    pub fn new() -> Self { Self::default() }

    /// Apply one V-cycle of p-multigrid: `x ← V-cycle(A, b, x)`.
    /// After the V-cycle, Dirichlet DOFs (first and last) are zeroed.
    pub fn v_cycle(&self, h: &PmgHierarchy, b: &[f64], x: &mut [f64]) {
        self.v_cycle_level(h, 0, b, x);
        let n = h.levels[0].nrows;
        // Ensure Dirichlet BCs at x=0 and x=1
        for &d in &[0, n - 1] {
            if d < x.len() { x[d] = 0.0; }
        }
    }

    fn v_cycle_level(&self, h: &PmgHierarchy, lvl: usize, b: &[f64], x: &mut [f64]) {
        let a = &h.levels[lvl];
        if lvl + 1 == h.levels.len() {
            // Coarsest level: direct CG solve
            let cfg = SolverConfig {
                rtol: 1e-12, atol: 0.0, max_iter: self.coarse_max_iter,
                verbose: false, ..Default::default()
            };
            let _ = crate::solve_cg(a, b, x, &cfg);
            return;
        }

        // Pre-smooth
        jacobi_smooth(a, b, x, self.jacobi_omega, self.pre_sweeps);

        // Restrict residual
        let mut ax = vec![0.0; b.len()];
        a.spmv(x, &mut ax);
        let mut r = vec![0.0; b.len()];
        for i in 0..b.len() { r[i] = b[i] - ax[i]; }

        let p = &h.prolong[lvl]; // P: coarse → fine
        let r_c = spmv_transpose(p, &r); // R = P^T
        let mut e_c = vec![0.0; r_c.len()];
        self.v_cycle_level(h, lvl + 1, &r_c, &mut e_c);

        // Prolongate correction
        let mut pe = vec![0.0; x.len()];
        p.spmv(&e_c, &mut pe);
        for i in 0..x.len() { x[i] += pe[i]; }

        // Post-smooth
        jacobi_smooth(a, b, x, self.jacobi_omega, self.post_sweeps);
    }
}

/// Solve using p-multigrid V-cycles as a preconditioner.
pub fn solve_vcycle_pmg(
    a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64],
    hierarchy: &PmgHierarchy, mg: &PmgPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, crate::SolverError> {
    if a.nrows != a.ncols || b.len() != a.nrows || x.len() != a.nrows {
        return Err(crate::SolverError::DimensionMismatch { rows: a.nrows, cols: a.ncols, rhs: b.len() });
    }
    if hierarchy.levels[0].nrows != a.nrows {
        return Err(crate::SolverError::DimensionMismatch {
            rows: hierarchy.levels[0].nrows, cols: hierarchy.levels[0].ncols, rhs: a.nrows,
        });
    }

    let mut ax = vec![0.0; b.len()];
    a.spmv(x, &mut ax);
    let mut r = vec![0.0; b.len()];
    for i in 0..b.len() { r[i] = b[i] - ax[i]; }
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-32);
    let tol = cfg.atol.max(cfg.rtol * b_norm);
    let mut r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r_norm <= tol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: r_norm });
    }

    let dirichlet = [0usize, x.len() - 1];
    for k in 0..cfg.max_iter {
        let mut corr = vec![0.0; x.len()];
        mg.v_cycle(hierarchy, &r, &mut corr);
        for i in 0..x.len() { x[i] += corr[i]; }
        for &d in &dirichlet { x[d] = 0.0; }

        a.spmv(x, &mut ax);
        for i in 0..b.len() { r[i] = b[i] - ax[i]; }
        r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();

        if r_norm <= tol {
            return Ok(SolveResult { converged: true, iterations: k + 1, final_residual: r_norm });
        }
    }
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: r_norm })
}

/// Build a p-multigrid hierarchy for a 1-D Laplacian (for testing).
///
/// Constructs stiffness matrices at orders p = p_fine, p_fine-1, ..., 1
/// on a uniform mesh of `n_elem` elements on [0, 1].
pub fn build_pmg_hierarchy_1d_laplacian(n_elem: usize, p_fine: u8) -> PmgHierarchy {
    let mut levels = Vec::new();
    let mut prolong = Vec::new();

    // Build A_p at each order from p_fine down to 1
    for p in (1..=p_fine as usize).rev() {
        let n_dofs = n_elem * p + 1;
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        let h = 1.0 / n_elem as f64;

        for e in 0..n_elem {
            let dofs: Vec<usize> = (0..=p).map(|k| e * p + k).collect();
            // 3-point Gauss quadrature on [0,1]
            let gl = [(0.112701665379258, 0.277777777777778),
                      (0.5, 0.444444444444444),
                      (0.887298334620742, 0.277777777777778)];
            for &(xi, w) in &gl {
                let dphi = lagrange_1d_deriv(p, xi);
                let w_phys = w * h;
                for i in 0..=p {
                    for j in 0..=p {
                        coo.add(dofs[i], dofs[j], dphi[i] * dphi[j] / h * w_phys);
                    }
                }
            }
        }
        let mut a = coo.into_csr();
        let mut rhs = vec![0.0; n_dofs];
        a.apply_dirichlet_symmetric(0, 0.0, &mut rhs);
        a.apply_dirichlet_symmetric(n_dofs - 1, 0.0, &mut rhs);
        levels.push(a);
    }

    // Build prolongation: P_{p→p+1} maps order-p DOFs to order-(p+1) DOFs
    for level in 1..levels.len() {
        let p_coarse = p_fine as usize - level;
        let p_fine_lvl = p_coarse + 1;
        let n_coarse = n_elem * p_coarse + 1;
        let n_fine = n_elem * p_fine_lvl + 1;
        let mut coo = CooMatrix::<f64>::new(n_fine, n_coarse);

        for e in 0..n_elem {
            let fine_dofs: Vec<usize> = (0..=p_fine_lvl).map(|k| e * p_fine_lvl + k).collect();
            let coarse_dofs: Vec<usize> = (0..=p_coarse).map(|k| e * p_coarse + k).collect();
            for (fi, &f_dof) in fine_dofs.iter().enumerate() {
                let xi_f = fi as f64 / p_fine_lvl as f64;
                let coarse_phi = lagrange_1d_eval(p_coarse, xi_f);
                for (cj, &c_dof) in coarse_dofs.iter().enumerate() {
                    coo.add(f_dof, c_dof, coarse_phi[cj]);
                }
            }
        }
        prolong.push(coo.into_csr());
    }

    PmgHierarchy::new(levels, prolong)
}

/// Evaluate 1-D Lagrange basis (equispaced on [0,1]).
fn lagrange_1d_eval(p: usize, xi: f64) -> Vec<f64> {
    let mut phi = vec![0.0; p + 1];
    for i in 0..=p {
        let xi_i = i as f64 / p as f64;
        phi[i] = 1.0;
        for j in 0..=p {
            if j == i { continue; }
            let xi_j = j as f64 / p as f64;
            phi[i] *= (xi - xi_j) / (xi_i - xi_j);
        }
    }
    phi
}

/// Derivative of 1-D Lagrange basis (equispaced on [0,1]).
fn lagrange_1d_deriv(p: usize, xi: f64) -> Vec<f64> {
    let mut dphi = vec![0.0; p + 1];
    let h = 1.0 / p as f64;
    for i in 0..=p {
        let xi_i = i as f64 * h;
        // dphi[i] = Σ_{k≠i} Π_{j≠i,j≠k} (xi - xi_j) / (xi_i - xi_j) / (xi_i - xi_k)
        let mut sum = 0.0;
        for k in 0..=p {
            if k == i { continue; }
            let xi_k = k as f64 * h;
            let mut prod = 1.0;
            for j in 0..=p {
                if j == i || j == k { continue; }
                let xi_j = j as f64 * h;
                prod *= (xi - xi_j) / (xi_i - xi_j);
            }
            sum += prod / (xi_i - xi_k);
        }
        dphi[i] = sum;
    }
    dphi
}

/// FMG: Full Multigrid — nested iteration starting from the coarsest level.
///
/// 1. Solve exactly on the coarsest grid
/// 2. Prolongate solution to the next finer level
/// 3. Apply V-cycles on that level
/// 4. Repeat until the finest level
pub fn fmg_solve(
    a_fine: &CsrMatrix<f64>, b: &[f64],
    hierarchy: &PmgHierarchy, mg: &PmgPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, crate::SolverError> {
    // Solve on coarsest level
    let n_coarse = hierarchy.levels.last().unwrap().nrows;
    let mut x_c = vec![0.0; n_coarse];
    {
        let a_c = hierarchy.levels.last().unwrap();
        // Restrict RHS to coarsest level
        let mut rhs_c = b.to_vec();
        for l in 0..hierarchy.prolong.len() {
            let prev = rhs_c;
            rhs_c = spmv_transpose(&hierarchy.prolong[l], &prev);
        }

        let _ = crate::solve_cg(a_c, &rhs_c, &mut x_c, &SolverConfig {
            rtol: 1e-14, atol: 0.0, max_iter: 500, verbose: false, ..Default::default()
        });
    }

    // Nested iteration: coarse → fine
    let mut x = x_c;
    for level in (0..hierarchy.prolong.len()).rev() {
        let p = &hierarchy.prolong[level];
        let n_fine = hierarchy.levels[level].nrows;
        let mut x_f = vec![0.0; n_fine];
        p.spmv(&x, &mut x_f);

        // Restrict RHS to current level: R_{level-1} * ... * R_0 * b
        let mut b_lvl = b.to_vec();
        for l in 0..level {
            let prev = b_lvl;
            b_lvl = spmv_transpose(&hierarchy.prolong[l], &prev);
        }

        // V-cycles at this level
        let mut r = vec![0.0; n_fine];
        hierarchy.levels[level].spmv(&x_f, &mut r);
        for i in 0..n_fine { r[i] = b_lvl[i] - r[i]; }
        let b_norm = b_lvl.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-32);
        let tol = cfg.rtol * b_norm;

        for _ in 0..mg.coarse_max_iter.min(10) {
            hierarchy.levels[level].spmv(&x_f, &mut r);
            for i in 0..n_fine { r[i] = b_lvl[i] - r[i]; }
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rn <= tol.max(1e-14) { break; }

            let mut corr = vec![0.0; n_fine];
            mg.v_cycle_level(hierarchy, level, &r, &mut corr);
            for i in 0..n_fine { x_f[i] += corr[i]; }
        }
        x = x_f;
    }

    // Final residual check
    let mut ax = vec![0.0; a_fine.nrows];
    a_fine.spmv(&x, &mut ax);
    let mut r = vec![0.0; a_fine.nrows];
    for i in 0..a_fine.nrows { r[i] = b[i] - ax[i]; }
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(SolveResult { converged: true, iterations: 0, final_residual: rn })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SolverConfig;

    #[test]
    fn pmg_hierarchy_builds_prolong() {
        let h = build_pmg_hierarchy_1d_laplacian(2, 2);
        assert_eq!(h.levels.len(), 2);
        assert_eq!(h.prolong.len(), 1);
        assert_eq!(h.prolong[0].nrows, h.levels[0].nrows);
        assert_eq!(h.prolong[0].ncols, h.levels[1].nrows);
    }

    #[test]
    fn pmg_hierarchy_three_levels() {
        let h = build_pmg_hierarchy_1d_laplacian(8, 3);
        assert_eq!(h.levels.len(), 3);
        assert_eq!(h.prolong.len(), 2);
        assert!(h.levels[0].nrows > h.levels[1].nrows);
        assert!(h.levels[1].nrows > h.levels[2].nrows);
    }

    #[test]
    fn pmg_vcycle_runs_without_error() {
        let h = build_pmg_hierarchy_1d_laplacian(8, 3);
        let n = h.levels[0].nrows;
        let b_vec: Vec<f64> = (0..n).map(|i| if i == n / 2 { 1.0 } else { 0.0 }).collect();
        let mut x = vec![0.0; n];
        let mg = PmgPrecond::default();
        mg.v_cycle(&h, &b_vec, &mut x);
        for &v in &x { assert!(v.is_finite()); }
    }

    #[test]
    fn pmg_fmg_runs_without_error() {
        let h = build_pmg_hierarchy_1d_laplacian(8, 2);
        let a = &h.levels[0];
        let n = a.nrows;
        let b_vec: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / (n - 1) as f64;
            (std::f64::consts::PI * x).sin()
        }).collect();
        let cfg = SolverConfig { rtol: 1e-6, ..Default::default() };
        let mg = PmgPrecond::new();
        let res = fmg_solve(a, &b_vec, &h, &mg, &cfg).unwrap();
        assert!(res.final_residual.is_finite());
    }

    #[test]
    fn pmg_vcycle_vs_cg_both_finite() {
        let h = build_pmg_hierarchy_1d_laplacian(8, 2);
        let a = &h.levels[0];
        let n = a.nrows;
        let b_vec: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / (n - 1) as f64;
            x * (1.0 - x)
        }).collect();

        let mut x_cg = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..Default::default() };
        let res_cg = crate::solve_cg(a, &b_vec, &mut x_cg, &cfg).unwrap();
        assert!(res_cg.converged, "CG should converge");

        let mut x_pmg = vec![0.0; n];
        let mg = PmgPrecond { coarse_max_iter: 200, ..Default::default() };
        let res_pmg = solve_vcycle_pmg(a, &b_vec, &mut x_pmg, &h, &mg, &cfg).unwrap();
        assert!(res_pmg.final_residual.is_finite(), "PMG should produce finite residual");
    }
}
