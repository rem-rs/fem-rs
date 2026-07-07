//! Staggered phase-field fracture solver.
//!
//! Alternates between displacement solve (with Miehe spectral split in 2D)
//! and phase-field solve until convergence.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::pf_solver::{PhaseFieldSolver, StaggeredConfig, StaggeredStepResult};
//! ```

use crate::phasefield::{
    assemble_degraded_stiffness, assemble_miehe_stiffness_and_force,
    assemble_phase_field_system, compute_elastic_energy,
    compute_psi_plus, update_history_field,
};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

/// Configuration for the staggered phase-field solver.
#[derive(Debug, Clone)]
pub struct StaggeredConfig {
    pub at1: bool,
    pub max_iter: usize,
    pub tol_u: f64,
    pub tol_d: f64,
    pub verbose: bool,
}

impl Default for StaggeredConfig {
    fn default() -> Self {
        Self { at1: false, max_iter: 50, tol_u: 1e-6, tol_d: 1e-6, verbose: false }
    }
}

/// Result of one load step in the staggered solver.
#[derive(Debug, Clone)]
pub struct StaggeredStepResult {
    pub converged: bool,
    pub iterations: usize,
    pub u_norm: f64,
    pub d_norm: f64,
    pub psi_max: f64,
    pub d_max: f64,
}

/// Staggered phase-field fracture solver.
pub struct PhaseFieldSolver;

impl PhaseFieldSolver {
    /// Run one staggered load step.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_step<M: MeshTopology>(
        cfg: &StaggeredConfig, mesh: &M,
        space_u: &dyn FESpace<Mesh = M>, space_d: &dyn FESpace<Mesh = M>,
        u_elem_dofs: &[Vec<usize>], d_elem_dofs: &[Vec<usize>],
        u_n_ldofs: usize, d_n_ldofs: usize,
        u: &mut [f64], d: &mut [f64], history: &mut [f64],
        lambda: f64, mu: f64, g_c: f64, l: f64, kappa_eps: f64,
        quad_order: u8, rhs: &[f64],
        linear_cfg: &fem_solver::SolverConfig,
    ) -> StaggeredStepResult {
        let n_u = u.len(); let n_d = d.len();
        let dim = mesh.dim() as usize;
        let n_elems = mesh.n_elements();
        for iter in 0..cfg.max_iter {
            let u_old = u.to_vec(); let d_old = d.to_vec();
            let (k_uu, f_int) = if dim == 2 {
                assemble_miehe_stiffness_and_force(mesh, space_u, u_elem_dofs, u_n_ldofs,
                    u, d, space_d, d_elem_dofs, d_n_ldofs, lambda, mu, kappa_eps, quad_order)
            } else {
                let k = assemble_degraded_stiffness(mesh, space_u, u_elem_dofs, u_n_ldofs,
                    d, space_d, d_elem_dofs, d_n_ldofs, lambda, mu, kappa_eps, quad_order);
                let mut fi = vec![0.0; n_u];
                for i in 0..n_u { for p in k.row_ptr[i]..k.row_ptr[i+1] { fi[i] += k.values[p] * u[k.col_idx[p] as usize]; } }
                (k, fi)
            };
            let mut r = f_int; for i in 0..n_u { r[i] -= rhs[i]; }
            let mut du = vec![0.0; n_u];
            let neg_r: Vec<f64> = r.iter().map(|&v| -v).collect();
            let _ = fem_solver::solve_cg(&k_uu, &neg_r, &mut du, linear_cfg);
            for i in 0..n_u { u[i] += du[i]; }
            let n_qp = if dim == 2 {
                let (pp, n) = compute_psi_plus(mesh, space_u, u_elem_dofs, u_n_ldofs, u, lambda, mu, quad_order);
                update_history_field(history, &pp); n
            } else {
                let (pe, n) = compute_elastic_energy(mesh, space_u, u_elem_dofs, u_n_ldofs, u, lambda, mu, quad_order);
                update_history_field(history, &pe); n
            };
            let n_qp_per = if n_qp > 0 { n_qp } else { history.len() / n_elems.max(1) };
            let (k_dd, rhs_d) = assemble_phase_field_system(mesh, space_d, d_elem_dofs, d_n_ldofs,
                history, n_qp_per, g_c, l, quad_order);
            let mut dd = vec![0.0; n_d];
            let neg_rd: Vec<f64> = rhs_d.iter().map(|&v| -v).collect();
            let _ = fem_solver::solve_cg(&k_dd, &neg_rd, &mut dd, linear_cfg);
            for i in 0..n_d { d[i] = -dd[i]; }
            for di in d.iter_mut() { *di = di.clamp(0.0, 1.0); }
            let du_norm = (0..n_u).map(|i| (u[i]-u_old[i]).powi(2)).sum::<f64>().sqrt();
            let dd_norm = (0..n_d).map(|i| (d[i]-d_old[i]).powi(2)).sum::<f64>().sqrt();
            let u_nv = u.iter().map(|v| v.powi(2)).sum::<f64>().sqrt().max(1e-30);
            let d_nv = d.iter().map(|v| v.powi(2)).sum::<f64>().sqrt().max(1e-30);
            let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
            if cfg.verbose { eprintln!("[PF] iter {iter}: du={:.3e} dd={:.3e} d_max={:.4e}", du_norm/u_nv, dd_norm/d_nv, d_max); }
            if du_norm / u_nv < cfg.tol_u && dd_norm / d_nv < cfg.tol_d {
                return StaggeredStepResult {
                    converged: true, iterations: iter+1, u_norm: u_nv, d_norm: d_nv,
                    psi_max: history.iter().cloned().fold(0.0_f64, f64::max), d_max,
                };
            }
        }
        StaggeredStepResult {
            converged: false, iterations: cfg.max_iter,
            u_norm: u.iter().map(|v| v.powi(2)).sum::<f64>().sqrt(),
            d_norm: d.iter().map(|v| v.powi(2)).sum::<f64>().sqrt(),
            psi_max: history.iter().cloned().fold(0.0_f64, f64::max),
            d_max: d.iter().cloned().fold(0.0_f64, f64::max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, VectorH1Space};

    #[test]
    fn staggered_config_defaults_valid() {
        let cfg = StaggeredConfig::default();
        assert!(!cfg.at1);
        assert_eq!(cfg.max_iter, 50);
        assert!(cfg.tol_u > 0.0);
    }
}
