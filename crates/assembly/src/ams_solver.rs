//! AMS-preconditioned PCG for H(curl) systems.
//!
//! [`solve_hcurl_ams`] builds the discrete gradient `G : H¹ → H(curl)`,
//! eliminates essential (PEC) boundary degrees of freedom, and solves
//! the reduced SPD system with PCG + AMS (Auxiliary-space Maxwell Solver).
//!
//! This is the recommended solver for static Maxwell (curl-curl + mass)
//! problems — it converges in far fewer iterations than point preconditioners
//! (Jacobi, SSOR).

use fem_linalg::{CooMatrix, CsrMatrix, SolveResult, fem_to_linlvo_csr};
use fem_mesh::Mesh;
use fem_solver::{AmsSolverConfig, SolverConfig, solve_pcg_ams};
use fem_space::{
    H1Space,
    HCurlSpace,
    constraints::{eliminate_dirichlet, expand_from_reduced},
    fe_space::FESpace,
};

use crate::discrete_op::DiscreteLinearOperator;

/// Solve an H(curl) system with PCG + AMS preconditioner.
///
/// # Arguments
/// * `space` — H(curl) finite element space (the same space the system was assembled with)
/// * `mat`   — assembled H(curl) stiffness matrix (full, before BC elimination)
/// * `rhs`   — assembled right-hand side vector
/// * `pec_dofs` — DOFs constrained by PEC (n×E = 0), from [`boundary_dofs_hcurl`]
/// * `cfg`   — solver configuration (rtol, max_iter, …)
///
/// # Returns
/// `(solution, solve_result)` where `solution` is the full-length vector
/// (n_dofs, with zeros at PEC DOFs) and `solve_result` contains convergence info.
///
/// # Panics
/// Panics if the discrete gradient construction fails or the solver fails.
pub fn solve_hcurl_ams(
    space: &HCurlSpace<Mesh<2>>,
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    pec_dofs: &[u32],
    cfg: &SolverConfig,
) -> (Vec<f64>, SolveResult) {
    let n_dofs = space.n_dofs();

    if pec_dofs.is_empty() {
        // No PEC BCs — solve the full system directly with AMS.
        let h1_mesh = space.mesh().clone();
        let h1_space = H1Space::new(h1_mesh, 1);
        let g_full = DiscreteLinearOperator::gradient(&h1_space, space)
            .expect("discrete gradient construction failed");
        let g_linlvo = fem_to_linlvo_csr(&g_full);
        let ams_cfg = AmsSolverConfig {
            inner_cfg: cfg.clone(),
            ams_cfg: Default::default(),
        };
        let mut u = vec![0.0_f64; n_dofs];
        let res = solve_pcg_ams(mat, &g_linlvo, rhs, &mut u, &ams_cfg)
            .expect("AMS-PCG solve failed");
        (u, res)
    } else {
        // Eliminate PEC DOFs → reduced SPD system.
        let pec_vals = vec![0.0_f64; pec_dofs.len()];
        let (sys_mat, sys_rhs, free_map, constrained_map) =
            eliminate_dirichlet(mat, rhs, pec_dofs, &pec_vals);
        let n_sys = free_map.len();

        // Build the discrete gradient and restrict to free DOFs.
        let h1_mesh = space.mesh().clone();
        let h1_space = H1Space::new(h1_mesh, 1);
        let n_h1 = h1_space.n_dofs();
        let g_full = DiscreteLinearOperator::gradient(&h1_space, space)
            .expect("discrete gradient construction failed");

        let mut g_coo = CooMatrix::new(n_sys, n_h1);
        for (fi, &orig) in free_map.iter().enumerate() {
            let start = g_full.row_ptr[orig] as usize;
            let end = g_full.row_ptr[orig + 1] as usize;
            for p in start..end {
                g_coo.add(fi, g_full.col_idx[p] as usize, g_full.values[p]);
            }
        }
        let g_linlvo = fem_to_linlvo_csr(&g_coo.into_csr());

        let ams_cfg = AmsSolverConfig {
            inner_cfg: cfg.clone(),
            ams_cfg: Default::default(),
        };
        let mut x_red = vec![0.0_f64; n_sys];
        let res = solve_pcg_ams(&sys_mat, &g_linlvo, &sys_rhs, &mut x_red, &ams_cfg)
            .expect("AMS-PCG solve failed");

        let u = expand_from_reduced(&x_red, &free_map, &constrained_map, &pec_vals, n_dofs);
        (u, res)
    }
}
