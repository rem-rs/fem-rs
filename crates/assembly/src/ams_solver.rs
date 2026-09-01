//! AMS-preconditioned PCG for H(curl) systems.
//!
//! [`solve_hcurl_ams`] builds the discrete gradient `G : H¹ → H(curl)`,
//! eliminates essential (PEC) boundary degrees of freedom, and solves
//! the reduced SPD system with PCG + AMS (Auxiliary-space Maxwell Solver).
//!
//! This is the recommended solver for static Maxwell (curl-curl + mass)
//! problems — it converges in far fewer iterations than point preconditioners
//! (Jacobi, SSOR).

use fem_solver::amg::{AmgConfig, AmgSolver};
use fem_linalg::{CooMatrix, CsrMatrix, SolveResult, fem_to_linlvo_csr};
use fem_mesh::MeshTopology;
use fem_solver::{
    AmsSolverConfig, SolverConfig, solve_pcg_ams,
    eigen::{lobpcg_constrained_preconditioned, LobpcgConfig, EigenResult},
};
use fem_space::{
    H1Space,
    HCurlSpace,
    constraints::{eliminate_dirichlet, expand_from_reduced},
    fe_space::FESpace,
};
use linlvo::precond::{AmsPrecond, AmsConfig};
use linlvo::core::preconditioner::Preconditioner;
use linlvo::core::vector::DenseVec;
use nalgebra::DMatrix;

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
pub fn solve_hcurl_ams<M: MeshTopology + Clone + 'static>(
    space: &HCurlSpace<M>,
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

/// Solve the Maxwell eigenvalue problem `A x = λ M x` using gradient constraints
/// + AMG preconditioner + block oversampling.
///
/// Note: AMS is NOT used here as a LOBPCG preconditioner because its auxiliary-space
/// correction `G·P_v⁻¹·Gᵀ` maps the residual into range(G), which is precisely the
/// space projected out by the gradient constraints.  Hence AMS degenerates to the
/// diagonal smoother for this use case.  AMS + singularity_regularization is
/// instead intended for the future AME (Adaptive Multigrid Eigensolver)
/// implementation where the nullspace is handled internally without explicit
/// constraints — matching MFEM ex13p's HYPRE AME solver.
pub fn solve_hcurl_eigen(
    stiffness_free: &CsrMatrix<f64>,
    mass_free: &CsrMatrix<f64>,
    gradient_constraints: &DMatrix<f64>,
    gradient_full_for_ams: &CsrMatrix<f64>,
    dim: usize,
    k: usize,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    let mut cfg = cfg.clone();
    cfg.max_iter = cfg.max_iter.max(300);

    // Block oversampling (constraints consume DOFs, so cap at n - n_constraints)
    let n_constraints = gradient_constraints.ncols();
    let k_work = (k + 20).min(stiffness_free.nrows.saturating_sub(n_constraints));

    let a_clone = stiffness_free.clone();
    let precond: Box<dyn Fn(&DMatrix<f64>) -> DMatrix<f64> + Send + Sync> = if dim == 3 {
        // AMS preconditioner (auxiliary-space Maxwell): the right tool for the
        // 3-D ND1 curl-curl operator (Hiptmair-Xu with edge smoother + nodal
        // coarse space).  Plain algebraic multigrid converges very slowly on
        // the (non-elliptic) 3-D curl-curl system.
        let a_linlvo = fem_to_linlvo_csr(stiffness_free);
        let g_linlvo = fem_to_linlvo_csr(gradient_full_for_ams);
        let ams = AmsPrecond::new(&a_linlvo, &g_linlvo, Default::default())
            .map_err(|e| format!("AMS setup failed: {e}"))?;
        Box::new(move |r: &DMatrix<f64>| -> DMatrix<f64> {
            let nk = r.ncols(); let mut z = DMatrix::<f64>::zeros(r.nrows(), nk);
            for j in 0..nk {
                let rhs: Vec<f64> = r.column(j).iter().copied().collect();
                let mut x = vec![0.0; a_clone.nrows];
                let lr = DenseVec::from_vec(rhs);
                let mut lx = DenseVec::from_vec(x.clone());
                ams.apply_precond(&lr, &mut lx);
                x.copy_from_slice(lx.as_slice());
                for i in 0..z.nrows() { z[(i, j)] = x[i]; }
            }
            z
        })
    } else {
        // 2-D: AMG on the reduced curl-curl system (validated on the ex13 2-D
        // alignment, 12-14 significant digits).
        let amg = AmgSolver::setup(stiffness_free, AmgConfig::default());
        Box::new(move |r: &DMatrix<f64>| -> DMatrix<f64> {
            let nk = r.ncols(); let mut z = DMatrix::<f64>::zeros(r.nrows(), nk);
            for j in 0..nk {
                let rhs: Vec<f64> = r.column(j).iter().copied().collect();
                let mut x = vec![0.0; a_clone.nrows];
                if amg.solve(&a_clone, &rhs, &mut x, &SolverConfig { max_iter: 20, rtol: 1e-2, atol: 1e-12, verbose: false, ..SolverConfig::default() }).is_err() {
                    x.copy_from_slice(&rhs);
                }
                for i in 0..z.nrows() { z[(i, j)] = x[i]; }
            }
            z
        })
    };

    let result = lobpcg_constrained_preconditioned(
        stiffness_free, Some(mass_free), k_work, gradient_constraints, precond, &cfg,
    )?;

    let n_found = result.eigenvalues.len().min(k);
    Ok(EigenResult {
        eigenvalues: result.eigenvalues[..n_found].to_vec(),
        eigenvectors: DMatrix::from(result.eigenvectors.columns(0, n_found).to_owned()),
        iterations: result.iterations,
        converged: result.converged,
    })
}

/// AMS-preconditioned LOBPCG for the Maxwell eigenvalue problem.
///
/// Unlike [`solve_hcurl_eigen`], this version does NOT use explicit gradient
/// constraints — instead it relies on AMS's internal nullspace handling
/// (via `singularity_regularization`) together with LOBPCG's `nullspace_skip`
/// to exclude gradient modes.  This is a stepping stone toward a full AME
/// implementation.
///
/// **Caveat**: AMS on the constraint-complement only activates the diagonal
/// smoother because the auxiliary-space correction maps back into the
/// constraint space.  Use this only when testing against an AME-based
/// reference (MFEM ex13p).
#[allow(dead_code)]
pub fn solve_hcurl_eigen_ams(
    stiffness_free: &CsrMatrix<f64>,
    mass_free: &CsrMatrix<f64>,
    gradient_constraints: &DMatrix<f64>,
    gradient_full_for_ams: &CsrMatrix<f64>,
    k: usize,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    let mut cfg = cfg.clone();
    cfg.max_iter = cfg.max_iter.max(300);
    // Skip gradient nullspace modes (λ ≈ 0) in the Rayleigh-Ritz selection.
    // AMS preconditioning on the unconstrained system can introduce nullspace
    // components into the search space; nullspace_skip ensures the algorithm
    // converges to physical curl-curl modes rather than gradient nullspace.
    cfg.nullspace_skip = 1e-8;

    // Block oversampling (constraints consume DOFs, so cap at n - n_constraints)
    let n_constraints = gradient_constraints.ncols();
    let k_work = (k + 20).min(stiffness_free.nrows.saturating_sub(n_constraints));

    // ── AMS preconditioner with singularity regularization ──────────────────
    let a_linlvo = fem_to_linlvo_csr(stiffness_free);
    let g_linlvo = fem_to_linlvo_csr(gradient_full_for_ams);
    let ams = AmsPrecond::<f64>::new(
        &a_linlvo,
        &g_linlvo,
        AmsConfig {
            singularity_regularization: 1e-6,
            smoother_sweeps: 3,
            ..AmsConfig::default()
        },
    )
    .map_err(|e| format!("AMS setup for LOBPCG failed: {e}"))?;

    // Block preconditioner: apply AMS column-by-column to the residual block.
    let precond = move |r: &DMatrix<f64>| -> DMatrix<f64> {
        let n = r.nrows();
        let nvec = r.ncols();
        let mut z = DMatrix::<f64>::zeros(n, nvec);
        for j in 0..nvec {
            let rhs_dv = DenseVec::from_vec(r.column(j).iter().copied().collect());
            let mut z_dv = DenseVec::zeros(n);
            ams.apply_precond(&rhs_dv, &mut z_dv);
            let zs = z_dv.as_slice();
            for i in 0..n {
                z[(i, j)] = zs[i];
            }
        }
        z
    };

    let result = lobpcg_constrained_preconditioned(
        stiffness_free, Some(mass_free), k_work, gradient_constraints, precond, &cfg,
    )?;

    let n_found = result.eigenvalues.len().min(k);
    Ok(EigenResult {
        eigenvalues: result.eigenvalues[..n_found].to_vec(),
        eigenvectors: DMatrix::from(result.eigenvectors.columns(0, n_found).to_owned()),
        iterations: result.iterations,
        converged: result.converged,
    })
}
