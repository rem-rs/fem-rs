//! Parallel Restricted Additive Schwarz (RAS) preconditioning scaffolding.
//!
//! This module introduces a DDM-oriented preconditioner entrypoint for
//! parallel Krylov solvers.  The first implementation intentionally keeps the
//! local subdomain solve lightweight (diagonal scaling on owned rows) so the
//! integration path is stable while overlap/local-solver kernels evolve.

use fem_solver::{SolveResult, SolverConfig, SolverError};
use linger::{DenseVec, Ilu0Precond, Preconditioner as _};

use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

/// Local subdomain solver choice for RAS.
///
/// More options (ILU0 / sparse direct) can be added without changing the
/// external `par_solve_pcg_ras` interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasLocalSolverKind {
    /// Diagonal inverse on owned rows.
    DiagJacobi,
    /// Local ILU(0) on the owned-owned block.
    Ilu0,
}

/// Rank-aggregated diagnostics useful for large-scale RAS runs.
#[derive(Debug, Clone)]
pub struct RasHpcDiagnostics {
    pub n_ranks: usize,
    pub global_owned_dofs: usize,
    pub global_ghost_dofs: usize,
    pub global_diag_nnz: usize,
    pub global_offd_nnz: usize,
    pub mean_owned_dofs_per_rank: f64,
    pub mean_ghost_dofs_per_rank: f64,
    /// Coefficient of variation (std/mean) for owned DOFs per rank.
    pub owned_dofs_cv: f64,
    /// Coefficient of variation (std/mean) for ghost DOFs per rank.
    pub ghost_dofs_cv: f64,
}

/// Build an aggregated parallel-health summary for the current distributed matrix.
pub fn summarize_ras_hpc(a: &ParCsrMatrix) -> RasHpcDiagnostics {
    let comm = a.comm();
    let n_ranks = comm.size();

    let owned = a.n_owned as f64;
    let ghost = a.n_ghost as f64;
    let owned_sq = owned * owned;
    let ghost_sq = ghost * ghost;

    let sum_owned = comm.allreduce_sum_f64(owned);
    let sum_ghost = comm.allreduce_sum_f64(ghost);
    let sum_owned_sq = comm.allreduce_sum_f64(owned_sq);
    let sum_ghost_sq = comm.allreduce_sum_f64(ghost_sq);

    let mean_owned = sum_owned / n_ranks as f64;
    let mean_ghost = sum_ghost / n_ranks as f64;
    let var_owned = (sum_owned_sq / n_ranks as f64 - mean_owned * mean_owned).max(0.0);
    let var_ghost = (sum_ghost_sq / n_ranks as f64 - mean_ghost * mean_ghost).max(0.0);

    let owned_cv = if mean_owned > 1e-30 { var_owned.sqrt() / mean_owned } else { 0.0 };
    let ghost_cv = if mean_ghost > 1e-30 { var_ghost.sqrt() / mean_ghost } else { 0.0 };

    let local_diag_nnz = a.diag.values.len() as i64;
    let local_offd_nnz = a.offd.values.len() as i64;
    let global_diag_nnz = comm.allreduce_sum_i64(local_diag_nnz).max(0) as usize;
    let global_offd_nnz = comm.allreduce_sum_i64(local_offd_nnz).max(0) as usize;

    RasHpcDiagnostics {
        n_ranks,
        global_owned_dofs: sum_owned.max(0.0) as usize,
        global_ghost_dofs: sum_ghost.max(0.0) as usize,
        global_diag_nnz,
        global_offd_nnz,
        mean_owned_dofs_per_rank: mean_owned,
        mean_ghost_dofs_per_rank: mean_ghost,
        owned_dofs_cv: owned_cv,
        ghost_dofs_cv: ghost_cv,
    }
}

/// Configuration for the RAS preconditioner.
#[derive(Debug, Clone)]
pub struct RasConfig {
    /// Overlap depth between subdomains.
    ///
    /// Current MVP accepts `0` only; non-zero overlap is planned.
    pub overlap: usize,
    /// Damping factor for preconditioner application.
    pub omega: f64,
    /// Local subdomain solve strategy.
    pub local_solver: RasLocalSolverKind,
}

impl Default for RasConfig {
    fn default() -> Self {
        Self {
            overlap: 0,
            omega: 1.0,
            local_solver: RasLocalSolverKind::DiagJacobi,
        }
    }
}

/// RAS preconditioner object reused across iterations.
pub struct RasPrecond {
    kernel: RasKernel,
    omega: f64,
    overlap: usize,
    overlap_mat: Option<ParCsrMatrix>,
}

enum RasKernel {
    DiagJacobi { inv_diag: Vec<f64> },
    Ilu0 { ilu: Ilu0Precond<f64> },
}

impl std::fmt::Debug for RasPrecond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RasPrecond")
            .field("kernel", &self.kernel)
            .field("omega", &self.omega)
            .field("overlap", &self.overlap)
            .finish()
    }
}

impl std::fmt::Debug for RasKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RasKernel::DiagJacobi { inv_diag } => f
                .debug_struct("DiagJacobi")
                .field("n_diag", &inv_diag.len())
                .finish(),
            RasKernel::Ilu0 { .. } => f.write_str("Ilu0"),
        }
    }
}

impl RasPrecond {
    /// Build a reusable RAS preconditioner from a distributed matrix.
    pub fn build(a: &ParCsrMatrix, cfg: &RasConfig) -> Result<Self, SolverError> {
        if cfg.overlap > 1 {
            return Err(SolverError::Linger(
                format!("RAS currently supports overlap in {{0,1}}, got {}", cfg.overlap),
            ));
        }
        if !cfg.omega.is_finite() || cfg.omega <= 0.0 {
            return Err(SolverError::Linger(
                format!("RAS omega must be finite and > 0, got {}", cfg.omega),
            ));
        }

        let kernel = match cfg.local_solver {
            RasLocalSolverKind::DiagJacobi => {
                let inv_diag = a
                    .diagonal()
                    .into_iter()
                    .map(|d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
                    .collect();
                RasKernel::DiagJacobi { inv_diag }
            }
            RasLocalSolverKind::Ilu0 => {
                let local = fem_solver::fem_to_linger_csr(&a.diag);
                let ilu = Ilu0Precond::from_csr(&local).map_err(SolverError::from)?;
                RasKernel::Ilu0 { ilu }
            }
        };

        let overlap_mat = if cfg.overlap == 1 {
            Some(clone_par_csr(a))
        } else {
            None
        };

        Ok(Self {
            kernel,
            omega: cfg.omega,
            overlap: cfg.overlap,
            overlap_mat,
        })
    }

    /// Apply one RAS preconditioner step: `z = M^{-1} r`.
    pub fn apply(&self, r: &ParVector, z: &mut ParVector) {
        // Base local solve.
        self.apply_local_core(r, z);

        // overlap=1: multiplicative overlap correction using the same local
        // kernel on the residual after one full local coupled SpMV.
        if self.overlap == 1 {
            if let Some(a) = &self.overlap_mat {
                let mut z_work = z.clone_vec();
                z_work.update_ghosts();

                let mut az = ParVector::zeros_like(r);
                a.spmv(&mut z_work, &mut az);

                let mut corr_rhs = r.clone_vec();
                corr_rhs.axpy(-1.0, &az);

                let mut dz = ParVector::zeros_like(r);
                self.apply_local_core(&corr_rhs, &mut dz);
                z.axpy(1.0, &dz);
            }
        }
    }

    fn apply_local_core(&self, r: &ParVector, z: &mut ParVector) {
        let n = r.n_owned();
        match &self.kernel {
            RasKernel::DiagJacobi { inv_diag } => {
                for i in 0..n {
                    z.as_slice_mut()[i] = self.omega * inv_diag[i] * r.as_slice()[i];
                }
            }
            RasKernel::Ilu0 { ilu } => {
                let x_owned = DenseVec::from_vec(r.as_slice()[..n].to_vec());
                let mut y_owned = DenseVec::zeros(n);
                ilu.apply_precond(&x_owned, &mut y_owned);
                for i in 0..n {
                    z.as_slice_mut()[i] = self.omega * y_owned.as_slice()[i];
                }
            }
        }
        for zi in &mut z.as_slice_mut()[n..] {
            *zi = 0.0;
        }
    }
}

fn clone_par_csr(a: &ParCsrMatrix) -> ParCsrMatrix {
    ParCsrMatrix::from_blocks(
        a.diag.clone(),
        a.offd.clone(),
        a.n_owned,
        a.n_ghost,
        a.ghost_exchange_arc(),
        a.comm().clone(),
    )
}

/// Parallel PCG with RAS preconditioning.
///
/// Current MVP uses overlap=0 + diagonal local solve.  The API is kept stable
/// for follow-up overlap/local-kernel upgrades.
pub fn par_solve_pcg_ras(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    ras_cfg: &RasConfig,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;
    let precond = RasPrecond::build(a, ras_cfg)?;

    if cfg.verbose && x.comm().is_root() {
        let d = summarize_ras_hpc(a);
        log::info!(
            "par_pcg_ras diag: ranks={}, owned={}, ghost={}, nnz(diag/offd)={}/{}, cv(owned/ghost)={:.3}/{:.3}",
            d.n_ranks,
            d.global_owned_dofs,
            d.global_ghost_dofs,
            d.global_diag_nnz,
            d.global_offd_nnz,
            d.owned_dofs_cv,
            d.ghost_dofs_cv,
        );
    }

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    for i in 0..n {
        r.as_slice_mut()[i] = b.as_slice()[i] - ax.as_slice()[i];
    }

    // z = M^{-1} r
    let mut z = ParVector::zeros_like(b);
    precond.apply(&r, &mut z);

    let mut p = z.clone_vec();
    let mut rz = r.global_dot(&z);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..cfg.max_iter {
        a.spmv(&mut p, &mut ap);

        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / pap;

        x.axpy(alpha, &p);
        r.axpy(-alpha, &ap);

        let rr = r.global_dot(&r);
        let res_norm = rr.sqrt() / b_norm;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_pcg_ras iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < cfg.rtol || rr.sqrt() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        precond.apply(&r, &mut z);
        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;

        for i in 0..p.as_slice().len() {
            p.as_slice_mut()[i] = z.as_slice()[i] + beta * p.as_slice()[i];
        }
        rz = rz_new;
    }

    let final_res = r.global_dot(&r).sqrt() / b_norm;
    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: final_res,
    })
}

/// Parallel restarted GMRES with right RAS preconditioning.
///
/// This path targets general (possibly non-symmetric) systems and uses the
/// distributed dot/norm operations on [`ParVector`].
pub fn par_solve_gmres_ras(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    ras_cfg: &RasConfig,
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    if restart == 0 {
        return Err(SolverError::Linger("GMRES restart must be > 0".to_string()));
    }

    let precond = RasPrecond::build(a, ras_cfg)?;

    if cfg.verbose && x.comm().is_root() {
        let d = summarize_ras_hpc(a);
        log::info!(
            "par_gmres_ras diag: ranks={}, owned={}, ghost={}, nnz(diag/offd)={}/{}, cv(owned/ghost)={:.3}/{:.3}",
            d.n_ranks,
            d.global_owned_dofs,
            d.global_ghost_dofs,
            d.global_diag_nnz,
            d.global_offd_nnz,
            d.owned_dofs_cv,
            d.ghost_dofs_cv,
        );
    }
    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    // r = b - A*x
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    let mut r = b.clone_vec();
    r.axpy(-1.0, &ax);

    let mut iter_total = 0usize;
    let mut rel_res = r.global_norm() / b_norm;
    if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: rel_res,
        });
    }

    while iter_total < cfg.max_iter {
        let beta = r.global_norm();
        if beta < 1e-30 {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: 0.0,
            });
        }

        let mut v: Vec<ParVector> = (0..=restart).map(|_| ParVector::zeros_like(b)).collect();
        v[0].copy_from(&r);
        v[0].scale(1.0 / beta);

        let mut z_basis: Vec<ParVector> = (0..restart).map(|_| ParVector::zeros_like(b)).collect();
        let mut h = vec![vec![0.0_f64; restart]; restart + 1];
        let mut cs = vec![0.0_f64; restart];
        let mut sn = vec![0.0_f64; restart];
        let mut g = vec![0.0_f64; restart + 1];
        g[0] = beta;

        let mut inner_done = 0usize;
        let mut converged = false;

        for j in 0..restart {
            if iter_total >= cfg.max_iter {
                break;
            }

            // Right preconditioning: z_j = M^{-1} v_j
            precond.apply(&v[j], &mut z_basis[j]);

            // Arnoldi operator application: w = A z_j
            let mut w = ParVector::zeros_like(b);
            a.spmv(&mut z_basis[j], &mut w);

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[i][j] = v[i].global_dot(&w);
                w.axpy(-h[i][j], &v[i]);
            }

            h[j + 1][j] = w.global_norm();
            if h[j + 1][j] > 1e-30 {
                v[j + 1].copy_from(&w);
                v[j + 1].scale(1.0 / h[j + 1][j]);
            }

            // Apply previous Givens rotations.
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = tmp;
            }

            // New Givens rotation.
            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-30 {
                cs[j] = h[j][j] / denom;
                sn[j] = h[j + 1][j] / denom;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }

            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            h[j + 1][j] = 0.0;

            let g_next = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            g[j + 1] = g_next;

            iter_total += 1;
            inner_done = j + 1;
            rel_res = g[j + 1].abs() / b_norm;

            if cfg.verbose && x.comm().is_root() {
                log::info!("par_gmres_ras iter {}: residual = {:.3e}", iter_total, rel_res);
            }

            if rel_res < cfg.rtol || g[j + 1].abs() < cfg.atol {
                converged = true;
                break;
            }
        }

        if inner_done == 0 {
            break;
        }

        // Back-substitution: H(0..m,0..m) y = g(0..m)
        let m = inner_done;
        let mut y = vec![0.0_f64; m];
        for i in (0..m).rev() {
            let mut s = g[i];
            for k in (i + 1)..m {
                s -= h[i][k] * y[k];
            }
            let diag = h[i][i];
            if diag.abs() < 1e-30 {
                return Err(SolverError::Linger(
                    "par_gmres_ras breakdown: near-singular Hessenberg diagonal".to_string(),
                ));
            }
            y[i] = s / diag;
        }

        // Right-preconditioned update: x += Z_m y
        for i in 0..m {
            x.axpy(y[i], &z_basis[i]);
        }

        if converged {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: rel_res,
            });
        }

        // Restart residual.
        a.spmv(x, &mut ax);
        r.copy_from(b);
        r.axpy(-1.0, &ax);
        rel_res = r.global_norm() / b_norm;
        if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: rel_res,
            });
        }
    }

    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: rel_res,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_simplex::partition_simplex;
    use crate::par_solver::par_solve_pcg_jacobi;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_solver::SolverConfig;
    use fem_space::constraints::boundary_dofs;
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    #[test]
    fn ras_hpc_diag_serial_is_consistent() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let d = summarize_ras_hpc(&a_mat);

            assert_eq!(d.n_ranks, 1);
            assert!(d.global_owned_dofs > 0);
            assert_eq!(d.owned_dofs_cv, 0.0);
            assert!(d.global_diag_nnz > 0);
        });
    }

    #[test]
    fn ras_hpc_diag_two_ranks_is_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let d = summarize_ras_hpc(&a_mat);

            assert_eq!(d.n_ranks, 2);
            assert!(d.global_owned_dofs > 0);
            assert!(d.global_diag_nnz > 0);
            assert!(d.owned_dofs_cv.is_finite());
            assert!(d.ghost_dofs_cv.is_finite());
        });
    }

    #[test]
    fn par_pcg_ras_overlap_one_builds_and_gt_one_rejects() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let ok_cfg = RasConfig { overlap: 1, ..RasConfig::default() };
            let ok = RasPrecond::build(&a_mat, &ok_cfg);
            assert!(ok.is_ok(), "expected overlap=1 to be accepted");

            let bad_cfg = RasConfig { overlap: 2, ..RasConfig::default() };
            let err = RasPrecond::build(&a_mat, &bad_cfg).unwrap_err();
            assert!(err.to_string().contains("{0,1}"));
        });
    }

    #[test]
    fn par_pcg_ras_matches_jacobi_path_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u_ras = crate::par_vector::ParVector::zeros(&par_space);
            let mut u_jac = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };

            let ras_res = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u_ras,
                &RasConfig::default(),
                &s_cfg,
            )
            .unwrap();

            let jac_res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u_jac, &s_cfg).unwrap();

            assert!(ras_res.converged);
            assert!(jac_res.converged);
            assert!((ras_res.final_residual - jac_res.final_residual).abs() < 1e-10);
        });
    }

    #[test]
    fn par_pcg_ras_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig {
                rtol: 1e-8,
                max_iter: 2000,
                ..SolverConfig::default()
            };

            let res = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u,
                &RasConfig::default(),
                &s_cfg,
            )
            .unwrap();

            assert!(
                res.converged,
                "rank {}: RAS-PCG did not converge: {} iters, res={:.3e}",
                comm.rank(),
                res.iterations,
                res.final_residual
            );
        });
    }

    #[test]
    fn par_pcg_ras_not_worse_than_jacobi_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u_ras = crate::par_vector::ParVector::zeros(&par_space);
            let mut u_jac = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig {
                rtol: 1e-8,
                max_iter: 2000,
                ..SolverConfig::default()
            };

            let ras_res = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u_ras,
                &RasConfig::default(),
                &s_cfg,
            )
            .unwrap();

            let jac_res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u_jac, &s_cfg).unwrap();

            assert!(ras_res.converged);
            assert!(jac_res.converged);
            assert!(
                ras_res.iterations <= jac_res.iterations,
                "rank {}: expected RAS-PCG iters <= Jacobi-PCG iters, got {} > {}",
                comm.rank(),
                ras_res.iterations,
                jac_res.iterations
            );
        });
    }

    #[test]
    fn par_pcg_ras_ilu0_converges_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1000, ..SolverConfig::default() };
            let ras_cfg = RasConfig {
                local_solver: RasLocalSolverKind::Ilu0,
                ..RasConfig::default()
            };

            let res = par_solve_pcg_ras(&a_mat, &rhs, &mut u, &ras_cfg, &s_cfg).unwrap();
            assert!(res.converged, "RAS-PCG(ILU0) did not converge: {:?}", res);
        });
    }

    #[test]
    fn par_pcg_ras_ilu0_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1000, ..SolverConfig::default() };
            let ras_cfg = RasConfig {
                local_solver: RasLocalSolverKind::Ilu0,
                ..RasConfig::default()
            };

            let res = par_solve_pcg_ras(&a_mat, &rhs, &mut u, &ras_cfg, &s_cfg).unwrap();
            assert!(
                res.converged,
                "rank {}: RAS-PCG(ILU0) did not converge: {:?}",
                comm.rank(),
                res
            );
        });
    }

    #[test]
    fn par_pcg_ras_ilu0_and_diag_are_both_stable_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u_diag = crate::par_vector::ParVector::zeros(&par_space);
            let mut u_ilu = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1000, ..SolverConfig::default() };

            let res_diag = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u_diag,
                &RasConfig {
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
                &s_cfg,
            )
            .unwrap();

            let res_ilu = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u_ilu,
                &RasConfig {
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
                &s_cfg,
            )
            .unwrap();

            assert!(res_diag.converged);
            assert!(res_ilu.converged);
            assert!(
                res_diag.final_residual <= 1e-7,
                "rank {}: Diag-RAS residual too high: {:.3e}",
                comm.rank(),
                res_diag.final_residual
            );
            assert!(
                res_ilu.final_residual <= 1e-7,
                "rank {}: ILU0-RAS residual too high: {:.3e}",
                comm.rank(),
                res_ilu.final_residual
            );
        });
    }

    #[test]
    fn par_gmres_ras_converges_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1000, ..SolverConfig::default() };
            let res = par_solve_gmres_ras(
                &a_mat,
                &rhs,
                &mut u,
                &RasConfig::default(),
                30,
                &s_cfg,
            )
            .unwrap();

            assert!(res.converged, "GMRES-RAS did not converge: {:?}", res);
        });
    }

    #[test]
    fn par_gmres_ras_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1000, ..SolverConfig::default() };
            let res = par_solve_gmres_ras(
                &a_mat,
                &rhs,
                &mut u,
                &RasConfig::default(),
                30,
                &s_cfg,
            )
            .unwrap();

            assert!(
                res.converged,
                "rank {}: GMRES-RAS did not converge: {:?}",
                comm.rank(),
                res
            );
            assert!(
                res.final_residual <= 1e-7,
                "rank {}: GMRES-RAS residual too high: {:.3e}",
                comm.rank(),
                res.final_residual
            );
        });
    }

    #[test]
    fn par_pcg_ras_overlap_one_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..SolverConfig::default() };
            let ras_cfg = RasConfig { overlap: 1, ..RasConfig::default() };

            let res = par_solve_pcg_ras(&a_mat, &rhs, &mut u, &ras_cfg, &s_cfg).unwrap();
            assert!(
                res.converged,
                "rank {}: overlap=1 RAS-PCG did not converge: {:?}",
                comm.rank(),
                res
            );
        });
    }

    #[test]
    fn par_gmres_ras_overlap_one_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1200, ..SolverConfig::default() };
            let ras_cfg = RasConfig { overlap: 1, ..RasConfig::default() };

            let res = par_solve_gmres_ras(&a_mat, &rhs, &mut u, &ras_cfg, 30, &s_cfg).unwrap();
            assert!(
                res.converged,
                "rank {}: overlap=1 GMRES-RAS did not converge: {:?}",
                comm.rank(),
                res
            );
            assert!(
                res.final_residual <= 1e-7,
                "rank {}: overlap=1 GMRES-RAS residual too high: {:.3e}",
                comm.rank(),
                res.final_residual
            );
        });
    }

    #[test]
    fn par_pcg_ras_overlap_zero_and_one_are_stable_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u0 = crate::par_vector::ParVector::zeros(&par_space);
            let mut u1 = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..SolverConfig::default() };

            let res0 = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u0,
                &RasConfig { overlap: 0, ..RasConfig::default() },
                &s_cfg,
            )
            .unwrap();

            let res1 = par_solve_pcg_ras(
                &a_mat,
                &rhs,
                &mut u1,
                &RasConfig { overlap: 1, ..RasConfig::default() },
                &s_cfg,
            )
            .unwrap();

            assert!(res0.converged, "rank {}: overlap=0 RAS failed: {:?}", comm.rank(), res0);
            assert!(res1.converged, "rank {}: overlap=1 RAS failed: {:?}", comm.rank(), res1);
            assert!(res0.final_residual <= 1e-7);
            assert!(res1.final_residual <= 1e-7);
        });
    }

    #[test]
    fn par_gmres_ras_ilu0_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1200, ..SolverConfig::default() };
            let ras_cfg = RasConfig {
                overlap: 0,
                local_solver: RasLocalSolverKind::Ilu0,
                ..RasConfig::default()
            };

            let res = par_solve_gmres_ras(&a_mat, &rhs, &mut u, &ras_cfg, 30, &s_cfg).unwrap();
            assert!(
                res.converged,
                "rank {}: GMRES-RAS(ILU0,ov0) did not converge: {:?}",
                comm.rank(),
                res
            );
        });
    }

    #[test]
    fn par_gmres_ras_ilu0_overlap_one_converges_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));

        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = crate::par_vector::ParVector::zeros(&par_space);
            let s_cfg = SolverConfig { rtol: 1e-8, max_iter: 1200, ..SolverConfig::default() };
            let ras_cfg = RasConfig {
                overlap: 1,
                local_solver: RasLocalSolverKind::Ilu0,
                ..RasConfig::default()
            };

            let res = par_solve_gmres_ras(&a_mat, &rhs, &mut u, &ras_cfg, 30, &s_cfg).unwrap();
            assert!(
                res.converged,
                "rank {}: GMRES-RAS(ILU0,ov1) did not converge: {:?}",
                comm.rank(),
                res
            );
            assert!(res.final_residual <= 1e-7);
        });
    }
}
