//! Parallel iterative solvers.
//!
//! Provides parallel Conjugate Gradient (CG) and Jacobi-preconditioned CG
//! (PCG) that operate on [`ParCsrMatrix`] and [`ParVector`].

use fem_solver::{SolveResult, SolverConfig, SolverError};

use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

/// Parallel Conjugate Gradient solver for SPD systems.
///
/// Solves `A x = b` where `A` is a distributed SPD matrix.  All inner
/// products are global reductions over owned DOFs.
pub fn par_solve_cg(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    for i in 0..n {
        r.data[i] = b.data[i] - ax.data[i];
    }

    let mut p = r.clone_vec();
    let mut rr = r.global_dot(&r);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..cfg.max_iter {
        // ap = A * p
        a.spmv(&mut p, &mut ap);

        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rr / pap;

        // x += alpha * p
        x.axpy(alpha, &p);
        // r -= alpha * ap
        r.axpy(-alpha, &ap);

        let rr_new = r.global_dot(&r);
        let res_norm = rr_new.sqrt() / b_norm;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_cg iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < cfg.rtol || rr_new.sqrt() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        let beta = rr_new / rr;
        // p = r + beta * p
        for i in 0..p.data.len() {
            p.data[i] = r.data[i] + beta * p.data[i];
        }
        rr = rr_new;
    }

    let final_res = rr.sqrt() / b_norm;
    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: final_res,
    })
}

/// Parallel Jacobi-preconditioned Conjugate Gradient.
///
/// Uses diagonal scaling `M = diag(A)` as the preconditioner.
pub fn par_solve_pcg_jacobi(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;

    // Build inverse diagonal preconditioner.
    let diag = a.diagonal();
    let inv_diag: Vec<f64> = diag.iter()
        .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
        .collect();

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    for i in 0..n {
        r.data[i] = b.data[i] - ax.data[i];
    }

    // z = M^{-1} r
    let mut z = ParVector::zeros_like(b);
    for i in 0..n {
        z.data[i] = inv_diag[i] * r.data[i];
    }

    let mut p = z.clone_vec();
    let mut rz = r.global_dot(&z);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..cfg.max_iter {
        // ap = A * p
        a.spmv(&mut p, &mut ap);

        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap;

        // x += alpha * p
        x.axpy(alpha, &p);
        // r -= alpha * ap
        r.axpy(-alpha, &ap);

        let rr = r.global_dot(&r);
        let res_norm = rr.sqrt() / b_norm;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_pcg_jacobi iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < cfg.rtol || rr.sqrt() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        // z = M^{-1} r
        for i in 0..n {
            z.data[i] = inv_diag[i] * r.data[i];
        }

        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;

        // p = z + beta * p
        for i in 0..p.data.len() {
            p.data[i] = z.data[i] + beta * p.data[i];
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


/// Parallel MINRES solver for symmetric (possibly indefinite) systems.
///
/// Solves `A x = b` where `A` is a distributed symmetric matrix.
/// Uses the Lanczos-based MINRES algorithm (Choi-Paige-Saunders).
pub fn par_solve_minres(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;
    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(&mut x.clone_vec(), &mut ax);
    for i in 0..n { r.data[i] = b.data[i] - ax.data[i]; }

    let mut beta1 = r.global_norm();
    if beta1 / b_norm < cfg.rtol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: beta1 / b_norm });
    }

    // Lanczos vectors
    let mut v_old = ParVector::zeros_like(b);
    let mut v_cur = r.clone_vec();
    for i in 0..v_cur.len() { v_cur.data[i] /= beta1; }
    let mut v_new = ParVector::zeros_like(b);

    // MINRES recurrence scalars
    let mut _beta_prev = 0.0_f64;
    let mut beta_cur = beta1;
    let mut c_old = 1.0_f64;
    let mut c_cur = 1.0_f64;
    let mut s_old = 0.0_f64;
    let mut s_cur = 0.0_f64;

    // Direction vectors
    let mut w_prev = ParVector::zeros_like(b);
    let mut w_cur = ParVector::zeros_like(b);

    let mut res_norm = beta1 / b_norm;

    for iter in 0..cfg.max_iter {
        // Lanczos step: v_new = A*v_cur - beta_cur * v_old
        a.spmv(&mut v_cur, &mut v_new);
        let alpha = v_cur.global_dot(&v_new);
        for i in 0..n {
            v_new.data[i] -= alpha * v_cur.data[i] + beta_cur * v_old.data[i];
        }
        let beta_next = v_new.global_norm();

        // Apply previous Givens rotations
        let r1 = s_old * beta_cur;
        let r2 = c_old * c_cur * beta_cur + s_cur * alpha;
        let r3 = -s_old * s_cur * beta_cur + c_cur * alpha;

        // this step's value before new rotation
        let r3_hat = r3;
        let r4 = beta_next;

        // New Givens rotation to zero out beta_next
        let gamma = (r3_hat * r3_hat + r4 * r4).sqrt();
        let c_new = if gamma > 1e-30 { r3_hat / gamma } else { 1.0 };
        let s_new = if gamma > 1e-30 { r4 / gamma } else { 0.0 };

        // Update direction vectors
        let mut w_new = ParVector::zeros_like(b);
        if gamma.abs() > 1e-30 {
            for i in 0..n {
                w_new.data[i] = (v_cur.data[i] - r1 * w_prev.data[i] - r2 * w_cur.data[i]) / gamma;
            }
        }

        // Update solution: x += c_new * beta1 * ... * w_new
        // In MINRES, the update is: x += (c_new * phi) * w_new
        // where phi tracks the residual components
        let _phi = c_new * res_norm * b_norm;
        // Actually, simplified MINRES update:
        let tau = c_new * beta1;
        for i in 0..n { x.data[i] += tau * w_new.data[i]; }

        // Update residual norm
        res_norm = s_new.abs() * res_norm;
        beta1 = s_new * beta1;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_minres iter {}: residual = {:.3e}", iter + 1, res_norm / b_norm);
        }

        if res_norm / b_norm < cfg.rtol || res_norm < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm / b_norm,
            });
        }

        // Prepare for next iteration
        if beta_next.abs() > 1e-30 {
            for i in 0..v_new.len() { v_new.data[i] /= beta_next; }
        }

        // Shift vectors
        std::mem::swap(&mut v_old, &mut v_cur);
        std::mem::swap(&mut v_cur, &mut v_new);
        v_new = ParVector::zeros_like(b);

        std::mem::swap(&mut w_prev, &mut w_cur);
        w_cur = w_new;

        _beta_prev = beta_cur;
        beta_cur = beta_next;
        c_old = c_cur;
        c_cur = c_new;
        s_old = s_cur;
        s_cur = s_new;
    }

    // Compute true residual
    let mut true_r = b.clone_vec();
    let mut true_ax = ParVector::zeros_like(b);
    a.spmv(x, &mut true_ax);
    for i in 0..n { true_r.data[i] = b.data[i] - true_ax.data[i]; }
    let final_res = true_r.global_norm() / b_norm;

    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: final_res,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;
    use fem_space::constraints::boundary_dofs;

    #[test]
    fn par_cg_laplacian_serial() {
        // Single-rank parallel CG on a simple Poisson problem.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            // Apply Dirichlet BCs: u=0 on boundary.
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_cg(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged, "CG did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_pcg_jacobi_two_ranks() {
        // Two-rank parallel PCG on Poisson.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            // Apply Dirichlet BCs: u=0 on boundary.
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_pcg_jacobi_p2_two_ranks() {
        // Two-rank parallel PCG on Poisson with P2 elements.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = fem_space::dof_manager::DofManager::new(&local_mesh, 2);
            let local_space = H1Space::new(local_mesh, 2);
            let par_space = ParallelFESpace::new_with_dof_manager(
                local_space, &pmesh, &dm, comm.clone(),
            );

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 4);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 5);

            // Apply Dirichlet BCs: u=0 on boundary.
            // boundary_dofs returns DOF IDs in DofManager numbering.
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(),
                par_space.local_space().dof_manager(), &[1, 2, 3, 4]);
            let dof_part = par_space.dof_partition();
            for &d in &bc_dofs {
                let pid = dof_part.permute_dof(d) as usize;
                if pid < dof_part.n_owned_dofs {
                    a_mat.apply_dirichlet_row(pid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: P2 PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }
}
