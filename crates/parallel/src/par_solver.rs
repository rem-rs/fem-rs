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
}
