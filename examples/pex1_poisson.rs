//! # Parallel Example 1 -- Parallel Poisson (analogous to MFEM pex1)
//!
//! Solves -Laplacian(u) = f on [0,1]^2 in parallel using ThreadLauncher,
//! where f = 2pi^2 sin(pi*x) sin(pi*y) so that the exact solution is
//! u(x,y) = sin(pi*x) sin(pi*y).
//!
//! Usage:
//!   cargo run --example pex1_poisson

use std::f64::consts::PI;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace,
    par_solve_pcg_jacobi, partition_simplex,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace};
use fem_space::constraints::boundary_dofs;

fn main() {
    env_logger::init();

    let n_workers = 2;
    let mesh_n = 16;

    println!("=== fem-rs pex1: Parallel Poisson ===");
    println!("  Workers: {n_workers}, Mesh: {mesh_n}x{mesh_n}");

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        // 1. Build and partition mesh.
        let mesh = SimplexMesh::<2>::unit_square_tri(mesh_n);
        let par_mesh = partition_simplex(&mesh, &comm);

        let rank = comm.rank();
        if rank == 0 {
            println!("  Global nodes: {}, elements: {}",
                par_mesh.global_n_nodes(), par_mesh.global_n_elems());
        }

        // 2. Build parallel FE space (P1).
        let local_space = H1Space::new(par_mesh.local_mesh().clone(), 1);
        let par_space = ParallelFESpace::new(local_space, &par_mesh, comm.clone());

        if rank == 0 {
            println!("  Global DOFs: {}", par_space.n_global_dofs());
        }

        // 3. Parallel assembly.
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);

        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

        // 4. Apply Dirichlet BCs: u=0 on all boundary faces.
        let dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
        for &d in &bc_dofs {
            let lid = d as usize;
            if lid < par_space.dof_partition().n_owned_dofs {
                a_mat.apply_dirichlet_par(lid, 0.0, &mut rhs);
            }
        }

        // 5. Solve with parallel PCG (Jacobi preconditioner).
        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 5000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

        if rank == 0 {
            println!("  PCG: {} iters, residual = {:.3e}, converged = {}",
                res.iterations, res.final_residual, res.converged);
        }

        // 6. Compute L2 error against exact solution u_exact = sin(pi*x)*sin(pi*y).
        // Approximate: |u - u_exact|^2 at DOF points, sum over owned DOFs.
        let n_owned = par_space.dof_partition().n_owned_dofs;
        let dm = par_space.local_space().dof_manager();
        let mut local_err_sq = 0.0_f64;
        for lid in 0..n_owned {
            let coord = dm.dof_coord(lid as u32);
            let exact = (PI * coord[0]).sin() * (PI * coord[1]).sin();
            let diff = u.as_slice()[lid] - exact;
            local_err_sq += diff * diff;
        }
        let global_err_sq = comm.allreduce_sum_f64(local_err_sq);
        let n_global = par_space.n_global_dofs() as f64;
        let l2_err = (global_err_sq / n_global).sqrt();

        if rank == 0 {
            println!("  L2 error (pointwise): {:.6e}", l2_err);
            println!("=== Done ===");
        }
    });
}
