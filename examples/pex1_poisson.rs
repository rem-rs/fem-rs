//! # Parallel Example 1 -- Parallel Poisson (analogous to MFEM pex1)
//!
//! Solves -Laplacian(u) = f on [0,1]^2 in parallel using ThreadLauncher,
//! where f = 2pi^2 sin(pi*x) sin(pi*y) so that the exact solution is
//! u(x,y) = sin(pi*x) sin(pi*y).
//!
//! Usage:
//!   cargo run --example pex1_poisson            # P1 (default)
//!   cargo run --example pex1_poisson -- --p2    # P2

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
use fem_space::dof_manager::DofManager;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let use_p2 = args.iter().any(|a| a == "--p2");
    let order: u8 = if use_p2 { 2 } else { 1 };

    let n_workers = 2;
    let mesh_n = 16;

    println!("=== fem-rs pex1: Parallel Poisson (P{order}) ===");
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

        // 2. Build parallel FE space.
        let local_mesh = par_mesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, order);
        let local_space = H1Space::new(local_mesh, order);
        let par_space = if order >= 2 {
            ParallelFESpace::new_with_dof_manager(local_space, &par_mesh, &dm, comm.clone())
        } else {
            ParallelFESpace::new(local_space, &par_mesh, comm.clone())
        };

        if rank == 0 {
            println!("  Global DOFs: {}", par_space.n_global_dofs());
        }

        // 3. Parallel assembly.
        let quad_order = if order >= 2 { 4 } else { 3 };
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], quad_order);

        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let rhs_quad = if order >= 2 { 5 } else { 3 };
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], rhs_quad);

        // 4. Apply Dirichlet BCs: u=0 on all boundary faces.
        let bc_dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(par_space.local_space().mesh(), bc_dm, &[1, 2, 3, 4]);
        let dof_part = par_space.dof_partition();
        for &d in &bc_dofs {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs);
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
        let n_owned = dof_part.n_owned_dofs;
        let err_dm = par_space.local_space().dof_manager();
        let mut local_err_sq = 0.0_f64;
        for pid in 0..n_owned {
            // Map partition local ID back to DofManager ID for coordinate lookup.
            let dm_id = dof_part.unpermute_dof(pid as u32);
            let coord = err_dm.dof_coord(dm_id);
            let exact = (PI * coord[0]).sin() * (PI * coord[1]).sin();
            let diff_val = u.as_slice()[pid] - exact;
            local_err_sq += diff_val * diff_val;
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
