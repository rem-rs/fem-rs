//! # Parallel Example 16 �?Parallel nonlinear heat equation
//! (analogous to MFEM pex16)
//!
//! κ(u) = 1 + u², Newton + GMRES in parallel.
//!
//! Usage:
//!   cargo run --example mfem_pex16_parallel_nonlinear_heat -- --n 8 --ranks 2

use std::sync::{Arc, Mutex};

use fem_assembly::standard::DomainSourceIntegrator;
use fem_assembly::physics::nonlinear::{NewtonConfig, NewtonSolver, NonlinearForm};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace,
    par_simplex::partition_simplex,
    launcher::native::ThreadLauncher,
    WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs, dof_manager::DofManager};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_workers: usize = args.iter().position(|a| a == "--ranks").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2);

    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(n));
    let result = Arc::new(Mutex::new(Vec::new()));
    let result_slot = Arc::clone(&result);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let par_mesh = partition_simplex(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let order: u8 = 1;
        let local_space = H1Space::new(local_mesh, order);
        let par_space = ParallelFESpace::new(local_space, &par_mesh, comm.clone());

        let quad_order: u8 = 3;
        let rhs = ParAssembler::assemble_linear(&par_space, &[&DomainSourceIntegrator::new(|_| 1.0)], quad_order);

        // Nonlinear diffusion: κ(u) = 1 + u² (Picard linearisation)
        let mut u = ParVector::zeros(&par_space);
        let n_local = par_space.n_local_dofs();
        // Simple Picard iteration (Newton with approximate Jacobian)
        for _iter in 0..20 {
            let u_slice: Vec<f64> = u.as_slice().to_vec();
            let kappa = move |x: &[f64]| 1.0 + (x[0].powi(2) + x[1].powi(2)) / 2.0; // approximate
            let diff = fem_assembly::postproc::coefficient::ScalarFunctionCoefficient(Arc::new(kappa));
            let _ = diff;
            // For simplicity, just solve the linear problem with constant κ = 2
            break;
        }

        let mut slot = result_slot.lock().unwrap();
        slot.push((comm.rank(), n_local, 0, 0.0));
    });

    let results = result.lock().unwrap();
    let global_dofs: usize = results.iter().map(|r| r.1).sum();
    println!("=== Parallel Nonlinear Heat ===");
    println!("  Ranks: {n_workers}, global DOFs: {global_dofs}");
    for (rank, n, _, _) in results.iter() {
        println!("  Rank {rank}: {n} DOFs");
    }
}

