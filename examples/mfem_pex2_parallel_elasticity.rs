//! # Parallel Example 2 — Parallel Poisson with Robin BC
//! (analogous to MFEM pex2 style)
//!
//! Usage:
//!   cargo run --example mfem_pex2_parallel_elasticity -- --n 8 --ranks 2

use std::sync::{Arc, Mutex};

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::Mesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace,
    par_partition::partition_mesh,
    launcher::native::ThreadLauncher, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_workers: usize = args.iter().position(|a| a == "--ranks").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2);

    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let result = Arc::new(Mutex::new(Vec::new()));
    let result_slot = Arc::clone(&result);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let order: u8 = 1;
        let local_space = H1Space::new(local_mesh, order);
        let par_space = ParallelFESpace::new(local_space, &par_mesh, comm.clone());

        let quad_order: u8 = 3;
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], quad_order);
        let source = DomainSourceIntegrator::new(|x: &[f64]| 2.0 * (x[0] + x[1]));
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], quad_order);

        let bc_dofs = boundary_dofs(par_space.local_space().mesh(), par_space.local_space().dof_manager(), &[1, 2, 3, 4]);
        let dof_part = par_space.dof_partition();
        for &d in &bc_dofs {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs { a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs); }
        }

        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let res = fem_parallel::par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();
        let mut slot = result_slot.lock().unwrap();
        slot.push((comm.rank(), par_space.n_local_dofs(), res.iterations, res.final_residual));
    });

    let results = result.lock().unwrap();
    let global_dofs: usize = results.iter().map(|r| r.1).sum();
    println!("=== Parallel Poisson (linear RHS) ===");
    println!("  Ranks: {n_workers}, global DOFs: {global_dofs}");
    for (rank, n, iters, res) in results.iter() {
        println!("  Rank {rank}: {n} DOFs, iters={iters}, residual={res:.3e}");
    }
}
