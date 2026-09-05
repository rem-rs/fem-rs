//!
//! Parallel Example 0 — Parallel Poisson (1:1 port of MFEM ex0p.cpp).
//!
//! Solves -Delta u = 1 with homogeneous Dirichlet BCs.
//!
//! Usage:
//!   cargo run --release --example mfem_pex0_parallel_poisson -- --ranks 2
//!   cargo run --release --example mfem_pex0_parallel_poisson -- --ranks 4 -o 2

use std::sync::Arc;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_parallel::{
    par_partition::partition_mesh,
    launcher::native::ThreadLauncher,
    ParAssembler, ParVector, ParallelFESpace, par_solve_pcg_amg,
    WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace};
use fem_space::constraints::boundary_dofs;
use fem_space::dof_manager::DofManager;
use fem_parallel::par_amg::ParAmgConfig;

struct Args {
    mesh_file: String,
    order: u8,
    ranks: usize,
}

fn parse_args() -> Args {
    let mut mesh_file = "data/star.mesh".to_string();
    let mut order = 1u8;
    let mut ranks = 2usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => mesh_file = it.next().unwrap_or(mesh_file),
            "-o" | "--order" => order = it.next().and_then(|s| s.parse().ok()).unwrap_or(order),
            "--ranks" => ranks = it.next().and_then(|s| s.parse().ok()).unwrap_or(ranks),
            _ => {}
        }
    }
    Args { mesh_file, order, ranks }
}

fn main() {
    let args = parse_args();

    // 1:1 with MFEM ex0p: read mesh, refine once
    let reader = read_mfem_file(&args.mesh_file).expect("Failed to read mesh");
    let mut mesh: Mesh<2> = reader.mesh2d.expect("Mesh must be 2D");
    mesh = refine_uniform(&mesh);

    let global_n = mesh.n_nodes();
    let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));

    launcher.launch(move |comm| {
        let rank = comm.rank();

        // Partition mesh
        let par_mesh = partition_mesh(&mesh, &comm);

        // Create H1 space
        let local_mesh = par_mesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, args.order);
        let local_space = H1Space::new(local_mesh, args.order);
        let par_space = if args.order >= 2 {
            ParallelFESpace::new_with_dof_manager(local_space, &par_mesh, &dm, comm.clone())
        } else {
            ParallelFESpace::new(local_space, &par_mesh, comm.clone())
        };

        // Assemble
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let source: fn(&[f64]) -> f64 = |_x| 1.0;
        let source = DomainSourceIntegrator::new(source);

        let quad_order = if args.order >= 2 { 4 } else { 3 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], quad_order);
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], quad_order);

        // Apply Dirichlet BCs
        let bc_dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(
            par_space.local_space().mesh(),
            bc_dm,
            &par_space.local_space().mesh().unique_boundary_tags(),
        );
        let dof_part = par_space.dof_partition();
        for &d in &bc_dofs {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs);
            }
        }

        // Solve with PCG + AMG (matching MFEM ex0p's BoomerAMG)
        let mut u = ParVector::zeros(&par_space);
        let solver_cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        let amg_cfg = ParAmgConfig::default();
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &solver_cfg).unwrap();

        if rank == 0 {
            println!("Number of unknowns: {}", par_space.n_global_dofs());
            println!("PCG: {} iters, residual = {:.3e}, converged = {}", res.iterations, res.final_residual, res.converged);
        }
    });
}
