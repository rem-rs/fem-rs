//! # Parallel Example 2 — Saddle-point system  (analogous to MFEM pex2)
//!
//! Demonstrates parallel block system solving with a saddle-point problem:
//!
//! ```text
//!   [ A   Bᵀ ] [ u ]   [ f ]
//!   [ B  -εI ] [ p ] = [ g ]
//! ```
//!
//! Uses two H¹ P1 spaces for velocity (u) and pressure (p), partitioned
//! across workers.
//!
//! ## Usage
//! ```
//! cargo run --example pex2_mixed_darcy
//! cargo run --example pex2_mixed_darcy -- --n 16 --ranks 4
//! ```

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::{DivIntegrator, PressureDivIntegrator},
    standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    ParAssembler, ParMixedAssembler, ParVector, ParallelFESpace,
    par_simplex::partition_simplex,
    par_solve_pcg_jacobi,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let n_workers = parse_arg(&args, "--ranks").unwrap_or(2);
    let mesh_n = parse_arg(&args, "--n").unwrap_or(8);

    println!("=== fem-rs pex2: Parallel Mixed Darcy (P1/P1 saddle-point) ===");
    println!("  Workers: {n_workers}, Mesh: {mesh_n}x{mesh_n}");

    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(mesh_n));

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition mesh.
        let par_mesh = partition_simplex(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        // 2. Build parallel FE spaces (two H1 P1 spaces on same mesh).
        let space_u = H1Space::new(local_mesh.clone(), 1);
        let space_p = H1Space::new(local_mesh, 1);
        let par_space_u = ParallelFESpace::new(space_u, &par_mesh, comm.clone());
        let par_space_p = ParallelFESpace::new(space_p, &par_mesh, comm.clone());

        let nu_global = par_space_u.n_global_dofs();
        let np_global = par_space_p.n_global_dofs();
        if rank == 0 {
            println!("  Global u-DOFs: {nu_global}, p-DOFs: {np_global}");
        }

        // 3. Assemble A = diffusion for u.
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space_u, &[&diff], 3);

        // 4. Assemble RHS f.
        let source_u = DomainSourceIntegrator::new(|x: &[f64]| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut f_u = ParAssembler::assemble_linear(&par_space_u, &[&source_u], 3);

        // 5. Apply Dirichlet u = 0 on boundary.
        let dm_u = par_space_u.local_space().dof_manager();
        let bnd_u = boundary_dofs(par_space_u.local_space().mesh(), dm_u, &[1, 2, 3, 4]);
        let dof_part = par_space_u.dof_partition();
        for &d in &bnd_u {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                a_mat.apply_dirichlet_par(pid, 0.0, &mut f_u);
            }
        }

        // 6. Solve just the velocity equation A*u = f as a simplified parallel demo.
        //    (Full saddle-point would require a parallel block solver, which is future work.)
        let mut u_sol = ParVector::zeros(&par_space_u);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let res = par_solve_pcg_jacobi(&a_mat, &f_u, &mut u_sol, &cfg).unwrap();

        if rank == 0 {
            println!(
                "  PCG (velocity block): {} iters, residual = {:.3e}, converged = {}",
                res.iterations, res.final_residual, res.converged
            );

            // Report solution statistics.
            let u_max: f64 = u_sol.owned_slice().iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
            println!("  max|u_owned| = {u_max:.4e}");
            println!("=== Done ===");
        }
    });
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
