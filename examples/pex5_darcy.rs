//! # Parallel Example 5 — H(div) Darcy flow  (analogous to MFEM pex5)
//!
//! Solves the H(div) grad-div problem in parallel:
//!
//! ```text
//!   −∇(α ∇·F) + β F = f    in Ω = [0,1]²
//!                F·n = 0    on ∂Ω
//! ```
//!
//! Uses lowest-order Raviart-Thomas (RT0) elements.
//!
//! ## Usage
//! ```
//! cargo run --example pex5_darcy
//! cargo run --example pex5_darcy -- --n 16 --ranks 4
//! ```

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::{
    standard::{GradDivIntegrator, VectorMassIntegrator, VectorDomainLFIntegrator},
    coefficient::FnVectorCoeff,
};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    ParVectorAssembler, ParVector, ParallelFESpace,
    par_simplex::partition_simplex,
    par_solve_pcg_jacobi,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{HDivSpace, fe_space::FESpace};

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let n_workers = parse_arg(&args, "--ranks").unwrap_or(2);
    let mesh_n = parse_arg(&args, "--n").unwrap_or(16);

    println!("=== fem-rs pex5: Parallel H(div) Darcy (RT0) ===");
    println!("  Workers: {n_workers}, Mesh: {mesh_n}x{mesh_n}");

    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(mesh_n));

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition mesh.
        let par_mesh = partition_simplex(&mesh, &comm);

        // 2. Build parallel H(div) space.
        let local_space = HDivSpace::new(par_mesh.local_mesh().clone(), 0);
        let par_space = ParallelFESpace::new_for_edge_space(
            local_space, &par_mesh, comm.clone(),
        );

        if rank == 0 {
            println!("  Global DOFs: {}", par_space.n_global_dofs());
        }

        // 3. Assemble α(∇·F,∇·G) + β(F,G).
        let alpha = 1.0;
        let beta = 1.0;
        let grad_div = GradDivIntegrator { kappa: alpha };
        let mass = VectorMassIntegrator { alpha: beta };
        let mut a_mat = ParVectorAssembler::assemble_bilinear(
            &par_space, &[&grad_div, &mass], 3,
        );

        // 4. Assemble RHS.
        // Manufactured: F = (sin(πx)cos(πy), −cos(πx)sin(πy)), ∇·F = 0
        // f = βF
        let source = VectorDomainLFIntegrator {
            f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                out[0] =  beta * (PI * x[0]).sin() * (PI * x[1]).cos();
                out[1] = -beta * (PI * x[0]).cos() * (PI * x[1]).sin();
            }),
        };
        let mut rhs = ParVectorAssembler::assemble_linear(&par_space, &[&source], 3);

        // 5. No essential BCs needed (natural F·n = 0 for this manufactured solution).
        // But we need to ensure the system is well-posed. For RT0 with grad-div + mass,
        // the system is SPD so PCG works directly.

        // 6. Solve with parallel PCG + Jacobi.
        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
        let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

        if rank == 0 {
            println!(
                "  PCG: {} iters, residual = {:.3e}, converged = {}",
                res.iterations, res.final_residual, res.converged
            );
            let h = 1.0 / mesh_n as f64;
            println!("  h = {h:.4e}");
            println!("=== Done ===");
        }
    });
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
