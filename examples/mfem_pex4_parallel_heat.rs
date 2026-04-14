//! # Parallel Example 4 — Parallel implicit heat equation time-stepping
//!
//! Extends `mfem_ex10_heat_equation` to run in parallel using `ThreadLauncher`.
//!
//! Solves the time-dependent heat equation on a distributed mesh:
//!
//! ```text
//!   ∂u/∂t − κ Δu = 0    in Ω = [0,1]², t ∈ [0, T]
//!             u = 0    on ∂Ω
//!             u = u₀   at t = 0    (u₀ = sin(πx)sin(πy))
//! ```
//!
//! Spatial discretization: P1 H¹ elements (or P2 with `--p2`).
//! Time integration: Implicit Euler (unconditionally stable).
//!
//! At each step we solve the parallel system:
//! ```text
//!   (M + dt K) u^{n+1} = M u^n
//! ```
//! using parallel PCG + Jacobi.
//!
//! Exact solution: `u(x,y,t) = e^{−2π²κt} sin(πx)sin(πy)`.
//!
//! ## Usage
//! ```
//! cargo run --example pex4_parallel_heat
//! cargo run --example pex4_parallel_heat -- --ranks 4 --n 32 --dt 0.005 --T 0.2
//! cargo run --example pex4_parallel_heat -- --p2 --n 16 --T 0.1
//! ```

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    par_simplex::partition_simplex, par_solve_pcg_jacobi, ParAssembler, ParVector, ParallelFESpace,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::dof_manager::DofManager;
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space};

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let n_workers = parse_arg(&args, "--ranks").unwrap_or(2);
    let mesh_n = parse_arg(&args, "--n").unwrap_or(16);
    let use_p2 = args.iter().any(|a| a == "--p2");
    let dt = parse_f64(&args, "--dt").unwrap_or(0.01);
    let t_end = parse_f64(&args, "--T").unwrap_or(0.1);
    let kappa = parse_f64(&args, "--kappa").unwrap_or(1.0);
    let order: u8 = if use_p2 { 2 } else { 1 };

    println!("=== fem-rs pex4: Parallel implicit heat equation ===");
    println!("  Workers: {n_workers}, Mesh: {mesh_n}×{mesh_n}, P{order}");
    println!(
        "  κ = {kappa}, dt = {dt}, T = {t_end}, steps = {}",
        (t_end / dt).ceil() as usize
    );

    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(mesh_n));

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition mesh.
        let par_mesh = partition_simplex(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        // 2. Build parallel FE space.
        let space = H1Space::new(local_mesh, order);
        let par_space = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let n_global = par_space.n_global_dofs();
        if rank == 0 {
            println!("  Global DOFs: {n_global}");
        }

        let quad_order = order * 2 + 1;
        let dof_part = par_space.dof_partition();
        let bnd = boundary_dofs(par_space.local_space().mesh(), par_space.local_space().dof_manager(), &[1, 2, 3, 4]);

        // 3. Assemble M (mass matrix).
        let mut par_m = ParAssembler::assemble_bilinear(
            &par_space,
            &[&MassIntegrator { rho: 1.0 }],
            quad_order,
        );

        // 4. Assemble sys = M + dt*K  (Implicit Euler system matrix).
        //    Pass both M and dt*K integrators in a single assembly call,
        //    so the assembler accumulates them into one matrix.
        let mut par_sys = ParAssembler::assemble_bilinear(
            &par_space,
            &[
                &MassIntegrator { rho: 1.0 },
                &DiffusionIntegrator { kappa: dt * kappa },
            ],
            quad_order,
        );

        // 5. Apply Dirichlet BCs to both par_m (for rhs computation) and par_sys.
        for &d in &bnd {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                // Zero the boundary row in sys (enforce u=0 strongly).
                par_sys.apply_dirichlet_par(pid, 0.0, &mut ParVector::zeros(&par_space));
                // Also zero the boundary row in M so that M*u^n contributes
                // zero RHS at boundary DOFs (Dirichlet data is 0 here).
                par_m.apply_dirichlet_par(pid, 0.0, &mut ParVector::zeros(&par_space));
            }
        }

        // 6. Initial condition: u₀ = sin(πx)sin(πy) interpolated at owned DOFs.
        let mut u = {
            let n_local = dof_part.n_owned_dofs + dof_part.n_ghost_dofs;
            let dm = par_space.local_space().dof_manager();
            let mut data = vec![0.0_f64; n_local];
            for i in 0..dof_part.n_owned_dofs {
                // unpermute_dof maps partition-local owned index → dm-local DOF id
                let dm_dof = dof_part.unpermute_dof(i as u32);
                let x = dm.dof_coord(dm_dof);
                data[i] = (PI * x[0]).sin() * (PI * x[1]).sin();
            }
            // Zero boundary DOFs.
            for &d in &bnd {
                let pid = dof_part.permute_dof(d) as usize;
                if pid < dof_part.n_owned_dofs {
                    data[pid] = 0.0;
                }
            }
            ParVector::from_local(data, &par_space)
        };

        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 5_000,
            verbose: false,
            ..SolverConfig::default()
        };

        // 7. Time-stepping: Implicit Euler.
        //    (M + dt K) u^{n+1} = M u^n
        let n_steps = (t_end / dt).ceil() as usize;
        let mut t = 0.0_f64;

        for step in 0..n_steps {
            // Compute rhs = M * u^n
            let mut rhs = ParVector::zeros(&par_space);
            par_m.spmv(&mut u, &mut rhs);

            // Zero boundary entries in rhs (Dirichlet = 0 enforced).
            for &d in &bnd {
                let pid = dof_part.permute_dof(d) as usize;
                if pid < dof_part.n_owned_dofs {
                    rhs.owned_slice_mut()[pid] = 0.0;
                }
            }

            let mut u_new = ParVector::zeros(&par_space);
            let res = par_solve_pcg_jacobi(&par_sys, &rhs, &mut u_new, &cfg)
                .expect("parallel PCG failed");

            if rank == 0 && step % 10 == 0 {
                log::debug!(
                    "  Step {}/{}: iters={}, res={:.3e}",
                    step + 1,
                    n_steps,
                    res.iterations,
                    res.final_residual
                );
            }

            // Copy owned DOFs.
            u.owned_slice_mut().copy_from_slice(u_new.owned_slice());
            t += dt;
        }

        // 8. Compare to exact solution at owned DOFs.
        let decay = (-2.0 * PI * PI * kappa * t_end).exp();
        let dm = par_space.local_space().dof_manager();
        let local_err2: f64 = (0..dof_part.n_owned_dofs)
            .map(|i| {
                let dm_dof = dof_part.unpermute_dof(i as u32);
                let x = dm.dof_coord(dm_dof);
                let u_ex = decay * (PI * x[0]).sin() * (PI * x[1]).sin();
                (u.owned_slice()[i] - u_ex).powi(2)
            })
            .sum();
        let local_count = dof_part.n_owned_dofs as f64;

        // Global reduction.
        let total_err2 = comm.allreduce_sum_f64(local_err2);
        let total_n = comm.allreduce_sum_f64(local_count);
        let rms_err = (total_err2 / total_n).sqrt();

        if rank == 0 {
            println!("  t = {t:.4e},  nodal RMS error vs exact = {rms_err:.4e}");
            println!("  (exact decay factor = {decay:.6e})");
            println!("Done.");
        }
    });
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}

fn parse_f64(args: &[String], flag: &str) -> Option<f64> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}
