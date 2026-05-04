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
//! cargo run --example mfem_pex4_parallel_heat
//! cargo run --example mfem_pex4_parallel_heat -- --ranks 4 --n 32 --dt 0.005 --T 0.2
//! cargo run --example mfem_pex4_parallel_heat -- --p2 --n 16 --T 0.1
//! ```

use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

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

    let args = parse_args();

    println!("=== fem-rs mfem_pex4: Parallel implicit heat equation ===");
    println!("  Workers: {}, Mesh: {}x{}, P{}", args.n_workers, args.mesh_n, args.mesh_n, args.order);
    println!(
        "  kappa = {}, dt = {}, T = {}, steps = {}",
        args.kappa,
        args.dt,
        args.t_end,
        (args.t_end / args.dt).ceil() as usize
    );

    let result = run_case(
        args.mesh_n,
        args.n_workers,
        args.order,
        args.dt,
        args.t_end,
        args.kappa,
        1.0,
    );

    println!("  Global DOFs: {}", result.global_dofs);
    println!("  t = {:.4e}, nodal RMS error vs exact = {:.4e}", result.final_time, result.rms_err);
    println!("  ||u(T)||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("  (exact decay factor = {:.6e})", result.decay);
    println!("Done.");
}

struct Args {
    n_workers: usize,
    mesh_n: usize,
    order: u8,
    dt: f64,
    t_end: f64,
    kappa: f64,
}

struct RunResult {
    global_dofs: usize,
    final_time: f64,
    rms_err: f64,
    solution_norm: f64,
    solution_checksum: f64,
    decay: f64,
}

fn run_case(
    mesh_n: usize,
    n_workers: usize,
    order: u8,
    dt: f64,
    t_end: f64,
    kappa: f64,
    initial_scale: f64,
) -> RunResult {
    let result = Arc::new(Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);

    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(mesh_n));

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition mesh.
        let par_mesh = partition_simplex(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        // 2. Build parallel FE space.
        let dof_manager = if order > 1 {
            Some(DofManager::new(&local_mesh, order))
        } else {
            None
        };
        let space = H1Space::new(local_mesh, order);
        let par_space = if let Some(ref dm) = dof_manager {
            ParallelFESpace::new_with_dof_manager(space, &par_mesh, dm, comm.clone())
        } else {
            ParallelFESpace::new(space, &par_mesh, comm.clone())
        };

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
                data[i] = initial_scale * (PI * x[0]).sin() * (PI * x[1]).sin();
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
        let decay = (-2.0 * PI * PI * kappa * t).exp();
        let dm = par_space.local_space().dof_manager();
        let local_err2: f64 = (0..dof_part.n_owned_dofs)
            .map(|i| {
                let dm_dof = dof_part.unpermute_dof(i as u32);
                let x = dm.dof_coord(dm_dof);
                let u_ex = initial_scale * decay * (PI * x[0]).sin() * (PI * x[1]).sin();
                (u.owned_slice()[i] - u_ex).powi(2)
            })
            .sum();
        let local_count = dof_part.n_owned_dofs as f64;
        let local_checksum: f64 = u.owned_slice()
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let gid = dof_part.global_dof(i as u32) as f64 + 1.0;
                gid * value
            })
            .sum();

        // Global reduction.
        let total_err2 = comm.allreduce_sum_f64(local_err2);
        let total_n = comm.allreduce_sum_f64(local_count);
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);
        let solution_norm = u.global_norm();
        let rms_err = (total_err2 / total_n).sqrt();

        if rank == 0 {
            *result_slot.lock().expect("mfem_pex4 result mutex poisoned") = Some(RunResult {
                global_dofs: par_space.n_global_dofs(),
                final_time: t,
                rms_err,
                solution_norm,
                solution_checksum,
                decay,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("mfem_pex4 result mutex poisoned after launch")
        .take()
        .expect("rank 0 did not publish mfem_pex4 result");
    final_result
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let use_p2 = args.iter().any(|a| a == "--p2");
    Args {
        n_workers: parse_arg(&args, "--ranks").unwrap_or(2),
        mesh_n: parse_arg(&args, "--n").unwrap_or(16),
        order: if use_p2 { 2 } else { 1 },
        dt: parse_f64(&args, "--dt").unwrap_or(0.01),
        t_end: parse_f64(&args, "--T").unwrap_or(0.1),
        kappa: parse_f64(&args, "--kappa").unwrap_or(1.0),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pex4_parallel_heat_coarse_case_has_reasonable_error() {
        let result = run_case(8, 2, 1, 0.01, 0.1, 1.0, 1.0);
        assert_eq!(result.global_dofs, 81);
        assert!((result.final_time - 0.1).abs() < 1.0e-12);
        assert!(result.rms_err < 1.0e-2, "RMS error too large: {}", result.rms_err);
        assert!(result.solution_norm > 1.0e-2, "solution norm too small: {}", result.solution_norm);
    }

    #[test]
    fn pex4_parallel_heat_partition_is_invariant_across_one_two_and_four_ranks() {
        let serial = run_case(8, 1, 1, 0.01, 0.1, 1.0, 1.0);
        let parallel2 = run_case(8, 2, 1, 0.01, 0.1, 1.0, 1.0);
        let parallel4 = run_case(8, 4, 1, 0.01, 0.1, 1.0, 1.0);
        assert_eq!(serial.global_dofs, parallel2.global_dofs);
        assert_eq!(serial.global_dofs, parallel4.global_dofs);
        assert!((serial.rms_err - parallel2.rms_err).abs() < 1.0e-12,
            "RMS mismatch: serial={} parallel2={}", serial.rms_err, parallel2.rms_err);
        assert!((serial.rms_err - parallel4.rms_err).abs() < 1.0e-12,
            "RMS mismatch: serial={} parallel4={}", serial.rms_err, parallel4.rms_err);
        assert!((serial.solution_norm - parallel2.solution_norm).abs() < 1.0e-12,
            "solution norm mismatch: serial={} parallel2={}", serial.solution_norm, parallel2.solution_norm);
        assert!((serial.solution_norm - parallel4.solution_norm).abs() < 1.0e-12,
            "solution norm mismatch: serial={} parallel4={}", serial.solution_norm, parallel4.solution_norm);
        assert!((serial.solution_checksum - parallel2.solution_checksum).abs() < 1.0e-10,
            "checksum mismatch: serial={} parallel2={}", serial.solution_checksum, parallel2.solution_checksum);
        assert!((serial.solution_checksum - parallel4.solution_checksum).abs() < 1.0e-10,
            "checksum mismatch: serial={} parallel4={}", serial.solution_checksum, parallel4.solution_checksum);
    }

    #[test]
    fn pex4_parallel_heat_smaller_dt_improves_accuracy() {
        let coarse_dt = run_case(8, 2, 1, 0.01, 0.1, 1.0, 1.0);
        let fine_dt = run_case(8, 2, 1, 0.005, 0.1, 1.0, 1.0);
        assert!(fine_dt.rms_err < coarse_dt.rms_err,
            "expected smaller dt to reduce RMS error: coarse={} fine={}",
            coarse_dt.rms_err, fine_dt.rms_err);
    }

    #[test]
    fn pex4_parallel_heat_sign_reversed_initial_condition_flips_solution() {
        let positive = run_case(8, 2, 1, 0.01, 0.1, 1.0, 1.0);
        let negative = run_case(8, 2, 1, 0.01, 0.1, 1.0, -1.0);
        assert!((positive.rms_err - negative.rms_err).abs() < 1.0e-12,
            "RMS error should be sign-invariant: positive={} negative={}", positive.rms_err, negative.rms_err);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-12,
            "solution norm should be sign-invariant: positive={} negative={}", positive.solution_norm, negative.solution_norm);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "checksum should flip sign: positive={} negative={}", positive.solution_checksum, negative.solution_checksum);
    }

    #[test]
    fn pex4_parallel_heat_kappa_scan_matches_expected_decay_trend() {
        let fast_decay = run_case(8, 2, 1, 0.01, 0.1, 1.0, 1.0);
        let slow_decay = run_case(8, 2, 1, 0.01, 0.1, 0.5, 1.0);
        assert!(slow_decay.decay > fast_decay.decay,
            "smaller kappa should decay less: fast={} slow={}", fast_decay.decay, slow_decay.decay);
        assert!(slow_decay.solution_norm > fast_decay.solution_norm,
            "smaller kappa should retain a larger solution norm: fast={} slow={}",
            fast_decay.solution_norm,
            slow_decay.solution_norm);
        assert!(slow_decay.rms_err < fast_decay.rms_err,
            "smaller kappa should reduce the time-integration error for the same dt: fast={} slow={}",
            fast_decay.rms_err,
            slow_decay.rms_err);
    }

    #[test]
    fn pex4_parallel_heat_p2_case_runs_with_expected_global_dofs() {
        let result = run_case(8, 2, 2, 0.01, 0.1, 1.0, 1.0);
        assert_eq!(result.global_dofs, 289);
        assert!((result.final_time - 0.1).abs() < 1.0e-12);
        assert!(result.rms_err < 1.5e-2, "P2 RMS error too large: {}", result.rms_err);
        assert!(result.solution_norm > 1.0, "P2 solution norm too small: {}", result.solution_norm);
    }

    #[test]
    fn pex4_parallel_heat_p2_partition_is_invariant_between_two_and_four_ranks() {
        let two = run_case(8, 2, 2, 0.01, 0.1, 1.0, 1.0);
        let four = run_case(8, 4, 2, 0.01, 0.1, 1.0, 1.0);
        assert_eq!(two.global_dofs, four.global_dofs);
        assert!((two.rms_err - four.rms_err).abs() < 1.0e-12,
            "P2 RMS mismatch: two={} four={}", two.rms_err, four.rms_err);
        assert!((two.solution_norm - four.solution_norm).abs() < 1.0e-12,
            "P2 solution norm mismatch: two={} four={}", two.solution_norm, four.solution_norm);
        assert!((two.solution_checksum - four.solution_checksum).abs() < 1.0e-10,
            "P2 checksum mismatch: two={} four={}", two.solution_checksum, four.solution_checksum);
    }

    #[test]
    fn pex4_parallel_heat_larger_initial_scale_gives_larger_solution_norm() {
        let small = run_case(8, 2, 1, 0.01, 0.1, 1.0, 0.5);
        let large = run_case(8, 2, 1, 0.01, 0.1, 1.0, 2.0);
        assert!(large.solution_norm > small.solution_norm,
            "expected larger initial scale to produce a larger solution norm: small={} large={}",
            small.solution_norm, large.solution_norm);
    }
}
