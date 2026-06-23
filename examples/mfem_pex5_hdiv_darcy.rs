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
//! cargo run --example mfem_pex5_darcy
//! cargo run --example mfem_pex5_darcy -- --n 16 --ranks 4
//! ```

use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

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
use fem_space::HDivSpace;

fn main() {
    env_logger::init();

    let args = parse_args();

    println!("=== fem-rs mfem_pex5: Parallel H(div) Darcy (RT0) ===");
    println!("  Workers: {}, Mesh: {}x{}", args.n_workers, args.mesh_n, args.mesh_n);

    let result = run_case(args.mesh_n, args.n_workers, 1.0, 1.0, 1.0);

    println!("  Global DOFs: {}", result.global_dofs);
    println!(
        "  PCG: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  h = {:.4e}", result.h);
    println!("  ||rhs||_L2 = {:.4e}", result.rhs_norm);
    println!("  ||u||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("=== Done ===");
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    rhs_norm: f64,
    solution_norm: f64,
    solution_checksum: f64,
}

fn run_case(mesh_n: usize, n_workers: usize, alpha: f64, beta: f64, source_scale: f64) -> RunResult {
    let result = Arc::new(Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);

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

        // 3. Assemble α(∇·F,∇·G) + β(F,G).
        let grad_div = GradDivIntegrator { kappa: alpha };
        let mass = VectorMassIntegrator { alpha: beta };
        let a_mat = ParVectorAssembler::assemble_bilinear(
            &par_space, &[&grad_div, &mass], 3,
        );

        // 4. Assemble RHS.
        // Manufactured: F = (sin(πx)cos(πy), −cos(πx)sin(πy)), ∇·F = 0
        // f = βF
        let source = VectorDomainLFIntegrator {
            f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                out[0] =  source_scale * beta * (PI * x[0]).sin() * (PI * x[1]).cos();
                out[1] = -source_scale * beta * (PI * x[0]).cos() * (PI * x[1]).sin();
            }),
        };
        let rhs = ParVectorAssembler::assemble_linear(&par_space, &[&source], 3);

        // 5. No essential BCs needed (natural F·n = 0 for this manufactured solution).
        // But we need to ensure the system is well-posed. For RT0 with grad-div + mass,
        // the system is SPD so PCG works directly.

        // 6. Solve with parallel PCG + Jacobi.
        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
        let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

        let dof_part = par_space.dof_partition();
        let local_checksum: f64 = u.owned_slice()
            .iter()
            .enumerate()
            .map(|(lid, value)| {
                let gid = dof_part.global_dof(lid as u32) as f64 + 1.0;
                gid * value
            })
            .sum();
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);
        let rhs_norm = rhs.global_norm();
        let solution_norm = u.global_norm();

        if rank == 0 {
            *result_slot.lock().expect("mfem_pex5 result mutex poisoned") = Some(RunResult {
                global_dofs: par_space.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                h: 1.0 / mesh_n as f64,
                rhs_norm,
                solution_norm,
                solution_checksum,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("mfem_pex5 result mutex poisoned after launch")
        .take()
        .expect("rank 0 did not publish mfem_pex5 result");
    final_result
}

struct Args {
    mesh_n: usize,
    n_workers: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    Args {
        mesh_n: parse_arg(&args, "--n").unwrap_or(16),
        n_workers: parse_arg(&args, "--ranks").unwrap_or(2),
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pex5_darcy_coarse_parallel_case_converges() {
        let result = run_case(8, 2, 1.0, 1.0, 1.0);
        assert!(result.converged);
        assert!(result.global_dofs == 208, "unexpected global dof count: {}", result.global_dofs);
        assert!(result.final_residual < 1.0e-7, "residual too large: {}", result.final_residual);
        assert!(result.rhs_norm > 1.0e-2, "rhs norm too small: {}", result.rhs_norm);
        assert!(result.solution_norm > 1.0e-4, "solution norm too small: {}", result.solution_norm);
    }

    #[test]
    fn pex5_darcy_partition_is_invariant_between_one_two_and_four_ranks() {
        let serial = run_case(8, 1, 1.0, 1.0, 1.0);
        let parallel2 = run_case(8, 2, 1.0, 1.0, 1.0);
        let parallel4 = run_case(8, 4, 1.0, 1.0, 1.0);

        assert!(serial.converged && parallel2.converged && parallel4.converged);
        assert_eq!(serial.global_dofs, parallel2.global_dofs);
        assert_eq!(serial.global_dofs, parallel4.global_dofs);
        assert!((serial.rhs_norm - parallel2.rhs_norm).abs() < 1.0e-10);
        assert!((serial.rhs_norm - parallel4.rhs_norm).abs() < 1.0e-10);
        assert!((serial.solution_norm - parallel2.solution_norm).abs() < 1.0e-8,
            "solution norm mismatch: serial={} parallel2={}",
            serial.solution_norm, parallel2.solution_norm);
        assert!((serial.solution_norm - parallel4.solution_norm).abs() < 1.0e-8,
            "solution norm mismatch: serial={} parallel4={}",
            serial.solution_norm, parallel4.solution_norm);
        assert!((serial.solution_checksum - parallel2.solution_checksum).abs() < 1.0e-8,
            "solution checksum mismatch: serial={} parallel2={}",
            serial.solution_checksum, parallel2.solution_checksum);
        assert!((serial.solution_checksum - parallel4.solution_checksum).abs() < 1.0e-8,
            "solution checksum mismatch: serial={} parallel4={}",
            serial.solution_checksum, parallel4.solution_checksum);
    }

    #[test]
    fn pex5_darcy_solution_scales_linearly_with_source() {
        let unit = run_case(8, 2, 1.0, 1.0, 1.0);
        let doubled = run_case(8, 2, 1.0, 1.0, 2.0);

        assert!(unit.converged && doubled.converged);
        assert!((doubled.rhs_norm / unit.rhs_norm - 2.0).abs() < 1.0e-10);
        assert!((doubled.solution_norm / unit.solution_norm - 2.0).abs() < 5.0e-6,
            "solution norm ratio mismatch: unit={} doubled={}",
            unit.solution_norm, doubled.solution_norm);
        assert!((doubled.solution_checksum / unit.solution_checksum - 2.0).abs() < 5.0e-6,
            "solution checksum ratio mismatch: unit={} doubled={}",
            unit.solution_checksum, doubled.solution_checksum);
    }

    #[test]
    fn pex5_darcy_sign_reversed_source_flips_solution() {
        let positive = run_case(8, 2, 1.0, 1.0, 1.0);
        let negative = run_case(8, 2, 1.0, 1.0, -1.0);

        assert!(positive.converged && negative.converged);
        assert!((positive.rhs_norm - negative.rhs_norm).abs() < 1.0e-10);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-10);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-8,
            "expected odd checksum symmetry: positive={} negative={}",
            positive.solution_checksum, negative.solution_checksum);
    }

    #[test]
    fn pex5_darcy_divergence_free_manufactured_solution_is_alpha_invariant() {
        let alpha1 = run_case(8, 2, 1.0, 1.0, 1.0);
        let alpha10 = run_case(8, 2, 10.0, 1.0, 1.0);

        assert!(alpha1.converged && alpha10.converged);
        assert!((alpha1.rhs_norm - alpha10.rhs_norm).abs() < 1.0e-12);
        assert!((alpha1.solution_norm - alpha10.solution_norm).abs() < 1.0e-5,
            "solution norm should be alpha-invariant for the divergence-free manufactured field: alpha1={} alpha10={}",
            alpha1.solution_norm,
            alpha10.solution_norm);
        assert!((alpha1.solution_checksum - alpha10.solution_checksum).abs() < 1.0e-2,
            "solution checksum should be alpha-invariant: alpha1={} alpha10={}",
            alpha1.solution_checksum,
            alpha10.solution_checksum);
    }

    #[test]
    fn pex5_darcy_coarser_mesh_has_larger_h_and_fewer_dofs() {
        let coarse = run_case(4, 2, 1.0, 1.0, 1.0);
        let fine = run_case(8, 2, 1.0, 1.0, 1.0);
        assert!(coarse.converged && fine.converged);
        assert!(coarse.h > fine.h,
            "expected coarser mesh to have larger h: coarse={} fine={}", coarse.h, fine.h);
        assert!(coarse.global_dofs < fine.global_dofs,
            "expected coarser mesh to have fewer DOFs: coarse={} fine={}", coarse.global_dofs, fine.global_dofs);
    }

    #[test]
    fn pex5_darcy_finer_mesh_has_smaller_h_and_more_dofs() {
        let standard = run_case(8, 2, 1.0, 1.0, 1.0);
        let fine = run_case(16, 2, 1.0, 1.0, 1.0);
        assert!(standard.converged && fine.converged);
        assert!(fine.h < standard.h,
            "expected finer mesh to have smaller h: standard={} fine={}", standard.h, fine.h);
        assert!(fine.global_dofs > standard.global_dofs,
            "expected finer mesh to have more DOFs: standard={} fine={}", standard.global_dofs, fine.global_dofs);
    }

    #[test]
    fn pex5_darcy_zero_source_gives_trivial_solution() {
        let result = run_case(8, 2, 1.0, 1.0, 0.0);
        assert!(result.converged);
        assert!(result.rhs_norm < 1.0e-12, "rhs norm should vanish: {}", result.rhs_norm);
        assert!(result.solution_norm < 1.0e-12, "solution norm should vanish: {}", result.solution_norm);
        assert!(result.solution_checksum.abs() < 1.0e-12,
            "solution checksum should vanish: {}", result.solution_checksum);
    }
}
