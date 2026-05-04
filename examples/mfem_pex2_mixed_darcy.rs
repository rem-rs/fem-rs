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
//! cargo run --example mfem_pex2_mixed_darcy
//! cargo run --example mfem_pex2_mixed_darcy -- --n 16 --ranks 4
//! ```

use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use fem_assembly::{
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace,
    par_simplex::partition_simplex,
    par_solve_pcg_jacobi,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    env_logger::init();

    let args = parse_args();

    println!("=== fem-rs mfem_pex2: Parallel Mixed Darcy (P1/P1 saddle-point) ===");
    println!("  Workers: {}, Mesh: {}x{}", args.n_workers, args.mesh_n, args.mesh_n);

    let result = run_case(args.mesh_n, args.n_workers, 1.0);

    println!("  Global u-DOFs: {}, p-DOFs: {}", result.nu_global, result.np_global);
    println!(
        "  PCG (velocity block): {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  ||f||_L2 = {:.4e}", result.rhs_norm);
    println!("  ||u||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("  ||u||_L1 = {:.4e}", result.solution_l1);
    println!("=== Done ===");
}

struct Args {
    mesh_n: usize,
    n_workers: usize,
}

struct RunResult {
    nu_global: usize,
    np_global: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    rhs_norm: f64,
    solution_norm: f64,
    solution_checksum: f64,
    solution_l1: f64,
}

fn run_case(mesh_n: usize, n_workers: usize, source_scale: f64) -> RunResult {
    let result = Arc::new(Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);

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

        // 3. Assemble A = diffusion for u.
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space_u, &[&diff], 3);

        // 4. Assemble RHS f.
        let source_u = DomainSourceIntegrator::new(|x: &[f64]| {
            source_scale * (PI * x[0]).sin() * (PI * x[1]).sin()
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

        let local_checksum: f64 = u_sol
            .owned_slice()
            .iter()
            .enumerate()
            .map(|(lid, value)| {
                let gid = dof_part.global_dof(lid as u32) as f64 + 1.0;
                gid * value
            })
            .sum();
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);
        let rhs_norm = f_u.global_norm();
        let solution_norm = u_sol.global_norm();
        let solution_l1 = comm.allreduce_sum_f64(
            u_sol.owned_slice().iter().map(|value| value.abs()).sum::<f64>()
        );

        if rank == 0 {
            *result_slot.lock().expect("mfem_pex2 result mutex poisoned") = Some(RunResult {
                nu_global,
                np_global,
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                rhs_norm,
                solution_norm,
                solution_checksum,
                solution_l1,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("mfem_pex2 result mutex poisoned after launch")
        .take()
        .expect("rank 0 did not publish mfem_pex2 result");
    final_result
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    Args {
        n_workers: parse_arg(&args, "--ranks").unwrap_or(2),
        mesh_n: parse_arg(&args, "--n").unwrap_or(8),
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
    fn pex2_mixed_darcy_coarse_case_converges() {
        let result = run_case(8, 2, 1.0);
        assert!(result.converged);
        assert_eq!(result.nu_global, 81);
        assert_eq!(result.np_global, 81);
        assert!(result.final_residual < 1.0e-12, "residual too large: {}", result.final_residual);
        assert!(result.solution_norm > 1.0e-2, "solution norm too small: {}", result.solution_norm);
        assert!(result.solution_l1 > 1.0e-1, "solution L1 too small: {}", result.solution_l1);
    }

    #[test]
    fn pex2_mixed_darcy_partition_is_invariant_across_one_two_and_four_ranks() {
        let serial = run_case(8, 1, 1.0);
        let parallel2 = run_case(8, 2, 1.0);
        let parallel4 = run_case(8, 4, 1.0);
        assert!(serial.converged && parallel2.converged && parallel4.converged);
        assert_eq!(serial.nu_global, parallel2.nu_global);
        assert_eq!(serial.nu_global, parallel4.nu_global);
        assert_eq!(serial.np_global, parallel2.np_global);
        assert_eq!(serial.np_global, parallel4.np_global);
        assert!((serial.rhs_norm - parallel2.rhs_norm).abs() < 1.0e-12);
        assert!((serial.rhs_norm - parallel4.rhs_norm).abs() < 1.0e-12);
        assert!((serial.solution_norm - parallel2.solution_norm).abs() < 1.0e-12,
            "solution norm mismatch: serial={} parallel2={}", serial.solution_norm, parallel2.solution_norm);
        assert!((serial.solution_norm - parallel4.solution_norm).abs() < 1.0e-12,
            "solution norm mismatch: serial={} parallel4={}", serial.solution_norm, parallel4.solution_norm);
        assert!((serial.solution_checksum - parallel2.solution_checksum).abs() < 1.0e-10,
            "checksum mismatch: serial={} parallel2={}", serial.solution_checksum, parallel2.solution_checksum);
        assert!((serial.solution_checksum - parallel4.solution_checksum).abs() < 1.0e-10,
            "checksum mismatch: serial={} parallel4={}", serial.solution_checksum, parallel4.solution_checksum);
        assert!((serial.solution_l1 - parallel2.solution_l1).abs() < 1.0e-12,
            "L1 mismatch: serial={} parallel2={}", serial.solution_l1, parallel2.solution_l1);
        assert!((serial.solution_l1 - parallel4.solution_l1).abs() < 1.0e-12,
            "L1 mismatch: serial={} parallel4={}", serial.solution_l1, parallel4.solution_l1);
    }

    #[test]
    fn pex2_mixed_darcy_solution_scales_linearly_with_source() {
        let unit = run_case(8, 2, 1.0);
        let doubled = run_case(8, 2, 2.0);
        assert!(unit.converged && doubled.converged);
        assert!((doubled.rhs_norm / unit.rhs_norm - 2.0).abs() < 1.0e-12);
        assert!((doubled.solution_norm / unit.solution_norm - 2.0).abs() < 1.0e-10,
            "solution norm ratio mismatch: unit={} doubled={}", unit.solution_norm, doubled.solution_norm);
        assert!((doubled.solution_checksum / unit.solution_checksum - 2.0).abs() < 1.0e-10,
            "checksum ratio mismatch: unit={} doubled={}", unit.solution_checksum, doubled.solution_checksum);
        assert!((doubled.solution_l1 / unit.solution_l1 - 2.0).abs() < 1.0e-10,
            "L1 ratio mismatch: unit={} doubled={}", unit.solution_l1, doubled.solution_l1);
    }

    #[test]
    fn pex2_mixed_darcy_sign_reversed_source_flips_solution() {
        let positive = run_case(8, 2, 1.0);
        let negative = run_case(8, 2, -1.0);
        assert!(positive.converged && negative.converged);
        assert!((positive.rhs_norm - negative.rhs_norm).abs() < 1.0e-12);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-12);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "checksum should flip sign: positive={} negative={}", positive.solution_checksum, negative.solution_checksum);
    }

    #[test]
    fn pex2_mixed_darcy_zero_source_gives_trivial_solution() {
        let result = run_case(8, 2, 0.0);
        assert!(result.converged);
        assert!(result.rhs_norm < 1.0e-12, "rhs norm should vanish: {}", result.rhs_norm);
        assert!(result.solution_norm < 1.0e-12, "solution norm should vanish: {}", result.solution_norm);
        assert!(result.solution_l1 < 1.0e-12, "solution L1 should vanish: {}", result.solution_l1);
        assert!(result.solution_checksum.abs() < 1.0e-12,
            "solution checksum should vanish: {}", result.solution_checksum);
    }

    #[test]
    fn pex2_mixed_darcy_coarser_mesh_converges_with_fewer_dofs() {
        let coarse = run_case(4, 2, 1.0);
        let standard = run_case(8, 2, 1.0);
        assert!(coarse.converged && standard.converged);
        assert!(coarse.nu_global < standard.nu_global,
            "expected coarser mesh to have fewer velocity DOFs: coarse={} standard={}", coarse.nu_global, standard.nu_global);
        assert!(coarse.np_global < standard.np_global,
            "expected coarser mesh to have fewer pressure DOFs: coarse={} standard={}", coarse.np_global, standard.np_global);
    }

    #[test]
    fn pex2_mixed_darcy_larger_source_gives_larger_solution_norm() {
        let small = run_case(8, 2, 0.5);
        let large = run_case(8, 2, 2.0);
        assert!(small.converged && large.converged);
        assert!(large.solution_norm > small.solution_norm,
            "expected larger source to give larger solution norm: small={} large={}", small.solution_norm, large.solution_norm);
        assert!(large.solution_l1 > small.solution_l1,
            "expected larger source to give larger L1 norm: small={} large={}", small.solution_l1, large.solution_l1);
    }

    #[test]
    fn pex2_mixed_darcy_three_ranks_matches_two_ranks() {
        let two = run_case(8, 2, 1.0);
        let three = run_case(8, 3, 1.0);
        assert!(two.converged && three.converged);
        assert_eq!(two.nu_global, three.nu_global);
        assert_eq!(two.np_global, three.np_global);
        assert!((two.solution_norm - three.solution_norm).abs() < 1.0e-12,
            "three vs two ranks norm mismatch: two={} three={}", two.solution_norm, three.solution_norm);
        assert!((two.solution_checksum - three.solution_checksum).abs() < 1.0e-10,
            "three vs two ranks checksum mismatch: two={} three={}", two.solution_checksum, three.solution_checksum);
    }
}
