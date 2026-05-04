//! # Parallel Example 1 -- Parallel Poisson (analogous to MFEM pex1)
//!
//! Solves -Laplacian(u) = f on [0,1]^2 in parallel using ThreadLauncher,
//! where f = 2pi^2 sin(pi*x) sin(pi*y) so that the exact solution is
//! u(x,y) = sin(pi*x) sin(pi*y).
//!
//! Usage:
//!   cargo run --example mfem_pex1_poisson                             # P1, 2 ranks, 16×16
//!   cargo run --example mfem_pex1_poisson -- --p2                     # P2 elements
//!   cargo run --example mfem_pex1_poisson -- --n 32 --ranks 4         # 32×32 mesh, 4 ranks
//!   cargo run --example mfem_pex1_poisson -- --metis                  # METIS graph partitioner
//!   cargo run --example mfem_pex1_poisson -- --streaming              # streaming partition
//!   cargo run --example mfem_pex1_poisson -- --metis --streaming      # METIS + streaming

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    MetisOptions, ParAssembler, ParVector, ParallelFESpace,
    par_simplex::{partition_simplex, partition_simplex_streaming},
    metis::{partition_simplex_metis, partition_simplex_metis_streaming},
    par_solve_pcg_jacobi,
    WorkerConfig,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace};
use fem_space::constraints::boundary_dofs;
use fem_space::dof_manager::DofManager;

#[derive(Clone, Copy)]
struct RunArgs {
    order: u8,
    n_workers: usize,
    mesh_n: usize,
    use_metis: bool,
    use_streaming: bool,
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    l2_err: f64,
    solution_norm: f64,
    solution_sum: f64,
    solution_checksum: f64,
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let use_p2 = args.iter().any(|a| a == "--p2");
    let use_metis = args.iter().any(|a| a == "--metis");
    let use_streaming = args.iter().any(|a| a == "--streaming");
    let run = RunArgs {
        order: if use_p2 { 2 } else { 1 },
        n_workers: parse_arg(&args, "--ranks").unwrap_or(2),
        mesh_n: parse_arg(&args, "--n").unwrap_or(16),
        use_metis,
        use_streaming,
    };

    let partitioner_name = match (use_metis, use_streaming) {
        (false, false) => "contiguous",
        (false, true)  => "contiguous+streaming",
        (true,  false) => "METIS",
        (true,  true)  => "METIS+streaming",
    };

    println!("=== fem-rs mfem_pex1: Parallel Poisson (P{}) ===", run.order);
    println!("  Workers: {}, Mesh: {}x{}, Partitioner: {}", run.n_workers, run.mesh_n, run.mesh_n, partitioner_name);

    let result = run_case(run);
    println!("  Global DOFs: {}", result.global_dofs);
    println!("  PCG: {} iters, residual = {:.3e}, converged = {}", result.iterations, result.final_residual, result.converged);
    println!("  L2 error (pointwise): {:.6e}", result.l2_err);
    println!("  ||u||_2 = {:.6e}, sum = {:.8e}, checksum = {:.8e}", result.solution_norm, result.solution_sum, result.solution_checksum);
    println!("=== Done ===");
}

fn run_case(run: RunArgs) -> RunResult {
    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(run.mesh_n));
    let result = Arc::new(std::sync::Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);

    let launcher = ThreadLauncher::new(WorkerConfig::new(run.n_workers));
    launcher.launch(move |comm| {
        // 1. Build and partition mesh.
        let par_mesh = if run.use_streaming {
            let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
            if run.use_metis {
                partition_simplex_metis_streaming(mesh_opt, &comm, &MetisOptions::default())
                    .expect("METIS streaming partition failed")
            } else {
                partition_simplex_streaming(mesh_opt, &comm)
                    .expect("streaming partition failed")
            }
        } else if run.use_metis {
            partition_simplex_metis(&mesh, &comm, &MetisOptions::default())
        } else {
            partition_simplex(&mesh, &comm)
        };

        let rank = comm.rank();

        // 2. Build parallel FE space.
        let local_mesh = par_mesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, run.order);
        let local_space = H1Space::new(local_mesh, run.order);
        let par_space = if run.order >= 2 {
            ParallelFESpace::new_with_dof_manager(local_space, &par_mesh, &dm, comm.clone())
        } else {
            ParallelFESpace::new(local_space, &par_mesh, comm.clone())
        };

        // 3. Parallel assembly.
        let quad_order = if run.order >= 2 { 4 } else { 3 };
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], quad_order);

        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let rhs_quad = if run.order >= 2 { 5 } else { 3 };
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

        // 6. Compute L2 error against exact solution u_exact = sin(pi*x)*sin(pi*y).
        let n_owned = dof_part.n_owned_dofs;
        let err_dm = par_space.local_space().dof_manager();
        let mut local_err_sq = 0.0_f64;
        for pid in 0..n_owned {
            let dm_id = dof_part.unpermute_dof(pid as u32);
            let coord = err_dm.dof_coord(dm_id);
            let exact = (PI * coord[0]).sin() * (PI * coord[1]).sin();
            let diff_val = u.as_slice()[pid] - exact;
            local_err_sq += diff_val * diff_val;
        }
        let global_err_sq = comm.allreduce_sum_f64(local_err_sq);
        let n_global = par_space.n_global_dofs() as f64;
        let l2_err = (global_err_sq / n_global).sqrt();
        let solution_norm = u.global_norm();
        let solution_sum = comm.allreduce_sum_f64(u.as_slice()[..n_owned].iter().sum::<f64>());
        let local_checksum: f64 = (0..n_owned)
            .map(|pid| {
                let gid = dof_part.global_dof(pid as u32) as f64 + 1.0;
                gid * u.as_slice()[pid]
            })
            .sum();
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);

        if rank == 0 {
            *result_slot.lock().expect("mfem_pex1 result mutex poisoned") = Some(RunResult {
                global_dofs: par_space.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                l2_err,
                solution_norm,
                solution_sum,
                solution_checksum,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("mfem_pex1 result mutex poisoned after launch")
        .take()
        .expect("rank 0 did not publish mfem_pex1 result");
    final_result
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_args(order: u8, n_workers: usize, use_metis: bool, use_streaming: bool) -> RunArgs {
        RunArgs {
            order,
            n_workers,
            mesh_n: 8,
            use_metis,
            use_streaming,
        }
    }

    fn assert_matching_results(label: &str, lhs: &RunResult, rhs: &RunResult) {
        assert!(lhs.converged && rhs.converged, "{label}: expected both runs to converge");
        assert_eq!(lhs.global_dofs, rhs.global_dofs, "{label}: global DOF mismatch");
        assert!(
            (lhs.l2_err - rhs.l2_err).abs() < 1.0e-12,
            "{label}: L2 mismatch lhs={} rhs={}",
            lhs.l2_err,
            rhs.l2_err
        );
        assert!(
            (lhs.solution_norm - rhs.solution_norm).abs() < 1.0e-12,
            "{label}: solution norm mismatch lhs={} rhs={}",
            lhs.solution_norm,
            rhs.solution_norm
        );
        assert!(
            (lhs.solution_sum - rhs.solution_sum).abs() < 1.0e-12,
            "{label}: solution sum mismatch lhs={} rhs={}",
            lhs.solution_sum,
            rhs.solution_sum
        );
    }

    #[test]
    fn pex1_poisson_p1_coarse_case_converges() {
        let result = run_case(run_args(1, 2, false, false));
        assert!(result.converged);
        assert_eq!(result.global_dofs, 81);
        assert!(result.final_residual < 1.0e-8, "residual too large: {}", result.final_residual);
        assert!(result.l2_err < 1.0e-2, "P1 L2 error too large: {}", result.l2_err);
    }

    #[test]
    fn pex1_poisson_p2_coarse_case_converges() {
        let result = run_case(run_args(2, 2, false, false));
        assert!(result.converged);
        assert_eq!(result.global_dofs, 289);
        assert!(result.final_residual < 1.0e-8, "residual too large: {}", result.final_residual);
        assert!(result.l2_err < 2.0e-4, "P2 L2 error too large: {}", result.l2_err);
    }

    #[test]
    fn pex1_poisson_p2_partition_is_invariant_between_two_and_four_ranks() {
        let two = run_case(run_args(2, 2, false, false));
        let four = run_case(run_args(2, 4, false, false));
        assert_matching_results("two-vs-four", &two, &four);
        assert!((two.solution_checksum - four.solution_checksum).abs() < 1.0e-10,
            "P2 checksum mismatch: two={} four={}", two.solution_checksum, four.solution_checksum);
        assert!((two.final_residual - four.final_residual).abs() < 1.0e-8,
            "P2 residual mismatch: two={} four={}", two.final_residual, four.final_residual);
    }

    #[test]
    fn pex1_poisson_streaming_partition_matches_replicated_partition() {
        let replicated = run_case(run_args(1, 2, false, false));
        let streaming = run_case(run_args(1, 2, false, true));

        assert_matching_results("replicated-vs-streaming", &replicated, &streaming);
        assert!(
            (replicated.solution_checksum - streaming.solution_checksum).abs() < 1.0e-10,
            "checksum mismatch: replicated={} streaming={}",
            replicated.solution_checksum,
            streaming.solution_checksum
        );
        assert!(
            (replicated.final_residual - streaming.final_residual).abs() < 1.0e-8,
            "residual mismatch: replicated={} streaming={}",
            replicated.final_residual,
            streaming.final_residual
        );
    }

    #[test]
    fn pex1_poisson_metis_variants_match_contiguous_baseline() {
        let baseline = run_case(run_args(2, 2, false, false));
        let metis = run_case(run_args(2, 2, true, false));
        let metis_streaming = run_case(run_args(2, 2, true, true));

        assert_matching_results("baseline-vs-metis", &baseline, &metis);
        assert_matching_results("baseline-vs-metis-streaming", &baseline, &metis_streaming);
        assert!((metis.solution_checksum - metis_streaming.solution_checksum).abs() < 1.0e-10,
            "METIS checksum mismatch: metis={} metis+streaming={}",
            metis.solution_checksum,
            metis_streaming.solution_checksum);
        assert!(
            (baseline.final_residual - metis.final_residual).abs() < 1.0e-8,
            "baseline/metis residual mismatch: baseline={} metis={}",
            baseline.final_residual,
            metis.final_residual
        );
        assert!(
            (baseline.final_residual - metis_streaming.final_residual).abs() < 1.0e-8,
            "baseline/metis+streaming residual mismatch: baseline={} metis+streaming={}",
            baseline.final_residual,
            metis_streaming.final_residual
        );
    }

    #[test]
    fn pex1_poisson_coarser_mesh_gives_larger_l2_error() {
        let coarse = run_case(RunArgs { order: 1, n_workers: 2, mesh_n: 4, use_metis: false, use_streaming: false });
        let fine = run_case(run_args(1, 2, false, false)); // mesh_n=8
        assert!(coarse.converged && fine.converged);
        assert!(coarse.l2_err > fine.l2_err,
            "expected coarser mesh to yield larger L2 error: coarse={} fine={}", coarse.l2_err, fine.l2_err);
        assert!(coarse.global_dofs < fine.global_dofs,
            "expected coarser mesh to have fewer global DOFs: coarse={} fine={}", coarse.global_dofs, fine.global_dofs);
    }

    #[test]
    fn pex1_poisson_p1_single_worker_matches_two_workers() {
        let one = run_case(RunArgs { order: 1, n_workers: 1, mesh_n: 8, use_metis: false, use_streaming: false });
        let two = run_case(run_args(1, 2, false, false));
        assert!(one.converged && two.converged);
        assert_eq!(one.global_dofs, two.global_dofs);
        assert!((one.l2_err - two.l2_err).abs() < 1.0e-12,
            "single vs two workers L2 error mismatch: one={} two={}", one.l2_err, two.l2_err);
        assert!((one.solution_norm - two.solution_norm).abs() < 1.0e-12,
            "single vs two workers norm mismatch: one={} two={}", one.solution_norm, two.solution_norm);
    }

    #[test]
    fn pex1_poisson_p2_finer_mesh_reduces_l2_error() {
        let coarse = run_case(run_args(2, 2, false, false)); // mesh_n=8
        let fine = run_case(RunArgs { order: 2, n_workers: 2, mesh_n: 16, use_metis: false, use_streaming: false });
        assert!(coarse.converged && fine.converged);
        assert!(fine.l2_err < coarse.l2_err,
            "expected finer mesh to reduce L2 error: coarse={} fine={}", coarse.l2_err, fine.l2_err);
        assert!(fine.global_dofs > coarse.global_dofs,
            "expected finer mesh to have more DOFs: coarse={} fine={}", coarse.global_dofs, fine.global_dofs);
    }
}
