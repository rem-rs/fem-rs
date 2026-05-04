//! # Parallel Example 3 — Maxwell cavity  (analogous to MFEM pex3)
//!
//! Solves the vector curl-curl + mass problem in parallel:
//!
//! ```text
//!   ∇×(∇×E) + E = f    in Ω = [0,1]²
//!          n×E = 0    on ∂Ω
//! ```
//!
//! Uses H(curl) Nédélec ND1 edge elements, partitioned across workers.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_pex3_maxwell
//! cargo run --example mfem_pex3_maxwell -- --n 16 --ranks 4
//! cargo run --example mfem_pex3_maxwell -- --n 16 --ranks 4 --solver jacobi
//! ```

use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use fem_examples::maxwell::marker_to_tags;
use fem_assembly::{
    standard::{CurlCurlIntegrator, VectorMassIntegrator, VectorMassTensorIntegrator},
    coefficient::ConstantMatrixCoeff,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
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
use fem_space::{HCurlSpace, fe_space::FESpace};
use fem_space::constraints::boundary_dofs_hcurl;

#[derive(Clone, Copy)]
struct RunArgs {
    n_workers: usize,
    mesh_n: usize,
    solver: SolverKind,
    has_pml: bool,
    pml_thickness: f64,
    sigma_max: f64,
    source_scale: f64,
}

#[derive(Clone, Copy, Debug)]
struct CaseResult {
    n_global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_l2: f64,
    solution_checksum: f64,
}

#[derive(Clone, Copy)]
enum SolverKind {
    Jacobi,
    Ams,
}

impl SolverKind {
    fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "ams" => Self::Ams,
            _ => Self::Jacobi,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Jacobi => "jacobi",
            Self::Ams => "ams",
        }
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let n_workers = parse_arg(&args, "--ranks").unwrap_or(2);
    let mesh_n = parse_arg(&args, "--n").unwrap_or(16);
    let solver = parse_solver(&args);
    let pml_thickness = parse_float(&args, "--pml-thickness").unwrap_or(0.2);
    let sigma_max = parse_float(&args, "--sigma-max").unwrap_or(2.0);
    let source_scale = parse_float(&args, "--source-scale").unwrap_or(1.0);
    let has_pml = args.iter().any(|a| a == "--pml");

    let run = RunArgs {
        n_workers,
        mesh_n,
        solver,
        has_pml,
        pml_thickness,
        sigma_max,
        source_scale,
    };
    let result = solve_case(run);

    println!("=== fem-rs mfem_pex3: Parallel Maxwell (ND1) ===");
    println!("  Workers: {} Mesh: {}x{}", run.n_workers, run.mesh_n, run.mesh_n);
    println!("  Solver: {}", run.solver.as_str());
    if run.has_pml {
        println!("  PML-like damping: thickness={}, sigma_max={}", run.pml_thickness, run.sigma_max);
    }
    println!("  Global DOFs: {}", result.n_global_dofs);
    println!(
        "  PCG: {} iters, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  Solution ||u||_2 = {:.3e}", result.solution_l2);
    println!("  checksum = {:.8e}", result.solution_checksum);
    let h = 1.0 / run.mesh_n as f64;
    println!("  h = {h:.4e}  (expected O(h) error for ND1)");
    println!("=== Done ===");
}

fn solve_case(run: RunArgs) -> CaseResult {
    let mesh = Arc::new(SimplexMesh::<2>::unit_square_tri(run.mesh_n));
    let out = Arc::new(Mutex::new(None::<CaseResult>));
    let out_closure = Arc::clone(&out);

    let launcher = ThreadLauncher::new(WorkerConfig::new(run.n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition mesh.
        let par_mesh = partition_simplex(&mesh, &comm);

        // 2. Build parallel H(curl) space.
        let local_space = HCurlSpace::new(par_mesh.local_mesh().clone(), 1);
        let par_space = ParallelFESpace::new_for_edge_space(
            local_space, &par_mesh, comm.clone(),
        );

        let n_global_dofs = par_space.n_global_dofs();

        // 3. Assemble (∇×∇× + damped mass).
        let curl_curl = CurlCurlIntegrator { mu: 1.0 };
        let mut a_mat = if run.has_pml {
            let tau_coeff = pml_mass_tensor(run.pml_thickness, run.sigma_max);
            let vec_mass = VectorMassTensorIntegrator { alpha: tau_coeff };
            ParVectorAssembler::assemble_bilinear(
                &par_space, &[&curl_curl, &vec_mass], 4,
            )
        } else {
            let vec_mass = VectorMassIntegrator { alpha: 1.0 };
            ParVectorAssembler::assemble_bilinear(
                &par_space, &[&curl_curl, &vec_mass], 4,
            )
        };

        // 4. Assemble RHS: f = ((1+π²)sin(πy), (1+π²)sin(πx)).
        let source = MaxwellSource {
            scale: run.source_scale,
        };
        let mut rhs = ParVectorAssembler::assemble_linear(&par_space, &[&source], 4);

        // 5. Apply n×E = 0 on all boundary edges.
        // MFEM-style essential boundary marker (`ess_bdr`): all boundary attributes.
        let bdr_attrs = [1, 2, 3, 4];
        let ess_bdr = [1, 1, 1, 1];
        let ess_tags = marker_to_tags(&bdr_attrs, &ess_bdr);
        let bnd = boundary_dofs_hcurl(
            par_space.local_space().mesh(),
            par_space.local_space(),
            &ess_tags,
        );
        let dof_part = par_space.dof_partition();
        for &d in &bnd {
            let pid = dof_part.permute_dof(d) as usize;
            if pid < dof_part.n_owned_dofs {
                a_mat.apply_dirichlet_par(pid, 0.0, &mut rhs);
            }
        }

        // 6. Solve with selected parallel solver.
        let mut u = ParVector::zeros(&par_space);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
        let res = match run.solver {
            SolverKind::Jacobi => par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap(),
            SolverKind::Ams => {
                if rank == 0 {
                    eprintln!("  [warn] --solver ams requested, but parallel AMS is not implemented yet; falling back to jacobi");
                }
                par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap()
            }
        };

        let solution_l2 = u.global_norm();
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

        if rank == 0 {
            let mut guard = out_closure.lock().expect("lock case result");
            *guard = Some(CaseResult {
                n_global_dofs,
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                solution_l2,
                solution_checksum,
            });
        }
    });

    let guard = out.lock().expect("lock final result");
    guard.expect("rank 0 case result missing")
}

// ─── Manufactured source ────────────────────────────────────────────────────

struct MaxwellSource {
    scale: f64,
}

impl VectorLinearIntegrator for MaxwellSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        let coeff = 1.0 + PI * PI;
        let fx = self.scale * coeff * (PI * x[1]).sin();
        let fy = self.scale * coeff * (PI * x[0]).sin();

        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_float(args: &[String], flag: &str) -> Option<f64> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

/// Construct a 2×2 PML damping tensor [1+σ, 0; 0, 1+σ] as ConstantMatrixCoeff.
/// Uses a simple uniform damping profile at the boundaries.
fn pml_mass_tensor(_thickness: f64, sigma_max: f64) -> ConstantMatrixCoeff {
    // For simplicity, use uniform isotropic damping σ = 0.5 * sigma_max.
    let sigma = 0.5 * sigma_max;
    let diag = [1.0 + sigma, 0.0, 0.0, 1.0 + sigma];
    ConstantMatrixCoeff(diag.to_vec())
}

fn parse_solver(args: &[String]) -> SolverKind {
    args.iter()
        .position(|a| a == "--solver")
        .and_then(|i| args.get(i + 1))
        .map(|s| SolverKind::from_str(s))
        .unwrap_or(SolverKind::Jacobi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_diff(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn pex3_maxwell_parallel_jacobi_converges() {
        let r = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        assert!(r.converged, "parallel pex3 solve did not converge");
        assert!(r.final_residual < 1e-7, "residual too large: {}", r.final_residual);
        assert!(r.n_global_dofs > 0);
        assert!(r.solution_l2.is_finite() && r.solution_l2 > 0.0);
        assert!(r.solution_checksum.is_finite() && r.solution_checksum.abs() > 1.0);
    }

    #[test]
    fn pex3_maxwell_pml_reduces_solution_norm() {
        let baseline = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        let pml = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: true,
            pml_thickness: 0.2,
            sigma_max: 4.0,
            source_scale: 1.0,
        });

        assert!(baseline.converged && pml.converged);
        assert!(
            pml.solution_l2 < baseline.solution_l2,
            "expected PML damping to reduce ||u||2: baseline={} pml={}",
            baseline.solution_l2,
            pml.solution_l2
        );
    }

    #[test]
    fn pex3_maxwell_solution_is_stable_across_one_two_and_four_worker_partitions() {
        let one_rank = solve_case(RunArgs {
            n_workers: 1,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        let two_ranks = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        let four_ranks = solve_case(RunArgs {
            n_workers: 4,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });

        assert!(one_rank.converged && two_ranks.converged && four_ranks.converged);
        assert_eq!(one_rank.n_global_dofs, two_ranks.n_global_dofs);
        assert_eq!(one_rank.n_global_dofs, four_ranks.n_global_dofs);
        assert!(
            rel_diff(one_rank.solution_l2, two_ranks.solution_l2) < 1.0e-8,
            "expected partition-independent solution norm: ranks1={} ranks2={}",
            one_rank.solution_l2,
            two_ranks.solution_l2,
        );
        assert!(
            rel_diff(one_rank.solution_l2, four_ranks.solution_l2) < 1.0e-8,
            "expected partition-independent solution norm: ranks1={} ranks4={}",
            one_rank.solution_l2,
            four_ranks.solution_l2
        );
        assert!(
            rel_diff(one_rank.solution_checksum, two_ranks.solution_checksum) < 1.0e-8,
            "expected partition-independent checksum: ranks1={} ranks2={}",
            one_rank.solution_checksum,
            two_ranks.solution_checksum
        );
        assert!(
            rel_diff(one_rank.solution_checksum, four_ranks.solution_checksum) < 1.0e-8,
            "expected partition-independent checksum: ranks1={} ranks4={}",
            one_rank.solution_checksum,
            four_ranks.solution_checksum
        );
    }

    #[test]
    fn pex3_maxwell_stronger_pml_further_reduces_solution_norm() {
        let moderate = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: true,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        let strong = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: true,
            pml_thickness: 0.2,
            sigma_max: 4.0,
            source_scale: 1.0,
        });

        assert!(moderate.converged && strong.converged);
        assert!(
            strong.solution_l2 < moderate.solution_l2,
            "expected stronger PML to reduce ||u||2 further: moderate={} strong={}",
            moderate.solution_l2,
            strong.solution_l2
        );
    }

    #[test]
    fn pex3_maxwell_solution_scales_linearly_with_source_amplitude() {
        let half = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 0.5,
        });
        let full = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });

        assert!(half.converged && full.converged);
        let ratio = full.solution_l2 / half.solution_l2.max(1.0e-30);
        assert!(
            (ratio - 2.0).abs() < 1.0e-6,
            "expected parallel Maxwell solution norm to scale linearly with source amplitude, got ratio {}",
            ratio
        );
        let checksum_ratio = full.solution_checksum / half.solution_checksum;
        assert!(
            (checksum_ratio - 2.0).abs() < 1.0e-6,
            "expected parallel Maxwell checksum to scale linearly with source amplitude, got ratio {}",
            checksum_ratio
        );
    }

    #[test]
    fn pex3_maxwell_sign_reversed_source_flips_solution_checksum() {
        let positive = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        let negative = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: -1.0,
        });

        assert!(positive.converged && negative.converged);
        assert!((positive.solution_l2 - negative.solution_l2).abs() < 1.0e-10,
            "solution norm should be sign-invariant: positive={} negative={}",
            positive.solution_l2,
            negative.solution_l2);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-8,
            "checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
    }

    #[test]
    fn pex3_maxwell_zero_source_gives_trivial_solution() {
        let zero = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 0.0,
        });

        assert!(zero.converged);
        assert!(zero.solution_l2 < 1.0e-12,
            "zero source should give trivial solution norm, got {}",
            zero.solution_l2);
        assert!(zero.solution_checksum.abs() < 1.0e-12,
            "zero source should give trivial checksum, got {}",
            zero.solution_checksum);
    }

    #[test]
    fn pex3_maxwell_finer_mesh_gives_more_global_dofs() {
        let coarse = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 8,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        let fine = solve_case(RunArgs {
            n_workers: 2,
            mesh_n: 12,
            solver: SolverKind::Jacobi,
            has_pml: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            source_scale: 1.0,
        });
        assert!(coarse.converged && fine.converged);
        assert!(fine.n_global_dofs > coarse.n_global_dofs,
            "expected finer mesh to have more global DOFs: coarse={} fine={}",
            coarse.n_global_dofs, fine.n_global_dofs);
    }
}
