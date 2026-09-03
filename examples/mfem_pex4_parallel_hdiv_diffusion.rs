//! # Parallel Example 4 — Parallel H(div) Diffusion (1:1 port of MFEM pex4 / ex4p.cpp)
//!
//! Solves the 2D H(div) diffusion problem
//! `-grad(alpha div F) + beta F = f` on data/star.mesh with
//! `F·n = <projected exact normal>` on all external boundaries, matching
//! MFEM `examples/ex4p.cpp` defaults:
//! serial `ref_levels` (≤1000 elems → 2) then 2 parallel uniform refinements
//! (4 total, done serially before partitioning — same global mesh), RT0
//! (RT_FECollection(0, 2)) H(div) space, alpha = beta = 1,
//! exact solution `F = (cos(κx)sin(κy), cos(κy)sin(κx))` with κ = π,
//! `f = (1+2κ²)·F`.  CG + AMS/ADS in C++ (2D → AMS); Rust uses
//! PCG + Jacobi (same physical problem, different preconditioner).
//!
//! Usage:
//!   cargo run --release --example mfem_pex4_parallel_hdiv_diffusion
//!   cargo run --release --example mfem_pex4_parallel_hdiv_diffusion -- --ranks 4
//!   cargo run --release --example mfem_pex4_parallel_hdiv_diffusion -- --dump-sol output/pex4_sol.txt

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::standard::{GradDivIntegrator, VectorMassIntegrator};
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_assembly::{VectorAssembler, hdiv_error::compute_hdiv_l2_error_owned};
use fem_element::{raviart_thomas::{QuadRTk, TriRTk}, reference::VectorReferenceElement};
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::{permute_csr, permute_vec};
use fem_parallel::{
    ParAmgConfig, ParCsrMatrix, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
    par_partition::partition_mesh, par_solve_pcg_amg,
};
use fem_parallel::par_mesh::ParallelMesh;
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::fe_space::FESpace;
use fem_space::HDivSpace;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let dump_sol = parse_arg_str(&args, "--dump-sol");

    println!("=== fem-rs mfem_pex4: Parallel H(div) Diffusion ===");
    println!("  Workers: {}, Mesh: star.mesh x4 (1:1 MFEM ex4p), RT0", n_workers);

    let result = run_case(n_workers, dump_sol);
    println!("Number of finite element unknowns: {}", result.global_dofs);
    println!("Size of linear system: {}", result.global_dofs);
    println!(
        "  PCG: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    let avg = result
        .final_residual
        .powf(1.0 / (result.iterations.max(1) as f64));
    println!("  Average reduction factor = {:.6}", avg);
    println!("|| F_h - F ||_{{L^2}} = {}", format!("{:.7}", result.l2_err));
    println!("=== Done ===");
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    l2_err: f64,
}

fn run_case(n_workers: usize, dump_sol: Option<String>) -> RunResult {
    // 1:1 with MFEM ex4p.cpp: star.mesh + serial ref_levels (≤1000 elems:
    // floor(log(1000/20)/log(2)/2) = 2) + 2 parallel refinements = 4 total
    // uniform refinements → 5120 quads / 10400 RT0 dofs.
    let mfem = fem_io::mfem::read_mfem_file("data/star.mesh")
        .expect("failed to read data/star.mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("star.mesh must be 2-D");
    for _ in 0..4 {
        mesh = refine_uniform(&mesh);
    }
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let kappa = PI; // freq = 1 → kappa = freq * M_PI

        // 1. Partition + RT0 H(div) space (edge DOFs, sign corrections).
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        let local_space = HDivSpace::new(par_mesh.local_mesh().clone(), 0);
        let par_space =
            ParallelFESpace::new_for_edge_space(local_space, &par_mesh, comm.clone());
        let dof_part = par_space.dof_partition();
        let n_owned = dof_part.n_owned_dofs;
        let n_dm = par_space.local_space().n_dofs();

        // 2. Essential BCs: all external boundaries (non-homogeneous:
        //    F·n = exact projected values, like ex4p's x.ProjectCoefficient).
        let mesh_ref = par_space.local_space().mesh();
        let all_tags: Vec<i32> = mesh_ref.unique_boundary_tags();
        let ess = boundary_dofs_hdiv(mesh_ref, par_space.local_space(), &all_tags);

        // 3. RHS: b(v) = ∫ f·v, f = (1+2κ²)·F_exact  (local, dm order).
        let quad_order = 4u8; // RT0 → MFEM ex4 uses order*2+2
        let source = MaxwellHSource { kappa };
        let local_rhs = VectorAssembler::assemble_linear(
            par_space.local_space(), &[&source], quad_order,
        );
        let permuted_rhs = permute_vec(&local_rhs, dof_part);
        let mut rhs = ParVector::from_local_raw(
            permuted_rhs,
            n_owned,
            par_space.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // 4. Stiffness: a(u,v) = ∫ (div u)(div v) + u·v  (alpha=beta=1).
        let grad_div = GradDivIntegrator { kappa: 1.0 };
        let vec_mass = VectorMassIntegrator { alpha: 1.0 };
        let local_mat = VectorAssembler::assemble_bilinear(
            par_space.local_space(), &[&grad_div, &vec_mass], quad_order,
        );
        let permuted_mat = permute_csr(&local_mat, dof_part);
        let mut a_mat = ParCsrMatrix::from_local_matrix(
            &permuted_mat,
            n_owned,
            par_space.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // 5. Initial guess x = projection of F_exact (dm order → partition order).
        let x0 = par_space.local_space().interpolate_vector(&|p| {
            exact_f(p, kappa).to_vec()
        });
        let x0_perm = permute_vec(x0.as_slice(), dof_part);
        let mut u = ParVector::from_local_raw(
            x0_perm,
            n_owned,
            par_space.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // 6. Essential BCs with the projected boundary values.
        //    DIAG_KEEP elimination (symmetric, matching C++ FormLinearSystem)
        //    + PCG + AMG (C++: HyprePCG + HypreAMS).
        let ess_global: Vec<u32> = ess
            .iter()
            .map(|&d| dof_part.global_dof(dof_part.permute_dof(d)))
            .collect();
        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        for r in 0..comm.size() as i32 {
            if r == comm.rank() { continue; }
            let mut bytes = Vec::with_capacity(ess_global.len() * 4);
            for &g in &ess_global {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
            sends.push((r, bytes));
        }
        let incoming = comm.alltoallv_bytes(&sends);
        let mut all_bnd: std::collections::HashSet<u32> = ess_global.iter().copied().collect();
        for (_, bytes) in incoming {
            for chunk in bytes.chunks_exact(4) {
                all_bnd.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        let clamped: Vec<usize> = (0..dof_part.n_owned_dofs)
            .filter(|&pid| all_bnd.contains(&dof_part.global_dof(pid as u32)))
            .collect();
        for &pid in &clamped {
            let bc_val = u.owned_slice()[pid];
            a_mat.apply_dirichlet_par_keep_diag(pid, bc_val, &mut rhs);
        }

        // 7. Solve: PCG + AMG (C++: HyprePCG + HypreAMS, rtol 1e-12).
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 5000,
            verbose: false,
            ..SolverConfig::default()
        };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            block_size: 1,
            use_global_aggregation: false,
            ..ParAmgConfig::default()
        };
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg).unwrap();

        // 8. L2 error over owned elements (u partition order → dm order).
        //    compute_hdiv_l2_error_owned already returns the sqrt'ed norm, so
        //    square before the global sum then sqrt again.
        let mut u_dm = vec![0.0_f64; n_dm];
        {
            // Fill owned AND ghost DOFs (owned elements reference ghost DOFs
            // across partition boundaries; refresh ghosts from owners first).
            let mut u_full = u.clone_vec();
            u_full.update_ghosts();
            let needs_sign = dof_part.needs_sign_correction();
            for pid in 0..dof_part.n_total_dofs() {
                let dm = dof_part.unpermute_dof(pid as u32) as usize;
                // permute_vec multiplied by sign_correction when going dm →
                // partition, so invert it back (sign² = 1).
                let s = if needs_sign {
                    dof_part.sign_correction(dm as u32)
                } else {
                    1.0
                };
                u_dm[dm] = u_full.as_slice()[pid] * s;
            }
        }
        let local_err = compute_hdiv_l2_error_owned(
            par_space.local_space(),
            &u_dm,
            |p| exact_f(p, kappa),
            &|e: u32| par_mesh.partition().elem_owner[e as usize] == rank,
        );
        let l2_err = comm.allreduce_sum_f64(local_err * local_err).sqrt();

        // 9. Optional global solution dump (like MFEM sol.XXXXXX).
        if let Some(ref path) = dump_sol {
            let owned: Vec<(u32, f64)> = (0..n_owned)
                .map(|pid| (dof_part.global_dof(pid as u32), u.owned_slice()[pid]))
                .collect();
            if rank == 0 {
                let mut all: Vec<(u32, f64)> = owned.clone();
                for src in 1..comm.size() as i32 {
                    let gids: Vec<u32> = comm.recv(src, 81);
                    let vals: Vec<f64> = comm.recv(src, 82);
                    all.extend(gids.into_iter().zip(vals));
                }
                all.sort_unstable_by_key(|&(g, _)| g);
                let mut s = String::new();
                for (_, v) in all {
                    s.push_str(&format!("{:.10e}\n", v));
                }
                std::fs::write(path, s).expect("failed to write solution dump");
            } else {
                let gids: Vec<u32> = owned.iter().map(|&(g, _)| g).collect();
                let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                comm.send(0, 81, &gids);
                comm.send(0, 82, &vals);
            }
        }

        if rank == 0 {
            *result_slot.lock().expect("pex4 result mutex poisoned") = Some(RunResult {
                global_dofs: par_space.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                l2_err,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("pex4 result mutex poisoned after launch")
        .take()
        .expect("rank 0 did not publish pex4 result");
    final_result
}

/// f = (1 + 2κ²)·(cos(κx)sin(κy), cos(κy)sin(κx)) — ex4p's f_exact.
struct MaxwellHSource {
    kappa: f64,
}

impl VectorLinearIntegrator for MaxwellHSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        let k = self.kappa;
        let temp = 1.0 + 2.0 * k * k;
        let fx = temp * (k * x[0]).cos() * (k * x[1]).sin();
        let fy = temp * (k * x[1]).cos() * (k * x[0]).sin();
        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

/// Exact solution F = (cos(κx)sin(κy), cos(κy)sin(κx)) — ex4p's F_exact.
fn exact_f(x: &[f64], kappa: f64) -> [f64; 2] {
    let k = kappa;
    [
        (k * x[0]).cos() * (k * x[1]).sin(),
        (k * x[1]).cos() * (k * x[0]).sin(),
    ]
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_arg_str<'a>(args: &'a [String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

// Keep imports used by the owned-element L2 error helper below.
#[allow(unused)]
fn _imports_for_error(_p: &ParallelMesh<Mesh<2>>) {
    let _ = TriRTk::new(0);
    let _ = QuadRTk::new(0);
    let _ = 0.0_f64;
}
