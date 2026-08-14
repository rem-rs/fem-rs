//! Parallel DG Poisson (1:1 translation of MFEM ex14p)
//!
//! Solves −Δu = 1 with homogeneous Dirichlet BCs using SIP-DG on the L² space:
//!
//! ```text
//!   a = ∫ κ ∇u·∇v + interior/boundary face terms of DGDiffusionIntegrator(κ, σ, k)
//!   b = ∫ 1·v  (the DGDirichletLFIntegrator term is zero for homogeneous BCs:
//!               ex14p adds DGDirichletLFIntegrator(zero, one, σ, k), where the
//!               first coefficient is the Dirichlet data uD = 0, so w = 0)
//! ```
//!
//! Parallel layout (following MFEM ex14p / pex9 conventions):
//! - serial mesh read on every rank, `-rs` serial uniform refinements
//!   (auto: floor(log(10000/NE)/log(2)/dim) — note ex14p caps at **10000**
//!   elements, unlike serial ex14's 50000);
//! - `partition_mesh` + `-rp` parallel uniform refinements;
//! - DG L² space on **Gauss-Legendre** nodes (ex14p uses GaussLegendre unless
//!   `-pa` partial assembly is requested — not implemented here);
//! - local DG assembly over owned + ghost elements (DgAssembler), interior
//!   faces normalized by global element id (e1 = smaller gid) so both ranks
//!   holding a cross-rank face assemble bit-identical entries;
//! - packed via `ParCsrMatrix::from_local_matrix` / `ParVector::from_local_raw`
//!   (L2 `from_l2_space` partition order == element order, identity permute);
//! - solved with PCG + AMG V-cycle (C++: HypreCG + BoomerAMG, rtol 1e-12).
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex14_parallel_dg_poisson -- --ranks 1
//! cargo run --release --example mfem_pex14_parallel_dg_poisson -- --ranks 4
//! cargo run --release --example mfem_pex14_parallel_dg_poisson -- --ranks 2 -m data/star.mesh -o 2
//! ```

use std::sync::{Arc, Mutex};

use fem_assembly::{
    Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_parallel::{
    DofPartition, ParAmgConfig, ParCsrMatrix, ParVector, ParallelFESpace, SmootherType,
    WorkerConfig, launcher::native::ThreadLauncher, par_partition::partition_mesh,
    par_refine::par_uniform_refine, par_solve_pcg_amg,
};
use fem_solver::SolverConfig;
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2) as usize;
    let mesh_file: String = parse_arg_str(&args, "-m")
        .unwrap_or_else(|| "data/star.mesh".to_string());
    let ser_ref_levels: i32 = parse_arg(&args, "-rs").unwrap_or(-1);
    let par_ref_levels: i32 = parse_arg(&args, "-rp").unwrap_or(2);
    let order: u8 = parse_arg(&args, "-o").map(|o| o as u8).unwrap_or(1);
    let sigma: f64 = parse_arg_f64(&args, "-s").unwrap_or(-1.0);
    let kappa: f64 = parse_arg_f64(&args, "-k").unwrap_or(-1.0);
    let eta: f64 = parse_arg_f64(&args, "-e").unwrap_or(0.0);
    let dump_sol = parse_arg_str(&args, "--dump-sol");
    if eta > 0.0 {
        panic!("mfem_pex14: BR2 (eta > 0) is not implemented (same as serial ex14)");
    }
    let kappa = if kappa < 0.0 {
        (order as f64 + 1.0).powi(2)
    } else {
        kappa
    };
    let dim = 2usize;

    println!("=== fem-rs mfem_pex14: Parallel DG Poisson ===");

    // ── 1. Serial mesh + serial refinements (ex14p: auto cap at 10000) ──────
    let mfem = read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("failed to read {mesh_file}: {e}"));
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("mesh must be 2-D");
    let ref_levels = if ser_ref_levels < 0 {
        ((10000.0_f64 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / dim as f64).floor()
            as i32
    } else {
        ser_ref_levels
    };
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }
    let mesh = Arc::new(mesh);

    let result = Arc::new(Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // ── 2. Partition + parallel uniform refinements ──────────────────────
        let mut par_mesh = partition_mesh(&mesh_arc, &comm);
        for _ in 0..par_ref_levels {
            par_mesh = par_uniform_refine(&par_mesh);
        }
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition();

        // ── 3. DG L² space (Gauss-Legendre nodes, like ex14p non-PA) ─────────
        let space_local = L2Space::new(local_mesh, order);
        let part = DofPartition::from_l2_space(&space_local, partition, &comm);
        let ps = ParallelFESpace::new_with_dof_partition(space_local, part, comm.clone());
        let n_owned = ps.dof_partition().n_owned_dofs;
        let ghost = ps.dof_ghost_exchange_arc();

        // ── 4. RHS: b = ∫ 1·v  (boundary DGDirichletLF is 0: uD = 0) ─────────
        let quad_order = order * 2 + 1;
        let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
        let rhs_local = Assembler::assemble_linear(ps.local_space(), &[&source], quad_order);
        let rhs = ParVector::from_local_raw(rhs_local, n_owned, Arc::clone(&ghost), comm.clone());

        // ── 5. A: Diffusion + interior/boundary DGDiffusion (SIP) ────────────
        let mut ifl = InteriorFaceList::build(ps.local_space().mesh());
        // Normalize cross-rank interior faces by global element id (pex9
        // convention): both ranks assemble the same e1/e2 roles, so the
        // face contribution is bit-identical (each elmat block only flips
        // sign under the swap, and |·| products are exact).  Only faces
        // whose elements live on different ranks need this — same-rank
        // faces must keep their local e1/e2 roles (identical to the
        // serial assembly, and the local element order is not the global
        // element order after partitioning).
        for f in ifl.faces.iter_mut() {
            let ol = partition.elem_owner[f.elem_left as usize];
            let or = partition.elem_owner[f.elem_right as usize];
            if ol != or && partition.global_elem(f.elem_left) > partition.global_elem(f.elem_right)
            {
                std::mem::swap(&mut f.elem_left, &mut f.elem_right);
            }
        }
        let a_local =
            DgAssembler::assemble_dg(ps.local_space(), &ifl, 1.0, sigma, kappa, quad_order, None);
        let a_mat = ParCsrMatrix::from_local_matrix(
            &a_local,
            n_owned,
            Arc::clone(&ghost),
            comm.clone(),
        );

        // ── 6. Solve: PCG + AMG V-cycle (C++: CG + BoomerAMG, rtol 1e-12) ───
        let mut u = ParVector::zeros(&ps);
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 500,
            verbose: false,
            ..SolverConfig::default()
        };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 1,
            n_post_smooth: 1,
            smoothed_prolongation: true,
            // Scalar DG system (1 unknown per element-dof): scalar AMG.
            block_size: 1,
            // DG penalty faces couple DOFs across rank boundaries strongly;
            // ghost-aware (cross-rank) aggregation keeps the coarse
            // hierarchy consistent near partition interfaces (default local
            // aggregation stagnates at ~6e-11 after 500 PCG iterations).
            use_global_aggregation: true,
            ..ParAmgConfig::default()
        };
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg)
            .expect("PCG+AMG failed");

        // ── 7. Global metrics + optional dump (global dof order) ─────────────
        let dof_part = ps.dof_partition();
        let solution_norm = u.global_norm();
        let solution_sum =
            comm.allreduce_sum_f64(u.as_slice()[..n_owned].iter().sum::<f64>());
        let local_checksum: f64 = (0..n_owned)
            .map(|pid| {
                let gid = dof_part.global_dof(pid as u32) as f64 + 1.0;
                gid * u.as_slice()[pid]
            })
            .sum();
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);

        if let Some(ref path) = dump_sol {
            let owned: Vec<(u32, f64)> = (0..n_owned)
                .map(|pid| (dof_part.global_dof(pid as u32), u.as_slice()[pid]))
                .collect();
            if rank == 0 {
                let mut all: Vec<(u32, f64)> = owned.clone();
                for src in 1..comm.size() as i32 {
                    let gids: Vec<u32> = comm.recv(src, 91);
                    let vals: Vec<f64> = comm.recv(src, 92);
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
                comm.send(0, 91, &gids);
                comm.send(0, 92, &vals);
            }
        }

        if rank == 0 {
            *result_slot.lock().expect("pex14 result mutex poisoned") = Some(RunResult {
                global_dofs: ps.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                solution_norm,
                solution_sum,
                solution_checksum,
            });
        }
    });

    let r = result.lock().expect("pex14 result mutex after launch").take();
    match r {
        Some(r) => {
            println!("Number of unknowns: {}", r.global_dofs);
            println!(
                "  PCG: {} iters, residual = {:.3e}, converged = {}",
                r.iterations, r.final_residual, r.converged
            );
            let avg = r
                .final_residual
                .powf(1.0 / (r.iterations.max(1) as f64));
            println!("  Average reduction factor = {:.6}", avg);
            println!(
                "  ||u||_2 = {:.6e}, sum = {:.8e}, checksum = {:.8e}",
                r.solution_norm, r.solution_sum, r.solution_checksum
            );
            println!("=== Done ===");
        }
        None => panic!("pex14: no result from workers"),
    }
}

struct RunResult {
    global_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_norm: f64,
    solution_sum: f64,
    solution_checksum: f64,
}

fn parse_arg(args: &[String], name: &str) -> Option<i32> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_arg_f64(args: &[String], name: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_arg_str(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
