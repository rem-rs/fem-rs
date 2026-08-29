//! # Parallel Example 6 — Parallel AMR Poisson (aligned with MFEM pex6 / ex6p.cpp)
//!
//! Solves the Poisson problem -Δu = 1 on `data/star.mesh` with homogeneous
//! Dirichlet BCs on all boundary attributes, in an AMR loop that mirrors the
//! MFEM ex6p structure (order-1 H1, ZZ error estimator, Dörfler marking with
//! `SetTotalErrorFraction(0.7)`, parallel non-conforming refinement):
//!
//! ```text
//!   repeat:
//!     1. assemble + solve  PCG + AMG (C++: CGSolver + BoomerAMG)
//!     2. ZZ gradient-recovery error indicators (per element)
//!     3. mark the smallest set of elements covering 70% of the total error
//!     4. stop when nothing is marked or the dof budget is reached
//!     5. parallel non-conforming refinement (par_refine_marked), prolongate u
//! ```
//!
//! The refinement uses the framework's cross-rank NC refine
//! ([`par_refine_marked`]): marks are gathered globally, every rank refines
//! its locally visible marked elements, and the partition (global ids,
//! owners, ghost layer — including cross-rank midpoints of coarse edges) is
//! rebuilt to match the serial NC-refinement sequence.  After each refine
//! round the mesh is load-rebalanced ([`par_repartition`], SFC + alltoallv)
//! like C++ `ex6p`'s `pmesh->Rebalance()` — element gids are preserved, so
//! the next round's refine continues with the same global numbering.
//!
//! Usage:
//!   cargo run --release --example mfem_pex6_parallel_amr
//!   cargo run --release --example mfem_pex6_parallel_amr -- --ranks 4 -md 50000
//!   cargo run --release --example mfem_pex6_parallel_amr -- --ranks 4 --no-rebalance

use std::collections::BTreeSet;
use std::sync::Arc;

use fem_assembly::postproc::l2_zz::l2_zz_estimator;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_core::{ElemId, Rank};
use fem_mesh::amr::{detect_hanging_quad, HangingNodeConstraint};
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{ParAmgConfig, SmootherType, par_solve_pcg_amg};
use fem_parallel::par_amr::{par_refine_marked, par_repartition};
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{Comm, ParAssembler, ParVector, ParallelFESpace, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::H1Space;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let max_dofs: usize = parse_arg(&args, "-md").unwrap_or(100_000);
    // C++ ex6p has no `-r` option: it starts from the coarse mesh (star.mesh,
    // 20 quads).  `-r` is only a debugging aid, default 0.
    let ref_levels: usize = parse_arg(&args, "-r").unwrap_or(0);
    // C++ ex6p: `if (do_rebalance) pmesh->Rebalance();` after each refine.
    let do_rebalance = !args.iter().any(|a| a == "--no-rebalance");

    println!("=== fem-rs mfem_pex6: Parallel AMR Poisson (H1 P1, ZZ + Dörfler 0.7) ===");

    // Read + serially refine star.mesh identically on every rank (replicated
    // partitioner), then partition.  ex6p starts from the coarse mesh and
    // refines adaptively; -r gives a controllable starting resolution.
    let mfem = fem_io::mfem::read_mfem_file("data/star.mesh")
        .expect("failed to read data/star.mesh");
    let mut mesh0: Mesh<2> = mfem.mesh2d.expect("star.mesh must be 2-D");
    // star.mesh is a quad mesh; the framework's NC refine supports both
    // Tri3 (NCState) and Quad4 (NCStateQuad) natively, so use it directly
    // (ex6p uses MFEM's nonconforming Quad4 refinement).
    for _ in 0..ref_levels {
        mesh0 = refine_uniform(&mesh0);
    }
    let mesh0 = Arc::new(mesh0);

    let result = Arc::new(std::sync::Mutex::new(None::<(usize, usize)>)); // (final dofs, iters)
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh0);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let mut par_mesh = partition_mesh(&mesh_arc, &comm);
        // Hanging-node constraints are re-detected from the current mesh
        // topology each round (partition rebuilds renumber local ids, so a
        // carried-over NCStateQuad would desync across ranks).

        let mut it = 0usize;
        let mut final_dofs = 0usize;
        let mut total_iters = 0usize;
        // Hanging-node constraints from the previous round's refinement
        // (ids are local node ids of the current local mesh).
        let mut hanging_constraints: Vec<HangingNodeConstraint> = Vec::new();
        loop {
            let local_mesh = par_mesh.local_mesh();
            let partition = par_mesh.partition();
            let n_local_elems = partition.n_owned_elems + partition.n_ghost_elems;
            if n_local_elems == 0 {
                break;
            }

            // 1. FE space, assembly, solve (C++: PCG + BoomerAMG, rtol 1e-6).
            let lm = local_mesh.clone();
            let ps = ParallelFESpace::new(H1Space::new(lm, 1u8), &par_mesh, comm.clone());
            let mut a_mat =
                ParAssembler::assemble_bilinear(&ps, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
            let mut rhs =
                ParAssembler::assemble_linear(&ps, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);
            let dm = ps.local_space().dof_manager();
            let ess = boundary_dofs(local_mesh, dm, &local_mesh.unique_boundary_tags());
            let dp = ps.dof_partition();
            // Drop constraints referencing nodes outside the current DOF
            // range (extra-ghost nodes: in the mesh but not DOFs — their
            // owning rank handles them).  Mesh::n_nodes may include them, so
            // filter against the DOF partition instead.
            let n_dofs = dp.n_total_dofs();
            hanging_constraints.retain(|c| {
                c.constrained < n_dofs
                    && c.parent_a < n_dofs
                    && c.parent_b < n_dofs
                    && c.extra.iter().all(|&(p, _)| p < n_dofs)
            });
            // Apply hanging-node constraints from the previous refinement
            // (MFEM NC spaces constrain hanging dofs implicitly: PᵀKP, Pᵀf).
            if !hanging_constraints.is_empty() {
                a_mat.apply_hanging_constraints(&hanging_constraints, &mut rhs, &dp);
            }
            for &d in &ess {
                let p = dp.permute_dof(d) as usize;
                if p < dp.n_owned_dofs {
                    a_mat.apply_dirichlet_par(p, 0.0, &mut rhs);
                }
            }
            let mut u = ParVector::zeros(&ps);
            let amg_cfg = ParAmgConfig {
                smoother: SmootherType::SymmetricGaussSeidel,
                ..Default::default()
            };
            let cfg = SolverConfig {
                rtol: 1e-6,
                max_iter: 2000,
                verbose: false,
                ..SolverConfig::default()
            };
            let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg)
                .expect("par_solve_pcg_amg failed");
            // C++ ex6p prints `GlobalTrueVSize` — the true dof count after
            // eliminating the hanging-node constraints (the raw `n_global_dofs`
            // counts hanging dofs as independent unknowns).  Subtract the
            // globally-owned hanging dof count.
            let n_hanging_owned = hanging_constraints
                .iter()
                .filter(|c| dp.is_owned_dof(c.constrained as u32))
                .count();
            let n_hanging_global =
                comm.allreduce_sum_i64(n_hanging_owned as i64) as usize;
            let global_dofs = ps.n_global_dofs().saturating_sub(n_hanging_global);
            final_dofs = global_dofs;
            total_iters += res.iterations;
            if rank == 0 {
                println!(
                    "AMR iteration {it}: unknowns = {global_dofs}, PCG iters = {}, res = {:.3e}",
                    res.iterations, res.final_residual
                );
            }

            // 2. dm-order nodal solution (H1 P1: partition dof order == node order).
            //    Recover hanging-node values (constrained dofs were set to 0
            //    by the identity rows): u[c] = 0.5·(u[a] + u[b]).
            //    ParVector is in partition order — permute node ids.
            u.update_ghosts();
            for c in &hanging_constraints {
                let pa = dp.permute_dof(c.parent_a as u32) as usize;
                let pb = dp.permute_dof(c.parent_b as u32) as usize;
                let pc = dp.permute_dof(c.constrained as u32) as usize;
                let v = 0.5 * (u.as_slice()[pa] + u.as_slice()[pb]);
                u.as_slice_mut()[pc] = v;
            }
            let mut u_dm = vec![0.0_f64; local_mesh.n_nodes()];
            for pid in 0..dp.n_total_dofs() {
                let dmid = dp.unpermute_dof(pid as u32) as usize;
                u_dm[dmid] = u.as_slice()[pid];
            }

            // 3. L2-projection ZZ error indicators (C++ ex6p:
            //    L2ZienkiewiczZhuEstimator, flux → RT0 smooth).
            //    np1: serial estimator (validated bit-for-bit vs C++).
            //    np>1: parallel estimator — the RT0 projection must be
            //    solved on the GLOBAL (cross-rank) space, otherwise each
            //    rank's local projection differs from C++ and the marking
            //    drifts (pex6 deep-water: np2 it2 marked 12 vs np1 40).
            let eta = if comm.size() > 1 {
                fem_parallel::par_l2zz::l2_zz_estimator_parallel(&par_mesh, &comm, &u_dm)
            } else {
                l2_zz_estimator(local_mesh, &u_dm)
            };
            let owned_gids: Vec<u32> = (0..partition.n_owned_elems)
                .map(|e| partition.global_elem(e as u32))
                .collect();
            let owned_eta: Vec<f64> = eta[..partition.n_owned_elems].to_vec();
            let global_eta =
                gather_global_eta(&comm, par_mesh.global_n_elems(), &owned_gids, &owned_eta);
            // C++ ex6p: `ThresholdRefiner::SetTotalErrorFraction(0.7)` with the
            // default `total_norm_p = infinity()` marks every element with
            // `η_K > 0.7 · ‖η‖_∞ = 0.7 · max_K η_K` (NOT Dörfler accumulation —
            // that is a different refinement strategy).
            let global_max = global_eta.iter().cloned().fold(0.0_f64, f64::max);
            let threshold = 0.7 * global_max;
            let marked_global: BTreeSet<ElemId> = global_eta
                .iter()
                .enumerate()
                .filter(|&(_, &e)| e > threshold)
                .map(|(i, _)| i as ElemId)
                .collect();
            // `marked_global` is a replicated global set (gather_global_eta
            // broadcasts), so its length is already the global mark count —
            // an allreduce SUM would count it once per rank.
            let n_marked = marked_global.len();
            if rank == 0 {
                println!(
                    "  marked {n_marked} / {} elements",
                    par_mesh.global_n_elems()
                );
            }

            // 4. Stop when nothing is marked or the dof budget is reached.
            if n_marked == 0 || global_dofs >= max_dofs {
                if rank == 0 && n_marked == 0 {
                    println!("Stopping criterion satisfied. Stop.");
                }
                if rank == 0 && global_dofs >= max_dofs {
                    println!("Reached the maximum number of dofs. Stop.");
                }
                break;
            }

            // 5. Local mark list (owned + ghost elements, globally consistent)
            //    and parallel non-conforming refinement with solution
            //    prolongation as the next iteration's initial guess.
            let marked_local: Vec<ElemId> = (0..n_local_elems)
                .filter(|&e| marked_global.contains(&partition.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let r = par_refine_marked(&par_mesh, fem_mesh::amr::NCState::new(), &marked_local, Some(&u_dm))
                .expect("par_refine_marked failed");
            par_mesh = r.par_mesh;

            // 5b. Load rebalancing after refinement (C++: pmesh->Rebalance()).
            //     Element gids are preserved, so the next round's marks and
            //     the NC-refinement sequence stay globally consistent.
            if do_rebalance {
                par_mesh = par_repartition(par_mesh)
                    .expect("par_repartition failed");
            }

            // Detect hanging constraints on the FINAL mesh of this round
            // (after repartitioning — it renumbers local ids, so detecting
            // earlier would leave stale ids).  Topology+coordinate detection
            // (no refinement history): safe on every rank.
            // Drop constraints referencing extra-ghost nodes (id ≥ n_nodes:
            // partition-rebuild artifacts, not DOFs — their owning rank
            // handles them).
            let n_local_nodes = par_mesh.local_mesh().n_nodes();
            hanging_constraints = detect_hanging_quad(par_mesh.local_mesh());
            hanging_constraints.retain(|c| {
                c.constrained < n_local_nodes
                    && c.parent_a < n_local_nodes
                    && c.parent_b < n_local_nodes
                    && c.extra.iter().all(|&(p, _)| p < n_local_nodes)
            });
            if do_rebalance && rank == 0 {
                println!(
                    "  rebalanced: {} elements across {} ranks",
                    par_mesh.global_n_elems(),
                    comm.size()
                );
            }
            it += 1;
        }

        if rank == 0 {
            *result_slot.lock().expect("pex6 mutex") = Some((final_dofs, total_iters));
        }
    });

    let (final_dofs, total_iters) = result
        .lock()
        .expect("pex6 mutex after launch")
        .take()
        .expect("rank 0 did not publish pex6 result");
    println!("=== Done: final unknowns = {final_dofs}, total PCG iters = {total_iters} ===");
}

/// Gather per-element ZZ errors for *owned* elements into a global array
/// (indexed by global element id) replicated on every rank.
///
/// Collective `alltoallv` (not root-collect): a point-to-point root gather
/// would deadlock when one rank runs ahead and blocks in `recv` while the
/// other is still inside a collective (e.g. the PCG solve).
fn gather_global_eta(
    comm: &Comm,
    n_global: usize,
    owned_gids: &[u32],
    owned_eta: &[f64],
) -> Vec<f64> {
    let mut payload = Vec::with_capacity(owned_gids.len() * 12);
    for (&g, &e) in owned_gids.iter().zip(owned_eta) {
        payload.extend_from_slice(&g.to_le_bytes());
        payload.extend_from_slice(&e.to_le_bytes());
    }
    let mut global = vec![0.0_f64; n_global];
    if comm.size() > 1 {
        let sends: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
            .map(|r| (r, payload.clone()))
            .collect();
        for (_src, bytes) in comm.alltoallv_bytes(&sends) {
            for chunk in bytes.chunks_exact(12) {
                let g = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let e = f64::from_le_bytes(chunk[4..12].try_into().unwrap());
                global[g as usize] = e;
            }
        }
    } else {
        for (&g, &e) in owned_gids.iter().zip(owned_eta) {
            global[g as usize] = e;
        }
    }
    global
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
