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

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_core::{ElemId, Rank};
use fem_mesh::amr::{NCState, dorfler_mark, zz_estimator};
use fem_mesh::{ElementType, Mesh, refine_uniform};
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
    let ref_levels: usize = parse_arg(&args, "-r").unwrap_or(1);
    // C++ ex6p: `if (do_rebalance) pmesh->Rebalance();` after each refine.
    let do_rebalance = !args.iter().any(|a| a == "--no-rebalance");

    println!("=== fem-rs mfem_pex6: Parallel AMR Poisson (H1 P1, ZZ + Dörfler 0.7) ===");

    // Read + serially refine star.mesh identically on every rank (replicated
    // partitioner), then partition.  ex6p starts from the coarse mesh and
    // refines adaptively; -r gives a controllable starting resolution.
    let mfem = fem_io::mfem::read_mfem_file("data/star.mesh")
        .expect("failed to read data/star.mesh");
    let mut mesh0: Mesh<2> = mfem.mesh2d.expect("star.mesh must be 2-D");
    // star.mesh is a quad mesh; the framework's non-conforming NC refine is
    // Tri3-only, so triangulate it (split each quad along its diagonal; the
    // domain geometry is unchanged).  ex6p uses the quad mesh directly with
    // MFEM's nonconforming Quad4 refinement.
    mesh0 = quad_mesh_to_tri(&mesh0);
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
        let mut nc = NCState::new();

        let mut it = 0usize;
        let mut final_dofs = 0usize;
        let mut total_iters = 0usize;
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
            let global_dofs = ps.n_global_dofs();
            final_dofs = global_dofs;
            total_iters += res.iterations;
            if rank == 0 {
                println!(
                    "AMR iteration {it}: unknowns = {global_dofs}, PCG iters = {}, res = {:.3e}",
                    res.iterations, res.final_residual
                );
            }

            // 2. dm-order nodal solution (H1 P1: partition dof order == node order).
            u.update_ghosts();
            let mut u_dm = vec![0.0_f64; local_mesh.n_nodes()];
            for pid in 0..dp.n_total_dofs() {
                let dmid = dp.unpermute_dof(pid as u32) as usize;
                u_dm[dmid] = u.as_slice()[pid];
            }

            // 3. ZZ error indicators + global Dörfler(0.7) marking.
            let eta = zz_estimator(local_mesh, &u_dm);
            let owned_gids: Vec<u32> = (0..partition.n_owned_elems)
                .map(|e| partition.global_elem(e as u32))
                .collect();
            let owned_eta: Vec<f64> = eta[..partition.n_owned_elems].to_vec();
            let global_eta =
                gather_global_eta(&comm, par_mesh.global_n_elems(), &owned_gids, &owned_eta);
            let marked_global: BTreeSet<ElemId> =
                dorfler_mark(&global_eta, 0.7).into_iter().collect();
            let n_marked =
                comm.allreduce_sum_i64(marked_global.len() as i64) as usize;
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
            let r = par_refine_marked(&par_mesh, nc, &marked_local, Some(&u_dm))
                .expect("par_refine_marked failed");
            par_mesh = r.par_mesh;
            nc = r.nc_state;

            // 5b. Load rebalancing after refinement (C++: pmesh->Rebalance()).
            //     Element gids are preserved, so the next round's marks and
            //     the NC-refinement sequence stay globally consistent.
            if do_rebalance {
                par_mesh = par_repartition(par_mesh)
                    .expect("par_repartition failed");
                if rank == 0 {
                    println!(
                        "  rebalanced: {} elements across {} ranks",
                        par_mesh.global_n_elems(),
                        comm.size()
                    );
                }
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

/// Split every Quad4 element of a pure-quad mesh into two Tri3 elements
/// along the (0,1,2)-(0,2,3) diagonal.  Nodes and boundary faces are kept
/// unchanged, so the domain geometry is identical.
fn quad_mesh_to_tri(mesh: &Mesh<2>) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Quad4,
        "quad_mesh_to_tri: expected Quad4 mesh"
    );
    let n = mesh.n_elems();
    let mut conn = Vec::with_capacity(n * 6);
    let mut tags = Vec::with_capacity(n * 2);
    for e in 0..n as u32 {
        let ns = mesh.elem_nodes(e);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);
        conn.extend_from_slice(&[a, b, c]);
        conn.extend_from_slice(&[a, c, d]);
        tags.push(mesh.elem_tags[e as usize]);
        tags.push(mesh.elem_tags[e as usize]);
    }
    Mesh::uniform(
        mesh.coords.clone(),
        conn,
        tags,
        ElementType::Tri3,
        mesh.face_conn.clone(),
        mesh.face_tags.clone(),
        ElementType::Line2,
    )
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
