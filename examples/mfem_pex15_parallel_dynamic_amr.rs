//! # MFEM Example 15p — Parallel Dynamic AMR Heat Equation (1:1)
//!
//! Parallel version of ex15: time-dependent Poisson with a prescribed
//! spherical-front solution, adaptive mesh refinement (threshold-based
//! `ThresholdRefiner`), derefinement (`ThresholdDerefiner`) and load
//! rebalancing after every mesh change.
//!
//! 1:1 with C++ `ex15p.cpp` (MFEM v4.9):
//! - H1 order 2, `L2ZienkiewiczZhuEstimator` (L2(2) flux → RT1 smooth
//!   projection → per-element L1 distance), pure local threshold
//!   (`SetTotalErrorFraction(0.0)`), `SetLocalErrorGoal(max_elem_error)`.
//! - Solve: PCG(1e-12, 200) + BoomerAMG.
//! - Refinement: parallel non-conforming refine (`par_refine_marked_ordered`
//!   + `limit_nc_level_quad` fixpoint) followed by `par_repartition_with_hanging`.
//! - Derefinement: topological 4-children groups with aggregated error below
//!   `hysteresis · max_elem_error`, executed globally on rank 0 and
//!   re-partitioned.
//! - `Iteration: N, number of unknowns: <GlobalTrueVSize>, total error: ...`
//!   printed exactly like C++.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex15_parallel_dynamic_amr -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex15_parallel_dynamic_amr -- --ranks 4 -m data/star.mesh -no-vis
//! ```

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};use fem_core::{ElemId, NodeId, Rank};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_mesh::amr::{HangingNodeConstraint, detect_hanging_quad, limit_nc_level_quad};
use fem_mesh::topology::MeshTopology;
use fem_parallel::Comm;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{ParAmgConfig, SmootherType, par_solve_pcg_amg};
use fem_parallel::par_amr::{par_refine_marked_ordered, par_repartition_with_hanging};
use fem_parallel::par_l2zz_rt1::l2_zz_rt1_estimator_global;
use fem_parallel::par_mesh::ParallelMesh;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{ParAssembler, ParVector, ParallelFESpace, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::H1Space;
use fem_space::constraints::{boundary_dofs, p2_hanging_constraints, recover_hanging_values};
use fem_space::fe_space::FESpace;

const ALPHA: f64 = 0.02;

fn front(x: f64, y: f64, z: f64, t: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    (-0.5 * ((r - t) / ALPHA).powi(2)).exp()
}

fn front_laplace(x: f64, y: f64, z: f64, t: f64, dim: i32) -> f64 {
    let x2 = x * x; let y2 = y * y; let z2 = z * z; let t2 = t * t;
    let r = (x2 + y2 + z2).sqrt();
    let a2 = ALPHA * ALPHA; let a4 = a2 * a2;
    let r_term = if r < 1e-30 { 0.0 } else { -2.0 * t * (x2 + y2 + z2 - (dim as f64 - 1.0) * a2 / 2.0) / r };
    -(-0.5 * ((r - t) / ALPHA).powi(2)).exp() / a4 * (r_term + x2 + y2 + z2 + t2 - dim as f64 * a2)
}

fn bdr_func(pt: &[f64], t: f64) -> f64 {
    let x = pt[0]; let y = pt[1]; let z = if pt.len() == 3 { pt[2] } else { 0.0 };
    front(x, y, z, t)
}

fn rhs_func(pt: &[f64], t: f64) -> f64 {
    let x = pt[0]; let y = pt[1]; let z = if pt.len() == 3 { pt[2] } else { 0.0 };
    front_laplace(x, y, z, t, pt.len() as i32)
}

/// C++ `std::cout << v` with default precision 6 (`%g`-style).
fn fmt_g6(v: f64) -> String {
    let p = 6usize;
    let sci = format!("{:.5e}", v);
    let (mant, exp) = sci.split_once('e').expect("sci format");
    let exp: i32 = exp.parse().expect("exp");
    let neg = mant.starts_with('-');
    let mant = mant.trim_start_matches('-');
    let mut digits: Vec<char> = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    while digits.len() > 1 && digits[digits.len() - 1] == '0' {
        digits.pop();
    }
    let mut out = String::new();
    if neg { out.push('-'); }
    if exp >= -4 && exp < p as i32 {
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if int_len >= digits.len() {
                out.push_str(&digits.iter().collect::<String>());
                out.push_str(&"0".repeat(int_len - digits.len()));
            } else {
                out.push_str(&digits[..int_len].iter().collect::<String>());
                out.push('.');
                out.push_str(&digits[int_len..].iter().collect::<String>());
            }
        } else {
            out.push('0');
            out.push('.');
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits.iter().collect::<String>());
        }
    } else {
        out.push(digits[0]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..].iter().collect::<String>());
        }
        out.push('e');
        if exp < 0 { out.push('-'); } else { out.push('+'); }
        let e = exp.abs();
        if e < 10 { out.push('0'); }
        out.push_str(&e.to_string());
    }
    out
}

// ─── Args ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Args {
    mesh: String,
    problem: i32,
    nfeatures: i32,
    order: u8,
    max_elem_error: f64,
    hysteresis: f64,
    ref_levels: usize,
    nc_limit: u32,
    t_final: f64,
    estimator: i32,
    ranks: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/star-hilbert.mesh".into(),
            problem: 0,
            nfeatures: 1,
            order: 2,
            max_elem_error: 1.0e-4,
            hysteresis: 0.25,
            ref_levels: 0,
            nc_limit: 3,
            t_final: 1.0,
            estimator: 0,
            ranks: 2,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => if let Some(v) = it.next() { a.mesh = v; }
                "-p" | "--problem" => if let Some(v) = it.next() { a.problem = v.parse().unwrap_or(0); }
                "-n" | "--nfeatures" => if let Some(v) = it.next() { a.nfeatures = v.parse().unwrap_or(1); }
                "-o" | "--order" => if let Some(v) = it.next() { a.order = v.parse().unwrap_or(2); }
                "-e" | "--max-err" => if let Some(v) = it.next() { a.max_elem_error = v.parse().unwrap_or(1.0e-4); }
                "-y" | "--hysteresis" => if let Some(v) = it.next() { a.hysteresis = v.parse().unwrap_or(0.25); }
                "-r" | "--ref-levels" | "-rs" | "--refine-serial" => if let Some(v) = it.next() { a.ref_levels = v.parse().unwrap_or(0); }
                "-l" | "--nc-limit" => if let Some(v) = it.next() { a.nc_limit = v.parse().unwrap_or(3); }
                "-tf" | "--t-final" => if let Some(v) = it.next() { a.t_final = v.parse().unwrap_or(1.0); }
                "-est" | "--estimator" => if let Some(v) = it.next() { a.estimator = v.parse().unwrap_or(0); }
                "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
                "-visit" | "--visit-datafiles" | "-no-visit" | "--no-visit-datafiles" => {}
                "--ranks" | "--np" => if let Some(v) = it.next() { a.ranks = v.parse().unwrap_or(2); }
                _ => {}
            }
        }
        a
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let mesh0: Mesh<2> = mfem.mesh2d.expect("expected 2D mesh");
    // C++: serial uniform refinement before partitioning.
    let mut m0 = mesh0;
    for _ in 0..args.ref_levels {
        m0 = fem_mesh::refine_uniform(&m0);
    }
    let mesh0 = Arc::new(m0);

    let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let mut par_mesh = partition_mesh(&mesh0, &comm);
        // MFEM UpdateVertices node creation order (gid → order), threaded
        // across AMR rounds for the RT edge-orientation bookkeeping.
        let mut creation_order: HashMap<u32, u32> =
            (0..mesh0.n_nodes() as u32).map(|g| (g, g)).collect();
        let mut refine_tree = RefineTree::default();

        let dt = 0.01;
        let mut time = 0.0;
        while time < args.t_final + 1e-10 {
            if rank == 0 {
                println!("\nTime {}\n\nRefinement:", fmt_g6(time));
            }

            let mut ref_it = 1usize;
            let mut last_global_eta: Vec<f64> = Vec::new();
            loop {
                // ── Solve on the current mesh (P2, PCG + AMG) ─────────────
                let (global_dofs, u_dm, hang_global) =
                    solve_p2_system(&par_mesh, &comm, time, &creation_order);
                if rank == 0 {
                    print!("Iteration: {ref_it}, number of unknowns: {global_dofs}");
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }

                // ── RT1 L2ZZ error indicators ─────────────────────────────
                if std::env::var("PEX15_TRACE").is_ok() && rank == 0 {
                    let lm = par_mesh.local_mesh();
                    let bad = (0..lm.n_nodes() as u32)
                        .filter(|&n| { let c = lm.coords_of(n as NodeId); !c[0].is_finite() || !c[1].is_finite() })
                        .count();
                    let min_e = par_mesh.global_n_elems();
                    eprintln!("[pex15] mesh nodes={} bad_coords={bad} elems={min_e}", lm.n_nodes());
                }
                let eta_owned = l2_zz_rt1_estimator_global(&par_mesh, &comm, &u_dm, &hang_global);
                if std::env::var("PEX15_DBG").is_ok() && rank == 0 && ref_it == 1 {
                    eprintln!("[dbg] eta_owned[0..5]={:?}", &eta_owned[..5.min(eta_owned.len())]);
                }
                if std::env::var("PEX15_TRACE").is_ok() && rank == 0 {
                    let nnan = eta_owned.iter().filter(|v| !v.is_finite()).count();
                    let nmax = eta_owned.iter().cloned().fold(0.0f64, |m, v| if v.is_finite() { m.max(v) } else { m });
                    eprintln!("[pex15] eta owned={} nan={nnan} max={nmax:.4e}", eta_owned.len());
                    if nnan == 0 && nmax > 0.1 {
                        let i = eta_owned.iter().position(|&v| v == nmax).unwrap();
                        let part = par_mesh.partition();
                        let g = part.global_elem(i as u32);
                        let lm = par_mesh.local_mesh();
                        let ns = lm.elem_nodes(i as ElemId);
                        let c = |k: usize| lm.coords_of(ns[k]);
                        eprintln!("[pex15]   max-eta elem local={i} gid={g} corners=({:.4},{:.4})({:.4},{:.4})({:.4},{:.4})({:.4},{:.4})",
                            c(0)[0], c(0)[1], c(1)[0], c(1)[1], c(2)[0], c(2)[1], c(3)[0], c(3)[1]);
                    }
                }
                let (global_eta, total_err) = gather_global_eta(
                    &comm, &par_mesh, &eta_owned,
                );
                last_global_eta = global_eta.clone();
                if rank == 0 {
                    println!(", total error: {}", fmt_g6(total_err));
                }

                // ── Threshold marking (pure local: η > max_elem_error) ────
                let marked_global: BTreeSet<ElemId> = global_eta
                    .iter()
                    .enumerate()
                    .filter(|&(_, &e)| e > args.max_elem_error)
                    .map(|(i, _)| i as ElemId)
                    .collect();
                let n_marked = marked_global.len();
                if std::env::var("PEX15_TRACE").is_ok() && rank == 0 {
                    eprintln!("[pex15] t={time} it={ref_it} dofs={global_dofs} ne={} marked={n_marked} total_err={total_err:.6e}",
                        global_eta.len());
                }
                if n_marked == 0 {
                    break;
                }

                // ── Parallel NC refine + LimitNCLevel fixpoint + rebalance ─
                let partition = par_mesh.partition();
                let n_local = partition.n_owned_elems + partition.n_ghost_elems;
                let mut to_refine: Vec<ElemId> = (0..n_local)
                    .filter(|&e| marked_global.contains(&partition.global_elem(e as u32)))
                    .map(|e| e as ElemId)
                    .collect();
                if std::env::var("PEX15_DBG").is_ok() && rank == 0 {
                    let lm = par_mesh.local_mesh();
                    eprintln!("[dbg-mark] t={time} it={ref_it} n_marked={n_marked} to_refine={}",
                        to_refine.len());
                    for &e in to_refine.iter().take(10) {
                        let g = partition.global_elem(e);
                        let ns = lm.elem_nodes(e);
                        let cstr = |n: NodeId| {
                            let c = lm.coords_of(n);
                            format!("({:.6},{:.6})", c[0], c[1])
                        };
                        eprintln!("[dbg-mark]   gid={g} nodes {ns:?} coords {}{}{}{}",
                            cstr(ns[0]), cstr(ns[1]), cstr(ns[2]), cstr(ns[3]));
                    }
                    // Dump elements gid 145/281 if present (pre-refine view).
                    for g in [145u32, 281] {
                        if let Some(e) = (0..lm.n_elems() as u32)
                            .find(|&e| partition.global_elem(e) == g)
                        {
                            let ns = lm.elem_nodes(e as ElemId);
                            let cstr = |n: NodeId| {
                                let c = lm.coords_of(n);
                                format!("({:.9},{:.9})", c[0], c[1])
                            };
                            eprintln!("[dbg-mark] pre-refine gid={g} nodes {ns:?} coords {}{}{}{}",
                                cstr(ns[0]), cstr(ns[1]), cstr(ns[2]), cstr(ns[3]));
                        }
                    }
                }
                loop {
                    let before_parents = collect_refine_parents(&par_mesh);
                    let r = par_refine_marked_ordered(
                        &par_mesh, fem_mesh::amr::NCState::new(), &to_refine, None,
                        &creation_order,
                    )
                    .expect("pex15 par_refine_marked_ordered failed");
                    creation_order = r.creation_order;
                    let hanging_edges = r.hanging_edges.clone();
                    let new_pm = par_repartition_with_hanging(r.par_mesh, &hanging_edges)
                        .expect("pex15 par_repartition_with_hanging failed");
                    // Record parent → children provenance for derefinement.
                    update_refine_tree(&mut refine_tree, &before_parents, &new_pm, &comm);
                    let extra: Vec<ElemId> = limit_nc_level_quad(
                        new_pm.local_mesh(), args.nc_limit,
                    )
                    .into_iter()
                    .map(|e| e as ElemId)
                    .collect();
                    par_mesh = new_pm;
                    if std::env::var("PEX15_TRACE").is_ok() && rank == 0 {
                        let lm2 = par_mesh.local_mesh();
                        let cmax = (0..lm2.n_elems() as u32)
                            .flat_map(|e| lm2.elem_nodes(e as fem_core::ElemId).iter().copied())
                            .max()
                            .unwrap_or(0);
                        eprintln!("[pex15-refine] after refine: elems={} nodes={} conn_max={cmax}",
                            lm2.n_elems(), lm2.n_nodes());
                        // Duplicate-coordinate node check + overlapping
                        // elements (same centre, 4-corner average).
                        let part2 = par_mesh.partition();
                        let mut coord_map: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
                        for n in 0..lm2.n_nodes() as u32 {
                            let c = lm2.coords_of(n as NodeId);
                            coord_map.entry(((c[0] * 1e6).round() as i64, (c[1] * 1e6).round() as i64))
                                .or_default().push(n);
                        }
                        let dupn: Vec<_> = coord_map.iter().filter(|(_, v)| v.len() > 1).collect();
                        if !dupn.is_empty() {
                            eprintln!("[pex15-refine] DUP NODES: {}", dupn.len());
                            for (k, v) in dupn.iter().take(8) {
                                let g: Vec<u32> = v.iter().map(|&n| part2.global_node(n)).collect();
                                eprintln!("[pex15-refine]   coord {k:?} local {v:?} gids {g:?}");
                            }
                        }
                        let mut cmap2: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
                        for e in 0..lm2.n_elems() as u32 {
                            let ns = lm2.elem_nodes(e as ElemId);
                            let mut sx = 0.0f64; let mut sy = 0.0f64;
                            for &n in ns { let c = lm2.coords_of(n); sx += c[0]; sy += c[1]; }
                            cmap2.entry(((sx / 4.0 * 1e6).round() as i64, (sy / 4.0 * 1e6).round() as i64))
                                .or_default().push(part2.global_elem(e));
                        }
                        let dup2: Vec<_> = cmap2.iter().filter(|(_, v)| v.len() > 1).collect();
                        if !dup2.is_empty() {
                            eprintln!("[pex15-refine] OVERLAP ELEMS: {} (may be false positive)", dup2.len());
                            for (k, v) in dup2.iter().take(8) {
                                // Dedup: repeated gid = same element (partition
                                // lookup or loop artifact), not an overlap.
                                let uniq: std::collections::BTreeSet<u32> = v.iter().copied().collect();
                                if uniq.len() <= 1 { continue; }
                                eprintln!("[pex15-refine]   centre {k:?} elems {v:?}");
                                for &g in &uniq {
                                    let e = (0..lm2.n_elems() as u32)
                                        .find(|&e| part2.global_elem(e) == g).unwrap();
                                    let ns = lm2.elem_nodes(e as ElemId);
                                    let cstr = |n: NodeId| {
                                        let c = lm2.coords_of(n);
                                        format!("({:.6},{:.6})", c[0], c[1])
                                    };
                                    eprintln!("[pex15-refine]     elem {g} nodes {ns:?} coords {}{}{}{}",
                                        cstr(ns[0]), cstr(ns[1]), cstr(ns[2]), cstr(ns[3]));
                                }
                            }
                        }
                    }
                    if extra.is_empty() {
                        break;
                    }
                    to_refine = extra;
                }
                ref_it += 1;
            }

            // ── Derefinement (C++ `derefiner.Apply(pmesh)`) ───────────────
            // Uses the η of the last inner iteration (the estimator was not
            // Reset(), so the errors from the final solve apply to the
            // current mesh).
            let derefined = if !last_global_eta.is_empty() {
                if std::env::var("PEX15_TRACE").is_ok() && rank == 0 {
                    // Overlap check: any two elements sharing 3+ corner
                    // coordinates (or same centre)?
                    let lm = par_mesh.local_mesh();
                    let part = par_mesh.partition();
                    let mut centre_map: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
                    for e in 0..lm.n_elems() as u32 {
                        let ns = lm.elem_nodes(e as ElemId);
                        let mut sx = 0.0f64; let mut sy = 0.0f64;
                        for &n in ns { let c = lm.coords_of(n); sx += c[0]; sy += c[1]; }
                        centre_map.entry(((sx / 4.0 * 1e6).round() as i64, (sy / 4.0 * 1e6).round() as i64))
                            .or_default().push(part.global_elem(e));
                    }
                    let dup: Vec<_> = centre_map.iter().filter(|(_, v)| v.len() > 1).collect();
                    if !dup.is_empty() {
                        eprintln!("[pex15-deref] OVERLAP: {} duplicate-centre groups:", dup.len());
                        for (k, v) in dup.iter().take(10) {
                            eprintln!("[pex15-deref]   centre {k:?} elems {v:?}");
                        }
                    }
                    let cand: Vec<u32> = refine_tree.records.iter()
                        .filter(|&(_, ch)| ch.iter().all(|&c| (c as usize) < last_global_eta.len()))
                        .map(|(&p, _)| p).collect();
                    eprintln!("[pex15-deref] tree={} candidates={} thresh={:.3e}",
                        refine_tree.records.len(), cand.len(),
                        args.hysteresis * args.max_elem_error);
                }
                parallel_derefine(
                    &mut par_mesh, &comm, &refine_tree, &last_global_eta,
                    args.hysteresis * args.max_elem_error,
                )
            } else {
                false
            };
            if derefined && rank == 0 {
                println!("\nDerefined elements.");
            }
            if derefined {
                // The rebuilt mesh renumbered nodes (compaction) — reset the
                // MFEM UpdateVertices creation order to the new identity.
                let n = par_mesh.local_mesh().n_nodes();
                creation_order = (0..n as u32).map(|g| (g, g)).collect();
                if std::env::var("PEX15_TRACE").is_ok() && rank == 0 {
                    // Overlap check on the rebuilt mesh (dedup by gid).
                    let lm = par_mesh.local_mesh();
                    let part = par_mesh.partition();
                    // Elements referencing nodes 531/488/277/487 or gid 110.
                    for e in 0..lm.n_elems() as u32 {
                        let g = part.global_elem(e);
                        let ns = lm.elem_nodes(e as ElemId);
                        if g == 110 || ns.contains(&531) || ns.contains(&488) {
                            let cstr = |n: NodeId| {
                                let c = lm.coords_of(n);
                                format!("({:.6},{:.6})", c[0], c[1])
                            };
                            eprintln!("[pex15-deref]   elem {e} gid {g} nodes {ns:?} coords {}{}{}{}",
                                cstr(ns[0]), cstr(ns[1]), cstr(ns[2]), cstr(ns[3]));
                        }
                    }
                    let mut cmap3: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
                    for e in 0..lm.n_elems() as u32 {
                        let ns = lm.elem_nodes(e as ElemId);
                        let mut sx = 0.0f64; let mut sy = 0.0f64;
                        for &n in ns { let c = lm.coords_of(n); sx += c[0]; sy += c[1]; }
                        cmap3.entry(((sx / 4.0 * 1e6).round() as i64, (sy / 4.0 * 1e6).round() as i64))
                            .or_default().push(part.global_elem(e));
                    }
                    let mut real_dup = 0usize;
                    for (k, v) in cmap3.iter() {
                        let uniq: std::collections::BTreeSet<u32> = v.iter().copied().collect();
                        if uniq.len() > 1 {
                            real_dup += 1;
                            eprintln!("[pex15-deref] REBUILT OVERLAP centre {k:?} elems {v:?}");
                        }
                    }
                    eprintln!("[pex15-deref] rebuilt mesh overlap groups = {real_dup}");
                }
            }

            time += dt;
        }
    });
}

// ─── P2 solve (assembly + constraints + PCG/AMG) ─────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_p2_system(
    par_mesh: &ParallelMesh<Mesh<2>>,
    comm: &Comm,
    time: f64,
    creation_order: &HashMap<u32, u32>,
) -> (usize, Vec<f64>, Vec<(u32, u32, u32)>) {
    let lm = par_mesh.local_mesh().clone();
    let h1 = H1Space::new(lm.clone(), 2u8);
    let dm0 = fem_space::dof_manager::DofManager::new(&lm, 2);
    let ps = ParallelFESpace::new_with_dof_manager(h1, par_mesh, &dm0, comm.clone());
    let quad_order = 5u8; // 2·order + 1
    let mut a_mat = ParAssembler::assemble_bilinear(
        &ps, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order,
    );
    let rhs_fn = |pt: &[f64]| rhs_func(pt, time);
    let mut rhs = ParAssembler::assemble_linear(
        &ps, &[&DomainSourceIntegrator::new(rhs_fn)], quad_order,
    );

    // ── Hanging-node constraints (P2 DOF level) ───────────────────────────
    // Detect locally, merge the global master-edge table (a 2nd-level
    // hanging edge whose fine elements live on another rank would be
    // missed), then map back to local node ids for this rank.
    let partition = par_mesh.partition();
    let hc_local = detect_hanging_quad(&lm);
    let hang_global = merge_hanging_global(comm, partition, &hc_local);
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
    let mut hc_local2: Vec<HangingNodeConstraint> = Vec::new();
    for &(pa, pb, mid) in &hang_global {
        let (Some(la), Some(lb), Some(lm_)) = (
            partition.local_node(pa),
            partition.local_node(pb),
            partition.local_node(mid),
        ) else { continue };
        midpoints.insert((la.min(lb), la.max(lb)), lm_);
        hc_local2.push(HangingNodeConstraint::new_weighted(
            lm_ as usize, la as usize, lb as usize, 0.5, 0.5, vec![],
        ));
    }
    let dm = ps.local_space().dof_manager();
    let dp = ps.dof_partition();
    let n_dofs = dp.n_total_dofs();
    let p2_hc: Vec<HangingNodeConstraint> = if args_order() == 2 && !hc_local2.is_empty() {
        let hc = p2_hanging_constraints(&hc_local2, dm, &midpoints);
        hc.into_iter()
            .filter(|c| {
                c.constrained < n_dofs
                    && c.parent_a < n_dofs
                    && c.parent_b < n_dofs
                    && c.extra.iter().all(|&(p, _)| p < n_dofs)
            })
            .collect()
    } else {
        Vec::new()
    };
    if !p2_hc.is_empty() {
        a_mat.apply_hanging_constraints(&p2_hc, &mut rhs, &dp);
    }
    if std::env::var("PEX15_DBG").is_ok() {
        eprintln!("[dbg-p2] t={time} p2_hc={} n_dofs={n_dofs} n_owned={}", p2_hc.len(), dp.n_owned_dofs);
    }

    // ── Dirichlet BC (all boundaries, time-dependent) ─────────────────────
    let ess = boundary_dofs(&lm, dm, &lm.unique_boundary_tags());
    for &d in &ess {
        let p = dp.permute_dof(d) as usize;
        if p < dp.n_owned_dofs {
            let coord = dm.dof_coord(d);
            a_mat.apply_dirichlet_par(p, bdr_func(&coord, time), &mut rhs);
        }
    }

    // ── Solve: PCG + BoomerAMG (C++: tol 1e-12, max 200) ──────────────────
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
    let solve_res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg)
        .expect("pex15 par_solve_pcg_amg failed");
    if std::env::var("PEX15_DBG").is_ok() {
        eprintln!("[dbg-solve] t={time} iters={} res={:.3e}", solve_res.iterations, solve_res.final_residual);
    }

    // ── GlobalTrueVSize: raw global dofs minus owned hanging dofs ─────────
    let n_hanging_owned = p2_hc
        .iter()
        .filter(|c| dp.is_owned_dof(c.constrained as u32))
        .count();
    let n_hanging_global = comm.allreduce_sum_i64(n_hanging_owned as i64) as usize;
    let global_dofs = ps.n_global_dofs().saturating_sub(n_hanging_global);

    // ── Solution in dm order, hanging values recovered (MFEM
    //    RecoverFEMSolution: P expansion of the true-dof solution) ────────
    u.update_ghosts();
    let mut u_dm = vec![0.0f64; n_dofs];
    for pid in 0..dp.n_total_dofs() {
        let dmid = dp.unpermute_dof(pid as u32) as usize;
        u_dm[dmid] = u.as_slice()[pid];
    }
    if !p2_hc.is_empty() {
        recover_hanging_values(&mut u_dm, &p2_hc);
    }

    let _ = creation_order;
    (global_dofs, u_dm, hang_global)
}

fn args_order() -> u8 {
    2
}

// ─── Parallel derefinement ───────────────────────────────────────────────────
//
// MFEM ex15p's `ThresholdDerefiner` coarsens any 4-children group (from the
// NC-refinement tree) whose aggregated error is below `hysteresis·goal`.
// A parallel topology-only detection cannot distinguish a "coarsenable
// group" (4 children of a refined parent) from a plain coarse-mesh patch of
// 4 elements sharing a corner — so we track the global parent→children
// provenance across refine rounds (`RefineTree`, populated by comparing the
// mesh before/after each parallel refine) and only coarsen recorded groups
// whose 4 children are all current leaves.
//
// Execution is global on rank 0 (remove the 4 children, restore the parent
// element at its original gid — the gid set stays dense 0..N-1), the new
// mesh is broadcast and every rank re-partitions.

/// Global parent → 4 children element records (gids), plus the parent's
/// own corners as **coordinates** (the parallel rebuild compacts node gids,
/// so gids captured at refine time are not valid after a derefinement
/// rebuild; coordinates are).
#[derive(Default)]
struct RefineTree {
    records: HashMap<u32, [u32; 4]>,
    parent_corners: HashMap<u32, [[f64; 2]; 4]>,
    /// Child element **centres** (XY-split coordinates) recorded at refine
    /// time.  Element gids are renumbered by every later refine, so a
    /// recorded child gid can point at an unrelated element by the time the
    /// tree is used for derefinement — re-locate each child in the current
    /// mesh by its centre coordinate instead.
    child_centres: HashMap<u32, [[f64; 2]; 4]>,
}

/// Populate/refresh `tree` from one parallel refine round: every refined
/// parent (gid present before, absent after) gets its 4 children (the
/// elements whose corners contain the parent's center) recorded, merged
/// across ranks.
///
/// `before_parents` = owned element (gid, center) of the pre-refine mesh,
/// collected by [`collect_refine_parents`].
fn update_refine_tree(
    tree: &mut RefineTree,
    before_parents: &[(u32, [f64; 2], [[f64; 2]; 4])],
    after: &ParallelMesh<Mesh<2>>,
    comm: &Comm,
) {
    let mesh_after = after.local_mesh();
    let part_after = after.partition();
    if std::env::var("PEX15_TRACE").is_ok() {
        eprintln!("[pex15-tree] before_parents={} after_elems={}",
            before_parents.len(), mesh_after.n_elems());
        if let Some(&(pg, pc, _corners)) = before_parents.first() {
            eprintln!("[pex15-tree] parent {pg} center = ({:.9}, {:.9})", pc[0], pc[1]);
            let mut shown = 0;
            for e in 0..mesh_after.n_elems() as u32 {
                if shown >= 4 { break; }
                let ns = mesh_after.elem_nodes(e as ElemId);
                let c = mesh_after.coords_of(ns[0]);
                eprintln!("[pex15-tree]   after elem {e} (gid {}) corner0 = ({:.9}, {:.9})",
                    part_after.global_elem(e), c[0], c[1]);
                shown += 1;
            }
            for e in 0..mesh_after.n_elems() as u32 {
                let ns = mesh_after.elem_nodes(e as ElemId);
                let mut has = false;
                for &n in ns {
                    let c = mesh_after.coords_of(n);
                    if (c[0] - pc[0]).abs() < 1e-6 && (c[1] - pc[1]).abs() < 1e-6 {
                        has = true;
                    }
                }
                if has {
                    let c = mesh_after.coords_of(ns[0]);
                    eprintln!("[pex15-tree]   after elem {e} (gid {}) contains center, corner0 = ({:.9}, {:.9})",
                        part_after.global_elem(e), c[0], c[1]);
                }
            }
        }
    }

    // For each parent, find children among the after-mesh elements whose
    // corner set contains the parent center (1e-9 tolerance, like
    // detect_hanging_quad) AND whose geometry matches a direct child —
    // ≥2 of the parent's edge midpoints (by coordinate) among its corners.
    // A multi-level refine (a refined child refined further by
    // limit_nc_level propagation) makes the grandCHILDREN also contain the
    // parent center; the midpoint count separates them (a grandchild holds
    // ≤1 parent edge midpoint — its own parent's other corners are the
    // parent's ¼ points / centre).  Recording a grandchild group as the
    // parent's children makes a later derefine remove the wrong elements
    // (pex15 t=0: recovered parents coexist with their children → next
    // refine duplicates them).
    let mut payload: Vec<u8> = Vec::new();
    let mut found: Vec<(u32, u32, [f64; 2], [[f64; 2]; 4])> = Vec::new(); // (parent, child, child centre, parent corners)
    for &(pg, pc, corners) in before_parents {
        // NOTE: no "parent still present" gid check — the parallel rebuild
        // reuses the parent's gid for its first child, so the gid survives
        // a refine.  A parent is refined iff its center became a corner of
        // some after-elements; an unrefined parent's center is never a
        // corner (it lies strictly inside the parent).
        // Parent edge midpoints (coordinates, 1e-9 match).
        let p_mid: Vec<[f64; 2]> = (0..4)
            .map(|i| {
                let a = corners[i];
                let b = corners[(i + 1) % 4];
                [0.5 * a[0] + 0.5 * b[0], 0.5 * a[1] + 0.5 * b[1]]
            })
            .collect();
        let mut children: Vec<u32> = Vec::new();
        let mut centres: Vec<[f64; 2]> = Vec::new();
        for e in 0..mesh_after.n_elems() as u32 {
            let g = part_after.global_elem(e);
            let ns = mesh_after.elem_nodes(e as ElemId);
            let mut has = false;
            for &n in ns {
                let c = mesh_after.coords_of(n);
                if (c[0] - pc[0]).abs() < 1e-6 && (c[1] - pc[1]).abs() < 1e-6 {
                    has = true;
                    break;
                }
            }
            if !has {
                continue;
            }
            // Direct-child check: ≥2 parent edge midpoints among corners.
            let n_mid = p_mid
                .iter()
                .filter(|m| {
                    ns.iter().any(|&n| {
                        let c = mesh_after.coords_of(n);
                        (c[0] - m[0]).abs() < 1e-9 && (c[1] - m[1]).abs() < 1e-9
                    })
                })
                .count();
            if n_mid >= 2 {
                children.push(g);
                // XY-split centre of the child (0.5·mid01 + 0.5·mid23) —
                // used later to re-locate the child after gid renumbering.
                let c0 = mesh_after.coords_of(ns[0]);
                let c1 = mesh_after.coords_of(ns[1]);
                let c2 = mesh_after.coords_of(ns[2]);
                let c3 = mesh_after.coords_of(ns[3]);
                let (m01x, m01y) = (0.5 * c0[0] + 0.5 * c1[0], 0.5 * c0[1] + 0.5 * c1[1]);
                let (m23x, m23y) = (0.5 * c2[0] + 0.5 * c3[0], 0.5 * c2[1] + 0.5 * c3[1]);
                centres.push([0.5 * m01x + 0.5 * m23x, 0.5 * m01y + 0.5 * m23y]);
            }
        }
        if children.len() >= 4 {
            // Sort children and their centres in lockstep by gid.
            let mut idx: Vec<usize> = (0..children.len()).collect();
            idx.sort_by_key(|&i| children[i]);
            for &i in idx.iter().take(4) {
                found.push((pg, children[i], centres[i], corners));
            }
        } else if std::env::var("PEX15_TRACE").is_ok() && children.len() > 0 {
            eprintln!("[pex15-tree] parent {pg} children={children:?} (<4)");
        }
    }
    for &(p, c, ctr, crns) in &found {
        payload.extend_from_slice(&p.to_le_bytes());
        payload.extend_from_slice(&c.to_le_bytes());
        payload.extend_from_slice(&ctr[0].to_le_bytes());
        payload.extend_from_slice(&ctr[1].to_le_bytes());
        for cr in &crns {
            payload.extend_from_slice(&cr[0].to_le_bytes());
            payload.extend_from_slice(&cr[1].to_le_bytes());
        }
    }
    // Merge across ranks.
    let mut pairs: Vec<(u32, u32, [f64; 2], [[f64; 2]; 4])> = found;
    if comm.size() > 1 {
        let sends: Vec<(Rank, Vec<u8>)> =
            (0..comm.size() as i32).map(|r| (r, payload.clone())).collect();
        for (_src, bytes) in comm.alltoallv_bytes(&sends) {
            for chunk in bytes.chunks_exact(88) {
                let mut crns = [[0.0f64; 2]; 4];
                for (j, cr) in crns.iter_mut().enumerate() {
                    cr[0] = f64::from_le_bytes(chunk[24 + j * 16..32 + j * 16].try_into().unwrap());
                    cr[1] = f64::from_le_bytes(chunk[32 + j * 16..40 + j * 16].try_into().unwrap());
                }
                pairs.push((
                    u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                    [
                        f64::from_le_bytes(chunk[8..16].try_into().unwrap()),
                        f64::from_le_bytes(chunk[16..24].try_into().unwrap()),
                    ],
                    crns,
                ));
            }
        }
    }
    // Aggregate parent → up to 4 children (gid + centre, lockstep).
    let mut by_parent: HashMap<u32, Vec<(u32, [f64; 2], [[f64; 2]; 4])>> = HashMap::new();
    for &(p, c, ctr, crns) in &pairs {
        by_parent.entry(p).or_default().push((c, ctr, crns));
    }
    for (p, mut cs) in by_parent {
        cs.sort_by_key(|&(c, _, _)| c);
        cs.dedup_by_key(|&mut (c, _, _)| c);
        if cs.len() >= 4 {
            tree.records.insert(p, [cs[0].0, cs[1].0, cs[2].0, cs[3].0]);
            tree.child_centres.insert(p, [cs[0].1, cs[1].1, cs[2].1, cs[3].1]);
            tree.parent_corners.insert(p, cs[0].2);
        }
    }
}

/// Collect owned elements (gid, center, corners) — the pre-refine parent
/// candidates.
fn collect_refine_parents(par_mesh: &ParallelMesh<Mesh<2>>) -> Vec<(u32, [f64; 2], [[f64; 2]; 4])> {
    let mesh = par_mesh.local_mesh();
    let part = par_mesh.partition();
    let mut out = Vec::with_capacity(part.n_owned_elems);
    for e in 0..part.n_owned_elems as u32 {
        let g = part.global_elem(e);
        let ns = mesh.elem_nodes(e as ElemId);
        let c0 = mesh.coords_of(ns[0]);
        let c2 = mesh.coords_of(ns[2]);
        let corners = [
            mesh.coords_of(ns[0]),
            mesh.coords_of(ns[1]),
            mesh.coords_of(ns[2]),
            mesh.coords_of(ns[3]),
        ];
        out.push((g, [0.5 * (c0[0] + c2[0]), 0.5 * (c0[1] + c2[1])], corners));
    }
    out
}

/// Derefinement step (C++ `ThresholdDerefiner::Apply`): choose groups whose
/// 4 children are all current leaves with aggregated error below the
/// threshold, execute globally on rank 0, broadcast the new mesh and
/// re-partition on every rank.  Returns true if anything was coarsened.
fn parallel_derefine(
    par_mesh: &mut ParallelMesh<Mesh<2>>,
    comm: &Comm,
    tree: &RefineTree,
    global_eta: &[f64],
    threshold: f64,
) -> bool {
    let partition = par_mesh.partition();
    let n_elems = par_mesh.global_n_elems();
    let mesh = par_mesh.local_mesh();

    // Candidate parents: recorded groups (children relocated by centre
    // coordinates later, on rank 0 — the recorded gids are stale after gid
    // renumbering).
    let mut candidates: Vec<u32> = tree.records.keys().copied().collect();
    candidates.sort_unstable();
    if candidates.is_empty() {
        return false;
    }

    // 2. Global mesh data → rank 0 (elements: gid → tag+corners; nodes:
    //    gid → coords).  Every rank submits its local (owned+ghost) view.
    //    IMPORTANT: every rank must CALL alltoallv_bytes (collective); only
    //    rank 0 parses the result.  A `if comm.rank() == 0` guard around the
    //    call deadlocks np2 (rank 1 never joins the collective).
    let mut elems_g: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut nodes_g: HashMap<u32, [f64; 2]> = HashMap::new();
    {
        let mut elem_payload = Vec::new();
        for e in 0..mesh.n_elems() as u32 {
            let g = partition.global_elem(e);
            let ns = mesh.elem_nodes(e as ElemId);
            let tag = mesh.element_tag(e as ElemId) as u32;
            elem_payload.extend_from_slice(&g.to_le_bytes());
            elem_payload.extend_from_slice(&tag.to_le_bytes());
            for &n in ns {
                elem_payload.extend_from_slice(&partition.global_node(n).to_le_bytes());
            }
        }
        let elem_sends: Vec<(Rank, Vec<u8>)> =
            (0..comm.size() as i32).map(|r| (r, elem_payload.clone())).collect();
        let recv_e: Vec<(Rank, Vec<u8>)> = if comm.size() > 1 {
            comm.alltoallv_bytes(&elem_sends)
        } else {
            vec![(0, elem_payload)]
        };
        if comm.rank() == 0 {
            for (_src, bytes) in recv_e {
                for rec in bytes.chunks_exact(24) {
                    let g = u32::from_le_bytes(rec[0..4].try_into().unwrap());
                    let tag = u32::from_le_bytes(rec[4..8].try_into().unwrap());
                    let mut cs = Vec::with_capacity(4);
                    for k in 0..4 {
                        cs.push(u32::from_le_bytes(rec[8 + k * 4..12 + k * 4].try_into().unwrap()));
                    }
                    elems_g.insert(g, vec![tag, cs[0], cs[1], cs[2], cs[3]]);
                }
            }
        }
        let mut node_payload = Vec::new();
        let n_nodes_part = partition.global_node_ids.len();
        for n in 0..n_nodes_part as u32 {
            let g = partition.global_node(n);
            let c = mesh.coords_of(n as NodeId);
            node_payload.extend_from_slice(&g.to_le_bytes());
            node_payload.extend_from_slice(&c[0].to_le_bytes());
            node_payload.extend_from_slice(&c[1].to_le_bytes());
        }
        let node_sends: Vec<(Rank, Vec<u8>)> =
            (0..comm.size() as i32).map(|r| (r, node_payload.clone())).collect();
        let recv_n: Vec<(Rank, Vec<u8>)> = if comm.size() > 1 {
            comm.alltoallv_bytes(&node_sends)
        } else {
            vec![(0, node_payload)]
        };
        if comm.rank() == 0 {
            for (_src, bytes) in recv_n {
                for rec in bytes.chunks_exact(20) {
                    let g = u32::from_le_bytes(rec[0..4].try_into().unwrap());
                    let x = f64::from_le_bytes(rec[4..12].try_into().unwrap());
                    let y = f64::from_le_bytes(rec[12..20].try_into().unwrap());
                    nodes_g.insert(g, [x, y]);
                }
            }
        }
    }

    // 3. Rank 0: choose groups + execute.
    let mut chosen: Vec<u32> = Vec::new();
    let mut bcast: Vec<u8> = Vec::new();
    if comm.rank() == 0 {
        // Quantized node-coordinate set for the leaf check.
        let node_set: std::collections::HashSet<(i64, i64)> = nodes_g
            .iter()
            .map(|(_, c)| ((c[0] * 1e6).round() as i64, (c[1] * 1e6).round() as i64))
            .collect();
        // Element XY-split centre from its 4 corner gids (same formula as
        // NCStateQuad::refine).
        let elem_centre = |rec: &[u32], nodes: &HashMap<u32, [f64; 2]>| -> Option<[f64; 2]> {
            let n1 = nodes.get(&rec[1])?;
            let n2 = nodes.get(&rec[2])?;
            let n3 = nodes.get(&rec[3])?;
            let n4 = nodes.get(&rec[4])?;
            let (m01x, m01y) = (0.5 * n1[0] + 0.5 * n2[0], 0.5 * n1[1] + 0.5 * n2[1]);
            let (m23x, m23y) = (0.5 * n3[0] + 0.5 * n4[0], 0.5 * n3[1] + 0.5 * n4[1]);
            Some([0.5 * m01x + 0.5 * m23x, 0.5 * m01y + 0.5 * m23y])
        };
        // Quantized element-centre → gid index (1e-6), for O(1) child
        // re-location instead of a per-group scan of all elements (np2
        // derefine was spending minutes).
        let mut centre_index: HashMap<(i64, i64), u32> = HashMap::new();
        for (g, rec) in &elems_g {
            if let Some(ec) = elem_centre(rec, &nodes_g) {
                centre_index
                    .entry(((ec[0] * 1e6).round() as i64, (ec[1] * 1e6).round() as i64))
                    .or_insert(*g);
            }
        }
        // Re-locate the 4 children of a recorded group in the CURRENT mesh
        // by matching the recorded child centres (recorded gids are stale
        // after every refine renumbers the element ids).
        let relocate = |centres: &[[f64; 2]; 4]| -> Option<[u32; 4]> {
            let mut out = [0u32; 4];
            for (k, ctr) in centres.iter().enumerate() {
                out[k] = *centre_index
                    .get(&((ctr[0] * 1e6).round() as i64, (ctr[1] * 1e6).round() as i64))?;
            }
            Some(out)
        };
        let mut n_over_thresh = 0usize;
        let mut n_not_leaf = 0usize;
        let mut n_missing_elem = 0usize;
        let mut n_stale_gid = 0usize;
        for &p in &candidates {
            let Some(centres) = tree.child_centres.get(&p) else {
                n_missing_elem += 1;
                continue;
            };
            let Some(ch) = relocate(centres) else {
                n_stale_gid += 1;
                continue;
            };
            let agg: f64 = ch.iter().map(|&c| global_eta[c as usize]).sum();
            if agg >= threshold {
                n_over_thresh += 1;
                continue;
            }
            let mut all_leaf = true;
            for &c in &ch {
                let Some(rec) = elems_g.get(&c) else {
                    n_missing_elem += 1;
                    all_leaf = false;
                    break;
                };
                // Guard against elements referencing nodes outside the
                // partition table (extra nodes): treat them as non-leaf.
                let (Some(n1), Some(n2), Some(n3), Some(n4)) = (
                    nodes_g.get(&rec[1]),
                    nodes_g.get(&rec[2]),
                    nodes_g.get(&rec[3]),
                    nodes_g.get(&rec[4]),
                ) else {
                    all_leaf = false;
                    break;
                };
                // A refined quad's centre node sits at the XY-split centre
                // (0.5·mid01 + 0.5·mid23, each midpoint 0.5·pa + 0.5·pb) —
                // NOT the 4-corner average, which differs by >1e-6 on the
                // curved star.mesh elements.  A child that was itself
                // refined has its centre node present in the mesh; using the
                // average here misses it (quantization) and wrongly treats
                // the child as a leaf, so the parent group gets coarsened
                // while the grandchild survives → the recovered parent
                // coexists with its children (pex15 it1: overlapping
                // elements after the next refine).
                let (m01x, m01y) = (0.5 * n1[0] + 0.5 * n2[0], 0.5 * n1[1] + 0.5 * n2[1]);
                let (m23x, m23y) = (0.5 * n3[0] + 0.5 * n4[0], 0.5 * n3[1] + 0.5 * n4[1]);
                let cx = 0.5 * m01x + 0.5 * m23x;
                let cy = 0.5 * m01y + 0.5 * m23y;
                if node_set.contains(&((cx * 1e6).round() as i64, (cy * 1e6).round() as i64)) {
                    all_leaf = false;
                    break;
                }
            }
            if all_leaf {
                chosen.push(p);
            } else if std::env::var("PEX15_TRACE").is_ok() {
                n_not_leaf += 1;
            }
        }
        if std::env::var("PEX15_TRACE").is_ok() {
            eprintln!("[pex15-deref] chosen={} over_thresh={n_over_thresh} not_leaf={n_not_leaf} missing={n_missing_elem} stale_gid={n_stale_gid}",
                chosen.len());
        }
        // Remove children, restore parents (corners = child k's corner k).
        // gid reuse across refine rounds can make two recorded groups share
        // current elements (a grandchild occupies an old child's gid slot) —
        // skip a group whose children were already removed by another group.
        let mut elems = elems_g;
        let mut executed: Vec<u32> = Vec::new();
        let mut next_gid = elems.keys().max().copied().unwrap_or(0) + 1;
        for &p in &chosen {
            // Re-locate the children in the current mesh by centre
            // coordinates (recorded gids are stale after gid renumbering).
            let Some(centres) = tree.child_centres.get(&p) else { continue };
            let Some(ch) = relocate(centres) else { continue };
            if ch.iter().any(|&c| !elems.contains_key(&c)) {
                continue;
            }
            // Parent corners captured at refine time as coordinates — the
            // rebuild compacted node gids, so match by coordinate to the
            // current node set.
            let parent_corners = tree.parent_corners[&p];
            let mut corner_gids = [0u32; 4];
            for k in 0..4 {
                let c = parent_corners[k];
                corner_gids[k] = nodes_g
                    .iter()
                    .find(|(_, nc)| (nc[0] - c[0]).abs() < 1e-9 && (nc[1] - c[1]).abs() < 1e-9)
                    .map(|(g, _)| *g)
                    .unwrap_or_else(|| panic!("parent corner ({}, {}) not found", c[0], c[1]));
            }
            let tag = elems[&ch[0]][0];
            if std::env::var("PEX15_DBG").is_ok() && (p == 186 || p == 110) {
                let c = |g: u32| nodes_g.get(&g).copied().unwrap_or([f64::NAN, f64::NAN]);
                eprintln!(
                    "[dbg-deref-exec] p={p} children={ch:?} parent_corners_gids={corner_gids:?} \
                     p_corners=({:.9},{:.9})({:.9},{:.9})({:.9},{:.9})({:.9},{:.9})",
                    c(corner_gids[0])[0], c(corner_gids[0])[1],
                    c(corner_gids[1])[0], c(corner_gids[1])[1],
                    c(corner_gids[2])[0], c(corner_gids[2])[1],
                    c(corner_gids[3])[0], c(corner_gids[3])[1],
                );
                for &cc in &ch {
                    if let Some(rec) = elems.get(&cc) {
                        let c2 = |g: u32| nodes_g.get(&g).copied().unwrap_or([f64::NAN, f64::NAN]);
                        eprintln!(
                            "[dbg-deref-exec]   child {cc} nodes {:?} coords ({:.6},{:.6})({:.6},{:.6})({:.6},{:.6})({:.6},{:.6})",
                            &rec[1..], c2(rec[1])[0], c2(rec[1])[1], c2(rec[2])[0], c2(rec[2])[1],
                            c2(rec[3])[0], c2(rec[3])[1], c2(rec[4])[0], c2(rec[4])[1]
                        );
                    }
                }
            }
            elems.remove(&ch[0]);
            elems.remove(&ch[1]);
            elems.remove(&ch[2]);
            elems.remove(&ch[3]);
            // The recorded parent gid `p` may be held by an unrelated element
            // (gid spaces differ across refine rounds) — inserting would
            // silently drop that element.  Use a fresh gid instead; the gids
            // are renumbered by the rebuild anyway (they only order the
            // sorted output).
            let pg_new = if elems.contains_key(&p) {
                next_gid += 1;
                next_gid - 1
            } else {
                p
            };
            elems.insert(pg_new, vec![tag, corner_gids[0], corner_gids[1], corner_gids[2], corner_gids[3]]);
            executed.push(p);
        }
        let chosen = executed;
        // Node compaction: collect referenced nodes in ascending gid.
        let mut ref_nodes: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for rec in elems.values() {
            for k in 0..4 {
                ref_nodes.insert(rec[1 + k]);
            }
        }
        let new_node: HashMap<u32, u32> = ref_nodes
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i as u32))
            .collect();
        // Elements in ascending gid order.
        let mut sorted: Vec<(&u32, &Vec<u32>)> = elems.iter().collect();
        sorted.sort_by_key(|&(g, _)| *g);
        let n_elem = sorted.len() as u32;
        let n_node = ref_nodes.len() as u32;
        bcast.extend_from_slice(&n_elem.to_le_bytes());
        bcast.extend_from_slice(&n_node.to_le_bytes());
        for (g, rec) in sorted {
            bcast.extend_from_slice(&g.to_le_bytes());
            bcast.extend_from_slice(&rec[0].to_le_bytes());
            for k in 0..4 {
                bcast.extend_from_slice(&new_node[&rec[1 + k]].to_le_bytes());
            }
        }
        for &g in &ref_nodes {
            let c = nodes_g[&g];
            bcast.extend_from_slice(&c[0].to_le_bytes());
            bcast.extend_from_slice(&c[1].to_le_bytes());
        }
    }

    // 4. Broadcast + rebuild + re-partition on every rank.
    if comm.size() > 1 {
        comm.broadcast_bytes(0, &mut bcast);
    }
    if bcast.is_empty() {
        return false;
    }
    let mut off = 0usize;
    let rd = |off: &mut usize, n: usize| -> Vec<u8> {
        let v = bcast[*off..*off + n].to_vec();
        *off += n;
        v
    };
    let n_elem = u32::from_le_bytes(rd(&mut off, 4).try_into().unwrap()) as usize;
    let n_node = u32::from_le_bytes(rd(&mut off, 4).try_into().unwrap()) as usize;
    let mut coords = Vec::with_capacity(n_node * 2);
    let mut conn = Vec::with_capacity(n_elem * 4);
    let mut elem_tags = Vec::with_capacity(n_elem);
    for _ in 0..n_elem {
        let _g = u32::from_le_bytes(rd(&mut off, 4).try_into().unwrap());
        let tag = u32::from_le_bytes(rd(&mut off, 4).try_into().unwrap());
        elem_tags.push(tag as i32);
        for _ in 0..4 {
            conn.push(u32::from_le_bytes(rd(&mut off, 4).try_into().unwrap()));
        }
    }
    if std::env::var("PEX15_TRACE").is_ok() {
        let cmax = conn.iter().max().copied().unwrap_or(0);
        eprintln!("[pex15-deref] rebuild parse: n_elem={n_elem} n_node={n_node} conn_max={cmax}");
        // Check for degenerate elements (duplicate corners) in the new mesh.
        let mut deg = 0usize;
        for e in 0..n_elem {
            let ns = &conn[e * 4..e * 4 + 4];
            let mut dup = false;
            for i in 0..4 {
                for j in (i + 1)..4 {
                    if ns[i] == ns[j] { dup = true; }
                }
            }
            if dup { deg += 1; }
        }
        eprintln!("[pex15-deref] degenerate elems = {deg}");
    }
    for _ in 0..n_node {
        let x = f64::from_le_bytes(rd(&mut off, 8).try_into().unwrap());
        let y = f64::from_le_bytes(rd(&mut off, 8).try_into().unwrap());
        coords.push(x);
        coords.push(y);
    }
    // Boundary edges (unshared edges, tag 1).
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for e in 0..n_elem {
        for k in 0..4 {
            let (a, b) = (conn[e * 4 + k], conn[e * 4 + (k + 1) % 4]);
            let key = (a.min(b), a.max(b));
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    for e in 0..n_elem {
        for k in 0..4 {
            let (a, b) = (conn[e * 4 + k], conn[e * 4 + (k + 1) % 4]);
            let key = (a.min(b), a.max(b));
            if edge_count[&key] == 1 {
                face_conn.push(a);
                face_conn.push(b);
                face_tags.push(1);
            }
        }
    }
    let new_mesh = Mesh::<2>::uniform(
        coords, conn, elem_tags, fem_mesh::element_type::ElementType::Quad4,
        face_conn, face_tags, fem_mesh::element_type::ElementType::Line2,
    );
    if std::env::var("PEX15_TRACE").is_ok() {
        // Sanity: element areas must be positive.
        let mut min_det = f64::MAX;
        for e in 0..new_mesh.n_elems() as u32 {
            let ns = new_mesh.elem_nodes(e as ElemId);
            let c = |i: usize| new_mesh.coords_of(ns[i]);
            let det = (c(1)[0] - c(0)[0]) * (c(3)[1] - c(0)[1])
                - (c(1)[1] - c(0)[1]) * (c(3)[0] - c(0)[0]);
            min_det = min_det.min(det);
        }
        eprintln!("[pex15-deref] rebuilt mesh: elems={} nodes={} min_det={:.6e}",
            new_mesh.n_elems(), new_mesh.n_nodes(), min_det);
    }
    let arc = Arc::new(new_mesh);
    *par_mesh = partition_mesh(&arc, comm);
    !chosen.is_empty()
}

/// Merge per-rank hanging constraints into a global `(pa, pb, mid)` table
/// (global node ids), deduplicated, replicated on every rank.
fn merge_hanging_global(
    comm: &Comm,
    partition: &fem_parallel::MeshPartition,
    hc: &[HangingNodeConstraint],
) -> Vec<(u32, u32, u32)> {
    let mut table: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    let n_nodes_part = partition.global_node_ids.len();
    for c in hc {
        // Detect may report nodes beyond the partition table (extra nodes).
        if (c.parent_a as usize) >= n_nodes_part
            || (c.parent_b as usize) >= n_nodes_part
            || (c.constrained as usize) >= n_nodes_part
        {
            continue;
        }
        let pa = partition.global_node(c.parent_a as u32);
        let pb = partition.global_node(c.parent_b as u32);
        let mid = partition.global_node(c.constrained as u32);
        table.insert((pa.min(pb), pa.max(pb), mid));
    }
    if comm.size() > 1 {
        let mut payload = Vec::with_capacity(table.len() * 12);
        for &(a, b, m) in &table {
            payload.extend_from_slice(&a.to_le_bytes());
            payload.extend_from_slice(&b.to_le_bytes());
            payload.extend_from_slice(&m.to_le_bytes());
        }
        let sends: Vec<(Rank, Vec<u8>)> =
            (0..comm.size() as i32).map(|r| (r, payload.clone())).collect();
        for (_src, bytes) in comm.alltoallv_bytes(&sends) {
            for chunk in bytes.chunks_exact(12) {
                let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let m = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                table.insert((a.min(b), a.max(b), m));
            }
        }
    }
    table.into_iter().collect()
}

/// Gather the per-element errors into a global (replicated) vector indexed
/// by global element id; returns also the L1 total error (norm_p = 1).
fn gather_global_eta(
    comm: &Comm,
    par_mesh: &ParallelMesh<Mesh<2>>,
    eta_owned: &[f64],
) -> (Vec<f64>, f64) {
    let partition = par_mesh.partition();
    let n_elems = par_mesh.global_n_elems();
    let mut global = vec![0.0f64; n_elems];
    let mut local_sum = 0.0;
    let mut payload = Vec::with_capacity(eta_owned.len() * 12);
    for (e, &v) in eta_owned.iter().enumerate() {
        let g = partition.global_elem(e as u32);
        payload.extend_from_slice(&g.to_le_bytes());
        payload.extend_from_slice(&v.to_le_bytes());
        local_sum += v;
    }
    let total = comm.allreduce_sum_f64(local_sum);
    if comm.size() > 1 {
        let sends: Vec<(Rank, Vec<u8>)> =
            (0..comm.size() as i32).map(|r| (r, payload.clone())).collect();
        for (_src, bytes) in comm.alltoallv_bytes(&sends) {
            for chunk in bytes.chunks_exact(12) {
                let g = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let e = f64::from_le_bytes(chunk[4..12].try_into().unwrap());
                global[g as usize] = e;
            }
        }
    } else {
        for (e, &v) in eta_owned.iter().enumerate() {
            global[partition.global_elem(e as u32) as usize] = v;
        }
    }
    (global, total)
}
