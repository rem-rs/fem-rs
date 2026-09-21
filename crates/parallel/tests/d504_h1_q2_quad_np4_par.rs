//! D504 — the 2-D quad Q2 distributed essential-set path at np = 4.
//!
//! Registered defect (round 54, `tmp/d124/d504_quad_q2_np4_deadlock.log`):
//! `ParallelFESpace::new` on `Mesh::<2>::unit_square_quad(3)` at np = 4 hangs —
//! all four ranks finish `partition_mesh`, none returns from the space
//! constructor.  `tmp/d124/oracle_report.md` could therefore only pin the quad
//! Q2 boundary set at np = 2.
//!
//! Root cause (r55 lane 1, traced with a per-rank trace): the **contiguous-block
//! sub-mesh extractor** (`crates/parallel/src/par_partition.rs`
//! `extract_submesh_for_rank`, `chunk = n_elems.div_ceil(n_ranks)`) gives 9
//! quads at np = 4 as `3,3,3,0` — rank 3 owns no element, no node, no ghost
//! dof.  The three ghost-dof exchange helpers in
//! `crates/parallel/src/dof_partition.rs` guarded their `alltoallv_bytes`
//! **collectives** with `ghost…is_empty()`, so the empty rank skipped the
//! exchange while ranks 0..2 blocked inside it.
//!
//! Fix: the three helpers now only short-circuit for `comm.size() <= 1`; every
//! rank joins the exchange (an empty request list is a legal alltoallv
//! payload).  `par_partition.rs`'s block distribution is left untouched — it is
//! a load-balance question (registered as D527), not a correctness one, and
//! changing it would shift every np >= 2 baseline in the repo.
//!
//! MFEM MPI oracle (`tmp/r55/d504_quad_q2_probe.cpp`, mfem410_mpi, dumps in
//! `tmp/r55/q2_np{1,2,4}.txt` / `q2_4_np{1,2,4}.txt`):
//!
//! ```text
//! mesh=3x3 quad order=2 np=1 ne=9  GlobalTrueVSize=49 sum_ess_true=24 union_keys=24
//! mesh=3x3 quad order=2 np=2 ne=9  GlobalTrueVSize=49 sum_ess_true=24 union_keys=24
//! mesh=3x3 quad order=2 np=4 ne=9  GlobalTrueVSize=49 sum_ess_true=24 union_keys=24
//! mesh=4x4 quad order=2 np=1..4 ne=16 GlobalTrueVSize=81 sum_ess_true=32 union_keys=32
//! ```
//! (the 3x3/4x4 key tables are byte-identical across np; `diff` archived in
//! `tmp/r55/`).
//!
//! The 24 MFEM keys of the 3x3 mesh are pinned inline below (`MFEM_Q2_KEYS`).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::ParAssembler;
use fem_parallel::par_csr::ParCsrMatrix;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector::ParVector;
use fem_parallel::WorkerConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::dof_manager::DofManager;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

type Key = [i64; 3];

/// MFEM `GetBoundaryTrueDofs` union keys, `PROBE mesh=3x3 quad order=2`
/// (24 = 12 boundary vertices + 12 boundary edge midpoints).
const MFEM_Q2_KEYS: &[[i64; 2]] = &[
    [0, 0],
    [0, 166667],
    [0, 333333],
    [0, 500000],
    [0, 666667],
    [0, 833333],
    [0, 1000000],
    [166667, 0],
    [166667, 1000000],
    [333333, 0],
    [333333, 1000000],
    [500000, 0],
    [500000, 1000000],
    [666667, 0],
    [666667, 1000000],
    [833333, 0],
    [833333, 1000000],
    [1000000, 0],
    [1000000, 166667],
    [1000000, 333333],
    [1000000, 500000],
    [1000000, 666667],
    [1000000, 833333],
    [1000000, 1000000],
];

fn key3(c: &[f64]) -> Key {
    [
        (c[0] * 1e6).round() as i64,
        (c[1] * 1e6).round() as i64,
        (c.get(2).copied().unwrap_or(0.0) * 1e6).round() as i64,
    ]
}

fn mfem_key_set() -> HashSet<Key> {
    MFEM_Q2_KEYS.iter().map(|k| [k[0], k[1], 0]).collect()
}

/// Per-rank statistics of the parallel space: the serial whole-mesh essential
/// set, the union of the rank-owned essential true dofs (keyed by dof
/// coordinates), and the owned/ghost dof counts per rank.
fn run_quad<const DIM: usize>(
    mesh: Mesh<DIM>,
    n_ranks: usize,
    order: u8,
    tags: &[i32],
    tag: i32,
) -> (HashSet<Key>, Vec<(usize, usize)>) {
    let serial_dm = DofManager::new(&mesh, order);
    let serial: HashSet<Key> = boundary_dofs(&mesh, &serial_dm, tags)
        .into_iter()
        .map(|d| key3(&serial_dm.dof_coord(d)))
        .collect();

    let out: Arc<Mutex<Option<(HashSet<Key>, Vec<(usize, usize)>)>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    let tags_vec: Vec<i32> = tags.to_vec();
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let dm = DofManager::new(&lm, order);
        let space = H1Space::new(lm.clone(), order);
        let par = ParallelFESpace::new(space, &pmesh, comm.clone());
        let dp = par.dof_partition();
        let local = par.essential_true_dofs(&tags_vec);
        let mut mine: HashSet<Key> = HashSet::new();
        for d in local {
            let pid = dp.permute_dof(d);
            if (pid as usize) < dp.n_owned_dofs {
                mine.insert(key3(&dm.dof_coord(d)));
            }
        }
        if comm.rank() == 0 {
            let mut union: HashSet<Key> = mine;
            let mut stats: Vec<(usize, usize)> = vec![(dp.n_owned_dofs, dp.n_total_dofs())];
            for src in 1..comm.size() as i32 {
                let flat: Vec<i64> = comm.recv(src, tag);
                union.extend(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
                let st: Vec<i64> = comm.recv(src, tag + 1);
                stats.push((st[0] as usize, st[1] as usize));
            }
            // 1. every rank's set is a subset of the serial set (no bogus dof),
            // 2. the union is exactly the serial set (no missed cross-rank dof).
            assert!(
                union.is_subset(&serial),
                "np{n_ranks}: union contains {} keys outside the serial set",
                union.difference(&serial).count()
            );
            assert_eq!(
                union, serial,
                "np{n_ranks}: union != serial whole-mesh essential set \
                 (missing {})",
                serial.difference(&union).count()
            );
            *out2.lock().unwrap() = Some((union, stats));
        } else {
            let flat: Vec<i64> = mine.iter().flat_map(|k| k.to_vec()).collect();
            comm.send(0, tag, &flat);
            comm.send(0, tag + 1, &[dp.n_owned_dofs as i64, dp.n_total_dofs() as i64]);
        }
    });
    let result = out.lock().unwrap().take().unwrap();
    result
}

// ── the minimal deadlock reproducer ──────────────────────────────────────────

/// Pre-fix this test never returns: on `unit_square_quad(3)` at np = 4 the
/// sub-mesh extractor hands rank 3 an empty partition, and the (then
/// rank-conditional) ghost-dof alltoallv left ranks 0..2 blocked forever.
/// Reproduce the hang by running this test under a shell timeout
/// (`timeout 180 cargo test -p fem-parallel --test d504_h1_q2_quad_np4_par
/// d504_h1_q2_quad_np4_empty_rank -- --test-threads=1 --nocapture`).
#[test]
fn d504_h1_q2_quad_np4_empty_rank() {
    let (union, stats) = run_quad(Mesh::<2>::unit_square_quad(3), 4, 2, &[1, 2, 3, 4], 700);
    assert_eq!(union.len(), 24, "np4 union must be MFEM's 24 essential dofs");
    assert_eq!(stats.len(), 4, "one stat entry per rank");
    assert_eq!(
        stats.iter().map(|s| s.0).sum::<usize>(),
        49,
        "owned dofs must partition the 49 serial dofs"
    );
    assert!(
        stats.iter().any(|s| s.0 == 0),
        "this mesh at np=4 is expected to leave one rank empty \
         (the D504 trigger); stats = {stats:?}"
    );
}

// ── MFEM comparison + np invariance ──────────────────────────────────────────

#[test]
fn d504_h1_q2_quad_boundary_set_matches_mfem_np1_np2_np4() {
    let mesh = Mesh::<2>::unit_square_quad(3);
    let mfem = mfem_key_set();
    assert_eq!(mfem.len(), 24);
    for np in [1usize, 2, 4] {
        let (union, _) = run_quad(mesh.clone(), np, 2, &[1, 2, 3, 4], 701);
        assert_eq!(
            union, mfem,
            "np{np}: fem-rs essential true-dof set differs from \
             MFEM ParFiniteElementSpace::GetBoundaryTrueDofs"
        );
    }
}

/// The 4x4 quad mesh gives every rank elements (`16 / 4 = 4`), so this covers
/// the halo with four **active** ranks — the empty-rank case of the test above
/// degenerates to three active ranks.
#[test]
fn d504_h1_q2_quad_four_active_ranks_boundary_set() {
    let mesh = Mesh::<2>::unit_square_quad(4);
    for np in [1usize, 2, 4] {
        let (union, stats) = run_quad(mesh.clone(), np, 2, &[1, 2, 3, 4], 702);
        assert_eq!(
            union.len(),
            32,
            "np{np}: MFEM pins the 4x4 quad Q2 essential set at 32"
        );
        if np == 4 {
            assert!(
                stats.iter().all(|s| s.0 > 0),
                "all four ranks must own dofs here; stats = {stats:?}"
            );
        }
    }
}

// ── solve drift (user-visible symptom) ───────────────────────────────────────

/// Solve `u − Δu = 1` with an inhomogeneous Dirichlet lift on the 4x4 quad mesh
/// and compare the owned solutions across np — the pre-D124/D504 pipeline
/// produced a nonsymmetric operator (missed cross-rank essential columns) and
/// diverged.
fn solve_quad(n_ranks: usize) -> HashMap<Key, f64> {
    let mesh = Mesh::<2>::unit_square_quad(4);
    let order = 2u8;
    let g = |x: &[f64]| x[0] * x[0] * x[0] + 2.0 * x[1] + (5.0 * x[0]).sin() * (4.0 * x[1]).sin();
    let out: Arc<Mutex<Option<HashMap<Key, f64>>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let dm = DofManager::new(&lm, order);
        let space = H1Space::new(lm.clone(), order);
        let par = ParallelFESpace::new_with_dof_manager(space, &pmesh, &dm, comm.clone());
        let dp = par.dof_partition();
        let n_total = dp.n_total_dofs();
        let ghost_arc = par.dof_ghost_exchange_arc();

        let mass = MassIntegrator { rho: 1.0 };
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&par, &[&mass, &diff], 2 * order + 1);
        let src = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
        let mut b = ParAssembler::assemble_linear(&par, &[&src], 2 * order + 2);

        let all_tags: Vec<i32> = (1..=4).collect();
        let mut owned_ess: Vec<(usize, f64)> = Vec::new();
        let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
        for &d in par.essential_true_dofs(&all_tags).iter() {
            let pid = dp.permute_dof(d) as usize;
            let val = g(dm.dof_coord(d));
            if pid < dp.n_owned_dofs {
                owned_ess.push((pid, val));
            } else {
                ghost_ess.push((pid - dp.n_owned_dofs, val));
            }
        }
        for &(pid, val) in &owned_ess {
            a.apply_dirichlet_par_keep_diag(pid, val, &mut b);
        }
        a.apply_ghost_ess_columns(&ghost_ess, &mut b);

        let mk = || {
            ParVector::from_local_raw(
                vec![0.0; n_total],
                dp.n_owned_dofs,
                ghost_arc.clone(),
                comm.clone(),
            )
        };
        let mut x = mk();
        let mut r = mk();
        let mut p = mk();
        let mut ap = mk();
        let dot_owned = |v: &ParVector, w: &ParVector| -> f64 {
            let local: f64 = v.as_slice()[..dp.n_owned_dofs]
                .iter()
                .zip(&w.as_slice()[..dp.n_owned_dofs])
                .map(|(&a, &b)| a * b)
                .sum();
            comm.allreduce_sum_f64(local)
        };
        let spmv = |a: &ParCsrMatrix, v: &mut ParVector, y: &mut ParVector| {
            v.update_ghosts();
            a.spmv(v, y);
        };
        spmv(&a, &mut x, &mut r);
        for i in 0..dp.n_owned_dofs {
            r.as_slice_mut()[i] = b.as_slice()[i] - r.as_slice()[i];
        }
        let mut rr = dot_owned(&r, &r);
        let rr0 = rr.max(1e-30);
        p.as_slice_mut().copy_from_slice(r.as_slice());
        for _ in 0..5000 {
            spmv(&a, &mut p, &mut ap);
            let alpha = rr / dot_owned(&p, &ap).max(1e-300);
            for i in 0..dp.n_owned_dofs {
                x.as_slice_mut()[i] += alpha * p.as_slice()[i];
                r.as_slice_mut()[i] -= alpha * ap.as_slice()[i];
            }
            let rr_new = dot_owned(&r, &r);
            if rr_new.sqrt() / rr0.sqrt() < 1e-12 {
                break;
            }
            let beta = rr_new / rr;
            for i in 0..dp.n_owned_dofs {
                p.as_slice_mut()[i] = r.as_slice()[i] + beta * p.as_slice()[i];
            }
            rr = rr_new;
        }
        x.update_ghosts();
        let map: HashMap<Key, f64> = (0..dp.n_owned_dofs)
            .map(|pid| {
                let d = dp.unpermute_dof(pid as u32);
                (key3(&dm.dof_coord(d)), x.as_slice()[pid])
            })
            .collect();
        if comm.rank() == 0 {
            let mut all = map;
            for src in 1..comm.size() as i32 {
                let flat: Vec<f64> = comm.recv(src, 704);
                for c in flat.chunks_exact(4) {
                    all.insert([c[0] as i64, c[1] as i64, c[2] as i64], c[3]);
                }
            }
            *out2.lock().unwrap() = Some(all);
        } else {
            let flat: Vec<f64> = map
                .iter()
                .flat_map(|(k, &v)| [k[0] as f64, k[1] as f64, k[2] as f64, v])
                .collect();
            comm.send(0, 704, &flat);
        }
    });
    let result = out.lock().unwrap().take().unwrap();
    result
}

#[test]
fn d504_h1_q2_quad_dirichlet_solve_np1_np2_np4_agree() {
    let s1 = solve_quad(1);
    assert_eq!(s1.len(), 81, "4x4 quad Q2 has 81 dofs");
    for np in [2usize, 4] {
        let sn = solve_quad(np);
        assert_eq!(s1.len(), sn.len(), "dof count mismatch np1 vs np{np}");
        let mut max_diff = 0.0_f64;
        let mut n_nonfinite = 0usize;
        for (k, &v1) in &s1 {
            let v2 = sn[k];
            if !v2.is_finite() {
                n_nonfinite += 1;
                continue;
            }
            max_diff = max_diff.max((v1 - v2).abs());
        }
        assert_eq!(n_nonfinite, 0, "np{np} solution has non-finite entries");
        assert!(
            max_diff < 1e-9,
            "solution drift np1 vs np{np}: max|Δu| = {max_diff:.3e}"
        );
    }
}
