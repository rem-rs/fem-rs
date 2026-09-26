//! D814-4 (round 80 Lane C) — the DOF partition's pyramid `NDk` (k ≥ 2)
//! face-block order must match `HCurlSpace`'s slot table and MFEM.
//!
//! ## The defect this file pins
//!
//! `nd_face_blocks_for_elem` (`crates/parallel/src/dof_partition.rs`) listed a
//! pyramid's four apex-triangle blocks **before** its base-quad block, while
//! `HCurlSpace` fills the element's slot table base-quad-first — MFEM's
//! `ND_FuentesPyramidElement` constructor orders the slots
//! `edges (8k) → base quad (2k(k−1)) → apex tris (4 × k(k−1)) → interior`
//! (fe_nd.cpp; runtime probe `$HOME/work/d80c/d814_pyramid_nd_dump.txt`:
//! octahedron p = 2, element slots 16..19 = face 0 {1,2,3,4}, 20..27 = the
//! apex tris).  The DP's Pass B slices `space.element_dofs(e)` with an `off`
//! cursor accumulated **in `nd_face_blocks_for_elem` order**, so every pyramid
//! face block read the *wrong slots*: the base-quad block read the last two
//! apex-tri blocks, the first two tri blocks read the base quad, and the last
//! two tri blocks read the first two tris — each face's DOFs keyed to the
//! **wrong face** and, because the two pyramids of a shared base quad shift in
//! opposite element orders on the two ranks, the same DOF landed on different
//! faces per rank.  Ownership, ghost keys and global ids scrambled.
//!
//! ## The fixture (why it has teeth)
//!
//! The D347 octahedron: two positively oriented pyramids sharing the base quad
//! `{1,2,3,4}` — enumerated `(4,3,2,1)` in element 0 and `(1,2,3,4)` in
//! element 1 (the reversed case).  At np = 2 each rank owns one pyramid and
//! carries the other as a ghost, so the shared quad's face DOFs go through the
//! anchor-local path whose slot slicing was wrong.  At np = 1 every face's
//! anchor is local and the (wrong) slicing is at least rank-consistent, so the
//! functional test also pins that np = 1 does not regress.
//!
//! ## Measured
//!
//! Pre-fix: np = 2 cross-rank entry agreement fails (and the functional moves);
//! post-fix all green.  `cargo test --release -p fem-parallel --test
//! d814_pyramid_nd_face_block_order_par -- --nocapture`.

use std::sync::{Arc, Mutex};

use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_mesh::{ElementType, Mesh};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector_assembler::ParVectorAssembler;
use fem_parallel::WorkerConfig;
use fem_space::HCurlSpace;

/// The D347 octahedron: vertex 0 the lower apex `(0,0,−1)`, 1..4 the equator,
/// 5 the upper apex `(0,0,1)`; element 0 = `[4,3,2,1,0]` (base enumerated
/// `4,3,2,1`), element 1 = `[1,2,3,4,5]`.  The shared quad face `{1,2,3,4}` is
/// interior; the eight apex triangles are the boundary.
fn octahedron() -> Mesh<3> {
    let coords = vec![
        0., 0., -1., // 0 lower apex
        1., 0., 0.,  // 1
        0., 1., 0.,  // 2
        -1., 0., 0., // 3
        0., -1., 0., // 4
        0., 0., 1.,  // 5 upper apex
    ];
    let conn: Vec<u32> = vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 5];
    let face_conn: Vec<u32> = vec![
        4, 3, 0, // e0: (0,1,4)
        3, 2, 0, // e0: (1,2,4)
        2, 1, 0, // e0: (2,3,4)
        1, 4, 0, // e0: (3,0,4)
        1, 2, 5, // e1: (0,1,4)
        2, 3, 5, // e1: (1,2,4)
        3, 4, 5, // e1: (2,3,4)
        4, 1, 5, // e1: (3,0,4)
    ];
    let nf = face_conn.len() / 3;
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        ElementType::Pyramid5,
        face_conn,
        (1..=nf as i32).collect(),
        ElementType::Tri3,
    )
}

/// `Σ_ij M_ij` of the NDk mass matrix: serial (unpartitioned), or summed over
/// the ranks' `diag` + `offd` blocks.
fn entry_sum(k: u8, n_ranks: Option<usize>) -> f64 {
    let mesh = octahedron();
    let integ = VectorMassIntegrator { alpha: 1.0 };
    match n_ranks {
        None => VectorAssembler::assemble_bilinear(&HCurlSpace::new(mesh, k), &[&integ], 6)
            .values
            .iter()
            .sum(),
        Some(np) => {
            let out: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
            let out_rank = Arc::clone(&out);
            ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let lm = pmesh.local_mesh().clone();
                let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
                let m = ParVectorAssembler::assemble_bilinear(&s, &[&integ], 6);
                let local = m.diag_block().values.iter().sum::<f64>()
                    + m.offd_block().values.iter().sum::<f64>();
                *out_rank.lock().unwrap() += local;
            });
            let r = *out.lock().unwrap();
            r
        }
    }
}

/// The teeth: the assembled `Σ_ij M_ij = ∫|Σ_i φ_i|²` of the NDk mass matrix
/// agrees with the serial value at np = 1 **and** np = 2.  With the mis-ordered
/// pyramid face blocks the two ranks keyed the shared quad's DOFs (and each
/// apex tri's DOFs) to different faces, so the np = 2 operator's functional
/// moved (or the ghost exchange failed outright).
#[test]
fn d814_pyramid_nd_mass_functional_is_rank_count_independent() {
    for k in [2u8, 3] {
        let ser = entry_sum(k, None);
        let np1 = entry_sum(k, Some(1));
        let np2 = entry_sum(k, Some(2));
        println!(
            "D814 k={k} serial={ser:.17e} np1={np1:.17e} (rel {:.3e}) np2={np2:.17e} \
             (rel {:.3e})",
            (ser - np1).abs() / ser.abs(),
            (ser - np2).abs() / ser.abs(),
        );
        assert!(
            (ser - np1).abs() <= 1e-12 * ser.abs(),
            "k={k}: np = 1 must reproduce the serial assembly, {ser:.17e} vs {np1:.17e}"
        );
        assert!(
            (ser - np2).abs() <= 1e-12 * ser.abs(),
            "k={k}: the parallel assembly must equal the serial one — the pyramid face \
             blocks of the DOF partition must slice the space's slot table in \
             HCurlSpace/MFEM order (base quad before the apex tris).  Measured \
             {ser:.17e} vs {np2:.17e} (rel {:.3e})",
            (ser - np2).abs() / ser.abs()
        );
    }
}

/// The teeth, at entry level: the two ranks' permuted DM mass matrices, mapped
/// to the DP's global DOF ids, must agree **entry by entry** — with the
/// mis-ordered blocks the two ranks keyed the shared quad's face DOFs to
/// different (wrong) faces, so the exchange handed back gids of unrelated DOFs.
#[test]
fn d814_pyramid_nd_cross_rank_global_entries_agree() {
    for k in [2u8, 3] {
        let mesh = octahedron();
        let integ = VectorMassIntegrator { alpha: 1.0 };
        type RankMap = (usize, Vec<((u32, u32), f64)>);
        let maps: Arc<Mutex<Vec<RankMap>>> = Arc::new(Mutex::new(Vec::new()));
        let maps_rank = Arc::clone(&maps);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank() as usize;
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
            let dp = s.dof_partition();
            let m = VectorAssembler::assemble_bilinear(s.local_space(), &[&integ], 6);
            let perm = fem_parallel::par_assembler::permute_csr(&m, dp);
            let mut v: Vec<((u32, u32), f64)> = Vec::new();
            for r in 0..perm.nrows {
                for kk in perm.row_ptr[r]..perm.row_ptr[r + 1] {
                    let c = perm.col_idx[kk] as usize;
                    v.push((
                        (dp.global_dof(r as u32), dp.global_dof(c as u32)),
                        perm.values[kk],
                    ));
                }
            }
            v.sort_by_key(|e| e.0);
            maps_rank.lock().unwrap().push((rank, v));
        });
        let maps = maps.lock().unwrap().clone();
        let m0: std::collections::HashMap<(u32, u32), f64> =
            maps.iter().find(|m| m.0 == 0).expect("rank 0").1.iter().cloned().collect();
        let m1: std::collections::HashMap<(u32, u32), f64> =
            maps.iter().find(|m| m.0 == 1).expect("rank 1").1.iter().cloned().collect();
        assert!(
            !m0.is_empty() && !m1.is_empty(),
            "k={k}: non-vacuity — both ranks must assemble entries"
        );
        let mut diffs = Vec::new();
        for (key, v0) in &m0 {
            let v1 = m1.get(key).copied().unwrap_or(0.0);
            let tol = 1e-12 * v0.abs().max(v1.abs()).max(1e-12);
            if (v0 - v1).abs() > tol {
                diffs.push((*key, *v0, v1));
            }
        }
        println!(
            "D814 k={k} cross-rank: {} / {} entries each, {} disagreements",
            m0.len(),
            m1.len(),
            diffs.len()
        );
        assert!(
            diffs.is_empty(),
            "k={k}: the ranks disagree on {} of {} global entries (first: {:?}) — the DP's \
             pyramid face-block order must match HCurlSpace's slot table (base quad first)",
            diffs.len(),
            m0.len(),
            diffs.iter().take(4).collect::<Vec<_>>()
        );
    }
}

/// Owned-DOF conservation: the per-rank owned counts must still partition the
/// global DOF set — `Σ_ranks n_owned == serial ndofs`.  With the scrambled
/// face keys the ownership came from the *wrong* face's entry, so ranks could
/// claim the same DOF twice (or nobody did).
#[test]
fn d814_pyramid_nd_owned_counts_partition_the_space() {
    for k in [2u8, 3] {
        let mesh = octahedron();
        let serial_ndofs = HCurlSpace::new(mesh.clone(), k).n_dofs();
        let counts: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let counts_rank = Arc::clone(&counts);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
            counts_rank
                .lock()
                .unwrap()
                .push(s.dof_partition().n_owned_dofs);
        });
        let counts = counts.lock().unwrap().clone();
        let sum: usize = counts.iter().sum();
        println!(
            "D814 k={k} serial ndofs={serial_ndofs} np2 owned counts={counts:?} sum={sum}"
        );
        assert_eq!(
            counts.len(),
            2,
            "k={k}: both ranks must report an owned count"
        );
        assert_eq!(
            sum, serial_ndofs,
            "k={k}: Σ n_owned over the ranks must equal the serial DOF count — with the \
             mis-ordered pyramid face blocks the ownership came from the wrong face's \
             entry, so DOFs were double-claimed or dropped"
        );
    }
}
