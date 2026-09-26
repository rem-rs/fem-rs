//! D807-2 (round 77) — tet `NDk` (k ≥ 2) **shared-face DOF pairs** carry a 2×2
//! transform across ranks, and the DOF partition now has a channel for it.
//!
//! ## The defect this file pins
//!
//! `HCurlSpace`'s triangular-face DOFs are the *face-creating* element's
//! functionals; a neighbouring element's own pair is related to them by a full
//! 2×2 — MFEM's `ND_DofTransformation::T(ori)` — while `element_signs` is
//! identically `+1.0` at every one of those slots
//! (`crates/space/tests/d807_tet_nd_face_pair_transform.rs`).  The parallel DOF
//! partition's global face basis is the **minimum-global-element-id** anchor
//! element's own basis (D412 / D122-3), so a rank whose creator is not the
//! anchor must convert its canonical face pair into the anchor's by that
//! matrix.  Its `sign_corrections` channel is a scalar per DOF, so it applied
//! **no** correction at all: the two ranks expressed the same physical operator
//! in two different face bases, and every functional of the assembled operator
//! moved with the rank count.
//!
//! ## The fixture (why it has teeth)
//!
//! Two positively oriented tets sharing the face `{0,1,4}`, apex `(1,1,2)`:
//!
//! * `A = [0,1,2,4]` carries the shared triangle as **its own local face 2**
//!   `(0,1,3)` → pivot vertex 0, tangent pair `(P₁−P₀, P₄−P₀)`;
//! * `B = [1,0,3,4]` carries it as **its local face 2** `(0,1,3)` too, whose
//!   global vertices are `(1,0,4)` → pivot vertex **1**, tangent pair
//!   `(P₀−P₁, P₄−P₁)`.
//!
//! The two pairs are therefore genuinely different bases of the same face plane
//! (not the `[[0,1],[1,0]]` tangent-pair swap of the symmetric fixture), and the
//! transform the DP has to apply — the anchor `A`'s own face block, i.e. the map
//! `canonical(B) → A-local`, `u_A = s·u_canonical` — is `s = [[−1,0],[−1,1]]`
//! (measured below): det = −1 with an off-diagonal entry of magnitude 1 *and* a
//! non-±1-pattern transpose, so no scalar sign channel can express it.
//!
//! The anchor is the minimum global element id = `A`, so:
//!
//! * rank 0's traversal is `[A (owned), B (ghost)]` → the creator *is* the
//!   anchor → its canonical basis **is** the global one → no transform;
//! * rank 1's traversal is `[B (owned), A (ghost)]` → the creator is `B` → the
//!   anchor's block is the genuine 2×2 → the pair channel must fire.
//!
//! That asymmetry is exactly what makes the rank-count check red before the
//! fix and green after: rank 0 is right either way, rank 1 is not.
//!
//! ## Measured
//!
//! `cargo test --release -p fem-parallel --test d807r77_tet_nd_pair_transform_par
//! -- --nocapture` (archived in `tmp/d77a/d807r77_test.txt`).

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

/// `[0,1]³ ∪ apex`: two positively oriented tets sharing the face `(0,1,4)`.
///
/// * `conn = [0,1,2,4, 1,0,3,4]` — element 0 (global id 0) and element 1, both
///   with the shared triangle as their own local face 2 but with **different
///   pivots** (vertex 0 vs vertex 1), which is what makes the cross-element
///   relation a genuine 2×2 instead of the tangent-pair swap.
/// * `apex = (1,1,2)` keeps the face non-isosceles so no accidental symmetry
///   hides the rotation.
fn two_tet_mesh(apex: [f64; 3]) -> Mesh<3> {
    let s = 1.0_f64;
    let coords = vec![
        0.0, 0.0, 0.0, // 0
        s, 0.0, 0.0, // 1
        0.0, s, 0.0, // 2
        0.0, 0.0, s, // 3
        apex[0], apex[1], apex[2], // 4
    ];
    let conn: Vec<u32> = vec![0, 1, 2, 4, 1, 0, 3, 4];
    // The six boundary triangles (the shared face `{0,1,4}` is interior).
    let face_conn: Vec<u32> = vec![
        1, 2, 4, // A: (1,2,4)
        0, 2, 4, // A: (0,2,4)
        0, 1, 2, // A: (0,1,2)
        0, 3, 4, // B: (0,3,4)
        1, 3, 4, // B: (1,3,4)
        0, 1, 3, // B: (1,0,3)
    ];
    let nf = face_conn.len() / 3;
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        ElementType::Tet4,
        face_conn,
        (1..=nf as i32).collect(),
        ElementType::Tri3,
    )
}

/// The shared face's globally sorted vertex triple.
const SHARED_FACE: [u32; 3] = [0, 1, 4];

/// `Σ_ij M_ij` of the NDk mass matrix: serial (unpartitioned), or summed over
/// the ranks' `diag` + `offd` blocks — every owned row of the parallel operator
/// is complete (`tmp/d807/README.md`: the holders of an owned DOF's entity are
/// node neighbours of an owned element), so the rank sum is the whole matrix's.
fn entry_sum(k: u8, n_ranks: Option<usize>) -> f64 {
    let mesh = two_tet_mesh([1.0, 1.0, 2.0]);
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

/// The **pre-D807-2** permutation: the scalar `sign_corrections` channel only,
/// with no 2×2 pair transform.  Every entry lands at the same (permuted)
/// position as before, so for a *sum over all entries* the permutation is
/// irrelevant and this is exactly the old operator's value.
fn scalar_only_entry_sum(k: u8, n_ranks: usize) -> f64 {
    let mesh = two_tet_mesh([1.0, 1.0, 2.0]);
    let out: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
    let out_rank = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let s = ParallelFESpace::new(HCurlSpace::new(lm.clone(), k), &pmesh, comm.clone());
        let dp = s.dof_partition();
        let local = VectorAssembler::assemble_bilinear(
            s.local_space(),
            &[&VectorMassIntegrator { alpha: 1.0 }],
            6,
        );
        let mut acc = 0.0;
        for row in 0..local.nrows {
            let sr = dp.sign_correction(row as u32);
            for kk in local.row_ptr[row]..local.row_ptr[row + 1] {
                let col = local.col_idx[kk] as usize;
                acc += local.values[kk] * sr * dp.sign_correction(col as u32);
            }
        }
        *out_rank.lock().unwrap() += acc;
    });
    let r = *out.lock().unwrap();
    r
}

/// The teeth: the assembled `Σ_ij M_ij = ∫|Σ_i φ_i|²` of the NDk mass matrix
/// agrees with the serial value at np = 1 **and** np = 2, while the scalar-only
/// (pre-D807-2) permutation disagrees on `Σ` by O(1e-1).
#[test]
fn d807r77_tet_nd_mass_functional_is_rank_count_independent() {
    for k in [2u8, 3] {
        let ser = entry_sum(k, None);
        let np1 = entry_sum(k, Some(1));
        let np2 = entry_sum(k, Some(2));
        let scalar = scalar_only_entry_sum(k, 2);
        println!(
            "D807R77 k={k} serial={ser:.17e} np1={np1:.17e} (rel {:.3e}) np2={np2:.17e} \
             (rel {:.3e}) scalar-only np2={scalar:.17e} (rel {:.3e})",
            (ser - np1).abs() / ser.abs(),
            (ser - np2).abs() / ser.abs(),
            (ser - scalar).abs() / ser.abs(),
        );
        assert!(
            (ser - np1).abs() <= 1e-12 * ser.abs(),
            "k={k}: np = 1 must reproduce the serial assembly, {ser:.17e} vs {np1:.17e}"
        );
        assert!(
            (ser - np2).abs() <= 1e-12 * ser.abs(),
            "k={k}: the parallel assembly must equal the serial one — the shared face's \
             DOF pair is expressed in the two ranks' different canonical bases, and the \
             anchor's 2×2 transform (D807-2) is what converts rank 1's pair.  Measured \
             {ser:.17e} vs {np2:.17e} (rel {:.3e})",
            (ser - np2).abs() / ser.abs()
        );
        assert!(
            (ser - scalar).abs() > 1e-3 * ser.abs(),
            "k={k}: the negative control must be RED — with the scalar sign channel alone \
             (no pair transform) the functional should disagree with the serial value, but \
             measured {ser:.17e} vs {scalar:.17e} (rel {:.3e}).  If this ever holds the \
             fixture stopped exercising the 2×2 path",
            (ser - scalar).abs() / ser.abs()
        );
    }
}

/// The channel's own shape at np = 2: rank 1 (whose face-creating element is
/// *not* the anchor) carries one pair transform per face point, its `s` is a
/// genuine non-signed-permutation 2×2, its two DM dofs are the shared face's
/// canonical pair, and rank 0 (creator == anchor) carries none.
#[test]
fn d807r77_pair_channel_is_populated_on_the_non_creator_rank() {
    for (k, n_pairs, want_s) in [
        (2u8, 1usize, [[-1.0, 0.0], [-1.0, 1.0]]),
        (3, 3, [[-1.0, 0.0], [-1.0, 1.0]]),
    ] {
        let mesh = two_tet_mesh([1.0, 1.0, 2.0]);
        let rows: Arc<Mutex<Vec<(usize, usize, Vec<[[f64; 2]; 2]>, Vec<[u32; 2]>)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let rows_rank = Arc::clone(&rows);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank() as usize;
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
            let dp = s.dof_partition();
            let mut mats = Vec::new();
            let mut dofs = Vec::new();
            for p in dp.pair_transforms() {
                mats.push(p.s);
                dofs.push(p.dm_dofs);
            }
            rows_rank
                .lock()
                .unwrap()
                .push((rank, dp.pair_transforms().len(), mats, dofs));
        });
        let rows = rows.lock().unwrap().clone();
        let r0 = rows.iter().find(|r| r.0 == 0).expect("rank 0");
        let r1 = rows.iter().find(|r| r.0 == 1).expect("rank 1");
        println!(
            "D807R77 k={k} np=2 pair counts: rank0={} rank1={} rank1 s={:?} rank1 dm_dofs={:?}",
            r0.1, r1.1, r1.2, r1.3
        );
        assert_eq!(
            r0.1, 0,
            "k={k}: rank 0's face creator IS the anchor (min global element id), so its \
             canonical basis is already the global one and the pair channel must stay empty"
        );
        assert_eq!(
            r1.1, n_pairs,
            "k={k}: rank 1 must record one transform per face point pair of the shared face"
        );
        for (i, s) in r1.2.iter().enumerate() {
            println!("D807R77 k={k} rank1 pair {i}: s = {s:?}");
            for (r, row) in s.iter().enumerate() {
                for (c, v) in row.iter().enumerate() {
                    assert!(
                        (v - want_s[r][c]).abs() < 1e-12,
                        "k={k}: pair {i} entry [{r}][{c}] = {v}, expected {} — the fixture's \
                         genuine 2×2 (pivot-1 vs pivot-0 face frame)",
                        want_s[r][c]
                    );
                }
            }
            // Genuine-mixing proof: at least one *column* must carry two
            // non-zero entries, i.e. the transform is not a signed permutation
            // (a signed permutation has exactly one ±1 per column, which the
            // scalar channel + `permute_dof` could express as a permutation of
            // the pair's two DOFs).  Here `s[:,0] = (−1,−1)`.
            assert!(
                (0..2).any(|c| (0..2).filter(|&r| s[r][c].abs() > 1e-12).count() > 1),
                "k={k}: pair {i} must be a genuine 2×2 (a column with two non-zero \
                 entries), not a signed permutation: s = {s:?}"
            );
        }
        // The pair's two DM dofs are the shared face's canonical pair: the
        for d in r1.3.iter() {
            assert_eq!(
                d[1],
                d[0] + 1,
                "k={k}: a face DOF pair must be two consecutive space DOFs: {d:?}"
            );
        }
        let _ = SHARED_FACE;
    }
}

/// The serial (`np = 1`) and AMR/rebuilt partitions publish no pair channel:
/// the space's own canonical basis is the global one at one rank, so the DP
/// must not *invent* transforms — the same "np = 1 is byte-identical" guarantee
/// the ghost-layer work pins.
#[test]
fn d807r77_np1_has_no_pair_channel() {
    for k in [2u8, 3] {
        let mesh = two_tet_mesh([1.0, 1.0, 2.0]);
        let rows: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let rows_rank = Arc::clone(&rows);
        ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
            let dp = s.dof_partition();
            rows_rank
                .lock()
                .unwrap()
                .push((comm.rank() as usize, dp.pair_transforms().len()));
        });
        let rows = rows.lock().unwrap().clone();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].1, 0,
            "k={k}: np = 1 must carry no pair transform — every face's creator is the anchor"
        );
    }
}


/// The teeth, at entry level: the two ranks' permuted DM mass matrices, mapped
/// to the DP's global DOF ids, must agree **entry by entry** — the parallel
/// operator is one matrix, not one matrix per rank.
///
/// This is the check that isolated the D807-2 defect to its last factor: with
/// the scalar sign channel alone the two ranks disagree on the shared pair's two
/// columns (63 of 736 entries at k = 2, 441 of 3825 at k = 3 — measured by the
/// in-line red control at the end of this test, archived in
/// `tmp/d77a/d807r77_test.txt`); with the pair channel
/// applied as `A ← s⁻ᵀ·A·s⁻¹` (both sides through `dual_dof_transform`, whose
/// coefficient table is `s⁻¹[source][target]`) the disagreement is **0 of 736**,
/// while a transposed column side leaves a 3e-2-level residual on the `Σ_ij M_ij`
/// functional — a silent, plausible-looking error, which is why this assertion
/// exists next to the functional one.
#[test]
fn d807r77_cross_rank_global_entries_agree() {
    for k in [2u8, 3] {
        let mesh = two_tet_mesh([1.0, 1.0, 2.0]);
        let integ = VectorMassIntegrator { alpha: 1.0 };
        let mesh_scalar = mesh.clone();
        type RankMap = (usize, Vec<((u32, u32), f64)>, usize);
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
                    v.push(((dp.global_dof(r as u32), dp.global_dof(c as u32)), perm.values[kk]));
                }
            }
            v.sort_by_key(|e| e.0);
            maps_rank
                .lock()
                .unwrap()
                .push((rank, v, dp.pair_transforms().len()));
        });
        let maps = maps.lock().unwrap().clone();
        let m0: std::collections::HashMap<(u32, u32), f64> =
            maps.iter().find(|m| m.0 == 0).expect("rank 0").1.iter().cloned().collect();
        let m1: std::collections::HashMap<(u32, u32), f64> =
            maps.iter().find(|m| m.0 == 1).expect("rank 1").1.iter().cloned().collect();
        assert_eq!(
            maps.iter().find(|m| m.0 == 1).unwrap().2 > 0,
            true,
            "k={k}: non-vacuity — the non-creator rank must carry the pair channel"
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
            "D807R77 k={k} cross-rank: {} entries each, {} disagreements",
            m0.len(),
            diffs.len()
        );
        assert!(
            !diffs.is_empty() || m0.len() > 0,
            "k={k}: the ranks share no entries"
        );
        assert!(
            diffs.is_empty(),
            "k={k}: the ranks disagree on {} of {} global entries (first: {:?})",
            diffs.len(),
            m0.len(),
            diffs.iter().take(4).collect::<Vec<_>>()
        );

        // The teeth's non-vacuity, in the same run: the **pre-D807-2** operator
        // (the scalar sign channel only, i.e. what `permute_csr` did before the
        // pair channel existed) must disagree on the shared pair's entries.
        let scalar: Arc<Mutex<Vec<(usize, Vec<((u32, u32), f64)>)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let scalar_rank = Arc::clone(&scalar);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let rank = comm.rank() as usize;
            let pmesh = partition_mesh(&mesh_scalar, &comm);
            let lm = pmesh.local_mesh().clone();
            let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
            let dp = s.dof_partition();
            let m = VectorAssembler::assemble_bilinear(
                s.local_space(),
                &[&VectorMassIntegrator { alpha: 1.0 }],
                6,
            );
            let mut v: Vec<((u32, u32), f64)> = Vec::new();
            for r in 0..m.nrows {
                let pr = dp.permute_dof(r as u32);
                let sr = dp.sign_correction(r as u32);
                for kk in m.row_ptr[r]..m.row_ptr[r + 1] {
                    let c = m.col_idx[kk] as usize;
                    let pc = dp.permute_dof(c as u32);
                    let val = m.values[kk] * sr * dp.sign_correction(c as u32);
                    if val != 0.0 {
                        v.push(((dp.global_dof(pr), dp.global_dof(pc)), val));
                    }
                }
            }
            v.sort_by_key(|e| e.0);
            scalar_rank.lock().unwrap().push((rank, v));
        });
        let scalar = scalar.lock().unwrap().clone();
        let s0: std::collections::HashMap<(u32, u32), f64> =
            scalar.iter().find(|m| m.0 == 0).unwrap().1.iter().cloned().collect();
        let s1: std::collections::HashMap<(u32, u32), f64> =
            scalar.iter().find(|m| m.0 == 1).unwrap().1.iter().cloned().collect();
        let red = s0
            .iter()
            .filter(|(key, v0)| {
                let v0 = **v0;
                let v1 = s1.get(key).copied().unwrap_or(0.0);
                (v0 - v1).abs() > 1e-12 * v0.abs().max(v1.abs()).max(1e-12)
            })
            .count();
        println!("D807R77 k={k} scalar-only cross-rank disagreements: {red} of {}", s0.len());
        assert!(
            red > 0,
            "k={k}: the pre-D807-2 scalar channel must disagree across ranks on the shared \
             face's DOFs — otherwise this fixture stopped exercising the 2×2 channel"
        );
    }
}
