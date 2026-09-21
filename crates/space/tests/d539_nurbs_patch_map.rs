//! D539: `NurbsExtension::patch_dof` — MFEM's `NURBSPatchMap::operator()` in
//! DOF mode — must return a **legal** global DOF over its *full* index domain
//! (`0 <= multi[d] < NCP[d]`), in 1-D exactly as in 2-D/3-D, and it must agree
//! with the element DOF table route ([`NurbsExtension::patch_local_dofs`]).
//!
//! Before the fix the raw `NURBSPatchMap` value was returned un-compacted:
//! fem-rs' raw 1-D numbering carries the interior slot of the mesh's unique
//! edge (MFEM's 1-D patch topology has *no* edge entities —
//! `patchTopo->GetNEdges() == 0` — so its raw numbering does not), that slot
//! stays inactive in the element-table compaction, and e.g. the once-refined
//! `segment-nurbs.mesh` (NCP 3, 3 active DOFs) had `patch_dof(0,[1]) == 3` —
//! one past the end (`index out of bounds: len is 3 but index is 3`, round 55
//! ③路).  MFEM's element table pushes every `p2g` value through the
//! `activeDof` compaction; fem-rs applies both maps in `patch_dof`
//! (2-D/3-D conforming offsets are fully used, so the compaction is the
//! identity there and nothing changes).
//!
//! MFEM 4.10 truth (`tmp/d531/d539_probe.cpp`, compiled against
//! `/home/quan/mfem410_ser`): the once-refined segment's compacted patch map
//! is `[0, 2, 1]` — the interior control point is DOF 2 and the last one
//! DOF 1 (MFEM's `Or1D(i1, I, opatch = 0)` reversal) — and the refined-twice
//! map is `[0, 4, 3, 2, 1]`.

use fem_space::nurbs_extension::NurbsExtension;

const SEGMENT: &str = include_str!("../../../data/segment-nurbs.mesh");
const SQUARE: &str = include_str!("../../../data/square-nurbs.mesh");
const DISC: &str = include_str!("../../../data/disc-nurbs.mesh");
const SQUARE_PATCH: &str = include_str!("../../../data/square-disc-nurbs-patch.mesh");
const BALL: &str = include_str!("../../../data/ball-nurbs.mesh");

/// The full `0..NCP` index domain of every patch, `patch_dof` against
/// `patch_local_dofs` (the element-table route), and every value in range.
fn check_patch_maps(ext: &NurbsExtension) {
    let dim = ext.dim();
    for p in 0..ext.n_patches() {
        let kvs = ext.patch_knot_vectors(p).expect("patch knot vectors");
        let ncps: Vec<usize> = kvs.iter().map(|k| k.ncp()).collect();
        let local = ext.patch_local_dofs(p).expect("patch local dofs");
        let lookup: std::collections::HashMap<&[usize], usize> =
            local.iter().map(|(m, g)| (m.as_slice(), *g)).collect();
        assert_eq!(lookup.len(), ncps.iter().product::<usize>(), "patch {p}");

        let mut multi = vec![0usize; dim];
        'domains: loop {
            let dof = ext
                .patch_dof(p, &multi)
                .unwrap_or_else(|e| panic!("patch {p} dof {multi:?}: {e}"));
            assert!(dof < ext.n_dofs(), "patch {p} dof {multi:?} = {dof} out of range");
            assert_eq!(
                Some(&dof),
                lookup.get(multi.as_slice()),
                "patch {p} dof {multi:?} disagrees with the element table"
            );
            // Odometer over the per-direction NCP ranges.
            for d in 0..dim {
                multi[d] += 1;
                if multi[d] < ncps[d] {
                    continue 'domains;
                }
                multi[d] = 0;
            }
            break;
        }
    }
}

#[test]
fn refined_1d_patch_dof_is_legal_and_matches_mfem() {
    // The D539 minimal failing case: the once-refined segment is a 3-DOF
    // 1-D extension whose raw interior slot is 3 — one past GetNDof() == 3.
    for (levels, n_dofs, ncp, patch_map) in [
        (0usize, 2usize, 2usize, &[0usize, 1][..]),
        (1, 3, 3, &[0, 2, 1]),
        (2, 5, 5, &[0, 4, 3, 2, 1]),
    ] {
        let mut ext = NurbsExtension::from_mesh_str(SEGMENT).expect("segment");
        for _ in 0..levels {
            ext.uniform_refinement(2).expect("uniform refinement");
        }
        assert_eq!(ext.n_dofs(), n_dofs, "refined {levels}x: GetNDof");
        assert_eq!(
            ext.patch_knot_vectors(0).expect("kv")[0].ncp(),
            ncp,
            "refined {levels}x: NCP"
        );
        check_patch_maps(&ext);
        let got: Vec<usize> = (0..ncp)
            .map(|i| ext.patch_dof(0, &[i]).expect("patch dof"))
            .collect();
        assert_eq!(got, patch_map, "refined {levels}x: MFEM 4.10 patch map");
    }
}

#[test]
fn patch_dof_matches_the_element_table_2d_3d() {
    // Unrefined and refined, over the full index domain of every patch.
    for (text, name) in [
        (SQUARE, "square-nurbs"),
        (DISC, "disc-nurbs"),
        (SQUARE_PATCH, "square-disc-nurbs-patch"),
        (BALL, "ball-nurbs"),
    ] {
        for levels in [0usize, 1] {
            let mut ext = NurbsExtension::from_mesh_str(text)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for _ in 0..levels {
                ext.uniform_refinement(2).expect("uniform refinement");
            }
            check_patch_maps(&ext);
        }
    }
}
