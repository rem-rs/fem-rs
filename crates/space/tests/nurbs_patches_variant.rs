//! Round-52 lane 5 (D143): the `patches` mesh-file variant of
//! `NurbsExtension::from_mesh_str` — MFEM `NURBSExtension::Load`'s `patches`
//! branch (`NURBSPatch` blocks + unique-knot-vector reconstruction through
//! `GetPatchDirectionEdges` / `CheckKVDirection` / `KnotVector::Flip`).
//!
//! Fixture: `data/square-disc-nurbs-patch.mesh` — 5 patch blocks, no
//! `weights` section, no node block (MFEM derives the nodal space from the
//! patches).
//!
//! Truth: MFEM 4.10 probe (`tmp/d143/probe.cpp`, dumps in
//! `tmp/d143/probe_truth.txt`): dim=2 NE=9 NBE=10 NV=14, NP=5 NKV=4 NDof=38
//! GNE=9 order=2, unique kv ncp `[4,3,3,3]`, and `NURBSPatchMap` (Dof mode)
//! patch-0/1 grids as pinned below.

use fem_space::NurbsExtension;

const MESH: &str = include_str!("../../../data/square-disc-nurbs-patch.mesh");

#[test]
fn patches_variant_loads_square_disc() {
    let ext =
        NurbsExtension::from_mesh_str(MESH).expect("the `patches` variant must load");

    assert_eq!(ext.dim(), 2);
    assert_eq!(ext.n_patches(), 5, "GetNP");
    assert_eq!(ext.n_knot_vectors(), 4, "GetNKV");
    assert_eq!(ext.n_dofs(), 38, "GetNDof");
    assert_eq!(ext.n_elements(), 9, "GetNE / GetGNE");
    assert_eq!(ext.n_bdr_elements(), 10, "GetNBE");
    assert_eq!(ext.n_vertices(), 14, "GetNV");

    // Unique knot vectors: all order 2, NCP [4, 3, 3, 3] (probe kv orders).
    let orders: Vec<usize> = (0..ext.n_knot_vectors()).map(|i| ext.knot_vector(i).order()).collect();
    assert_eq!(orders, vec![2, 2, 2, 2]);
    let ncps: Vec<usize> = (0..ext.n_knot_vectors()).map(|i| ext.knot_vector(i).ncp()).collect();
    assert_eq!(ncps, vec![4, 3, 3, 3]);

    // `NURBSPatchMap` (Dof mode) grids, i fastest: probe patch 0
    // `0 18 3 10 29 12 11 30 13 4 19 7` and patch 1
    // `3 20 2 12 31 16 13 32 17 7 21 6`.
    let grid = |p: usize, want: &[[usize; 3]]| {
        for (j, row) in want.iter().enumerate() {
            for (i, &dof) in row.iter().enumerate() {
                assert_eq!(
                    ext.patch_dof(p, &[i, j]).unwrap(),
                    dof,
                    "patch {p} dof ({i},{j})"
                );
            }
        }
    };
    grid(0, &[[0, 18, 3], [10, 29, 12], [11, 30, 13], [4, 19, 7]]);
    grid(1, &[[3, 20, 2], [12, 31, 16], [13, 32, 17], [7, 21, 6]]);

    // The patch blocks carry their own rational weights, so the *analysis*
    // weights stay unit (MFEM reads no `weights` section for this flavour).
    assert_eq!(ext.weights().len(), 38);
    assert!(ext.weights().iter().all(|&w| w == 1.0));
}

#[test]
fn knotvectors_variant_still_rejected_nothing() {
    // Sanity: the knotvectors variant keeps working (square-nurbs.mesh).
    const KV_MESH: &str = include_str!("../../../data/square-nurbs.mesh");
    let ext = NurbsExtension::from_mesh_str(KV_MESH).expect("knotvectors variant");
    assert_eq!(ext.n_patches(), 1);
    assert_eq!(ext.n_knot_vectors(), 2);
}
