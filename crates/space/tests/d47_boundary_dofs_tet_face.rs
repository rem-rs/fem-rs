//! D47 regression: `boundary_dofs_hcurl` must return the **tet triangular-face**
//! DOFs of a tet `NDk` (k ≥ 2) space, not just its edge DOFs.
//!
//! `HCurlSpace`'s tet face DOFs (2 per face point, `k(k-1)/2` points per face)
//! carry the interior part of the tangential trace, so omitting them leaves a
//! PEC / essential boundary condition incomplete and the discrete solution
//! O(1) wrong (MFEM constrains every boundary DOF via
//! `FiniteElementSpace::GetEssentialTrueDofs`).
//!
//! The expected DOF count is stated purely in mesh combinatorics,
//! `k · (#boundary edges) + k(k-1) · (#boundary faces)`, and the test also pins
//! the membership of the returned set against an independently rebuilt one.

use std::collections::BTreeSet;

use fem_core::types::DofId;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::{EdgeKey, FaceKey};
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::HCurlSpace;

/// Unique edges of the tagged boundary faces, from the mesh alone.
fn boundary_edges<M: MeshTopology>(mesh: &M, tags: &[i32]) -> BTreeSet<EdgeKey> {
    let mut out = BTreeSet::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) {
            continue;
        }
        let ns = mesh.face_nodes(f);
        for i in 0..ns.len() {
            out.insert(EdgeKey::new(ns[i], ns[(i + 1) % ns.len()]));
        }
    }
    out
}

/// Tagged boundary faces, from the mesh alone.
fn boundary_faces<M: MeshTopology>(mesh: &M, tags: &[i32]) -> Vec<Vec<u32>> {
    (0..mesh.n_boundary_faces() as u32)
        .filter(|&f| tags.contains(&mesh.face_tag(f)))
        .map(|f| mesh.face_nodes(f).to_vec())
        .collect()
}

#[test]
fn d47_tet_face_dofs_are_essential() {
    for (lvl, k) in [(1usize, 2u8), (1, 3), (2, 2)] {
        let mesh = Mesh::<3>::unit_cube_tet(lvl);
        let space = HCurlSpace::new(mesh.clone(), k);
        let tags = mesh.unique_boundary_tags();
        assert!(!tags.is_empty(), "mesh must have tagged boundary faces");

        let be = boundary_edges(&mesh, &tags);
        let bf = boundary_faces(&mesh, &tags);
        let got = boundary_dofs_hcurl(&mesh, &space, &tags);

        // Closed-form expectation: k DOFs per boundary edge (ND1 .. ND3 all
        // carry one point-value DOF per Gauss-Legendre edge point, and fem-rs
        // uses exactly k of them) plus `k(k-1)` per triangular boundary face
        // (2 per face point, `k(k-1)/2` points).
        let expected_len = k as usize * be.len() + (k as usize) * (k as usize - 1) * bf.len();
        assert_eq!(
            got.len(),
            expected_len,
            "order {k}, lvl {lvl}: {} boundary DOFs, expected {expected_len} \
             ({} boundary edges, {} boundary faces) — tet face DOFs missing?",
            got.len(),
            be.len(),
            bf.len(),
        );

        // Independent membership check.
        let mut want: BTreeSet<DofId> = BTreeSet::new();
        for e in &be {
            for d in space.edge_dofs(*e).expect("boundary edge must have DOFs") {
                want.insert(d);
            }
        }
        let nfd = (k as DofId) * (k as DofId - 1);
        for f in &bf {
            assert_eq!(f.len(), 3, "tet boundary faces are triangles");
            let first = space
                .face_dof(FaceKey::new(f[0], f[1], f[2]))
                .expect("boundary tet face must have a face-DOF block (k >= 2)");
            for m in 0..nfd {
                want.insert(first + m);
            }
        }
        let got_set: BTreeSet<DofId> = got.iter().copied().collect();
        assert_eq!(got_set, want, "order {k}, lvl {lvl}: DOF set mismatch");
        // Sorted + deduplicated by contract.
        assert!(got.windows(2).all(|w| w[0] < w[1]), "must be strictly increasing");
    }
}

#[test]
fn d47_order_one_is_unchanged() {
    // ND1 has no face DOFs: the boundary set is exactly the boundary edges'.
    let mesh = Mesh::<3>::unit_cube_tet(1);
    let space = HCurlSpace::new(mesh.clone(), 1);
    let tags = mesh.unique_boundary_tags();
    let be = boundary_edges(&mesh, &tags);
    let got = boundary_dofs_hcurl(&mesh, &space, &tags);
    assert_eq!(got.len(), be.len());
    assert!(mesh.dim() == 3 && space.order() == 1);
}
