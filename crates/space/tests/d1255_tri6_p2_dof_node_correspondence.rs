//! D1255 (round 118) — Tri6 P2 space (`DofManager::build_q2_tri`, D838):
//! **space dof ids are NOT mesh node ids**.
//!
//! Bisect verdict for `mfem_ex7_surface_poisson`'s L2 regression
//! 9.4026555405e-3 → 1.0904089632e-1 (r84 `c5a06dd6` → r116 `e4a819c5`):
//!
//! | commit | value |
//! |---|---|
//! | r84 `c5a06dd6` | 9.4026555405e-3 (anchor, good) |
//! | D819-A `39145fb0` | 9.4026555405e-3 (unchanged) |
//! | D820-1/2 `0eb677d6` | rc=101 (Tri6 corner view collapsed the old hack to 66 dofs) |
//! | **D838 `17a319c8` (first-bad)** | **1.0904089632e-1** |
//!
//! Root cause: D838 moved ex7's Tri6 space from "order 1 over all row nodes"
//! (dof id == node id) to the row-aware P2 build `build_q2_tri`, whose edge
//! dofs are numbered in element-scan order — a *permutation* of the mesh
//! node ids. The example's `tri6_l2_error` kept indexing the solution vector
//! by mesh node id (`u[node]` instead of `u[dofs[k]]`), so the error was
//! evaluated against a permuted solution (the solve itself was always
//! dof-consistent: fixed probe run reproduces 9.4026530972e-3 with the
//! identical PCG trajectory, iter0 1.25008 / ARF 0.493151).
//!
//! This pin locks the fem-space side of that contract so consumers cannot
//! silently assume dof-id == node-id on quadratic Tri6 rows:
//!
//! 1. `build_q2_tri` yields `n_vertex + n_edge` dofs (no interior at P2);
//! 2. the dof numbering is a real permutation of node ids (tooth: reverting
//!    to node-id-ordered dofs — the historical hack — turns this red);
//! 3. positional correspondence `dofs[e][k] ↔ row node ns[e][k]` holds
//!    exactly (`dof_coord(dofs[e][k]) == node_coords(ns[e][k])`);
//! 4. interpolating a nodal vector **by dofs** reproduces the nodal values
//!    (P2 Lagrange identity at the row nodes);
//! 5. negative control: the same nodal vector scattered **by node id** does
//!    NOT interpolate correctly on this permuted numbering — exactly the
//!    D1255 mechanism (`u[node]` mis-indexing), kept red on purpose.

use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Two Tri6 triangles in 3-D (planar z=0) sharing edge (c1,c2), with node
/// ids deliberately ordered so that `build_q2_tri`'s element-scan edge-dof
/// numbering diverges from the node-id order: elem1's midsides get ids 4-5,
/// elem0's get 6-8 (the shared m12 = 8 created last). This makes the D1255
/// permutation concrete instead of incidental.
///
/// rows: elem0 = (c0,c1,c2, m01,m12,m20) = (0,1,2, 6,8,7)
///       elem1 = (c1,c3,c2, m13,m32,m21) = (1,3,2, 4,5,8)   [m21 == m12 == 8]
fn two_tri6_mesh() -> Mesh<3> {
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, // 0  c0
        1.0, 0.0, 0.0, // 1  c1
        0.0, 1.0, 0.0, // 2  c2
        1.0, 1.0, 0.0, // 3  c3
        1.0, 0.5, 0.0, // 4  m13
        0.5, 1.0, 0.0, // 5  m32
        0.5, 0.0, 0.0, // 6  m01
        0.0, 0.5, 0.0, // 7  m20
        0.5, 0.5, 0.0, // 8  m12 (shared)
    ];
    let conn: Vec<u32> = vec![
        0, 1, 2, 6, 8, 7, // elem0
        1, 3, 2, 4, 5, 8, // elem1
    ];
    Mesh {
        coords,
        conn,
        elem_tags: vec![1, 2],
        elem_type: ElementType::Tri6,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
        nc_leaf_states: None,
        nc_face_ids: None,
    }
}

/// Equispaced P2 triangle basis at `(xi, eta)` — mirrors
/// `fem_assembly::boundary::surface_tri6::p2_basis_tri6` (inlined so the
/// fem-space pin does not depend on fem-assembly). Dof order:
/// `[v0, v1, v2, mid01, mid12, mid20]` = `build_q2_tri`'s row layout.
fn p2_basis_tri6(xi: f64, eta: f64) -> [f64; 6] {
    let s = 1.0 - xi - eta;
    [
        2.0 * s * (s - 0.5),
        2.0 * xi * (xi - 0.5),
        2.0 * eta * (eta - 0.5),
        4.0 * xi * s,
        4.0 * xi * eta,
        4.0 * eta * s,
    ]
}

/// The six row-node reference positions, in `build_q2_tri` row order
/// `[v0, v1, v2, mid01, mid12, mid20]`.
const ROW_NODE_XI: [(f64, f64); 6] =
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)];

#[test]
fn d1255_tri6_p2_dof_count_and_vertex_split() {
    let mesh = two_tri6_mesh();
    let space = H1Space::new(mesh, 2);
    // 4 corners + 5 edges (2 triangles sharing one edge); no interior at P2.
    assert_eq!(space.n_dofs(), 9);
    assert_eq!(space.dof_manager().n_vertex_dofs, 4);
}

#[test]
fn d1255_tri6_p2_dofs_are_a_real_permutation_of_node_ids() {
    // Tooth: the historical "order 1 over all row nodes" hack had
    // dof id == node id; `build_q2_tri`'s element-scan edge numbering does
    // not. If anyone reverts to node-id-ordered dofs this goes red — which
    // is the state that made ex7's node-id-indexed error reader silently
    // wrong (D1255).
    let mesh = two_tri6_mesh();
    let ns0: Vec<u32> = mesh.element_nodes(0).to_vec();
    let ns1: Vec<u32> = mesh.element_nodes(1).to_vec();
    let space = H1Space::new(mesh, 2);
    let d0: Vec<u32> = space.element_dofs_u32(0).to_vec();
    let d1: Vec<u32> = space.element_dofs_u32(1).to_vec();
    let mismatched = (0..6).filter(|&k| d0[k] != ns0[k]).count()
        + (0..6).filter(|&k| d1[k] != ns1[k]).count();
    assert!(
        mismatched > 0,
        "build_q2_tri dofs must not be node-id-ordered on this fixture \
         (d0={d0:?}, ns0={ns0:?}, d1={d1:?}, ns1={ns1:?})"
    );
    // And the specific adversarial placements hold: vertex dofs are the
    // corner view (position 0: node 0 -> dof 0), but elem0's first edge dof
    // (position 3, row node m01 = 6) got dof id 4 from the element scan.
    assert_eq!((d0[0], ns0[0]), (0, 0));
    assert_eq!((d0[3], ns0[3]), (4, 6));
}

#[test]
fn d1255_tri6_p2_dof_node_positional_correspondence() {
    // The contract any Tri6 consumer relies on: position k of the element
    // dof row belongs to row node ns[k], with identical physical coordinates.
    let space = H1Space::new(two_tri6_mesh(), 2);
    let mesh = space.mesh();
    let dim = space.dof_manager().dim;
    for e in 0..2u32 {
        let ns = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        assert_eq!(dofs.len(), 6);
        for k in 0..6 {
            let dc = space.dof_manager().dof_coord(dofs[k]);
            let nc = mesh.node_coords(ns[k]);
            assert_eq!(&dc[..dim], &nc[..dim], "elem {e} position {k}");
        }
    }
}

#[test]
fn d1255_tri6_p2_interpolates_by_dofs_not_node_ids() {
    let space = H1Space::new(two_tri6_mesh(), 2);
    let mesh = space.mesh();
    let nodal: Vec<f64> = (0..mesh.n_nodes() as usize)
        .map(|j| {
            let c = mesh.node_coords(j as u32);
            2.0 * c[0] * c[1] - 3.0 * c[2] + 0.25
        })
        .collect();

    // Scattered BY DOFS (correct): u[dofs[k]] = nodal[ns[k]].
    let mut by_dof = vec![0.0_f64; space.n_dofs()];
    for e in 0..2u32 {
        let ns = mesh.element_nodes(e);
        let dofs = space.element_dofs(e).to_vec();
        for k in 0..6 {
            by_dof[dofs[k] as usize] = nodal[ns[k] as usize];
        }
    }

    // P2 Lagrange identity: evaluating at each row node's reference position
    // reproduces the nodal value exactly.
    for e in 0..2u32 {
        let ns = mesh.element_nodes(e);
        let dofs = space.element_dofs(e).to_vec();
        for (k, &(xi, eta)) in ROW_NODE_XI.iter().enumerate() {
            let phi = p2_basis_tri6(xi, eta);
            let uh = (0..6).map(|i| phi[i] * by_dof[dofs[i] as usize]).sum::<f64>();
            assert!(
                (uh - nodal[ns[k] as usize]).abs() < 1e-14,
                "elem {e} node {k}: uh={uh} vs nodal={}",
                nodal[ns[k] as usize]
            );
        }
    }

    // Negative control (the D1255 mechanism): the SAME nodal values scattered
    // BY NODE ID and read through the dof rows do not interpolate — on this
    // permuted fixture the mis-indexing is loud.
    let by_node = nodal.clone();
    let mut worst = 0.0_f64;
    for e in 0..2u32 {
        let ns = mesh.element_nodes(e);
        let dofs = space.element_dofs(e).to_vec();
        for (k, &(xi, eta)) in ROW_NODE_XI.iter().enumerate() {
            let phi = p2_basis_tri6(xi, eta);
            let uh = (0..6).map(|i| phi[i] * by_node[dofs[i] as usize]).sum::<f64>();
            let ue = nodal[ns[k] as usize];
            worst = worst.max((uh - ue).abs());
        }
    }
    assert!(
        worst > 1e-3,
        "node-id-scattered values must NOT interpolate through dof rows on a \
         permuted numbering (worst |diff| = {worst}); if this ever goes ~0 the \
         permutation pin above is the one that should have caught it"
    );
}
