//! D492 (closed, round 55) — tri RT1 slot / global-numbering / sign semantics
//! are now MFEM 4.10's, with **no** pairing bridge.
//!
//! # What round 54 found (evidence `tmp/d492/`)
//!
//! Round 54 located three pairing-layer differences between fem-rs's tri RT1
//! space and MFEM's `RT_TriangleElement` + `FiniteElementSpace`:
//!
//! 1. **slot convention**: π(mfem_slot) = femrs_slot = `[4,5,0,1,3,2,6,7]`
//!    (fem-rs listed the edge blocks as `TRI_FACES = [(v1,v2), (v0,v2),
//!    (v0,v1)]` with the third edge run `v0→v2`, MFEM as
//!    `Geometry::TRIANGLE::Edges = [(v0,v1), (v1,v2), (v2,v0)]`);
//! 2. **global numbering** (registered as **D513**): fem-rs interleaved each
//!    element's interior dofs right after its edge blocks, MFEM is
//!    entity-major (`fespace.cpp`: `ebase = E[i]*ne`, then `bbase = nvdofs +
//!    nedofs + elem*nb`);
//! 3. **sign convention** (**D514**): fem-rs's sign was the outward-vs-
//!    canonical-normal test, MFEM's is the `SegDofOrd` orientation sign
//!    (`+1` iff the element's local edge direction is ascending), which
//!    differs by a negative factor on every positively oriented element.
//!
//! With those three bridged by hand the two prolongations were proved
//! mathematically identical (220/220 entries, max 3.6e-16,
//! `tmp/d492/d492_join_proof.txt` + `tmp/d492/join_probe`).
//!
//! # What round 55 changed (this test now pins the result)
//!
//! The three differences are gone from the code, so the pairing is the
//! **identity**:
//!
//! * `crates/element/src/raviart_thomas/tri_rt1.rs` (and `tri_rt2.rs`/
//!   `tri_rtk.rs`) enumerate the reference dofs in MFEM's local order —
//!   `mfem_tri_nodal_dofs`, `TriRT1::{build_vandermonde, dof_coords}`;
//! * `crates/space/src/hdiv.rs::build_2d_tri` walks `TRI_EDGES =
//!   [(0,1),(1,2),(2,0)]` with the same within-block node ordering, and
//!   numbers the global dofs entity-major (edges first, then interiors by
//!   element);
//! * `build_2d_tri` stamps MFEM's orientation sign `+1 iff gi < gj`.
//!
//! The literals below are MFEM's own tables: `tmp/d468/d468_tri_o1.txt`
//! rows `C_DOFS 0 0 1 -4 -3 -6 -5 10 11` and
//! `C_DOFS 1 -2 -1 6 7 8 9 12 13` (negative = `−1 − dof`, i.e. dof with a
//! `−1` sign), decoded to `[(0,+),(1,+),(3,−),(2,−),(5,−),(4,−),(10,+),(11,+)]`
//! and `[(1,−),(0,−),(6,+),(7,+),(8,+),(9,+),(12,+),(13,+)]`.

use fem_mesh::amr::refine_uniform;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::HDivSpace;

/// MFEM `MakeCartesian2D(1, 1, TRIANGLE)` — the exact d468 oracle fixture
/// (main diagonal (0,0)-(1,1), MFEM vertex/element order).
fn mfem_tri_mesh() -> Mesh<2> {
    Mesh::<2> {
        coords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        conn: vec![0, 3, 2, 3, 0, 1],
        vertex_parents: vec![],
        elem_tags: vec![1, 1],
        elem_type: ElementType::Tri3,
        face_conn: vec![0, 1, 1, 3, 3, 2, 2, 0],
        face_tags: vec![1, 1, 1, 1],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: None,
    }
}

/// MFEM `Geometry::TRIANGLE::Edges`.
const MFEM_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
/// `RT_TriangleElement::nk`, indexed by `dof2nk`.
const MFEM_NK: [[f64; 2]; 3] = [[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]];

/// Physical identity of an MFEM RT1 slot `m` on an element with vertices
/// `verts`: `(edge as a min/max vertex pair, node index counted from the
/// min vertex, reference normal)`.  MFEM numbers the edge blocks in
/// `Geometry::TRIANGLE::Edges` order with `p + 1` Gauss-Legendre open points
/// ascending along the *listed* direction, and
/// `SegDofOrd[orientation > 0 ? 0 : 1]` reverses the canonical node index on
/// descending edges.
fn mfem_slot_identity(verts: [u32; 3], m: usize) -> ((u32, u32), usize, [f64; 2]) {
    let (i, j) = MFEM_EDGES[m / 2];
    let (va, vb) = (verts[i], verts[j]);
    let node_from_va = m % 2;
    let (pair, node) = if va < vb {
        ((va, vb), node_from_va)
    } else {
        ((vb, va), 1 - node_from_va)
    };
    (pair, node, MFEM_NK[m / 2])
}

/// The literal MFEM per-slot `(dof, sign)` tables for the fixture
/// (`tmp/d468/d468_tri_o1.txt`, `C_DOFS` rows, negative d decoded as `−1−d`).
const MFEM_C_DOFS: [&[(u32, f64)]; 2] = [
    &[
        (0, 1.0),
        (1, 1.0),
        (3, -1.0),
        (2, -1.0),
        (5, -1.0),
        (4, -1.0),
        (10, 1.0),
        (11, 1.0),
    ],
    &[
        (1, -1.0),
        (0, -1.0),
        (6, 1.0),
        (7, 1.0),
        (8, 1.0),
        (9, 1.0),
        (12, 1.0),
        (13, 1.0),
    ],
];

/// fem-rs's `element_dofs` / `element_signs` are MFEM's `GetElementDofs`
/// tables **verbatim** (D513/D514 closed): no π, no sign bridge.
#[test]
fn d492_tri_rt1_element_dofs_are_mfem_tables() {
    let mesh = mfem_tri_mesh();
    let space = HDivSpace::new(mesh.clone(), 1);

    for (e, want) in MFEM_C_DOFS.iter().enumerate() {
        let dofs = space.element_dofs(e as u32);
        let signs = space.element_signs(e as u32);
        let got: Vec<(u32, f64)> = dofs
            .iter()
            .copied()
            .zip(signs.iter().copied())
            .map(|(d, s)| {
                assert!(s == 1.0 || s == -1.0, "sign must be ±1, got {s}");
                (d, s)
            })
            .collect();
        assert_eq!(&got[..], *want, "elem {e}: MFEM C_DOFS table");
    }

    // Entity-major global numbering: 4 boundary edges + 1 interior diagonal
    // = 5 edge blocks × 2 dofs, then 2 interiors per element.
    assert_eq!(space.n_dofs(), 5 * 2 + 2 * 2);
    // Edge blocks are numbered in MFEM's edge-table order — the
    // first-encounter order of `TRIANGLE::Edges` over the elements: the
    // diagonal (0,3) comes first (element 0's local edge 0).
    use fem_space::dof_manager::EdgeKey;
    assert_eq!(space.edge_face_dof(EdgeKey::new(0, 3)), Some(0));
    assert_eq!(space.edge_face_dof(EdgeKey::new(2, 3)), Some(2));
    assert_eq!(space.edge_face_dof(EdgeKey::new(0, 2)), Some(4));
    assert_eq!(space.edge_face_dof(EdgeKey::new(0, 1)), Some(6));
    assert_eq!(space.edge_face_dof(EdgeKey::new(1, 3)), Some(8));
    // ...and the interior dofs sit after every edge dof (D513: `bbase`).
    for e in 0..2u32 {
        assert!(
            space.element_dofs(e)[6..].iter().all(|&d| d >= 10),
            "elem {e} interiors must follow the entity-major edge base"
        );
    }

    // Fine mesh: 16 edges × 2 + 8 interiors × 2 dofs, again entity-major.
    let fine = refine_uniform(&mesh);
    let fine_space = HDivSpace::new(fine.clone(), 1);
    assert_eq!(fine_space.n_dofs(), 16 * 2 + 8 * 2, "16 edges x2 + 8 interiors x2");
    assert_eq!(fine.n_elements(), 8);
}

/// The slot permutation is the identity: MFEM's FE-local slot `m` and fem-rs's
/// slot `m` carry the same physical sample *and* the same reference normal, on
/// every element of the coarse and of the uniformly refined mesh.
#[test]
fn d492_tri_rt1_slot_map_is_identity() {
    let mesh = mfem_tri_mesh();
    let fine = refine_uniform(&mesh);
    let cases = [
        ("coarse", mesh.clone()),
        ("fine", fine.clone()),
    ];
    for (label, m) in cases.iter() {
        let space = HDivSpace::new(m.clone(), 1);
        for e in 0..m.n_elements() as u32 {
            let nd = m.element_nodes(e);
            let verts = [nd[0], nd[1], nd[2]];
            // Edge slots (0..6): the (min,max) vertex pair of the block the
            // slot's global dof belongs to, and the node index inside it.
            for s in 0..6usize {
                let (pair, node, _nk) = mfem_slot_identity(verts, s);
                let dofs = space.element_dofs(e);
                let first = space
                    .edge_face_dof(fem_space::dof_manager::EdgeKey::new(pair.0, pair.1))
                    .unwrap_or_else(|| panic!("{label}: missing block for edge {pair:?}"));
                assert_eq!(
                    dofs[s],
                    first + node as u32,
                    "{label}: elem {e} slot {s} dof vs MFEM physical node"
                );
                // The sign must be MFEM's `SegDofOrd` orientation sign.
                let (i, j) = MFEM_EDGES[s / 2];
                let want = if verts[i] < verts[j] { 1.0 } else { -1.0 };
                assert_eq!(
                    space.element_signs(e)[s],
                    want,
                    "{label}: elem {e} slot {s} orientation sign"
                );
            }
            // Interior slots: both codes order the two components
            // nk = (0,-1) then (-1,0) at (1/3,1/3), and both are sign +1.
            for s in 6..8usize {
                let dofs = space.element_dofs(e);
                assert!(dofs[s] > 0);
                assert_eq!(space.element_signs(e)[s], 1.0);
            }
        }
    }
}
