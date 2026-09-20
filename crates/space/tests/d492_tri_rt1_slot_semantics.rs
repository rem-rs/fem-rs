//! D492 — tri RT1 within-edge face-block layout: fem-rs slot convention vs
//! MFEM `RT_TriangleElement` FE local dof order.
//!
//! # Root cause (located this round, evidence `tmp/d492/`)
//!
//! The d468 tri RT1 oracle (`crates/assembly/tests/d468_hdiv_prolongation_mfem_parity.rs`,
//! `#[ignore]`d) pairs each MFEM global dof with `(element, MFEM slot)` from
//! the probe dump (`tmp/d468/d468_tri_o1.txt`, decoded against the empirical
//! dump `tmp/d492/d492_tri_o1_dump.txt`) and then resolves fem-rs dof =
//! `element_dofs(elem)[slot]`.  That pairing is only value-correct when the
//! two slot conventions coincide.  They do **not**:
//!
//! * MFEM `RT_TriangleElement` (`fem/fe/fe_rt.cpp`, ctor) numbers its FE-local
//!   dofs as edge blocks in `Geometry::TRIANGLE::Edges` order
//!   `[(v0,v1), (v1,v2), (v2,v0)]`, `p+1` Gauss-Legendre nodes per block
//!   ascending along the listed local edge direction, then the interior
//!   (normal (0,-1) then (-1,0) at (1/3,1/3)).
//! * fem-rs `HDivSpace::build_2d_tri` (D34, shared with `TriRT1`/`TriRT2` in
//!   `crates/element`) numbers slots as `TRI_FACES = [(v1,v2), (v0,v2),
//!   (v0,v1)]`, ascending along each listed direction, then the interior in
//!   the same (0,-1), (-1,0) order.
//!
//! Same normals, same per-edge node directions — different block order (and
//! the third edge runs (v2,v0) in MFEM vs (v0,v2) in fem-rs).  The element-
//! invariant slot permutation is `π(mfem_slot) = femrs_slot =
//! [4, 5, 0, 1, 3, 2, 6, 7]` for order 1 (hyp and bottom blocks swap; the
//! left block is additionally reversed).  The *global* numbering additionally
//! differs in layout: MFEM numbers all edge blocks entity-major first
//! (`edge e → {2e, 2e+1}`, probe dump `edge 0 … dofs [0,1]`), while
//! `build_2d_tri` interleaves each element's interior dofs right after its
//! edges (pinned below: elem 0's interiors are 6..7, elem 1's 12..13).
//!
//! # Why the fix is not space-only
//!
//! `build_2d_tri` must keep listing slots in the *reference element's* basis
//! order (`TriRT1`/`TriRT2` in crates/element), because the vector assembler,
//! `interpolate_vector`, `discrete_op` and `transfer` all pair slot-for-slot
//! with that basis.  Aligning with MFEM therefore requires permuting
//! `crates/element::raviart_thomas::tri_rt1/2` (incl. `mfem_tri_nodal_dofs`)
//! **together with** `hdiv.rs::build_2d_tri`; afterwards the d468 oracle's
//! `resolve_dof_map` becomes correct with π = identity.  This test pins the
//! current convention and the machine-checked bridge so the realigning round
//! can flip it.

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

/// Physical identity of a 2-D RT1 edge slot: `(min vertex, max vertex, node
/// index counted from the min vertex)` — the same key MFEM's canonical edge
/// blocks and fem-rs's edge blocks both use (block index i ↔ t = bop[i] from
/// the smaller endpoint in both codes).
///
/// * MFEM slot `m`: edge block `m/2` over local edges `J = [(0,1), (1,2),
///   (2,0)]`, node `m%2` at `t = bop[m%2]` from the block's first endpoint
///   (`fe_rt.cpp` ctor: `Nodes.IntPoint(o).Set2(bop[i], 0.)` etc.).
/// * fem-rs slot `s`: edge block `s/2` over `TRI_FACES = [(1,2), (0,2),
///   (0,1)]`, node `s%2` at `t = bop[s%2]` from the block's first endpoint
///   (`build_2d_tri`: slots ascend along (li→lj), reversed when `gi > gj`).
fn slot_identity(verts: [u32; 3], mfem_slot: bool, k: usize) -> (u32, u32, usize) {
    const MFEM_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    const TRI_FACES: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
    const N: usize = 2; // order-1: 2 nodes per edge
    let (a, b) = if mfem_slot {
        MFEM_EDGES[k / 2]
    } else {
        TRI_FACES[k / 2]
    };
    let (va, vb) = (verts[a], verts[b]);
    // node index within the block, k%2, counted from `va`; flip to count from
    // the min endpoint.
    let node_from_va = k % 2;
    if va < vb {
        (va, vb, node_from_va)
    } else {
        (vb, va, N - 1 - node_from_va)
    }
}

/// MFEM slot → fem-rs slot permutation for tri RT1: MFEM blocks
/// [bottom(0,1) | hyp(1,2) | left(2,0) | int] vs fem-rs
/// [hyp(1,2) | left(0,2) | bottom(0,1) | int] with the left block reversed.
const PI_MFEM_TO_FEMRS: [usize; 8] = [4, 5, 0, 1, 3, 2, 6, 7];

/// The π bridge must hold on every element: fem-rs slot `PI_MFEM_TO_FEMRS[m]`
/// carries the same physical edge flux identity as MFEM slot `m` (edge slots
/// 0..6; the two interior slots map identity-wise — π[6]=6, π[7]=7 — since
/// both codes order them (0,-1) then (-1,0) at (1/3,1/3)).
fn assert_pi_bridge(space: &HDivSpace<Mesh<2>>, mesh: &Mesh<2>) {
    for e in 0..mesh.n_elements() as u32 {
        let verts: [u32; 3] = {
            let nd = mesh.element_nodes(e);
            [nd[0], nd[1], nd[2]]
        };
        for m in 0..6usize {
            let mfem = slot_identity(verts, true, m);
            let femrs = slot_identity(verts, false, PI_MFEM_TO_FEMRS[m]);
            assert_eq!(
                mfem, femrs,
                "elem {e}: MFEM slot {m} identity {mfem:?} != fem-rs slot {} identity {femrs:?}",
                PI_MFEM_TO_FEMRS[m]
            );
        }
    }
}

/// Pin fem-rs's exact slot tables + signs on the coarse 2-triangle mesh and
/// the π bridge; then assert the bridge on every element of the uniformly
/// refined mesh as well (the permutation is element-invariant).
///
/// The literal tables cross-check the probe decode in
/// `tmp/d492/d492_tri_o1_dump.txt`: MFEM elem 0's decoded slot→(dof, sign)
/// list is `[(0,+),(1,+),(3,-),(2,-),(5,-),(4,-),(10,+),(11,+)]` and elem 1's
/// `[(1,-),(0,-),(6,+),(7,+),(8,+),(9,+),(12,+),(13,+)]`.
#[test]
fn d492_tri_rt1_slot_tables_and_mfem_bridge() {
    let mesh = mfem_tri_mesh();
    let space = HDivSpace::new(mesh.clone(), 1);

    // fem-rs edge numbering (single-pass, TRI_FACES first-encounter, interiors
    // interleaved per element — NOT MFEM's entity-major blocks): elem 0's edges
    // top {2,3} → 0..1, left {0,2} → 2..3, diagonal {0,3} → 4..5, then its
    // interiors 6..7; elem 1's bottom {0,1} → 8..9, right {1,3} → 10..11
    // (diagonal shared), then interiors 12..13.
    // Elem 0 (conn [0,3,2]): (v1,v2)=(3,2) reversed, (v0,v2)=(0,2) straight,
    // (v0,v1)=(0,3) straight.
    let dofs0 = space.element_dofs(0);
    assert_eq!(dofs0, &[1, 0, 2, 3, 4, 5, 6, 7], "elem 0 slot table");
    assert_eq!(space.element_signs(0), &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]);

    // Elem 1 (conn [3,0,1]): (v1,v2)=(0,1) straight, (v0,v2)=(3,1) reversed,
    // (v0,v1)=(3,0) reversed (shares the diagonal block).
    assert_eq!(space.element_dofs(1), &[8, 9, 11, 10, 5, 4, 12, 13], "elem 1 slot table");
    assert_eq!(space.element_signs(1), &[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);

    assert_pi_bridge(&space, &mesh);

    // Fine mesh: the bridge is purely combinatorial (slot patterns), so it
    // holds on every refined element too.
    let fine = refine_uniform(&mesh);
    let fine_space = HDivSpace::new(fine.clone(), 1);
    assert_eq!(fine_space.n_dofs(), 16 * 2 + 8 * 2, "16 edges x2 + 8 interiors x2");
    assert_pi_bridge(&fine_space, &fine);
}
