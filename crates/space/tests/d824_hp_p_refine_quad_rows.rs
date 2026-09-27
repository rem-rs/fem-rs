//! D824-2 (round 84, fix agent A) — hp (variable-order) spaces dispatch by
//! **element type**, not by connectivity row length.
//!
//! `p_refine.rs` classified every element by `element_nodes(e).len()`; a
//! Quad8 row (8 nodes) fell into the hex arm, a Quad9 row (9 nodes) into the
//! triangle arm, and the builder's geometry assert rejected every quadratic
//! row outright (`unsupported element geometry`), so hp was unreachable on
//! Quad8/Quad9/Tri6/Tet10/Hex20/Hex27/Prism15/Prism18 meshes.  Since D824 the
//! dispatch is by `mesh.element_type(e)` over the rows' **corner** view (one
//! MFEM geometry per family — the same doctrine as the D824-1 P1 fix).
//!
//! MFEM 4.10 ground truth (probe `tmp/d84fixA/hp_quad_probe.cpp` on the
//! curved Quad9 fixture, `EnsureNCMesh` + SetElementOrder(0,2)/(1,1), output
//! `tmp/d84fixA/probe_hp_quad9.txt`):
//!
//! ```text
//! NDofs=11 NTrueDofs=10
//! elem 0 (9 dofs): 0 1 4 3 6 7 8 9 10     (4 corners + 4 edge dofs + 1 interior)
//! elem 1 (4 dofs): 1 2 5 4                (p1: corners only)
//! constrained 7 <- (1, +0.5) (4, +0.5)    (shared edge p2 midpoint <- endpoints)
//! ```
//!
//! fem-rs numbers hp vertex dofs by node-id identity (the historical
//! convention, exact for linear meshes), so its absolute ids carry the
//! quadratic fixture's node gaps; the tests pin the structure that MFEM
//! pins — per-element row shape, variant layout, NDofs *net of the unused
//! node ids* (20 − 9 = 11 = MFEM's NDofs) and the constraint weights — plus
//! exact parity with the already-good Quad4 path.
//!
//! ```text
//! cargo test -p fem-space --test d824_hp_p_refine_quad_rows -- --nocapture
//! ```

use fem_io::gmsh::read_msh;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::p_refine::{build_variable_order_dof_manager, detect_p_constraints};

const FIXTURE_Q9: &str = include_str!("../../../data/d819_quad9_curved.msh");
const FIXTURE_TET10: &str = include_str!("../../../data/d824_tet10_pair.msh");
const FIXTURE_TRI6: &str = include_str!("../../../data/d824_tri6_pair.msh");

/// The round-84 lane A fem-rs-internal Quad8 fixture (Gmsh type 16 — MFEM
/// aborts on that file, so the cell is a fem-rs-side row label).
const FIXTURE_Q8: &str = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n\
1 0.0000000000000000e+00 0.0000000000000000e+00 0\n\
2 1.0000000000000000e+00 0.0000000000000000e+00 0\n\
3 1.0700000000000001e+00 1.0500000000000000e+00 0\n\
4 -5.0000000000000000e-02 9.4000000000000000e-01 0\n\
5 5.0000000000000000e-01 3.0000000000000000e-03 0\n\
6 1.0300000000000000e+00 5.1000000000000000e-01 0\n\
7 5.2000000000000000e-01 1.0100000000000000e+00 0\n\
8 -2.0000000000000000e-02 4.7000000000000000e-01 0\n\
$EndNodes\n$Elements\n1\n1 16 1 1 1 2 3 4 5 6 7 8\n$EndElements\n";

fn quad9_mesh() -> fem_mesh::Mesh<2> {
    read_msh(FIXTURE_Q9.as_bytes()).expect("parse Quad9 fixture").into_2d().expect("2-D")
}

/// hp on the curved **Quad9** mesh (orders [2, 1]): the builder no longer
/// panics, cell 0 (p2) carries 4 corners + 4 edge dofs + 1 interior over the
/// row's corners, cell 1 (p1) carries its 4 corners only, and the shared
/// edge's p2-variant midpoint is constrained to the endpoints with the MFEM
/// 0.5/0.5 GLL weights (probe above; MFEM NDofs 11 = fem-rs 20 − 9 gaps).
#[test]
fn d824_hp_quad9_rows_match_mfem_structure() {
    let mesh = quad9_mesh();
    assert_eq!(mesh.element_type(0), ElementType::Quad9);
    let orders: Vec<u8> = vec![2, 1];
    let dm = build_variable_order_dof_manager(&mesh, &orders);

    // MFEM: NDofs=11 on the compacted corner set; fem-rs keeps node-id
    // identity, so 20 dofs − 9 unused quadratic-geometry nodes = 11.
    assert_eq!(dm.n_dofs, 20, "15 nodes + 4 edge dofs + 1 interior");
    assert_eq!(dm.n_dofs - (mesh.n_nodes() - 6), 11, "net of gaps = MFEM NDofs");

    // Cell 0 (p2): corners [0,2,12,10], edge dofs 15..19, interior 19.
    assert_eq!(dm.element_dofs(0), &[0, 2, 12, 10, 15, 16, 17, 18, 19]);
    // Cell 1 (p1): its 4 corners only.
    assert_eq!(dm.element_dofs(1), &[2, 4, 14, 12]);

    // The shared edge (2,12) holds two variants; the p2 variant dof 16 is
    // constrained to the endpoints with the GLL midpoint weights.
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 1, "one mixed-order edge");
    assert_eq!(constraints[0].constrained, 16);
    let mut parents = constraints[0].parents.clone();
    parents.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        parents,
        vec![(2, 0.5), (12, 0.5)],
        "MFEM: constrained 7 <- (1, +0.5) (4, +0.5)"
    );
}

/// hp on a **Quad8** mesh (pre-fix: the builder panicked at the geometry
/// assert after silently minting triangle edges).  Post-fix the single p2
/// cell numbers 4 corners + 4 edge dofs + 1 interior, with the row's
/// mid-edge geometry nodes (4..7) left out of the space.
#[test]
fn d824_hp_quad8_mesh_builder_works() {
    let msh = read_msh(FIXTURE_Q8.as_bytes()).expect("parse Quad8 fixture");
    let mesh = msh.into_2d().expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Quad8);
    assert_eq!(mesh.n_nodes(), 8);

    let dm = build_variable_order_dof_manager(&mesh, &[2]);
    assert_eq!(dm.n_dofs, 8 + 4 + 1, "4 corners (node ids) + 4 edge dofs + interior");
    assert_eq!(dm.element_dofs(0), &[0, 1, 2, 3, 8, 9, 10, 11, 12]);

    let constraints = detect_p_constraints(&dm, &mesh, &[2]);
    assert!(constraints.is_empty(), "uniform order: no constraints");
}

/// One-geometry parity: a straight **Quad4** two-cell mesh with orders [2, 1]
/// produces the same variant structure (and hence the same dof count net of
/// vertex gaps) as the Quad9 rows — MFEM's single SQUARE geometry.
#[test]
fn d824_hp_quad4_parity_with_quad9_rows() {
    let mesh = fem_mesh::Mesh::<2>::uniform(
        vec![
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, // bottom
            0.0, 1.0, 1.0, 1.0, 2.0, 1.0, // top
        ],
        vec![0, 1, 4, 3, 1, 2, 5, 4],
        vec![1, 1],
        ElementType::Quad4,
        vec![],
        vec![],
        ElementType::Line2,
    );
    assert_eq!(mesh.n_elements(), 2);
    let orders: Vec<u8> = vec![2, 1];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    // 6 vertices + 4 edge dofs (3 cell-0 edges at p2 + shared p2 variant)
    // + 1 interior = 11 = MFEM's compacted count.
    assert_eq!(dm.n_dofs, 11);
    assert_eq!(dm.element_dofs(0).len(), 9);
    assert_eq!(dm.element_dofs(1).len(), 4);

    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 1);
    assert_eq!(constraints[0].constrained, dm.element_dofs(0)[5], "shared edge dof");
    let mut parents = constraints[0].parents.clone();
    parents.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(parents, vec![(1, 0.5), (4, 0.5)]);
}

/// Census (discipline ⑰): hp now also accepts Tet10 rows — cell 0 p1, cell 1
/// p2: 5 vertices + 6 edge dofs (3 shared-edge p2 variants + 3 cell-1 outer
/// edges), no face/volume dofs at these orders.
#[test]
fn d824_hp_tet10_rows_accepted() {
    let msh = read_msh(FIXTURE_TET10.as_bytes()).expect("parse Tet10 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tet10);
    let orders: Vec<u8> = vec![1, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);

    assert_eq!(dm.n_dofs, 14 + 6, "14 node ids + 6 edge-variant dofs");
    assert_eq!(dm.n_dofs - (mesh.n_nodes() - 5), 11, "net of gaps = 5 corners + 6 edges");
    assert_eq!(dm.element_dofs(0).len(), 4, "p1 cell: corners only");
    assert_eq!(dm.element_dofs(1).len(), 10, "p2 cell: 4 corners + 6 edges");

    // The 3 mixed-order (shared face) edges constrain their p2 variant dofs
    // to the endpoint pair with the GLL midpoint weights.
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 3);
    for c in &constraints {
        let mut parents = c.parents.clone();
        parents.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(parents.len(), 2);
        assert!((parents[0].1 - 0.5).abs() < 1e-14);
        assert!((parents[1].1 - 0.5).abs() < 1e-14);
    }
}

/// Census: hp accepts Tri6 rows too — cell 0 p1, cell 1 p2: 4 vertices + 3
/// edge dofs (shared edge p2 variant + the two cell-1-only edges).
#[test]
fn d824_hp_tri6_rows_accepted() {
    let msh = read_msh(FIXTURE_TRI6.as_bytes()).expect("parse Tri6 fixture");
    let mesh = msh.into_2d().expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tri6);
    let orders: Vec<u8> = vec![1, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);

    assert_eq!(dm.n_dofs, 9 + 3, "9 node ids + 3 edge dofs");
    assert_eq!(dm.n_dofs - (mesh.n_nodes() - 4), 7, "net of gaps = 4 corners + 3 edges");
    assert_eq!(dm.element_dofs(0).len(), 3, "p1 cell: corners only");
    assert_eq!(dm.element_dofs(1).len(), 6, "p2 cell: 3 corners + 3 edges");

    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 1, "the shared edge is the only mixed edge");
}
