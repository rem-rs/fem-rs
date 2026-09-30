//! D824-1 (round 84, fix agent A) — P1 spaces on **quadratic geometry rows of
//! every family** number the corner vertices only.
//!
//! Round-84 lane A fixed the Quad8/Quad9 P1 (D820-A); the same family disease
//! remained for the other quadratic row labels: `build_p1` treated all row
//! nodes as P1 dofs (Tri6 → 6, Tet10 → 10, Hex27 → 27, Prism18 → 18,
//! Pyramid13 → 13 per cell) where MFEM keeps one geometry per family and
//! carries only the row's **corner** nodes in the vertex table.
//!
//! MFEM 4.10 ground truth (probe `tmp/d84fixA/h1p1_probe.cpp`, WSL
//! `$HOME/mfem410_ser`, straight-sided two-cell Gmsh fixtures
//! `data/d824_*_pair.msh`, outputs `tmp/d84fixA/probe_*.txt`):
//!
//! ```text
//! tri6    (Gmsh 9):  NE=2 NV=4   H1(1) NDofs=4    rows 0 1 2 | 0 2 3
//! tet10   (Gmsh 11): NE=2 NV=5   H1(1) NDofs=5    rows 0 1 2 3 | 1 2 3 4
//! hex27   (Gmsh 12): NE=2 NV=12  H1(1) NDofs=12   rows 0..7 | 4..11
//! prism18 (Gmsh 13): NE=2 NV=8   H1(1) NDofs=8    rows 0 1 2 4 5 6 | 1 3 2 5 7 6
//! pyramid (curved):  NE=48 NV=35 (d235 mesh + SetCurvature(2), probe
//!                    `tmp/d84fixA/probe_pyr235.txt`) — H1(1) NDofs=35,
//!                    5 dofs per cell (the element's vertex list).
//! ```
//!
//! The vertex ids are the compacted corner set in ascending node order, and
//! the per-cell row is the element's corner list *in local vertex order* —
//! for Prism18 the fem-rs row is the layer-major `PrismPk` lattice (D319), so
//! the corners sit at slots `[0, 1, 2, 12, 13, 14]`, NOT at the row prefix
//! (the frozen `GMSH_PERM_PRISM18` inverse is consistent with MFEM's
//! `HOPrismMapping`/`WedgeToGmshPrism` slot-for-slot: file slots 3..5 are the
//! top-triangle corners).
//!
//! ```text
//! cargo test -p fem-space --test d824_ho_row_p1_corner_view -- --nocapture
//! ```

use fem_element::lagrange::PyramidBasisType;
use fem_io::gmsh::read_msh;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::dof_manager::DofManager;
use fem_space::fe_space::FESpace;
use fem_space::h1::H1Space;
use fem_space::ref_elem::h1_field_element;

const FIXTURE_TRI6: &str = include_str!("../../../data/d824_tri6_pair.msh");
const FIXTURE_TET10: &str = include_str!("../../../data/d824_tet10_pair.msh");
const FIXTURE_HEX27: &str = include_str!("../../../data/d824_hex27_pair.msh");
const FIXTURE_PRISM18: &str = include_str!("../../../data/d824_prism18_pair.msh");
const FIXTURE_PYR13: &str = include_str!("../../../data/d824_pyr13_pair.msh");

/// MFEM 4.10 H1(1) mass oracles (`MassIntegrator`, explicit order-10 rule;
/// probe `tmp/d84fixA/h1p1_probe.cpp`, trimmed `M i j value` dumps).
const ORACLE_HEX27: &str = include_str!("oracle_d824_h1_mass_hex27.txt");
const ORACLE_PRISM18: &str = include_str!("oracle_d824_h1_mass_prism18.txt");
const ORACLE_TET10: &str = include_str!("oracle_d824_h1_mass_tet10.txt");
const ORACLE_TRI6: &str = include_str!("oracle_d824_h1_mass_tri6.txt");

/// Compare an assembled H¹(1) mass against the MFEM 4.10 oracle.
///
/// Both sides integrate the degree-2 mass integrand on the straight order-2
/// geometry with an explicit order-10 rule â exactly, up to floating-point
/// round-off â so the oracles must be reproduced to 1e-12 (the order-40
/// band first probed here accumulated ~1e-10 of summation noise on the
/// 56k-point order-40 prism rule, on both sides).
#[track_caller]
fn assert_mass_matches_mfem(
    oracle: &str,
    n_dofs: usize,
    mass: &fem_linalg::CsrMatrix<f64>,
    tol: f64,
    what: &str,
) {
    let mut checked = 0usize;
    for line in oracle.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"M") {
            continue;
        }
        let (i, j): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
        let want: f64 = f[3].parse().unwrap();
        let got = mass.get(i, j);
        assert!(
            (got - want).abs() <= tol,
            "{what} M[{i}][{j}]: {got} vs MFEM {want}"
        );
        checked += 1;
    }
    assert!(checked > 0, "{what}: oracle parsed");
    assert_eq!(mass.nrows, n_dofs, "{what}: matrix size vs NDofs");
}

/// Tri6 (Gmsh type 9, 2-D): P1 numbers the 4 mesh vertices (3 per cell), not
/// the 9 geometry nodes.  Pre-fix this numbered the whole 6-node rows.
#[test]
fn d824_tri6_p1_numbers_corner_vertices_only() {
    let msh = read_msh(FIXTURE_TRI6.as_bytes()).expect("parse Tri6 fixture");
    let mesh = msh.into_2d().expect("2-D mesh");
    assert_eq!(mesh.n_elements(), 2);
    assert_eq!(mesh.element_type(0), ElementType::Tri6);
    assert_eq!(mesh.n_nodes(), 9, "fixture carries 9 geometry nodes");

    let space = H1Space::new(mesh, 1);
    assert_eq!(space.n_dofs(), 4, "MFEM probe: H1(1) NDofs = NV = 4");
    assert_eq!(space.element_dofs(0), &[0, 1, 2], "cell 0 corners");
    assert_eq!(space.element_dofs(1), &[0, 2, 3], "cell 1 corners");

    // The 5 mid-edge geometry nodes are not P1 dofs (pre-fix state asserted
    // negatively: they must never enter the vertex dof table).
    let dm = space.dof_manager();
    assert_eq!(dm.n_vertex_dofs, 4);
    assert_eq!(dm.dof_coords.len(), 4 * 2);
}

/// Tet10 (Gmsh type 11): P1 numbers the 5 mesh vertices (4 per cell), not the
/// 14 geometry nodes.  The reader permutes rows to `H1TetPk` slot order, so
/// the corners keep the row prefix.
#[test]
fn d824_tet10_p1_numbers_corner_vertices_only() {
    let msh = read_msh(FIXTURE_TET10.as_bytes()).expect("parse Tet10 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Tet10);
    assert_eq!(mesh.n_nodes(), 14);

    let space = H1Space::new(mesh, 1);
    assert_eq!(space.n_dofs(), 5, "MFEM probe: H1(1) NDofs = NV = 5");
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3]);
    assert_eq!(space.element_dofs(1), &[1, 2, 3, 4]);
}

/// Hex27 (Gmsh type 12): P1 numbers the 12 mesh vertices (8 per cell), not
/// the 45 geometry nodes.
#[test]
fn d824_hex27_p1_numbers_corner_vertices_only() {
    let msh = read_msh(FIXTURE_HEX27.as_bytes()).expect("parse Hex27 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Hex27);
    assert_eq!(mesh.n_nodes(), 45);

    let space = H1Space::new(mesh, 1);
    assert_eq!(space.n_dofs(), 12, "MFEM probe: H1(1) NDofs = NV = 12");
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(space.element_dofs(1), &[4, 5, 6, 7, 8, 9, 10, 11]);
}

/// Prism18 (Gmsh type 13) — the NON-PREFIX family: the fem-rs row is the
/// layer-major `PrismPk` lattice (D319) with the 6 corners at slots
/// `[0, 1, 2, 12, 13, 14]`; P1 must still number exactly the compacted corner
/// set in vertex order (MFEM probe rows below).
#[test]
fn d824_prism18_p1_numbers_corner_vertices_only() {
    let msh = read_msh(FIXTURE_PRISM18.as_bytes()).expect("parse Prism18 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Prism18);
    assert_eq!(mesh.n_nodes(), 27);

    let space = H1Space::new(mesh, 1);
    assert_eq!(space.n_dofs(), 8, "MFEM probe: H1(1) NDofs = NV = 8");
    // Cell 0's 4th dof is the TOP corner (node id 4), not row slot 3's
    // bottom-edge midpoint (node id 8) — the non-prefix proof.
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 4, 5, 6], "MFEM elem 0");
    assert_eq!(space.element_dofs(1), &[1, 3, 2, 5, 7, 6], "MFEM elem 1");
}

/// Pyramid13 (Gmsh code 19): P1 numbers the 6 mesh vertices (5 per cell), not
/// the 18 geometry nodes.  MFEM parity for this family is count-level: MFEM's
/// Gmsh reader takes only the 14-node code-14 pyramid and re-orders the
/// winding on read (probe `tmp/d84fixA/probe_pyr14.txt`), but the semantic —
/// a curved (order-2) PYRAMID mesh carries NV = the corner set and H1(1)
/// numbers 5 dofs per cell over it — is pinned on MFEM's own d235 mesh
/// (`tmp/d84fixA/probe_pyr235.txt`: 48 pyramids + SetCurvature(2) →
/// NV=35 = NDofs, rows = the cells' vertex lists).
#[test]
fn d824_pyr13_p1_numbers_corner_vertices_only() {
    let msh = read_msh(FIXTURE_PYR13.as_bytes()).expect("parse Pyramid13 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Pyramid13);
    assert_eq!(mesh.n_nodes(), 18);

    let space = H1Space::new(mesh, 1);
    assert_eq!(space.n_dofs(), 6, "P1 = the 6 corners of the bipyramid");
    // Corners in each row's local vertex order as COMPACTED dof ids: the
    // corner view is the ascending node set {0,1,2,3,4,13}, so the second
    // apex (node 13) carries dof id 5.
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4]);
    assert_eq!(space.element_dofs(1), &[1, 0, 3, 2, 5]);
}

/// Census (discipline ⑰): the remaining quadratic row labels `Hex20` and
/// `Prism15` take the same corner view (8 of 20 / 6 of 15 row nodes).
#[test]
fn d824_hex20_prism15_corner_view_counts() {
    type NodeId = u32;
    let mut coords: Vec<f64> = Vec::new();
    let push = |p: [f64; 3], coords: &mut Vec<f64>| coords.extend_from_slice(&p);
    for z in 0..2usize {
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            push([x, y, z as f64], &mut coords);
        }
    }
    // Hex20 row: 8 corners (nodes 0..7) + 12 edge mids appended.
    for k in 0..12u32 {
        push([0.5 + 0.01 * k as f64, -1.0, -2.0], &mut coords);
    }
    let hex20: fem_mesh::Mesh<3> = fem_mesh::Mesh {
        coords,
        conn: (0..20u32).collect(),
        elem_tags: vec![1],
        elem_type: ElementType::Hex20,
        face_conn: vec![0, 3, 2, 1],
        face_tags: vec![1],
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    };
    let space = H1Space::new(hex20, 1);
    assert_eq!(space.n_dofs(), 8, "Hex20 row of 20: P1 = 8 corners");
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4, 5, 6, 7]);

    // Prism15 row: 6 corners (nodes 0..5) + 9 edge mids appended.
    let mut coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // bottom tri
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, // top tri
    ];
    for k in 0..9u32 {
        coords.extend_from_slice(&[0.5 + 0.01 * k as f64, -1.0, -2.0]);
    }
    let prism15: fem_mesh::Mesh<3> = fem_mesh::Mesh {
        coords,
        conn: (0..15u32).collect(),
        elem_tags: vec![1],
        elem_type: ElementType::Prism15,
        face_conn: vec![0, 2, 1],
        face_tags: vec![1],
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    };
    let space = H1Space::new(prism15, 1);
    assert_eq!(space.n_dofs(), 6, "Prism15 row of 15: P1 = 6 corners");
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3, 4, 5]);
}

/// A mesh mixing linear and quadratic rows keeps the compacted corner view:
/// the Tet4 row donates all its nodes, the Tet10 row only its 4 corners
/// (MFEM `RemoveUnusedVertices` semantics — probe: NV excludes the mid-edge
/// geometry nodes of the quadratic cell).
#[test]
fn d824_mixed_linear_quadratic_rows_corner_view() {
    type NodeId = u32;
    // Tet4 (A,B,C,D) + Tet10 (B,C,D,E) sharing the face BCD.  Node ids follow
    // first discovery: 5 corners, then the 6 Tet10 row mids (H1TetPk edge
    // order (B,C) (B,D) (B,E) (C,D) (C,E) (D,E)).
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, // A = 0
        1.0, 0.0, 0.0, // B = 1
        0.0, 1.0, 0.0, // C = 2
        0.0, 0.0, 1.0, // D = 3
        1.0, 1.0, 1.0, // E = 4
        0.5, 0.5, 0.0, // mBC = 5
        0.5, 0.0, 0.5, // mBD = 6
        1.0, 0.5, 0.5, // mBE = 7
        0.0, 0.5, 0.5, // mCD = 8
        0.5, 1.0, 0.5, // mCE = 9
        0.5, 0.5, 1.0, // mDE = 10
    ];
    let tet10_row: [NodeId; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mesh: fem_mesh::Mesh<3> = fem_mesh::Mesh {
        coords,
        conn: [0u32, 1, 2, 3].into_iter().chain(tet10_row).collect(),
        elem_tags: vec![1, 1],
        elem_type: ElementType::Tet4,
        face_conn: vec![0, 2, 1],
        face_tags: vec![1],
        face_type: ElementType::Tri3,
        elem_types: Some(vec![ElementType::Tet4, ElementType::Tet10]),
        elem_offsets: Some(vec![0, 4, 14]), // conn offsets: Tet4 row of 4, Tet10 row of 10
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    };
    let space = H1Space::new(mesh, 1);
    assert_eq!(space.n_dofs(), 5, "corner set of both rows");
    assert_eq!(space.element_dofs(0), &[0, 1, 2, 3], "linear row untouched");
    assert_eq!(space.element_dofs(1), &[1, 2, 3, 4], "quadratic row corners");
}

/// The `h1_field_element` P1 pairing (the element the assembler slots against
/// the space's dof rows) is the per-family linear element on every quadratic
/// row label — 3/4/8/6/5 dofs, corners at the reference corners (⑰).
#[test]
fn d824_h1_field_element_p1_counts_per_family() {
    let pyr = PyramidBasisType::default();
    let cases: &[(ElementType, usize)] = &[
        (ElementType::Tri6, 3),
        (ElementType::Tet10, 4),
        (ElementType::Hex20, 8),
        (ElementType::Hex27, 8),
        (ElementType::Prism15, 6),
        (ElementType::Prism18, 6),
        (ElementType::Pyramid13, 5),
    ];
    for (t, want) in cases {
        let fe = h1_field_element(*t, 1, pyr);
        assert_eq!(fe.n_dofs(), *want, "{t:?} P1 field element dof count");
    }
    // The Hex27 P1 lattice is the Q1 hex corners.
    let hex = h1_field_element(ElementType::Hex27, 1, pyr);
    let mut corners = hex.dof_coords();
    corners.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])).then(a[2].total_cmp(&b[2])));
    assert_eq!(
        corners,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ]
    );
}

/// End to end (red → green): the assembled global H¹(1) mass matrix on the
/// Hex27 pair matches MFEM 4.10 entry-wise (both sides integrate the same P1
/// space on the same straight order-2 geometry with an order-10 rule).
#[test]
fn d824_global_mass_matrix_hex27_matches_mfem() {
    let msh = read_msh(FIXTURE_HEX27.as_bytes()).expect("parse Hex27 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    let space = H1Space::new(mesh, 1);
    let mass = fem_assembly::assembler::Assembler::assemble_bilinear(
        &space,
        &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }],
        10,
    );
    assert_mass_matches_mfem(ORACLE_HEX27, space.n_dofs(), &mass, 1e-12, "Hex27 H1(1)");
}

/// End to end (red → green): the Prism18 mass matrix (the non-prefix corner
/// family) matches MFEM entry-wise.
#[test]
fn d824_global_mass_matrix_prism18_matches_mfem() {
    let msh = read_msh(FIXTURE_PRISM18.as_bytes()).expect("parse Prism18 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    let space = H1Space::new(mesh, 1);
    let mass = fem_assembly::assembler::Assembler::assemble_bilinear(
        &space,
        &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }],
        10,
    );
    assert_mass_matches_mfem(ORACLE_PRISM18, space.n_dofs(), &mass, 1e-12, "Prism18 H1(1)");
}

/// End to end: Tet10 and Tri6 mass matrices match MFEM entry-wise as well.
#[test]
fn d824_global_mass_matrix_tet10_tri6_match_mfem() {
    {
        let msh = read_msh(FIXTURE_TET10.as_bytes()).expect("parse Tet10 fixture");
        let mesh = msh.into_3d().expect("3-D mesh");
        let space = H1Space::new(mesh, 1);
        let mass = fem_assembly::assembler::Assembler::assemble_bilinear(
            &space,
            &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }],
            10,
        );
        assert_mass_matches_mfem(ORACLE_TET10, space.n_dofs(), &mass, 1e-12, "Tet10 H1(1)");
    }
    {
        let msh = read_msh(FIXTURE_TRI6.as_bytes()).expect("parse Tri6 fixture");
        let mesh = msh.into_2d().expect("2-D mesh");
        let space = H1Space::new(mesh, 1);
        let mass = fem_assembly::assembler::Assembler::assemble_bilinear(
            &space,
            &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }],
            10,
        );
        assert_mass_matches_mfem(ORACLE_TRI6, space.n_dofs(), &mass, 1e-12, "Tri6 H1(1)");
    }
}

/// Negative pin (the pre-fix state, stated positively): a quadratic row's
/// non-corner geometry nodes are never P1 dofs — the DofManager at order 1 on
/// the Prism18 fixture numbers exactly the 8 corners.
#[test]
fn d824_p1_never_numbers_quadratic_geometry_nodes() {
    let msh = read_msh(FIXTURE_PRISM18.as_bytes()).expect("parse Prism18 fixture");
    let mesh = msh.into_3d().expect("3-D mesh");
    let dm = DofManager::new(&mesh, 1);
    assert_eq!(dm.n_dofs, 8);
    assert_eq!(dm.dof_coords.len(), 8 * 3);
}
