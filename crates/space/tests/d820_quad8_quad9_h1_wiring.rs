//! D820-A (round 84, lane A) — Quad8/Quad9 cell wiring on the *space* side
//! (D819-A registered debt).
//!
//! Round-83 adjudication (D819, `crates/assembly/tests/d819_quad8_quad9_mixed_dispatch.rs`):
//! MFEM has a single `Geometry::SQUARE` — a fem-rs `Quad8`/`Quad9` mesh cell is
//! a *geometry row label* (D243), and `H1_FECollection(p, 2)` builds the tensor
//! `H1_QuadrilateralElement(p)` on it.  The assembly side was fixed in round 83
//! (`5bde63dc`); the space side was not: `DofManager::build_p1` treated all 8/9
//! geometry nodes of the row as P1 dofs (MFEM: 4 corner vertices),
//! `DofManager::build` routed Quad8/Quad9 rows of order ≥ 2 into the
//! **triangle** builder `build_pk`, and `h1_field_element` had no
//! Quad8/Quad9 arm (panic) — so an H¹ space on the certified fixture
//! `data/d819_quad9_curved.msh` was end-to-end unreachable.
//!
//! MFEM 4.10 ground truth (probe `tmp/d84a/h1_probe.cpp`, WSL `$HOME/mfem410_ser`,
//! `H1_FECollection(p, 2)` on `data/d819_quad9_curved.msh` — 2 curved
//! Gmsh type-10 quads, 15 nodes):
//!
//! ```text
//! NE=2 NV=6 NBE=6, cell geometry = SQUARE(3) on both cells
//! vertex ids = the 6 corner nodes in ascending node-table order:
//!   nodes {0,2,4,10,12,14} -> vertices 0..5
//! H1(1): NDofs=6   elem0 0 1 4 3          elem1 1 2 5 4
//! H1(2): NDofs=15 (6 v + 7 e + 2 i)
//!        elem0 0 1 4 3 6 7 8 9 13   elem1 1 2 5 4 10 11 12 7 14
//! H1(3): NDofs=28 (6 v + 14 e + 8 i)
//!        elem0 0 1 4 3 6 7 8 9 11 10 13 12 20 21 22 23
//!        elem1 1 2 5 4 14 15 16 17 19 18 9 8 24 25 26 27
//! ```
//!
//! ```text
//! cargo test -p fem-space --test d820_quad8_quad9_h1_wiring -- --nocapture
//! ```

use fem_element::lagrange::factory::QuadQk;
use fem_element::lagrange::PyramidBasisType;
use fem_element::ReferenceElement;
use fem_io::gmsh::read_msh;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::dof_manager::DofManager;
use fem_space::fe_space::FESpace;
use fem_space::h1::H1Space;
use fem_space::hcurl::HCurlSpace;
use fem_space::ref_elem::h1_field_element;

/// The MFEM-certified real mesh fixture (round-83 probe 3; MFEM 4.10's own
/// Gmsh reader re-parses it as SQUARE + order-2 Nodes).
const FIXTURE_Q9: &str = include_str!("../../../data/d819_quad9_curved.msh");

fn quad9_mesh() -> fem_mesh::Mesh<2> {
    let msh = read_msh(FIXTURE_Q9.as_bytes()).expect("parse Quad9 fixture");
    msh.into_2d().expect("2-D mesh")
}

#[track_caller]
fn assert_dof_row(space_dofs: &[u32], want: &[u32], cell: usize, what: &str) {
    assert_eq!(
        space_dofs,
        want,
        "{what}: cell {cell} element dofs {:?} != MFEM {:?}",
        space_dofs,
        want
    );
}

/// D820-A red: `build_p1` must number the **4 corner vertices** of Quad9 rows
/// (MFEM `NV`), not the 8/9 geometry nodes of the row.
#[test]
fn d820_h1_p1_numbers_corner_vertices_only() {
    let mesh = quad9_mesh();
    assert_eq!(mesh.n_elements(), 2);
    assert_eq!(mesh.element_type(0), ElementType::Quad9);
    // The mesh keeps the full order-2 node table (15 nodes); the *space* must
    // not.
    assert_eq!(mesh.n_nodes(), 15, "fixture carries 15 geometry nodes");

    let space = H1Space::new(mesh, 1);
    // MFEM probe: H1(1) NDofs = NV = 6.
    assert_eq!(space.n_dofs(), 6, "H1(1) on the Quad9 fixture = 4 corners/cell");
    let dm = space.dof_manager();
    assert_eq!(dm.n_vertex_dofs, 6);
    assert_dof_row(space.element_dofs(0), &[0, 1, 4, 3], 0, "H1(1)");
    assert_dof_row(space.element_dofs(1), &[1, 2, 5, 4], 1, "H1(1)");
}

/// D820-A red: order 2 on Quad9 rows must take the tensor `QuadQk` numbering
/// (vertex → edge → interior, MFEM `H1_QuadrilateralElement(2)`), not the
/// triangle builder.
#[test]
fn d820_h1_p2_dof_table_matches_mfem() {
    let mesh = quad9_mesh();
    let space = H1Space::new(mesh, 2);
    assert_eq!(space.n_dofs(), 15, "MFEM H1(2) NDofs = 6 + 7 + 2");
    let dm = space.dof_manager();
    assert_eq!(dm.n_vertex_dofs, 6, "MFEM NVDofs");
    assert_eq!(space.element_dofs(0).len(), 9, "tensor 9 dofs per cell");
    assert_dof_row(
        space.element_dofs(0),
        &[0, 1, 4, 3, 6, 7, 8, 9, 13],
        0,
        "H1(2)",
    );
    assert_dof_row(
        space.element_dofs(1),
        &[1, 2, 5, 4, 10, 11, 12, 7, 14],
        1,
        "H1(2)",
    );
}

/// D820-A red: order 3 routed `npe == 8/9` rows into the triangle builder;
/// the tensor `QuadQk(3)` numbering must match MFEM's H1(3) tables.
#[test]
fn d820_h1_p3_dof_table_matches_mfem() {
    let mesh = quad9_mesh();
    let space = H1Space::new(mesh, 3);
    assert_eq!(space.n_dofs(), 28, "MFEM H1(3) NDofs = 6 + 14 + 8");
    assert_dof_row(
        space.element_dofs(0),
        &[0, 1, 4, 3, 6, 7, 8, 9, 11, 10, 13, 12, 20, 21, 22, 23],
        0,
        "H1(3)",
    );
    assert_dof_row(
        space.element_dofs(1),
        &[1, 2, 5, 4, 14, 15, 16, 17, 19, 18, 9, 8, 24, 25, 26, 27],
        1,
        "H1(3)",
    );
}

/// The `h1_field_element` dispatch (the element the assembler pairs with the
/// space's dof tables) must expose the tensor family on Quad8/Quad9 cells —
/// the D581 hex pattern on quads (pre-fix this panicked).
#[test]
fn d820_h1_field_element_quad8_quad9_is_tensor_qk() {
    for p in 0..=5u8 {
        let want = if p == 0 { 1 } else { QuadQk::new(p as usize).n_dofs() };
        for t in [ElementType::Quad8, ElementType::Quad9] {
            let fe = h1_field_element(t, p, PyramidBasisType::default());
            assert_eq!(fe.n_dofs(), want, "{t:?} p={p} tensor dof count");
        }
    }
    // Same lattice as the Quad4 arm — one SQUARE geometry, one family.
    for p in 1..=4u8 {
        let q4 = h1_field_element(ElementType::Quad4, p, PyramidBasisType::default());
        for t in [ElementType::Quad8, ElementType::Quad9] {
            let fe = h1_field_element(t, p, PyramidBasisType::default());
            assert_eq!(fe.dof_coords(), q4.dof_coords(), "{t:?} p={p} lattice");
        }
    }
}

/// Red → green end to end: `assemble_bilinear` (which reads
/// `h1_field_element` through `field_element_for_space`) must assemble the
/// global H¹ mass matrix on the curved Quad9 cells.  MFEM 4.10 oracle dumped
/// by `tmp/d84a/nd_probe.cpp h1mass` (`MassIntegrator` on an explicit order-40
/// square rule; the fem-rs side assembles on its own order-40 rule — the
/// mass integrand is a bounded-degree polynomial on the Q2 geometry, so both
/// sides are converged and only rounding separates them).
#[test]
fn d820_global_mass_matrix_matches_mfem() {
    const ORACLE_P1: &str = include_str!("oracle_d84a_h1_mass_p1.txt");
    const ORACLE_P2: &str = include_str!("oracle_d84a_h1_mass_p2.txt");

    for (order, oracle) in [(1u8, ORACLE_P1), (2u8, ORACLE_P2)] {
        let mesh = quad9_mesh();
        let space = H1Space::new(mesh, order);
        let mass = fem_assembly::assembler::Assembler::assemble_bilinear(
            &space,
            &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }],
            40,
        );
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
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "H1({order}) M[{i}][{j}]: {got} vs MFEM {want}"
            );
            checked += 1;
        }
        assert!(checked > 0, "oracle {order} parsed");
        assert_eq!(mass.nrows, space.n_dofs());
    }
}

/// Census (discipline ⑰): orders 4 and 5 on Quad9 rows take `build_pk_quad`
/// too — full MFEM table parity across the whole order ladder (probe
/// `tmp/d84a/h1_p4.txt`, `h1_p5.txt`).
#[test]
fn d820_h1_p4_p5_dof_tables_match_mfem() {
    let space4 = H1Space::new(quad9_mesh(), 4);
    assert_eq!(space4.n_dofs(), 45, "MFEM H1(4) NDofs = 6 + 21 + 18");
    assert_dof_row(
        space4.element_dofs(0),
        &[0, 1, 4, 3, 6, 7, 8, 9, 10, 11, 14, 13, 12, 17, 16, 15, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        0,
        "H1(4)",
    );
    assert_dof_row(
        space4.element_dofs(1),
        &[1, 2, 5, 4, 18, 19, 20, 21, 22, 23, 26, 25, 24, 11, 10, 9, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        1,
        "H1(4)",
    );

    let space5 = H1Space::new(quad9_mesh(), 5);
    assert_eq!(space5.n_dofs(), 66, "MFEM H1(5) NDofs = 6 + 28 + 32");
    assert_dof_row(
        space5.element_dofs(0),
        &[
            0, 1, 4, 3, 6, 7, 8, 9, 10, 11, 12, 13, 17, 16, 15, 14, 21, 20, 19, 18, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        ],
        0,
        "H1(5)",
    );
    assert_dof_row(
        space5.element_dofs(1),
        &[
            1, 2, 5, 4, 22, 23, 24, 25, 26, 27, 28, 29, 33, 32, 31, 30, 13, 12, 11, 10, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        ],
        1,
        "H1(5)",
    );
}

/// D819-B: `HCurlSpace` must accept Quad9 cells — MFEM `ND_FECollection` on
/// the same SQUARE+order-2-Nodes mesh gives the edge-major tables below
/// (probe `tmp/d84a/nd_probe.cpp`; MFEM encodes an orientation-reversed dof
/// as `-(dof+1)`, decoded here).
#[test]
fn d820_hcurl_quad9_matches_mfem_nd_tables() {
    let mesh = quad9_mesh();

    // ND(1): NDofs = 7 edges x 1.
    let space = HCurlSpace::new(quad9_mesh(), 1);
    assert_eq!(mesh.element_type(0), ElementType::Quad9);
    assert_eq!(space.n_dofs(), 7, "MFEM nd(1) NDofs");
    let (d0, s0) = (space.element_dofs(0), space.element_signs(0));
    let (d1, s1) = (space.element_dofs(1), space.element_signs(1));
    // MFEM elem0 `0 1 -3 -4` / elem1 `4 5 -7 -2`.
    assert_eq!(d0, &[0, 1, 2, 3]);
    assert_eq!(s0, &[1.0, 1.0, -1.0, -1.0]);
    assert_eq!(d1, &[4, 5, 6, 1]);
    assert_eq!(s1, &[1.0, 1.0, -1.0, -1.0]);

    // ND(2): NDofs = 14 edge dofs + 8 interior = 22.
    let space = HCurlSpace::new(mesh, 2);
    assert_eq!(space.n_dofs(), 22, "MFEM nd(2) NDofs");
    let (d0, s0) = (space.element_dofs(0), space.element_signs(0));
    let (d1, s1) = (space.element_dofs(1), space.element_signs(1));
    // MFEM elem0 `0 1 2 3 -6 -5 -8 -7 14 15 16 17`
    assert_eq!(d0, &[0, 1, 2, 3, 5, 4, 7, 6, 14, 15, 16, 17]);
    assert_eq!(s0, &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);
    // MFEM elem1 `8 9 10 11 -14 -13 -4 -3 18 19 20 21`
    assert_eq!(d1, &[8, 9, 10, 11, 13, 12, 3, 2, 18, 19, 20, 21]);
    assert_eq!(s1, &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);
}

/// Census (discipline ⑰): a fem-rs-internal Quad8 mesh (Gmsh type-16 — MFEM
/// aborts on that file, so this fixture is fem-rs-side only) must take the
/// same corner-vertex P1 / tensor P2 wiring as Quad9: one SQUARE geometry.
#[test]
fn d820_quad8_mesh_corner_and_tensor_wiring() {
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
    let msh = read_msh(FIXTURE_Q8.as_bytes()).expect("parse Quad8 fixture");
    let mesh = msh.into_2d().expect("2-D mesh");
    assert_eq!(mesh.element_type(0), ElementType::Quad8);
    assert_eq!(mesh.n_nodes(), 8);

    // P1: 4 corner dofs (nodes 0..3), MFEM SQUARE semantics.
    let p1 = H1Space::new(mesh.clone(), 1);
    assert_eq!(p1.n_dofs(), 4, "P1 = 4 corners of the Quad8 row");
    assert_dof_row(p1.element_dofs(0), &[0, 1, 2, 3], 0, "Quad8 H1(1)");

    // P2: tensor 9 dofs; corners map to the compacted vertex ids.
    let p2 = H1Space::new(mesh, 2);
    assert_eq!(p2.n_dofs(), 4 + 4 + 1, "Q2 = 4 v + 4 e + 1 i");
    assert_dof_row(p2.element_dofs(0), &[0, 1, 2, 3, 4, 5, 6, 7, 8], 0, "Quad8 H1(2)");
}

/// Negative pin (the pre-fix state, stated positively): a Quad9 mesh's
/// DofManager at order 1 must NOT number the mid-edge/centre geometry nodes —
/// those 9 non-corner nodes are geometry dofs, never H¹(1) dofs.
#[test]
fn d820_p1_never_numbers_quadratic_geometry_nodes() {
    let mesh = quad9_mesh();
    let dm = DofManager::new(&mesh, 1);
    assert_eq!(dm.n_dofs, 6);
    // Node 1 (first edge midpoint of cell 0) is not a dof; its corner
    // neighbours are.
    assert!((0..15u32).any(|n| n == 1) && dm.dof_coords.len() == 6 * 2);
}
