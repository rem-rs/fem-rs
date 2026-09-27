//! D824-A (round 85, lane C) — hp (variable-order) spaces accept PYRAMID rows.
//!
//! Round-84 D820-2 made `p_refine.rs` dispatch by element type and released
//! Tri6/Tet10/Hex20/Hex27/Prism15/Prism18 rows, but pyramid rows were still
//! rejected by the `build_variable_order_dof_manager` geometry assert (the
//! round-84 debt list item 1), and the pyramid `elem_uses_gll` value was
//! unproven (debt item 3, D824-C).
//!
//! MFEM 4.10 ground truth (probe `tmp/d85c/pyr_hp_probe.cpp`, mixed-order
//! H1 pyramid triple on `EnsureNCMesh` + SetElementOrder — outputs
//! `tmp/d85c/out_p211.txt` / `out_p311.txt`; a disjoint hex is attached in
//! the probe mesh only because `Mesh::EnsureNCMesh` skips meshes whose
//! meshgen bit is the pyramid-only 0x8, `mesh/mesh.cpp:11781`):
//!
//! ```text
//! p=[2,1,1]: NDofs=45 (pyramid-only part: 18)
//! elem 0 p2 (15 dofs): 0 1 2 3 4 | 4 base-edge + 4 lateral-edge | base quad face | interior
//! p=[3,1,1]: NDofs=104 (pyramid-only part: 40); elem 0 p3 (37 dofs)
//! constrained 16 <- (0, +0.723606798) (1, +0.276393202)      # GLL edge weights
//! constrained 56 <- (3, +0.5236) (2, +0.2) (1, +0.0764) (0, +0.2)   # Fuentes j-reversed base face
//! constrained 61 <- (1, +0.3333) (2, +0.3333) (4, +0.3333)   # shared TRI face (1,2,4)
//! ```
//!
//! Semantics pinned (fe_h1.cpp:1043-1150, `H1_FuentesPyramidElement`, the
//! `H1_FECollection` default pyramid family):
//! * edges: 8 blocks in `Geometry::Constants<PYRAMID>::Edges` order and
//!   direction `(0,1) (1,2) (3,2) (0,3) (0,4) (1,4) (2,4) (3,4)`
//!   (`fem/geom.cpp:1076`) — the p3 element row lists the (3,2)-edge dofs
//!   reversed against the stored ascending order;
//! * base quad face (face 0) interior layout `(cp[i], cp[p−j], 0)` — the y
//!   axis j-REVERSED against the hex tensor convention (the p3 probe weights
//!   above put the FIRST base-face dof at `(cp[1], cp[p−1])`, where the
//!   hex-style layout would sit at `(cp[1], cp[1])` with weight 0.5236 on
//!   v0 instead of 0.2);
//! * 4 triangular side faces in `Faces` order `(0,1,4) (1,2,4) (2,3,4)
//!   (3,0,4)` at the `H1_TriangleElement`/`H1TriPk` GLL barycentrics;
//! * interior `(p−1)³` Fuentes bubble nodes `cp[k]` layers, in-plane
//!   `(cp[i], cp[j])` with i fastest;
//! * D824-C: every 1-D distribution is CLOSED Gauss-Lobatto
//!   (`Poly1D::ClosedPoints(p, GaussLobatto)`, `fe_h1.cpp:1045`; probe p3
//!   edge nodes at `0.5·(1 ∓ 1/√5)`, not `1/3, 2/3`).
//!
//! D824-B counting basis (unchanged this round): hp vertex dofs keep node-id
//! identity, so on quadratic Pyramid13 rows the space's `n_dofs` counts the
//! unused mid-node ids (gaps).  MFEM's vertex table is the compacted corner
//! set, hence the structural comparison is `n_dofs − (n_nodes − n_corners)`
//! ("net of gaps") on the quadratic fixture and exact `n_dofs == MFEM NDofs`
//! on the linear one.
//!
//! ```text
//! cargo test -p fem-space --test d824_hp_p_refine_pyramid_rows -- --nocapture
//! ```

use fem_io::gmsh::read_msh;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::dof_manager::DofManager;
use fem_space::p_refine::{build_variable_order_dof_manager, detect_p_constraints};

const FIXTURE_PYR5: &str = include_str!("../../../data/d824_pyr5_hp_triple.msh");
const FIXTURE_PYR13: &str = include_str!("../../../data/d824_pyr13_hp_triple.msh");

/// cp(1) and cp(2) of the p3 closed Gauss-Lobatto points on [0, 1]:
/// `0.5·(1 ∓ 1/√5)`.
fn cp1() -> f64 {
    0.5 * (1.0 - 1.0 / 5.0_f64.sqrt())
}
fn cp2() -> f64 {
    0.5 * (1.0 + 1.0 / 5.0_f64.sqrt())
}

fn pyr5_mesh() -> fem_mesh::Mesh<3> {
    read_msh(FIXTURE_PYR5.as_bytes()).expect("parse Pyramid5 fixture").into_3d().expect("3-D")
}

fn pyr13_mesh() -> fem_mesh::Mesh<3> {
    read_msh(FIXTURE_PYR13.as_bytes()).expect("parse Pyramid13 fixture").into_3d().expect("3-D")
}

fn sorted(mut parents: Vec<(u32, f64)>) -> Vec<(u32, f64)> {
    parents.sort_by_key(|&(d, _)| d);
    parents
}

/// Parent list equality up to `1e-12` (the constraint weights come out of a
/// Lagrange-basis evaluation, so they are not bit-identical to the closed
/// products of the GLL constants).
fn assert_parents(actual: &[(u32, f64)], expected: &[(u32, f64)], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: parent count");
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.0, e.0, "{context}: parent dof");
        assert!((a.1 - e.1).abs() < 1e-12, "{context}: weight {} vs {}", a.1, e.1);
    }
}

/// hp on the straight Pyramid5 triple, orders [2, 1, 1]: cell 0 (p2) carries
/// 5 corners + 8 edge dofs + 1 base-face dof + 1 interior = the Fuentes p2
/// row; cells 1/2 (p1) carry their corners only.  The linear mesh has no
/// node-id gaps, so `n_dofs` is EXACTLY MFEM's NDofs for the pyramid part of
/// the probe mesh (18; probe run `out_p211.txt`: 45 total − 23 hex dofs).
#[test]
fn d824_hp_pyr5_p211_matches_mfem() {
    let mesh = pyr5_mesh();
    assert_eq!(mesh.n_nodes(), 8);
    assert_eq!(mesh.element_type(0), ElementType::Pyramid5);
    let orders: Vec<u8> = vec![2, 1, 1];
    let dm = build_variable_order_dof_manager(&mesh, &orders);

    assert_eq!(dm.n_dofs, 18, "8 corners + 8 edge dofs + 1 base face + 1 interior");

    // Cell 0 row: corners, 8 edge dofs in MFEM PYRAMID Edges order, base
    // quad face, interior (tri faces hold no dofs at p2).
    assert_eq!(
        dm.element_dofs(0),
        &[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    );
    // Cells 1/2: p1 corners only, input orientation preserved.
    assert_eq!(dm.element_dofs(1), &[0, 3, 2, 1, 5]);
    assert_eq!(dm.element_dofs(2), &[1, 2, 6, 7, 4]);

    // Constraints: the 6 mixed-order edges (4 base + the 2 laterals shared
    // with cell 2) with 0.5/0.5 weights, and the shared base quad face dof
    // on the p1 trace = 4 corners x 0.25 (probe dofs 16-19, 21, 22, 36).
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 7);

    let expect: Vec<(u32, Vec<(u32, f64)>)> = vec![
        (8, vec![(0, 0.5), (1, 0.5)]),   // base edge (0,1)  <- MFEM 16
        (9, vec![(1, 0.5), (2, 0.5)]),   // base edge (1,2)  <- MFEM 17
        (10, vec![(2, 0.5), (3, 0.5)]),  // base edge (2,3)  <- MFEM 18
        (11, vec![(0, 0.5), (3, 0.5)]),  // base edge (0,3)  <- MFEM 19
        (13, vec![(1, 0.5), (4, 0.5)]),  // lateral (1,4)    <- MFEM 21
        (14, vec![(2, 0.5), (4, 0.5)]),  // lateral (2,4)    <- MFEM 22
        (16, vec![(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]), // base face <- MFEM 36
    ];
    for (constrained, parents) in expect {
        let c = constraints.iter().find(|c| c.constrained == constrained)
            .unwrap_or_else(|| panic!("missing constraint on dof {constrained}"));
        assert_parents(&sorted(c.parents.clone()), &sorted(parents), "pyr5 p211");
    }
}

/// hp on the straight Pyramid5 triple, orders [3, 1, 1]: cell 0 (p3) is the
/// full 37-dof Fuentes row — the (3,2)-edge block runs REVERSED against the
/// stored edge order, the face block is [base quad x4 | tri faces in Faces
/// order], the interior holds (p-1)^3 = 8 — and `n_dofs` = 40 exactly equals
/// MFEM's pyramid-only NDofs (`out_p311.txt`: 104 − 64 hex dofs).
#[test]
fn d824_hp_pyr5_p311_matches_mfem() {
    let mesh = pyr5_mesh();
    let orders: Vec<u8> = vec![3, 1, 1];
    let dm = build_variable_order_dof_manager(&mesh, &orders);

    assert_eq!(dm.n_dofs, 40, "8 corners + 16 edges + 8 faces + 8 interior");

    // Cell 0 row (37 dofs).  Edge block: (0,1) (1,2) (3,2)! (0,3) then the
    // laterals; face block: base quad (0,1,2,3) then tri faces (0,1,4)
    // (1,2,4) (2,3,4) (3,0,4) — fem-rs dof ids: base face 24-27, tri faces
    // 28/(1,2,4)->30/(2,3,4)->31/(3,0,4)->29 (FaceKey-sorted variant store).
    let mut row: Vec<u32> = (0..5).collect(); // corners
    row.extend([8, 9, 10, 11, 13, 12, 14, 15]); // edges, (3,2) reversed
    row.extend([16, 17, 18, 19, 20, 21, 22, 23]); // laterals
    row.extend([24, 25, 26, 27]); // base quad face
    row.extend([28, 30, 31, 29]); // tri faces in Faces order
    row.extend(32..40); // interior
    assert_eq!(dm.element_dofs(0), &row, "cell 0 (p3) row");
    assert_eq!(dm.element_dofs(0).len(), 37, "Fuentes p3 dof count");
    assert_eq!(dm.element_dofs(1), &[0, 3, 2, 1, 5]);
    assert_eq!(dm.element_dofs(2), &[1, 2, 6, 7, 4]);

    // Constraints (17): GLL edge weights — the D824-C pin: with the wrong
    // (equispaced) node set these would be 2/3 + 1/3, not 0.7236 + 0.2764.
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 17, "6 edges x 2 + base face x4 + tri face x1");

    let gll_pairs: [(u32, u32, u32); 6] = [
        (8, 0, 1),   // (0,1) <- MFEM 16/17
        (10, 1, 2),  // (1,2) <- MFEM 18/19
        (12, 2, 3),  // (2,3) <- MFEM 20/21
        (14, 0, 3),  // (0,3) <- MFEM 22/23
        (18, 1, 4),  // (1,4) <- MFEM 26/27
        (20, 2, 4),  // (2,4) <- MFEM 28/29
    ];
    for &(first, a, b) in &gll_pairs {
        let c0 = constraints.iter().find(|c| c.constrained == first)
            .unwrap_or_else(|| panic!("missing edge constraint {first}"));
        assert_parents(
            &sorted(c0.parents.clone()),
            &sorted(vec![(a, cp2()), (b, cp1())]),
            "edge dof at t = cp1: GLL master weights",
        );
        let c1 = constraints.iter().find(|c| c.constrained == first + 1)
            .unwrap_or_else(|| panic!("missing edge constraint {}", first + 1));
        assert_parents(
            &sorted(c1.parents.clone()),
            &sorted(vec![(a, cp1()), (b, cp2())]),
            "edge dof at t = cp2",
        );
    }

    // Base quad face dofs at the Fuentes (cp[i], cp[p−j]) positions — the
    // j-REVERSED layout (probe dofs 56-59; the hex-style ascending tensor
    // would swap the 0.0764/0.5236 pattern of dofs 25/26).
    let a2 = cp1() * cp1(); // 0.0764
    let b2 = cp2() * cp2(); // 0.5236
    let ab = cp1() * cp2(); // 0.2
    let expect_face: [(u32, [(u32, f64); 4]); 4] = [
        (24, [(0, ab), (1, a2), (2, ab), (3, b2)]), // (cp1, cp2)
        (25, [(0, a2), (1, ab), (2, b2), (3, ab)]), // (cp2, cp2)
        (26, [(0, b2), (1, ab), (2, a2), (3, ab)]), // (cp1, cp1)
        (27, [(0, ab), (1, b2), (2, ab), (3, a2)]), // (cp2, cp1)
    ];
    for (constrained, parents) in expect_face {
        let c = constraints.iter().find(|c| c.constrained == constrained)
            .unwrap_or_else(|| panic!("missing base face constraint {constrained}"));
        assert_parents(&sorted(c.parents.clone()), &sorted(parents.to_vec()), "pyr5 p311 base face");
    }

    // The shared TRI face (1,2,4) p3 dof interpolates the p1 trace = its
    // three corners with 1/3 each (probe dof 61).
    let c = constraints.iter().find(|c| c.constrained == 30)
        .expect("missing shared tri face constraint");
    assert_parents(
        &sorted(c.parents.clone()),
        &sorted(vec![(1, 1.0 / 3.0), (2, 1.0 / 3.0), (4, 1.0 / 3.0)]),
        "shared tri face p3 dof",
    );
}

/// D824-B counting basis on quadratic Pyramid13 rows: vertex dofs keep
/// node-id identity, so the 17 unused mid-node geometry ids stay counted in
/// `n_dofs` (25 node ids + 16 edge + 8 face + 8 interior = 57) and the MFEM
/// comparison is NET OF GAPS: 57 − (25 − 8) = 40 = MFEM's NDofs.  The cell-0
/// row also pins the Pyramid13 corner view: its first five slots are the
/// row's first five node ids (Gmsh code-19 corners are a prefix), and the
/// constraint structure is the linear fixture's, bit for bit in weights.
#[test]
fn d824_hp_pyr13_p311_net_of_gaps() {
    let mesh = pyr13_mesh();
    assert_eq!(mesh.n_nodes(), 25);
    assert_eq!(mesh.element_type(0), ElementType::Pyramid13);
    let orders: Vec<u8> = vec![3, 1, 1];
    let dm = build_variable_order_dof_manager(&mesh, &orders);

    assert_eq!(dm.n_dofs, 25 + 16 + 8 + 8, "node ids + edge/face/interior dofs");
    assert_eq!(dm.n_dofs - (mesh.n_nodes() - 8), 40, "net of gaps = MFEM NDofs");

    let row0 = dm.element_dofs(0);
    assert_eq!(&row0[..5], &[0, 1, 2, 3, 4], "Pyramid13 corners = row prefix");
    assert_eq!(row0.len(), 37, "Fuentes p3 row over the corner view");
    // Full cell-0 row: edge dofs start AFTER all 25 node ids (25..41), the
    // (3,2)-edge block [29, 30] runs reversed (30, 29), then base face
    // 41-44, tri faces 45 / (1,2,4)->47 / (2,3,4)->48 / (3,0,4)->46,
    // interior 49..57.
    let mut row: Vec<u32> = (0..5).collect();
    row.extend([25, 26, 27, 28, 30, 29, 31, 32]);
    row.extend([33, 34, 35, 36, 37, 38, 39, 40]);
    row.extend([41, 42, 43, 44]);
    row.extend([45, 47, 48, 46]);
    row.extend(49..57);
    assert_eq!(row0, &row, "cell 0 (p3) row over Pyramid13 rows");

    // Same constraint weights as the straight Pyramid5 mesh (same corner
    // geometry): 17 constraints, GLL edges, Fuentes base face, tri face 1/3.
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert_eq!(constraints.len(), 17);
    let c = constraints.iter().find(|c| c.constrained == row0[5])
        .expect("first base-edge constraint");
    assert_parents(
        &sorted(c.parents.clone()),
        &sorted(vec![(0, cp2()), (1, cp1())]),
        "pyr13 first base-edge GLL weights",
    );
    let tri = constraints.iter().find(|c| c.parents.len() == 3)
        .expect("shared tri face constraint");
    assert_parents(
        &sorted(tri.parents.clone()),
        &sorted(vec![(1, 1.0 / 3.0), (2, 1.0 / 3.0), (4, 1.0 / 3.0)]),
        "pyr13 shared tri face",
    );
}

/// Zero-regression parity with the fixed-order path: a UNIFORM order through
/// the variable-order builder must produce the same dof count as the
/// fixed-order Fuentes `DofManager` (same entities, single variant each —
/// the stacked base-to-base pair shares its base quad face, so face dofs
/// dedup identically in both).
#[test]
fn d824_hp_pyr_uniform_parity_with_fixed_order() {
    let mesh = pyr5_mesh();
    for p in [1u8, 2, 3] {
        let dm_var = build_variable_order_dof_manager(&mesh, &vec![p; 3]);
        let dm_fix = DofManager::new(&mesh, p);
        assert_eq!(dm_var.n_dofs, dm_fix.n_dofs, "uniform p{p}: hp vs fixed order");
        assert_eq!(dm_var.element_dofs(0).len(), dm_fix.element_dofs(0).len(),
            "uniform p{p}: cell 0 row length");
        assert!(detect_p_constraints(&dm_var, &mesh, &vec![p; 3]).is_empty(),
            "uniform p{p}: no constraints");
    }
}
