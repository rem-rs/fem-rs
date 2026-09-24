//! D761 — `error_estimate.rs::geom_rule` + `vertex_shapes` on the `Hex20` /
//! `Hex27` labels (and the same-class quadratic connectivity rows).
//!
//! # The defect
//!
//! * `geom_rule` covered `Tri3/Tri6/Quad4/Tet4/Tet10/Hex8` only — the
//!   quadratic-hex labels hit `other => panic!("geom_rule: unsupported element
//!   type …")` (a *loud* failure, but a hard stop on a reachable path: the
//!   Gmsh reader maps element type 17 → `Hex20` and keeps a 20-node
//!   connectivity row);
//! * `vertex_shapes` dispatched on the **raw `npe`** (`nodes.len()`): `3`, `4`
//!   (2-D and 3-D), `8` were covered and *every quadratic row* — `Tet10` (10),
//!   `Hex20` (20), `Hex27` (27), `Prism18` (18), `Pyramid13` (13) — fell into
//!   `_ => {}` and **silently returned all-zero vertex weights**.  `phys_point`
//!   then evaluated the exact solution at the ORIGIN, and the `Lp` estimator
//!   degenerated to `|u_ex(0) − u_h(ξ_q)|`.  `Tet10` has a `geom_rule` arm, so
//!   that silent wrongness was reachable *today*; the hex labels reached the
//!   same arm as soon as the panic above was lifted.
//!
//! # Fix under test
//!
//! * `geom_rule(Hex20 | Hex27, ·) = hex_rule(·)` — one CUBE geometry, one H¹
//!   family on MFEM's `[0,1]³` hex frame for all three hexahedral cell types
//!   (D581/D721), so the `Hex8` rule is the family rule.  MFEM 4.10's own
//!   `IntRules.Get(CUBE, ·)` is `[0,1]³`-native with weight sum 1 (probe
//!   `tmp/d760/zz_rule_probe.cpp`), i.e. `hex_rule`'s frame.  No serendipity
//!   shape function or rule is involved (the C-lane `serendipity.rs` surface
//!   is untouched);
//! * `vertex_shapes` reads the family's linear reference element from
//!   `fem_space::ref_elem` (`h1_simplex_slots(·, 1)` / `fixed_order_tensor` /
//!   `h1_prism_slots(1)` / `h1_pyramid_slots(1)`) and lays the corner weights
//!   on the first `elem_vertex_count` slots of the row — the extra nodes of a
//!   quadratic row are weighted 0 (the file's linear element-geometry model),
//!   and an unsupported family panics instead of zeroing.
//!
//! # Acceptance
//!
//! 1. **reachability**: a Gmsh v4.1 file with a 20-node hex (type 17) reads to
//!    a `Hex20` element with a 20-node conn, and the estimator path runs on it.
//!    Pre-fix this test panics inside `geom_rule`;
//! 2. **affine exactness on a translated cell**: `u_ex` affine is exactly
//!    representable in P1, so `lp_error_estimator` must return ~0 on the
//!    translated cube for the `Hex20` and `Hex27` labels (and for the
//!    order-2 geometry-table path).  A zero-weight `phys_point` (the D761
//!    silent zero) would report `|u_ex(0) − u_h| ≈ |∇u·x| ≈ 16`, and any
//!    frame/order confusion in `geom_rule` shows up as a volume error;
//! 3. **same-geometry consistency**: the same physical cube written as
//!    `Hex8` / `Hex20` / `Hex27` must give the *same* `Lp` value for the same
//!    `u_ex` (same Gaussian rule, same geometry map, same 8 corner dofs);
//! 4. **curved-table sanity**: a warped order-2 `Hex27` table keeps the
//!    estimator paths finite, positive and in the same ballpark as the
//!    straight cell.
//!
//! Run:
//!   cargo test -p fem-assembly --test d761_hex2027_geom_rule -- --nocapture

use fem_assembly::postproc::error_estimate::{
    lp_error_estimator, zz_estimator_aniso, AnisotropicErrorEstimator,
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_element::lagrange::HexQk;
use fem_element::ReferenceElement;
use fem_io::gmsh::read_msh;
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::GeometryData;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Exactly representable in the P1 space of a straight (or affine-mapped) cell.
fn u_affine(x: &[f64]) -> f64 { 2.0 + 3.0 * x[0] - 1.5 * x[1] + 0.5 * x[2] }
/// Not representable — gives a nonzero Lp/ZZ value to compare across labels.
fn u_quad(x: &[f64]) -> f64 { 0.5 + x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2] }
/// Smooth, non-polynomial, and NOT degenerate on the fixture lattice: the
/// nodal-gradient recovery is genuinely different from the per-element flux
/// (`sin(π·x)` would vanish at every half-integer lattice point of these
/// translated fixtures, where the P1 space only samples corners).
fn u_wave(x: &[f64]) -> f64 {
    (std::f64::consts::PI * (x[0] - 10.314)).sin()
        + (std::f64::consts::PI * (x[1] - 20.271)).sin()
        + (std::f64::consts::PI * (x[2] - 30.161)).sin()
}

const ORIGIN_OFFSET: [f64; 3] = [10.0, 20.0, 30.0];

fn shifted(p: [f64; 3]) -> [f64; 3] { [p[0] + ORIGIN_OFFSET[0], p[1] + ORIGIN_OFFSET[1], p[2] + ORIGIN_OFFSET[2]] }

/// Standard hex ring: MFEM `CUBE::Vertices` order = the fem-rs conn order.
fn ring() -> [[f64; 3]; 8] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

/// The 12 hex edges in the Gmsh / VTK mid-node order (bottom ring, top ring,
/// vertical) — the extra-node order a 20-node hex row carries.
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

fn mid(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])] }

fn hex_row(elem_type: ElementType, mapped: &dyn Fn([f64; 3]) -> [f64; 3]) -> Mesh<3> {
    let c = ring();
    let mut nodes: Vec<[f64; 3]> = c.iter().map(|&p| mapped(p)).collect();
    match elem_type {
        ElementType::Hex8 => {}
        ElementType::Hex20 => {
            for &(a, b) in &HEX_EDGES {
                nodes.push(mapped(mid(c[a], c[b])));
            }
        }
        ElementType::Hex27 => {
            for &(a, b) in &HEX_EDGES {
                nodes.push(mapped(mid(c[a], c[b])));
            }
            // 6 face centres (MFEM `H1_HexahedronElement(2)` order: the four
            // "diagonal" pairs z-, y-, x+, y+, x-, z+) then the volume centre.
            nodes.push(mapped(mid(c[0], c[2])));
            nodes.push(mapped(mid(c[0], c[5])));
            nodes.push(mapped(mid(c[1], c[6])));
            nodes.push(mapped(mid(c[2], c[7])));
            nodes.push(mapped(mid(c[0], c[7])));
            nodes.push(mapped(mid(c[4], c[6])));
            nodes.push(mapped([0.5, 0.5, 0.5]));
        }
        other => panic!("unsupported fixture label {other:?}"),
    }
    let npe = elem_type.nodes_per_element();
    let mut coords = Vec::with_capacity(nodes.len() * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        coords,
        (0..npe as u32).collect(),
        vec![1],
        elem_type,
        vec![],
        vec![],
        ElementType::Quad4,
    )
}

/// Gmsh v4.1 ASCII, one 20-node hex (type 17): 8 corners in ring order, then
/// the 12 edge mids in Gmsh order.  `Hex20` parses verbatim
/// (`gmsh_node_permutation(Hex20) = None`, no geometry table — the reader's
/// D341 exclusion for the incomplete serendipity families).
fn gmsh_hex20_fixture() -> String {
    let c = ring();
    let mut pts: Vec<[f64; 3]> = c.iter().map(|&p| shifted(p)).collect();
    for &(a, b) in &HEX_EDGES {
        pts.push(shifted(mid(c[a], c[b])));
    }
    let mut s = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Entities\n0 0 0 1\n$EndEntities\n$Nodes\n1 20 1 20\n3 1 0 20\n",
    );
    for k in 1..=20 {
        s.push_str(&format!("{k}\n"));
    }
    for p in &pts {
        s.push_str(&format!("{:.17e} {:.17e} {:.17e}\n", p[0], p[1], p[2]));
    }
    s.push_str("$EndNodes\n$Elements\n1 1 1 1\n3 1 17 1\n");
    s.push('1');
    for k in 1..=20 {
        s.push_str(&format!(" {k}"));
    }
    s.push_str("\n$EndElements\n");
    s
}

/// Attach an order-2 geometry table on the `HexQk(2)` lattice (the D614
/// construction: `HexQk::new(2).dof_coords()` is `[0,1]³`-native since D757).
fn attach_curved_hex27(mesh: &mut Mesh<3>, map: &dyn Fn([f64; 3]) -> [f64; 3]) {
    let family = HexQk::new(2);
    assert_eq!(family.n_dofs(), 27);
    let lattice = family.dof_coords();
    let mut coords = Vec::with_capacity(27 * 3);
    for dc in &lattice {
        coords.extend_from_slice(&map([dc[0], dc[1], dc[2]]));
    }
    mesh.geometry = Some(GeometryData {
        order: 2,
        conn: (0..27u32).collect(),
        nodes_per_elem: 27,
        coords,
        n_nodes: 27,
    });
}

/// A row of **two** adjacent unit cubes sharing the x=1 face (node-identified
/// on the half-integer lattice), so the nodal recovery genuinely averages
/// across elements and `zz_estimator_aniso` has a nonzero energy.
fn hex_row2(elem_type: ElementType) -> Mesh<3> {
    let c = ring();
    let mut local: Vec<[f64; 3]> = c.to_vec();
    if !matches!(elem_type, ElementType::Hex8) {
        for &(a, b) in &HEX_EDGES {
            local.push(mid(c[a], c[b]));
        }
    }
    if matches!(elem_type, ElementType::Hex27) {
        local.push(mid(c[0], c[2]));
        local.push(mid(c[0], c[5]));
        local.push(mid(c[1], c[6]));
        local.push(mid(c[2], c[7]));
        local.push(mid(c[0], c[7]));
        local.push(mid(c[4], c[6]));
        local.push([0.5, 0.5, 0.5]);
    }
    let mut map: std::collections::HashMap<(i64, i64, i64), u32> = Default::default();
    let mut coords: Vec<f64> = Vec::new();
    let mut conn: Vec<u32> = Vec::new();
    for e in 0..2u32 {
        for p in &local {
            let q = [p[0] + e as f64 + ORIGIN_OFFSET[0], p[1] + ORIGIN_OFFSET[1], p[2] + ORIGIN_OFFSET[2]];
            let key = (
                (2.0 * q[0]).round() as i64,
                (2.0 * q[1]).round() as i64,
                (2.0 * q[2]).round() as i64,
            );
            let id = match map.get(&key) {
                Some(&id) => id,
                None => {
                    let id = (coords.len() / 3) as u32;
                    coords.extend_from_slice(&q);
                    map.insert(key, id);
                    id
                }
            };
            conn.push(id);
        }
    }
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        elem_type,
        vec![],
        vec![],
        ElementType::Quad4,
    )
}

fn run_lp(mesh: &Mesh<3>, u: &dyn Fn(&[f64]) -> f64, p: f64) -> f64 {
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&space, space.interpolate(u).as_slice().to_vec());
    lp_error_estimator(&gf, p, u).total_error
}

/// A single translated unit tetrahedron: `Tet4` (4 corners) or `Tet10` with
/// the conn in the fem-rs tet10 order `[v0 v1 v2 v3 | m01 m02 m03 m12 m13 m23]`
/// (`p_refine_tet4_to_tet10`, the D235 layout).  `Tet10` is the D761 arm that
/// was reachable **without** any `geom_rule` change — `geom_rule(Tet10, ·)`
/// already worked, so the all-zero `vertex_shapes` row fed `phys_point`
/// directly.
fn tet_row(elem_type: ElementType) -> Mesh<3> {
    let c = [shifted([0.0, 0.0, 0.0]), shifted([1.0, 0.0, 0.0]), shifted([0.0, 1.0, 0.0]), shifted([0.0, 0.0, 1.0])];
    let mid = |a: [f64; 3], b: [f64; 3]| [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
    let mut nodes: Vec<[f64; 3]> = c.to_vec();
    if elem_type == ElementType::Tet10 {
        for (a, b) in [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            nodes.push(mid(c[a], c[b]));
        }
    }
    let mut coords = Vec::with_capacity(nodes.len() * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    let npe = elem_type.nodes_per_element();
    assert_eq!(npe, nodes.len(), "fixture conn row length");
    Mesh::<3>::uniform(
        coords,
        (0..npe as u32).collect(),
        vec![1],
        elem_type,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

#[test]
fn d761_tet10_quadratic_row_is_affine_exact() {
    // The D761 silent-zero arm on the label that was reachable pre-fix:
    // `phys_point` with a 10-entry row used to return all-zero weights, so the
    // exact solution was evaluated at the ORIGIN.  With the affine `u_ex` on a
    // cell translated to (10,20,30) that is a ~16-wide Lp error instead of 0.
    let lp10 = run_lp(&tet_row(ElementType::Tet10), &u_affine, 1.0);
    let lp4 = run_lp(&tet_row(ElementType::Tet4), &u_affine, 1.0);
    eprintln!("D761 Tet10/Lp(p=1) affine = {lp10:.6e}, Tet4 = {lp4:.6e}");
    assert!(lp10 <= 1e-12, "Tet10 affine exactness: Lp = {lp10:.6e}");
    assert!(lp4 <= 1e-12, "Tet4 affine exactness: Lp = {lp4:.6e}");

    // Same field, non-representable: the quadratic row must agree with the
    // linear twin (same 4 corner dofs, same geometry, same tet rule).
    let q10 = run_lp(&tet_row(ElementType::Tet10), &u_quad, 1.0);
    let q4 = run_lp(&tet_row(ElementType::Tet4), &u_quad, 1.0);
    eprintln!("D761 u_quad Lp(p=1): Tet10 {q10:.17e}  Tet4 {q4:.17e}");
    assert!((q10 - q4).abs() <= 1e-12 * q4.abs(), "Tet10 must match the Tet4 value");
}

#[test]
fn d761_gmsh_hex20_reaches_the_estimator_and_is_affine_exact() {
    let msh = read_msh(gmsh_hex20_fixture().as_bytes()).expect("parse Hex20 fixture");
    let mesh = msh.mesh3d.expect("3D mesh");
    assert_eq!(mesh.elem_type, ElementType::Hex20, "Gmsh type 17 → Hex20 label");
    assert_eq!(mesh.n_elems(), 1);
    assert_eq!(mesh.element_nodes(0).len(), 20, "20-node connectivity row");
    assert_eq!(mesh.geom_order(), 1, "no geometry table for the serendipity label");

    // The element sits at (10,20,30)…(11,21,31): the origin is NOT in it, so a
    // zero-weight `phys_point` (D761) cannot hide behind a small |∇u·x|.
    let lp = run_lp(&mesh, &u_affine, 1.0);
    eprintln!("D761 Gmsh Hex20: Lp(p=1) affine = {lp:.6e} (must be ~0)");
    assert!(lp <= 1e-12, "Hex20 affine exactness: Lp = {lp:.6e}");

    // The estimator paths that share the rule/shape helpers run too.
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&space, space.interpolate(&u_quad).as_slice().to_vec());
    let aniso = zz_estimator_aniso(&gf);
    eprintln!(
        "D761 Gmsh Hex20: zz_estimator_aniso eta[0] = {:.6e} (finite, > 0)",
        aniso.eta[0]
    );
    assert!(aniso.eta[0].is_finite() && aniso.eta[0] > 0.0);
}

#[test]
fn d761_hex2027_affine_exact_on_the_translated_cell() {
    for et in [ElementType::Hex20, ElementType::Hex27] {
        let mesh = hex_row(et, &|p| shifted(p));
        let lp = run_lp(&mesh, &u_affine, 1.0);
        eprintln!("D761 straight {et:?}: Lp(p=1) affine = {lp:.6e}");
        assert!(lp <= 1e-12, "{et:?} affine exactness: Lp = {lp:.6e}");
    }
}

#[test]
fn d761_hex_labels_agree_with_the_linear_family() {
    // Same physical cube, same P1 space (the first 8 conn entries are the
    // corners in every row), same Gaussian rule and geometry map ⇒ the same
    // Lp value for a non-representable u_ex.
    let v8 = run_lp(&hex_row(ElementType::Hex8, &|p| shifted(p)), &u_quad, 1.0);
    let v20 = run_lp(&hex_row(ElementType::Hex20, &|p| shifted(p)), &u_quad, 1.0);
    let v27 = run_lp(&hex_row(ElementType::Hex27, &|p| shifted(p)), &u_quad, 1.0);
    eprintln!("D761 u=x²+2y²+3z² Lp(p=1): Hex8 {v8:.17e}  Hex20 {v20:.17e}  Hex27 {v27:.17e}");
    assert!((v8 - v20).abs() <= 1e-12 * v8.abs(), "Hex20 must match the Hex8 value");
    assert!((v8 - v27).abs() <= 1e-12 * v8.abs(), "Hex27 must match the Hex8 value");
    assert!(v8 > 1e-3, "the fixture field must be non-representable (v8 = {v8:.3e})");
}

#[test]
fn d761_curved_hex27_geometry_table_affine_exact() {
    // An affine geometry map written as an order-2 table (all 27 nodes on it):
    // the geometry is exactly the trilinear/affine map the linear vertex model
    // reconstructs, so affine exactness must hold through the curved path too.
    let map = |p: [f64; 3]| {
        [
            10.0 + 2.0 * p[0] + 0.3 * p[1],
            20.0 + 1.5 * p[1] + 0.2 * p[2],
            30.0 + 3.0 * p[2],
        ]
    };
    let mut mesh = hex_row(ElementType::Hex27, &|p| p);
    attach_curved_hex27(&mut mesh, &map);
    assert_eq!(mesh.geom_order(), 2);
    let lp = run_lp(&mesh, &u_affine, 1.0);
    eprintln!("D761 affine order-2 Hex27 table: Lp(p=1) affine = {lp:.6e}");
    assert!(lp <= 1e-12, "affine-mapped Hex27 must stay exactly representable: {lp:.6e}");

    // The warped table (the D614 `G_hex`) — no MFEM value exists for the Lp
    // estimator on it; require a finite, positive value of the same order as
    // the straight cell.
    let warp = |p: [f64; 3]| {
        [
            p[0] + 0.08 * p[1] * (1.0 - p[1]),
            p[1] + 0.06 * p[2] * (1.0 - p[2]),
            p[2] + 0.05 * p[0] * (1.0 - p[0]),
        ]
    };
    let straight = run_lp(&hex_row(ElementType::Hex27, &|p| shifted(p)), &u_quad, 1.0);
    let mut warped = hex_row(ElementType::Hex27, &|p| p);
    attach_curved_hex27(&mut warped, &warp);
    let curved = run_lp(&warped, &u_quad, 1.0);
    eprintln!("D761 warped Hex27: Lp(p=1) = {curved:.17e} vs straight {straight:.17e}");
    assert!(curved.is_finite() && curved > 0.0);
    assert!(
        curved > 0.25 * straight && curved < 4.0 * straight,
        "warped Hex27 Lp {curved:.6e} outside the straight-cell ballpark {straight:.6e}"
    );
}

#[test]
fn d761_hex27_aniso_estimator_runs_on_the_quadratic_row() {
    // `zz_estimator_aniso` samples `ref_vertex_coords` per vertex and
    // interpolates the recovery with `vertex_shapes` on the conn row: for a
    // Hex27 row both must use the 8 corners (and skip the 19 extra nodes).
    // Two cells (shared face) put the recovery off the per-element flux, and
    // the Hex8 twin of the same physical mesh must give the same estimator —
    // the same `hex_rule` points, the same 8 corner dofs, the same geometry.
    let mesh27 = hex_row2(ElementType::Hex27);
    let mesh8 = hex_row2(ElementType::Hex8);
    let s27 = H1Space::new(mesh27.clone(), 1);
    let s8 = H1Space::new(mesh8.clone(), 1);
    let g27 = GridFunction::new(&s27, s27.interpolate(&u_wave).as_slice().to_vec());
    let g8 = GridFunction::new(&s8, s8.interpolate(&u_wave).as_slice().to_vec());
    let i27 = zz_estimator_aniso(&g27);
    let i8 = zz_estimator_aniso(&g8);
    eprintln!(
        "D761 2-cell zz_estimator_aniso eta: Hex27 {:?} (flags {:?})  vs Hex8 {:?} (flags {:?})",
        i27.eta,
        i27.get_anisotropic_flags(),
        i8.eta,
        i8.get_anisotropic_flags()
    );
    for (a, b) in i27.eta.iter().zip(&i8.eta) {
        assert!(a.is_finite() && *a > 1e-3, "Hex27 eta = {a:.3e}");
        assert!(
            (a - b).abs() <= 1e-12 * a.abs(),
            "Hex27 and Hex8 must agree on the same geometry: {a:.17e} vs {b:.17e}"
        );
    }
    assert_eq!(
        i27.get_anisotropic_flags(),
        i8.get_anisotropic_flags(),
        "aniso flags must not depend on the conn row length"
    );
}
