//! D793-3 — `postproc::grid_function::element_jacobian`'s `needs_iso` gate
//! listed only the families `element_jacobian_at` used to special-case
//! (`Quad4/Quad8/Quad9`, `Hex8/Hex20`, `Prism6/Prism15`, `Pyramid5`), so
//! `compute_coeff_l2_norm` / `compute_coeff_l2_norm_first_n` /
//! `compute_element_l2_errors` fell into the affine corner-difference
//! `simplex_jacobian` for `Tri3/Tri6/Tet4/Tet10` (silently straight geometry on
//! a curved simplex — inconsistent with the D787-fixed `compute_l2_error_l2`
//! right next door) and for `Prism18/Hex27/Pyramid13`, where that map is not
//! merely inaccurate but **singular** (`det J ≡ 0`: the vertex tables list the
//! base diagonal in slots 1..3 — the D353 defect class), so the norm came back
//! as `0.0` on a perfectly valid straight mesh.
//!
//! The gate is now family-complete (D787 made `element_jacobian_at` family
//! aware; D793-3 routes every non-surface cell family through it).
//!
//! # Truth
//!
//! * Curved simplex: MFEM 4.10 `ComputeLpNorm(2.0, u_cf, mesh, irs)` with
//!   `irs[geom] = IntRules.Get(geom, 12)` and `u = 1 + x + 2y (+ 3z)` from
//!   `tmp/d793/d793_probe.cpp` (`tmp/d793/mfem_d793_{tet,tri}_p2.txt`) — the
//!   same fixture (unit cell + `set_curvature(2)` + D319 warp) and the same
//!   rule order as the D787 consumer test.  The norm of the *interpolant* of an
//!   affine field is a genuine geometry probe: on a curved cell `u_h(x(ξ))` is
//!   not `u(x)` inside the element, so `∫u_h²` has full sensitivity to the map.
//! * Element error: `compute_element_l2_errors` of `u_h ≡ 1` against the exact
//!   `0` is `√(∫ dΩ)` = `√(element volume)`, and the curved tet's MFEM volume
//!   (`∫|det J|`, 8- and 3-point rules agreeing) is the D787 dump
//!   `1.8298472222222222e-1`.
//! * Straight high-order cells: the unit cube `Hex27` (volume 1) and the unit
//!   `Pyramid13` (volume 1/3, MFEM's `GetElementVolume` truth of the D581/D614
//!   round) — the `det J ≡ 0` class.

use fem_assembly::postproc::grid_function::{
    compute_coeff_l2_norm, compute_coeff_l2_norm_first_n, GridFunction,
};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// MFEM 4.10 `mfem_d793_tet_p2.txt`: `ComputeLpNorm(2,u)`, rule order 12,
/// curved tet P2, `u = 1 + x + 2y + 3z`.
const MFEM_TET_P2_LPNORM_U: f64 = 1.1059035638904782;
/// MFEM 4.10 `mfem_d793_tri_p2.txt`: same on the curved tri P2
/// (`u = 1 + x + 2y`).
const MFEM_TRI_P2_LPNORM_U: f64 = 1.6182459639992921;
/// MFEM 4.10 `tmp/d787/mfem_truth_tet.txt`: `∫|det J|` of the curved tet P2
/// (8- and 3-point rules agree — the map is a polynomial).
const MFEM_TET_P2_VOLUME: f64 = 1.8298472222222222e-1;

const TOL: f64 = 1e-13;

fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn g2(x: [f64; 2]) -> [f64; 2] {
    [
        x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1],
        x[1] + 0.05 * x[0] * x[1],
    ]
}

fn warp_mesh<const D: usize>(mesh: &mut Mesh<D>, f: fn([f64; D]) -> [f64; D]) {
    let n_geom = mesh.geometry.as_ref().expect("curved mesh keeps a table").coords.len() / D;
    {
        let geo = mesh.geometry.as_mut().expect("curved table");
        for k in 0..n_geom {
            let mut x = [0.0_f64; D];
            x.copy_from_slice(&geo.coords[k * D..(k + 1) * D]);
            let y = f(x);
            geo.coords[k * D..(k + 1) * D].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; D];
        x.copy_from_slice(&mesh.coords[k * D..(k + 1) * D]);
        let y = f(x);
        mesh.coords[k * D..(k + 1) * D].copy_from_slice(&y);
    }
}

fn curved_unit_tet() -> Mesh<3> {
    let mut m = Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        vec![1; 4],
        ElementType::Tri3,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g3);
    m
}

fn curved_unit_tri() -> Mesh<2> {
    let mut m = Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1; 3],
        ElementType::Line2,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g2);
    m
}

/// Straight unit-cube `Hex27`: 8 corners (MFEM `CUBE::Vertices` ring), 12 edge
/// mids (VTK order), 6 face centres, 1 centre — the D614 fixture.
fn straight_unit_hex27() -> Mesh<3> {
    let c: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (c[a][0] + c[b][0]),
            0.5 * (c[a][1] + c[b][1]),
            0.5 * (c[a][2] + c[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = c.to_vec();
    for (a, b) in [
        (0usize, 1usize),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        nodes.push(mid(a, b));
    }
    for (a, b) in [(0usize, 2usize), (0, 5), (1, 6), (2, 7), (0, 7), (4, 6)] {
        nodes.push(mid(a, b));
    }
    nodes.push([0.5, 0.5, 0.5]);
    let mut coords = Vec::with_capacity(27 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        coords,
        (0..27).collect(),
        vec![1],
        ElementType::Hex27,
        vec![],
        vec![],
        ElementType::Quad4,
    )
}

/// Straight unit `Pyramid13`: 5 vertices + 8 edge mids — the D614 fixture.
fn straight_unit_pyramid13() -> Mesh<3> {
    let v: [[f64; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (v[a][0] + v[b][0]),
            0.5 * (v[a][1] + v[b][1]),
            0.5 * (v[a][2] + v[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = v.to_vec();
    for (a, b) in [
        (0usize, 1usize),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ] {
        nodes.push(mid(a, b));
    }
    let mut coords = Vec::with_capacity(13 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        coords,
        (0..13).collect(),
        vec![1],
        ElementType::Pyramid13,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Straight unit right prism `Prism18` (volume 1/2): the `d353` unit-prism
/// vertices + the 9 edge midpoints + the 3 quad-face centres of the 18-slot
/// row.  Only the 6 vertex slots are read on a straight cell.
fn straight_unit_prism18() -> Mesh<3> {
    let v: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (v[a][0] + v[b][0]),
            0.5 * (v[a][1] + v[b][1]),
            0.5 * (v[a][2] + v[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = v.to_vec();
    for (a, b) in [
        (0usize, 1usize),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
    ] {
        nodes.push(mid(a, b));
    }
    for quad in [[0usize, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]] {
        let mut c = [0.0_f64; 3];
        for &k in quad.iter() {
            for d in 0..3 {
                c[d] += 0.25 * v[k][d];
            }
        }
        nodes.push(c);
    }
    let mut coords = Vec::with_capacity(18 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        coords,
        (0..18).collect(),
        vec![1],
        ElementType::Prism18,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

#[test]
fn d793_coeff_l2_norm_curved_tet_matches_mfem() {
    let mesh = curved_unit_tet();
    assert_eq!(mesh.geom_order(), 2);
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) =
        &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2];
    let norm = compute_coeff_l2_norm(&mesh, exact, 12);
    eprintln!("D793-3 curved tet P2 ComputeLpNorm(2,u): {norm:.17e}");
    let rel = (norm - MFEM_TET_P2_LPNORM_U).abs() / MFEM_TET_P2_LPNORM_U;
    assert!(
        rel <= TOL,
        "curved tet coefficient L2 norm: got {norm:.17e}, MFEM {MFEM_TET_P2_LPNORM_U:.17e} \
         (rel {rel:.3e}) — the measure is not the cell's own geometry map"
    );
    let first_n = compute_coeff_l2_norm_first_n(&mesh, exact, 12, 1);
    assert!(
        (first_n - norm).abs() <= TOL,
        "compute_coeff_l2_norm_first_n(1) = {first_n:.17e} vs {norm:.17e}"
    );
}

#[test]
fn d793_coeff_l2_norm_curved_tri_matches_mfem() {
    let mesh = curved_unit_tri();
    assert_eq!(mesh.geom_order(), 2);
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1];
    let norm = compute_coeff_l2_norm(&mesh, exact, 12);
    eprintln!("D793-3 curved tri P2 ComputeLpNorm(2,u): {norm:.17e}");
    let rel = (norm - MFEM_TRI_P2_LPNORM_U).abs() / MFEM_TRI_P2_LPNORM_U;
    assert!(
        rel <= TOL,
        "curved tri coefficient L2 norm: got {norm:.17e}, MFEM {MFEM_TRI_P2_LPNORM_U:.17e} \
         (rel {rel:.3e})"
    );
}

/// `compute_element_l2_errors` runs through the same `element_jacobian` helper:
/// the error of `u_h ≡ 1` against `0` is `√(∫dΩ)`, i.e. `√(volume)`.
#[test]
fn d793_element_l2_errors_curved_tet_matches_mfem_volume() {
    let mesh = curved_unit_tet();
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&space, vec![1.0_f64; space.n_dofs()]);
    let errors = gf.compute_element_l2_errors(&|_x: &[f64]| 0.0, 12);
    assert_eq!(errors.len(), 1);
    let want = MFEM_TET_P2_VOLUME.sqrt();
    eprintln!("D793-3 curved tet element L2 error of 1 vs 0: {:.17e}", errors[0]);
    let rel = (errors[0] - want).abs() / want;
    assert!(
        rel <= TOL,
        "curved tet element L2 error: got {:.17e}, MFEM √vol = {want:.17e} (rel {rel:.3e})",
        errors[0]
    );
}

/// The families `Prism18/Hex27/Pyramid13` were missing from the gate *and* from
/// the iso list: on a straight unit cube and a straight unit pyramid the affine
/// corner map is singular (`det J ≡ 0`), so the norm used to come back `0.0`
/// instead of `√(volume)`.
#[test]
fn d793_coeff_l2_norm_straight_high_order_cells_not_silently_zero() {
    let one: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = &|_x: &[f64]| 1.0;
    let hex27 = straight_unit_hex27();
    let norm = compute_coeff_l2_norm(&hex27, one, 12);
    eprintln!("D793-3 straight Hex27 ||1||_L2: {norm:.17e} (want 1)");
    assert!(
        (norm - 1.0).abs() < 1e-13,
        "straight unit-cube Hex27 ||1||_L2 = {norm:.17e}, want 1.0 (0.0 = the D353 \
         singular-corner-Jacobian class)"
    );
    let pyr13 = straight_unit_pyramid13();
    let norm = compute_coeff_l2_norm(&pyr13, one, 12);
    let want = (1.0_f64 / 3.0).sqrt();
    eprintln!("D793-3 straight Pyramid13 ||1||_L2: {norm:.17e} (want {want:.17e})");
    assert!(
        (norm - want).abs() < 1e-13,
        "straight unit Pyramid13 ||1||_L2 = {norm:.17e}, want {want:.17e}"
    );
    // `Prism18` took the same affine arm (its vertex row repeats the base
    // diagonal in slots 1..3), which for this cell yields the *cube's* unit
    // determinant instead of the prism's 1/2 — the norm therefore has to be
    // `√(1/2)`, not 1.  (This pin was added after the fix; the pre-fix value
    // `1.0` follows from those three corner columns analytically.)
    let prism18 = straight_unit_prism18();
    let norm = compute_coeff_l2_norm(&prism18, one, 12);
    let want = 0.5_f64.sqrt();
    eprintln!("D793-3 straight Prism18 ||1||_L2: {norm:.17e} (want {want:.17e})");
    assert!(
        (norm - want).abs() < 1e-13,
        "straight unit Prism18 ||1||_L2 = {norm:.17e}, want {want:.17e} (1.0 = the \
         D353 singular/affine-corner class: the vertex row's slots 1..3 are the \
         base diagonal + the extrusion axis, which the old arm read as a unit cube)"
    );
}
