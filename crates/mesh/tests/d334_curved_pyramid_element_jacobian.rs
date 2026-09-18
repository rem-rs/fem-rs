//! D334 — curved-pyramid geometry in `element_jacobian_at` / `geometry_jacobian`.
//!
//! Both helpers are "linear (P1) reference element + the element's geometry
//! coordinates" utilities.  For a **curved** pyramid (`geom_order >= 2`) that
//! pairing is simply wrong: `Mesh::set_curvature_pyramid5` writes the geometry
//! table in the slot order of `h1_pyramid_element(g, Fuentes)` (D347, MFEM's
//! `SetCurvature` default family), so the table has `p(p^2+3)+1` entries
//! (15 at `g = 2`) while the P1 element has 5 — and `raw_geometry_nodes` then
//! fell back to the plain corner vertices, so the curvature was invisible *and*
//! the base corners were twisted (round 45 measured `∫|det J| = 0.1738` for the
//! unit pyramid, whose isoparametric value is 1/3).
//!
//! The fix adds the order-`g` arm; these tests pin the integral, the identity map
//! of the unit pyramid, the agreement with `Mesh::element_jacobian` (the
//! isoparametric path the assembler uses — the two must not drift) and the fact
//! that a displaced *quadratic* node is now visible.
//!
//! Evidence: `tmp/d343/EVIDENCE.md` §4; the Fuentes node table and the
//! `SetCurvature` oracle are in `tmp/d347/EVIDENCE.md`.

use fem_mesh::element_type::ElementType;
use fem_mesh::transformation::{element_jacobian_at, geometry_jacobian};
use fem_mesh::{Mesh, MeshTopology};

const UNIT_VERTICES_FLAT: [f64; 15] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
];

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        UNIT_VERTICES_FLAT.to_vec(),
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `∫|det J|` over the unit pyramid, from `element_jacobian_at`.
fn volume_element_jacobian(mesh: &Mesh<3>, e: u32, order: u8) -> f64 {
    let rule = fem_element::quadrature::pyramid_rule(order);
    rule.points
        .iter()
        .zip(rule.weights.iter())
        .map(|(xi, w)| w * element_jacobian_at(mesh, e, xi, 3).0.determinant().abs())
        .sum()
}

/// `∫ det J` over the unit pyramid, from `geometry_jacobian`.
fn volume_geometry_jacobian(mesh: &Mesh<3>, e: u32, order: u8) -> f64 {
    let rule = fem_element::quadrature::pyramid_rule(order);
    rule.points
        .iter()
        .zip(rule.weights.iter())
        .map(|(xi, w)| {
            let (det, _inv) = geometry_jacobian(mesh, e, xi, 3);
            w * det
        })
        .sum()
}

const SAMPLES: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.3],
    [0.25, 0.25, 0.5],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [0.5, 0.25, 0.25],
    [0.05, 0.05, 0.9],
];

/// A curved but *geometrically identical* pyramid: `set_curvature(2)` on the
/// unit pyramid places every quadratic node on the trivial map, so the
/// isoparametric geometry is the identity and `det J ≡ 1`.
#[test]
fn d334_curved_unit_pyramid_has_unit_jacobian() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);
    assert_eq!(mesh.geom_order(), 2);
    assert_eq!(mesh.geometry_nodes(0).len(), 15, "Fuentes P2: p(p^2+3)+1");

    for xi in SAMPLES {
        let (jac, x) = element_jacobian_at(&mesh, 0, &xi, 3);
        let det = jac.determinant();
        assert!(
            (det - 1.0).abs() < 1e-12,
            "unit pyramid det J at {xi:?} = {det:.15} (D334: used to be twisted)"
        );
        for i in 0..3 {
            assert!(
                (x[i] - xi[i]).abs() < 1e-12,
                "identity map broken at {xi:?}: x = {x:?}"
            );
        }
    }

    let vol = volume_element_jacobian(&mesh, 0, 8);
    eprintln!("D334 curved unit pyramid: ∫|det J| (element_jacobian_at) = {vol:.15}");
    assert!(
        (vol - 1.0 / 3.0).abs() < 1e-12,
        "∫|det J| = {vol:.15}, expected 1/3 (round 45's pre-fix value was 0.1738)"
    );
    let vol2 = volume_geometry_jacobian(&mesh, 0, 8);
    eprintln!("D334 curved unit pyramid: ∫det J (geometry_jacobian) = {vol2:.15}");
    assert!((vol2 - 1.0 / 3.0).abs() < 1e-12, "geometry_jacobian: {vol2:.15}");
}

/// The new arm must agree with `Mesh::element_jacobian` — the isoparametric
/// path the assembler uses (same Fuentes element, same slot order).  A drift
/// between the two would put the field side and the geometry side on different
/// maps.
#[test]
fn d334_curved_pyramid_agrees_with_element_jacobian() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);

    let (mut wx, mut wd) = (0.0_f64, 0.0_f64);
    for xi in SAMPLES {
        let (j_at, x_at) = element_jacobian_at(&mesh, 0, &xi, 3);
        let (j_ref, det_ref, x_ref) = mesh.element_jacobian(0, &xi);
        for i in 0..3 {
            wx = wx.max((x_at[i] - x_ref[i]).abs());
            for d in 0..3 {
                wx = wx.max((j_at[(i, d)] - j_ref[(i, d)]).abs());
            }
        }
        wd = wd.max((j_at.determinant() - det_ref).abs());
        let (det_g, _inv) = geometry_jacobian(&mesh, 0, &xi, 3);
        wd = wd.max((det_g - det_ref).abs());
    }
    eprintln!("D334 vs Mesh::element_jacobian: max |ΔJ/x| = {wx:.3e}, max |Δdet J| = {wd:.3e}");
    assert!(wx < 1e-13, "element_jacobian_at drifted from element_jacobian: {wx:.3e}");
    assert!(wd < 1e-13, "determinants drifted: {wd:.3e}");
}

/// Displacing the **base-centre** quadratic node (Fuentes slot 13 — a node the
/// P1 vertex table cannot represent) must change the geometry: this is the
/// sensitivity the old fallback silently discarded.
#[test]
fn d334_displaced_quadratic_node_is_visible() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);
    let baseline = volume_element_jacobian(&mesh, 0, 8);

    // `set_curvature_pyramid5` lays the geometry nodes out with the Fuentes
    // element's `dof_coords()`; slot 13 is the base centre (0.5, 0.5, 0).
    let re = fem_element::lagrange::h1_pyramid_element(2, fem_element::lagrange::PyramidBasisType::default());
    let coord = &re.dof_coords()[13];
    assert!(
        (coord[0] - 0.5).abs() < 1e-12 && (coord[1] - 0.5).abs() < 1e-12 && coord[2].abs() < 1e-12,
        "slot 13 must be the base centre, got {coord:?}"
    );

    // Find the geometry node of slot 13 and push it up in z.
    let node = mesh.geometry_nodes(0)[13] as usize;
    {
        let geo = mesh.geometry.as_mut().expect("geometry table");
        geo.coords[node * 3 + 2] += 0.25;
    }
    let moved = volume_element_jacobian(&mesh, 0, 8);
    eprintln!("D334 displaced base-centre node: {baseline:.15} -> {moved:.15}");
    assert!(
        (moved - baseline).abs() > 1e-3,
        "a displaced quadratic node must move the geometry ({baseline:.15} -> {moved:.15})"
    );
    // Both accessors still describe the same map.
    let d = (moved - volume_geometry_jacobian(&mesh, 0, 8)).abs();
    assert!(d < 1e-12, "element_jacobian_at and geometry_jacobian disagree by {d:.3e}");
}

/// Straight pyramids keep the D331 behaviour (layer permutation + P1 basis).
#[test]
fn d334_straight_pyramid_unchanged() {
    let mesh = unit_pyramid();
    assert_eq!(mesh.geom_order(), 1);
    let vol = volume_element_jacobian(&mesh, 0, 8);
    assert!((vol - 1.0 / 3.0).abs() < 1e-13, "straight pyramid volume {vol:.15}");
    let (jac, x) = element_jacobian_at(&mesh, 0, &[0.25, 0.25, 0.5], 3);
    assert!((jac.determinant() - 1.0).abs() < 1e-13);
    assert!((x[0] - 0.25).abs() < 1e-13 && (x[2] - 0.5).abs() < 1e-13);
}

/// Replica of the **pre-D334** fallback, so the "before" number is measured
/// here instead of quoted: the P1 (`PyramidPk(1)`, layer-ordered) element
/// evaluated over the raw **vertex-ordered** corner table (the old
/// `raw_geometry_nodes` → `element_nodes` fallback, with
/// `straight_pyramid_layer_nodes` declining at `geom_order > 1`).
fn pre_d334_volume(mesh: &Mesh<3>, e: u32, order: u8) -> f64 {
    let rule = fem_element::quadrature::pyramid_rule(order);
    rule.points
        .iter()
        .zip(rule.weights.iter())
        .map(|(xi, w)| w * pre_d334_map(mesh, e, xi).0.determinant().abs())
        .sum()
}

/// `(J, x)` of the pre-D334 fallback at `xi`.
fn pre_d334_map(mesh: &Mesh<3>, e: u32, xi: &[f64]) -> (nalgebra::DMatrix<f64>, [f64; 3]) {
    use fem_element::ReferenceElement;
    let re = ElementType::Pyramid5.ref_elem(1);
    let npe = re.n_dofs();
    let nodes = mesh.element_nodes(e).to_vec();
    let mut phi = vec![0.0_f64; npe];
    let mut grad = vec![0.0_f64; npe * 3];
    re.eval_basis(xi, &mut phi);
    re.eval_grad_basis(xi, &mut grad);
    let mut jac = nalgebra::DMatrix::<f64>::zeros(3, 3);
    let mut x = [0.0_f64; 3];
    for k in 0..npe {
        let c = mesh.geom_coords_of(nodes[k]);
        for i in 0..3 {
            x[i] += c[i] * phi[k];
            for j in 0..3 {
                jac[(i, j)] += c[i] * grad[k * 3 + j];
            }
        }
    }
    (jac, x)
}

/// The bug, measured: on the curved unit pyramid the old fallback integrates
/// `0.173755809543588` (round 45's number) — the same twisted-base value D331
/// removed for *straight* pyramids, plus total blindness to the quadratic
/// nodes — while the new arm gives exactly `1/3`.
/// The bug, measured.
///
/// The pre-D334 fallback paired the layer-ordered `PyramidPk(1)` basis with the
/// **vertex-ordered** corner table (the same twist D331 removed for straight
/// pyramids), so the unit pyramid's own reference point `(1,1,0)` maps to a
/// *different* vertex, and `∫|det J|` is not `1/3`.
///
/// Note on the recorded number: D331's `∫|det J| = 0.173755809543588` is
/// **quadrature-order dependent** — the twisted map reverses orientation inside
/// the element, so the `|det J|` integrand changes sign and no absolute-value
/// sum is stable.  Measured here (order: value): `2: 0.192450090`, `4:
/// 0.143443828`, `6: 0.173755810` (the recorded figure), `8: 0.157475073`,
/// `10: 0.169982349`.  This test therefore measures the fallback at several
/// orders and asserts the new arm is `1/3` at all of them, instead of pinning
/// one order's number as if it were a property of the map.
#[test]
fn d334_pre_fix_fallback_is_a_different_map() {
    let mut mesh = unit_pyramid();
    mesh.set_curvature(2);

    // The fallback maps the reference corner (1,1,0) to the other base corner.
    let (_, x) = pre_d334_map(&mesh, 0, &[1.0, 1.0, 0.0]);
    eprintln!("D334 fallback: x(1,1,0) = {x:?} (the map does not reproduce its own corner)");
    assert!(
        (x[0] - 1.0).abs() > 0.5,
        "the pre-fix fallback is supposed to be a different map, got {x:?}"
    );

    for order in [2u8, 4, 6, 8, 10] {
        let before = pre_d334_volume(&mesh, 0, order);
        let after = volume_element_jacobian(&mesh, 0, order);
        eprintln!("D334 order={order}: fallback = {before:.15}, fixed = {after:.15}");
        assert!((after - 1.0 / 3.0).abs() < 1e-12, "order={order}: after = {after:.15}");
    }
}
