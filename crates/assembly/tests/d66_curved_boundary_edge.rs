//! D66 regression: on **2-D** meshes with curved geometry the boundary
//! integrals must use the boundary element's own **curved** mapping, not the
//! affine chord of the edge's two vertices (the 2-D counterpart of the D59
//! 3-D face fix).
//!
//! Fixture: a quad mesh of the unit square whose bottom edge (tag 1) is the
//! graph `y = 4h·x(1−x)`.  The parabola is quadratic, so the order-2
//! isoparametric geometry reproduces it *exactly* and the arc length is
//! analytic:
//!
//! ```text
//! L = ∫₀¹ √(1 + (4h − 8hx)²) dx = √(1+16h²)/2 + asinh(4h)/(8h)
//! ```
//!
//! `y` vanishes at both corners, so the corner-only chord measures the straight
//! unit segment (length 1) and misses the curved length by ≈ 8h²/3.

use fem_assembly::standard::NeumannIntegrator;
use fem_assembly::Assembler;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::{FESpace, H1Space};

/// 2×2 quad mesh of the unit square with the bottom edge curved by `h`
/// (`y = 4h·x(1−x)` on the bottom row of geometry nodes).
fn build_curved_square(h: f64) -> Mesh<2> {
    let mut mesh = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
    mesh.set_curvature(2);
    // Displace the bottom-row (y = 0) geometry nodes onto the graph of the
    // parabola; every other row is left untouched.
    let geo = mesh.geometry.as_mut().expect("set_curvature built geometry");
    let d = 2usize;
    for k in 0..geo.n_nodes {
        if geo.coords[k * d + 1] == 0.0 {
            let x = geo.coords[k * d];
            geo.coords[k * d + 1] = 4.0 * h * x * (1.0 - x);
        }
    }
    mesh
}

/// Arc length of `y = 4h·x(1−x)` on `[0, 1]`, closed form.
fn analytic_arc_length(h: f64) -> f64 {
    (1.0 + 16.0 * h * h).sqrt() / 2.0 + (4.0 * h).asinh() / (8.0 * h)
}

/// `∫_Γ g ds` over boundary tag 1 (the bottom edge).
fn boundary_integral<F>(mesh: &Mesh<2>, g: F) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    let space = H1Space::new(mesh.clone(), 1);
    // P1 space: the face dofs are the edge's two vertex nodes.
    let face_dofs = |f: u32| mesh.face_nodes(f).to_vec();
    let load = Assembler::assemble_boundary_linear(
        space.n_dofs(),
        mesh,
        &face_dofs,
        1,
        &[&NeumannIntegrator::new(g)],
        &[1],
        16,
    );
    load.as_slice().iter().sum()
}

/// Strong-curvature fixture: the chord is visibly wrong (5.7·10⁻² short).
///
/// D74: the tolerance is 10⁻¹² (machine-precision level).  It was 10⁻⁷ only
/// because the 1-D face quadrature used to be silently capped at 4 Gauss
/// points (degree 7) for any requested order, and the √-integrand of an arc
/// length is not polynomial, so its quadrature error bottomed out at
/// ≈ 4.9·10⁻⁹ here (measured 1.0571159335080944 vs analytic 1.0571159384280695).
/// With the uncapped rule the measured arc length is 1.0571159384280713,
/// i.e. 1.8·10⁻¹⁵ from the analytic value.
#[test]
fn d66_quad_curved_bottom_edge_arc_length_matches_analytic() {
    let h = 0.15;
    let mesh = build_curved_square(h);
    assert_eq!(mesh.geom_order(), 2, "fixture must have curved geometry");
    let expected = analytic_arc_length(h);
    let measured = boundary_integral(&mesh, |_x, _n| 1.0);
    // The corner-only chord would measure the straight unit segment.
    assert!(
        (measured - 1.0).abs() > 1.0e-2,
        "probe: curved arc length must visibly differ from the chord value 1.0 (got {measured})"
    );
    assert!(
        (measured - expected).abs() < 1.0e-12,
        "curved boundary arc length {measured:.16} vs analytic {expected:.16}"
    );
}

/// Mild-curvature fixture: the analytic arc length is reproduced to machine
/// precision (the chord is still off by 3.3·10⁻³, so the test discriminates).
/// The capped rule's error scales like `h⁸` (≈ 1.1·10⁻¹³ here vs 4.9·10⁻⁹ on
/// the strong-curvature fixture), so the 10⁻¹² bar is met with margin.
#[test]
fn d66_quad_small_curvature_arc_length_is_machine_precise() {
    let h = 0.035;
    let mesh = build_curved_square(h);
    let expected = analytic_arc_length(h);
    let measured = boundary_integral(&mesh, |_x, _n| 1.0);
    assert!(
        (measured - 1.0).abs() > 1.0e-3,
        "probe: curved arc length must differ from the chord value 1.0 (got {measured})"
    );
    assert!(
        (measured - expected).abs() < 1.0e-12,
        "curved boundary arc length {measured:.16} vs analytic {expected:.16}"
    );
}

/// With `u = (0, 1)` the constant-field flux is `∫_Γ u·n ds = ∫ (n·u) ds`.
/// Since `n ds = (t0_y, −t0_x) dξ`, the `y` component integrates to
/// `−∫ t0_x dξ = −(x(1) − x(0)) = −1` for the unit-x bottom edge — exactly,
/// and independently of the curvature.  It pins the *normalisation* and the
/// *outward orientation* of the curved edge's normal (and that the curved
/// geometry keeps the endpoints at the corners).
#[test]
fn d66_quad_curved_bottom_edge_constant_flux_matches_divergence_theorem() {
    let mesh = build_curved_square(0.15);
    let measured = boundary_integral(&mesh, |_x, n| n[1]);
    assert!(
        (measured + 1.0).abs() < 1.0e-12,
        "∫ u·n ds on the curved bottom edge = {measured:.16}, expected −1"
    );
}

/// A straight (`geom_order == 1`) quad mesh must keep the historical
/// chord/order-1 path: the arc length is 1 and the constant flux is −1.
///
/// D74: the bar moved from 1e-15 to 5e-15.  The boundary quadrature request
/// here is order 16, which used to be silently capped at the 4-point (degree-7)
/// rule whose weights sum to 1 bit-exactly; it is now the correct 9-point
/// Gauss rule whose Newton-iterated weights carry O(10⁻¹⁵) round-off in the
/// sum (measured: length = 1.0000000000000016).  MFEM's double-precision
/// `QuadratureFunctions1D::GaussLegendre` has the same property (its MPFR path
/// exists precisely to go beyond it), so the geometric pin is kept at a
/// round-off level, not a bit-exact one.
#[test]
fn d66_straight_quad_bottom_edge_is_unchanged() {
    let mesh = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
    assert_eq!(mesh.geom_order(), 1);
    let length = boundary_integral(&mesh, |_x, _n| 1.0);
    assert!(
        (length - 1.0).abs() < 5.0e-15,
        "straight bottom edge length {length:.16}, expected 1"
    );
    let flux = boundary_integral(&mesh, |_x, n| n[1]);
    assert!(
        (flux + 1.0).abs() < 5.0e-15,
        "straight bottom edge ∫ u·n ds {flux:.16}, expected −1"
    );
}
