//! D250: the `GridFunction` error arms (`compute_h1_error`, `compute_w1_error`,
//! `compute_l1_error`) and `postproc/error_estimate.rs::geom_jacobian` must use
//! the **isoparametric (element-wise) geometric Jacobian** — the D242 fix
//! (`geo_ref_elem_from_mesh` + `isoparametric_jacobian`) applied to the scalar
//! error norms and the AMR estimator geometry.
//!
//! Pre-fix the H1/W1/L1 arms used the corner-difference `simplex_jacobian` for
//! every element: on hexes that matrix is the **[0,1]³→physical** map (full
//! edge length per axis) while the hex solution bases (`HexQk`) and their
//! quadrature (`hex_rule`) live on **[-1,1]³** — every hex gradient came out at
//! half strength and the sampled physical points `x0 + J·ξ` fell outside the
//! element (an exactly-interpolated affine field scored h1 ≈ 8.7e0 instead of
//! 0).  Warped quads/hexes additionally have a non-constant Jacobian that a
//! single corner difference cannot represent.
//!
//! # Acceptance (MFEM 4.10 parity, probe `tmp/d264/d250_probe.cpp`)
//!
//! Fields with exact FE interpolants must give h1/w1/l1 ≤ 1e-14 on every
//! family (C++ prints 1e-15..1e-16); the non-representable `x²` fields pin the
//! absolute values against the C++ order-8 numbers to 1e-12.

use fem_assembly::GridFunction;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

const TOL_ZERO: f64 = 1e-14;
const TOL_CPP: f64 = 1e-12;
const Q: u8 = 8; // matches the C++ probe's explicit order-8 rules

fn u_lin3(x: &[f64]) -> f64 { 1.0 + 2.0 * x[0] - 3.0 * x[1] + 5.0 * x[2] }
fn grad_lin3(x: &[f64]) -> Vec<f64> { let _ = x; vec![2.0, -3.0, 5.0] }
fn u_lin2(x: &[f64]) -> f64 { 1.0 + 2.0 * x[0] - 3.0 * x[1] }
fn grad_lin2(x: &[f64]) -> Vec<f64> { let _ = x; vec![2.0, -3.0] }
fn u_hexq2(x: &[f64]) -> f64 { 1.0 + x[0] * x[0] - 3.0 * x[1] * x[2] }
fn grad_hexq2(x: &[f64]) -> Vec<f64> { vec![2.0 * x[0], -3.0 * x[2], -3.0 * x[1]] }
fn u_quadq2(x: &[f64]) -> f64 { 1.0 + x[0] * x[0] - 3.0 * x[0] * x[1] }
fn grad_quadq2(x: &[f64]) -> Vec<f64> { vec![2.0 * x[0] - 3.0 * x[1], -3.0 * x[0]] }
fn u_x2z(x: &[f64]) -> f64 { x[0] * x[0] + x[2] }
fn grad_x2z(x: &[f64]) -> Vec<f64> { vec![2.0 * x[0], 0.0, 1.0] }
fn u_x2(x: &[f64]) -> f64 { x[0] * x[0] }
fn grad_x2(x: &[f64]) -> Vec<f64> { vec![2.0 * x[0], 0.0] }
fn u_tri2(x: &[f64]) -> f64 { 0.5 + 2.0 * x[0] - 3.0 * x[1] + x[0] * x[1] }
fn grad_tri2(x: &[f64]) -> Vec<f64> { vec![2.0 + x[1], -3.0 + x[0]] }

/// The four error norms of one mesh/field pair (h1 semi, w1, l1, l2).
fn norms<S: FESpace>(
    gf: &GridFunction<'_, S>,
    vg: &dyn Fn(&[f64]) -> Vec<f64>,
    uf: &dyn Fn(&[f64]) -> f64,
) -> [f64; 4] {
    [
        gf.compute_h1_error(vg, Q),
        gf.compute_w1_error(vg, Q),
        gf.compute_l1_error(uf, Q),
        gf.compute_l2_error(uf, Q),
    ]
}

fn report(tag: &str, n: [f64; 4]) {
    eprintln!("{tag:<10} h1 {:+.6e}  w1 {:+.6e}  l1 {:+.6e}  l2 {:+.6e}", n[0], n[1], n[2], n[3]);
}

/// MFEM HEXBOX/HEXQ2/HEXX2P1 mesh: 2×1×1 hex box (unit cube).
fn hex_box() -> Mesh<3> {
    Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false)
}

#[test]
fn d250_hex_p1_affine_norms_zero() {
    let mesh = hex_box();
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin3).as_slice().to_vec());
    let n = norms(&gf, &grad_lin3, &u_lin3);
    report("HEXBOX", n);
    for v in n { assert!(v <= TOL_ZERO, "HEXBOX h1/w1/l1/l2 must be ≤{TOL_ZERO:.0e}: {n:?}"); }
}

#[test]
fn d250_hex_p2_quadratic_norms_zero() {
    let mesh = hex_box();
    let h1 = H1Space::new(mesh.clone(), 2);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_hexq2).as_slice().to_vec());
    let n = norms(&gf, &grad_hexq2, &u_hexq2);
    report("HEXQ2", n);
    for v in n { assert!(v <= TOL_ZERO, "HEXQ2 h1/w1/l1/l2 must be ≤{TOL_ZERO:.0e}: {n:?}"); }
}

/// C++ (order-8 rules): L1 4.1666666666666657e-2, L2 4.5643546458763826e-2,
/// H1semi 2.8867513459481298e-1, W1 2.3621260909976921e-1.
#[test]
fn d250_hex_x2_p1_nonrepresentable_matches_cpp() {
    let mesh = hex_box();
    let h1 = H1Space::new(mesh.clone(), 1);
    // MFEM's ProjectCoefficient on a nodal H1 space is the nodal interpolation
    // at the dof points, so `interpolate` builds the same field the C++ probe
    // projects (verified: all four norms match its row exactly).
    let gf = GridFunction::new(&h1, h1.interpolate(&u_x2z).as_slice().to_vec());
    let n = norms(&gf, &grad_x2z, &u_x2z);
    report("HEXX2P1", n);
    let cpp = [2.8867513459481298e-1, 2.3621260909976921e-1, 4.1666666666666657e-2, 4.5643546458763826e-2];
    for (got, want) in n.iter().zip(cpp) {
        assert!((got - want).abs() <= TOL_CPP, "HEXX2P1 vs C++: got {got:.17e} want {want:.17e}");
    }
}

#[test]
fn d250_hex_warped_affine_norms_zero() {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    for (nv, d) in [(6usize, [0.25, -0.2, 0.3]), (5, [0.1, 0.05, 0.2]), (7, [-0.05, 0.15, -0.1])] {
        for c in 0..3 { mesh.coords[nv * 3 + c] += d[c]; }
    }
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin3).as_slice().to_vec());
    let n = norms(&gf, &grad_lin3, &u_lin3);
    report("HEXWARP", n);
    // C++ 1.7e-15/1.7e-15 — the locate-free norm sums pick up a few ulps of
    // rounding on the warped geometry; the sharp gate stays 1e-14.
    for v in n { assert!(v <= TOL_ZERO, "HEXWARP h1/w1/l1/l2 must be ≤{TOL_ZERO:.0e}: {n:?}"); }
}

#[test]
fn d250_quad_warped_affine_norms_zero() {
    let mut mesh = Mesh::<2>::unit_square_quad(2);
    mesh.coords[4 * 2] += 0.12;
    mesh.coords[4 * 2 + 1] += 0.15;
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin2).as_slice().to_vec());
    let n = norms(&gf, &grad_lin2, &u_lin2);
    report("QUADWARP", n);
    for v in n { assert!(v <= TOL_ZERO, "QUADWARP h1/w1/l1/l2 must be ≤{TOL_ZERO:.0e}: {n:?}"); }
}

/// C++ (order-8 rules): L1 4.5266666666666677e-2, L2 5.0967963794263299e-2,
/// H1semi 3.0575697256918866e-1, W1 3.0338807991822492e-1.
#[test]
fn d250_quad_x2_p1_nonrepresentable_matches_cpp() {
    let mut mesh = Mesh::<2>::unit_square_quad(2);
    mesh.coords[4 * 2] += 0.12;
    mesh.coords[4 * 2 + 1] += 0.15;
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_x2).as_slice().to_vec());
    let n = norms(&gf, &grad_x2, &u_x2);
    report("QUADX2P1", n);
    let cpp = [3.0575697256918866e-1, 3.0338807991822492e-1, 4.5266666666666677e-2, 5.0967963794263299e-2];
    for (got, want) in n.iter().zip(cpp) {
        assert!((got - want).abs() <= TOL_CPP, "QUADX2P1 vs C++: got {got:.17e} want {want:.17e}");
    }
}

/// Simplex controls: the corner-difference fallback is exact there and must
/// stay untouched (C++ TETBOX/TRI2D rows print 0..1.6e-15).
#[test]
fn d250_simplex_controls_unchanged() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin3).as_slice().to_vec());
    let n = norms(&gf, &grad_lin3, &u_lin3);
    report("TETBOX", n);
    for v in n { assert!(v <= TOL_ZERO, "TET control must be ≤{TOL_ZERO:.0e}: {n:?}"); }

    let mesh = Mesh::<2>::unit_square_tri(3);
    let h1 = H1Space::new(mesh.clone(), 2);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_tri2).as_slice().to_vec());
    let n = norms(&gf, &grad_tri2, &u_tri2);
    report("TRI2D", n);
    for v in n { assert!(v <= TOL_ZERO, "TRI control must be ≤{TOL_ZERO:.0e}: {n:?}"); }
}

// ── error_estimate.rs (D250: geom_jacobian isoparametric arm) ────────────────

/// `lp_error_estimator` must agree with `compute_l1_error` on the *same*
/// quadrature rule (both hex_rule(5) here): after the D250 fix both walk the
/// isoparametric geometry, so their totals coincide to rounding.  Pre-fix the
/// estimator's hex det was the [0,1]³ corner difference — exactly 8× the true
/// |det J| — so the two totals differed by that factor.  (The |u_h − u|
/// integrand of the x² field has interior sign changes, so absolute values are
/// rule-dependent; comparing two integrators on the *same* rule is the sharp
/// geometry pin.)
#[test]
fn d250_lp_estimator_matches_l1_error_same_rule() {
    use fem_assembly::postproc::error_estimate::lp_error_estimator;

    let mesh = hex_box();
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_x2z).as_slice().to_vec());
    let ind = lp_error_estimator(&gf, 1.0, &u_x2z);
    let l1 = gf.compute_l1_error(&u_x2z, 5);
    eprintln!("HEXX2P1 Lp(p=1) total = {:.17e}  vs compute_l1_error = {l1:.17e}", ind.total_error);
    assert!(
        (ind.total_error - l1).abs() <= 1e-12,
        "Lp p=1 total {} vs compute_l1_error {}",
        ind.total_error,
        l1
    );
}

/// `zz_estimator` on the hex box with an exactly-representable affine field:
/// the recovery is exact, so every indicator must vanish to rounding
/// (exercises `evaluate_gradient_at_element` end-to-end from the estimator).
#[test]
fn d250_zz_estimator_hex_affine_is_exact() {
    use fem_assembly::postproc::error_estimate::zz_estimator;

    let mesh = hex_box();
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin3).as_slice().to_vec());
    let ind = zz_estimator(&gf);
    eprintln!("HEXBOX ZZ total = {:.3e}", ind.total_error);
    assert!(ind.total_error <= 1e-13, "ZZ total {} must be ~0", ind.total_error);
}
