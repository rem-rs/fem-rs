//! Integration tests for the fem-element crate.
//!
//! Checks partition-of-unity, gradient consistency, and quadrature weight sums
//! for each reference element type.

use fem_element::{lagrange::{TetP1, TriP1, TriP2, QuadQ1}, ReferenceElement};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Sum of all basis functions evaluated at a quadrature point.
fn sum_basis(re: &dyn ReferenceElement, xi: &[f64]) -> f64 {
    let mut phi = vec![0.0_f64; re.n_dofs()];
    re.eval_basis(xi, &mut phi);
    phi.iter().sum()
}

/// Check partition of unity at all quadrature points.
fn check_partition_of_unity(re: &dyn ReferenceElement, quad_order: u8, tol: f64) {
    let quad = re.quadrature(quad_order);
    for (qi, xi) in quad.points.iter().enumerate() {
        let s = sum_basis(re, xi);
        assert!(
            (s - 1.0).abs() < tol,
            "partition-of-unity violated at qp {qi}: sum = {s:.6}, xi = {xi:?}"
        );
    }
}

// ─── Partition of unity ───────────────────────────────────────────────────────

#[test]
fn tri_p1_partition_of_unity() {
    check_partition_of_unity(&TriP1, 5, 1e-13);
}

#[test]
fn tri_p2_partition_of_unity() {
    check_partition_of_unity(&TriP2, 7, 1e-13);
}

#[test]
fn quad_q1_partition_of_unity() {
    check_partition_of_unity(&QuadQ1, 5, 1e-13);
}

#[test]
fn tet_p1_partition_of_unity() {
    check_partition_of_unity(&TetP1, 5, 1e-13);
}

// ─── Quadrature weight sum ────────────────────────────────────────────────────

/// Weights should sum to the reference element's area/volume.
fn check_weight_sum(re: &dyn ReferenceElement, quad_order: u8, expected_area: f64, tol: f64) {
    let quad = re.quadrature(quad_order);
    let total: f64 = quad.weights.iter().sum();
    assert!(
        (total - expected_area).abs() < tol,
        "quadrature weights sum to {total:.6e}, expected {expected_area:.6e}"
    );
}

#[test]
fn tri_p1_quadrature_weight_sum() {
    // Reference triangle area = 0.5
    check_weight_sum(&TriP1, 5, 0.5, 1e-13);
}

#[test]
fn tri_p2_quadrature_weight_sum() {
    check_weight_sum(&TriP2, 7, 0.5, 1e-13);
}

#[test]
fn quad_q1_quadrature_weight_sum() {
    // Reference square [-1,1]² area = 4
    check_weight_sum(&QuadQ1, 5, 4.0, 1e-13);
}

#[test]
fn tet_p1_quadrature_weight_sum() {
    // Reference tetrahedron volume = 1/6
    check_weight_sum(&TetP1, 5, 1.0 / 6.0, 1e-13);
}

// ─── Gradient consistency ─────────────────────────────────────────────────────

/// For a linear function u = a·x + b, the FE gradient at any point should
/// exactly reproduce a (since P1 interpolates linears exactly).
/// We test this using the reference-space gradient formula:
///   ∇_ref u = Σ_i u_i * ∇φ_i
/// where u_i are the nodal values of u at the DOF locations.
#[test]
fn tri_p1_grad_linear_function() {
    // u(ξ,η) = 2ξ + 3η  → ∂u/∂ξ = 2, ∂u/∂η = 3
    let re = TriP1;
    // P1 DOF locations on reference triangle: (0,0), (1,0), (0,1)
    let dof_pts = [[0.0f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let u_vals: Vec<f64> = dof_pts.iter().map(|p| 2.0 * p[0] + 3.0 * p[1]).collect();

    let quad = re.quadrature(3);
    let n = re.n_dofs();
    let mut grad_ref = vec![0.0f64; n * 2];

    for xi in &quad.points {
        re.eval_grad_basis(xi, &mut grad_ref);

        let mut du_dxi = 0.0;
        let mut du_deta = 0.0;
        for i in 0..n {
            du_dxi += u_vals[i] * grad_ref[i * 2];
            du_deta += u_vals[i] * grad_ref[i * 2 + 1];
        }
        assert!((du_dxi - 2.0).abs() < 1e-12, "∂u/∂ξ = {du_dxi:.6}, expected 2.0");
        assert!((du_deta - 3.0).abs() < 1e-12, "∂u/∂η = {du_deta:.6}, expected 3.0");
    }
}

#[test]
fn quad_q1_grad_bilinear_function() {
    // u(ξ,η) = ξ (constant in η) → ∂u/∂ξ = 1, ∂u/∂η = 0
    // Q1 DOF locations: (-1,-1),(1,-1),(1,1),(-1,1)
    let re = QuadQ1;
    let dof_pts = [[-1.0f64, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let u_vals: Vec<f64> = dof_pts.iter().map(|p| p[0]).collect(); // u = ξ

    let quad = re.quadrature(3);
    let n = re.n_dofs();
    let mut grad_ref = vec![0.0f64; n * 2];

    for xi in &quad.points {
        re.eval_grad_basis(xi, &mut grad_ref);
        let mut du_dxi = 0.0;
        let mut du_deta = 0.0;
        for i in 0..n {
            du_dxi += u_vals[i] * grad_ref[i * 2];
            du_deta += u_vals[i] * grad_ref[i * 2 + 1];
        }
        assert!((du_dxi - 1.0).abs() < 1e-12, "Q1 ∂u/∂ξ = {du_dxi:.6}, expected 1.0 at xi={xi:?}");
        assert!(du_deta.abs() < 1e-12, "Q1 ∂u/∂η = {du_deta:.6}, expected 0.0 at xi={xi:?}");
    }
}

// ─── Interpolation exactness ──────────────────────────────────────────────────

/// P1 integrates constant functions exactly.
#[test]
fn tri_p1_integrates_constant_exactly() {
    let re = TriP1;
    let quad = re.quadrature(1);
    let mut phi = vec![0.0_f64; re.n_dofs()];

    // ∫ 1 dΩ over reference triangle = 0.5
    let area: f64 = quad.points.iter().zip(quad.weights.iter()).map(|(xi, &w)| {
        re.eval_basis(xi, &mut phi);
        w * phi.iter().sum::<f64>()
    }).sum();
    assert!((area - 0.5).abs() < 1e-13, "∫1 dΩ = {area:.6e}, expected 0.5");
}
