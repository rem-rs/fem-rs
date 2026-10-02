//! D998: `DiscreteLinearOperator::gradient` for `H¹(P3) → H(curl) ND3` on
//! 3-D hexahedra — the pex32 `-o 3` arm (MFEM ex32p builds the H1 space at
//! the ND order, so `-o 3` needs the P3→ND3 discrete gradient; before this
//! arm `gradient` returned `UnsupportedH1Order(3)` and pex32 panicked,
//! `rs32_fichera_o3.err` round-106).
//!
//! Same two independent checks as the D110 P2→ND2 entry:
//!
//! 1. **Exactness.** For `p ∈ P3` whose gradient lies in `(P2)³ ⊆ ND3`, the
//!    DOF vector `G · p_h` must equal the Nédélec interpolant of `∇p`,
//!    computed by `HCurlSpace::interpolate_vector` through a completely
//!    separate code path (canonical functionals + `J·t̂` tangents).
//! 2. **Order discipline.** H1 order 4 keeps failing with a *gradient* order
//!    error (no silent fallback), and the (2, ≠2) pair keeps failing with the
//!    order-mismatch error.

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace};

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn matvec(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows];
    a.spmv(x, &mut y);
    y
}

/// `G p_h == Π_ND3(∇p)` for cubic p whose gradients are exactly representable:
/// `p = x₀x₁x₂` (∇p = (x₁x₂, x₀x₂, x₀x₁)) and
/// `p = x₀³ + 2x₁x₂x₀ + x₂` (∇p = (3x₀² + 2x₁x₂, 2x₀x₂, 2x₀x₁ + 1)).
#[test]
fn p3_nd3_gradient_hex3d_matches_nd_interpolation() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 3);
    let nd = HCurlSpace::new(mesh.clone(), 3);

    let g = DiscreteLinearOperator::gradient(&h1, &nd)
        .expect("3-D hex P3→ND3 gradient must assemble (D998)");
    assert_eq!(g.nrows, nd.n_dofs(), "rows = ND3 DOFs");
    assert_eq!(g.ncols, h1.n_dofs(), "columns = P3 DOFs");

    // p = x₀ x₁ x₂  →  ∇p = (x₁ x₂, x₀ x₂, x₀ x₁)
    let p_dofs = h1.interpolate(&|x| x[0] * x[1] * x[2]);
    let g_p = matvec(&g, p_dofs.as_slice());
    let nd_exact = nd.interpolate_vector(&|x| vec![x[1] * x[2], x[0] * x[2], x[0] * x[1]]);
    let dev = max_abs_diff(&g_p, nd_exact.as_slice());
    assert!(
        dev < 1e-12,
        "G·(P3 interpolant of x₀x₁x₂) vs Π_ND3(∇p): max dev {dev:.3e}"
    );

    // p = x₀³ + 2 x₁ x₂ x₀ + x₂
    //   →  ∇p = (3x₀² + 2x₁x₂, 2x₀x₂, 2x₀x₁ + 1)
    let p_dofs = h1.interpolate(&|x| {
        x[0] * x[0] * x[0] + 2.0 * x[1] * x[2] * x[0] + x[2]
    });
    let g_p = matvec(&g, p_dofs.as_slice());
    let nd_exact = nd.interpolate_vector(&|x| {
        vec![
            3.0 * x[0] * x[0] + 2.0 * x[1] * x[2],
            2.0 * x[0] * x[2],
            2.0 * x[0] * x[1] + 1.0,
        ]
    });
    let dev = max_abs_diff(&g_p, nd_exact.as_slice());
    assert!(
        dev < 1e-12,
        "G·(P3 interpolant of x₀³+2x₀x₁x₂+x₂) vs Π_ND3(∇p): max dev {dev:.3e}"
    );
}

/// Gradient of a constant must vanish: `G · 1 = 0` (row sums are the σ_s
/// functionals of the zero field).
#[test]
fn p3_nd3_gradient_kills_constants() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 3);
    let nd = HCurlSpace::new(mesh.clone(), 3);
    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("P3→ND3 gradient");

    let ones = vec![1.0; h1.n_dofs()];
    let g_ones = matvec(&g, &ones);
    let dev = g_ones.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(dev < 1e-12, "G·1 must vanish, max |row sum| = {dev:.3e}");
}

/// Order discipline: order 4 keeps an explicit error, and a (2, ≠2) pair is
/// rejected with the unsupported-H(curl)-order error (the historic variant,
/// pinned by the in-file tests).
#[test]
fn p3_nd3_gradient_rejects_wrong_orders() {
    let mesh = Mesh::<3>::unit_cube_hex(1);

    let h1 = H1Space::new(mesh.clone(), 4);
    let nd = HCurlSpace::new(mesh.clone(), 4);
    let err = DiscreteLinearOperator::gradient(&h1, &nd).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("H1 space must be order"),
        "expected the gradient H1-order error, got: {msg}"
    );

    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh, 1);
    let err = DiscreteLinearOperator::gradient(&h1, &nd).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("H(curl) space must be order"),
        "expected the unsupported-H(curl)-order error, got: {msg}"
    );
}
