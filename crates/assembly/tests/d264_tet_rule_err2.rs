//! D264: with the MFEM 4.10 Witherden-Vincent tables active for
//! `tet_rule(8/9/10)` (previously the Grundmann-Moller fallback with weights
//! down to -1.17e-1), the error integrals of an exactly-interpolated field on
//! a tet mesh are sums of non-negative terms — no more tiny-negative err²
//! turning `sqrt` into NaN (the pre-fix symptom was NaN clamped to 0 in
//! `compute_l2_error`).
//!
//! A NaN would surface as exactly `0.0` after the defensive `max(0.0)`
//! clamp, so the assertions below require a **strictly positive** error below
//! 1e-9: `0 < err < 1e-9` fails both for NaN→0 and for genuine garbage.

use fem_assembly::GridFunction;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Quadratic with an exact P2 interpolant: u = 1 + 2x² − 3y·z + x.
fn u(x: &[f64]) -> f64 {
    1.0 + 2.0 * x[0] * x[0] - 3.0 * x[1] * x[2] + x[0]
}

fn l2_error(q: u8) -> f64 {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = H1Space::new(mesh.clone(), 2);
    let gf = GridFunction::new(&space, space.interpolate(&u).as_slice().to_vec());
    gf.compute_l2_error(&u, q)
}

#[test]
fn d264_tet_exact_field_err_nonnegative_and_tiny_q8_to_q10() {
    for q in [8u8, 9, 10] {
        let err = l2_error(q);
        assert!(
            err.is_finite() && err > 0.0 && err < 1e-9,
            "q={q}: exact-field L2 error must be 0 < err < 1e-9 (NaN→0 after the \
             defensive clamp would show up as exactly 0), got {err:e}"
        );
    }
}

#[test]
fn d264_tet_rule_8_exactness_beats_p3_field() {
    // A cubic field on a P3 space: the WV order-8 rule integrates the err²
    // (degree ≤ 6 polynomial) exactly, so the error equals the C++-style
    // analytic value 0 up to roundoff — the pre-fix GM rule (negative
    // weights) produced NaN-clamped zeros for the same input.
    let mesh = Mesh::<3>::unit_cube_tet(1);
    let space = H1Space::new(mesh.clone(), 3);
    let cub = |x: &[f64]| x[0] * x[0] * x[0] + 2.0 * x[1] * x[2];
    let gf = GridFunction::new(&space, space.interpolate(&cub).as_slice().to_vec());
    let err = gf.compute_l2_error(&cub, 8);
    assert!(
        err.is_finite() && err >= 0.0 && err < 1e-9,
        "cubic P3 field, q=8: got {err:e}"
    );
}
