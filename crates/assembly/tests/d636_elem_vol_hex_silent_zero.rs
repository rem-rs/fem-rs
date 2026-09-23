//! D636 — `error_estimate::elem_vol` measured hex (and prism/pyramid)
//! volumes with the "first 4 vertices" tetrahedron formula.  On a hex the
//! first four corners are coplanar, so `det ≡ 0` and every 3-D non-simplex
//! cell got volume 0 — the ZZ-family estimators weigh η² by it
//! (`η_K = sqrt(err²·V_K)`, `η_K = sqrt(energy·V_K)`, `η²_K = h²·f²·V_K`),
//! so their indicators were systematically squashed to zero on hex meshes.
//!
//! Fix: the 3-D non-simplex arm integrates `∫|det J|` through the family-true
//! isoparametric `geom_jacobian` (the D235/D614 single source of truth),
//! sampled with the element family's own quadrature.
//!
//! Oracle: unit cube, one hex, `f ≡ 1`: `residual_estimator`'s interior term
//! is `η² = h²·f²·V = 3·1·1` → `η = √3`; `zz_estimator` on `u = x²` must give
//! strictly positive η on every hex.

use fem_assembly::postproc::error_estimate::{residual_estimator, zz_estimator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

#[test]
fn d636_zz_estimator_hex_indicators_not_silently_zero() {
    // 2×2×2 hexes; a field whose per-element gradient differs from its
    // neighbours' average, so a healthy ZZ recovery gives η > 0 (with
    // `u = x²` on a uniform grid the ZZ recovery is exact — superconvergence —
    // and η = 0 for the *right* reasons, which would mask the D636 defect).
    let m = Mesh::<3>::unit_cube_hex(2);
    let s = H1Space::new(m.clone(), 1);
    let d = s.interpolate(&|x| {
        (std::f64::consts::PI * x[0]).sin()
            + (std::f64::consts::PI * x[1]).sin()
            + (std::f64::consts::PI * x[2]).sin()
    });
    let gf = GridFunction::new(&s, d.as_slice().to_vec());
    let eta = zz_estimator(&gf).eta;
    assert_eq!(eta.len(), m.n_elems());
    for (e, v) in eta.iter().enumerate() {
        assert!(
            *v > 1e-10,
            "hex {e}: ZZ indicator {v} — elem_vol collapsed to 0 (D636)"
        );
    }
}

#[test]
fn d636_residual_estimator_hex_volume_matches_h2_f2_v() {
    // One unit hex: h = √3 (space diagonal), V = 1 → η = √(h²·f²·V) = √3.
    let m = Mesh::<3>::unit_cube_hex(1);
    assert_eq!(m.n_elems(), 1, "unit_cube_hex(1) = one hex");
    let s = H1Space::new(m.clone(), 1);
    let d = s.interpolate(&|_x| 0.0);
    let gf = GridFunction::new(&s, d.as_slice().to_vec());
    let eta = residual_estimator(&gf, &|_x| 1.0).eta;
    assert!(
        (eta[0] - 3.0f64.sqrt()).abs() < 1e-10,
        "η = {} (expected √3 = h·f·√V on the unit hex)",
        eta[0]
    );
}
