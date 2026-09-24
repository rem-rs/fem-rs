//! D696 batch-3 red pins: the `det.abs()` → signed-det conversions in the
//! postproc family (`error_estimate.rs` elem_vol + ZZ kernels,
//! `flux_recovery.rs` ComputeFluxEnergy, `postprocess.rs` H¹ error / element
//! volume / scalar integration, `grid_function.rs` L¹/L²/H¹ error kernels)
//! must carry the **MFEM signed `Trans.Weight()` semantics**.
//!
//! MFEM ground truth (4.10): `ElementTransformation::EvalWeight` (eltrans.cpp:30)
//! returns `Jacobian().Weight()` and `DenseMatrix::Weight`'s square-matrix
//! branch is `return Det();` — the upstream `fabs(Det())` is **commented out**
//! (linalg/densemat.cpp); the 3×2 surface branch `sqrt(E·G−F·F)` is
//! intrinsically nonnegative.  The D679 precedent: inverted tet
//! `GetElementVolume` = −1/6.
//!
//! Analytic pin: on the {positive, positive, inverted}-triangle mesh every
//! station integrates the same constant class, so the accumulated measure is
//! `+1/2 + 1/2 − 1/2 = +1/2` where `abs()` weights would give `3/2`.  The
//! estimator stations additionally pin the inverted-element **NaN signature**
//! (`sqrt(negative element energy)`), which is exactly MFEM's behavior on an
//! inverted cell.
//!
//! Run:
//!   cargo test -p fem-assembly --test d696_signed_det_pins_batch3 -- --nocapture

use fem_assembly::postproc::error_estimate::zz_estimator;
use fem_assembly::postproc::flux_recovery::zz_estimator_mfem;
use fem_assembly::postproc::grid_function::{compute_coeff_l2_norm, GridFunction};
use fem_assembly::postproc::postprocess::compute_h1_error;
use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Unit-square corners with three P1 triangles: `[0,1,2]` (+1/2),
/// `[0,2,3]` (+1/2), `[1,0,2]` (node swap → det = −1, −1/2).
fn pos_pos_inv_mesh() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 1, 2, 0, 2, 3, 1, 0, 2],
        vec![1, 1, 1],
        ElementType::Tri3,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

const SQRT_HALF: f64 = 0.7071067811865476; // sqrt(+1/2), the signed measure
const SQRT_1_5: f64 = 1.224744871391589; // sqrt(3/2), what abs() would give

/// compute_coeff_l2_norm(1) = √(Σ_e ±|K|) — the grid_function.rs L² kernel.
#[test]
fn d696_b3_coeff_l2_norm_signed_pin() {
    let mesh = pos_pos_inv_mesh();
    let norm = compute_coeff_l2_norm(&mesh, &|_p| 1.0, 2);
    println!("coeff L2 norm: {norm:.14e} (signed {SQRT_HALF}, abs {SQRT_1_5})");
    assert!(
        (norm - SQRT_HALF).abs() < 1e-12,
        "compute_coeff_l2_norm must carry MFEM's signed Trans.Weight(): {norm}"
    );
}

/// H¹ error of u_h ≡ 0 against ∇u = (1,0) = √(∫1) — the postprocess.rs kernel.
#[test]
fn d696_b3_h1_error_signed_pin() {
    let mesh = pos_pos_inv_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = vec![0.0_f64; space.n_dofs()];
    let err = compute_h1_error(&space, &dofs, |_p| vec![1.0, 0.0], 2);
    println!("H1 error: {err:.14e} (signed {SQRT_HALF}, abs {SQRT_1_5})");
    assert!(
        (err - SQRT_HALF).abs() < 1e-12,
        "compute_h1_error must carry MFEM's signed Trans.Weight(): {err}"
    );
}

/// The estimator stations: on the inverted triangle the element energy
/// `‖∇u_h − R(∇u_h)‖²·V_K` is NEGATIVE (V_K < 0, recovery residual > 0), so
/// `η_K = sqrt(·)` is NaN — exactly MFEM's signed-Weight behavior.  The two
/// positive triangles stay finite.
#[test]
fn d696_b3_zz_estimator_inverted_nan_pin() {
    let mesh = pos_pos_inv_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::from_projection(&space, &|p: &[f64]| p[0], 2);
    let ind = zz_estimator(&gf);
    println!(
        "zz eta: finite [{:.6e}, {:.6e}] inverted {} (NaN = signed signature)",
        ind.eta[0], ind.eta[1], ind.eta[2]
    );
    assert!(ind.eta[0].is_finite() && ind.eta[0] >= 0.0, "positive tri eta");
    assert!(ind.eta[1].is_finite() && ind.eta[1] >= 0.0, "positive tri eta");
    assert!(
        ind.eta[2].is_nan(),
        "inverted tri must carry the MFEM signed-Weight NaN signature: {}",
        ind.eta[2]
    );
}

/// The ComputeFluxEnergy station (flux_recovery.rs, MFEM
/// `DiffusionIntegrator::ComputeFluxEnergy`): the MFEM-style ZZ estimator on
/// the inverted triangle must reproduce the same NaN signature.
#[test]
fn d696_b3_flux_energy_zz_mfem_inverted_nan_pin() {
    let mesh = pos_pos_inv_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::from_projection(&space, &|p: &[f64]| p[0], 2);
    let integrator = DiffusionIntegrator { kappa: 1.0 };
    let ind = zz_estimator_mfem(&gf, &integrator);
    println!(
        "zz(MFEM) eta: finite [{:.6e}, {:.6e}] inverted {} (NaN = signed signature)",
        ind.eta[0], ind.eta[1], ind.eta[2]
    );
    assert!(ind.eta[0].is_finite() && ind.eta[0] >= 0.0, "positive tri eta");
    assert!(ind.eta[1].is_finite() && ind.eta[1] >= 0.0, "positive tri eta");
    assert!(
        ind.eta[2].is_nan(),
        "inverted tri ComputeFluxEnergy must be signed (NaN): {}",
        ind.eta[2]
    );
}
