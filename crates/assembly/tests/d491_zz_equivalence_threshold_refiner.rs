//! D491 + D404-deepening (D497) verification probes.
//!
//! # D491 — `zz_estimator_nodal` vs `zz_estimator_mfem_nc`
//!
//! Round 52 registered the suspicion that the two ZZ estimators are
//! "bit-identical duplicates".  They are not:
//!
//! * They are **mathematically identical** (the same MFEM
//!   `SumFluxAndCount` + `ComputeFluxEnergy` three-step ZZ algorithm), but
//!   the floating-point operation order differs — `zz_estimator_nodal`
//!   rotates *transform-first* (`grad_phys_i = J⁻ᵀ·∂φ_i`, then
//!   `∇u = Σ u_i·grad_phys_i`), `zz_estimator_mfem_nc` goes *combine-first*
//!   (`∇ξu = Σ u_i·∂φ_i`, then `flux = J⁻¹·∇ξu`) to match MFEM
//!   `DiffusionIntegrator::ComputeElementFlux` bit-for-bit.  The d404
//!   estimator-variants probe prints both at 4 significant digits with
//!   equal-looking values while their near-threshold mark counts differ
//!   (level 0: `0.5·RMS` marks 10 vs 12 of 20) — rounding noise, not
//!   duplication.  The first test pins the measured agreement.
//! * Both have live, disjoint consumers (`examples/mfem_ex6_flux_recovery`
//!   + d185 + d404_probe vs `amr_refiner` + d353/d365 + d404_probe), and
//!   their feature sets differ (aniso flags + `ref_elem_vol` sampling vs
//!   `FluxRecovery` genericity + Bergot pyramid sampling), so neither can
//!   be deleted or merged (conclusion recorded on `zz_estimator_nodal`'s
//!   doc).
//! * The genuine literal-copy pair is `zz_estimator_mfem` vs
//!   `zz_estimator_mfem_nc` (identical bodies; the NC variant only adds the
//!   D455-unused `constraints` parameter).  The second test pins their
//!   bit-identity as the invariant that documents the (deliberate,
//!   MFEM-parity) redundancy; merging them needs an edit in
//!   `flux_recovery.rs` (outside this round's authorized files).
//!
//! # D497 — `ThresholdRefiner` full MFEM marking semantics
//!
//! `amr_refiner::ThresholdRefiner` now implements MFEM
//! `ThresholdRefiner::MarkWithoutRefining` (`mesh/mesh_operators.cpp`)
//! verbatim:
//!
//! ```text
//! total_err = (Σ η_i^p)^(1/p)   (p = total_norm_p, ∞ → max)
//! threshold = max(total_err·total_fraction·N^(−1/p), local_err_goal)  p < ∞
//! threshold = max(total_err·total_fraction,          local_err_goal)  p = ∞
//! mark: η_i > threshold
//! STOP: NE ≥ max_elements | total_err ≤ total_err_goal | no marks
//! ```
//!
//! The tests pin: the p = 2 threshold ≡ [`rms_mark`] marking, the default
//! p = ∞ (η > 0.5·max) rule, the `local_err_goal` **floor** semantics (incl.
//! ex15's fraction-0 purely-local configuration), the three STOP criteria,
//! and the aniso path using the same family.

use std::f64::consts::PI;

use fem_assembly::postproc::amr_refiner::ThresholdRefiner;
use fem_assembly::postproc::error_estimate::zz_estimator_nodal;
use fem_assembly::postproc::flux_recovery::{zz_estimator_mfem, zz_estimator_mfem_nc};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::amr::NCState;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// The D404 Poisson MMS field.
fn mms(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }

/// ── D491 ─────────────────────────────────────────────────────────────────────
///
/// `zz_estimator_nodal` and `zz_estimator_mfem_nc` agree to fp-reordering
/// noise (max relative Δη ≲ 1e-12) on the D404 conforming MMS mesh — they
/// are the same mathematics, not the same code path.  Prints the exact
/// deviation and the bit-equality count for the debt record.
#[test]
fn d491_nodal_vs_mfem_nc_mathematically_equivalent() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&mms);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    let int = DiffusionIntegrator { kappa: 1.0 };

    let a = zz_estimator_nodal(&gf, &[]);
    let b = zz_estimator_mfem_nc(&gf, &int, &[]);
    assert_eq!(a.eta.len(), b.eta.len());

    let ne = a.eta.len();
    let max_eta = a.eta.iter().cloned().fold(0.0_f64, f64::max);
    let max_diff = a
        .eta
        .iter()
        .zip(&b.eta)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    let bit_eq = a
        .eta
        .iter()
        .zip(&b.eta)
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    println!(
        "D491: ne={ne}  ‖η‖_nodal={:.6e}  ‖η‖_mfem_nc={:.6e}  max|Δη|={max_diff:.3e}  \
         rel={:.3e}  bit-equal={bit_eq}/{ne}",
        a.total_error,
        b.total_error,
        max_diff / max_eta
    );

    // Same mathematics ⇒ agreement to fp reordering noise only.
    assert!(
        max_diff / max_eta < 1e-10,
        "nodal and mfem_nc ZZ must agree to fp-reordering noise, got rel={:.3e}",
        max_diff / max_eta
    );
    assert!(
        (a.total_error - b.total_error).abs() < 1e-10 * b.total_error,
        "total errors diverge: {} vs {}",
        a.total_error,
        b.total_error
    );
}

/// ── D491 ─────────────────────────────────────────────────────────────────────
///
/// The genuine redundancy: `zz_estimator_mfem` ≡ `zz_estimator_mfem_nc`
/// bit-for-bit for empty constraints (identical bodies; the NC variant only
/// carries the D455-unused `constraints` parameter, kept for MFEM parity —
/// an H1 flux space makes MFEM's primal hanging-node transforms no-ops).
/// Pinning the bit-identity documents the equivalence and catches any future
/// divergence between the two bodies.
#[test]
fn d491_mfem_and_mfem_nc_bit_identical_for_empty_constraints() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&mms);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    let int = DiffusionIntegrator { kappa: 1.0 };

    let a = zz_estimator_mfem(&gf, &int);
    let b = zz_estimator_mfem_nc(&gf, &int, &[]);
    assert_eq!(a.eta.len(), b.eta.len());
    for (i, (x, y)) in a.eta.iter().zip(&b.eta).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "eta[{i}] diverged: {x} vs {y} — the mfem/mfem_nc bodies are no longer identical"
        );
    }
}

/// ── D497 ─────────────────────────────────────────────────────────────────────
///
/// `total_norm_p = 2` + `total_fraction = 0.5`: the refiner's threshold is
/// `0.5·‖η‖₂/√N` and its marking set coincides with
/// [`fem_assembly::postproc::error_estimate::ElementIndicators::rms_mark`]
/// (the D404 rule) — the round-52 temporary is now a first-class
/// `ThresholdRefiner` configuration.
#[test]
fn d497_threshold_refiner_p2_threshold_matches_rms_mark() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let ne = mesh.n_elements();
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&mms);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    let int = DiffusionIntegrator { kappa: 1.0 };

    // Reference indicators: the exact call the refiner performs internally.
    let ind = zz_estimator_mfem_nc(&gf, &int, &[]);
    let expected_marks = ind.rms_mark(0.5);
    // Mirror `compute_threshold`'s association: (total·fraction)·N^(−1/p).
    let expected_threshold = ind.total_error * 0.5 * (ne as f64).powf(-1.0 / 2.0);
    assert!(!expected_marks.is_empty(), "test field must mark something");

    let mut mesh = mesh;
    let mut nc_state = NCState::new();
    let mut refiner = ThresholdRefiner::new(false);
    refiner.set_total_error_norm_p(2.0);
    refiner.set_total_error_fraction(0.5);
    refiner.apply(&mut mesh, &mut nc_state, &gf, &int, Some(&[]));

    let thr = refiner.threshold();
    assert!(
        (thr - expected_threshold).abs() <= 1e-12 * expected_threshold,
        "p2 threshold {thr} vs rms formula {expected_threshold}"
    );
    assert_eq!(
        refiner.last_marked, expected_marks,
        "p2 marking must equal rms_mark(0.5)"
    );
    assert!(!refiner.stop(), "marks were produced, loop must continue");
    assert!(mesh.n_elements() > ne, "marked elements must refine the mesh");
}

/// ── D497 ─────────────────────────────────────────────────────────────────────
///
/// Defaults (p = ∞, fraction = 0.5, goal = 0) mark `η > 0.5·‖η‖_∞`, and
/// `local_err_goal` is a **floor** on the computed threshold — with
/// `fraction = 0` the rule degenerates to the purely-local threshold (MFEM
/// ex15's configuration, and the pre-D497 fem-rs behaviour).
#[test]
fn d497_threshold_refiner_default_linf_and_local_goal_floor() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let ne0 = mesh.n_elements();
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&mms);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    let int = DiffusionIntegrator { kappa: 1.0 };
    let ind = zz_estimator_mfem_nc(&gf, &int, &[]);
    let max_eta = ind.eta.iter().cloned().fold(0.0_f64, f64::max);

    // (a) Default rule: η > 0.5·max(η).
    let mut mesh_a = mesh.clone();
    let mut nc_a = NCState::new();
    let mut r = ThresholdRefiner::new(false);
    r.apply(&mut mesh_a, &mut nc_a, &gf, &int, Some(&[]));
    assert!(
        (r.threshold() - 0.5 * max_eta).abs() <= 1e-15 * max_eta,
        "default threshold {} vs 0.5·max {:#e}",
        r.threshold(),
        0.5 * max_eta
    );
    let want: Vec<u32> = ind
        .eta
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e > 0.5 * max_eta)
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(r.last_marked, want, "default marking = 0.5·max rule");
    assert!(!r.stop());

    // (b) fraction = 0 + goal g ⇒ threshold == g (ex15 purely-local rule).
    let mut mesh_b = mesh.clone();
    let mut nc_b = NCState::new();
    let mut r = ThresholdRefiner::new(false);
    r.set_total_error_fraction(0.0);
    r.set_local_error_goal(1.0e-2);
    r.apply(&mut mesh_b, &mut nc_b, &gf, &int, Some(&[]));
    assert_eq!(r.threshold(), 1.0e-2, "fraction-0 threshold must equal the goal");
    let want: Vec<u32> = ind
        .eta
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e > 1.0e-2)
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(r.last_marked, want);

    // (c) The goal floors the fraction term: goal above every η ⇒ nothing
    //     marked (STOP), threshold == goal.
    let mut mesh_c = mesh;
    let mut nc_c = NCState::new();
    let mut r = ThresholdRefiner::new(false);
    r.set_local_error_goal(2.0 * max_eta);
    r.apply(&mut mesh_c, &mut nc_c, &gf, &int, Some(&[]));
    assert_eq!(r.threshold(), 2.0 * max_eta);
    assert!(r.stop(), "no η above the goal floor ⇒ STOP");
    assert_eq!(mesh_c.n_elements(), ne0, "STOP must not refine");
}

/// ── D497 ─────────────────────────────────────────────────────────────────────
///
/// The three MFEM STOP criteria: `max_elements` (stops *before* estimating),
/// `total_err_goal` (stops after estimating), and the degenerate
/// `total_err = 0 ≤ goal = 0` zero-field case.
#[test]
fn d497_threshold_refiner_stop_criteria() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let ne0 = mesh.n_elements();
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&mms);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    let int = DiffusionIntegrator { kappa: 1.0 };

    // (a) max_elements reached on the input mesh ⇒ STOP, no estimation.
    let mut m = mesh.clone();
    let mut nc = NCState::new();
    let mut r = ThresholdRefiner::new(false);
    r.set_max_elements(ne0 as u64);
    r.apply(&mut m, &mut nc, &gf, &int, Some(&[]));
    assert!(r.stop(), "NE >= max_elements must STOP");
    assert_eq!(m.n_elements(), ne0, "max_elements STOP must not refine");
    assert!(r.eta.is_empty(), "max_elements STOP skips estimation (MFEM)");
    assert_eq!(r.threshold(), 0.0, "threshold reset at MarkWithoutRefining entry");

    // (b) total_err_goal above the total error ⇒ STOP after estimating.
    let mut r = ThresholdRefiner::new(false);
    r.set_total_error_goal(1.0e9);
    r.apply(&mut m, &mut nc, &gf, &int, Some(&[]));
    assert!(r.stop(), "total_err <= total_err_goal must STOP");
    assert_eq!(m.n_elements(), ne0);
    assert!(!r.eta.is_empty(), "estimation ran before the goal STOP");
    assert_eq!(r.threshold(), 0.0);

    // (c) Zero field ⇒ total_err = 0 <= default goal 0 ⇒ STOP (the legacy
    //     no-marks-→-stop behaviour of a converged/exact solution).
    let dz = space.interpolate(&|_| 0.0);
    let gf0 = GridFunction::new(&space, dz.as_slice().to_vec());
    let mut m0 = mesh;
    let mut r = ThresholdRefiner::new(false);
    r.apply(&mut m0, &mut nc, &gf0, &int, Some(&[]));
    assert!(r.stop(), "zero indicators must STOP");
    assert_eq!(m0.n_elements(), ne0);
}

/// ── D497 ─────────────────────────────────────────────────────────────────────
///
/// The aniso path uses the same MFEM threshold family: in MFEM the
/// anisotropic estimator only overrides each marked `Refinement`'s *type*
/// (`ref.SetType(aniso_flags[ref.index])`); the marking threshold is the same
/// p-norm rule.
#[test]
fn d497_threshold_refiner_aniso_uses_pnorm_threshold_family() {
    let mesh = Mesh::<2>::unit_square_quad(2);
    let ne = mesh.n_elements();
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&mms);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());

    let ind = zz_estimator_nodal(&gf, &[]);
    let expected_marks = ind.rms_mark(0.5);
    let expected_threshold = ind.total_error * 0.5 * (ne as f64).powf(-1.0 / 2.0);
    assert!(!expected_marks.is_empty(), "test field must mark something");

    let mut m = mesh;
    let mut refiner = ThresholdRefiner::new(false);
    refiner.set_total_error_norm_p(2.0);
    refiner.set_total_error_fraction(0.5);
    refiner.apply_aniso(&mut m, &ind);

    let thr = refiner.threshold();
    assert!(
        (thr - expected_threshold).abs() <= 1e-12 * expected_threshold,
        "aniso p2 threshold {thr} vs rms formula {expected_threshold}"
    );
    assert_eq!(
        refiner.last_marked, expected_marks,
        "aniso marking must equal the same p2 rule"
    );
    assert!(!refiner.stop());
    assert!(m.n_elements() > ne, "aniso marks must refine the quad mesh");
}
