//! D1237 regression pins: the **real-valued** `DpgWeakForm` face quadrature is
//! sized per trace integrator the MFEM way (`test_fe.GetOrder() +
//! trial_face_fe.GetOrder()`, bilininteg.cpp TraceIntegrator :4429 /
//! NormalTraceIntegrator :4480 / TangentTraceIntegrator :4582), not with one
//! fixed order — the D1221 fix of the complex twin
//! (`complex_dpg_weakform.rs`), mirrored onto the real weak form.
//!
//! Defect (round 116): `dpg_weakform.rs` still defaulted to a single face
//! rule of order 4.  On quad faces `quad_rule_01(4)` is a 3×3 Gauss rule
//! (exact degree 5), so any trial/test pair with `p_face + test_order > 5`
//! was under-integrated — the same latent trap that corrupted the complex
//! acoustics solves at `-o ≥ 2 -do ≥ 1` (D1221).  Every production caller of
//! the real weak form (the ex31-family serial miniapps `dpg_poisson_2d` /
//! `pconvection_diffusion`, the parallel `pdiffusion` / `pconvection_diffusion`
//! via `ParDpgWeakForm`, and the NC/SC test suites) passes the value
//! explicitly — for the ultraweak Poisson family both face pairs need exactly
//! `2p + do − 1` = `test_order + p − 1`, so the historical single-rule call
//! was exact and those tiers are byte-identical before/after the default
//! change.  The pins below lock that:
//!
//! 1. `explicit_face_rule_unchanged_2d` — the ex31-family serial builder
//!    (explicit `set_face_quad_order(test_order + p − 1)`, exactly what the
//!    miniapps do) at `p = 2` and `p = 3`: pinned matrix invariants.  The
//!    explicit path maps to the same rule it did before D1237 by
//!    construction; the constants freeze it.
//! 2. `auto_equals_explicit_mfem_rule_3d` — on a 3-D hex ultraweak Poisson
//!    (`p = 3, do = 1`, both face pairs need degree 6) the new default (auto)
//!    must assemble a **bitwise identical** matrix to the explicit
//!    `set_face_quad_order(6)` call — i.e. auto selects exactly the MFEM rule.
//! 3. `order4_trap_sensitivity_3d` — the same 3-D case with the historical
//!    `set_face_quad_order(4)` must move the matrix by > 1e-3 relative, i.e.
//!    the pin catches the under-integration defect.
//!
//! Tolerances: bitwise for pin 2 (same rule ⇒ same summation), `1e-3`
//! relative for pin 3 (quadrature-order mistakes on degree-6 integrands move
//! entries by percent-level; round 115 measured the same magnitude on the
//! complex twin).

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTraceIntegrator, DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::{DpgSystem, DpgWeakForm};
use fem_mesh::Mesh;

/// `-(σ, τ)` adapter — MFEM `TransposeIntegrator(VectorFEMassIntegrator(-1))`
/// (copied from the miniapp; the integrator is not part of the public API).
struct NegVectorMass;
impl fem_assembly::dpg::dpg_integrators::DpgBilinear2 for NegVectorMass {
    fn assemble2(
        &self,
        ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
        trial: &fem_assembly::dpg::dpg_basis::VolVals,
        test: &fem_assembly::dpg::dpg_basis::VolVals,
        m: &mut [f64],
    ) {
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for k in 0..nt {
            for c in 0..d {
                for j in 0..nsc {
                    m[k * nc + c * nsc + j] -= ctx.w * test.phi[k * d + c] * trial.phi[j];
                }
            }
        }
    }
}

/// Ultraweak DPG Poisson form, 1:1 with `miniapps/dpg/dpg_poisson_2d.rs`
/// `build` (including its explicit `set_face_quad_order(test_order + p − 1)`).
fn build_poisson_2d_explicit(mesh: &Mesh<2>, p: u8) -> DpgWeakForm<Mesh<2>> {
    let test_order = p + 1; // delta_order = 1
    let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh.clone());
    let u = a.add_trial_scalar_space(p - 1);
    let sig = a.add_trial_vector_space(p - 1, 2);
    let hatu = a.add_trial_trace_space_h1(p);
    let hatsig = a.add_trial_trace_space(p - 1);
    let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
    let v = a.add_test_space(VolKind::Scalar, test_order);

    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
        u,
        tau,
    );
    a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
    a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
    a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
    a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
    a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
    a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau);
    a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
    a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
    // The miniapp's exact calls:
    a.set_quad_order(2 * test_order);
    a.set_face_quad_order(test_order + p - 1);
    a
}

/// Ultraweak DPG Poisson form, dim `dim` (2 or 3), NO explicit face rule —
/// the assembly uses whatever the weak form's default policy is.  The trial
/// and test blocks mirror the miniapp family; both face pairs
/// (hatu × tau, hatsig × v) couple degree `2p + do − 1` on any mesh.
fn build_poisson<const D: usize>(mesh: &Mesh<D>, p: u8, delta_order: u8) -> DpgWeakForm<Mesh<D>>
where
    Mesh<D>: fem_mesh::MeshTopology + Clone + 'static,
{
    let test_order = p + delta_order;
    let mut a: DpgWeakForm<Mesh<D>> = DpgWeakForm::new(mesh.clone());
    let u = a.add_trial_scalar_space(p - 1);
    let sig = a.add_trial_vector_space(p - 1, D);
    let hatu = a.add_trial_trace_space_h1(p);
    let hatsig = a.add_trial_trace_space(p - 1);
    let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
    let v = a.add_test_space(VolKind::Scalar, test_order);

    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
        u,
        tau,
    );
    a.add_trial_integrator(Box::new(NegVectorMass), sig, tau);
    a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
    a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
    a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v);
    a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
    a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau);
    a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
    a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v);
    a
}

/// `(size, nnz, Frobenius norm)` of the assembled normal system.
fn system_invariants(sys: &DpgSystem) -> (usize, usize, f64) {
    let mat = sys.matrix();
    let fro: f64 = mat.values.iter().map(|&v| v * v).sum::<f64>().sqrt();
    (mat.nrows, mat.values.len(), fro)
}

/// Assemble the full normal system (no BCs, no condensation).
fn assemble_full<M: fem_mesh::MeshTopology + Clone + 'static>(
    mut a: DpgWeakForm<M>,
) -> DpgSystem {
    a.assemble();
    let n = a.size();
    let (sys, _b, _x) = a.form_linear_system(&[], &vec![0.0; n], false);
    sys
}

/// Pin 1: the ex31-family serial tier (explicit face rule, exactly the
/// miniapps' call) is unchanged by D1237 — the explicit path feeds the same
/// rule order to the same `face_quadrature` as before.  Constants captured on
/// the D1237 build; pre-D1237 they are identical by construction (see module
/// docs), so any drift here is a regression of the explicit path itself.
#[test]
fn explicit_face_rule_unchanged_2d() {
    // (p, size, nnz, frobenius)
    const PINS: &[(u8, usize, usize, f64)] = &[
        (2, 93, 2317, 10.916_709_129_054_590),
        (3, 177, 8289, 13.608_574_348_238_127),
    ];
    let mesh = Mesh::<2>::unit_square_quad(2);
    for &(p, size, nnz, fro) in PINS {
        let sys = assemble_full(build_poisson_2d_explicit(&mesh, p));
        let (s, nz, f) = system_invariants(&sys);
        println!("CAPTURE p{p}: size={s} nnz={nz} fro={f:.17e}");
        assert_eq!(s, size, "p{p}: system size");
        assert_eq!(nz, nnz, "p{p}: nnz");
        assert!(
            (f - fro).abs() <= 1e-9 * fro.abs().max(1.0),
            "p{p}: Frobenius norm {f:.17e} vs pinned {fro:.17e}"
        );
    }
}

/// Pin 2: on the 3-D hex ultraweak Poisson (`p = 3, do = 1`; both face pairs
/// need degree `2·3 + 1 − 1 = 6` on the quad faces) the DEFAULT (auto,
/// per-pair MFEM sizing) must assemble a bitwise-identical matrix to the
/// explicit single rule `set_face_quad_order(6)` — auto selects exactly the
/// MFEM rule, so callers can either trust the default or state it.
#[test]
fn auto_equals_explicit_mfem_rule_3d() {
    let mesh = Mesh::<3>::unit_cube_hex(1);

    let mut auto = build_poisson(&mesh, 3, 1);
    auto.assemble();
    let n = auto.size();
    let (sys_auto, _, _) = auto.form_linear_system(&[], &vec![0.0; n], false);

    let mut expl = build_poisson(&mesh, 3, 1);
    expl.set_face_quad_order(6); // = test_order + p − 1 = 4 + 3 − 1, both pairs
    expl.assemble();
    let (sys_expl, _, _) = expl.form_linear_system(&[], &vec![0.0; n], false);

    let (ma, me) = (sys_auto.matrix(), sys_expl.matrix());
    assert_eq!(ma.nrows, me.nrows);
    assert_eq!(ma.values.len(), me.values.len(), "nnz must match");
    for (i, (&va, &ve)) in ma.values.iter().zip(me.values.iter()).enumerate() {
        assert_eq!(
            va.to_bits(),
            ve.to_bits(),
            "entry {i} differs: auto {va:.17e} vs explicit {ve:.17e}"
        );
    }
}

/// Pin 3: the historical fixed order 4 (3×3 Gauss on quads, exact degree 5)
/// under-integrates the degree-6 face pairs of `p = 3, do = 1` — the matrix
/// must move by > 1e-3 relative, i.e. this pin FAILS on a pre-D1237 build
/// (where the default WAS the fixed-4 rule).
#[test]
fn order4_trap_sensitivity_3d() {
    let mesh = Mesh::<3>::unit_cube_hex(1);

    let mut auto = build_poisson(&mesh, 3, 1);
    auto.assemble();
    let n = auto.size();
    let (sys_auto, _, _) = auto.form_linear_system(&[], &vec![0.0; n], false);

    let mut broken = build_poisson(&mesh, 3, 1);
    broken.set_face_quad_order(4); // the historical default: exact degree 5
    broken.assemble();
    let (sys_broken, _, _) = broken.form_linear_system(&[], &vec![0.0; n], false);

    let (ma, mb) = (sys_auto.matrix(), sys_broken.matrix());
    assert_eq!(ma.values.len(), mb.values.len(), "nnz must match");
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (&va, &vb) in ma.values.iter().zip(mb.values.iter()) {
        worst = worst.max((va - vb).abs());
        scale = scale.max(va.abs());
    }
    assert!(
        worst > 1e-3 * scale,
        "fixed-4 rule must move the matrix (> 1e-3 relative), got {worst:.3e} vs scale {scale:.3e}"
    );
}
