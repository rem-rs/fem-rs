//! D696 batch-1 red pins: the `det.abs()` → signed-det conversions in
//! `vector_assembler.rs` (2-D RT×ND curl pairings, D667 stragglers),
//! `partial.rs` (PA mass / PA diffusion / lumped mass / Hcurl matrix-free)
//! and `mixed/mod.rs` (hdiv_l2_mixed / hcurl_h1_mixed / hcurl_h1_gradient /
//! h1_hdiv_mixed) must carry the **MFEM signed `Trans.Weight()` semantics**
//! (D679/D652 precedent).  Every pin is `abs == signed` on positively
//! oriented elements and differs ONLY on inverted (det < 0) ones, so all
//! positive-det byte-pins (ex8 DPG, ex4/ex5, d652/d667 parity) are untouched
//! by construction.
//!
//! Pin form: the entry-sum functional `1ᵀM1` is invariant under the vertex
//! permutation an orientation flip induces on the reference-to-physical map,
//! so **signed semantics ⇔ `1ᵀM1(inverted) = −1ᵀM1(positive)`** (the old
//! `abs()` gave equality).  Where an analytic value exists (affine tri mass /
//! diffusion = −1/2, D679) it is pinned directly.
//!
//! Run:
//!   cargo test -p fem-assembly --test d696_signed_det_pins_batch1 -- --nocapture

use fem_assembly::mixed::{
    assemble_h1_hdiv_mixed, assemble_hcurl_h1_gradient, assemble_hcurl_h1_mixed,
    HCurlH1CurlIntegrator, HDivL2DivIntegrator, H1HDivIntegrator,
    MixedVectorGradientIntegrator,
};
use fem_assembly::partial::{
    HcurlMatrixFreeOperator, LumpedMassOperator, PADiffusionOperator, PAMassOperator,
    MatFreeOperator,
};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::{FESpace, HCurlSpace, HDivSpace, H1Space, L2Space};

/// One positively (CCW) and one negatively (CW) oriented unit triangle /
/// square, built directly through `Mesh::uniform` (no reader-side orientation
/// fixups — the d679 pattern).
fn positive_tri() -> Mesh<2> {
    // v0=(0,0) v1=(1,0) v2=(0,1); CCW conn [0,1,2] → det = +1.
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

fn inverted_tri() -> Mesh<2> {
    // Same two physical vertices as positive_tri but CW conn [0,2,1] → det = −1.
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 2, 1],
        vec![1],
        ElementType::Tri3,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

fn positive_quad() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Quad4,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

fn inverted_quad() -> Mesh<2> {
    // CW conn [0,3,2,1] → det = −1 (d679 pattern).
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 3, 2, 1],
        vec![1],
        ElementType::Quad4,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

/// Sum of all entries of a CSR matrix (`1ᵀM1` without assembling 1 twice).
fn entry_sum(m: &CsrMatrix<f64>) -> f64 {
    let ones = vec![1.0_f64; m.ncols];
    let mut y = vec![0.0_f64; m.nrows];
    m.spmv(&ones, &mut y);
    y.iter().sum()
}

#[test]
fn d696_pa_mass_inverted_tri_pin() {
    // partial.rs PAMassOperator — MFEM `MassIntegrator` signed Weight:
    // 1ᵀM1 = −∫1 = −1/2 on the inverted tri (D679 analytic value).
    let mesh = inverted_tri();
    let space = H1Space::new(mesh.clone(), 1);
    let pa = PAMassOperator::new(space, 1.0_f64, 3);
    let n = MatFreeOperator::n_dofs(&pa);
    let one = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; n];
    pa.apply(&one, &mut y);
    let s: f64 = y.iter().sum();
    println!("inverted tri PA mass 1ᵀM1 = {s:.12e} (MFEM signed −0.5)");
    assert!((s + 0.5).abs() < 1e-12, "PA mass must follow signed det: {s}");
}

#[test]
fn d696_pa_diffusion_inverted_tri_pin() {
    // partial.rs PADiffusionOperator — MFEM `DiffusionIntegrator` signed
    // Weight: uᵀKu (u = x) = −1/2 (D679 analytic value on the assembled path).
    let mesh = inverted_tri();
    let space = H1Space::new(mesh.clone(), 1);
    let pa = PADiffusionOperator::new(space, 1.0_f64, 3);
    let n = MatFreeOperator::n_dofs(&pa);
    let x: Vec<f64> = (0..n)
        .map(|d| mesh.node_coords(d as u32)[0])
        .collect();
    let mut y = vec![0.0_f64; n];
    pa.apply(&x, &mut y);
    let s: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    println!("inverted tri PA diffusion uᵀKu = {s:.12e} (MFEM signed −0.5)");
    assert!((s + 0.5).abs() < 1e-12, "PA diffusion must follow signed det: {s}");
}

#[test]
fn d696_lumped_mass_inverted_tri_pin() {
    // partial.rs LumpedMassOperator — row sums of the consistent mass:
    // Σ_i diag_i = ∫(Σφ)² = −1/2 signed on the inverted tri.
    let mesh = inverted_tri();
    let space = H1Space::new(mesh.clone(), 1);
    let lumped = LumpedMassOperator::assemble(&space, 1.0, 3);
    let s: f64 = lumped.diag.iter().sum();
    println!("inverted tri lumped Σdiag = {s:.12e} (MFEM signed −0.5)");
    assert!((s + 0.5).abs() < 1e-12, "lumped mass must follow signed det: {s}");
}

#[test]
fn d696_hcurl_matrixfree_inverted_tri_pin() {
    // partial.rs HcurlMatrixFreeOperator (curl-curl + vector mass) — the
    // entry sum must negate under orientation flip (abs() gave equality).
    let mu_inv = 1.0;
    let alpha = 1.0;
    let qo = 4;
    let pos = HcurlMatrixFreeOperator::new(&HCurlSpace::new(positive_tri(), 1), mu_inv, alpha, qo);
    let neg = HcurlMatrixFreeOperator::new(&HCurlSpace::new(inverted_tri(), 1), mu_inv, alpha, qo);
    let n = MatFreeOperator::n_dofs(&pos);
    let one = vec![1.0_f64; n];
    let mut yp = vec![0.0_f64; n];
    let mut yn = vec![0.0_f64; n];
    pos.apply(&one, &mut yp);
    neg.apply(&one, &mut yn);
    let (sp, sn): (f64, f64) = (yp.iter().sum(), yn.iter().sum());
    println!("hcurl matfree 1ᵀK1: positive {sp:.6e} inverted {sn:.6e}");
    assert!(sp.abs() > 1e-12, "pin must be non-degenerate");
    assert!(
        (sn + sp).abs() < 1e-12 * (1.0 + sp.abs()),
        "hcurl matrix-free must negate on inverted tri: {sp} vs {sn}"
    );
}

#[test]
fn d696_curl_pairing_2d_inverted_tri_pins() {
    // vector_assembler.rs `assemble_curl_hdiv_pairing_2d_{nd2_rt2,nd1_rt0}`
    // (D667 stragglers) — MFEM `MixedScalarCurlIntegrator` signed Weight.
    let pos_hcurl = HCurlSpace::new(positive_tri(), 2);
    let pos_hdiv = HDivSpace::new(positive_tri(), 2);
    let neg_hcurl = HCurlSpace::new(inverted_tri(), 2);
    let neg_hdiv = HDivSpace::new(inverted_tri(), 2);
    let c_pos = VectorAssembler::assemble_curl_hdiv_pairing_2d_nd2_rt2(&pos_hcurl, &pos_hdiv, 6);
    let c_neg = VectorAssembler::assemble_curl_hdiv_pairing_2d_nd2_rt2(&neg_hcurl, &neg_hdiv, 6);
    let (sp, sn) = (entry_sum(&c_pos), entry_sum(&c_neg));
    println!("nd2_rt2 pairing ΣC: positive {sp:.6e} inverted {sn:.6e}");
    assert!(sp.abs() > 1e-12, "pin must be non-degenerate");
    assert!(
        (sn + sp).abs() < 1e-12 * (1.0 + sp.abs()),
        "nd2_rt2 pairing must negate on inverted tri: {sp} vs {sn}"
    );

    let pos_hcurl = HCurlSpace::new(positive_tri(), 1);
    let pos_hdiv = HDivSpace::new(positive_tri(), 0);
    let neg_hcurl = HCurlSpace::new(inverted_tri(), 1);
    let neg_hdiv = HDivSpace::new(inverted_tri(), 0);
    let c_pos = VectorAssembler::assemble_curl_hdiv_pairing_2d_nd1_rt0(&pos_hcurl, &pos_hdiv, 6);
    let c_neg = VectorAssembler::assemble_curl_hdiv_pairing_2d_nd1_rt0(&neg_hcurl, &neg_hdiv, 6);
    let (sp, sn) = (entry_sum(&c_pos), entry_sum(&c_neg));
    println!("nd1_rt0 pairing ΣC: positive {sp:.6e} inverted {sn:.6e}");
    assert!(sp.abs() > 1e-12, "pin must be non-degenerate");
    assert!(
        (sn + sp).abs() < 1e-12 * (1.0 + sp.abs()),
        "nd1_rt0 pairing must negate on inverted tri: {sp} vs {sn}"
    );
}

#[test]
fn d696_hdiv_l2_mixed_inverted_tri_and_quad_pins() {
    // mixed/mod.rs `assemble_hdiv_l2_mixed` — affine (tri) and isoparametric
    // (quad) branches; MFEM `VectorFEDivergenceIntegrator`-family signed
    // Weight.  (MixedAssembler is the module-level export used by callers.)
    let pos_l2 = L2Space::new(positive_tri(), 0);
    let pos_vel = HDivSpace::new(positive_tri(), 0);
    let neg_l2 = L2Space::new(inverted_tri(), 0);
    let neg_vel = HDivSpace::new(inverted_tri(), 0);
    let b_pos = fem_assembly::mixed::assemble_hdiv_l2_mixed(
        &pos_l2, &pos_vel, &[&HDivL2DivIntegrator], 4,
    );
    let b_neg = fem_assembly::mixed::assemble_hdiv_l2_mixed(
        &neg_l2, &neg_vel, &[&HDivL2DivIntegrator], 4,
    );
    let (sp, sn) = (entry_sum(&b_pos), entry_sum(&b_neg));
    println!("hdiv_l2 tri ΣB: positive {sp:.6e} inverted {sn:.6e}");
    assert!(sp.abs() > 1e-12, "pin must be non-degenerate");
    assert!(
        (sn + sp).abs() < 1e-12 * (1.0 + sp.abs()),
        "hdiv_l2 mixed must negate on inverted tri: {sp} vs {sn}"
    );

    // Isoparametric branch (Quad4 always takes use_iso).
    let pos_l2 = L2Space::new(positive_quad(), 0);
    let pos_vel = HDivSpace::new(positive_quad(), 0);
    let neg_l2 = L2Space::new(inverted_quad(), 0);
    let neg_vel = HDivSpace::new(inverted_quad(), 0);
    let b_pos = fem_assembly::mixed::assemble_hdiv_l2_mixed(
        &pos_l2, &pos_vel, &[&HDivL2DivIntegrator], 4,
    );
    let b_neg = fem_assembly::mixed::assemble_hdiv_l2_mixed(
        &neg_l2, &neg_vel, &[&HDivL2DivIntegrator], 4,
    );
    let (sp, sn) = (entry_sum(&b_pos), entry_sum(&b_neg));
    println!("hdiv_l2 quad ΣB: positive {sp:.6e} inverted {sn:.6e}");
    assert!(sp.abs() > 1e-12, "pin must be non-degenerate");
    assert!(
        (sn + sp).abs() < 1e-12 * (1.0 + sp.abs()),
        "hdiv_l2 mixed must negate on inverted quad: {sp} vs {sn}"
    );
}

#[test]
fn d696_hcurl_h1_mixed_inverted_tri_pin() {
    // mixed/mod.rs `assemble_hcurl_h1_mixed` — MFEM mixed curl coupling
    // signed Weight.
    let pos_h1 = H1Space::new(positive_tri(), 1);
    let pos_nd = HCurlSpace::new(positive_tri(), 1);
    let neg_h1 = H1Space::new(inverted_tri(), 1);
    let neg_nd = HCurlSpace::new(inverted_tri(), 1);
    let m_pos = assemble_hcurl_h1_mixed(&pos_h1, &pos_nd, &[&HCurlH1CurlIntegrator], 4);
    let m_neg = assemble_hcurl_h1_mixed(&neg_h1, &neg_nd, &[&HCurlH1CurlIntegrator], 4);
    let (sp, sn) = (entry_sum(&m_pos), entry_sum(&m_neg));
    println!("hcurl_h1 ΣM: positive {sp:.6e} inverted {sn:.6e}");
    assert!(sp.abs() > 1e-12, "pin must be non-degenerate");
    assert!(
        (sn + sp).abs() < 1e-12 * (1.0 + sp.abs()),
        "hcurl_h1 mixed must negate on inverted tri: {sp} vs {sn}"
    );
}

#[test]
fn d696_hcurl_h1_gradient_inverted_tri_pin() {
    // mixed/mod.rs `assemble_hcurl_h1_gradient` (ND gauging block) — signed
    // Weight.  Convention-free functional through the transpose: with v =
    // ND dofs of the constant field e_x (exact in ND1), Gᵀ·v has H¹ rows at
    // the same physical vertices on both meshes and equals the SIGNED
    // integral ∫ ∂φ_i/∂x dV = ∓(1/2, −1/2, 0) — positive tri negative area
    // sign, inverted tri flipped by the signed det.  (The direct G·u rows
    // are ND edge dofs whose canonical slots re-order between the two
    // meshes, so the transpose carries the analytic pin instead.)
    let pos_nd = HCurlSpace::new(positive_tri(), 1);
    let pos_h1 = H1Space::new(positive_tri(), 1);
    let neg_nd = HCurlSpace::new(inverted_tri(), 1);
    let neg_h1 = H1Space::new(inverted_tri(), 1);
    let v_pos = pos_nd
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let v_neg = neg_nd
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let g_pos = assemble_hcurl_h1_gradient(&pos_nd, &pos_h1, 4);
    let g_neg = assemble_hcurl_h1_gradient(&neg_nd, &neg_h1, 4);
    let gt_pos = g_pos.transpose();
    let gt_neg = g_neg.transpose();
    let n = gt_pos.nrows;
    let mut zp = vec![0.0_f64; n];
    let mut zn = vec![0.0_f64; n];
    gt_pos.spmv(&v_pos, &mut zp);
    gt_neg.spmv(&v_neg, &mut zn);
    println!("hcurl_h1 grad Gᵀ·e_x: positive {zp:?} inverted {zn:?} (MFEM signed ∫∂φ/∂x)");
    let want_p = [-0.5_f64, 0.5, 0.0];
    for (i, &w) in want_p.iter().enumerate().take(n.min(3)) {
        assert!(
            (zp[i] - w).abs() < 1e-12,
            "positive tri must match MFEM signed ∫∂φ{i}/∂x = {w}, got {}",
            zp[i]
        );
        assert!(
            (zn[i] + w).abs() < 1e-12,
            "inverted tri must negate MFEM signed ∫∂φ{i}/∂x, got {} vs {w}",
            zn[i]
        );
    }
}

#[test]
fn d696_h1_hdiv_mixed_inverted_tri_and_quad_pins() {
    // mixed/mod.rs `assemble_h1_hdiv_mixed` — MFEM
    // `MixedVectorGradientIntegrator` (∫ σ ∇φ·w) signed Weight on both
    // branches.  Functional: M·v with v = HDiv dofs of the constant field
    // e_x (exactly representable in RT0), rows indexed by H¹ dofs of the
    // same physical vertices on both meshes ⇒ signed semantics ⇔
    // (M_neg v) == −(M_pos v).  (1ᵀM1 is analytically zero for this form,
    // hence the field functional.)  The positive-tri value is pinned
    // analytically: ∫∂φ_i/∂x = Area·(−1, +1, 0) = (−1/2, +1/2, 0).
    let integrators: [&dyn H1HDivIntegrator; 1] = [&MixedVectorGradientIntegrator { sigma: 1.0 }];

    let pos_h1 = H1Space::new(positive_tri(), 1);
    let pos_rt = HDivSpace::new(positive_tri(), 0);
    let neg_h1 = H1Space::new(inverted_tri(), 1);
    let neg_rt = HDivSpace::new(inverted_tri(), 0);
    let v_pos = pos_rt
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let v_neg = neg_rt
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let m_pos = assemble_h1_hdiv_mixed(&pos_h1, &pos_rt, &integrators[..], 4);
    let m_neg = assemble_h1_hdiv_mixed(&neg_h1, &neg_rt, &integrators[..], 4);
    let n = m_pos.nrows;
    let mut zp = vec![0.0_f64; n];
    let mut zn = vec![0.0_f64; n];
    m_pos.spmv(&v_pos, &mut zp);
    m_neg.spmv(&v_neg, &mut zn);
    println!("h1_hdiv tri M·e_x: positive {zp:?} inverted {zn:?} (MFEM signed)");
    for (i, (&a, &b)) in zp.iter().zip(zn.iter()).enumerate().take(3) {
        assert!(
            (b + a).abs() < 1e-12 * (1.0 + a.abs()),
            "h1_hdiv mixed must negate on inverted tri at row {i}: {a} vs {b}"
        );
    }
    assert!((zp[0] + 0.5).abs() < 1e-12, "analytic ∫∂φ0/∂x = −1/2, got {}", zp[0]);
    assert!((zp[1] - 0.5).abs() < 1e-12, "analytic ∫∂φ1/∂x = +1/2, got {}", zp[1]);
    assert!(zp[2].abs() < 1e-12, "analytic ∫∂φ2/∂x = 0, got {}", zp[2]);

    // Isoparametric branch (Quad4 always takes use_iso).
    let pos_h1 = H1Space::new(positive_quad(), 1);
    let pos_rt = HDivSpace::new(positive_quad(), 0);
    let neg_h1 = H1Space::new(inverted_quad(), 1);
    let neg_rt = HDivSpace::new(inverted_quad(), 0);
    let v_pos = pos_rt
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let v_neg = neg_rt
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let m_pos = assemble_h1_hdiv_mixed(&pos_h1, &pos_rt, &integrators[..], 4);
    let m_neg = assemble_h1_hdiv_mixed(&neg_h1, &neg_rt, &integrators[..], 4);
    let n = m_pos.nrows;
    let mut zp = vec![0.0_f64; n];
    let mut zn = vec![0.0_f64; n];
    m_pos.spmv(&v_pos, &mut zp);
    m_neg.spmv(&v_neg, &mut zn);
    let norm: f64 = zp.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("h1_hdiv quad ‖M·e_x‖ = {norm:.6e}");
    assert!(norm > 1e-12, "pin must be non-degenerate");
    for i in 0..n {
        assert!(
            (zn[i] + zp[i]).abs() < 1e-12 * (1.0 + zp[i].abs()),
            "h1_hdiv mixed must negate on inverted quad at row {i}: {} vs {}",
            zp[i], zn[i]
        );
    }
}

/// Positive-orientation invariance guard: on positively oriented meshes the
/// conversion must be bitwise inert (abs == signed for det > 0) — the ex8 /
/// ex4 / ex5 / d652 / d667 red lines depend on that.  Pins the analytic
/// h1_hdiv value ∫∂φ_i/∂x = (−1/2, +1/2, 0).
#[test]
fn d696_positive_orientation_inertness_guard() {
    let h1 = H1Space::new(positive_tri(), 1);
    let rt = HDivSpace::new(positive_tri(), 0);
    let v = rt
        .interpolate_vector(&|_x: &[f64]| vec![1.0, 0.0])
        .as_slice()
        .to_vec();
    let m = assemble_h1_hdiv_mixed(
        &h1,
        &rt,
        &[&MixedVectorGradientIntegrator { sigma: 1.0 }],
        4,
    );
    let mut z = vec![0.0_f64; 3];
    m.spmv(&v, &mut z);
    println!("positive tri h1_hdiv M·e_x = {z:?} (abs-inert analytic values)");
    assert!((z[0] + 0.5).abs() < 1e-12);
    assert!((z[1] - 0.5).abs() < 1e-12);
    assert!(z[2].abs() < 1e-12);
}
