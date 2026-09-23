//! D679 red pins: the `det.abs()` → signed-det conversions in
//! `assembler.rs` / `hdiv_error.rs` / `complex.rs` must carry the **MFEM
//! signed `Trans.Weight()` semantics** (MFEM 4.10 probe, tmp/d667: inverted
//! tet `GetElementVolume` = −1/6, "NOT FIXED").  Every pin below is
//! `abs == signed` on positively oriented elements and differs ONLY on
//! inverted (det < 0) ones — so all positive-det byte-pins (ex8 DPG, ex4/ex5,
//! d652/d667 parity) are untouched by construction.
//!
//! | pin | site family | MFEM signed expectation | old `abs()` value |
//! |-----|-------------|-------------------------|-------------------|
//! | inverted tri mass sum       | affine branch phys_weight | −1/2 | +1/2 |
//! | inverted tri diffusion eᵀKe | affine branch weight      | −1   | +1   |
//! | inverted tri linear sum     | linear affine branch      | −1/2 | +1/2 |
//! | inverted quad mass sum      | isoparametric phys_weight | −1   | +1   |
//! | inverted quad linear sum    | linear isoparametric      | −1   | +1   |
//! | two-quad L2 error of const  | hdiv_error scalar site    | 0    | √2   |
//!
//! Run:
//!   cargo test -p fem-assembly --test d679_signed_det_pins -- --nocapture

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::constraints::boundary_dofs;
use fem_space::{FESpace, H1Space};

/// One positively (CCW) and one negatively (CW) oriented unit triangle /
/// square, built directly through `Mesh::uniform` (no reader-side orientation
/// fixups — the d37 pattern).
fn inverted_tri() -> Mesh<2> {
    // v0=(0,0) v1=(1,0) v2=(0,1); CW conn [0,2,1] → det = −1.
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

fn inverted_quad() -> Mesh<2> {
    // v0=(0,0) v1=(1,0) v2=(1,1) v3=(0,1); CW conn [0,3,2,1] → det = −1.
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

#[test]
fn d679_inverted_element_signed_weight_pins() {
    // ── affine branch (tri, sites 1127/1145 + linear 1279) ──
    let mesh = inverted_tri();
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
    let one = space.interpolate(&|_x: &[f64]| 1.0).as_slice().to_vec();
    let mut me = vec![0.0f64; n];
    m.spmv(&one, &mut me);
    let mass_sum: f64 = one.iter().zip(me.iter()).map(|(a, b)| a * b).sum();
    println!("inverted tri: 1^T M 1 = {mass_sum:.12e} (MFEM signed −0.5)");
    assert!((mass_sum + 0.5).abs() < 1e-12, "mass must follow signed det: {mass_sum}");

    let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    // u_h = x  ⇒  ∫ |∇u|² dV with signed det = −(1/2)·1 = −0.5.
    let ux = space.interpolate(&|x: &[f64]| x[0]).as_slice().to_vec();
    let mut ke = vec![0.0f64; n];
    k.spmv(&ux, &mut ke);
    let kq: f64 = ux.iter().zip(ke.iter()).map(|(a, b)| a * b).sum();
    println!("inverted tri: u^T K u (u=x) = {kq:.12e} (MFEM signed −0.5)");
    assert!((kq + 0.5).abs() < 1e-12, "diffusion must follow signed det: {kq}");

    let b = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_x: &[f64]| 1.0)], 3);
    let lin_sum: f64 = b.as_slice().iter().sum();
    println!("inverted tri: linear sum = {lin_sum:.12e} (MFEM signed −0.5)");
    assert!((lin_sum + 0.5).abs() < 1e-12, "linear form must follow signed det: {lin_sum}");

    // ── isoparametric branch (quad, sites 1169/1181 + linear 1292) ──
    let mesh = inverted_quad();
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
    let one = space.interpolate(&|_x: &[f64]| 1.0).as_slice().to_vec();
    let mut me = vec![0.0f64; n];
    m.spmv(&one, &mut me);
    let mass_sum: f64 = one.iter().zip(me.iter()).map(|(a, b)| a * b).sum();
    println!("inverted quad: 1^T M 1 = {mass_sum:.12e} (MFEM signed −1)");
    assert!((mass_sum + 1.0).abs() < 1e-12, "mass must follow signed det: {mass_sum}");

    let b = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_x: &[f64]| 1.0)], 3);
    let lin_sum: f64 = b.as_slice().iter().sum();
    println!("inverted quad: linear sum = {lin_sum:.12e} (MFEM signed −1)");
    assert!((lin_sum + 1.0).abs() < 1e-12, "linear form must follow signed det: {lin_sum}");

    // ── hdiv_error scalar L2 site: positive + inverted pair cancels ──
    // Two-quad mesh [CCW, CW]: u_h ≡ 1 against exact 0 gives
    // err² = +1 + (−1) = 0 under signed weights (the d37 inverted-pair
    // signature), vs √2 under the old abs clamp.
    let coords = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0];
    let conn = vec![0, 1, 2, 3, 2, 5, 4, 1]; // second quad CW (det = −1)
    let pair = Mesh::<2>::uniform(coords, conn, vec![1, 1], ElementType::Quad4, vec![], vec![], ElementType::Line2);
    let space2 = H1Space::new(pair.clone(), 1);
    let dm = space2.dof_manager();
    let all_tags: Vec<i32> = pair.unique_boundary_tags();
    let _ = boundary_dofs(&pair, dm, &all_tags);
    let u1 = space2.interpolate(&|_x: &[f64]| 1.0);
    let (err, _det_probe) = {
        // direct err² via the same jacobian_and_point path as
        // hdiv_error::scalar_l2_error_impl — read it back through the public
        // API by integrating against exact 0.
        (fem_assembly::hdiv_error::compute_l2_error_scalar(&space2, u1.as_slice(), &|_| 0.0), 0.0f64)
    };
    println!("pair [CCW, CW]: L2 error of const 1 vs 0 = {err:.12e} (signed expectation 0)");
    assert!(err < 1e-12, "signed det must cancel the pair: {err}");
}
