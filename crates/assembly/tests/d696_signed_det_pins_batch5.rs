//! D696 batch-5 red pins: the D727/D728 det-sign adjudications on the tail
//! stations (`dgmassinv.rs`, `complex.rs` — the round-69 D-lane lot).
//!
//! | pin | station | MFEM counterpart / verdict |
//! |-----|---------|----------------------------|
//! | inverted tri DG mass block  | `dgmassinv.rs` assemble | `BilinearForm` + `MassIntegrator` (dgmassinv.cpp:67-68,114), `ip.weight·Ttr.Weight()` **signed** → block = exact negation of the positive twin (abs() would re-positive it) |
//! | inverted tri complex Helmholtz block | `complex.rs` `NativeComplexAssembler::assemble` | re/im `BilinearForm` integrators, `ip.weight·Ttr.Weight()` **signed** → block = exact negation of the positive twin |
//! | two-quad L2 error of const  | `complex.rs` `compute_l2_error` | MFEM takes `fabs` of every ELEMENT sum (gridfunc.cpp:3449-3453, complex_fem.cpp:296-298) ⇒ on affine elements **abs is the MFEM semantics in any orientation** — retained; plain signed would cancel to 0 |
//!
//! The guard-class station `hdiv_error.rs` `|det| > 1e-80` (D679 verdict,
//! annotated in-source) and the cut-cell metric
//! `cut/moment_fitting.rs::det_j` (no MFEM core counterpart; Gram
//! determinant naturally non-negative — annotated in-source) retain abs by
//! adjudication and carry no pin: no affine fixture separates them from
//! their MFEM semantics.
//!
//! Run:
//!   cargo test -p fem-assembly --test d696_signed_det_pins_batch5 -- --nocapture

use fem_assembly::complex::{ComplexGridFunction, HelmholtzIntegrator, NativeComplexAssembler};
use fem_assembly::dgmassinv::DGMassInverse;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::{FESpace, H1Space, L2Space};

/// One positively (CCW) oriented unit right triangle.
fn pos_tri() -> Mesh<2> {
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

/// The same triangle with a node swap (CW conn) → det = −1.
fn inv_tri() -> Mesh<2> {
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

/// One CCW unit square plus one CW unit square sharing an edge.
fn pos_inv_quad_mesh() -> Mesh<2> {
    // v0..v3: square A [0,1,2,3] (CCW, det = +1);
    // v1,v2,v4,v5: square B [1,2,5,4] (CW, det = −1).
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
        vec![0, 1, 2, 3, 1, 2, 5, 4],
        vec![1, 1],
        ElementType::Quad4,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

#[test]
fn d696_b5_dgmassinv_inverted_element_signed_pin() {
    let pos = L2Space::new(pos_tri(), 1);
    let inv = L2Space::new(inv_tri(), 1);
    let m_pos = DGMassInverse::new(&pos, 1.0_f64, 3);
    let m_inv = DGMassInverse::new(&inv, 1.0_f64, 3);

    let b_pos = m_pos.element_mass(0);
    let b_inv = m_inv.element_mass(0);
    let d_pos = m_pos.element_diagonal();
    let d_inv = m_inv.element_diagonal();

    // The inverted block carries the SIGNED area: every entry is the
    // negation of the positive twin's (up to determinant-path rounding —
    // nalgebra's LU determinant is not a pure sign flip; an abs() regression
    // shows up as a SAME-sign entry, O(1) above the tolerance).  Off-diagonal
    // entries may be negative in either orientation (Gram matrix of the L2
    // basis); only the negation relation is universal.
    for k in 0..b_pos.len() {
        assert!(
            (b_inv[k] + b_pos[k]).abs() <= 1e-14 * b_pos[k].abs().max(1.0),
            "inverted DG mass block entry {k}: {} != −{} (abs() regression?)",
            b_inv[k], b_pos[k]
        );
    }
    // Jacobi diagonal follows the same sign convention (MFEM
    // `AssembleDiagonal` + `Reciprocal`).
    for (i, (&dp, &dn)) in d_pos.iter().zip(d_inv.iter()).enumerate() {
        assert!(
            (dn + dp).abs() <= 1e-14 * dp.abs().max(1.0) && dp > 0.0 && dn < 0.0,
            "inverted DG mass diag [{i}]: {dn} vs positive {dp}"
        );
    }
}

#[test]
fn d696_b5_native_complex_assembler_inverted_element_signed_pin() {
    let integ = HelmholtzIntegrator {
        kappa_re: 1.0,
        kappa_im: 0.5,
        rho: 1.0,
        omega: 2.0,
    };

    let space_pos = H1Space::new(pos_tri(), 1);
    let space_inv = H1Space::new(inv_tri(), 1);
    let sys_pos = NativeComplexAssembler::assemble(&space_pos, &[&integ], 3);
    let sys_inv = NativeComplexAssembler::assemble(&space_inv, &[&integ], 3);

    assert_eq!(sys_pos.mat.re_vals.len(), sys_inv.mat.re_vals.len());
    assert_eq!(sys_pos.mat.im_vals.len(), sys_inv.mat.im_vals.len());
    // Negation up to determinant-path rounding (an abs() regression shows up
    // as a SAME-sign block, O(1) above the tolerance).  Entries that are
    // exactly zero (grad-couplings that vanish for P1) stay zero in both
    // orientations.
    let check = |tag: &str, k: usize, p: f64, n: f64| {
        assert!(
            (n + p).abs() <= 1e-14 * p.abs().max(1.0),
            "inverted complex block ({tag}) entry {k}: {n} vs positive {p} (abs() regression?)"
        );
        assert!(
            p == 0.0 || (p > 0.0) == (n < 0.0),
            "sign convention broken at ({tag}, {k}): {n} vs {p}"
        );
    };
    for (k, (&p, &n)) in sys_pos.mat.re_vals.iter().zip(sys_inv.mat.re_vals.iter()).enumerate() {
        check("re", k, p, n);
    }
    for (k, (&p, &n)) in sys_pos.mat.im_vals.iter().zip(sys_inv.mat.im_vals.iter()).enumerate() {
        check("im", k, p, n);
    }
}

#[test]
fn d696_b5_compute_l2_error_retains_abs_on_inverted_quad() {
    // uh = 0, exact = 1 over {+quad, −quad}: each unit-square element
    // contributes |−1| = +1 to the MFEM per-element-fabs accumulation, total
    // ∫1² = 2 over the two squares ⇒ error √2 (gridfunc.cpp:3449-3453).
    // Plain signed weights would cancel the inverted element and give 0 —
    // the D727/D728 retention signature.
    let mesh = pos_inv_quad_mesh();
    let space = H1Space::new(mesh, 1);
    let uh = ComplexGridFunction {
        u_re: vec![0.0; space.n_dofs()],
        u_im: vec![0.0; space.n_dofs()],
    };
    let (er, ei) = uh.compute_l2_error(&|_p| 1.0, &|_p| 0.0, 3, &space);
    assert_eq!(er.to_bits(), 2.0_f64.sqrt().to_bits(), "L2 error retention pin (er)");
    assert_eq!(ei.to_bits(), 0.0_f64.to_bits(), "L2 error retention pin (ei)");
}
