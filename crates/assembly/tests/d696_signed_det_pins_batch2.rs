//! D696 batch-2 red pins: the `det.abs()` → signed-det conversions in the
//! DPG family (`dpg_framework.rs` element_volume + Poisson2D/3D forms,
//! `dpg_2d.rs`, `dpg_3d.rs`, `dpg_stokes.rs`, `dpg_elasticity.rs`,
//! `dpg_maxwell.rs`) must carry the **MFEM signed `Trans.Weight()`
//! semantics** (D679 precedent: inverted tet `GetElementVolume` = −1/6).
//! All sites are simplex hand-rolled weak forms whose quadrature weight is
//! MFEM's `ip.weight * Trans.Weight()`.
//!
//! Analytic pin: the P3 test Gram block `M_V` is the H¹ inner product, and
//! the P3 partition of unity gives `Σφ = 1`, `Σ∇φ = 0`, so
//! `1ᵀM_V1 = ∫1 dV = ±vol` — **+1/2** on the positively oriented unit tri,
//! **−1/2** on the inverted one (D679 value), and ±1/6 on the tets.
//! `abs()` weights would keep both positive.
//!
//! Run:
//!   cargo test -p fem-assembly --test d696_signed_det_pins_batch2 -- --nocapture

use fem_assembly::dpg::dpg_framework::{Poisson2DForm, Poisson3DForm};
use fem_assembly::dpg::BilinearForm;
use fem_mesh::{element_type::ElementType, Mesh};

fn positive_tri() -> Mesh<2> {
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

fn positive_tet() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

fn inverted_tet() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 2, 1, 3],
        vec![1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `1ᵀM_V1` over the (square) test Gram block.
fn gram_sum(bm_and_mv: &(Vec<f64>, Vec<f64>), n_test: usize) -> f64 {
    let (_, mv) = bm_and_mv;
    let mut ones = vec![1.0_f64; n_test];
    let mut y = vec![0.0_f64; n_test];
    for r in 0..n_test {
        for c in 0..n_test {
            y[r] += mv[r * n_test + c] * ones[c];
        }
    }
    ones.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[test]
fn d696_poisson2d_gram_signed_pin() {
    // TriP3 test Gram (P3 has 10 dofs): 1ᵀM_V1 = ∫1 = ±1/2.
    let pos = gram_sum(&Poisson2DForm.eval(&positive_tri(), 0, 10, 3), 10);
    let neg = gram_sum(&Poisson2DForm.eval(&inverted_tri(), 0, 10, 3), 10);
    println!("poisson2d 1ᵀM_V1: positive {pos:.12e} inverted {neg:.12e} (±0.5)");
    assert!((pos - 0.5).abs() < 1e-12, "positive tri Gram sum must be +1/2: {pos}");
    assert!((neg + 0.5).abs() < 1e-12, "inverted tri Gram sum must be −1/2 (signed): {neg}");
}

#[test]
fn d696_poisson3d_gram_signed_pin() {
    // TetP3 test Gram (P3 has 20 dofs): 1ᵀM_V1 = ∫1 = ±1/6.
    let pos = gram_sum(&Poisson3DForm.eval(&positive_tet(), 0, 20, 4), 20);
    let neg = gram_sum(&Poisson3DForm.eval(&inverted_tet(), 0, 20, 4), 20);
    println!("poisson3d 1ᵀM_V1: positive {pos:.12e} inverted {neg:.12e} (±1/6)");
    assert!((pos - 1.0 / 6.0).abs() < 1e-12, "positive tet Gram sum must be +1/6: {pos}");
    assert!(
        (neg + 1.0 / 6.0).abs() < 1e-12,
        "inverted tet Gram sum must be −1/6 (signed, D679 GetElementVolume): {neg}"
    );
}
