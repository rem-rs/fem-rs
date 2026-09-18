//! D333 — the element factory's hex RT element must be MFEM's **default**
//! `RT_FECollection` variant.
//!
//! MFEM's `RT_FECollection(p, 3)` builds
//! `RT_HexahedronElement(p, GaussLobatto, GaussLegendre)`
//! (`fem/fe_coll.hpp`), and `fem_assembly::vector_assembler::vec_ref_elem`
//! pins exactly that for `(HDiv, Hex8)` since D245
//! (`HexRTk::new_gauss_legendre`).  `fem_element::vec_ref_elem` used to return
//! `HexRTk::new` — the `(GaussLobatto, IntegratedGLL)` pair that only the LOR
//! stack (`fem/lor/lor.cpp:317` `CheckBasisType`) wants — for **every** order.
//! D333 makes the two agree.
//!
//! The disagreement was latent (the factory is reached only as the order-0
//! quadrature provider at `postproc/grid_function.rs`'s P0 L² path, where the
//! two variants' 1-point open modes coincide), but it is exactly the kind of
//! drift that turns into a silent wrong answer the first time someone uses the
//! factory as a *basis*.
//!
//! This test pins the contract from the outside: the factory's element must be
//! basis-identical to `HexRTk::new_gauss_legendre(p)` and, for `p >= 1`,
//! *different* from `HexRTk::new(p)` (so a revert to the IntegratedGLL variant
//! fails here).
//!
//! The LOR side is deliberately untouched — `fem_assembly::lor_factory` names
//! `HexRTk::new` explicitly (its lib tests
//! `lor_rt_pcg_iterations_mesh_independent` / `lor_nd_pcg_iterations_mesh_independent`
//! are the guard).

use fem_element::lagrange::factory::{vec_ref_elem, VecFamily};
use fem_element::raviart_thomas::HexRTk;
use fem_element::{ElemType, VectorReferenceElement};

/// Basis values of `el` at a spread of reference points.
fn values(el: &dyn VectorReferenceElement) -> Vec<Vec<f64>> {
    let pts = [
        [0.0, 0.0, 0.0],
        [-0.5, 0.25, 0.75],
        [0.125, -0.875, 0.375],
        [1.0, -1.0, 1.0],
        [-0.9, 0.9, -0.1],
    ];
    let n = el.n_dofs();
    pts.iter()
        .map(|p| {
            let mut out = vec![0.0; n * 3];
            el.eval_basis_vec(p, &mut out);
            out
        })
        .collect()
}

#[test]
fn hex_rt_factory_matches_the_mfem_default_variant() {
    for p in 0u8..=4 {
        let got = vec_ref_elem(VecFamily::RaviartThomas, ElemType::Hex, p);
        let want = HexRTk::new_gauss_legendre(p as usize);
        assert_eq!(got.n_dofs(), want.n_dofs(), "p={p}: dof count");
        assert_eq!(
            values(got.as_ref()),
            values(&want),
            "p={p}: fem_element::vec_ref_elem must be HexRTk::new_gauss_legendre \
             (MFEM `RT_HexahedronElement(p, GaussLobatto, GaussLegendre)`)"
        );
    }
}

#[test]
fn hex_rt_factory_is_not_the_integrated_gll_variant() {
    // The two variants differ at *every* order: the IntegratedGLL open modes
    // carry a second `partial_open` factor, so `new`'s tensor modes are 4x (or
    // 16x, for two open axes) `new_gauss_legendre`'s at p = 0 already.
    for p in 0u8..=4 {
        let got = vec_ref_elem(VecFamily::RaviartThomas, ElemType::Hex, p);
        assert_ne!(
            values(got.as_ref()),
            values(&HexRTk::new(p as usize)),
            "p={p}: the factory must NOT return the LOR-only IntegratedGLL variant"
        );
    }
}

/// The D333 change is **behaviour-neutral for the one internal caller**:
/// `postproc/grid_function.rs`'s P0 L² path uses the factory only as a
/// *quadrature* provider, and `HexRTk::quadrature` is `hex_rule(order)`,
/// independent of the open-basis variant.
#[test]
fn hex_rt_quadrature_is_variant_independent() {
    for p in 0u8..=6 {
        let a = HexRTk::new(p as usize).quadrature(4);
        let b = HexRTk::new_gauss_legendre(p as usize).quadrature(4);
        assert_eq!(a.points, b.points, "p={p}: quadrature points");
        assert_eq!(a.weights, b.weights, "p={p}: quadrature weights");
    }
}

/// The LOR pin itself: `HexRTk::new` is still the IntegratedGLL variant, and
/// `lor_factory`'s comment says MFEM's `CheckBasisType` requires it.  If this
/// ever changes, `lor_rt_pcg_iterations_mesh_independent` must be revisited.
#[test]
fn hex_rtk_new_is_still_integrated_gll() {
    let igll = HexRTk::new(2);
    let gl = HexRTk::new_gauss_legendre(2);
    assert_ne!(values(&igll), values(&gl), "the two variants must stay distinct");
    assert_eq!(igll.n_dofs(), gl.n_dofs(), "same dof count/layout");
}
