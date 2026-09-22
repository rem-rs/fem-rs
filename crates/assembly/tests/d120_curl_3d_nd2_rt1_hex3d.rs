//! D120: `DiscreteLinearOperator::curl_3d` for `H(curl) ND2 → H(div) RT1` on
//! 3-D hexahedra.
//!
//! Before this entry point the order-2 discrete curl was tetrahedron-only: on
//! a hex mesh the ND2 face-dof rows needed a tet face anchor, so the assembly
//! panicked with `"tet face must have an interpolation anchor"`.  That blocked
//! the joule / tesla / volta miniapps at `-o 2` on hex meshes
//! (`curl->Mult(E, dB)`, joule_solver.cpp:585).
//!
//! # MFEM parity (the acceptance gate)
//!
//! MFEM assembles the discrete curl with `DiscreteLinearOperator` +
//! `CurlInterpolator` (`fem/bilininteg.hpp:4130`).  Because `curl` maps the
//! order-2 hexahedral Nédélec space *into* the order-1 Raviart-Thomas space
//! (`curl Q_{1,2,2}×Q_{2,1,2}×Q_{2,2,1} ⊂ Q_{2,1,1}×Q_{1,2,1}×Q_{1,1,2}`), the
//! assembled matrix is the RT1 interpolant of the curl and is *reference-element*
//! only: the covariant pullback pairs `(J⁻ᵀφ̂)·(Jτ̂) = φ̂·τ̂` and
//! `(1/detJ)·J curl φ̂ · adj(J) n̂ = curl φ̂ · n̂`, so the local matrix does not
//! depend on the physical geometry at all.  The probe
//! `tmp/d120/d120_curl_probe.cpp` (MFEM 4.10, `$HOME/work/d120/`) confirms this
//! empirically: the assembled matrices on a straight hex (`hex1f.mesh`) and a
//! strongly warped hex (`hexw.mesh`) agree to 15 digits.
//!
//! The entry-for-entry MFEM parity gate lives in the sibling target
//! `d120_mfem_curl_hex1_triplets.rs` (verbatim probe table `CURL 36 54 216`
//! from `tmp/d120/curl_hex1f_o2.txt`); this file pins the operator's
//! semantics on multi-element hex meshes.

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

fn matvec(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows];
    a.spmv(x, &mut y);
    y
}

/// The de Rham semantics: for `F = (y², z², x²)` the curl is
/// `curl F = (−2z, −2x, −2y)`, which lies *exactly* in the hex RT1 space, so
/// `C · Π_ND2(F)` must equal the RT1 interpolant of `curl F` — a completely
/// independent code path (`HDivSpace::interpolate_vector`).
#[test]
fn hex_nd2_rt1_commutes_with_interpolation() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let rt = HDivSpace::new(mesh.clone(), 1);

    let c = DiscreteLinearOperator::curl_3d(&nd, &rt)
        .expect("hex curl_3d(ND2→RT1) must assemble (D120)");

    let nd_f = nd.interpolate_vector(&|x| vec![x[1] * x[1], x[2] * x[2], x[0] * x[0]]);
    let rt_curl = rt.interpolate_vector(&|x| vec![-2.0 * x[2], -2.0 * x[0], -2.0 * x[1]]);

    let c_f = matvec(&c, nd_f.as_slice());
    let dev: f64 = c_f
        .iter()
        .zip(rt_curl.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dev < 1e-10,
        "C·Π_ND2(F) vs Π_RT1(curl F): max dev {dev:.3e}"
    );
}

/// The assembled operator must cover multi-element meshes (shared RT1 faces,
/// slot reversal, both spaces' sign tables).  Geometry-independence of the
/// reference-only local matrix is pinned on the MFEM side:
/// `tmp/d120/curl_hexwf_o2.txt` (warped hex) ≡ `curl_hex1f_o2.txt` (straight)
/// to 15 digits.
#[test]
fn hex_nd2_rt1_assembles_on_multi_hex_meshes() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let rt = HDivSpace::new(mesh.clone(), 1);
    let c = DiscreteLinearOperator::curl_3d(&nd, &rt)
        .expect("hex curl_3d(ND2→RT1) must assemble (D120)");
    assert_eq!(c.nrows, rt.n_dofs());
    assert_eq!(c.ncols, nd.n_dofs());
    assert!(c.nnz() > 0);
}

/// `curl(grad p) = 0`: with the D110 P2→ND2 hex gradient in place, the
/// commuting diagram `curl_3d(ND2 → RT1) · gradient(P2 → ND2) ≡ 0` must close
/// at order 2 on hexahedra (the d110 test could only pin this at order 1).
#[test]
fn hex_curl_of_gradient_is_zero_at_order_2() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let rt = HDivSpace::new(mesh.clone(), 1);

    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("P2→ND2 hex gradient");
    let c = DiscreteLinearOperator::curl_3d(&nd, &rt)
        .expect("hex curl_3d(ND2→RT1) must assemble (D120)");
    assert_eq!(c.ncols, g.nrows);

    let mut max_dev = 0.0_f64;
    for j in 0..h1.n_dofs() {
        let mut e = vec![0.0; h1.n_dofs()];
        e[j] = 1.0;
        let gcol = matvec(&g, &e);
        let cg = matvec(&c, &gcol);
        for v in cg {
            max_dev = max_dev.max(v.abs());
        }
    }
    assert!(
        max_dev < 1e-10,
        "order-2 curl(grad) must vanish on hexes: max |C·G| = {max_dev:.3e}"
    );
}
