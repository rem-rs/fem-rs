//! MFEM-nodal Raviart-Thomas RT2 element on the reference tetrahedron — 36 DOFs.
//!
//! Thin wrapper over the shared MFEM `RT_TetrahedronElement` port in
//! [`super::tet_rt1`]: nodal flux functionals at the Gauss-Legendre face nodes
//! (order `p = 2` per face direction) plus 12 interior component samples
//! (order `p − 1 = 1`), slot-ordered faces-first in MFEM `FaceVert` order
//! `(1,2,3), (0,3,2), (0,1,3), (0,2,1)` — matching `HDivSpace::build_3d_tet`.

use super::tet_rt1::{eval_nodal_tet_basis, eval_nodal_tet_div, nodal_tet_dof_coords};
use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT2 H(div) element on the reference tetrahedron — 36 DOFs,
/// order 2, MFEM nodal flux-dual semantics.
pub struct TetRT2;

impl VectorReferenceElement for TetRT2 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        (2 + 1) * (2 + 2) * (2 + 4) / 2 // 36
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        eval_nodal_tet_basis(2, xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        eval_nodal_tet_div(2, xi, div_vals);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        nodal_tet_dof_coords(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raviart_thomas::tet_rt1::TET_NK;

    /// RT2 nodal duality: D_m(Φ_i) = Φ_i(x_m)·nk_m = δ_mi.
    #[test]
    fn tet_rt2_nodal_duality() {
        let e = TetRT2;
        let n = e.n_dofs();
        let bop = crate::quadrature::gauss_legendre_01(3).0;
        let iop = crate::quadrature::gauss_legendre_01(2).0;
        let mut dual = vec![0.0f64; n * n];
        let mut phi = vec![0.0f64; n * 3];
        let mut m = 0usize;
        for (f, nk) in TET_NK.iter().enumerate() {
            for j in 0..=2 {
                for i in 0..=(2 - j) {
                    let w = bop[i] + bop[j] + bop[2 - i - j];
                    let b0 = bop[2 - i - j] / w;
                    let b1 = bop[i] / w;
                    let b2 = bop[j] / w;
                    let pt: [f64; 3] = match f {
                        0 => [b0, b1, b2],
                        1 => [0.0, b2, b1],
                        2 => [b1, 0.0, b2],
                        _ => [b2, b1, 0.0],
                    };
                    e.eval_basis_vec(&pt, &mut phi);
                    for ii in 0..n {
                        dual[m * n + ii] = phi[ii * 3] * nk[0]
                            + phi[ii * 3 + 1] * nk[1]
                            + phi[ii * 3 + 2] * nk[2];
                    }
                    m += 1;
                }
            }
        }
        for d in 0..2 {
            for j in 0..(2 - d) {
                for i in 0..(2 - d - j) {
                    let w = iop[i] + iop[j] + iop[d] + iop[1 - i - j - d];
                    let pt = [iop[i] / w, iop[j] / w, iop[d] / w];
                    for nk in &TET_NK[1..4] {
                        e.eval_basis_vec(&pt, &mut phi);
                        for ii in 0..n {
                            dual[m * n + ii] = phi[ii * 3] * nk[0]
                                + phi[ii * 3 + 1] * nk[1]
                                + phi[ii * 3 + 2] * nk[2];
                        }
                        m += 1;
                    }
                }
            }
        }
        assert_eq!(m, n);
        for (a, row) in dual.chunks(n).enumerate() {
            for (b, val) in row.iter().enumerate() {
                let exp = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (val - exp).abs() < 1e-9,
                    "D_{a}(phi_{b}) = {}, expected {exp}",
                    val
                );
            }
        }
    }

    #[test]
    fn tet_rt2_n_dofs() {
        assert_eq!(TetRT2.n_dofs(), 36);
    }

    #[test]
    fn tet_rt2_basis_finite() {
        let e = TetRT2;
        let mut v = vec![0.0; 36 * 3];
        for xi in &[
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.25, 0.25, 0.25],
            vec![0.0, 0.5, 0.5],
        ] {
            e.eval_basis_vec(xi, &mut v);
            for val in &v {
                assert!(val.is_finite(), "non-finite at {xi:?}: {val}");
            }
        }
    }

    #[test]
    fn tet_rt2_div_finite() {
        let e = TetRT2;
        let mut div = vec![0.0; 36];
        let qr = e.quadrature(3);
        for xi in &qr.points {
            e.eval_div(xi, &mut div);
            for d in &div {
                assert!(d.is_finite());
            }
        }
    }
}
