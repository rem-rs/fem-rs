//! Raviart-Thomas RT1 element on the reference hexahedron `[0,1]^3`.
//!
//! # Space: RT₁ = Q_{2,1,1} × Q_{1,2,1} × Q_{1,1,2}
//! dim = 3×2×2 + 2×3×2 + 2×2×3 = 12 + 12 + 12 = 36 DOFs.
//!
//! # DOFs (36 total)
//! - 4 normal-flux moments per face × 6 faces = 24 face DOFs
//! - 12 interior DOFs
//!
//! Delegates to the generic `HexRTk::new(1)` implementation.

use crate::gll_basis::{gl_nodes_01, gll_nodes_01};
use crate::quadrature::hex_rule;
use crate::raviart_thomas::hex_rtk::{free_axes, HEX_RT_FACES};
use crate::raviart_thomas::HexRTk;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// MFEM `RT_HexahedronElement(k)` nodal dof table — reference sample points
/// and (unnormalised) reference normals on the `[0,1]³` hex, in the slot
/// order the H(div) space uses (faces in `HEX_FACES` order with the
/// `HEX_RT_FACES` frame reversals, `(k+1)²` Gauss points each, then the
/// interior closed×open×open grid: x-normal, y-normal, z-normal blocks with
/// the `HexRTk` orientation-flip sign folded into the normal).
///
/// Published (D494) as the order-generic counterpart of
/// `tet_rt1::mfem_nodal_dofs` so the assembly crate's MFEM-exact
/// prolongation can consume the order-1 rows without duplicating the
/// enumeration rules.  D721: the points are MFEM's `[0,1]³` node values (the
/// open Gauss-Legendre / closed Gauss-Lobatto points of
/// `Poly_1D::OpenPoints` / `Poly_1D::ClosedPoints`).
pub fn mfem_hex_nodal_dofs(k: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let gl = gl_nodes_01(k + 1);
    let m = k + 1;
    let axis = |i: usize| match i {
        0 => [1.0, 0.0, 0.0],
        1 => [0.0, 1.0, 0.0],
        _ => [0.0, 0.0, 1.0],
    };
    let mut pts = Vec::new();
    let mut nks = Vec::new();
    for &(nc, at_max, _s, f1, f2) in &HEX_RT_FACES {
        let cnorm = if at_max { 1.0 } else { 0.0 };
        let (a1, a2) = free_axes(nc);
        let mut nk = [0.0_f64; 3];
        nk[nc] = if at_max { 1.0 } else { -1.0 };
        for j in 0..m {
            let q = if f2 { m - 1 - j } else { j };
            for i in 0..m {
                let p = if f1 { m - 1 - i } else { i };
                let mut xi = [0.0_f64; 3];
                xi[nc] = cnorm;
                xi[a1] = gl[p];
                xi[a2] = gl[q];
                pts.push(xi);
                nks.push(nk);
            }
        }
    }
    if k >= 1 {
        let cp = gll_nodes_01(k + 1);
        // `HexRTk` bakes MFEM's reference orientation flips into the interior
        // basis (closed index <= k/2 is negative), so the dual flux samples
        // flip their normal with them.
        for l in 0..m {
            for j in 0..m {
                for i in 1..=k {
                    let s = if i <= k / 2 { -1.0 } else { 1.0 };
                    let mut nk = axis(0);
                    for d in 0..3 {
                        nk[d] *= s;
                    }
                    pts.push([cp[i], gl[j], gl[l]]);
                    nks.push(nk);
                }
            }
        }
        for l in 0..m {
            for j in 1..=k {
                let s = if j <= k / 2 { -1.0 } else { 1.0 };
                for i in 0..m {
                    let mut nk = axis(1);
                    for d in 0..3 {
                        nk[d] *= s;
                    }
                    pts.push([gl[i], cp[j], gl[l]]);
                    nks.push(nk);
                }
            }
        }
        for l in 1..=k {
            let s = if l <= k / 2 { -1.0 } else { 1.0 };
            for j in 0..m {
                for i in 0..m {
                    let mut nk = axis(2);
                    for d in 0..3 {
                        nk[d] *= s;
                    }
                    pts.push([gl[i], gl[j], cp[l]]);
                    nks.push(nk);
                }
            }
        }
    }
    (pts, nks)
}

/// Raviart-Thomas RT1 H(div) element on the reference hexahedron — 36 DOFs, order 1.
pub struct HexRT1;

impl VectorReferenceElement for HexRT1 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        36
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        HexRTk::new(1).eval_basis_vec(xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        HexRTk::new(1).eval_div(xi, div_vals);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        (0..36).map(|_| vec![0.0, 0.0, 0.0]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_rt1_n_dofs() {
        assert_eq!(HexRT1.n_dofs(), 36);
    }

    #[test]
    fn mfem_hex_nodal_dofs_table_is_unisolvent() {
        // One row per basis dof; every row must sample some basis function's
        // normal trace nontrivially (nodal sanity for the prolongation's
        // W⁻¹ dual map).  The rows must pair with `HexRTk`, whose slot
        // layout (HEX_RT_FACES frame + interior blocks) the table mirrors.
        for k in 0..=2usize {
            let (pts, nks) = mfem_hex_nodal_dofs(k);
            let basis = HexRTk::new(k);
            assert_eq!(pts.len(), basis.n_dofs());
            assert_eq!(nks.len(), basis.n_dofs());
            let n = basis.n_dofs();
            let mut phi = vec![0.0_f64; n * 3];
            for (i, (p, nk)) in pts.iter().zip(nks.iter()).enumerate() {
                basis.eval_basis_vec(p, &mut phi);
                let mut best = 0.0_f64;
                for j in 0..n {
                    let d = phi[j * 3] * nk[0] + phi[j * 3 + 1] * nk[1] + phi[j * 3 + 2] * nk[2];
                    best = best.max(d.abs());
                }
                assert!(best > 1e-6, "k={k}: nodal row {i} annihilates the basis");
            }
        }
    }

    #[test]
    fn hex_rt1_basis_finite() {
        let elem = HexRT1;
        let mut v = vec![0.0; 36 * 3];
        for xi in &[vec![0., 0., 0.], vec![1., -1., 0.5], vec![-0.5, 0.5, 1.]] {
            elem.eval_basis_vec(xi, &mut v);
            for &val in &v {
                assert!(val.is_finite(), "non-finite at {xi:?}: {val}");
            }
        }
    }

    #[test]
    fn hex_rt1_div_finite() {
        let elem = HexRT1;
        let mut div = vec![0.0; 36];
        let qr = elem.quadrature(3);
        for xi in &qr.points {
            elem.eval_div(xi, &mut div);
            for &d in &div {
                assert!(d.is_finite());
            }
        }
    }
}
