//! Raviart-Thomas RT1 element on the reference quadrilateral `[0,1]^2`.
//!
//! A 1:1 port of MFEM's `RT_QuadrilateralElement` with `p = 1`
//! (fem/fe/fe_rt.cpp), the element behind `RT_FECollection(1, 2)` — the one
//! ex40 (and every current MFEM H(div) example) actually uses.
//!
//! # Tensor-product structure
//!
//! 12 DOFs = 8 edge normal traces + 4 interior moments.  The basis is a
//! tensor product of 1-D bases on `[0,1]`:
//!
//! - closed basis `c_i` (Gauss-Lobatto nodes `{0, 1/2, 1}`, `Poly_1D`):
//!   - `c0(t) = 2t² − 3t + 1`, `c1(t) = −4t² + 4t`, `c2(t) = 2t² − t`
//! - open basis `o_j` (Gauss-Legendre nodes `{a, b}`, `a = (1−1/√3)/2`,
//!   `b = 1−a`, `d = b−a = 1/√3`):
//!   - `o0(t) = (b−t)/d`, `o1(t) = (t−a)/d`
//!
//! x-components `s·c_i(x)·o_j(y)`, y-components `s·o_i(x)·c_j(y)`, with the
//! signs and DOF permutation taken from MFEM's `dof_map` (p = 1):
//! `[-8, 8, 2, -7, -10, 3, -1, -2, -11, 11, 5, 4]`.

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Open (Gauss-Legendre) nodes and width for p = 1.
const A: f64 = 0.21132486540518713; // (1 - 1/√3)/2
const B: f64 = 0.7886751345948129; // (1 + 1/√3)/2
const D: f64 = 0.5773502691896258; // b - a = 1/√3

#[inline]
fn c0(t: f64) -> f64 {
    2.0 * t * t - 3.0 * t + 1.0
}
#[inline]
fn c1(t: f64) -> f64 {
    -4.0 * t * t + 4.0 * t
}
#[inline]
fn c2(t: f64) -> f64 {
    2.0 * t * t - t
}
#[inline]
fn dc0(t: f64) -> f64 {
    4.0 * t - 3.0
}
#[inline]
fn dc1(t: f64) -> f64 {
    4.0 - 8.0 * t
}
#[inline]
fn dc2(t: f64) -> f64 {
    4.0 * t - 1.0
}
#[inline]
fn o0(t: f64) -> f64 {
    (B - t) / D
}
#[inline]
fn o1(t: f64) -> f64 {
    (t - A) / D
}

/// Quadrilateral RT1 element — 12 DOFs (MFEM `RT_QuadrilateralElement(1)`).
pub struct QuadRT1;

impl VectorReferenceElement for QuadRT1 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        12
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        // x-components (from dof_map): s·c_i(x)·o_j(y)
        values[0] = 0.0;
        values[1] = -o0(x) * c0(y);      // dof0
        values[0 + 2] = 0.0;
        values[1 + 2] = -o1(x) * c0(y);  // dof1
        values[4] = c2(x) * o0(y);       // dof2
        values[5] = 0.0;
        values[6] = c2(x) * o1(y);       // dof3
        values[7] = 0.0;
        values[8] = 0.0;
        values[9] = o1(x) * c2(y);       // dof4
        values[10] = 0.0;
        values[11] = o0(x) * c2(y);      // dof5
        values[12] = -c0(x) * o1(y);     // dof6
        values[13] = 0.0;
        values[14] = -c0(x) * o0(y);     // dof7
        values[15] = 0.0;
        values[16] = c1(x) * o0(y);      // dof8
        values[17] = 0.0;
        values[18] = -c1(x) * o1(y);     // dof9
        values[19] = 0.0;
        values[20] = 0.0;
        values[21] = -o0(x) * c1(y);     // dof10
        values[22] = 0.0;
        values[23] = o1(x) * c1(y);      // dof11
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        div_vals[0] = -o0(x) * dc0(y);
        div_vals[1] = -o1(x) * dc0(y);
        div_vals[2] = dc2(x) * o0(y);
        div_vals[3] = dc2(x) * o1(y);
        div_vals[4] = o1(x) * dc2(y);
        div_vals[5] = o0(x) * dc2(y);
        div_vals[6] = -dc0(x) * o1(y);
        div_vals[7] = -dc0(x) * o0(y);
        div_vals[8] = dc1(x) * o0(y);
        div_vals[9] = -dc1(x) * o1(y);
        div_vals[10] = -o0(x) * dc1(y);
        div_vals[11] = o1(x) * dc1(y);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    /// MFEM `RT_QuadrilateralElement(1)` node coordinates (edge traces at
    /// the two Gauss points, interiors at the tensor grid).
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![A, 0.0],        // 0 bottom (x = a)
            vec![B, 0.0],        // 1 bottom (x = b)
            vec![1.0, A],        // 2 right  (y = a)
            vec![1.0, B],        // 3 right  (y = b)
            vec![B, 1.0],        // 4 top    (x = b)
            vec![A, 1.0],        // 5 top    (x = a)
            vec![0.0, B],        // 6 left   (y = b)
            vec![0.0, A],        // 7 left   (y = a)
            vec![0.5, A],        // 8 interior
            vec![0.5, B],        // 9 interior
            vec![A, 0.5],        // 10 interior
            vec![B, 0.5],        // 11 interior
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_rt1_n_dofs() {
        assert_eq!(QuadRT1.n_dofs(), 12);
    }

    #[test]
    fn quad_rt1_values_finite() {
        let elem = QuadRT1;
        let mut vals = vec![0.0; 24];
        let q = elem.quadrature(4);
        for pt in &q.points {
            elem.eval_basis_vec(pt, &mut vals);
            for (i, &v) in vals.iter().enumerate() {
                assert!(v.is_finite(), "value[{i}] = {v}");
            }
        }
    }

    #[test]
    fn quad_rt1_nodal_basis() {
        // MFEM RT_QuadrilateralElement edge DOFs are nodal normal traces at
        // the node coordinates above: DOF_j(Φ_i) = δ_ij for j = 0..7 (the
        // interior DOFs 8..11 are integral moments, not nodal).
        let elem = QuadRT1;
        let coords = elem.dof_coords();
        let norms = [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let mut vals = vec![0.0; 24];
        for j in 0..8 {
            elem.eval_basis_vec(&coords[j], &mut vals);
            let edge = j / 2;
            for i in 0..12 {
                let tr = vals[i * 2] * norms[edge][0] + vals[i * 2 + 1] * norms[edge][1];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (tr - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {tr}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rt1_mass_diagonal_matches_mfem() {
        // MFEM single-unit-square RT1 mass: edge diag = 1/15, interior = 4/15.
        let elem = QuadRT1;
        let qr = elem.quadrature(5);
        let mut vals = vec![0.0; 24];
        let mut diag = vec![0.0; 12];
        for (q, pt) in qr.points.iter().enumerate() {
            elem.eval_basis_vec(pt, &mut vals);
            let w = qr.weights[q];
            for i in 0..12 {
                diag[i] += w * (vals[i * 2] * vals[i * 2] + vals[i * 2 + 1] * vals[i * 2 + 1]);
            }
        }
        for i in 0..8 {
            assert!((diag[i] - 1.0 / 15.0).abs() < 1e-13, "edge diag[{i}] = {}", diag[i]);
        }
        for i in 8..12 {
            assert!((diag[i] - 4.0 / 15.0).abs() < 1e-13, "interior diag[{i}] = {}", diag[i]);
        }
    }
}
