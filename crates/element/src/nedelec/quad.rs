//! Nedelec-I lowest-order element on the reference quadrilateral `[0,1]^2`.
//!
//! Local edge ordering (counter-clockwise):
//! - e0: (0,0) -> (1,0)
//! - e1: (1,0) -> (1,1)
//! - e2: (1,1) -> (0,1)
//! - e3: (0,1) -> (0,0)
//!
//! The basis is a 1:1 port of MFEM's `ND1QuadFiniteElement::CalcVShape`
//! (fem/fe/fe_nd.cpp, `ND_QuadrilateralElement` with p=1) on `[0,1]²`, so
//! each edge tangential moment is nodal: `DOF_j(Phi_i) = delta_ij`, where
//! DOF is the tangential trace along the oriented edge.

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Lowest-order H(curl) element on reference quad, 4 edge DOFs.
pub struct QuadND1;

impl VectorReferenceElement for QuadND1 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        4
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // e0 (bottom, +x): (1-y, 0)
        values[0] = 1.0 - y;
        values[1] = 0.0;

        // e1 (right, +y): (0, x)
        values[2] = 0.0;
        values[3] = x;

        // e2 (top, -x): (-y, 0)
        values[4] = -y;
        values[5] = 0.0;

        // e3 (left, -y): (0, x-1)
        values[6] = 0.0;
        values[7] = x - 1.0;
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        // scalar curl in 2D: dFy/dx - dFx/dy = +1 for every edge mode
        curl_vals[0] = 1.0;
        curl_vals[1] = 1.0;
        curl_vals[2] = 1.0;
        curl_vals[3] = 1.0;
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0],
            vec![1.0, 0.5],
            vec![0.5, 1.0],
            vec![0.0, 0.5],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd1_curl_constant() {
        let elem = QuadND1;
        let qr = elem.quadrature(3);
        let mut curl = vec![0.0; elem.n_dofs()];
        for pt in &qr.points {
            elem.eval_curl(pt, &mut curl);
            for (i, &c) in curl.iter().enumerate() {
                assert!((c - 1.0).abs() < 1e-13, "curl[{i}] = {c}");
            }
        }
    }

    #[test]
    fn nd1_nodal_edge_moments() {
        let elem = QuadND1;
        let mut vals = vec![0.0; elem.n_dofs() * 2];

        // Unit tangents and edge lengths for local edge ordering on [0,1]².
        let tangents = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let edge_len = [1.0, 1.0, 1.0, 1.0];

        for (j, (mid, (t, l))) in elem
            .dof_coords()
            .iter()
            .zip(tangents.iter().zip(edge_len.iter()))
            .enumerate()
        {
            elem.eval_basis_vec(mid, &mut vals);
            for i in 0..elem.n_dofs() {
                let dof = (vals[i * 2] * t[0] + vals[i * 2 + 1] * t[1]) * l;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }
}
