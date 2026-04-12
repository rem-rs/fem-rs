//! Second-order tensor-product H(curl) element on reference quad `[-1,1]^2`.
//!
//! This implementation uses 2 edge moments per edge (8 DOFs total).

use crate::quadrature::quad_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Second-order H(curl) element on reference quad, 8 edge-based DOFs.
pub struct QuadND2;

impl VectorReferenceElement for QuadND2 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 8 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // Two edge modes per side (8 total): split ND1 edge traces by linear factors.
        // bottom edge (y=-1), +x
        values[0] = 0.125 * (1.0 - y) * (1.0 - x);
        values[1] = 0.0;
        values[2] = 0.125 * (1.0 - y) * (1.0 + x);
        values[3] = 0.0;

        // right edge (x=+1), +y
        values[4] = 0.0;
        values[5] = 0.125 * (1.0 + x) * (1.0 - y);
        values[6] = 0.0;
        values[7] = 0.125 * (1.0 + x) * (1.0 + y);

        // top edge (y=+1), -x
        values[8] = -0.125 * (1.0 + y) * (1.0 + x);
        values[9] = 0.0;
        values[10] = -0.125 * (1.0 + y) * (1.0 - x);
        values[11] = 0.0;

        // left edge (x=-1), -y
        values[12] = 0.0;
        values[13] = -0.125 * (1.0 - x) * (1.0 + y);
        values[14] = 0.0;
        values[15] = -0.125 * (1.0 - x) * (1.0 - y);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // scalar curl in 2D: dFy/dx - dFx/dy
        curl_vals[0] = 0.125 * (1.0 - x);
        curl_vals[1] = 0.125 * (1.0 + x);
        curl_vals[2] = 0.125 * (1.0 - y);
        curl_vals[3] = 0.125 * (1.0 + y);
        curl_vals[4] = -0.125 * (1.0 + x);
        curl_vals[5] = -0.125 * (1.0 - x);
        curl_vals[6] = -0.125 * (1.0 + y);
        curl_vals[7] = -0.125 * (1.0 - y);
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![-0.5, -1.0], vec![0.5, -1.0],
            vec![1.0, -0.5], vec![1.0, 0.5],
            vec![0.5, 1.0], vec![-0.5, 1.0],
            vec![-1.0, 0.5], vec![-1.0, -0.5],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd2_quad_basis_and_curl_are_finite() {
        let elem = QuadND2;
        let qr = elem.quadrature(4);
        let mut phi = vec![0.0; elem.n_dofs() * 2];
        let mut curl = vec![0.0; elem.n_dofs()];
        for xi in &qr.points {
            elem.eval_basis_vec(xi, &mut phi);
            elem.eval_curl(xi, &mut curl);
            assert!(phi.iter().all(|v| v.is_finite()));
            assert!(curl.iter().all(|v| v.is_finite()));
        }
    }
}
