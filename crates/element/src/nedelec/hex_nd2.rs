//! Second-order tensor-product H(curl) element on reference hex `[-1,1]^3`.
//!
//! This implementation provides 2 edge moments per edge (24 DOFs total).

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};
use crate::nedelec::hex::HexND1;

/// Second-order H(curl) element on reference hex, 24 edge-based DOFs.
pub struct HexND2;

impl VectorReferenceElement for HexND2 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 24 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let mut low = vec![0.0_f64; 12 * 3];
        HexND1.eval_basis_vec(xi, &mut low);

        // Duplicate each ND1 edge mode into two ND2 edge modes with linear split
        // along the edge direction. This yields two independent edge moments.
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        for i in 0..24 * 3 {
            values[i] = 0.0;
        }

        for e in 0..12 {
            let (s0, s1) = match e {
                0..=3 => (0.5 * (1.0 - x), 0.5 * (1.0 + x)),
                4..=7 => (0.5 * (1.0 - y), 0.5 * (1.0 + y)),
                _ => (0.5 * (1.0 - z), 0.5 * (1.0 + z)),
            };
            for d in 0..3 {
                values[(2 * e) * 3 + d] = s0 * low[e * 3 + d];
                values[(2 * e + 1) * 3 + d] = s1 * low[e * 3 + d];
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let mut low_curl = vec![0.0_f64; 12 * 3];
        HexND1.eval_curl(xi, &mut low_curl);

        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        for i in 0..24 * 3 {
            curl_vals[i] = 0.0;
        }

        for e in 0..12 {
            let (s0, s1) = match e {
                0..=3 => (0.5 * (1.0 - x), 0.5 * (1.0 + x)),
                4..=7 => (0.5 * (1.0 - y), 0.5 * (1.0 + y)),
                _ => (0.5 * (1.0 - z), 0.5 * (1.0 + z)),
            };
            for d in 0..3 {
                curl_vals[(2 * e) * 3 + d] = s0 * low_curl[e * 3 + d];
                curl_vals[(2 * e + 1) * 3 + d] = s1 * low_curl[e * 3 + d];
            }
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(24);
        let edge_mids = vec![
            vec![0.0, -1.0, -1.0], vec![0.0, 1.0, -1.0], vec![0.0, -1.0, 1.0], vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, -1.0], vec![-1.0, 0.0, -1.0], vec![1.0, 0.0, 1.0], vec![-1.0, 0.0, 1.0],
            vec![-1.0, -1.0, 0.0], vec![1.0, -1.0, 0.0], vec![1.0, 1.0, 0.0], vec![-1.0, 1.0, 0.0],
        ];
        for p in edge_mids {
            out.push(p.clone());
            out.push(p);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd2_hex_basis_and_curl_are_finite() {
        let elem = HexND2;
        let qr = elem.quadrature(3);
        let mut phi = vec![0.0; elem.n_dofs() * 3];
        let mut curl = vec![0.0; elem.n_dofs() * 3];
        for xi in &qr.points {
            elem.eval_basis_vec(xi, &mut phi);
            elem.eval_curl(xi, &mut curl);
            assert!(phi.iter().all(|v| v.is_finite()));
            assert!(curl.iter().all(|v| v.is_finite()));
        }
    }
}
