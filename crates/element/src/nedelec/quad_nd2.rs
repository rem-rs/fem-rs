//! Second-order tensor-product H(curl) element on reference quad `[0,1]^2`.
//!
//! This implementation uses 2 tangential edge moments per edge (8 DOFs
//! total).  The basis is the `[-1,1]²` edge-trace splitting transported to
//! the MFEM `[0,1]²` reference domain (Piola-equivalent: `φ'(x) = 2φ(2x-1)`,
//! so physical traces are unchanged).

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Second-order H(curl) element on reference quad, 8 edge-based DOFs.
pub struct QuadND2;

impl VectorReferenceElement for QuadND2 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        8
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // Two edge modes per side (8 total): split ND1 edge traces by linear factors.
        // bottom edge (y=0), +x
        values[0] = (1.0 - y) * (1.0 - x);
        values[1] = 0.0;
        values[2] = x * (1.0 - y);
        values[3] = 0.0;

        // right edge (x=1), +y
        values[4] = 0.0;
        values[5] = x * (1.0 - y);
        values[6] = 0.0;
        values[7] = x * y;

        // top edge (y=1), -x
        values[8] = -x * y;
        values[9] = 0.0;
        values[10] = -(1.0 - x) * y;
        values[11] = 0.0;

        // left edge (x=0), -y
        values[12] = 0.0;
        values[13] = -(1.0 - x) * y;
        values[14] = 0.0;
        values[15] = -(1.0 - x) * (1.0 - y);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // scalar curl in 2D: dFy/dx - dFx/dy
        curl_vals[0] = 1.0 - x;   // Φ0 = ((1-y)(1-x), 0)
        curl_vals[1] = x;         // Φ1 = (x(1-y), 0)
        curl_vals[2] = 1.0 - y;   // Φ2 = (0, x(1-y))
        curl_vals[3] = y;         // Φ3 = (0, xy)
        curl_vals[4] = x;         // Φ4 = (-xy, 0)
        curl_vals[5] = 1.0 - x;   // Φ5 = (-y(1-x), 0)
        curl_vals[6] = y;         // Φ6 = (0, -y(1-x))
        curl_vals[7] = 1.0 - y;   // Φ7 = (0, -(1-x)(1-y))
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
        // Two Gauss points 1/3, 2/3 per edge.
        vec![
            vec![1.0 / 3.0, 0.0],
            vec![2.0 / 3.0, 0.0],
            vec![1.0, 1.0 / 3.0],
            vec![1.0, 2.0 / 3.0],
            vec![1.0 / 3.0, 1.0],
            vec![2.0 / 3.0, 1.0],
            vec![0.0, 1.0 / 3.0],
            vec![0.0, 2.0 / 3.0],
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

    #[test]
    fn nd2_splits_nd1_traces() {
        // The 8 edge modes split the 4 ND1 traces into pairs:
        //   bottom: Φ0+Φ1 = (1-y, 0)     right: Φ2+Φ3 = (0, x)
        //   top:    Φ4+Φ5 = (-y, 0)      left:  Φ6+Φ7 = (0, x-1)
        let elem = QuadND2;
        let nd1 = crate::nedelec::QuadNDk::new(1);
        let mut vals = vec![0.0; 16];
        let mut ref1 = vec![0.0; 8];
        let mut curl = vec![0.0; 8];
        let mut curl1 = vec![0.0; 4];
        for xi in elem.quadrature(5).points {
            let xi: &[f64] = &xi;
            elem.eval_basis_vec(xi, &mut vals);
            elem.eval_curl(xi, &mut curl);
            nd1.eval_basis_vec(xi, &mut ref1);
            nd1.eval_curl(xi, &mut curl1);
            for d in 0..2 {
                let sum0 = vals[0 * 2 + d] + vals[1 * 2 + d];
                let sum1 = vals[2 * 2 + d] + vals[3 * 2 + d];
                let sum2 = vals[4 * 2 + d] + vals[5 * 2 + d];
                let sum3 = vals[6 * 2 + d] + vals[7 * 2 + d];
                assert!((sum0 - ref1[0 * 2 + d]).abs() < 1e-13, "bottom comp {d}");
                assert!((sum1 - ref1[1 * 2 + d]).abs() < 1e-13, "right comp {d}");
                assert!((sum2 - ref1[2 * 2 + d]).abs() < 1e-13, "top comp {d}");
                assert!((sum3 - ref1[3 * 2 + d]).abs() < 1e-13, "left comp {d}");
            }
            for i in 0..4 {
                assert!(
                    (curl[2 * i] + curl[2 * i + 1] - curl1[i]).abs() < 1e-13,
                    "curl pair {i}"
                );
            }
        }
    }
}
