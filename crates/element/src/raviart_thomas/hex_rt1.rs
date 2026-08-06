//! Raviart-Thomas RT1 element on the reference hexahedron `[-1,1]^3`.
//!
//! # Space: RT₁ = Q_{2,1,1} × Q_{1,2,1} × Q_{1,1,2}
//! dim = 3×2×2 + 2×3×2 + 2×2×3 = 12 + 12 + 12 = 36 DOFs.
//!
//! # DOFs (36 total)
//! - 4 normal-flux moments per face × 6 faces = 24 face DOFs
//! - 12 interior DOFs
//!
//! Delegates to the generic `HexRTk::new(1)` implementation.

use crate::quadrature::hex_rule;
use crate::raviart_thomas::HexRTk;
use crate::reference::{QuadratureRule, VectorReferenceElement};

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
