//! Raviart-Thomas RT2 element on the reference tetrahedron.
//!
//! # Space: RT₂ = P₂³ ⊕ x·P̃₂
//! dim = 30 + 6 = 36 DOFs.
//!
//! # DOFs (36 total)
//! - 6 normal-flux face moments per face × 4 faces = 24 face DOFs
//! - 12 interior DOFs (∫ u · w dV for w ∈ (P₁)³)
//!
//! Delegates to the generic `TetRTk::new(2)` implementation.

use crate::quadrature::tet_rule;
use crate::raviart_thomas::TetRTk;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT2 H(div) element on the reference tetrahedron — 36 DOFs, order 2.
pub struct TetRT2;

impl VectorReferenceElement for TetRT2 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 36 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        TetRTk::new(2).eval_basis_vec(xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        TetRTk::new(2).eval_div(xi, div_vals);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        (0..36).map(|_| vec![0.25, 0.25, 0.25]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_rt2_n_dofs() {
        assert_eq!(TetRT2.n_dofs(), 36);
    }

    #[test]
    fn tet_rt2_basis_finite() {
        let elem = TetRT2;
        let mut v = vec![0.0; 36 * 3];
        for xi in &[
            vec![0., 0., 0.], vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.],
            vec![0.25, 0.25, 0.25], vec![0.0, 0.5, 0.5],
        ] {
            elem.eval_basis_vec(xi, &mut v);
            for &val in &v { assert!(val.is_finite(), "non-finite at {xi:?}: {val}"); }
        }
    }

    #[test]
    fn tet_rt2_div_finite() {
        let elem = TetRT2;
        let mut div = vec![0.0; 36];
        let qr = elem.quadrature(3);
        for xi in &qr.points {
            elem.eval_div(xi, &mut div);
            for &d in &div { assert!(d.is_finite()); }
        }
    }
}
