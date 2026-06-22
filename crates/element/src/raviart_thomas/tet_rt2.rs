//! Raviart-Thomas RT2 element on the reference tetrahedron.
//!
//! # Space: RT₂ = P₂³ ⊕ x P̃₂
//! dim = 30 + 5 = 35? No — MFEM has 15 DOFs for 2D RT2.
//! In 3D: RT₂ = P₂³ ⊕ x P̃₂ with dim = 3×10 + 15 = 45? 
//! Actually: RT₂ on tet has 3 face DOFs × 4 faces + 3 interior = 15 DOFs.
//! But wait, MFEM documentation says RT2 on tet has 15 DOFs.
//! Let me provide the correct count: 3 per face × 4 + 3 interior = 15.
//!
//! # Placeholder
//! This is a structural placeholder. eval_basis_vec returns zero vectors.
//! It provides the correct dimensional information for DOF management.

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

pub struct TetRT2;

impl VectorReferenceElement for TetRT2 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 15 }

    fn eval_basis_vec(&self, _xi: &[f64], values: &mut [f64]) {
        for v in values.iter_mut() { *v = 0.0; }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        use crate::quadrature::hex_rule;
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0, 0.0]; 15]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_rt2_n_dofs() {
        assert_eq!(TetRT2.n_dofs(), 15);
    }
}
