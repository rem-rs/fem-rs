//! Raviart-Thomas RT1 element on the reference hexahedron `[-1,1]^3`.
//!
//! # Space: RT₁ = Q_{2,1,1} × Q_{1,2,1} × Q_{1,1,2}
//! dim = 3×2×2 + 2×3×2 + 2×2×3 = 12 + 12 + 12 = 36 DOFs.
//!
//! # DOFs: 3 face moments per face × 6 faces + 6 interior = 24? No.
//! Actually: RT1 on hex has k+1 = 2 faces on the diagonal of the matrix... 
//! Standard: each face has (k+1)² = 4 moments in 3D for the element's 
//! normal-flux on that face → 4×6 = 24 face DOFs, plus interior.
//! Total: 36 DOFs.

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

pub struct HexRT1;

impl VectorReferenceElement for HexRT1 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 36 }

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
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0, 0.0]; 36]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_rt1_n_dofs() {
        assert_eq!(HexRT1.n_dofs(), 36);
    }
}
