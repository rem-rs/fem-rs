//! Raviart-Thomas RT0 element on the reference hexahedron `[-1,1]^3`.
//!
//! # Reference element geometry
//!
//! Vertices (MFEM convention):
//! v₀=(-1,-1,-1), v₁=(1,-1,-1), v₂=(1,1,-1), v₃=(-1,1,-1),
//! v₄=(-1,-1, 1), v₅=(1,-1, 1), v₆=(1,1, 1), v₇=(-1,1, 1).
//!
//! Faces with outward unit normals:
//! - f₀: z=−1  (bottom),  n̂ = ( 0,  0, −1),  area = 4
//! - f₁: z= 1  (top),     n̂ = ( 0,  0,  1),  area = 4
//! - f₂: y=−1  (near),    n̂ = ( 0, −1,  0),  area = 4
//! - f₃: y= 1  (far),     n̂ = ( 0,  1,  0),  area = 4
//! - f₄: x=−1  (left),    n̂ = (−1,  0,  0),  area = 4
//! - f₅: x= 1  (right),   n̂ = ( 1,  0,  0),  area = 4
//!
//! # Basis functions
//!
//! The six RT0 basis functions on `[-1,1]³`:
//!
//! ```text
//! Φ₀ = (0, 0, (ζ−1)/8)   — face f₀ (z=−1)
//! Φ₁ = (0, 0, (1+ζ)/8)   — face f₁ (z= 1)
//! Φ₂ = (0, (η−1)/8, 0)   — face f₂ (y=−1)
//! Φ₃ = (0, (1+η)/8, 0)   — face f₃ (y= 1)
//! Φ₄ = ((ξ−1)/8, 0, 0)   — face f₄ (x=−1)
//! Φ₅ = ((1+ξ)/8, 0, 0)   — face f₅ (x= 1)
//! ```
//!
//! Each satisfies `∫_{f_j} Φ_i · n̂_j dS = δ_ij`.
//! Each has `div Φ_i = 1/8` (constant on the reference element).

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT0 H(div) element on the reference hexahedron — 6 face DOFs.
///
/// Reference domain: `[-1,1]³`.
pub struct HexRT0;

impl VectorReferenceElement for HexRT0 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 6 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        values[0] = 0.0; values[1] = 0.0; values[2] = (z - 1.0) / 8.0;
        values[3] = 0.0; values[4] = 0.0; values[5] = (1.0 + z) / 8.0;
        values[6] = 0.0; values[7] = (y - 1.0) / 8.0; values[8] = 0.0;
        values[9] = 0.0; values[10] = (1.0 + y) / 8.0; values[11] = 0.0;
        values[12] = (x - 1.0) / 8.0; values[13] = 0.0; values[14] = 0.0;
        values[15] = (1.0 + x) / 8.0; values[16] = 0.0; values[17] = 0.0;
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.125;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, -1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt0_div_constant() {
        let elem = HexRT0;
        let mut div = vec![0.0; 6];
        for pt in &elem.quadrature(3).points {
            elem.eval_div(pt, &mut div);
            for (i, &d) in div.iter().enumerate() {
                assert!((d - 0.125).abs() < 1e-13, "div[{i}] = {d}");
            }
        }
    }

    #[test]
    fn rt0_nodal_basis() {
        let elem = HexRT0;
        let faces: [([f64; 3], f64); 6] = [
            ([0.0, 0.0, -1.0], 4.0),
            ([0.0, 0.0, 1.0], 4.0),
            ([0.0, -1.0, 0.0], 4.0),
            ([0.0, 1.0, 0.0], 4.0),
            ([-1.0, 0.0, 0.0], 4.0),
            ([1.0, 0.0, 0.0], 4.0),
        ];

        let mids = elem.dof_coords();
        let mut vals = vec![0.0; 18];
        for (j, (normal, area)) in faces.iter().enumerate() {
            elem.eval_basis_vec(&mids[j], &mut vals);
            for i in 0..6 {
                let dot = vals[i * 3] * normal[0]
                    + vals[i * 3 + 1] * normal[1]
                    + vals[i * 3 + 2] * normal[2];
                let dof = dot * area;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rt0_divergence_theorem_consistency() {
        let elem = HexRT0;
        let qr = elem.quadrature(3);
        let mut div = vec![0.0; 6];
        for i in 0..6 {
            let mut integral = 0.0;
            for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
                elem.eval_div(pt, &mut div);
                integral += div[i] * w;
            }
            assert!((integral - 1.0).abs() < 1e-12, "div Φ_{i} = {integral}");
        }
    }
}
