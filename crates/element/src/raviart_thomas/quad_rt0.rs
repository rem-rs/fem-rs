//! Raviart-Thomas RT0 element on the reference quadrilateral `[0,1]^2`.
//!
//! # Reference element geometry
//!
//! Vertices: v₀=(0,0), v₁=(1,0), v₂=(1,1), v₃=(0,1).
//!
//! Faces (edges in 2-D, counter-clockwise) with outward unit normals:
//! - f₀: v₀→v₁  (bottom),  n̂₀ = ( 0, −1),  length = 1
//! - f₁: v₁→v₂  (right),   n̂₁ = ( 1,  0),  length = 1
//! - f₂: v₂→v₃  (top),     n̂₂ = ( 0,  1),  length = 1
//! - f₃: v₃→v₀  (left),    n̂₃ = (−1,  0),  length = 1
//!
//! # Basis functions
//!
//! The four RT0 basis functions on `[0,1]²` are a 1:1 port of MFEM's
//! `RT0QuadFiniteElement::CalcVShape` (fem/fe/fe_fixed_order.cpp); they
//! satisfy `∫_{f_j} Φ_i · n̂_j ds = δ_ij`:
//!
//! - Φ₀ = (0, y−1)  — face f₀ (bottom)
//! - Φ₁ = (x, 0)    — face f₁ (right)
//! - Φ₂ = (0, y)    — face f₂ (top)
//! - Φ₃ = (x−1, 0)  — face f₃ (left)
//!
//! Each has `div Φ_i = 1` (constant on the reference element).

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT0 H(div) element on the reference quadrilateral — 4 face DOFs.
///
/// Reference domain: `[0,1]²` (matches MFEM and the `QuadQk` geometry).
pub struct QuadRT0;

impl VectorReferenceElement for QuadRT0 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        0
    }
    fn n_dofs(&self) -> usize {
        4
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        // Φ₀ = (0, y−1) — bottom face
        values[0] = 0.0;
        values[1] = y - 1.0;
        // Φ₁ = (x, 0) — right face
        values[2] = x;
        values[3] = 0.0;
        // Φ₂ = (0, y) — top face
        values[4] = 0.0;
        values[5] = y;
        // Φ₃ = (x−1, 0) — left face
        values[6] = x - 1.0;
        values[7] = 0.0;
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 1.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    /// Midpoint of each face in CCW order.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0], // f₀ bottom midpoint
            vec![1.0, 0.5], // f₁ right midpoint
            vec![0.5, 1.0], // f₂ top midpoint
            vec![0.0, 0.5], // f₃ left midpoint
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt0_div_constant() {
        let elem = QuadRT0;
        let mut div = vec![0.0; 4];
        for pt in &elem.quadrature(4).points {
            elem.eval_div(pt, &mut div);
            for (i, &d) in div.iter().enumerate() {
                assert!((d - 1.0).abs() < 1e-13, "div[{i}] = {d}");
            }
        }
    }

    #[test]
    fn rt0_nodal_basis() {
        let elem = QuadRT0;
        // Faces in CCW order: (normal, length)
        let faces: [([f64; 2], f64); 4] = [
            ([0.0, -1.0], 1.0), // f₀ bottom
            ([1.0, 0.0], 1.0),  // f₁ right
            ([0.0, 1.0], 1.0),  // f₂ top
            ([-1.0, 0.0], 1.0), // f₃ left
        ];

        let mids = elem.dof_coords();
        let mut vals = vec![0.0; 8];
        for (j, (normal, len)) in faces.iter().enumerate() {
            elem.eval_basis_vec(&mids[j], &mut vals);
            for i in 0..4 {
                let dof = (vals[i * 2] * normal[0] + vals[i * 2 + 1] * normal[1]) * len;
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
        let elem = QuadRT0;
        let qr = elem.quadrature(4);
        let mut div = vec![0.0; 4];
        for i in 0..4 {
            let mut integral = 0.0;
            for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
                elem.eval_div(pt, &mut div);
                integral += div[i] * w;
            }
            assert!((integral - 1.0).abs() < 1e-12, "∫div Φ_{i} = {integral}");
        }
    }
}
