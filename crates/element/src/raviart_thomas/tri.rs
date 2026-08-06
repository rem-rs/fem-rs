//! Raviart-Thomas RT0 element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! # Reference element geometry
//!
//! Vertices: v₀=(0,0), v₁=(1,0), v₂=(0,1).
//!
//! Faces (edges in 2-D) with outward unit normals:
//! - f₀: v₁→v₂  (opposite v₀),  n̂₀ = (1,  1)/√2
//! - f₁: v₀→v₂  (opposite v₁),  n̂₁ = (−1, 0)
//! - f₂: v₀→v₁  (opposite v₂),  n̂₂ = (0, −1)
//!
//! # Basis functions
//!
//! The three RT0 basis functions on the reference triangle are:
//!
//! ```text
//! Φ₀ = (ξ,     η  )              — associated with f₀ (opposite v₀)
//! Φ₁ = (ξ−1,   η  )              — associated with f₁ (opposite v₁)
//! Φ₂ = (ξ,     η−1)              — associated with f₂ (opposite v₂)
//! ```
//!
//! These have the **Piola form**: Φᵢ = (constant part) such that
//! `DOF_j(Φᵢ) = δᵢⱼ`, i.e. the normal flux of Φᵢ through face j equals 1
//! if i=j and 0 otherwise.
//!
//! The divergence of each RT0 basis is constant = 2 (on the reference triangle
//! with area 1/2, so the integral of the divergence equals 1, consistent with
//! one unit of flux exiting through one face).

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT0 H(div) element on the reference triangle — 3 face DOFs.
///
/// Reference domain: triangle with vertices (0,0), (1,0), (0,1).
///
/// Basis functions:
/// - Φ₀ = (ξ,   η  )   — face f₀ (opposite v₀)
/// - Φ₁ = (ξ−1, η  )   — face f₁ (opposite v₁)
/// - Φ₂ = (ξ,   η−1)   — face f₂ (opposite v₂)
///
/// Each basis function has **div = 2** (constant on the reference element).
pub struct TriRT0;

impl VectorReferenceElement for TriRT0 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        0
    }
    fn n_dofs(&self) -> usize {
        3
    }

    /// `values[i*2 + c]` = component c of RT0 basis function i.
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        // Φ₀ = (ξ, η)
        values[0] = x;
        values[1] = y;
        // Φ₁ = (ξ−1, η)
        values[2] = x - 1.0;
        values[3] = y;
        // Φ₂ = (ξ, η−1)
        values[4] = x;
        values[5] = y - 1.0;
    }

    /// Curl of RT0 functions: `curl(Φ_x, Φ_y) = ∂Φ_y/∂ξ − ∂Φ_x/∂η`.
    /// For RT0: all curls are 0 (div-conforming, not curl-conforming).
    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    /// Divergence: constant 2 for all RT0 basis functions.
    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        // div Φᵢ = ∂Φᵢ_x/∂ξ + ∂Φᵢ_y/∂η = 1 + 1 = 2
        for v in div_vals.iter_mut() {
            *v = 2.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    /// DOF sites: midpoints of the three faces (edges).
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.5], // midpoint of f₀: v₁→v₂
            vec![0.0, 0.5], // midpoint of f₁: v₀→v₂
            vec![0.5, 0.0], // midpoint of f₂: v₀→v₁
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// div = 2 for all RT0 basis functions (constant).
    #[test]
    fn rt0_div_constant() {
        let elem = TriRT0;
        let mut div = vec![0.0; 3];
        for pt in &elem.quadrature(4).points {
            elem.eval_div(pt, &mut div);
            for (i, &d) in div.iter().enumerate() {
                assert!((d - 2.0).abs() < 1e-13, "div[{i}] = {d}");
            }
        }
    }

    /// Nodal basis property: DOF_j(Φᵢ) = δᵢⱼ.
    ///
    /// Outward normals and edge lengths for the reference triangle:
    /// - f₀ (v₁→v₂): length √2, outward normal (1,1)/√2
    /// - f₁ (v₀→v₂): length 1,  outward normal (−1,0)
    /// - f₂ (v₀→v₁): length 1,  outward normal (0,−1)
    #[test]
    fn rt0_nodal_basis() {
        let elem = TriRT0;
        // (outward_normal, edge_length)
        let faces: [([f64; 2], f64); 3] = [
            ([1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()], 2f64.sqrt()),
            ([-1.0, 0.0], 1.0),
            ([0.0, -1.0], 1.0),
        ];

        let mids = elem.dof_coords();
        let mut vals = vec![0.0; 6];
        for (j, (normal, len)) in faces.iter().enumerate() {
            elem.eval_basis_vec(&mids[j], &mut vals);
            for i in 0..3 {
                let dof = (vals[i * 2] * normal[0] + vals[i * 2 + 1] * normal[1]) * len;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// Flux through each face equals 1 for the corresponding basis function
    /// (verified using the divergence theorem: ∫∫ div Φᵢ dA = ∫∂ Φᵢ·n ds).
    /// ∫∫ div Φᵢ dA = 2 × (area of ref triangle) = 2 × 0.5 = 1. ✓
    #[test]
    fn rt0_divergence_theorem_consistency() {
        let elem = TriRT0;
        let qr = elem.quadrature(4);
        let mut div = vec![0.0; 3];
        for i in 0..3 {
            let mut integral = 0.0;
            for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
                elem.eval_div(pt, &mut div);
                integral += div[i] * w;
            }
            // integral of div = 2 * area(ref tri) = 2 * 0.5 = 1
            assert!((integral - 1.0).abs() < 1e-12, "∫div Φ_{i} = {integral}");
        }
    }
}
