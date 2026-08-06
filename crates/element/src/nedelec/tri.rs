//! Nedelec-I element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! # Reference element geometry
//!
//! Vertices:  v₀=(0,0),  v₁=(1,0),  v₂=(0,1)
//!
//! Edges (with tangent direction used for DOF sign convention):
//! - e₀: v₀→v₁,  tangent t̂₀ = (1,0)
//! - e₁: v₁→v₂,  tangent t̂₁ = (−1,1)/√2
//! - e₂: v₂→v₀,  tangent t̂₂ = (0,−1)
//!
//! # Basis functions
//!
//! The three lowest-order Nedelec-I basis functions on the reference triangle are:
//!
//! ```text
//!   Φ₀(ξ,η) = (  η,  −ξ )   ← associated with edge e₀ (v₀→v₁)
//!   Φ₁(ξ,η) = (  η,  1−ξ−η ) ← wait — let's use the standard monomial basis below
//! ```
//!
//! The canonical lowest-order Nedelec-I basis on the reference triangle
//! (Nédélec 1980, also MFEM `ND_TriangleElement` order 1) is:
//!
//! ```text
//!   Φ₀ = [  1−η,   ξ  ]
//!   Φ₁ = [ −η,    ξ  ]
//!   Φ₂ = [  η,   1−ξ ]
//! ```
//!
//! Wait — the standard presentation uses the **Whitney form** based on barycentric
//! coordinates.  Let λ₀=1−ξ−η, λ₁=ξ, λ₂=η.
//!
//! Whitney 1-forms: `w_{ij} = λᵢ ∇λⱼ − λⱼ ∇λᵢ`
//!
//! Edge e₀ (λ₀,λ₁): `Φ₀ = λ₀ ∇λ₁ − λ₁ ∇λ₀`
//!   = (1−ξ−η)(1,0) − ξ(−1,−1) = (1−η, ξ)  ... wait, let's compute carefully.
//!
//! ∇λ₀ = (−1,−1), ∇λ₁ = (1,0), ∇λ₂ = (0,1).
//!
//! ```text
//!   Φ₀ = λ₀ ∇λ₁ − λ₁ ∇λ₀ = (1−ξ−η)(1,0) − ξ(−1,−1) = (1−η, ξ)
//!   Φ₁ = λ₁ ∇λ₂ − λ₂ ∇λ₁ = ξ(0,1) − η(1,0)          = (−η, ξ)
//!   Φ₂ = λ₀ ∇λ₂ − λ₂ ∇λ₀ = (1−ξ−η)(0,1) − η(−1,−1)  = (η, 1−ξ)
//! ```
//!
//! DOF i is the tangential moment on edge i:
//! `DOF_i(u) = ∫_{e_i} u · t̂_i ds`
//!
//! The tangent vectors (in the direction of increasing parameter):
//! - e₀: v₀→v₁: t̂₀ = (1, 0)
//! - e₁: v₁→v₂: t̂₁ = (−1, 1)/√2
//! - e₂: v₀→v₂: t̂₂ = (0, 1)
//!
//! One can verify that `DOF_j(Φᵢ) = δᵢⱼ` with the above definitions.

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Nedelec first-kind H(curl) element on the reference triangle — 3 edge DOFs.
///
/// Reference domain: triangle with vertices (0,0), (1,0), (0,1).
///
/// Basis functions (Whitney 1-forms `w_{ij} = λᵢ ∇λⱼ − λⱼ ∇λᵢ`):
/// - Φ₀ = w₀₁ = (1−η,  ξ)    — edge e₀: v₀→v₁, curl = +2
/// - Φ₁ = w₁₂ = (−η,   ξ)    — edge e₁: v₁→v₂, curl = +2
/// - Φ₂ = w₀₂ = ( η,  1−ξ)   — edge e₂: v₀→v₂, curl = −2
///
/// The scalar 2-D curl of a vector field (Φ_x, Φ_y) is `∂Φ_y/∂ξ − ∂Φ_x/∂η`.
pub struct TriND1;

impl VectorReferenceElement for TriND1 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        3
    }

    /// `values[i*2 + c]` = component c of basis function i.
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        // Φ₀ = w_{01} = (1−η, ξ)
        values[0] = 1.0 - y;
        values[1] = x;
        // Φ₁ = w_{12} = (−η, ξ)
        values[2] = -y;
        values[3] = x;
        // Φ₂ = w_{02} = (η, 1−ξ)
        values[4] = y;
        values[5] = 1.0 - x;
    }

    /// 2-D scalar curl: `curl_vals[i] = ∂Φᵢ_y/∂ξ − ∂Φᵢ_x/∂η`
    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        // Φ₀ = (1−η, ξ):  ∂(ξ)/∂ξ − ∂(1−η)/∂η = 1 − (−1) = 2
        curl_vals[0] = 2.0;
        // Φ₁ = (−η, ξ):   ∂(ξ)/∂ξ − ∂(−η)/∂η  = 1 − (−1) = 2
        curl_vals[1] = 2.0;
        // Φ₂ = (η, 1−ξ):  ∂(1−ξ)/∂ξ − ∂(η)/∂η = −1 − 1 = −2
        curl_vals[2] = -2.0;
    }

    /// Divergence — not the natural operator for H(curl); returns zeros.
    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    /// DOF sites: midpoints of the three edges.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0], // midpoint of e₀: v₀→v₁
            vec![0.5, 0.5], // midpoint of e₁: v₁→v₂
            vec![0.0, 0.5], // midpoint of e₂: v₀→v₂
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// curl of ND1 basis functions on the reference triangle: [2, 2, -2].
    #[test]
    fn nd1_curl_constant() {
        let elem = TriND1;
        let mut curl = vec![0.0; 3];
        let expected = [2.0, 2.0, -2.0];
        let qr = elem.quadrature(3);
        for pt in &qr.points {
            elem.eval_curl(pt, &mut curl);
            for (i, &c) in curl.iter().enumerate() {
                assert!(
                    (c - expected[i]).abs() < 1e-13,
                    "curl[{i}] = {c}, expected {}",
                    expected[i]
                );
            }
        }
    }

    /// Nodal basis property: DOF_j(Φᵢ) = δᵢⱼ.
    ///
    /// For the tangential DOF on edge eⱼ we approximate the line integral by
    /// evaluating Φᵢ at the edge midpoint and dotting with the edge tangent.
    /// For lowest-order ND1 the integrand is linear so the midpoint rule is exact.
    #[test]
    fn nd1_nodal_basis() {
        let elem = TriND1;
        // Edge tangents (unit, direction of local edge):
        // e₀: v₀→v₁  tangent (1,0)       length 1
        // e₁: v₁→v₂  tangent (−1,1)/√2   length √2
        // e₂: v₀→v₂  tangent (0,1)       length 1
        let tangents: [[f64; 2]; 3] = [
            [1.0, 0.0],
            [-1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()],
            [0.0, 1.0],
        ];
        let edge_len = [1.0_f64, 2f64.sqrt(), 1.0_f64];

        let mut vals = vec![0.0; 6];
        for (j, (mid, (t, l))) in elem
            .dof_coords()
            .iter()
            .zip(tangents.iter().zip(edge_len.iter()))
            .enumerate()
        {
            elem.eval_basis_vec(mid, &mut vals);
            for i in 0..3 {
                // tangential component at midpoint × edge length = line integral
                let dof = (vals[i * 2] * t[0] + vals[i * 2 + 1] * t[1]) * l;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// The vector sum Σ Φᵢ = (1−η−η+η, ξ+ξ+1−ξ) = (1−η, ξ+1).
    /// Not a fixed constant, but a useful sanity-check that the definitions are consistent.
    #[test]
    fn nd1_basis_values_at_centroid() {
        let elem = TriND1;
        let centroid = [1.0 / 3.0, 1.0 / 3.0];
        let mut vals = vec![0.0; 6];
        elem.eval_basis_vec(&centroid, &mut vals);
        // Φ₀ = (1−1/3, 1/3) = (2/3, 1/3)
        assert!((vals[0] - 2.0 / 3.0).abs() < 1e-14);
        assert!((vals[1] - 1.0 / 3.0).abs() < 1e-14);
        // Φ₁ = (−1/3, 1/3)
        assert!((vals[2] + 1.0 / 3.0).abs() < 1e-14);
        assert!((vals[3] - 1.0 / 3.0).abs() < 1e-14);
        // Φ₂ = (η, 1−ξ) = (1/3, 2/3)
        assert!((vals[4] - 1.0 / 3.0).abs() < 1e-14);
        assert!((vals[5] - 2.0 / 3.0).abs() < 1e-14);
    }
}
