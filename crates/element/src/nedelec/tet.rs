//! Nedelec-I element on the reference tetrahedron.
//!
//! Reference vertices: v₀=(0,0,0), v₁=(1,0,0), v₂=(0,1,0), v₃=(0,0,1).
//!
//! # Edges (and DOF association)
//! | DOF | Edge        | from → to |
//! |-----|-------------|-----------|
//! | 0   | e₀₁         | v₀ → v₁  |
//! | 1   | e₀₂         | v₀ → v₂  |
//! | 2   | e₀₃         | v₀ → v₃  |
//! | 3   | e₁₂         | v₁ → v₂  |
//! | 4   | e₁₃         | v₁ → v₃  |
//! | 5   | e₂₃         | v₂ → v₃  |
//!
//! # Basis functions (Whitney 1-forms)
//! With barycentric coordinates λ₀=1−ξ−η−ζ, λ₁=ξ, λ₂=η, λ₃=ζ and
//! ∇λ₀=(−1,−1,−1), ∇λ₁=(1,0,0), ∇λ₂=(0,1,0), ∇λ₃=(0,0,1):
//!
//! `Φᵢⱼ = λᵢ ∇λⱼ − λⱼ ∇λᵢ`
//!
//! ```text
//! Φ₀₁ = λ₀∇λ₁ − λ₁∇λ₀ = (1−ξ−η−ζ)(1,0,0) − ξ(−1,−1,−1) = (1−η−ζ, ξ, ξ)
//! Φ₀₂ = λ₀∇λ₂ − λ₂∇λ₀ = (1−ξ−η−ζ)(0,1,0) − η(−1,−1,−1) = (η, 1−ξ−ζ, η)
//! Φ₀₃ = λ₀∇λ₃ − λ₃∇λ₀ = (1−ξ−η−ζ)(0,0,1) − ζ(−1,−1,−1) = (ζ, ζ, 1−ξ−η)
//! Φ₁₂ = λ₁∇λ₂ − λ₂∇λ₁ = ξ(0,1,0) − η(1,0,0)             = (−η, ξ, 0)
//! Φ₁₃ = λ₁∇λ₃ − λ₃∇λ₁ = ξ(0,0,1) − ζ(1,0,0)             = (−ζ, 0, ξ)
//! Φ₂₃ = λ₂∇λ₃ − λ₃∇λ₂ = η(0,0,1) − ζ(0,1,0)             = (0, −ζ, η)
//! ```
//!
//! # Curl
//! `(curl Φ)ₖ = εₖᵢⱼ ∂Φⱼ/∂xᵢ`
//!
//! All Whitney 1-forms have **constant curl** on the reference element:
//! `curl Φᵢⱼ = 2 (∇λᵢ × ∇λⱼ)`
//!
//! ```text
//! curl Φ₀₁ = 2 ∇λ₀ × ∇λ₁ = 2 (−1,−1,−1)×(1,0,0) = 2(0,−1,1) → (0,−2,2)  — wait
//! ```
//!
//! Let us compute with the determinant formula ∇λᵢ × ∇λⱼ:
//! - ∇λ₀ × ∇λ₁ = (−1,−1,−1)×(1,0,0) = (0·0−(−1)·0, (−1)·1−(−1)·0, (−1)·0−(−1)·1) = (0,−1,1)
//! - ∇λ₀ × ∇λ₂ = (−1,−1,−1)×(0,1,0) = ((−1)·0−(−1)·1, (−1)·0−(−1)·0, (−1)·1−(−1)·0) = (1,0,−1)
//! - ∇λ₀ × ∇λ₃ = (−1,−1,−1)×(0,0,1) = ((−1)·1−(−1)·0, (−1)·0−(−1)·1, (−1)·0−(−1)·0) = (−1,1,0)
//! - ∇λ₁ × ∇λ₂ = (1,0,0)×(0,1,0)     = (0,0,1)
//! - ∇λ₁ × ∇λ₃ = (1,0,0)×(0,0,1)     = (0,−1,0)
//! - ∇λ₂ × ∇λ₃ = (0,1,0)×(0,0,1)     = (1,0,0)

use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Nedelec first-kind H(curl) element on the reference tetrahedron — 6 edge DOFs.
///
/// Reference domain: tetrahedron with vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1).
pub struct TetND1;

impl VectorReferenceElement for TetND1 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        6
    }

    /// `values[i*3 + c]` = component c of basis function i.
    ///
    /// DOF ordering: e₀₁, e₀₂, e₀₃, e₁₂, e₁₃, e₂₃.
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        // Φ₀₁ = (1−y−z, x, x)
        values[0] = 1.0 - y - z;
        values[1] = x;
        values[2] = x;
        // Φ₀₂ = (y, 1−x−z, y)
        values[3] = y;
        values[4] = 1.0 - x - z;
        values[5] = y;
        // Φ₀₃ = (z, z, 1−x−y)
        values[6] = z;
        values[7] = z;
        values[8] = 1.0 - x - y;
        // Φ₁₂ = (−y, x, 0)
        values[9] = -y;
        values[10] = x;
        values[11] = 0.0;
        // Φ₁₃ = (−z, 0, x)
        values[12] = -z;
        values[13] = 0.0;
        values[14] = x;
        // Φ₂₃ = (0, −z, y)
        values[15] = 0.0;
        values[16] = -z;
        values[17] = y;
    }

    /// Constant curls of each Whitney 1-form: `curl_vals[i*3 + c]`.
    ///
    /// curl Φᵢⱼ = 2 (∇λᵢ × ∇λⱼ).
    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        // e₀₁: 2*(0,−1,1)
        curl_vals[0] = 0.0;
        curl_vals[1] = -2.0;
        curl_vals[2] = 2.0;
        // e₀₂: 2*(1,0,−1)
        curl_vals[3] = 2.0;
        curl_vals[4] = 0.0;
        curl_vals[5] = -2.0;
        // e₀₃: 2*(−1,1,0)
        curl_vals[6] = -2.0;
        curl_vals[7] = 2.0;
        curl_vals[8] = 0.0;
        // e₁₂: 2*(0,0,1)
        curl_vals[9] = 0.0;
        curl_vals[10] = 0.0;
        curl_vals[11] = 2.0;
        // e₁₃: 2*(0,−1,0)
        curl_vals[12] = 0.0;
        curl_vals[13] = -2.0;
        curl_vals[14] = 0.0;
        // e₂₃: 2*(1,0,0)
        curl_vals[15] = 2.0;
        curl_vals[16] = 0.0;
        curl_vals[17] = 0.0;
    }

    /// Divergence — zero for Whitney 1-forms (not the natural operator).
    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }

    /// DOF sites: midpoints of the six edges.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0, 0.0], // e₀₁
            vec![0.0, 0.5, 0.0], // e₀₂
            vec![0.0, 0.0, 0.5], // e₀₃
            vec![0.5, 0.5, 0.0], // e₁₂
            vec![0.5, 0.0, 0.5], // e₁₃
            vec![0.0, 0.5, 0.5], // e₂₃
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All curls are constant — verify at several quadrature points.
    #[test]
    fn tet_nd1_curl_constant() {
        let elem = TetND1;
        let mut curl = vec![0.0; 18];
        let qr = elem.quadrature(3);
        // Expected constant curls
        let expected: [[f64; 3]; 6] = [
            [0.0, -2.0, 2.0],
            [2.0, 0.0, -2.0],
            [-2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, -2.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        for pt in &qr.points {
            elem.eval_curl(pt, &mut curl);
            for (i, exp) in expected.iter().enumerate() {
                for c in 0..3 {
                    let got = curl[i * 3 + c];
                    assert!(
                        (got - exp[c]).abs() < 1e-13,
                        "curl[{i}][{c}] = {got}, expected {}",
                        exp[c]
                    );
                }
            }
        }
    }

    /// Nodal basis property: DOF_j(Φᵢ) = δᵢⱼ.
    /// DOF = tangential component at edge midpoint × edge length.
    #[test]
    fn tet_nd1_nodal_basis() {
        let elem = TetND1;
        // Edge tangents (unit) and lengths
        let edges: [([f64; 3], [f64; 3]); 6] = [
            // (from, to)
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), // e₀₁
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // e₀₂
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), // e₀₃
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // e₁₂
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), // e₁₃
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), // e₂₃
        ];

        let mut vals = vec![0.0; 18];
        for (j, (from, to)) in edges.iter().enumerate() {
            let dx = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
            let len = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
            let t = [dx[0] / len, dx[1] / len, dx[2] / len];
            let mid = elem.dof_coords()[j].clone();
            elem.eval_basis_vec(&mid, &mut vals);
            for i in 0..6 {
                let dof =
                    (vals[i * 3] * t[0] + vals[i * 3 + 1] * t[1] + vals[i * 3 + 2] * t[2]) * len;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// Basis values at origin.
    #[test]
    fn tet_nd1_at_origin() {
        let elem = TetND1;
        let mut vals = vec![0.0; 18];
        elem.eval_basis_vec(&[0.0, 0.0, 0.0], &mut vals);
        // Φ₀₁(0) = (1,0,0)
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!(vals[1].abs() < 1e-14);
        assert!(vals[2].abs() < 1e-14);
        // Φ₀₂(0) = (0,1,0)
        assert!(vals[3].abs() < 1e-14);
        assert!((vals[4] - 1.0).abs() < 1e-14);
        assert!(vals[5].abs() < 1e-14);
        // Φ₀₃(0) = (0,0,1)
        assert!(vals[6].abs() < 1e-14);
        assert!(vals[7].abs() < 1e-14);
        assert!((vals[8] - 1.0).abs() < 1e-14);
    }
}
