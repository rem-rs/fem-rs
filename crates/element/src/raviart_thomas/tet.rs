//! Raviart-Thomas RT0 element on the reference tetrahedron.
//!
//! Reference vertices: v₀=(0,0,0), v₁=(1,0,0), v₂=(0,1,0), v₃=(0,0,1).
//!
//! # Faces (and DOF association)
//! | DOF | Face        | Opposite vertex | Outward normal        | Area  |
//! |-----|-------------|-----------------|-----------------------|-------|
//! | 0   | f₀₁₂₃      | v₀              | (1,1,1)/√3            | √3/2  |
//! | 1   | f₀₀₂₃      | v₁              | (−1,0,0)              | 1/2   |
//! | 2   | f₀₀₁₃      | v₂              | (0,−1,0)              | 1/2   |
//! | 3   | f₀₀₁₂      | v₃              | (0,0,−1)              | 1/2   |
//!
//! # Basis functions
//! The four RT0 basis functions on the reference tetrahedron are:
//!
//! ```text
//! Φ₀ = (ξ,   η,   ζ  )              — face f₀ (opposite v₀; normal points out)
//! Φ₁ = (ξ−1, η,   ζ  )              — face f₁ (opposite v₁)
//! Φ₂ = (ξ,   η−1, ζ  )              — face f₂ (opposite v₂)
//! Φ₃ = (ξ,   η,   ζ−1)              — face f₃ (opposite v₃)
//! ```
//!
//! Each has divergence = 3 (constant), and the integral over the reference
//! tetrahedron (volume 1/6) is 3×(1/6) = 1/2, matching the face area of
//! the three coordinate faces; for face f₀ (area √3/2) a scaling factor
//! is absorbed into the DOF definition.
//!
//! # DOF definition
//! `DOF_j(u) = ∫_{fⱼ} u · n̂ⱼ ds`  (outward normal flux through face j).

use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT0 H(div) element on the reference tetrahedron — 4 face DOFs.
///
/// Reference domain: tetrahedron with vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1).
///
/// Basis functions (scaled so that `DOF_j(Φᵢ) = δᵢⱼ`):
/// - Φ₀ = 2(ξ,   η,   ζ  )   — face opposite v₀
/// - Φ₁ = 2(ξ−1, η,   ζ  )   — face opposite v₁
/// - Φ₂ = 2(ξ,   η−1, ζ  )   — face opposite v₂
/// - Φ₃ = 2(ξ,   η,   ζ−1)   — face opposite v₃
///
/// Each has **div = 6**.
pub struct TetRT0;

impl VectorReferenceElement for TetRT0 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 0 }
    fn n_dofs(&self) -> usize  { 4 }

    /// `values[i*3 + c]` = component c of RT0 basis function i.
    ///
    /// Basis matches MFEM `RT_TetrahedronElement(0)` (fe_rt.cpp CalcVShape:
    /// T⁻¹ interpolation with nk = {1,1,1}/{-1,0,0}/{0,-1,0}/{0,0,-1}, face
    /// points (1/3,1/3,1/3) etc., c = 1/4):
    ///   Φ₀ = (x, y, z), Φ₁ = (x−1, y, z), Φ₂ = (x, y−1, z), Φ₃ = (x, y, z−1)
    /// (Rust used to scale these by 2 — that gave 4× mass entries vs MFEM.)
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        // Φ₀ = (x, y, z)
        values[0] = x;           values[1]  = y;           values[2]  = z;
        // Φ₁ = (x−1, y, z)
        values[3] = x-1.0;       values[4]  = y;           values[5]  = z;
        // Φ₂ = (x, y−1, z)
        values[6] = x;           values[7]  = y-1.0;       values[8]  = z;
        // Φ₃ = (x, y, z−1)
        values[9] = x;           values[10] = y;           values[11] = z-1.0;
    }

    /// Curl is not the natural operator for H(div) — returns zeros.
    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    /// Divergence: 3 for all RT0 basis functions (MFEM: div of (x,y,z) etc.).
    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() { *v = 3.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    /// DOF sites: centroids of the four faces.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![1.0/3.0, 1.0/3.0, 1.0/3.0], // centroid of f₀: v₁v₂v₃
            vec![0.0,     1.0/3.0, 1.0/3.0], // centroid of f₁: v₀v₂v₃
            vec![1.0/3.0, 0.0,     1.0/3.0], // centroid of f₂: v₀v₁v₃
            vec![1.0/3.0, 1.0/3.0, 0.0    ], // centroid of f₃: v₀v₁v₂
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// div = 3 for all RT0 basis functions (MFEM RT_TetrahedronElement(0)).
    #[test]
    fn tet_rt0_div_constant() {
        let elem = TetRT0;
        let mut div = vec![0.0; 4];
        for pt in &elem.quadrature(4).points {
            elem.eval_div(pt, &mut div);
            for (i, &d) in div.iter().enumerate() {
                assert!((d - 3.0).abs() < 1e-13, "div[{i}] = {d}");
            }
        }
    }

    /// MFEM RT0 interpolation DOFs: dof_j(Φᵢ) = δᵢⱼ where
    /// dof_j(u) = u(ip_j)·nk_j with the face interpolation points
    /// (1/3,1/3,1/3), (0,1/3,1/3), (1/3,0,1/3), (1/3,1/3,0) and
    /// nk = (1,1,1), (-1,0,0), (0,-1,0), (0,0,-1).
    #[test]
    fn tet_rt0_nodal_basis() {
        let elem = TetRT0;
        let ips: [[f64; 3]; 4] = [
            [1.0/3.0, 1.0/3.0, 1.0/3.0],
            [0.0,     1.0/3.0, 1.0/3.0],
            [1.0/3.0, 0.0,     1.0/3.0],
            [1.0/3.0, 1.0/3.0, 0.0],
        ];
        let nk: [[f64; 3]; 4] = [
            [1.0, 1.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
        ];
        let mut vals = vec![0.0; 12];
        for (j, ip) in ips.iter().enumerate() {
            elem.eval_basis_vec(ip, &mut vals);
            for i in 0..4 {
                let dof = vals[i*3]*nk[j][0] + vals[i*3+1]*nk[j][1] + vals[i*3+2]*nk[j][2];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// Divergence theorem: ∫∫∫ div Φᵢ dV = 3 × (1/6) = 1/2.
    #[test]
    fn tet_rt0_div_integral() {
        let elem = TetRT0;
        let qr = elem.quadrature(4);
        let mut div = vec![0.0; 4];
        for i in 0..4 {
            let mut integral = 0.0;
            for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
                elem.eval_div(pt, &mut div);
                integral += div[i] * w;
            }
            // 3 × (1/6) = 1/2
            assert!((integral - 0.5).abs() < 1e-12, "∫div Φ_{i} = {integral}");
        }
    }
}
