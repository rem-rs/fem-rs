//! Nedelec-I element on the reference triangular prism.
//!
//! Reference domain: (ξ, η, ζ) where (η,ζ) ∈ unit triangle and ξ ∈ [0,1].
//!
//! Vertices:
//!   V₀=(0,0,0)  V₁=(0,1,0)  V₂=(0,0,1)   — bottom triangle (ξ=0)
//!   V₃=(1,0,0)  V₄=(1,1,0)  V₅=(1,0,1)   — top triangle (ξ=1)
//!
//! Edges (9 total):
//!   E₀: V₀→V₁  E₁: V₀→V₂  E₂: V₁→V₂   — bottom triangle
//!   E₃: V₃→V₄  E₄: V₃→V₅  E₅: V₄→V₅   — top triangle
//!   E₆: V₀→V₃  E₇: V₁→V₄  E₈: V₂→V₅   — vertical
//!
//! # Whitney 1-form construction
//!
//! Barycentric-like coordinates λ₀…λ₅ (partition of unity on the prism):
//!   λ₀ = (1-ξ)(1-η-ζ)   λ₁ = (1-ξ)η      λ₂ = (1-ξ)ζ
//!   λ₃ = ξ(1-η-ζ)       λ₄ = ξη          λ₅ = ξζ
//!
//! Each basis function Φᵢⱼ = λᵢ∇λⱼ − λⱼ∇λᵢ is a Whitney 1-form
//! associated with edge (i→j).

use crate::quadrature::prism_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Nedelec first-kind H(curl) element on the reference triangular prism — 9 edge DOFs.
pub struct PrismND1;

/// Evalute the 6 barycentric-like coordinates and their gradients at (ξ,η,ζ).
fn barycentric(xi: f64, eta: f64, zeta: f64) -> ([f64; 6], [[f64; 3]; 6]) {
    let a = 1.0 - xi;
    let b = 1.0 - eta - zeta;
    // λ values
    let lam = [
        a * b,           // λ₀ = (1-ξ)(1-η-ζ)
        a * eta,         // λ₁ = (1-ξ)η
        a * zeta,        // λ₂ = (1-ξ)ζ
        xi * b,          // λ₃ = ξ(1-η-ζ)
        xi * eta,        // λ₄ = ξη
        xi * zeta,       // λ₅ = ξζ
    ];
    // Gradients ∇λ = (∂/∂ξ, ∂/∂η, ∂/∂ζ)
    let grad = [
        [-b, -a, -a],           // ∇λ₀
        [-eta, a, 0.0],         // ∇λ₁
        [-zeta, 0.0, a],        // ∇λ₂
        [b, -xi, -xi],          // ∇λ₃
        [eta, xi, 0.0],         // ∇λ₄
        [zeta, 0.0, xi],        // ∇λ₅
    ];
    (lam, grad)
}

impl VectorReferenceElement for PrismND1 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 9 }

    /// values[i*3 + c] = component c of basis function i.
    ///
    /// DOF → edge mapping:
    ///   0:E₀  1:E₁  2:E₂  3:E₃  4:E₄  5:E₅  6:E₆  7:E₇  8:E₈
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (lam, grad) = barycentric(xi[0], xi[1], xi[2]);

        // Edge index → (i,j) vertex pair from the Whitney form
        let edges: [(usize, usize); 9] = [
            (0, 1), (0, 2), (1, 2),
            (3, 4), (3, 5), (4, 5),
            (0, 3), (1, 4), (2, 5),
        ];

        for (e, &(i, j)) in edges.iter().enumerate() {
            let li = lam[i]; let lj = lam[j];
            let gix = grad[i]; let gjx = grad[j];
            values[e * 3]     = li * gjx[0] - lj * gix[0];
            values[e * 3 + 1] = li * gjx[1] - lj * gix[1];
            values[e * 3 + 2] = li * gjx[2] - lj * gix[2];
        }
    }

    /// curl of each Whitney 1-form: curl(λᵢ∇λⱼ − λⱼ∇λᵢ) = 2(∇λᵢ × ∇λⱼ)
    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let (_, grad) = barycentric(xi[0], xi[1], xi[2]);
        let edges: [(usize, usize); 9] = [
            (0, 1), (0, 2), (1, 2),
            (3, 4), (3, 5), (4, 5),
            (0, 3), (1, 4), (2, 5),
        ];

        for (e, &(i, j)) in edges.iter().enumerate() {
            let gi = grad[i];
            let gj = grad[j];
            // ∇λᵢ × ∇λⱼ  (cross product)
            let cx = gi[1]*gj[2] - gi[2]*gj[1];
            let cy = gi[2]*gj[0] - gi[0]*gj[2];
            let cz = gi[0]*gj[1] - gi[1]*gj[0];
            curl_vals[e * 3]     = 2.0 * cx;
            curl_vals[e * 3 + 1] = 2.0 * cy;
            curl_vals[e * 3 + 2] = 2.0 * cz;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { prism_rule(order) }

    /// DOF sites: edge midpoints of the 9 edges.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.5, 0.0],  // E₀: V₀→V₁
            vec![0.0, 0.0, 0.5],  // E₁: V₀→V₂
            vec![0.0, 0.5, 0.5],  // E₂: V₁→V₂
            vec![1.0, 0.5, 0.0],  // E₃: V₃→V₄
            vec![1.0, 0.0, 0.5],  // E₄: V₃→V₅
            vec![1.0, 0.5, 0.5],  // E₅: V₄→V₅
            vec![0.5, 0.0, 0.0],  // E₆: V₀→V₃
            vec![0.5, 1.0, 0.0],  // E₇: V₁→V₄
            vec![0.5, 0.0, 1.0],  // E₈: V₂→V₅
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prism_nd1_basis_finite() {
        let elem = PrismND1;
        let mut vals = vec![0.0; 27];
        let pts = &[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [0.5, 0.25, 0.25], [0.2, 0.3, 0.1],
        ];
        for p in pts {
            elem.eval_basis_vec(p, &mut vals);
            for v in vals.iter() { assert!(v.is_finite(), "non-finite at {p:?}"); }
        }
    }

    #[test]
    fn prism_nd1_curl_finite() {
        let elem = PrismND1;
        let mut curl = vec![0.0; 27];
        let qr = elem.quadrature(2);
        for xi in &qr.points {
            elem.eval_curl(xi, &mut curl);
            for v in curl.iter() { assert!(v.is_finite(), "non-finite curl at {xi:?}"); }
        }
    }

    /// Nodal basis: DOF_j(Φ_i) = δ_{ij}, where DOF_j = ∫_{E_j} Φ·t̂ ds.
    /// For the lowest-order element, the integrand is linear in the param,
    /// so 2-point Gauss quadrature is exact.
    #[test]
    fn prism_nd1_nodal_basis() {
        let elem = PrismND1;
        // Edge definitions: (start, end)
        let edge_data: [([f64; 3], [f64; 3]); 9] = [
            ([0.0,0.0,0.0], [0.0,1.0,0.0]),  // E₀
            ([0.0,0.0,0.0], [0.0,0.0,1.0]),  // E₁
            ([0.0,1.0,0.0], [0.0,0.0,1.0]),  // E₂
            ([1.0,0.0,0.0], [1.0,1.0,0.0]),  // E₃
            ([1.0,0.0,0.0], [1.0,0.0,1.0]),  // E₄
            ([1.0,1.0,0.0], [1.0,0.0,1.0]),  // E₅
            ([0.0,0.0,0.0], [1.0,0.0,0.0]),  // E₆
            ([0.0,1.0,0.0], [1.0,1.0,0.0]),  // E₇
            ([0.0,0.0,1.0], [1.0,0.0,1.0]),  // E₈
        ];
        // 2-point Gauss-Legendre on [0,1]
        let gl_pts = [0.21132486540518713, 0.7886751345948129];
        let gl_wts = [0.5, 0.5];

        let mut vals = vec![0.0; 27];
        for (j, (start, end)) in edge_data.iter().enumerate() {
            let dx = [end[0]-start[0], end[1]-start[1], end[2]-start[2]];
            let mut moments = [0.0f64; 9];
            for (&t, &w) in gl_pts.iter().zip(gl_wts.iter()) {
                let pt = [
                    start[0] + t*dx[0],
                    start[1] + t*dx[1],
                    start[2] + t*dx[2],
                ];
                elem.eval_basis_vec(&pt, &mut vals);
                for i in 0..9 {
                    let tang = vals[i*3]*dx[0] + vals[i*3+1]*dx[1] + vals[i*3+2]*dx[2];
                    moments[i] += w * tang;
                }
            }
            for i in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (moments[i] - expected).abs() < 1e-12,
                    "DOF_{j}(Φ_{i}) = {}, expected {expected}", moments[i]
                );
            }
        }
    }

    #[test]
    fn prism_nd1_n_dofs() {
        assert_eq!(PrismND1.n_dofs(), 9);
    }
}
