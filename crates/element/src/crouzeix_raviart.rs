//! Crouzeix-Raviart (CR) nonconforming finite element.
//!
//! The CR triangle element has 3 DOFs at edge midpoints (one per edge).
//! Basis functions satisfy φ_i(m_j) = δ_ij where m_j is the midpoint of edge j.
//!
//! For the vector-valued version (Stokes velocity): 6 DOFs (2 per edge midpoint).
//!
//! ## CR1 scalar basis on the reference triangle
//!
//! Edge midpoints in (x,y) coordinates:
//!   m₀ = (0.5, 0) — edge opposite vertex 0 (between vertices 1 and 2)
//!   m₁ = (0, 0.5) — edge opposite vertex 1 (between vertices 0 and 2)
//!   m₂ = (0.5, 0.5) — edge opposite vertex 2 (between vertices 0 and 1)
//!
//! Basis functions: φ_i(x,y) = 1 - 2·s_i where s_i is the barycentric
//! coordinate of the vertex opposite edge i.  Equivalent to linear Lagrange
//! polynomials evaluated at midpoints.

use crate::VectorReferenceElement;

pub struct CrouzeixRaviart1 { _priv: () }

impl CrouzeixRaviart1 {
    pub fn new() -> Self { CrouzeixRaviart1 { _priv: () } }
}

/// Scalar CR1 basis on the reference triangle (3 DOFs at edge midpoints).
///
/// Reference triangle: v₀=(0,0), v₁=(1,0), v₂=(0,1)
/// Edge 0 (v₀-v₁, bottom): midpoint (0.5, 0)
/// Edge 1 (v₁-v₂, diagonal): midpoint (0.5, 0.5)
/// Edge 2 (v₀-v₂, left): midpoint (0, 0.5)
pub fn cr1_basis(xi: &[f64], vals: &mut [f64]) {
    vals[0] = 1.0 - 2.0 * xi[1];          // φ₀ at bottom edge midpoint
    vals[1] = 2.0 * (xi[0] + xi[1]) - 1.0; // φ₁ at diagonal edge midpoint
    vals[2] = 1.0 - 2.0 * xi[0];          // φ₂ at left edge midpoint
}

/// Scalar CR1 gradient on the reference triangle.
pub fn cr1_grad(_xi: &[f64], grads: &mut [f64]) {
    grads[0] = 0.0; grads[1] = -2.0;  // ∇φ₀
    grads[2] = 2.0; grads[3] = 2.0;   // ∇φ₁
    grads[4] = -2.0; grads[5] = 0.0;  // ∇φ₂
}

/// Vector CR1 element: 6 DOFs (2 components × 3 edges) for Stokes velocity.
pub struct CrouzeixRaviartVec1 { _priv: () }

impl CrouzeixRaviartVec1 {
    pub fn new() -> Self { CrouzeixRaviartVec1 { _priv: () } }
}

impl VectorReferenceElement for CrouzeixRaviartVec1 {
    fn n_dofs(&self) -> usize { 6 }
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 1 }

    fn quadrature(&self, order: u8) -> crate::QuadratureRule {
        crate::quadrature::tri_rule(order)
    }

    fn eval_basis_vec(&self, xi: &[f64], vals: &mut [f64]) {
        // 3 edge DOFs, 2 components each, interleaved: [u₀, v₀, u₁, v₁, u₂, v₂]
        let mut phi = [0.0_f64; 3];
        cr1_basis(xi, &mut phi);
        for i in 0..3 {
            vals[i * 2] = phi[i];     // x-component
            vals[i * 2 + 1] = phi[i]; // y-component (same basis)
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl: &mut [f64]) {
        // Φ_i = (φ_i, φ_i), curl = ∂φ/∂x - ∂φ/∂y
        // φ₀=1-2y: curl₀ = 0-(-2) = 2.  Wait, that makes no sense either.
        // Actually for the vector element Φ_i = (φ_i, 0) or (0, φ_i) depending.
        // Standard CR vector element pairs each scalar DOF with one component axis.
        // Let's use a simpler interleaving: [φ₀,0, φ₁,0, φ₂,0, 0,φ₀, 0,φ₁, 0,φ₂]
        // That's 12 DOFs, not 6. For 6 DOFs (standard Stokes CR1):
        // Interleaved by edge: [u₀, v₀, u₁, v₁, u₂, v₂] where each u_i, v_i share the same φ_i.
        // curl: ∂v_i/∂x - ∂u_i/∂y = 0 - 0 = 0 for our gradients? 
        // v_i = φ_i, ∂φ_i/∂x = 0 (for φ₀), -2 (for φ₂), 2 (for φ₁)
        // u_i = φ_i, ∂φ_i/∂y = -2 (for φ₀), 2 (for φ₁), 0 (for φ₂)
        // curl = ∂v/∂x - ∂u/∂y:
        // edge 0 (φ₀): curl = 0 - (-2) = 2
        // edge 1 (φ₁): curl = 2 - 2 = 0
        // edge 2 (φ₂): curl = -2 - 0 = -2
        for comp in 0..2 {
            curl[comp * 3] = 2.0;
            curl[comp * 3 + 1] = 0.0;
            curl[comp * 3 + 2] = -2.0;
        }
    }

    fn eval_div(&self, _xi: &[f64], div: &mut [f64]) {
        // div(φ_i, φ_i) = ∂φ_i/∂x + ∂φ_i/∂y
        // φ₀: 0 + (-2) = -2
        // φ₁: 2 + 2 = 4
        // φ₂: -2 + 0 = -2
        for i in 0..6 {
            div[i] = if i % 3 == 0 { -2.0 } else if i % 3 == 1 { 4.0 } else { -2.0 };
        }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        // 6 DOFs: 2 components × 3 edges, interleaved [u₀,v₀, u₁,v₁, u₂,v₂]
        vec![
            vec![0.5, 0.0], vec![0.5, 0.0],  // edge 0 (bottom)
            vec![0.5, 0.5], vec![0.5, 0.5],  // edge 1 (diagonal)
            vec![0.0, 0.5], vec![0.0, 0.5],  // edge 2 (left)
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cr1_basis_interpolates_midpoints() {
        let mut phi = [0.0; 3];
        // Edge 0 (bottom): (0.5, 0)
        cr1_basis(&[0.5, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14, "φ₀(0.5,0) = {}", phi[0]);
        assert!((phi[1] - 0.0).abs() < 1e-14, "φ₁(0.5,0) = {}", phi[1]);
        assert!((phi[2] - 0.0).abs() < 1e-14, "φ₂(0.5,0) = {}", phi[2]);

        // Edge 1 (diagonal): (0.5, 0.5)
        cr1_basis(&[0.5, 0.5], &mut phi);
        assert!((phi[0] - 0.0).abs() < 1e-14, "φ₀(0.5,0.5) = {}", phi[0]);
        assert!((phi[1] - 1.0).abs() < 1e-14, "φ₁(0.5,0.5) = {}", phi[1]);
        assert!((phi[2] - 0.0).abs() < 1e-14, "φ₂(0.5,0.5) = {}", phi[2]);

        // Edge 2 (left): (0, 0.5)
        cr1_basis(&[0.0, 0.5], &mut phi);
        assert!((phi[0] - 0.0).abs() < 1e-14, "φ₀(0,0.5) = {}", phi[0]);
        assert!((phi[1] - 0.0).abs() < 1e-14, "φ₁(0,0.5) = {}", phi[1]);
        assert!((phi[2] - 1.0).abs() < 1e-14, "φ₂(0,0.5) = {}", phi[2]);
    }

    #[test]
    fn cr1_sum_to_one() {
        let mut phi = [0.0; 3];
        cr1_basis(&[0.2, 0.3], &mut phi);
        let s: f64 = phi.iter().sum();
        assert!((s - 1.0).abs() < 1e-14, "sum = {}", s);
    }

    #[test]
    fn cr_vec_n_dofs() {
        let e = CrouzeixRaviartVec1::new();
        assert_eq!(e.n_dofs(), 6);
        assert_eq!(e.dim(), 2);
    }
}
