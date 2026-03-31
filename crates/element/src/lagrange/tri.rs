//! Lagrange elements on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! Barycentric coordinates: λ₁ = 1−ξ−η,  λ₂ = ξ,  λ₃ = η

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── P1 ───────────────────────────────────────────────────────────────────────

/// Linear Lagrange element on the reference triangle — 3 DOFs at vertices.
///
/// Basis:
/// - φ₀ = 1−ξ−η  (vertex 0: origin)
/// - φ₁ = ξ       (vertex 1: (1,0))
/// - φ₂ = η       (vertex 2: (0,1))
pub struct TriP1;

impl ReferenceElement for TriP1 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 3 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        values[0] = 1.0 - x - y;
        values[1] = x;
        values[2] = y;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // row-major [3×2]: grads[i*2 + j]
        grads[0] = -1.0;  grads[1] = -1.0;  // ∇φ₀
        grads[2] =  1.0;  grads[3] =  0.0;  // ∇φ₁
        grads[4] =  0.0;  grads[5] =  1.0;  // ∇φ₂
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]
    }
}

// ─── P2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on the reference triangle — 6 DOFs.
///
/// DOF ordering:
/// - 0: vertex (0,0)   — φ₀ = λ₁(2λ₁−1)
/// - 1: vertex (1,0)   — φ₁ = λ₂(2λ₂−1)
/// - 2: vertex (0,1)   — φ₂ = λ₃(2λ₃−1)
/// - 3: edge midpoint (0.5, 0)   — φ₃ = 4λ₁λ₂
/// - 4: edge midpoint (0.5, 0.5) — φ₄ = 4λ₂λ₃
/// - 5: edge midpoint (0, 0.5)   — φ₅ = 4λ₁λ₃
pub struct TriP2;

impl ReferenceElement for TriP2 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize  { 6 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let l1 = 1.0 - x - y;
        let l2 = x;
        let l3 = y;
        values[0] = l1 * (2.0 * l1 - 1.0);
        values[1] = l2 * (2.0 * l2 - 1.0);
        values[2] = l3 * (2.0 * l3 - 1.0);
        values[3] = 4.0 * l1 * l2;
        values[4] = 4.0 * l2 * l3;
        values[5] = 4.0 * l1 * l3;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        // ∂φ₀/∂ξ = (4ξ+4η−3),  ∂φ₀/∂η = same
        grads[0]  = 4.0 * x + 4.0 * y - 3.0;
        grads[1]  = 4.0 * x + 4.0 * y - 3.0;
        // ∂φ₁/∂ξ = 4ξ−1,  ∂φ₁/∂η = 0
        grads[2]  = 4.0 * x - 1.0;
        grads[3]  = 0.0;
        // ∂φ₂/∂ξ = 0,  ∂φ₂/∂η = 4η−1
        grads[4]  = 0.0;
        grads[5]  = 4.0 * y - 1.0;
        // ∂φ₃/∂ξ = 4(1−2ξ−η),  ∂φ₃/∂η = −4ξ
        grads[6]  = 4.0 * (1.0 - 2.0 * x - y);
        grads[7]  = -4.0 * x;
        // ∂φ₄/∂ξ = 4η,  ∂φ₄/∂η = 4ξ
        grads[8]  = 4.0 * y;
        grads[9]  = 4.0 * x;
        // ∂φ₅/∂ξ = −4η,  ∂φ₅/∂η = 4(1−ξ−2η)
        grads[10] = -4.0 * y;
        grads[11] = 4.0 * (1.0 - x - 2.0 * y);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
            vec![0.5, 0.0], vec![0.5, 0.5], vec![0.0, 0.5],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(5);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(5);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-12, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn tri_p1_pou()       { check_pou(&TriP1); }
    #[test] fn tri_p1_grad_zero() { check_grad_zero(&TriP1); }
    #[test] fn tri_p2_pou()       { check_pou(&TriP2); }
    #[test] fn tri_p2_grad_zero() { check_grad_zero(&TriP2); }

    #[test]
    fn tri_p1_vertex_dofs() {
        let mut phi = vec![0.0; 3];
        TriP1.eval_basis(&[0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        assert!(phi[1].abs() < 1e-14);
        assert!(phi[2].abs() < 1e-14);

        TriP1.eval_basis(&[1.0, 0.0], &mut phi);
        assert!(phi[0].abs() < 1e-14);
        assert!((phi[1] - 1.0).abs() < 1e-14);
        assert!(phi[2].abs() < 1e-14);
    }

    #[test]
    fn tri_p2_vertex_and_edge_dofs() {
        let mut phi = vec![0.0; 6];
        // At vertex 0: φ₀=1, rest=0
        TriP2.eval_basis(&[0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        for i in 1..6 { assert!(phi[i].abs() < 1e-14, "phi[{i}]={}", phi[i]); }
        // At edge midpoint (0.5, 0): φ₃=1, rest=0
        TriP2.eval_basis(&[0.5, 0.0], &mut phi);
        for i in [0, 1, 2, 4, 5] { assert!(phi[i].abs() < 1e-14, "phi[{i}]={}", phi[i]); }
        assert!((phi[3] - 1.0).abs() < 1e-14);
    }
}
