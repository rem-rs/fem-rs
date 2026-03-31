//! Lagrange elements on the reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
//!
//! Barycentric coordinates: λ₁=1−ξ−η−ζ, λ₂=ξ, λ₃=η, λ₄=ζ

use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── P1 ───────────────────────────────────────────────────────────────────────

/// Linear Lagrange element on the reference tetrahedron — 4 DOFs at vertices.
///
/// Basis:
/// - φ₀ = 1−ξ−η−ζ  (vertex (0,0,0))
/// - φ₁ = ξ          (vertex (1,0,0))
/// - φ₂ = η          (vertex (0,1,0))
/// - φ₃ = ζ          (vertex (0,0,1))
pub struct TetP1;

impl ReferenceElement for TetP1 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 4 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        values[0] = 1.0 - x - y - z;
        values[1] = x;
        values[2] = y;
        values[3] = z;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // row-major [4×3]: grads[i*3 + j]
        // ∇φ₀ = (-1,-1,-1)
        grads[0]  = -1.0;  grads[1]  = -1.0;  grads[2]  = -1.0;
        // ∇φ₁ = (1,0,0)
        grads[3]  =  1.0;  grads[4]  =  0.0;  grads[5]  =  0.0;
        // ∇φ₂ = (0,1,0)
        grads[6]  =  0.0;  grads[7]  =  1.0;  grads[8]  =  0.0;
        // ∇φ₃ = (0,0,1)
        grads[9]  =  0.0;  grads[10] =  0.0;  grads[11] =  1.0;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(4);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(4);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-13, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn tet_p1_pou()       { check_pou(&TetP1); }
    #[test] fn tet_p1_grad_zero() { check_grad_zero(&TetP1); }

    #[test]
    fn tet_p1_vertex_dofs() {
        let mut phi = vec![0.0; 4];
        TetP1.eval_basis(&[0.0, 0.0, 0.0], &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        for i in 1..4 { assert!(phi[i].abs() < 1e-14); }

        TetP1.eval_basis(&[1.0, 0.0, 0.0], &mut phi);
        assert!(phi[0].abs() < 1e-14);
        assert!((phi[1] - 1.0).abs() < 1e-14);
    }
}
