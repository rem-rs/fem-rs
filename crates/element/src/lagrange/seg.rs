//! Lagrange elements on the reference segment `[0, 1]`.

use crate::quadrature::seg_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── P1 ───────────────────────────────────────────────────────────────────────

/// Linear Lagrange element on `[0, 1]` — 2 DOFs at the vertices.
///
/// Basis:  φ₀ = 1 − ξ,  φ₁ = ξ
pub struct SegP1;

impl ReferenceElement for SegP1 {
    fn dim(&self)   -> u8    { 1 }
    fn order(&self) -> u8    { 1 }
    fn n_dofs(&self) -> usize { 2 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        values[0] = 1.0 - x;
        values[1] = x;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // grads[i*1 + 0] = ∂φᵢ/∂ξ
        grads[0] = -1.0;
        grads[1] =  1.0;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0]]
    }
}

// ─── P2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on `[0, 1]` — 3 DOFs: two vertices + midpoint.
///
/// DOF order: 0 (ξ=0), 1 (ξ=1), 2 (ξ=½)
///
/// Basis:
/// - φ₀ = (1−ξ)(1−2ξ)
/// - φ₁ = ξ(2ξ−1)
/// - φ₂ = 4ξ(1−ξ)
pub struct SegP2;

impl ReferenceElement for SegP2 {
    fn dim(&self)   -> u8    { 1 }
    fn order(&self) -> u8    { 2 }
    fn n_dofs(&self) -> usize { 3 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        values[0] = (1.0 - x) * (1.0 - 2.0 * x);
        values[1] = x * (2.0 * x - 1.0);
        values[2] = 4.0 * x * (1.0 - x);
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let x = xi[0];
        grads[0] = -3.0 + 4.0 * x;
        grads[1] =  4.0 * x - 1.0;
        grads[2] =  4.0 - 8.0 * x;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![0.5]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_partition_of_unity(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(elem.order() * 2);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "POU failed at {:?}: sum={s}", pt);
        }
    }

    fn check_grad_sum_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(elem.order() * 2);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-13, "grad sum d={d} != 0: {s} at {:?}", pt);
            }
        }
    }

    #[test]
    fn seg_p1_partition_of_unity() { check_partition_of_unity(&SegP1); }
    #[test]
    fn seg_p1_grad_sum_zero()      { check_grad_sum_zero(&SegP1); }
    #[test]
    fn seg_p2_partition_of_unity() { check_partition_of_unity(&SegP2); }
    #[test]
    fn seg_p2_grad_sum_zero()      { check_grad_sum_zero(&SegP2); }

    #[test]
    fn seg_p2_recovers_linear() {
        // φ₂ at ξ=0.5 should be 1.0; vertex functions should be 0.
        let mut phi = vec![0.0; 3];
        SegP2.eval_basis(&[0.5], &mut phi);
        assert!((phi[0]).abs() < 1e-14);
        assert!((phi[1]).abs() < 1e-14);
        assert!((phi[2] - 1.0).abs() < 1e-14);
    }
}
