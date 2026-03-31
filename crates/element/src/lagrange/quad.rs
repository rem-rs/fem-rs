//! Lagrange elements on the reference quadrilateral `[-1,1]²`.

use crate::quadrature::quad_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Q1 ───────────────────────────────────────────────────────────────────────

/// Bilinear Lagrange element on the reference quad `[-1,1]²` — 4 DOFs.
///
/// Node ordering (counter-clockwise):
/// - 0: (−1,−1)
/// - 1: (+1,−1)
/// - 2: (+1,+1)
/// - 3: (−1,+1)
///
/// Basis: φᵢ = (1 + ξᵢ ξ)(1 + ηᵢ η) / 4
pub struct QuadQ1;

/// Node coordinates (ξ, η) of the 4 Q1 nodes.
const Q1_NODES: [(f64, f64); 4] = [
    (-1.0, -1.0),
    ( 1.0, -1.0),
    ( 1.0,  1.0),
    (-1.0,  1.0),
];

impl ReferenceElement for QuadQ1 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 4 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in Q1_NODES.iter().enumerate() {
            values[i] = 0.25 * (1.0 + xi_i * x) * (1.0 + eta_i * y);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in Q1_NODES.iter().enumerate() {
            grads[i * 2]     = 0.25 * xi_i  * (1.0 + eta_i * y);
            grads[i * 2 + 1] = 0.25 * eta_i * (1.0 + xi_i  * x);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q1_NODES.iter().map(|&(x, y)| vec![x, y]).collect()
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
                assert!(s.abs() < 1e-12, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn quad_q1_pou()       { check_pou(&QuadQ1); }
    #[test] fn quad_q1_grad_zero() { check_grad_zero(&QuadQ1); }

    #[test]
    fn quad_q1_node_dofs() {
        let mut phi = vec![0.0; 4];
        for (i, &(x, y)) in Q1_NODES.iter().enumerate() {
            QuadQ1.eval_basis(&[x, y], &mut phi);
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-14,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }
}
