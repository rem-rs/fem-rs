//! Lagrange elements on the reference hexahedron `[-1,1]³`.

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Q1 ───────────────────────────────────────────────────────────────────────

/// Trilinear Lagrange element on the reference hex `[-1,1]³` — 8 DOFs.
///
/// Node ordering: bottom face (z=−1) then top face (z=+1), each as a
/// counter-clockwise quad starting from (−1,−1).
///
/// | Index | (ξ, η, ζ)      |
/// |-------|----------------|
/// | 0     | (−1, −1, −1)   |
/// | 1     | (+1, −1, −1)   |
/// | 2     | (+1, +1, −1)   |
/// | 3     | (−1, +1, −1)   |
/// | 4     | (−1, −1, +1)   |
/// | 5     | (+1, −1, +1)   |
/// | 6     | (+1, +1, +1)   |
/// | 7     | (−1, +1, +1)   |
///
/// Basis: φᵢ = (1 + ξᵢ ξ)(1 + ηᵢ η)(1 + ζᵢ ζ) / 8
pub struct HexQ1;

const Q1_NODES: [(f64, f64, f64); 8] = [
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    ( 1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
    ( 1.0, -1.0,  1.0),
    ( 1.0,  1.0,  1.0),
    (-1.0,  1.0,  1.0),
];

impl ReferenceElement for HexQ1 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 8 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        for (i, &(xi_i, eta_i, zeta_i)) in Q1_NODES.iter().enumerate() {
            values[i] = 0.125
                * (1.0 + xi_i   * x)
                * (1.0 + eta_i  * y)
                * (1.0 + zeta_i * z);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        for (i, &(xi_i, eta_i, zeta_i)) in Q1_NODES.iter().enumerate() {
            let f_xi   = 1.0 + xi_i   * x;
            let f_eta  = 1.0 + eta_i  * y;
            let f_zeta = 1.0 + zeta_i * z;
            grads[i * 3]     = 0.125 * xi_i   * f_eta  * f_zeta;
            grads[i * 3 + 1] = 0.125 * eta_i  * f_xi   * f_zeta;
            grads[i * 3 + 2] = 0.125 * zeta_i * f_xi   * f_eta;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { hex_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q1_NODES.iter().map(|&(x, y, z)| vec![x, y, z]).collect()
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
            assert!((s - 1.0).abs() < 1e-12, "POU failed sum={s}");
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
                assert!(s.abs() < 1e-11, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn hex_q1_pou()       { check_pou(&HexQ1); }
    #[test] fn hex_q1_grad_zero() { check_grad_zero(&HexQ1); }

    #[test]
    fn hex_q1_node_dofs() {
        let mut phi = vec![0.0; 8];
        for (i, &(x, y, z)) in Q1_NODES.iter().enumerate() {
            HexQ1.eval_basis(&[x, y, z], &mut phi);
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }
}
