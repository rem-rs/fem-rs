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

// ─── P2 ───────────────────────────────────────────────────────────────────────

/// Quadratic Lagrange element on the reference tetrahedron — 10 DOFs.
///
/// Barycentric coordinates: λ₁=1−ξ−η−ζ, λ₂=ξ, λ₃=η, λ₄=ζ
///
/// DOF ordering:
/// - 0: vertex (0,0,0)   — φ₀ = λ₁(2λ₁−1)
/// - 1: vertex (1,0,0)   — φ₁ = λ₂(2λ₂−1)
/// - 2: vertex (0,1,0)   — φ₂ = λ₃(2λ₃−1)
/// - 3: vertex (0,0,1)   — φ₃ = λ₄(2λ₄−1)
/// - 4: edge midpoint (1/2,0,0)   — φ₄ = 4λ₁λ₂
/// - 5: edge midpoint (0,1/2,0)   — φ₅ = 4λ₁λ₃
/// - 6: edge midpoint (0,0,1/2)   — φ₆ = 4λ₁λ₄
/// - 7: edge midpoint (1/2,1/2,0) — φ₇ = 4λ₂λ₃
/// - 8: edge midpoint (1/2,0,1/2) — φ₈ = 4λ₂λ₄
/// - 9: edge midpoint (0,1/2,1/2) — φ₉ = 4λ₃λ₄
pub struct TetP2;

impl ReferenceElement for TetP2 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize  { 10 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let l1 = 1.0 - x - y - z;
        let l2 = x;
        let l3 = y;
        let l4 = z;
        // Vertex DOFs
        values[0] = l1 * (2.0 * l1 - 1.0);
        values[1] = l2 * (2.0 * l2 - 1.0);
        values[2] = l3 * (2.0 * l3 - 1.0);
        values[3] = l4 * (2.0 * l4 - 1.0);
        // Edge DOFs
        values[4] = 4.0 * l1 * l2;
        values[5] = 4.0 * l1 * l3;
        values[6] = 4.0 * l1 * l4;
        values[7] = 4.0 * l2 * l3;
        values[8] = 4.0 * l2 * l4;
        values[9] = 4.0 * l3 * l4;
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads layout: row-major [10×3], grads[i*3 + j]
        // ∂λ₁/∂ξ=-1, ∂λ₁/∂η=-1, ∂λ₁/∂ζ=-1
        // ∂λ₂/∂ξ= 1, ∂λ₂/∂η= 0, ∂λ₂/∂ζ= 0
        // ∂λ₃/∂ξ= 0, ∂λ₃/∂η= 1, ∂λ₃/∂ζ= 0
        // ∂λ₄/∂ξ= 0, ∂λ₄/∂η= 0, ∂λ₄/∂ζ= 1
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let l1 = 1.0 - x - y - z;
        let l2 = x;
        let l3 = y;
        let l4 = z;
        // φ₀ = l1(2l1-1): ∂/∂λ₁ = 4l1-1; all partials = (4l1-1)*(-1)
        let d0 = 4.0 * l1 - 1.0;
        grads[0] = -d0;  grads[1] = -d0;  grads[2] = -d0;
        // φ₁ = l2(2l2-1): ∂/∂λ₂ = 4l2-1; ξ only
        let d1 = 4.0 * l2 - 1.0;
        grads[3] = d1;   grads[4] = 0.0;  grads[5] = 0.0;
        // φ₂ = l3(2l3-1): η only
        let d2 = 4.0 * l3 - 1.0;
        grads[6] = 0.0;  grads[7] = d2;   grads[8] = 0.0;
        // φ₃ = l4(2l4-1): ζ only
        let d3 = 4.0 * l4 - 1.0;
        grads[9] = 0.0;  grads[10] = 0.0; grads[11] = d3;
        // φ₄ = 4l1l2: ∂/∂ξ=4(l1-l2), ∂/∂η=-4l2, ∂/∂ζ=-4l2
        grads[12] = 4.0 * (l1 - l2);   grads[13] = -4.0 * l2;          grads[14] = -4.0 * l2;
        // φ₅ = 4l1l3: ∂/∂ξ=-4l3, ∂/∂η=4(l1-l3), ∂/∂ζ=-4l3
        grads[15] = -4.0 * l3;          grads[16] = 4.0 * (l1 - l3);   grads[17] = -4.0 * l3;
        // φ₆ = 4l1l4: ∂/∂ξ=-4l4, ∂/∂η=-4l4, ∂/∂ζ=4(l1-l4)
        grads[18] = -4.0 * l4;          grads[19] = -4.0 * l4;          grads[20] = 4.0 * (l1 - l4);
        // φ₇ = 4l2l3: ∂/∂ξ=4l3, ∂/∂η=4l2, ∂/∂ζ=0
        grads[21] = 4.0 * l3;           grads[22] = 4.0 * l2;           grads[23] = 0.0;
        // φ₈ = 4l2l4: ∂/∂ξ=4l4, ∂/∂η=0, ∂/∂ζ=4l2
        grads[24] = 4.0 * l4;           grads[25] = 0.0;                grads[26] = 4.0 * l2;
        // φ₉ = 4l3l4: ∂/∂ξ=0, ∂/∂η=4l4, ∂/∂ζ=4l3
        grads[27] = 0.0;                grads[28] = 4.0 * l4;           grads[29] = 4.0 * l3;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.5],
            vec![0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.0, 0.5, 0.5],
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

    #[test] fn tet_p2_pou()       { check_pou(&TetP2); }
    #[test] fn tet_p2_grad_zero() { check_grad_zero(&TetP2); }

    #[test]
    fn tet_p2_vertex_dofs() {
        let verts = [[0.0f64,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        for (vi, vp) in verts.iter().enumerate() {
            let mut phi = vec![0.0; 10];
            TetP2.eval_basis(vp, &mut phi);
            assert!((phi[vi] - 1.0).abs() < 1e-14, "vertex {vi}: phi={}", phi[vi]);
            for j in 0..10 { if j != vi { assert!(phi[j].abs() < 1e-14, "vertex {vi}, phi[{j}]={}", phi[j]); } }
        }
    }

    #[test]
    fn tet_p2_edge_midpoint_dofs() {
        // Edge midpoints and their DOF indices (DOFs 4-9)
        let edges = [
            ([0.5f64, 0.0, 0.0], 4usize),  // λ₁λ₂ midpoint
            ([0.0, 0.5, 0.0],    5),
            ([0.0, 0.0, 0.5],    6),
            ([0.5, 0.5, 0.0],    7),
            ([0.5, 0.0, 0.5],    8),
            ([0.0, 0.5, 0.5],    9),
        ];
        for (pt, di) in &edges {
            let mut phi = vec![0.0; 10];
            TetP2.eval_basis(pt, &mut phi);
            assert!((phi[*di] - 1.0).abs() < 1e-14, "edge dof {di}: phi={}", phi[*di]);
            for j in 0..10 { if j != *di { assert!(phi[j].abs() < 1e-13, "edge dof {di}, phi[{j}]={}", phi[j]); } }
        }
    }
}
