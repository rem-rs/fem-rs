//! Rannacher-Turek Q1_rot nonconforming quadrilateral element.
//!
//! Reference domain: [-1,1]². Shape functions: Span{1, x, y, x²-y²}.
//! DOFs: edge-average values (4 per edge). Vector version: 8 DOFs for Stokes.
//!
//! Built via Vandermonde from monomials + face-average DOFs with Gauss quadrature.

use crate::reference::{QuadratureRule, VectorReferenceElement};

const EDGE_GEOM: [([f64; 2], [f64; 2]); 4] = [
    ([-1.0, -1.0], [1.0, -1.0]),  // bottom
    ([1.0, -1.0],  [1.0,  1.0]),  // right
    ([-1.0,  1.0], [1.0,  1.0]),  // top   (parameterized -1→1)
    ([-1.0, -1.0], [-1.0, 1.0]),  // left
];

/// Monomial evaluation at (x, y).
fn eval_mm(x: f64, y: f64) -> [f64; 4] {
    [1.0, x, y, x*x - y*y]
}

/// Build 4×4 Vandermonde: DOF_i(m_j) where DOF is edge-average.
fn build_vm() -> [[f64; 4]; 4] {
    let (gp, gw) = gauss4();
    let mut v = [[0.0_f64; 4]; 4];
    for (ei, &(s, e)) in EDGE_GEOM.iter().enumerate() {
        let dx = e[0] - s[0];
        let dy = e[1] - s[1];
        let len = (dx*dx + dy*dy).sqrt();
        for (&t, &w) in gp.iter().zip(gw.iter()) {
            let x = s[0] + t * dx;
            let y = s[1] + t * dy;
            let m_vals = eval_mm(x, y);
            for j in 0..4 {
                // DOF_i(m_j) = ∫ edge m_j(edge(t)) dt / length
                v[ei][j] += w * m_vals[j] * len;
            }
        }
        // Divide by edge length for average
        for j in 0..4 { v[ei][j] /= len; }
    }
    v
}

fn gauss4() -> ([f64; 4], [f64; 4]) {
    ([0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970263],
     [0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269])
}

/// Invert 4×4 matrix.
fn inv4(mut a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut inv = [[0.0_f64; 4]; 4];
    for i in 0..4 { inv[i][i] = 1.0; }
    for c in 0..4 {
        let mut mr = c; let mut mv = a[c][c].abs();
        for r in (c+1)..4 { let x = a[r][c].abs(); if x > mv { mv = x; mr = r; } }
        a.swap(c, mr); inv.swap(c, mr);
        let ip = 1.0 / a[c][c];
        for j in 0..4 { a[c][j] *= ip; inv[c][j] *= ip; }
        for r in 0..4 { if r == c { continue; } let f = a[r][c];
            for j in 0..4 { a[r][j] -= f * a[c][j]; inv[r][j] -= f * inv[c][j]; }
        }
    }
    inv
}

fn coeff() -> &'static [[f64; 4]; 4] {
    use std::sync::OnceLock;
    static C: OnceLock<[[f64; 4]; 4]> = OnceLock::new();
    C.get_or_init(|| {
        let v = build_vm();
        let vi = inv4(v);
        // Transpose: C[i][j] = vi[j][i]
        let mut c = [[0.0_f64; 4]; 4];
        for i in 0..4 { for j in 0..4 { c[i][j] = vi[j][i]; } }
        c
    })
}

fn eval_all_monos(x: f64, y: f64) -> [f64; 4] { eval_mm(x, y) }

// ─── Scalar Q1_rot ─────────────────────────────────────────────────────────

pub struct QuadQ1Rot;

impl QuadQ1Rot {
    pub fn eval_basis(xi: &[f64], vals: &mut [f64]) {
        let c = coeff();
        let mv = eval_all_monos(xi[0], xi[1]);
        for i in 0..4 {
            vals[i] = c[i][0]*mv[0] + c[i][1]*mv[1] + c[i][2]*mv[2] + c[i][3]*mv[3];
        }
    }

    pub fn eval_grad_basis(xi: &[f64], grads: &mut [f64]) {
        let c = coeff();
        let (x, y) = (xi[0], xi[1]);
        let dm = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0*x, -2.0*y]];
        for i in 0..4 {
            grads[i*2]   = c[i][0]*dm[0][0] + c[i][1]*dm[1][0] + c[i][2]*dm[2][0] + c[i][3]*dm[3][0];
            grads[i*2+1] = c[i][0]*dm[0][1] + c[i][1]*dm[1][1] + c[i][2]*dm[2][1] + c[i][3]*dm[3][1];
        }
    }
}

// ─── Vector Q1_rot (8 DOFs: 2 comps × 4 edges) ─────────────────────────────

pub struct QuadQ1RotVec;

impl VectorReferenceElement for QuadQ1RotVec {
    fn n_dofs(&self) -> usize { 8 }
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 1 }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let o = (order as usize).max(3);
        let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(o);
        let nq = x1d.len();
        let mut pts = Vec::with_capacity(nq * nq);
        let mut wts = Vec::with_capacity(nq * nq);
        for i in 0..nq { for j in 0..nq { pts.push(vec![x1d[i], x1d[j]]); wts.push(w1d[i] * w1d[j]); }}
        QuadratureRule { points: pts, weights: wts }
    }

    fn eval_basis_vec(&self, xi: &[f64], vals: &mut [f64]) {
        let mut phi = [0.0_f64; 4];
        QuadQ1Rot::eval_basis(xi, &mut phi);
        for i in 0..4 { vals[i*2] = phi[i]; vals[i*2+1] = phi[i]; }
    }

    fn eval_curl(&self, xi: &[f64], curl: &mut [f64]) {
        let mut g = [0.0_f64; 8];
        QuadQ1Rot::eval_grad_basis(xi, &mut g);
        for i in 0..4 { curl[i] = g[i*2] + g[i*2+1]; } // dΦ/dx + dΦ/dy (since u=v=Φ)
    }

    fn eval_div(&self, xi: &[f64], div: &mut [f64]) {
        let mut g = [0.0_f64; 8];
        QuadQ1Rot::eval_grad_basis(xi, &mut g);
        for i in 0..4 { div[i] = g[i*2] + g[i*2+1]; }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0, -1.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]]
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn q1rot_partition_of_unity() {
        let mut phi = [0.0_f64; 4];
        for &p in &[[0.0, 0.0], [0.5, 0.3], [-0.7, 0.8], [1.0, -1.0]] {
            QuadQ1Rot::eval_basis(&p, &mut phi);
            assert!((phi.iter().sum::<f64>() - 1.0).abs() < 1e-12, "POU failed at {p:?}");
        }
    }

    #[test] fn q1rot_edge_avg_delta() {
        // Q1_rot DOFs are edge averages: ∫_edge_j φ_i dt / len_j = δ_ij.
        let (gp, gw) = gauss4();
        let mut phi = [0.0_f64; 4];
        for (ei, &(s, e)) in EDGE_GEOM.iter().enumerate() {
            let dx = e[0] - s[0]; let dy = e[1] - s[1];
            let len = (dx*dx + dy*dy).sqrt();
            let mut avg = [0.0_f64; 4];
            for (&t, &w) in gp.iter().zip(gw.iter()) {
                let pt = [s[0] + t*dx, s[1] + t*dy];
                QuadQ1Rot::eval_basis(&pt, &mut phi);
                for i in 0..4 { avg[i] += w * phi[i] * len; }
            }
            for i in 0..4 {
                let expected = if i == ei { 1.0 } else { 0.0 };
                assert!((avg[i] / len - expected).abs() < 1e-12,
                    "edge {ei} DOF {i}: avg={}", avg[i] / len);
            }
        }
    }

    #[test] fn q1rot_basis_values_at_origins() {
        let mut phi = [0.0_f64; 4];
        QuadQ1Rot::eval_basis(&[0.0, 0.0], &mut phi);
        for v in &phi { assert!(v.is_finite()); }
    }

    #[test] fn q1rot_vec_size() { assert_eq!(QuadQ1RotVec.n_dofs(), 8); }

    #[test] fn q1rot_vec_basis_finite() {
        let e = QuadQ1RotVec; let mut v = vec![0.0; 8];
        for p in &e.quadrature(3).points { e.eval_basis_vec(p, &mut v); for x in &v { assert!(x.is_finite()); } }
    }

    #[test] fn q1rot_vec_curl_finite() {
        let e = QuadQ1RotVec; let mut c = vec![0.0; 4];
        for p in &e.quadrature(3).points { e.eval_curl(p, &mut c); for x in &c { assert!(x.is_finite()); } }
    }
}
