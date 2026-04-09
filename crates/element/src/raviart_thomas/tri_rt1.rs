//! Raviart-Thomas RT1 element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! # Space
//! `RT₁ = P₁² ⊕ x P̃₁`  where `x P̃₁ = { (ξ·p, η·p) : p ∈ P̃₁ }`
//! dim = 6 + 2 = 8.
//!
//! # DOF functionals (8 total)
//! 2 normal-flux moments per edge (3 edges × 2 = 6) + 2 interior moments:
//!
//! | DOF | Location | Functional |
//! |-----|----------|------------|
//! | 0   | edge f₀ (v₁v₂, hyp.) | ∫ Φ·n̂₀ dσ     (n̂₀=(1,1)/√2, len=√2) |
//! | 1   | edge f₀  | ∫ Φ·n̂₀ · t dσ  (t = param along edge) |
//! | 2   | edge f₁ (v₀v₂, left) | ∫ Φ·n̂₁ dσ   (n̂₁=(-1,0)) |
//! | 3   | edge f₁  | ∫ Φ·n̂₁ · t dσ |
//! | 4   | edge f₂ (v₀v₁, bot.) | ∫ Φ·n̂₂ dσ   (n̂₂=(0,-1)) |
//! | 5   | edge f₂  | ∫ Φ·n̂₂ · t dσ |
//! | 6   | interior | ∫_T Φ_x dA |
//! | 7   | interior | ∫_T Φ_y dA |

use std::sync::OnceLock;

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

static COEFF: OnceLock<[[f64; 8]; 8]> = OnceLock::new();

/// Evaluate the 8 RT1 monomials at (x,y).
/// P₁² monomials: (1,0),(ξ,0),(η,0),(0,1),(0,ξ),(0,η)
/// x P̃₁ monomials: (ξ²,ξη),(ξη,η²)
fn eval_monomials(x: f64, y: f64, vals: &mut [f64; 16]) {
    vals[0] = 1.0; vals[1] = 0.0;   // (1,0)
    vals[2] = x;   vals[3] = 0.0;   // (ξ,0)
    vals[4] = y;   vals[5] = 0.0;   // (η,0)
    vals[6] = 0.0; vals[7] = 1.0;   // (0,1)
    vals[8] = 0.0; vals[9] = x;     // (0,ξ)
    vals[10] = 0.0; vals[11] = y;   // (0,η)
    vals[12] = x*x; vals[13] = x*y; // (ξ²,ξη)
    vals[14] = x*y; vals[15] = y*y; // (ξη,η²)
}

/// div(m_j) = ∂m_j_x/∂ξ + ∂m_j_y/∂η
fn eval_monomial_divs(x: f64, y: f64, divs: &mut [f64; 8]) {
    divs[0] = 0.0; // div(1,0)=0
    divs[1] = 1.0; // div(ξ,0)=1
    divs[2] = 0.0; // div(η,0)=0
    divs[3] = 0.0; // div(0,1)=0
    divs[4] = 0.0; // div(0,ξ)=0
    divs[5] = 1.0; // div(0,η)=1
    divs[6] = 2.0*x + y; // div(ξ²,ξη) = 2ξ + ξ = 3ξ? wait: ∂ξ²/∂ξ + ∂ξη/∂η = 2ξ + ξ = 3ξ
    // Actually: ∂(ξ²)/∂ξ = 2ξ, ∂(ξη)/∂η = ξ → div = 2ξ + ξ = 3ξ
    divs[6] = 3.0*x;
    // div(ξη, η²) = ∂ξη/∂ξ + ∂η²/∂η = η + 2η = 3η
    divs[7] = 3.0*y;
}

/// Build 8×8 Vandermonde matrix V[k][j] = DOF_k(m_j).
fn build_vandermonde() -> [[f64; 8]; 8] {
    let mut v = [[0.0f64; 8]; 8];

    // 4-point Gauss-Legendre on [0,1]
    let sq6_5 = (6.0f64 / 5.0).sqrt();
    let ta = ((3.0 - 2.0 * sq6_5) / 7.0).sqrt();
    let tb = ((3.0 + 2.0 * sq6_5) / 7.0).sqrt();
    let wa = (18.0 + 30.0f64.sqrt()) / 36.0;
    let wb = (18.0 - 30.0f64.sqrt()) / 36.0;
    let gl_pts = [0.5*(1.0-tb), 0.5*(1.0-ta), 0.5*(1.0+ta), 0.5*(1.0+tb)];
    let gl_wts = [0.5*wb, 0.5*wa, 0.5*wa, 0.5*wb];

    let mut mono = [0.0f64; 16];

    // --- Edge f₀: hypotenuse v₁→v₂, param t: (1-t, t), n̂=(1,1), edge length=√2 ---
    // DOF_0 = ∫₀¹ [(m_j)_x + (m_j)_y](1-t,t) · (1/√2) · √2 dt = ∫₀¹ [(m_j)_x+(m_j)_y](1-t,t) dt
    // DOF_1 = ∫₀¹ [(m_j)_x + (m_j)_y](1-t,t) · t dt
    // (The √2 from edge length and 1/√2 from unit normal cancel.)
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(1.0-t, t, &mut mono);
        for j in 0..8 {
            let nflux = mono[j*2] + mono[j*2+1]; // (n_x=1, n_y=1) dotted (unnormalized)
            v[0][j] += w * nflux;
            v[1][j] += w * nflux * t;
        }
    }
    // --- Edge f₁: left edge v₀→v₂, param t: (0,t), n̂=(-1,0), length=1 ---
    // DOF_2 = ∫₀¹ (−(m_j)_x)(0,t) dt
    // DOF_3 = ∫₀¹ (−(m_j)_x)(0,t) · t dt
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(0.0, t, &mut mono);
        for j in 0..8 {
            let nflux = -mono[j*2]; // n=(−1,0)
            v[2][j] += w * nflux;
            v[3][j] += w * nflux * t;
        }
    }
    // --- Edge f₂: bottom edge v₀→v₁, param t: (t,0), n̂=(0,-1), length=1 ---
    // DOF_4 = ∫₀¹ (−(m_j)_y)(t,0) dt
    // DOF_5 = ∫₀¹ (−(m_j)_y)(t,0) · t dt
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(t, 0.0, &mut mono);
        for j in 0..8 {
            let nflux = -mono[j*2+1]; // n=(0,−1)
            v[4][j] += w * nflux;
            v[5][j] += w * nflux * t;
        }
    }
    // --- Interior DOFs: ∫_T (m_j)_x dA and ∫_T (m_j)_y dA ---
    let qr = tri_rule(5);
    for (xi, w) in qr.points.iter().zip(qr.weights.iter()) {
        eval_monomials(xi[0], xi[1], &mut mono);
        for j in 0..8 {
            v[6][j] += w * mono[j*2];
            v[7][j] += w * mono[j*2+1];
        }
    }

    v
}

fn invert_8x8(a: [[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 16]; 8];
    for i in 0..8 {
        for j in 0..8 { m[i][j] = a[i][j]; }
        m[i][8 + i] = 1.0;
    }
    for col in 0..8 {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..8 {
            if m[row][col].abs() > max_val { max_val = m[row][col].abs(); max_row = row; }
        }
        m.swap(col, max_row);
        let inv = 1.0 / m[col][col];
        assert!(inv.is_finite(), "TriRT1 Vandermonde matrix is singular");
        for j in 0..16 { m[col][j] *= inv; }
        for row in 0..8 {
            if row == col { continue; }
            let f = m[row][col];
            for j in 0..16 { let d = f * m[col][j]; m[row][j] -= d; }
        }
    }
    let mut r = [[0.0f64; 8]; 8];
    for i in 0..8 { for j in 0..8 { r[i][j] = m[i][8+j]; } }
    r
}

fn transpose_8x8(a: [[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut t = [[0.0f64; 8]; 8];
    for i in 0..8 { for j in 0..8 { t[i][j] = a[j][i]; } }
    t
}

fn coeff() -> &'static [[f64; 8]; 8] {
    COEFF.get_or_init(|| transpose_8x8(invert_8x8(build_vandermonde())))
}

// ─── TriRT1 ──────────────────────────────────────────────────────────────────

/// Raviart-Thomas RT1 H(div) element on the reference triangle — 8 DOFs, order 1.
pub struct TriRT1;

impl VectorReferenceElement for TriRT1 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize { 8 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();
        let mut mono = [0.0f64; 16];
        eval_monomials(x, y, &mut mono);
        for i in 0..8 {
            let mut vx = 0.0; let mut vy = 0.0;
            for j in 0..8 {
                vx += c[i][j] * mono[j*2];
                vy += c[i][j] * mono[j*2+1];
            }
            values[i*2]   = vx;
            values[i*2+1] = vy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();
        let mut md = [0.0f64; 8];
        eval_monomial_divs(x, y, &mut md);
        for i in 0..8 {
            let mut s = 0.0;
            for j in 0..8 { s += c[i][j] * md[j]; }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            // edge f₀: hypotenuse, params 1/3 and 2/3 → (2/3,1/3) and (1/3,2/3)
            vec![2.0/3.0, 1.0/3.0],
            vec![1.0/3.0, 2.0/3.0],
            // edge f₁: left edge, (0,1/3) and (0,2/3)
            vec![0.0, 1.0/3.0],
            vec![0.0, 2.0/3.0],
            // edge f₂: bottom edge, (1/3,0) and (2/3,0)
            vec![1.0/3.0, 0.0],
            vec![2.0/3.0, 0.0],
            // interior
            vec![1.0/3.0, 1.0/3.0],
            vec![0.25, 0.25],
        ]
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt1_coeff_computed() {
        let c = coeff();
        let diag: f64 = (0..8).map(|i| c[i][i].abs()).sum();
        assert!(diag > 0.1);
    }

    #[test]
    fn rt1_basis_finite() {
        let elem = TriRT1;
        let mut v = vec![0.0; 16];
        for xi in &[[0.0,0.0],[1.0,0.0],[0.0,1.0],[0.25,0.25],[1./3.,1./3.]] {
            elem.eval_basis_vec(xi, &mut v);
            for &val in &v { assert!(val.is_finite()); }
        }
    }

    /// Nodal basis: DOF_k(Φ_i) = δ_{ki} via numerical integration.
    #[test]
    fn rt1_nodal_basis() {
        let elem = TriRT1;
        let sq6_5 = (6.0f64 / 5.0).sqrt();
        let ta = ((3.0 - 2.0 * sq6_5) / 7.0).sqrt();
        let tb = ((3.0 + 2.0 * sq6_5) / 7.0).sqrt();
        let wa = (18.0 + 30.0f64.sqrt()) / 36.0;
        let wb = (18.0 - 30.0f64.sqrt()) / 36.0;
        let gl_pts = [0.5*(1.0-tb), 0.5*(1.0-ta), 0.5*(1.0+ta), 0.5*(1.0+tb)];
        let gl_wts = [0.5*wb, 0.5*wa, 0.5*wa, 0.5*wb];

        let mut vals = vec![0.0; 16];
        let mut dof_mat = [[0.0f64; 8]; 8];

        // Edge f₀: hypotenuse (1-t,t), normal (1,1)
        for k in 0..4 {
            let (t, w) = (gl_pts[k], gl_wts[k]);
            elem.eval_basis_vec(&[1.0-t, t], &mut vals);
            for i in 0..8 {
                let nf = vals[i*2] + vals[i*2+1];
                dof_mat[0][i] += w * nf;
                dof_mat[1][i] += w * nf * t;
            }
        }
        // Edge f₁: left (0,t), normal (-1,0)
        for k in 0..4 {
            let (t, w) = (gl_pts[k], gl_wts[k]);
            elem.eval_basis_vec(&[0.0, t], &mut vals);
            for i in 0..8 {
                let nf = -vals[i*2];
                dof_mat[2][i] += w * nf;
                dof_mat[3][i] += w * nf * t;
            }
        }
        // Edge f₂: bottom (t,0), normal (0,-1)
        for k in 0..4 {
            let (t, w) = (gl_pts[k], gl_wts[k]);
            elem.eval_basis_vec(&[t, 0.0], &mut vals);
            for i in 0..8 {
                let nf = -vals[i*2+1];
                dof_mat[4][i] += w * nf;
                dof_mat[5][i] += w * nf * t;
            }
        }
        // Interior
        let qr = elem.quadrature(6);
        for (xi, w) in qr.points.iter().zip(qr.weights.iter()) {
            elem.eval_basis_vec(xi, &mut vals);
            for i in 0..8 {
                dof_mat[6][i] += w * vals[i*2];
                dof_mat[7][i] += w * vals[i*2+1];
            }
        }
        for k in 0..8 {
            for i in 0..8 {
                let exp = if i == k { 1.0 } else { 0.0 };
                assert!((dof_mat[k][i] - exp).abs() < 1e-9,
                    "DOF_{k}(Phi_{i}) = {}, expected {exp}", dof_mat[k][i]);
            }
        }
    }
}
