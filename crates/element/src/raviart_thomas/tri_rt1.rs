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
    vals[0] = 1.0;
    vals[1] = 0.0; // (1,0)
    vals[2] = x;
    vals[3] = 0.0; // (ξ,0)
    vals[4] = y;
    vals[5] = 0.0; // (η,0)
    vals[6] = 0.0;
    vals[7] = 1.0; // (0,1)
    vals[8] = 0.0;
    vals[9] = x; // (0,ξ)
    vals[10] = 0.0;
    vals[11] = y; // (0,η)
    vals[12] = x * x;
    vals[13] = x * y; // (ξ²,ξη)
    vals[14] = x * y;
    vals[15] = y * y; // (ξη,η²)
}

/// div(m_j) = ∂m_j_x/∂ξ + ∂m_j_y/∂η
fn eval_monomial_divs(x: f64, y: f64, divs: &mut [f64; 8]) {
    divs[0] = 0.0; // div(1,0)=0
    divs[1] = 1.0; // div(ξ,0)=1
    divs[2] = 0.0; // div(η,0)=0
    divs[3] = 0.0; // div(0,1)=0
    divs[4] = 0.0; // div(0,ξ)=0
    divs[5] = 1.0; // div(0,η)=1
    divs[6] = 2.0 * x + y; // div(ξ²,ξη) = 2ξ + ξ = 3ξ? wait: ∂ξ²/∂ξ + ∂ξη/∂η = 2ξ + ξ = 3ξ
                           // Actually: ∂(ξ²)/∂ξ = 2ξ, ∂(ξη)/∂η = ξ → div = 2ξ + ξ = 3ξ
    divs[6] = 3.0 * x;
    // div(ξη, η²) = ∂ξη/∂ξ + ∂η²/∂η = η + 2η = 3η
    divs[7] = 3.0 * y;
}

/// Build 8×8 Vandermonde matrix V[k][j] = DOF_k(m_j).
///
/// Matches MFEM `RT_TriangleElement` (fem/fe/fe_rt.cpp): **nodal** DOFs.
/// Edge DOF nodes are the Gauss-Legendre open points on each edge
/// (`poly1d.OpenPoints(1)` → (1∓1/√3)/2 = 0.2113248654, 0.7886751346);
/// interior DOFs are both at (1/3, 1/3).  The functional is the point
/// evaluation of the normal component φ·nk with the *unnormalised* edge
/// normals `nk = {0,-1, 1,1, -1,0}` (hypotenuse normal has length √2 —
/// MFEM keeps it unnormalised, which affects the basis scaling).
fn build_vandermonde() -> [[f64; 8]; 8] {
    let mut v = [[0.0f64; 8]; 8];

    // Gauss-Legendre 2-point nodes on [0,1] (poly1d.OpenPoints(1))
    let gl = [0.5 * (1.0 - 1.0 / 3.0f64.sqrt()), 0.5 * (1.0 + 1.0 / 3.0f64.sqrt())];

    let mut mono = [0.0f64; 16];

    // Edge 0: bottom, nodes (t, 0), nk = (0,-1)
    for (k, &t) in gl.iter().enumerate() {
        eval_monomials(t, 0.0, &mut mono);
        for j in 0..8 {
            v[k][j] = -mono[j * 2 + 1]; // m_j · (0,-1)
        }
    }
    // Edge 1: hypotenuse, nodes (t2, t1) with (t_hi, t_lo) pairs, nk = (1,1)
    for k in 0..2 {
        let (x, y) = (gl[1 - k], gl[k]); // MFEM: Set2(bop[p-i], bop[i]) for i=0..p
        eval_monomials(x, y, &mut mono);
        for j in 0..8 {
            v[2 + k][j] = mono[j * 2] + mono[j * 2 + 1]; // m_j · (1,1)
        }
    }
    // Edge 2: left, nodes (0, t), nk = (-1,0)
    for k in 0..2 {
        let t = gl[1 - k]; // MFEM: Set2(0., bop[p-i])
        eval_monomials(0.0, t, &mut mono);
        for j in 0..8 {
            v[4 + k][j] = -mono[j * 2]; // m_j · (-1,0)
        }
    }
    // Interior DOFs: both at (1/3, 1/3); dof2nk = 0 → nk=(0,-1), dof2nk = 2 → nk=(-1,0)
    eval_monomials(1.0 / 3.0, 1.0 / 3.0, &mut mono);
    for j in 0..8 {
        v[6][j] = -mono[j * 2 + 1]; // nk[0]=(0,-1)
        v[7][j] = -mono[j * 2];     // nk[2]=(-1,0)
    }

    v
}

fn invert_8x8(a: [[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 16]; 8];
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = a[i][j];
        }
        m[i][8 + i] = 1.0;
    }
    for col in 0..8 {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..8 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        m.swap(col, max_row);
        let inv = 1.0 / m[col][col];
        assert!(inv.is_finite(), "TriRT1 Vandermonde matrix is singular");
        for j in 0..16 {
            m[col][j] *= inv;
        }
        for row in 0..8 {
            if row == col {
                continue;
            }
            let f = m[row][col];
            for j in 0..16 {
                let d = f * m[col][j];
                m[row][j] -= d;
            }
        }
    }
    let mut r = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            r[i][j] = m[i][8 + j];
        }
    }
    r
}

fn transpose_8x8(a: [[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut t = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn coeff() -> &'static [[f64; 8]; 8] {
    COEFF.get_or_init(|| transpose_8x8(invert_8x8(build_vandermonde())))
}

// ─── TriRT1 ──────────────────────────────────────────────────────────────────

/// Raviart-Thomas RT1 H(div) element on the reference triangle — 8 DOFs, order 1.
pub struct TriRT1;

impl VectorReferenceElement for TriRT1 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        8
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();
        let mut mono = [0.0f64; 16];
        eval_monomials(x, y, &mut mono);
        for i in 0..8 {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for j in 0..8 {
                vx += c[i][j] * mono[j * 2];
                vy += c[i][j] * mono[j * 2 + 1];
            }
            values[i * 2] = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();
        let mut md = [0.0f64; 8];
        eval_monomial_divs(x, y, &mut md);
        for i in 0..8 {
            let mut s = 0.0;
            for j in 0..8 {
                s += c[i][j] * md[j];
            }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let gl_lo = 0.5 * (1.0 - 1.0 / 3.0f64.sqrt());
        let gl_hi = 0.5 * (1.0 + 1.0 / 3.0f64.sqrt());
        vec![
            // Edge 0 (bottom, (0,0)→(1,0)): GL nodes
            vec![gl_lo, 0.0],
            vec![gl_hi, 0.0],
            // Edge 1 (hypotenuse, (1,0)→(0,1)): MFEM Set2(bop[p-i], bop[i])
            vec![gl_hi, gl_lo],
            vec![gl_lo, gl_hi],
            // Edge 2 (left, (0,1)→(0,0)): MFEM Set2(0., bop[p-i])
            vec![0.0, gl_hi],
            vec![0.0, gl_lo],
            // Interior: both at (1/3, 1/3)
            vec![1.0 / 3.0, 1.0 / 3.0],
            vec![1.0 / 3.0, 1.0 / 3.0],
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
        for xi in &[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [1. / 3., 1. / 3.],
        ] {
            elem.eval_basis_vec(xi, &mut v);
            for &val in &v {
                assert!(val.is_finite());
            }
        }
    }

    /// Nodal basis (MFEM RT_TriangleElement semantics): the DOF functional
    /// is the *point evaluation* of the normal component at the node, i.e.
    /// DOF_k(Φ_i) = Φ_i(node_k)·nk_k = δ_{ki}.  Nodes are the Gauss-Legendre
    /// open points on each edge; interior DOFs both at (1/3,1/3).
    #[test]
    fn rt1_nodal_basis() {
        let elem = TriRT1;
        let gl_lo = 0.5 * (1.0 - 1.0 / 3.0f64.sqrt());
        let gl_hi = 0.5 * (1.0 + 1.0 / 3.0f64.sqrt());
        // node_k + normal (unnormalised nk, as MFEM fe_rt.cpp)
        let nodes: [(f64, f64, f64, f64); 8] = [
            (gl_lo, 0.0, 0.0, -1.0),  // edge0 (0,1)
            (gl_hi, 0.0, 0.0, -1.0),
            (gl_hi, gl_lo, 1.0, 1.0), // edge1 (1,2)
            (gl_lo, gl_hi, 1.0, 1.0),
            (0.0, gl_hi, -1.0, 0.0),  // edge2 (2,0)
            (0.0, gl_lo, -1.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0, -1.0), // interior, dof2nk=0 → nk[0]
            (1.0 / 3.0, 1.0 / 3.0, -1.0, 0.0), // interior, dof2nk=2 → nk[2]
        ];
        let mut vals = vec![0.0; 16];
        for (k, &(x, y, nx, ny)) in nodes.iter().enumerate() {
            elem.eval_basis_vec(&[x, y], &mut vals);
            for i in 0..8 {
                let nf = vals[i * 2] * nx + vals[i * 2 + 1] * ny;
                let exp = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (nf - exp).abs() < 1e-9,
                    "DOF_{k}(Phi_{i}) = {nf}, expected {exp}",
                );
            }
        }
    }
}
