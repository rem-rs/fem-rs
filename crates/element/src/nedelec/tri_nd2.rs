//! Nedelec-I order-2 element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! # Space
//! `N₂ = P₁² ⊕ x^⊥ P̃₁`  (dim = 6 + 2 = 8)
//!
//! Monomial basis (component-wise):
//! ```text
//! m₀ = (1, 0)       m₁ = (ξ, 0)      m₂ = (η, 0)
//! m₃ = (0, 1)       m₄ = (0, ξ)      m₅ = (0, η)
//! m₆ = (−ξη, ξ²)   m₇ = (−η², ξη)
//! ```
//!
//! # DOF functionals (8 total)
//! Two tangential moments per edge, two interior moments:
//!
//! | DOF | Edge / interior | Functional |
//! |-----|-----------------|------------|
//! | 0   | e₀ (η=0, ξ↑)   | ∫₀¹ Φ_x(ξ,0) dξ           |
//! | 1   | e₀              | ∫₀¹ Φ_x(ξ,0) · ξ dξ       |
//! | 2   | e₁ (1−t,t)      | ∫₀¹ [−Φ_x + Φ_y](1−t,t) dt       |
//! | 3   | e₁              | ∫₀¹ [−Φ_x + Φ_y](1−t,t) · t dt   |
//! | 4   | e₂ (ξ=0, η↑)   | ∫₀¹ Φ_y(0,η) dη           |
//! | 5   | e₂              | ∫₀¹ Φ_y(0,η) · η dη       |
//! | 6   | interior        | ∫_T Φ_x dA                |
//! | 7   | interior        | ∫_T Φ_y dA                |

use std::sync::OnceLock;

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Coefficient matrix (cached) ────────────────────────────────────────────

/// Column-major coefficient matrix C = V⁻¹.
/// `Φ_i(ξ,η) = Σ_j C[i][j] · m_j(ξ,η)`
static COEFF: OnceLock<[[f64; 8]; 8]> = OnceLock::new();

/// Build the Vandermonde matrix V, where V[i,j] = DOF_i(m_j).
fn build_vandermonde() -> [[f64; 8]; 8] {
    // Reference triangle integrals:
    //   ∫_T 1    = 1/2
    //   ∫_T ξ    = 1/6
    //   ∫_T η    = 1/6
    //   ∫_T ξ²   = 1/12
    //   ∫_T η²   = 1/12
    //   ∫_T ξη   = 1/24
    //
    // Row 0: DOF_0 = ∫₀¹ (m_j)_x(ξ,0) dξ
    let r0 = [1.0, 0.5, 0.0,  0.0, 0.0, 0.0,  0.0,       0.0];
    // Row 1: DOF_1 = ∫₀¹ (m_j)_x(ξ,0) · ξ dξ
    let r1 = [0.5, 1.0/3.0, 0.0,  0.0, 0.0, 0.0,  0.0,  0.0];
    // Row 2: DOF_2 = ∫₀¹ [−(m_j)_x + (m_j)_y](1−t, t) dt
    let r2 = [-1.0, -0.5, -0.5,  1.0, 0.5, 0.5,  0.5,  0.5];
    // Row 3: DOF_3 = ∫₀¹ [−(m_j)_x + (m_j)_y](1−t, t) · t dt
    let r3 = [-0.5, -1.0/6.0, -1.0/3.0,  0.5, 1.0/6.0, 1.0/3.0,  1.0/6.0, 1.0/3.0];
    // Row 4: DOF_4 = ∫₀¹ (m_j)_y(0, η) dη
    let r4 = [0.0, 0.0, 0.0,  1.0, 0.0, 0.5,  0.0,  0.0];
    // Row 5: DOF_5 = ∫₀¹ (m_j)_y(0, η) · η dη
    let r5 = [0.0, 0.0, 0.0,  0.5, 0.0, 1.0/3.0,  0.0,  0.0];
    // Row 6: DOF_6 = ∫_T (m_j)_x dA
    let r6 = [0.5, 1.0/6.0, 1.0/6.0,  0.0, 0.0, 0.0,  -1.0/24.0, -1.0/12.0];
    // Row 7: DOF_7 = ∫_T (m_j)_y dA
    let r7 = [0.0, 0.0, 0.0,  0.5, 1.0/6.0, 1.0/6.0,  1.0/12.0, 1.0/24.0];

    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// Invert an 8×8 matrix (row-major) using Gauss-Jordan elimination.
fn invert_8x8(a: [[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 16]; 8];
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = a[i][j];
        }
        m[i][8 + i] = 1.0; // augment with identity
    }

    for col in 0..8 {
        // Pivot: find the row with the largest absolute value in this column
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..8 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        m.swap(col, max_row);

        let pivot = m[col][col];
        assert!(pivot.abs() > 1e-14, "TriND2 Vandermonde matrix is singular");
        let inv_pivot = 1.0 / pivot;
        for j in 0..16 {
            m[col][j] *= inv_pivot;
        }
        for row in 0..8 {
            if row == col { continue; }
            let factor = m[row][col];
            for j in 0..16 {
                let delta = factor * m[col][j];
                m[row][j] -= delta;
            }
        }
    }

    let mut result = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i][j] = m[i][8 + j];
        }
    }
    result
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
    // DOF_k(Φ_i) = Σ_j C[i][j] V[k][j] = (C V^T)_{ik} = δ_{ik}
    // ⟹ C = (V^T)^{-1} = (V^{-1})^T
    COEFF.get_or_init(|| transpose_8x8(invert_8x8(build_vandermonde())))
}

// ─── Monomial evaluators ────────────────────────────────────────────────────

/// Evaluate the 8 monomial vectors at (x,y) and store into `vals[i*2], vals[i*2+1]`.
#[inline]
fn eval_monomials(x: f64, y: f64, vals: &mut [f64; 16]) {
    // m₀ = (1, 0)
    vals[0] = 1.0;  vals[1] = 0.0;
    // m₁ = (ξ, 0)
    vals[2] = x;    vals[3] = 0.0;
    // m₂ = (η, 0)
    vals[4] = y;    vals[5] = 0.0;
    // m₃ = (0, 1)
    vals[6] = 0.0;  vals[7] = 1.0;
    // m₄ = (0, ξ)
    vals[8] = 0.0;  vals[9] = x;
    // m₅ = (0, η)
    vals[10] = 0.0; vals[11] = y;
    // m₆ = (−ξη, ξ²)
    vals[12] = -x * y; vals[13] = x * x;
    // m₇ = (−η², ξη)
    vals[14] = -y * y; vals[15] = x * y;
}

/// Scalar curl of each monomial at (x,y): curl(m_j) = ∂(m_j)_y/∂ξ − ∂(m_j)_x/∂η.
/// Returns [0, 0, -1, 0, 1, 0, 3ξ, 3η]
#[inline]
fn eval_monomial_curls(x: f64, y: f64, curls: &mut [f64; 8]) {
    curls[0] = 0.0;
    curls[1] = 0.0;
    curls[2] = -1.0; // ∂(0)/∂ξ − ∂(η)/∂η = 0 − 1
    curls[3] = 0.0;
    curls[4] = 1.0;  // ∂(ξ)/∂ξ − ∂(0)/∂η = 1 − 0
    curls[5] = 0.0;
    curls[6] = 3.0 * x; // ∂(ξ²)/∂ξ − ∂(−ξη)/∂η = 2ξ − (−ξ) = 3ξ
    curls[7] = 3.0 * y; // ∂(ξη)/∂ξ − ∂(−η²)/∂η = η − (−2η) = 3η
}

// ─── TriND2 ──────────────────────────────────────────────────────────────────

/// Nédélec first-kind H(curl) element on the reference triangle — 8 DOFs, order 2.
///
/// Reference domain: triangle with vertices (0,0), (1,0), (0,1).
///
/// DOF layout (2 per edge, 2 interior):
/// - DOFs 0–1: edge e₀ (v₀→v₁, bottom)
/// - DOFs 2–3: edge e₁ (v₁→v₂, hypotenuse)
/// - DOFs 4–5: edge e₂ (v₀→v₂, left)
/// - DOFs 6–7: interior bubble moments
pub struct TriND2;

impl VectorReferenceElement for TriND2 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize { 8 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();

        let mut mono = [0.0f64; 16];
        eval_monomials(x, y, &mut mono);

        // Φ_i(ξ,η) = Σ_j C[i][j] · m_j(ξ,η)
        for i in 0..8 {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for j in 0..8 {
                vx += c[i][j] * mono[j * 2];
                vy += c[i][j] * mono[j * 2 + 1];
            }
            values[i * 2]     = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();

        let mut mc = [0.0f64; 8];
        eval_monomial_curls(x, y, &mut mc);

        // curl(Φ_i) = Σ_j C[i][j] · curl(m_j)
        for i in 0..8 {
            let mut s = 0.0;
            for j in 0..8 {
                s += c[i][j] * mc[j];
            }
            curl_vals[i] = s;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        // H(curl) elements have zero divergence in the natural sense.
        for v in div_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }

    /// DOF sites:
    /// - 2 Gauss points per edge (at 1/3 and 2/3 along each edge)
    /// - 2 interior points (barycentric coordinates)
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            // Edge e₀ (η=0): ξ = 1/3 and 2/3
            vec![1.0/3.0, 0.0],
            vec![2.0/3.0, 0.0],
            // Edge e₁ (v₁→v₂): param t=1/3 → (2/3, 1/3), t=2/3 → (1/3, 2/3)
            vec![2.0/3.0, 1.0/3.0],
            vec![1.0/3.0, 2.0/3.0],
            // Edge e₂ (ξ=0): η = 1/3 and 2/3
            vec![0.0, 1.0/3.0],
            vec![0.0, 2.0/3.0],
            // Interior
            vec![1.0/3.0, 1.0/3.0],
            vec![1.0/4.0, 1.0/4.0],
        ]
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd2_coeff_matrix_is_computed() {
        // Just trigger the OnceLock computation; if it panics the matrix is singular.
        let c = coeff();
        // Diagonal should not all be zero
        let diag_sum: f64 = (0..8).map(|i| c[i][i].abs()).sum();
        assert!(diag_sum > 0.1, "coefficient matrix diagonal is unexpectedly small");
    }

    /// Nodal basis property: DOF_j(Φᵢ) ≈ δᵢⱼ.
    /// We approximate each edge DOF via 4-point Gauss quadrature on [0,1].
    #[test]
    fn nd2_nodal_basis() {
        let elem = TriND2;
        let mut vals = vec![0.0; 16];

        // 4-point Gauss-Legendre on [0,1]
        let sq6_5 = (6.0f64 / 5.0).sqrt();
        let ta = ((3.0 - 2.0 * sq6_5) / 7.0).sqrt();
        let tb = ((3.0 + 2.0 * sq6_5) / 7.0).sqrt();
        let wa = (18.0 + 30.0f64.sqrt()) / 36.0;
        let wb = (18.0 - 30.0f64.sqrt()) / 36.0;
        let gl_pts = [0.5*(1.0-tb), 0.5*(1.0-ta), 0.5*(1.0+ta), 0.5*(1.0+tb)];
        let gl_wts = [0.5*wb, 0.5*wa, 0.5*wa, 0.5*wb];

        // DOF matrix (DOF_j applies to basis function i): should be identity
        let mut dof_mat = [[0.0f64; 8]; 8];

        // --- Edge e₀: y=0, x in [0,1], tangent=(1,0), weight moments 1 and x ---
        for (&t, &w) in gl_pts.iter().zip(gl_wts.iter()) {
            elem.eval_basis_vec(&[t, 0.0], &mut vals);
            for i in 0..8 {
                let tang = vals[i * 2]; // Φ_x
                dof_mat[0][i] += w * tang;         // ∫ Φ_x dξ
                dof_mat[1][i] += w * tang * t;     // ∫ Φ_x · ξ dξ
            }
        }

        // --- Edge e₁: param (1-t, t), tangent=(-1,1), moment 1 and t ---
        for (&t, &w) in gl_pts.iter().zip(gl_wts.iter()) {
            let xi = [1.0 - t, t];
            elem.eval_basis_vec(&xi, &mut vals);
            for i in 0..8 {
                let tang = -vals[i * 2] + vals[i * 2 + 1];
                dof_mat[2][i] += w * tang;
                dof_mat[3][i] += w * tang * t;
            }
        }

        // --- Edge e₂: x=0, y in [0,1], tangent=(0,1), moments 1 and y ---
        for (&t, &w) in gl_pts.iter().zip(gl_wts.iter()) {
            elem.eval_basis_vec(&[0.0, t], &mut vals);
            for i in 0..8 {
                let tang = vals[i * 2 + 1]; // Φ_y
                dof_mat[4][i] += w * tang;
                dof_mat[5][i] += w * tang * t;
            }
        }

        // --- Interior DOFs: ∫_T Φ_x dA and ∫_T Φ_y dA via triangle quadrature ---
        let qr = elem.quadrature(6);
        for (xi, w) in qr.points.iter().zip(qr.weights.iter()) {
            elem.eval_basis_vec(xi, &mut vals);
            for i in 0..8 {
                dof_mat[6][i] += w * vals[i * 2];
                dof_mat[7][i] += w * vals[i * 2 + 1];
            }
        }

        // Check that dof_mat ≈ identity
        for j in 0..8 {
            for i in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof_mat[j][i] - expected).abs() < 1e-10,
                    "DOF_{j}(Phi_{i}) = {}, expected {expected}", dof_mat[j][i]
                );
            }
        }
    }

    /// Curl should be linear in (ξ,η) — check at several points.
    #[test]
    fn nd2_curl_is_linear() {
        let elem = TriND2;
        let curl = vec![0.0; 8];
        let curl2 = vec![0.0; 8];

        let pts = [
            [0.1, 0.1],
            [0.5, 0.2],
            [0.2, 0.5],
            [1.0/3.0, 1.0/3.0],
        ];
        // Evaluate curl at p and at 2*p; for linear functions curl(2p)=2*curl(p) only at origin...
        // Better test: verify curl changes linearly between two points.
        // curl(t*p1 + (1-t)*p2) = t*curl(p1) + (1-t)*curl(p2)
        let p1 = [0.1, 0.2_f64];
        let p2 = [0.3, 0.1_f64];
        let t = 0.4f64;
        let pm = [t * p1[0] + (1.0-t)*p2[0], t * p1[1] + (1.0-t)*p2[1]];
        let mut c1 = vec![0.0; 8];
        let mut c2 = vec![0.0; 8];
        let mut cm = vec![0.0; 8];
        elem.eval_curl(&p1, &mut c1);
        elem.eval_curl(&p2, &mut c2);
        elem.eval_curl(&pm, &mut cm);
        for i in 0..8 {
            let interp = t * c1[i] + (1.0-t) * c2[i];
            assert!(
                (cm[i] - interp).abs() < 1e-12,
                "curl is not linear for basis {i}: {}, expected {interp}", cm[i]
            );
        }
        let _ = pts; // suppress warning
        let _ = curl; let _ = curl2;
    }

    #[test]
    fn nd2_basis_values_finite() {
        let elem = TriND2;
        let mut vals = vec![0.0; 16];
        for xi in &[[0.0,0.0],[1.0,0.0],[0.0,1.0],[0.25,0.25],[1.0/3.0,1.0/3.0]] {
            elem.eval_basis_vec(xi, &mut vals);
            for v in &vals {
                assert!(v.is_finite(), "non-finite value at {xi:?}: {v}");
            }
        }
    }
}
