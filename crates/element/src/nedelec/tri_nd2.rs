//! Nedelec-I order-2 element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! 1:1 port of MFEM `ND_TriangleElement(2)` (`fe_nd.cpp`).
//!
//! # Space
//! `N₂ = P₁² ⊕ s·x^⊥ P̃₁`  (dim = 6 + 2 = 8)
//!
//! # DOF semantics (D32 fix — MFEM nodal point-value functionals)
//!
//! Every DOF is a **point evaluation of the tangential component** at the DOF
//! point `x_i = FE::Nodes` along a fixed reference tangent `t̂_i` taken from
//! MFEM's `tk` table (the reference edge direction vectors, *unnormalized*):
//!
//! ```text
//! σ_i(Φ) = Φ(x_i) · t̂_i
//! ```
//!
//! Edge DOF points are the 2-point Gauss-Legendre nodes on `[0,1]`
//! (MFEM `OpenPoints(1)`, symmetric about `t = 1/2`), ordered ascending along
//! each edge's named direction `(v_i → v_j)`.  Because the point set is
//! reflection invariant (`1 − t_m = t_{1−m}`) the edge DOFs transform under an
//! edge reversal as a **signed anti-diagonal permutation**
//! (`σ^rev_m = −σ_{1−m}`), which is what makes the cross-element pairing
//! conforming.  (The previous integral-moment functionals `∫Φ·t̂ t^m dt` are
//! NOT reflection invariant — reversal mixes moments binomially — which
//! polluted curl-curl solutions on split diagonals.)
//!
//! # DOF layout (MFEM slot order)
//!
//! | DOF | Edge / interior | Point          | Tangent    | Functional       |
//! |-----|-----------------|----------------|------------|------------------|
//! | 0   | e₀ (v₀→v₁)      | (t₀, 0)        | (1, 0)     | Φ_x(t₀,0)        |
//! | 1   | e₀              | (t₁, 0)        | (1, 0)     | Φ_x(t₁,0)        |
//! | 2   | e₁ (v₁→v₂)      | (1−t₀, t₀)     | (−1, 1)    | (−Φ_x+Φ_y)(pt)   |
//! | 3   | e₁              | (1−t₁, t₁)     | (−1, 1)    | (−Φ_x+Φ_y)(pt)   |
//! | 4   | e₂ (v₂→v₀)      | (0, 1−t₀)      | (0, −1)    | −Φ_y(0,1−t₀)     |
//! | 5   | e₂              | (0, 1−t₁)      | (0, −1)    | −Φ_y(0,1−t₁)     |
//! | 6   | interior        | (1/3, 1/3)     | (1, 0)     | Φ_x(1/3,1/3)     |
//! | 7   | interior        | (1/3, 1/3)     | (0, 1)     | Φ_y(1/3,1/3)     |
//!
//! with `t₀ = (1−1/√3)/2`, `t₁ = (1+1/√3)/2`.

use std::sync::OnceLock;

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Coefficient matrix (cached) ────────────────────────────────────────────

/// Column-major coefficient matrix C = V⁻¹.
/// `Φ_i(ξ,η) = Σ_j C[i][j] · m_j(ξ,η)`
static COEFF: OnceLock<[[f64; 8]; 8]> = OnceLock::new();

/// 2-point Gauss-Legendre nodes on `[0,1]` — MFEM `OpenPoints(1)` (GaussLegendre).
pub(crate) const GL2: [f64; 2] = [0.21132486540518710671, 0.78867513459481286553];

/// Barycentric coordinates of the ND2 interior point (MFEM `iop` centroid).
const INTERIOR: f64 = 1.0 / 3.0;

/// Build the Vandermonde matrix V, where V[i,j] = DOF_i(m_j).
///
/// Every row is an exact point evaluation `t̂_i · m_j(x_i)` — no quadrature.
fn build_vandermonde() -> [[f64; 8]; 8] {
    let mut v = [[0.0f64; 8]; 8];
    let mut mono = [0.0f64; 16];

    let mut tangential_row = |row: usize, x: f64, y: f64, tx: f64, ty: f64| {
        eval_monomials(x, y, &mut mono);
        for (j, chunk) in mono.chunks_exact(2).enumerate() {
            v[row][j] = chunk[0] * tx + chunk[1] * ty;
        }
    };

    // Edge e₀ (v₀→v₁, η=0), tangent (1,0).
    tangential_row(0, GL2[0], 0.0, 1.0, 0.0);
    tangential_row(1, GL2[1], 0.0, 1.0, 0.0);
    // Edge e₁ (v₁→v₂), point (1−t, t), tangent (−1,1).
    tangential_row(2, 1.0 - GL2[0], GL2[0], -1.0, 1.0);
    tangential_row(3, 1.0 - GL2[1], GL2[1], -1.0, 1.0);
    // Edge e₂ (v₂→v₀), point (0, 1−t), tangent (0,−1).
    tangential_row(4, 0.0, 1.0 - GL2[0], 0.0, -1.0);
    tangential_row(5, 0.0, 1.0 - GL2[1], 0.0, -1.0);
    // Interior (1/3,1/3): tangents (1,0), (0,1).
    tangential_row(6, INTERIOR, INTERIOR, 1.0, 0.0);
    tangential_row(7, INTERIOR, INTERIOR, 0.0, 1.0);

    v
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
            if row == col {
                continue;
            }
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
    vals[0] = 1.0;
    vals[1] = 0.0;
    // m₁ = (ξ, 0)
    vals[2] = x;
    vals[3] = 0.0;
    // m₂ = (η, 0)
    vals[4] = y;
    vals[5] = 0.0;
    // m₃ = (0, 1)
    vals[6] = 0.0;
    vals[7] = 1.0;
    // m₄ = (0, ξ)
    vals[8] = 0.0;
    vals[9] = x;
    // m₅ = (0, η)
    vals[10] = 0.0;
    vals[11] = y;
    // m₆ = (−ξη, ξ²)
    vals[12] = -x * y;
    vals[13] = x * x;
    // m₇ = (−η², ξη)
    vals[14] = -y * y;
    vals[15] = x * y;
}

/// Scalar curl of each monomial at (x,y): curl(m_j) = ∂(m_j)_y/∂ξ − ∂(m_j)_x/∂η.
/// Returns [0, 0, -1, 0, 1, 0, 3ξ, 3η]
#[inline]
fn eval_monomial_curls(x: f64, y: f64, curls: &mut [f64; 8]) {
    curls[0] = 0.0;
    curls[1] = 0.0;
    curls[2] = -1.0; // ∂(0)/∂ξ − ∂(η)/∂η = 0 − 1
    curls[3] = 0.0;
    curls[4] = 1.0; // ∂(ξ)/∂ξ − ∂(0)/∂η = 1 − 0
    curls[5] = 0.0;
    curls[6] = 3.0 * x; // ∂(ξ²)/∂ξ − ∂(−ξη)/∂η = 2ξ − (−ξ) = 3ξ
    curls[7] = 3.0 * y; // ∂(ξη)/∂ξ − ∂(−η²)/∂η = η − (−2η) = 3η
}

// ─── TriND2 ──────────────────────────────────────────────────────────────────

/// Nédélec first-kind H(curl) element on the reference triangle — 8 DOFs, order 2.
///
/// Reference domain: triangle with vertices (0,0), (1,0), (0,1).
///
/// DOF layout (2 point-value DOFs per edge, 2 interior point values; MFEM
/// `ND_TriangleElement(2)` slot order — see the module docs).
pub struct TriND2;

impl TriND2 {
    /// Coefficient matrix `C[i][j]` on the reference triangle with
    /// `Φ_i(ξ,η) = Σ_j C[i][j] · m_j(ξ,η)` for the eight monomial vectors `m_j`.
    #[inline]
    pub fn monomial_coeff_matrix() -> &'static [[f64; 8]; 8] {
        coeff()
    }
}

impl VectorReferenceElement for TriND2 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        8
    }

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
            values[i * 2] = vx;
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
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    /// DOF sites (MFEM `FE::Nodes`): 2 Gauss points per edge ordered along the
    /// edge's named direction, plus the two interior point values at (1/3,1/3).
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let (t0, t1) = (GL2[0], GL2[1]);
        vec![
            // Edge e₀ (v₀→v₁, η=0)
            vec![t0, 0.0],
            vec![t1, 0.0],
            // Edge e₁ (v₁→v₂): point (1−t, t)
            vec![1.0 - t0, t0],
            vec![1.0 - t1, t1],
            // Edge e₂ (v₂→v₀): point (0, 1−t)
            vec![0.0, 1.0 - t0],
            vec![0.0, 1.0 - t1],
            // Interior
            vec![INTERIOR, INTERIOR],
            vec![INTERIOR, INTERIOR],
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
        assert!(
            diag_sum > 0.1,
            "coefficient matrix diagonal is unexpectedly small"
        );
    }

    /// Nodal basis property: DOF_j(Φᵢ) = δᵢⱼ exactly (point-value functionals).
    #[test]
    fn nd2_nodal_basis_point_values() {
        let elem = TriND2;
        let mut vals = vec![0.0; 16];

        // (point, tangent) per DOF, mirroring the Vandermonde rows.
        let dofs: [([f64; 2], [f64; 2]); 8] = [
            ([GL2[0], 0.0], [1.0, 0.0]),
            ([GL2[1], 0.0], [1.0, 0.0]),
            ([1.0 - GL2[0], GL2[0]], [-1.0, 1.0]),
            ([1.0 - GL2[1], GL2[1]], [-1.0, 1.0]),
            ([0.0, 1.0 - GL2[0]], [0.0, -1.0]),
            ([0.0, 1.0 - GL2[1]], [0.0, -1.0]),
            ([INTERIOR, INTERIOR], [1.0, 0.0]),
            ([INTERIOR, INTERIOR], [0.0, 1.0]),
        ];

        let mut dof_mat = [[0.0f64; 8]; 8];
        for (j, (pt, tang)) in dofs.iter().enumerate() {
            elem.eval_basis_vec(pt, &mut vals);
            for i in 0..8 {
                dof_mat[j][i] = vals[i * 2] * tang[0] + vals[i * 2 + 1] * tang[1];
            }
        }

        for j in 0..8 {
            for i in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof_mat[j][i] - expected).abs() < 1e-14,
                    "DOF_{j}(Phi_{i}) = {}, expected {expected}",
                    dof_mat[j][i]
                );
            }
        }
    }

    /// 1:1 check against the C++ MFEM `ND_TriangleElement(2)` dump
    /// (`tmp/d32/nd2_dump.cpp`, WSL `$HOME/work/nd2_dump`):
    /// VShape at (0.137, 0.421) and curl at the same point.
    #[test]
    fn nd2_matches_mfem_dump() {
        let elem = TriND2;
        let mut phi = vec![0.0; 16];
        elem.eval_basis_vec(&[0.137, 0.421], &mut phi);
        let mfem_phi: [[f64; 2]; 8] = [
            [0.076797256181313028, 0.018171371497132931],
            [-0.22907425618131308, -0.054202371497133008],
            [0.17216846137808234, -0.056026316410445164],
            [-0.034922461378082706, 0.01136431641044499],
            [-0.1163279694051417, -0.23845852160721462],
            [-0.13164103059485799, -0.2698484783927852],
            [1.2895229999999998, -0.0086310000000002322],
            [-0.38521499999999942, 0.53635500000000047],
        ];
        for i in 0..8 {
            assert!(
                (phi[i * 2] - mfem_phi[i][0]).abs() < 1e-13,
                "phi[{i}]_x = {}, expected {}",
                phi[i * 2],
                mfem_phi[i][0]
            );
            assert!(
                (phi[i * 2 + 1] - mfem_phi[i][1]).abs() < 1e-13,
                "phi[{i}]_y = {}, expected {}",
                phi[i * 2 + 1],
                mfem_phi[i][1]
            );
        }

        let mut curl = vec![0.0; 8];
        elem.eval_curl(&[0.137, 0.421], &mut curl);
        let mfem_curl = [
            1.3979132444627611,
            -0.1869132444627612,
            -0.22685364402434116,
            1.2488536440243421,
            1.8289403995615803,
            1.9380596004384196,
            -0.18899999999999942,
            2.7449999999999992,
        ];
        for i in 0..8 {
            assert!(
                (curl[i] - mfem_curl[i]).abs() < 1e-13,
                "curl[{i}] = {}, expected {}",
                curl[i],
                mfem_curl[i]
            );
        }
    }

    /// At the interior point the two interior DOFs evaluate to the unit
    /// vectors (point-value DOFs): verified against the MFEM dump.
    #[test]
    fn nd2_interior_point_delta() {
        let elem = TriND2;
        let mut vals = vec![0.0; 16];
        elem.eval_basis_vec(&[INTERIOR, INTERIOR], &mut vals);
        for i in 0..6 {
            assert!(vals[i * 2].abs() < 1e-14);
            assert!(vals[i * 2 + 1].abs() < 1e-14);
        }
        assert!((vals[12] - 1.0).abs() < 1e-14 && vals[13].abs() < 1e-14);
        assert!(vals[14].abs() < 1e-14 && (vals[15] - 1.0).abs() < 1e-14);
    }

    /// Curl should be linear in (ξ,η) — check at several points.
    #[test]
    fn nd2_curl_is_linear() {
        let elem = TriND2;
        let p1 = [0.1, 0.2_f64];
        let p2 = [0.3, 0.1_f64];
        let t = 0.4f64;
        let pm = [t * p1[0] + (1.0 - t) * p2[0], t * p1[1] + (1.0 - t) * p2[1]];
        let mut c1 = vec![0.0; 8];
        let mut c2 = vec![0.0; 8];
        let mut cm = vec![0.0; 8];
        elem.eval_curl(&p1, &mut c1);
        elem.eval_curl(&p2, &mut c2);
        elem.eval_curl(&pm, &mut cm);
        for i in 0..8 {
            let interp = t * c1[i] + (1.0 - t) * c2[i];
            assert!(
                (cm[i] - interp).abs() < 1e-12,
                "curl is not linear for basis {i}: {}, expected {interp}",
                cm[i]
            );
        }
    }

    #[test]
    fn nd2_basis_values_finite() {
        let elem = TriND2;
        let mut vals = vec![0.0; 16];
        for xi in &[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [1.0 / 3.0, 1.0 / 3.0],
        ] {
            elem.eval_basis_vec(xi, &mut vals);
            for v in &vals {
                assert!(v.is_finite(), "non-finite value at {xi:?}: {v}");
            }
        }
    }
}
