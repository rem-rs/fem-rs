//! Nedelec-I order-2 element on the reference tetrahedron.
//!
//! 1:1 port of MFEM `ND_TetrahedronElement(2)` (`fe_nd.cpp`).
//!
//! Reference vertices: v₀=(0,0,0), v₁=(1,0,0), v₂=(0,1,0), v₃=(0,0,1).
//!
//! # Space
//! `N₂ = P₁³ ⊕ x^⊥ P̃₁`  (dim = 12 + 8 = 20)
//!
//! # DOF semantics (D32 fix — MFEM nodal point-value functionals)
//!
//! Every DOF is a **point evaluation of the tangential component** at the DOF
//! point `x_i = FE::Nodes` along a fixed reference tangent `t̂_i` from MFEM's
//! `tk` table (reference edge/face direction vectors, *unnormalized*):
//!
//! ```text
//! σ_i(Φ) = Φ(x_i) · t̂_i
//! ```
//!
//! * Edge DOFs (12): 2 per edge at the Gauss-Legendre points
//!   `t₀ = (1−1/√3)/2`, `t₁ = (1+1/√3)/2` (MFEM `OpenPoints(1)`), ordered
//!   ascending along each edge's named direction.  The GL point set is
//!   symmetric about 1/2, so an edge reversal maps the pair to a **signed
//!   anti-diagonal permutation** (`σ^rev_m = −σ_{1−m}`) — conforming pairing
//!   (the old integral moments `∫Φ·t̂ t^m` mix binomially under reversal).
//! * Face DOFs (8): 2 per face at the face centroid `(1/3,1/3,1/3)` in face
//!   barycentric coordinates, along two face edge-direction tangents (MFEM
//!   `dof2tk` pairs).  Cross-element face pairing between differently oriented
//!   tets requires MFEM's 2×2 `ND_DofTransformation` face rotations — the
//!   element semantics is identical to MFEM; see `crates/space/src/hcurl.rs`
//!   for the pairing conventions.
//!
//! # DOF layout (MFEM slot order)
//! ```text
//! 0..2   e₀₁ (v₀→v₁)  tang (1,0,0)     points (t,0,0)
//! 2..4   e₀₂ (v₀→v₂)  tang (0,1,0)     points (0,t,0)
//! 4..6   e₀₃ (v₀→v₃)  tang (0,0,1)     points (0,0,t)
//! 6..8   e₁₂ (v₁→v₂)  tang (−1,1,0)    points (1−t,t,0)
//! 8..10  e₁₃ (v₁→v₃)  tang (−1,0,1)    points (1−t,0,t)
//! 10..12 e₂₃ (v₂→v₃)  tang (0,−1,1)    points (0,1−t,t)
//! 12,13  face (1,2,3) at (1/3,1/3,1/3), tang (−1,1,0), (−1,0,1)
//! 14,15  face (0,3,2) at (0,1/3,1/3),   tang (0,0,1),   (0,1,0)
//! 16,17  face (0,1,3) at (1/3,0,1/3),   tang (1,0,0),   (0,0,1)
//! 18,19  face (0,2,1) at (1/3,1/3,0),   tang (0,1,0),   (1,0,0)
//! ```

use std::sync::OnceLock;

use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Coefficient matrix (cached) ────────────────────────────────────────────

static COEFF: OnceLock<[[f64; 20]; 20]> = OnceLock::new();

/// 2-point Gauss-Legendre nodes on `[0,1]` — MFEM `OpenPoints(1)` (GaussLegendre).
const GL2: [f64; 2] = [0.21132486540518710671, 0.78867513459481286553];

/// Face centroid barycentric coordinate (MFEM `fop`/`w` for p=2).
const FC: f64 = 1.0 / 3.0;

/// Evaluate the 20 monomials at (x,y,z).
/// Layout: monos[j*3], monos[j*3+1], monos[j*3+2] = components of m_j.
///
/// P₁³ monomials (j=0..11):
///   j=0: (1,0,0)  j=1: (ξ,0,0)  j=2: (η,0,0)  j=3: (ζ,0,0)
///   j=4: (0,1,0)  j=5: (0,ξ,0)  j=6: (0,η,0)  j=7: (0,ζ,0)
///   j=8: (0,0,1)  j=9: (0,0,ξ)  j=10:(0,0,η)  j=11:(0,0,ζ)
/// x^⊥ P̃₁ monomials (j=12..19):
///   j=12: (-ξη, ξ², 0)   j=13: (-ξζ, 0, ξ²)
///   j=14: (-η², ξη, 0)   j=15: (0, -ηζ, η²)
///   j=16: (-ζ², 0, ξζ)   j=17: (0, -ζ², ηζ)
///   j=18: (-ηζ, ξζ, 0)   j=19: (-ζη, 0, ξη)
fn eval_monomials(x: f64, y: f64, z: f64, vals: &mut [f64]) {
    // P₁³ (j=0..11)
    vals[0] = 1.0;
    vals[1] = 0.0;
    vals[2] = 0.0;
    vals[3] = x;
    vals[4] = 0.0;
    vals[5] = 0.0;
    vals[6] = y;
    vals[7] = 0.0;
    vals[8] = 0.0;
    vals[9] = z;
    vals[10] = 0.0;
    vals[11] = 0.0;
    vals[12] = 0.0;
    vals[13] = 1.0;
    vals[14] = 0.0;
    vals[15] = 0.0;
    vals[16] = x;
    vals[17] = 0.0;
    vals[18] = 0.0;
    vals[19] = y;
    vals[20] = 0.0;
    vals[21] = 0.0;
    vals[22] = z;
    vals[23] = 0.0;
    vals[24] = 0.0;
    vals[25] = 0.0;
    vals[26] = 1.0;
    vals[27] = 0.0;
    vals[28] = 0.0;
    vals[29] = x;
    vals[30] = 0.0;
    vals[31] = 0.0;
    vals[32] = y;
    vals[33] = 0.0;
    vals[34] = 0.0;
    vals[35] = z;

    // x^⊥ P̃₁ (j=12..19)
    vals[36] = -x * y;
    vals[37] = x * x;
    vals[38] = 0.0;
    vals[39] = -z * x;
    vals[40] = 0.0;
    vals[41] = x * x;
    vals[42] = -y * y;
    vals[43] = x * y;
    vals[44] = 0.0;
    vals[45] = 0.0;
    vals[46] = -y * z;
    vals[47] = y * y;
    vals[48] = -z * z;
    vals[49] = 0.0;
    vals[50] = z * x;
    vals[51] = 0.0;
    vals[52] = -z * z;
    vals[53] = y * z;
    vals[54] = -y * z;
    vals[55] = x * z;
    vals[56] = 0.0;
    vals[57] = -z * y;
    vals[58] = 0.0;
    vals[59] = x * y;
}

/// Curl of each monomial: curl(m) = (∂m_z/∂y − ∂m_y/∂z, ∂m_x/∂z − ∂m_z/∂x, ∂m_y/∂x − ∂m_x/∂y)
fn eval_monomial_curls(x: f64, y: f64, z: f64, curls: &mut [f64]) {
    // j=0..11 (P₁³)
    curls[0] = 0.0;
    curls[1] = 0.0;
    curls[2] = 0.0;
    curls[3] = 0.0;
    curls[4] = 0.0;
    curls[5] = 0.0;
    curls[6] = 0.0;
    curls[7] = 0.0;
    curls[8] = -1.0;
    curls[9] = 0.0;
    curls[10] = 1.0;
    curls[11] = 0.0;
    curls[12] = 0.0;
    curls[13] = 0.0;
    curls[14] = 0.0;
    curls[15] = 0.0;
    curls[16] = 0.0;
    curls[17] = 1.0;
    curls[18] = 0.0;
    curls[19] = 0.0;
    curls[20] = 0.0;
    curls[21] = -1.0;
    curls[22] = 0.0;
    curls[23] = 0.0;
    curls[24] = 0.0;
    curls[25] = 0.0;
    curls[26] = 0.0;
    curls[27] = 0.0;
    curls[28] = -1.0;
    curls[29] = 0.0;
    curls[30] = 1.0;
    curls[31] = 0.0;
    curls[32] = 0.0;
    curls[33] = 0.0;
    curls[34] = 0.0;
    curls[35] = 0.0;
    // j=12: (-ξη, ξ², 0) → (0, 0, 3ξ)
    curls[36] = 0.0;
    curls[37] = 0.0;
    curls[38] = 3.0 * x;
    // j=13: (-ζξ, 0, ξ²) → (0, −3ξ, 0)
    curls[39] = 0.0;
    curls[40] = -3.0 * x;
    curls[41] = 0.0;
    // j=14: (-η², ξη, 0) → (0, 0, 3η)
    curls[42] = 0.0;
    curls[43] = 0.0;
    curls[44] = 3.0 * y;
    // j=15: (0, -ηζ, η²) → (3η, 0, 0)
    curls[45] = 3.0 * y;
    curls[46] = 0.0;
    curls[47] = 0.0;
    // j=16: (-ζ², 0, ζξ) → (0, −3ζ, 0)
    curls[48] = 0.0;
    curls[49] = -3.0 * z;
    curls[50] = 0.0;
    // j=17: (0, -ζ², ηζ) → (3ζ, 0, 0)
    curls[51] = 3.0 * z;
    curls[52] = 0.0;
    curls[53] = 0.0;
    // j=18: (-ηζ, ξζ, 0) → (−ξ, −y, 2z)
    curls[54] = -x;
    curls[55] = -y;
    curls[56] = 2.0 * z;
    // j=19: (-ζη, 0, ξη) → (x, −2y, z)
    curls[57] = x;
    curls[58] = -2.0 * y;
    curls[59] = z;
}

/// Build the 20×20 Vandermonde matrix.
///
/// Every row is an exact point evaluation `t̂_i · m_j(x_i)` (MFEM `tk` tangents
/// at the `FE::Nodes` points) — no quadrature.
fn build_vandermonde() -> [[f64; 20]; 20] {
    let mut v = [[0.0f64; 20]; 20];
    let mut mono = [0.0f64; 60];

    let mut tangential_row = |row: usize, p: [f64; 3], t: [f64; 3]| {
        eval_monomials(p[0], p[1], p[2], &mut mono);
        for (j, chunk) in mono.chunks_exact(3).enumerate() {
            v[row][j] = chunk[0] * t[0] + chunk[1] * t[1] + chunk[2] * t[2];
        }
    };

    let mut row = 0usize;
    // ── Edges (MFEM order; points ascending along each edge direction) ──
    // e₀₁ (v₀→v₁)
    for &t in &GL2 {
        tangential_row(row, [t, 0.0, 0.0], [1.0, 0.0, 0.0]);
        row += 1;
    }
    // e₀₂ (v₀→v₂)
    for &t in &GL2 {
        tangential_row(row, [0.0, t, 0.0], [0.0, 1.0, 0.0]);
        row += 1;
    }
    // e₀₃ (v₀→v₃)
    for &t in &GL2 {
        tangential_row(row, [0.0, 0.0, t], [0.0, 0.0, 1.0]);
        row += 1;
    }
    // e₁₂ (v₁→v₂)
    for &t in &GL2 {
        tangential_row(row, [1.0 - t, t, 0.0], [-1.0, 1.0, 0.0]);
        row += 1;
    }
    // e₁₃ (v₁→v₃)
    for &t in &GL2 {
        tangential_row(row, [1.0 - t, 0.0, t], [-1.0, 0.0, 1.0]);
        row += 1;
    }
    // e₂₃ (v₂→v₃)
    for &t in &GL2 {
        tangential_row(row, [0.0, 1.0 - t, t], [0.0, -1.0, 1.0]);
        row += 1;
    }
    assert_eq!(row, 12);

    // ── Faces (MFEM face order & dof2tk tangent pairs; centroids) ──
    // face (1,2,3): tang (−1,1,0), (−1,0,1)
    tangential_row(row, [FC, FC, FC], [-1.0, 1.0, 0.0]);
    row += 1;
    tangential_row(row, [FC, FC, FC], [-1.0, 0.0, 1.0]);
    row += 1;
    // face (0,3,2): tang (0,0,1), (0,1,0)
    tangential_row(row, [0.0, FC, FC], [0.0, 0.0, 1.0]);
    row += 1;
    tangential_row(row, [0.0, FC, FC], [0.0, 1.0, 0.0]);
    row += 1;
    // face (0,1,3): tang (1,0,0), (0,0,1)
    tangential_row(row, [FC, 0.0, FC], [1.0, 0.0, 0.0]);
    row += 1;
    tangential_row(row, [FC, 0.0, FC], [0.0, 0.0, 1.0]);
    row += 1;
    // face (0,2,1): tang (0,1,0), (1,0,0)
    tangential_row(row, [FC, FC, 0.0], [0.0, 1.0, 0.0]);
    row += 1;
    tangential_row(row, [FC, FC, 0.0], [1.0, 0.0, 0.0]);

    v
}

/// Invert a 20×20 matrix using Gauss-Jordan elimination.
fn invert_20x20(a: [[f64; 20]; 20]) -> [[f64; 20]; 20] {
    let n = 20usize;
    let mut m = vec![[0.0f64; 40]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = a[i][j];
        }
        m[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..n {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        m.swap(col, max_row);
        let pivot = m[col][col];
        assert!(
            pivot.abs() > 1e-14,
            "TetND2 Vandermonde matrix is singular (col={col})"
        );
        let inv = 1.0 / pivot;
        for j in 0..2 * n {
            m[col][j] *= inv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = m[row][col];
            for j in 0..2 * n {
                let d = f * m[col][j];
                m[row][j] -= d;
            }
        }
    }
    let mut r = [[0.0f64; 20]; 20];
    for i in 0..n {
        for j in 0..n {
            r[i][j] = m[i][n + j];
        }
    }
    r
}

fn transpose_20x20(a: [[f64; 20]; 20]) -> [[f64; 20]; 20] {
    let mut t = [[0.0f64; 20]; 20];
    for i in 0..20 {
        for j in 0..20 {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn coeff() -> &'static [[f64; 20]; 20] {
    COEFF.get_or_init(|| transpose_20x20(invert_20x20(build_vandermonde())))
}

// ─── TetND2 ─────────────────────────────────────────────────────────────────

/// Nédélec first-kind H(curl) element on the reference tetrahedron — 20 DOFs, order 2.
///
/// Point-value DOFs per MFEM `ND_TetrahedronElement(2)` (see module docs).
pub struct TetND2;

impl VectorReferenceElement for TetND2 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        20
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut mono = [0.0f64; 60];
        eval_monomials(x, y, z, &mut mono);
        for i in 0..20 {
            let mut vx = 0.0;
            let mut vy = 0.0;
            let mut vz = 0.0;
            for j in 0..20 {
                vx += c[i][j] * mono[j * 3];
                vy += c[i][j] * mono[j * 3 + 1];
                vz += c[i][j] * mono[j * 3 + 2];
            }
            values[i * 3] = vx;
            values[i * 3 + 1] = vy;
            values[i * 3 + 2] = vz;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut mc = [0.0f64; 60];
        eval_monomial_curls(x, y, z, &mut mc);
        for i in 0..20 {
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut cz = 0.0;
            for j in 0..20 {
                cx += c[i][j] * mc[j * 3];
                cy += c[i][j] * mc[j * 3 + 1];
                cz += c[i][j] * mc[j * 3 + 2];
            }
            curl_vals[i * 3] = cx;
            curl_vals[i * 3 + 1] = cy;
            curl_vals[i * 3 + 2] = cz;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        use crate::quadrature::tet_rule;
        tet_rule(order)
    }

    /// DOF sites (MFEM `FE::Nodes`).
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let mut coords = Vec::with_capacity(20);
        // Edge DOFs, ascending along each edge direction.
        let edges: [([f64; 3], [f64; 3]); 6] = [
            ([0., 0., 0.], [1., 0., 0.]),
            ([0., 0., 0.], [0., 1., 0.]),
            ([0., 0., 0.], [0., 0., 1.]),
            ([1., 0., 0.], [0., 1., 0.]),
            ([1., 0., 0.], [0., 0., 1.]),
            ([0., 1., 0.], [0., 0., 1.]),
        ];
        for (a, b) in &edges {
            for &t in &GL2 {
                coords.push(vec![
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                ]);
            }
        }
        // Face DOFs at the centroids, MFEM face order.
        for f in &[
            [FC, FC, FC],
            [0.0, FC, FC],
            [FC, 0.0, FC],
            [FC, FC, 0.0],
        ] {
            coords.push(f.to_vec());
            coords.push(f.to_vec());
        }
        coords
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_nd2_coeff_computed() {
        let c = coeff();
        let diag_sum: f64 = (0..20).map(|i| c[i][i].abs()).sum();
        assert!(diag_sum > 0.1, "coefficient matrix diagonal is small");
    }

    #[test]
    fn tet_nd2_basis_finite() {
        let elem = TetND2;
        let mut vals = vec![0.0; 20 * 3];
        for xi in &[
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.25, 0.25, 0.25],
            vec![1. / 3., 1. / 3., 0.0],
        ] {
            elem.eval_basis_vec(xi, &mut vals);
            for &v in &vals {
                assert!(v.is_finite(), "non-finite at {xi:?}: {v}");
            }
        }
    }

    #[test]
    fn tet_nd2_curl_finite() {
        let elem = TetND2;
        let mut curl = vec![0.0; 20 * 3];
        let qr = elem.quadrature(4);
        for xi in &qr.points {
            elem.eval_curl(xi, &mut curl);
            for &v in &curl {
                assert!(v.is_finite(), "non-finite curl at {xi:?}: {v}");
            }
        }
    }

    /// Nodal basis: DOF_k(Φ_i) = δ_{ki} exactly (point-value functionals).
    #[test]
    fn tet_nd2_nodal_point_values() {
        let elem = TetND2;
        let coords = elem.dof_coords();
        let tangents: [[f64; 3]; 20] = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let mut vals = vec![0.0f64; 60];
        for (k, (pt, tang)) in coords.iter().zip(tangents.iter()).enumerate() {
            elem.eval_basis_vec(pt, &mut vals);
            for i in 0..20 {
                let sigma = vals[i * 3] * tang[0] + vals[i * 3 + 1] * tang[1] + vals[i * 3 + 2] * tang[2];
                let expected = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (sigma - expected).abs() < 1e-12,
                    "DOF_{k}(Phi_{i}) = {sigma}, expected {expected}"
                );
            }
        }
    }

    /// 1:1 check against the C++ MFEM `ND_TetrahedronElement(2)` dump
    /// (`tmp/d32/nd2_dump.cpp`): VShape at (0.111, 0.253, 0.198).
    #[test]
    fn tet_nd2_matches_mfem_dump() {
        let elem = TetND2;
        let mut phi = vec![0.0; 60];
        elem.eval_basis_vec(&[0.111, 0.253, 0.198], &mut phi);
        let mfem_phi: [[f64; 3]; 20] = [
            [0.058572978563594003, 0.011842624081163766, 0.011842624081163537],
            [-0.25236997856359378, -0.051025624081163817, -0.051025624081163679],
            [0.04976881902413053, 0.13592985749278366, 0.049768819024130578],
            [-0.031299819024130544, -0.08548685749278373, -0.031299819024130579],
            [0.032045527187836292, 0.032045527187836362, 0.10293411763365684],
            [-0.050261527187836469, -0.050261527187836573, -0.16144611763365663],
            [0.14597482865635963, -0.064044292414450352, 0.0],
            [0.083749171343640086, -0.036743707585549658, 0.0],
            [0.12114515360559067, 0.0, -0.067914707324346249],
            [0.091308846394409102, 0.0, -0.051188292675653647],
            [0.0, 0.054621983352787361, -0.069794756506339428],
            [0.0, 0.073484016647212505, -0.093896243493660381],
            [-0.15028199999999997, 0.13186800000000012, -0.084249000000000157],
            [-0.15028200000000014, -0.065934000000000118, 0.16849800000000031],
            [0.15028200000000017, -0.10988999999999967, 0.81516599999999961],
            [0.15028199999999997, 0.67062599999999994, -0.18215999999999974],
            [0.58627799999999974, 0.065934000000000006, -0.079919999999999741],
            [-0.19423799999999963, 0.065934000000000131, 0.35764199999999968],
            [-0.24819299999999977, 0.37595699999999993, 0.084249000000000088],
            [0.74913299999999949, -0.061604999999999938, 0.084249000000000254],
        ];
        for i in 0..20 {
            for d in 0..3 {
                let got = phi[i * 3 + d];
                let exp = mfem_phi[i][d];
                assert!(
                    (got - exp).abs() < 1e-13,
                    "phi[{i}][{d}] = {got}, expected {exp}"
                );
            }
        }
    }

    /// Edge DOF points must be symmetric about the edge midpoint — the
    /// property that makes the reversal pairing a signed anti-diagonal.
    #[test]
    fn tet_nd2_edge_points_symmetric() {
        let coords = dof_coords_of_edges();
        for e in 0..6 {
            let a = &coords[2 * e];
            let b = &coords[2 * e + 1];
            let (lo, hi) = edge_vertices(e);
            for d in 0..3 {
                // midpoint of the GL pair must equal the edge midpoint
                let mid = 0.5 * (a[d] + b[d]);
                let edge_mid = 0.5 * (lo[d] + hi[d]);
                assert!((mid - edge_mid).abs() < 1e-14, "edge {e} dim {d}");
            }
        }
    }

    fn dof_coords_of_edges() -> Vec<Vec<f64>> {
        TetND2.dof_coords()[..12].to_vec()
    }

    fn edge_vertices(e: usize) -> ([f64; 3], [f64; 3]) {
        [
            ([0., 0., 0.], [1., 0., 0.]),
            ([0., 0., 0.], [0., 1., 0.]),
            ([0., 0., 0.], [0., 0., 1.]),
            ([1., 0., 0.], [0., 1., 0.]),
            ([1., 0., 0.], [0., 0., 1.]),
            ([0., 1., 0.], [0., 0., 1.]),
        ][e]
    }
}
