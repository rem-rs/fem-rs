//! MFEM `RT_TetrahedronElement` port: **nodal** flux-dual RT bases on the
//! reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
//!
//! This module provides the shared generic construction used by [`TetRT1`]
//! (15 DOFs) and [`TetRT2`] (36 DOFs).  It reproduces MFEM's nodal DOF
//! semantics exactly (`fem/fe/fe_rt.cpp`):
//!
//! - **Nodes** sit on the faces at the Gauss-Legendre open points
//!   `poly1d.OpenPoints(p)` (degree `p = k` per face direction) following the
//!   MFEM per-face point patterns, plus interior points at degree `p-1` with
//!   one component sample (nk ∈ {−x, −y, −z}) per point.
//! - **Functionals** are point evaluations `Φ(x_m) · nk_m` with the MFEM
//!   unnormalised face-normal table `nk = {1,1,1, −1,0,0, 0,−1,0, 0,0,−1}`.
//! - The basis functions are the duals of these functionals
//!   (`D_m(Φ_i) = δ_mi`), built by inverting the nodal Vandermonde matrix.
//!
//! Slot order (matches `HDivSpace::build_3d_tet`): faces in MFEM
//! `FaceVert` order `(1,2,3), (0,3,2), (0,1,3), (0,2,1)`, `(k+1)(k+2)/2`
//! grid nodes per face, then the interior component samples.

use std::sync::OnceLock;

use nalgebra::DMatrix;

use crate::quadrature::{gauss_legendre_01, tet_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// MFEM `Geometry::Constants<TETRAHEDRON>::FaceVert` (local face vertex order).
pub(super) const TET_FACE_VERTS: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/// MFEM `RT_TetrahedronElement::nk`: unnormalised normals for `dof2nk` 0..3.
pub(super) const TET_NK: [[f64; 3]; 4] = [
    [1.0, 1.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
];

/// Number of RT_k DOFs on the reference tetrahedron.
pub(super) fn nodal_tet_n_dofs(k: usize) -> usize {
    (k + 1) * (k + 2) * (k + 4) / 2
}

/// MFEM nodal dof table for order `k`: `(reference points, reference
/// normals)` — the canonical slot/dof definition shared by the basis
/// construction, `HDivSpace::interpolate_vector` and the discrete operators
/// (`crates/assembly/src/discrete_op.rs`).
pub fn mfem_nodal_dofs(k: usize) -> &'static (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    static CACHE: [OnceLock<(Vec<[f64; 3]>, Vec<[f64; 3]>)>; 5] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[k].get_or_init(|| {
        let mut pts = Vec::new();
        let mut nks = Vec::new();
        let bop = gauss_legendre_01(k + 1).0;
        let p = k;
        // Faces in MFEM FaceVert order, MFEM per-face point patterns.
        for (f, nk) in TET_NK.iter().enumerate() {
            for j in 0..=p {
                for i in 0..=(p - j) {
                    let w = bop[i] + bop[j] + bop[p - i - j];
                    let b0 = bop[p - i - j] / w;
                    let b1 = bop[i] / w;
                    let b2 = bop[j] / w;
                    let pt: [f64; 3] = match f {
                        0 => [b0, b1, b2], // (1,2,3): Set3(bop[p-i-j], bop[i], bop[j])/w
                        1 => [0.0, b2, b1], // (0,3,2): Set3(0, bop[j], bop[i])/w
                        2 => [b1, 0.0, b2], // (0,1,3): Set3(bop[i], 0, bop[j])/w
                        _ => [b2, b1, 0.0], // (0,2,1): Set3(bop[j], bop[i], 0)/w
                    };
                    pts.push(pt);
                    nks.push(*nk);
                }
            }
        }
        // Interior: degree p−1 points, one component sample per point
        // (dof2nk = 1, 2, 3 → nk = −x, −y, −z).
        if k >= 1 {
            let iop = gauss_legendre_01(k).0;
            for d in 0..k {
                for j in 0..(k - d) {
                    for i in 0..(k - d - j) {
                        let w = iop[i] + iop[j] + iop[d] + iop[k - 1 - i - j - d];
                        let pt = [iop[i] / w, iop[j] / w, iop[d] / w];
                        for nk in &TET_NK[1..4] {
                            pts.push(pt);
                            nks.push(*nk);
                        }
                    }
                }
            }
        }
        (pts, nks)
    })
}

struct NodalTetData {
    /// `(V⁻¹)ᵀ`: row `i` holds the monomial coefficients of basis function `i`.
    coeff: Vec<f64>,
}

fn nodal_tet_data(k: usize) -> &'static NodalTetData {
    static CACHE: [OnceLock<Option<NodalTetData>>; 5] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    let slot = CACHE[k].get_or_init(|| {
        let n = nodal_tet_n_dofs(k);
        let n3 = (k + 1) * (k + 2) * (k + 3) / 2; // [P_k]³ monomials (interleaved)
        let nb = (k + 1) * (k + 2) / 2; // homogeneous bubbles
        let mt = n3 + nb;
        debug_assert_eq!(mt, n);

        let mut mono = vec![0.0f64; mt * 3];
        // Vandermonde V[m][j] = D_m(mono_j) = mono_j(x_m)·nk_m.
        let mut v = vec![0.0f64; n * n];

        let (dof_pts, dof_nks) = mfem_nodal_dofs(k);
        for (m, (pt, nk)) in dof_pts.iter().zip(dof_nks.iter()).enumerate() {
            fill_mono_row(k, pt[0], pt[1], pt[2], n3, &mut mono);
            for jj in 0..mt {
                v[m * n + jj] =
                    mono[jj * 3] * nk[0] + mono[jj * 3 + 1] * nk[1] + mono[jj * 3 + 2] * nk[2];
            }
        }

        // coeff = (V⁻¹)ᵀ so that D_m(Φ_i) = Σ_j coeff[i][j]·V[m][j] = δ_mi.
        let dm = DMatrix::from_row_slice(n, n, &v);
        let inv = dm
            .try_inverse()
            .expect("MFEM nodal tet RT Vandermonde is singular");
        let mut coeff = vec![0.0f64; n * n];
        for (r, row) in coeff.chunks_mut(n).enumerate() {
            for (c, val) in row.iter_mut().enumerate() {
                *val = inv[(c, r)];
            }
        }
        Some(NodalTetData { coeff })
    });
    slot.as_ref()
        .expect("nodal tet data present for all k in 0..=4")
}

/// Fill `mono` with the RT_k monomial generators at `(x, y, z)`.
fn fill_mono_row(k: usize, x: f64, y: f64, z: f64, n3: usize, mono: &mut [f64]) {
    let mut idx = 0usize;
    // [P_k]³ monomials, component-interleaved.
    for deg in 0..=k {
        for a in 0..=deg {
            for b in 0..=(deg - a) {
                let c = deg - a - b;
                let m = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
                mono[idx * 3] = 0.0;
                mono[idx * 3 + 1] = 0.0;
                mono[idx * 3 + 2] = 0.0;
                mono[idx * 3] = m;
                idx += 1;
                mono[idx * 3] = 0.0;
                mono[idx * 3 + 1] = 0.0;
                mono[idx * 3 + 2] = 0.0;
                mono[idx * 3 + 1] = m;
                idx += 1;
                mono[idx * 3] = 0.0;
                mono[idx * 3 + 1] = 0.0;
                mono[idx * 3 + 2] = 0.0;
                mono[idx * 3 + 2] = m;
                idx += 1;
            }
        }
    }
    debug_assert_eq!(idx, n3);
    // Homogeneous bubbles (x·m, y·m, z·m), degree k.
    for a in 0..=k {
        for b in 0..=(k - a) {
            let c = k - a - b;
            let m = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            mono[idx * 3] = x * m;
            mono[idx * 3 + 1] = y * m;
            mono[idx * 3 + 2] = z * m;
            idx += 1;
        }
    }
}

/// Divergence of each monomial generator (same layout as [`fill_mono_row`]).
fn fill_mono_div_row(k: usize, x: f64, y: f64, z: f64, n3: usize, divs: &mut [f64]) {
    let mut idx = 0usize;
    for deg in 0..=k {
        for a in 0..=deg {
            for b in 0..=(deg - a) {
                let c = deg - a - b;
                // Only the "active" component contributes.
                let dx = if a > 0 {
                    (a as f64) * x.powi(a as i32 - 1) * y.powi(b as i32) * z.powi(c as i32)
                } else {
                    0.0
                };
                let dy = if b > 0 {
                    (b as f64) * x.powi(a as i32) * y.powi(b as i32 - 1) * z.powi(c as i32)
                } else {
                    0.0
                };
                let dz = if c > 0 {
                    (c as f64) * x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32 - 1)
                } else {
                    0.0
                };
                divs[idx] = dx;
                idx += 1;
                divs[idx] = dy;
                idx += 1;
                divs[idx] = dz;
                idx += 1;
            }
        }
    }
    debug_assert_eq!(idx, n3);
    for a in 0..=k {
        for b in 0..=(k - a) {
            let c = k - a - b;
            let m = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            // div((x·m, y·m, z·m)) = (3 + a + b + c)·m = (k+3)·m
            divs[idx] = (k as f64 + 3.0) * m;
            idx += 1;
        }
    }
}

/// Reference node coordinates in slot order (MFEM `Nodes.IntPoint`).
pub(super) fn nodal_tet_dof_coords(k: usize) -> Vec<Vec<f64>> {
    let (pts, _) = mfem_nodal_dofs(k);
    pts.iter().map(|p| p.to_vec()).collect()
}

pub(super) fn eval_nodal_tet_basis(k: usize, xi: &[f64], values: &mut [f64]) {
    let n3 = (k + 1) * (k + 2) * (k + 3) / 2;
    let d = nodal_tet_data(k);
    let n = nodal_tet_n_dofs(k);
    let mut mono = vec![0.0f64; n * 3];
    fill_mono_row(k, xi[0], xi[1], xi[2], n3, &mut mono);
    for i in 0..n {
        let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
        for (j, c) in d.coeff[i * n..i * n + n].iter().enumerate() {
            vx += c * mono[j * 3];
            vy += c * mono[j * 3 + 1];
            vz += c * mono[j * 3 + 2];
        }
        values[i * 3] = vx;
        values[i * 3 + 1] = vy;
        values[i * 3 + 2] = vz;
    }
}

pub(super) fn eval_nodal_tet_div(k: usize, xi: &[f64], div_vals: &mut [f64]) {
    let d = nodal_tet_data(k);
    let n = nodal_tet_n_dofs(k);
    let n3 = (k + 1) * (k + 2) * (k + 3) / 2;
    let mut divs = vec![0.0f64; n];
    fill_mono_div_row(k, xi[0], xi[1], xi[2], n3, &mut divs);
    for i in 0..n {
        let mut s = 0.0;
        for (j, c) in d.coeff[i * n..i * n + n].iter().enumerate() {
            s += c * divs[j];
        }
        div_vals[i] = s;
    }
}

// ─── TetRT1 ─────────────────────────────────────────────────────────────────

/// Raviart-Thomas RT1 H(div) element on the reference tetrahedron — 15 DOFs,
/// order 1, MFEM nodal flux-dual semantics.
pub struct TetRT1;

impl VectorReferenceElement for TetRT1 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        nodal_tet_n_dofs(1)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        eval_nodal_tet_basis(1, xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        eval_nodal_tet_div(1, xi, div_vals);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        nodal_tet_dof_coords(1)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluates the order-`k` nodal basis through the trait object used by
    /// the space/assembler (TetRT1 for k=1; the shared helpers for k=2, which
    /// live in `tet_rt2::TetRT2`).
    fn eval_for(k: usize, xi: &[f64], n: usize, values: &mut [f64]) {
        if k == 1 {
            TetRT1.eval_basis_vec(xi, values);
        } else {
            eval_nodal_tet_basis(k, xi, values);
        }
        let _ = n;
    }

    /// Nodal duality for every order: `D_m(Φ_i) = Φ_i(x_m)·nk_m = δ_mi`.
    #[test]
    fn nodal_duality() {
        for k in 1..=2usize {
            let n = nodal_tet_n_dofs(k);
            let bop = gauss_legendre_01(k + 1).0;
            let p = k;
            let mut dual = vec![0.0f64; n * n];
            let mut phi = vec![0.0f64; n * 3];
            let mut m = 0usize;
            for (f, nk) in TET_NK.iter().enumerate() {
                for j in 0..=p {
                    for i in 0..=(p - j) {
                        let w = bop[i] + bop[j] + bop[p - i - j];
                        let b0 = bop[p - i - j] / w;
                        let b1 = bop[i] / w;
                        let b2 = bop[j] / w;
                        let pt: [f64; 3] = match f {
                            0 => [b0, b1, b2],
                            1 => [0.0, b2, b1],
                            2 => [b1, 0.0, b2],
                            _ => [b2, b1, 0.0],
                        };
                        eval_for(k, &pt, n, &mut phi);
                        for ii in 0..n {
                            dual[m * n + ii] = phi[ii * 3] * nk[0]
                                + phi[ii * 3 + 1] * nk[1]
                                + phi[ii * 3 + 2] * nk[2];
                        }
                        m += 1;
                    }
                }
            }
            let iop = gauss_legendre_01(k).0;
            for d in 0..k {
                for j in 0..(k - d) {
                    for i in 0..(k - d - j) {
                        let w = iop[i] + iop[j] + iop[d] + iop[k - 1 - i - j - d];
                        let pt = [iop[i] / w, iop[j] / w, iop[d] / w];
                        for nk in &TET_NK[1..4] {
                            eval_for(k, &pt, n, &mut phi);
                            for ii in 0..n {
                                dual[m * n + ii] = phi[ii * 3] * nk[0]
                                    + phi[ii * 3 + 1] * nk[1]
                                    + phi[ii * 3 + 2] * nk[2];
                            }
                            m += 1;
                        }
                    }
                }
            }
            assert_eq!(m, n);
            for (a, row) in dual.chunks(n).enumerate() {
                for (b, val) in row.iter().enumerate() {
                    let exp = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (val - exp).abs() < 1e-9,
                        "k={k}: D_{a}(phi_{b}) = {}, expected {exp}",
                        val
                    );
                }
            }
        }
    }

    /// Per-face normal support: face-supported basis functions vanish on the
    /// other faces; interior ones on all faces.
    #[test]
    fn per_face_normal_support() {
        let pts: [[f64; 3]; 4] = [
            [0.42, 0.27, 0.31],
            [0.0, 0.3, 0.4],
            [0.3, 0.0, 0.4],
            [0.3, 0.4, 0.0],
        ];
        for k in 1..=2usize {
            let n = nodal_tet_n_dofs(k);
            let face_dofs = (k + 1) * (k + 2) / 2;
            let mut phi = vec![0.0f64; n * 3];
            for (f, pt) in pts.iter().enumerate() {
                eval_for(k, pt, n, &mut phi);
                for j in 0..n {
                    let flux = phi[j * 3] * TET_NK[f][0]
                        + phi[j * 3 + 1] * TET_NK[f][1]
                        + phi[j * 3 + 2] * TET_NK[f][2];
                    let in_block = j < 4 * face_dofs && j / face_dofs == f;
                    if !in_block {
                        assert!(
                            flux.abs() < 1e-9,
                            "k={k}: face {f}: basis {j} normal trace = {flux}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn basis_and_div_finite() {
        for k in 1..=2usize {
            let n = nodal_tet_n_dofs(k);
            let mut v = vec![0.0; n * 3];
            let mut d = vec![0.0; n];
            let qr = tet_rule(4);
            let extra: Vec<Vec<f64>> = vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ];
            let pts: Vec<&Vec<f64>> = qr.points.iter().chain(extra.iter()).collect();
            for xi in &pts {
                eval_for(k, xi, n, &mut v);
                eval_nodal_tet_div(k, xi, &mut d);
                for val in v.iter().chain(d.iter()) {
                    assert!(val.is_finite(), "k={k} at {xi:?}");
                }
            }
        }
    }
}
