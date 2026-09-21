//! MFEM `RT_TetrahedronElement` port: **nodal** flux-dual RT bases on the
//! reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
//!
//! This module provides the shared MFEM nodal dof table used by [`TetRT1`]
//! (15 DOFs) and [`super::tet_rt2::TetRT2`] (36 DOFs), and the order-generic
//! [`TetRTNodal`] alias.  It reproduces MFEM's nodal DOF semantics exactly
//! (`fem/fe/fe_rt.cpp:899-999`):
//!
//! - **Nodes** sit on the faces at the Gauss-Legendre open points
//!   `poly1d.OpenPoints(p)` (degree `p = k` per face direction) following the
//!   MFEM per-face point patterns, plus interior points at degree `p-1` with
//!   one component sample (nk ∈ {−x, −y, −z}) per point.
//! - **Functionals** are point evaluations `Φ(x_m) · nk_m` with the MFEM
//!   unnormalised face-normal table `nk = {1,1,1, −1,0,0, 0,−1,0, 0,0,−1}`.
//! - The basis functions are the duals of these functionals
//!   (`D_m(Φ_i) = δ_mi`, MFEM `Ti.Factor(T)`); the basis construction itself
//!   (MFEM's Chebyshev-product Vandermonde) lives in
//!   [`super::tet_rtk`] since D540 and serves every tet RT order.
//!
//! Slot order (matches `HDivSpace::build_3d_tet`): faces in MFEM
//! `FaceVert` order `(1,2,3), (0,3,2), (0,1,3), (0,2,1)`, `(k+1)(k+2)/2`
//! grid nodes per face, then the interior component samples.

use std::sync::OnceLock;

use crate::quadrature::gauss_legendre_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};
use crate::raviart_thomas::tet_rtk::{eval_mfem_rt_tet_basis, eval_mfem_rt_tet_div};

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
/// construction (`tet_rtk`), `HDivSpace::interpolate_vector` and the discrete
/// operators (`crates/assembly/src/discrete_op.rs`).
///
/// D560/D571: at `k = 0` the table carries MFEM's **fixed-order**
/// `RT0TetFiniteElement` duals (`fem/fe/fe_fixed_order.cpp:6298`): the tet
/// RT0 space MFEM actually serves — `RT0_3DFECollection` (fe_coll.hpp:1473),
/// i.e. every legacy RT0 script/golden — stores `dof = f·adj(J)·n̂|F|` with
/// `nk = {{.5,.5,.5}, {-.5,0,0}, {0,-.5,0}, {0,0,-.5}}` (face normals scaled
/// by the reference-face area) and a basis twice the generic
/// `RT_TetrahedronElement(0)`'s (`CalcVShape = 2(x,y,z), 2(x−1,y,z), …`,
/// `fe_fixed_order.cpp:6267-6287`).  Both scalings flow from this one table:
/// the `TetRTk::new(0)` Vandermonde is built from these `nk` (basis ×2),
/// while `HDivSpace::interpolate_vector`'s flux rows read them directly
/// (functional ×½) — so the stored dof is exactly `RT0TetFiniteElement::
/// Project`'s (`fe_fixed_order.cpp:6356-6370`) and `W = φ·nk` stays the
/// identity.  Orders `k ≥ 1` keep the generic `RT_TetrahedronElement(k)`
/// table (`nk` unnormalised), which is what `RT_FECollection(p, 3)` serves
/// (`fe_coll.cpp:2575`).
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
                    // D560/D571: k = 0 carries `RT0TetFiniteElement::nk`
                    // (n̂·|F|, `fe_fixed_order.cpp:6298`) — see the fn docs.
                    nks.push(if k == 0 {
                        [nk[0] * 0.5, nk[1] * 0.5, nk[2] * 0.5]
                    } else {
                        *nk
                    });
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

/// Reference node coordinates in slot order (MFEM `Nodes.IntPoint`).
pub(super) fn nodal_tet_dof_coords(k: usize) -> Vec<Vec<f64>> {
    let (pts, _) = mfem_nodal_dofs(k);
    pts.iter().map(|p| p.to_vec()).collect()
}

// ─── TetRT1 ─────────────────────────────────────────────────────────────────

/// Raviart-Thomas RT1 H(div) element on the reference tetrahedron — 15 DOFs,
/// order 1, MFEM nodal flux-dual semantics (D540 point-dual engine).
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
        eval_mfem_rt_tet_basis(1, xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        eval_mfem_rt_tet_div(1, xi, div_vals);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        crate::quadrature::tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        nodal_tet_dof_coords(1)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::raviart_thomas::tet_rtk::TetRTk;

    /// Nodal duality for every order: `D_m(Φ_i) = Φ_i(x_m)·nk_m = δ_mi`.
    #[test]
    fn nodal_duality() {
        for k in 1..=2usize {
            let n = nodal_tet_n_dofs(k);
            let (pts, nks) = mfem_nodal_dofs(k);
            let mut dual = vec![0.0f64; n * n];
            let mut phi = vec![0.0f64; n * 3];
            let e = TetRTk::new(k);
            for (m, (pt, nk)) in pts.iter().zip(nks.iter()).enumerate() {
                e.eval_basis_vec(pt, &mut phi);
                for (ii, row) in dual[m * n..m * n + n].iter_mut().enumerate() {
                    *row = phi[ii * 3] * nk[0] + phi[ii * 3 + 1] * nk[1] + phi[ii * 3 + 2] * nk[2];
                }
            }
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
            let e = TetRTk::new(k);
            for (f, pt) in pts.iter().enumerate() {
                e.eval_basis_vec(pt, &mut phi);
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
            let qr = crate::quadrature::tet_rule(4);
            let extra: Vec<Vec<f64>> = vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ];
            let pts: Vec<&Vec<f64>> = qr.points.iter().chain(extra.iter()).collect();
            let e = TetRTk::new(k);
            for xi in &pts {
                e.eval_basis_vec(xi, &mut v);
                e.eval_div(xi, &mut d);
                for val in v.iter().chain(d.iter()) {
                    assert!(val.is_finite(), "k={k} at {xi:?}");
                }
            }
        }
    }
}
