//! Rannacher-Turek Q1_rot nonconforming quadrilateral element.
//!
//! Reference domain: [-1,1]². Shape functions: Span{1, x, y, x²-y²}.
//! DOFs: edge-average values (4 per edge). Vector version: 8 DOFs for Stokes.
//!
//! Built via Vandermonde from monomials + face-average DOFs with Gauss quadrature.

#[allow(unused_imports)]
use crate::quadrature;
use crate::reference::{QuadratureRule, ReferenceElement, VectorReferenceElement};

const EDGE_GEOM: [([f64; 2], [f64; 2]); 4] = [
    ([-1.0, -1.0], [1.0, -1.0]), // bottom
    ([1.0, -1.0], [1.0, 1.0]),   // right
    ([-1.0, 1.0], [1.0, 1.0]),   // top   (parameterized -1→1)
    ([-1.0, -1.0], [-1.0, 1.0]), // left
];

/// Monomial evaluation at (x, y).
fn eval_mm(x: f64, y: f64) -> [f64; 4] {
    [1.0, x, y, x * x - y * y]
}

/// Build 4×4 Vandermonde: DOF_i(m_j) where DOF is edge-average.
fn build_vm() -> [[f64; 4]; 4] {
    let (gp, gw) = gauss4();
    let mut v = [[0.0_f64; 4]; 4];
    for (ei, &(s, e)) in EDGE_GEOM.iter().enumerate() {
        let dx = e[0] - s[0];
        let dy = e[1] - s[1];
        let len = (dx * dx + dy * dy).sqrt();
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
        for j in 0..4 {
            v[ei][j] /= len;
        }
    }
    v
}

fn gauss4() -> ([f64; 4], [f64; 4]) {
    (
        [
            0.0694318442029737,
            0.3300094782075719,
            0.6699905217924281,
            0.9305681557970263,
        ],
        [
            0.1739274225687269,
            0.3260725774312731,
            0.3260725774312731,
            0.1739274225687269,
        ],
    )
}

/// Invert 4×4 matrix.
fn inv4(mut a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut inv = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        inv[i][i] = 1.0;
    }
    for c in 0..4 {
        let mut mr = c;
        let mut mv = a[c][c].abs();
        for r in (c + 1)..4 {
            let x = a[r][c].abs();
            if x > mv {
                mv = x;
                mr = r;
            }
        }
        a.swap(c, mr);
        inv.swap(c, mr);
        let ip = 1.0 / a[c][c];
        for j in 0..4 {
            a[c][j] *= ip;
            inv[c][j] *= ip;
        }
        for r in 0..4 {
            if r == c {
                continue;
            }
            let f = a[r][c];
            for j in 0..4 {
                a[r][j] -= f * a[c][j];
                inv[r][j] -= f * inv[c][j];
            }
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
        for i in 0..4 {
            for j in 0..4 {
                c[i][j] = vi[j][i];
            }
        }
        c
    })
}

fn eval_all_monos(x: f64, y: f64) -> [f64; 4] {
    eval_mm(x, y)
}

// ─── Scalar Q1_rot ─────────────────────────────────────────────────────────

pub struct QuadQ1Rot;

impl QuadQ1Rot {
    pub fn eval_basis(xi: &[f64], vals: &mut [f64]) {
        let c = coeff();
        let mv = eval_all_monos(xi[0], xi[1]);
        for i in 0..4 {
            vals[i] = c[i][0] * mv[0] + c[i][1] * mv[1] + c[i][2] * mv[2] + c[i][3] * mv[3];
        }
    }

    pub fn eval_grad_basis(xi: &[f64], grads: &mut [f64]) {
        let c = coeff();
        let (x, y) = (xi[0], xi[1]);
        let dm = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0 * x, -2.0 * y]];
        for i in 0..4 {
            grads[i * 2] =
                c[i][0] * dm[0][0] + c[i][1] * dm[1][0] + c[i][2] * dm[2][0] + c[i][3] * dm[3][0];
            grads[i * 2 + 1] =
                c[i][0] * dm[0][1] + c[i][1] * dm[1][1] + c[i][2] * dm[2][1] + c[i][3] * dm[3][1];
        }
    }
}

/// Scalar Q1_rot reference element on [-1,1]² (4 edge-average DOFs).
pub struct Q1RotRef;
impl ReferenceElement for Q1RotRef {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        4
    }
    fn eval_basis(&self, xi: &[f64], vals: &mut [f64]) {
        QuadQ1Rot::eval_basis(xi, vals);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        QuadQ1Rot::eval_grad_basis(xi, grads);
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        let o = (order as usize).max(3);
        let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(o);
        let nq = x1d.len();
        let mut pts = Vec::with_capacity(nq * nq);
        let mut wts = Vec::with_capacity(nq * nq);
        for i in 0..nq {
            for j in 0..nq {
                pts.push(vec![x1d[i], x1d[j]]);
                wts.push(w1d[i] * w1d[j]);
            }
        }
        QuadratureRule {
            points: pts,
            weights: wts,
        }
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, -1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ]
    }
}

// ─── Vector Q1_rot (8 DOFs: 2 comps × 4 edges) ─────────────────────────────

pub struct QuadQ1RotVec;

impl VectorReferenceElement for QuadQ1RotVec {
    fn n_dofs(&self) -> usize {
        8
    }
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let o = (order as usize).max(3);
        let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(o);
        let nq = x1d.len();
        let mut pts = Vec::with_capacity(nq * nq);
        let mut wts = Vec::with_capacity(nq * nq);
        for i in 0..nq {
            for j in 0..nq {
                pts.push(vec![x1d[i], x1d[j]]);
                wts.push(w1d[i] * w1d[j]);
            }
        }
        QuadratureRule {
            points: pts,
            weights: wts,
        }
    }

    fn eval_basis_vec(&self, xi: &[f64], vals: &mut [f64]) {
        let mut phi = [0.0_f64; 4];
        QuadQ1Rot::eval_basis(xi, &mut phi);
        for i in 0..4 {
            vals[i * 2] = phi[i];
            vals[i * 2 + 1] = phi[i];
        }
    }

    fn eval_curl(&self, xi: &[f64], curl: &mut [f64]) {
        let mut g = [0.0_f64; 8];
        QuadQ1Rot::eval_grad_basis(xi, &mut g);
        for i in 0..4 {
            curl[i] = g[i * 2] + g[i * 2 + 1];
        } // dΦ/dx + dΦ/dy (since u=v=Φ)
    }

    fn eval_div(&self, xi: &[f64], div: &mut [f64]) {
        let mut g = [0.0_f64; 8];
        QuadQ1Rot::eval_grad_basis(xi, &mut g);
        for i in 0..4 {
            div[i] = g[i * 2] + g[i * 2 + 1];
        }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, -1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ]
    }
}

// ─── 3D Rannacher-Turek: RotTriLinearHex (6 face-midpoint dofs) ─────────────

/// MFEM `RotTriLinearHexFiniteElement` — the 3-D arm of
/// `LinearNonConf3DFECollection` (`fem/fe_coll.hpp:1044`): rotated bilinear
/// (Rannacher–Turek) hexahedron element, 6 dofs = face-midpoint *values* on
/// the reference hex `[0,1]^3`.
///
/// Space: span{1, x, y, z, x²−y², y²−z²} in the MFEM cube coordinates
/// rescaled to `[-1,1]³`; branching-free closed form.  Ported verbatim from
/// MFEM 4.10 `fem/fe/fe_fixed_order.cpp:6763` (`CalcShape` :6791,
/// `CalcDShape` :6808); dof numbering = MFEM `Nodes` order (bottom, front,
/// right, back, left, top face centers).  Note this element's dofs are point
/// values at face centers — unlike the 2-D `Q1RotRef` whose dofs are edge
/// *averages* (Stokes usage).
///
/// Unlike fem-rs' own `HexQ1`/`HexQk` ([-1,1]³), the reference domain follows
/// MFEM's cube convention `[0,1]³` (same as `HexRT1`/`mfem_hex_nodal_dofs`)
/// so the probe acceptance is a 1:1 point-set comparison.
pub struct RotTriLinearHex;

impl RotTriLinearHex {
    /// MFEM's affine rescale of the `[0,1]³` reference hex onto `[-1,1]³`.
    #[inline]
    fn unit2center(u: f64) -> f64 {
        2.0 * u - 1.0
    }
}

impl ReferenceElement for RotTriLinearHex {
    fn dim(&self) -> u8 {
        3
    }
    /// MFEM declares `NodalFiniteElement(3, CUBE, 6, 2, Qk)`.
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        6
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = Self::unit2center(xi[0]);
        let y = Self::unit2center(xi[1]);
        let z = Self::unit2center(xi[2]);
        let f5 = x * x - y * y;
        let f6 = y * y - z * z;

        // Verbatim MFEM expressions (operation order preserved).
        values[0] = (1.0 / 6.0) * (1.0 - 3.0 * z - f5 - 2.0 * f6);
        values[1] = (1.0 / 6.0) * (1.0 - 3.0 * y - f5 + f6);
        values[2] = (1.0 / 6.0) * (1.0 + 3.0 * x + 2.0 * f5 + f6);
        values[3] = (1.0 / 6.0) * (1.0 + 3.0 * y - f5 + f6);
        values[4] = (1.0 / 6.0) * (1.0 - 3.0 * x + 2.0 * f5 + f6);
        values[5] = (1.0 / 6.0) * (1.0 + 3.0 * z - f5 - 2.0 * f6);
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // grads[i * 3 + j] = dphi_i/dxi_j (MFEM reference [0,1]^3 coords),
        // MFEM dshape(i,j) with a = 2/3.
        const A: f64 = 2.0 / 3.0;
        let xt = A * (1.0 - 2.0 * xi[0]);
        let yt = A * (1.0 - 2.0 * xi[1]);
        let zt = A * (1.0 - 2.0 * xi[2]);

        grads[0 * 3] = xt;
        grads[0 * 3 + 1] = yt;
        grads[0 * 3 + 2] = -1.0 - 2.0 * zt;

        grads[1 * 3] = xt;
        grads[1 * 3 + 1] = -1.0 - 2.0 * yt;
        grads[1 * 3 + 2] = zt;

        grads[2 * 3] = 1.0 - 2.0 * xt;
        grads[2 * 3 + 1] = yt;
        grads[2 * 3 + 2] = zt;

        grads[3 * 3] = xt;
        grads[3 * 3 + 1] = 1.0 - 2.0 * yt;
        grads[3 * 3 + 2] = zt;

        grads[4 * 3] = -1.0 - 2.0 * xt;
        grads[4 * 3 + 1] = yt;
        grads[4 * 3 + 2] = zt;

        grads[5 * 3] = xt;
        grads[5 * 3 + 1] = yt;
        grads[5 * 3 + 2] = 1.0 - 2.0 * zt;
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        crate::quadrature::hex_rule(order)
    }

    /// MFEM `Nodes.IntPoint` face centers: bottom, front, right, back, left,
    /// top.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![1.0, 0.5, 0.5],
            vec![0.5, 1.0, 0.5],
            vec![0.0, 0.5, 0.5],
            vec![0.5, 0.5, 1.0],
        ]
    }
}

// ─── 3D nonconforming P1 tetrahedron (4 face-center dofs) ──────────────────

/// MFEM `P1TetNonConfFiniteElement` — the TETRAHEDRON arm of
/// `LinearNonConf3DFECollection` (`fem/fe_coll.hpp:1048`): the lowest-order
/// nonconforming Crouzeix–Raviart-type tetrahedron element, 4 dofs = point
/// values at the *centroids of the four faces* of the reference tet
/// `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
///
/// Space: span{1, x, y, z} composed as `φ₀ = 1 − 3L₀`, `φᵢ = 1 − 3Lᵢ` with
/// `L₀ = 1 − x − y − z`, `L₁ = x`, `L₂ = y`, `L₃ = z` — continuous across
/// inter-element faces only in the edge-average sense (the function value at
/// a face is its centroid value, so the trace is not single-valued: hence
/// "nonconforming").  Ported verbatim from MFEM 4.10
/// `fem/fe/fe_fixed_order.cpp:2959` (`CalcShape` :2980, `CalcDShape`
/// :2992); dof numbering = MFEM `Nodes` order (face opposite vertex 0 first,
/// then faces x=0, y=0, z=0).  Branch-free closed form.
///
/// MFEM declares `NodalFiniteElement(3, TETRAHEDRON, 4, 1)` (order tag 1).
pub struct P1TetNonConf;

impl ReferenceElement for P1TetNonConf {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        4
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        // Verbatim MFEM operation order: L0 = 1.0 - L1 - L2 - L3.
        let l1 = xi[0];
        let l2 = xi[1];
        let l3 = xi[2];
        let l0 = 1.0 - l1 - l2 - l3;
        values[0] = 1.0 - 3.0 * l0;
        values[1] = 1.0 - 3.0 * l1;
        values[2] = 1.0 - 3.0 * l2;
        values[3] = 1.0 - 3.0 * l3;
    }

    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) {
        // Constant dshape table, MFEM rows verbatim.
        grads[0 * 3..0 * 3 + 3].copy_from_slice(&[3.0, 3.0, 3.0]);
        grads[1 * 3..1 * 3 + 3].copy_from_slice(&[-3.0, 0.0, 0.0]);
        grads[2 * 3..2 * 3 + 3].copy_from_slice(&[0.0, -3.0, 0.0]);
        grads[3 * 3..3 * 3 + 3].copy_from_slice(&[0.0, 0.0, -3.0]);
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        crate::quadrature::tet_rule(order)
    }

    /// MFEM `Nodes.IntPoint`: centroid of the face opposite vertex 0, then
    /// centroids of the faces x = 0, y = 0, z = 0.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        const T: f64 = 0.33333333333333333333;
        vec![vec![T, T, T], vec![0.0, T, T], vec![T, 0.0, T], vec![T, T, 0.0]]
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q1rot_partition_of_unity() {
        let mut phi = [0.0_f64; 4];
        for &p in &[[0.0, 0.0], [0.5, 0.3], [-0.7, 0.8], [1.0, -1.0]] {
            QuadQ1Rot::eval_basis(&p, &mut phi);
            assert!(
                (phi.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                "POU failed at {p:?}"
            );
        }
    }

    #[test]
    fn q1rot_edge_avg_delta() {
        // Q1_rot DOFs are edge averages: ∫_edge_j φ_i dt / len_j = δ_ij.
        let (gp, gw) = gauss4();
        let mut phi = [0.0_f64; 4];
        for (ei, &(s, e)) in EDGE_GEOM.iter().enumerate() {
            let dx = e[0] - s[0];
            let dy = e[1] - s[1];
            let len = (dx * dx + dy * dy).sqrt();
            let mut avg = [0.0_f64; 4];
            for (&t, &w) in gp.iter().zip(gw.iter()) {
                let pt = [s[0] + t * dx, s[1] + t * dy];
                QuadQ1Rot::eval_basis(&pt, &mut phi);
                for i in 0..4 {
                    avg[i] += w * phi[i] * len;
                }
            }
            for i in 0..4 {
                let expected = if i == ei { 1.0 } else { 0.0 };
                assert!(
                    (avg[i] / len - expected).abs() < 1e-12,
                    "edge {ei} DOF {i}: avg={}",
                    avg[i] / len
                );
            }
        }
    }

    #[test]
    fn q1rot_basis_values_at_origins() {
        let mut phi = [0.0_f64; 4];
        QuadQ1Rot::eval_basis(&[0.0, 0.0], &mut phi);
        for v in &phi {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn q1rot_vec_size() {
        assert_eq!(QuadQ1RotVec.n_dofs(), 8);
    }

    #[test]
    fn q1rot_vec_basis_finite() {
        let e = QuadQ1RotVec;
        let mut v = vec![0.0; 8];
        for p in &e.quadrature(3).points {
            e.eval_basis_vec(p, &mut v);
            for x in &v {
                assert!(x.is_finite());
            }
        }
    }

    #[test]
    fn q1rot_vec_curl_finite() {
        let e = QuadQ1RotVec;
        let mut c = vec![0.0; 4];
        for p in &e.quadrature(3).points {
            e.eval_curl(p, &mut c);
            for x in &c {
                assert!(x.is_finite());
            }
        }
    }

    #[test]
    fn p1tet_nonconf_partition_of_unity() {
        let e = P1TetNonConf;
        let mut v = vec![0.0; 4];
        for p in &[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.5, 0.25, 0.0],
            [0.0, 0.5, 0.5],
        ] {
            e.eval_basis(p, &mut v);
            let s: f64 = v.iter().sum();
            assert!((s - 1.0).abs() < 1e-15, "POU at {p:?}: sum={s}");
        }
    }

    #[test]
    fn p1tet_nonconf_face_center_delta() {
        // Nodal dof = face-centroid value; the 1/3 coordinates are inexact
        // binary doubles, so the delta property holds to ~1 ulp of 1.
        let e = P1TetNonConf;
        let nodes = e.dof_coords();
        let n = e.n_dofs();
        let mut v = vec![0.0; n];
        for (j, node) in nodes.iter().enumerate() {
            e.eval_basis(node, &mut v);
            for (i, x) in v.iter().enumerate() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (x - expect).abs() < 1e-15,
                    "phi_{i} at node_{j} = {x} (expect {expect})"
                );
            }
        }
    }

    #[test]
    fn p1tet_nonconf_gradient_sum_zero() {
        let e = P1TetNonConf;
        let mut g = vec![0.0; 12];
        for p in &e.quadrature(3).points {
            e.eval_grad_basis(p, &mut g);
            for j in 0..3 {
                let s: f64 = (0..4).map(|i| g[i * 3 + j]).sum();
                assert_eq!(s, 0.0, "sum dphi dir {j} at {p:?}");
            }
        }
    }

    #[test]
    fn p1tet_nonconf_interior_finite() {
        let e = P1TetNonConf;
        let mut v = vec![0.0; 4];
        let mut g = vec![0.0; 12];
        for p in &e.quadrature(5).points {
            e.eval_basis(p, &mut v);
            e.eval_grad_basis(p, &mut g);
            assert!(v.iter().all(|x| x.is_finite()));
            assert!(g.iter().all(|x| x.is_finite()));
        }
    }
}
