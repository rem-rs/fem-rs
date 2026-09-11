//! Arbitrary-order Nedelec-I element on the reference tetrahedron.
//!
//! 1:1 port of MFEM `ND_TetrahedronElement(p)` (`fem/fe/fe_nd.cpp`, MFEM 4.10).
//!
//! Reference vertices: v₀=(0,0,0), v₁=(1,0,0), v₂=(0,1,0), v₃=(0,0,1).
//!
//! # Space
//! `N_p = P_{p-1}³ ⊕ x^⊥ P̃_{p-2}`, dim = `p(p+2)(p+3)/2`: `6p` edge DOFs,
//! `4p(p−1)` face DOFs, `p(p−1)(p−2)/2·3` interior DOFs.
//!
//! # Construction (MFEM `ND_TetrahedronElement` verbatim)
//!
//! 1. **Raw basis `u`** built from the hierarchical 1-D basis
//!    `shape_x/shape_y/shape_z/shape_l = Poly_1D::CalcChebyshev(p−1, ·)`
//!    (`T_j(2x−1)` etc.), with `c = 1/4`:
//!    * `s·e_x, s·e_y, s·e_z` for `s = shape_x(i)shape_y(j)shape_z(k)shape_l(l)`,
//!      `i+j+k ≤ p−1`, `l = p−1−i−j−k`;
//!    * `s·(y−c, −(x−c), 0)` and `s·(z−c, 0, −(x−c))` for
//!      `s = shape_x(p−1−j−k)shape_y(j)shape_z(k)`;
//!    * `s·(0, z−c, −(y−c))` for `s = shape_y(p−1−k)shape_z(k)`.
//! 2. **Square Vandermonde** `A[b][m] = σ_m(u_b)`, inverted exactly as MFEM
//!    does (`Ti.Factor(T)`): `Φ_i = Σ_b A⁻¹[i][b]·u_b`.
//!
//! # DOF semantics (D38 — MFEM nodal point-value functionals)
//!
//! Every DOF is a **point evaluation of the tangential component**
//!
//! ```text
//! σ_i(Φ) = Φ(x_i) · t̂_i
//! ```
//!
//! at the MFEM `FE::Nodes` point `x_i` along the fixed *unnormalized*
//! reference tangent `t̂_i = tk + 3*dof2tk[i]` with MFEM's `tk` table
//! `(1,0,0) (0,1,0) (0,0,1) (−1,1,0) (−1,0,1) (0,−1,1)`:
//!
//! * **Edges** (`p` per edge): Gauss-Legendre points `eop = OpenPoints(p−1)`,
//!   ascending along each edge's named direction; edge (1,2) reads
//!   `(eop[p−1−i], eop[i], 0)` etc. (MFEM's `Nodes` construction).
//! * **Faces** (`p(p−1)` per face): the face-interior GL points
//!   `(fop[i], fop[j], fop[p−2−i−j])/w`, `fop = OpenPoints(p−2)`, with MFEM's
//!   `dof2tk` tangent pairs — face (1,2,3): `(−1,1,0)`, `(−1,0,1)`;
//!   (0,3,2): `(0,0,1)`, `(0,1,0)`; (0,1,3): `(1,0,0)`, `(0,0,1)`;
//!   (0,2,1): `(0,1,0)`, `(1,0,0)`.
//! * **Interior** (`3` per point): `iop = OpenPoints(p−3)` barycentric points
//!   with tangents `(1,0,0)`, `(0,1,0)`, `(0,0,1)`.
//!
//! The GL point sets are symmetric about the edge midpoint, so an edge
//! reversal maps the edge DOFs to a **signed anti-diagonal permutation**
//! (MFEM `SegDofOrd`), which is what makes cross-element edge pairing
//! conforming.  The face DOF pairs at a shared face are related between the
//! two adjacent elements by a full 2×2 change of basis (MFEM's
//! `ND_DofTransformation::T(ori)` family) — not a signed permutation — see
//! `HCurlSpace` for the cross-element pairing.
//!
//! `TetND2` is the explicit order-2 specialization of the same functionals.

use crate::quadrature::{gauss_legendre_01, tet_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};
use std::sync::OnceLock;

pub(crate) use crate::nedelec::tri_ndk::{chebyshev_d, invert_dense};

/// Bubble centre used by MFEM's raw basis (`ND_TetrahedronElement::c`).
const C_BUBBLE: f64 = 1.0 / 4.0;

struct TetNDkData {
    /// Row-major `n × n`: `Φ_i = Σ_b ti[i][b]·u_b` (`ti = A⁻¹`, MFEM `Ti`).
    ti: Vec<f64>,
    n: usize,
}

/// `(point, tangent)` of every local DOF in MFEM `Nodes.IntPoint`/`dof2tk`
/// order — the single source of truth for the element's DOF layout.
pub(crate) fn dof_table(k: usize) -> Vec<([f64; 3], [f64; 3])> {
    let mut out = Vec::with_capacity(k * (k + 2) * (k + 3) / 2);
    let eop = gauss_legendre_01(k).0; // OpenPoints(k-1): k nodes
    let pm1 = k - 1;
    // Edges, in MFEM Geometry::Constants<TETRAHEDRON>::Edges order.
    for i in 0..k {
        out.push(([eop[i], 0.0, 0.0], [1.0, 0.0, 0.0])); // (0,1)
    }
    for i in 0..k {
        out.push(([0.0, eop[i], 0.0], [0.0, 1.0, 0.0])); // (0,2)
    }
    for i in 0..k {
        out.push(([0.0, 0.0, eop[i]], [0.0, 0.0, 1.0])); // (0,3)
    }
    for i in 0..k {
        out.push(([eop[pm1 - i], eop[i], 0.0], [-1.0, 1.0, 0.0])); // (1,2)
    }
    for i in 0..k {
        out.push(([eop[pm1 - i], 0.0, eop[i]], [-1.0, 0.0, 1.0])); // (1,3)
    }
    for i in 0..k {
        out.push(([0.0, eop[pm1 - i], eop[i]], [0.0, -1.0, 1.0])); // (2,3)
    }
    // Faces (MFEM face order (1,2,3), (0,3,2), (0,1,3), (0,2,1)).
    if k >= 2 {
        let fop = gauss_legendre_01(k - 1).0; // OpenPoints(k-2): k-1 nodes
        let pm2 = k - 2;
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let w = fop[i] + fop[j] + fop[pm2 - i - j];
                out.push((
                    [fop[pm2 - i - j] / w, fop[i] / w, fop[j] / w],
                    [-1.0, 1.0, 0.0],
                ));
                out.push((
                    [fop[pm2 - i - j] / w, fop[i] / w, fop[j] / w],
                    [-1.0, 0.0, 1.0],
                ));
            }
        }
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let w = fop[i] + fop[j] + fop[pm2 - i - j];
                out.push(([0.0, fop[j] / w, fop[i] / w], [0.0, 0.0, 1.0]));
                out.push(([0.0, fop[j] / w, fop[i] / w], [0.0, 1.0, 0.0]));
            }
        }
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let w = fop[i] + fop[j] + fop[pm2 - i - j];
                out.push(([fop[i] / w, 0.0, fop[j] / w], [1.0, 0.0, 0.0]));
                out.push(([fop[i] / w, 0.0, fop[j] / w], [0.0, 0.0, 1.0]));
            }
        }
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let w = fop[i] + fop[j] + fop[pm2 - i - j];
                out.push(([fop[j] / w, fop[i] / w, 0.0], [0.0, 1.0, 0.0]));
                out.push(([fop[j] / w, fop[i] / w, 0.0], [1.0, 0.0, 0.0]));
            }
        }
    }
    // Interior.
    if k >= 3 {
        let iop = gauss_legendre_01(k - 2).0; // OpenPoints(k-3): k-2 nodes
        let pm3 = k - 3;
        for kk in 0..=pm3 {
            for j in 0..=(pm3 - kk) {
                for i in 0..=(pm3 - kk - j) {
                    let w = iop[i] + iop[j] + iop[kk] + iop[pm3 - i - j - kk];
                    let p = [iop[i] / w, iop[j] / w, iop[kk] / w];
                    out.push((p, [1.0, 0.0, 0.0]));
                    out.push((p, [0.0, 1.0, 0.0]));
                    out.push((p, [0.0, 0.0, 1.0]));
                }
            }
        }
    }
    out
}

/// Evaluate the raw `u` basis (MFEM `ND_TetrahedronElement::CalcVShape`) into
/// the flat array `ub[b*3 .. b*3+3]`.
fn eval_u(x: f64, y: f64, z: f64, pm1: usize, ub: &mut [f64]) {
    let (tx, _) = chebyshev_d(pm1, x);
    let (ty, _) = chebyshev_d(pm1, y);
    let (tz, _) = chebyshev_d(pm1, z);
    let (tl, _) = chebyshev_d(pm1, 1.0 - x - y - z);
    let mut b = 0usize;
    for k in 0..=pm1 {
        for j in 0..=(pm1 - k) {
            for i in 0..=(pm1 - j - k) {
                let s = tx[i] * ty[j] * tz[k] * tl[pm1 - i - j - k];
                ub[b * 3] = s;
                ub[b * 3 + 1] = 0.0;
                ub[b * 3 + 2] = 0.0;
                ub[b * 3 + 3] = 0.0;
                ub[b * 3 + 4] = s;
                ub[b * 3 + 5] = 0.0;
                ub[b * 3 + 6] = 0.0;
                ub[b * 3 + 7] = 0.0;
                ub[b * 3 + 8] = s;
                b += 3;
            }
        }
    }
    for k in 0..=pm1 {
        for j in 0..=(pm1 - k) {
            let s = tx[pm1 - j - k] * ty[j] * tz[k];
            ub[b * 3] = s * (y - C_BUBBLE);
            ub[b * 3 + 1] = -s * (x - C_BUBBLE);
            ub[b * 3 + 2] = 0.0;
            ub[b * 3 + 3] = s * (z - C_BUBBLE);
            ub[b * 3 + 4] = 0.0;
            ub[b * 3 + 5] = -s * (x - C_BUBBLE);
            b += 2;
        }
    }
    for k in 0..=pm1 {
        let s = ty[pm1 - k] * tz[k];
        ub[b * 3] = 0.0;
        ub[b * 3 + 1] = s * (z - C_BUBBLE);
        ub[b * 3 + 2] = -s * (y - C_BUBBLE);
        b += 1;
    }
    debug_assert_eq!(b * 3, ub.len());
}

/// Evaluate the raw `u` basis curls (MFEM
/// `ND_TetrahedronElement::CalcCurlShape`).
fn eval_u_curl(x: f64, y: f64, z: f64, pm1: usize, uc: &mut [f64]) {
    let (tx, dx) = chebyshev_d(pm1, x);
    let (ty, dy) = chebyshev_d(pm1, y);
    let (tz, dz) = chebyshev_d(pm1, z);
    let (tl, dl) = chebyshev_d(pm1, 1.0 - x - y - z);
    let (px, py, pz) = (x - C_BUBBLE, y - C_BUBBLE, z - C_BUBBLE);
    let mut b = 0usize;
    for k in 0..=pm1 {
        for j in 0..=(pm1 - k) {
            for i in 0..=(pm1 - j - k) {
                let l = pm1 - i - j - k;
                let ddx = (dx[i] * tl[l] - tx[i] * dl[l]) * ty[j] * tz[k];
                let ddy = (dy[j] * tl[l] - ty[j] * dl[l]) * tx[i] * tz[k];
                let ddz = (dz[k] * tl[l] - tz[k] * dl[l]) * tx[i] * ty[j];
                uc[b * 3] = 0.0;
                uc[b * 3 + 1] = ddz;
                uc[b * 3 + 2] = -ddy;
                uc[b * 3 + 3] = -ddz;
                uc[b * 3 + 4] = 0.0;
                uc[b * 3 + 5] = ddx;
                uc[b * 3 + 6] = ddy;
                uc[b * 3 + 7] = -ddx;
                uc[b * 3 + 8] = 0.0;
                b += 3;
            }
        }
    }
    for k in 0..=pm1 {
        for j in 0..=(pm1 - k) {
            let i = pm1 - j - k;
            // curl of s*(y−c, −(x−c), 0), s = tx[i]·ty[j]·tz[k]
            uc[b * 3] = tx[i] * px * ty[j] * dz[k];
            uc[b * 3 + 1] = tx[i] * ty[j] * py * dz[k];
            uc[b * 3 + 2] = -((dx[i] * px + tx[i]) * ty[j] * tz[k]
                + (dy[j] * py + ty[j]) * tx[i] * tz[k]);
            // curl of s*(z−c, 0, −(x−c))
            uc[b * 3 + 3] = -tx[i] * px * dy[j] * tz[k];
            uc[b * 3 + 4] = tx[i] * ty[j] * (dz[k] * pz + tz[k])
                + (dx[i] * px + tx[i]) * ty[j] * tz[k];
            uc[b * 3 + 5] = -tx[i] * dy[j] * tz[k] * pz;
            b += 2;
        }
    }
    for k in 0..=pm1 {
        let j = pm1 - k;
        // curl of s*(0, z−c, −(y−c)), s = ty[j]·tz[k]
        uc[b * 3] = -((dy[j] * py + ty[j]) * tz[k] + ty[j] * (dz[k] * pz + tz[k]));
        uc[b * 3 + 1] = 0.0;
        uc[b * 3 + 2] = 0.0;
        b += 1;
    }
    debug_assert_eq!(b * 3, uc.len());
}

fn tet_data(k: usize) -> &'static TetNDkData {
    static CACHE: [OnceLock<TetNDkData>; 9] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[k - 1].get_or_init(|| build_tet_data(k))
}

fn build_tet_data(k: usize) -> TetNDkData {
    let n = k * (k + 2) * (k + 3) / 2;
    let table = dof_table(k);
    let mut a = vec![0.0_f64; n * n]; // A[b][m] = σ_m(u_b)
    let mut ub = vec![0.0_f64; 3 * n];
    for (m, (x, t)) in table.iter().enumerate() {
        eval_u(x[0], x[1], x[2], k - 1, &mut ub);
        for b in 0..n {
            a[b * n + m] =
                ub[b * 3] * t[0] + ub[b * 3 + 1] * t[1] + ub[b * 3 + 2] * t[2];
        }
    }
    TetNDkData {
        ti: invert_dense(n, &a, "TetNDk"),
        n,
    }
}

pub struct TetNDk {
    order: usize,
}

impl TetNDk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "TetNDk requires order ≥ 1");
        TetNDk { order: p }
    }

    /// Reference tangents `t̂_i` of every local DOF (MFEM `tk`/`dof2tk`).
    pub fn dof_tangents(&self) -> Vec<[f64; 3]> {
        dof_table(self.order).into_iter().map(|(_, t)| t).collect()
    }
}

impl VectorReferenceElement for TetNDk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        let k = self.order;
        k * (k + 2) * (k + 3) / 2
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;

        if k == 1 {
            // MFEM 4.10 Nedelec1TetFiniteElement::CalcVShape
            let (x, y, z) = (xi[0], xi[1], xi[2]);
            values[0] = 1.0 - y - z; values[1] = x; values[2] = x;
            values[3] = y; values[4] = 1.0 - x - z; values[5] = y;
            values[6] = z; values[7] = z; values[8] = 1.0 - x - y;
            values[9] = -y; values[10] = x; values[11] = 0.0;
            values[12] = -z; values[13] = 0.0; values[14] = x;
            values[15] = 0.0; values[16] = -z; values[17] = y;
            return;
        }

        let d = tet_data(k);
        let mut ub = vec![0.0_f64; 3 * d.n];
        eval_u(xi[0], xi[1], xi[2], k - 1, &mut ub);
        for i in 0..d.n {
            let mut v = [0.0_f64; 3];
            for b in 0..d.n {
                let c = d.ti[i * d.n + b];
                for comp in 0..3 {
                    v[comp] += c * ub[b * 3 + comp];
                }
            }
            values[i * 3] = v[0];
            values[i * 3 + 1] = v[1];
            values[i * 3 + 2] = v[2];
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order;
        let n = k * (k + 2) * (k + 3) / 2;

        if k == 1 {
            // MFEM 4.10 Nedelec1TetFiniteElement::CalcCurlShape
            curl_vals[0] = 0.0; curl_vals[1] = -2.0; curl_vals[2] = 2.0;
            curl_vals[3] = 2.0; curl_vals[4] = 0.0; curl_vals[5] = -2.0;
            curl_vals[6] = -2.0; curl_vals[7] = 2.0; curl_vals[8] = 0.0;
            curl_vals[9] = 0.0; curl_vals[10] = 0.0; curl_vals[11] = 2.0;
            curl_vals[12] = 0.0; curl_vals[13] = -2.0; curl_vals[14] = 0.0;
            curl_vals[15] = 2.0; curl_vals[16] = 0.0; curl_vals[17] = 0.0;
            return;
        }

        let d = tet_data(k);
        let mut uc = vec![0.0_f64; 3 * n];
        eval_u_curl(xi[0], xi[1], xi[2], k - 1, &mut uc);
        for i in 0..n {
            let mut c = [0.0_f64; 3];
            for b in 0..n {
                let f = d.ti[i * n + b];
                for comp in 0..3 {
                    c[comp] += f * uc[b * 3 + comp];
                }
            }
            curl_vals[i * 3] = c[0];
            curl_vals[i * 3 + 1] = c[1];
            curl_vals[i * 3 + 2] = c[2];
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }

    /// DOF sites (MFEM `FE::Nodes`).
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        dof_table(self.order).into_iter().map(|(p, _)| p.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_ndk_coeff() {
        for k in 1..=4 {
            let d = tet_data(k);
            let mut s = 0.0;
            for i in 0..d.n {
                s += d.ti[i * d.n + i].abs();
            }
            assert!(s > 0.1, "k={k}");
        }
    }

    #[test]
    fn tet_ndk_finite() {
        for k in 1..=3 {
            let e = TetNDk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            let mut c = vec![0.0; n * 3];
            for p in &[(0.25, 0.25, 0.25), (0.1, 0.2, 0.15), (0.5, 0.1, 0.1)] {
                e.eval_basis_vec(&[p.0, p.1, p.2], &mut v);
                e.eval_curl(&[p.0, p.1, p.2], &mut c);
                for val in v.iter().chain(c.iter()) {
                    assert!(val.is_finite(), "k={k}");
                }
            }
        }
    }

    /// Nodal property: `σ_j(Φ_i) = δ_ij` with the element's own
    /// `(dof_coords, dof_tangents)` table — the defining property of the
    /// point-value DOF semantics.
    #[test]
    fn nodal_basis_is_delta() {
        for k in 2..=4usize {
            let elem = TetNDk::new(k);
            let n = elem.n_dofs();
            let coords = elem.dof_coords();
            let tangents = elem.dof_tangents();
            let mut vals = vec![0.0; n * 3];
            for j in 0..n {
                elem.eval_basis_vec(&coords[j], &mut vals);
                for i in 0..n {
                    let s = vals[i * 3] * tangents[j][0]
                        + vals[i * 3 + 1] * tangents[j][1]
                        + vals[i * 3 + 2] * tangents[j][2];
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (s - expect).abs() < 1e-10,
                        "k={k}: DOF_{j}(Phi_{i}) = {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn dof_counts() {
        for k in 1..=5usize {
            let e = TetNDk::new(k);
            assert_eq!(e.n_dofs(), k * (k + 2) * (k + 3) / 2);
            assert_eq!(e.dof_coords().len(), e.n_dofs());
            assert_eq!(e.dof_tangents().len(), e.n_dofs());
        }
    }
}
