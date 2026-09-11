//! Arbitrary-order Nedelec-I element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! 1:1 port of MFEM `ND_TriangleElement(p)` (`fem/fe/fe_nd.cpp`, MFEM 4.10).
//!
//! # Space
//! `N_p = P_{p-1}² ⊕ x^⊥ P̃_{p-1}` (dim = `p(p+2)`): `3p` edge DOFs (p per
//! edge) plus `p(p-1)` interior (bubble) DOFs.
//!
//! # Construction (MFEM `ND_TriangleElement` verbatim)
//!
//! 1. **Raw basis `u`** (`p(p+2)` vector functions built from the hierarchical
//!    1-D basis `shape_x/shape_y/shape_l = Poly_1D::CalcChebyshev(p−1, ·)`,
//!    i.e. `T_j(2x−1)`, `T_j(2y−1)`, `T_j(2(1−x−y)−1)`):
//!    * `s·e_x` and `s·e_y` for `s = shape_x(i)·shape_y(j)·shape_l(p−1−i−j)`,
//!      `i+j ≤ p−1`;
//!    * `s·(y−c, −(x−c))` for `s = shape_x(p−1−j)·shape_y(j)`, `c = 1/3`.
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
//! reference tangent `t̂_i = tk + 2*dof2tk[i]`:
//!
//! | DOFs                | Points `x_i`                    | Tangent `t̂_i` |
//! |---------------------|---------------------------------|---------------|
//! | edge (0,1) `0..p`   | `(eop[i], 0)`                   | `(1, 0)`  |
//! | edge (1,2) `p..2p`  | `(eop[p−1−i], eop[i])`          | `(−1, 1)` |
//! | edge (2,0) `2p..3p` | `(0, eop[p−1−i])`               | `(0, −1)` |
//! | interior `3p..`     | barycentric GL points           | `(1, 0)`, `(0, 1)` |
//!
//! with `eop = OpenPoints(p−1)` the p-point Gauss-Legendre rule on `[0,1]` and
//! the interior points the barycentric combinations of `iop = OpenPoints(p−2)`
//! (`(iop[i], iop[j], iop[p−2−i−j])/w`, `w = Σ iop`) — exactly MFEM's
//! `Nodes.IntPoint` construction.  The interior DOFs are element-owned (a
//! triangle has no shared 2-D face), so their tangent pair needs no
//! orientation bookkeeping.
//!
//! The Gauss-Legendre point set is symmetric about `1/2`, so an edge reversal
//! maps the edge DOFs to a **signed anti-diagonal permutation**
//! (`σ^rev_m = −σ_{p−1−m}`) — MFEM's own `SegDofOrd` encoding, which is what
//! makes `HCurlSpace` cross-element edge pairing conforming.  (The pre-D38
//! integral moments `∫Φ·t̂ t^m dt` are not reflection invariant: reversal mixes
//! them binomially, which scalar signs cannot express.)
//!
//! `TriND2` is the explicit order-2 specialization of the same functionals;
//! the two agree to round-off (same functionals ⟹ same dual basis).

use crate::quadrature::{gauss_legendre_01, tri_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};
use std::sync::OnceLock;

/// Chebyshev basis `T_0..T_p` on `[0,1]` (MFEM `Poly_1D::CalcChebyshev`).
fn chebyshev(p: usize, x: f64) -> Vec<f64> {
    let mut u = vec![0.0_f64; p + 1];
    u[0] = 1.0;
    if p == 0 {
        return u;
    }
    let z = 2.0 * x - 1.0;
    u[1] = z;
    for n in 1..p {
        u[n + 1] = 2.0 * z * u[n] - u[n - 1];
    }
    u
}

/// Chebyshev basis and its derivative w.r.t. `x` (MFEM
/// `Poly_1D::CalcChebyshev(p, x, u, d)`).
pub(crate) fn chebyshev_d(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0_f64; p + 1];
    let mut d = vec![0.0_f64; p + 1];
    u[0] = 1.0;
    if p == 0 {
        return (u, d);
    }
    let z = 2.0 * x - 1.0;
    u[1] = z;
    d[1] = 2.0;
    for n in 1..p {
        u[n + 1] = 2.0 * z * u[n] - u[n - 1];
        d[n + 1] = (n + 1) as f64 * (z * d[n] / n as f64 + 2.0 * u[n]);
    }
    (u, d)
}

/// Dense `n × n` Gauss-Jordan inverse (row-major, partial pivoting); the shared
/// linear-algebra step of the tri/tet Nédélec `Ti.Factor(T)`.
pub(crate) fn invert_dense(n: usize, a: &[f64], tag: &str) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a[i * n + j];
        }
        m[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col * 2 * n + col].abs();
        for r in col + 1..n {
            let v = m[r * 2 * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        assert!(
            best > 1e-13,
            "{tag}: singular Vandermonde (column {col}, pivot {best:e})"
        );
        if piv != col {
            for j in 0..2 * n {
                m.swap(col * 2 * n + j, piv * 2 * n + j);
            }
        }
        let p = m[col * 2 * n + col];
        for j in 0..2 * n {
            m[col * 2 * n + j] /= p;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * 2 * n + col];
            if f != 0.0 {
                for j in 0..2 * n {
                    m[r * 2 * n + j] -= f * m[col * 2 * n + j];
                }
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = m[i * 2 * n + n + j];
        }
    }
    inv
}

/// Bubble centre used by MFEM's raw basis (`ND_TriangleElement::c`).
const C_BUBBLE: f64 = 1.0 / 3.0;

struct TriNDkData {
    /// Row-major `n × n`: `Φ_i = Σ_b ti[i][b]·u_b` (`ti = A⁻¹`, MFEM `Ti`).
    ti: Vec<f64>,
    n: usize,
}

/// `(point, tangent)` of every local DOF, in MFEM `Nodes.IntPoint`/`dof2tk`
/// slot order.  Shared by the Vandermonde builder and `dof_coords` /
/// `dof_tangents`, so the element cannot drift from its own DOF layout.
pub(crate) fn dof_table(k: usize) -> Vec<([f64; 2], [f64; 2])> {
    let mut out = Vec::with_capacity(k * (k + 2));
    let eop = gauss_legendre_01(k).0; // MFEM poly1d.OpenPoints(k-1): k nodes
    let pm1 = k - 1;
    for i in 0..k {
        out.push(([eop[i], 0.0], [1.0, 0.0])); // (0,1)
    }
    for i in 0..k {
        out.push(([eop[pm1 - i], eop[i]], [-1.0, 1.0])); // (1,2)
    }
    for i in 0..k {
        out.push(([0.0, eop[pm1 - i]], [0.0, -1.0])); // (2,0)
    }
    if k >= 2 {
        let iop = gauss_legendre_01(k - 1).0; // MFEM OpenPoints(k-2): k-1 nodes
        let pm2 = k - 2;
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let w = iop[i] + iop[j] + iop[pm2 - i - j];
                let p = [iop[i] / w, iop[j] / w];
                out.push((p, [1.0, 0.0]));
                out.push((p, [0.0, 1.0]));
            }
        }
    }
    out
}

/// Evaluate the raw `u` basis (MFEM `ND_TriangleElement::CalcVShape`) into the
/// flat array `ub[b*2], ub[b*2+1]`.
fn eval_u(x: f64, y: f64, pm1: usize, ub: &mut [f64]) {
    let (tx, _) = chebyshev_d(pm1, x);
    let (ty, _) = chebyshev_d(pm1, y);
    let (tl, _) = chebyshev_d(pm1, 1.0 - x - y);
    let mut b = 0usize;
    for j in 0..=pm1 {
        for i in 0..=(pm1 - j) {
            let s = tx[i] * ty[j] * tl[pm1 - i - j];
            ub[b * 2] = s;
            ub[b * 2 + 1] = 0.0;
            ub[b * 2 + 2] = 0.0;
            ub[b * 2 + 3] = s;
            b += 2;
        }
    }
    for j in 0..=pm1 {
        let s = tx[pm1 - j] * ty[j];
        ub[b * 2] = s * (y - C_BUBBLE);
        ub[b * 2 + 1] = -s * (x - C_BUBBLE);
        b += 1;
    }
    debug_assert_eq!(b * 2, ub.len());
}

/// Evaluate the raw `u` basis curls (MFEM
/// `ND_TriangleElement::CalcCurlShape`).
fn eval_u_curl(x: f64, y: f64, pm1: usize, uc: &mut [f64]) {
    let (tx, dx) = chebyshev_d(pm1, x);
    let (ty, dy) = chebyshev_d(pm1, y);
    let (tl, dl) = chebyshev_d(pm1, 1.0 - x - y);
    let mut b = 0usize;
    for j in 0..=pm1 {
        for i in 0..=(pm1 - j) {
            let l = pm1 - i - j;
            let ddx = (dx[i] * tl[l] - tx[i] * dl[l]) * ty[j];
            let ddy = (dy[j] * tl[l] - ty[j] * dl[l]) * tx[i];
            uc[b] = -ddy;
            uc[b + 1] = ddx;
            b += 2;
        }
    }
    for j in 0..=pm1 {
        let i = pm1 - j;
        uc[b] = -((dx[i] * (x - C_BUBBLE) + tx[i]) * ty[j]
            + (dy[j] * (y - C_BUBBLE) + ty[j]) * tx[i]);
        b += 1;
    }
    debug_assert_eq!(b, uc.len());
}

fn tri_data(k: usize) -> &'static TriNDkData {
    static CACHE: [OnceLock<TriNDkData>; 9] = [
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
    CACHE[k - 1].get_or_init(|| build_tri_data(k))
}

fn build_tri_data(k: usize) -> TriNDkData {
    let n = k * (k + 2);
    let pm1 = k - 1;
    let table = dof_table(k);
    let mut a = vec![0.0_f64; n * n]; // A[b][m] = σ_m(u_b)
    let mut ub = vec![0.0_f64; 2 * n];
    for (m, (x, t)) in table.iter().enumerate() {
        eval_u(x[0], x[1], pm1, &mut ub);
        for b in 0..n {
            a[b * n + m] = ub[b * 2] * t[0] + ub[b * 2 + 1] * t[1];
        }
    }
    TriNDkData {
        ti: invert_dense(n, &a, "TriNDk"),
        n,
    }
}

/// Arbitrary-order Nedelec-I element on the reference triangle.
///
/// DOF layout = MFEM `ND_TriangleElement(p)` (`Nodes` + `dof2tk`).
pub struct TriNDk {
    order: usize,
}

impl TriNDk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "TriNDk requires order ≥ 1");
        TriNDk { order: p }
    }

    /// Reference tangents `t̂_i` of every local DOF (MFEM `tk`/`dof2tk`).
    pub fn dof_tangents(&self) -> Vec<[f64; 2]> {
        if self.order == 1 {
            return vec![[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]];
        }
        dof_table(self.order).into_iter().map(|(_, t)| t).collect()
    }
}

impl VectorReferenceElement for TriNDk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.order * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;

        // Special case k=1: Whitney 1-forms (matches TriND1)
        if k == 1 {
            let x = xi[0];
            let y = xi[1];
            // Φ₀ = w_{01} = (1−η, ξ)
            values[0] = 1.0 - y;
            values[1] = x;
            // Φ₁ = w_{12} = (−η, ξ)
            values[2] = -y;
            values[3] = x;
            // Φ₂ = w_{02} = (η, 1−ξ)
            values[4] = y;
            values[5] = 1.0 - x;
            return;
        }

        let d = tri_data(k);
        let (x, y) = (xi[0], xi[1]);
        let mut ub = vec![0.0_f64; 2 * d.n];
        eval_u(x, y, k - 1, &mut ub);
        for i in 0..d.n {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for b in 0..d.n {
                let c = d.ti[i * d.n + b];
                vx += c * ub[b * 2];
                vy += c * ub[b * 2 + 1];
            }
            values[i * 2] = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order;

        // Special case k=1: constant curl [2, 2, −2]
        if k == 1 {
            curl_vals[0] = 2.0;
            curl_vals[1] = 2.0;
            curl_vals[2] = -2.0;
            return;
        }

        let d = tri_data(k);
        let mut uc = vec![0.0_f64; d.n];
        eval_u_curl(xi[0], xi[1], k - 1, &mut uc);
        for i in 0..d.n {
            let mut s = 0.0;
            for b in 0..d.n {
                s += d.ti[i * d.n + b] * uc[b];
            }
            curl_vals[i] = s;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    /// DOF sites (MFEM `FE::Nodes`): Gauss-Legendre points along every edge
    /// (ascending along each edge's named direction) plus the barycentric GL
    /// interior point values.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;

        // Special case k=1: edge midpoints (matches TriND1)
        if k == 1 {
            return vec![vec![0.5, 0.0], vec![0.5, 0.5], vec![0.0, 0.5]];
        }

        dof_table(k).into_iter().map(|(p, _)| p.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd1_curl_constant() {
        let elem = TriNDk::new(1);
        let mut curl = vec![0.0; 3];
        let expected = [2.0, 2.0, -2.0];
        let qr = elem.quadrature(3);
        for pt in &qr.points {
            elem.eval_curl(pt, &mut curl);
            for (i, &c) in curl.iter().enumerate() {
                assert!(
                    (c - expected[i]).abs() < 1e-13,
                    "curl[{i}] = {c}, expected {}",
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn nd1_nodal_basis() {
        let elem = TriNDk::new(1);
        let tangents: [[f64; 2]; 3] = [
            [1.0, 0.0],
            [-1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()],
            [0.0, 1.0],
        ];
        let edge_len = [1.0_f64, 2f64.sqrt(), 1.0_f64];

        let mut vals = vec![0.0; 6];
        for (j, (mid, (t, l))) in elem
            .dof_coords()
            .iter()
            .zip(tangents.iter().zip(edge_len.iter()))
            .enumerate()
        {
            elem.eval_basis_vec(mid, &mut vals);
            for i in 0..3 {
                let dof = (vals[i * 2] * t[0] + vals[i * 2 + 1] * t[1]) * l;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// Nodal property for all orders: `σ_j(Φ_i) = δ_ij` with the element's own
    /// `(dof_coords, dof_tangents)` table.
    #[test]
    fn nodal_basis_is_delta() {
        for k in 2..=5usize {
            let elem = TriNDk::new(k);
            let n = elem.n_dofs();
            let coords = elem.dof_coords();
            let tangents = elem.dof_tangents();
            let mut vals = vec![0.0; n * 2];
            for j in 0..n {
                elem.eval_basis_vec(&coords[j], &mut vals);
                for i in 0..n {
                    let s = vals[i * 2] * tangents[j][0] + vals[i * 2 + 1] * tangents[j][1];
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (s - expect).abs() < 1e-10,
                        "k={k}: DOF_{j}(Phi_{i}) = {s}"
                    );
                }
            }
        }
    }

    /// DOF count and DOF-site count agree with MFEM `p(p+2)`.
    #[test]
    fn dof_counts() {
        for k in 1..=5usize {
            let e = TriNDk::new(k);
            assert_eq!(e.n_dofs(), k * (k + 2));
            assert_eq!(e.dof_coords().len(), k * (k + 2));
            assert_eq!(e.dof_tangents().len(), k * (k + 2));
        }
    }
}
