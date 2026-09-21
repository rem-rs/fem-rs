//! Arbitrary-order RT_k on the reference triangle — MFEM `RT_TriangleElement`
//! construction, verbatim (D529).
//!
//! MFEM (`fem/fe/fe_rt.cpp`) builds the basis from the **point-value duality**
//! `D_k(v) = v(node_k)·nk_k` over the nodal sample table
//! ([`mfem_tri_nodal_dofs`]): with `u_o` running over the `[P_k]²` Chebyshev
//! products `T_i(2x−1)·T_j(2y−1)·T_{k−i−j}(2(1−x−y)−1)` (two components each)
//! plus the bubble components `(x−c)·s, (y−c)·s` with `s = T_i·T_{k−i}` and
//! `c = 1/3` (MFEM evaluates `Poly_1D::CalcBasis` = `CalcChebyshev`), the flux
//! matrix `T(o,k) = u_o(node_k)·nk_k` is factored once
//! (`Ti.Factor(T)`) and every basis function is `φ_k = Σ_o (T⁻¹)_{k,o}·u_o`
//! (`CalcVShape`: `Ti.Mult(u, shape)`; `CalcDivShape` likewise on the
//! divergences of `u_o`).  The dof values are therefore exactly MFEM's —
//! `W[i][j] = φ_j(node_i)·nk_i = I` against the sample table.
//!
//! fem-rs used to build the *moment* duals `∫(Φ·n)t^p dt` instead: the same
//! space, but a different dual basis, so any path reading tri RTk dof values
//! disagreed with MFEM while the `P = B·W⁻¹` machinery hid the difference
//! (D529, round 55: `TriRTk(1)` gave `max|W−I| = 3.5e+0`).

use crate::reference::VectorReferenceElement;
use crate::raviart_thomas::tri_rt1::mfem_tri_nodal_dofs;
use std::sync::OnceLock;

/// MFEM `RT_TriangleElement::nk` — the reference flux directions indexed by
/// `dof2nk`: bottom (v0,v1) → (0,−1), hypotenuse (v1,v2) → (1,1) (unnormalised
/// as in MFEM), left (v2,v0) → (−1,0).
const NK: [[f64; 2]; 3] = [[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]];

/// MFEM `RT_TriangleElement::c` — the bubble shift.
const C: f64 = 1.0 / 3.0;

struct TriRTkData {
    /// `Ti = T⁻¹` in row-major `n×n` layout, `T(o,k) = u_o(node_k)·nk_k`
    /// (MFEM `DenseMatrix Ti` of `RT_TriangleElement`).
    ti: Vec<f64>,
}

/// MFEM `Poly_1D::CalcBasis` = `CalcChebyshev` (`fe_base.cpp:2376`): the
/// Chebyshev polynomials `T_i(2x−1)` through degree `p` with `x`-derivatives,
/// via `T_{i+1} = 2z·T_i − T_{i−1}` (`z = 2x−1`) and
/// `d_{i+1} = (i+1)·(z·d_i/i + 2·T_i)`.
///
/// The basis is load-bearing, not cosmetic: the bubble span
/// `{T_i(x)·T_{k−i}(y)}` differs from the Legendre-product span (it carries
/// mixed linear terms), and only with it are the MFEM nodal functionals
/// independent on the RT space — a Legendre-product `T` is exactly singular
/// for `k ≥ 2`.
fn cheb_all(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut v = Vec::with_capacity(p + 1);
    let mut d = Vec::with_capacity(p + 1);
    v.push(1.0);
    d.push(0.0);
    if p >= 1 {
        v.push(2.0 * x - 1.0);
        d.push(2.0);
    }
    for i in 1..p {
        let z = 2.0 * x - 1.0;
        v.push(2.0 * z * v[i] - v[i - 1]);
        d.push((i + 1) as f64 * (z * d[i] / i as f64 + 2.0 * v[i]));
    }
    (v, d)
}

/// Solve `T·X = I` with a partial-pivot LU (MFEM `DenseMatrix::Factor` +
/// `CalcInverse` semantics), returning `X` row-major.
fn invert(n: usize, t: &[f64]) -> Vec<f64> {
    let mut lu = t.to_vec();
    let mut x = vec![0.0_f64; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut pr = c;
        let mut pv = lu[c * n + c].abs();
        for r in c + 1..n {
            let v = lu[r * n + c].abs();
            if v > pv {
                pv = v;
                pr = r;
            }
        }
        assert!(pv > 1e-300, "TriRTk: singular T matrix at column {c}");
        if pr != c {
            for j in 0..n {
                lu.swap(c * n + j, pr * n + j);
                x.swap(c * n + j, pr * n + j);
            }
        }
        let inv = 1.0 / lu[c * n + c];
        for r in c + 1..n {
            let f = lu[r * n + c] * inv;
            lu[r * n + c] = f;
            if f != 0.0 {
                for j in (c + 1)..n {
                    lu[r * n + j] -= f * lu[c * n + j];
                }
                for j in 0..n {
                    x[r * n + j] -= f * x[c * n + j];
                }
            }
        }
    }
    for c in 0..n {
        for r in (0..n).rev() {
            let mut s = x[r * n + c];
            for j in (r + 1)..n {
                s -= lu[r * n + j] * x[j * n + c];
            }
            x[r * n + c] = s / lu[r * n + r];
        }
    }
    x
}

/// `u_o(ip)` in MFEM's row layout: for `j = 0..=k`, `i + j ≤ k` the pair
/// `(s, 0), (0, s)` with `s = T_i(x)·T_j(y)·T_{k−i−j}(1−x−y)` (Chebyshev,
/// `T_m(w) = T_m(2w−1)`), then for `i = 0..=k` the bubble
/// `((x−c)s, (y−c)s)` with `s = T_i(x)·T_{k−i}(y)`.
/// Interleaved `(o, component)` pairs, `2n` entries.
fn eval_u(k: usize, x: f64, y: f64) -> Vec<f64> {
    let (lx, _) = cheb_all(k, x);
    let (ly, _) = cheb_all(k, y);
    let (ll, _) = cheb_all(k, 1.0 - x - y);
    let mut u = Vec::with_capacity(2 * (k + 1) * (k + 3));
    for j in 0..=k {
        for i in 0..=(k - j) {
            let s = lx[i] * ly[j] * ll[k - i - j];
            u.push(s);
            u.push(0.0);
            u.push(0.0);
            u.push(s);
        }
    }
    for i in 0..=k {
        let s = lx[i] * ly[k - i];
        u.push((x - C) * s);
        u.push((y - C) * s);
    }
    u
}

/// `∇·u_o(ip)` in the same `o` layout (MFEM `CalcDivShape`): the `[P_k]²`
/// blocks contribute `∂_x s` / `∂_y s` (with `∂_x(1−x−y) = ∂_y(1−x−y) = −1`),
/// the bubble contributes `2s + (x−c)·∂_x s + (y−c)·∂_y s`.
fn eval_div_u(k: usize, x: f64, y: f64) -> Vec<f64> {
    let (lx, dx) = cheb_all(k, x);
    let (ly, dy) = cheb_all(k, y);
    let (ll, dl) = cheb_all(k, 1.0 - x - y);
    let mut divu = Vec::with_capacity((k + 1) * (k + 3));
    for j in 0..=k {
        for i in 0..=(k - j) {
            let kk = k - i - j;
            divu.push((dx[i] * ll[kk] - lx[i] * dl[kk]) * ly[j]);
            divu.push((dy[j] * ll[kk] - ly[j] * dl[kk]) * lx[i]);
        }
    }
    for i in 0..=k {
        let j = k - i;
        divu.push(
            (lx[i] + (x - C) * dx[i]) * ly[j] + (ly[j] + (y - C) * dy[j]) * lx[i],
        );
    }
    divu
}

fn tri_data(k: usize) -> &'static TriRTkData {
    static CACHE: [OnceLock<TriRTkData>; 9] = [
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
    CACHE[k].get_or_init(|| {
        let n = (k + 1) * (k + 3);
        let (nodes, nks) = mfem_tri_nodal_dofs(k);
        debug_assert_eq!(nodes.len(), n, "TriRTk({k}): sample table size");
        // T(o,k) = u_o(node_k)·nk_k, rows `o` = basis functions, columns
        // `k` = dofs — verbatim MFEM `RT_TriangleElement::RT_TriangleElement`.
        let mut t = vec![0.0_f64; n * n];
        for (kk, (ip, nk)) in nodes.iter().zip(nks.iter()).enumerate() {
            let (lx, _) = cheb_all(k, ip[0]);
            let (ly, _) = cheb_all(k, ip[1]);
            let (ll, _) = cheb_all(k, 1.0 - ip[0] - ip[1]);
            let (nx, ny) = (nk[0], nk[1]);
            let mut o = 0usize;
            for j in 0..=k {
                for i in 0..=(k - j) {
                    let s = lx[i] * ly[j] * ll[k - i - j];
                    t[o * n + kk] = s * nx;
                    o += 1;
                    t[o * n + kk] = s * ny;
                    o += 1;
                }
            }
            for i in 0..=k {
                let s = lx[i] * ly[k - i];
                t[o * n + kk] = s * ((ip[0] - C) * nx + (ip[1] - C) * ny);
                o += 1;
            }
            debug_assert_eq!(o, n, "TriRTk({k}): u row count");
        }
        TriRTkData { ti: invert(n, &t) }
    })
}

/// Arbitrary-order Raviart-Thomas RT_k H(div) element on the reference triangle.
pub struct TriRTk {
    order: usize,
}

impl TriRTk {
    pub fn new(p: usize) -> Self {
        TriRTk { order: p }
    }
}

impl VectorReferenceElement for TriRTk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 3)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;
        let x = xi[0];
        let y = xi[1];

        // Special case k=0: classical Piola form, in MFEM's edge-block order
        // (bottom (v0,v1), hypotenuse (v1,v2), left (v2,v0)).
        if k == 0 {
            // Φ₀ = (ξ, η−1)  — bottoms edge (0,1), nk = (0,−1)
            values[0] = x;
            values[1] = y - 1.0;
            // Φ₁ = (ξ, η)  — hypotenuse (1,2), nk = (1,1)
            values[2] = x;
            values[3] = y;
            // Φ₂ = (ξ−1, η)  — left (2,0), nk = (−1,0)
            values[4] = x - 1.0;
            values[5] = y;
            return;
        }

        let d = tri_data(k);
        let n = self.n_dofs();
        let u = eval_u(k, x, y);
        for (i, row) in d.ti.chunks_exact(n).enumerate() {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for (o, &c) in row.iter().enumerate() {
                if c != 0.0 {
                    vx += c * u[o * 2];
                    vy += c * u[o * 2 + 1];
                }
            }
            values[i * 2] = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order;

        // Special case k=0: div = 2 for all basis functions
        if k == 0 {
            for v in div_vals.iter_mut() {
                *v = 2.0;
            }
            return;
        }

        let d = tri_data(k);
        let n = self.n_dofs();
        let divu = eval_div_u(k, xi[0], xi[1]);
        for (i, row) in d.ti.chunks_exact(n).enumerate() {
            let mut s = 0.0;
            for (o, &c) in row.iter().enumerate() {
                if c != 0.0 {
                    s += c * divu[o];
                }
            }
            div_vals[i] = s;
        }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;

        // Special case k=0: edge midpoints (matches TriRT0), in MFEM's
        // edge-block order (bottom, hypotenuse, left).
        if k == 0 {
            return vec![
                vec![0.5, 0.0],
                vec![0.5, 0.5],
                vec![0.0, 0.5],
            ];
        }

        // MFEM `RT_TriangleElement` node table: edge blocks on the
        // Gauss-Legendre open points, then the interior component samples
        // (two dofs per interior point, `dof2nk` 0 then 2).
        mfem_tri_nodal_dofs(k)
            .0
            .iter()
            .map(|p| vec![p[0], p[1]])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt0_div_constant() {
        let elem = TriRTk::new(0);
        let mut div = vec![0.0; 3];
        for pt in &elem.quadrature(4).points {
            elem.eval_div(pt, &mut div);
            for (i, &d) in div.iter().enumerate() {
                assert!((d - 2.0).abs() < 1e-13, "div[{i}] = {d}");
            }
        }
    }

    #[test]
    fn rt0_nodal_basis() {
        let elem = TriRTk::new(0);
        // MFEM edge-block order: bottom (0,1), hypotenuse (1,2), left (2,0).
        let faces: [([f64; 2], f64); 3] = [
            ([0.0, -1.0], 1.0),
            ([1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()], 2f64.sqrt()),
            ([-1.0, 0.0], 1.0),
        ];

        let mids = elem.dof_coords();
        let mut vals = vec![0.0; 6];
        for (j, (normal, len)) in faces.iter().enumerate() {
            elem.eval_basis_vec(&mids[j], &mut vals);
            for i in 0..3 {
                let dof = (vals[i * 2] * normal[0] + vals[i * 2 + 1] * normal[1]) * len;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// D44 regression: `eval_basis_vec` must evaluate **all** `k ≥ 1` orders
    /// without panicking or producing non-finite values.
    #[test]
    fn eval_all_orders_is_in_bounds() {
        for k in 0..=8usize {
            let e = TriRTk::new(k);
            let n = e.n_dofs();
            let mut vals = vec![0.0f64; n * 2];
            let mut div = vec![0.0f64; n];
            for pt in &e.quadrature(4).points {
                e.eval_basis_vec(pt, &mut vals);
                e.eval_div(pt, &mut div);
                for &v in vals.iter().chain(div.iter()) {
                    assert!(v.is_finite(), "k={k}: non-finite basis value at {pt:?}");
                }
            }
        }
    }


    /// D529 regression: the basis must be **point-dual** to MFEM's nodal
    /// sample table, `W[i][j] = φ_j(node_i)·nk_i = δ_ij` — the dof values are
    /// MFEM's.  Round ≤55 built the moment duals `∫(Φ·n)t^p dt` instead, and
    /// `TriRTk(1)`/`TriRTk(2)` scored `max|W−I| = 3.5e+0`/`1.8e+1` here.
    #[test]
    fn basis_dual_to_point_functionals() {
        for k in 0..=4usize {
            let e = TriRTk::new(k);
            let n = e.n_dofs();
            let (nodes, nks) = mfem_tri_nodal_dofs(k);
            assert_eq!(nodes.len(), n, "k={k}: sample table size");
            let mut max_off = 0.0_f64;
            let mut phi = vec![0.0f64; n * 2];
            for (i, (pt, nk)) in nodes.iter().zip(nks.iter()).enumerate() {
                e.eval_basis_vec(pt, &mut phi);
                for j in 0..n {
                    let d = phi[j * 2] * nk[0] + phi[j * 2 + 1] * nk[1];
                    let want = if i == j { 1.0 } else { 0.0 };
                    max_off = max_off.max((d - want).abs());
                }
            }
            assert!(
                max_off <= 1e-12,
                "k={k}: max |W - I| = {max_off:.3e}, expected point-dual (≤1e-12)"
            );
        }
    }

    /// D529 consistency pin: the generic `TriRTk` construction reproduces the
    /// independently built (and MFEM-verified, d468) low-order elements
    /// slot-for-slot.
    #[test]
    fn matches_low_order_mfem_elements() {
        use crate::raviart_thomas::{TriRT1, TriRT2};

        for (k, lo) in [(1usize, &TriRT1 as &dyn VectorReferenceElement), (2, &TriRT2)] {
            let e = TriRTk::new(k);
            assert_eq!(e.n_dofs(), lo.n_dofs());
            let n = e.n_dofs();
            let mut a = vec![0.0f64; n * 2];
            let mut b = vec![0.0f64; n * 2];
            let mut max_diff = 0.0_f64;
            for pt in &e.quadrature(8).points {
                e.eval_basis_vec(pt, &mut a);
                lo.eval_basis_vec(pt, &mut b);
                for (x, y) in a.iter().zip(b.iter()) {
                    max_diff = max_diff.max((x - y).abs());
                }
            }
            assert!(
                max_diff <= 1e-11,
                "k={k}: TriRTk vs low-order element max diff {max_diff:.3e}"
            );
        }
    }
}
