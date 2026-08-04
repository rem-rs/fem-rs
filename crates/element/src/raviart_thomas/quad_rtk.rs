//! Arbitrary-order Raviart-Thomas element on the reference quadrilateral
//! `[0,1]²` — a 1:1 port of MFEM's `RT_QuadrilateralElement` (fem/fe/fe_rt.cpp),
//! the element behind `RT_FECollection(k, 2)` for every order.
//!
//! # Tensor-product structure
//!
//! RT_k = Q_{k+1,k} × Q_{k,k+1}, `dof = 2(k+1)(k+2)`:
//!
//! - **x components**: `s·c_i(x)·o_j(y)`,  i = 0..k+1 (closed), j = 0..k (open)
//! - **y components**: `s·o_i(x)·c_j(y)`,  i = 0..k (open),  j = 0..k+1 (closed)
//!
//! with the closed 1-D basis `c` = Gauss-Lobatto-Legendre nodes on `[0,1]`
//! (degree k+1, k+2 points) and the open basis `o` = Gauss-Legendre nodes
//! (degree k, k+1 points), evaluated with the same stable-centre barycentric
//! formula as MFEM `Poly_1D::Basis::Eval` (Barycentric).
//!
//! Signs and DOF permutation come from MFEM's `dof_map` (built by the
//! constructor loop in fe_rt.cpp), which encodes the normal-moment ordering
//! and orientation flips.
//!
//! # Verified degenerations
//! - k = 0: closed = linear GLL `{0,1}`, open = constant → the RT0 quad
//!   basis `(0,y-1)/(x,0)/(0,y)/(x-1,0)`.
//! - k = 1: identical to [`QuadRT1`](super::QuadRT1) (single-square mass
//!   matrices match to machine precision).

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};
use std::sync::OnceLock;

// ─── 1-D node generation (MFEM QuadratureFunctions1D, intrules.cpp) ────────

/// Gauss-Legendre nodes on `[0,1]` (MFEM `GaussLegendre`, bit-identical:
/// hard-coded for n ≤ 3, Newton with `cos(π(i−1/4)/(n+1/2))` start + the
/// round-off-safe `xi = ((1−z)+dz)/2` mapping otherwise).
// The final `pp` assignment is kept for MFEM bit-identity (the post-done
// iteration recomputes pp exactly as the C++ does, even though we only use
// the nodes).
#[allow(unused_assignments)]
fn gl_nodes(n: usize) -> Vec<f64> {
    match n {
        1 => vec![0.5],
        2 => vec![0.21132486540518711775, 0.78867513459481288225],
        3 => vec![0.11270166537925831148, 0.5, 0.88729833462074168852],
        _ => {
            let m = (n + 1) / 2;
            let mut pts = vec![0.0; n];
            for i in 1..=m {
                let mut z = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
                let mut pp = 0.0;
                let mut xi = 0.0;
                let mut done = false;
                loop {
                    // p1 = P_n(z), p2 = P_{n-1}(z)
                    let mut p2 = 1.0;
                    let mut p1 = z;
                    for j in 2..=n {
                        let p3 = p2;
                        p2 = p1;
                        p1 = ((2 * j - 1) as f64 * z * p2 - (j - 1) as f64 * p3) / j as f64;
                    }
                    pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
                    if done {
                        break;
                    }
                    let dz = p1 / pp;
                    if dz.abs() < 1e-16 {
                        done = true;
                        // map the new point (z-dz) to [0,1] without round-off:
                        xi = ((1.0 - z) + dz) / 2.0;
                    }
                    z -= dz;
                }
                pts[i - 1] = xi;
                pts[n - i] = 1.0 - xi;
            }
            pts
        }
    }
}

/// Gauss-Lobatto-Legendre nodes on `[0,1]` (MFEM `GaussLobatto`,
/// bit-identical: Newton on the roots of P'_{n-1} with the Chebyshev start
/// `sin(π(i/(n−1) − 1/2))` and the symmetric `1−z_i` partner).
// Same as gl_nodes: the trailing `pl` computation mirrors the C++ loop.
#[allow(unused_assignments)]
fn gll_nodes(n: usize) -> Vec<f64> {
    let mut pts = vec![0.0; n];
    if n == 1 {
        pts[0] = 0.5;
        return pts;
    }
    pts[0] = 0.0;
    pts[n - 1] = 1.0;
    for i in 1..=(n - 1) / 2 {
        let mut xi = (std::f64::consts::PI * (i as f64 / (n - 1) as f64 - 0.5)).sin();
        let mut zi = 0.0;
        let mut pl = 0.0;
        let mut done = false;
        for _ in 0..8 {
            // p_l = P_{n-1}(x_i)
            let mut plm1 = 1.0;
            pl = xi;
            for l in 1..n - 1 {
                let plp1 = ((2 * l + 1) as f64 * xi * pl - l as f64 * plm1) / (l + 1) as f64;
                plm1 = pl;
                pl = plp1;
            }
            if done {
                break;
            }
            let dx = (xi * pl - plm1) / (n as f64 * pl);
            if dx.abs() < 1e-16 {
                done = true;
                zi = ((1.0 + xi) - dx) / 2.0;
            }
            xi -= dx;
        }
        pts[i] = zi;
        pts[n - 1 - i] = 1.0 - zi;
    }
    pts
}

// ─── 1-D barycentric Lagrange (MFEM Poly_1D::Basis, Barycentric) ────────────

/// Stable-centre barycentric Lagrange values and derivatives at `y` on the
/// nodes `nodes` (`[0,1]`) — bit-identical to MFEM `Poly_1D::Basis::Eval`
/// (Barycentric), generalised from `QuadQk::mfem_bary_1d`.
fn bary_eval(nodes: &[f64], y: f64) -> (Vec<f64>, Vec<f64>) {
    let n = nodes.len();
    let p = n - 1;
    // Barycentric weights w_i = 1/∏_{j≠i}(x_i − x_j)
    let mut w = vec![1.0; n];
    for i in 0..n {
        for j in 0..n {
            if j != i {
                w[i] *= nodes[i] - nodes[j];
            }
        }
        w[i] = 1.0 / w[i];
    }
    // Stable centre k: lk = ∏ over the nodes on one side of y.
    let mut k = 0usize;
    let mut lk = 1.0;
    while k < p {
        if y >= (nodes[k] + nodes[k + 1]) / 2.0 {
            lk *= y - nodes[k];
            k += 1;
        } else {
            for i in k + 1..=p {
                lk *= y - nodes[i];
            }
            break;
        }
    }
    let l = lk * (y - nodes[k]);
    let mut sk = 0.0;
    let mut u = vec![0.0; n];
    for i in 0..k {
        let si = 1.0 / (y - nodes[i]);
        sk += si;
        u[i] = l * si * w[i];
    }
    u[k] = lk * w[k];
    for i in k + 1..=p {
        let si = 1.0 / (y - nodes[i]);
        sk += si;
        u[i] = l * si * w[i];
    }
    let lp = l * sk + lk;
    let mut d = vec![0.0; n];
    for i in 0..k {
        d[i] = (lp * w[i] - u[i]) / (y - nodes[i]);
    }
    d[k] = sk * u[k];
    for i in k + 1..=p {
        d[i] = (lp * w[i] - u[i]) / (y - nodes[i]);
    }
    (u, d)
}

// ─── dof_map (MFEM RT_QuadrilateralElement constructor) ─────────────────────

/// Build the DOF map: `dof_map[o]` = signed DOF index for the `o`-th
/// tensor-product slot (x components first, then y components), with the
/// orientation flips encoded as negative values (`−1−idx`).
fn build_dof_map(p: usize) -> Vec<i32> {
    let dof = 2 * (p + 1) * (p + 2);
    let dof2 = dof / 2;
    let mut dm = vec![0i32; dof];
    let mut o = 0i32;
    // edges
    for i in 0..=p {
        // (0,1) bottom, y = 0
        dm[dof2 + i + 0 * (p + 1)] = o;
        o += 1;
    }
    for i in 0..=p {
        // (1,2) right, x = 1
        dm[0 + (p + 1) + i * (p + 2)] = o;
        o += 1;
    }
    for i in 0..=p {
        // (2,3) top, y = 1
        dm[dof2 + (p - i) + (p + 1) * (p + 1)] = o;
        o += 1;
    }
    for i in 0..=p {
        // (3,0) left, x = 0
        dm[0 + 0 + (p - i) * (p + 2)] = o;
        o += 1;
    }
    // interior
    for j in 0..=p {
        for i in 1..=p {
            dm[i + j * (p + 2)] = o;
            o += 1;
        }
    }
    for j in 1..=p {
        for i in 0..=p {
            dm[dof2 + i + j * (p + 1)] = o;
            o += 1;
        }
    }
    // dof orientations
    // x-components
    for j in 0..=p {
        for i in 0..=p / 2 {
            let idx = i + j * (p + 2);
            dm[idx] = -1 - dm[idx];
        }
    }
    if p % 2 == 1 {
        for j in p / 2 + 1..=p {
            let idx = (p / 2 + 1) + j * (p + 2);
            dm[idx] = -1 - dm[idx];
        }
    }
    // y-components
    for j in 0..=p / 2 {
        for i in 0..=p {
            let idx = dof2 + i + j * (p + 1);
            dm[idx] = -1 - dm[idx];
        }
    }
    if p % 2 == 1 {
        for i in 0..=p / 2 {
            let idx = dof2 + i + (p / 2 + 1) * (p + 1);
            dm[idx] = -1 - dm[idx];
        }
    }
    dm
}

#[inline]
fn decode(v: i32) -> (usize, f64) {
    if v < 0 {
        ((-1 - v) as usize, -1.0)
    } else {
        (v as usize, 1.0)
    }
}

// ─── Element data ────────────────────────────────────────────────────────────

struct QuadRTkData {
    p: usize,
    dof: usize,
    dof_map: Vec<i32>,
    /// closed (GLL) nodes on [0,1], k+2 of them
    cp: Vec<f64>,
    /// open (GL) nodes on [0,1], k+1 of them
    op: Vec<f64>,
}

fn rt_data(k: usize) -> &'static QuadRTkData {
    static CACHE: [OnceLock<QuadRTkData>; 7] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[k].get_or_init(|| QuadRTkData {
        p: k,
        dof: 2 * (k + 1) * (k + 2),
        dof_map: build_dof_map(k),
        cp: gll_nodes(k + 2),
        op: gl_nodes(k + 1),
    })
}

pub struct QuadRTk {
    order: usize,
}
impl QuadRTk {
    pub fn new(p: usize) -> Self {
        assert!(p < 7, "QuadRTk: order {p} exceeds supported range");
        QuadRTk { order: p }
    }
}

impl VectorReferenceElement for QuadRTk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        2 * (self.order + 1) * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let d = rt_data(self.order);
        let p = d.p;
        let (cx, _) = bary_eval(&d.cp, xi[0]);
        let (cy, _) = bary_eval(&d.cp, xi[1]);
        let (ox, _) = bary_eval(&d.op, xi[0]);
        let (oy, _) = bary_eval(&d.op, xi[1]);
        values.fill(0.0);
        let mut o = 0;
        // x components: j = 0..p (open y), i = 0..p+1 (closed x)
        for j in 0..=p {
            for i in 0..=p + 1 {
                let (idx, s) = decode(d.dof_map[o]);
                o += 1;
                values[idx * 2] = s * cx[i] * oy[j];
            }
        }
        // y components: j = 0..p+1 (closed y), i = 0..p (open x)
        for j in 0..=p + 1 {
            for i in 0..=p {
                let (idx, s) = decode(d.dof_map[o]);
                o += 1;
                values[idx * 2 + 1] = s * ox[i] * cy[j];
            }
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let d = rt_data(self.order);
        let p = d.p;
        let (_cx, dcx) = bary_eval(&d.cp, xi[0]);
        let (cy, dcy) = bary_eval(&d.cp, xi[1]);
        let (ox, _) = bary_eval(&d.op, xi[0]);
        let (oy, _) = bary_eval(&d.op, xi[1]);
        let mut o = 0;
        // x components: d/dx [c_i(x)·o_j(y)] = dc_i(x)·o_j(y)
        for j in 0..=p {
            for i in 0..=p + 1 {
                let (idx, s) = decode(d.dof_map[o]);
                o += 1;
                div_vals[idx] = s * dcx[i] * oy[j];
            }
        }
        // y components: d/dy [o_i(x)·c_j(y)] = o_i(x)·dc_j(y)
        for j in 0..=p + 1 {
            for i in 0..=p {
                let (idx, s) = decode(d.dof_map[o]);
                o += 1;
                div_vals[idx] = s * ox[i] * dcy[j];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    /// MFEM `RT_QuadrilateralElement` node coordinates: `(cp[i], op[j])` /
    /// `(op[i], cp[j])` in tensor-product order, permuted by `dof_map`.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let d = rt_data(self.order);
        let p = d.p;
        let mut c = vec![vec![0.0; 2]; d.dof];
        let mut o = 0;
        for j in 0..=p {
            for i in 0..=p + 1 {
                let (idx, _) = decode(d.dof_map[o]);
                o += 1;
                c[idx] = vec![d.cp[i], d.op[j]];
            }
        }
        for j in 0..=p + 1 {
            for i in 0..=p {
                let (idx, _) = decode(d.dof_map[o]);
                o += 1;
                c[idx] = vec![d.op[i], d.cp[j]];
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_dofs() {
        assert_eq!(QuadRTk::new(0).n_dofs(), 4);
        assert_eq!(QuadRTk::new(1).n_dofs(), 12);
        assert_eq!(QuadRTk::new(2).n_dofs(), 24);
        assert_eq!(QuadRTk::new(3).n_dofs(), 40);
    }

    #[test]
    fn matches_quad_rt0() {
        // k = 0 must reproduce QuadRT0 exactly: (0,y-1)/(x,0)/(0,y)/(x-1,0).
        let k = QuadRTk::new(0);
        let r0 = crate::raviart_thomas::QuadRT0;
        let mut vk = vec![0.0; 8];
        let mut v0 = vec![0.0; 8];
        let mut dk = vec![0.0; 4];
        let mut d0 = vec![0.0; 4];
        for xi in k.quadrature(5).points {
            k.eval_basis_vec(&xi, &mut vk);
            k.eval_div(&xi, &mut dk);
            r0.eval_basis_vec(&xi, &mut v0);
            r0.eval_div(&xi, &mut d0);
            for i in 0..8 {
                assert!((vk[i] - v0[i]).abs() < 1e-14, "phi[{i}] k0={} rt0={}", vk[i], v0[i]);
            }
            for i in 0..4 {
                assert!((dk[i] - d0[i]).abs() < 1e-14, "div[{i}]");
            }
        }
    }

    #[test]
    fn matches_quad_rt1() {
        // k = 1 must reproduce QuadRT1 (MFEM RT_QuadrilateralElement(1)).
        let k = QuadRTk::new(1);
        let r1 = crate::raviart_thomas::QuadRT1;
        let mut vk = vec![0.0; 24];
        let mut v1 = vec![0.0; 24];
        let mut dk = vec![0.0; 12];
        let mut d1 = vec![0.0; 12];
        for xi in k.quadrature(5).points {
            k.eval_basis_vec(&xi, &mut vk);
            k.eval_div(&xi, &mut dk);
            r1.eval_basis_vec(&xi, &mut v1);
            r1.eval_div(&xi, &mut d1);
            for i in 0..24 {
                assert!(
                    (vk[i] - v1[i]).abs() < 1e-14,
                    "phi[{i}] k1={} rt1={}",
                    vk[i],
                    v1[i]
                );
            }
            for i in 0..12 {
                assert!((dk[i] - d1[i]).abs() < 1e-14, "div[{i}]");
            }
        }
    }

    #[test]
    fn finite() {
        for k in 0..=4 {
            let e = QuadRTk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 2];
            let mut d = vec![0.0; n];
            for p in &[(0.2, 0.3), (0.7, 0.2), (0.5, 0.5)] {
                e.eval_basis_vec(&[p.0, p.1], &mut v);
                e.eval_div(&[p.0, p.1], &mut d);
                for &val in v.iter().chain(d.iter()) {
                    assert!(val.is_finite(), "k={k} at ({},{})", p.0, p.1);
                }
            }
        }
    }

    #[test]
    fn coeff_non_singular() {
        // The basis must be unisolvent: no zero columns in the nodal matrix.
        for k in 1..=4 {
            let e = QuadRTk::new(k);
            let coords = e.dof_coords();
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 2];
            let mut col_max = vec![0.0_f64; n];
            for cd in &coords {
                e.eval_basis_vec(cd, &mut v);
                for i in 0..n {
                    let m = v[i * 2].abs().max(v[i * 2 + 1].abs());
                    col_max[i] = col_max[i].max(m);
                }
            }
            for i in 0..n {
                assert!(col_max[i] > 1e-6, "k={k}: dof {i} has no nodal value");
            }
        }
    }
}
