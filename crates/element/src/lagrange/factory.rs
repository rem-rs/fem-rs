//! Arbitrary-order Lagrange finite elements via monomial Vandermonde precomputation.
//!
//! Provides [`SegPk`], [`TriPk`], [`TetPk`], [`QuadQk`], and [`HexQk`] — each
//! implementing [`ReferenceElement`] for any polynomial order `p ≥ 1`.
//!
//! # Approach
//! For simplex elements (Seg, Tri, Tet) equispaced nodal points are used.
//! Basis function evaluation is performed by precomputing Lagrange coefficients
//! from the monomial Vandermonde matrix at construction time, then evaluating
//! the polynomial at each query point.
//!
//! For tensor-product elements (Quad, Hex) the 1-D Lagrange basis on `[-1,1]`
//! is evaluated via a stable product formula.
//!
//! # DOF ordering
//! Each element type follows the MFEM-compatible ordering:
//! 1. Vertices first
//! 2. Edge interior DOFs (for each edge, ordered from lower to higher vertex)
//! 3. Face interior DOFs (for each face)
//! 4. Volume interior DOFs (for each element)

use super::prism::PrismPk;
use super::pyramid::PyramidPk;
use crate::quadrature::{hex_rule, quad_rule_01, seg_rule, tet_rule, tri_rule};
use crate::reference::{QuadratureRule, ReferenceElement, VectorReferenceElement};
use crate::serendipity::{HexSerendipityPk, QuadSerendipityPk};
use nalgebra::DMatrix;

// ─── Helpers: equispaced nodes ────────────────────────────────────────────────

fn equispaced_nodes_1d(p: usize) -> Vec<f64> {
    (0..=p).map(|i| i as f64 / p as f64).collect()
}

fn equispaced_nodes_tri(p: usize) -> Vec<[f64; 2]> {
    let mut nodes = Vec::with_capacity((p + 1) * (p + 2) / 2);
    nodes.push([0.0, 0.0]);
    nodes.push([1.0, 0.0]);
    nodes.push([0.0, 1.0]);
    if p == 1 {
        return nodes;
    }
    for k in 1..p {
        nodes.push([k as f64 / p as f64, 0.0]);
    }
    for k in 1..p {
        let t = k as f64 / p as f64;
        nodes.push([1.0 - t, t]);
    }
    for k in 1..p {
        nodes.push([0.0, k as f64 / p as f64]);
    }
    // Interior nodes: i from p-2 down to 1, j from p-1-i down to 1 (matches TriP4 IJK ordering)
    for i in (1..=(p - 2)).rev() {
        for j in (1..=(p - 1 - i)).rev() {
            nodes.push([i as f64 / p as f64, j as f64 / p as f64]);
        }
    }
    debug_assert_eq!(nodes.len(), (p + 1) * (p + 2) / 2);
    nodes
}

fn equispaced_nodes_tet(p: usize) -> Vec<[f64; 3]> {
    let mut nodes = Vec::with_capacity((p + 1) * (p + 2) * (p + 3) / 6);
    nodes.push([0.0, 0.0, 0.0]);
    nodes.push([1.0, 0.0, 0.0]);
    nodes.push([0.0, 1.0, 0.0]);
    nodes.push([0.0, 0.0, 1.0]);
    if p == 1 {
        return nodes;
    }
    for k in 1..p {
        nodes.push([k as f64 / p as f64, 0.0, 0.0]);
    }
    for k in 1..p {
        nodes.push([0.0, k as f64 / p as f64, 0.0]);
    }
    for k in 1..p {
        nodes.push([0.0, 0.0, k as f64 / p as f64]);
    }
    for k in 1..p {
        let t = k as f64 / p as f64;
        nodes.push([1.0 - t, t, 0.0]);
    }
    for k in 1..p {
        let t = k as f64 / p as f64;
        nodes.push([1.0 - t, 0.0, t]);
    }
    for k in 1..p {
        let t = k as f64 / p as f64;
        nodes.push([0.0, 1.0 - t, t]);
    }
    for j in 1..=(p.saturating_sub(2)) {
        for i in 1..=(p - 1 - j) {
            nodes.push([i as f64 / p as f64, j as f64 / p as f64, 0.0]);
        }
    }
    for k in 1..=(p.saturating_sub(2)) {
        for i in 1..=(p - 1 - k) {
            nodes.push([i as f64 / p as f64, 0.0, k as f64 / p as f64]);
        }
    }
    for k in 1..=(p.saturating_sub(2)) {
        for j in 1..=(p - 1 - k) {
            nodes.push([0.0, j as f64 / p as f64, k as f64 / p as f64]);
        }
    }
    for k in 1..=(p.saturating_sub(2)) {
        for j in 1..=(p - 1 - k) {
            let fj = j as f64 / p as f64;
            let fk = k as f64 / p as f64;
            nodes.push([1.0 - fj - fk, fj, fk]);
        }
    }
    for k in 1..=(p.saturating_sub(3)) {
        for j in 1..=(p - 2 - k) {
            for i in 1..=(p - 1 - j - k) {
                nodes.push([
                    i as f64 / p as f64,
                    j as f64 / p as f64,
                    k as f64 / p as f64,
                ]);
            }
        }
    }
    let expected = (p + 1) * (p + 2) * (p + 3) / 6;
    debug_assert_eq!(
        nodes.len(),
        expected,
        "tet p={p}: got {} nodes, expected {expected}",
        nodes.len()
    );
    nodes
}

// ─── 1D Lagrange helpers (direct formula, no Vandermonde) ──────────────────────

// ─── SegPk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on `[0,1]` — `(p+1)` DOFs.
///
/// DOF ordering: ξ = 0, 1/p, 2/p, …, 1 (equispaced, vertices first).
pub struct SegPk {
    order: usize,
}

impl SegPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        Self { order: p }
    }
}

impl ReferenceElement for SegPk {
    fn dim(&self) -> u8 {
        1
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.order + 1
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let t = p as f64 * xi[0];
        for dof_idx in 0..=p {
            values[dof_idx] = lagrange_val(dof_idx, p, t);
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.order;
        let t = p as f64 * xi[0];
        for dof_idx in 0..=p {
            grads[dof_idx] = p as f64 * lagrange_deriv(dof_idx, p, t);
        }
    }
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        let p = self.order;
        let t = p as f64 * xi[0];
        let p2 = (p as f64) * (p as f64);
        for dof_idx in 0..=p {
            hess[dof_idx] = p2 * lagrange_hess(dof_idx, p, t);
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        seg_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        equispaced_nodes_1d(self.order)
            .iter()
            .map(|&x| vec![x])
            .collect()
    }
}

// ─── 1D Lagrange helpers (direct formula, no Vandermonde) ──────────────────────

/// Evaluate the standard degree-p Lagrange polynomial through integer nodes
/// {0, 1, ..., p}: l_n(t) = Π_{m≠n} (t-m)/(n-m).
/// Used by SegPk (1D elements) where the standard Lagrange basis is correct.
pub fn lagrange_val(n: usize, p: usize, t: f64) -> f64 {
    let mut val = 1.0;
    let tn = n as f64;
    for m in 0..=p {
        if m != n {
            val *= (t - m as f64) / (tn - m as f64);
        }
    }
    val
}

/// Derivative of the standard degree-p Lagrange polynomial l_n'(t).
pub fn lagrange_deriv(n: usize, p: usize, t: f64) -> f64 {
    let mut sum = 0.0;
    let tn = n as f64;
    for k in 0..=p {
        if k != n {
            let mut term = 1.0;
            let tk = k as f64;
            for m in 0..=p {
                if m != n && m != k {
                    term *= (t - m as f64) / (tn - m as f64);
                }
            }
            sum += term / (tn - tk);
        }
    }
    sum
}

/// Second derivative of the degree-p Lagrange polynomial `l_n(t)`.
/// `l_n''(t) = l_n(t) * [(Σ 1/(t-m))² - Σ 1/(t-m)²]`.
pub fn lagrange_hess(n: usize, p: usize, t: f64) -> f64 {
    let mut s1 = 0.0; // Σ 1/(t-m)
    let mut s2 = 0.0; // Σ 1/(t-m)²
    for m in 0..=p {
        if m != n {
            let inv = 1.0 / (t - m as f64);
            s1 += inv;
            s2 += inv * inv;
        }
    }
    lagrange_val(n, p, t) * (s1 * s1 - s2)
}

/// Rising-factorial basis L_n(t) = Π_{a=0}^{n-1} (t - a) / (n - a), with L_0(t) = 1.
/// This is the correct building block for simplex Lagrange elements, NOT the
/// standard Lagrange polynomial through integer nodes.
pub fn rising_val(n: usize, t: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut val = 1.0;
    for a in 0..n {
        val *= (t - a as f64) / (n as f64 - a as f64);
    }
    val
}

/// Derivative of the rising-factorial basis L_n'(t).
/// L_n(t) = Π_{a=0}^{n-1} (t-a)/(n-a), so
/// L_n'(t) = Σ_{b=0}^{n-1} 1/(n-b) · Π_{a≠b} (t-a)/(n-a)
pub fn rising_deriv(n: usize, t: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut sum = 0.0;
    for b in 0..n {
        let mut term = 1.0;
        for a in 0..n {
            if a != b {
                term *= (t - a as f64) / (nf - a as f64);
            }
        }
        sum += term / (nf - b as f64);
    }
    sum
}

/// Second derivative of the rising-factorial basis L_n''(t).
/// L_n''(t) = L_n(t) * [(Σ 1/(t-a))² - Σ 1/(t-a)²]
pub fn rising_hess(n: usize, t: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let ln = rising_val(n, t);
    let _nf = n as f64;
    let mut s1 = 0.0; // Σ 1/(t-a)
    let mut s2 = 0.0; // Σ 1/(t-a)²
    for a in 0..n {
        let inv = 1.0 / (t - a as f64);
        s1 += inv;
        s2 += inv * inv;
    }
    ln * (s1 * s1 - s2)
}

// ─── TriPk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference triangle `(0,0),(1,0),(0,1)` —
/// `(p+1)(p+2)/2` DOFs.
///
/// Uses the direct barycentric Lagrange formula (stable for any order) instead
/// of a Vandermonde-based coefficient approach (which becomes ill-conditioned
/// at p ≥ 3 for simplex elements).
pub struct TriPk {
    order: usize,
    nodes: Vec<[f64; 2]>,
    ijk: Vec<(usize, usize, usize)>,
}

impl TriPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes = equispaced_nodes_tri(p);
        let ijk: Vec<(usize, usize, usize)> = nodes
            .iter()
            .map(|n| {
                let i = (n[0] * p as f64).round() as usize;
                let j = (n[1] * p as f64).round() as usize;
                (i, j, p - i - j)
            })
            .collect();
        Self {
            order: p,
            nodes,
            ijk,
        }
    }
}

impl ReferenceElement for TriPk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 2) / 2
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in self.ijk.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2);
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in self.ijk.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            // ∂φ/∂ξ = p·(di·vj·vk - vi·vj·dk)
            grads[dof_idx * 2] = pf * (di * vj * vk - vi * vj * dk);
            // ∂φ/∂η = p·(vi·dj·vk - vi·vj·dk)
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let pf2 = pf * pf;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in self.ijk.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            let hii = rising_hess(i, t0);
            let hjj = rising_hess(j, t1);
            let hkk = rising_hess(k, t2);
            let base = dof_idx * 4;
            // ∂²φ/∂ξ², ∂²φ/∂ξ∂η, ∂²φ/∂η∂ξ, ∂²φ/∂η²
            hess[base] = pf2 * (hii * vj * vk - 2.0 * di * vj * dk + vi * vj * hkk);
            hess[base + 1] = pf2 * (di * dj * vk - di * vj * dk - vi * dj * dk + vi * vj * hkk);
            hess[base + 2] = hess[base + 1];
            hess[base + 3] = pf2 * (vi * hjj * vk - 2.0 * vi * dj * dk + vi * vj * hkk);
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── H1TriPk: MFEM H1_TriangleElement (Gauss-Lobatto nodes) ────────────────

/// MFEM `H1_TriangleElement(p)` clone on the reference triangle
/// `(0,0),(1,0),(0,1)` — `(p+1)(p+2)/2` DOFs ordered vertices → edges →
/// interior, with the edge/interior DOFs at **Gauss-Lobatto closed points**
/// (MFEM `H1_FECollection` default `BasisType::GaussLobatto`), NOT the
/// equispaced points used by [`TriPk`] (which serves the DG/L2 paths).
///
/// Node layout mirrors `H1_TriangleElement::H1_TriangleElement` (fe_h1.cpp):
/// - vertices: `(0,0)`, `(1,0)`, `(0,1)`
/// - edge 0 (v0→v1): `(cp[i], 0)` for i = 1..p-1
/// - edge 1 (v1→v2): `(cp[p-i], cp[i])` for i = 1..p-1
/// - edge 2 (v2→v0): `(0, cp[p-i])` for i = 1..p-1  (from the **v2** end!)
/// - interior: `(cp[i]/w, cp[j]/w)`, w = cp[i]+cp[j]+cp[p-i-j],
///   j = 1..p-1, i = 1..p-1-j
///
/// where `cp` are the `p+1` GLL closed points on `[0,1]`.  The basis is the
/// Vandermonde inverse of the lexicographic product basis
/// `s_o(ξ) = L_i(ξ0)·L_j(ξ1)·L_k(1−ξ0−ξ1)` (o = `idx(i,j)`, k = p−i−j, `L` =
/// 1-D GLL Lagrange polynomials), exactly as MFEM computes it, so the DOFs
/// are nodal values at `nodes`.
pub struct H1TriPk {
    order: usize,
    nodes: Vec<[f64; 2]>,
    gll: Vec<f64>, // GLL closed points on [0,1], size p+1
    lex: Vec<(usize, usize)>, // lex (i,j) with i+j <= p (MFEM idx order)
    ti: Vec<f64>, // dof×dof row-major: φ_k = Σ_o ti[k·dof+o]·s_o
}

/// Evaluate the 1-D Lagrange basis on `nodes` at `x`.
fn lag1d_on(nodes: &[f64], x: f64, out: &mut [f64]) {
    let n = nodes.len();
    for i in 0..n {
        let mut v = 1.0;
        for m in 0..n {
            if m != i {
                v *= (x - nodes[m]) / (nodes[i] - nodes[m]);
            }
        }
        out[i] = v;
    }
}

/// Evaluate the 1-D Lagrange basis derivatives on `nodes` at `x`.
fn dlag1d_on(nodes: &[f64], x: f64, out: &mut [f64]) {
    let n = nodes.len();
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut v = 1.0 / (nodes[i] - nodes[j]);
            for m in 0..n {
                if m != i && m != j {
                    v *= (x - nodes[m]) / (nodes[i] - nodes[m]);
                }
            }
            s += v;
        }
        out[i] = s;
    }
}

impl H1TriPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        // GLL closed points on [-1,1], mapped to [0,1] (MFEM poly1d::ClosedPoints).
        let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
        let gll: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
        Self::with_closed_points(p, gll)
    }

    /// The **closed-uniform** (`BasisType::ClosedUniform`) twin of
    /// [`H1TriPk::new`]: same MFEM slot order (vertices → edges → interior,
    /// edge 2 counted from its `v2` end, interior at the `w`-normalised
    /// barycentric nodes), but the 1-D points are the *equispaced* `i/p`.
    ///
    /// This is MFEM's legacy `Quadratic`/`Cubic` `Quadratic2DFiniteElement` /
    /// `Cubic2DFiniteElement` node placement.  `fem-io` uses it (D112) to
    /// reinterpret an MFEM `nodes` grid function written with a *legacy*
    /// `FiniteElementCollection` name (`Linear`/`Quadratic`/`Cubic`).
    pub fn new_closed_uniform(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        let cp: Vec<f64> = (0..=p).map(|i| i as f64 / p as f64).collect();
        Self::with_closed_points(p, cp)
    }

    /// Build the element for an arbitrary increasing set of `p+1` 1-D nodes on
    /// `[0,1]`.  The nodes of the triangle are the MFEM `H1_TriangleElement`
    /// placement of `cp` (see the struct docs) and the basis is the unique
    /// nodal Lagrange basis there, so the choice of `cp` only moves the DOF
    /// positions — the slot semantics never change.
    fn with_closed_points(p: usize, gll: Vec<f64>) -> Self {
        let mut nodes: Vec<[f64; 2]> = Vec::with_capacity((p + 1) * (p + 2) / 2);
        nodes.push([gll[0], gll[0]]);
        nodes.push([gll[p], gll[0]]);
        nodes.push([gll[0], gll[p]]);
        for i in 1..p {
            nodes.push([gll[i], gll[0]]);
        }
        for i in 1..p {
            nodes.push([gll[p - i], gll[i]]);
        }
        for i in 1..p {
            nodes.push([gll[0], gll[p - i]]);
        }
        for j in 1..p {
            for i in 1..(p - j) {
                let w = gll[i] + gll[j] + gll[p - i - j];
                nodes.push([gll[i] / w, gll[j] / w]);
            }
        }

        // Lex order with MFEM's idx(i,j) = ((2p+3-j)·j)/2 + i.
        let p2p3 = 2 * p + 3;
        let mut lex = Vec::with_capacity(nodes.len());
        for j in 0..=p {
            for i in 0..=(p - j) {
                let o = (p2p3 - j) * j / 2 + i;
                lex.push((i, j));
                debug_assert_eq!(lex.len() - 1, o, "H1TriPk lex idx mismatch");
            }
        }
        debug_assert_eq!(lex.len(), nodes.len());

        // Vandermonde on the *monomial* basis m_o = x^i·y^j (i+j <= p, lex
        // order).  Using the 1-D GLL tensor product lx·ly·ll here produced
        // basis functions of degree 3p (not p): interpolation at the nodes
        // held, but the partition of unity failed away from the nodes
        // (Σφ ≈ 0.78 at [0.2,0.3] for p=3), corrupting every H1 assembly
        // of order >= 3 (P3 Poisson L2 error ~0.5).
        let n = nodes.len();
        let mut t = DMatrix::<f64>::zeros(n, n);
        for (k, node) in nodes.iter().enumerate() {
            for (o, &(i, j)) in lex.iter().enumerate() {
                // monomial x^i y^j evaluated at node k
                t[(o, k)] = node[0].powi(i as i32) * node[1].powi(j as i32);
            }
        }
        let ti_m = t
            .try_inverse()
            .expect("H1TriPk: singular Vandermonde matrix");
        let mut ti = vec![0.0; n * n];
        for k in 0..n {
            for o in 0..n {
                ti[k * n + o] = ti_m[(k, o)];
            }
        }
        Self { order: p, nodes, gll, lex, ti }
    }
}

impl ReferenceElement for H1TriPk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.nodes.len()
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let n = self.nodes.len();
        let (x, y) = (xi[0], xi[1]);
        // Monomial basis m_o = x^i·y^j (i+j <= p), matching the Vandermonde
        // used in `new`.  (The old 1-D GLL tensor product produced degree-3p
        // functions that failed the partition of unity.)
        let mut s = vec![0.0; n];
        let mut xp = vec![1.0; p + 1];
        let mut yp = vec![1.0; p + 1];
        for i in 1..=p {
            xp[i] = xp[i - 1] * x;
            yp[i] = yp[i - 1] * y;
        }
        for (o, &(i, j)) in self.lex.iter().enumerate() {
            s[o] = xp[i] * yp[j];
        }
        for k in 0..n {
            let mut acc = 0.0;
            for o in 0..n {
                acc += self.ti[k * n + o] * s[o];
            }
            values[k] = acc;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.order;
        let n = self.nodes.len();
        let (x, y) = (xi[0], xi[1]);
        // Derivatives of the monomial basis: ∂m/∂x = i·x^{i-1}·y^j,
        // ∂m/∂y = j·x^i·y^{j-1}.
        let mut xp = vec![1.0; p + 1];
        let mut yp = vec![1.0; p + 1];
        for i in 1..=p {
            xp[i] = xp[i - 1] * x;
            yp[i] = yp[i - 1] * y;
        }
        let mut ds = vec![0.0; n * 2];
        for (o, &(i, j)) in self.lex.iter().enumerate() {
            let dx = if i > 0 { (i as f64) * xp[i - 1] * yp[j] } else { 0.0 };
            let dy = if j > 0 { (j as f64) * xp[i] * yp[j - 1] } else { 0.0 };
            ds[o * 2] = dx;
            ds[o * 2 + 1] = dy;
        }
        for k in 0..n {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for o in 0..n {
                gx += self.ti[k * n + o] * ds[o * 2];
                gy += self.ti[k * n + o] * ds[o * 2 + 1];
            }
            grads[k * 2] = gx;
            grads[k * 2 + 1] = gy;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── H1TetPk: MFEM H1_TetrahedronElement (Gauss-Lobatto nodes) ──────────────

/// Integer barycentric coordinates of MFEM `H1_TetrahedronElement(p)`'s node
/// enumeration, in the order `(λ₁, λ₂, λ₃, λ₄)` with
/// `λ₁ = 1-x-y-z, λ₂ = x, λ₃ = y, λ₄ = z` (so every entry sums to `p`).
///
/// The enumeration is `fem/fe/fe_h1.cpp`'s: vertices, then the six edges in
/// `Geometry::TETRAHEDRON` order `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`,
/// then the four faces in `Mesh::GenerateFaces`/`TET_FACES` order
/// `{1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}`, then the interior in MFEM's
/// `(k, j, i)` nested order.
///
/// These labels are what the mesh loader keys a geometric DOF by (the *slot*
/// of the reference element is a position; the label is its identity), so they
/// are exposed separately from [`H1TetPk`]'s node coordinates — a position has
/// to be rounded to an integer grid, a label does not.
pub fn h1_tet_slot_labels(p: usize) -> Vec<[usize; 4]> {
    assert!(p >= 1, "h1_tet_slot_labels: order must be >= 1");
    let mut labels: Vec<[usize; 4]> = Vec::with_capacity((p + 1) * (p + 2) * (p + 3) / 6);
    // vertices
    labels.push([p, 0, 0, 0]);
    labels.push([0, p, 0, 0]);
    labels.push([0, 0, p, 0]);
    labels.push([0, 0, 0, p]);
    // edges (see `Tetrahedron::edges` in mesh/tetrahedron.cpp)
    for i in 1..p {
        labels.push([p - i, i, 0, 0]);
    }
    for i in 1..p {
        labels.push([p - i, 0, i, 0]);
    }
    for i in 1..p {
        labels.push([p - i, 0, 0, i]);
    }
    for i in 1..p {
        labels.push([0, p - i, i, 0]);
    }
    for i in 1..p {
        labels.push([0, p - i, 0, i]);
    }
    for i in 1..p {
        labels.push([0, 0, p - i, i]);
    }
    // faces (see `Mesh::GenerateFaces`)
    for j in 1..p {
        for i in 1..(p - j) {
            labels.push([0, p - i - j, i, j]);
        }
    }
    for j in 1..p {
        for i in 1..(p - j) {
            labels.push([p - i - j, 0, j, i]);
        }
    }
    for j in 1..p {
        for i in 1..(p - j) {
            labels.push([p - i - j, i, 0, j]);
        }
    }
    for j in 1..p {
        for i in 1..(p - j) {
            labels.push([p - i - j, j, i, 0]);
        }
    }
    // interior
    for k in 1..p {
        for j in 1..(p - k) {
            for i in 1..(p - j - k) {
                labels.push([p - i - j - k, i, j, k]);
            }
        }
    }
    debug_assert_eq!(labels.len(), (p + 1) * (p + 2) * (p + 3) / 6);
    labels
}

/// MFEM `H1_TetrahedronElement(p)` clone on the reference tetrahedron
/// `(0,0,0),(1,0,0),(0,1,0),(0,0,1)` — `(p+1)(p+2)(p+3)/6` DOFs at the **closed
/// Gauss-Lobatto points** (MFEM `H1_FECollection`'s default
/// `BasisType::GaussLobatto`), in MFEM's DOF order (see
/// [`h1_tet_slot_labels`]).
///
/// This is the tetrahedron analogue of [`H1TriPk`], and it is *not*
/// [`TetPk`]: `TetPk` uses **equispaced** nodes on the integer barycentric grid
/// (so its DOFs sit at `k/p` along every edge).  For `p ≤ 2` the two coincide
/// (the closed Gauss-Lobatto points of `p = 2` are the midpoints); from `p = 3`
/// on they differ (`1/3` vs `0.2764`), which changes the isoparametric map of a
/// curved mesh and, one level up, the H¹ basis itself.
///
/// The basis is the Vandermonde inverse of MFEM's *hierarchical* product basis
/// `s_o = T_i(x)·T_j(y)·T_k(z)·T_l(1−x−y−z)` (o enumerated with `k` outer, `j`
/// middle, `i` inner; `l = p−i−j−k`; `T_n` = the Chebyshev polynomial of
/// degree `n` evaluated at `2ξ−1`, i.e. exactly `Poly_1D::CalcBasis`), so the
/// functions are the unique nodal Lagrange basis at `nodes` and reproduce
/// MFEM's `CalcShape`/`CalcDShape` up to round-off.  Using the Chebyshev basis
/// rather than monomials keeps the Vandermonde inversion of the `p = 6`
/// (84 DOF) element well conditioned.
pub struct H1TetPk {
    inner: std::sync::Arc<H1TetPkInner>,
}

struct H1TetPkInner {
    order: usize,
    nodes: Vec<[f64; 3]>,
    lex: Vec<(usize, usize, usize)>,
    /// Row-major DOF × DOF: `φ_m = Σ_o ti[m·dof+o]·s_o`.
    ti: Vec<f64>,
}

/// Chebyshev polynomials `T_0..T_p` evaluated at `2x−1` (MFEM
/// `Poly_1D::CalcChebyshev`).
fn cheb_1d(p: usize, x: f64) -> Vec<f64> {
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

/// Values and first derivatives of `T_0..T_p` at `2x−1` (MFEM
/// `Poly_1D::CalcChebyshev(p, x, u, d)`).
fn cheb_1d_d(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
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
        d[n + 1] = (n as f64 + 1.0) * (z * d[n] / n as f64 + 2.0 * u[n]);
    }
    (u, d)
}

fn h1_tet_pk_build(p: usize) -> H1TetPkInner {
    // Closed Gauss-Lobatto points on [0,1] (MFEM `poly1d.ClosedPoints(p)`).
    let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
    let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
    h1_tet_pk_build_with_points(p, &cp)
}

/// Build the `H1_TetrahedronElement(p)`-slot element for an arbitrary
/// increasing set of `p+1` 1-D nodes `cp` on `[0,1]`.
///
/// The node *placement* (which barycentric label each slot carries) never
/// depends on `cp`, so this is the shared body of [`H1TetPk::new`] (MFEM's
/// closed Gauss-Lobatto points) and [`H1TetPk::new_closed_uniform`] (the
/// equispaced `i/p` of the legacy `CubicFECollection`).
fn h1_tet_pk_build_with_points(p: usize, cp: &[f64]) -> H1TetPkInner {
    let labels = h1_tet_slot_labels(p);
    let mut nodes: Vec<[f64; 3]> = labels
        .iter()
        .map(|&[l, i, j, k]| {
            let w = cp[i] + cp[j] + cp[k] + cp[l];
            if w == 0.0 {
                // A vertex: exactly one barycentric coordinate is 1.
                [cp[i], cp[j], cp[k]]
            } else {
                // `w` is 1 for an edge node (two nonzero coordinates summing
                // to p) and the MFEM normalisation otherwise.
                [cp[i] / w, cp[j] / w, cp[k] / w]
            }
        })
        .collect();
    // Vertices: the formula above divides by `cp[l] = 1` (or `cp[i] = 1`), which
    // is exact — but keep the explicit positions so the vertices are bit-exact.
    nodes[0] = [cp[0], cp[0], cp[0]];
    nodes[1] = [cp[p], cp[0], cp[0]];
    nodes[2] = [cp[0], cp[p], cp[0]];
    nodes[3] = [cp[0], cp[0], cp[p]];

    // MFEM's basis enumeration: k outer, j middle, i inner.
    let mut lex = Vec::with_capacity(nodes.len());
    for k in 0..=p {
        for j in 0..=(p - k) {
            for i in 0..=(p - j - k) {
                lex.push((i, j, k));
            }
        }
    }
    debug_assert_eq!(lex.len(), nodes.len());

    let n = nodes.len();
    let mut t = DMatrix::<f64>::zeros(n, n);
    for (m, node) in nodes.iter().enumerate() {
        let l = 1.0 - node[0] - node[1] - node[2];
        let sx = cheb_1d(p, node[0]);
        let sy = cheb_1d(p, node[1]);
        let sz = cheb_1d(p, node[2]);
        let sl = cheb_1d(p, l);
        for (o, &(i, j, k)) in lex.iter().enumerate() {
            t[(o, m)] = sx[i] * sy[j] * sz[k] * sl[p - i - j - k];
        }
    }
    let ti_m = t.try_inverse().expect("H1TetPk: singular Vandermonde matrix");
    let mut ti = vec![0.0; n * n];
    for m in 0..n {
        for o in 0..n {
            ti[m * n + o] = ti_m[(m, o)];
        }
    }
    H1TetPkInner { order: p, nodes, lex, ti }
}

impl H1TetPk {
    /// Build (or fetch from the per-order cache) the element.
    ///
    /// `geo_ref_elem` builds the geometry element **once per element**, and the
    /// Vandermonde inversion is `O(dof³)` (≈ 6·10⁵ flops at `p = 6`), so the
    /// tables are cached per order and shared.
    pub fn new(p: usize) -> Self {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<H1TetPkInner>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let inner = {
            let mut m = cache.lock().expect("H1TetPk cache poisoned");
            m.entry(p).or_insert_with(|| Arc::new(h1_tet_pk_build(p))).clone()
        };
        Self { inner }
    }

    /// The **closed-uniform** (`BasisType::ClosedUniform`) twin of
    /// [`H1TetPk::new`]: same MFEM slot order (see [`h1_tet_slot_labels`]), but
    /// the 1-D points are the *equispaced* `i/p`, so the tetrahedron's DOFs sit
    /// on the integer barycentric lattice.
    ///
    /// This is MFEM's legacy `Quadratic`/`Cubic` `Quadratic3DFiniteElement` /
    /// `Cubic3DFiniteElement` node placement.  `fem-io` uses it (D112) to
    /// reinterpret an MFEM `nodes` grid function written with a *legacy*
    /// `FiniteElementCollection` name (`Linear`/`Quadratic`/`Cubic`).
    pub fn new_closed_uniform(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        let cp: Vec<f64> = (0..=p).map(|i| i as f64 / p as f64).collect();
        Self {
            inner: std::sync::Arc::new(h1_tet_pk_build_with_points(p, &cp)),
        }
    }

    /// MFEM's `H1_TetrahedronElement(p)` node enumeration — see
    /// [`h1_tet_slot_labels`].
    pub fn slot_labels(p: usize) -> Vec<[usize; 4]> {
        h1_tet_slot_labels(p)
    }
}

impl ReferenceElement for H1TetPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.inner.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.inner.nodes.len()
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.inner.order;
        let n = self.inner.nodes.len();
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let sx = cheb_1d(p, x);
        let sy = cheb_1d(p, y);
        let sz = cheb_1d(p, z);
        let sl = cheb_1d(p, 1.0 - x - y - z);
        let mut u = vec![0.0; n];
        for (o, &(i, j, k)) in self.inner.lex.iter().enumerate() {
            u[o] = sx[i] * sy[j] * sz[k] * sl[p - i - j - k];
        }
        for m in 0..n {
            let mut acc = 0.0;
            for o in 0..n {
                acc += self.inner.ti[m * n + o] * u[o];
            }
            values[m] = acc;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.inner.order;
        let n = self.inner.nodes.len();
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (sx, dx) = cheb_1d_d(p, x);
        let (sy, dy) = cheb_1d_d(p, y);
        let (sz, dz) = cheb_1d_d(p, z);
        let (sl, dl) = cheb_1d_d(p, 1.0 - x - y - z);
        // MFEM `H1_TetrahedronElement::CalcDShape`: the `1-x-y-z` coordinate
        // contributes `-dshape_l` to every partial derivative.
        let mut du = vec![0.0_f64; n * 3];
        for (o, &(i, j, k)) in self.inner.lex.iter().enumerate() {
            let l = p - i - j - k;
            du[o * 3] = (dx[i] * sl[l] - sx[i] * dl[l]) * sy[j] * sz[k];
            du[o * 3 + 1] = (dy[j] * sl[l] - sy[j] * dl[l]) * sx[i] * sz[k];
            du[o * 3 + 2] = (dz[k] * sl[l] - sz[k] * dl[l]) * sx[i] * sy[j];
        }
        for m in 0..n {
            let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
            for o in 0..n {
                let c = self.inner.ti[m * n + o];
                gx += c * du[o * 3];
                gy += c * du[o * 3 + 1];
                gz += c * du[o * 3 + 2];
            }
            grads[m * 3] = gx;
            grads[m * 3 + 1] = gy;
            grads[m * 3 + 2] = gz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.inner
            .nodes
            .iter()
            .map(|c| vec![c[0], c[1], c[2]])
            .collect()
    }
}

// ─── TetPk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference tetrahedron —
/// `(p+1)(p+2)(p+3)/6` DOFs.
///
/// Uses the direct barycentric Lagrange formula (stable for any order) instead
/// of a Vandermonde-based coefficient approach.
pub struct TetPk {
    order: usize,
    nodes: Vec<[f64; 3]>,
    ijkl: Vec<(usize, usize, usize, usize)>,
}

impl TetPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes = equispaced_nodes_tet(p);
        let ijkl: Vec<(usize, usize, usize, usize)> = nodes
            .iter()
            .map(|n| {
                let i = (n[0] * p as f64).round() as usize;
                let j = (n[1] * p as f64).round() as usize;
                let k = (n[2] * p as f64).round() as usize;
                (i, j, k, p - i - j - k)
            })
            .collect();
        Self {
            order: p,
            nodes,
            ijkl,
        }
    }
}

impl ReferenceElement for TetPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 2) * (self.order + 3) / 6
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * xi[2];
        let t3 = pf * (1.0 - xi[0] - xi[1] - xi[2]);
        for (dof_idx, &(i, j, k, l)) in self.ijkl.iter().enumerate() {
            values[dof_idx] =
                rising_val(i, t0) * rising_val(j, t1) * rising_val(k, t2) * rising_val(l, t3);
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * xi[2];
        let t3 = pf * (1.0 - xi[0] - xi[1] - xi[2]);
        for (dof_idx, &(i, j, k, l)) in self.ijkl.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let vl = rising_val(l, t3);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            let dl = rising_deriv(l, t3);
            grads[dof_idx * 3] = pf * (di * vj * vk * vl - vi * vj * vk * dl);
            grads[dof_idx * 3 + 1] = pf * (vi * dj * vk * vl - vi * vj * vk * dl);
            grads[dof_idx * 3 + 2] = pf * (vi * vj * dk * vl - vi * vj * vk * dl);
        }
    }
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let pf2 = pf * pf;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * xi[2];
        let t3 = pf * (1.0 - xi[0] - xi[1] - xi[2]);
        for (dof_idx, &(i, j, k, l)) in self.ijkl.iter().enumerate() {
            let vi = rising_val(i, t0);
            let vj = rising_val(j, t1);
            let vk = rising_val(k, t2);
            let vl = rising_val(l, t3);
            let di = rising_deriv(i, t0);
            let dj = rising_deriv(j, t1);
            let dk = rising_deriv(k, t2);
            let dl = rising_deriv(l, t3);
            let hii = rising_hess(i, t0);
            let hjj = rising_hess(j, t1);
            let hkk = rising_hess(k, t2);
            let hll = rising_hess(l, t3);
            let base = dof_idx * 9;
            // d²φ/dξ², d²φ/dξdη, d²φ/dξdζ
            hess[base] = pf2 * (hii * vj * vk * vl - 2.0 * di * vj * vk * dl + vi * vj * vk * hll);
            hess[base + 1] = pf2
                * (di * dj * vk * vl - di * vj * vk * dl - vi * dj * vk * dl + vi * vj * vk * hll);
            hess[base + 2] = pf2
                * (di * vj * dk * vl - di * vj * vk * dl - vi * vj * dk * dl + vi * vj * vk * hll);
            // d²φ/dηdξ, d²φ/dη², d²φ/dηdζ
            hess[base + 3] = hess[base + 1];
            hess[base + 4] =
                pf2 * (vi * hjj * vk * vl - 2.0 * vi * dj * vk * dl + vi * vj * vk * hll);
            hess[base + 5] = pf2
                * (vi * dj * dk * vl - vi * dj * vk * dl - vi * vj * dk * dl + vi * vj * vk * hll);
            // d²φ/dζdξ, d²φ/dζdη, d²φ/dζ²
            hess[base + 6] = hess[base + 2];
            hess[base + 7] = hess[base + 5];
            hess[base + 8] =
                pf2 * (vi * vj * hkk * vl - 2.0 * vi * vj * dk * dl + vi * vj * vk * hll);
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}

// ─── Lagrange1D: shared 1D barycentric basis for Quad and Hex ──────────────

/// Pre-computed 1D equispaced Lagrange basis on `[-1, 1]` (barycentric form).
///
/// Shared by [`QuadQk`] and [`HexQk`] to avoid duplicating the O(p) evaluation
/// methods (`val`, `val_d`, `val_d_h`).
pub(crate) struct Lagrange1D {
    pub(crate) nodes: Vec<f64>,
    pub(crate) bary_w: Vec<f64>,
}

impl Lagrange1D {
    pub(crate) fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        // Use Gauss-Lobatto-Legendre nodes matching MFEM's H1_FECollection
        // (BasisType::GaussLobatto).  For p=1,2 these are identical to
        // equispaced; for p>=3 the GLL nodes cluster near the boundaries,
        // eliminating Runge oscillations and matching MFEM's diagonal
        // preconditioner spectral properties.
        let (nodes, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
        Self::from_nodes(nodes)
    }

    /// Build a 1D Lagrange interpolant on arbitrary nodes on `[-1,1]`.
    pub(crate) fn from_nodes(nodes: Vec<f64>) -> Self {
        let n = nodes.len();
        let mut bary_w = vec![1.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                if j != i {
                    bary_w[i] *= nodes[i] - nodes[j];
                }
            }
            bary_w[i] = 1.0 / bary_w[i];
        }
        Self { nodes, bary_w }
    }

    fn ell(&self, x: f64) -> f64 {
        let mut e = 1.0_f64;
        for &xj in &self.nodes {
            e *= x - xj;
        }
        e
    }

    /// Evaluate all 1D Lagrange basis values at `x` in O(p).
    pub(crate) fn val(&self, x: f64) -> Vec<f64> {
        let n = self.nodes.len();
        let mut vals = vec![0.0_f64; n];
        let ell = self.ell(x);
        if ell.abs() < 1e-30 {
            for (i, &xi) in self.nodes.iter().enumerate() {
                if (x - xi).abs() < 1e-30 {
                    vals[i] = 1.0;
                    break;
                }
            }
            return vals;
        }
        for (i, &xi) in self.nodes.iter().enumerate() {
            vals[i] = ell * self.bary_w[i] / (x - xi);
        }
        vals
    }

    /// Evaluate values and first derivatives.
    pub(crate) fn val_d(&self, x: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.nodes.len();
        // Exact evaluation AT a node: the generic barycentric derivative
        // formula hits a removable 0·∞ singularity there (l_i(x_k)=0 times
        // Σ1/(x−x_j)=∞), so use the closed forms
        //   l_k'(x_k) = Σ_{j≠k} 1/(x_k−x_j),
        //   l_i'(x_k) = (w_i/w_k)/(x_k−x_i)   (i ≠ k).
        if let Some(k) = self.nodes.iter().position(|&xj| (x - xj).abs() < 1e-14) {
            let mut vals = vec![0.0_f64; n];
            vals[k] = 1.0;
            let mut ders = vec![0.0_f64; n];
            for i in 0..n {
                ders[i] = if i == k {
                    self.nodes
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != k)
                        .map(|(_, &xj)| 1.0 / (x - xj))
                        .sum()
                } else {
                    self.bary_w[i] / self.bary_w[k] / (x - self.nodes[i])
                };
            }
            return (vals, ders);
        }
        let vals = self.val(x);
        let n = self.nodes.len();
        let mut ders = vec![0.0_f64; n];
        let mut sum_inv = 0.0_f64;
        for &xj in &self.nodes {
            sum_inv += 1.0 / (x - xj);
        }
        for i in 0..n {
            ders[i] = vals[i] * (sum_inv - 1.0 / (x - self.nodes[i]));
        }
        (vals, ders)
    }

    /// Evaluate values, first derivatives, and second derivatives.
    pub(crate) fn val_d_h(&self, x: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Exact evaluation AT a node (same singularity as in `val_d`), with
        //   l_k''(x_k) = (Σ_{j≠k} 1/(x_k−x_j))² − Σ_{j≠k} 1/(x_k−x_j)²,
        //   l_i''(x_k) = 2·l_i'(x_k)·Σ_{j∉{i,k}} 1/(x_k−x_j)   (i ≠ k).
        if let Some(k) = self.nodes.iter().position(|&xj| (x - xj).abs() < 1e-14) {
            let n = self.nodes.len();
            let mut vals = vec![0.0_f64; n];
            vals[k] = 1.0;
            let mut ders = vec![0.0_f64; n];
            let mut hess = vec![0.0_f64; n];
            for i in 0..n {
                if i == k {
                    let mut s = 0.0_f64;
                    let mut t = 0.0_f64;
                    for (j, &xj) in self.nodes.iter().enumerate() {
                        if j != k {
                            s += 1.0 / (x - xj);
                            t += 1.0 / ((x - xj) * (x - xj));
                        }
                    }
                    ders[i] = s;
                    hess[i] = s * s - t;
                } else {
                    let li = self.bary_w[i] / self.bary_w[k] / (x - self.nodes[i]);
                    let r: f64 = self
                        .nodes
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i && j != k)
                        .map(|(_, &xj)| 1.0 / (x - xj))
                        .sum();
                    ders[i] = li;
                    hess[i] = 2.0 * li * r;
                }
            }
            return (vals, ders, hess);
        }
        let vals = self.val(x);
        let n = self.nodes.len();
        let mut ders = vec![0.0_f64; n];
        let mut hess = vec![0.0_f64; n];
        let mut s = 0.0_f64;
        let mut t = 0.0_f64;
        for &xj in &self.nodes {
            let inv = 1.0 / (x - xj);
            s += inv;
            t += inv * inv;
        }
        for i in 0..n {
            let inv_i = 1.0 / (x - self.nodes[i]);
            ders[i] = vals[i] * (s - inv_i);
            hess[i] = vals[i] * ((s - inv_i) * (s - inv_i) - t + inv_i * inv_i);
        }
        (vals, ders, hess)
    }
}

// ─── QuadPosQk (Bernstein / H1 Positive basis) ─────────────────────────────

/// Binomial coefficient `C(n, k)` as an integer.
fn binom_coeff(n: usize, k: usize) -> usize {
    let k = k.min(n - k);
    let mut c = 1usize;
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}

/// 1D Bernstein basis values at `x ∈ [0,1]` — bit-identical to MFEM's
/// `Poly_1D::CalcBinomTerms(p, x, 1-x, u)` (same accumulation order).
fn bernstein_1d(p: usize, x: f64) -> Vec<f64> {
    let mut u = vec![0.0_f64; p + 1];
    if p == 0 {
        u[0] = 1.0;
        return u;
    }
    let y = 1.0 - x;
    let mut z = x;
    let mut i = 1usize;
    while i < p {
        u[i] = binom_coeff(p, i) as f64 * z;
        z *= x;
        i += 1;
    }
    u[p] = z;
    z = y;
    i -= 1;
    while i > 0 {
        u[i] *= z;
        z *= y;
        i -= 1;
    }
    u[0] = z;
    u
}

/// 1D Bernstein basis values and derivatives — bit-identical to MFEM's
/// `Poly_1D::CalcBinomTerms(p, x, 1-x, u, d)`.
fn bernstein_1d_d(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0_f64; p + 1];
    let mut d = vec![0.0_f64; p + 1];
    if p == 0 {
        u[0] = 1.0;
        d[0] = 0.0;
        return (u, d);
    }
    let y = 1.0 - x;
    let xpy = x + y;
    let ptx = p as f64 * x;
    let mut z = 1.0;
    let mut i = 1usize;
    while i < p {
        d[i] = binom_coeff(p, i) as f64 * z * (i as f64 * xpy - ptx);
        z *= x;
        u[i] = binom_coeff(p, i) as f64 * z;
        i += 1;
    }
    d[p] = p as f64 * z;
    u[p] = z * x;
    z = 1.0;
    i -= 1;
    while i > 0 {
        d[i] *= z;
        z *= y;
        u[i] *= z;
        i -= 1;
    }
    d[0] = -(p as f64) * z;
    u[0] = z * y;
    (u, d)
}

/// Map a lexicographic tensor index `(ix, iy)` to the H1 DOF ordering
/// (vertices → edges → interior), identical to `QuadQk::node_to_dof` and to
/// MFEM's `H1_DOF_MAP`.
fn pos_dof_map(p: usize, ix: usize, iy: usize) -> usize {
    let x = ix as f64 / p as f64;
    let y = iy as f64 / p as f64;
    let tol = 1e-12;
    let on_xmin = x.abs() < tol;
    let on_xmax = (x - 1.0).abs() < tol;
    let on_ymin = y.abs() < tol;
    let on_ymax = (y - 1.0).abs() < tol;
    let on_boundary = on_xmin || on_xmax || on_ymin || on_ymax;
    if on_boundary {
        if on_xmin && on_ymin {
            return 0;
        }
        if on_xmax && on_ymin {
            return 1;
        }
        if on_xmax && on_ymax {
            return 2;
        }
        if on_xmin && on_ymax {
            return 3;
        }
        let mut idx = 4usize;
        if on_ymin {
            return idx + (ix - 1);
        }
        idx += p - 1;
        if on_xmax {
            return idx + (iy - 1);
        }
        idx += p - 1;
        if on_ymax {
            return idx + (p - 1 - ix);
        }
        idx += p - 1;
        if on_xmin {
            return idx + (p - 1 - iy);
        }
        unreachable!()
    } else {
        let base = 4 + 4 * (p - 1);
        base + (iy - 1) * (p - 1) + (ix - 1)
    }
}

/// Arbitrary-order Bernstein (H1 "Positive") element on the reference quad
/// `[0,1]²` — `(p+1)²` DOFs at the **equidistant** nodes `(i/p, j/p)`, ordered
/// by the H1 DOF map (vertices → edges → interior).
///
/// Matches MFEM's `H1Pos_QuadrilateralElement` (`H1_FECollection` with
/// `BasisType::Positive`): the basis functions are the Bernstein polynomials
/// `B_i^p(x)·B_j^p(y)`, and DOFs hold **Bernstein coefficients** (not nodal
/// values) — for `p = 2` the interior basis is `4x(1-x)y(1-y)`, i.e. a quarter
/// of the interior GLL bubble, which is exactly the 16× diagonal difference
/// observed between MFEM's `Positive` elasticity assembly and the standard GLL
/// assembly.
pub struct QuadPosQk {
    order: usize,
}

impl QuadPosQk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        QuadPosQk { order: p }
    }
}

impl ReferenceElement for QuadPosQk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 1) }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let bx = bernstein_1d(p, xi[0]);
        let by = bernstein_1d(p, xi[1]);
        // Lexicographic order (j slowest), reordered by H1 map.
        for j in 0..=p {
            for i in 0..=p {
                values[pos_dof_map(p, i, j)] = bx[i] * by[j];
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.order;
        let (bx, dbx) = bernstein_1d_d(p, xi[0]);
        let (by, dby) = bernstein_1d_d(p, xi[1]);
        for j in 0..=p {
            for i in 0..=p {
                let dof = pos_dof_map(p, i, j);
                grads[dof * 2] = dbx[i] * by[j];
                grads[dof * 2 + 1] = bx[i] * dby[j];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order;
        let n = (p + 1) * (p + 1);
        let mut coords = vec![vec![0.0_f64; 2]; n];
        for j in 0..=p {
            for i in 0..=p {
                let dof = pos_dof_map(p, i, j);
                coords[dof] = vec![i as f64 / p as f64, j as f64 / p as f64];
            }
        }
        coords
    }
}

// ─── QuadQk ──────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference quad `[0,1]²` — `(p+1)²` DOFs.
///
/// Uses Gauss-Lobatto-Legendre (GLL) nodes matching MFEM's `H1_FECollection`
/// with `BasisType::GaussLobatto`.  Basis values and derivatives use the same
/// stable-centre barycentric formula as MFEM's `Poly_1D::Basis::Eval`
/// (Barycentric) **directly on `[0,1]`** — bit-identical to the C++ values
/// (previously evaluated on `[-1,1]` via `ξ = 2·x − 1` with a chain factor,
/// which differed from MFEM by ~1 ulp and propagated into matrix entries).
pub struct QuadQk {
    order: usize,
    lag1d: Lagrange1D,
    /// GLL nodes on `[0,1]` (`0.5·(lag1d.nodes+1)`), used by the MFEM
    /// barycentric evaluation and `node_to_dof`.
    gll01: Vec<f64>,
    /// DOF ordering: `false` = MFEM H1_FECollection topological order
    /// (vertices → edges → interior, see [`QuadQk::node_to_dof`]);
    /// `true` = lexicographic tensor-product order `ix + iy*(p+1)` (x fastest),
    /// which is what MFEM's `DG_FECollection`/`L2_FECollection` use.
    lex: bool,
    /// `true` for [`QuadQk::new_closed_uniform`], whose 1-D nodes live on
    /// `[0,1]` (the element's own reference frame) rather than on `[-1,1]`
    /// (`gll01` is then already the node set).  Only `eval_hessian` needs to
    /// know: it is the one path that evaluates `lag1d` at `2x−1`.
    uniform: bool,
}

impl QuadQk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        let lag1d = Lagrange1D::new(p);
        let gll01 = lag1d.nodes.iter().map(|&x| 0.5 * (x + 1.0)).collect();
        Self {
            order: p,
            lag1d,
            gll01,
            lex: false,
            uniform: false,
        }
    }

    /// The **closed-uniform** (`BasisType::ClosedUniform`) twin of
    /// [`QuadQk::new`]: the same H1 slot order on `[0,1]²`, but the 1-D DOF
    /// nodes are the *equispaced* points `i/p` instead of the Gauss-Lobatto
    /// points.
    ///
    /// This is MFEM's legacy `Quadratic`/`Cubic` `BiQuadratic2DFiniteElement` /
    /// `BiCubic2DFiniteElement` node placement.  `fem-io` uses it (D112) to
    /// reinterpret an MFEM `nodes` grid function written with a *legacy*
    /// `FiniteElementCollection` name (`Linear`/`Quadratic`/`Cubic`) and to
    /// re-interpolate it onto the Gauss-Lobatto nodes the rest of the library
    /// assumes.
    pub fn new_closed_uniform(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        let nodes: Vec<f64> = (0..=p).map(|i| i as f64 / p as f64).collect();
        Self {
            order: p,
            lag1d: Lagrange1D::from_nodes(nodes.clone()),
            gll01: nodes,
            lex: false,
            uniform: true,
        }
    }

    /// QuadQk with **lexicographic** (tensor-product) DOF ordering — matches
    /// MFEM `DG_FECollection`/`L2_FECollection` (per-element DOFs are the GLL
    /// nodes in row-major order `ix + iy*(p+1)`, x fastest).  The default
    /// [`QuadQk::new`] keeps the H1 topological ordering.
    pub fn new_lex(p: usize) -> Self {
        let mut q = QuadQk::new(p);
        q.lex = true;
        q
    }

    /// DOF index for the tensor node `(ix, iy)`.
    fn dof_index(&self, ix: usize, iy: usize) -> usize {
        if self.lex {
            ix + iy * (self.order + 1)
        } else {
            self.node_to_dof(ix, iy)
        }
    }

    /// Map a point `x` on `[0,1]` to `[-1,1]` (legacy path, kept for the
    /// second-derivative evaluation which still uses `Lagrange1D`).
    fn to_std(&self, x: f64) -> f64 {
        2.0 * x - 1.0
    }

    /// Chain-rule factor for first derivatives: d/dx = 2 · d/dξ.
    fn grad_factor(&self) -> f64 {
        2.0
    }

    /// Chain-rule factor for second derivatives: d²/dx² = 4 · d²/dξ².
    fn hess_factor(&self) -> f64 {
        4.0
    }

    /// MFEM `Poly_1D::Basis::Eval` (Barycentric) on `[0,1]`: stable-centre
    /// barycentric Lagrange values and first derivatives at `y`.
    ///
    /// This mirrors MFEM's *value+derivative* overload
    /// (`Eval(y, u, d)`), which computes the values as `u(i) = l·si·w(i)`
    /// with `si = 1/(y−x(i))`.  [`QuadQk::mfem_bary_val`] mirrors the
    /// value-only overload (`Eval(y, u)`) used by `CalcShape`, which instead
    /// divides: `u(i) = l·w(i)/(y−x(i))` — the two differ by 1 ulp and
    /// picking the wrong one breaks bit-identical assembly.
    fn mfem_bary_1d(&self, y: f64) -> (Vec<f64>, Vec<f64>) {
        let p = self.order;
        let n = p + 1;
        let x = &self.gll01;
        // Barycentric weights — MFEM Poly_1D::Basis::Basis(Barycentric)
        // accumulates with the j<i double loop (`w(i) *= xij; w(j) *= -xij`)
        // then takes one reciprocal; matching the exact multiply order keeps
        // the weights bit-identical (a `j != i` full loop permutes the
        // accumulation order and differs by ~1 ulp).
        let mut w = vec![1.0; n];
        for i in 0..n {
            for j in 0..i {
                let xij = x[i] - x[j];
                w[i] *= xij;
                w[j] *= -xij;
            }
        }
        for i in 0..n {
            w[i] = 1.0 / w[i];
        }
        // Stable centre k: lk = ∏ over the nodes on one side of y.
        let mut k = 0usize;
        let mut lk = 1.0;
        while k < p {
            if y >= (x[k] + x[k + 1]) / 2.0 {
                lk *= y - x[k];
                k += 1;
            } else {
                for i in k + 1..=p {
                    lk *= y - x[i];
                }
                break;
            }
        }
        let l = lk * (y - x[k]);
        let mut sk = 0.0;
        let mut u = vec![0.0; n];
        for i in 0..k {
            // MFEM Poly_1D::Basis::Eval(y, u, d) (value+derivative overload)
            // uses the reciprocal-multiplication form `u(i) = l·si·w(i)` with
            // `si = 1/(y−x(i))` (fe_base.cpp:1905) — NOT the division form
            // used by the value-only overload (which `mfem_bary_val` mirrors).
            let si = 1.0 / (y - x[i]);
            sk += si;
            u[i] = l * si * w[i];
        }
        u[k] = lk * w[k];
        for i in k + 1..=p {
            let si = 1.0 / (y - x[i]);
            sk += si;
            u[i] = l * si * w[i];
        }
        let lp = l * sk + lk;
        let mut d = vec![0.0; n];
        for i in 0..k {
            d[i] = (lp * w[i] - u[i]) / (y - x[i]);
        }
        d[k] = sk * u[k];
        for i in k + 1..=p {
            d[i] = (lp * w[i] - u[i]) / (y - x[i]);
        }
        (u, d)
    }

    /// MFEM `Poly_1D::Basis::Eval(y, u)` (value-only overload, used by
    /// `CalcShape`): barycentric Lagrange values with the **division** form
    /// `u(i) = l·w(i)/(y − x(i))` — bit-for-bit different (1 ulp) from the
    /// reciprocal-multiplication form in [`QuadQk::mfem_bary_1d`].
    fn mfem_bary_val(&self, y: f64) -> Vec<f64> {
        let p = self.order;
        let n = p + 1;
        let x = &self.gll01;
        // Barycentric weights — MFEM Poly_1D::Basis::Basis(Barycentric)
        // accumulates with the j<i double loop (`w(i) *= xij; w(j) *= -xij`)
        // then takes one reciprocal; matching the exact multiply order keeps
        // the weights bit-identical.
        let mut w = vec![1.0; n];
        for i in 0..n {
            for j in 0..i {
                let xij = x[i] - x[j];
                w[i] *= xij;
                w[j] *= -xij;
            }
        }
        for i in 0..n {
            w[i] = 1.0 / w[i];
        }
        // Stable centre k (identical to mfem_bary_1d).
        let mut k = 0usize;
        let mut lk = 1.0;
        while k < p {
            if y >= (x[k] + x[k + 1]) / 2.0 {
                lk *= y - x[k];
                k += 1;
            } else {
                for i in k + 1..=p {
                    lk *= y - x[i];
                }
                break;
            }
        }
        let l = lk * (y - x[k]);
        let mut u = vec![0.0; n];
        // MFEM value-only Eval: u(i) = l * w(i) / (y - x(i)).
        for i in 0..k {
            u[i] = l * w[i] / (y - x[i]);
        }
        u[k] = lk * w[k];
        for i in k + 1..=p {
            u[i] = l * w[i] / (y - x[i]);
        }
        u
    }

    fn node_to_dof(&self, ix: usize, iy: usize) -> usize {
        let p = self.order;
        let x = self.gll01[ix]; // [0,1]
        let y = self.gll01[iy];
        let tol = 1e-12;
        let on_xmin = x.abs() < tol;
        let on_xmax = (x - 1.0).abs() < tol;
        let on_ymin = y.abs() < tol;
        let on_ymax = (y - 1.0).abs() < tol;
        let on_boundary = on_xmin || on_xmax || on_ymin || on_ymax;

        if on_boundary {
            if on_xmin && on_ymin {
                return 0;
            }
            if on_xmax && on_ymin {
                return 1;
            }
            if on_xmax && on_ymax {
                return 2;
            }
            if on_xmin && on_ymax {
                return 3;
            }
            let mut idx = 4usize;
            if on_ymin {
                return idx + (ix - 1);
            }
            idx += p - 1;
            if on_xmax {
                return idx + (iy - 1);
            }
            idx += p - 1;
            if on_ymax {
                return idx + (p - 1 - ix);
            }
            idx += p - 1;
            if on_xmin {
                return idx + (p - 1 - iy);
            }
            unreachable!()
        } else {
            let base = 4 + 4 * (p - 1);
            base + (iy - 1) * (p - 1) + (ix - 1)
        }
    }

    fn all_dof_coords(&self) -> Vec<[f64; 2]> {
        let p = self.order;
        let n = (p + 1) * (p + 1);
        let mut coords = vec![[0.0, 0.0]; n];
        for iy in 0..=p {
            for ix in 0..=p {
                let dof = self.dof_index(ix, iy);
                coords[dof] = [self.gll01[ix], self.gll01[iy]];
            }
        }
        coords
    }
}

impl ReferenceElement for QuadQk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 1)
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        // MFEM [0,1] barycentric basis (no chain-rule mapping).  MFEM's
        // `CalcShape` uses the *value-only* `Poly_1D::Basis::Eval` overload
        // (division form) — see `mfem_bary_val`.
        let lx = self.mfem_bary_val(xi[0]);
        let ly = self.mfem_bary_val(xi[1]);
        let p = self.order;
        for iy in 0..=p {
            for ix in 0..=p {
                values[self.dof_index(ix, iy)] = lx[ix] * ly[iy];
            }
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        // MFEM [0,1] barycentric basis; gradients are exact (no chain factor).
        let (lx, dlx) = self.mfem_bary_1d(xi[0]);
        let (ly, dly) = self.mfem_bary_1d(xi[1]);
        let p = self.order;
        for iy in 0..=p {
            for ix in 0..=p {
                let dof = self.dof_index(ix, iy);
                grads[dof * 2] = dlx[ix] * ly[iy];
                grads[dof * 2 + 1] = lx[ix] * dly[iy];
            }
        }
    }
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        // `lag1d` holds the 1-D nodes in the Gauss-Lobatto `[-1,1]` frame for
        // `QuadQk::new` (map xi from `[0,1]`, chain rule d²/dx² = 4·d²/dξ²) and
        // in the element's own `[0,1]` frame for
        // [`QuadQk::new_closed_uniform`] (no map, factor 1).
        let (x, y, fac) = if self.uniform {
            (xi[0], xi[1], 1.0)
        } else {
            (self.to_std(xi[0]), self.to_std(xi[1]), self.hess_factor())
        };
        let (lx, dlx, hlx) = self.lag1d.val_d_h(x);
        let (ly, dly, hly) = self.lag1d.val_d_h(y);
        let p = self.order;
        for iy in 0..=p {
            for ix in 0..=p {
                let dof = self.node_to_dof(ix, iy);
                let base = dof * 4;
                hess[base] = fac * hlx[ix] * ly[iy];
                hess[base + 1] = fac * dlx[ix] * dly[iy];
                hess[base + 2] = hess[base + 1];
                hess[base + 3] = fac * lx[ix] * hly[iy];
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.all_dof_coords().iter().map(|c| c.to_vec()).collect()
    }
}

/// Arbitrary-order L² Lagrange element on `[0,1]²` with **Gauss-Legendre**
/// nodes — `(p+1)²` DOFs, matching MFEM's `L2_FECollection` default
/// `BasisType::GaussLegendre` (used by e.g. `CoefficientRefiner` and L2
/// projection error estimation).  Unlike [`QuadQk`] (Gauss-Lobatto nodes
/// including the boundary), GL nodes are interior-only.
///
/// The 1D basis uses the direct Lagrange formula `Π_{j≠i}(x−x_j)/(x_i−x_j)`
/// on `[0,1]` (no reference-domain mapping), so the values match MFEM's L2
/// element bit-for-bit.
/// Direct 1D Lagrange values through `nodes` at `x`:
/// `l_i(x) = Π_{j≠i} (x−x_j)/(x_i−x_j)`.
///
/// Shared by the Gauss-Legendre L² tensor elements ([`QuadL2GL`], [`HexL2GL`]);
/// MFEM's L2 tensor elements evaluate the same nodal Lagrange product on
/// `[0,1]` via `Poly_1D::Basis`, so the values match at the GL nodes.
fn lagrange_1d_val(nodes: &[f64], x: f64) -> Vec<f64> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let mut v = 1.0;
            for (j, &xj) in nodes.iter().enumerate() {
                if j != i {
                    v *= (x - xj) / (xi - xj);
                }
            }
            v
        })
        .collect()
}

/// Direct 1D Lagrange derivative values through `nodes` at `x`.
fn lagrange_1d_der(nodes: &[f64], x: f64) -> Vec<f64> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let mut s = 0.0;
            for (m, &xm) in nodes.iter().enumerate() {
                if m == i {
                    continue;
                }
                let mut t = 1.0 / (xi - xm);
                for (j, &xj) in nodes.iter().enumerate() {
                    if j != i && j != m {
                        t *= (x - xj) / (xi - xj);
                    }
                }
                s += t;
            }
            s
        })
        .collect()
}

pub struct QuadL2GL {
    order: usize,
    nodes: Vec<f64>, // Gauss-Legendre nodes on [0,1]
}

impl QuadL2GL {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        // Gauss-Legendre nodes on [-1,1] mapped to [0,1]
        let (nodes, _w) = crate::quadrature::gauss_legendre_arbitrary(p + 1);
        let nodes = nodes.iter().map(|x| 0.5 * (x + 1.0)).collect();
        Self { order: p, nodes }
    }

    /// Direct 1D Lagrange values at `x` (on [0,1]).
    fn lag1d_val(&self, x: f64) -> Vec<f64> {
        lagrange_1d_val(&self.nodes, x)
    }

    /// Direct 1D Lagrange derivative values at `x`.
    fn lag1d_der(&self, x: f64) -> Vec<f64> {
        lagrange_1d_der(&self.nodes, x)
    }

    /// Tensor-product DOF ordering: dof = iy*(p+1) + ix (x varies fastest,
    /// matching MFEM's L2 tensor order `for (j) for (i) shape(o++) = sx(i)*sy(j)`
    /// and bit-identical summation).
    fn node_to_dof(&self, ix: usize, iy: usize) -> usize {
        iy * (self.order + 1) + ix
    }

    fn all_dof_coords(&self) -> Vec<[f64; 2]> {
        let p = self.order;
        let n = (p + 1) * (p + 1);
        let mut coords = vec![[0.0, 0.0]; n];
        for ix in 0..=p {
            for iy in 0..=p {
                let dof = self.node_to_dof(ix, iy);
                coords[dof] = [self.nodes[ix], self.nodes[iy]];
            }
        }
        coords
    }

    /// Evaluate the 1D Lagrange basis (values, derivatives) at `t ∈ [0,1]`.
    /// Exposed so assemblers can reproduce MFEM's bit-identical `(dof·l_x)·l_y`
    /// summation order instead of a pre-multiplied tensor basis.
    pub fn eval_1d(&self, t: f64) -> (Vec<f64>, Vec<f64>) {
        (self.lag1d_val(t), self.lag1d_der(t))
    }

    /// Tensor-product DOF index for the `(ix, iy)` node.
    pub fn dof_index(&self, ix: usize, iy: usize) -> usize {
        self.node_to_dof(ix, iy)
    }
}

impl ReferenceElement for QuadL2GL {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 1)
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (lx, ly) = (self.lag1d_val(xi[0]), self.lag1d_val(xi[1]));
        let p = self.order;
        for ix in 0..=p {
            for iy in 0..=p {
                values[self.node_to_dof(ix, iy)] = lx[ix] * ly[iy];
            }
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (lx, dlx) = (self.lag1d_val(xi[0]), self.lag1d_der(xi[0]));
        let (ly, dly) = (self.lag1d_val(xi[1]), self.lag1d_der(xi[1]));
        let p = self.order;
        for ix in 0..=p {
            for iy in 0..=p {
                let dof = self.node_to_dof(ix, iy);
                grads[dof * 2] = dlx[ix] * ly[iy];
                grads[dof * 2 + 1] = lx[ix] * dly[iy];
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.all_dof_coords().iter().map(|c| c.to_vec()).collect()
    }
}

// ─── HexL2GL ─────────────────────────────────────────────────────────────────

/// Arbitrary-order L² Lagrange element on the reference hex `[-1,1]³` with
/// **Gauss-Legendre** nodes — `(p+1)³` DOFs, matching MFEM's
/// `L2_HexahedronElement` (`L2_FECollection`'s default
/// `BasisType::GaussLegendre`): interior-only GL points, lexicographic
/// tensor-product DOF order (`L2_DOF_MAP`: dof = `ix + iy·(p+1) + iz·(p+1)²`,
/// x fastest) — NOT the H1 topological ordering of [`HexQk`].
///
/// MFEM places the GL nodes on `[0,1]³` (`Poly_1D::OpenPoints`, ascending);
/// this element keeps the fem-rs hex convention `[-1,1]³` (same domain as
/// [`HexQ1`]/[`HexQk`] and `hex_rule`), i.e. the MFEM basis composed with the
/// affine map `ξ = 2x−1` — the same polynomial space with identical
/// quadrature/Jacobian treatment in the assembler.  The 1D basis uses the
/// direct Lagrange formula [`lagrange_1d_val`] on the GL nodes (same
/// construction as [`QuadL2GL`]).
pub struct HexL2GL {
    order: usize,
    nodes: Vec<f64>, // Gauss-Legendre nodes on [-1,1], ascending (MFEM order)
}

impl HexL2GL {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        // Gauss-Legendre nodes on [-1,1].  `gauss_legendre_arbitrary` returns
        // descending nodes for n > 4 (Newton from the largest root) and the
        // hard-coded ascending table for n <= 4; MFEM's
        // `QuadratureFunctions1D::GaussLegendre` is always ascending, so sort
        // to make the lexicographic DOF numbering match MFEM's.
        let (mut nodes, _w) = crate::quadrature::gauss_legendre_arbitrary(p + 1);
        if nodes.len() > 1 && nodes[0] > nodes[nodes.len() - 1] {
            nodes.reverse();
        }
        Self { order: p, nodes }
    }

    /// Tensor-product DOF ordering: dof = `ix + iy·(p+1) + iz·(p+1)²`
    /// (x varies fastest), matching MFEM's `L2_HexahedronElement`
    /// `for (k) for (j) for (i) shape(o++) = sx(i)·sy(j)·sz(k)`.
    fn node_to_dof(&self, ix: usize, iy: usize, iz: usize) -> usize {
        let p1 = self.order + 1;
        ix + iy * p1 + iz * p1 * p1
    }

    fn all_dof_coords(&self) -> Vec<[f64; 3]> {
        let p = self.order;
        let n = (p + 1) * (p + 1) * (p + 1);
        let mut coords = vec![[0.0, 0.0, 0.0]; n];
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.node_to_dof(ix, iy, iz);
                    coords[dof] = [self.nodes[ix], self.nodes[iy], self.nodes[iz]];
                }
            }
        }
        coords
    }

    /// Tensor-product DOF index for the `(ix, iy, iz)` node.
    pub fn dof_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        self.node_to_dof(ix, iy, iz)
    }
}

impl ReferenceElement for HexL2GL {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 1) * (self.order + 1)
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (lx, ly, lz) = (
            lagrange_1d_val(&self.nodes, xi[0]),
            lagrange_1d_val(&self.nodes, xi[1]),
            lagrange_1d_val(&self.nodes, xi[2]),
        );
        let p = self.order;
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    values[self.node_to_dof(ix, iy, iz)] = lx[ix] * ly[iy] * lz[iz];
                }
            }
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (lx, ly, lz) = (
            lagrange_1d_val(&self.nodes, xi[0]),
            lagrange_1d_val(&self.nodes, xi[1]),
            lagrange_1d_val(&self.nodes, xi[2]),
        );
        let (dlx, dly, dlz) = (
            lagrange_1d_der(&self.nodes, xi[0]),
            lagrange_1d_der(&self.nodes, xi[1]),
            lagrange_1d_der(&self.nodes, xi[2]),
        );
        let p = self.order;
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.node_to_dof(ix, iy, iz);
                    grads[dof * 3] = dlx[ix] * ly[iy] * lz[iz];
                    grads[dof * 3 + 1] = lx[ix] * dly[iy] * lz[iz];
                    grads[dof * 3 + 2] = lx[ix] * ly[iy] * dlz[iz];
                }
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        // [-1,1]³ tensor GL rule — same reference domain as the hex geometry
        // element, so the assembler's `geom_quad_point` pass-through is exact.
        hex_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.all_dof_coords().iter().map(|c| c.to_vec()).collect()
    }
}

// ─── HexQk ───────────────────────────────────────────────────────────────────

/// MFEM `Geometry::Constants<Geometry::CUBE>::Vertices` reference-cube corners
/// (`{0,1}³`, 0 = low side of the axis): local vertices 0..3 are the bottom
/// ring counter-clockwise, 4..7 the top ring.
const HEX_VERT_SIDES: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// MFEM `Constants<Geometry::CUBE>::Edges`: local edge `k` runs from local
/// vertex `HEX_EDGE_VERTS[k][0]` to `HEX_EDGE_VERTS[k][1]`.  Every entry
/// starts on the low side of its varying axis, so the H1 edge blocks run in
/// ascending tensor-index order (asserted in [`HexQk::node_to_dof`]).
const HEX_EDGE_VERTS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [3, 2],
    [0, 3],
    [4, 5],
    [5, 6],
    [7, 6],
    [4, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// MFEM `Constants<Geometry::CUBE>::FaceVert`: local face `f` lists its four
/// local vertices in canonical face order (reference square
/// `(0,0) → (1,0) → (1,1) → (0,1)`).
const HEX_FACE_VERTS: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

/// Pre-D31 fem-rs H1 slot order for `p == 2`, still in force because
/// `DofManager::build_q2_hex` (`crates/space/src/dof_manager.rs`) hard-codes it
/// in its element-local `EDGES`/`FACES` tables: slot `s` carries the tensor
/// node `LEGACY_P2_SLOTS[s]` (vertices 0..8 in ring order, then the 12 edges of
/// that function's `EDGES` in order, then its 6 `FACES` in order, then the
/// element centre).  Verified against the pre-D31
/// `hex_qk_to_mfem_h1_perm(2)` hard table (`17,18,19,16, 11,9,13,15, 8,10,14,12,
/// 24,22,21,23, 20,25, 26`).  `p >= 3` uses the MFEM order unconditionally;
/// p = 2 joins it as soon as the crates/space table is switched (see
/// `tests::hex_qk_p2_keeps_legacy_slot_order_until_dofmanager_follows`).
const LEGACY_P2_SLOTS: [[usize; 3]; 27] = [
    [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
    [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2],
    [2, 0, 1], [2, 2, 1], [0, 2, 1], [0, 0, 1],
    [0, 1, 0], [2, 1, 0], [2, 1, 2], [0, 1, 2],
    [1, 0, 0], [1, 2, 0], [1, 2, 2], [1, 0, 2],
    [0, 1, 1], [2, 1, 1], [1, 0, 1], [1, 2, 1],
    [1, 1, 0], [1, 1, 2], [1, 1, 1],
];

/// Arbitrary-order Lagrange element on the reference hex `[-1,1]³` — `(p+1)³` DOFs.
pub struct HexQk {
    order: usize,
    lag1d: Lagrange1D,
    /// DOF ordering: `false` = H1 *topological* order (vertices → edges →
    /// faces → interior) in **MFEM `H1_HexahedronElement` order** for
    /// `p >= 3` — see [`HexQk::node_to_dof`], pinned against the C++ node dump
    /// by `tests::hex_qk_dof_coords_match_mfem_node_dump` and
    /// `tests::hex_qk_slots_match_mfem_h1_dof_map`.  `p == 2` is the one order
    /// still on the pre-D31 fem-rs order because
    /// `DofManager::build_q2_hex` (`crates/space`) hard-codes it (see
    /// [`LEGACY_P2_SLOTS`]); the fem-rs global numbering otherwise
    /// (`DofManager::build_pk_hex`, p >= 3) derives its edge/face slot runs
    /// from `HexQk::dof_coords()` and follows this layout automatically.
    ///
    /// `true` = lexicographic tensor-product order `ix + iy·(p+1) + iz·(p+1)²`
    /// (x fastest), which is what MFEM's `DG_FECollection`/`L2_FECollection`
    /// with `BasisType::GaussLobatto` use (`L2_DOF_MAP`).
    lex: bool,
}

impl HexQk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        Self {
            order: p,
            lag1d: Lagrange1D::new(p),
            lex: false,
        }
    }

    /// HexQk with **lexicographic** (tensor-product) DOF ordering — matches
    /// MFEM `L2_FECollection`/`DG_FECollection` with `BasisType::GaussLobatto`
    /// (per-element DOFs are the GLL nodes in row-major order
    /// `ix + iy·(p+1) + iz·(p+1)²`, x fastest).  The default [`HexQk::new`]
    /// keeps the H1 topological ordering.
    pub fn new_lex(p: usize) -> Self {
        let mut h = HexQk::new(p);
        h.lex = true;
        h
    }

    /// DOF index for the tensor node `(ix, iy, iz)`.
    fn dof_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        if self.lex {
            let p1 = self.order + 1;
            ix + iy * p1 + iz * p1 * p1
        } else {
            self.node_to_dof(ix, iy, iz)
        }
    }

    /// H1 *topological* DOF index of the tensor node `(ix, iy, iz)` — the
    /// MFEM `H1_HexahedronElement` slot order (`TensorBasisElement`,
    /// `H1_DOF_MAP`, dims == 3):
    ///
    /// * slots `0..8`: the 8 vertices in `CUBE::Vertices` order;
    /// * then 12 edge blocks of `p-1` slots in `CUBE::Edges` order, each block
    ///   running from its first stored local vertex to the second at the 1-D
    ///   GLL parameters `1..=p-1` (block-internal order never reversed);
    /// * then 6 face blocks of `(p-1)²` in `CUBE::FaceVert` order, block slot
    ///   `(j-1)(p-1) + (i-1)` where `i` runs along the face's first stored
    ///   edge direction and `j` along the second;
    /// * then the `(p-1)³` interior slots with `iz` outermost, `iy` next, `ix`
    ///   fastest.
    ///
    /// This is MFEM 4.9/4.10 bit-for-bit (the `HEX p=2..5` dumps in
    /// `tmp/gll_ref/fe_nodes_cpp.txt` are pinned slot-by-slot by
    /// `tests::hex_qk_dof_coords_match_mfem_node_dump`).  The pre-D31 fem-rs
    /// order differed: it enumerated edge blocks by an internal
    /// `(face_i, face_j)` signature table and faces as
    /// `xmin/xmax/ymin/ymax/zmin/zmax`.  The global numbering in
    /// `DofManager::build_pk_hex` is derived from `HexQk::dof_coords()`, so it
    /// follows this layout automatically.
    fn node_to_dof(&self, ix: usize, iy: usize, iz: usize) -> usize {
        // p == 2 is the one order where the fem-rs global numbering does *not*
        // follow this element: `DofManager::build_q2_hex` (crates/space) has
        // the pre-D31 slot order hard-coded in its `EDGES`/`FACES` tables, so
        // the two must switch together.
        if self.order == 2 {
            return LEGACY_P2_SLOTS
                .iter()
                .position(|s| s == &[ix, iy, iz])
                .expect("hex p=2 slot");
        }
        let p = self.order;
        let e = p.saturating_sub(1); // slots per edge == face row count
        let idx = [ix, iy, iz];
        let on_bnd = [
            ix == 0 || ix == p,
            iy == 0 || iy == p,
            iz == 0 || iz == p,
        ];
        let n_bnd = on_bnd.iter().filter(|&&b| b).count();
        let side = [
            usize::from(ix == p),
            usize::from(iy == p),
            usize::from(iz == p),
        ];

        // Vertex: on all three pairs of boundary planes (this is also every
        // slot of the p == 1 element).
        if n_bnd == 3 {
            return HEX_VERT_SIDES
                .iter()
                .position(|v| v == &side)
                .expect("hex vertex corner");
        }

        // Edge: exactly one free axis.  The edge is the `CUBE::Edges` entry
        // whose endpoints differ along that axis and agree on the other two.
        if n_bnd == 2 {
            let av = (0..3).find(|&d| !on_bnd[d]).expect("hex edge axis");
            let bnd: Vec<usize> = (0..3).filter(|&d| d != av).collect();
            let k = HEX_EDGE_VERTS
                .iter()
                .position(|&[la, lb]| {
                    HEX_VERT_SIDES[la][av] != HEX_VERT_SIDES[lb][av]
                        && HEX_VERT_SIDES[la][bnd[0]] == side[bnd[0]]
                        && HEX_VERT_SIDES[la][bnd[1]] == side[bnd[1]]
                })
                .expect("hex edge");
            // `CUBE::Edges` starts every edge on the low side of its varying
            // axis, so each block runs in ascending tensor-index order.
            debug_assert_eq!(HEX_VERT_SIDES[HEX_EDGE_VERTS[k][0]][av], 0);
            return 8 + k * e + (idx[av] - 1);
        }

        // Face: exactly one boundary axis.  The face is the `FaceVert` row
        // whose four vertices all carry that side.
        if n_bnd == 1 {
            let ax = (0..3).find(|&d| on_bnd[d]).expect("hex face axis");
            let f = HEX_FACE_VERTS
                .iter()
                .position(|fv| fv.iter().all(|&v| HEX_VERT_SIDES[v][ax] == side[ax]))
                .expect("hex face");
            // `(row, col)` = `(j-1, i-1)` of MFEM's `for (j) for (i)` block
            // enumeration, with the (i, j) tensor indices of each face.
            let (row, col) = match f {
                0 => (p - iy - 1, ix - 1),     // z = low:  (i, p-j, 0)
                1 => (iz - 1, ix - 1),         // y = low:  (i, 0, j)
                2 => (iz - 1, iy - 1),         // x = high: (p, i, j)
                3 => (iz - 1, p - ix - 1),     // y = high: (p-i, p, j)
                4 => (iz - 1, p - iy - 1),     // x = low:  (0, p-i, j)
                _ => (iy - 1, ix - 1),         // z = high: (i, j, p)
            };
            return 8 + 12 * e + f * e * e + row * e + col;
        }

        // Interior: `iz` outermost, then `iy`, `ix` fastest.
        let base = 8 + 12 * e + 6 * e * e;
        base + (iz - 1) * e * e + (iy - 1) * e + (ix - 1)
    }

    fn all_dof_coords(&self) -> Vec<[f64; 3]> {
        let p = self.order;
        let n = (p + 1) * (p + 1) * (p + 1);
        let mut coords = vec![[0.0, 0.0, 0.0]; n];
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.dof_index(ix, iy, iz);
                    coords[dof] = [
                        self.lag1d.nodes[ix],
                        self.lag1d.nodes[iy],
                        self.lag1d.nodes[iz],
                    ];
                }
            }
        }
        coords
    }
}

impl ReferenceElement for HexQk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        let p = self.order + 1;
        p * p * p
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (lx, ly, lz) = (
            self.lag1d.val(xi[0]),
            self.lag1d.val(xi[1]),
            self.lag1d.val(xi[2]),
        );
        let p = self.order;
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    values[self.dof_index(ix, iy, iz)] = lx[ix] * ly[iy] * lz[iz];
                }
            }
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (lx, dlx) = self.lag1d.val_d(xi[0]);
        let (ly, dly) = self.lag1d.val_d(xi[1]);
        let (lz, dlz) = self.lag1d.val_d(xi[2]);
        let p = self.order;
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.dof_index(ix, iy, iz);
                    grads[dof * 3] = dlx[ix] * ly[iy] * lz[iz];
                    grads[dof * 3 + 1] = lx[ix] * dly[iy] * lz[iz];
                    grads[dof * 3 + 2] = lx[ix] * ly[iy] * dlz[iz];
                }
            }
        }
    }
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        let (lx, dlx, hlx) = self.lag1d.val_d_h(xi[0]);
        let (ly, dly, hly) = self.lag1d.val_d_h(xi[1]);
        let (lz, dlz, hlz) = self.lag1d.val_d_h(xi[2]);
        let p = self.order;
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.dof_index(ix, iy, iz);
                    let b = dof * 9;
                    hess[b] = hlx[ix] * ly[iy] * lz[iz];
                    hess[b + 1] = dlx[ix] * dly[iy] * lz[iz];
                    hess[b + 2] = dlx[ix] * ly[iy] * dlz[iz];
                    hess[b + 3] = hess[b + 1];
                    hess[b + 4] = lx[ix] * hly[iy] * lz[iz];
                    hess[b + 5] = lx[ix] * dly[iy] * dlz[iz];
                    hess[b + 6] = hess[b + 2];
                    hess[b + 7] = hess[b + 5];
                    hess[b + 8] = lx[ix] * ly[iy] * hlz[iz];
                }
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.all_dof_coords().iter().map(|c| c.to_vec()).collect()
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/// Element type identifier for the factory function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    Seg,
    Tri,
    Tet,
    Quad,
    Hex,
    Prism,
    Pyramid,
    QuadSerendipity,
    HexSerendipity,
}

/// Create a reference element of the given type and order.
pub fn ref_elem(etype: ElemType, order: u8) -> Box<dyn ReferenceElement> {
    match etype {
        ElemType::Seg => Box::new(SegPk::new(order as usize)),
        ElemType::Tri => Box::new(TriPk::new(order as usize)),
        ElemType::Tet => Box::new(TetPk::new(order as usize)),
        ElemType::Quad => Box::new(QuadQk::new(order as usize)),
        ElemType::Hex => Box::new(HexQk::new(order as usize)),
        ElemType::Prism => Box::new(PrismPk::new(order as usize)),
        ElemType::Pyramid => Box::new(PyramidPk::new(order as usize)),
        ElemType::QuadSerendipity => Box::new(QuadSerendipityPk::new(order as usize)),
        ElemType::HexSerendipity => Box::new(HexSerendipityPk::new(order as usize)),
    }
}

pub type LagrangeSegment = SegPk;
pub type LagrangeTriangle = TriPk;
pub type LagrangeTetrahedron = TetPk;
pub type LagrangeQuad = QuadQk;
pub type LagrangeHex = HexQk;
pub type LagrangePrism = PrismPk;
pub type LagrangePyramid = PyramidPk;

/// Number of DOFs for a simplex element of given dimension and order.
pub fn n_dofs_simplex(dim: usize, order: usize) -> usize {
    let mut num = 1usize;
    for i in 1..=dim {
        num = num * (order + i) / i;
    }
    num
}

/// Family of vector-valued reference elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecFamily {
    Nedelec,
    RaviartThomas,
    BrezziDouglasMarini,
    /// NURBS H(div) space (divergence-conforming IGA, requires knot vectors).
    NURBS_HDiv,
    /// NURBS H(curl) space (curl-conforming IGA, requires knot vectors).
    NURBS_HCurl,
}

/// Create a vector-valued reference element by family, type, and order.
pub fn vec_ref_elem(
    family: VecFamily,
    etype: ElemType,
    order: u8,
) -> Box<dyn VectorReferenceElement> {
    let p = order as usize;
    match (family, etype) {
        (VecFamily::Nedelec, ElemType::Tri) => Box::new(crate::nedelec::TriNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Quad) if p == 1 => Box::new(crate::nedelec::QuadNDk::new(1)),
        (VecFamily::Nedelec, ElemType::Quad) => Box::new(crate::nedelec::QuadNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Tet) => Box::new(crate::nedelec::TetNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Hex) => Box::new(crate::nedelec::HexNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Prism) => Box::new(crate::nedelec::PrismNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Pyramid) => Box::new(crate::nedelec::PyraNDk::new(p)),
        (VecFamily::RaviartThomas, ElemType::Tri) if p == 0 => {
            Box::new(crate::raviart_thomas::TriRTk::new(0))
        }
        (VecFamily::RaviartThomas, ElemType::Quad) if p == 0 => {
            Box::new(crate::raviart_thomas::QuadRTk::new(0))
        }
        (VecFamily::RaviartThomas, ElemType::Tet) if p == 0 => {
            Box::new(crate::raviart_thomas::TetRTk::new(0))
        }
        (VecFamily::RaviartThomas, ElemType::Hex) if p == 0 => {
            Box::new(crate::raviart_thomas::HexRTk::new(0))
        }
        (VecFamily::RaviartThomas, ElemType::Tri) => {
            Box::new(crate::raviart_thomas::TriRTk::new(p))
        }
        (VecFamily::RaviartThomas, ElemType::Quad) => {
            Box::new(crate::raviart_thomas::QuadRTk::new(p))
        }
        (VecFamily::RaviartThomas, ElemType::Tet) => {
            Box::new(crate::raviart_thomas::TetRTk::new(p))
        }
        (VecFamily::RaviartThomas, ElemType::Hex) => {
            Box::new(crate::raviart_thomas::HexRTk::new(p))
        }
        (VecFamily::RaviartThomas, ElemType::Prism) => {
            Box::new(crate::raviart_thomas::PrismRTk::new(p))
        }
        (VecFamily::RaviartThomas, ElemType::Pyramid) => {
            Box::new(crate::raviart_thomas::PyraRTk::new(p))
        }
        (VecFamily::BrezziDouglasMarini, ElemType::Tri) => {
            Box::new(crate::brezzi_douglas_marini::TriBDMk::new(p))
        }
        (VecFamily::BrezziDouglasMarini, ElemType::Quad) => {
            Box::new(crate::brezzi_douglas_marini::QuadBDMk::new(p))
        }
        (VecFamily::BrezziDouglasMarini, ElemType::Tet) => {
            Box::new(crate::brezzi_douglas_marini::TetBDMk::new(p))
        }
        (VecFamily::BrezziDouglasMarini, ElemType::Hex) => {
            Box::new(crate::brezzi_douglas_marini::HexBDMk::new(p))
        }
        _ => panic!("vec_ref_elem: unsupported (family={family:?}, type={etype:?})"),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// MFEM `H1_HexahedronElement` slot order expressed as *tensor indices*:
    /// entry `s` holds the `(ix, iy, iz)` GLL tensor node that MFEM's
    /// `H1_DOF_MAP` puts into slot `s`.  Transcribed from
    /// `TensorBasisElement::TensorBasisElement` (`fem/fe/fe_base.cpp`,
    /// dims == 3) using `Constants<Geometry::CUBE>::Vertices`/`Edges`/
    /// `FaceVert` — vertices 0..3 bottom ring, 4..7 top ring; edges in
    /// `CUBE::Edges` order, each block running from its first stored vertex to
    /// the second; faces in `FaceVert` order with `(j)` outer and `(i)` inner;
    /// interior with `(k)` outer, `(i)` fastest.
    ///
    /// This is the reference for
    /// [`tests::hex_qk_slots_match_mfem_h1_dof_map`]; the numeric ground truth
    /// is the C++ dump in [`tests::hex_qk_dof_coords_match_mfem_node_dump`].
    fn mfem_h1_hex_slot_nodes(p: usize) -> Vec<(usize, usize, usize)> {
        let mut m: Vec<(usize, usize, usize)> = Vec::with_capacity((p + 1) * (p + 1) * (p + 1));
        // vertices: bottom ring 0..3, top ring 4..7
        m.extend([
            (0, 0, 0), (p, 0, 0), (p, p, 0), (0, p, 0),
            (0, 0, p), (p, 0, p), (p, p, p), (0, p, p),
        ]);
        // edges: (0,1) (1,2) (3,2) (0,3) (4,5) (5,6) (7,6) (4,7)
        //        (0,4) (1,5) (2,6) (3,7)
        for i in 1..p { m.push((i, 0, 0)); }
        for i in 1..p { m.push((p, i, 0)); }
        for i in 1..p { m.push((i, p, 0)); }
        for i in 1..p { m.push((0, i, 0)); }
        for i in 1..p { m.push((i, 0, p)); }
        for i in 1..p { m.push((p, i, p)); }
        for i in 1..p { m.push((i, p, p)); }
        for i in 1..p { m.push((0, i, p)); }
        for i in 1..p { m.push((0, 0, i)); }
        for i in 1..p { m.push((p, 0, i)); }
        for i in 1..p { m.push((p, p, i)); }
        for i in 1..p { m.push((0, p, i)); }
        // faces: (3,2,1,0) (0,1,5,4) (1,2,6,5) (2,3,7,6) (3,0,4,7) (4,5,6,7)
        for j in 1..p { for i in 1..p { m.push((i, p - j, 0)); } }
        for j in 1..p { for i in 1..p { m.push((i, 0, j)); } }
        for j in 1..p { for i in 1..p { m.push((p, i, j)); } }
        for j in 1..p { for i in 1..p { m.push((p - i, p, j)); } }
        for j in 1..p { for i in 1..p { m.push((0, p - i, j)); } }
        for j in 1..p { for i in 1..p { m.push((i, j, p)); } }
        // interior
        for k in 1..p {
            for j in 1..p {
                for i in 1..p { m.push((i, j, k)); }
            }
        }
        m
    }

    /// D31 pin: `HexQk`'s H1 slot order **is** MFEM's
    /// `H1_HexahedronElement` order, i.e. slot `s` of `dof_coords()` is the
    /// tensor node [`mfem_h1_hex_slot_nodes`] assigns to `s`, and
    /// `dof_index` inverts it.  Checked for p = 1 and p = 3..=5 (p = 2 is the
    /// documented legacy exception, see
    /// [`hex_qk_p2_keeps_legacy_slot_order_until_dofmanager_follows`]), plus
    /// bijectivity.
    #[test]
    fn hex_qk_slots_match_mfem_h1_dof_map() {
        for p in [1usize, 3, 4, 5] {
            let hex = HexQk::new(p);
            let n = (p + 1) * (p + 1) * (p + 1);
            let mfem = mfem_h1_hex_slot_nodes(p);
            assert_eq!(mfem.len(), n, "p={p}: MFEM slot count");
            assert_eq!(hex.n_dofs(), n, "p={p}: HexQk dof count");

            // The slot table is a bijection of the tensor grid …
            let mut seen: Vec<usize> = mfem
                .iter()
                .map(|&(i, j, k)| i + j * (p + 1) + k * (p + 1) * (p + 1))
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "p={p}: slot table is a bijection");

            // … and `dof_index` is exactly its inverse.
            for (slot, &(i, j, k)) in mfem.iter().enumerate() {
                assert_eq!(
                    hex.dof_index(i, j, k),
                    slot,
                    "p={p}: HexQk tensor node ({i},{j},{k}) must be MFEM slot {slot}"
                );
            }

            // `dof_coords()` carries the GLL tensor node of the slot's own
            // tensor index (HexQk lives on [-1,1]³, MFEM on [0,1]³).
            let coords = hex.dof_coords();
            for (slot, &(i, j, k)) in mfem.iter().enumerate() {
                let want = [
                    hex.lag1d.nodes[i],
                    hex.lag1d.nodes[j],
                    hex.lag1d.nodes[k],
                ];
                for d in 0..3 {
                    assert!(
                        (coords[slot][d] - want[d]).abs() < 1e-15,
                        "p={p}: slot {slot} coord {d} = {} != GLL tensor node {want:?}",
                        coords[slot][d]
                    );
                }
            }
        }
    }

    /// D31 numeric ground truth: the per-slot node coordinates of MFEM 4.9/4.10
    /// `H1_HexahedronElement` on `[0,1]³`, dumped from C++ (`node slot N xyz
    /// …` in `tmp/gll_ref/fe_nodes_cpp.txt`, harness
    /// `tmp/gll_ref/probe_fe_nodes.cpp`).  `HexQk::dof_coords()` mapped to
    /// `[0,1]³` must reproduce them slot by slot — this pins the *whole*
    /// layout (vertex/edge/face/interior block order and every block-internal
    /// offset), not just its block structure.  p = 3 is used because it is the
    /// lowest order that distinguishes the block-internal offsets (two slots
    /// per edge, a 2×2 face block); p = 2's MFEM dump is reproduced by
    /// `HexQk::new_lex(2)`-independent data once the p = 2 exception is lifted.
    #[test]
    fn hex_qk_dof_coords_match_mfem_node_dump() {
        const HEX_P2: [[f64; 3]; 27] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5],
        ];
        const HEX_P3: [[f64; 3]; 64] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.27639320225002106, 0.0, 0.0],
            [0.72360679774997894, 0.0, 0.0],
            [1.0, 0.27639320225002106, 0.0],
            [1.0, 0.72360679774997894, 0.0],
            [0.27639320225002106, 1.0, 0.0],
            [0.72360679774997894, 1.0, 0.0],
            [0.0, 0.27639320225002106, 0.0],
            [0.0, 0.72360679774997894, 0.0],
            [0.27639320225002106, 0.0, 1.0],
            [0.72360679774997894, 0.0, 1.0],
            [1.0, 0.27639320225002106, 1.0],
            [1.0, 0.72360679774997894, 1.0],
            [0.27639320225002106, 1.0, 1.0],
            [0.72360679774997894, 1.0, 1.0],
            [0.0, 0.27639320225002106, 1.0],
            [0.0, 0.72360679774997894, 1.0],
            [0.0, 0.0, 0.27639320225002106],
            [0.0, 0.0, 0.72360679774997894],
            [1.0, 0.0, 0.27639320225002106],
            [1.0, 0.0, 0.72360679774997894],
            [1.0, 1.0, 0.27639320225002106],
            [1.0, 1.0, 0.72360679774997894],
            [0.0, 1.0, 0.27639320225002106],
            [0.0, 1.0, 0.72360679774997894],
            [0.27639320225002106, 0.72360679774997894, 0.0],
            [0.72360679774997894, 0.72360679774997894, 0.0],
            [0.27639320225002106, 0.27639320225002106, 0.0],
            [0.72360679774997894, 0.27639320225002106, 0.0],
            [0.27639320225002106, 0.0, 0.27639320225002106],
            [0.72360679774997894, 0.0, 0.27639320225002106],
            [0.27639320225002106, 0.0, 0.72360679774997894],
            [0.72360679774997894, 0.0, 0.72360679774997894],
            [1.0, 0.27639320225002106, 0.27639320225002106],
            [1.0, 0.72360679774997894, 0.27639320225002106],
            [1.0, 0.27639320225002106, 0.72360679774997894],
            [1.0, 0.72360679774997894, 0.72360679774997894],
            [0.72360679774997894, 1.0, 0.27639320225002106],
            [0.27639320225002106, 1.0, 0.27639320225002106],
            [0.72360679774997894, 1.0, 0.72360679774997894],
            [0.27639320225002106, 1.0, 0.72360679774997894],
            [0.0, 0.72360679774997894, 0.27639320225002106],
            [0.0, 0.27639320225002106, 0.27639320225002106],
            [0.0, 0.72360679774997894, 0.72360679774997894],
            [0.0, 0.27639320225002106, 0.72360679774997894],
            [0.27639320225002106, 0.27639320225002106, 1.0],
            [0.72360679774997894, 0.27639320225002106, 1.0],
            [0.27639320225002106, 0.72360679774997894, 1.0],
            [0.72360679774997894, 0.72360679774997894, 1.0],
            [0.27639320225002106, 0.27639320225002106, 0.27639320225002106],
            [0.72360679774997894, 0.27639320225002106, 0.27639320225002106],
            [0.27639320225002106, 0.72360679774997894, 0.27639320225002106],
            [0.72360679774997894, 0.72360679774997894, 0.27639320225002106],
            [0.27639320225002106, 0.27639320225002106, 0.72360679774997894],
            [0.72360679774997894, 0.27639320225002106, 0.72360679774997894],
            [0.27639320225002106, 0.72360679774997894, 0.72360679774997894],
            [0.72360679774997894, 0.72360679774997894, 0.72360679774997894],
        ];

        let p = 3usize;
        let hex = HexQk::new(p);
        let coords = hex.dof_coords();
        assert_eq!(coords.len(), HEX_P3.len(), "p={p}: slot count");
        let mut max_err = 0.0_f64;
        for (slot, want) in HEX_P3.iter().enumerate() {
            for d in 0..3 {
                let got = 0.5 * (coords[slot][d] + 1.0); // [-1,1] → [0,1]
                max_err = max_err.max((got - want[d]).abs());
                assert!(
                    (got - want[d]).abs() < 1e-15,
                    "p={p} slot {slot} coord {d}: got {got} want {} (C++ dump)",
                    want[d]
                );
            }
        }
        assert!(max_err <= 1e-16, "p={p}: dump match must be bit-exact");

        // p = 2 is the documented legacy exception.  Its slot order is pinned
        // against the same C++ dump through the pre-D31 permutation table
        // (`hex_qk_to_mfem_h1_perm(p=2)`, itself taken from the old test):
        // legacy slot h holds the tensor node MFEM puts in slot PERM[h].
        const PERM_P2: [usize; 27] = [
            0, 1, 2, 3, 4, 5, 6, 7,
            17, 18, 19, 16,
            11, 9, 13, 15,
            8, 10, 14, 12,
            24, 22, 21, 23,
            20, 25,
            26,
        ];
        let hex2 = HexQk::new(2);
        let coords2 = hex2.dof_coords();
        assert_eq!(coords2.len(), HEX_P2.len());
        for (slot, &mfem_slot) in PERM_P2.iter().enumerate() {
            for d in 0..3 {
                let got = 0.5 * (coords2[slot][d] + 1.0);
                assert!(
                    (got - HEX_P2[mfem_slot][d]).abs() < 1e-15,
                    "p=2 legacy slot {slot} coord {d}: got {got} want {} (MFEM slot {mfem_slot})",
                    HEX_P2[mfem_slot][d]
                );
            }
        }
        // … and it is *not* the MFEM order, i.e. the exception is real and the
        // `DofManager::build_q2_hex` fix is still outstanding.  Once that fix
        // lands this assertion (and `LEGACY_P2_SLOTS`) must be deleted and p=2
        // folded into the loop above.
        assert!(PERM_P2.iter().copied().ne(0..27usize));
    }

    /// D100 pin: `HexQk::eval_basis` and `HexQk::dof_coords` are both derived
    /// from `dof_index`, so slot `k`'s basis function must be the Kronecker
    /// delta at `dof_coords()[k]` — for **every** slot, including slot 0 (the
    /// reference corner `(-1,-1,-1)`).  This is the element-side half of the
    /// "Hex8-P2 first basis function" investigation: the anomaly reported in
    /// round 26 (`2x+3y+5z` projected on `H1Space<Mesh<3>>` Hex8-P2 returning
    /// `6.3e-15` at dof 0 instead of an exact `0`) is **not** a coordinate or
    /// basis defect — the L² projection of any P2-representable function is
    /// accurate to ~1e-13 *absolute* at all 27 dofs, and dof 0's error merely
    /// looks large because its exact value is zero.
    #[test]
    fn hex_q2_basis_is_kronecker_at_dof_coords() {
        let e = HexQk::new(2);
        let coords = e.dof_coords();
        assert_eq!(coords.len(), 27);
        assert_eq!(coords[0], vec![-1.0, -1.0, -1.0], "slot 0 = reference corner");
        // The legacy p=2 slot order (`LEGACY_P2_SLOTS`) must still cover all 27
        // tensor nodes exactly once.
        let mut nodes: Vec<[i32; 3]> = coords
            .iter()
            .map(|c| [c[0] as i32, c[1] as i32, c[2] as i32])
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes.len(), 27, "p=2 slots must be a bijection");
        for (k, x) in coords.iter().enumerate() {
            let mut phi = vec![0.0_f64; 27];
            e.eval_basis(x, &mut phi);
            for (j, &v) in phi.iter().enumerate() {
                let want = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (v - want).abs() < 1e-14,
                    "slot {k} at node {x:?}: phi[{j}] = {v} (want {want})"
                );
            }
        }
    }

    fn check_pou(elem: &dyn ReferenceElement) {
        let order = elem.order() as usize;
        let rule = elem.quadrature((2 * order as u8 + 2).min(15));
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "POU failed for dim={} p={order} at {:?}: sum={s}",
                elem.dim(),
                pt
            );
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let order = elem.order() as usize;
        let rule = elem.quadrature((2 * order as u8 + 2).min(15));
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(
                    s.abs() < 1e-10,
                    "grad sum d={d} = {s} for dim={} p={order}",
                    elem.dim()
                );
            }
        }
    }

    fn check_nodal_interp(elem: &dyn ReferenceElement) {
        let coords = elem.dof_coords();
        let n = elem.n_dofs();
        let mut phi = vec![0.0_f64; n];
        for (i, coord) in coords.iter().enumerate() {
            elem.eval_basis(coord, &mut phi);
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-10,
                    "nodal interp: node {i}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    // ── SegPk ─────────────────────────────────────────────────────────────

    #[test]
    fn seg_pk_pou() {
        for p in 1..=5 {
            check_pou(&SegPk::new(p));
        }
    }
    #[test]
    fn seg_pk_grad_zero() {
        for p in 1..=5 {
            check_grad_zero(&SegPk::new(p));
        }
    }
    #[test]
    fn seg_pk_nodal_interp() {
        for p in 1..=5 {
            check_nodal_interp(&SegPk::new(p));
        }
    }

    // ── TriPk ─────────────────────────────────────────────────────────────
    // Note: gradient_fd tests p=1..=5 (direct Lagrange formula, no Vandermonde).
    // grad_zero uses &dyn ReferenceElement trait object which may overflow
    // Windows debug stack when called in a loop; tested at single order.
    // p=5 passes gradient_fd with 1e-5 tolerance on all DOFs.

    #[test]
    fn tri_pk_pou() {
        check_pou(&TriPk::new(3));
    }
    #[test]
    fn tri_pk_grad_zero() {
        check_grad_zero(&TriPk::new(3));
    }
    #[test]
    fn tri_pk_nodal_interp() {
        check_nodal_interp(&TriPk::new(3));
    }

    // ── TetPk ─────────────────────────────────────────────────────────────

    #[test]
    fn tet_pk_pou() {
        check_pou(&TetPk::new(3));
    }
    #[test]
    fn tet_pk_grad_zero() {
        check_grad_zero(&TetPk::new(3));
    }
    #[test]
    fn tet_pk_nodal_interp() {
        check_nodal_interp(&TetPk::new(3));
    }
    #[test]
    fn tet_pk_n_dofs() {
        for p in 1..=8 {
            assert_eq!(TetPk::new(p).n_dofs(), (p + 1) * (p + 2) * (p + 3) / 6);
        }
    }

    #[test]
    fn tet_pk_matches_p1() {
        use crate::lagrange::TetP1;
        let pk = TetPk::new(1);
        let n = 4;
        let mut v1 = vec![0.0; n];
        let mut v2 = vec![0.0; n];
        for &(x, y, z) in &[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.25, 0.25, 0.25),
        ] {
            pk.eval_basis(&[x, y, z], &mut v1);
            TetP1.eval_basis(&[x, y, z], &mut v2);
            for i in 0..n {
                assert!((v1[i] - v2[i]).abs() < 1e-13, "tet p=1 ({x},{y},{z}) i={i}");
            }
        }
    }

    #[test]
    fn tet_pk_matches_p2() {
        use crate::lagrange::TetP2;
        let pk = TetPk::new(2);
        let n = 10;
        let mut v1 = vec![0.0; n];
        let mut v2 = vec![0.0; n];
        for &(x, y, z) in &[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.2, 0.3, 0.1),
        ] {
            pk.eval_basis(&[x, y, z], &mut v1);
            TetP2.eval_basis(&[x, y, z], &mut v2);
            for i in 0..n {
                assert!((v1[i] - v2[i]).abs() < 1e-12, "tet p=2 ({x},{y},{z}) i={i}");
            }
        }
    }

    // ── QuadQk ────────────────────────────────────────────────────────────

    #[test]
    fn quad_qk_pou() {
        for p in 1..=6 {
            check_pou(&QuadQk::new(p));
        }
    }
    #[test]
    fn quad_qk_grad_zero() {
        for p in 1..=6 {
            check_grad_zero(&QuadQk::new(p));
        }
    }
    #[test]
    fn quad_qk_nodal_interp() {
        for p in 1..=6 {
            check_nodal_interp(&QuadQk::new(p));
        }
    }
    #[test]
    fn quad_qk_n_dofs() {
        for p in 1..=8 {
            assert_eq!(QuadQk::new(p).n_dofs(), (p + 1) * (p + 1));
        }
    }

    #[test]
    fn quad_qk_matches_q1() {
        use crate::lagrange::QuadQ1;
        let qk = QuadQk::new(1);
        let n = 4;
        let mut v1 = vec![0.0; n];
        let mut v2 = vec![0.0; n];
        // QuadQk uses [0,1]²; QuadQ1 uses [-1,1]².
        // φ_QuadQk(x,y) = φ_QuadQ1(2x-1, 2y-1)
        for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.65, 0.25)] {
            qk.eval_basis(&[x, y], &mut v1);
            QuadQ1.eval_basis(&[2.0 * x - 1.0, 2.0 * y - 1.0], &mut v2);
            for i in 0..n {
                assert!((v1[i] - v2[i]).abs() < 1e-13, "Q1 ({x},{y}) i={i}");
            }
        }
    }

    #[test]
    fn quad_qk_matches_q2() {
        use crate::lagrange::QuadQ2;
        let qk = QuadQk::new(2);
        let n = 9;
        let mut v1 = vec![0.0; n];
        let mut v2 = vec![0.0; n];
        // QuadQk uses [0,1]²; QuadQ2 uses [-1,1]².
        // φ_QuadQk(x,y) = φ_QuadQ2(2x-1, 2y-1)
        for &(x, y) in &[
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 1.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (0.65, 0.25),
        ] {
            qk.eval_basis(&[x, y], &mut v1);
            QuadQ2.eval_basis(&[2.0 * x - 1.0, 2.0 * y - 1.0], &mut v2);
            for i in 0..n {
                assert!((v1[i] - v2[i]).abs() < 1e-12, "Q2 ({x},{y}) i={i}");
            }
        }
    }

    // ── HexQk ─────────────────────────────────────────────────────────────

    #[test]
    fn hex_qk_pou() {
        for p in 1..=4 {
            check_pou(&HexQk::new(p));
        }
    }
    #[test]
    fn hex_qk_grad_zero() {
        for p in 1..=4 {
            check_grad_zero(&HexQk::new(p));
        }
    }
    #[test]
    fn hex_qk_nodal_interp() {
        for p in 1..=4 {
            check_nodal_interp(&HexQk::new(p));
        }
    }
    #[test]
    fn hex_qk_n_dofs() {
        for p in 1..=6 {
            let pp = p + 1;
            assert_eq!(HexQk::new(p).n_dofs(), pp * pp * pp);
        }
    }

    #[test]
    fn hex_qk_matches_q1() {
        use crate::lagrange::HexQ1;
        let qk = HexQk::new(1);
        let n = 8;
        let mut v1 = vec![0.0; n];
        let mut v2 = vec![0.0; n];
        for &(x, y, z) in &[
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (0.3, -0.5, 0.7),
        ] {
            qk.eval_basis(&[x, y, z], &mut v1);
            HexQ1.eval_basis(&[x, y, z], &mut v2);
            for i in 0..n {
                assert!((v1[i] - v2[i]).abs() < 1e-13, "H1 ({x},{y},{z}) i={i}");
            }
        }
    }

    // ── HexL2GL ───────────────────────────────────────────────────────────

    #[test]
    fn hex_l2gl_n_dofs() {
        for p in 1..=8 {
            assert_eq!(HexL2GL::new(p).n_dofs(), (p + 1) * (p + 1) * (p + 1));
        }
    }

    #[test]
    fn hex_l2gl_pou() {
        for p in 1..=4 {
            check_pou(&HexL2GL::new(p));
        }
    }

    #[test]
    fn hex_l2gl_grad_zero() {
        for p in 1..=3 {
            check_grad_zero(&HexL2GL::new(p));
        }
    }

    #[test]
    fn hex_l2gl_nodal_interp() {
        for p in 1..=4 {
            check_nodal_interp(&HexL2GL::new(p));
        }
    }

    /// The GL nodes must be ascending (MFEM `QuadratureFunctions1D::
    /// GaussLegendre` order) so the lexicographic DOF numbering matches
    /// MFEM's `L2_HexahedronElement` `L2_DOF_MAP`.
    #[test]
    fn hex_l2gl_nodes_ascending() {
        for p in 1..=8 {
            let (nodes, w) = crate::quadrature::gauss_legendre_arbitrary(p + 1);
            let mut srt = nodes.clone();
            srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(nodes, srt, "p={p}: HexL2GL node order must be ascending");
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-12);
        }
    }

    /// `HexL2GL` dof 0 sits at the (min, min, min) GL corner node and dof
    /// `(p+1)² − 1` completes the first z-layer (x fastest), mirroring MFEM's
    /// `for (k) for (j) for (i) shape(o++) = sx(i)·sy(j)·sz(k)`.
    #[test]
    fn hex_l2gl_lex_layout() {
        let p = 3usize;
        let el = HexL2GL::new(p);
        let coords = el.dof_coords();
        // dof index = ix + iy*(p+1) + iz*(p+1)^2
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let o = el.dof_index(ix, iy, iz);
                    assert_eq!(coords[o][0], coords[ix][0]);
                    assert_eq!(coords[o][1], coords[iy * (p + 1)][1]);
                    assert_eq!(coords[o][2], coords[iz * (p + 1) * (p + 1)][2]);
                }
            }
        }
    }

    /// `HexQk::new_lex` reorders the GLL tensor nodes lexicographically; the
    /// nodal interpolation property must hold in the lex layout too.
    #[test]
    fn hex_qk_lex_nodal_interp() {
        for p in 1..=3 {
            check_nodal_interp(&HexQk::new_lex(p));
        }
    }

    #[test]
    fn hex_qk_lex_coords() {
        // p=1: GLL nodes are the corners; lex order must be
        // (−,−,−),(+,−,−),(−,+,−),(+,+,−),(−,−,+),(+,−,+),(−,+,+),(+,+,+).
        let lex = HexQk::new_lex(1);
        let coords = lex.dof_coords();
        let expected: Vec<Vec<f64>> = [(-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
            .iter()
            .map(|&(x, y, z)| vec![x, y, z])
            .collect();
        assert_eq!(coords, expected);
        // Same set of nodes as the default (H1-ordered) HexQk(1).
        let h1 = HexQk::new(1);
        let mut a = lex.dof_coords();
        let mut b = h1.dof_coords();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b);
    }

    // ── Factory ───────────────────────────────────────────────────────────

    #[test]
    fn ref_elem_factory() {
        assert_eq!(ref_elem(ElemType::Seg, 5).n_dofs(), 6);
        assert_eq!(ref_elem(ElemType::Tri, 4).n_dofs(), 15);
        assert_eq!(ref_elem(ElemType::Tet, 3).n_dofs(), 20);
        assert_eq!(ref_elem(ElemType::Quad, 5).n_dofs(), 36);
        assert_eq!(ref_elem(ElemType::Hex, 3).n_dofs(), 64);
        assert_eq!(ref_elem(ElemType::Prism, 1).n_dofs(), 6);
        assert_eq!(ref_elem(ElemType::Prism, 2).n_dofs(), 18);
        assert_eq!(ref_elem(ElemType::Pyramid, 1).n_dofs(), 5);
        assert_eq!(ref_elem(ElemType::Pyramid, 2).n_dofs(), 14);
    }

    #[test]
    fn n_dofs_simplex_formula() {
        assert_eq!(n_dofs_simplex(1, 1), 2);
        assert_eq!(n_dofs_simplex(1, 5), 6);
        assert_eq!(n_dofs_simplex(2, 1), 3);
        assert_eq!(n_dofs_simplex(2, 3), 10);
        assert_eq!(n_dofs_simplex(2, 6), 28);
        assert_eq!(n_dofs_simplex(3, 1), 4);
        assert_eq!(n_dofs_simplex(3, 3), 20);
        assert_eq!(n_dofs_simplex(3, 5), 56);
    }

    // ── Gradient FD checks ────────────────────────────────────────────────

    #[test]
    fn seg_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=4 {
            // Monomial Vandermonde conditioning degrades for p > 5
            let elem = SegPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut grads) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for &x in &[0.1, 0.5, 0.9] {
                elem.eval_basis(&[x], &mut vc);
                elem.eval_basis(&[x + h], &mut vx);
                elem.eval_grad_basis(&[x], &mut grads);
                for i in 0..n {
                    let fd = (vx[i] - vc[i]) / h;
                    assert!((grads[i] - fd).abs() < 1e-4, "p={p} x={x} i={i}");
                }
            }
        }
    }

    #[test]
    fn tri_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=5 {
            let elem = TriPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut grads) =
                (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n * 2]);
            for &(x, y) in &[(0.2, 0.3), (0.5, 0.2), (1.0 / 3.0, 1.0 / 3.0)] {
                elem.eval_basis(&[x, y], &mut vc);
                elem.eval_basis(&[x + h, y], &mut vx);
                elem.eval_basis(&[x, y + h], &mut vy);
                elem.eval_grad_basis(&[x, y], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    assert!(
                        (grads[i * 2] - fd_x).abs() < 1e-5,
                        "p={p} ({x},{y}) i={i} gx: analytic={} fd={}",
                        grads[i * 2],
                        fd_x
                    );
                    assert!(
                        (grads[i * 2 + 1] - fd_y).abs() < 1e-5,
                        "p={p} ({x},{y}) i={i} gy: analytic={} fd={}",
                        grads[i * 2 + 1],
                        fd_y
                    );
                }
            }
        }
    }

    #[test]
    fn tet_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=5 {
            let elem = TetPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * 3],
            );
            for &(x, y, z) in &[(0.15, 0.2, 0.25), (0.3, 0.3, 0.1)] {
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x + h, y, z], &mut vx);
                elem.eval_basis(&[x, y + h, z], &mut vy);
                elem.eval_basis(&[x, y, z + h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!(
                        (grads[i * 3] - fd_x).abs() < 1e-5,
                        "p={p} ({x},{y},{z}) i={i} gx"
                    );
                    assert!(
                        (grads[i * 3 + 1] - fd_y).abs() < 1e-5,
                        "p={p} ({x},{y},{z}) i={i} gy"
                    );
                    assert!(
                        (grads[i * 3 + 2] - fd_z).abs() < 1e-5,
                        "p={p} ({x},{y},{z}) i={i} gz"
                    );
                }
            }
        }
    }

    // ── PrismPk ────────────────────────────────────────────────────────────

    #[test]
    fn prism_pk_pou() {
        check_pou(&PrismPk::new(2));
    }
    #[test]
    fn prism_pk_grad_zero() {
        check_grad_zero(&PrismPk::new(2));
    }
    #[test]
    fn prism_pk_nodal_interp() {
        check_nodal_interp(&PrismPk::new(2));
    }
    #[test]
    fn prism_pk_n_dofs() {
        for p in 1..=5 {
            let n_tri = (p + 1) * (p + 2) / 2;
            assert_eq!(PrismPk::new(p).n_dofs(), (p + 1) * n_tri);
        }
    }

    #[test]
    fn prism_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=3 {
            let elem = PrismPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * 3],
            );
            let test_pts: &[[f64; 3]] = if p == 1 {
                &[[0.3, 0.2, 0.1]]
            } else {
                &[[0.2, 0.3, 0.15], [0.5, 0.1, 0.25]]
            };
            for pt in test_pts {
                let (x, y, z) = (pt[0], pt[1], pt[2]);
                if y + z > 0.95 {
                    continue;
                }
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x + h, y, z], &mut vx);
                elem.eval_basis(&[x, y + h, z], &mut vy);
                elem.eval_basis(&[x, y, z + h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!(
                        (grads[i * 3] - fd_x).abs() < 1e-5,
                        "p={p} ({x},{y},{z}) i={i} gx"
                    );
                    assert!(
                        (grads[i * 3 + 1] - fd_y).abs() < 1e-5,
                        "p={p} ({x},{y},{z}) i={i} gy"
                    );
                    assert!(
                        (grads[i * 3 + 2] - fd_z).abs() < 1e-5,
                        "p={p} ({x},{y},{z}) i={i} gz"
                    );
                }
            }
        }
    }

    // ── PyramidPk ──────────────────────────────────────────────────────────

    #[test]
    fn pyramid_pk_pou() {
        check_pou(&PyramidPk::new(2));
    }
    #[test]
    fn pyramid_pk_grad_zero() {
        check_grad_zero(&PyramidPk::new(2));
    }
    #[test]
    fn pyramid_pk_nodal_interp() {
        check_nodal_interp(&PyramidPk::new(2));
    }
    #[test]
    fn pyramid_pk_n_dofs() {
        assert_eq!(PyramidPk::new(1).n_dofs(), 5);
        assert_eq!(PyramidPk::new(2).n_dofs(), 14);
        assert_eq!(PyramidPk::new(3).n_dofs(), 30);
    }

    #[test]
    fn pyramid_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=3 {
            let elem = PyramidPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * 3],
            );
            let test_pts: &[[f64; 3]] = if p == 1 {
                &[[0.1, 0.1, 0.1]]
            } else {
                &[[0.15, 0.15, 0.1], [0.08, 0.08, 0.2]]
            };
            for pt in test_pts {
                let (x, y, z) = (pt[0], pt[1], pt[2]);
                if x + z > 0.8 || y + z > 0.8 {
                    continue;
                }
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x + h, y, z], &mut vx);
                elem.eval_basis(&[x, y + h, z], &mut vy);
                elem.eval_basis(&[x, y, z + h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!(
                        (grads[i * 3] - fd_x).abs() < 1e-4,
                        "p={p} ({x},{y},{z}) i={i} gx"
                    );
                    assert!(
                        (grads[i * 3 + 1] - fd_y).abs() < 1e-4,
                        "p={p} ({x},{y},{z}) i={i} gy"
                    );
                    assert!(
                        (grads[i * 3 + 2] - fd_z).abs() < 4e-4,
                        "p={p} ({x},{y},{z}) i={i} gz"
                    );
                }
            }
        }
    }

    /// Regression test: `Lagrange1D::val_d` / `val_d_h` used to return 0/NaN
    /// when evaluated exactly at a node (removable 0·∞ singularity in the
    /// barycentric derivative formula). Verify against closed-form values and
    /// finite differences.
    #[test]
    fn lagrange1d_derivatives_at_nodes() {
        for p in [1usize, 2, 3, 4] {
            let lag = Lagrange1D::new(p);
            for (k, &xk) in lag.nodes.iter().enumerate() {
                let (vals, ders) = lag.val_d(xk);
                let (vals2, ders2, hess) = lag.val_d_h(xk);
                assert_eq!(vals, vals2);
                assert_eq!(ders, ders2);
                // Values: l_i(x_k) = δ_ik
                for i in 0..=p {
                    let want = if i == k { 1.0 } else { 0.0 };
                    assert!((vals[i] - want).abs() < 1e-14, "p={p} k={k} i={i} val");
                }
                // Derivatives must be finite and match central FD of `val`.
                let h = 1e-7;
                let vp = lag.val(xk + h);
                let vm = lag.val(xk - h);
                for i in 0..=p {
                    assert!(ders[i].is_finite(), "p={p} k={k} i={i} der not finite");
                    assert!(hess[i].is_finite(), "p={p} k={k} i={i} hess not finite");
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    assert!(
                        (ders[i] - fd).abs() < 1e-6,
                        "p={p} k={k} i={i}: der={} fd={fd}",
                        ders[i]
                    );
                }
                // Partition of unity: Σ l_i' = 0, Σ l_i'' = 0.
                assert!(ders.iter().sum::<f64>().abs() < 1e-12, "p={p} k={k} Σder");
                assert!(hess.iter().sum::<f64>().abs() < 1e-10, "p={p} k={k} Σhess");
            }
        }
        // Closed forms: p=2, nodes {-1,0,1}, at x=-1:
        //   l_0 = x(x-1)/2 → l_0'(-1) = -3/2, l_0''(-1) = 1
        //   l_1 = 1-x²    → l_1'(-1) = 2,    l_1''(-1) = -2
        //   l_2 = x(x+1)/2 → l_2'(-1) = -1/2, l_2''(-1) = 1
        let lag = Lagrange1D::new(2);
        let (_, ders, hess) = lag.val_d_h(-1.0);
        assert!((ders[0] + 1.5).abs() < 1e-14);
        assert!((ders[1] - 2.0).abs() < 1e-14);
        assert!((ders[2] + 0.5).abs() < 1e-14);
        assert!((hess[0] - 1.0).abs() < 1e-14);
        assert!((hess[1] + 2.0).abs() < 1e-14);
        assert!((hess[2] - 1.0).abs() < 1e-14);
    }

    /// QuadQk/HexQk gradients and Hessians at element vertices (uses the fixed
    /// node path through `eval_grad_basis` / `eval_hessian`).
    #[test]
    fn quad_hex_grad_at_vertices_finite() {
        for p in [1usize, 2, 4] {
            let q = QuadQk::new(p);
            let coords = q.dof_coords();
            let mut g = vec![0.0_f64; q.n_dofs() * 2];
            let mut hs = vec![0.0_f64; q.n_dofs() * 4];
            for xi in &coords {
                q.eval_grad_basis(xi, &mut g);
                assert!(
                    g.iter().all(|v| v.is_finite()),
                    "QuadQk p={p} grad at {xi:?}"
                );
                q.eval_hessian(xi, &mut hs);
                assert!(
                    hs.iter().all(|v| v.is_finite()),
                    "QuadQk p={p} hess at {xi:?}"
                );
                // Σ ∇φᵢ = 0 (partition of unity)
                let (sx, sy): (f64, f64) = (0..q.n_dofs())
                    .map(|i| (g[i * 2], g[i * 2 + 1]))
                    .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                assert!(
                    sx.abs() < 1e-12 && sy.abs() < 1e-12,
                    "QuadQk p={p} Σgrad at {xi:?}"
                );
            }
            let hx = HexQk::new(p.min(2));
            let coords3 = hx.dof_coords();
            let mut g3 = vec![0.0_f64; hx.n_dofs() * 3];
            for xi in &coords3 {
                hx.eval_grad_basis(xi, &mut g3);
                assert!(g3.iter().all(|v| v.is_finite()), "HexQk grad at {xi:?}");
            }
        }
    }
}
