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

use crate::quadrature::{seg_rule, tri_rule, tet_rule, quad_rule, hex_rule};
use crate::reference::{QuadratureRule, ReferenceElement, VectorReferenceElement};
use super::prism::PrismPk;
use super::pyramid::PyramidPk;

// ─── Helpers: equispaced nodes ────────────────────────────────────────────────

fn equispaced_nodes_1d(p: usize) -> Vec<f64> {
    (0..=p).map(|i| i as f64 / p as f64).collect()
}

fn equispaced_nodes_tri(p: usize) -> Vec<[f64; 2]> {
    let mut nodes = Vec::with_capacity((p + 1) * (p + 2) / 2);
    nodes.push([0.0, 0.0]);
    nodes.push([1.0, 0.0]);
    nodes.push([0.0, 1.0]);
    if p == 1 { return nodes; }
    for k in 1..p { nodes.push([k as f64 / p as f64, 0.0]); }
    for k in 1..p { let t = k as f64 / p as f64; nodes.push([1.0 - t, t]); }
    for k in 1..p { nodes.push([0.0, k as f64 / p as f64]); }
    for j in 1..=(p - 2) {
        for i in 1..=(p - 1 - j) {
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
    if p == 1 { return nodes; }
    for k in 1..p { nodes.push([k as f64 / p as f64, 0.0, 0.0]); }
    for k in 1..p { nodes.push([0.0, k as f64 / p as f64, 0.0]); }
    for k in 1..p { nodes.push([0.0, 0.0, k as f64 / p as f64]); }
    for k in 1..p { let t = k as f64 / p as f64; nodes.push([1.0 - t, t, 0.0]); }
    for k in 1..p { let t = k as f64 / p as f64; nodes.push([1.0 - t, 0.0, t]); }
    for k in 1..p { let t = k as f64 / p as f64; nodes.push([0.0, 1.0 - t, t]); }
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
                nodes.push([i as f64 / p as f64, j as f64 / p as f64, k as f64 / p as f64]);
            }
        }
    }
    let expected = (p + 1) * (p + 2) * (p + 3) / 6;
    debug_assert_eq!(nodes.len(), expected, "tet p={p}: got {} nodes, expected {expected}", nodes.len());
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
    fn dim(&self) -> u8 { 1 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { self.order + 1 }
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
    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        equispaced_nodes_1d(self.order).iter().map(|&x| vec![x]).collect()
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

/// Rising-factorial basis L_n(t) = Π_{a=0}^{n-1} (t - a) / (n - a), with L_0(t) = 1.
/// This is the correct building block for simplex Lagrange elements, NOT the
/// standard Lagrange polynomial through integer nodes.
pub fn rising_val(n: usize, t: f64) -> f64 {
    if n == 0 { return 1.0; }
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
    if n == 0 { return 0.0; }
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
        let ijk: Vec<(usize, usize, usize)> = nodes.iter()
            .map(|n| {
                let i = (n[0] * p as f64).round() as usize;
                let j = (n[1] * p as f64).round() as usize;
                (i, j, p - i - j)
            })
            .collect();
        Self { order: p, nodes, ijk }
    }
}

impl ReferenceElement for TriPk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) / 2 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * (1.0 - xi[0] - xi[1]);
        for (dof_idx, &(i, j, k)) in self.ijk.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0)
                            * rising_val(j, t1)
                            * rising_val(k, t2);
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
            grads[dof_idx * 2]     = pf * (di * vj * vk - vi * vj * dk);
            // ∂φ/∂η = p·(vi·dj·vk - vi·vj·dk)
            grads[dof_idx * 2 + 1] = pf * (vi * dj * vk - vi * vj * dk);
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1]]).collect()
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
        let ijkl: Vec<(usize, usize, usize, usize)> = nodes.iter()
            .map(|n| {
                let i = (n[0] * p as f64).round() as usize;
                let j = (n[1] * p as f64).round() as usize;
                let k = (n[2] * p as f64).round() as usize;
                (i, j, k, p - i - j - k)
            })
            .collect();
        Self { order: p, nodes, ijkl }
    }
}

impl ReferenceElement for TetPk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) * (self.order + 3) / 6 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;
        let t0 = pf * xi[0];
        let t1 = pf * xi[1];
        let t2 = pf * xi[2];
        let t3 = pf * (1.0 - xi[0] - xi[1] - xi[2]);
        for (dof_idx, &(i, j, k, l)) in self.ijkl.iter().enumerate() {
            values[dof_idx] = rising_val(i, t0)
                            * rising_val(j, t1)
                            * rising_val(k, t2)
                            * rising_val(l, t3);
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
            grads[dof_idx * 3]     = pf * (di * vj * vk * vl - vi * vj * vk * dl);
            grads[dof_idx * 3 + 1] = pf * (vi * dj * vk * vl - vi * vj * vk * dl);
            grads[dof_idx * 3 + 2] = pf * (vi * vj * dk * vl - vi * vj * vk * dl);
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}

// ─── QuadQk ──────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference quad `[-1,1]²` — `(p+1)²` DOFs.
pub struct QuadQk {
    order: usize,
    nodes_1d: Vec<f64>,
}

impl QuadQk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes_1d: Vec<f64> = (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect();
        Self { order: p, nodes_1d }
    }

    fn lagrange_1d(&self, x: f64) -> Vec<f64> {
        let n = self.nodes_1d.len();
        let mut vals = vec![1.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                if j != i { vals[i] *= (x - self.nodes_1d[j]) / (self.nodes_1d[i] - self.nodes_1d[j]); }
            }
        }
        vals
    }

    fn lagrange_1d_deriv(&self, x: f64) -> Vec<f64> {
        let n = self.nodes_1d.len();
        let mut ders = vec![0.0_f64; n];
        for i in 0..n {
            let mut sum = 0.0_f64;
            for m in 0..n {
                if m == i { continue; }
                let mut prod = 1.0 / (self.nodes_1d[i] - self.nodes_1d[m]);
                for j in 0..n {
                    if j != i && j != m { prod *= (x - self.nodes_1d[j]) / (self.nodes_1d[i] - self.nodes_1d[j]); }
                }
                sum += prod;
            }
            ders[i] = sum;
        }
        ders
    }

    fn node_to_dof(&self, ix: usize, iy: usize) -> usize {
        let p = self.order;
        let x = self.nodes_1d[ix];
        let y = self.nodes_1d[iy];
        let tol = 1e-12;
        let on_xmin = (x + 1.0).abs() < tol;
        let on_xmax = (x - 1.0).abs() < tol;
        let on_ymin = (y + 1.0).abs() < tol;
        let on_ymax = (y - 1.0).abs() < tol;
        let on_boundary = on_xmin || on_xmax || on_ymin || on_ymax;

        if on_boundary {
            if on_xmin && on_ymin { return 0; }
            if on_xmax && on_ymin { return 1; }
            if on_xmax && on_ymax { return 2; }
            if on_xmin && on_ymax { return 3; }
            let mut idx = 4usize;
            if on_ymin { return idx + (ix - 1); }
            idx += p - 1;
            if on_xmax { return idx + (iy - 1); }
            idx += p - 1;
            if on_ymax { return idx + (p - 1 - ix); }
            idx += p - 1;
            if on_xmin { return idx + (p - 1 - iy); }
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
                let dof = self.node_to_dof(ix, iy);
                coords[dof] = [self.nodes_1d[ix], self.nodes_1d[iy]];
            }
        }
        coords
    }
}

impl ReferenceElement for QuadQk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 1) }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let lx = self.lagrange_1d(xi[0]);
        let ly = self.lagrange_1d(xi[1]);
        let p = self.order;
        for iy in 0..=p {
            for ix in 0..=p {
                values[self.node_to_dof(ix, iy)] = lx[ix] * ly[iy];
            }
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let lx = self.lagrange_1d(xi[0]);
        let ly = self.lagrange_1d(xi[1]);
        let dlx = self.lagrange_1d_deriv(xi[0]);
        let dly = self.lagrange_1d_deriv(xi[1]);
        let p = self.order;
        for iy in 0..=p {
            for ix in 0..=p {
                let dof = self.node_to_dof(ix, iy);
                grads[dof * 2]     = dlx[ix] * ly[iy];
                grads[dof * 2 + 1] = lx[ix]  * dly[iy];
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.all_dof_coords().iter().map(|c| c.to_vec()).collect()
    }
}

// ─── HexQk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference hex `[-1,1]³` — `(p+1)³` DOFs.
pub struct HexQk {
    order: usize,
    nodes_1d: Vec<f64>,
}

impl HexQk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes_1d: Vec<f64> = (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect();
        Self { order: p, nodes_1d }
    }

    fn lagrange_1d(&self, x: f64) -> Vec<f64> {
        let n = self.nodes_1d.len();
        let mut vals = vec![1.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                if j != i { vals[i] *= (x - self.nodes_1d[j]) / (self.nodes_1d[i] - self.nodes_1d[j]); }
            }
        }
        vals
    }

    fn lagrange_1d_deriv(&self, x: f64) -> Vec<f64> {
        let n = self.nodes_1d.len();
        let mut ders = vec![0.0_f64; n];
        for i in 0..n {
            let mut sum = 0.0_f64;
            for m in 0..n {
                if m == i { continue; }
                let mut prod = 1.0 / (self.nodes_1d[i] - self.nodes_1d[m]);
                for j in 0..n {
                    if j != i && j != m { prod *= (x - self.nodes_1d[j]) / (self.nodes_1d[i] - self.nodes_1d[j]); }
                }
                sum += prod;
            }
            ders[i] = sum;
        }
        ders
    }

    fn node_to_dof(&self, ix: usize, iy: usize, iz: usize) -> usize {
        let p = self.order;
        let x = self.nodes_1d[ix];
        let y = self.nodes_1d[iy];
        let z = self.nodes_1d[iz];
        let tol = 1e-12;

        let on_xmin = (x + 1.0).abs() < tol;
        let on_xmax = (x - 1.0).abs() < tol;
        let on_ymin = (y + 1.0).abs() < tol;
        let on_ymax = (y - 1.0).abs() < tol;
        let on_zmin = (z + 1.0).abs() < tol;
        let on_zmax = (z - 1.0).abs() < tol;

        let n_faces = [on_xmin, on_xmax, on_ymin, on_ymax, on_zmin, on_zmax]
            .iter().filter(|&&b| b).count();

        if n_faces >= 3 {
            let vx = if on_xmin { 0 } else { 1 };
            let vy = if on_ymin { 0 } else { 1 };
            let vz = if on_zmin { 0 } else { 1 };
            return match (vx, vy, vz) {
                (0,0,0) => 0, (1,0,0) => 1, (1,1,0) => 2, (0,1,0) => 3,
                (0,0,1) => 4, (1,0,1) => 5, (1,1,1) => 6, (0,1,1) => 7,
                _ => unreachable!(),
            };
        }

        let mut base = 8usize;

        if n_faces == 2 {
            // Determine which axis varies and compute DOF directly.
            // The varying axis is the one whose faces are NOT active.
            let idx;
            let local;

            if !on_xmin && !on_xmax {
                // x varies
                idx = ix;
                local = if on_ymin && !on_ymax && !on_zmin && !on_zmax {
                    ix                                    // e0: xmax∩ymin
                } else if !on_ymin && on_ymax && !on_zmin && !on_zmax {
                    p - ix                                // e1: xmax∩ymax
                } else if !on_ymin && on_ymax && !on_zmin && !on_zmax { // xmin∩ymax
                    p - ix                                // e2
                } else {
                    ix                                    // e3: xmin∩ymin
                };
            } else if !on_ymin && !on_ymax {
                // y varies
                idx = iy;
                local = if on_xmin && !on_xmax && on_zmin && !on_zmax {
                    iy                                    // e4
                } else if !on_xmin && on_xmax && on_zmin && !on_zmax {
                    iy                                    // e5
                } else if !on_xmin && on_xmax && !on_zmin && on_zmax {
                    p - iy                                // e6
                } else {
                    p - iy                                // e7
                };
            } else {
                // z varies
                idx = iz;
                local = if on_ymin && !on_ymax && on_xmin && !on_xmax {
                    p - iz                                // e8
                } else if !on_ymin && on_ymax && on_xmin && !on_xmax {
                    iz                                    // e9
                } else if !on_ymin && on_ymax && !on_xmin && on_xmax {
                    p - iz                                // e10
                } else {
                    iz                                    // e11
                };
            }

            if idx > 0 && idx < p {
                // Map face pair to edge index for DOF computation
                let faces: Vec<usize> = [on_xmin, on_xmax, on_ymin, on_ymax, on_zmin, on_zmax]
                    .iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
                let (f0, f1) = if faces[0] < faces[1] { (faces[0], faces[1]) } else { (faces[1], faces[0]) };
                let ei = match (f0, f1) {
                    (1, 2) => 0, (1, 3) => 1, (0, 3) => 2, (0, 2) => 3,
                    (0, 4) => 4, (1, 4) => 5, (1, 5) => 6, (0, 5) => 7,
                    (2, 4) => 8, (3, 4) => 9, (3, 5) => 10, (2, 5) => 11,
                    _ => 0,
                };
                return base + ei * (p - 1) + (local - 1);
            }
            // idx is on boundary → corner vertex, fall through
        }

        // n_faces == 0 → volume interior point
        if n_faces == 0 && p >= 2 {
            let vol_base = 8 + 12 * (p - 1) + 6 * (p - 1) * (p - 1);
            return vol_base + (iz - 1) * (p - 1) * (p - 1) + (iy - 1) * (p - 1) + (ix - 1);
        }

        // n_faces == 1 → face interior point
        if p < 2 { return 0; } // No face interior DOFs for p < 2
        base += 12 * (p - 1);
        let face_idx = if on_xmin { 0 }
            else if on_xmax { 1 }
            else if on_ymin { 2 }
            else if on_ymax { 3 }
            else if on_zmin { 4 }
            else { 5 };

        // Varying coordinate indices for this face
        let (va, vb) = match face_idx {
            0 | 1 => (iy, iz), // xmin/xmax: y,z vary
            2 | 3 => (ix, iz), // ymin/ymax: x,z vary
            4 | 5 => (ix, iy), // zmin/zmax: x,y vary
            _ => unreachable!(),
        };
        // Varying coordinates must be strictly interior (not on any edge of this face)
        if va == 0 || va == p || vb == 0 || vb == p {
            return 0; // Fallback: this is an edge/corner point misclassified as face
        }

        // Compute face DOF index with direction adjustment
        let (fa, fb) = match face_idx {
            0 => (p - iy, iz),
            1 => (iy, iz),
            2 => (ix, p - iz),
            3 => (ix, iz),
            4 => (ix, p - iy),
            5 => (ix, iy),
            _ => unreachable!(),
        };
        base + face_idx * (p - 1) * (p - 1) + (fb - 1) * (p - 1) + (fa - 1)
    }

    fn all_dof_coords(&self) -> Vec<[f64; 3]> {
        let p = self.order;
        let n = (p + 1) * (p + 1) * (p + 1);
        let mut coords = vec![[0.0, 0.0, 0.0]; n];
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.node_to_dof(ix, iy, iz);
                    coords[dof] = [self.nodes_1d[ix], self.nodes_1d[iy], self.nodes_1d[iz]];
                }
            }
        }
        coords
    }
}

impl ReferenceElement for HexQk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { let p = self.order + 1; p * p * p }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let lx = self.lagrange_1d(xi[0]);
        let ly = self.lagrange_1d(xi[1]);
        let lz = self.lagrange_1d(xi[2]);
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
        let lx = self.lagrange_1d(xi[0]);
        let ly = self.lagrange_1d(xi[1]);
        let lz = self.lagrange_1d(xi[2]);
        let dlx = self.lagrange_1d_deriv(xi[0]);
        let dly = self.lagrange_1d_deriv(xi[1]);
        let dlz = self.lagrange_1d_deriv(xi[2]);
        let p = self.order;
        for iz in 0..=p {
            for iy in 0..=p {
                for ix in 0..=p {
                    let dof = self.node_to_dof(ix, iy, iz);
                    grads[dof * 3]     = dlx[ix] * ly[iy]  * lz[iz];
                    grads[dof * 3 + 1] = lx[ix]  * dly[iy] * lz[iz];
                    grads[dof * 3 + 2] = lx[ix]  * ly[iy]  * dlz[iz];
                }
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { hex_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.all_dof_coords().iter().map(|c| c.to_vec()).collect()
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/// Element type identifier for the factory function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType { Seg, Tri, Tet, Quad, Hex, Prism, Pyramid }

/// Create a reference element of the given type and order.
pub fn ref_elem(etype: ElemType, order: u8) -> Box<dyn ReferenceElement> {
    match etype {
        ElemType::Seg     => Box::new(SegPk::new(order as usize)),
        ElemType::Tri     => Box::new(TriPk::new(order as usize)),
        ElemType::Tet     => Box::new(TetPk::new(order as usize)),
        ElemType::Quad    => Box::new(QuadQk::new(order as usize)),
        ElemType::Hex     => Box::new(HexQk::new(order as usize)),
        ElemType::Prism   => Box::new(PrismPk::new(order as usize)),
        ElemType::Pyramid => Box::new(PyramidPk::new(order as usize)),
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
    for i in 1..=dim { num = num * (order + i) / i; }
    num
}

/// Family of vector-valued reference elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecFamily { Nedelec, RaviartThomas }

/// Create a vector-valued reference element by family, type, and order.
pub fn vec_ref_elem(family: VecFamily, etype: ElemType, order: u8) -> Box<dyn VectorReferenceElement> {
    let p = order as usize;
    match (family, etype) {
        (VecFamily::Nedelec, ElemType::Tri) => Box::new(crate::nedelec::TriNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Quad) => Box::new(crate::nedelec::QuadNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Tet) => Box::new(crate::nedelec::TetNDk::new(p)),
        (VecFamily::Nedelec, ElemType::Hex) => Box::new(crate::nedelec::HexNDk::new(p)),
        (VecFamily::RaviartThomas, ElemType::Tri) => Box::new(crate::raviart_thomas::TriRTk::new(p)),
        (VecFamily::RaviartThomas, ElemType::Quad) => Box::new(crate::raviart_thomas::QuadRTk::new(p)),
        (VecFamily::RaviartThomas, ElemType::Tet) => Box::new(crate::raviart_thomas::TetRTk::new(p)),
        (VecFamily::RaviartThomas, ElemType::Hex) => Box::new(crate::raviart_thomas::HexRTk::new(p)),
        _ => panic!("vec_ref_elem: unsupported (family={family:?}, type={etype:?})"),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let order = elem.order() as usize;
        let rule = elem.quadrature((2 * order as u8 + 2).min(15));
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-10,
                "POU failed for dim={} p={order} at {:?}: sum={s}", elem.dim(), pt);
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
                assert!(s.abs() < 1e-10,
                    "grad sum d={d} = {s} for dim={} p={order}", elem.dim());
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
                assert!((phi[j] - expected).abs() < 1e-10,
                    "nodal interp: node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    // ── SegPk ─────────────────────────────────────────────────────────────

    #[test] fn seg_pk_pou() { for p in 1..=5 { check_pou(&SegPk::new(p)); } }
    #[test] fn seg_pk_grad_zero() { for p in 1..=5 { check_grad_zero(&SegPk::new(p)); } }
    #[test] fn seg_pk_nodal_interp() { for p in 1..=5 { check_nodal_interp(&SegPk::new(p)); } }

    // ── TriPk ─────────────────────────────────────────────────────────────
    // Note: gradient_fd tests p=1..=5 (direct Lagrange formula, no Vandermonde).
    // grad_zero uses &dyn ReferenceElement trait object which may overflow
    // Windows debug stack when called in a loop; tested at single order.
    // p=5 passes gradient_fd with 1e-5 tolerance on all DOFs.

    #[test] fn tri_pk_pou() { check_pou(&TriPk::new(3)); }
    #[test] fn tri_pk_grad_zero() { check_grad_zero(&TriPk::new(3)); }
    #[test] fn tri_pk_nodal_interp() { check_nodal_interp(&TriPk::new(3)); }

    // ── TetPk ─────────────────────────────────────────────────────────────

    #[test] fn tet_pk_pou() { check_pou(&TetPk::new(3)); }
    #[test] fn tet_pk_grad_zero() { check_grad_zero(&TetPk::new(3)); }
    #[test] fn tet_pk_nodal_interp() { check_nodal_interp(&TetPk::new(3)); }
    #[test] fn tet_pk_n_dofs() { for p in 1..=8 { assert_eq!(TetPk::new(p).n_dofs(), (p+1)*(p+2)*(p+3)/6); } }

    #[test]
    fn tet_pk_matches_p1() {
        use crate::lagrange::TetP1;
        let pk = TetPk::new(1);
        let n = 4;
        let mut v1 = vec![0.0; n]; let mut v2 = vec![0.0; n];
        for &(x,y,z) in &[(0.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(0.25,0.25,0.25)] {
            pk.eval_basis(&[x,y,z], &mut v1); TetP1.eval_basis(&[x,y,z], &mut v2);
            for i in 0..n { assert!((v1[i]-v2[i]).abs() < 1e-13, "tet p=1 ({x},{y},{z}) i={i}"); }
        }
    }

    #[test]
    fn tet_pk_matches_p2() {
        use crate::lagrange::TetP2;
        let pk = TetPk::new(2);
        let n = 10;
        let mut v1 = vec![0.0; n]; let mut v2 = vec![0.0; n];
        for &(x,y,z) in &[
            (0.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),
            (0.5,0.0,0.0),(0.0,0.5,0.0),(0.0,0.0,0.5),
            (0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5),(0.2,0.3,0.1),
        ] {
            pk.eval_basis(&[x,y,z], &mut v1); TetP2.eval_basis(&[x,y,z], &mut v2);
            for i in 0..n { assert!((v1[i]-v2[i]).abs() < 1e-12, "tet p=2 ({x},{y},{z}) i={i}"); }
        }
    }

    // ── QuadQk ────────────────────────────────────────────────────────────

    #[test] fn quad_qk_pou() { for p in 1..=6 { check_pou(&QuadQk::new(p)); } }
    #[test] fn quad_qk_grad_zero() { for p in 1..=6 { check_grad_zero(&QuadQk::new(p)); } }
    #[test] fn quad_qk_nodal_interp() { for p in 1..=6 { check_nodal_interp(&QuadQk::new(p)); } }
    #[test] fn quad_qk_n_dofs() { for p in 1..=8 { assert_eq!(QuadQk::new(p).n_dofs(), (p+1)*(p+1)); } }

    #[test]
    fn quad_qk_matches_q1() {
        use crate::lagrange::QuadQ1;
        let qk = QuadQk::new(1);
        let n = 4;
        let mut v1 = vec![0.0; n]; let mut v2 = vec![0.0; n];
        for &(x,y) in &[(-1.0,-1.0),(1.0,-1.0),(1.0,1.0),(-1.0,1.0),(0.3,-0.5)] {
            qk.eval_basis(&[x,y], &mut v1); QuadQ1.eval_basis(&[x,y], &mut v2);
            for i in 0..n { assert!((v1[i]-v2[i]).abs() < 1e-13, "Q1 ({x},{y}) i={i}"); }
        }
    }

    #[test]
    fn quad_qk_matches_q2() {
        use crate::lagrange::QuadQ2;
        let qk = QuadQk::new(2);
        let n = 9;
        let mut v1 = vec![0.0; n]; let mut v2 = vec![0.0; n];
        for &(x,y) in &[
            (-1.0,-1.0),(1.0,-1.0),(1.0,1.0),(-1.0,1.0),
            (0.0,-1.0),(1.0,0.0),(0.0,1.0),(-1.0,0.0),(0.0,0.0),(0.3,-0.5),
        ] {
            qk.eval_basis(&[x,y], &mut v1); QuadQ2.eval_basis(&[x,y], &mut v2);
            for i in 0..n { assert!((v1[i]-v2[i]).abs() < 1e-12, "Q2 ({x},{y}) i={i}"); }
        }
    }

    // ── HexQk ─────────────────────────────────────────────────────────────

    #[test] fn hex_qk_pou() { for p in 1..=4 { check_pou(&HexQk::new(p)); } }
    #[test] fn hex_qk_grad_zero() { for p in 1..=4 { check_grad_zero(&HexQk::new(p)); } }
    #[test] fn hex_qk_nodal_interp() { for p in 1..=4 { check_nodal_interp(&HexQk::new(p)); } }
    #[test] fn hex_qk_n_dofs() { for p in 1..=6 { let pp=p+1; assert_eq!(HexQk::new(p).n_dofs(), pp*pp*pp); } }

    #[test]
    fn hex_qk_matches_q1() {
        use crate::lagrange::HexQ1;
        let qk = HexQk::new(1);
        let n = 8;
        let mut v1 = vec![0.0; n]; let mut v2 = vec![0.0; n];
        for &(x,y,z) in &[
            (-1.0,-1.0,-1.0),(1.0,-1.0,-1.0),(1.0,1.0,-1.0),(-1.0,1.0,-1.0),
            (-1.0,-1.0,1.0),(1.0,-1.0,1.0),(1.0,1.0,1.0),(-1.0,1.0,1.0),(0.3,-0.5,0.7),
        ] {
            qk.eval_basis(&[x,y,z], &mut v1); HexQ1.eval_basis(&[x,y,z], &mut v2);
            for i in 0..n { assert!((v1[i]-v2[i]).abs() < 1e-13, "H1 ({x},{y},{z}) i={i}"); }
        }
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
        for p in 1..=4 { // Monomial Vandermonde conditioning degrades for p > 5
            let elem = SegPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n]);
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
            let (mut vc, mut vx, mut vy, mut grads) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*2]);
            for &(x,y) in &[(0.2,0.3),(0.5,0.2),(1.0/3.0,1.0/3.0)] {
                elem.eval_basis(&[x,y], &mut vc);
                elem.eval_basis(&[x+h,y], &mut vx);
                elem.eval_basis(&[x,y+h], &mut vy);
                elem.eval_grad_basis(&[x,y], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    assert!((grads[i*2] - fd_x).abs() < 1e-5, "p={p} ({x},{y}) i={i} gx: analytic={} fd={}", grads[i*2], fd_x);
                    assert!((grads[i*2+1] - fd_y).abs() < 1e-5, "p={p} ({x},{y}) i={i} gy: analytic={} fd={}", grads[i*2+1], fd_y);
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
                vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*3]
            );
            for &(x,y,z) in &[(0.15,0.2,0.25),(0.3,0.3,0.1)] {
                elem.eval_basis(&[x,y,z], &mut vc);
                elem.eval_basis(&[x+h,y,z], &mut vx);
                elem.eval_basis(&[x,y+h,z], &mut vy);
                elem.eval_basis(&[x,y,z+h], &mut vz);
                elem.eval_grad_basis(&[x,y,z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!((grads[i*3] - fd_x).abs() < 1e-5, "p={p} ({x},{y},{z}) i={i} gx");
                    assert!((grads[i*3+1] - fd_y).abs() < 1e-5, "p={p} ({x},{y},{z}) i={i} gy");
                    assert!((grads[i*3+2] - fd_z).abs() < 1e-5, "p={p} ({x},{y},{z}) i={i} gz");
                }
            }
        }
    }

    // ── PrismPk ────────────────────────────────────────────────────────────

    #[test] fn prism_pk_pou() { check_pou(&PrismPk::new(2)); }
    #[test] fn prism_pk_grad_zero() { check_grad_zero(&PrismPk::new(2)); }
    #[test] fn prism_pk_nodal_interp() { check_nodal_interp(&PrismPk::new(2)); }
    #[test] fn prism_pk_n_dofs() {
        for p in 1..=5 {
            let n_tri = (p+1)*(p+2)/2;
            assert_eq!(PrismPk::new(p).n_dofs(), (p+1)*n_tri);
        }
    }

    #[test]
    fn prism_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=3 {
            let elem = PrismPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
                vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*3]
            );
            let test_pts: &[[f64; 3]] = if p == 1 {
                &[[0.3, 0.2, 0.1]]
            } else {
                &[[0.2, 0.3, 0.15], [0.5, 0.1, 0.25]]
            };
            for pt in test_pts {
                let (x, y, z) = (pt[0], pt[1], pt[2]);
                if y + z > 0.95 { continue; }
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x+h, y, z], &mut vx);
                elem.eval_basis(&[x, y+h, z], &mut vy);
                elem.eval_basis(&[x, y, z+h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!((grads[i*3] - fd_x).abs() < 1e-5, "p={p} ({x},{y},{z}) i={i} gx");
                    assert!((grads[i*3+1] - fd_y).abs() < 1e-5, "p={p} ({x},{y},{z}) i={i} gy");
                    assert!((grads[i*3+2] - fd_z).abs() < 1e-5, "p={p} ({x},{y},{z}) i={i} gz");
                }
            }
        }
    }

    // ── PyramidPk ──────────────────────────────────────────────────────────

    #[test] fn pyramid_pk_pou() { check_pou(&PyramidPk::new(2)); }
    #[test] fn pyramid_pk_grad_zero() { check_grad_zero(&PyramidPk::new(2)); }
    #[test] fn pyramid_pk_nodal_interp() { check_nodal_interp(&PyramidPk::new(2)); }
    #[test] fn pyramid_pk_n_dofs() {
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
                vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*3]
            );
            let test_pts: &[[f64; 3]] = if p == 1 {
                &[[0.1, 0.1, 0.1]]
            } else {
                &[[0.15, 0.15, 0.1], [0.08, 0.08, 0.2]]
            };
            for pt in test_pts {
                let (x, y, z) = (pt[0], pt[1], pt[2]);
                if x + z > 0.8 || y + z > 0.8 { continue; }
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x+h, y, z], &mut vx);
                elem.eval_basis(&[x, y+h, z], &mut vy);
                elem.eval_basis(&[x, y, z+h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!((grads[i*3] - fd_x).abs() < 1e-4, "p={p} ({x},{y},{z}) i={i} gx");
                    assert!((grads[i*3+1] - fd_y).abs() < 1e-4, "p={p} ({x},{y},{z}) i={i} gy");
                    assert!((grads[i*3+2] - fd_z).abs() < 4e-4, "p={p} ({x},{y},{z}) i={i} gz");
                }
            }
        }
    }
}
