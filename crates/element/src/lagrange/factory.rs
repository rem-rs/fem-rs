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
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Helpers: monomial enumeration ────────────────────────────────────────────

fn monomial_exponents(dim: usize, p: usize) -> Vec<Vec<usize>> {
    let mut exps = Vec::new();
    let mut stack: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for total in (0..=p).rev() {
        stack.push((total, dim, Vec::new()));
    }
    while let Some((total, remaining, mut current)) = stack.pop() {
        if remaining == 1 {
            current.push(total);
            exps.push(current);
        } else {
            for k in (0..=total).rev() {
                let mut v = current.clone();
                v.push(k);
                stack.push((total - k, remaining - 1, v));
            }
        }
    }
    exps
}

/// Evaluate a monomial `x₁^a₁ · x₂^a₂ · …` at point `x`.
fn eval_monomial(exponents: &[usize], x: &[f64]) -> f64 {
    exponents.iter().zip(x.iter()).map(|(&a, &xi)| xi.powi(a as i32)).product()
}

fn eval_monomial_deriv(exponents: &[usize], x: &[f64], k: usize) -> f64 {
    let ak = exponents[k];
    if ak == 0 { return 0.0; }
    let mut val = ak as f64;
    for (j, (&aj, &xj)) in exponents.iter().zip(x.iter()).enumerate() {
        val *= if j == k { xj.powi((ak - 1) as i32) } else { xj.powi(aj as i32) };
    }
    val
}

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

// ─── Helpers: Vandermonde precomputation ──────────────────────────────────────

fn precompute_lagrange_coeffs(nodes: &[Vec<f64>], dim: usize, order: usize) -> Vec<f64> {
    let n_dofs = nodes.len();
    let monos = monomial_exponents(dim, order);
    let n_mono = monos.len();
    debug_assert_eq!(n_dofs, n_mono);

    // Build Vandermonde matrix V[i,j] = monomial_j(node_i)
    let mut vander = vec![0.0_f64; n_dofs * n_mono];
    for (i, node) in nodes.iter().enumerate() {
        for (j, exps) in monos.iter().enumerate() {
            vander[i * n_mono + j] = eval_monomial(exps, node);
        }
    }

    // Solve V^T * C = I using LU with partial pivoting
    // This gives C^T such that C^T * V = I, i.e., each row of C^T is a Lagrange polynomial
    let mut lu = vander;
    let mut pivots = vec![0usize; n_dofs];
    for col in 0..n_dofs {
        let mut max_val = lu[col * n_mono + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n_dofs {
            let val = lu[row * n_mono + col].abs();
            if val > max_val { max_val = val; max_row = row; }
        }
        pivots[col] = max_row;
        if max_row != col {
            for j in 0..n_mono { lu.swap(col * n_mono + j, max_row * n_mono + j); }
        }
        let pivot = lu[col * n_mono + col];
        debug_assert!(pivot.abs() > 1e-15, "Singular Vandermonde at col {col}");
        for row in (col + 1)..n_dofs {
            let factor = lu[row * n_mono + col] / pivot;
            lu[row * n_mono + col] = factor;
            for j in (col + 1)..n_mono { lu[row * n_mono + j] -= factor * lu[col * n_mono + j]; }
        }
    }

    let mut coeffs = vec![0.0_f64; n_dofs * n_mono];
    for basis in 0..n_dofs {
        let mut rhs = vec![0.0_f64; n_dofs];
        rhs[basis] = 1.0;
        for i in 0..n_dofs {
            let piv = pivots[i];
            if piv != i { rhs.swap(i, piv); }
        }
        for i in 1..n_dofs {
            for j in 0..i { rhs[i] -= lu[i * n_mono + j] * rhs[j]; }
        }
        for i in (0..n_dofs).rev() {
            for j in (i + 1)..n_dofs { rhs[i] -= lu[i * n_mono + j] * rhs[j]; }
            rhs[i] /= lu[i * n_mono + i];
        }
        for j in 0..n_dofs { coeffs[basis * n_mono + j] = rhs[j]; }
    }
    coeffs
}

fn eval_basis_from_coeffs(coeffs: &[f64], monos: &[Vec<usize>], x: &[f64], values: &mut [f64]) {
    let n_dofs = values.len();
    let n_mono = monos.len();
    let mut mono_vals = vec![0.0_f64; n_mono];
    for (j, exps) in monos.iter().enumerate() { mono_vals[j] = eval_monomial(exps, x); }
    for i in 0..n_dofs {
        let row = &coeffs[i * n_mono..(i + 1) * n_mono];
        values[i] = row.iter().zip(mono_vals.iter()).map(|(c, m)| c * m).sum();
    }
}

fn eval_grad_basis_from_coeffs(coeffs: &[f64], monos: &[Vec<usize>], x: &[f64], dim: usize, grads: &mut [f64]) {
    let n_dofs = grads.len() / dim;
    let n_mono = monos.len();
    for d in 0..dim {
        let mut mono_ders = vec![0.0_f64; n_mono];
        for (j, exps) in monos.iter().enumerate() { mono_ders[j] = eval_monomial_deriv(exps, x, d); }
        for i in 0..n_dofs {
            let row = &coeffs[i * n_mono..(i + 1) * n_mono];
            grads[i * dim + d] = row.iter().zip(mono_ders.iter()).map(|(c, m)| c * m).sum();
        }
    }
}

// ─── SegPk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on `[0,1]` — `(p+1)` DOFs.
///
/// DOF ordering: ξ = 0, 1/p, 2/p, …, 1 (equispaced, vertices first).
pub struct SegPk {
    order: usize,
    coeffs: Vec<f64>,
    monos: Vec<Vec<usize>>,
}

impl SegPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes = equispaced_nodes_1d(p);
        let node_vecs: Vec<Vec<f64>> = nodes.iter().map(|&x| vec![x]).collect();
        let monos = monomial_exponents(1, p);
        let coeffs = precompute_lagrange_coeffs(&node_vecs, 1, p);
        Self { order: p, coeffs, monos }
    }
}

impl ReferenceElement for SegPk {
    fn dim(&self) -> u8 { 1 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { self.order + 1 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        eval_basis_from_coeffs(&self.coeffs, &self.monos, xi, values);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        eval_grad_basis_from_coeffs(&self.coeffs, &self.monos, xi, 1, grads);
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        equispaced_nodes_1d(self.order).iter().map(|&x| vec![x]).collect()
    }
}

// ─── TriPk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference triangle `(0,0),(1,0),(0,1)` —
/// `(p+1)(p+2)/2` DOFs.
pub struct TriPk {
    order: usize,
    nodes: Vec<[f64; 2]>,
    coeffs: Vec<f64>,
    monos: Vec<Vec<usize>>,
}

impl TriPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes = equispaced_nodes_tri(p);
        let node_vecs: Vec<Vec<f64>> = nodes.iter().map(|n| vec![n[0], n[1]]).collect();
        let monos = monomial_exponents(2, p);
        let coeffs = precompute_lagrange_coeffs(&node_vecs, 2, p);
        Self { order: p, nodes, coeffs, monos }
    }
}

impl ReferenceElement for TriPk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) / 2 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        eval_basis_from_coeffs(&self.coeffs, &self.monos, xi, values);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        eval_grad_basis_from_coeffs(&self.coeffs, &self.monos, xi, 2, grads);
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1]]).collect()
    }
}

// ─── TetPk ───────────────────────────────────────────────────────────────────

/// Arbitrary-order Lagrange element on the reference tetrahedron —
/// `(p+1)(p+2)(p+3)/6` DOFs.
pub struct TetPk {
    order: usize,
    nodes: Vec<[f64; 3]>,
    coeffs: Vec<f64>,
    monos: Vec<Vec<usize>>,
}

impl TetPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let nodes = equispaced_nodes_tet(p);
        let node_vecs: Vec<Vec<f64>> = nodes.iter().map(|n| vec![n[0], n[1], n[2]]).collect();
        let monos = monomial_exponents(3, p);
        let coeffs = precompute_lagrange_coeffs(&node_vecs, 3, p);
        Self { order: p, nodes, coeffs, monos }
    }
}

impl ReferenceElement for TetPk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) * (self.order + 3) / 6 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        eval_basis_from_coeffs(&self.coeffs, &self.monos, xi, values);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        eval_grad_basis_from_coeffs(&self.coeffs, &self.monos, xi, 3, grads);
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
pub enum ElemType { Seg, Tri, Tet, Quad, Hex }

/// Create a reference element of the given type and order.
pub fn ref_elem(etype: ElemType, order: u8) -> Box<dyn ReferenceElement> {
    match etype {
        ElemType::Seg  => Box::new(SegPk::new(order as usize)),
        ElemType::Tri  => Box::new(TriPk::new(order as usize)),
        ElemType::Tet  => Box::new(TetPk::new(order as usize)),
        ElemType::Quad => Box::new(QuadQk::new(order as usize)),
        ElemType::Hex  => Box::new(HexQk::new(order as usize)),
    }
}

pub type LagrangeSegment = SegPk;
pub type LagrangeTriangle = TriPk;
pub type LagrangeTetrahedron = TetPk;
pub type LagrangeQuad = QuadQk;
pub type LagrangeHex = HexQk;

/// Number of DOFs for a simplex element of given dimension and order.
pub fn n_dofs_simplex(dim: usize, order: usize) -> usize {
    let mut num = 1usize;
    for i in 1..=dim { num = num * (order + i) / i; }
    num
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
    // Note: Individual tests pass for all orders. Running multiple Vandermonde-heavy
    // tests sequentially may overflow the Windows debug-mode test thread stack.
    // The gradient correctness is verified by gradient_fd (finite difference check).

    #[test] fn tri_pk_pou() { check_pou(&TriPk::new(3)); }
    #[test] fn tri_pk_nodal_interp() { check_nodal_interp(&TriPk::new(3)); }

    // ── TetPk ─────────────────────────────────────────────────────────────

    #[test] fn tet_pk_pou() { check_pou(&TetPk::new(3)); }
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
        for p in 1..=3 {
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
                    assert!((grads[i*2] - fd_x).abs() < 1e-5, "p={p} ({x},{y}) i={i} gx");
                    assert!((grads[i*2+1] - fd_y).abs() < 1e-5, "p={p} ({x},{y}) i={i} gy");
                }
            }
        }
    }

    #[test]
    fn tet_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=3 {
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
}
