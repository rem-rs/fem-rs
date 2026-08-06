//! Lagrange elements on the reference quadrilateral `[-1,1]²`.

use crate::quadrature::quad_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Q1 ───────────────────────────────────────────────────────────────────────

/// Bilinear Lagrange element on the reference quad `[-1,1]²` — 4 DOFs.
///
/// Node ordering (counter-clockwise):
/// - 0: (−1,−1)
/// - 1: (+1,−1)
/// - 2: (+1,+1)
/// - 3: (−1,+1)
///
/// Basis: φᵢ = (1 + ξᵢ ξ)(1 + ηᵢ η) / 4
pub struct QuadQ1;

/// Node coordinates (ξ, η) of the 4 Q1 nodes.
const Q1_NODES: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

impl ReferenceElement for QuadQ1 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        4
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in Q1_NODES.iter().enumerate() {
            values[i] = 0.25 * (1.0 + xi_i * x) * (1.0 + eta_i * y);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in Q1_NODES.iter().enumerate() {
            grads[i * 2] = 0.25 * xi_i * (1.0 + eta_i * y);
            grads[i * 2 + 1] = 0.25 * eta_i * (1.0 + xi_i * x);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q1_NODES.iter().map(|&(x, y)| vec![x, y]).collect()
    }
}

// ─── Q2 ───────────────────────────────────────────────────────────────────────

/// Biquadratic serendipity — 9-node Lagrange element on the reference quad `[-1,1]²`.
///
/// Node ordering:
/// - 0: (−1,−1)  corner
/// - 1: (+1,−1)  corner
/// - 2: (+1,+1)  corner
/// - 3: (−1,+1)  corner
/// - 4: ( 0,−1)  edge midpoint
/// - 5: (+1, 0)  edge midpoint
/// - 6: ( 0,+1)  edge midpoint
/// - 7: (−1, 0)  edge midpoint
/// - 8: ( 0, 0)  interior
///
/// Basis: standard tensor-product Lagrange polynomials through the nine nodes.
/// For node at (ξᵢ, ηᵢ), φᵢ = Lᵢ(ξ) · Lᵢ(η) where Lᵢ is the 1-D Lagrange polynomial.
pub struct QuadQ2;

/// Node coordinates (ξ, η) of the 9 Q2 nodes.
const Q2_NODES: [(f64, f64); 9] = [
    (-1.0, -1.0), // 0
    (1.0, -1.0),  // 1
    (1.0, 1.0),   // 2
    (-1.0, 1.0),  // 3
    (0.0, -1.0),  // 4
    (1.0, 0.0),   // 5
    (0.0, 1.0),   // 6
    (-1.0, 0.0),  // 7
    (0.0, 0.0),   // 8
];

/// Evaluate the three 1-D quadratic Lagrange polynomials on [-1,1]
/// through nodes ξ=-1, ξ=0, ξ=+1:
/// L₀(ξ) = ξ(ξ−1)/2,  L₁(ξ) = 1−ξ²,  L₂(ξ) = ξ(ξ+1)/2
///
/// Returns [L0, L1, L2] and their derivatives [L0', L1', L2'].
#[inline]
fn q2_1d(x: f64) -> ([f64; 3], [f64; 3]) {
    let vals = [
        0.5 * x * (x - 1.0), // L₋₁ (node at -1)
        1.0 - x * x,         // L₀  (node at  0)
        0.5 * x * (x + 1.0), // L₊₁ (node at +1)
    ];
    let ders = [
        0.5 * (2.0 * x - 1.0), // L₋₁'
        -2.0 * x,              // L₀'
        0.5 * (2.0 * x + 1.0), // L₊₁'
    ];
    (vals, ders)
}

/// Map (ξᵢ, ηᵢ) → indices into the 1-D basis:
/// ξ coordinate: -1 → index 0, 0 → index 1, +1 → index 2
/// η coordinate: same.
#[inline]
fn coord_to_idx(c: f64) -> usize {
    if c < -0.5 {
        0
    } else if c > 0.5 {
        2
    } else {
        1
    }
}

impl ReferenceElement for QuadQ2 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        9
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, _) = q2_1d(x);
        let (ly, _) = q2_1d(y);
        for (i, &(xi_i, eta_i)) in Q2_NODES.iter().enumerate() {
            let ix = coord_to_idx(xi_i);
            let iy = coord_to_idx(eta_i);
            values[i] = lx[ix] * ly[iy];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, dlx) = q2_1d(x);
        let (ly, dly) = q2_1d(y);
        for (i, &(xi_i, eta_i)) in Q2_NODES.iter().enumerate() {
            let ix = coord_to_idx(xi_i);
            let iy = coord_to_idx(eta_i);
            grads[i * 2] = dlx[ix] * ly[iy]; // ∂φᵢ/∂ξ
            grads[i * 2 + 1] = lx[ix] * dly[iy]; // ∂φᵢ/∂η
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q2_NODES.iter().map(|&(x, y)| vec![x, y]).collect()
    }
}

// ─── Q3 ───────────────────────────────────────────────────────────────────────

/// Bicubic Lagrange element on the reference quad `[-1,1]²` — 16 DOFs.
///
/// DOF ordering (MFEM-compatible): vertices → bottom/right/top/left edges → interior.
/// 1-D nodes at ξ ∈ {-1, -1/3, 1/3, 1}.
pub struct QuadQ3;

fn lagrange_1d_p3(x: f64) -> ([f64; 4], [f64; 4]) {
    const N: [f64; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut v = [1.0; 4];
    let mut d = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            if j != i {
                v[i] *= (x - N[j]) / (N[i] - N[j]);
            }
        }
        let mut s = 0.0;
        for m in 0..4 {
            if m == i {
                continue;
            }
            let mut t = 1.0 / (N[i] - N[m]);
            for j in 0..4 {
                if j != i && j != m {
                    t *= (x - N[j]) / (N[i] - N[j]);
                }
            }
            s += t;
        }
        d[i] = s;
    }
    (v, d)
}

fn p3_dof(ix: usize, iy: usize) -> usize {
    const P: usize = 3;
    let (vx, vy) = (ix == 0 || ix == P, iy == 0 || iy == P);
    if vx && vy {
        match (ix, iy) {
            (0, 0) => 0,
            (P, 0) => 1,
            (P, P) => 2,
            (0, P) => 3,
            _ => 0,
        }
    } else if vx || vy {
        let base = 4;
        let n_edge = P - 1;
        if iy == 0 {
            base + (ix - 1)
        } else if ix == P {
            base + n_edge + (iy - 1)
        } else if iy == P {
            base + 2 * n_edge + (P - 1 - ix)
        } else {
            base + 3 * n_edge + (P - 1 - iy)
        }
    } else {
        4 + 4 * (P - 1) + (iy - 1) * (P - 1) + (ix - 1)
    }
}

impl ReferenceElement for QuadQ3 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        3
    }
    fn n_dofs(&self) -> usize {
        16
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, _) = lagrange_1d_p3(x);
        let (ly, _) = lagrange_1d_p3(y);
        for iy in 0..4 {
            for ix in 0..4 {
                values[p3_dof(ix, iy)] = lx[ix] * ly[iy];
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, dlx) = lagrange_1d_p3(x);
        let (ly, dly) = lagrange_1d_p3(y);
        for iy in 0..4 {
            for ix in 0..4 {
                let dof = p3_dof(ix, iy);
                grads[dof * 2] = dlx[ix] * ly[iy];
                grads[dof * 2 + 1] = lx[ix] * dly[iy];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let n = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
        let mut c = vec![[0.0; 2]; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                c[p3_dof(ix, iy)] = [n[ix], n[iy]];
            }
        }
        c.iter().map(|&[x, y]| vec![x, y]).collect()
    }
}

// ─── Q4 ───────────────────────────────────────────────────────────────────────

/// Biquartic Lagrange element on the reference quad `[-1,1]²` — 25 DOFs.
pub struct QuadQ4;

fn lagrange_1d_p4(x: f64) -> ([f64; 5], [f64; 5]) {
    const N: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let mut v = [1.0; 5];
    let mut d = [0.0; 5];
    for i in 0..5 {
        for j in 0..5 {
            if j != i {
                v[i] *= (x - N[j]) / (N[i] - N[j]);
            }
        }
        let mut s = 0.0;
        for m in 0..5 {
            if m == i {
                continue;
            }
            let mut t = 1.0 / (N[i] - N[m]);
            for j in 0..5 {
                if j != i && j != m {
                    t *= (x - N[j]) / (N[i] - N[j]);
                }
            }
            s += t;
        }
        d[i] = s;
    }
    (v, d)
}

fn p4_dof(ix: usize, iy: usize) -> usize {
    const P: usize = 4;
    let (vx, vy) = (ix == 0 || ix == P, iy == 0 || iy == P);
    if vx && vy {
        match (ix, iy) {
            (0, 0) => 0,
            (P, 0) => 1,
            (P, P) => 2,
            (0, P) => 3,
            _ => 0,
        }
    } else if vx || vy {
        let base = 4;
        let n_edge = P - 1;
        if iy == 0 {
            base + (ix - 1)
        } else if ix == P {
            base + n_edge + (iy - 1)
        } else if iy == P {
            base + 2 * n_edge + (P - 1 - ix)
        } else {
            base + 3 * n_edge + (P - 1 - iy)
        }
    } else {
        4 + 4 * (P - 1) + (iy - 1) * (P - 1) + (ix - 1)
    }
}

impl ReferenceElement for QuadQ4 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        4
    }
    fn n_dofs(&self) -> usize {
        25
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, _) = lagrange_1d_p4(x);
        let (ly, _) = lagrange_1d_p4(y);
        for iy in 0..5 {
            for ix in 0..5 {
                values[p4_dof(ix, iy)] = lx[ix] * ly[iy];
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, dlx) = lagrange_1d_p4(x);
        let (ly, dly) = lagrange_1d_p4(y);
        for iy in 0..5 {
            for ix in 0..5 {
                let dof = p4_dof(ix, iy);
                grads[dof * 2] = dlx[ix] * ly[iy];
                grads[dof * 2 + 1] = lx[ix] * dly[iy];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let n = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut c = vec![[0.0; 2]; 25];
        for iy in 0..5 {
            for ix in 0..5 {
                c[p4_dof(ix, iy)] = [n[ix], n[iy]];
            }
        }
        c.iter().map(|&[x, y]| vec![x, y]).collect()
    }
}

// ─── Serendipity elements (vertices + edges only, no interior) ───────────

use std::sync::OnceLock;

/// Monomials in the serendipity space S_p: ξⁱηʲ with i,j≤p and (i≤1 or j≤1).
fn serendipity_monomials(p: usize) -> Vec<(usize, usize)> {
    let mut m = Vec::new();
    for i in 0..=p {
        for j in 0..=p {
            if i <= 1 || j <= 1 {
                m.push((i, j));
            }
        }
    }
    m
}

/// Boundary DOF positions for serendipity quadrilateral of order p.
fn serendipity_dofs(p: usize) -> Vec<(f64, f64)> {
    let node_1d: Vec<f64> = (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect();
    let mut dofs = Vec::with_capacity(4 * p);
    // vertices
    dofs.push((node_1d[0], node_1d[0]));
    dofs.push((node_1d[p], node_1d[0]));
    dofs.push((node_1d[p], node_1d[p]));
    dofs.push((node_1d[0], node_1d[p]));
    // bottom edge (y=-1)
    for ix in 1..p {
        dofs.push((node_1d[ix], node_1d[0]));
    }
    // right edge (x=1)
    for iy in 1..p {
        dofs.push((node_1d[p], node_1d[iy]));
    }
    // top edge (y=1, reversed)
    for ix in (1..p).rev() {
        dofs.push((node_1d[ix], node_1d[p]));
    }
    // left edge (x=-1, reversed)
    for iy in (1..p).rev() {
        dofs.push((node_1d[0], node_1d[iy]));
    }
    assert_eq!(dofs.len(), 4 * p);
    dofs
}

/// Precomputed serendipity basis: stores the inverse Vandermonde matrix
/// mapping monomial coefficients → Lagrange basis values.
struct SerendipityData {
    coeffs: Vec<f64>, // [n_dofs][n_mono] row-major
    monomials: Vec<(usize, usize)>,
    dofs: Vec<(f64, f64)>,
}

impl SerendipityData {
    fn new(p: usize) -> Self {
        let dofs = serendipity_dofs(p);
        let n = dofs.len();
        let monomials = serendipity_monomials(p);
        assert_eq!(monomials.len(), n, "S_{p} dimension should be 4p");

        // Build V: V[k][s] = m_s(ξₖ, ηₖ)
        let mut v = vec![vec![0.0; n]; n];
        for (k, &(xk, yk)) in dofs.iter().enumerate() {
            for (s, &(i, j)) in monomials.iter().enumerate() {
                v[k][s] = xk.powi(i as i32) * yk.powi(j as i32);
            }
        }

        // Invert V (Gauss-Jordan).
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            inv[i][i] = 1.0;
        }
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&a, &b| v[a][col].abs().partial_cmp(&v[b][col].abs()).unwrap())
                .unwrap();
            v.swap(col, piv);
            inv.swap(col, piv);
            let scl = v[col][col];
            for j in 0..n {
                v[col][j] /= scl;
                inv[col][j] /= scl;
            }
            for row in 0..n {
                if row != col {
                    let f = v[row][col];
                    for j in 0..n {
                        v[row][j] -= f * v[col][j];
                        inv[row][j] -= f * inv[col][j];
                    }
                }
            }
        }
        // Store TRANSPOSED inverse so that eval_basis[k] = Σₛ coeffs[k*n+s]·mono_vals[s] works correctly.
        // Need: φₖ(ξₘ,ηₘ) = Σₛ coeffs[k][s]·V[m][s] = δ_{km} → coeffs[k][s] = (V⁻¹)[s][k]
        let mut coeffs = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                coeffs[i * n + j] = inv[j][i];
            }
        }
        Self {
            coeffs,
            monomials,
            dofs,
        }
    }

    fn eval_basis(&self, xi: f64, eta: f64, values: &mut [f64]) {
        let n = self.dofs.len();
        // Evaluate monomials at (xi, eta)
        let mut mono_vals = Vec::with_capacity(n);
        for &(i, j) in &self.monomials {
            mono_vals.push(xi.powi(i as i32) * eta.powi(j as i32));
        }
        // Compute basis values = coeffs × mono_vals
        for k in 0..n {
            let mut sum = 0.0;
            for s in 0..n {
                sum += self.coeffs[k * n + s] * mono_vals[s];
            }
            values[k] = sum;
        }
    }

    fn eval_grad_basis(&self, xi: f64, eta: f64, grads: &mut [f64]) {
        let n = self.dofs.len();
        let mut dm_dxi = Vec::with_capacity(n);
        let mut dm_deta = Vec::with_capacity(n);
        for &(i, j) in &self.monomials {
            let xi_p = xi.powi(i as i32);
            let eta_p = eta.powi(j as i32);
            dm_dxi.push(if i > 0 {
                (i as f64) * xi.powi((i - 1) as i32) * eta_p
            } else {
                0.0
            });
            dm_deta.push(if j > 0 {
                (j as f64) * xi_p * eta.powi((j - 1) as i32)
            } else {
                0.0
            });
        }
        for k in 0..n {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for s in 0..n {
                let c = self.coeffs[k * n + s];
                gx += c * dm_dxi[s];
                gy += c * dm_deta[s];
            }
            grads[k * 2] = gx;
            grads[k * 2 + 1] = gy;
        }
    }
}

fn sp_data(p: usize) -> &'static SerendipityData {
    static CACHE: [OnceLock<SerendipityData>; 5] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[p - 1].get_or_init(|| SerendipityData::new(p))
}

/// P1 serendipity quadrilateral — 4 DOFs at vertices. (Same as Q1.)
pub struct QuadP1;

impl ReferenceElement for QuadP1 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        4
    }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        values[0] = 0.25 * (1.0 - x) * (1.0 - y);
        values[1] = 0.25 * (1.0 + x) * (1.0 - y);
        values[2] = 0.25 * (1.0 + x) * (1.0 + y);
        values[3] = 0.25 * (1.0 - x) * (1.0 + y);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        grads[0] = -0.25 * (1.0 - y);
        grads[1] = -0.25 * (1.0 - x);
        grads[2] = 0.25 * (1.0 - y);
        grads[3] = -0.25 * (1.0 + x);
        grads[4] = 0.25 * (1.0 + y);
        grads[5] = 0.25 * (1.0 + x);
        grads[6] = -0.25 * (1.0 + y);
        grads[7] = 0.25 * (1.0 - x);
    }
    fn quadrature(&self, o: u8) -> QuadratureRule {
        quad_rule(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![-1., -1.], vec![1., -1.], vec![1., 1.], vec![-1., 1.]]
    }
}

macro_rules! impl_quad_p {
    ($name:ident, $p:literal) => {
        impl ReferenceElement for $name {
            fn dim(&self) -> u8 {
                2
            }
            fn order(&self) -> u8 {
                $p
            }
            fn n_dofs(&self) -> usize {
                4 * $p as usize
            }
            fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
                let data = sp_data($p);
                data.eval_basis(xi[0], xi[1], values);
            }
            fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
                let data = sp_data($p);
                data.eval_grad_basis(xi[0], xi[1], grads);
            }
            fn quadrature(&self, o: u8) -> QuadratureRule {
                quad_rule(o)
            }
            fn dof_coords(&self) -> Vec<Vec<f64>> {
                serendipity_dofs($p)
                    .iter()
                    .map(|&(x, y)| vec![x, y])
                    .collect()
            }
        }
    };
}

/// P2 serendipity quadrilateral — 8 DOFs.
pub struct QuadP2;
impl_quad_p!(QuadP2, 2);

/// P3 serendipity quadrilateral — 12 DOFs.
pub struct QuadP3;
impl_quad_p!(QuadP3, 3);

/// P4 serendipity quadrilateral — 16 DOFs.
pub struct QuadP4;
impl_quad_p!(QuadP4, 4);

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(4);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(4);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-12, "grad sum d={d} = {s}");
            }
        }
    }

    #[test]
    fn quad_q1_pou() {
        check_pou(&QuadQ1);
    }
    #[test]
    fn quad_q1_grad_zero() {
        check_grad_zero(&QuadQ1);
    }

    #[test]
    fn quad_q1_node_dofs() {
        let mut phi = vec![0.0; 4];
        for (i, &(x, y)) in Q1_NODES.iter().enumerate() {
            QuadQ1.eval_basis(&[x, y], &mut phi);
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-14,
                    "node {i}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    #[test]
    fn quad_q2_pou() {
        check_pou(&QuadQ2);
    }
    #[test]
    fn quad_q2_grad_zero() {
        check_grad_zero(&QuadQ2);
    }

    #[test]
    fn quad_q2_node_dofs() {
        let mut phi = vec![0.0; 9];
        for (i, &(x, y)) in Q2_NODES.iter().enumerate() {
            QuadQ2.eval_basis(&[x, y], &mut phi);
            for j in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    // ── QuadQ3 ────────────────────────────────────────────────────────────

    #[test]
    fn quad_q3_pou() {
        check_pou(&QuadQ3);
    }
    #[test]
    fn quad_q3_grad_zero() {
        check_grad_zero(&QuadQ3);
    }
    #[test]
    fn quad_q3_n_dofs() {
        assert_eq!(QuadQ3.n_dofs(), 16);
    }

    #[test]
    fn quad_q3_matches_quad_qk() {
        use crate::lagrange::factory::QuadQk;
        assert_eq!(QuadQ3.n_dofs(), QuadQk::new(3).n_dofs());
    }

    #[test]
    fn quad_q3_nodal_interp() {
        let coords = QuadQ3.dof_coords();
        let mut phi = vec![0.0; 16];
        for (i, c) in coords.iter().enumerate() {
            QuadQ3.eval_basis(&[c[0], c[1]], &mut phi);
            for j in 0..16 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12);
            }
        }
    }

    // ── QuadQ4 ────────────────────────────────────────────────────────────

    #[test]
    fn quad_q4_pou() {
        check_pou(&QuadQ4);
    }
    #[test]
    fn quad_q4_grad_zero() {
        check_grad_zero(&QuadQ4);
    }
    #[test]
    fn quad_q4_n_dofs() {
        assert_eq!(QuadQ4.n_dofs(), 25);
    }

    #[test]
    fn quad_q4_matches_quad_qk() {
        use crate::lagrange::factory::QuadQk;
        assert_eq!(QuadQ4.n_dofs(), QuadQk::new(4).n_dofs());
    }

    #[test]
    fn quad_q4_nodal_interp() {
        let coords = QuadQ4.dof_coords();
        let mut phi = vec![0.0; 25];
        for (i, c) in coords.iter().enumerate() {
            QuadQ4.eval_basis(&[c[0], c[1]], &mut phi);
            for j in 0..25 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-12);
            }
        }
    }

    // ── QuadP serendipity ─────────────────────────────────────────────────

    #[test]
    fn quad_p1_pou() {
        check_pou(&QuadP1);
    }
    #[test]
    fn quad_p1_grad_zero() {
        check_grad_zero(&QuadP1);
    }
    #[test]
    fn quad_p1_n_dofs() {
        assert_eq!(QuadP1.n_dofs(), 4);
    }

    #[test]
    fn quad_p2_pou() {
        check_pou(&QuadP2);
    }
    #[test]
    fn quad_p2_grad_zero() {
        check_grad_zero(&QuadP2);
    }
    #[test]
    fn quad_p2_n_dofs() {
        assert_eq!(QuadP2.n_dofs(), 8);
    }

    #[test]
    fn quad_p3_pou() {
        check_pou(&QuadP3);
    }
    #[test]
    fn quad_p3_grad_zero() {
        check_grad_zero(&QuadP3);
    }
    #[test]
    fn quad_p3_n_dofs() {
        assert_eq!(QuadP3.n_dofs(), 12);
    }

    #[test]
    fn quad_p4_pou() {
        check_pou(&QuadP4);
    }
    #[test]
    fn quad_p4_grad_zero() {
        check_grad_zero(&QuadP4);
    }
    #[test]
    fn quad_p4_n_dofs() {
        assert_eq!(QuadP4.n_dofs(), 16);
    }

    #[test]
    fn quad_p2_nodal_interp() {
        let c = QuadP2.dof_coords();
        let mut phi = vec![0.0; 8];
        for (i, coord) in c.iter().enumerate() {
            QuadP2.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..8 {
                assert!((phi[j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn quad_p3_nodal_interp() {
        let c = QuadP3.dof_coords();
        let mut phi = vec![0.0; 12];
        for (i, coord) in c.iter().enumerate() {
            QuadP3.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..12 {
                assert!((phi[j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn quad_p4_nodal_interp() {
        let c = QuadP4.dof_coords();
        let mut phi = vec![0.0; 16];
        for (i, coord) in c.iter().enumerate() {
            QuadP4.eval_basis(&[coord[0], coord[1]], &mut phi);
            for j in 0..16 {
                assert!((phi[j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
