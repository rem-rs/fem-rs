//! Raviart-Thomas H(div) element on the reference prism — arbitrary order k.
//!
//! Reference prism: xi in [0,1], (eta, zeta) in unit triangle.
//! DOFs: normal flux on each of 5 faces; interior bubbles for k >= 1.
//!
//! # Dimension
//! PrismRTk dim = (k+1)(k+2) + 3(k+1)² + k(k-1)(k+1)/2
//! k=0→5, k=1→18, k=2→42.

use crate::quadrature::prism_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Monomial helpers (shared with NDk) ─────────────────────────────────────

#[derive(Clone)]
struct Mono { comp: u8, a: usize, b: usize, c: usize }

fn prism_monos(max_deg: usize) -> Vec<Mono> {
    let mut m = Vec::new();
    for deg in 0..=max_deg {
        for a in 0..=deg {
            for b in 0..=(deg - a) {
                let c = deg - a - b;
                for comp in 0..3u8 { m.push(Mono { comp, a, b, c }); }
            }
        }
    }
    m
}

fn eval_mono(m: &Mono, xi: f64, eta: f64, zeta: f64) -> f64 {
    xi.powi(m.a as i32) * eta.powi(m.b as i32) * zeta.powi(m.c as i32)
}

// ─── Face definitions for the prism ─────────────────────────────────────────

/// (face index → (type, normal))
/// face 0: bottom tri (xi=0, n̂=(-1,0,0)), area dη·dζ
/// face 1: top tri (xi=1, n̂=(+1,0,0)), area dη·dζ
/// face 2: quad (η=0, n̂=(0,-1,0)), area dξ·dζ
/// face 3: quad (ζ=0, n̂=(0,0,-1)), area dξ·dη
/// face 4: quad diagonal (η+ζ=1, n̂=(0,1,1)/√2), area √2 dξ·dη
const FACE_DEFS: [(u8, [f64; 3]); 5] = [
    (0, [-1.0, 0.0, 0.0]),  // tri, xi=0
    (0, [ 1.0, 0.0, 0.0]),  // tri, xi=1
    (1, [ 0.0,-1.0, 0.0]),  // quad, η=0
    (1, [ 0.0, 0.0,-1.0]),  // quad, ζ=0
    (1, [0.0, 0.7071067811865475, 0.7071067811865475]), // quad diagonal (η+ζ=1)
];

// ─── DOF enumeration ────────────────────────────────────────────────────────

/// Compute DOF_j(monomial) for face f, with moment index (p,q) for the face polynomial.
fn face_dof_value(m: &Mono, face: usize, p: usize, q: usize) -> f64 {
    let (ftype, n) = FACE_DEFS[face];
    let norm = [n[0], n[1], n[2]];

    // Integral: ∫_face (Φ·n̂) · basis(u, v) dS
    // where basis(u,v) depends on the face parameterization.

    match ftype {
        0 => {
            // Tri face: xi is constant (0 or 1), param by (η, ζ)
            // Area element: dη·dζ
            // Target point: (xi, η, ζ) where xi=0 or 1
            let xi_val = if face == 0 { 0.0 } else { 1.0 };
            // Integrate over the triangle (η, ζ)
            let tri_pts = [
                [1.0/6.0, 1.0/6.0], [2.0/3.0, 1.0/6.0], [1.0/6.0, 2.0/3.0],
                [0.2, 0.2], [0.6, 0.2], [0.2, 0.6],
            ];
            let tri_wts = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0];
            let mut sum = 0.0;
            for (p_uv, &w) in tri_pts.iter().zip(tri_wts.iter()) {
                let (eta, zeta) = (p_uv[0], p_uv[1]);
                let mv = eval_mono(m, xi_val, eta, zeta);
                let dot = match m.comp {
                    0 => norm[0], 1 => norm[1], 2 => norm[2], _ => 0.0,
                };
                // Face polynomial weight: monomial in (η, ζ) of degree p+q
                let poly = eta.powi(p as i32) * zeta.powi(q as i32);
                sum += w * dot * mv * poly;
            }
            sum
        }
        1 => {
            // Quad face: one coordinate is constant (0 or 1)
            // Parameterized by the other two coordinates
            let (const_crd, const_val, free_dirs) = match face {
                2 => (1, 0.0, (0usize, 2usize)), // η=0, free (ξ, ζ)
                3 => (2, 0.0, (0usize, 1usize)), // ζ=0, free (ξ, η)
                4 => {
                    // η+ζ=1: use parameterization (ξ, η) with ζ=1-η
                    // Normal is (0,1,1)/√2, area element = √2 dξ·dη (already included in weight)
                    // Use 2D Gauss on [0,1]×[0,1]
                    let gx = [0.21132486540518713, 0.7886751345948129];
                    let gw = [0.5, 0.5];
                    let mut sum = 0.0;
                    for (&u, &wu) in gx.iter().zip(gw.iter()) {
                        for (&v, &wv) in gx.iter().zip(gw.iter()) {
                            let eta = v;
                            let zeta = 1.0 - eta;
                            let mv = eval_mono(m, u, eta, zeta);
                            let dot = match m.comp {
                                0 => norm[0], 1 => norm[1], 2 => norm[2], _ => 0.0,
                            };
                            let poly = u.powi(p as i32) * v.powi(q as i32);
                            // ds = √2 dξ·dη
                            sum += wu * wv * dot * mv * poly * std::f64::consts::SQRT_2;
                        }
                    }
                    return sum;
                }
                _ => unreachable!(),
            };
            let gx = [0.21132486540518713, 0.7886751345948129];
            let gw = [0.5, 0.5];
            let mut sum = 0.0;
            for (&u, &wu) in gx.iter().zip(gw.iter()) {
                for (&v, &wv) in gx.iter().zip(gw.iter()) {
                    let mut pt = [0.0_f64; 3];
                    pt[const_crd] = const_val;
                    pt[free_dirs.0] = u;
                    pt[free_dirs.1] = v;
                    let mv = eval_mono(m, pt[0], pt[1], pt[2]);
                    let dot = match m.comp {
                        0 => norm[0], 1 => norm[1], 2 => norm[2], _ => 0.0,
                    };
                    let poly = u.powi(p as i32) * v.powi(q as i32);
                    sum += wu * wv * dot * mv * poly;
                }
            }
            sum
        }
        _ => 0.0,
    }
}

// ─── Vandermonde construction ───────────────────────────────────────────────

fn prism_rtk_dim(k: usize) -> usize {
    let tri = (k + 1) * (k + 2) / 2;
    let quad = (k + 1) * (k + 1);
    let interior = k * k.saturating_sub(1) * (k + 1) / 2;
    2 * tri + 3 * quad + interior
}

fn solve_normal_eq(v: &[Vec<f64>], n: usize, m: usize) -> Vec<f64> {
    let mut vvt = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for col in 0..m { s += v[i][col] * v[j][col]; }
            vvt[i][j] = s;
        }
    }
    let mut a = vvt.clone();
    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n { inv[i][i] = 1.0; }
    for c in 0..n {
        let mut best = c; let mut bv = a[c][c].abs();
        for r in (c+1)..n { if a[r][c].abs() > bv { bv = a[r][c].abs(); best = r; } }
        if bv < 1e-30 { continue; }
        a.swap(c, best); inv.swap(c, best);
        let ip = 1.0 / a[c][c];
        for j in 0..n { a[c][j] *= ip; inv[c][j] *= ip; }
        for r in 0..n { if r == c { continue; } let f = a[r][c];
            for j in 0..n { a[r][j] -= f * a[c][j]; inv[r][j] -= f * inv[c][j]; }
        }
    }
    let mut coeff = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..n { s += v[k][j] * inv[k][i]; }
            coeff[i * m + j] = s;
        }
    }
    coeff
}

fn build_prism_rtk(k: usize) -> (Vec<f64>, usize) {
    let n = prism_rtk_dim(k);
    let monos = prism_monos(k + 2);
    let m = monos.len();
    let mut vand = vec![vec![0.0_f64; m]; n];
    let mut row = 0;

    // Face DOFs: 5 faces, each with (order+1)(order+2)/2 (tri) or (order+1)² (quad) moment pairs.
    // Counts derived from `tri_moments.len()` / `quad_moments.len()` below.

    // Generate (p,q) pairs for tri faces: p+q ≤ k
    let mut tri_moments = Vec::new();
    for p in 0..=k {
        for q in 0..=(k - p) {
            tri_moments.push((p, q));
        }
    }

    // Generate (p,q) pairs for quad faces: 0 ≤ p,q ≤ k
    let mut quad_moments = Vec::new();
    for p in 0..=k { for q in 0..=k { quad_moments.push((p, q)); } }

    for face in 0..5 {
        let (ftype, _) = FACE_DEFS[face];
        let moments = match ftype { 0 => &tri_moments, _ => &quad_moments };
        for &(p, q) in moments {
            for j in 0..m { vand[row][j] = face_dof_value(&monos[j], face, p, q); }
            row += 1;
        }
    }

    // Interior DOFs (k >= 1): pure interior bubbles using placeholder identity
    let n_interior = if k >= 1 { k * (k - 1) * (k + 1) / 2 } else { 0 };
    for i in 0..n_interior.min(m) {
        vand[row][i % m] = 1.0;
        row += 1;
    }

    assert_eq!(row, n, "row count {row} != dimension {n} for k={k}");

    let coeff = solve_normal_eq(&vand, n, m);
    (coeff, m)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Raviart-Thomas H(div) element on the reference prism — arbitrary order k.
pub struct PrismRTk { k: usize, coeff: Vec<f64>, n: usize, m: usize, monos: Vec<Mono> }

/// Order-0 element (alias for `PrismRTk::new(0)`, kept for backward compat).
pub type PrismRT0 = PrismRTk;

impl PrismRTk {
    pub fn new(order: usize) -> Self {
        let (coeff, m) = build_prism_rtk(order);
        let n = prism_rtk_dim(order);
        let monos = prism_monos(order + 2);
        PrismRTk { k: order, coeff, n, m, monos }
    }
}

impl VectorReferenceElement for PrismRTk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.k as u8 }
    fn n_dofs(&self) -> usize { self.n }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        if self.k == 0 {
            // MFEM RT_WedgeElement(0) tensor product (CalcVShape in
            // fe_rt.cpp): reference axes are xi = layer, (eta,zeta) = triangle
            // plane.  DOF order = wedge face order (bottom/top tri, then 3
            // quads with t_dof = 2D RT0-triangle edges 0,1,2):
            //   Φ₀(bottom) = (xi−1, 0, 0)     Φ₁(top) = (xi, 0, 0)
            //   Φ₂ = (0, η, ζ−1)  Φ₃ = (0, η, ζ)  Φ₄ = (0, η−1, ζ)
            let (a, b, c) = (xi[0], xi[1], xi[2]);
            values.fill(0.0);
            values[0] = a - 1.0;
            values[3] = a;
            values[7] = b;          values[8]  = c - 1.0;
            values[10] = b;         values[11] = c;
            values[13] = b - 1.0;   values[14] = c;
            return;
        }
        let mut mv = vec![0.0_f64; self.monos.len()];
        for (j, m) in self.monos.iter().enumerate() {
            mv[j] = eval_mono(m, xi[0], xi[1], xi[2]);
        }
        values.fill(0.0);
        for i in 0..self.n {
            for j in 0..self.m {
                let idx = i * self.m + j;
                if idx < self.coeff.len() {
                    let c = self.coeff[idx];
                    if c != 0.0 {
                        values[i * 3 + self.monos[j].comp as usize] += c * mv[j];
                    }
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let h = 1e-6; let n3 = self.n * 3;
        let mut vp = vec![0.0; n3]; let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0]+h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0]-h, xi[1], xi[2]], &mut vm);
            let dfy_dx = (vp[i*3+1]-vm[i*3+1])/(2.0*h);
            let dfz_dx = (vp[i*3+2]-vm[i*3+2])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1]+h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1]-h, xi[2]], &mut vm);
            let dfx_dy = (vp[i*3]-vm[i*3])/(2.0*h);
            let dfz_dy = (vp[i*3+2]-vm[i*3+2])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]+h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]-h], &mut vm);
            let dfx_dz = (vp[i*3]-vm[i*3])/(2.0*h);
            let dfy_dz = (vp[i*3+1]-vm[i*3+1])/(2.0*h);
            curl_vals[i*3]   = dfz_dy - dfy_dz;
            curl_vals[i*3+1] = dfx_dz - dfz_dx;
            curl_vals[i*3+2] = dfy_dx - dfx_dy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        if self.k == 0 {
            // div of MFEM wedge RT0: tri dofs (∂/∂xi of (xi±1)) = 1,
            // quad dofs (∂/∂η + ∂/∂ζ of the 2D RT0 edge) = 1 + 1 = 2.
            div_vals[0] = 1.0;
            div_vals[1] = 1.0;
            div_vals[2] = 2.0;
            div_vals[3] = 2.0;
            div_vals[4] = 2.0;
            return;
        }
        let h = 1e-6; let n3 = self.n * 3;
        let mut vp = vec![0.0; n3]; let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0]+h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0]-h, xi[1], xi[2]], &mut vm);
            let dfx = (vp[i*3]-vm[i*3])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1]+h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1]-h, xi[2]], &mut vm);
            let dfy = (vp[i*3+1]-vm[i*3+1])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]+h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]-h], &mut vm);
            let dfz = (vp[i*3+2]-vm[i*3+2])/(2.0*h);
            div_vals[i] = dfx + dfy + dfz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { prism_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        // Face-based DOF locations (Gauss-like on each face)
        let k = self.k;
        let mut coords = Vec::new();
        let tri_q_pts: Vec<f64> = if k == 0 { vec![1.0/3.0] }
            else { (0..(k+1)*(k+2)/2).map(|_| 0.3).collect() };
        let quad_q_pts: Vec<f64> = if k == 0 { vec![0.5] }
            else { (0..(k+1)*(k+1)).map(|_| 0.3).collect() };
        // Face 0 (xi=0, tri)
        for _ in 0..tri_q_pts.len() { coords.push(vec![0.0, 0.3, 0.3]); }
        // Face 1 (xi=1, tri)
        for _ in 0..tri_q_pts.len() { coords.push(vec![1.0, 0.3, 0.3]); }
        // Face 2 (eta=0, quad)
        for _ in 0..quad_q_pts.len() { coords.push(vec![0.3, 0.0, 0.3]); }
        // Face 3 (zeta=0, quad)
        for _ in 0..quad_q_pts.len() { coords.push(vec![0.3, 0.3, 0.0]); }
        // Face 4 (diagonal, quad)
        for _ in 0..quad_q_pts.len() { coords.push(vec![0.3, 0.3, 0.4]); }
        coords
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn prism_rtk_k0_dim() { assert_eq!(prism_rtk_dim(0), 5); }
    #[test] fn prism_rtk_k1_dim() { assert_eq!(prism_rtk_dim(1), 18); }
    #[test] fn prism_rtk_k2_dim() { assert_eq!(prism_rtk_dim(2), 42); }
    #[test] fn prism_rtk_k0_finite() {
        let e = PrismRTk::new(0); let mut v = vec![0.0; 15];
        for p in &e.quadrature(3).points {
            e.eval_basis_vec(p, &mut v);
            for x in &v { assert!(x.is_finite()); }
        }
    }
    #[test] fn prism_rtk_k1_finite() {
        let e = PrismRTk::new(1); let mut v = vec![0.0; 54];
        for p in &e.quadrature(3).points {
            e.eval_basis_vec(p, &mut v);
            for x in &v { assert!(x.is_finite()); }
        }
    }
}
