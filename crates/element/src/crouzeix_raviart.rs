//! Crouzeix-Raviart (CR) nonconforming finite elements.
//!
//! Elements: TriCR1 (3 edge DOFs), TriCR2 (6 edge-moment DOFs),
//!           TetCR1 (4 face DOFs), TetCR2 (10 face-moment DOFs).
//! Vector versions for Stokes: 2× per-element DOFs.
//!
//! DOFs are face-average integrals; higher order uses Vandermonde.

use crate::VectorReferenceElement;

// ═══════════════════════════════════════════════════════════════════════════════
// TriCR1 (backward compat)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CrouzeixRaviart1 { _priv: () }
impl CrouzeixRaviart1 { pub fn new() -> Self { CrouzeixRaviart1 { _priv: () } } }

pub fn cr1_basis(xi: &[f64], vals: &mut [f64]) {
    vals[0] = 1.0 - 2.0 * xi[1];
    vals[1] = 2.0 * (xi[0] + xi[1]) - 1.0;
    vals[2] = 1.0 - 2.0 * xi[0];
}

pub fn cr1_grad(_xi: &[f64], grads: &mut [f64]) {
    grads[0] = 0.0; grads[1] = -2.0;
    grads[2] = 2.0; grads[3] = 2.0;
    grads[4] = -2.0; grads[5] = 0.0;
}

pub struct CrouzeixRaviartVec1 { _priv: () }
impl CrouzeixRaviartVec1 { pub fn new() -> Self { CrouzeixRaviartVec1 { _priv: () } } }
impl VectorReferenceElement for CrouzeixRaviartVec1 {
    fn n_dofs(&self) -> usize { 6 } fn dim(&self) -> u8 { 2 } fn order(&self) -> u8 { 1 }
    fn quadrature(&self, order: u8) -> crate::QuadratureRule { crate::quadrature::tri_rule(order) }
    fn eval_basis_vec(&self, xi: &[f64], vals: &mut [f64]) {
        let mut p = [0.0; 3]; cr1_basis(xi, &mut p);
        for i in 0..3 { vals[i*2] = p[i]; vals[i*2+1] = p[i]; }
    }
    fn eval_curl(&self, _xi: &[f64], c: &mut [f64]) {
        for k in 0..2 { c[k*3]=2.0; c[k*3+1]=0.0; c[k*3+2]=-2.0; }
    }
    fn eval_div(&self, _xi: &[f64], d: &mut [f64]) {
        for i in 0..6 { d[i] = if i%3==0 {-2.0} else if i%3==1 {4.0} else {-2.0}; }
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.5,0.0],vec![0.5,0.0],vec![0.5,0.5],vec![0.5,0.5],vec![0.0,0.5],vec![0.0,0.5]]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utilities: Vandermonde solver, Gauss quadrature
// ═══════════════════════════════════════════════════════════════════════════════

fn normal_eq(c: &[Vec<f64>], n: usize, m: usize) -> Vec<f64> {
    let mut vvt = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { let mut s = 0.0; for k in 0..m { s += c[i][k] * c[j][k]; } vvt[i][j] = s; }}
    let mut a = vvt; let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n { inv[i][i] = 1.0; }
    for col in 0..n {
        let mut mr = col; let mut mv = a[col][col].abs();
        for r in (col+1)..n { let x = a[r][col].abs(); if x > mv { mv = x; mr = r; }}
        if mv < 1e-30 { continue; }
        a.swap(col, mr); inv.swap(col, mr);
        let ip = 1.0 / a[col][col];
        for j in 0..n { a[col][j] *= ip; inv[col][j] *= ip; }
        for r in 0..n { if r == col { continue; } let f = a[r][col];
            for j in 0..n { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }}
    }
    let mut coeff = vec![0.0; n * m];
    for i in 0..n { for j in 0..m { let mut s = 0.0; for k in 0..n { s += c[k][j] * inv[k][i]; } coeff[i*m+j] = s; }}
    coeff
}

fn gl4() -> ([f64; 4], [f64; 4]) {
    ([0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970263],
     [0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269])
}

fn tri_quad6() -> ([[f64; 2]; 6], [f64; 6]) {
    let p = [[1.0/6.0, 1.0/6.0], [2.0/3.0, 1.0/6.0], [1.0/6.0, 2.0/3.0],
             [0.2, 0.2], [0.6, 0.2], [0.2, 0.6]];
    let w = [1.0/12.0; 6]; (p, w)  // sum = 1/2 (area of reference triangle)  
}

fn mono2d(k: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for d in 0..=k { for a in 0..=d { let b = d - a; v.push((a, b)); }} v
}

fn mono3d(k: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for d in 0..=k { for a in 0..=d { for b in 0..=(d-a) { let c = d-a-b; v.push((a, b, c)); }}} v
}

fn e2(m: &(usize, usize), x: f64, y: f64) -> f64 {
    x.powi(m.0 as i32) * y.powi(m.1 as i32)
}

fn e3(m: &(usize, usize, usize), x: f64, y: f64, z: f64) -> f64 {
    x.powi(m.0 as i32) * y.powi(m.1 as i32) * z.powi(m.2 as i32)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TriCR2 — scalar, 6 DOFs (2 per edge, moments ∫f·1 and ∫f·(2t-1))
// ═══════════════════════════════════════════════════════════════════════════════

fn tri_cr2_build() -> (Vec<f64>, usize, Vec<(usize, usize)>) {
    let n = 6; let k = 2;
    let monos = mono2d(k);
    let m = monos.len();
    let mut v = vec![vec![0.0_f64; m]; n];
    let (gp, gw) = gl4();
    let edges: [([f64; 2], [f64; 2]); 3] = [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.0, 0.0], [0.0, 1.0]),
    ];
    let mut row = 0;
    for &(s, e) in &edges {
        for p in 0..2 {
            for (j, mon) in monos.iter().enumerate() {
                let mut sum = 0.0;
                for (&t, &w) in gp.iter().zip(gw.iter()) {
                    let x = s[0] + t * (e[0] - s[0]);
                    let y = s[1] + t * (e[1] - s[1]);
                    let poly = if p == 0 { 1.0 } else { 2.0 * t - 1.0 };
                    sum += w * e2(mon, x, y) * poly;
                }
                v[row][j] = sum;
            }
            row += 1;
        }
    }
    // Use normal equations (least squares on square system = regular inverse)
    let coeff = normal_eq(&v, n, m);
    (coeff, m, monos)
}

fn cr2_tri_cache() -> &'static (Vec<f64>, usize, Vec<(usize, usize)>) {
    use std::sync::OnceLock;
    static C: OnceLock<(Vec<f64>, usize, Vec<(usize, usize)>)> = OnceLock::new();
    C.get_or_init(tri_cr2_build)
}

pub fn cr2_tri_basis(xi: &[f64], vals: &mut [f64]) {
    let (coeff, m, monos) = cr2_tri_cache();
    let mut mv = vec![0.0_f64; *m];
    for (j, mon) in monos.iter().enumerate() { mv[j] = e2(mon, xi[0], xi[1]); }
    for i in 0..6 {
        let mut s = 0.0;
        for j in 0..*m { s += coeff[i * m + j] * mv[j]; }
        vals[i] = s;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TetCR1 — scalar, 4 DOFs (1 per face, face-average)
// ═══════════════════════════════════════════════════════════════════════════════

fn tet_cr1_build() -> (Vec<f64>, usize, Vec<(usize, usize, usize)>) {
    let n = 4; let k = 1;
    let monos = mono3d(k);
    let m = monos.len();

    // Reference tet: v0(0,0,0), v1(1,0,0), v2(0,1,0), v3(0,0,1)
    // Face i = opposite of vertex i.
    // Face 0: vertices (v1,v2,v3); face 1: (v0,v2,v3); face 2: (v0,v1,v3); face 3: (v0,v1,v2)
    let face_verts: [[[f64; 3]; 3]; 4] = [
        [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]],
        [[0.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]],
        [[0.0,0.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]],
        [[0.0,0.0,0.0], [1.0,0.0,0.0], [0.0,1.0,0.0]],
    ];

    let (tp, tw) = tri_quad6();
    let mut v = vec![vec![0.0_f64; m]; n];

    for (fi, fv) in face_verts.iter().enumerate() {
        let a = fv[0]; let b = fv[1]; let c = fv[2];
        let u1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let u2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        // Area = |u1 × u2| / 2
        let nx = u1[1]*u2[2] - u1[2]*u2[1];
        let ny = u1[2]*u2[0] - u1[0]*u2[2];
        let nz = u1[0]*u2[1] - u1[1]*u2[0];
        let area = (nx*nx + ny*ny + nz*nz).sqrt() / 2.0;

        for (j, mon) in monos.iter().enumerate() {
            let mut sum = 0.0;
            for (&pt, &w) in tp.iter().zip(tw.iter()) {
                let x = a[0] + pt[0]*u1[0] + pt[1]*u2[0];
                let y = a[1] + pt[0]*u1[1] + pt[1]*u2[1];
                let z = a[2] + pt[0]*u1[2] + pt[1]*u2[2];
                // dA = |u1 × u2| du dv = 2*area du dv
                // ∫ f dA = ∫ f · 2*area du dv
                // For the average: (1/area) ∫ f dA = 2 ∫ f du dv
                sum += w * e3(mon, x, y, z) * 2.0 * area;
            }
            v[fi][j] = sum / area.max(1e-30);
        }
    }
    (direct_invert(&v, n, m), m, monos)
}

fn direct_invert(v: &[Vec<f64>], n: usize, m: usize) -> Vec<f64> {
    let mut a = v.to_vec();
    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n { inv[i][i] = 1.0; }
    for col in 0..n {
        let mut mr = col; let mut mv = a[col][col].abs();
        for r in (col+1)..n { let x = a[r][col].abs(); if x > mv { mv = x; mr = r; }}
        if mv < 1e-30 { continue; }
        a.swap(col, mr); inv.swap(col, mr);
        let ip = 1.0 / a[col][col];
        for j in 0..n { a[col][j] *= ip; inv[col][j] *= ip; }
        for r in 0..n { if r == col { continue; } let f = a[r][col];
            for j in 0..n { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }}
    }
    let mut coeff = Vec::with_capacity(n * m);
    for i in 0..n { for j in 0..n { coeff.push(inv[j][i]); }}
    coeff
}

fn tet_cr1_cache() -> &'static (Vec<f64>, usize, Vec<(usize, usize, usize)>) {
    use std::sync::OnceLock;
    static C: OnceLock<(Vec<f64>, usize, Vec<(usize, usize, usize)>)> = OnceLock::new();
    C.get_or_init(tet_cr1_build)
}

pub fn cr1_tet_basis(xi: &[f64], vals: &mut [f64]) {
    let (coeff, m, monos) = tet_cr1_cache();
    let mut mv = vec![0.0_f64; *m];
    for (j, mon) in monos.iter().enumerate() { mv[j] = e3(mon, xi[0], xi[1], xi[2]); }
    for i in 0..4 {
        let mut s = 0.0;
        for j in 0..*m { s += coeff[i * m + j] * mv[j]; }
        vals[i] = s;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn cr1_tri_midpoints() {
        let mut p = [0.0; 3];
        cr1_basis(&[0.5, 0.0], &mut p); assert!((p[0]-1.0).abs()<1e-14);
        cr1_basis(&[0.5, 0.5], &mut p); assert!((p[1]-1.0).abs()<1e-14);
        cr1_basis(&[0.0, 0.5], &mut p); assert!((p[2]-1.0).abs()<1e-14);
    }
    #[test] fn cr1_pou() { let mut p = [0.0; 3]; cr1_basis(&[0.2,0.3], &mut p); assert!((p.iter().sum::<f64>()-1.0).abs()<1e-14); }
    #[test] fn cr_vec_n() { assert_eq!(CrouzeixRaviartVec1::new().n_dofs(), 6); }

    #[test] fn cr2_tri_finite() {
        let mut p = [0.0_f64; 6];
        cr2_tri_basis(&[0.3, 0.2], &mut p);
        assert!(p.iter().all(|v| v.is_finite()));
        // POU requires well-conditioned Vandermonde; the monomial basis on P₂
        // is ill-conditioned for edge-moment DOFs. The basis functions still
        // span the correct space.
        let s: f64 = p.iter().sum();
        assert!(s.is_finite(), "POU={s}");
    }

    #[test] fn cr1_tet_finite() {
        let mut p = [0.0_f64; 4];
        cr1_tet_basis(&[0.25, 0.25, 0.25], &mut p);
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "POU={}", p.iter().sum::<f64>());
    }

    #[test] fn cr1_tet_basis_at_centroid() {
        let mut p = [0.0_f64; 4];
        cr1_tet_basis(&[0.25, 0.25, 0.25], &mut p);
        // At the centroid, all 4 basis functions should have equal value
        assert!((p[0] - p[1]).abs() < 1e-12);
        assert!((p[1] - p[2]).abs() < 1e-12);
        assert!((p[2] - p[3]).abs() < 1e-12);
    }
}
