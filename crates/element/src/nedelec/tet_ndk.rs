//! Arbitrary-order Nedelec-I on reference tetrahedron via Vandermonde.
//! N_k: dim = k(k+2)(k+3)/2. DOFs: edge k×6, face k(k−1)×4, volume interior.

use std::sync::OnceLock;
use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

struct TetNDkData { coeff: Vec<f64>, n: usize, monomap: Vec<usize> }

fn beta_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 1).fold(1.0, |p, i| p * i as f64)
}

fn tet_int(a: usize, b: usize, c: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        * (1..=c).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + c + 3).fold(1.0, |p, i| p * i as f64)
}

fn tet_data(k: usize) -> &'static TetNDkData {
    static CACHE: [OnceLock<TetNDkData>; 9] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    CACHE[k - 1].get_or_init(|| {
        let n = k * (k + 2) * (k + 3) / 2;
        struct M { c: usize, a: usize, b: usize, d: usize }
        let mut ms = Vec::new();
        for deg in 0..=k { for a in 0..=deg { for b in 0..=(deg - a) { let d = deg - a - b; for c in 0..3 { ms.push(M { c, a, b, d }); } } } }
        let mt = ms.len();
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0usize;
        // Edges: 6 edges × k DOFs
        let edge_def: [(usize, usize); 6] = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];
        for &(va, vb) in &edge_def {
            for p in 0..k {
                for (j, m) in ms.iter().enumerate() {
                    let val = match (va, vb, m.c) {
                        (0, 1, 0) if m.b == 0 && m.d == 0 => 1.0 / (m.a as f64 + p as f64 + 1.0),
                        (0, 2, 1) if m.a == 0 && m.d == 0 => 1.0 / (m.b as f64 + p as f64 + 1.0),
                        (0, 3, 2) if m.a == 0 && m.b == 0 => 1.0 / (m.d as f64 + p as f64 + 1.0),
                        (1, 2, 0) if m.d == 0 => -beta_int(m.a, m.b + p),
                        (1, 2, 1) if m.d == 0 => beta_int(m.a, m.b + p),
                        (1, 3, 0) if m.b == 0 => -beta_int(m.a, m.d + p),
                        (1, 3, 2) if m.b == 0 => beta_int(m.a, m.d + p),
                        (2, 3, 1) if m.a == 0 => -beta_int(m.b, m.d + p),
                        (2, 3, 2) if m.a == 0 => beta_int(m.b, m.d + p),
                        _ => 0.0,
                    };
                    v[di][j] = val;
                }
                di += 1;
            }
        }
        // Faces: 4 faces × k(k-1) DOFs — use Gaussian quadrature on the face
        if k >= 2 {
            // Face map: (x_fn, y_fn, z_fn, t1x,t1y,t1z, t2x,t2y,t2z)
            // Face 0: vertices (1,2,3) = (1,0,0),(0,1,0),(0,0,1). Param: x=1-u-v, y=u, z=v
            // Face 1: vertices (0,2,3) = (0,0,0),(0,1,0),(0,0,1). Param: x=0, y=u, z=v
            // Face 2: vertices (0,1,3) = (0,0,0),(1,0,0),(0,0,1). Param: x=u, y=0, z=v
            // Face 3: vertices (0,1,2) = (0,0,0),(1,0,0),(0,1,0). Param: x=u, y=v, z=0
            // Actually the face vertices differ from TriNDk's face definition.
            // Let me use the more careful mapping.
            // Face 0: vertices (1,2,3) = (1,0,0),(0,1,0),(0,0,1). Param: x=1-u-v, y=u, z=v
            // Face 1: vertices (0,2,3) = (0,0,0),(0,1,0),(0,0,1). Param: x=0, y=u, z=v
            // Face 2: vertices (0,1,3) = (0,0,0),(1,0,0),(0,0,1). Param: x=u, y=0, z=v
            // Face 3: vertices (0,1,2) = (0,0,0),(1,0,0),(0,1,0). Param: x=u, y=v, z=0
            // Tangents for each face: t1 = ∂/∂u, t2 = ∂/∂v
            let face_map: [(fn(f64,f64)->f64, fn(f64,f64)->f64, fn(f64,f64)->f64,
                            f64,f64,f64, f64,f64,f64); 4] = [
                (|u,v| 1.0-u-v, |u,_v| u, |_u,v| v, -1.0,1.0,0.0, -1.0,0.0,1.0),
                (|_,_| 0.0,     |u,_v| u, |_u,v| v,  0.0,1.0,0.0,  0.0,0.0,1.0),
                (|u,_v| u,      |_,_| 0.0, |_u,v| v, 1.0,0.0,0.0,  0.0,0.0,1.0),
                (|u,_v| u,      |_u,v| v, |_,_| 0.0, 1.0,0.0,0.0,  0.0,1.0,0.0),
            ];
            let qr = crate::quadrature::tri_rule_arbitrary((2 * k) as u8);
            for fi in 0..4 {
                let (xf, yf, zf, t1x, t1y, t1z, t2x, t2y, t2z) = face_map[fi];
                let tans = [(t1x, t1y, t1z), (t2x, t2y, t2z)];
                for deg in 0..=(k - 2) {
                    for ix in 0..=deg {
                        let iy = deg - ix;
                        for &(tx, ty, tz) in &tans {
                            for (j, m) in ms.iter().enumerate() {
                                let mut sum = 0.0;
                                for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                                    let (u, v) = (qp[0], qp[1]);
                                    let (x, y, z) = (xf(u, v), yf(u, v), zf(u, v));
                                    let val = x.powi(m.a as i32) * y.powi(m.b as i32) * z.powi(m.d as i32);
                                    let tang = match m.c { 0 => tx, 1 => ty, _ => tz };
                                    sum += w * tang * val * u.powi(ix as i32) * v.powi(iy as i32);
                                }
                                v[di][j] = sum;
                            }
                            di += 1;
                        }
                    }
                }
            }
        }
        // Volume DOFs: (k-2)(k-1)k/2 × 3 components
        if k >= 3 {
            for deg in 0..=(k - 3) {
                for a in 0..=deg { for b in 0..=(deg - a) { let d = deg - a - b;
                    for comp in 0..3 {
                        for (j, m) in ms.iter().enumerate() {
                            v[di][j] = if m.c == comp { tet_int(m.a + a, m.b + b, m.d + d) } else { 0.0 };
                        }
                        di += 1;
                    }
                }}
            }
        }
        assert_eq!(di, n);

        // Gauss-Jordan with column pivoting
        let mut cp: Vec<usize> = (0..mt).collect();
        let mut r = vec![vec![0.0; n + mt]; n];
        for i in 0..n { for j in 0..mt { r[i][j] = v[i][j]; } r[i][mt + i] = 1.0; }
        let mut sel = Vec::new();
        for c in 0..n {
            let mut bc = c; let mut bv = 0.0_f64;
            for cc in c..mt { let mut mr = 0.0_f64; for rr in c..n { mr = mr.max(r[rr][cc].abs()); } if mr > bv { bv = mr; bc = cc; } }
            cp.swap(c, bc); for rr in 0..n { r[rr].swap(c, bc); }
            let mut pr = c; let mut pv = r[c][c].abs();
            for rr in c + 1..n { if r[rr][c].abs() > pv { pv = r[rr][c].abs(); pr = rr; } }
            r.swap(c, pr);
            let piv = r[c][c]; assert!(piv.abs() > 1e-14, "TetNDk({k}) singular at col {c}");
            let ip = 1.0 / piv; for j in c..n + mt { r[c][j] *= ip; }
            for rr in 0..n { if rr != c { let f = r[rr][c]; for j in c..n + mt { r[rr][j] -= f * r[c][j]; } } }
            sel.push(c);
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n { for j in 0..n { coeff[i * n + j] = r[i][mt + sel[j]]; } }
        TetNDkData { coeff, n, monomap: sel }
    })
}

pub struct TetNDk { order: usize }
impl TetNDk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); TetNDk { order: p } }
}

impl VectorReferenceElement for TetNDk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { let k = self.order; k * (k + 2) * (k + 3) / 2 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order; let d = tet_data(k); let n = d.n;
        let x = xi[0]; let y = xi[1]; let z = xi[2];
        let mut mv = vec![0.0; ((k+1)*(k+2)*(k+3)/2) * 3];
        let mut idx = 0usize;
        for deg in 0..=k { for a in 0..=deg { for b in 0..=(deg - a) { let c = deg - a - b;
            let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            for comp in 0..3 {
                mv[idx * 3 + comp] = if comp == 0 { v } else if comp == 1 { v } else { v };
                // Actually only the matching component should be set
                mv[idx * 3] = 0.0; mv[idx * 3 + 1] = 0.0; mv[idx * 3 + 2] = 0.0;
                mv[idx * 3 + comp] = v;
                idx += 1;
            }
        }}}
        for i in 0..n {
            let mut vx = 0.0; let mut vy = 0.0; let mut vz = 0.0;
            for (ji, &s) in d.monomap.iter().enumerate() {
                let c = d.coeff[i * n + ji];
                vx += c * mv[s * 3]; vy += c * mv[s * 3 + 1]; vz += c * mv[s * 3 + 2];
            }
            values[i * 3] = vx; values[i * 3 + 1] = vy; values[i * 3 + 2] = vz;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order; let d = tet_data(k); let n = d.n;
        let x = xi[0]; let y = xi[1]; let z = xi[2];
        let mut cm = vec![0.0; d.monomap.len() * 3];
        for (ji, &s) in d.monomap.iter().enumerate() {
            let mut rem = s; let mut deg = 0usize;
            loop { let n_ad = 3 * (deg + 1) * (deg + 2) / 2; if rem < n_ad { break; } rem -= n_ad; deg += 1; }
            let comp = rem % 3; let inner = rem / 3;
            let mut r2 = inner; let mut a = 0usize;
            loop { let n_r = deg - a + 1; if r2 < n_r { break; } r2 -= n_r; a += 1; }
            let b = r2; let c = deg - a - b;
            // curl of (x^a y^b z^c, 0, 0): (0, c·x^a y^b z^(c-1), -b·x^a y^(b-1) z^c)
            // curl of (0, x^a y^b z^c, 0): (-c·x^a y^b z^(c-1), 0, a·x^(a-1) y^b z^c)
            // curl of (0, 0, x^a y^b z^c): (b·x^a y^(b-1) z^c, -a·x^(a-1) y^b z^c, 0)
            let xp = x.powi(a as i32); let yp = y.powi(b as i32); let zp = z.powi(c as i32);
            let xm1 = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
            let ym1 = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
            let zm1 = if c > 0 { z.powi((c - 1) as i32) } else { 0.0 };
            let fa = a as f64; let fb = b as f64; let fc = c as f64;
            cm[ji * 3] = match comp { 0 => 0.0, 1 => -fc * xp * yp * zm1, _ => fb * xp * ym1 * zp };
            cm[ji * 3 + 1] = match comp { 0 => fc * xp * yp * zm1, 1 => 0.0, _ => -fa * xm1 * yp * zp };
            cm[ji * 3 + 2] = match comp { 0 => -fb * xp * ym1 * zp, 1 => fa * xm1 * yp * zp, _ => 0.0 };
        }
        for i in 0..n {
            let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
            for ji in 0..d.monomap.len() { let c = d.coeff[i * n + ji];
                cx += c * cm[ji * 3]; cy += c * cm[ji * 3 + 1]; cz += c * cm[ji * 3 + 2];
            }
            curl_vals[i * 3] = cx; curl_vals[i * 3 + 1] = cy; curl_vals[i * 3 + 2] = cz;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) { for v in div_vals.iter_mut() { *v = 0.0; } }
    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order; let n = k * (k + 2) * (k + 3) / 2;
        let mut c = Vec::with_capacity(n);
        for _ in 0..6 * k { c.push(vec![0.25, 0.25, 0.25]); }
        for _ in 0..4 * k * (k - 1) { c.push(vec![0.25, 0.25, 0.25]); }
        for _ in 0..n - c.len() { c.push(vec![0.25, 0.25, 0.25]); }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn tet_ndk_coeff() { for k in 1..=4 { let d = tet_data(k); let mut s = 0.0; for i in 0..d.n { s += d.coeff[i * d.n + i].abs(); } assert!(s > 0.1, "k={k}"); } }
    #[test] fn tet_ndk_finite() { for k in 1..=3 { let e = TetNDk::new(k); let n = e.n_dofs(); let mut v = vec![0.0; n * 3]; let mut c = vec![0.0; n * 3];
        for p in &[(0.25,0.25,0.25),(0.1,0.2,0.15),(0.5,0.1,0.1)] { e.eval_basis_vec(&[p.0,p.1,p.2], &mut v); e.eval_curl(&[p.0,p.1,p.2], &mut c);
            for val in v.iter().chain(c.iter()) { assert!(val.is_finite(), "k={k}"); } } } }
}
