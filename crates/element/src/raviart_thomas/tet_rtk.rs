//! Arbitrary-order Raviart-Thomas on reference tetrahedron via Vandermonde.
//! RT_k = [P_k]³ + (x,y,z)·P_k, dim = (k+1)(k+2)(k+4)/2.
//! DOFs: face flux 2(k+1)(k+2) + interior k(k+1)(k+2)/2.

use std::sync::OnceLock;
use crate::reference::VectorReferenceElement;

struct TetRTkData { coeff: Vec<f64>, n: usize, monomap: Vec<usize> }

fn tet_int(a: usize, b: usize, c: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        * (1..=c).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + c + 3).fold(1.0, |p, i| p * i as f64)
}

/// Decompose a [P_k]³ monomial index j into (deg, comp, a, b, c)
fn decode_pk3(j: usize) -> (usize, usize, usize, usize, usize) {
    let mut rem = j;
    let mut deg = 0;
    loop { let na = 3 * (deg + 1) * (deg + 2) / 2; if rem < na { break; } rem -= na; deg += 1; }
    let comp = rem % 3;
    rem /= 3;
    let mut a = 0;
    loop { let nxt = deg - a + 1; if rem < nxt { break; } rem -= nxt; a += 1; }
    (deg, comp, a, rem, deg - a - rem)
}

/// Decompose a bubble monomial index j into (deg, a, b, c)
fn decode_bub(j: usize) -> (usize, usize, usize, usize) {
    let mut rem = j;
    let mut deg = 0;
    loop { let na = (deg + 1) * (deg + 2) / 2; if rem < na { break; } rem -= na; deg += 1; }
    let mut a = 0;
    loop { let nxt = deg - a + 1; if rem < nxt { break; } rem -= nxt; a += 1; }
    (deg, a, rem, deg - a - rem)
}

fn tet_data(k: usize) -> &'static TetRTkData {
    static CACHE: [OnceLock<TetRTkData>; 9] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    CACHE[k].get_or_init(|| {
        let n = (k + 1) * (k + 2) * (k + 4) / 2;
        let n3 = (k + 1) * (k + 2) * (k + 3) / 2; // [P_k]³ count
        let nb = (k + 1) * (k + 2) * (k + 3) / 6; // bubble count
        let mt = n3 + nb;
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0usize;
        let qr = crate::quadrature::tri_rule_arbitrary(((3 * k) as u8).max(2));

        // Helper: evaluate [P_k]³ monomial at (x,y,z)
        let mv_pk3 = |j: usize, x: f64, y: f64, z: f64| -> f64 {
            let (_, _, a, b, c) = decode_pk3(j);
            x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32)
        };
        // Helper: evaluate bubble monomial at (x,y,z), returns (Φ_x, Φ_y, Φ_z)
        let mv_bub = |j: usize, x: f64, y: f64, z: f64| -> (f64, f64, f64) {
            let (_, a, b, c) = decode_bub(j);
            let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            (x * v, y * v, z * v)
        };

        for d in 0..=k { for a in 0..=d { let b = d - a;
            // Face 0: ∫ (Φ_x+Φ_y+Φ_z) u^a v^b dA on x+y+z=1
            for j in 0..n3 {
                let mut s = 0.0;
                for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                    let (u, v) = (qp[0], qp[1]);
                    s += w * mv_pk3(j, 1.0-u-v, u, v) * u.powi(a as i32) * v.powi(b as i32);
                }
                v[di][j] = s;
            }
            for j in 0..nb {
                let jj = n3 + j;
                let mut s = 0.0;
                for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                    let (u, v) = (qp[0], qp[1]); let x = 1.0-u-v; let y = u; let z = v;
                    let (fx, fy, fz) = mv_bub(j, x, y, z);
                    s += w * (fx + fy + fz) * u.powi(a as i32) * v.powi(b as i32);
                }
                v[di][jj] = s;
            }
            di += 1;
        }}

        for d in 0..=k { for a in 0..=d { let b = d - a;
            // Face 1: x=0, normal (-1,0,0). ∫ (-Φ_x) y^a z^b dA on x=0
            for j in 0..n3 {
                let (_, comp, am, bm, cm) = decode_pk3(j);
                if comp == 0 && am == 0 {
                    let mut s = 0.0;
                    for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                        let (u, v) = (qp[0], qp[1]);
                        s += w * (-1.0) * u.powi((bm + a) as i32) * v.powi((cm + b) as i32);
                    }
                    v[di][j] = s;
                }
            }
            // Bubble: Φ_x = x^(a+1) y^b z^c = 0 on x=0
            for j in 0..nb { v[di][n3 + j] = 0.0; }
            di += 1;
        }}

        for d in 0..=k { for a in 0..=d { let b = d - a;
            // Face 2: y=0, normal (0,-1,0). ∫ (-Φ_y) x^a z^b dA
            for j in 0..n3 {
                let (_, comp, am, bm, cm) = decode_pk3(j);
                if comp == 1 && bm == 0 {
                    let mut s = 0.0;
                    for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                        let (u, v) = (qp[0], qp[1]);
                        s += w * (-1.0) * u.powi((am + a) as i32) * v.powi((cm + b) as i32);
                    }
                    v[di][j] = s;
                }
            }
            for j in 0..nb { v[di][n3 + j] = 0.0; }
            di += 1;
        }}

        for d in 0..=k { for a in 0..=d { let b = d - a;
            // Face 3: z=0, normal (0,0,-1). ∫ (-Φ_z) x^a y^b dA
            for j in 0..n3 {
                let (_, comp, am, bm, cm) = decode_pk3(j);
                if comp == 2 && cm == 0 {
                    let mut s = 0.0;
                    for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                        let (u, v) = (qp[0], qp[1]);
                        s += w * (-1.0) * u.powi((am + a) as i32) * v.powi((bm + b) as i32);
                    }
                    v[di][j] = s;
                }
            }
            for j in 0..nb { v[di][n3 + j] = 0.0; }
            di += 1;
        }}

        if k >= 1 {
            for d in 0..=(k - 1) { for a in 0..=d { for b in 0..=(d - a) { let c = d - a - b;
                for comp in 0..3 {
                    for j in 0..n3 {
                        let (_, mc, am, bm, cm) = decode_pk3(j);
                        v[di][j] = if mc == comp { tet_int(am + a, bm + b, cm + c) } else { 0.0 };
                    }
                    for j in 0..nb {
                        let (_, am, bm, cm) = decode_bub(j);
                        let (ea, eb, ec) = match comp {
                            0 => (am + 1 + a, bm + b, cm + c),
                            1 => (am + a, bm + 1 + b, cm + c),
                            _ => (am + a, bm + b, cm + 1 + c),
                        };
                        v[di][n3 + j] = tet_int(ea, eb, ec);
                    }
                    di += 1;
                }
            }}}
        }

        assert_eq!(di, n);

        let mut cp: Vec<usize> = (0..mt).collect();
        let mut row = vec![vec![0.0; n + mt]; n];
        for i in 0..n { for j in 0..mt { row[i][j] = v[i][j]; } row[i][mt + i] = 1.0; }
        let mut sel = Vec::new();
        for c in 0..n {
            let mut bc = c; let mut bv = 0.0_f64;
            for cc in c..mt { let mut mr = 0.0_f64; for rr in c..n { mr = mr.max(row[rr][cc].abs()); } if mr > bv { bv = mr; bc = cc; } }
            cp.swap(c, bc); for rr in 0..n { row[rr].swap(c, bc); }
            let mut pr = c; let mut pv = row[c][c].abs();
            for rr in c + 1..n { if row[rr][c].abs() > pv { pv = row[rr][c].abs(); pr = rr; } }
            row.swap(c, pr);
            let piv = row[c][c]; assert!(piv.abs() > 1e-14, "TetRTk({k}) singular at col {c}");
            let ip = 1.0 / piv; for j in c..n + mt { row[c][j] *= ip; }
            for rr in 0..n { if rr != c { let f = row[rr][c]; for j in c..n + mt { row[rr][j] -= f * row[c][j]; } } }
            sel.push(c);
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n { for j in 0..n { coeff[i * n + j] = row[i][mt + sel[j]]; } }
        TetRTkData { coeff, n, monomap: sel }
    })
}

pub struct TetRTk { order: usize }
impl TetRTk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); TetRTk { order: p } }
}

impl VectorReferenceElement for TetRTk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) * (self.order + 4) / 2 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order; let d = tet_data(k); let n = d.n;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let n3 = (k + 1) * (k + 2) * (k + 3) / 2;
        let nb = (k + 1) * (k + 2) * (k + 3) / 6;
        let mt = n3 + nb;
        let mut mv = vec![0.0; mt * 3];
        let mut idx = 0;
        for dg in 0..=k { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            for comp in 0..3 { mv[idx * 3] = 0.0; mv[idx * 3 + 1] = 0.0; mv[idx * 3 + 2] = 0.0; mv[idx * 3 + comp] = v; idx += 1; }
        }}}
        for dg in 0..=k { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            mv[idx * 3] = x * v; mv[idx * 3 + 1] = y * v; mv[idx * 3 + 2] = z * v; idx += 1;
        }}}
        for i in 0..n {
            let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
            for (ji, &s) in d.monomap.iter().enumerate() {
                let c = d.coeff[i * n + ji];
                vx += c * mv[s * 3]; vy += c * mv[s * 3 + 1]; vz += c * mv[s * 3 + 2];
            }
            values[i * 3] = vx; values[i * 3 + 1] = vy; values[i * 3 + 2] = vz;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order; let d = tet_data(k); let n = d.n;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let n3 = (k + 1) * (k + 2) * (k + 3) / 2;
        let nb = (k + 1) * (k + 2) * (k + 3) / 6;
        let mt = n3 + nb;
        let mut dm = vec![0.0; mt];
        let mut idx = 0;
        for dg in 0..=k { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            let xm1 = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
            let ym1 = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
            let zm1 = if c > 0 { z.powi((c - 1) as i32) } else { 0.0 };
            let xp = x.powi(a as i32); let yp = y.powi(b as i32); let zp = z.powi(c as i32);
            dm[idx] = (a as f64) * xm1 * yp * zp; idx += 1;
            dm[idx] = (b as f64) * xp * ym1 * zp; idx += 1;
            dm[idx] = (c as f64) * xp * yp * zm1; idx += 1;
        }}}
        for dg in 0..=k { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
            dm[idx] = (a + b + c + 3) as f64 * v; idx += 1;
        }}}
        for i in 0..n {
            let mut s = 0.0;
            for (ji, &sel) in d.monomap.iter().enumerate() { s += d.coeff[i * n + ji] * dm[sel]; }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::tet_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let n = self.n_dofs();
        (0..n).map(|_| vec![0.25, 0.25, 0.25]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn coeff_non_singular() { for k in 1..=4 { let d = tet_data(k); let mut s = 0.0; for i in 0..d.n { s += d.coeff[i * d.n + i].abs(); } assert!(s > 0.1, "k={k}"); } }
    #[test] fn finite() { for k in 1..=3 { let e = TetRTk::new(k); let n = e.n_dofs(); let mut v = vec![0.0; n*3]; let mut d = vec![0.0; n];
        for p in &[(0.25,0.25,0.25),(0.1,0.2,0.15),(0.5,0.1,0.1)] { e.eval_basis_vec(&[p.0,p.1,p.2], &mut v); e.eval_div(&[p.0,p.1,p.2], &mut d);
            for &val in v.iter().chain(d.iter()) { assert!(val.is_finite()); } } } }
}
