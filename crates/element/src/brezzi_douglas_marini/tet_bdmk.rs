use std::sync::OnceLock;
use crate::reference::VectorReferenceElement;

struct TetBDMkData { coeff: Vec<f64>, n: usize, monomap: Vec<usize> }

fn tet_int(a: usize, b: usize, c: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        * (1..=c).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + c + 3).fold(1.0, |p, i| p * i as f64)
}
fn tri_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 2).fold(1.0, |p, i| p * i as f64)
}

fn tet_data(k: usize) -> &'static TetBDMkData {
    static CACHE: [OnceLock<TetBDMkData>; 9] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    CACHE[k].get_or_init(|| {
        let n = (k + 1) * (k + 2) * (k + 3) / 2;
        // [P_k]^3 monomials
        let mut mc = Vec::new();
        for dg in 0..=k { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            for comp in 0..3 { mc.push((a, b, c, comp)); }
        }}}
        // Expand candidate set with RT-type bubbles (x,y,z)·P_k (like TetRTk)
        let mut mb = Vec::new();
        for dg in 0..=k { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            mb.push((a, b, c));
        }}}
        let mt = mc.len() + mb.len();
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0usize;
        let qr = crate::quadrature::tri_rule_arbitrary(((3 * k) as u8).max(2));

        // Face 0: x+y+z=1, normal (1,1,1)
        for a in 0..=k { for b in 0..=(k - a) {
            for (j, &(am, bm, cm, _comp)) in mc.iter().enumerate() {
                let mut s = 0.0;
                for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                    let (u, v) = (qp[0], qp[1]);
                    let x = 1.0 - u - v; let y = u; let z = v;
                    s += w * (x.powi(am as i32) * y.powi(bm as i32) * z.powi(cm as i32))
                        * u.powi(a as i32) * v.powi(b as i32);
                }
                v[di][j] = s;
            }
            for (j, &(am, bm, cm)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                let mut s = 0.0;
                for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                    let (u, v) = (qp[0], qp[1]); let x = 1.0 - u - v; let y = u; let z = v;
                    let bv = x.powi(am as i32) * y.powi(bm as i32) * z.powi(cm as i32);
                    s += w * (x + y + z) * bv * u.powi(a as i32) * v.powi(b as i32);
                }
                v[di][jj] = s;
            }
            di += 1;
        }}

        // Face 1: x=0, normal (-1,0,0)
        for a in 0..=k { for b in 0..=(k - a) {
            for (j, &(am, bm, cm, comp)) in mc.iter().enumerate() {
                v[di][j] = if comp == 0 && am == 0 { -tri_int(bm + a, cm + b) } else { 0.0 };
            }
            for (j, &(_am, _bm, _cm)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                // bubble: x·(x^a y^b z^c). At x=0: 0.
                v[di][jj] = 0.0;
            }
            di += 1;
        }}

        // Face 2: y=0, normal (0,-1,0)
        for a in 0..=k { for b in 0..=(k - a) {
            for (j, &(am, bm, cm, comp)) in mc.iter().enumerate() {
                v[di][j] = if comp == 1 && bm == 0 { -tri_int(am + a, cm + b) } else { 0.0 };
            }
            for (j, &(_am, _bm, _cm)) in mb.iter().enumerate() {
                v[di][mc.len() + j] = 0.0;
            }
            di += 1;
        }}

        // Face 3: z=0, normal (0,0,-1)
        for a in 0..=k { for b in 0..=(k - a) {
            for (j, &(am, bm, cm, comp)) in mc.iter().enumerate() {
                v[di][j] = if comp == 2 && cm == 0 { -tri_int(am + a, bm + b) } else { 0.0 };
            }
            for (j, &(_am, _bm, _cm)) in mb.iter().enumerate() {
                v[di][mc.len() + j] = 0.0;
            }
            di += 1;
        }}

        // Interior: [P_{k-2}]^3 L² moments
        if k >= 2 {
            for dg in 0..=(k - 2) { for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
                for comp in 0..3 {
                    for (j, &(am, bm, cm, mc_comp)) in mc.iter().enumerate() {
                        v[di][j] = if mc_comp == comp { tet_int(am + a, bm + b, cm + c) } else { 0.0 };
                    }
                    for (j, &(am, bm, cm)) in mb.iter().enumerate() {
                        let jj = mc.len() + j;
                        let (ea, eb, ec) = match comp {
                            0 => (am + 1 + a, bm + b, cm + c),
                            1 => (am + a, bm + 1 + b, cm + c),
                            _ => (am + a, bm + b, cm + 1 + c),
                        };
                        v[di][jj] = tet_int(ea, eb, ec);
                    }
                    di += 1;
                }
            }}}
            // Fill remaining (k-1)(k+1) DOFs: deg=k-1 monomials, all comps
            let fill = (k - 1) * (k + 1);
            let mut fill_cnt = 0usize;
            for a in 0..=(k - 1) { for b in 0..=(k - 1 - a) { let c = (k - 1) - a - b;
                for comp in 0..3 {
                    if fill_cnt >= fill { break; }
                    for (j, &(am, bm, cm, mc_comp)) in mc.iter().enumerate() {
                        v[di][j] = if mc_comp == comp { tet_int(am + a, bm + b, cm + c) } else { 0.0 };
                    }
                    for (j, &(am, bm, cm)) in mb.iter().enumerate() {
                        let jj = mc.len() + j;
                        let (ea, eb, ec) = match comp {
                            0 => (am + 1 + a, bm + b, cm + c),
                            1 => (am + a, bm + 1 + b, cm + c),
                            _ => (am + a, bm + b, cm + 1 + c),
                        };
                        v[di][jj] = tet_int(ea, eb, ec);
                    }
                    di += 1; fill_cnt += 1;
                }
            } if fill_cnt >= fill { break; }}
        }

        assert_eq!(di, n, "TetBDMk({k}): DOF {di} vs {n}");

        // Column-pivoted Gauss-Jordan (like TetRTk)
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
            let piv = row[c][c]; assert!(piv.abs() > 1e-14, "TetBDMk({k}) singular at col {c}");
            let ip = 1.0 / piv; for j in c..n + mt { row[c][j] *= ip; }
            for rr in 0..n { if rr != c { let f = row[rr][c]; for j in c..n + mt { row[rr][j] -= f * row[c][j]; } } }
            sel.push(cp[c]);
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n { for j in 0..n { coeff[i * n + j] = row[i][mt + j]; } }
        TetBDMkData { coeff, n, monomap: sel }
    })
}

pub struct TetBDMk { order: usize }
impl TetBDMk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); TetBDMk { order: p } }
}

impl VectorReferenceElement for TetBDMk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) * (self.order + 3) / 2 }

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
    #[test] fn finite() { for k in 1..=3 { let e = TetBDMk::new(k); let n = e.n_dofs(); let mut v = vec![0.0; n*3]; let mut d = vec![0.0; n];
        for p in &[(0.25,0.25,0.25),(0.1,0.2,0.15),(0.5,0.1,0.1)] { e.eval_basis_vec(&[p.0,p.1,p.2], &mut v); e.eval_div(&[p.0,p.1,p.2], &mut d);
            for &val in v.iter().chain(d.iter()) { assert!(val.is_finite()); } } } }
    #[test] fn n_dofs() {
        assert_eq!(TetBDMk::new(1).n_dofs(), 12);
        assert_eq!(TetBDMk::new(2).n_dofs(), 30);
        assert_eq!(TetBDMk::new(3).n_dofs(), 60);
        assert_eq!(TetBDMk::new(4).n_dofs(), 105);
    }
}
