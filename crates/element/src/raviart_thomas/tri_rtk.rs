//! Arbitrary-order RT_k on reference triangle, Vandermonde construction.

use std::sync::OnceLock;
use crate::reference::VectorReferenceElement;

struct TriRTkData { coeff: Vec<f64>, n: usize, monomap: Vec<usize> }

fn tri_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 2).fold(1.0, |p, i| p * i as f64)
}

fn beta_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 1).fold(1.0, |p, i| p * i as f64)
}

fn tri_data(k: usize) -> &'static TriRTkData {
    static CACHE: [OnceLock<TriRTkData>; 6] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    CACHE[k].get_or_init(|| {
        let n = (k + 1) * (k + 3);
        // Enumerate monomials: comp=0,1 for P_k + bubble
        let mut mc = Vec::new(); // (a, b, comp) for P_k entries, comp=0,1
        let mut mb = Vec::new(); // (a, b) for bubble entries
        for deg in 0..=k { for a in 0..=deg { let b = deg - a;
            mc.push((a, b, 0)); mc.push((a, b, 1));
            mb.push((a, b));
        }}
        let mt = mc.len() + mb.len();
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0usize;

        // Edge 0: hypotenuse (1-t,t), normal (1,1)/√2, length √2 → ∫(Φ_x+Φ_y)dt
        for p in 0..=k {
            // P_k entries
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                let val = if comp == 0 { beta_int(a, b + p) } else { beta_int(a, b + p) };
                // Φ_x + Φ_y: both comps contribute x^a y^b · t^p
                v[di][j] = beta_int(a, b + p);
            }
            // Bubble entries
            for (j, &(a, b)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                // (x·x^a y^b, y·x^a y^b): Φ_x+Φ_y = x^(a+1)y^b + x^a y^(b+1)
                // At (1-t,t): (1-t)^(a+1)t^b + (1-t)^a t^(b+1)
                v[di][jj] = beta_int(a + 1, b + p) + beta_int(a, b + p + 1);
            }
            di += 1;
        }

        // Edge 1: left edge (0,t), normal (-1,0)
        for p in 0..=k {
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                // Φ·n = -Φ_x. Comp 0 contributes -x^a y^b. At (0,t): -0^a t^b.
                // Only comp=0, a=0 gives nonzero: -t^(b+p)
                let val = if comp == 0 && a == 0 { -1.0 / (b + p + 1) as f64 } else { 0.0 };
                v[di][j] = val;
            }
            for (j, &(a, b)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                // Φ·n = -x^(a+1)y^b. At (0,t): -(0)^(a+1)·t^b = 0.
                v[di][jj] = 0.0;
            }
            di += 1;
        }

        // Edge 2: bottom edge (t,0), normal (0,-1)
        for p in 0..=k {
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                // Φ·n = -Φ_y = -x^a y^b. At (t,0): -t^a·0^b.
                // Only comp=1, b=0 gives nonzero: -t^(a+p)
                let val = if comp == 1 && b == 0 { -1.0 / (a + p + 1) as f64 } else { 0.0 };
                v[di][j] = val;
            }
            for (j, &(a, b)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                // Φ·n = -x^a y^(b+1). At (t,0): -t^a·0^(b+1) = 0.
                v[di][jj] = 0.0;
            }
            di += 1;
        }

        // Interior DOFs: ∫ Φ_x·x^ix y^iy dA and ∫ Φ_y·x^ix y^iy dA for ix+iy ≤ k-1
        if k >= 1 {
            for deg in 0..=(k - 1) { for ix in 0..=deg { let iy = deg - ix;
                for comp in 0..2 {
                    for (j, &(a, b, mc_comp)) in mc.iter().enumerate() {
                        let val = if mc_comp == comp { tri_int(a + ix, b + iy) } else { 0.0 };
                        v[di][j] = val;
                    }
                    for (j, &(a, b)) in mb.iter().enumerate() {
                        let jj = mc.len() + j;
                        // bubble: (x^(a+1)y^b, x^a y^(b+1))
                        let val = if comp == 0 { tri_int(a + 1 + ix, b + iy) } else { tri_int(a + ix, b + 1 + iy) };
                        v[di][jj] = val;
                    }
                    di += 1;
                }
            }}
        }

        assert_eq!(di, n, "TriRTk({k}): DOF count {di} vs {n}");

        // Gauss-Jordan with column pivoting
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
            let piv = row[c][c]; assert!(piv.abs() > 1e-14, "TriRTk({k}) singular at col {c}");
            let ip = 1.0 / piv; for j in c..n + mt { row[c][j] *= ip; }
            for rr in 0..n { if rr != c { let f = row[rr][c]; for j in c..n + mt { row[rr][j] -= f * row[c][j]; } } }
            sel.push(c);
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n { for j in 0..n { coeff[i * n + j] = row[i][mt + sel[j]]; } }
        TriRTkData { coeff, n, monomap: sel }
    })
}

pub struct TriRTk { order: usize }
impl TriRTk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); TriRTk { order: p } }
}

impl VectorReferenceElement for TriRTk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 3) }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order; let d = tri_data(k); let n = d.n;
        let x = xi[0]; let y = xi[1];
        let mt = (k + 1) * (k + 2) + (k + 1) * (k + 2) / 2; // P_k entries + bubble entries
        let mut mv = vec![0.0; mt * 2];
        let mut idx = 0usize;
        // P_k entries: comp=0 and comp=1
        for deg in 0..=k { for a in 0..=deg { let b = deg - a; let v = x.powi(a as i32) * y.powi(b as i32);
            mv[idx * 2] = v; mv[idx * 2 + 1] = 0.0; idx += 1;
            mv[idx * 2] = 0.0; mv[idx * 2 + 1] = v; idx += 1;
        }}
        // Bubble entries: (x^(a+1) y^b, x^a y^(b+1))
        for deg in 0..=k { for a in 0..=deg { let b = deg - a; let v = x.powi(a as i32) * y.powi(b as i32);
            mv[idx * 2] = x * v; mv[idx * 2 + 1] = y * v; idx += 1;
        }}
        for i in 0..n {
            let mut vx = 0.0; let mut vy = 0.0;
            for (ji, &s) in d.monomap.iter().enumerate() {
                let c = d.coeff[i * n + ji];
                vx += c * mv[s * 2]; vy += c * mv[s * 2 + 1];
            }
            values[i * 2] = vx; values[i * 2 + 1] = vy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order; let d = tri_data(k); let n = d.n;
        let x = xi[0]; let y = xi[1];
        let mt = (k + 1) * (k + 2) + (k + 1) * (k + 2) / 2;
        let mut dm = vec![0.0; mt];
        let mut idx = 0usize;
        // Div of P_k entries:
        // comp=0 (x^a y^b, 0): div = a·x^(a-1) y^b (if a>0, else 0)
        // comp=1 (0, x^a y^b): div = b·x^a y^(b-1) (if b>0, else 0)
        for deg in 0..=k { for a in 0..=deg { let b = deg - a;
            let xp = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
            let yp = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
            dm[idx] = (a as f64) * xp * y.powi(b as i32); idx += 1;
            dm[idx] = (b as f64) * x.powi(a as i32) * yp; idx += 1;
        }}
        // Div of bubble entries:
        // div(x^(a+1) y^b, x^a y^(b+1)) = (a+1)x^a y^b + (b+1)x^a y^b = (a+b+2)x^a y^b
        for deg in 0..=k { for a in 0..=deg { let b = deg - a;
            let v = x.powi(a as i32) * y.powi(b as i32);
            dm[idx] = (a + b + 2) as f64 * v; idx += 1;
        }}
        for i in 0..n {
            let mut s = 0.0;
            for (ji, &sel) in d.monomap.iter().enumerate() {
                s += d.coeff[i * n + ji] * dm[sel];
            }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::tri_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order; let n = (k + 1) * (k + 3);
        let mut c = Vec::with_capacity(n);
        // 3 edges × (k+1) DOFs at equispaced points
        for p in 0..=k { let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![1.0 - t, t]); // edge 0: hypotenuse
        }
        for p in 0..=k { let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![0.0, t]); // edge 1: left
        }
        for p in 0..=k { let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![t, 0.0]); // edge 2: bottom
        }
        // Interior: k(k+1) DOFs at barycentric coords
        let remaining = n - c.len();
        for i in 0..remaining {
            c.push(vec![1.0 / 3.0, 1.0 / 3.0]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn coeff_non_singular() { for k in 1..=4 { let d = tri_data(k); let mut s = 0.0; for i in 0..d.n { s += d.coeff[i * d.n + i].abs(); } assert!(s > 0.1, "k={k}"); } }
    #[test] fn finite() { for k in 1..=3 { let e = TriRTk::new(k); let n = e.n_dofs(); let mut v = vec![0.0; n*2]; let mut d = vec![0.0; n];
        for p in &[(0.25,0.25),(0.1,0.2),(0.5,0.1)] { e.eval_basis_vec(&[p.0,p.1], &mut v); e.eval_div(&[p.0,p.1], &mut d);
            for &val in v.iter().chain(d.iter()) { assert!(val.is_finite()); } } } }
}
