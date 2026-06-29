use std::sync::OnceLock;
use crate::reference::VectorReferenceElement;

struct TriBDMkData { coeff: Vec<f64>, n: usize }

fn tri_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 2).fold(1.0, |p, i| p * i as f64)
}
fn beta_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 1).fold(1.0, |p, i| p * i as f64)
}

fn tri_data(k: usize) -> &'static TriBDMkData {
    static CACHE: [OnceLock<TriBDMkData>; 9] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    CACHE[k].get_or_init(|| {
        let n = (k + 1) * (k + 2); // dim([P_k]^2)
        // Monomials: (a, b, comp) for comp=0,1, a+b ≤ k
        let mut mc = Vec::new();
        for deg in 0..=k { for a in 0..=deg { let b = deg - a;
            mc.push((a, b, 0)); mc.push((a, b, 1));
        }}
        assert_eq!(mc.len(), n);
        let mut v = vec![vec![0.0; n]; n];
        let mut di = 0usize;

        // Edge 0: hypotenuse (1-t,t), normal (1,1)/√2
        // ∫ (Φ_x+Φ_y)·t^p dt  (1/√2 absorbed into coefficient)
        for p in 0..=k {
            for (j, &(a, b, _comp)) in mc.iter().enumerate() {
                v[di][j] = beta_int(a, b + p);
            }
            di += 1;
        }

        // Edge 1: left edge (0,t), normal (-1,0)
        for p in 0..=k {
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                let val = if comp == 0 && a == 0 { -1.0 / (b + p + 1) as f64 } else { 0.0 };
                v[di][j] = val;
            }
            di += 1;
        }

        // Edge 2: bottom edge (t,0), normal (0,-1)
        for p in 0..=k {
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                let val = if comp == 1 && b == 0 { -1.0 / (a + p + 1) as f64 } else { 0.0 };
                v[di][j] = val;
            }
            di += 1;
        }

        // Interior: k(k-1) from [P_{k-2}]^2 + (k-1) from deg=k-1 comp=0 fill
        if k >= 2 {
            // [P_{k-2}]^2: both comps, all monomials with degree ≤ k-2
            for deg in 0..=(k - 2) { for ix in 0..=deg { let iy = deg - ix;
                for comp in 0..2 {
                    for (j, &(a, b, mc_comp)) in mc.iter().enumerate() {
                        v[di][j] = if mc_comp == comp { tri_int(a + ix, b + iy) } else { 0.0 };
                    }
                    di += 1;
                }
            }}
            // Fill remaining k-1 DOFs: comp 0, deg=k-1 monomials (ix=0..k-2, iy=k-1-ix)
            for ix in 0..=(k - 2) { let iy = (k - 1) - ix;
                for (j, &(a, b, mc_comp)) in mc.iter().enumerate() {
                    v[di][j] = if mc_comp == 0 { tri_int(a + ix, b + iy) } else { 0.0 };
                }
                di += 1;
            }
        }

        assert_eq!(di, n, "TriBDMk({k}): DOF count {di} vs {n}");

        // Gauss-Jordan (square matrix, no column pivoting)
        let mut row = vec![vec![0.0; n + n]; n];
        for i in 0..n { for j in 0..n { row[i][j] = v[i][j]; } row[i][n + i] = 1.0; }
        for c in 0..n {
            let mut pr = c; let mut pv = row[c][c].abs();
            for rr in c + 1..n { if row[rr][c].abs() > pv { pv = row[rr][c].abs(); pr = rr; } }
            row.swap(c, pr);
            let piv = row[c][c]; assert!(piv.abs() > 1e-14, "TriBDMk({k}) singular at col {c}");
            let ip = 1.0 / piv; for j in c..n + n { row[c][j] *= ip; }
            for rr in 0..n { if rr != c { let f = row[rr][c]; for j in c..n + n { row[rr][j] -= f * row[c][j]; } } }
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n { for j in 0..n { coeff[i * n + j] = row[i][n + j]; } }
        TriBDMkData { coeff, n }
    })
}

pub struct TriBDMk { order: usize }
impl TriBDMk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); TriBDMk { order: p } }
}

impl VectorReferenceElement for TriBDMk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * (self.order + 2) }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order; let d = tri_data(k); let n = d.n;
        let x = xi[0]; let y = xi[1];
        let mut mv = vec![0.0; n * 2];
        let mut idx = 0usize;
        for deg in 0..=k { for a in 0..=deg { let b = deg - a; let v = x.powi(a as i32) * y.powi(b as i32);
            mv[idx * 2] = v; mv[idx * 2 + 1] = 0.0; idx += 1;
            mv[idx * 2] = 0.0; mv[idx * 2 + 1] = v; idx += 1;
        }}
        for i in 0..n {
            let mut vx = 0.0; let mut vy = 0.0;
            for j in 0..n {
                let c = d.coeff[i * n + j];
                vx += c * mv[j * 2]; vy += c * mv[j * 2 + 1];
            }
            values[i * 2] = vx; values[i * 2 + 1] = vy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order; let d = tri_data(k); let n = d.n;
        let x = xi[0]; let y = xi[1];
        let mut dm = vec![0.0; n];
        let mut idx = 0usize;
        for deg in 0..=k { for a in 0..=deg { let b = deg - a;
            let xp = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
            let yp = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
            dm[idx] = (a as f64) * xp * y.powi(b as i32); idx += 1;
            dm[idx] = (b as f64) * x.powi(a as i32) * yp; idx += 1;
        }}
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n { s += d.coeff[i * n + j] * dm[j]; }
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
        let k = self.order; let n = (k + 1) * (k + 2);
        let mut c = Vec::with_capacity(n);
        for p in 0..=k { let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![1.0 - t, t]);
        }
        for p in 0..=k { let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![0.0, t]);
        }
        for p in 0..=k { let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![t, 0.0]);
        }
        while c.len() < n { c.push(vec![1.0 / 3.0, 1.0 / 3.0]); }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn coeff_non_singular() { for k in 1..=4 { let d = tri_data(k); let mut s = 0.0; for i in 0..d.n { s += d.coeff[i * d.n + i].abs(); } assert!(s > 0.1, "k={k}"); } }
    #[test] fn finite() { for k in 1..=4 { let e = TriBDMk::new(k); let n = e.n_dofs(); let mut v = vec![0.0; n*2]; let mut d = vec![0.0; n];
        for p in &[(0.25,0.25),(0.1,0.2),(0.5,0.1)] { e.eval_basis_vec(&[p.0,p.1], &mut v); e.eval_div(&[p.0,p.1], &mut d);
            for &val in v.iter().chain(d.iter()) { assert!(val.is_finite(),"k={k} at {p:?}"); } } } }
    #[test] fn n_dofs() {
        assert_eq!(TriBDMk::new(1).n_dofs(), 6);
        assert_eq!(TriBDMk::new(2).n_dofs(), 12);
        assert_eq!(TriBDMk::new(3).n_dofs(), 20);
    }
}
