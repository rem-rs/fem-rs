//! Arbitrary-order RT_k on reference quadrilateral [0,1]² via Vandermonde.
//! RT_k = Q_{k+1,k} × Q_{k,k+1}, dim = 2(k+1)(k+2).
//! DOFs: edge flux 4(k+1) + interior x-moments k(k+1) + y-moments k(k+1).

use crate::reference::VectorReferenceElement;
use std::sync::OnceLock;

struct QuadRTkData {
    coeff: Vec<f64>,
    n: usize,
}

/// Gauss-Legendre nodes/weights on [0,1]
fn gl_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    crate::quadrature::gauss_legendre_01_arbitrary(n)
}

fn quad_data(k: usize) -> &'static QuadRTkData {
    static CACHE: [OnceLock<QuadRTkData>; 9] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[k].get_or_init(|| {
        let n = 2 * (k + 1) * (k + 2);
        let nx = (k + 2) * (k + 1); // x-comp monomials: degree k+1 in x, k in y
        let ny = (k + 1) * (k + 2); // y-comp monomials: degree k in x, k+1 in y
        let mt = nx + ny;
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0usize;
        let nq = (3 * k + 3).max(4);
        let (gx, gw) = gl_1d(nq);
        // Tensor-product 2D quadrature for interior
        let mut qp2 = Vec::new();
        let mut qw2 = Vec::new();
        for (&xi, &wi) in gx.iter().zip(gw.iter()) {
            for (&xj, &wj) in gx.iter().zip(gw.iter()) {
                qp2.push((xi, xj));
                qw2.push(wi * wj);
            }
        }

        // Edge 0 (bottom): y=0, n=(0,-1). ∫ -Φ_y(x,0)·x^p dx
        for p in 0..=k {
            // x-comp monomials contribute nothing to Φ_y
            for j in 0..nx {
                v[di][j] = 0.0;
            }
            // y-comp monomials: Φ_y = x^a y^b
            for j in 0..ny {
                let jj = nx + j;
                let a = j / (k + 2);
                let b = j % (k + 2);
                let yv = if b == 0 { 1.0 } else { 0.0 }; // y^b at y=0
                let mut s = 0.0;
                for (&xi, &wi) in gx.iter().zip(gw.iter()) {
                    s += wi * (-yv) * xi.powi((a + p) as i32);
                }
                v[di][jj] = s;
            }
            di += 1;
        }
        // Edge 1 (right): x=1, n=(1,0). ∫ Φ_x(1,y)·y^p dy
        for p in 0..=k {
            for j in 0..nx {
                let a = j / (k + 1);
                let b = j % (k + 1);
                let xv = 1.0_f64.powi(a as i32); // x^a at x=1
                let mut s = 0.0;
                for (&yi, &wi) in gx.iter().zip(gw.iter()) {
                    s += wi * xv * yi.powi((b + p) as i32);
                }
                v[di][j] = s;
            }
            for j in 0..ny {
                v[di][nx + j] = 0.0;
            }
            di += 1;
        }
        // Edge 2 (top): y=1, n=(0,1). ∫ Φ_y(x,1)·x^p dx
        for p in 0..=k {
            for j in 0..nx {
                v[di][j] = 0.0;
            }
            for j in 0..ny {
                let jj = nx + j;
                let a = j / (k + 2);
                let b = j % (k + 2);
                let yv = 1.0_f64.powi(b as i32); // y^b at y=1
                let mut s = 0.0;
                for (&xi, &wi) in gx.iter().zip(gw.iter()) {
                    s += wi * yv * xi.powi((a + p) as i32);
                }
                v[di][jj] = s;
            }
            di += 1;
        }
        // Edge 3 (left): x=0, n=(-1,0). ∫ -Φ_x(0,y)·y^p dy
        for p in 0..=k {
            for j in 0..nx {
                let a = j / (k + 1);
                let b = j % (k + 1);
                let xv = if a == 0 { 1.0 } else { 0.0 }; // x^a at x=0
                let mut s = 0.0;
                for (&yi, &wi) in gx.iter().zip(gw.iter()) {
                    s += wi * (-xv) * yi.powi((b + p) as i32);
                }
                v[di][j] = s;
            }
            for j in 0..ny {
                v[di][nx + j] = 0.0;
            }
            di += 1;
        }
        // Interior: ∫ Φ_x·x^a y^b dA for a=0..k-1, b=0..k
        if k >= 1 {
            for a in 0..k {
                for b in 0..=k {
                    for j in 0..nx {
                        let am = j / (k + 1);
                        let bm = j % (k + 1);
                        let mut s = 0.0;
                        for (xy, wi) in qp2.iter().zip(qw2.iter()) {
                            let (xi, yi) = *xy;
                            s += wi * xi.powi((am + a) as i32) * yi.powi((bm + b) as i32);
                        }
                        v[di][j] = s;
                    }
                    for j in 0..ny {
                        v[di][nx + j] = 0.0;
                    }
                    di += 1;
                }
            }
            for a in 0..=k {
                for b in 0..k {
                    for j in 0..nx {
                        v[di][j] = 0.0;
                    }
                    for j in 0..ny {
                        let jj = nx + j;
                        let am = j / (k + 2);
                        let bm = j % (k + 2);
                        let mut s = 0.0;
                        for (xy, wi) in qp2.iter().zip(qw2.iter()) {
                            let (xi, yi) = *xy;
                            s += wi * xi.powi((am + a) as i32) * yi.powi((bm + b) as i32);
                        }
                        v[di][jj] = s;
                    }
                    di += 1;
                }
            }
        }

        assert_eq!(di, n);

        // Gauss-Jordan (square invert)
        let mut row = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                row[i][j] = v[i][j];
            }
            row[i][n + i] = 1.0;
        }
        for c in 0..n {
            let mut pr = c;
            let mut pv = row[c][c].abs();
            for rr in c + 1..n {
                if row[rr][c].abs() > pv {
                    pv = row[rr][c].abs();
                    pr = rr;
                }
            }
            row.swap(c, pr);
            let piv = row[c][c];
            assert!(piv.abs() > 1e-14, "QuadRTk({k}) singular at col {c}");
            let ip = 1.0 / piv;
            for j in c..2 * n {
                row[c][j] *= ip;
            }
            for rr in 0..n {
                if rr != c {
                    let f = row[rr][c];
                    for j in c..2 * n {
                        row[rr][j] -= f * row[c][j];
                    }
                }
            }
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                coeff[i * n + j] = row[i][n + j];
            }
        }
        QuadRTkData { coeff, n }
    })
}

pub struct QuadRTk {
    order: usize,
}
impl QuadRTk {
    pub fn new(p: usize) -> Self {
        QuadRTk { order: p }
    }
}

impl VectorReferenceElement for QuadRTk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        2 * (self.order + 1) * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;
        let d = quad_data(k);
        let n = d.n;
        let x = xi[0];
        let y = xi[1];
        let nx = (k + 2) * (k + 1);
        let ny = (k + 1) * (k + 2);
        let mut mv = vec![0.0; (nx + ny) * 2];
        // x-comp monomials: (x^a y^b, 0)
        let mut idx = 0;
        for a in 0..=k + 1 {
            for b in 0..=k {
                let v = x.powi(a as i32) * y.powi(b as i32);
                mv[idx * 2] = v;
                mv[idx * 2 + 1] = 0.0;
                idx += 1;
            }
        }
        // y-comp monomials: (0, x^a y^b)
        for a in 0..=k {
            for b in 0..=k + 1 {
                let v = x.powi(a as i32) * y.powi(b as i32);
                mv[idx * 2] = 0.0;
                mv[idx * 2 + 1] = v;
                idx += 1;
            }
        }
        for i in 0..n {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for j in 0..n {
                let c = d.coeff[i * n + j];
                vx += c * mv[j * 2];
                vy += c * mv[j * 2 + 1];
            }
            values[i * 2] = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order;
        let d = quad_data(k);
        let n = d.n;
        let x = xi[0];
        let y = xi[1];
        let nx = (k + 2) * (k + 1);
        let ny = (k + 1) * (k + 2);
        let mt = nx + ny;
        let mut dm = vec![0.0; mt];
        // x-comp div: ∂/∂x (x^a y^b) = a·x^(a-1) y^b
        let mut idx = 0;
        for a in 0..=k + 1 {
            for b in 0..=k {
                let xm1 = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
                dm[idx] = (a as f64) * xm1 * y.powi(b as i32);
                idx += 1;
            }
        }
        // y-comp div: ∂/∂y (x^a y^b) = b·x^a y^(b-1)
        for a in 0..=k {
            for b in 0..=k + 1 {
                let ym1 = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
                dm[idx] = (b as f64) * x.powi(a as i32) * ym1;
                idx += 1;
            }
        }
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += d.coeff[i * n + j] * dm[j];
            }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::quad_rule_01(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;
        let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        let step = 1.0 / (k + 2) as f64;
        for p in 0..=k {
            c.push(vec![step * (p + 1) as f64, 0.0]);
        }
        for p in 0..=k {
            c.push(vec![1.0, step * (p + 1) as f64]);
        }
        for p in 0..=k {
            c.push(vec![step * (p + 1) as f64, 1.0]);
        }
        for p in 0..=k {
            c.push(vec![0.0, step * (p + 1) as f64]);
        }
        for _ in c.len()..n {
            c.push(vec![0.5, 0.5]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn n_dofs() {
        assert_eq!(QuadRTk::new(0).n_dofs(), 4);
        assert_eq!(QuadRTk::new(1).n_dofs(), 12);
        assert_eq!(QuadRTk::new(2).n_dofs(), 24);
        assert_eq!(QuadRTk::new(3).n_dofs(), 40);
    }
    #[test]
    fn coeff_non_singular() {
        for k in 1..=4 {
            let d = quad_data(k);
            let n = d.n;
            let mut ok = true;
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += d.coeff[j * n + i].abs();
                }
                if s < 1e-10 {
                    ok = false;
                }
            }
            assert!(ok, "k={k}");
        }
    }
    #[test]
    fn finite() {
        for k in 0..=3 {
            let e = QuadRTk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 2];
            let mut d = vec![0.0; n];
            for p in &[(0.2, 0.3), (0.7, 0.2), (0.5, 0.5)] {
                e.eval_basis_vec(&[p.0, p.1], &mut v);
                e.eval_div(&[p.0, p.1], &mut d);
                for &val in v.iter().chain(d.iter()) {
                    assert!(val.is_finite(), "k={k} at ({},{})", p.0, p.1);
                }
            }
        }
    }
}
