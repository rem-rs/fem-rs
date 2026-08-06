use crate::quadrature::{tet_rule, tri_rule};
use crate::reference::{QuadratureRule, ReferenceElement};

// Bernstein on triangle: B_ijk = (i+j+k)!/(i!j!k!) · x^i · y^j · (1-x-y)^k
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut r = 1.0;
    for i in 1..=k {
        r *= (n - i + 1) as f64 / i as f64;
    }
    r
}
fn multinomial3(i: usize, j: usize, k: usize) -> f64 {
    let n = i + j + k;
    binom(n, i) * binom(n - i, j)
}

pub struct BernsteinTriPk {
    p: usize,
    ijk: Vec<(usize, usize, usize)>,
}

impl BernsteinTriPk {
    pub fn new(p: usize) -> Self {
        let mut ijk = Vec::new();
        for k in 0..=p {
            for j in 0..=(p - k) {
                let i = p - j - k;
                ijk.push((i, j, k));
            }
        }
        Self { p, ijk }
    }
}

impl ReferenceElement for BernsteinTriPk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        (self.p + 1) * (self.p + 2) / 2
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let z = 1.0 - x - y;
        for (idx, &(i, j, k)) in self.ijk.iter().enumerate() {
            let c = multinomial3(i, j, k);
            values[idx] = c * x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let z = 1.0 - x - y;
        for (idx, &(i, j, k)) in self.ijk.iter().enumerate() {
            let c = multinomial3(i, j, k);
            let _b = c * x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32);
            let dx = if i > 0 {
                i as f64 * c * x.powi(i as i32 - 1) * y.powi(j as i32) * z.powi(k as i32)
                    - (if k > 0 {
                        k as f64 * c * x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32 - 1)
                    } else {
                        0.0
                    })
            } else {
                -(if k > 0 {
                    k as f64 * c * x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32 - 1)
                } else {
                    0.0
                })
            };
            let dy = if j > 0 {
                j as f64 * c * x.powi(i as i32) * y.powi(j as i32 - 1) * z.powi(k as i32)
                    - (if k > 0 {
                        k as f64 * c * x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32 - 1)
                    } else {
                        0.0
                    })
            } else {
                -(if k > 0 {
                    k as f64 * c * x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32 - 1)
                } else {
                    0.0
                })
            };
            grads[idx * 2] = dx;
            grads[idx * 2 + 1] = dy;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let mut c = Vec::new();
        for k in 0..=self.p {
            for j in 0..=(self.p - k) {
                let i = self.p - j - k;
                if self.p > 0 {
                    c.push(vec![i as f64 / self.p as f64, j as f64 / self.p as f64]);
                } else {
                    c.push(vec![1.0 / 3.0, 1.0 / 3.0]);
                }
            }
        }
        c
    }
}

// ─── Bernstein on tetrahedron ────────────────────────────────────────────────

fn multinomial4(i: usize, j: usize, k: usize, l: usize) -> f64 {
    let n = i + j + k + l;
    binom(n, i) * binom(n - i, j) * binom(n - i - j, k)
}

pub struct BernsteinTetPk {
    p: usize,
    ijkl: Vec<(usize, usize, usize, usize)>,
}

impl BernsteinTetPk {
    pub fn new(p: usize) -> Self {
        let mut ijkl = Vec::new();
        for l in 0..=p {
            for k in 0..=(p - l) {
                for j in 0..=(p - k - l) {
                    let i = p - j - k - l;
                    ijkl.push((i, j, k, l));
                }
            }
        }
        Self { p, ijkl }
    }
}

impl ReferenceElement for BernsteinTetPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        (self.p + 1) * (self.p + 2) * (self.p + 3) / 6
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let w = 1.0 - x - y - z;
        for (idx, &(i, j, k, l)) in self.ijkl.iter().enumerate() {
            values[idx] = multinomial4(i, j, k, l)
                * x.powi(i as i32)
                * y.powi(j as i32)
                * z.powi(k as i32)
                * w.powi(l as i32);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let w = 1.0 - x - y - z;
        for (idx, &(i, j, k, l)) in self.ijkl.iter().enumerate() {
            let c = multinomial4(i, j, k, l);
            let xi = if i > 0 { x.powi(i as i32 - 1) } else { 0.0 };
            let xi1 = if i > 0 { x.powi(i as i32) } else { 1.0 };
            let yj = y.powi(j as i32);
            let yj1 = if j > 0 { y.powi(j as i32 - 1) } else { 0.0 };
            let zk = z.powi(k as i32);
            let zk1 = if k > 0 { z.powi(k as i32 - 1) } else { 0.0 };
            let wl = w.powi(l as i32);
            let wl1 = if l > 0 { w.powi(l as i32 - 1) } else { 0.0 };

            let mut dx = 0.0;
            let mut dy = 0.0;
            let mut dz = 0.0;
            if i > 0 {
                dx += i as f64 * c * xi * yj * zk * wl;
            }
            if l > 0 {
                dx -= l as f64 * c * xi1 * yj * zk * wl1;
            }
            if j > 0 {
                dy += j as f64 * c * xi1 * yj1 * zk * wl;
            }
            if l > 0 {
                dy -= l as f64 * c * xi1 * yj * zk * wl1;
            }
            if k > 0 {
                dz += k as f64 * c * xi1 * yj * zk1 * wl;
            }
            if l > 0 {
                dz -= l as f64 * c * xi1 * yj * zk * wl1;
            }
            grads[idx * 3] = dx;
            grads[idx * 3 + 1] = dy;
            grads[idx * 3 + 2] = dz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let mut c = Vec::new();
        for l in 0..=self.p {
            for k in 0..=(self.p - l) {
                for j in 0..=(self.p - k - l) {
                    let i = self.p - j - k - l;
                    if self.p > 0 {
                        c.push(vec![
                            i as f64 / self.p as f64,
                            j as f64 / self.p as f64,
                            k as f64 / self.p as f64,
                        ]);
                    } else {
                        c.push(vec![0.25, 0.25, 0.25]);
                    }
                }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pou(e: &dyn ReferenceElement) {
        let q = e.quadrature(4);
        let mut p = vec![0.0; e.n_dofs()];
        for pt in &q.points {
            e.eval_basis(pt, &mut p);
            let s: f64 = p.iter().sum();
            assert!((s - 1.).abs() < 1e-12, "POU={s}");
        }
    }
    fn ndofs(e: &dyn ReferenceElement, exp: usize) {
        assert_eq!(e.n_dofs(), exp);
    }

    #[test]
    fn bt1() {
        ndofs(&BernsteinTriPk::new(1), 3);
        pou(&BernsteinTriPk::new(1));
    }
    #[test]
    fn bt2() {
        ndofs(&BernsteinTriPk::new(2), 6);
        pou(&BernsteinTriPk::new(2));
    }
    #[test]
    fn bt3() {
        ndofs(&BernsteinTriPk::new(3), 10);
        pou(&BernsteinTriPk::new(3));
    }
    #[test]
    fn bt4() {
        ndofs(&BernsteinTriPk::new(4), 15);
        pou(&BernsteinTriPk::new(4));
    }
    #[test]
    fn btet1() {
        ndofs(&BernsteinTetPk::new(1), 4);
        pou(&BernsteinTetPk::new(1));
    }
    #[test]
    fn btet2() {
        ndofs(&BernsteinTetPk::new(2), 10);
        pou(&BernsteinTetPk::new(2));
    }
    #[test]
    fn btet3() {
        ndofs(&BernsteinTetPk::new(3), 20);
        pou(&BernsteinTetPk::new(3));
    }
}
