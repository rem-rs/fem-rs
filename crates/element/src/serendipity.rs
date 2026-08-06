use crate::quadrature::{hex_rule, quad_rule};
use crate::reference::{QuadratureRule, ReferenceElement};

fn nd(p: usize) -> Vec<f64> {
    (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect()
}

// Quad serendipity: monomials {x^i y^j : i=0 or i=p or j=0 or j=p}
fn mono_ij(p: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for j in 0..=p {
        for i in 0..=p {
            if i == 0 || i == p || j == 0 || j == p {
                v.push((i, j));
            }
        }
    }
    v
}
fn nodes_2d(p: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for j in 0..=p {
        for i in 0..=p {
            if i == 0 || i == p || j == 0 || j == p {
                v.push((i, j));
            }
        }
    }
    v
}
fn pow(x: f64, e: usize) -> f64 {
    if e == 0 {
        1.
    } else {
        x.powi(e as i32)
    }
}

fn build_coef2d(p: usize) -> Vec<f64> {
    let n = 4 * p;
    let xv = nd(p);
    let mi = mono_ij(p);
    let nds = nodes_2d(p);
    let mut v = vec![0.; n * n];
    for r in 0..n {
        let (ni, nj) = nds[r];
        let xi = xv[ni] + 1.;
        let et = xv[nj] + 1.;
        for c in 0..n {
            let (mi, mj) = mi[c];
            v[r * n + c] = pow(xi, mi) * pow(et, mj);
        }
    }
    let mut inv = vec![0.; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.;
    }
    let mut a = v;
    for c in 0..n {
        let mut mr = c;
        let mut mv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let x = a[r * n + c].abs();
            if x > mv {
                mv = x;
                mr = r
            }
        }
        for j in 0..n {
            a.swap(c * n + j, mr * n + j);
            inv.swap(c * n + j, mr * n + j);
        }
        let pv = a[c * n + c];
        let ip = 1. / pv;
        for j in 0..n {
            a[c * n + j] *= ip;
            inv[c * n + j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= f * a[c * n + j];
                inv[r * n + j] -= f * inv[c * n + j];
            }
        }
    }
    let mut c = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = inv[j * n + i];
        }
    }
    c
}

pub struct QuadSerendipityPk {
    p: usize,
    co: Vec<f64>,
    mi: Vec<(usize, usize)>,
    nds: Vec<(usize, usize)>,
}

impl QuadSerendipityPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        Self {
            p,
            co: build_coef2d(p),
            mi: mono_ij(p),
            nds: nodes_2d(p),
        }
    }
}

impl ReferenceElement for QuadSerendipityPk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        4 * self.p
    }
    fn eval_basis(&self, xi: &[f64], vals: &mut [f64]) {
        let n = self.n_dofs();
        for i in 0..n {
            let mut s = 0.;
            for j in 0..n {
                let (mi, mj) = self.mi[j];
                s += self.co[i * n + j] * pow(xi[0] + 1., mi) * pow(xi[1] + 1., mj);
            }
            vals[i] = s;
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let n = self.n_dofs();
        let u = xi[0] + 1.;
        let v = xi[1] + 1.;
        for i in 0..n {
            let mut sx = 0.;
            let mut sy = 0.;
            for j in 0..n {
                let (mi, mj) = self.mi[j];
                let mx = if mi == 0 {
                    0.
                } else {
                    mi as f64 * pow(u, mi - 1) * pow(v, mj)
                };
                let my = if mj == 0 {
                    0.
                } else {
                    mj as f64 * pow(u, mi) * pow(v, mj - 1)
                };
                sx += self.co[i * n + j] * mx;
                sy += self.co[i * n + j] * my;
            }
            grads[i * 2] = sx;
            grads[i * 2 + 1] = sy;
        }
    }
    fn quadrature(&self, o: u8) -> QuadratureRule {
        quad_rule(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let xv = nd(self.p);
        let mut c = Vec::new();
        for &(i, j) in &self.nds {
            c.push(vec![xv[i], xv[j]]);
        }
        c
    }
}

// Hex serendipity
fn mi3(p: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for k in 0..=p {
        for j in 0..=p {
            for i in 0..=p {
                if i == 0 || i == p || j == 0 || j == p || k == 0 || k == p {
                    v.push((i, j, k));
                }
            }
        }
    }
    v
}
fn nd3(p: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for k in 0..=p {
        for j in 0..=p {
            for i in 0..=p {
                if i == 0 || i == p || j == 0 || j == p || k == 0 || k == p {
                    v.push((i, j, k));
                }
            }
        }
    }
    v
}

fn build_hex(p: usize) -> Vec<f64> {
    let n = {
        let e = p.saturating_sub(1);
        8 + 12 * e + 6 * e * e
    };
    let xv = nd(p);
    let m = mi3(p);
    let d = nd3(p);
    let mut v = vec![0.; n * n];
    for r in 0..n {
        let (ni, nj, nk) = d[r];
        let (xi, et, zt) = (xv[ni] + 1., xv[nj] + 1., xv[nk] + 1.);
        for c in 0..n {
            let (mi, mj, mk) = m[c];
            v[r * n + c] = pow(xi, mi) * pow(et, mj) * pow(zt, mk);
        }
    }
    let mut inv = vec![0.; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.;
    }
    let mut a = v;
    for c in 0..n {
        let mut mr = c;
        let mut mv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let x = a[r * n + c].abs();
            if x > mv {
                mv = x;
                mr = r
            }
        }
        for j in 0..n {
            a.swap(c * n + j, mr * n + j);
            inv.swap(c * n + j, mr * n + j);
        }
        let pv = a[c * n + c];
        if pv.abs() < 1e-14 {
            continue;
        }
        let ip = 1. / pv;
        for j in 0..n {
            a[c * n + j] *= ip;
            inv[c * n + j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= f * a[c * n + j];
                inv[r * n + j] -= f * inv[c * n + j];
            }
        }
    }
    let mut cc = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            cc[i * n + j] = inv[j * n + i];
        }
    }
    cc
}

pub struct HexSerendipityPk {
    p: usize,
    co: Vec<f64>,
    mi: Vec<(usize, usize, usize)>,
    nds: Vec<(usize, usize, usize)>,
}
impl HexSerendipityPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        Self {
            p,
            co: build_hex(p),
            mi: mi3(p),
            nds: nd3(p),
        }
    }
    fn n(&self) -> usize {
        let e = self.p.saturating_sub(1);
        8 + 12 * e + 6 * e * e
    }
}

impl ReferenceElement for HexSerendipityPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        self.n()
    }
    fn eval_basis(&self, xi: &[f64], vals: &mut [f64]) {
        let n = self.n();
        let u = xi[0] + 1.;
        let v = xi[1] + 1.;
        let w = xi[2] + 1.;
        for i in 0..n {
            let mut s = 0.;
            for j in 0..n {
                let (mi, mj, mk) = self.mi[j];
                s += self.co[i * n + j] * pow(u, mi) * pow(v, mj) * pow(w, mk);
            }
            vals[i] = s;
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], g: &mut [f64]) {
        let n = self.n();
        let u = xi[0] + 1.;
        let v = xi[1] + 1.;
        let w = xi[2] + 1.;
        for i in 0..n {
            let mut sx = 0.;
            let mut sy = 0.;
            let mut sz = 0.;
            for j in 0..n {
                let (mi, mj, mk) = self.mi[j];
                let mx = if mi == 0 {
                    0.
                } else {
                    (mi as f64) * pow(u, mi - 1) * pow(v, mj) * pow(w, mk)
                };
                let my = if mj == 0 {
                    0.
                } else {
                    (mj as f64) * pow(u, mi) * pow(v, mj - 1) * pow(w, mk)
                };
                let mz = if mk == 0 {
                    0.
                } else {
                    (mk as f64) * pow(u, mi) * pow(v, mj) * pow(w, mk - 1)
                };
                sx += self.co[i * n + j] * mx;
                sy += self.co[i * n + j] * my;
                sz += self.co[i * n + j] * mz;
            }
            g[i * 3] = sx;
            g[i * 3 + 1] = sy;
            g[i * 3 + 2] = sz;
        }
    }
    fn quadrature(&self, o: u8) -> QuadratureRule {
        hex_rule(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let xv = nd(self.p);
        let mut c = Vec::new();
        for &(i, j, k) in &self.nds {
            c.push(vec![xv[i], xv[j], xv[k]]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pou(e: &dyn ReferenceElement) {
        let q = e.quadrature(4);
        let mut p = vec![0.; e.n_dofs()];
        for pt in &q.points {
            e.eval_basis(pt, &mut p);
            let s: f64 = p.iter().sum();
            assert!((s - 1.).abs() < 1e-12, "POU={s}");
        }
    }
    fn interp(e: &dyn ReferenceElement) {
        let c = e.dof_coords();
        let n = e.n_dofs();
        let mut p = vec![0.; n];
        for (i, cc) in c.iter().enumerate() {
            e.eval_basis(cc, &mut p);
            for j in 0..n {
                let exp = if i == j { 1. } else { 0. };
                assert!((p[j] - exp).abs() < 1e-12);
            }
        }
    }
    fn grad(e: &dyn ReferenceElement, pts: &[[f64; 2]]) {
        let h = 1e-7;
        let n = e.n_dofs();
        let (mut vc, mut vx, mut vy, mut g) =
            (vec![0.; n], vec![0.; n], vec![0.; n], vec![0.; n * 2]);
        for p in pts {
            let (x, y) = (p[0], p[1]);
            e.eval_basis(&[x, y], &mut vc);
            e.eval_basis(&[x + h, y], &mut vx);
            e.eval_basis(&[x, y + h], &mut vy);
            e.eval_grad_basis(&[x, y], &mut g);
            for i in 0..n {
                assert!((g[i * 2] - (vx[i] - vc[i]) / h).abs() < 1e-5);
                assert!((g[i * 2 + 1] - (vy[i] - vc[i]) / h).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn q1() {
        assert_eq!(QuadSerendipityPk::new(1).n_dofs(), 4);
        pou(&QuadSerendipityPk::new(1));
        interp(&QuadSerendipityPk::new(1));
    }
    #[test]
    fn q2() {
        assert_eq!(QuadSerendipityPk::new(2).n_dofs(), 8);
        pou(&QuadSerendipityPk::new(2));
        interp(&QuadSerendipityPk::new(2));
        grad(&QuadSerendipityPk::new(2), &[[0.2, 0.3], [-0.5, 0.7]]);
    }
    #[test]
    fn q3() {
        assert_eq!(QuadSerendipityPk::new(3).n_dofs(), 12);
        pou(&QuadSerendipityPk::new(3));
        interp(&QuadSerendipityPk::new(3));
    }
    #[test]
    fn q4() {
        assert_eq!(QuadSerendipityPk::new(4).n_dofs(), 16);
        pou(&QuadSerendipityPk::new(4));
        interp(&QuadSerendipityPk::new(4));
    }
    #[test]
    fn h1() {
        assert_eq!(HexSerendipityPk::new(1).n_dofs(), 8);
        pou(&HexSerendipityPk::new(1));
        interp(&HexSerendipityPk::new(1));
    }
    #[test]
    fn h2() {
        assert_eq!(HexSerendipityPk::new(2).n_dofs(), 26);
        pou(&HexSerendipityPk::new(2));
        interp(&HexSerendipityPk::new(2));
    }
    #[test]
    fn h3() {
        assert_eq!(HexSerendipityPk::new(3).n_dofs(), 56);
        pou(&HexSerendipityPk::new(3));
    }
}
