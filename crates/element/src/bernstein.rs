//! Bernstein polynomial basis.
//!
//! B_{i,p}(t) = C(p,i) · t^i · (1-t)^{p-i},  t ∈ [0,1]
//!
//! Used for Bézier curves, IGA, and as an alternative well-conditioned
//! basis for high-order finite elements.

use crate::quadrature::seg_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

/// Precomputed binomial coefficients: binom[n][k] = C(n,k) for n ≤ MAX_DEG.
const MAX_DEG: usize = 15;
static BINOM: [[f64; MAX_DEG + 1]; MAX_DEG + 1] = {
    let mut b = [[0.0; MAX_DEG + 1]; MAX_DEG + 1];
    let mut n = 0;
    while n <= MAX_DEG {
        b[n][0] = 1.0;
        b[n][n] = 1.0;
        let mut k = 1;
        while k < n {
            b[n][k] = b[n - 1][k - 1] + b[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    b
};

/// Evaluate all degree-p Bernstein basis polynomials at t ∈ [0,1].
/// values[i] = B_{i,p}(t)
pub fn bernstein_vals(p: usize, t: f64) -> Vec<f64> {
    let mut v = vec![0.0; p + 1];
    match p {
        0 => v[0] = 1.0,
        1 => { v[0] = 1.0 - t; v[1] = t; }
        _ => {
            // de Casteljau / recurrence: B_{i,p} = (1-t)·B_{i,p-1} + t·B_{i-1,p-1}
            let mut prev = vec![1.0]; // p=0
            for deg in 1..=p {
                let mut cur = vec![0.0; deg + 1];
                cur[0] = (1.0 - t) * prev[0];
                for i in 1..deg {
                    cur[i] = (1.0 - t) * prev[i] + t * prev[i - 1];
                }
                cur[deg] = t * prev[deg - 1];
                prev = cur;
            }
            v.copy_from_slice(&prev);
        }
    }
    v
}

/// Evaluate all degree-p Bernstein basis derivatives at t ∈ [0,1].
/// ders[i] = d/dt B_{i,p}(t)
pub fn bernstein_ders(p: usize, t: f64) -> Vec<f64> {
    if p == 0 { return vec![0.0]; }
    let v_low = bernstein_vals(p - 1, t);
    let mut d = Vec::with_capacity(p + 1);
    let pf = p as f64;
    d.push(-pf * v_low[0]);
    for i in 1..p {
        d.push(pf * (v_low[i - 1] - v_low[i]));
    }
    d.push(pf * v_low[p - 1]);
    d
}

/// Evaluate all 2D tensor-product Bernstein basis functions at (ξ, η) ∈ [0,1]².
/// values[j*(p+1)+i] = B_{i,p}(ξ)·B_{j,p}(η)  (row-major, j outer, i inner)
pub fn bernstein_vals_2d(p: usize, xi: f64, eta: f64) -> Vec<f64> {
    let vx = bernstein_vals(p, xi);
    let vy = bernstein_vals(p, eta);
    let np1 = p + 1;
    let mut vals = Vec::with_capacity(np1 * np1);
    for j in 0..np1 {
        for i in 0..np1 {
            vals.push(vx[i] * vy[j]);
        }
    }
    vals
}

/// Evaluate all 3D tensor-product Bernstein basis functions at (ξ,η,ζ) ∈ [0,1]³.
/// values[k*(p+1)² + j*(p+1) + i] = B_{i,p}(ξ)·B_{j,p}(η)·B_{k,p}(ζ)
pub fn bernstein_vals_3d(p: usize, xi: f64, eta: f64, zeta: f64) -> Vec<f64> {
    let vx = bernstein_vals(p, xi);
    let vy = bernstein_vals(p, eta);
    let vz = bernstein_vals(p, zeta);
    let np1 = p + 1;
    let np12 = np1 * np1;
    let mut vals = Vec::with_capacity(np12 * np1);
    for k in 0..np1 {
        for j in 0..np1 {
            for i in 0..np1 {
                vals.push(vx[i] * vy[j] * vz[k]);
            }
        }
    }
    vals
}

/// Evaluate all 2D Bernstein basis values and gradients at (ξ, η) ∈ [0,1]².
/// Returns (values, grads) where:
///   values[idx] = B_{i,p}(ξ)·B_{j,p}(η)
///   grads[idx*2+0] = ∂/∂ξ B_{i,p}(ξ)·B_{j,p}(η)
///   grads[idx*2+1] = B_{i,p}(ξ)·∂/∂η B_{j,p}(η)
/// with idx = j*(p+1)+i.
pub fn bernstein_ders_2d(p: usize, xi: f64, eta: f64) -> (Vec<f64>, Vec<f64>) {
    let vx = bernstein_vals(p, xi);
    let vy = bernstein_vals(p, eta);
    let dx = bernstein_ders(p, xi);
    let dy = bernstein_ders(p, eta);
    let np1 = p + 1;
    let n = np1 * np1;
    let mut vals = Vec::with_capacity(n);
    let mut grads = vec![0.0; n * 2];
    for j in 0..np1 {
        for i in 0..np1 {
            let idx = j * np1 + i;
            vals.push(vx[i] * vy[j]);
            grads[idx * 2]     = dx[i] * vy[j];     // d/dξ
            grads[idx * 2 + 1] = vx[i] * dy[j];     // d/dη
        }
    }
    (vals, grads)
}

/// Compute the binomial coefficient C(n,k) as f64.
fn binom(n: usize, k: usize) -> f64 {
    if k > n || k > MAX_DEG || n > MAX_DEG { return 0.0; }
    BINOM[n][k]
}

/// Reference element on [0,1] using Bernstein basis of given order.
pub struct BernsteinSegPk {
    order: usize,
}

impl BernsteinSegPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "BernsteinSegPk: order must be ≥ 1");
        Self { order: p }
    }
}

impl ReferenceElement for BernsteinSegPk {
    fn dim(&self) -> u8 { 1 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { self.order + 1 }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let v = bernstein_vals(self.order, xi[0]);
        values.copy_from_slice(&v);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let d = bernstein_ders(self.order, xi[0]);
        grads.copy_from_slice(&d);
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { seg_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        (0..=self.order).map(|i| vec![i as f64 / self.order as f64]).collect()
    }
}

/// Reference element on [-1,1]² using tensor-product Bernstein basis.
pub struct BernsteinQuadPk {
    p: usize,
}

impl BernsteinQuadPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "BernsteinQuadPk: order must be ≥ 1");
        Self { p }
    }
}

impl ReferenceElement for BernsteinQuadPk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.p as u8 }
    fn n_dofs(&self) -> usize { (self.p + 1) * (self.p + 1) }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let t = (xi[0] + 1.0) * 0.5;
        let u = (xi[1] + 1.0) * 0.5;
        let v = bernstein_vals_2d(self.p, t, u);
        values.copy_from_slice(&v);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let t = (xi[0] + 1.0) * 0.5;
        let u = (xi[1] + 1.0) * 0.5;
        let (_, g) = bernstein_ders_2d(self.p, t, u);
        // chain rule: d/dξ = d/dt · 1/2,  d/dη = d/du · 1/2
        let np1 = self.p + 1;
        let n = np1 * np1;
        for idx in 0..n {
            grads[idx * 2]     = g[idx * 2]     * 0.5;
            grads[idx * 2 + 1] = g[idx * 2 + 1] * 0.5;
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        crate::quadrature::quad_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let np1 = self.p + 1;
        let mut coords = Vec::with_capacity(np1 * np1);
        for j in 0..np1 {
            let eta = -1.0 + 2.0 * j as f64 / self.p as f64;
            for i in 0..np1 {
                let xi = -1.0 + 2.0 * i as f64 / self.p as f64;
                coords.push(vec![xi, eta]);
            }
        }
        coords
    }
}

/// Reference element on [-1,1]³ using tensor-product Bernstein basis.
pub struct BernsteinHexPk {
    p: usize,
}

impl BernsteinHexPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "BernsteinHexPk: order must be ≥ 1");
        Self { p }
    }
}

impl ReferenceElement for BernsteinHexPk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.p as u8 }
    fn n_dofs(&self) -> usize { (self.p + 1) * (self.p + 1) * (self.p + 1) }
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let t = (xi[0] + 1.0) * 0.5;
        let u = (xi[1] + 1.0) * 0.5;
        let v = (xi[2] + 1.0) * 0.5;
        let vals = bernstein_vals_3d(self.p, t, u, v);
        values.copy_from_slice(&vals);
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let t = (xi[0] + 1.0) * 0.5;
        let u = (xi[1] + 1.0) * 0.5;
        let v = (xi[2] + 1.0) * 0.5;
        let vx = bernstein_vals(self.p, t);
        let vy = bernstein_vals(self.p, u);
        let vz = bernstein_vals(self.p, v);
        let dx = bernstein_ders(self.p, t);
        let dy = bernstein_ders(self.p, u);
        let dz = bernstein_ders(self.p, v);
        let np1 = self.p + 1;
        let np12 = np1 * np1;
        for k in 0..np1 {
            for j in 0..np1 {
                for i in 0..np1 {
                    let idx = k * np12 + j * np1 + i;
                    grads[idx * 3]     = dx[i] * vy[j] * vz[k] * 0.5;
                    grads[idx * 3 + 1] = vx[i] * dy[j] * vz[k] * 0.5;
                    grads[idx * 3 + 2] = vx[i] * vy[j] * dz[k] * 0.5;
                }
            }
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        crate::quadrature::hex_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let np1 = self.p + 1;
        let np12 = np1 * np1;
        let mut coords = Vec::with_capacity(np12 * np1);
        for k in 0..np1 {
            let zeta = -1.0 + 2.0 * k as f64 / self.p as f64;
            for j in 0..np1 {
                let eta = -1.0 + 2.0 * j as f64 / self.p as f64;
                for i in 0..np1 {
                    let xi = -1.0 + 2.0 * i as f64 / self.p as f64;
                    coords.push(vec![xi, eta, zeta]);
                }
            }
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernstein_p0() {
        let v = bernstein_vals(0, 0.5);
        assert!((v[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bernstein_p1() {
        let v = bernstein_vals(1, 0.3);
        assert!((v[0] - 0.7).abs() < 1e-14);
        assert!((v[1] - 0.3).abs() < 1e-14);
    }

    #[test]
    fn bernstein_p2_endpoints() {
        let v0 = bernstein_vals(2, 0.0);
        assert!((v0[0] - 1.0).abs() < 1e-14);
        assert!((v0[1] - 0.0).abs() < 1e-14);
        assert!((v0[2] - 0.0).abs() < 1e-14);
        let v1 = bernstein_vals(2, 1.0);
        assert!((v1[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bernstein_partition_of_unity() {
        for p in 0..=8 {
            for &t in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let v = bernstein_vals(p, t);
                let s: f64 = v.iter().sum();
                assert!((s - 1.0).abs() < 1e-13, "p={p} t={t}: sum={s}");
            }
        }
    }

    #[test]
    fn bernstein_ders_fd() {
        let h = 1e-8;
        for p in 1..=5 {
            for &t in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let d_analytic = bernstein_ders(p, t);
                let vp = bernstein_vals(p, t + h);
                let vm = bernstein_vals(p, t - h);
                for i in 0..=p {
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    assert!((d_analytic[i] - fd).abs() < 1e-7,
                        "p={p} t={t} i={i}: analytic={} fd={}", d_analytic[i], fd);
                }
            }
        }
    }

    #[test]
    fn bernstein_seg_element() {
        let elem = BernsteinSegPk::new(3);
        assert_eq!(elem.n_dofs(), 4);
        let mut v = vec![0.0; 4];
        elem.eval_basis(&[0.5], &mut v);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-14);
        elem.eval_basis(&[0.0], &mut v);
        assert!((v[0] - 1.0).abs() < 1e-14);
        elem.eval_basis(&[1.0], &mut v);
        assert!((v[3] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bernstein_quad_pou() {
        for p in 1..=3 {
            let elem = BernsteinQuadPk::new(p);
            let n = elem.n_dofs();
            let mut v = vec![0.0; n];
            for &xi in &[-0.8, 0.0, 0.5] {
                for &eta in &[-0.6, 0.0, 0.7] {
                    elem.eval_basis(&[xi, eta], &mut v);
                    let s: f64 = v.iter().sum();
                    assert!((s - 1.0).abs() < 1e-13, "Quad P{p} at ({xi},{eta}): sum={s}");
                }
            }
        }
    }

    #[test]
    fn bernstein_quad_n_dofs() {
        for p in 1..=3 {
            let elem = BernsteinQuadPk::new(p);
            assert_eq!(elem.n_dofs(), (p + 1) * (p + 1));
        }
    }

    #[test]
    fn bernstein_quad_grad_fd() {
        let h = 1e-8;
        let elem = BernsteinQuadPk::new(2);
        let n = elem.n_dofs();
        let mut v = vec![0.0; n];
        let mut g = vec![0.0; n * 2];
        let pts = [-0.5, 0.0, 0.5];
        for &xi in &pts {
            for &eta in &pts {
                elem.eval_grad_basis(&[xi, eta], &mut g);
                // FD in ξ
                let mut vp = vec![0.0; n];
                let mut vm = vec![0.0; n];
                elem.eval_basis(&[xi + h, eta], &mut vp);
                elem.eval_basis(&[xi - h, eta], &mut vm);
                for i in 0..n {
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    assert!((g[i * 2] - fd).abs() < 1e-6,
                        "Quad P2 d/dξ at ({xi},{eta}) i={i}: analytic={} fd={}", g[i*2], fd);
                }
                // FD in η
                elem.eval_basis(&[xi, eta + h], &mut vp);
                elem.eval_basis(&[xi, eta - h], &mut vm);
                for i in 0..n {
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    assert!((g[i * 2 + 1] - fd).abs() < 1e-6,
                        "Quad P2 d/dη at ({xi},{eta}) i={i}: analytic={} fd={}", g[i*2+1], fd);
                }
            }
        }
    }

    #[test]
    fn bernstein_hex_pou() {
        for p in 1..=2 {
            let elem = BernsteinHexPk::new(p);
            let n = elem.n_dofs();
            let mut v = vec![0.0; n];
            for &xi in &[-0.5, 0.3] {
                for &eta in &[-0.4, 0.5] {
                    for &zeta in &[-0.6, 0.4] {
                        elem.eval_basis(&[xi, eta, zeta], &mut v);
                        let s: f64 = v.iter().sum();
                        assert!((s - 1.0).abs() < 1e-13,
                            "Hex P{p} at ({xi},{eta},{zeta}): sum={s}");
                    }
                }
            }
        }
    }

    #[test]
    fn bernstein_hex_n_dofs() {
        for p in 1..=2 {
            let elem = BernsteinHexPk::new(p);
            assert_eq!(elem.n_dofs(), (p + 1) * (p + 1) * (p + 1));
        }
    }

    #[test]
    fn bernstein_hex_grad_fd() {
        let h = 1e-8;
        let elem = BernsteinHexPk::new(2);
        let n = elem.n_dofs();
        let mut g = vec![0.0; n * 3];
        let pts = [-0.4, 0.2, 0.5];
        for &xi in &pts {
            for &eta in &pts {
                for &zeta in &pts {
                    elem.eval_grad_basis(&[xi, eta, zeta], &mut g);
                    // FD in ξ
                    let mut vp = vec![0.0; n];
                    let mut vm = vec![0.0; n];
                    elem.eval_basis(&[xi + h, eta, zeta], &mut vp);
                    elem.eval_basis(&[xi - h, eta, zeta], &mut vm);
                    for i in 0..n {
                        let fd = (vp[i] - vm[i]) / (2.0 * h);
                        assert!((g[i * 3] - fd).abs() < 1e-6,
                            "Hex P2 d/dξ at ({xi},{eta},{zeta}) i={i}: analytic={} fd={}", g[i*3], fd);
                    }
                    // FD in η
                    elem.eval_basis(&[xi, eta + h, zeta], &mut vp);
                    elem.eval_basis(&[xi, eta - h, zeta], &mut vm);
                    for i in 0..n {
                        let fd = (vp[i] - vm[i]) / (2.0 * h);
                        assert!((g[i * 3 + 1] - fd).abs() < 1e-6,
                            "Hex P2 d/dη at ({xi},{eta},{zeta}) i={i}: analytic={} fd={}", g[i*3+1], fd);
                    }
                    // FD in ζ
                    elem.eval_basis(&[xi, eta, zeta + h], &mut vp);
                    elem.eval_basis(&[xi, eta, zeta - h], &mut vm);
                    for i in 0..n {
                        let fd = (vp[i] - vm[i]) / (2.0 * h);
                        assert!((g[i * 3 + 2] - fd).abs() < 1e-6,
                            "Hex P2 d/dζ at ({xi},{eta},{zeta}) i={i}: analytic={} fd={}", g[i*3+2], fd);
                    }
                }
            }
        }
    }
}
