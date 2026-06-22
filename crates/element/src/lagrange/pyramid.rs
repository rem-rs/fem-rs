//! Arbitrary-order Lagrange element on the reference pyramid.
//!
//! Reference pyramid: vertices (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1).
//! Domain: x ∈ [0, 1-z], y ∈ [0, 1-z], z ∈ [0,1]. Volume = 1/3.
//!
//! Uses the collapsed-coordinate formulation:
//! - Collapsed coordinates: r = x/(1-z), s = y/(1-z), t = z, all in [0,1].
//! - Nodes are placed at equispaced grid points: (r,s,t) = (i/p, j/p, k/p) where
//!   i,j ≤ p-k. Total DOFs = Σ_{k=0}^{p} (p-k+1)² = (p+1)(p+2)(2p+3)/6.
//!
//! Basis: φ(x,y,z) = L_k(t) · L_i^{(p-k)}(r) · L_j^{(p-k)}(s)
//! where L_n^{(d)} is the standard degree-d Lagrange polynomial through
//! equispaced nodes on [0,1].

use crate::quadrature::pyramid_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

fn lagrange_1d_val(i: usize, degree: usize, xi: f64) -> f64 {
    if degree == 0 { return 1.0; }
    let d = degree as f64;
    let t = d * xi;
    let mut val = 1.0;
    let tn = i as f64;
    for m in 0..=degree {
        if m != i {
            val *= (t - m as f64) / (tn - m as f64);
        }
    }
    val
}

fn lagrange_1d_deriv(i: usize, degree: usize, xi: f64) -> f64 {
    if degree == 0 { return 0.0; }
    let d = degree as f64;
    let t = d * xi;
    let mut sum = 0.0;
    let tn = i as f64;
    for k in 0..=degree {
        if k != i {
            let mut term = 1.0;
            for m in 0..=degree {
                if m != i && m != k {
                    term *= (t - m as f64) / (tn - m as f64);
                }
            }
            sum += term / (tn - k as f64);
        }
    }
    d * sum
}

/// Arbitrary-order Lagrange element on the reference pyramid.
///
/// DOF ordering: layer-by-layer from base (k=0) to apex (k=p).
/// Within each layer, row-major ordering over the (p-k+1)×(p-k+1) grid.
pub struct PyramidPk {
    order: usize,
    layer_offset: Vec<usize>,
}

impl PyramidPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let mut layer_offset = Vec::with_capacity(p + 2);
        let mut off = 0usize;
        for k in 0..=p {
            layer_offset.push(off);
            let n = p - k + 1;
            off += n * n;
        }
        layer_offset.push(off);
        Self { order: p, layer_offset }
    }

    fn layer_n(&self, k: usize) -> usize {
        self.order - k + 1
    }

    fn n_dofs_total(&self) -> usize {
        *self.layer_offset.last().unwrap()
    }

    fn dof_index(&self, k: usize, i: usize, j: usize) -> usize {
        let n = self.layer_n(k);
        self.layer_offset[k] + j * n + i
    }

    #[allow(dead_code)]
    fn for_each_dof<F: FnMut(usize, usize, usize, usize)>(&self, mut f: F) {
        let p = self.order;
        for k in 0..=p {
            let n = p - k + 1;
            for j in 0..n {
                for i in 0..n {
                    f(k, i, j, self.dof_index(k, i, j));
                }
            }
        }
    }
}

impl ReferenceElement for PyramidPk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { self.n_dofs_total() }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let p = self.order;

        if (z - 1.0).abs() < 1e-14 {
            for v in values.iter_mut() { *v = 0.0; }
            let apex = self.dof_index(p, 0, 0);
            values[apex] = 1.0;
            return;
        }

        let inv_one_minus_z = 1.0 / (1.0 - z);
        let r = x * inv_one_minus_z;
        let s = y * inv_one_minus_z;

        for k in 0..=p {
            let lz = lagrange_1d_val(k, p, z);
            let layer_deg = p - k;
            let n = layer_deg + 1;
            for j in 0..n {
                let ls = lagrange_1d_val(j, layer_deg, s);
                for i in 0..n {
                    let lr = lagrange_1d_val(i, layer_deg, r);
                    values[self.dof_index(k, i, j)] = lz * lr * ls;
                }
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let p = self.order;

        if (z - 1.0).abs() < 1e-14 {
            for g in grads.iter_mut() { *g = 0.0; }
            return;
        }

        let inv_one_minus_z = 1.0 / (1.0 - z);

        for k in 0..=p {
            let lz = lagrange_1d_val(k, p, z);
            let dlz = lagrange_1d_deriv(k, p, z);
            let layer_deg = p - k;
            let n = layer_deg + 1;

            let (mut lr, mut ls) = if n > 1 {
                (vec![0.0; n], vec![0.0; n])
            } else {
                // Special case for apex layer (n=1): constant basis functions
                // pre-evaluated lr/ls arrays
                let mut lr = Vec::with_capacity(1);
                let mut ls = Vec::with_capacity(1);
                lr.push(1.0);
                ls.push(1.0);
                (lr, ls)
            };

            let r = x * inv_one_minus_z;
            let s_val = y * inv_one_minus_z;

            for i in 0..n { lr[i] = lagrange_1d_val(i, layer_deg, r); }
            for j in 0..n { ls[j] = lagrange_1d_val(j, layer_deg, s_val); }

            let (mut dlr, mut dls) = if n > 1 {
                (vec![0.0; n], vec![0.0; n])
            } else {
                (vec![0.0; 1], vec![0.0; 1])
            };

            for i in 0..n { dlr[i] = lagrange_1d_deriv(i, layer_deg, r); }
            for j in 0..n { dls[j] = lagrange_1d_deriv(j, layer_deg, s_val); }

            for j in 0..n {
                for i in 0..n {
                    let dof = self.dof_index(k, i, j);
                    let lr_i = lr[i];
                    let ls_j = ls[j];
                    let dlr_i = dlr[i];
                    let dls_j = dls[j];

                    grads[dof * 3]     = lz * dlr_i * ls_j * inv_one_minus_z;
                    grads[dof * 3 + 1] = lz * lr_i * dls_j * inv_one_minus_z;
                    grads[dof * 3 + 2] = dlz * lr_i * ls_j
                        + lz * dlr_i * ls_j * x * inv_one_minus_z * inv_one_minus_z
                        + lz * lr_i * dls_j * y * inv_one_minus_z * inv_one_minus_z;
                }
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { pyramid_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order;
        let mut coords = Vec::with_capacity(self.n_dofs_total());
        for k in 0..=p {
            let z = k as f64 / p as f64;
            let n = p - k + 1;
            for j in 0..n {
                let y = j as f64 / p as f64;
                for i in 0..n {
                    let x = i as f64 / p as f64;
                    coords.push(vec![x, y, z]);
                }
            }
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let order = elem.order();
        let rule = elem.quadrature((2 * order + 2).min(15));
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-10,
                "POU failed at {:?}: sum={s}", pt);
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let order = elem.order();
        let rule = elem.quadrature((2 * order + 2).min(15));
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-10,
                    "grad sum d={d} = {s} at {:?}", pt);
            }
        }
    }

    fn check_nodal_interp(elem: &dyn ReferenceElement) {
        let coords = elem.dof_coords();
        let n = elem.n_dofs();
        let mut phi = vec![0.0_f64; n];
        for (i, coord) in coords.iter().enumerate() {
            elem.eval_basis(coord, &mut phi);
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-10,
                    "nodal interp: node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn pyramid_pou() { for p in 1..=3 { check_pou(&PyramidPk::new(p)); } }
    #[test]
    fn pyramid_grad_zero() { for p in 1..=3 { check_grad_zero(&PyramidPk::new(p)); } }
    #[test]
    fn pyramid_nodal_interp() { for p in 1..=3 { check_nodal_interp(&PyramidPk::new(p)); } }
    #[test]
    fn pyramid_n_dofs() {
        assert_eq!(PyramidPk::new(1).n_dofs(), 5);
        assert_eq!(PyramidPk::new(2).n_dofs(), 14);
        assert_eq!(PyramidPk::new(3).n_dofs(), 30);
        // (p+1)(p+2)(2p+3)/6
        assert_eq!(PyramidPk::new(4).n_dofs(), 55);
    }

    #[test]
    fn pyramid_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=3 {
            let elem = PyramidPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
                vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n*3]
            );
            let test_pts: &[[f64; 3]] = if p == 1 {
                &[[0.3, 0.2, 0.1]]
            } else {
                &[[0.2, 0.3, 0.15], [0.4, 0.1, 0.25]]
            };
            for pt in test_pts {
                let (x, y, z) = (pt[0], pt[1], pt[2]);
                if x + z > 0.95 || y + z > 0.95 { continue; }
                if z > 0.8 { continue; }
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x+h, y, z], &mut vx);
                elem.eval_basis(&[x, y+h, z], &mut vy);
                elem.eval_basis(&[x, y, z+h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!((grads[i*3] - fd_x).abs() < 1e-4, "p={p} ({x},{y},{z}) i={i} gx");
                    assert!((grads[i*3+1] - fd_y).abs() < 1e-4, "p={p} ({x},{y},{z}) i={i} gy");
                    assert!((grads[i*3+2] - fd_z).abs() < 1e-4, "p={p} ({x},{y},{z}) i={i} gz");
                }
            }
        }
    }
}
