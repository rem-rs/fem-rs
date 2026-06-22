//! Arbitrary-order Nedelec-I element on the reference hexahedron [-1,1]³.
//! Edge-based: k DOFs/edge × 12 edges = 12k DOFs.

use crate::reference::VectorReferenceElement;

fn lag(nodes: &[f64], j: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (i, &ni) in nodes.iter().enumerate() {
        if i != j { v *= (x - ni) / (nodes[j] - ni); }
    }
    v
}

fn hat(y: f64, y0: f64) -> f64 { 0.5 * (1.0 + y0 * y) }
fn hat_d(y: f64, y0: f64) -> f64 { 0.5 * y0 }

pub struct HexNDk { order: usize }

impl HexNDk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); HexNDk { order: p } }
    fn nodes(&self) -> Vec<f64> {
        let p = self.order;
        (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect()
    }
}

impl VectorReferenceElement for HexNDk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { 12 * self.order }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order; let nodes = self.nodes();
        let x = xi[0]; let y = xi[1]; let z = xi[2];
        values.fill(0.0);
        let yz = [(-1.0,-1.0),(1.0,-1.0),(1.0,1.0),(-1.0,1.0)];
        // x-directed: edges 0..3
        for (ei, &(y0, z0)) in yz.iter().enumerate() {
            let hy = hat(y, y0); let hz = hat(z, z0);
            for j in 0..p { let d = ei * p + j; values[d * 3] = lag(&nodes, j, x) * hy * hz; }
        }
        // y-directed: edges 4..7
        for (ei, &(x0, z0)) in yz.iter().enumerate() {
            let hx = hat(x, x0); let hz = hat(z, z0); let b = 4 * p;
            for j in 0..p { let d = b + ei * p + j; values[d * 3 + 1] = lag(&nodes, j, y) * hx * hz; }
        }
        // z-directed: edges 8..11
        for (ei, &(x0, y0)) in yz.iter().enumerate() {
            let hx = hat(x, x0); let hy = hat(y, y0); let b = 8 * p;
            for j in 0..p { let d = b + ei * p + j; values[d * 3 + 2] = lag(&nodes, j, z) * hx * hy; }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let p = self.order; let nodes = self.nodes();
        let x = xi[0]; let y = xi[1]; let z = xi[2];
        curl_vals.fill(0.0);
        let yz = [(-1.0,-1.0),(1.0,-1.0),(1.0,1.0),(-1.0,1.0)];
        // x-directed: curl = (0, L_x·h_y·dh_z/dz, -L_x·dh_y/dy·h_z)
        for (ei, &(y0, z0)) in yz.iter().enumerate() {
            let hy = hat(y, y0); let hz = hat(z, z0);
            let dhy = hat_d(y, y0); let dhz = hat_d(z, z0);
            for j in 0..p { let d = ei * p + j; let lx = lag(&nodes, j, x);
                curl_vals[d * 3 + 1] = lx * hy * dhz; curl_vals[d * 3 + 2] = -lx * dhy * hz; }
        }
        // y-directed: curl = (-L_y·h_x·dh_z/dz, 0, L_y·dh_x/dx·h_z)
        for (ei, &(x0, z0)) in yz.iter().enumerate() {
            let hx = hat(x, x0); let hz = hat(z, z0);
            let dhx = hat_d(x, x0); let dhz = hat_d(z, z0); let b = 4 * p;
            for j in 0..p { let d = b + ei * p + j; let ly = lag(&nodes, j, y);
                curl_vals[d * 3] = -ly * hx * dhz; curl_vals[d * 3 + 2] = ly * dhx * hz; }
        }
        // z-directed: curl = (L_z·dh_x/dx·h_y, -L_z·h_x·dh_y/dy, 0)
        for (ei, &(x0, y0)) in yz.iter().enumerate() {
            let hx = hat(x, x0); let hy = hat(y, y0);
            let dhx = hat_d(x, x0); let dhy = hat_d(y, y0); let b = 8 * p;
            for j in 0..p { let d = b + ei * p + j; let lz = lag(&nodes, j, z);
                curl_vals[d * 3] = lz * dhx * hy; curl_vals[d * 3 + 1] = -lz * hx * dhy; }
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order; let nodes = self.nodes();
        let mut c = Vec::with_capacity(12 * p);
        let yz = [(-1.0,-1.0),(1.0,-1.0),(1.0,1.0),(-1.0,1.0)];
        for &(y0, z0) in &yz { for j in 0..p { c.push(vec![nodes[j], y0, z0]); } }
        for &(x0, z0) in &yz { for j in 0..p { c.push(vec![x0, nodes[j], z0]); } }
        for &(x0, y0) in &yz { for j in 0..p { c.push(vec![x0, y0, nodes[j]]); } }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn n_dofs() {
        assert_eq!(HexNDk::new(1).n_dofs(), 12);
        assert_eq!(HexNDk::new(2).n_dofs(), 24);
        assert_eq!(HexNDk::new(3).n_dofs(), 36);
    }
    #[test] fn finite() {
        for k in 1..=3 { let e = HexNDk::new(k); let n = e.n_dofs();
            let mut v = vec![0.0; n*3]; let mut c = vec![0.0; n*3];
            for p in &[(0.3,-0.5,0.7),(-0.1,0.2,-0.3),(0.0,0.0,0.0)] {
                e.eval_basis_vec(&[p.0,p.1,p.2], &mut v); e.eval_curl(&[p.0,p.1,p.2], &mut c);
                for &val in v.iter().chain(c.iter()) { assert!(val.is_finite()); }
            }
        }
    }
    #[test] fn nodal_interp() {
        let e = HexNDk::new(2); let n = 24; let coords = e.dof_coords();
        let mut v = vec![0.0; n*3];
        for (i, cd) in coords.iter().enumerate() {
            e.eval_basis_vec(&[cd[0], cd[1], cd[2]], &mut v);
            let ei = i / 2;
            let ti = v[i*3 + if ei < 4 { 0 } else if ei < 8 { 1 } else { 2 }];
            assert!(ti.abs() > 0.5, "DOF {i}: self={ti}");
        }
    }
}
