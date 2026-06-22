//! Arbitrary-order Nedelec-I element on the reference quadrilateral [-1,1]².
//! Edge-based: k DOFs per edge × 4 edges = 4k DOFs.

use crate::reference::VectorReferenceElement;

fn lag(nodes: &[f64], j: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (i, &ni) in nodes.iter().enumerate() { if i != j { v *= (x - ni) / (nodes[j] - ni); } }
    v
}

fn hat(y: f64, y0: f64) -> f64 { 0.5 * (1.0 + y0 * y) }
fn hat_d(y: f64, y0: f64) -> f64 { 0.5 * y0 }

pub struct QuadNDk { order: usize }

impl QuadNDk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); QuadNDk { order: p } }
    fn nodes(&self) -> Vec<f64> {
        let p = self.order;
        (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect()
    }
}

impl VectorReferenceElement for QuadNDk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { 4 * self.order }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order; let nodes = self.nodes();
        let x = xi[0]; let y = xi[1];
        values.fill(0.0);
        // e0: bottom (y=-1), tangent x
        let hy = hat(y, -1.0);
        for j in 0..p { values[(0 * p + j) * 2] = lag(&nodes, j, x) * hy; }
        // e1: right (x=1), tangent y
        let hx = hat(x, 1.0);
        for j in 0..p { values[(1 * p + j) * 2 + 1] = lag(&nodes, j, y) * hx; }
        // e2: top (y=1), tangent -x
        let hy2 = hat(y, 1.0);
        for j in 0..p { values[(2 * p + j) * 2] = -lag(&nodes, j, x) * hy2; }
        // e3: left (x=-1), tangent -y
        let hx2 = hat(x, -1.0);
        for j in 0..p { values[(3 * p + j) * 2 + 1] = -lag(&nodes, j, y) * hx2; }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let p = self.order; let nodes = self.nodes();
        let x = xi[0]; let y = xi[1];
        curl_vals.fill(0.0);
        // e0: Φ=(L·hat(y,-1), 0), curl = -∂Φ_x/∂y = -L·hat_d(y,-1)
        let dhy = hat_d(y, -1.0);
        for j in 0..p { curl_vals[0 * p + j] = -lag(&nodes, j, x) * dhy; }
        // e1: Φ=(0, L·hat(x,1)), curl = ∂Φ_y/∂x = L·hat_d(x,1)
        let dhx = hat_d(x, 1.0);
        for j in 0..p { curl_vals[1 * p + j] = lag(&nodes, j, y) * dhx; }
        // e2: Φ=(-L·hat(y,1), 0), curl = -∂Φ_x/∂y = -( -L·hat_d(y,1)) = L·hat_d(y,1)
        let dhy2 = hat_d(y, 1.0);
        for j in 0..p { curl_vals[2 * p + j] = lag(&nodes, j, x) * dhy2; }
        // e3: Φ=(0, -L·hat(x,-1)), curl = ∂Φ_y/∂x = -L·hat_d(x,-1)
        let dhx2 = hat_d(x, -1.0);
        for j in 0..p { curl_vals[3 * p + j] = -lag(&nodes, j, y) * dhx2; }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() { *v = 0.0; }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order; let nodes = self.nodes();
        let mut c = Vec::with_capacity(4 * p);
        for j in 0..p { c.push(vec![nodes[j], -1.0]); }
        for j in 0..p { c.push(vec![1.0, nodes[j]]); }
        for j in 0..p { c.push(vec![nodes[j], 1.0]); }
        for j in 0..p { c.push(vec![-1.0, nodes[j]]); }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn n_dofs() {
        assert_eq!(QuadNDk::new(1).n_dofs(), 4);
        assert_eq!(QuadNDk::new(2).n_dofs(), 8);
        assert_eq!(QuadNDk::new(3).n_dofs(), 12);
    }
    #[test] fn finite() {
        for k in 1..=3 { let e = QuadNDk::new(k); let n = e.n_dofs();
            let mut v = vec![0.0; n*2]; let mut c = vec![0.0; n];
            for p in &[(0.3,-0.5),(-0.1,0.2),(0.0,0.0)] {
                e.eval_basis_vec(&[p.0,p.1], &mut v); e.eval_curl(&[p.0,p.1], &mut c);
                for &val in v.iter().chain(c.iter()) { assert!(val.is_finite()); }
            }
        }
    }
    #[test] fn nodal_interp() {
        let e = QuadNDk::new(2); let n = 8; let coords = e.dof_coords();
        let mut v = vec![0.0; n*2];
        for (i, cd) in coords.iter().enumerate() {
            e.eval_basis_vec(&[cd[0], cd[1]], &mut v);
            let ei = i / 2;
            let ti = match ei { 0 | 2 => v[i*2], _ => v[i*2+1] };
            assert!(ti.abs() > 0.5, "DOF {i} (edge {ei}): self-tang={ti}");
        }
    }
}
