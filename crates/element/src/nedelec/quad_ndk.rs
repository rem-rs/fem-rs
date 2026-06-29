//! Nedelec-I element on reference quadrilateral [-1,1]².
//! ND_k: dim = 2k(k+1). Edge: 4k DOFs (k per edge). Interior: 2k(k-1) DOFs (k≥2).
//! Uses tensor-product construction: Lagrange × hat for edges, Lagrange × bubble for interior.

use crate::reference::VectorReferenceElement;

fn lag(nodes: &[f64], j: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (i, &ni) in nodes.iter().enumerate() { if i != j { v *= (x - ni) / (nodes[j] - ni); } }
    v
}

fn lag_deriv(nodes: &[f64], j: usize, x: f64) -> f64 {
    let p = nodes.len() - 1;
    let mut s = 0.0;
    for m in 0..=p {
        if m == j { continue; }
        let mut num = 1.0; let mut den = 1.0;
        for i in 0..=p { if i == j || i == m { continue; } num *= x - nodes[i]; den *= nodes[j] - nodes[i]; }
        s += num / (den * (nodes[j] - nodes[m]));
    }
    s
}

fn hat(y: f64, y0: f64) -> f64 { 0.5 * (1.0 + y0 * y) }
fn hat_d(_y: f64, y0: f64) -> f64 { 0.5 * y0 }

pub struct QuadNDk { order: usize }
impl QuadNDk {
    pub fn new(p: usize) -> Self { assert!(p >= 1); QuadNDk { order: p } }
    fn nodes(&self) -> Vec<f64> { let p = self.order; (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect() }
}

impl VectorReferenceElement for QuadNDk {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { 2 * self.order * (self.order + 1) }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order; let n = self.n_dofs(); let nodes = self.nodes();
        let x = xi[0]; let y = xi[1];
        values.fill(0.0);

        // Edge 0 (y=-1): tangent +x
        let hy = hat(y, -1.0);
        for j in 0..p { values[(0 * p + j) * 2] = lag(&nodes, j, x) * hy; }
        // Edge 1 (x=1): tangent +y
        let hx = hat(x, 1.0);
        for j in 0..p { values[(1 * p + j) * 2 + 1] = lag(&nodes, j, y) * hx; }
        // Edge 2 (y=1): tangent -x
        let hy2 = hat(y, 1.0);
        for j in 0..p { values[(2 * p + j) * 2] = -lag(&nodes, j, x) * hy2; }
        // Edge 3 (x=-1): tangent -y
        let hx2 = hat(x, -1.0);
        for j in 0..p { values[(3 * p + j) * 2 + 1] = -lag(&nodes, j, y) * hx2; }

        // Interior x-comp: l_j(x) · (1-y²) · y^i for i=0..p-2, j=0..p-1
        let mut off = 4 * p;
        if p >= 2 {
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32); let ym = 1.0 - y * y;
                for j in 0..p {
                    values[off * 2] = lag(&nodes, j, x) * ym * yi;
                    off += 1;
                }
            }
            // Interior y-comp: (1-x²) · x^i · l_j(y) for i=0..p-2, j=0..p-1
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32); let xm = 1.0 - x * x;
                for j in 0..p {
                    values[off * 2 + 1] = xm * xi * lag(&nodes, j, y);
                    off += 1;
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let p = self.order; let n = self.n_dofs(); let nodes = self.nodes();
        let x = xi[0]; let y = xi[1];
        curl_vals.fill(0.0);

        // Edge curls
        for j in 0..p {
            let lx = lag(&nodes, j, x); let ly = lag(&nodes, j, y);
            let dlx = lag_deriv(&nodes, j, x); let dly = lag_deriv(&nodes, j, y);
            let dhy = hat_d(y, -1.0); let dhx = hat_d(x, 1.0);
            let dhy2 = hat_d(y, 1.0); let dhx2 = hat_d(x, -1.0);
            // e0: Φ=(lx·hat(y,-1), 0) → curl = -lx · dhy
            curl_vals[0 * p + j] = -lx * dhy;
            // e1: Φ=(0, ly·hat(x,1)) → curl = ly · dhx
            curl_vals[1 * p + j] = ly * dhx;
            // e2: Φ=(-lx·hat(y,1), 0) → curl = -(-lx·dhy2) = lx·dhy2
            curl_vals[2 * p + j] = lx * dhy2;
            // e3: Φ=(0, -ly·hat(x,-1)) → curl = -ly·dhx2
            curl_vals[3 * p + j] = -ly * dhx2;
        }

        // Interior curls
        if p >= 2 {
            let mut off = 4 * p;
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32); let ym = 1.0 - y * y;
                let dyi = if i > 0 { i as f64 * y.powi((i - 1) as i32) } else { 0.0 };
                for j in 0..p {
                    let lx = lag(&nodes, j, x);
                    // Φ = (lx·(1-y²)·y^i, 0), curl = -∂Φ_x/∂y = -lx·[-2y·y^i + (1-y²)·i·y^(i-1)]
                    curl_vals[off] = -lx * (-2.0 * y * yi + ym * dyi);
                    off += 1;
                }
            }
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32); let xm = 1.0 - x * x;
                let dxi = if i > 0 { i as f64 * x.powi((i - 1) as i32) } else { 0.0 };
                for j in 0..p {
                    let ly = lag(&nodes, j, y);
                    // Φ = (0, (1-x²)·x^i·ly), curl = ∂Φ_y/∂x = [-2x·x^i + (1-x²)·i·x^(i-1)] · ly
                    curl_vals[off] = (-2.0 * x * xi + xm * dxi) * ly;
                    off += 1;
                }
            }
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) { for v in div_vals.iter_mut() { *v = 0.0; } }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order; let nodes = self.nodes(); let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        for j in 0..p { c.push(vec![nodes[j], -1.0]); }
        for j in 0..p { c.push(vec![1.0, nodes[j]]); }
        for j in 0..p { c.push(vec![nodes[j], 1.0]); }
        for j in 0..p { c.push(vec![-1.0, nodes[j]]); }
        while c.len() < n { c.push(vec![0.0, 0.0]); }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn n_dofs() {
        assert_eq!(QuadNDk::new(1).n_dofs(), 4);
        assert_eq!(QuadNDk::new(2).n_dofs(), 12);
        assert_eq!(QuadNDk::new(3).n_dofs(), 24);
    }
    #[test] fn finite() {
        for k in 1..=4 { let e = QuadNDk::new(k); let n = e.n_dofs();
            let mut v = vec![0.0; n*2]; let mut c = vec![0.0; n];
            for p in &[(0.3,-0.5),(-0.1,0.2),(0.0,0.0)] { e.eval_basis_vec(&[p.0,p.1], &mut v);
                for &val in v.iter() { assert!(val.is_finite()); }
                e.eval_curl(&[p.0,p.1], &mut c);
                for &val in c.iter() { assert!(val.is_finite()); }
            }
        }
    }
    #[test] fn edge_interp() {
        let e = QuadNDk::new(2); let coords = e.dof_coords(); let n = 12; let mut v = vec![0.0; n*2];
        for (i, cd) in coords.iter().enumerate() { if i >= 8 { break; }
            e.eval_basis_vec(&[cd[0], cd[1]], &mut v);
            let ei = i / 2;
            let ti = match ei { 0 | 2 => v[i*2], _ => v[i*2+1] };
            assert!(ti.abs() > 0.5, "DOF {i} edge {ei}: self-tang={ti}");
        }
    }
}
