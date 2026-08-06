//! BDM_k reference element on the reference quadrilateral [-1,1]².
//!
//! Pure tensor-product construction.
//! DOFs: 4(k+1) edge normal moments + k(k-1) interior L² moments (k ≥ 2).
//! Total dim = (k+1)(k+2).

use crate::VectorReferenceElement;

struct QuadBDMkData {
    k: usize,
    n: usize,
    #[allow(dead_code)]
    n_interior: usize,
}

static CACHE: [std::sync::OnceLock<QuadBDMkData>; 9] = [
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
];

fn quad_data(k: usize) -> &'static QuadBDMkData {
    CACHE[k].get_or_init(|| {
        let n_edge = 4 * (k + 1);
        let n_int = if k >= 2 { k * (k - 1) } else { 0 };
        QuadBDMkData {
            k,
            n: n_edge + n_int,
            n_interior: n_int,
        }
    })
}

pub struct QuadBDMk {
    order: usize,
}

impl QuadBDMk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        QuadBDMk { order: p }
    }
    fn nodes(&self) -> Vec<f64> {
        let (p, _) = crate::quadrature::gauss_lobatto_arbitrary(self.order + 2);
        p[..self.order + 1].to_vec()
    }
    fn lagrange(&self, i: usize, t: f64) -> f64 {
        let nodes = self.nodes();
        let k = self.order;
        let mut num = 1.0;
        let mut den = 1.0;
        for j in 0..=k {
            if j == i {
                continue;
            }
            num *= t - nodes[j];
            den *= nodes[i] - nodes[j];
        }
        num / den
    }
    fn dlagrange(&self, i: usize, t: f64) -> f64 {
        let nodes = self.nodes();
        let k = self.order;
        let mut sum = 0.0;
        for j in 0..=k {
            if j == i {
                continue;
            }
            let mut num = 1.0;
            let mut den = 1.0;
            for jj in 0..=k {
                if jj == i || jj == j {
                    continue;
                }
                num *= t - nodes[jj];
                den *= nodes[i] - nodes[jj];
            }
            sum += num / den;
        }
        sum
    }
}

impl VectorReferenceElement for QuadBDMk {
    fn n_dofs(&self) -> usize {
        quad_data(self.order).n
    }
    fn dim(&self) -> u8 {
        2u8
    }
    fn order(&self) -> u8 {
        self.order as u8
    }

    fn quadrature(&self, order: u8) -> crate::QuadratureRule {
        let o = order.max(self.order as u8 + 1) as usize;
        let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(o);
        let nq = x1d.len();
        let mut pts = Vec::with_capacity(nq * nq);
        let mut wts = Vec::with_capacity(nq * nq);
        for i in 0..nq {
            for j in 0..nq {
                pts.push(vec![x1d[i], x1d[j]]);
                wts.push(w1d[i] * w1d[j]);
            }
        }
        crate::QuadratureRule {
            points: pts,
            weights: wts,
        }
    }

    fn eval_basis_vec(&self, xi: &[f64], vals: &mut [f64]) {
        let d = quad_data(self.order);
        let k = d.k;
        let n = d.n;
        let (x, y) = (xi[0], xi[1]);
        let mut idx = 0;

        // Edge 0: y = -1, normal (0, -1) → comp (0, -L_i(x))
        for i in 0..=k {
            if idx < n {
                vals[idx * 2] = 0.0;
                vals[idx * 2 + 1] = -self.lagrange(i, x);
                idx += 1;
            }
        }
        // Edge 1: x = 1, normal (1, 0) → comp (L_i(y), 0)
        for i in 0..=k {
            if idx < n {
                vals[idx * 2] = self.lagrange(i, y);
                vals[idx * 2 + 1] = 0.0;
                idx += 1;
            }
        }
        // Edge 2: y = 1, normal (0, 1) → comp (0, L_i(x))
        for i in 0..=k {
            if idx < n {
                vals[idx * 2] = 0.0;
                vals[idx * 2 + 1] = self.lagrange(i, x);
                idx += 1;
            }
        }
        // Edge 3: x = -1, normal (-1, 0) → comp (-L_i(y), 0)
        for i in 0..=k {
            if idx < n {
                vals[idx * 2] = -self.lagrange(i, y);
                vals[idx * 2 + 1] = 0.0;
                idx += 1;
            }
        }

        // Interior: (1-x²)(1-y²)·x^p·y^q for both components
        if k >= 2 {
            for p in 0..k - 1 {
                for q in 0..k - 1 {
                    if idx < n {
                        let b = (1.0 - x * x) * (1.0 - y * y);
                        vals[idx * 2] = b * x.powi(p as i32) * y.powi(q as i32);
                        vals[idx * 2 + 1] = b * x.powi(p as i32) * y.powi(q as i32);
                        idx += 1;
                    }
                }
            }
        }
        while idx < n {
            vals[idx * 2] = 0.0;
            vals[idx * 2 + 1] = 0.0;
            idx += 1;
        }
    }

    fn eval_div(&self, xi: &[f64], div: &mut [f64]) {
        let d = quad_data(self.order);
        let k = d.k;
        let n = d.n;
        let (x, y) = (xi[0], xi[1]);
        let mut idx = 0;
        for _ in 0..4 * (k + 1) {
            if idx < n {
                div[idx] = 0.0;
                idx += 1;
            }
        }
        if k >= 2 {
            for p in 0..k - 1 {
                for q in 0..k - 1 {
                    if idx < n {
                        let b = (1.0 - x * x) * (1.0 - y * y);
                        let dx_b = -2.0 * x * (1.0 - y * y);
                        let dy_b = -2.0 * y * (1.0 - x * x);
                        let mx = x.powi(p as i32);
                        let my = y.powi(q as i32);
                        let dmx = if p > 0 {
                            p as f64 * x.powi(p as i32 - 1)
                        } else {
                            0.0
                        };
                        let dmy = if q > 0 {
                            q as f64 * y.powi(q as i32 - 1)
                        } else {
                            0.0
                        };
                        div[idx] =
                            (dx_b * mx * my + b * dmx * my) + (dy_b * mx * my + b * mx * dmy);
                        idx += 1;
                    }
                }
            }
        }
        while idx < n {
            div[idx] = 0.0;
            idx += 1;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl: &mut [f64]) {
        // 2D scalar curl: ∂Φ_y/∂x - ∂Φ_x/∂y
        let d = quad_data(self.order);
        let k = d.k;
        let n = d.n;
        let (x, y) = (xi[0], xi[1]);
        let mut idx = 0;
        // Edge 0: comp=(0,-L_i(x)), curl=∂(-L_i)/∂x - ∂(0)/∂y = -L_i'(x)
        for i in 0..=k {
            if idx < n {
                curl[idx] = -self.dlagrange(i, x);
                idx += 1;
            }
        }
        // Edge 1: comp=(L_i(y),0), curl=∂(0)/∂x - ∂(L_i(y))/∂y = -L_i'(y)
        for i in 0..=k {
            if idx < n {
                curl[idx] = -self.dlagrange(i, y);
                idx += 1;
            }
        }
        // Edge 2: comp=(0,L_i(x)), curl=∂(L_i(x))/∂x - ∂(0)/∂y = L_i'(x)
        for i in 0..=k {
            if idx < n {
                curl[idx] = self.dlagrange(i, x);
                idx += 1;
            }
        }
        // Edge 3: comp=(-L_i(y),0), curl=∂(0)/∂x - ∂(-L_i(y))/∂y = L_i'(y)
        for i in 0..=k {
            if idx < n {
                curl[idx] = self.dlagrange(i, y);
                idx += 1;
            }
        }
        // Interior: comp=(b·x^p·y^q, b·x^p·y^q), curl=∂(b·x^p·y^q)/∂x - ∂(b·x^p·y^q)/∂y
        // = (dx_b·x^p·y^q + b·p·x^{p-1}·y^q) - (dy_b·x^p·y^q + b·x^p·q·y^{q-1})
        if k >= 2 {
            for p in 0..k - 1 {
                for q in 0..k - 1 {
                    if idx < n {
                        let b = (1.0 - x * x) * (1.0 - y * y);
                        let dx_b = -2.0 * x * (1.0 - y * y);
                        let dy_b = -2.0 * y * (1.0 - x * x);
                        let mx = x.powi(p as i32);
                        let my = y.powi(q as i32);
                        let dmx = if p > 0 {
                            p as f64 * x.powi(p as i32 - 1)
                        } else {
                            0.0
                        };
                        let dmy = if q > 0 {
                            q as f64 * y.powi(q as i32 - 1)
                        } else {
                            0.0
                        };
                        curl[idx] =
                            (dx_b * mx * my + b * dmx * my) - (dy_b * mx * my + b * mx * dmy);
                        idx += 1;
                    }
                }
            }
        }
        while idx < n {
            curl[idx] = 0.0;
            idx += 1;
        }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let d = quad_data(self.order);
        let k = d.k;
        let n = d.n;
        let nodes = if k >= 1 {
            let (p, _) = crate::quadrature::gauss_lobatto_arbitrary(k + 2);
            p[..=k].to_vec()
        } else {
            vec![]
        };
        let mut coords = Vec::with_capacity(n);
        // Edge 0: y=-1
        for i in 0..=k {
            coords.push(vec![nodes[i], -1.0]);
        }
        // Edge 1: x=1
        for i in 0..=k {
            coords.push(vec![1.0, nodes[i]]);
        }
        // Edge 2: y=1
        for i in 0..=k {
            coords.push(vec![nodes[i], 1.0]);
        }
        // Edge 3: x=-1
        for i in 0..=k {
            coords.push(vec![-1.0, nodes[i]]);
        }
        // Interior: bubble centers
        for p in 0..k - 1 {
            for q in 0..k - 1 {
                coords.push(vec![
                    2.0 * p as f64 / (k - 1) as f64 - 1.0,
                    2.0 * q as f64 / (k - 1) as f64 - 1.0,
                ]);
            }
        }
        while coords.len() < n {
            coords.push(vec![0.0, 0.0]);
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_bdmk_dof_counts() {
        assert_eq!(QuadBDMk::new(1).n_dofs(), 8);
        assert_eq!(QuadBDMk::new(2).n_dofs(), 14);
        assert_eq!(QuadBDMk::new(3).n_dofs(), 22);
    }

    #[test]
    fn quad_bdmk_finite() {
        for k in 1..=3 {
            let e = QuadBDMk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 2];
            let mut d = vec![0.0; n];
            e.eval_basis_vec(&[0.3, -0.2], &mut v);
            e.eval_div(&[0.3, -0.2], &mut d);
            for val in &v {
                assert!(val.is_finite(), "k={k}");
            }
            for val in &d {
                assert!(val.is_finite(), "k={k} div");
            }
        }
    }
}
