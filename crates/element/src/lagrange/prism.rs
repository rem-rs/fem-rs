//! Arbitrary-order Lagrange element on the reference triangular prism.
//!
//! Reference domain: (ξ, η, ζ) where (η, ζ) ∈ unit triangle `(0,0),(1,0),(0,1)`
//! and ξ ∈ [0,1] (extrusion direction). Volume = 0.5.
//!
//! The basis is the tensor product of a 1D Lagrange basis on [0,1] in ξ and the
//! triangular Lagrange basis in (η, ζ) (identical to [`TriPk`](super::TriPk)).
//!
//! DOF count: `(p+1) × (p+1)(p+2)/2`.

use crate::lagrange::factory::{rising_val, rising_deriv};
use crate::quadrature::prism_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

fn lagrange_val(n: usize, p: usize, t: f64) -> f64 {
    if p == 0 { return 1.0; }
    let mut val = 1.0;
    let tn = n as f64;
    for m in 0..=p {
        if m != n {
            val *= (t - m as f64) / (tn - m as f64);
        }
    }
    val
}

fn lagrange_deriv(n: usize, p: usize, t: f64) -> f64 {
    if p == 0 { return 0.0; }
    let mut sum = 0.0;
    let tn = n as f64;
    for k in 0..=p {
        if k != n {
            let mut term = 1.0;
            for m in 0..=p {
                if m != n && m != k {
                    term *= (t - m as f64) / (tn - m as f64);
                }
            }
            sum += term / (tn - k as f64);
        }
    }
    sum
}

/// Arbitrary-order Lagrange element on the reference triangular prism.
///
/// DOF ordering: layer-by-layer, where layer `k` (ξ = k/p) contains the full set
/// of triangular DOFs. Within each layer, ordering follows [`TriPk`].
pub struct PrismPk {
    order: usize,
    tri_nodes: Vec<[f64; 2]>,
    tri_ijk: Vec<(usize, usize, usize)>,
    n_tri: usize,
}

impl PrismPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let tri_nodes = equispaced_nodes_tri(p);
        let n_tri = tri_nodes.len();
        let tri_ijk: Vec<(usize, usize, usize)> = tri_nodes.iter()
            .map(|n| {
                let i = (n[0] * p as f64).round() as usize;
                let j = (n[1] * p as f64).round() as usize;
                (i, j, p - i - j)
            })
            .collect();
        Self { order: p, tri_nodes, tri_ijk, n_tri }
    }

    fn dof_index(&self, k: usize, tri_dof: usize) -> usize {
        k * self.n_tri + tri_dof
    }

    fn tri_coords(&self) -> &[[f64; 2]] {
        &self.tri_nodes
    }
}

fn equispaced_nodes_tri(p: usize) -> Vec<[f64; 2]> {
    let mut nodes = Vec::with_capacity((p + 1) * (p + 2) / 2);
    nodes.push([0.0, 0.0]);
    nodes.push([1.0, 0.0]);
    nodes.push([0.0, 1.0]);
    if p == 1 { return nodes; }
    for k in 1..p { nodes.push([k as f64 / p as f64, 0.0]); }
    for k in 1..p { let t = k as f64 / p as f64; nodes.push([1.0 - t, t]); }
    for k in 1..p { nodes.push([0.0, k as f64 / p as f64]); }
    for j in 1..=(p - 2) {
        for i in 1..=(p - 1 - j) {
            nodes.push([i as f64 / p as f64, j as f64 / p as f64]);
        }
    }
    nodes
}

impl ReferenceElement for PrismPk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.order as u8 }
    fn n_dofs(&self) -> usize { (self.order + 1) * self.n_tri }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;

        let t_xi = pf * xi[0];
        let t_eta = pf * xi[1];
        let t_zeta = pf * xi[2];
        let t_rest = pf * (1.0 - xi[1] - xi[2]);

        for k in 0..=p {
            let lx = lagrange_val(k, p, t_xi);
            for (tri_dof, &(i, j, r)) in self.tri_ijk.iter().enumerate() {
                let phi_tri = rising_val(i, t_eta)
                            * rising_val(j, t_zeta)
                            * rising_val(r, t_rest);
                values[self.dof_index(k, tri_dof)] = lx * phi_tri;
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.order;
        let pf = p as f64;

        let t_xi = pf * xi[0];
        let t_eta = pf * xi[1];
        let t_zeta = pf * xi[2];
        let t_rest = pf * (1.0 - xi[1] - xi[2]);

        for k in 0..=p {
            let lx = lagrange_val(k, p, t_xi);
            let dlx = p as f64 * lagrange_deriv(k, p, t_xi);
            for (tri_dof, &(i, j, r)) in self.tri_ijk.iter().enumerate() {
                let vi = rising_val(i, t_eta);
                let vj = rising_val(j, t_zeta);
                let vr = rising_val(r, t_rest);
                let di = rising_deriv(i, t_eta);
                let dj = rising_deriv(j, t_zeta);
                let dr = rising_deriv(r, t_rest);
                let phi_tri = vi * vj * vr;

                let dof = self.dof_index(k, tri_dof);
                grads[dof * 3]     = dlx * phi_tri;
                grads[dof * 3 + 1] = lx * pf * (di * vj * vr - vi * vj * dr);
                grads[dof * 3 + 2] = lx * pf * (vi * dj * vr - vi * vj * dr);
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { prism_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order;
        let mut coords = Vec::with_capacity(self.n_dofs());
        let tri = self.tri_coords();
        for k in 0..=p {
            let xi0 = k as f64 / p as f64;
            for tc in tri.iter() {
                coords.push(vec![xi0, tc[0], tc[1]]);
            }
        }
        coords
    }
}
