//! Vector FE divergence integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

pub struct VectorFEDivergenceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for VectorFEDivergenceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.coeff.eval(&ctx);
        let n_nodes = n / dim;
        for j in 0..n_nodes {
            let vj = qp.phi[j];
            for k in 0..n_nodes {
                let mut div_u = 0.0;
                for c in 0..dim { div_u += qp.grad_phys[k * dim + c]; }
                k_elem[j * n + k] += w * vj * div_u;
            }
        }
    }
}
