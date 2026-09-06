//! DG Dirichlet boundary condition linear form integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{LinearIntegrator, QpData};

pub struct DGDirichletLFIntegrator<C: ScalarCoeff = f64> {
    pub g: C,
}

impl<C: ScalarCoeff> LinearIntegrator for DGDirichletLFIntegrator<C> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.g.eval(&ctx);
        for i in 0..n { f_elem[i] += w * qp.phi[i]; }
    }
}
