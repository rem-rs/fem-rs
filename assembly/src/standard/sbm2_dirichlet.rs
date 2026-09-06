//! SBM2 boundary condition integrators.
use crate::integrator::{BilinearIntegrator, LinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

pub struct SBM2DirichletLFIntegrator<C: ScalarCoeff = f64> {
    pub g: C,
}

impl<C: ScalarCoeff> LinearIntegrator for SBM2DirichletLFIntegrator<C> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.g.eval(&ctx);
        for i in 0..n { f_elem[i] += w * qp.phi[i]; }
    }
}

pub struct SBM2DirichletIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for SBM2DirichletIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}
