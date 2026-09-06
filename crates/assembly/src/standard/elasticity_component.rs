//! Elasticity component integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

pub struct ElasticityComponentIntegrator<C: ScalarCoeff = f64> {
    pub mu: C,
    pub lambda: C,
    pub component: usize,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for ElasticityComponentIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mu = self.mu.eval(&ctx);
        let comp = self.component;
        for i in 0..n {
            for j in 0..n {
                let strain = qp.phi_vec[i * dim + comp] * qp.phi_vec[j * dim + comp];
                k_elem[i * n + j] += w * mu * strain;
            }
        }
    }
}
