//! Hyperelastic nonlinear integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

pub struct HyperelasticNLFIntegrator<C: ScalarCoeff = f64> {
    pub mu: C,
    pub lambda: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for HyperelasticNLFIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mu = self.mu.eval(&ctx);
        let lambda = self.lambda.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                let mut strain = 0.0;
                for c in 0..dim { strain += qp.phi_vec[i * dim + c] * qp.phi_vec[j * dim + c]; }
                let trace = if i == j { mu } else { 0.0 };
                k_elem[i * n + j] += w * (mu * strain + lambda * trace);
            }
        }
    }
}
