//! Second diffusion integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

pub struct Diffusion2Integrator<C: ScalarCoeff = f64> {
    pub kappa: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for Diffusion2Integrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.kappa.eval(&ctx);
        // Simplified: use gradient outer product instead of hessian
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim { dot += qp.phi_vec[i * dim + c] * qp.phi_vec[j * dim + c]; }
                k_elem[i * n + j] += w * dot;
            }
        }
    }
}
