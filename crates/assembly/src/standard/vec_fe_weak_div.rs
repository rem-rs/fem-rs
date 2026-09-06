//! Vector FE weak divergence integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

pub struct VectorFEWeakDivergenceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for VectorFEWeakDivergenceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n {
            let div_i = qp.div[i];
            for j in 0..n {
                let div_j = qp.div[j];
                k_elem[i * n + j] += -w * div_i * div_j;
            }
        }
    }
}
