//! Vector diffusion integrator for mixed formulations.
use crate::integrator::QpData;
use crate::mixed::MixedBilinearIntegrator;
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

pub struct MixedVectorDiffusionIntegrator<C: ScalarCoeff = f64> {
    pub kappa: C,
}

impl<C: ScalarCoeff> MixedBilinearIntegrator for MixedVectorDiffusionIntegrator<C> {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &dyn std::any::Any, m_elem: &mut [f64]) {
        let qp_col = qp_col.downcast_ref::<QpData<'_>>().unwrap();
        let n_row = qp_row.n_dofs;
        let n_col = qp_col.n_dofs;
        let dim = qp_col.dim;
        let ctx = CoeffCtx::from_qp(qp_row.x_phys, dim, qp_row.elem_id, qp_row.elem_tag, None, None);
        let w = qp_row.weight * self.kappa.eval(&ctx);
        for i in 0..n_row {
            for j in 0..n_col {
                let mut dot = 0.0;
                for c in 0..dim { dot += qp_row.grad_phys[i * dim + c] * qp_col.grad_phys[j * dim + c]; }
                m_elem[i * n_col + j] += w * dot;
            }
        }
    }
}
