//! Mixed scalar weak gradient integrator.
use crate::integrator::QpData;
use crate::mixed::MixedBilinearIntegrator;

pub struct MixedScalarWeakGradientIntegrator;

impl MixedBilinearIntegrator for MixedScalarWeakGradientIntegrator {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m_elem: &mut [f64]) {
        let n_p = qp_row.n_dofs;
        let n_u = qp_col.n_dofs;
        let dim = qp_col.dim;
        let w = qp_col.weight;
        for j in 0..n_p {
            let pj = qp_row.phi[j];
            for k in 0..n_u {
                let mut div_u = 0.0;
                for c in 0..dim { div_u += qp_col.grad_phys[k * dim + c]; }
                m_elem[j * n_u + k] += -w * pj * div_u;
            }
        }
    }
}
