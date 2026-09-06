//! Mixed dot product integrator.
use crate::integrator::QpData;
use crate::mixed::MixedBilinearIntegrator;

pub struct MixedDotProductIntegrator;

impl MixedBilinearIntegrator for MixedDotProductIntegrator {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m_elem: &mut [f64]) {
        let n_row = qp_row.n_dofs;
        let n_col = qp_col.n_dofs;
        let w = qp_row.weight;
        for i in 0..n_row {
            for j in 0..n_col {
                m_elem[i * n_col + j] += w * qp_row.phi[i] * qp_col.phi[j];
            }
        }
    }
}
