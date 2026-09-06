//! Mixed scalar mass integrator.
use crate::integrator::QpData;
use crate::mixed::MixedBilinearIntegrator;

pub struct MixedScalarMassIntegrator;

impl MixedBilinearIntegrator for MixedScalarMassIntegrator {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m_elem: &mut [f64]) {
        let n_r = qp_row.n_dofs;
        let n_c = qp_col.n_dofs;
        let w = qp_row.weight;
        for j in 0..n_r {
            let pj = qp_row.phi[j];
            for i in 0..n_c {
                m_elem[j * n_c + i] += w * pj * qp_col.phi[i];
            }
        }
    }
}
