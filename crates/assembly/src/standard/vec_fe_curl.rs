//! Vector FE curl integrator.
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

pub struct VectorFECurlIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for VectorFECurlIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        let curl_dim = if dim == 2 { 1 } else { 3 };
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..curl_dim { dot += qp.curl[i * curl_dim + c] * qp.curl[j * curl_dim + c]; }
                k_elem[i * n + j] += w * dot;
            }
        }
    }
}
