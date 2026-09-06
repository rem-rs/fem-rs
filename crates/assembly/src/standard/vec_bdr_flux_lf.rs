//! Vector boundary flux linear form integrator.
use crate::postproc::coefficient::{CoeffCtx, VectorCoeff};
use crate::vector_integrator::{VectorLinearIntegrator, VectorQpData};

pub struct VectorBoundaryFluxLFIntegrator<V: VectorCoeff> {
    pub f: V,
}

impl<V: VectorCoeff> VectorLinearIntegrator for VectorBoundaryFluxLFIntegrator<V> {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mut f_buf = vec![0.0; dim];
        self.f.eval(&ctx, &mut f_buf);
        for i in 0..n {
            let mut dot = 0.0;
            for c in 0..dim { dot += f_buf[c] * qp.phi_vec[i * dim + c]; }
            f_elem[i] += w * dot;
        }
    }
}
