//! Vector boundary linear form integrator.
use crate::postproc::coefficient::CoeffCtx;
use crate::postproc::coefficient::VectorCoeff;
use crate::integrator::LinearIntegrator;
use crate::integrator::QpData;

pub struct VectorBoundaryLFIntegrator<V: VectorCoeff> {
    pub f: V,
}

impl<V: VectorCoeff> LinearIntegrator for VectorBoundaryLFIntegrator<V> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mut f_buf = vec![0.0; qp.dim];
        self.f.eval(&ctx, &mut f_buf);
        for i in 0..n {
            let mut dot = 0.0;
            for c in 0..qp.dim { dot += f_buf[c] * qp.phi[i]; }
            f_elem[i] += w * dot;
        }
    }
}
