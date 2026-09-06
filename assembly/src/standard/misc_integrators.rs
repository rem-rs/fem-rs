//! Miscellaneous missing integrators.
use crate::integrator::{BilinearIntegrator, LinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff, VectorCoeff};

pub struct VectorDivergenceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for VectorDivergenceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        let n_nodes = n / dim;
        for i in 0..n_nodes {
            let mut div_i = 0.0;
            for c in 0..dim { div_i += qp.grad_phys[i * dim + c]; }
            for j in 0..n_nodes {
                let mut div_j = 0.0;
                for c in 0..dim { div_j += qp.grad_phys[j * dim + c]; }
                k_elem[i * n + j] += w * div_i * div_j;
            }
        }
    }
}

pub struct VectorConvectionNLFIntegrator<V: VectorCoeff> {
    pub velocity: V,
}

impl<V: VectorCoeff> BilinearIntegrator for VectorConvectionNLFIntegrator<V> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mut v_buf = vec![0.0; dim];
        self.velocity.eval(&ctx, &mut v_buf);
        for i in 0..n {
            for j in 0..n {
                let mut conv = 0.0;
                for c in 0..dim { conv += v_buf[c] * qp.grad_phys[j * dim + c]; }
                k_elem[i * n + j] += w * qp.phi[i] * conv;
            }
        }
    }
}

pub struct WhiteGaussianNoiseDomainLFIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> LinearIntegrator for WhiteGaussianNoiseDomainLFIntegrator<C> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, Some(qp.phi), qp.elem_dofs);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n { f_elem[i] += w * qp.phi[i]; }
    }
}

pub struct NormalTraceJumpIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for NormalTraceJumpIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

pub struct NonconservativeDGTraceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for NonconservativeDGTraceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

pub struct MixedWeakGradDotIntegrator;

impl BilinearIntegrator for MixedWeakGradDotIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let w = qp.weight;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim { dot += qp.grad_phys[i * dim + c] * qp.phi[j]; }
                k_elem[i * n + j] += w * dot;
            }
        }
    }
}
