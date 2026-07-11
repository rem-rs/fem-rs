//! Tensor (anisotropic) diffusion bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω (σ ∇u) · ∇v dx
//! ```
//!
//! where `σ` is a dim×dim matrix coefficient.  For isotropic diffusion use
//! [`crate::standard::DiffusionIntegrator`] instead.

use crate::integrator::{BilinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, MatrixCoeff};

/// Bilinear integrator for anisotropic diffusion `(σ ∇u)·∇v`.
///
/// `sigma` is a dim×dim matrix coefficient (row-major, length dim²).
/// For isotropic (scalar) diffusion, use [`crate::standard::DiffusionIntegrator`].
pub struct TensorDiffusionIntegrator<M: MatrixCoeff = crate::postproc::coefficient::ConstantMatrixCoeff> {
    pub sigma: M,
}

impl<M: MatrixCoeff> BilinearIntegrator for TensorDiffusionIntegrator<M> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let w = qp.weight;
        let grad = qp.grad_phys;

        // Evaluate sigma matrix at the quadrature point
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );
        let mut s = vec![0.0; dim * dim];
        self.sigma.eval(&ctx, &mut s);

        for i in 0..n {
            for j in 0..n {
                // ∇φ_i · σ · ∇φ_j = Σₐ Σ_b grad[i*dim+a] * sigma[a*dim+b] * grad[j*dim+b]
                let mut val = 0.0;
                for a in 0..dim {
                    let ga = grad[i * dim + a];
                    if ga == 0.0 { continue; }
                    for b in 0..dim {
                        val += ga * s[a * dim + b] * grad[j * dim + b];
                    }
                }
                k_elem[i * n + j] += w * val;
            }
        }
    }
}
