//! Vector mass bilinear form integrator for H(curl) / H(div) spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω α u · v dx
//! ```
//!
//! where `u` and `v` are vector-valued basis functions.

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

/// Bilinear integrator for the vector mass operator `α u·v`.
///
/// Used in Maxwell cavity eigenvalue problems (`∇×∇×E + E = f`)
/// and as the B-matrix in H(div) mixed formulations.
pub struct VectorMassIntegrator<C: ScalarCoeff = f64> {
    /// Scalar mass coefficient (α).
    pub alpha: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for VectorMassIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            None, None,
        );
        let w_a = qp.weight * self.alpha.eval(&ctx);

        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..d {
                    dot += qp.phi_vec[i * d + c] * qp.phi_vec[j * d + c];
                }
                k_elem[i * n + j] += w_a * dot;
            }
        }
    }
}
