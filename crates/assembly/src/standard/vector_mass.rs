//! Vector mass bilinear form integrator for H(curl) / H(div) spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω α u · v dx
//! ```
//!
//! where `u` and `v` are vector-valued basis functions.

use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

/// Bilinear integrator for the vector mass operator `α u·v`.
///
/// Used in Maxwell cavity eigenvalue problems (`∇×∇×E + E = f`)
/// and as the B-matrix in H(div) mixed formulations.
pub struct VectorMassIntegrator {
    /// Scalar mass coefficient (α).
    pub alpha: f64,
}

impl VectorBilinearIntegrator for VectorMassIntegrator {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let w_a = qp.weight * self.alpha;

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
