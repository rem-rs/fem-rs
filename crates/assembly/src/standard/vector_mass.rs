//! Vector mass bilinear form integrators for H(curl) / H(div) spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω α u · v dx          (isotropic, α scalar)
//! a(u, v) = ∫_Ω (A u) · v dx        (anisotropic, A tensor dim×dim)
//! ```
//!
//! where `u` and `v` are vector-valued basis functions.

use crate::coefficient::{CoeffCtx, MatrixCoeff, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

// ─── Isotropic ───────────────────────────────────────────────────────────────

/// Bilinear integrator for the isotropic vector mass operator `α u·v`.
///
/// `α` is a **scalar** coefficient.  Use [`VectorMassTensorIntegrator`] for
/// anisotropic (tensor-valued) permittivity or other material tensors.
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

// ─── Anisotropic (tensor) ────────────────────────────────────────────────────

/// Bilinear integrator for the anisotropic vector mass operator `(A u)·v`.
///
/// `A` is a **matrix** coefficient (dim×dim, row-major).  This handles:
/// - Anisotropic electric permittivity tensor ε (electromagnetic problems)
/// - Anisotropic mass density tensors (structural dynamics)
/// - General symmetric positive-definite tensor weights
///
/// # Formula
///
/// ```text
/// a(u, v) = ∫_Ω (A(x) u) · v dx   where A is dim×dim at each point
/// ```
///
/// # Example (anisotropic permittivity)
/// ```rust,ignore
/// use fem_assembly::coefficient::ConstantMatrixCoeff;
/// use fem_assembly::standard::VectorMassTensorIntegrator;
/// // ε_r = diag(2, 1) — uniaxial medium
/// let integ = VectorMassTensorIntegrator {
///     alpha: ConstantMatrixCoeff(vec![2.0, 0.0, 0.0, 1.0]),
/// };
/// ```
pub struct VectorMassTensorIntegrator<C: MatrixCoeff> {
    /// Matrix mass coefficient (dim×dim, row-major).
    pub alpha: C,
}

impl<C: MatrixCoeff> VectorBilinearIntegrator for VectorMassTensorIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n   = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, dim, qp.elem_id, qp.elem_tag,
            None, None,
        );
        let w = qp.weight;

        // Evaluate tensor A into a local stack buffer (max 3×3 = 9).
        let mut a_buf = [0.0_f64; 9];
        self.alpha.eval(&ctx, &mut a_buf[..dim * dim]);

        for i in 0..n {
            // A φᵢ: matrix-vector product, result in `au` (length dim).
            let mut au = [0.0_f64; 3];
            for r in 0..dim {
                for c in 0..dim {
                    au[r] += a_buf[r * dim + c] * qp.phi_vec[i * dim + c];
                }
            }
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += au[c] * qp.phi_vec[j * dim + c];
                }
                k_elem[i * n + j] += w * dot;
            }
        }
    }
}
