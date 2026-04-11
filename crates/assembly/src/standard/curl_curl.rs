//! Curl-curl bilinear form integrators for H(curl) spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω μ (∇×u) · (∇×v) dx          (isotropic,  μ scalar)
//! a(u, v) = ∫_Ω (M ∇×u) · (∇×v) dx           (anisotropic, M tensor)
//! ```
//!
//! In 2-D, `∇×u` is a scalar (no tensor action needed for 2-D).
//! In 3-D, `∇×u` is a 3-vector; the tensor integrator computes `(M c_u)·c_v`
//! where `c_u = curl u` and `M` is the 3×3 permeability inverse tensor.

use crate::coefficient::{CoeffCtx, MatrixCoeff, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

// ─── Isotropic ───────────────────────────────────────────────────────────────

/// Bilinear integrator for the curl-curl operator `μ (∇×u)·(∇×v)`.
///
/// `μ` is a **scalar** coefficient (isotropic permeability).  For vacuum,
/// `μ = μ₀ ≈ 4π×10⁻⁷`.  Use [`CurlCurlTensorIntegrator`] for anisotropic
/// (tensor-valued) permeability.
///
/// Used for Maxwell equations, eddy-current problems, and electromagnetic
/// cavity eigenvalue problems.
pub struct CurlCurlIntegrator<C: ScalarCoeff = f64> {
    /// Scalar permeability coefficient (μ).
    pub mu: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for CurlCurlIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            None, None,
        );
        let w_mu = qp.weight * self.mu.eval(&ctx);

        if qp.dim == 2 {
            // Scalar curl: curl[i] is a single f64.
            for i in 0..n {
                for j in 0..n {
                    k_elem[i * n + j] += w_mu * qp.curl[i] * qp.curl[j];
                }
            }
        } else {
            // 3-D vector curl: curl[i*3 + c] is component c of curl of basis i.
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for c in 0..3 {
                        dot += qp.curl[i * 3 + c] * qp.curl[j * 3 + c];
                    }
                    k_elem[i * n + j] += w_mu * dot;
                }
            }
        }
    }
}

// ─── Anisotropic (tensor) ────────────────────────────────────────────────────

/// Bilinear integrator for the anisotropic curl-curl operator `(M ∇×u)·(∇×v)`.
///
/// `M` is a **matrix** coefficient (dim×dim, row-major).  This handles:
/// - Anisotropic magnetic permeability tensors (iron cores, ferrites)
/// - Anisotropic reluctivity ν = μ⁻¹ for eddy-current problems
/// - General tensor-valued material parameters
///
/// # 2-D note
///
/// In 2-D, `∇×u` is a scalar, so the tensor reduces to a scalar factor:
/// `a(u,v) = ∫ M₀₀ curl_u curl_v dx`.  Only `M[0]` (the (0,0) entry) is used.
///
/// # 3-D formula
///
/// ```text
/// a(u,v) = ∫_Ω (M c_u)·c_v dx   where c_u = curl u ∈ ℝ³
/// ```
///
/// # Example (isotropic via tensor)
/// ```rust,ignore
/// use fem_assembly::coefficient::ScalarMatrixCoeff;
/// use fem_assembly::standard::CurlCurlTensorIntegrator;
/// // Equivalent to CurlCurlIntegrator { mu: 1.0 }
/// let integ = CurlCurlTensorIntegrator { mu: ScalarMatrixCoeff(1.0_f64) };
/// ```
///
/// # Example (anisotropic iron core)
/// ```rust,ignore
/// use fem_assembly::coefficient::{ConstantMatrixCoeff, PWConstCoeff, FnMatrixCoeff};
/// use fem_assembly::standard::CurlCurlTensorIntegrator;
/// // μ_r = 1000 in x, 500 in y (anisotropic ferrite)
/// let mu_tensor = ConstantMatrixCoeff(vec![1000.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 1.0]);
/// let integ = CurlCurlTensorIntegrator { mu: mu_tensor };
/// ```
pub struct CurlCurlTensorIntegrator<C: MatrixCoeff> {
    /// Matrix permeability coefficient (dim×dim, row-major).
    pub mu: C,
}

impl<C: MatrixCoeff> VectorBilinearIntegrator for CurlCurlTensorIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n   = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, dim, qp.elem_id, qp.elem_tag,
            None, None,
        );

        if dim == 2 {
            // Scalar curl in 2-D: only M[0,0] contributes.
            let mut m_buf = [0.0_f64; 4];
            self.mu.eval(&ctx, &mut m_buf);
            let w_mu00 = qp.weight * m_buf[0];
            for i in 0..n {
                for j in 0..n {
                    k_elem[i * n + j] += w_mu00 * qp.curl[i] * qp.curl[j];
                }
            }
        } else {
            // 3-D vector curl: k_ij += w * (M c_i) · c_j
            let mut m_buf = [0.0_f64; 9];
            self.mu.eval(&ctx, &mut m_buf);
            let w = qp.weight;

            for i in 0..n {
                // c_i = curl of basis i, length 3
                // M c_i: matrix-vector product
                let mut mc = [0.0_f64; 3];
                for r in 0..3 {
                    for c in 0..3 {
                        mc[r] += m_buf[r * 3 + c] * qp.curl[i * 3 + c];
                    }
                }
                for j in 0..n {
                    let mut dot = 0.0;
                    for c in 0..3 {
                        dot += mc[c] * qp.curl[j * 3 + c];
                    }
                    k_elem[i * n + j] += w * dot;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_assembler::VectorAssembler;
    use crate::standard::VectorMassIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_space::HCurlSpace;

    #[test]
    fn curl_curl_is_symmetric() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let integ = CurlCurlIntegrator { mu: 1.0 };
        let mat = VectorAssembler::assemble_bilinear(&space, &[&integ], 4);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
    }

    #[test]
    fn curl_curl_positive_semi_definite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let integ = CurlCurlIntegrator { mu: 1.0 };
        let mat = VectorAssembler::assemble_bilinear(&space, &[&integ], 4);
        for i in 0..mat.nrows {
            let diag = mat.get(i, i);
            assert!(diag >= -1e-14, "diagonal K[{i},{i}] = {diag} is negative");
        }
    }

    #[test]
    fn vector_mass_is_symmetric() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let integ = VectorMassIntegrator { alpha: 1.0 };
        let mat = VectorAssembler::assemble_bilinear(&space, &[&integ], 4);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12, "M[{i},{j}] - M[{j},{i}] = {diff}");
            }
        }
    }

    #[test]
    fn curl_curl_plus_mass_is_spd() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let cc = CurlCurlIntegrator { mu: 1.0 };
        let vm = VectorMassIntegrator { alpha: 1.0 };
        let mat = VectorAssembler::assemble_bilinear(&space, &[&cc, &vm], 4);
        for i in 0..mat.nrows {
            let diag = mat.get(i, i);
            assert!(diag > 1e-14, "diagonal (K+M)[{i},{i}] = {diag} should be positive");
        }
    }
}
