//! Curl-curl bilinear form integrator for H(curl) spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω μ (∇×u) · (∇×v) dx
//! ```
//!
//! In 2-D, `∇×u` is a scalar, so the integrand is `μ · curl_u · curl_v`.
//! In 3-D, `∇×u` is a 3-vector, so the integrand is `μ · (curl_u · curl_v)`.

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

/// Bilinear integrator for the curl-curl operator `μ (∇×u)·(∇×v)`.
///
/// Used for Maxwell equations, eddy-current problems, and electromagnetic
/// cavity eigenvalue problems.
pub struct CurlCurlIntegrator<C: ScalarCoeff = f64> {
    /// Permeability coefficient (μ).  For vacuum, μ = μ₀ ≈ 4π×10⁻⁷.
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
