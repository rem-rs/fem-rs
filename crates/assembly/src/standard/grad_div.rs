//! Grad-div bilinear form integrator for vector finite element spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω κ (∇·u)(∇·v) dx
//! ```
//!
//! This is the grad-div stabilization term, commonly used in Stokes/Navier-Stokes
//! to improve mass conservation, and in Maxwell problems.

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

/// Bilinear integrator for the grad-div operator `κ (∇·u)(∇·v)`.
///
/// Uses [`VectorQpData::div`] which provides the divergence of each
/// Piola-transformed basis function.
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::GradDivIntegrator;
/// let integ = GradDivIntegrator { kappa: 1.0 };
/// ```
pub struct GradDivIntegrator<C: ScalarCoeff = f64> {
    /// Scalar coefficient.
    pub kappa: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for GradDivIntegrator<C> {
    /// `K[i,j] += w · κ · div(φᵢ) · div(φⱼ)`
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            None, None,
        );
        let w_k = qp.weight * self.kappa.eval(&ctx);

        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w_k * qp.div[i] * qp.div[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_assembler::VectorAssembler;
    use fem_mesh::SimplexMesh;
    use fem_space::HDivSpace;

    #[test]
    fn grad_div_is_symmetric() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let integ = GradDivIntegrator { kappa: 1.0 };
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
    fn grad_div_positive_semi_definite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let integ = GradDivIntegrator { kappa: 1.0 };
        let mat = VectorAssembler::assemble_bilinear(&space, &[&integ], 4);
        for i in 0..mat.nrows {
            let diag = mat.get(i, i);
            assert!(diag >= -1e-14, "diagonal K[{i},{i}] = {diag} is negative");
        }
    }
}
