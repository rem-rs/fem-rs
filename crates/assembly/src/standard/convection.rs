//! Convection bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω (b · ∇u) v dx
//! ```
//!
//! where `b` is a vector-valued convection velocity field.

use crate::coefficient::{CoeffCtx, VectorCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the convection operator `(b · ∇u) v`.
///
/// The velocity field `b` is provided as a [`VectorCoeff`].  The resulting
/// matrix is **non-symmetric** (advection biases one direction).
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::ConvectionIntegrator;
/// use fem_assembly::coefficient::ConstantVectorCoeff;
/// // Uniform wind in x-direction
/// let integ = ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]) };
/// ```
pub struct ConvectionIntegrator<V: VectorCoeff> {
    /// Convection velocity field.
    pub velocity: V,
}

impl<V: VectorCoeff> BilinearIntegrator for ConvectionIntegrator<V> {
    /// `K_elem[i,j] += w · φᵢ · (b · ∇φⱼ)`
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );

        // Evaluate velocity at this QP.
        let mut b = [0.0_f64; 3];
        self.velocity.eval(&ctx, &mut b[..d]);

        for i in 0..n {
            let phi_i = qp.phi[i];
            for j in 0..n {
                // b · ∇φⱼ
                let mut b_dot_grad_j = 0.0;
                for k in 0..d {
                    b_dot_grad_j += b[k] * qp.grad_phys[j * d + k];
                }
                k_elem[i * n + j] += qp.weight * phi_i * b_dot_grad_j;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::coefficient::ConstantVectorCoeff;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    /// Convection matrix with uniform b = (1, 0) should be non-symmetric.
    #[test]
    fn convection_is_non_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = ConvectionIntegrator {
            velocity: ConstantVectorCoeff(vec![1.0, 0.0]),
        };
        let mat = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        // Check that at least one (i,j) pair is non-symmetric.
        let mut has_asymmetry = false;
        for i in 0..n {
            for j in 0..n {
                if (dense[i * n + j] - dense[j * n + i]).abs() > 1e-12 {
                    has_asymmetry = true;
                }
            }
        }
        assert!(has_asymmetry, "convection matrix should be non-symmetric");
    }
}
