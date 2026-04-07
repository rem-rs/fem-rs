//! Vector domain source linear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! F(v) = ∫_Ω f · v dx
//! ```
//!
//! where `f` is a vector-valued source and `v` is a vector basis function
//! from H(curl), H(div), or similar vector FE spaces.

use crate::coefficient::{CoeffCtx, VectorCoeff};
use crate::vector_integrator::{VectorLinearIntegrator, VectorQpData};

/// Linear integrator for a vector source term `∫ f · v dx`.
///
/// The source function `f` is provided as a [`VectorCoeff`].
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::VectorDomainLFIntegrator;
/// use fem_assembly::coefficient::ConstantVectorCoeff;
/// let integ = VectorDomainLFIntegrator {
///     f: ConstantVectorCoeff(vec![0.0, -9.81]),
/// };
/// ```
pub struct VectorDomainLFIntegrator<V: VectorCoeff> {
    /// Vector-valued source function.
    pub f: V,
}

impl<V: VectorCoeff> VectorLinearIntegrator for VectorDomainLFIntegrator<V> {
    /// `f_elem[i] += w · Σ_c f_c(x) · φᵢ_c(x)`
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            None, None,
        );

        let mut fval = [0.0_f64; 3];
        self.f.eval(&ctx, &mut fval[..d]);

        for i in 0..n {
            let mut dot = 0.0;
            for c in 0..d {
                dot += fval[c] * qp.phi_vec[i * d + c];
            }
            f_elem[i] += qp.weight * dot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficient::ConstantVectorCoeff;
    use crate::vector_assembler::VectorAssembler;
    use fem_mesh::SimplexMesh;
    use fem_space::HCurlSpace;

    /// ∫ f · v dx with constant f should produce a non-zero load vector.
    #[test]
    fn vector_source_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let integ = VectorDomainLFIntegrator {
            f: ConstantVectorCoeff(vec![1.0, 0.0]),
        };
        let rhs = VectorAssembler::assemble_linear(&space, &[&integ], 4);
        let norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "RHS should be non-zero, got norm = {norm}");
    }
}
