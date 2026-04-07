//! Boundary flux linear form integrators for vector FE spaces.
//!
//! These integrators compute boundary contributions of the form
//!
//! ```text
//! F(v) = ∫_Γ g(x) (v · n) ds
//! ```
//!
//! where `n` is the outward unit normal and `v` is a vector basis function.
//!
//! For H(div) Raviart-Thomas spaces, the boundary DOFs already represent
//! the normal flux component `v · n`.  This means boundary flux integrals
//! reduce to scalar integrals over face DOFs, and the existing
//! [`BoundaryLinearIntegrator`] / [`NeumannIntegrator`] infrastructure
//! can be reused directly.
//!
//! These types are provided for API completeness and naming consistency
//! with MFEM.

use crate::coefficient::{CoeffCtx, ScalarCoeff};
use crate::integrator::{BdQpData, BoundaryLinearIntegrator};

/// Boundary linear integrator for `∫_Γ g(x) (n · v) ds`.
///
/// Computes the boundary integral where `g` is a scalar coefficient and
/// the test function is dotted with the outward normal.  For scalar H¹
/// spaces, this acts on individual DOFs — the basis function value φᵢ
/// is already scalar, so the "normal dot" is implicit in the problem
/// formulation (e.g. Neumann BC `∂u/∂n = g`).
///
/// For H(div) RT spaces where DOFs represent normal flux, this is
/// equivalent to [`NeumannIntegrator`].
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::BoundaryNormalLFIntegrator;
/// let integ = BoundaryNormalLFIntegrator { g: 1.0 };
/// ```
pub struct BoundaryNormalLFIntegrator<C: ScalarCoeff = f64> {
    /// Scalar boundary source coefficient.
    pub g: C,
}

impl<C: ScalarCoeff> BoundaryLinearIntegrator for BoundaryNormalLFIntegrator<C> {
    /// `f_face[i] += w · g(x) · φᵢ`
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), None,
        );
        let w_g = qp.weight * self.g.eval(&ctx);
        for i in 0..qp.n_dofs {
            f_face[i] += w_g * qp.phi[i];
        }
    }
}

/// Boundary flux integrator `∫_Γ f(x) (v · n) ds` for H(div) RT spaces.
///
/// This is mathematically identical to [`BoundaryNormalLFIntegrator`] —
/// the name matches MFEM's `VectorFEBoundaryFluxLFIntegrator` for API
/// familiarity.
///
/// For RT0 spaces, boundary DOFs already represent the normal flux
/// component, so `v · n = φ_face` on the boundary.
pub type VectorFEBoundaryFluxLFIntegrator<C = f64> = BoundaryNormalLFIntegrator<C>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::{Assembler, face_dofs_p1};
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace};

    /// ∫_Γ 1 · φ ds summed over all DOFs should equal the boundary length (= 4).
    #[test]
    fn boundary_normal_integral() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = BoundaryNormalLFIntegrator { g: 1.0 };
        let tags: Vec<i32> = (1..=4).collect();
        let rhs = Assembler::assemble_boundary_linear(
            space.n_dofs(), space.mesh(), &face_dofs_p1(space.mesh()), 1,
            &[&integ], &tags, 3,
        );
        let total: f64 = rhs.iter().sum();
        assert!((total - 4.0).abs() < 1e-10, "∫_Γ 1 ds = {total}, expected 4.0");
    }
}
