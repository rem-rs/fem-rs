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

use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff, VectorCoeff};
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
/// # Naming warning
///
/// This type is **not** MFEM's `BoundaryNormalLFIntegrator`: MFEM's class of
/// that name takes a *vector* coefficient and computes `∫_Γ (v(x)·n) φᵢ ds`,
/// which is [`VectorBoundaryNormalLFIntegrator`] here.  This scalar variant
/// corresponds to MFEM's `BoundaryLFIntegrator` (`∫_Γ g(x) φᵢ ds`).  The
/// semantics are kept as-is because existing consumers depend on them.
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

/// Boundary linear integrator for `∫_Γ (v(x)·n) φᵢ ds` — MFEM's
/// `BoundaryNormalLFIntegrator(VectorCoefficient &)`.
///
/// `v` is an arbitrary [`VectorCoeff`] evaluated at the physical coordinates
/// of the quadrature point and `n` is the **unit outward normal** of the
/// boundary face.  `qp.weight` already carries MFEM's quadrature measure
/// (`ip.weight × |J_face|`), so the accumulated element vector is
/// `Σ_q ip.weight_q (v(x_q)·n_q) φᵢ(x_q)`, i.e. MFEM's
/// `elvect.Add(ip.weight*(Qvec*nor), shape)` with `nor = CalcOrtho(Tr.Jacobian())`
/// (the *unnormalised* face normal — the unit normal times `|J_face|`).
///
/// # Why a second type is needed
///
/// fem-rs's [`BoundaryNormalLFIntegrator`] computes the *scalar*
/// `∫_Γ g(x) φᵢ ds` (a `ScalarCoeff` field), which is MFEM's
/// `BoundaryLFIntegrator`.  Changing that type's semantics would silently
/// break its consumers, so the MFEM-compatible *vector* functional lives in
/// this separate type with an explicit name.
///
/// # Consumers
///
/// The Navier–Stokes miniapps (`miniapps/fluids/navier_*`) use it for the
/// pressure-space functionals `FText_bdr = ∫_Γ (FText·n) q ds` and
/// `g_bdr = ∫_Γ (u_D·n) q ds` of the split scheme
/// (`navier_solver.cpp`: `BoundaryNormalLFIntegrator(*FText_gfcoeff)` and
/// `BoundaryNormalLFIntegrator(*vel_dbc.coeff)`).  `assemble_boundary_linear`
/// must be given a face reference element of the matching order (see
/// `ref_elem_face`, which serves every order through
/// `assembler::face_dofs_h1`) and MFEM's default rule
/// `IntRules.Get(SEGMENT, 1*order + 1)` as `quad_order`.
pub struct VectorBoundaryNormalLFIntegrator<C: VectorCoeff> {
    /// The vector coefficient `v(x)`.
    pub v: C,
}

impl<C: VectorCoeff> BoundaryLinearIntegrator for VectorBoundaryNormalLFIntegrator<C> {
    /// `f_face[i] += w · (v(x)·n) · φᵢ`
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), None,
        );
        // At most 3 components; `qp.dim` of the face is the mesh dimension.
        let mut v = [0.0_f64; 3];
        self.v.eval(&ctx, &mut v[..qp.dim]);
        let vn: f64 = v[..qp.dim]
            .iter()
            .zip(qp.normal.iter())
            .map(|(a, b)| a * b)
            .sum();
        let w = qp.weight * vn;
        for i in 0..qp.n_dofs {
            f_face[i] += w * qp.phi[i];
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
    use fem_mesh::Mesh;
    use fem_space::{H1Space, fe_space::FESpace};

    /// ∫_Γ 1 · φ ds summed over all DOFs should equal the boundary length (= 4).
    #[test]
    fn boundary_normal_integral() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
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
