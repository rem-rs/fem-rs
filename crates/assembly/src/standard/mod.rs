//! Standard finite element integrators.
//!
//! Re-exports the most commonly used integrators for convenience.
//!
//! # Which quadrature weight?
//!
//! `QpData` carries three weights and they are **not** interchangeable.  Pick
//! by the shape of the integrand, not by the kind of element or space:
//!
//! | integrand | field | why |
//! |---|---|---|
//! | `∇φ·∇ψ` (times a coefficient), read through the assembler-provided `grad_phys` | `QpData::weight` | on the volume path `grad_phys = adj(J)ᵀ ∇φ`, so `weight × grad_phys` is already the physical measure — MFEM's `DiffusionIntegrator` (`w = ip.weight / Weight()`, `dshapedxt = dshape·AdjugateJacobian`) |
//! | `φ·φ`, `φ·coeff`, `ρ u v`, `curl·curl` on a physical basis — any physical volume form built from *values* | `QpData::phys_weight` | always `ip.weight · |det J|`; using `weight` here is off by `1/|det J|²` on curved/stretched elements (measured 9× on `det J = 1/3`) |
//! | `ip.weight` times a Jacobian factor the integrator supplies itself (e.g. `ConvectionIntegrator`, which scales the adjugate gradient) | `QpData::ref_weight` | the bare MFEM `ip.weight`, for MFEM's "measure implicit in the gradient" family |
//! | anything on a face (`BoundaryBilinearIntegrator` / `BoundaryLinearIntegrator`) | `BdQpData::weight` | single-weight layout, always the physical face measure |
//!
//! Rationale, the per-assembler-path table and the counter-example are in
//! `crate::integrator` under `QpData::weight`; the two helper macros below
//! (`scalar_bilinear_integrator!` → `qp.weight`,
//! `scalar_bilinear_integrator_phys!` → `qp.phys_weight`) encode the first two
//! rows of the table.

/// Helper macro for scalar bilinear integrators with a [`ScalarCoeff`] field.
///
/// Generates the struct definition, necessary imports, and the
/// [`BilinearIntegrator`] trait impl with the standard
/// `CoeffCtx::from_qp(x_phys, dim, elem_id, elem_tag, Some(phi), elem_dofs)`
/// preamble.
///
/// The last argument is a block with the four bindings available:
/// `|qp, k_elem, n, w|` where:
/// * `qp` — [`QpData`] reference
/// * `k_elem` — element matrix slice
/// * `n` — number of DOFs (`= qp.n_dofs`)
/// * `w` — weighted coefficient (`= qp.weight × coeff.eval(&ctx)`)
macro_rules! scalar_bilinear_integrator {
    ($name:ident, $field:ident, $doc:literal, |$qp:ident, $kelem:ident, $n:ident, $w:ident| $body:block) => {
        use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
        use crate::integrator::{BilinearIntegrator, QpData};

        #[doc = $doc]
        pub struct $name<C: ScalarCoeff = f64> {
            pub $field: C,
        }

        impl<C: ScalarCoeff> BilinearIntegrator for $name<C> {
            fn add_to_element_matrix(&self, $qp: &QpData<'_>, $kelem: &mut [f64]) {
                let $n = $qp.n_dofs;
                let ctx = CoeffCtx::from_qp(
                    $qp.x_phys, $qp.dim, $qp.elem_id, $qp.elem_tag,
                    Some($qp.phi), $qp.elem_dofs,
                );
                let $w = $qp.weight * self.$field.eval(&ctx);
                $body
            }
        }
    };
}

/// Like [`scalar_bilinear_integrator!`] but uses `qp.phys_weight` (the true
/// physical measure `quadrature weight × |det J|`) instead of `qp.weight`.
///
/// `qp.weight` is `ip.weight / |det J|` on the volume (linear *and*
/// isoparametric) assembler path — the DiffusionIntegrator convention, correct
/// when it multiplies the adjugate-scaled `grad_phys` but wrong by
/// `1/|det J|²` for a mass-type integrand.  `phys_weight` is always the
/// physical measure, so this is the macro for every `φ·φ`-shaped physical
/// volume form (`MassIntegrator`, `VectorFEMassIntegrator`, the DG jump/mass
/// terms, `SBM2*`, …).  See the module docs for the full selection table.
macro_rules! scalar_bilinear_integrator_phys {
    ($name:ident, $field:ident, $doc:literal, |$qp:ident, $kelem:ident, $n:ident, $w:ident| $body:block) => {
        use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
        use crate::integrator::{BilinearIntegrator, QpData};

        #[doc = $doc]
        pub struct $name<C: ScalarCoeff = f64> {
            pub $field: C,
        }

        impl<C: ScalarCoeff> BilinearIntegrator for $name<C> {
            fn add_to_element_matrix(&self, $qp: &QpData<'_>, $kelem: &mut [f64]) {
                let $n = $qp.n_dofs;
                let ctx = CoeffCtx::from_qp(
                    $qp.x_phys, $qp.dim, $qp.elem_id, $qp.elem_tag,
                    Some($qp.phi), $qp.elem_dofs,
                );
                let $w = $qp.phys_weight * self.$field.eval(&ctx);
                $body
            }
        }
    };
}

/// Helper macro for boundary scalar bilinear integrators with a [`ScalarCoeff`] field.
///
/// Like [`scalar_bilinear_integrator!`] but for [`BoundaryBilinearIntegrator`] on
/// [`BdQpData`], with `CoeffCtx::from_qp` receiving `Some(phi), None`.
macro_rules! boundary_scalar_bilinear {
    ($name:ident, $field:ident, $doc:literal, |$qp:ident, $kface:ident, $n:ident, $w:ident| $body:block) => {
        use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
        use crate::integrator::{BdQpData, BoundaryBilinearIntegrator};

        #[doc = $doc]
        pub struct $name<C: ScalarCoeff = f64> {
            pub $field: C,
        }

        impl<C: ScalarCoeff> BoundaryBilinearIntegrator for $name<C> {
            fn add_to_face_matrix(&self, $qp: &BdQpData<'_>, $kface: &mut [f64]) {
                let $n = $qp.n_dofs;
                let ctx = CoeffCtx::from_qp(
                    $qp.x_phys, $qp.dim, $qp.elem_id, $qp.elem_tag,
                    Some($qp.phi), None,
                );
                let $w = $qp.weight * self.$field.eval(&ctx);
                $body
            }
        }
    };
}

/// Helper macro for domain source (linear form) integrators with a closure.
///
/// Generates a struct generic over `F: Fn(&[f64]) -> f64 + Send + Sync` with a
/// `pub fn new(f: F) -> Self` constructor, plus the [`LinearIntegrator`] impl.
///
/// The body block receives `|qp, f_elem, n, w|` where `w = qp.weight × f(x)`.
macro_rules! domain_linear_closure {
    ($name:ident, $doc:literal, |$qp:ident, $felem:ident, $n:ident, $w:ident| $body:block) => {
        use crate::integrator::{LinearIntegrator, QpData};

        #[doc = $doc]
        pub struct $name<F: Fn(&[f64]) -> f64 + Send + Sync> {
            f: F,
        }

        impl<F: Fn(&[f64]) -> f64 + Send + Sync> $name<F> {
            pub fn new(f: F) -> Self { $name { f } }
        }

        impl<F: Fn(&[f64]) -> f64 + Send + Sync> LinearIntegrator for $name<F> {
            fn add_to_element_vector(&self, $qp: &QpData<'_>, $felem: &mut [f64]) {
                let $n = $qp.n_dofs;
                let $w = $qp.weight * (self.f)($qp.x_phys);
                $body
            }
        }
    };
}

/// Helper macro for boundary linear (Neumann) integrators with a closure.
///
/// Like [`domain_linear_closure!`] but for [`BoundaryLinearIntegrator`] on
/// [`BdQpData`]; the closure receives both coordinates and the outward normal.
/// Body block receives `|qp, f_face, n, w|` where `w = qp.weight × g(x, n)`.
macro_rules! boundary_linear_closure {
    ($name:ident, $doc:literal, |$qp:ident, $fface:ident, $n:ident, $w:ident| $body:block) => {
        use crate::integrator::{BdQpData, BoundaryLinearIntegrator};

        #[doc = $doc]
        pub struct $name<F: Fn(&[f64], &[f64]) -> f64 + Send + Sync> {
            g: F,
        }

        impl<F: Fn(&[f64], &[f64]) -> f64 + Send + Sync> $name<F> {
            pub fn new(g: F) -> Self { $name { g } }
        }

        impl<F: Fn(&[f64], &[f64]) -> f64 + Send + Sync> BoundaryLinearIntegrator for $name<F> {
            fn add_to_face_vector(&self, $qp: &BdQpData<'_>, $fface: &mut [f64]) {
                let $n = $qp.n_dofs;
                let $w = $qp.weight * (self.g)($qp.x_phys, $qp.normal);
                $body
            }
        }
    };
}

pub mod diffusion;
pub mod tensor_diffusion;
pub mod mass;
pub mod neumann;
pub mod source;
pub mod elasticity;
pub mod curl_curl;
pub mod vector_mass;
pub mod convection;
pub mod vector_diffusion;
pub mod vector_h1_mass;
pub mod vector_convection;
pub mod boundary_mass;
pub mod grad_div;
pub mod transpose;
pub mod sum;
pub mod vector_source;
pub mod boundary_flux;
pub mod tangential_boundary;
pub mod bbar;
pub mod infinite;
pub mod shell_mitc4;
pub mod vec_fe_divergence;
pub mod normal_trace;
pub mod vec_bdr_flux_lf;
pub mod dg_dirichlet_lf;
pub mod vec_fe_weak_div;
pub mod vec_fe_curl;
pub mod diffusion2;
pub mod hyperelastic_nl;
pub mod elasticity_component;

pub use diffusion::DiffusionIntegrator;
pub use tensor_diffusion::{TensorDiffusionIntegrator, AnisotropicDiffusionIntegrator};
pub use elasticity::ElasticityIntegrator;
pub use mass::MassIntegrator;
pub use neumann::NeumannIntegrator;
pub use source::DomainSourceIntegrator;
pub use source::DomainSourceIntegratorCoeff;
pub use curl_curl::{CurlCurlIntegrator, CurlCurlTensorIntegrator, AnisotropicCurlCurlIntegrator};
pub use vector_mass::{
    VectorMassIntegrator, VectorMassTensorIntegrator, VectorFEMassIntegrator,
};
pub use convection::{
    ConvectionIntegrator, MixedDirectionalDerivativeIntegrator, mfem_quad_order,
};
pub use vector_diffusion::VectorDiffusionIntegrator;
pub use vector_h1_mass::VectorH1MassIntegrator;
pub use vector_convection::VectorConvectionIntegrator;
pub use boundary_mass::BoundaryMassIntegrator;
pub use grad_div::GradDivIntegrator;
pub use transpose::TransposeIntegrator;
pub use sum::SumIntegrator;
pub use vector_source::VectorDomainLFIntegrator;
pub use bbar::{assemble_bbar_elasticity, FBarIntegrator};
pub use infinite::InfiniteDomainIntegrator;
pub use shell_mitc4::{mitc4_shell_stiffness, mitc4_shell_mass};
pub use boundary_flux::{
    BoundaryNormalLFIntegrator, VectorBoundaryNormalLFIntegrator, VectorFEBoundaryFluxLFIntegrator,
};
pub use tangential_boundary::TangentialTraceLFIntegrator;

pub use vec_fe_divergence::VectorFEDivergenceIntegrator;
pub use normal_trace::NormalTraceIntegrator;
pub use vec_bdr_flux_lf::VectorBoundaryFluxLFIntegrator;
pub use dg_dirichlet_lf::DGDirichletLFIntegrator;
pub use vec_fe_weak_div::VectorFEWeakDivergenceIntegrator;
pub use vec_fe_curl::VectorFECurlIntegrator;
pub use diffusion2::Diffusion2Integrator;
pub use hyperelastic_nl::HyperelasticNLFIntegrator;
pub use elasticity_component::ElasticityComponentIntegrator;
pub mod vector_boundary_lf;
pub mod sbm2_dirichlet;
pub mod sbm2_neumann;
pub mod sbm3_dirichlet;
pub mod sbm3_neumann;
pub mod misc_integrators;
/// Minimal serial MFEM `NonlinearForm` framework (promoted from
/// `dist_solver::filter` in D52) — the *assembly-side* form, distinct from
/// `physics::nonlinear::NonlinearForm` (the solver-side Newton form).
pub mod nonlinear_form;
pub mod vector_convection_nlf;

pub use vector_boundary_lf::VectorBoundaryLFIntegrator;
pub use sbm2_dirichlet::{SBM2DirichletLFIntegrator, SBM2DirichletIntegrator};
pub use sbm2_neumann::{SBM2NeumannLFIntegrator, SBM2NeumannIntegrator};
pub use sbm3_dirichlet::{Sbm3DirichletIntegrator, Sbm3DirichletLFIntegrator};
pub use sbm3_neumann::{Sbm3NeumannIntegrator, Sbm3NeumannLFIntegrator};
pub use misc_integrators::{
    VectorDivergenceIntegrator,
    WhiteGaussianNoiseDomainLFIntegrator, NormalTraceJumpIntegrator,
    NonconservativeDGTraceIntegrator, MixedWeakGradDotIntegrator,
    MixedWeakCurlCrossIntegrator, DivDivIntegrator,
};
pub use vector_convection_nlf::VectorConvectionNLFIntegrator;
