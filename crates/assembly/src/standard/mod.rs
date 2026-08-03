//! Standard finite element integrators.
//!
//! Re-exports the most commonly used integrators for convenience.

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

pub use diffusion::DiffusionIntegrator;
pub use tensor_diffusion::TensorDiffusionIntegrator;
pub use elasticity::ElasticityIntegrator;
pub use mass::MassIntegrator;
pub use neumann::NeumannIntegrator;
pub use source::DomainSourceIntegrator;
pub use source::DomainSourceIntegratorCoeff;
pub use curl_curl::{CurlCurlIntegrator, CurlCurlTensorIntegrator};
pub use vector_mass::{VectorMassIntegrator, VectorMassTensorIntegrator};
pub use convection::ConvectionIntegrator;
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
pub use boundary_flux::{BoundaryNormalLFIntegrator, VectorFEBoundaryFluxLFIntegrator};
pub use tangential_boundary::TangentialTraceLFIntegrator;
