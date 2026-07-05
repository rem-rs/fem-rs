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
        use crate::coefficient::{CoeffCtx, ScalarCoeff};
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

pub mod diffusion;
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

pub use diffusion::DiffusionIntegrator;
pub use elasticity::ElasticityIntegrator;
pub use mass::MassIntegrator;
pub use neumann::NeumannIntegrator;
pub use source::DomainSourceIntegrator;
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
pub use boundary_flux::{BoundaryNormalLFIntegrator, VectorFEBoundaryFluxLFIntegrator};
pub use tangential_boundary::TangentialTraceLFIntegrator;
