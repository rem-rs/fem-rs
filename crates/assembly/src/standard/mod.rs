//! Standard finite element integrators.
//!
//! Re-exports the most commonly used integrators for convenience.

pub mod diffusion;
pub mod mass;
pub mod neumann;
pub mod source;
pub mod elasticity;
pub mod curl_curl;
pub mod vector_mass;
pub mod convection;
pub mod vector_diffusion;
pub mod vector_convection;
pub mod boundary_mass;
pub mod grad_div;
pub mod transpose;
pub mod sum;
pub mod vector_source;
pub mod boundary_flux;

pub use diffusion::DiffusionIntegrator;
pub use elasticity::ElasticityIntegrator;
pub use mass::MassIntegrator;
pub use neumann::NeumannIntegrator;
pub use source::DomainSourceIntegrator;
pub use curl_curl::CurlCurlIntegrator;
pub use vector_mass::VectorMassIntegrator;
pub use convection::ConvectionIntegrator;
pub use vector_diffusion::VectorDiffusionIntegrator;
pub use vector_convection::VectorConvectionIntegrator;
pub use boundary_mass::BoundaryMassIntegrator;
pub use grad_div::GradDivIntegrator;
pub use transpose::TransposeIntegrator;
pub use sum::SumIntegrator;
pub use vector_source::VectorDomainLFIntegrator;
pub use boundary_flux::{BoundaryNormalLFIntegrator, VectorFEBoundaryFluxLFIntegrator};
