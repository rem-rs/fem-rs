//! Standard finite element integrators.
//!
//! Re-exports the most commonly used integrators for convenience.

pub mod diffusion;
pub mod mass;
pub mod neumann;
pub mod source;

pub use diffusion::DiffusionIntegrator;
pub use mass::MassIntegrator;
pub use neumann::NeumannIntegrator;
pub use source::DomainSourceIntegrator;
