//! # fem-space
//!
//! Finite element spaces: DOF management for H¹, L², H(curl), and H(div).
//!
//! ## Core components
//! - [`FESpace`] — trait shared by all finite element spaces
//! - [`DofManager`] — builds and stores element→global DOF maps (Lagrange)
//! - [`H1Space`] — continuous Lagrange space (P1 or P2 on triangular meshes)
//! - [`L2Space`] — discontinuous Lagrange space (P0 or P1 per element)
//! - [`VectorH1Space`] — vector-valued H¹ space ([H¹]^d) for elasticity / Stokes
//! - [`HCurlSpace`] — H(curl) Nédélec edge element space
//! - [`HDivSpace`] — H(div) Raviart-Thomas face element space
//! - [`apply_dirichlet`] — zero-out / set Dirichlet rows in a stiffness matrix

pub mod dof_manager;
pub mod fe_space;
pub mod h1;
pub mod h1_trace;
pub mod l2;
pub mod hcurl;
pub mod hdiv;
pub mod constraints;
pub mod vector_h1;

pub use dof_manager::{DofManager, EdgeKey, FaceKey};
pub use fe_space::{FESpace, SpaceType};
pub use h1::H1Space;
pub use h1_trace::H1TraceSpace;
pub use l2::L2Space;
pub use hcurl::HCurlSpace;
pub use hdiv::HDivSpace;
pub use vector_h1::VectorH1Space;
pub use constraints::{apply_dirichlet, apply_hanging_constraints, apply_hanging_face_constraints, recover_hanging_values, recover_hanging_face_values, prolongate_p2_hanging, boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv};

