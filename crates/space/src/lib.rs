//! # fem-space
//!
//! Finite element spaces: DOF management, H¹ continuous Lagrange, and L² DG.
//!
//! ## Core components
//! - [`FESpace`] — trait shared by all finite element spaces
//! - [`DofManager`] — builds and stores element→global DOF maps
//! - [`H1Space`] — continuous Lagrange space (P1 or P2 on triangular meshes)
//! - [`L2Space`] — discontinuous Lagrange space (P0 or P1 per element)
//! - [`apply_dirichlet`] — zero-out / set Dirichlet rows in a stiffness matrix

pub mod dof_manager;
pub mod fe_space;
pub mod h1;
pub mod l2;
pub mod constraints;

pub use dof_manager::DofManager;
pub use fe_space::{FESpace, SpaceType};
pub use h1::H1Space;
pub use l2::L2Space;
pub use constraints::{apply_dirichlet, boundary_dofs};
