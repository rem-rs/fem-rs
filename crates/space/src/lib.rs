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
pub mod ordering;
pub mod h1;
pub mod l2;
pub mod hcurl;
pub mod hdiv;
pub mod constraints;
pub mod iga;
pub mod iga_fe_space;
pub mod p_refine;
pub mod vector_h1;
pub mod make_refined;
pub mod block_fe_space;
pub mod dpg_trace;


pub use dof_manager::{DofManager, EdgeKey, FaceKey};
pub use fe_space::{FESpace, SpaceType};
pub use ordering::Ordering;
pub use h1::H1Space;

pub use l2::{L2Basis, L2Space};
pub use hcurl::HCurlSpace;
pub use hdiv::HDivSpace;
pub use iga_fe_space::{IgaFESpace1D, IgaFESpace2D, IgaFESpace3D, IgaSinglePatchMesh1D, IgaSinglePatchMesh2D, IgaSinglePatchMesh3D, IgaMultiPatchMesh2D, IgaMultiPatchMesh3D};
pub use vector_h1::VectorH1Space;
pub use constraints::{apply_dirichlet, apply_dirichlet_diag_one, eliminate_dirichlet, expand_from_reduced, apply_hanging_constraints, apply_hanging_face_constraints, recover_hanging_values, recover_hanging_face_values, prolongate_p2_hanging, build_h1_prolongation_matrix, boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv, identify_periodic_dof_pairs, apply_periodic};
pub use iga::{IgaBoundary2D, IgaBoundary3D, IgaSpace1D, IgaSpace2D, IgaSpace3D};
pub use block_fe_space::BlockFESpace;
pub use dpg_trace::{DpgTraceSpace, FaceInfo};
pub use make_refined::make_refined;

