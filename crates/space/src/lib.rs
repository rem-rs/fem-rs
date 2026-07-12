#![allow(clippy::needless_range_loop)]

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
pub mod restricted_hcurl;
pub mod hdiv;
pub mod constraints;
pub mod iga;
pub mod iga_fe_space;
pub mod nurbs_fe_collection;
pub mod p_refine;
pub mod vector_h1;
pub mod skeleton;
pub mod block_fe_space;
pub mod trace_spaces;
pub mod cr_space;
pub mod vector_cr;
pub mod vem;

pub mod complex;

pub use dof_manager::{DofManager, EdgeKey, FaceKey};
pub use fe_space::{FESpace, SpaceType};
pub use h1::H1Space;
pub use cr_space::CRSpace;
pub use vector_cr::VectorCRSpace;
pub use h1_trace::H1TraceSpace;
pub use l2::L2Space;
pub use hcurl::HCurlSpace;
pub use restricted_hcurl::RestrictedHCurlSpace;
pub use hdiv::HDivSpace;
pub use iga_fe_space::{IgaFESpace1D, IgaFESpace2D, IgaFESpace3D, IgaSinglePatchMesh1D, IgaSinglePatchMesh2D, IgaSinglePatchMesh3D, IgaMultiPatchMesh2D, IgaMultiPatchMesh3D};
pub use vector_h1::VectorH1Space;
pub use constraints::{apply_dirichlet, eliminate_dirichlet, expand_from_reduced, apply_hanging_constraints, apply_hanging_face_constraints, recover_hanging_values, recover_hanging_face_values, prolongate_p2_hanging, boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv, identify_periodic_dof_pairs, apply_periodic};
pub use iga::{IgaBoundary2D, IgaBoundary3D, IgaSpace1D, IgaSpace2D, IgaSpace3D};
pub use block_fe_space::BlockFESpace;
pub use trace_spaces::{HCurlTraceSpace, HDivTraceSpace};
pub use complex::{ComplexGridFunction, ComplexSpace, apply_complex_dirichlet};

