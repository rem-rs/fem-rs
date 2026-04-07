//! # fem-mesh
//!
//! Mesh topology and geometry for fem-rs.
//!
//! ## Modules
//! - [`element_type`] — `ElementType` enum (Tri3, Tet4, Hex8, …)
//! - [`boundary`]     — `BoundaryTag` and `PhysicalGroup`
//! - [`topology`]     — `MeshTopology` trait
//! - [`simplex`]      — `SimplexMesh<D>`: concrete unstructured mesh with built-in generators

pub mod amr;
pub mod boundary;
pub mod curved;
pub mod element_type;
pub mod simplex;
pub mod topology;
pub mod transformation;

pub use amr::{refine_marked, refine_nonconforming, refine_nonconforming_3d, refine_uniform, dorfler_mark, zz_estimator, kelly_estimator, prolongate_p1, HangingNodeConstraint, HangingFaceConstraint, NCState, NCState3D};
pub use boundary::{BoundaryTag, PhysicalGroup};
pub use curved::CurvedMesh;
pub use element_type::ElementType;
pub use simplex::SimplexMesh;
pub use topology::MeshTopology;
pub use transformation::ElementTransformation;
