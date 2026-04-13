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
pub mod lor;
pub mod simplex;
pub mod submesh;
pub mod topology;
pub mod transformation;

pub use amr::{refine_marked, refine_marked_with_tree, derefine_marked, DerefineTree, DerefineRecord, refine_nonconforming, refine_nonconforming_3d, refine_uniform, dorfler_mark, mark_for_derefinement, zz_estimator, kelly_estimator, prolongate_p1, restrict_to_coarse_p1, HangingNodeConstraint, HangingFaceConstraint, NCState, NCState3D};
pub use boundary::{BoundaryTag, NamedAttributeRegistry, NamedAttributeSet, PhysicalGroup};
pub use curved::CurvedMesh;
pub use element_type::ElementType;
pub use lor::LorMesh;
pub use simplex::SimplexMesh;
pub use submesh::{SubMesh, extract_submesh, extract_submesh_by_name};
pub use topology::MeshTopology;
pub use transformation::ElementTransformation;
