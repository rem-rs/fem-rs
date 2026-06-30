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
pub mod cad;
pub mod curved;
pub mod element_type;
pub mod lor;
pub mod moving_mesh;
pub mod point_locator;
pub mod simplex;
pub mod submesh;
pub mod tmop;
pub mod topology;
pub mod transformation;
pub mod step_iges;

pub use amr::{refine_marked, refine_marked_with_tree, derefine_marked, DerefineTree, DerefineRecord, refine_nonconforming, refine_nonconforming_3d, refine_nonconforming_quad, refine_nonconforming_hex, refine_nonconforming_quad_aniso, refine_nonconforming_hex_aniso, QuadRefineDir, HexRefineDir, refine_uniform, refine_uniform_3d, dorfler_mark, mark_for_derefinement, mark_for_p_refinement, p_refine_tri3_to_tri6, p_prolongate_p1_to_p2, zz_estimator, kelly_estimator, dwr_estimator, prolongate_p1, restrict_to_coarse_p1, HangingNodeConstraint, HangingFaceConstraint, NCState, NCState3D, NCStateQuad};
pub use boundary::{BoundaryTag, NamedAttributeRegistry, NamedAttributeSet, PhysicalGroup};
pub use cad::{CadShape, CadModel, AnalyticSurface, FacetedCadSurface, NurbsCadSurface2D, ProjectionConfig, project_boundary_to_cad, project_elevated_node};
pub use curved::{CurvedMesh, JacobianCache, CurvedElementTransformation, refine_curved_2d, refine_curved_3d, refine_curved_2d_nc, refine_curved_3d_nc};
pub use element_type::ElementType;
pub use lor::LorMesh;
pub use moving_mesh::{
	MeshMotionConfig,
	all_boundary_nodes,
	apply_node_displacement,
	boundary_nodes_with_tags,
	laplacian_smooth_2d,
};
pub use point_locator::{LocatedPoint2D, LocatedPoint3D, TetPointLocator, TriPointLocator};
pub use simplex::SimplexMesh;
pub use submesh::{SubMesh, extract_submesh, extract_submesh_by_name};
pub use topology::MeshTopology;
pub use transformation::ElementTransformation;
