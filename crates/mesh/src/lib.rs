#![allow(clippy::needless_range_loop)]
//! # fem-mesh
//!
//! Mesh topology and geometry for fem-rs.
//!
//! ## Modules
//! - [`element_type`] — `ElementType` enum (Tri3, Tet4, Hex8, …)
//! - [`boundary`]     — `BoundaryTag` and `PhysicalGroup`
//! - [`topology`]     — `MeshTopology` trait
//! - [`simplex`]      — [`Mesh<D>`]: concrete unstructured mesh with built-in generators

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
pub mod cut_cell;
pub mod hp_amr;
pub mod dec;
pub mod extrusion;
pub mod particle;
pub mod poly_mesh;
pub mod pumi_mesh;
pub mod size_function;
pub mod supermesh;
pub mod rebuild_boundary;

pub use amr::{refine_marked, closure_refine, closure_refine_default, refine_marked_with_tree, derefine_marked, DerefineTree, DerefineRecord, refine_nonconforming, refine_nonconforming_3d, refine_nonconforming_quad, refine_nonconforming_hex, refine_nonconforming_quad_aniso, refine_nonconforming_hex_aniso, refine_nonconforming_prism, refine_nonconforming_prism_aniso, refine_nonconforming_pyramid, refine_nonconforming_pyramid_aniso, refine_nonconforming_tri_aniso, refine_nonconforming_tet_aniso, QuadRefineDir, HexRefineDir, TriRefineDir, TetRefineDir, PrismRefineDir, PyramidRefineDir, refine_uniform, refine_uniform_3d, refine_prism6_uniform, refine_pyramid5_uniform, refine_hex8_uniform, refine_hex20_uniform, refine_hex27_uniform, zz_estimator, zz_estimator_3d, zz_estimator_3d_general, kelly_estimator, kelly_estimator_3d, kelly_estimator_3d_general, residual_estimator, residual_estimator_3d, residual_estimator_3d_general, dwr_estimator, dwr_estimator_3d_general, prolongate_p1, restrict_to_coarse_p1, HangingNodeConstraint, HangingFaceConstraint, HangingQuadFaceConstraint, NCState, NCState3D, NCStateQuad, NCStateHex, NCStatePrism, NCStatePyramid, make_conforming_tri};
pub use amr::{dorfler_mark, mark_for_derefinement, mark_for_p_refinement, p_refine_tri3_to_tri6, p_refine_tri6_to_tri10, p_refine_tet4_to_tet10, p_refine_tet10_to_tet20, p_refine_quad4_to_quad9, p_refine_hex8_to_hex20, p_refine_hex20_to_hex27, p_prolongate_p1_to_p2};
pub use hp_amr::{HpAction, hp_mark, compute_smoothness_indicator, mark_smooth_for_p_refinement, mark_rough_for_h_refinement};
pub use boundary::{BoundaryTag, NamedAttributeRegistry, NamedAttributeSet, PhysicalGroup};
pub use cad::{CadShape, CadModel, AnalyticSurface, FacetedCadSurface, NurbsCadSurface2D, TrimLoop, TrimmedNurbsSurface, ProjectionConfig, project_boundary_to_cad, project_elevated_node};
pub use curved::{CurvedMesh, JacobianCache, CurvedElementTransformation, refine_curved_2d, refine_curved_3d, refine_curved_3d_general, refine_curved_2d_nc, refine_curved_3d_nc, refine_curved_3d_nc_general, refine_curved_2d_nc_with_cad, refine_curved_3d_nc_with_cad};
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
pub use simplex::{Mesh, tet_volume};
pub use submesh::{SubMesh, SubMesh3D, extract_submesh, extract_submesh_3d, extract_submesh_by_name};
pub use topology::MeshTopology;
pub use transformation::ElementTransformation;
pub use extrusion::{extrude_tri3_to_prisms, extrude_quad4_to_hex8};
pub use particle::ParticleSet;
pub use size_function::{
    compute_element_sizes, compute_target_sizes,
    size_to_markers, smooth_size_field,
};
pub use supermesh::{build_supermesh, SupermeshElement};
