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
pub mod point_locator;
pub mod simplex;
pub mod submesh;
pub mod topology;
pub mod transformation;
pub mod extrusion;
pub mod findpts;
pub mod particle;
pub mod supermesh;
pub mod rebuild_boundary;
pub mod mfem_kernels;
pub mod tmop;

pub use tmop::invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};
pub use tmop::metrics::{
    TmopQualityMetric, TmopQualityMetric3D,
    TmopMetric001, TmopMetric002, TmopMetric007, TmopMetric009,
    TmopMetric014, TmopMetric022, TmopMetric050, TmopMetric055, TmopMetric056,
    TmopMetric058, TmopMetric077, TmopAMetric014, TmopAMetric050,
    TmopMetric301, TmopMetric302, TmopMetric303, TmopMetric304,
    TmopMetric315, TmopMetric316, TmopMetric318, TmopMetric321, TmopMetric323, TmopMetric360,
};
pub use tmop::target::{TargetConstructor, TargetType, ideal_shape_jac_2d, ideal_shape_jac_3d};
pub use tmop::check::{check_metric_2d, check_metric_3d, run_tmop_check_metric, MetricCheckResult};
pub use tmop::integrator::{TmopIntegrator2D, TmopIntegrator3D};

pub use amr::{mark_tet_mesh_for_refinement, tet_select_rt_debug, refine_marked, closure_refine, closure_refine_default, refine_marked_with_tree, derefine_marked, DerefineTree, DerefineRecord, refine_nonconforming, refine_nonconforming_3d, refine_nonconforming_quad, refine_nonconforming_hex, refine_nonconforming_quad_aniso, refine_nonconforming_hex_aniso, refine_nonconforming_prism, refine_nonconforming_prism_aniso, refine_nonconforming_pyramid, refine_nonconforming_pyramid_aniso, refine_nonconforming_tri_aniso, refine_nonconforming_tet_aniso, QuadRefineDir, HexRefineDir, TriRefineDir, TetRefineDir, PrismRefineDir, PyramidRefineDir, refine_uniform, refine_uniform_3d, refine_prism6_uniform, refine_pyramid5_uniform, refine_hex8_uniform, refine_hex20_uniform, refine_hex27_uniform, prolongate_p1, restrict_to_coarse_p1, HangingNodeConstraint, HangingFaceConstraint, HangingQuadFaceConstraint, limit_nc_level_quad, NCState, NCState3D, NCStateQuad, NCStateHex, make_conforming_tri};
pub use amr::{p_refine_tri3_to_tri6, p_refine_tri6_to_tri10, p_refine_tet4_to_tet10, p_refine_tet10_to_tet20, p_refine_quad4_to_quad9, p_refine_hex8_to_hex20, p_refine_hex20_to_hex27, p_prolongate_p1_to_p2};
pub use boundary::{BoundaryTag, NamedAttributeRegistry, NamedAttributeSet, PhysicalGroup};
pub use cad::{CadShape, CadModel, AnalyticSurface, FacetedCadSurface, NurbsCadSurface2D, TrimLoop, TrimmedNurbsSurface, ProjectionConfig, project_boundary_to_cad, project_elevated_node};
pub use curved::{CurvedMesh, JacobianCache, CurvedElementTransformation, refine_curved_2d, refine_curved_3d, refine_curved_3d_general, refine_curved_2d_nc, refine_curved_3d_nc, refine_curved_3d_nc_general, refine_curved_2d_nc_with_cad, refine_curved_3d_nc_with_cad};
pub use element_type::ElementType;
pub use point_locator::{LocatedPoint2D, LocatedPoint3D, TetPointLocator, TriPointLocator};
pub use findpts::{FindPoints, FindPointsOptions, LocatedPoint as FindPointResult};
pub use simplex::{Mesh, tet_volume};
pub use submesh::{SubMesh, SubMesh3D, BoundarySubMesh, extract_submesh, extract_submesh_3d, extract_submesh_by_name, extract_boundary_submesh};
pub use topology::MeshTopology;
pub use transformation::{ElementTransformation, geometry_jacobian, element_jacobian_at, xform_grads};
pub use extrusion::{extrude_tri3_to_prisms, extrude_quad4_to_hex8};
pub use particle::ParticleSet;
pub use supermesh::{build_supermesh, SupermeshElement};
pub mod nurbs_mesh;
