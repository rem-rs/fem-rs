//! Adaptive Mesh Refinement (AMR).
pub mod convergence;
pub use convergence::{ConvergenceStudy, ConvergenceRecord};
mod amr_inner;
mod bisect;
mod curved_hex;
mod curved_prism;
mod curved_quad;
mod curved_tet;
mod curved_tri;
pub mod nc_quad_tree;
mod p_refine;
mod refine_2d;
mod make_conforming;
mod refinement_tree;
mod schedule;
pub mod general_refinement;
pub mod sfc_ordering;
pub use amr_inner::*;
pub use refinement_tree::*;
pub use amr_inner::{
    refine_uniform_surface_tri3, refine_uniform_surface_quad4,
    refine_at_vertex_surface,
};
pub use bisect::*;
pub use p_refine::*;
pub use refine_2d::{
    closure_refine, closure_refine_default, general_refinement_quad,
    general_refinement_quad_aniso,
};
pub use make_conforming::make_conforming_tri;
pub use schedule::*;

