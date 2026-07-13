//! Adaptive Mesh Refinement (AMR).
mod amr_inner;
mod bisect;
mod estimators;
mod p_refine;
mod refine_2d;
mod make_conforming;
mod refinement_tree;
pub use amr_inner::*;
pub use amr_inner::{
    refine_uniform_surface_tri3, refine_uniform_surface_quad4,
    refine_at_vertex_surface,
};
pub use bisect::*;
pub use estimators::*;
pub use p_refine::*;
pub use refine_2d::{closure_refine, closure_refine_default};
pub use make_conforming::make_conforming_tri;
pub use refinement_tree::*;

