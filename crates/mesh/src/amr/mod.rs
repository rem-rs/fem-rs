//! Adaptive Mesh Refinement (AMR).
pub mod convergence;
pub use convergence::{ConvergenceStudy, ConvergenceRecord};
mod amr_inner;
pub mod anisotropy;
mod bisect;
mod dwr;
mod estimators;
mod forest_cache;
mod p_refine;
mod recovery;
mod refine_2d;
mod make_conforming;
mod refinement_tree;
mod schedule;
pub mod smoothness;
pub use amr_inner::*;
pub use amr_inner::{
    refine_uniform_surface_tri3, refine_uniform_surface_quad4,
    refine_at_vertex_surface,
};
pub use anisotropy::*;
pub use bisect::*;
pub use dwr::*;
pub use estimators::*;
pub use forest_cache::*;
pub use p_refine::*;
pub use recovery::*;
pub use refine_2d::{closure_refine, closure_refine_default};
pub use make_conforming::make_conforming_tri;
pub use refinement_tree::*;
pub use schedule::*;

