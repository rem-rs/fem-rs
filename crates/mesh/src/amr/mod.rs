//! Adaptive Mesh Refinement (AMR).
mod amr_inner;
mod bisect;
mod estimators;
mod p_refine;
mod refine_2d;
mod make_conforming;
pub use amr_inner::*;
pub use bisect::*;
pub use estimators::*;
pub use p_refine::*;
pub use refine_2d::{closure_refine, closure_refine_default};
pub use make_conforming::make_conforming_tri;

