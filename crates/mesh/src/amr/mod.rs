//! Adaptive Mesh Refinement (AMR).
mod amr_inner;
mod refine_2d;
mod make_conforming;
pub use amr_inner::*;
pub use make_conforming::make_conforming_tri;

