//! # fem-core
//!
//! Foundational types, traits, and error handling for the fem-rs workspace.
//! Every other crate in the workspace depends on this one.
//!
//! ## Modules
//! - [`scalar`]  — floating-point scalar abstraction (`f32` / `f64`)
//! - [`types`]   — index type aliases (`NodeId`, `ElemId`, `DofId`, `FaceId`)
//! - [`error`]   — `FemError` enum and `FemResult<T>` alias
//! - [`point`]   — coordinate and matrix type aliases (nalgebra re-exports)

pub mod error;
pub mod point;
pub mod scalar;
pub mod types;

// Flat re-exports for ergonomic use: `use fem_core::*` in other crates.
pub use error::{FemError, FemResult};
pub use point::{Coord2, Coord3, Mat2x2, Mat3x3, Vec2, Vec3};
pub use scalar::Scalar;
pub use types::{DofId, EdgeId, ElemId, FaceId, NodeId, Rank};
