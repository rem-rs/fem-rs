// Crate-level attribute: non_snake_case allowed for physics notation (e.g. C10, D1 coefficients).
// The unused attribute was removed — handle individual items with targeted #[allow(...)].
#![allow(non_snake_case)]
//! # fem-core
//!
//! Foundational types, traits, and error handling for the fem-rs workspace.
//! Every other crate in the workspace depends on this one.
//!
//! ## Modules
//! - [`scalar`]  — floating-point scalar abstraction (`f32` / `f64`)
//! - [`types`]   — index type aliases (`NodeId`, `ElemId`, `DofId`, `FaceId`)
//! - [`error`]   — `FemError` enum and `FemResult<T>` alias
//! - [`material`] — material response types and constitutive helpers
//!
//! # Examples
//!
//! ```
//! use fem_core::{FemError, FemResult, NodeId, ElemId, Scalar};
//!
//! // Index types are simple u32 wrappers
//! let node: NodeId = 5;
//! assert_eq!(node, 5);
//!
//! let elem: ElemId = 3;
//! assert_eq!(elem, 3);
//!
//! // FemResult<T> is the standard return type throughout fem-rs
//! fn safe_divide(a: f64, b: f64) -> FemResult<f64> {
//!     if b.abs() < 1e-300 {
//!         Err(FemError::Other("division by zero".into()))
//!     } else {
//!         Ok(a / b)
//!     }
//! }
//! assert!(safe_divide(1.0, 0.0).is_err());
//! assert!((safe_divide(6.0, 2.0).unwrap() - 3.0).abs() < 1e-15);
//!
//! // Scalar trait abstracts over f32/f64
//! fn double<T: Scalar>(x: T) -> T { x + x }
//! assert_eq!(double(2.0_f64), 4.0);
//! ```

pub mod error;
pub mod material;
pub mod scalar;
pub mod types;

// Flat re-exports for ergonomic use: `use fem_core::*` in other crates.
pub use error::{FemError, FemResult};
pub use material::{MaterialModel, FiniteStrainMaterial, MaterialResponse, linear_elastic_stiffness, DeformationGradient};
pub use scalar::Scalar;
pub use types::{DofId, EdgeId, ElemId, FaceId, NodeId, Rank};
