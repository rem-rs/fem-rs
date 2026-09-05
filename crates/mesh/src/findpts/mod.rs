//! FindPoints — spatial query for simplex meshes.
//!
//! Self-contained implementation — no external crates.
//!
//! Modules:
//! - [`bvh`] — Bounding Volume Hierarchy (AABB tree, median-split)
//! - [`newton`] — Newton iteration for inverse isoparametric mapping
//! - [`find_points`] — [`FindPoints`] query API combining BVH + Newton

pub mod bvh;
pub mod find_points;
pub mod newton;

pub use find_points::{FindPoints, FindPointsOptions, LocatedPoint};
