//! FindPoints — spatial query for simplex meshes.
//!
//! Self-contained implementation — no external crates.
//!
//! Modules:
//! - [`bvh`] — Bounding Volume Hierarchy (AABB tree, median-split)
//! - [`newton`] — Newton iteration for inverse isoparametric mapping
//! - [`find_points`] — [`FindPoints`] query API combining BVH + Newton
//! - [`gslib`] — [`GslibFindPoints`] serial locator with MFEM
//!   FindPointsGSLIB code/dist semantics for general (curved) meshes

pub mod bvh;
pub mod find_points;
pub mod gslib;
pub mod newton;

pub use find_points::{FindPoints, FindPointsOptions, LocatedPoint};
pub use gslib::{
    GslibFindPoints, GslibPoint, CODE_BORDER, CODE_INSIDE, CODE_NOT_FOUND,
    DEFAULT_BDR_TOL, DEFAULT_NEWT_TOL, STRICT_TOL,
};
