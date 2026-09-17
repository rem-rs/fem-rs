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
//!
//! # D228 — element choice on shared faces/corners (implementation-defined)
//!
//! A point sitting exactly on a face/edge/corner shared by several elements
//! is *contained* by all of them; which element a locator reports is a
//! tie-break rule, not a mathematical property.  The rules, per implementation:
//!
//! - **fem-rs [`GslibFindPoints`]** (this module): among the BVH candidate
//!   elements a *strictly inside* containment beats a *border* containment;
//!   among equals the smallest Newton residual wins; remaining ties resolve
//!   by BVH candidate order (see the module docs of [`gslib`]).
//! - **fem-rs `transformation::find_points` / `MeshTopology::locate`** (the
//!   legacy/serial path used by `get_values`): first hit wins — elements are
//!   probed in **element-index order** (bbox-filtered pass, then a full scan),
//!   Newton must converge to residual < 1e-10 and land within `eps = 1e-8`
//!   of the reference domain.
//! - **MFEM 4.10 without GSLIB** (`Mesh::FindPoints`, `mesh/mesh.cpp:14316`,
//!   the serial reference build): the element whose *center* is closest to
//!   the query point is tried first (strict `<`, so the lowest element index
//!   wins center-distance ties); if the inverse transform does not report
//!   `Inside`, the *vertex neighbours* of that element are tried in
//!   vertex-table row order (elements in index order per vertex), then NCMesh
//!   neighbours on nonconforming meshes.
//! - **MFEM 4.10 with GSLIB** (`FindPointsGSLIB`): the gslib library's own
//!   BVH/search decides; deterministic for a fixed build, but a *different*
//!   rule from all of the above.
//!
//! All four are internally deterministic, but they need not agree with each
//! other.  For **continuous** fields (H1, and H(curl)/H(div) fields at points
//! where the shared-face continuity holds) the choice is immaterial; for
//! **discontinuous** quantities — L2 fields, the normal component of H(div)
//! fields, the tangential components of H(curl) fields, and gradients of
//! discontinuous fields — the reported value depends on the chosen element.
//! Cross-code value comparisons must therefore use *interior* probe points
//! (evidence: `tmp/d158/` tet probe points (0.7, 0.6, 0.25) on `y = 0.6` and
//! (0.5, 0.5, 0.5) on `x = 0.5` — fem-rs picks tets 10/24, MFEM 20/34; see
//! `crates/io/tests/d269_findpoints_element_choice.rs`).

pub mod bvh;
pub mod find_points;
pub mod gslib;
pub mod incomplete;
pub mod locator_cache;
pub mod newton;

pub use find_points::{FindPoints, FindPointsOptions, LocatedPoint};
pub use gslib::{
    GslibFindPoints, GslibPoint, CODE_BORDER, CODE_INSIDE, CODE_NOT_FOUND,
    DEFAULT_BDR_TOL, DEFAULT_NEWT_TOL, STRICT_TOL,
};

// D224 routing helpers for `MeshTopology::locate` (crate-internal).
pub(crate) use gslib::to_factory_coords;
