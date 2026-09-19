//! Legacy module path for the NURBS knot-vector working API.
//!
//! The implementation used to live here; since D374 it lives in
//! [`crate::nurbs_patch`] together with the `NURBSPatch` object layer it
//! drives (the type was orphaned here — no consumers — and needed MFEM-parity
//! completions: `%g` print format, exact `GetSpan`/`CalcDnShape`, MFEM's
//! `DenseMatrix::Invert`-based `GetInterpolant`).  This module only re-exports
//! the moved type so the public path `fem_mesh::nurbs_mesh::NurbsKnotVector`
//! keeps resolving.

pub use crate::nurbs_patch::NurbsKnotVector;
