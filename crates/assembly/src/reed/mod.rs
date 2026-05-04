//! Partial-assembly and libCEED-style operators via the [`reed`] workspace crates.
//!
//! Enable with **`--features reed`** on `fem-assembly`. This keeps default builds
//! (including `fem-wasm`) free of the reed dependency graph unless explicitly requested.

pub mod context;
pub mod hcurl;
pub mod qfunction;
pub mod restriction;

pub use context::{CeedBackend, FemCeed, FemCeedError};
pub use hcurl::HcurlReedOperator;
pub use restriction::{mesh_to_elem_restriction, qdata_elem_restriction};
