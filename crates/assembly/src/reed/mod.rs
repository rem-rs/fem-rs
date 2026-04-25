//! Partial-assembly and libCEED-style operators via the [`reed`] workspace crates.
//!
//! Enable with **`--features reed`** on `fem-assembly`. This keeps default builds
//! (including `fem-wasm`) free of the reed dependency graph unless explicitly requested.
//!
//! ## Coordinated FEM + reed surface
//!
//! This module is the **integration root** for “fem-rs spaces + reed execution”:
//! - [`FemCeed`] — scalar H¹ mass / Poisson on **triangles** (`apply_mass_2d`, …) and **tets**
//!   (`apply_mass_3d`, …) via [`crate::assembler::Assembler`]; for iterations use the matching
//!   `cache_*` helpers and [`FemCeed::cache_h1_scalar_ops_2d`] / [`FemCeed::cache_h1_scalar_ops_3d`].
//!   CSR helpers such as [`FemCeed::assemble_curl_hdiv_nd2_rt2_csr`]
//!   delegate to [`crate::DiscreteLinearOperator`] so there is **one** ND2→RT2 curl kernel
//!   (shared with default builds).
//! - `fem_discrete` — [`assemble_mass_h1_2d`] / [`assemble_poisson_h1_2d`],
//!   [`assemble_mass_h1_3d`] / [`assemble_poisson_h1_3d`], [`h1_tri_quad_order`] /
//!   [`h1_tet_quad_order`] (aliases of [`crate::h1_quad_order_hint`]), plus
//!   [`assemble_curl_hdiv_pairing_2d_nd2_rt2`] for callers who import from `fem_assembly::reed` only.
//!
//! ## Crate root (`--features reed`)
//!
//! The same [`FemCeed`], cache handles, and `fem_discrete` assemblers are **re-exported** from
//! `fem_assembly` — use either `fem_assembly::FemCeed` or `fem_assembly::reed::FemCeed` according to
//! import style.  Quadrature hint mapping [`crate::h1_quad_order_hint`] is additionally available at
//! the crate root **without** enabling `reed`.

pub mod context;
pub mod hcurl;
pub mod fem_discrete;
pub mod qfunction;
pub mod restriction;

pub use context::{
    CachedH1Mass2d, CachedH1Mass3d, CachedH1Poisson2d, CachedH1Poisson3d, CachedH1ScalarOps2d,
    CachedH1ScalarOps3d, CeedBackend, FemCeed, FemCeedError,
};
pub use hcurl::HcurlReedOperator;
pub use fem_discrete::{
    assemble_curl_hdiv_pairing_2d_nd2_rt2, assemble_mass_h1_2d, assemble_mass_h1_3d,
    assemble_poisson_h1_2d, assemble_poisson_h1_3d, h1_tet_quad_order, h1_tri_quad_order,
    TRI_ND2_RT2_MIXED_QUAD_ORDER,
};
pub use restriction::{mesh_to_elem_restriction, qdata_elem_restriction};
