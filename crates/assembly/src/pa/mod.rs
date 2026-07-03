//! Partial Assembly (PA) — matrix-free element apply via
//! tensor-product sum-factorization.
//!
//! Supports Hex Qk / Quad Qk diffusion operator.
//! Apply PA(⋅) is equivalent to assembled SpMV up to machine precision.
//!
//! # Design
//! - `PaData` holds per-element precomputed data (J⁻ᵀ, detJ, κ, ...)
//! - `pa_apply` computes y = A·x element-by-element without forming A
//! - Default feature: no extra deps beyond fem-{mesh,space,linalg}

mod hex_q1;
mod quad_q1;
mod q2;
mod q3;
mod q4;
mod tet4;
pub mod types;

pub use hex_q1::{build_hex_q1_pa_data, pa_apply_hex_q1};
pub use quad_q1::{build_quad_q1_pa_data, pa_apply_quad_q1};
pub use q2::{build_hex_q2_pa_data, build_quad_q2_pa_data, pa_apply_hex_q2, pa_apply_quad_q2};
pub use q3::{build_hex_q3_pa_data, pa_apply_hex_q3, pa_apply_hex_q3_sf};
pub use q4::{build_hex_q4_pa_data, pa_apply_hex_q4};
pub use tet4::{build_tet4_pa_data, pa_apply_tet4};
