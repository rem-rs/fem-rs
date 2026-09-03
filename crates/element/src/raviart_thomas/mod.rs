//! Raviart-Thomas H(div) elements.
//!
//! These elements provide **normal continuity** across inter-element faces and are
//! the canonical choice for mixed formulations of Darcy flow, Stokes, and
//! incompressible elasticity.
//!
//! # DOF convention
//! Each DOF is a normal-flux moment on a face (edge in 2-D):
//! `DOF_i = ∫_{f_i} Φ · n̂ᵢ ds`
//!
//! # Available elements
//! | Type        | Domain      | DOFs | Order |
//! |-------------|-------------|------|-------|
//! | [`TriRT0`]  | triangle    | 3    | 0     |
//! | [`TriRT1`]  | triangle    | 8    | 1     |
//! | [`TriRT2`]  | triangle    | 15   | 2     |
//! | [`QuadRT0`] | quadrilateral | 4  | 0     |
//! | [`TetRT0`]  | tetrahedron | 4    | 0     |
//! | [`TetRT1`]  | tetrahedron | 15   | 1     |
//! | [`HexRT0`]  | hexahedron  | 6    | 0     |
//! | [`HexRT1`]  | hexahedron  | 36   | 1     |
//! | [`TetRT2`]  | tetrahedron | 36   | 2     |

pub mod hex_rt1;
pub mod hex_rtk;
pub mod prism;
pub mod pyramid;
pub mod quad_rt1;
pub mod quad_rtk;
pub mod tet_rt1;
pub mod tet_rt2;
pub mod tet_rtk;
pub mod tri_rt1;
pub mod tri_rt2;
pub mod tri_rtk;

pub use hex_rt1::HexRT1;
pub use hex_rtk::HexRTk;
pub use prism::{PrismRT0, PrismRTk};
pub use pyramid::{PyraRT0, PyraRTk};
pub use quad_rt1::QuadRT1;
pub use quad_rtk::QuadRTk;
pub use tet_rt1::TetRT1;
pub use tet_rt2::TetRT2;
pub use tet_rtk::TetRTk;
pub use tri_rt1::TriRT1;
pub use tri_rt2::TriRT2;
pub use tri_rtk::TriRTk;

/// RT0 on quadrilateral — type alias for `QuadRTk` with order 0.
///
/// This provides the same Piola basis functions as the former `QuadRT0`:
/// Φ₀=(0,y-1), Φ₁=(x,0), Φ₂=(0,y), Φ₃=(x-1,0), with div=1.
pub type QuadRT0 = QuadRTk;
