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

pub mod tri;
pub mod tri_rt1;
pub mod tri_rt2;
pub mod tri_rtk;
pub mod quad_rt0;
pub mod quad_rt1;
pub mod quad_rtk;
pub mod tet;
pub mod tet_rt1;
pub mod tet_rt2;
pub mod tet_rtk;
pub mod hex_rt0;
pub mod hex_rt1;
pub mod hex_rtk;

pub use tri::TriRT0;
pub use tri_rt1::TriRT1;
pub use tri_rt2::TriRT2;
pub use tri_rtk::TriRTk;
pub use quad_rt0::QuadRT0;
pub use quad_rt1::QuadRT1;
pub use quad_rtk::QuadRTk;
pub use tet::TetRT0;
pub use tet_rt1::TetRT1;
pub use tet_rt2::TetRT2;
pub use tet_rtk::TetRTk;
pub use hex_rt0::HexRT0;
pub use hex_rt1::HexRT1;
pub use hex_rtk::HexRTk;
