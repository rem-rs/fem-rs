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
//! | [`TetRT0`]  | tetrahedron | 4    | 0     |

pub mod tri;
pub mod tri_rt1;
pub mod tet;
pub mod tet_rt1;

pub use tri::TriRT0;
pub use tri_rt1::TriRT1;
pub use tet::TetRT0;
pub use tet_rt1::TetRT1;
