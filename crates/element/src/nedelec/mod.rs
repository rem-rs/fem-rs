//! Nedelec (first-kind) H(curl) elements.
//!
//! These elements provide **tangential continuity** across inter-element edges/faces and
//! are the canonical choice for discretising the curl-curl operator that appears in
//! Maxwell's equations.
//!
//! # DOF convention
//! Each DOF is associated with an edge.  The DOF value equals the line-integral of the
//! vector field along that edge: `DOF_i = ∫_{e_i} Φ · t̂ ds`, where `t̂` is the unit
//! tangent of edge `i`.
//!
//! # Available elements
//! | Type       | Domain       | DOFs | Order |
//! |-----------|--------------|------|-------|
//! | [`TriND1`] | triangle     | 3    | 1     |
//! | [`TetND1`] | tetrahedron  | 6    | 1     |

pub mod hex_nd2;
pub mod hex_ndk;
pub mod prism;
pub mod pyramid;
pub mod quad_nd2;
pub mod quad_ndk;
pub mod tet_nd2;
pub mod tet_ndk;
pub mod tri_nd2;
pub mod tri_ndk;

pub use hex_nd2::HexND2;
pub use hex_ndk::HexNDk;
pub use prism::{PrismND1, PrismNDk};
pub use pyramid::{PyraND1, PyraNDk};
pub use quad_nd2::QuadND2;
pub use quad_ndk::QuadNDk;
pub use tet_nd2::TetND2;
pub use tet_ndk::TetNDk;
pub use tri_nd2::TriND2;
pub use tri_ndk::TriNDk;
