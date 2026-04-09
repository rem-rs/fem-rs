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

pub mod tri;
pub mod tri_nd2;
pub mod tet;
pub mod tet_nd2;

pub use tri::TriND1;
pub use tri_nd2::TriND2;
pub use tet::TetND1;
pub use tet_nd2::TetND2;
