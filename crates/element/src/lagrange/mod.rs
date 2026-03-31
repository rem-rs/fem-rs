//! Lagrange finite elements on standard reference domains.

pub mod seg;
pub mod tri;
pub mod tet;
pub mod quad;
pub mod hex;

pub use seg::{SegP1, SegP2};
pub use tri::{TriP1, TriP2};
pub use tet::TetP1;
pub use quad::QuadQ1;
pub use hex::HexQ1;
