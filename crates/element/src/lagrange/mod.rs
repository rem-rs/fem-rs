//! Lagrange finite elements on standard reference domains.

pub mod seg;
pub mod tri;
pub mod tet;
pub mod quad;
pub mod hex;
pub mod factory;

pub use seg::{SegP1, SegP2, SegP3};
pub use tri::{TriP1, TriP2, TriP3};
pub use tet::{TetP1, TetP2, TetP3};
pub use quad::{QuadQ1, QuadQ2};
pub use hex::HexQ1;
pub use factory::{SegPk, TriPk, TetPk, QuadQk, HexQk, ref_elem, ElemType,
                  LagrangeSegment, LagrangeTriangle, LagrangeTetrahedron,
                  LagrangeQuad, LagrangeHex};
