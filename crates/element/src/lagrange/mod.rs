//! Lagrange finite elements on standard reference domains.

pub mod factory;
pub mod hex;
pub mod prism;
pub mod pyramid;
pub mod quad;
pub mod seg;
pub mod tet;
pub mod tri;

pub use factory::{
    ref_elem, vec_ref_elem, ElemType, H1TriPk, HexQk, LagrangeHex, LagrangePrism, LagrangePyramid,
    LagrangeQuad, LagrangeSegment, LagrangeTetrahedron, LagrangeTriangle, QuadL2GL, QuadPosQk, QuadQk, SegPk,
    TetPk, TriPk, VecFamily,
};
pub use hex::{HexQ1, HexQ2, HexQ3};
pub use prism::PrismPk;
pub use pyramid::PyramidPk;
pub use quad::{QuadP1, QuadP2, QuadP3, QuadP4, QuadQ1, QuadQ2, QuadQ3, QuadQ4};
pub use seg::{SegP1, SegP2, SegP3, SegP4, SegP5, SegP6};
pub use tet::{TetP1, TetP2, TetP3, TetP4, TetP5, TetP6};
pub use tri::{TriP1, TriP10, TriP2, TriP3, TriP4, TriP5, TriP6, TriP7, TriP8, TriP9};
