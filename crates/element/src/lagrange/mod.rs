//! Lagrange finite elements on standard reference domains.

pub mod factory;
pub mod hex;
pub mod legacy;
pub mod prism;
pub mod pyramid;
pub mod pyramid_fuentes;
pub mod pyramid_l2;
pub mod quad;
pub mod seg;
pub mod tet;
pub mod tri;

pub use factory::{
    ref_elem, vec_ref_elem, ElemType, H1TetPk, H1TriPk, HexL2GL, HexQk, LagrangeHex, LagrangePrism,
    LagrangePyramid, LagrangeQuad, LagrangeSegment, LagrangeTetrahedron, LagrangeTriangle,
    QuadL2GL, QuadPosQk, QuadQk, SegPk, TetL2GL, TetPk, TriL2GL, TriPk, VecFamily,
};
pub use hex::{HexQ1, HexQ2, HexQ3};
pub use prism::{h1_prism_slots, H1PrismPk, H1PrismSlot, PrismPk, PRISM_EDGES};
pub use pyramid::{
    h1_pyramid_element, h1_pyramid_slot_labels, H1PyramidPk, PyramidBasisType, PyramidPk,
};
pub use pyramid_fuentes::{fuentes_pyramid_n_dofs, h1_fuentes_pyramid_nodes, H1FuentesPyramidPk};
pub use pyramid_l2::{l2_fuentes_pyramid_n_dofs, L2FuentesPyramidPk};
pub use quad::{QuadP1, QuadP2, QuadP3, QuadP4, QuadQ1, QuadQ2, QuadQ3, QuadQ4};
pub use seg::{SegP1, SegP2, SegP3, SegP4, SegP5, SegP6};
pub use tet::{TetP1, TetP2, TetP3, TetP4, TetP5, TetP6};
pub use tri::{TriP1, TriP10, TriP2, TriP3, TriP4, TriP5, TriP6, TriP7, TriP8, TriP9};
