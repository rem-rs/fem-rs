//! # fem-element
//!
//! Reference finite elements, quadrature rules, and Lagrange basis functions.
//!
//! ## Traits
//! - [`ReferenceElement`] — scalar-valued elements (Lagrange H1/L2).
//! - [`VectorReferenceElement`] — vector-valued elements (H(curl), H(div)).
//!
//! ## Lagrange elements
//! | Type      | Reference domain | DOFs |
//! |-----------|-----------------|------|
//! | [`SegP1`] | [0,1]           | 2    |
//! | [`SegP2`] | [0,1]           | 3    |
//! | [`SegP3`] | [0,1]           | 4    |
//! | [`TriP1`] | unit triangle   | 3    |
//! | [`TriP2`] | unit triangle   | 6    |
//! | [`TriP3`] | unit triangle   | 10   |
//! | [`TetP1`] | unit tet        | 4    |
//! | [`TetP2`] | unit tet        | 10   |
//! | [`TetP3`] | unit tet        | 20   |
//! | [`QuadQ1`]| [-1,1]²         | 4    |
//! | [`HexQ1`] | [-1,1]³         | 8    |
//!
//! ## H(curl) Nedelec elements
//! | Type       | Reference domain | DOFs |
//! |------------|-----------------|------|
//! | [`TriND1`] | unit triangle   | 3    |
//! | [`QuadND1`]| reference quad  | 4    |
//! | [`TetND1`] | unit tet        | 6    |
//!
//! ## H(div) Raviart-Thomas elements
//! | Type        | Reference domain | DOFs |
//! |-------------|-----------------|------|
//! | [`TriRT0`]  | unit triangle   | 3    |
//! | [`TetRT0`]  | unit tet        | 4    |

pub mod reference;
pub mod quadrature;
pub mod lagrange;
pub mod nedelec;
pub mod raviart_thomas;
pub mod nurbs;
pub mod tri6_geom;
pub mod iga;
pub mod basis_cache;

pub use reference::{QuadratureRule, ReferenceElement, VectorReferenceElement};
pub use quadrature::{TriQuadRule, tri_rule_named};
pub use lagrange::{HexQ1, HexQ2, HexQ3, QuadQ1, QuadQ2, QuadQ3, QuadQ4, QuadP1, QuadP2, QuadP3, QuadP4, SegP1, SegP2, SegP3, SegP4, SegP5, SegP6, TetP1, TetP2, TetP3, TetP4, TetP5, TetP6, TriP1, TriP2, TriP3, TriP4, TriP5, TriP6, TriP7, TriP8, TriP9, TriP10,
                   LagrangeSegment, LagrangeTriangle, LagrangeTetrahedron, LagrangeQuad, LagrangeHex,
                   LagrangePrism, LagrangePyramid,
                   SegPk, TriPk, TetPk, QuadQk, HexQk, PrismPk, PyramidPk, ref_elem, ElemType};
pub use nedelec::{TriND1, TriND2, TriNDk, QuadND1, QuadND2, HexND1, HexND2, HexNDk, TetND1, TetND2, TetNDk};
pub use raviart_thomas::{TriRT0, TetRT0, TriRT1, TriRT2, TetRT1, TetRT2, QuadRT0, HexRT0, HexRT1, QuadRT1};
pub use nurbs::{KnotVector, BSplineBasis1D, NurbsPatch2D, NurbsPatch3D,
                NurbsMesh2D, NurbsMesh3D, greville_abscissae};

