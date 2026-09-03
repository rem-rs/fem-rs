#![allow(clippy::needless_range_loop)]
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
//! 
//! | [`TetND1`] | unit tet        | 6    |
//!
//! ## H(div) Raviart-Thomas elements
//! | Type        | Reference domain | DOFs |
//! |-------------|-----------------|------|
//! | [`TriRT0`]  | unit triangle   | 3    |
//! | [`TetRT0`]  | unit tet        | 4    |

pub mod bernstein;
pub mod bezier_extraction;
pub mod brezzi_douglas_marini;
pub mod crouzeix_raviart;
pub mod iga;
pub mod lagrange;
pub mod nedelec;
pub mod nonconforming;
/// Legacy NURBS module — prefer `fem_element::iga` for new code.
/// The new `iga::KnotVector` (pure knot sequence) + `iga::BsplineBasis` supersede
/// the old degree-aware `nurbs::KnotVector`. Types like `NurbsPatch2D`,
/// `NurbsMesh2D`, etc. remain in `nurbs` but are re-exported from `iga` for
/// migration convenience.
pub mod nurbs;
pub mod quadrature;
pub mod raviart_thomas;
pub mod reference;
pub mod serendipity;

pub use bernstein::{bernstein_dders, bernstein_ders, bernstein_vals};
pub use brezzi_douglas_marini::{HexBDMk, QuadBDMk, TetBDMk, TriBDMk};
pub use crouzeix_raviart::{
    cr1_basis, cr1_grad, cr1_tet_basis, cr1_tet_grad, cr2_tet_basis, cr2_tet_grad, cr2_tri_basis,
    cr2_tri_grad, CrTet1, CrTet2, CrTri1, CrTri2, CrouzeixRaviart1, CrouzeixRaviartVec1,
};
pub use lagrange::{
    ref_elem, vec_ref_elem, ElemType, H1TriPk, HexQ1, HexQ2, HexQ3, HexQk, LagrangeHex, LagrangePrism,
    LagrangePyramid, LagrangeQuad, LagrangeSegment, LagrangeTetrahedron, LagrangeTriangle, PrismPk,
    PyramidPk, QuadL2GL, QuadP1, QuadP2, QuadP3, QuadP4, QuadQ1, QuadQ2, QuadQ3, QuadQ4, QuadQk,
    SegP1, SegP2, SegP3, SegP4, SegP5, SegP6, SegPk, TetP1, TetP2, TetP3, TetP4, TetP5, TetP6,
    TetPk, TriP1, TriP10, TriP2, TriP3, TriP4, TriP5, TriP6, TriP7, TriP8, TriP9, TriPk, VecFamily,
};
pub use nedelec::{
    HexND2, HexNDk, PrismND1, PrismNDk, PyraND1, PyraNDk, QuadND2, QuadNDk,
    TetND2, TetNDk, TriND2, TriNDk,
};
pub use nonconforming::{Q1RotRef, QuadQ1Rot, QuadQ1RotVec};
pub use raviart_thomas::{
    HexRT1, HexRTk, PrismRT0, PrismRTk, PyraRT0, PyraRTk, QuadRT1, QuadRTk,
    TetRT1, TetRT2, TetRTk, TriRT1, TriRT2, TriRTk,
};
pub use reference::{QuadratureRule, ReferenceElement, VectorReferenceElement};
pub use serendipity::{HexSerendipityPk, QuadSerendipityPk};
// Use fem_element::iga::* instead (migration target)
