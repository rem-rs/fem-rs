//! # fem-element
//!
//! Reference finite elements, quadrature rules, and Lagrange basis functions.
//!
//! ## Trait
//! [`ReferenceElement`] — implemented by every concrete element type.
//!
//! ## Lagrange elements
//! | Type      | Reference domain | DOFs |
//! |-----------|-----------------|------|
//! | [`SegP1`] | [0,1]           | 2    |
//! | [`SegP2`] | [0,1]           | 3    |
//! | [`TriP1`] | unit triangle   | 3    |
//! | [`TriP2`] | unit triangle   | 6    |
//! | [`TetP1`] | unit tet        | 4    |
//! | [`QuadQ1`]| [-1,1]²         | 4    |
//! | [`HexQ1`] | [-1,1]³         | 8    |

pub mod reference;
pub mod quadrature;
pub mod lagrange;

pub use reference::{QuadratureRule, ReferenceElement};
pub use lagrange::{HexQ1, QuadQ1, SegP1, SegP2, TetP1, TriP1, TriP2};
