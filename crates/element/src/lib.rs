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

pub use reference::{QuadratureRule, ReferenceElement, VectorReferenceElement};
pub use lagrange::{HexQ1, QuadQ1, QuadQ2, SegP1, SegP2, SegP3, TetP1, TetP2, TetP3, TriP1, TriP2, TriP3};
pub use nedelec::{TriND1, QuadND1, QuadND2, HexND1, HexND2, TetND1, TriND2, TetND2};
pub use raviart_thomas::{TriRT0, TetRT0, TriRT1, TetRT1};

