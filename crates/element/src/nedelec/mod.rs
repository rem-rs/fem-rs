//! Nedelec (first-kind) H(curl) elements.
//!
//! These elements provide **tangential continuity** across inter-element edges/faces and
//! are the canonical choice for discretising the curl-curl operator that appears in
//! Maxwell's equations.
//!
//! # DOF convention (order 2, 1:1 with MFEM `fe_nd.cpp`)
//! The order-2 elements (`TriND2`, `QuadND2`, `TetND2`) use MFEM's **nodal
//! point-value functionals**: `σ_i(Φ) = Φ(x_i)·t̂_i` at the DOF point
//! `x_i = FE::Nodes` (Gauss-Legendre points on the edges) along the fixed
//! reference tangent `t̂_i` (MFEM `tk` table).  The symmetric Gauss point sets
//! make the edge DOFs reflection-invariant: an edge reversal maps
//! `σ^rev_m = −σ_{k−1−m}`, so `HCurlSpace` pairs shared edges with a signed
//! anti-diagonal permutation (MFEM's own encoding).
//!
//! ND1 keeps the classic edge line-integral DOF `DOF_i = ∫_{e_i} Φ·t̂ ds`, and
//! the generic `*NDk` (k≥3) elements currently keep integral-moment edge DOFs
//! (pending the same nodal redesign — see round-14 D32 report).
//!
//! # Available elements
//! | Type       | Domain       | DOFs | Order |
//! |-----------|--------------|------|-------|
//! | [`TriND1`] | triangle     | 3    | 1     |
//! | [`TetND1`] | tetrahedron  | 6    | 1     |

pub mod hex;
pub mod hex_nd2;
pub mod hex_ndk;
pub mod prism;
pub mod pyramid;
pub mod quad;
pub mod quad_nd2;
pub mod quad_ndk;
pub mod tet;
pub mod tet_nd2;
pub mod tet_ndk;
pub mod tri;
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
pub use tri::TriND1;
pub use quad::QuadND1;
pub use hex::HexND1;
pub use tet::TetND1;
pub use tri_ndk::TriNDk;
