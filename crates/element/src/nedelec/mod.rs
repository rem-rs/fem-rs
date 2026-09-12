//! Nedelec (first-kind) H(curl) elements.
//!
//! These elements provide **tangential continuity** across inter-element edges/faces and
//! are the canonical choice for discretising the curl-curl operator that appears in
//! Maxwell's equations.
//!
//! # DOF convention (all orders, 1:1 with MFEM `fem/fe/fe_nd.cpp`)
//!
//! Every `*NDk` element uses MFEM's **nodal point-value functionals**:
//! `σ_i(Φ) = Φ(x_i)·t̂_i` at the DOF point `x_i = FE::Nodes` — the
//! Gauss-Legendre edge points plus MFEM's barycentric GL points for the
//! interior/face DOFs — along the fixed reference tangent `t̂_i` (MFEM `tk`
//! table).  The symmetric Gauss point sets make the edge DOFs
//! reflection-invariant: an edge reversal maps `σ^rev_m = −σ_{k−1−m}`, so
//! `HCurlSpace` pairs shared edges with a signed anti-diagonal permutation
//! (MFEM's own encoding).
//!
//! `HexNDk` (all orders `k ≥ 1`, round-15 D36) follows the same rule: it is a
//! 1:1 port of MFEM `ND_HexahedronElement(p, GaussLobatto, GaussLegendre)` —
//! the element `ND_FECollection(p, dim)` builds by default — with
//! Gauss-Legendre open points along each component direction and GLL closed
//! points across it; see `hex_ndk.rs`.  The LOR-compatible open basis
//! (`ND_HexahedronElement(p, GaussLobatto, IntegratedGLL)`, the pair MFEM
//! documents for LOR discretizations) is available as
//! [`HexNDk::new_integrated_gll`]; see [`NdOpenBasis`].
//!
//! ND1 keeps the classic edge line-integral DOF `DOF_i = ∫_{e_i} Φ·t̂ ds`.
//!
//! `TriNDk`/`TetNDk` (round-16 D38) follow the same nodal rule at **every**
//! order: the reference functions are built exactly as MFEM's
//! `ND_TriangleElement`/`ND_TetrahedronElement` do (hierarchical Chebyshev `u`
//! basis + `Ti = T⁻¹`), and the DOFs are the `FE::Nodes` point-value
//! functionals with the `dof2tk` reference tangents — edge DOFs at the
//! Gauss-Legendre open points, interior/face DOFs at MFEM's barycentric GL
//! points.  Shared tet face DOF *pairs* are related between adjacent elements
//! by a full 2×2 change of basis (MFEM `ND_DofTransformation`), see
//! `fem_space::hcurl::FaceDofBlock`.
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
pub use hex_ndk::{HexNDk, NdOpenBasis};
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
