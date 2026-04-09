//! # fem-linalg
//!
//! Sparse and dense linear algebra for fem-rs.
//!
//! ## Modules
//! - [`csr`]      — `CsrMatrix<T>`: CSR sparse matrix with SpMV and BC helpers
//! - [`coo`]      — `CooMatrix<T>`: coordinate-format accumulator → converts to CSR
//! - [`vector`]   — `Vector<T>`: heap vector with axpy, dot, norm
//! - [`sparsity`] — `SparsityPattern`: non-zero structure built from DOF connectivity
//! - [`dense`]    — small dense operations (LU factorisation, matmat) for coarse-grid solves
//! - [`block`]    — `BlockMatrix` / `BlockVector` for mixed / saddle-point problems
//!
//! ## Re-exports from `linger`
//! - `BlrMatrix`, `BlrBlock` — Block Low-Rank compression for direct solvers

pub mod coo;
pub mod csr;
pub mod dense;
pub mod sparsity;
pub mod vector;
pub mod block;

pub use coo::CooMatrix;
pub use csr::CsrMatrix;
pub use csr::spadd;
pub use sparsity::SparsityPattern;
pub use vector::Vector;
pub use block::{BlockMatrix, BlockVector};
pub use dense::DenseTensor;

// Re-exports from linger for Block Low-Rank compression
#[cfg(feature = "direct")]
#[doc(inline)]
pub use linger::direct::{BlrBlock, BlrMatrix, compress_block, compress_block_adaptive};
