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

pub mod coo;
pub mod csr;
pub mod dense;
pub mod sparsity;
pub mod vector;

pub use coo::CooMatrix;
pub use csr::CsrMatrix;
pub use sparsity::SparsityPattern;
pub use vector::Vector;
