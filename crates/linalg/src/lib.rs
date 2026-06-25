//! # fem-linalg
//!
//! Sparse and dense linear algebra for fem-rs.
//!
//! ## Modules
//! - [`csr`]      â€?`CsrMatrix<T>`: CSR sparse matrix with SpMV and BC helpers
//! - [`coo`]      â€?`CooMatrix<T>`: coordinate-format accumulator â†?converts to CSR
//! - [`vector`]   â€?`Vector<T>`: heap vector with axpy, dot, norm
//! - [`sparsity`] â€?`SparsityPattern`: non-zero structure built from DOF connectivity
//! - [`dense`]    â€?small dense operations (LU factorisation, matmat) for coarse-grid solves
//! - [`block`]    â€?`BlockMatrix` / `BlockVector` for mixed / saddle-point problems
//! - [`hmatrix`]  â€?H-matrix infrastructure: cluster tree, bounding boxes, block cluster tree
//!
//! ## Feature flags
//!
//! - **`parallel`** â€?Rayon-parallel `CsrMatrix::spmv` / `spmv_add` when the row count
//!   meets `spmv_parallel_min_rows()` (adaptive default: `128` on a single worker
//!   down to `16` on 8+ workers; override env
//!   `FEM_LINALG_SPMV_PARALLEL_MIN_ROWS`). For `f64`, serial and parallel paths use an
//!   8-way unrolled dot over each rowâ€™s nonzeros.
//!
//! ## Re-exports from `linlvo`
//! - `BlrMatrix`, `BlrBlock` â€?Block Low-Rank compression for direct solvers

pub mod complex_csr;
pub mod coo;
pub mod csr;
pub mod dense;
pub mod sparsity;
pub mod vector;
pub mod block;
pub mod pool;
pub mod hmatrix;

pub use coo::CooMatrix;
pub use csr::CsrMatrix;
pub use csr::{spadd, csr_spmm};
#[cfg(feature = "parallel")]
pub use csr::{csr_spmm_parallel, spadd_parallel, spmv_parallel_min_rows, FEM_LINALG_SPMV_PARALLEL_MIN_ROWS};
pub use sparsity::SparsityPattern;
pub use vector::Vector;
pub use block::{BlockMatrix, BlockVector};
pub use dense::DenseTensor;
pub use pool::{CooVectorPool, PooledCooVectors};

// Re-exports from linlvo for Block Low-Rank compression
#[cfg(feature = "direct")]
#[doc(inline)]
pub use linlvo::direct::{BlrBlock, BlrMatrix, compress_block, compress_block_adaptive};
