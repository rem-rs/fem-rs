//! # fem-linalg
//!
//! Sparse and dense linear algebra for fem-rs.
//!
//! ## Modules
//! - [`csr`]      �?`CsrMatrix<T>`: CSR sparse matrix with SpMV and BC helpers
//! - [`coo`]      �?`CooMatrix<T>`: coordinate-format accumulator �?converts to CSR
//! - [`vector`]   �?`Vector<T>`: heap vector with axpy, dot, norm
//! - [`sparsity`] �?`SparsityPattern`: non-zero structure built from DOF connectivity
//! - [`dense`]    �?small dense operations (LU factorisation, matmat) for coarse-grid solves
//! - [`block`]    �?`BlockMatrix` / `BlockVector` for mixed / saddle-point problems
//! - [`hmatrix`]  �?H-matrix infrastructure: cluster tree, bounding boxes, block cluster tree
//!
//! ## Feature flags
//!
//! - **`parallel`** �?Rayon-parallel `CsrMatrix::spmv` / `spmv_add` when the row count
//!   meets `spmv_parallel_min_rows()` (adaptive default: `128` on a single worker
//!   down to `16` on 8+ workers; override env
//!   `FEM_LINALG_SPMV_PARALLEL_MIN_ROWS`). For `f64`, serial and parallel paths use an
//!   8-way unrolled dot over each row’s nonzeros.
//!
//! ## Re-exports from `linlvo`
//! - `BlrMatrix`, `BlrBlock` �?Block Low-Rank compression for direct solvers

pub mod complex_csr;
pub mod coo;
pub mod csr;
pub mod dense;
pub mod sparsity;
pub mod vector;
pub mod block;
pub mod pool;
pub mod hmatrix;

#[cfg(feature = "direct")]
pub mod solver_types;

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

#[cfg(feature = "direct")]
pub use solver_types::{SolverConfig, SolverError, SolveResult, PrintLevel, fem_to_linlvo_csr, into_result};

// ─── Kani proof harnesses (formal verification) ────────────────────────
// These are only compiled by the Kani verifier (`cargo kani`).
// Normal `cargo build` / `cargo test` skips them entirely.
#[cfg(feature = "kani")]
#[allow(unexpected_cfgs)]
mod kani_proofs {
    use crate::CsrMatrix;
    use crate::CsrMatrix;

    /// Prove that SpMV never panics for a well-formed CSR matrix.
    #[kani::proof]
    fn csr_spmv_no_panic() {
        let n: usize = kani::any();
        kani::assume(n > 0 && n <= 50);
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i];
            row_ptr[i + 1] += 1; // diagonal
            col_idx.push(i as u32);
            values.push(2.0);
        }
        let a = CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values };
        let x: Vec<f64> = (0..n).map(|_| { let v: f64 = kani::any(); kani::assume(v.is_finite()); v }).collect();
        let mut y = vec![0.0; n];
        a.spmv(&x, &mut y);
        for v in &y { assert!(v.is_finite()); }
    }

    /// Prove that `row_ptr` is monotonically increasing.
    #[kani::proof]
    fn csr_row_ptr_monotonic() {
        let n: usize = kani::any();
        kani::assume(n > 0 && n <= 50);
        let nnz: usize = kani::any();
        kani::assume(nnz <= n);
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            let d: usize = kani::any();
            kani::assume(d <= 1);
            row_ptr[i + 1] = row_ptr[i] + d;
        }
        row_ptr[n] = nnz;
        for i in 0..n { assert!(row_ptr[i] <= row_ptr[i + 1]); }
    }
}
