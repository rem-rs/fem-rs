//! Block-structured matrices and vectors for mixed / saddle-point problems.
//!
//! A `BlockVector` splits a contiguous coefficient vector into named blocks;
//! a `BlockMatrix` holds a 2-D array of `CsrMatrix` sub-blocks that together
//! form the global system matrix.
//!
//! # Typical use
//! For a Stokes system (u ∈ H¹^d, p ∈ L²) the block layout is:
//! ```text
//! [ A   B^T ] [ u ]   [ f ]
//! [ B   0   ] [ p ] = [ g ]
//! ```
//!
//! ```rust,ignore
//! let mut bm = BlockMatrix::new(vec![n_u, n_p]);
//! bm.set(0, 0, a_uu);
//! bm.set(0, 1, b_t);
//! bm.set(1, 0, b);
//! let mut bv = BlockVector::new(vec![n_u, n_p]);
//! bv.block_mut(0).copy_from_slice(&rhs_u);
//! ```

use crate::CsrMatrix;

// ─── BlockVector ─────────────────────────────────────────────────────────────

/// A dense vector partitioned into named contiguous blocks.
///
/// Block `i` occupies indices `offsets[i] .. offsets[i+1]` in the underlying
/// flat storage.
#[derive(Debug, Clone)]
pub struct BlockVector {
    data:    Vec<f64>,
    offsets: Vec<usize>,
}

impl BlockVector {
    /// Create a zero block vector with the given block sizes.
    pub fn new(sizes: Vec<usize>) -> Self {
        let total: usize = sizes.iter().sum();
        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        offsets.push(0);
        for s in &sizes {
            offsets.push(offsets.last().unwrap() + s);
        }
        BlockVector { data: vec![0.0; total], offsets }
    }

    /// Number of blocks.
    pub fn n_blocks(&self) -> usize { self.offsets.len() - 1 }

    /// Total length (sum of all block sizes).
    pub fn len(&self) -> usize { self.data.len() }

    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Immutable view of block `i`.
    pub fn block(&self, i: usize) -> &[f64] {
        &self.data[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Mutable view of block `i`.
    pub fn block_mut(&mut self, i: usize) -> &mut [f64] {
        let (lo, hi) = (self.offsets[i], self.offsets[i + 1]);
        &mut self.data[lo..hi]
    }

    /// Flat immutable slice of the entire vector.
    pub fn as_slice(&self) -> &[f64] { &self.data }

    /// Flat mutable slice of the entire vector.
    pub fn as_slice_mut(&mut self) -> &mut [f64] { &mut self.data }

    /// Global byte offset for the start of block `i`.
    pub fn offset(&self, i: usize) -> usize { self.offsets[i] }

    /// Size (number of DOFs) of block `i`.
    pub fn block_size(&self, i: usize) -> usize {
        self.offsets[i + 1] - self.offsets[i]
    }
}

// ─── BlockMatrix ─────────────────────────────────────────────────────────────

/// A 2-D array of sparse sub-blocks forming a global system matrix.
///
/// An `n_blocks × n_blocks` matrix where each entry is an
/// `Option<CsrMatrix<f64>>`.  Entries can be `None` (treated as zero).
///
/// The sizes of the row/column block partitions must be consistent with the
/// actual sub-block dimensions when calling [`BlockMatrix::spmv`].
#[derive(Debug)]
pub struct BlockMatrix {
    /// Row / column block sizes (must match).
    pub row_sizes: Vec<usize>,
    pub col_sizes: Vec<usize>,
    /// Flat row-major storage of `Option<CsrMatrix>` entries.
    blocks: Vec<Option<CsrMatrix<f64>>>,
    n_row_blocks: usize,
    n_col_blocks: usize,
}

impl BlockMatrix {
    /// Create a zero block matrix with given row and column block sizes.
    pub fn new(row_sizes: Vec<usize>, col_sizes: Vec<usize>) -> Self {
        let nr = row_sizes.len();
        let nc = col_sizes.len();
        BlockMatrix {
            row_sizes,
            col_sizes,
            blocks: vec_none(nr * nc),
            n_row_blocks: nr,
            n_col_blocks: nc,
        }
    }

    /// Create a square block matrix (same sizes for rows and columns).
    pub fn new_square(sizes: Vec<usize>) -> Self {
        Self::new(sizes.clone(), sizes)
    }

    /// Set sub-block `(i, j)`.
    pub fn set(&mut self, i: usize, j: usize, block: CsrMatrix<f64>) {
        assert_eq!(block.nrows, self.row_sizes[i],
            "row block {i}: expected {} rows, got {}", self.row_sizes[i], block.nrows);
        assert_eq!(block.ncols, self.col_sizes[j],
            "col block {j}: expected {} cols, got {}", self.col_sizes[j], block.ncols);
        self.blocks[i * self.n_col_blocks + j] = Some(block);
    }

    /// Immutable reference to sub-block `(i, j)`, if set.
    pub fn get(&self, i: usize, j: usize) -> Option<&CsrMatrix<f64>> {
        self.blocks[i * self.n_col_blocks + j].as_ref()
    }

    /// Number of row blocks.
    pub fn n_row_blocks(&self) -> usize { self.n_row_blocks }

    /// Number of column blocks.
    pub fn n_col_blocks(&self) -> usize { self.n_col_blocks }

    /// Total number of rows (sum of row block sizes).
    pub fn total_rows(&self) -> usize { self.row_sizes.iter().sum() }

    /// Total number of columns (sum of col block sizes).
    pub fn total_cols(&self) -> usize { self.col_sizes.iter().sum() }

    /// Compute `y = A * x` where `x` and `y` are partitioned `BlockVector`s.
    ///
    /// `x` must have the same block sizes as [`BlockMatrix::col_sizes`];
    /// `y` must have the same block sizes as [`BlockMatrix::row_sizes`].
    pub fn spmv(&self, x: &BlockVector, y: &mut BlockVector) {
        // Zero y
        for v in y.as_slice_mut() { *v = 0.0; }

        for i in 0..self.n_row_blocks {
            let y_blk = y.block_mut(i);
            for j in 0..self.n_col_blocks {
                if let Some(a) = self.get(i, j) {
                    let x_blk = x.block(j);
                    let mut tmp = vec![0.0_f64; a.nrows];
                    a.spmv(x_blk, &mut tmp);
                    for (yi, &ti) in y_blk.iter_mut().zip(tmp.iter()) {
                        *yi += ti;
                    }
                }
            }
        }
    }

    /// Compute the 2-norm of `Ax - b`.
    pub fn residual_norm(&self, x: &BlockVector, b: &BlockVector) -> f64 {
        let mut ax = BlockVector::new(self.row_sizes.clone());
        self.spmv(x, &mut ax);
        let mut sum = 0.0;
        for (ai, bi) in ax.as_slice().iter().zip(b.as_slice().iter()) {
            let d = ai - bi;
            sum += d * d;
        }
        sum.sqrt()
    }
}

fn vec_none<T>(n: usize) -> Vec<Option<T>> {
    (0..n).map(|_| None).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooMatrix;

    fn diag2(n: usize, v: f64) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n { coo.add(i, i, v); }
        coo.into_csr()
    }

    #[test]
    fn block_vector_partition() {
        let mut bv = BlockVector::new(vec![3, 2]);
        bv.block_mut(0).copy_from_slice(&[1.0, 2.0, 3.0]);
        bv.block_mut(1).copy_from_slice(&[4.0, 5.0]);
        assert_eq!(bv.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(bv.block(0), &[1.0, 2.0, 3.0]);
        assert_eq!(bv.block(1), &[4.0, 5.0]);
    }

    #[test]
    fn block_matrix_diagonal_spmv() {
        let mut bm = BlockMatrix::new_square(vec![2, 3]);
        bm.set(0, 0, diag2(2, 2.0));
        bm.set(1, 1, diag2(3, 3.0));

        let mut x = BlockVector::new(vec![2, 3]);
        x.block_mut(0).copy_from_slice(&[1.0, 2.0]);
        x.block_mut(1).copy_from_slice(&[1.0, 1.0, 1.0]);

        let mut y = BlockVector::new(vec![2, 3]);
        bm.spmv(&x, &mut y);

        assert_eq!(y.block(0), &[2.0, 4.0]);
        assert_eq!(y.block(1), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn block_matrix_off_diagonal_spmv() {
        // [ 2I  I ] [ 1 ]   [ 3 ]
        // [ 0   I ] [ 1 ] = [ 1 ]
        let mut bm = BlockMatrix::new_square(vec![2, 2]);
        bm.set(0, 0, diag2(2, 2.0));
        bm.set(0, 1, diag2(2, 1.0));
        bm.set(1, 1, diag2(2, 1.0));

        let mut x = BlockVector::new(vec![2, 2]);
        x.block_mut(0).copy_from_slice(&[1.0, 1.0]);
        x.block_mut(1).copy_from_slice(&[1.0, 1.0]);

        let mut y = BlockVector::new(vec![2, 2]);
        bm.spmv(&x, &mut y);

        assert_eq!(y.block(0), &[3.0, 3.0]);
        assert_eq!(y.block(1), &[1.0, 1.0]);
    }

    #[test]
    fn block_vector_len_and_offset() {
        let bv = BlockVector::new(vec![4, 6]);
        assert_eq!(bv.len(), 10);
        assert_eq!(bv.offset(0), 0);
        assert_eq!(bv.offset(1), 4);
        assert_eq!(bv.block_size(0), 4);
        assert_eq!(bv.block_size(1), 6);
    }
}
