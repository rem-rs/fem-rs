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

    /// Build a block vector from explicit block offsets.
    ///
    /// Equivalent to MFEM's `BlockVector(const Array<int> &offsets)` — the
    /// joule/maxwell miniapps use it to lay out several FE spaces in one
    /// contiguous buffer (`Array<int> true_offset(7)` in `joule.cpp`):
    /// block `i` then occupies `offsets[i] .. offsets[i + 1]`.
    ///
    /// `offsets` must start at zero and be non-decreasing.  The block sizes
    /// implied by the offsets normally come from
    /// [`BlockFESpace`](fem_space::BlockFESpace)::`global_dof_offset`, but the
    /// coupling is deliberately left to the caller (this crate is a level
    /// below the FE spaces).
    pub fn from_offsets(offsets: &[usize]) -> Self {
        assert!(
            !offsets.is_empty(),
            "BlockVector::from_offsets: need at least one offset (got an empty slice)"
        );
        assert_eq!(
            offsets[0], 0,
            "BlockVector::from_offsets: offsets[0] must be 0, got {}", offsets[0]
        );
        for (i, w) in offsets.windows(2).enumerate() {
            assert!(
                w[1] >= w[0],
                "BlockVector::from_offsets: offsets must be non-decreasing \
                 (offsets[{}] = {} > offsets[{}] = {})", i, w[0], i + 1, w[1]
            );
        }
        BlockVector {
            data: vec![0.0; *offsets.last().unwrap()],
            offsets: offsets.to_vec(),
        }
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

    /// Immutable views of every block at once.
    ///
    /// The returned slices alias the block vector's storage (zero copy); the
    /// set of views is the whole buffer, so they are disjoint by construction.
    pub fn views(&self) -> Vec<&[f64]> {
        self.offsets
            .windows(2)
            .map(|w| &self.data[w[0]..w[1]])
            .collect()
    }

    /// Mutable views of every block at once — the multi-field counterpart of
    /// [`BlockVector::block_mut`].
    ///
    /// This is the Rust equivalent of the MFEM pattern in `joule.cpp`, where
    /// six `GridFunction`s are attached to one `BlockVector` (`MakeRef`) and
    /// written through simultaneously:
    /// ```text
    /// T_gf.MakeRef(&L2FESpace,    F, true_offset[0]);
    /// F_gf.MakeRef(&HDivFESpace,  F, true_offset[1]);
    /// ...
    /// ```
    /// Rather than handing out raw pointers (as MFEM's `MakeRef` does), the
    /// borrow checker is satisfied by splitting the single `&mut self` into
    /// `n_blocks()` disjoint `&mut [f64]` views — no aliasing, no `unsafe`.
    /// Writes through a view are visible in the underlying vector, and blocks
    /// are independent of one another.
    pub fn views_mut(&mut self) -> Vec<&mut [f64]> {
        let offsets = self.offsets.clone();
        let mut rest: &mut [f64] = &mut self.data;
        let mut out = Vec::with_capacity(offsets.len() - 1);
        for w in offsets.windows(2) {
            let (head, tail) = rest.split_at_mut(w[1] - w[0]);
            out.push(head);
            rest = tail;
        }
        out
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

    #[test]
    fn from_offsets_matches_sizes() {
        // MFEM's `BlockVector(Array<int>&)`: offsets {0, 3, 5, 5} == sizes {3, 2, 0}.
        let from_off = BlockVector::from_offsets(&[0, 3, 5, 5]);
        let from_sizes = BlockVector::new(vec![3, 2, 0]);
        assert_eq!(from_off.len(), from_sizes.len());
        assert_eq!(from_off.n_blocks(), from_sizes.n_blocks());
        for i in 0..from_sizes.n_blocks() {
            assert_eq!(from_off.offset(i), from_sizes.offset(i));
            assert_eq!(from_off.block_size(i), from_sizes.block_size(i));
        }
        assert_eq!(from_off.as_slice(), from_sizes.as_slice());
    }

    #[test]
    fn views_mut_write_through_and_disjoint() {
        let mut bv = BlockVector::new(vec![2, 3, 1]);

        // Write every block through its view only.
        {
            let mut views = bv.views_mut();
            assert_eq!(views.len(), 3);
            views[0].copy_from_slice(&[1.0, 2.0]);
            views[1].copy_from_slice(&[3.0, 4.0, 5.0]);
            views[2].copy_from_slice(&[6.0]);
        }

        // The writes are visible in the original vector ...
        assert_eq!(bv.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(bv.block(0), &[1.0, 2.0]);
        assert_eq!(bv.block(1), &[3.0, 4.0, 5.0]);
        assert_eq!(bv.block(2), &[6.0]);

        // ... and bit-for-bit identical to a per-block copy of the same data.
        let mut copy = BlockVector::new(vec![2, 3, 1]);
        copy.block_mut(0).copy_from_slice(&[1.0, 2.0]);
        copy.block_mut(1).copy_from_slice(&[3.0, 4.0, 5.0]);
        copy.block_mut(2).copy_from_slice(&[6.0]);
        for (a, b) in bv.as_slice().iter().zip(copy.as_slice().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        // Mutating block 1 must not disturb the other blocks.
        let before = (bv.block(0).to_vec(), bv.block(2).to_vec());
        for (k, v) in bv.views_mut()[1].iter_mut().enumerate() {
            *v = 100.0 + k as f64;
        }
        assert_eq!(bv.block(1), &[100.0, 101.0, 102.0]);
        assert_eq!(bv.block(0), before.0.as_slice());
        assert_eq!(bv.block(2), before.1.as_slice());
    }

    #[test]
    fn views_and_views_mut_partition_the_buffer() {
        let mut bv = BlockVector::new(vec![4, 0, 2]);
        let sizes: Vec<usize> = (0..bv.n_blocks()).map(|i| bv.block_size(i)).collect();
        {
            let imm: Vec<&[f64]> = bv.views();
            assert_eq!(imm.len(), 3);
            assert_eq!(imm.iter().map(|v| v.len()).sum::<usize>(), bv.len());
            assert!(imm[1].is_empty());
        }

        let mut off = 0usize;
        for (i, v) in bv.views_mut().iter_mut().enumerate() {
            assert_eq!(v.len(), sizes[i]);
            v.fill(i as f64);
            off += v.len();
        }
        assert_eq!(off, bv.len());
        assert_eq!(bv.as_slice(), &[0.0, 0.0, 0.0, 0.0, 2.0, 2.0]);
    }
}
