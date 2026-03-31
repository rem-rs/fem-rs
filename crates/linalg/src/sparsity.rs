use crate::csr::CsrMatrix;

/// Non-zero structure of a sparse matrix, built from DOF connectivity.
///
/// Used by `Assembler::new` to build the pattern once; subsequent assembly
/// loops only fill values without re-sorting.
#[derive(Debug, Clone)]
pub struct SparsityPattern {
    pub nrows: usize,
    pub ncols: usize,
    /// `row_ptr[i]..row_ptr[i+1]` indexes the entries for row `i` in `col_idx`.
    pub row_ptr: Vec<usize>,
    /// Sorted, deduplicated column indices per row.
    pub col_idx: Vec<u32>,
}

impl SparsityPattern {
    /// Build from element→DOF connectivity.
    ///
    /// `n_dofs`: total number of DOFs (rows == cols for symmetric problem).
    /// `element_dofs`: slice-of-slices; element_dofs[e] = DOF indices for element e.
    ///
    /// Each element contributes a full `k×k` block (all pairs from its DOFs).
    /// The pattern is symmetric: if (i,j) is added, (j,i) is too.
    pub fn from_element_dofs(n_dofs: usize, element_dofs: &[&[u32]]) -> Self {
        let mut pairs: Vec<(u32, u32)> = Vec::new();

        for &dofs in element_dofs {
            for &di in dofs {
                for &dj in dofs {
                    pairs.push((di, dj));
                    // Symmetric: also add (dj, di) — redundant only when di == dj,
                    // but dedup handles that.
                    if di != dj {
                        pairs.push((dj, di));
                    }
                }
            }
        }

        Self::build_from_pairs(n_dofs, n_dofs, pairs)
    }

    /// Build from a list of (row, col) pairs (not necessarily sorted or unique).
    pub fn from_pairs(nrows: usize, ncols: usize, pairs: &[(u32, u32)]) -> Self {
        Self::build_from_pairs(nrows, ncols, pairs.to_vec())
    }

    /// Shared internal constructor: sort, dedup, then build CSR structure.
    fn build_from_pairs(nrows: usize, ncols: usize, mut pairs: Vec<(u32, u32)>) -> Self {
        pairs.sort_unstable();
        pairs.dedup();

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_idx: Vec<u32> = Vec::with_capacity(pairs.len());

        for &(r, c) in &pairs {
            row_ptr[r as usize + 1] += 1;
            col_idx.push(c);
        }

        // Prefix sum to get actual row_ptr offsets
        for i in 1..=nrows {
            row_ptr[i] += row_ptr[i - 1];
        }

        Self { nrows, ncols, row_ptr, col_idx }
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize { self.col_idx.len() }

    /// Create a zero-initialized `CsrMatrix<T>` with this sparsity pattern.
    pub fn to_csr<T: Copy + Default>(&self) -> CsrMatrix<T> {
        CsrMatrix {
            nrows:   self.nrows,
            ncols:   self.ncols,
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_idx.clone(),
            values:  vec![T::default(); self.col_idx.len()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangular elements sharing an edge:
    ///
    ///   DOFs: 0, 1, 2, 3
    ///   elem 0: [0, 1, 2]
    ///   elem 1: [1, 2, 3]
    #[test]
    fn from_element_dofs_basic() {
        let e0: &[u32] = &[0, 1, 2];
        let e1: &[u32] = &[1, 2, 3];
        let sp = SparsityPattern::from_element_dofs(4, &[e0, e1]);

        assert_eq!(sp.nrows, 4);
        assert_eq!(sp.ncols, 4);

        // Every entry should be within bounds.
        for &c in &sp.col_idx {
            assert!(c < 4, "col index {c} out of range");
        }

        // row_ptr must be non-decreasing and end at nnz
        for i in 0..sp.nrows {
            assert!(sp.row_ptr[i] <= sp.row_ptr[i + 1]);
        }
        assert_eq!(sp.row_ptr[sp.nrows], sp.nnz());

        // The diagonal must be present for each DOF that appears in any element.
        for row in 0..sp.nrows {
            let start = sp.row_ptr[row];
            let end   = sp.row_ptr[row + 1];
            let has_diag = sp.col_idx[start..end].contains(&(row as u32));
            assert!(has_diag, "row {row} missing diagonal");
        }

        // Symmetry: if (i,j) is present then (j,i) must be too.
        for row in 0..sp.nrows {
            let start = sp.row_ptr[row];
            let end   = sp.row_ptr[row + 1];
            for &col in &sp.col_idx[start..end] {
                let col = col as usize;
                let cs = sp.row_ptr[col];
                let ce = sp.row_ptr[col + 1];
                let sym = sp.col_idx[cs..ce].contains(&(row as u32));
                assert!(sym, "({row},{col}) present but ({col},{row}) missing");
            }
        }
    }

    #[test]
    fn from_pairs_dedup_and_sort() {
        // Duplicate pairs must be deduplicated.
        let pairs = [(0u32, 1u32), (1, 0), (0, 1), (1, 1)];
        let sp = SparsityPattern::from_pairs(2, 2, &pairs);
        assert_eq!(sp.nnz(), 3, "expected 3 unique pairs: (0,1),(1,0),(1,1)");
    }

    #[test]
    fn to_csr_zero_values() {
        let e0: &[u32] = &[0, 1];
        let sp = SparsityPattern::from_element_dofs(2, &[e0]);
        let csr = sp.to_csr::<f64>();
        // All values must be zero
        for &v in &csr.values {
            assert_eq!(v, 0.0);
        }
        assert_eq!(csr.nnz(), sp.nnz());
    }

    #[test]
    fn single_element_full_block() {
        // One element with DOFs [0,1,2] → 3×3 full block (symmetric).
        let e: &[u32] = &[0, 1, 2];
        let sp = SparsityPattern::from_element_dofs(3, &[e]);
        // All 9 entries of the 3×3 block should be stored.
        assert_eq!(sp.nnz(), 9);
    }
}
