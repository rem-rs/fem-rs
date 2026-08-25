use fem_core::Scalar;
use crate::csr::CsrMatrix;

/// Coordinate-format sparse matrix (accumulates (row, col, value) triples).
///
/// Used during FEM assembly to collect element contributions before converting
/// to CSR with `into_csr()`.  Duplicate `(row, col)` entries are summed.
#[derive(Debug, Clone)]
pub struct CooMatrix<T> {
    pub nrows: usize,
    pub ncols: usize,
    rows: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<T>,
}

impl<T: Scalar> CooMatrix<T> {
    /// Create an empty sparse matrix of shape `nrows × ncols`.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, rows: Vec::new(), cols: Vec::new(), vals: Vec::new() }
    }

    /// Reserve capacity for `n` non-zeros (performance hint).
    pub fn reserve(&mut self, n: usize) {
        self.rows.reserve(n);
        self.cols.reserve(n);
        self.vals.reserve(n);
    }

    /// Clear all entries without releasing allocated memory (for reuse).
    #[inline]
    pub fn clear(&mut self) {
        self.rows.clear();
        self.cols.clear();
        self.vals.clear();
    }

    /// Add a scalar contribution `val` at position `(row, col)`.
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, val: T) {
        self.rows.push(row as u32);
        self.cols.push(col as u32);
        self.vals.push(val);
    }

    /// Add a dense `k × k` element matrix at the DOF index pairs in `dofs`.
    ///
    /// `k_elem` is row-major: `k_elem[i * k + j]` is the (i,j) entry.
    /// Uses batched Vec extension to minimise capacity checks.
    pub fn add_element_matrix(&mut self, dofs: &[usize], k_elem: &[T]) {
        let k = dofs.len();
        debug_assert_eq!(k_elem.len(), k * k);
        let n_entries = k * k;
        self.rows.reserve(n_entries);
        self.cols.reserve(n_entries);
        self.vals.reserve(n_entries);
        for i in 0..k {
            let row = dofs[i] as u32;
            for j in 0..k {
                self.rows.push(row);
                self.cols.push(dofs[j] as u32);
                self.vals.push(k_elem[i * k + j]);
            }
        }
    }

    /// Add a dense element load vector `f_elem` (length `k`) to positions `dofs`.
    pub fn add_element_vec_to_rhs(&self, dofs: &[usize], f_elem: &[T], rhs: &mut [T]) {
        debug_assert_eq!(dofs.len(), f_elem.len());
        for (&d, &v) in dofs.iter().zip(f_elem.iter()) {
            rhs[d] += v;
        }
    }

    /// Number of stored triplets (before deduplication).
    pub fn nnz_raw(&self) -> usize { self.vals.len() }

    /// Append all triplets from `other` (must have the same dimensions).
    ///
    /// Used to merge thread-local COO chunks before `into_csr()`.
    pub fn append(&mut self, mut other: Self) {
        assert_eq!(self.nrows, other.nrows, "coo append: nrows mismatch");
        assert_eq!(self.ncols, other.ncols, "coo append: ncols mismatch");
        self.rows.append(&mut other.rows);
        self.cols.append(&mut other.cols);
        self.vals.append(&mut other.vals);
    }

    /// Build a COO matrix from an existing CSR matrix by iterating over
    /// all stored entries.  Dimensions are preserved; duplicate (row, col)
    /// entries from the CSR are emitted as separate triplets (they will be
    /// summed on the next `into_csr()` call).
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        let nrows = csr.nrows;
        let ncols = csr.ncols;
        let nnz = csr.values.len();
        let mut coo = Self::new(nrows, ncols);
        coo.reserve(nnz);
        for row in 0..nrows {
            for j in csr.row_ptr[row]..csr.row_ptr[row + 1] {
                coo.rows.push(row as u32);
                coo.cols.push(csr.col_idx[j]);
                coo.vals.push(csr.values[j]);
            }
        }
        coo
    }

    /// Convert to CSR with columns sorted by index (traditional COO→CSR).
    /// Used by spadd (which requires sorted rows for its merge algorithm).
    pub fn into_csr_sorted(mut self) -> CsrMatrix<T> {
        let nnz = self.vals.len();
        let nrows = self.nrows;
        if nnz == 0 { return CsrMatrix::new_empty(nrows, self.ncols); }
        let mut keys_vals: Vec<(u64, T)> = Vec::with_capacity(nnz);
        for ((&r, &c), v) in self.rows.iter().zip(self.cols.iter()).zip(self.vals.drain(..)) {
            keys_vals.push(((r as u64) << 32 | c as u64, v));
        }
        self.rows = Vec::new(); self.cols = Vec::new();
        keys_vals.sort_unstable_by_key(|&(k, _)| k);
        let mut out_ptr = vec![0usize; nrows + 1];
        let mut out_col = Vec::<u32>::new();
        let mut out_val = Vec::<T>::new();
        let mut row = 0usize;
        let mut last_col: Option<u32> = None;
        for (k, v) in keys_vals {
            let r = (k >> 32) as usize;
            let c = (k & 0xFFFF_FFFF) as u32;
            if r != row { last_col = None; }
            while row < r { row += 1; out_ptr[row] = out_col.len(); }
            if last_col == Some(c) { *out_val.last_mut().unwrap() = *out_val.last().unwrap() + v; }
            else { last_col = Some(c); out_col.push(c); out_val.push(v); }
        }
        while row < nrows { row += 1; out_ptr[row] = out_col.len(); }
        CsrMatrix { nrows, ncols: self.ncols, row_ptr: out_ptr, col_idx: out_col, values: out_val }
    }

    /// Convert to CSR with insertion-order columns (matching MFEM).
    /// Two-phase: sort by (row, col) to merge duplicates, then
    /// sort each row by insertion order for MFEM-compatible spmv.
    pub fn into_csr(mut self) -> CsrMatrix<T> {
        let nnz = self.vals.len();
        let nrows = self.nrows;
        if nnz == 0 {
            return CsrMatrix::new_empty(nrows, self.ncols);
        }

        // Track insertion order (MFEM: CSR columns in insertion order, not sorted).
        struct Entry<T> { row: u32, col: u32, idx: u32, val: T }
        let mut entries: Vec<Entry<T>> = Vec::with_capacity(nnz);
        for (i, ((&r, &c), v)) in self.rows.iter().zip(self.cols.iter())
            .zip(self.vals.drain(..)).enumerate()
        {
            entries.push(Entry { row: r, col: c, idx: i as u32, val: v });
        }
        self.rows = Vec::new();
        self.cols = Vec::new();

        // Phase A: sort by (row, col, insertion idx) so that duplicate
        // (row, col) entries are summed in insertion order.  MFEM's open
        // SparseMatrix::AddSubMatrix accumulates element contributions
        // element-by-element in traversal order; `sort_unstable_by(row,col)`
        // would permute equal keys and change the summation order (hence the
        // last-ulp differences seen on multi-element diagonals).
        entries.sort_by(|a, b| {
            a.row.cmp(&b.row)
                .then(a.col.cmp(&b.col))
                .then(a.idx.cmp(&b.idx))
        });

        // Merge adjacent duplicates, keeping earliest insertion index.
        let mut merged: Vec<Entry<T>> = Vec::with_capacity(entries.len());
        for e in entries {
            if let Some(last) = merged.last_mut() {
                if last.row == e.row && last.col == e.col {
                    last.val = last.val + e.val;
                    last.idx = last.idx.min(e.idx);
                    continue;
                }
            }
            merged.push(e);
        }

        // Phase B: sort by (row, DESCENDING insertion_idx) — MFEM's open
        // SparseMatrix prepends new columns to the per-row linked list
        // (sparsemat.hpp SearchRow: node->Prev = Rows[row]; Rows[row] = node),
        // so the finalized CSR column order is the REVERSE of the insertion
        // order.  Matching this makes GS-smoother sweeps bit-identical.
        merged.sort_by(|a, b| a.row.cmp(&b.row).then(b.idx.cmp(&a.idx)));

        // Build CSR from merged entries.
        let mut out_ptr = vec![0usize; nrows + 1];
        let mut out_col = Vec::<u32>::new();
        let mut out_val = Vec::<T>::new();
        let mut row = 0usize;
        for e in &merged {
            let r = e.row as usize;
            while row < r { row += 1; out_ptr[row] = out_col.len(); }
            out_col.push(e.col);
            out_val.push(e.val);
        }
        while row < nrows { row += 1; out_ptr[row] = out_col.len(); }

        CsrMatrix { nrows, ncols: self.ncols, row_ptr: out_ptr, col_idx: out_col, values: out_val }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo_to_csr_asymmetric_patterns() {
        let mut ca = CooMatrix::<f64>::new(3, 3);
        ca.add(1, 0, 10.0); ca.add(2, 0, 20.0); ca.add(2, 1, 30.0);
        let a = ca.into_csr();
        assert_eq!(a.row_ptr, [0, 0, 1, 3], "A row_ptr");
        // MFEM open-matrix column order is the REVERSE of insertion order
        // (head-insert linked list): row 2 was inserted (2,0) then (2,1), so
        // the finalized column order is [1, 0].
        assert_eq!(a.col_idx, [0, 1, 0], "A col_idx");
        assert!((a.get(1, 0) - 10.0).abs() < 1e-14);
        assert!((a.get(2, 0) - 20.0).abs() < 1e-14);
        assert!((a.get(2, 1) - 30.0).abs() < 1e-14);
    }

    #[test]
    fn spadd_asymmetric_patterns() {
        let mut ca = CooMatrix::<f64>::new(3, 3);
        ca.add(1, 0, 10.0); ca.add(2, 0, 20.0); ca.add(2, 1, 30.0);
        let a = ca.into_csr();
        let mut cb = CooMatrix::<f64>::new(3, 3);
        cb.add(0, 1, 40.0); cb.add(0, 2, 50.0); cb.add(1, 2, 60.0);
        let b = cb.into_csr();
        let c = crate::spadd(&a, &b);
        assert!((c.get(1, 0) - 10.0).abs() < 1e-14);
        assert!((c.get(0, 1) - 40.0).abs() < 1e-14);
        assert!((c.get(2, 1) - 30.0).abs() < 1e-14);
    }

    #[test]
    fn coo_to_csr_sum_duplicates() {
        // 3×3 identity via duplicate entries
        let mut coo = CooMatrix::<f64>::new(3, 3);
        for i in 0..3 {
            coo.add(i, i, 0.5);
            coo.add(i, i, 0.5);
        }
        let csr = coo.into_csr();
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.values.len(), 3); // 3 unique (i,i)
        for i in 0..3 {
            assert!((csr.get(i, i) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn element_matrix_add() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        let k = [1.0, -1.0, -1.0, 1.0];
        coo.add_element_matrix(&[0, 1], &k);
        let csr = coo.into_csr();
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((csr.get(0, 1) + 1.0).abs() < 1e-14);
    }

    #[test]
    fn append_merges_triplets() {
        let mut a = CooMatrix::<f64>::new(2, 2);
        a.add(0, 0, 1.0);
        let mut b = CooMatrix::<f64>::new(2, 2);
        b.add(1, 1, 2.0);
        a.append(b);
        let csr = a.into_csr();
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((csr.get(1, 1) - 2.0).abs() < 1e-14);
    }
}
