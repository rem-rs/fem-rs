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

    /// Convert to CSR, summing duplicate entries.
    ///
    /// Sort by (row, col), then merge duplicates.
    /// Uses packed u64 keys `(row << 32 | col)` for cache-friendly sorting.
    /// For large matrices (≥ 16k nnz) with the `parallel` feature, uses Rayon
    /// parallel sort for additional speedup.
    pub fn into_csr(mut self) -> CsrMatrix<T> {
        let nnz = self.vals.len();
        let nrows = self.nrows;
        if nnz == 0 {
            return CsrMatrix::new_empty(nrows, self.ncols);
        }

        // 1. Count entries per row.
        let mut row_count = vec![0usize; nrows];
        for &r in &self.rows { row_count[r as usize] += 1; }

        // 2. Prefix sum -> row_ptr (temp).
        let mut row_ptr = vec![0usize; nrows + 1];
        for i in 0..nrows { row_ptr[i + 1] = row_ptr[i] + row_count[i]; }

        // 3. Scatter (row, col, val) to per-row bins.
        let mut col_idx = vec![0u32; nnz];
        let mut values  = vec![T::zero(); nnz];
        let mut cursor = row_ptr.clone();
        for ((&r, &c), v) in self.rows.iter().zip(self.cols.iter()).zip(self.vals.drain(..)) {
            let pos = cursor[r as usize];
            cursor[r as usize] = pos + 1;
            col_idx[pos] = c;
            values[pos] = v;
        }
        self.rows = Vec::new();
        self.cols = Vec::new();

        // 4. Sort each row and merge duplicates into final output.
        let mut out_col = Vec::<u32>::new();
        let mut out_val = Vec::<T>::new();
        let mut out_ptr = vec![0usize; nrows + 1];

        for row in 0..nrows {
            let start = row_ptr[row];
            let end   = row_ptr[row + 1];
            let len   = end - start;
            out_ptr[row] = out_col.len();
            if len < 2 {
                if len == 1 {
                    out_col.push(col_idx[start]);
                    out_val.push(values[start]);
                }
                continue;
            }

            // Pack and sort row by column index.
            let mut pairs: Vec<(u32, T)> = Vec::with_capacity(len);
            for k in start..end {
                pairs.push((col_idx[k], std::mem::replace(&mut values[k], T::zero())));
            }
            pairs.sort_unstable_by_key(|&(c, _)| c);

            // Compact duplicates into output.
            let mut last_col: Option<u32> = None;
            for (c, v) in pairs {
                if last_col == Some(c) {
                    *out_val.last_mut().unwrap() = *out_val.last().unwrap() + v;
                } else {
                    last_col = Some(c);
                    out_col.push(c);
                    out_val.push(v);
                }
            }
        }
        out_ptr[nrows] = out_col.len();
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
        assert_eq!(a.col_idx, [0, 0, 1], "A col_idx");
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
