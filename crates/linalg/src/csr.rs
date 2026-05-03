use std::any::TypeId;

use fem_core::Scalar;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use std::sync::OnceLock;

/// Environment variable for [`spmv_parallel_min_rows`]: minimum CSR row count before
/// Rayon parallelizes `spmv` / `spmv_add` (native `parallel` feature only).
#[cfg(feature = "parallel")]
pub const FEM_LINALG_SPMV_PARALLEL_MIN_ROWS: &str = "FEM_LINALG_SPMV_PARALLEL_MIN_ROWS";

#[cfg(feature = "parallel")]
const DEFAULT_SPMV_PARALLEL_MIN_ROWS: usize = 128;

#[cfg(feature = "parallel")]
static SPMV_PARALLEL_MIN_ROWS: OnceLock<usize> = OnceLock::new();

/// Minimum row count before using Rayon for SpMV (avoids thread overhead on tiny systems).
///
/// Default `128`. Override with [`FEM_LINALG_SPMV_PARALLEL_MIN_ROWS`] (must parse to a
/// positive integer; invalid values fall back to the default).
#[cfg(feature = "parallel")]
#[inline]
pub fn spmv_parallel_min_rows() -> usize {
    *SPMV_PARALLEL_MIN_ROWS.get_or_init(|| {
        std::env::var(FEM_LINALG_SPMV_PARALLEL_MIN_ROWS)
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_SPMV_PARALLEL_MIN_ROWS)
    })
}

#[inline]
fn csr_row_dot_f64(
    row_ptr: &[usize],
    col_idx: &[u32],
    values: &[f64],
    x: &[f64],
    row: usize,
) -> f64 {
    let start = row_ptr[row];
    let end = row_ptr[row + 1];
    let mut k = start;
    let mut sum = 0.0_f64;

    let end4 = start + (end - start) / 4 * 4;
    while k < end4 {
        sum += values[k] * x[col_idx[k] as usize]
            + values[k + 1] * x[col_idx[k + 1] as usize]
            + values[k + 2] * x[col_idx[k + 2] as usize]
            + values[k + 3] * x[col_idx[k + 3] as usize];
        k += 4;
    }
    while k < end {
        sum += values[k] * x[col_idx[k] as usize];
        k += 1;
    }
    sum
}

#[inline]
fn csr_row_dot_axpby_f64(
    row_ptr: &[usize],
    col_idx: &[u32],
    values: &[f64],
    x: &[f64],
    row: usize,
    alpha: f64,
    beta: f64,
    yi: f64,
) -> f64 {
    let s = csr_row_dot_f64(row_ptr, col_idx, values, x, row);
    alpha * s + beta * yi
}

/// Compressed Sparse Row matrix.
///
/// - `row_ptr[i]` = index into `col_idx`/`values` of the first entry in row `i`.
/// - `row_ptr[nrows]` = total number of stored non-zeros.
/// - `col_idx` and `values` are aligned: `values[k]` lives at column `col_idx[k]`.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    pub nrows:   usize,
    pub ncols:   usize,
    pub row_ptr: Vec<usize>,   // length nrows + 1
    pub col_idx: Vec<u32>,
    pub values:  Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Empty matrix with pre-allocated structure arrays.
    pub fn new_empty(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            values:  Vec::new(),
        }
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Get value at `(row, col)`.  Returns 0 if not stored.
    pub fn get(&self, row: usize, col: usize) -> T {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] as usize == col {
                return self.values[k];
            }
        }
        T::zero()
    }

    /// Mutable reference to value at `(row, col)`.
    /// Panics if the entry is not present in the sparsity pattern.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] as usize == col {
                return &mut self.values[k];
            }
        }
        panic!("CsrMatrix::get_mut: ({row},{col}) not in sparsity pattern");
    }

    // -----------------------------------------------------------------------
    // Matrix-vector products
    // -----------------------------------------------------------------------

    /// Compute `y = A x`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: `T` is `f64`; `CsrMatrix<T>` matches `CsrMatrix<f64>` layout.
            let m: &CsrMatrix<f64> =
                unsafe { &*(self as *const CsrMatrix<T> as *const CsrMatrix<f64>) };
            let x: &[f64] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let y: &mut [f64] =
                unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            #[cfg(feature = "parallel")]
            if self.nrows >= spmv_parallel_min_rows() {
                m.spmv_parallel_f64(x, y);
                return;
            }
            m.spmv_serial_f64(x, y);
            return;
        }
        #[cfg(feature = "parallel")]
        if self.nrows >= spmv_parallel_min_rows() {
            self.spmv_parallel(x, y);
            return;
        }
        self.spmv_serial(x, y);
    }

    /// Compute `y = α A x + β y`.
    pub fn spmv_add(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: `T` is `f64` (same layout as below).
            let m: &CsrMatrix<f64> =
                unsafe { &*(self as *const CsrMatrix<T> as *const CsrMatrix<f64>) };
            let x: &[f64] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let y: &mut [f64] =
                unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            let alpha = unsafe { std::ptr::read(&alpha as *const T as *const f64) };
            let beta = unsafe { std::ptr::read(&beta as *const T as *const f64) };
            #[cfg(feature = "parallel")]
            if self.nrows >= spmv_parallel_min_rows() {
                m.spmv_add_parallel_f64(alpha, x, beta, y);
                return;
            }
            m.spmv_add_serial_f64(alpha, x, beta, y);
            return;
        }
        #[cfg(feature = "parallel")]
        if self.nrows >= spmv_parallel_min_rows() {
            self.spmv_add_parallel(alpha, x, beta, y);
            return;
        }
        self.spmv_add_serial(alpha, x, beta, y);
    }

    fn spmv_serial(&self, x: &[T], y: &mut [T]) {
        for (row, yi) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            let mut s = T::zero();
            for k in start..end {
                s += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = s;
        }
    }

    fn spmv_add_serial(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        for (row, yi) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            let mut s = T::zero();
            for k in start..end {
                s += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = alpha * s + beta * *yi;
        }
    }

    #[cfg(feature = "parallel")]
    fn spmv_parallel(&self, x: &[T], y: &mut [T]) {
        // par_windows(2) on row_ptr avoids enumerate() index tracking.
        self.row_ptr.par_windows(2).zip(y.par_iter_mut()).for_each(|(w, yi)| {
            let (start, end) = (w[0], w[1]);
            let mut s = T::zero();
            for k in start..end {
                s += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = s;
        });
    }

    #[cfg(feature = "parallel")]
    fn spmv_add_parallel(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        self.row_ptr.par_windows(2).zip(y.par_iter_mut()).for_each(|(w, yi)| {
            let (start, end) = (w[0], w[1]);
            let mut s = T::zero();
            for k in start..end {
                s += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = alpha * s + beta * *yi;
        });
    }

    /// Diagonal vector `d[i] = A[i,i]`.
    pub fn diagonal(&self) -> Vec<T> {
        let mut d = vec![T::zero(); self.nrows];
        for row in 0..self.nrows {
            d[row] = self.get(row, row);
        }
        d
    }

    // -----------------------------------------------------------------------
    // Boundary condition helpers
    // -----------------------------------------------------------------------

    /// Apply a Dirichlet BC for DOF `row`:
    /// - Zero the entire row.
    /// - Set the diagonal to 1.
    /// - Modify `rhs[row] = prescribed_value`.
    ///
    /// Also subtracts the column contribution from other rows to maintain
    /// symmetry (the "symmetric elimination" approach).
    ///
    /// For symmetric FEM matrices, exploits the fact that A[j,row] ≠ 0 only if
    /// A[row,j] ≠ 0 — so we only visit neighbors of `row` in the sparsity graph,
    /// reducing cost from O(n) to O(nnz_per_row). This is O(1) for sparse FEM
    /// stiffness matrices regardless of problem size.
    pub fn apply_dirichlet_symmetric(
        &mut self,
        row: usize,
        value: T,
        rhs: &mut [T],
    ) {
        // For symmetric FEM matrices: A[j, row] != 0 iff A[row, j] != 0.
        // Collect (other_row, a_ij) pairs from the row's sparsity pattern.
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];

        // Subtract column `row` contribution from neighboring rows (only O(nnz_row) work).
        for k in start..end {
            let other_row = self.col_idx[k] as usize;
            if other_row == row { continue; }
            // a_ij = A[other_row, row]; by symmetry equals A[row, other_row] = values[k]
            let a_ij = self.values[k];
            if a_ij == T::zero() { continue; }
            rhs[other_row] -= a_ij * value;
            // Zero A[other_row, row] via binary search in that row (CSR is sorted).
            if let Some(pos) = self.find_entry(other_row, row) {
                self.values[pos] = T::zero();
            }
        }

        // Zero the entire row, then set diagonal to 1.
        for k in start..end {
            self.values[k] = T::zero();
        }
        if let Some(k) = self.find_entry(row, row) {
            self.values[k] = T::one();
        }
        rhs[row] = value;
    }

    /// Apply Dirichlet BC (row-zeroing only, not symmetric).
    ///
    /// Faster than symmetric elimination; use when symmetry is not required.
    pub fn apply_dirichlet_row_zeroing(
        &mut self,
        row: usize,
        value: T,
        rhs: &mut [T],
    ) {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            self.values[k] = T::zero();
        }
        if let Some(k) = self.find_entry(row, row) {
            self.values[k] = T::one();
        }
        rhs[row] = value;
    }

    fn find_entry(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] as usize == col { return Some(k); }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Debug
    // -----------------------------------------------------------------------

    /// Convert to a dense `nrows × ncols` row-major matrix (testing only).
    pub fn to_dense(&self) -> Vec<T> {
        let mut d = vec![T::zero(); self.nrows * self.ncols];
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            for k in start..end {
                d[row * self.ncols + self.col_idx[k] as usize] = self.values[k];
            }
        }
        d
    }

    // -----------------------------------------------------------------------
    // Matrix arithmetic
    // -----------------------------------------------------------------------

    /// Compute `self + other` (sparse matrix addition, union sparsity pattern).
    ///
    /// Equivalent to the free function [`spadd`].
    pub fn add(&self, other: &CsrMatrix<T>) -> CsrMatrix<T>
    where
        T: std::ops::Mul<Output = T>,
    {
        spadd(self, other)
    }

    /// Compute `alpha * self + beta * other`.
    ///
    /// Both matrices must have the same dimensions.  The result has the union
    /// of the sparsity patterns.
    pub fn axpby(&self, alpha: T, other: &CsrMatrix<T>, beta: T) -> CsrMatrix<T>
    where
        T: std::ops::Mul<Output = T>,
    {
        assert_eq!(self.nrows, other.nrows, "axpby: row count mismatch");
        assert_eq!(self.ncols, other.ncols, "axpby: col count mismatch");

        let m = self.nrows;
        let mut row_ptr = Vec::with_capacity(m + 1);
        let mut col_idx = Vec::new();
        let mut values  = Vec::new();

        row_ptr.push(0);

        for i in 0..m {
            let a_start = self.row_ptr[i];
            let a_end   = self.row_ptr[i + 1];
            let b_start = other.row_ptr[i];
            let b_end   = other.row_ptr[i + 1];

            let mut ja = a_start;
            let mut jb = b_start;

            while ja < a_end && jb < b_end {
                let ca = self.col_idx[ja];
                let cb = other.col_idx[jb];
                if ca < cb {
                    col_idx.push(ca);
                    values.push(alpha * self.values[ja]);
                    ja += 1;
                } else if ca > cb {
                    col_idx.push(cb);
                    values.push(beta * other.values[jb]);
                    jb += 1;
                } else {
                    col_idx.push(ca);
                    values.push(alpha * self.values[ja] + beta * other.values[jb]);
                    ja += 1;
                    jb += 1;
                }
            }
            while ja < a_end {
                col_idx.push(self.col_idx[ja]);
                values.push(alpha * self.values[ja]);
                ja += 1;
            }
            while jb < b_end {
                col_idx.push(other.col_idx[jb]);
                values.push(beta * other.values[jb]);
                jb += 1;
            }

            row_ptr.push(col_idx.len());
        }

        CsrMatrix { nrows: m, ncols: self.ncols, row_ptr, col_idx, values }
    }

    // -----------------------------------------------------------------------
    // Transpose
    // -----------------------------------------------------------------------

    /// Return the transpose `Aᵀ` as a new CSR matrix.
    ///
    /// `Aᵀ[j,i] = A[i,j]` for every stored entry.  The result has
    /// dimensions `ncols × nrows`.
    pub fn transpose(&self) -> CsrMatrix<T> {
        let m = self.nrows;
        let n = self.ncols;
        let nnz = self.nnz();

        // Count entries per column (= per row of Aᵀ).
        let mut col_count = vec![0usize; n];
        for &c in &self.col_idx {
            col_count[c as usize] += 1;
        }

        // Build row_ptr for Aᵀ (prefix sum of col_count).
        let mut t_row_ptr = vec![0usize; n + 1];
        for j in 0..n {
            t_row_ptr[j + 1] = t_row_ptr[j] + col_count[j];
        }

        // Scatter entries.
        let mut t_col_idx = vec![0u32; nnz];
        let mut t_values  = vec![T::zero(); nnz];
        let mut offset    = t_row_ptr.clone(); // write cursors
        for i in 0..m {
            let start = self.row_ptr[i];
            let end   = self.row_ptr[i + 1];
            for k in start..end {
                let j = self.col_idx[k] as usize;
                let pos = offset[j];
                t_col_idx[pos] = i as u32;
                t_values[pos]  = self.values[k];
                offset[j] += 1;
            }
        }

        CsrMatrix {
            nrows:   n,
            ncols:   m,
            row_ptr: t_row_ptr,
            col_idx: t_col_idx,
            values:  t_values,
        }
    }
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Sparse matrix addition: `C = A + B`.
///
/// Both matrices must have the same dimensions.  The result has the union
/// of the sparsity patterns; entries present in both are summed.
///
/// # Panics
/// Panics if `A` and `B` have different dimensions.
pub fn spadd<T: Scalar>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T> {
    assert_eq!(a.nrows, b.nrows, "spadd: row count mismatch");
    assert_eq!(a.ncols, b.ncols, "spadd: col count mismatch");

    let m = a.nrows;
    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_idx = Vec::new();
    let mut values  = Vec::new();

    row_ptr.push(0);

    for i in 0..m {
        let a_start = a.row_ptr[i];
        let a_end   = a.row_ptr[i + 1];
        let b_start = b.row_ptr[i];
        let b_end   = b.row_ptr[i + 1];

        let mut ja = a_start;
        let mut jb = b_start;

        // Merge two sorted column-index streams.
        while ja < a_end && jb < b_end {
            let ca = a.col_idx[ja];
            let cb = b.col_idx[jb];
            if ca < cb {
                col_idx.push(ca);
                values.push(a.values[ja]);
                ja += 1;
            } else if ca > cb {
                col_idx.push(cb);
                values.push(b.values[jb]);
                jb += 1;
            } else {
                // Same column — sum values.
                col_idx.push(ca);
                values.push(a.values[ja] + b.values[jb]);
                ja += 1;
                jb += 1;
            }
        }
        // Flush remaining entries.
        while ja < a_end {
            col_idx.push(a.col_idx[ja]);
            values.push(a.values[ja]);
            ja += 1;
        }
        while jb < b_end {
            col_idx.push(b.col_idx[jb]);
            values.push(b.values[jb]);
            jb += 1;
        }

        row_ptr.push(col_idx.len());
    }

    CsrMatrix {
        nrows: m,
        ncols: a.ncols,
        row_ptr,
        col_idx,
        values,
    }
}

impl CsrMatrix<f64> {
    pub(super) fn spmv_serial_f64(&self, x: &[f64], y: &mut [f64]) {
        for (row, yi) in y.iter_mut().enumerate() {
            *yi = csr_row_dot_f64(&self.row_ptr, &self.col_idx, &self.values, x, row);
        }
    }

    pub(super) fn spmv_add_serial_f64(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        for (row, yi) in y.iter_mut().enumerate() {
            *yi = csr_row_dot_axpby_f64(
                &self.row_ptr, &self.col_idx, &self.values, x, row, alpha, beta, *yi,
            );
        }
    }

    #[cfg(feature = "parallel")]
    pub(super) fn spmv_parallel_f64(&self, x: &[f64], y: &mut [f64]) {
        // Use par_windows(2) on row_ptr + zip to avoid enumerate overhead.
        self.row_ptr.par_windows(2).zip(y.par_iter_mut()).for_each(|(w, yi)| {
            let (start, end) = (w[0], w[1]);
            let mut k = start;
            let mut sum = 0.0_f64;
            let end4 = start + (end - start) / 4 * 4;
            while k < end4 {
                sum += self.values[k]     * x[self.col_idx[k]     as usize]
                     + self.values[k + 1] * x[self.col_idx[k + 1] as usize]
                     + self.values[k + 2] * x[self.col_idx[k + 2] as usize]
                     + self.values[k + 3] * x[self.col_idx[k + 3] as usize];
                k += 4;
            }
            while k < end {
                sum += self.values[k] * x[self.col_idx[k] as usize];
                k += 1;
            }
            *yi = sum;
        });
    }

    #[cfg(feature = "parallel")]
    pub(super) fn spmv_add_parallel_f64(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        self.row_ptr.par_windows(2).zip(y.par_iter_mut()).for_each(|(w, yi)| {
            let (start, end) = (w[0], w[1]);
            let mut k = start;
            let mut sum = 0.0_f64;
            let end4 = start + (end - start) / 4 * 4;
            while k < end4 {
                sum += self.values[k]     * x[self.col_idx[k]     as usize]
                     + self.values[k + 1] * x[self.col_idx[k + 1] as usize]
                     + self.values[k + 2] * x[self.col_idx[k + 2] as usize]
                     + self.values[k + 3] * x[self.col_idx[k + 3] as usize];
                k += 4;
            }
            while k < end {
                sum += self.values[k] * x[self.col_idx[k] as usize];
                k += 1;
            }
            *yi = alpha * sum + beta * *yi;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooMatrix;

    fn small_matrix() -> CsrMatrix<f64> {
        // [ 2 -1  0 ]
        // [-1  2 -1 ]
        // [ 0 -1  2 ]
        let mut c = CooMatrix::<f64>::new(3, 3);
        c.add(0, 0,  2.0); c.add(0, 1, -1.0);
        c.add(1, 0, -1.0); c.add(1, 1,  2.0); c.add(1, 2, -1.0);
        c.add(2, 1, -1.0); c.add(2, 2,  2.0);
        c.into_csr()
    }

    #[test]
    fn spmv_tridiag() {
        let a = small_matrix();
        let x = vec![1.0f64, 2.0, 3.0];
        let mut y = vec![0.0f64; 3];
        a.spmv(&x, &mut y);
        // [2-2, -1+4-3, -2+6] = [0, 0, 4]
        assert!((y[0]).abs() < 1e-14);
        assert!((y[1]).abs() < 1e-14);
        assert!((y[2] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn diagonal() {
        let a = small_matrix();
        let d = a.diagonal();
        assert_eq!(d, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn dirichlet_row_zeroing() {
        let mut a = small_matrix();
        let mut rhs = vec![1.0f64, 2.0, 3.0];
        a.apply_dirichlet_row_zeroing(0, 5.0, &mut rhs);
        assert!((a.get(0, 1)).abs() < 1e-14, "off-diag should be zero");
        assert!((a.get(0, 0) - 1.0).abs() < 1e-14, "diagonal should be 1");
        assert!((rhs[0] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn transpose_symmetric() {
        // Symmetric matrix → transpose should be identical.
        let a = small_matrix();
        let at = a.transpose();
        assert_eq!(at.nrows, a.ncols);
        assert_eq!(at.ncols, a.nrows);
        let da = a.to_dense();
        let dt = at.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert!((da[i * 3 + j] - dt[j * 3 + i]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn transpose_rectangular() {
        // 2×3 matrix
        let mut c = CooMatrix::<f64>::new(2, 3);
        c.add(0, 0, 1.0); c.add(0, 2, 3.0);
        c.add(1, 1, 4.0); c.add(1, 2, 5.0);
        let a = c.into_csr();
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        assert!((at.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((at.get(2, 0) - 3.0).abs() < 1e-14);
        assert!((at.get(1, 1) - 4.0).abs() < 1e-14);
        assert!((at.get(2, 1) - 5.0).abs() < 1e-14);
        assert!((at.get(0, 1)).abs() < 1e-14); // zero entry
    }

    #[test]
    fn transpose_double_is_identity() {
        let a = small_matrix();
        let att = a.transpose().transpose();
        let da = a.to_dense();
        let dt = att.to_dense();
        for k in 0..da.len() {
            assert!((da[k] - dt[k]).abs() < 1e-14);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn spmv_parallel_min_rows_positive() {
        assert!(spmv_parallel_min_rows() >= 1);
    }

    #[test]
    fn spadd_basic() {
        let a = small_matrix();
        let b = small_matrix();
        let c = super::spadd(&a, &b);
        // C should be 2*A
        let dc = c.to_dense();
        let da = a.to_dense();
        for k in 0..dc.len() {
            assert!((dc[k] - 2.0 * da[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn spadd_with_empty() {
        let a = small_matrix();
        let _b = CsrMatrix::<f64>::new_empty(3, 3);
        // Need row_ptr for b to have correct length
        let b = CooMatrix::<f64>::new(3, 3).into_csr();
        let c = super::spadd(&a, &b);
        let dc = c.to_dense();
        let da = a.to_dense();
        for k in 0..dc.len() {
            assert!((dc[k] - da[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn spadd_different_patterns() {
        // A has entries on diagonal, B has off-diagonal only
        let mut ca = CooMatrix::<f64>::new(2, 2);
        ca.add(0, 0, 1.0); ca.add(1, 1, 2.0);
        let a = ca.into_csr();

        let mut cb = CooMatrix::<f64>::new(2, 2);
        cb.add(0, 1, 3.0); cb.add(1, 0, 4.0);
        let b = cb.into_csr();

        let c = super::spadd(&a, &b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((c.get(0, 1) - 3.0).abs() < 1e-14);
        assert!((c.get(1, 0) - 4.0).abs() < 1e-14);
        assert!((c.get(1, 1) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn add_method_same_as_spadd() {
        let a = small_matrix();
        let b = small_matrix();
        let c_free = super::spadd(&a, &b);
        let c_method = a.add(&b);
        let df = c_free.to_dense();
        let dm = c_method.to_dense();
        assert_eq!(df.len(), dm.len());
        for (f, m) in df.iter().zip(dm.iter()) {
            assert!((f - m).abs() < 1e-14, "add method differs from spadd: {f} vs {m}");
        }
    }

    #[test]
    fn axpby_identity_alpha1_beta0() {
        // axpby(1, b, 0) should equal a
        let a = small_matrix();
        let b = small_matrix();
        let c = a.axpby(1.0, &b, 0.0);
        let da = a.to_dense();
        let dc = c.to_dense();
        for (ai, ci) in da.iter().zip(dc.iter()) {
            assert!((ai - ci).abs() < 1e-14);
        }
    }

    #[test]
    fn axpby_scaled() {
        // axpby(2, b, 3) = 2*a + 3*b
        let a = small_matrix();
        let b = small_matrix();
        let c = a.axpby(2.0, &b, 3.0);
        // a == b here, so 2*a + 3*a = 5*a
        let da = a.to_dense();
        let dc = c.to_dense();
        for (ai, ci) in da.iter().zip(dc.iter()) {
            assert!((5.0 * ai - ci).abs() < 1e-14);
        }
    }

    #[test]
    fn axpby_different_patterns() {
        let mut ca = CooMatrix::<f64>::new(2, 2);
        ca.add(0, 0, 1.0);
        let a = ca.into_csr();

        let mut cb = CooMatrix::<f64>::new(2, 2);
        cb.add(1, 1, 2.0);
        let b = cb.into_csr();

        let c = a.axpby(3.0, &b, 4.0);
        // (0,0) = 3*1 = 3, (1,1) = 4*2 = 8
        assert!((c.get(0, 0) - 3.0).abs() < 1e-14);
        assert!((c.get(1, 1) - 8.0).abs() < 1e-14);
        assert!((c.get(0, 1)).abs() < 1e-14);
        assert!((c.get(1, 0)).abs() < 1e-14);
    }
}
