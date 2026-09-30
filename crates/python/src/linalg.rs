use pyo3::prelude::*;
use fem_linalg::CsrMatrix;

/// Wrapper around ``CsrMatrix<f64>`` for Python.
#[pyclass(name = "CsrMatrix")]
pub struct PyCsrMatrix {
    pub inner: CsrMatrix<f64>,
}

#[pymethods]
impl PyCsrMatrix {
    #[new]
    pub fn new() -> Self {
        PyCsrMatrix {
            inner: CsrMatrix::new_empty(0, 0),
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a tuple ``(data, indices, indptr, shape)`` suitable for
    /// constructing ``scipy.sparse.csr_matrix`` in Python:
    ///
    /// ```python
    /// data, indices, indptr, shape = csr.to_scipy()
    /// A = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    /// ```
    pub fn to_scipy(&self) -> PyResult<(Vec<f64>, Vec<u32>, Vec<usize>, (usize, usize))> {
        Ok((
            self.inner.values.clone(),
            self.inner.col_idx.clone(),
            self.inner.row_ptr.clone(),
            (self.inner.nrows, self.inner.ncols),
        ))
    }

    /// Dense row-major copy of the matrix as a flat list of ``nrows·ncols``
    /// entries (entry ``(i, j)`` at index ``i·ncols + j``).  Delegates to
    /// ``fem_linalg::CsrMatrix::to_dense``.  Intended for small matrices
    /// (tests, diagnostics).
    pub fn to_dense(&self) -> Vec<f64> {
        self.inner.to_dense()
    }
}
