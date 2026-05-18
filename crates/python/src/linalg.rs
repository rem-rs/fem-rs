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
}
