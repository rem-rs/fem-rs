use pyo3::prelude::*;

#[pyclass]
pub struct PyCsrMatrix;

#[pymethods]
impl PyCsrMatrix {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
