use pyo3::prelude::*;

#[pyclass]
pub struct PyMesh;

#[pymethods]
impl PyMesh {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
