use pyo3::prelude::*;

#[pyclass]
pub struct PyH1Space;

#[pymethods]
impl PyH1Space {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
