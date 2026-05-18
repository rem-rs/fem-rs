use pyo3::prelude::*;

#[pyclass]
pub struct PySolveResult;

#[pymethods]
impl PySolveResult {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyfunction]
pub fn py_solve_cg() -> PyResult<()> {
    Ok(())
}

#[pyfunction]
pub fn py_solve_gmres() -> PyResult<()> {
    Ok(())
}

#[pyfunction]
pub fn py_solve_sparse_lu() -> PyResult<()> {
    Ok(())
}
