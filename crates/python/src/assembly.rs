use pyo3::prelude::*;

#[pyclass]
pub struct PyStiffnessIntegrator;

#[pymethods]
impl PyStiffnessIntegrator {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyclass]
pub struct PyMassIntegrator;

#[pymethods]
impl PyMassIntegrator {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyclass]
pub struct PyConstantLoad;

#[pymethods]
impl PyConstantLoad {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyfunction]
pub fn py_assemble_bilinear() -> PyResult<()> {
    Ok(())
}

#[pyfunction]
pub fn py_assemble_linear() -> PyResult<()> {
    Ok(())
}

#[pyfunction]
pub fn py_apply_dirichlet() -> PyResult<()> {
    Ok(())
}
