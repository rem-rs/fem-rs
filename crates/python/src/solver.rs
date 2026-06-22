use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArrayMethods};
use fem_solver::{
    solve_cg, solve_gmres, solve_sparse_lu, solve_sparse_cholesky,
    solve_pcg_jacobi, solve_bicgstab,
    SolverConfig, SolveResult,
};
use crate::linalg::PyCsrMatrix;

/// Result of an iterative solve.
#[pyclass(name = "SolveResult")]
pub struct PySolveResult {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub final_residual: f64,
}

impl From<SolveResult> for PySolveResult {
    fn from(r: SolveResult) -> Self {
        PySolveResult {
            converged: r.converged,
            iterations: r.iterations,
            final_residual: r.final_residual,
        }
    }
}

/// Solve Ax = b using Conjugate Gradient (SPD systems).
#[pyfunction]
#[pyo3(name = "solve_cg")]
#[pyo3(signature = (mat, b, x, tol=None, max_iter=None))]
pub fn py_solve_cg(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<PySolveResult> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_cg(&mat.inner, &rhs, &mut sol, &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    unsafe { x.as_slice_mut()?.copy_from_slice(&sol); }
    Ok(result.into())
}

/// Solve Ax = b using PCG + Jacobi preconditioner.
#[pyfunction]
#[pyo3(name = "solve_pcg_jacobi")]
#[pyo3(signature = (mat, b, x, tol=None, max_iter=None))]
pub fn py_solve_pcg_jacobi(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<PySolveResult> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_pcg_jacobi(&mat.inner, &rhs, &mut sol, &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    unsafe { x.as_slice_mut()?.copy_from_slice(&sol); }
    Ok(result.into())
}

/// Solve Ax = b using GMRES.
#[pyfunction]
#[pyo3(name = "solve_gmres")]
#[pyo3(signature = (mat, b, x, restart=None, tol=None, max_iter=None))]
pub fn py_solve_gmres(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    restart: Option<usize>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<PySolveResult> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_gmres(&mat.inner, &rhs, &mut sol, restart.unwrap_or(30), &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    unsafe { x.as_slice_mut()?.copy_from_slice(&sol); }
    Ok(result.into())
}

/// Solve Ax = b using BiCGSTAB.
#[pyfunction]
#[pyo3(name = "solve_bicgstab")]
#[pyo3(signature = (mat, b, x, tol=None, max_iter=None))]
pub fn py_solve_bicgstab(
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<PySolveResult> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_bicgstab(&mat.inner, &rhs, &mut sol, &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    unsafe { x.as_slice_mut()?.copy_from_slice(&sol); }
    Ok(result.into())
}

/// Solve Ax = b using sparse LU factorization (direct).
#[pyfunction]
#[pyo3(name = "solve_sparse_lu")]
pub fn py_solve_sparse_lu<'py>(
    py: Python<'py>,
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let x = solve_sparse_lu(&mat.inner, &rhs)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, x))
}

/// Solve Ax = b using sparse Cholesky (SPD, ~2x faster than LU).
#[pyfunction]
#[pyo3(name = "solve_sparse_cholesky")]
pub fn py_solve_sparse_cholesky<'py>(
    py: Python<'py>,
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let x = solve_sparse_cholesky(&mat.inner, &rhs)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, x))
}
