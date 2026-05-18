use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArrayMethods};
use fem_solver::{solve_cg, solve_gmres, solve_sparse_lu, SolverConfig, SolveResult};
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
///
/// Args:
///     mat: CsrMatrix — system matrix
///     b: ndarray — right-hand side
///     x: ndarray — initial guess (modified in-place to solution)
///     tol: float — relative tolerance (default 1e-8)
///     max_iter: int — maximum iterations (default 1000)
///
/// Returns:
///     SolveResult
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
    // SAFETY: The numpy array is guaranteed to be contiguous and
    // its lifetime is bounded by the function scope.
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_cg(&mat.inner, &rhs, &mut sol, &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // SAFETY: `sol` has the same length as `x`, and we are
    // writing back into the same buffer we read from.
    unsafe {
        let x_slice = x.as_slice_mut()?;
        x_slice.copy_from_slice(&sol);
    }
    Ok(result.into())
}

/// Solve Ax = b using GMRES (general non-symmetric systems).
///
/// Args:
///     mat: CsrMatrix — system matrix
///     b: ndarray — right-hand side
///     x: ndarray — initial guess (modified in-place to solution)
///     restart: int — Krylov subspace dimension (default 30)
///     tol: float — relative tolerance (default 1e-8)
///     max_iter: int — maximum iterations (default 1000)
///
/// Returns:
///     SolveResult
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
    // SAFETY: The numpy array is guaranteed to be contiguous and
    // its lifetime is bounded by the function scope.
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let mut sol = unsafe { x.as_slice()? }.to_vec();
    let cfg = SolverConfig {
        rtol: tol.unwrap_or(1e-8),
        max_iter: max_iter.unwrap_or(1000),
        ..Default::default()
    };
    let result = solve_gmres(&mat.inner, &rhs, &mut sol, restart.unwrap_or(30), &cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // SAFETY: `sol` has the same length as `x`, and we are
    // writing back into the same buffer we read from.
    unsafe {
        let x_slice = x.as_slice_mut()?;
        x_slice.copy_from_slice(&sol);
    }
    Ok(result.into())
}

/// Solve Ax = b using sparse LU factorization (direct solve).
///
/// Args:
///     mat: CsrMatrix — system matrix
///     b: ndarray — right-hand side
///
/// Returns:
///     ndarray — solution vector
#[pyfunction]
#[pyo3(name = "solve_sparse_lu")]
pub fn py_solve_sparse_lu<'py>(
    py: Python<'py>,
    mat: &PyCsrMatrix,
    b: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // SAFETY: The numpy array is guaranteed to be contiguous and
    // its lifetime is bounded by the function scope.
    let rhs = unsafe { b.as_slice()? }.to_vec();
    let x = solve_sparse_lu(&mat.inner, &rhs)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, x))
}
