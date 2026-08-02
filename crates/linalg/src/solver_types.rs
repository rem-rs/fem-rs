//! Solver configuration and error types (requires `direct` feature = linlvo).
//!
//! This module lives in `fem-linalg` so that `fem-amg` can use these types
//! without depending on `fem-solver` (avoiding a circular dependency).

/// Outcome returned by solvers.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
}

/// Solver error.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("solver did not converge in {max_iter} iterations (residual = {residual:.3e})")]
    ConvergenceFailed { max_iter: usize, residual: f64 },
    #[error("dimension mismatch: matrix is {rows}×{cols}, rhs has length {rhs}")]
    DimensionMismatch { rows: usize, cols: usize, rhs: usize },
    #[error("linlvo error: {0}")]
    Linlvo(String),
}

impl From<linlvo::SolverError> for SolverError {
    fn from(e: linlvo::SolverError) -> Self {
        match e {
            linlvo::SolverError::ConvergenceFailed { max_iter, residual } => {
                SolverError::ConvergenceFailed { max_iter, residual }
            }
            other => SolverError::Linlvo(other.to_string()),
        }
    }
}

/// Verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PrintLevel {
    #[default] Silent,
    Summary,
    Iterations,
    Debug,
}

/// Convergence parameters.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub rtol: f64,
    pub atol: f64,
    pub max_iter: usize,
    pub verbose: bool,
    pub print_level: PrintLevel,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 1_000, verbose: false, print_level: PrintLevel::Silent }
    }
}

impl SolverConfig {
    pub fn to_linlvo(&self) -> linlvo::SolverParams {
        let level = match self.effective_print_level() {
            PrintLevel::Silent => linlvo::VerboseLevel::Silent,
            PrintLevel::Summary => linlvo::VerboseLevel::Summary,
            _ => linlvo::VerboseLevel::Iterations,
        };
        linlvo::SolverParams { rtol: self.rtol, atol: self.atol, max_iter: self.max_iter, verbose: level, check_interval: 0 }
    }

    pub fn effective_print_level(&self) -> PrintLevel {
        if self.print_level != PrintLevel::Silent { self.print_level }
        else if self.verbose { PrintLevel::Iterations }
        else { PrintLevel::Silent }
    }
}

/// Convert `fem_linalg::CsrMatrix<T>` to `linlvo::sparse::CsrMatrix<T>`.
pub fn fem_to_linlvo_csr<T: linlvo::core::scalar::Scalar>(a: &crate::CsrMatrix<T>) -> linlvo::sparse::CsrMatrix<T> {
    linlvo::sparse::CsrMatrix::from_raw(
        a.nrows, a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

/// Convert a linlvo `SolverResult` to a `SolveResult`.
pub fn into_result(r: linlvo::SolverResult) -> SolveResult {
    SolveResult { converged: r.converged, iterations: r.iterations, final_residual: r.final_residual }
}
