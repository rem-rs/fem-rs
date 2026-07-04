/// All error variants produced by fem-rs.
#[derive(Debug, thiserror::Error)]
pub enum FemError {
    /// A mesh-level inconsistency (bad connectivity, inverted elements, etc.).
    #[error("mesh error: {0}")]
    Mesh(String),

    /// DOF-map inconsistency detected during DOF numbering or assembly.
    #[error("DOF mapping inconsistency: elem {elem}, local dof {dof}")]
    DofMapping { elem: usize, dof: usize },

    /// Iterative solver failed to reach the requested tolerance.
    #[error("solver did not converge after {0} iterations")]
    SolverDivergence(usize),

    /// Two quantities with incompatible sizes were combined.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    /// Standard I/O error (file not found, permission denied, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A Jacobian determinant was non-positive (inverted or degenerate element).
    #[error("non-positive Jacobian det={det:.6e} at element {elem}")]
    NegativeJacobian { elem: usize, det: f64 },

    /// Feature not yet implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),

    /// Catch-all generic error for cases that don't fit the other categories.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias — all public API functions return this.
pub type FemResult<T> = Result<T, FemError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = FemError::SolverDivergence(100);
        assert!(e.to_string().contains("100"));

        let e = FemError::DimMismatch { expected: 3, actual: 2 };
        assert!(e.to_string().contains("expected 3"));
        assert!(e.to_string().contains("got 2"));
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let fem_err: FemError = io_err.into();
        assert!(matches!(fem_err, FemError::Io(_)));
    }

    #[test]
    fn result_alias() {
        let ok: FemResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);
    }
}
