//! Backend selection primitives for operator execution.
//!
//! This module introduces a stable backend enum used by higher layers to
//! choose between classic assembled operators and reed-backed operator paths.

use fem_linalg::CsrMatrix;

/// Assembly/execution backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorBackend {
    /// Classic fem-rs assembly path (assembled sparse matrices).
    Native,
    /// reed/libCEED-style operator path.
    Reed,
}

impl OperatorBackend {
    /// Parse from user-facing backend name.
    ///
    /// Accepted values: `"native"`, `"reed"` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "native" => Some(Self::Native),
            "reed" => Some(Self::Reed),
            _ => None,
        }
    }

    /// Canonical backend name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Reed => "reed",
        }
    }
}

/// Minimal linear-operator interface used by backend-agnostic solver entrypoints.
pub trait LinearOperator {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Apply `y <- A * x`.
    fn apply(&self, x: &[f64], y: &mut [f64]);
}

/// Adapter that exposes a CSR matrix through [`LinearOperator`].
pub struct CsrLinearOperator<'a> {
    mat: &'a CsrMatrix<f64>,
}

impl<'a> CsrLinearOperator<'a> {
    /// Wrap a CSR matrix as a backend-agnostic linear operator.
    pub fn new(mat: &'a CsrMatrix<f64>) -> Self {
        Self { mat }
    }

    /// Borrow the underlying matrix.
    pub fn matrix(&self) -> &'a CsrMatrix<f64> {
        self.mat
    }
}

impl LinearOperator for CsrLinearOperator<'_> {
    fn nrows(&self) -> usize { self.mat.nrows }
    fn ncols(&self) -> usize { self.mat.ncols }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.mat.spmv(x, y);
    }
}

impl LinearOperator for CsrMatrix<f64> {
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.spmv(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    #[test]
    fn parse_backend_name() {
        assert_eq!(OperatorBackend::parse("native"), Some(OperatorBackend::Native));
        assert_eq!(OperatorBackend::parse("REED"), Some(OperatorBackend::Reed));
        assert_eq!(OperatorBackend::parse("other"), None);
    }

    #[test]
    fn csr_linear_operator_apply_matches_spmv() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(0, 1, 1.0);
        coo.add(1, 0, 1.0);
        coo.add(1, 1, 3.0);
        let a = coo.into_csr();

        let op = CsrLinearOperator::new(&a);
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 2];
        op.apply(&x, &mut y);

        assert!((y[0] - 4.0).abs() < 1e-12);
        assert!((y[1] - 7.0).abs() < 1e-12);
        assert_eq!(op.nrows(), 2);
        assert_eq!(op.ncols(), 2);
    }
}
