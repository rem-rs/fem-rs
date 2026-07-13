//! Convergence study helper for AMR tests.
//!
//! Provides [`ConvergenceStudy<D>`] and [`ConvergenceRecord`] for tracking
//! convergence rates during adaptive mesh refinement loops.

/// Record of one AMR step in a convergence study.
#[derive(Debug, Clone)]
pub struct ConvergenceRecord {
    pub step: usize,
    pub n_dofs: usize,
    pub n_elems: usize,
    pub l2_error: f64,
    pub h1_error: Option<f64>,
    pub n_h_refined: usize,
    pub n_p_refined: usize,
}

/// Drives an AMR loop and records convergence history.
///
/// # Type Parameters
/// - `D`: spatial dimension (2 or 3)
#[derive(Default)]
pub struct ConvergenceStudy<const D: usize> {
    pub records: Vec<ConvergenceRecord>,
}

impl<const D: usize> ConvergenceStudy<D> {
    /// Create a new empty convergence study.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Append a convergence record.
    pub fn push(&mut self, record: ConvergenceRecord) {
        self.records.push(record);
    }

    /// Return the most recent record, if any.
    pub fn last(&self) -> Option<&ConvergenceRecord> {
        self.records.last()
    }

    /// Estimate the L² convergence rate from the last two records.
    /// Returns the slope in the log-log plot: rate = Δlog(error) / Δlog(DOF).
    pub fn convergence_rate_l2(&self) -> Option<f64> {
        if self.records.len() < 2 {
            return None;
        }
        let a = &self.records[self.records.len() - 2];
        let b = self.records.last().unwrap();

        let log_n_a = (a.n_dofs as f64).ln();
        let log_n_b = (b.n_dofs as f64).ln();
        let log_e_a = a.l2_error.ln();
        let log_e_b = b.l2_error.ln();

        let dn = log_n_b - log_n_a;
        if dn.abs() < 1e-15 {
            return None;
        }
        Some((log_e_b - log_e_a) / dn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_study_tracks_records() {
        let mut study = ConvergenceStudy::<2>::new();
        assert_eq!(study.records.len(), 0);
        study.push(ConvergenceRecord {
            step: 0, n_dofs: 25, n_elems: 16,
            l2_error: 0.1, h1_error: Some(0.5),
            n_h_refined: 4, n_p_refined: 0,
        });
        assert_eq!(study.records.len(), 1);
        assert!((study.convergence_rate_l2().unwrap_or(0.0) - 0.0).abs() < 1e-10);
    }
}
