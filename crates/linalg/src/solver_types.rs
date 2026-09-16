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

/// Verbosity level — the MFEM 4.10 `IterativeSolver` legacy print scale
/// (`linalg/solvers.hpp` `PrintLevel` + `FromLegacyPrintLevel`,
/// `linalg/solvers.cpp:119`):
///
/// | variant        | MFEM level | errors | warnings | iterations | summary | first_and_last |
/// |----------------|------------|--------|----------|------------|---------|----------------|
/// | `Silent`       | -1         | no     | no       | no         | no      | no             |
/// | `Summary`      | 2          | yes    | yes      | no         | yes     | no             |
/// | `Iterations`   | 1          | yes    | yes      | yes        | no      | no             |
/// | `Debug`        | 1          | yes    | yes      | yes        | no      | no             |
/// | `WarningsOnly` | 0          | yes    | yes      | no         | no      | no             |
/// | `FirstAndLast` | 3          | yes    | yes      | no         | no      | yes            |
///
/// **Ordering caveat:** the derive order used to matter because legacy
/// consumers compared with `>=`; the last such consumer (`fem-solver`'s
/// `bpcg.rs`) was fixed to an explicit match in D216, so the order is now
/// only historical — do NOT reintroduce `>=`/`<` tests on this type: the
/// level-0/level-3 variants sort after `Iterations` even though their MFEM
/// levels (0 and 3) do not follow the derive order.  `Debug` is a fem-rs
/// extension of MFEM's level 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PrintLevel {
    #[default] Silent,
    Summary,
    Iterations,
    Debug,
    /// MFEM legacy level 0 (`PrintLevel().Errors().Warnings()`): warnings and
    /// errors only — NO iteration lines and no summary.
    WarningsOnly,
    /// MFEM legacy level 3 (`PrintLevel().Errors().Warnings().FirstAndLast()`):
    /// the first iteration line (with a `" ..."` suffix) and the final
    /// iteration line instead of the per-iteration history, plus the ARF line.
    FirstAndLast,
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
    /// Map the MFEM legacy print scale onto `linlvo`'s 3-level `VerboseLevel`.
    ///
    /// D218 — this mapping is **lossy** for two variants, because
    /// `linlvo::VerboseLevel` (`vendor/linger/src/core/solver.rs`) has only
    /// `Silent`/`Summary`/`Iterations`: no level-0 (warnings-only) tier, no
    /// level-3 (first-and-last) tier, no warnings channel of its own, and no
    /// output abstraction (its solvers `println!` directly behind
    /// `params.verbose == Iterations` / `!= Silent` checks, so a fem-rs-side
    /// wrapper cannot intercept or filter lines without changing the vendor
    /// crate).  Per variant (MFEM legacy level in parentheses):
    ///
    /// | `PrintLevel`    | → `linlvo` | exact? | why |
    /// |-----------------|------------|--------|-----|
    /// | `Silent` (-1)   | `Silent`     | yes    | -1 is linlvo's `Silent`. |
    /// | `WarningsOnly` (0) | `Silent`  | no (under-prints) | level 0 must show warnings and errors but NO summary; linlvo has no warnings-only channel — its only end-of-run chatter is gated `!= Silent`, i.e. the Summary tier, which would violate level 0's no-summary contract. Quiet is the honest downgrade. |
    /// | `Summary` (2)   | `Summary`    | yes (tier) | level 2 is linlvo's `Summary`. |
    /// | `Iterations` (1)| `Iterations` | yes (tier) | level 1 is linlvo's `Iterations`. |
    /// | `Debug` (1)     | `Iterations` | yes (tier) | fem-rs extension of level 1. |
    /// | `FirstAndLast` (3) | `Summary` | no (approximates) | level 3 prints the first and last iteration lines instead of the history; linlvo cannot do that. `Iterations` would wrongly give the FULL history; `Summary` gives only the end-of-run line(s) — the "last" half of first-and-last, strictly less than the history. |
    ///
    /// MFEM-faithful level-0/level-3 output exists only on the fem-rs-native
    /// MFEM-ported solvers, which do not go through this mapping:
    /// `crates/solver/src/iterative.rs` (CG/MINRES/GMRES/BiCGSTAB trailers)
    /// and `crates/solver/src/bpcg.rs` (D216).
    pub fn to_linlvo(&self) -> linlvo::SolverParams {
        let level = match self.effective_print_level() {
            PrintLevel::Silent | PrintLevel::WarningsOnly => linlvo::VerboseLevel::Silent,
            PrintLevel::Summary | PrintLevel::FirstAndLast => linlvo::VerboseLevel::Summary,
            PrintLevel::Iterations | PrintLevel::Debug => linlvo::VerboseLevel::Iterations,
        };
        linlvo::SolverParams { rtol: self.rtol, atol: self.atol, max_iter: self.max_iter, verbose: level, check_interval: 10 }
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

#[cfg(test)]
mod d218_mapping_tests {
    //! Pin the (lossy) `PrintLevel` → `linlvo::VerboseLevel` table, doc'd on
    //! [`SolverConfig::to_linlvo`]: `WarningsOnly` degrades to `Silent`
    //! (under-prints), `FirstAndLast` degrades to `Summary` (approximates).

    use super::*;

    fn level_of(cfg: &SolverConfig) -> linlvo::VerboseLevel {
        cfg.to_linlvo().verbose
    }

    #[test]
    fn mapping_table_is_pinned() {
        let base = SolverConfig::default();
        for (level, expected) in [
            (PrintLevel::Silent, linlvo::VerboseLevel::Silent),
            // lossy: level 0 has no warnings channel in linlvo → quiet
            (PrintLevel::WarningsOnly, linlvo::VerboseLevel::Silent),
            (PrintLevel::Summary, linlvo::VerboseLevel::Summary),
            (PrintLevel::Iterations, linlvo::VerboseLevel::Iterations),
            // Debug is MFEM level 1 in disguise
            (PrintLevel::Debug, linlvo::VerboseLevel::Iterations),
            // lossy: no first/last tier in linlvo → closest subset is Summary
            (PrintLevel::FirstAndLast, linlvo::VerboseLevel::Summary),
        ] {
            let cfg = SolverConfig { print_level: level, ..base.clone() };
            assert_eq!(
                level_of(&cfg),
                expected,
                "to_linlvo mapping changed for {level:?}"
            );
        }
    }

    #[test]
    fn verbose_flag_falls_back_to_iterations() {
        // `print_level: Silent` + `verbose: true` is the legacy flag combo;
        // effective_print_level raises it to level 1 → linlvo Iterations.
        let cfg = SolverConfig { print_level: PrintLevel::Silent, verbose: true, ..Default::default() };
        assert_eq!(level_of(&cfg), linlvo::VerboseLevel::Iterations);
    }

    #[test]
    fn convergence_params_pass_through() {
        let cfg = SolverConfig {
            rtol: 1e-6,
            atol: 1e-12,
            max_iter: 42,
            verbose: false,
            print_level: PrintLevel::Summary,
        };
        let p = cfg.to_linlvo();
        assert_eq!(p.rtol, 1e-6);
        assert_eq!(p.atol, 1e-12);
        assert_eq!(p.max_iter, 42);
        assert_eq!(p.check_interval, 10);
    }
}
