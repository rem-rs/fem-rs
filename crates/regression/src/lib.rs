//! Regression baseline testing framework for fem-rs.
//!
//! Stores known-good numerical results as JSON baselines and compares
//! current test outputs against them — catching silent numerical
//! regressions that loose-boundary tests would miss.
//!
//! # Workflow
//!
//! 1. **Create baseline** — run tests with `FEM_UPDATE_BASELINES=1`:
//!    ```bash
//!    FEM_UPDATE_BASELINES=1 cargo test --example mfem_ex1_poisson
//!    ```
//!    This writes a `.json` file to `tests/baselines/<test_name>.json`.
//!    **Commit the generated file** to the repository.
//!
//! 2. **Verify** — run tests normally (CI does this):
//!    ```bash
//!    cargo test --example mfem_ex1_poisson
//!    ```
//!    Each metric is compared against the stored baseline with
//!    relative + absolute tolerances. A mismatch **panics the test**.
//!
//! 3. **Update** — when results intentionally change (algorithm fix,
//!    new mesh, etc.), re-run step 1 and commit the updated baseline.
//!
//! # Example
//!
//! ```ignore
//! use fem_regression::regression;
//!
//! #[test]
//! fn ex1_regression() {
//!     let result = solve_case(8, 1, 1.0);
//!     regression("mfem_ex1_poisson")
//!         .check("l2_error_n8_p1", result.l2_error)
//!         .check_with("residual_n8_p1", result.final_residual, 1e-4, 1e-8)
//!         .finalize();
//! }
//! ```
//!
//! (The example above requires a baseline file; run with `FEM_UPDATE_BASELINES=1` to create one.)
//!
//! A minimal standalone doc-test showing the metric-building API:
//!
//! ```
//! use fem_regression::{regression, MetricEntry};
//! use std::collections::BTreeMap;
//!
//! // Demonstrate metric data structure — not a full regression check.
//! let metrics: BTreeMap<String, MetricEntry> = [
//!     ("pi".into(), MetricEntry { value: 3.14159, rtol: 1e-6, atol: 1e-10 }),
//! ].into();
//! assert!((metrics["pi"].value - std::f64::consts::PI).abs() < 1e-5);
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::fs;

// ─── Data types ─────────────────────────────────────────────────────────

/// A single numerical metric with its tolerance bounds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricEntry {
    /// The stored reference value.
    pub value: f64,
    /// Relative tolerance (default 1e-6).
    pub rtol: f64,
    /// Absolute tolerance (default 1e-10).
    pub atol: f64,
}

/// A complete baseline snapshot for one named test/example.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaselineFile {
    pub name: String,
    pub metrics: BTreeMap<String, MetricEntry>,
}

// ─── Builder ────────────────────────────────────────────────────────────

/// Builder for collecting metrics and finalising regression comparison.
pub struct RegressionCheck {
    name: String,
    baseline_path: PathBuf,
    metrics: BTreeMap<String, MetricEntry>,
}

/// Begin a regression check for the named test/example.
///
/// The baseline file is stored at `tests/baselines/{name}.json`.
pub fn regression(name: &str) -> RegressionCheck {
    let baseline_path = baseline_dir().join(format!("{}.json", name));
    RegressionCheck { name: name.into(), baseline_path, metrics: BTreeMap::new() }
}

impl RegressionCheck {
    /// Register a metric with default tolerances (rtol = 1e-6, atol = 1e-10).
    pub fn check(mut self, key: &str, value: f64) -> Self {
        self.metrics.insert(key.into(), MetricEntry { value, rtol: 1e-6, atol: 1e-10 });
        self
    }

    /// Register a metric with custom tolerances.
    pub fn check_with(mut self, key: &str, value: f64, rtol: f64, atol: f64) -> Self {
        self.metrics.insert(key.into(), MetricEntry { value, rtol, atol });
        self
    }

    /// Finalize the check.
    ///
    /// - If `FEM_UPDATE_BASELINES=1`: writes/updates the baseline file.
    /// - Otherwise: loads the stored baseline and compares every metric,
    ///   panicking on mismatches beyond tolerance.
    pub fn finalize(self) {
        if std::env::var("FEM_UPDATE_BASELINES").is_ok() {
            self.save();
        } else {
            self.verify();
        }
    }

    fn save(&self) {
        let file = BaselineFile {
            name: self.name.clone(),
            metrics: self.metrics.clone(),
        };
        if let Some(parent) = self.baseline_path.parent() {
            fs::create_dir_all(parent).expect("failed to create baseline directory");
        }
        let json = serde_json::to_string_pretty(&file)
            .expect("failed to serialize baseline");
        fs::write(&self.baseline_path, &json)
            .unwrap_or_else(|e| panic!("failed to write baseline to {}: {}", self.baseline_path.display(), e));
        eprintln!(
            "  [regression] BASELINE UPDATED: {}  ({} metrics)",
            self.baseline_path.display(),
            self.metrics.len()
        );
    }

    fn verify(&self) {
        let content = fs::read_to_string(&self.baseline_path)
            .unwrap_or_else(|_| {
                panic!(
                    "Regression baseline not found at {}.\n\
                     └─ Run with FEM_UPDATE_BASELINES=1 to create it.\n\
                     └─ e.g.: FEM_UPDATE_BASELINES=1 cargo test --example {}",
                    self.baseline_path.display(),
                    self.name
                );
            });

        let stored: BaselineFile = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("failed to parse baseline {}: {}", self.baseline_path.display(), e));

        let mut all_pass = true;

        for (key, current) in &self.metrics {
            match stored.metrics.get(key) {
                None => {
                    all_pass = false;
                    eprintln!(
                        "  [regression] MISSING: {}.{} — not in stored baseline",
                        self.name, key
                    );
                }
                Some(stored_entry) => {
                    let abs_diff = (current.value - stored_entry.value).abs();
                    let rel_diff = abs_diff / stored_entry.value.abs().max(f64::MIN_POSITIVE);
                    let atol = stored_entry.atol.max(current.atol); // use looser of the two
                    let rtol = stored_entry.rtol.max(current.rtol);

                    if abs_diff > atol && rel_diff > rtol {
                        all_pass = false;
                        eprintln!(
                            "  [regression] FAIL: {}.{}\n   \
                             current  = {:.10e}\n   \
                             baseline = {:.10e}\n   \
                             |diff|   = {:.3e}  (rtol={:.1e}, atol={:.1e})",
                            self.name, key,
                            current.value,
                            stored_entry.value,
                            abs_diff,
                            rtol,
                            atol,
                        );
                    }
                }
            }
        }

        for extra_key in stored.metrics.keys() {
            if !self.metrics.contains_key(extra_key) {
                // Not a failure — just informational. Stale baselines get cleaned up on update.
                eprintln!(
                    "  [regression] NOTE: {}.{} exists in stored baseline but was not checked",
                    self.name, extra_key
                );
            }
        }

        assert!(all_pass, "Regression baseline check FAILED for '{}'", self.name);
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Determine the baseline directory.
///
/// Order of precedence:
/// 1. `FEM_BASELINE_DIR` environment variable
/// 2. `tests/baselines/` relative to workspace root
fn baseline_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("FEM_BASELINE_DIR") {
        return PathBuf::from(dir);
    }
    workspace_root().join("tests").join("baselines")
}

/// Find the workspace root by walking up from cwd to find `Cargo.toml` containing `[workspace]`.
fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir()
        .expect("current_dir available");
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = fs::read_to_string(&candidate) {
                if content.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            // Fallback: current directory
            return std::env::current_dir().unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_baseline_path() -> PathBuf {
        std::env::temp_dir().join("fem_regression_test.json")
    }

    #[test]
    fn save_and_load_roundtrip() {
        let path = test_baseline_path();
        // Clean up from any previous run
        let _ = fs::remove_file(&path);

        // Save
        let check = RegressionCheck {
            name: "test_roundtrip".into(),
            baseline_path: path.clone(),
            metrics: BTreeMap::from([
                ("alpha".into(), MetricEntry { value: 1.234, rtol: 1e-6, atol: 1e-10 }),
                ("beta".into(), MetricEntry { value: 5.678e-5, rtol: 1e-4, atol: 1e-12 }),
            ]),
        };
        check.save();

        // Load and verify
        let loaded: BaselineFile = {
            let content = fs::read_to_string(&path).unwrap();
            serde_json::from_str(&content).unwrap()
        };
        assert_eq!(loaded.name, "test_roundtrip");
        assert_eq!(loaded.metrics.len(), 2);
        assert!((loaded.metrics["alpha"].value - 1.234).abs() < 1e-15);
        assert!((loaded.metrics["beta"].value - 5.678e-5).abs() < 1e-15);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn exact_match_passes() {
        // Build a baseline in a temp dir, then verify matching metrics
        let dir = std::env::temp_dir().join("fem_reg_test_exact");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("exact_match.json");

        let baseline = BaselineFile {
            name: "exact_match".into(),
            metrics: BTreeMap::from([
                ("x".into(), MetricEntry { value: 42.0, rtol: 1e-6, atol: 1e-10 }),
            ]),
        };
        fs::write(&path, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();

        // This should not panic
        RegressionCheck {
            name: "exact_match".into(),
            baseline_path: path.clone(),
            metrics: BTreeMap::from([
                ("x".into(), MetricEntry { value: 42.0, rtol: 1e-6, atol: 1e-10 }),
            ]),
        }
        .verify();

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "FAIL")]
    fn mismatch_panics() {
        let dir = std::env::temp_dir().join("fem_reg_test_mismatch");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("mismatch.json");

        let baseline = BaselineFile {
            name: "mismatch".into(),
            metrics: BTreeMap::from([
                ("x".into(), MetricEntry { value: 42.0, rtol: 1e-10, atol: 1e-12 }),
            ]),
        };
        fs::write(&path, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();

        RegressionCheck {
            name: "mismatch".into(),
            baseline_path: path.clone(),
            metrics: BTreeMap::from([
                ("x".into(), MetricEntry { value: 43.0, rtol: 1e-10, atol: 1e-12 }),
            ]),
        }
        .verify();

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn within_tolerance_passes() {
        let dir = std::env::temp_dir().join("fem_reg_test_tol");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("tol.json");

        let baseline = BaselineFile {
            name: "tol".into(),
            metrics: BTreeMap::from([
                ("x".into(), MetricEntry { value: 1.0, rtol: 1e-2, atol: 1e-12 }),
            ]),
        };
        fs::write(&path, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();

        // 1% relative tolerance should accept 1.009
        RegressionCheck {
            name: "tol".into(),
            baseline_path: path.clone(),
            metrics: BTreeMap::from([
                ("x".into(), MetricEntry { value: 1.009, rtol: 1e-2, atol: 1e-12 }),
            ]),
        }
        .verify();

        let _ = fs::remove_dir_all(&dir);
    }
}
