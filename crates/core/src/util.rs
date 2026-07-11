//! General-purpose numerical utilities used across fem-rs examples.
//!
//! These small functions are promoted here to eliminate the copy-paste
//! duplication found across many example files.

/// Weighted checksum for reproducibility testing.
///
/// Computes `Σ (i+1) · v[i]` — a linear functional that is sensitive to
/// both the magnitude and ordering of the entries.  Used in CI alignment
/// tests to detect numerical drift.
pub fn checksum(v: &[f64]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, &x)| (i as f64 + 1.0) * x)
        .sum()
}

/// Euclidean (L²) norm of a slice.
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Relative L² change between two slices: `‖b - a‖ / (1 + ‖a‖)`.
pub fn relative_change(a: &[f64], b: &[f64]) -> f64 {
    let diff: f64 = a.iter().zip(b.iter()).map(|(x, y)| (y - x).powi(2)).sum();
    let ref_norm: f64 = a.iter().map(|x| x * x).sum();
    diff.sqrt() / (1.0 + ref_norm.sqrt())
}

// ─── Spherical Bessel functions (jₙ, yₙ) ───────────────────────────────────
//
// Needed by ex25 (PML Maxwell) for the analytic solution of a spherical
// cavity.  Implemented via the upward recurrence for stability.

/// Spherical Bessel j₀(x) = sin(x)/x.
pub fn bessel_j0(x: f64) -> f64 {
    if x.abs() < 1e-60 { 1.0 } else { x.sin() / x }
}

/// Spherical Bessel j₁(x) = sin(x)/x² − cos(x)/x.
pub fn bessel_j1(x: f64) -> f64 {
    if x.abs() < 1e-60 { 0.0 } else { x.sin() / (x * x) - x.cos() / x }
}

/// Spherical Bessel j₂(x) = (3/x³ − 1/x)·sin(x) − (3/x²)·cos(x).
pub fn bessel_j2(x: f64) -> f64 {
    if x.abs() < 1e-60 { 0.0 }
    else { (3.0 / (x * x * x) - 1.0 / x) * x.sin() - (3.0 / (x * x)) * x.cos() }
}

/// Spherical Neumann y₀(x) = −cos(x)/x.
pub fn bessel_y0(x: f64) -> f64 {
    if x.abs() < 1e-60 { f64::NEG_INFINITY } else { -x.cos() / x }
}

/// Spherical Neumann y₁(x) = −cos(x)/x² − sin(x)/x.
pub fn bessel_y1(x: f64) -> f64 {
    if x.abs() < 1e-60 { f64::NEG_INFINITY } else { -x.cos() / (x * x) - x.sin() / x }
}

/// Spherical Neumann y₂(x) = (−3/x³ + 1/x)·cos(x) − (3/x²)·sin(x).
pub fn bessel_y2(x: f64) -> f64 {
    if x.abs() < 1e-60 { f64::NEG_INFINITY }
    else { (-3.0 / (x * x * x) + 1.0 / x) * x.cos() - (3.0 / (x * x)) * x.sin() }
}
