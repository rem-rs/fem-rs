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
