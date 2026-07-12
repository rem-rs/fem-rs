//! Shared utility functions for IGA modules.
//!
//! Consolidated definitions used across `iga.rs`, `iga_bezier.rs`,
//! `iga_gpu.rs`, and `contact_iga.rs` — eliminates four copies of
//! [`nonempty_spans`](fn@nonempty_spans).

use fem_element::quadrature::seg_rule;

/// Return `(span_index, left, right)` for each non-empty knot span.
///
/// `knots` must be a non-decreasing knot vector with at least 2 entries.
/// Only spans with `knots[span+1] > knots[span]` are returned (zero-length
/// spans from repeated knots are skipped).
pub fn nonempty_spans(knots: &[f64]) -> Vec<(usize, f64, f64)> {
    knots
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| if w[1] > w[0] { Some((i, w[0], w[1])) } else { None })
        .collect()
}

/// Gauss–Legendre points and weights on `[0, 1]` for a given quadrature order.
///
/// Delegates to [`seg_rule`] from the element crate.
pub fn gauss_01(order: u8) -> (Vec<f64>, Vec<f64>) {
    let seg = seg_rule(order);
    let pts: Vec<f64> = seg.points.iter().map(|p| p[0]).collect();
    (pts, seg.weights)
}
