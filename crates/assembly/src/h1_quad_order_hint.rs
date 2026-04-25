//! Map legacy quadrature **hints** (from older reed-cpu-style call sites) to `fem-element`
//! reference-simplex rule **orders** for scalar H¹ assembly.
//!
//! Used by [`crate::assembler::Assembler`] callers directly and by the **`reed`** feature's
//! `FemCeed` coordinated path so triangle and tet rules stay consistent.

/// Map legacy `q` hint plus scalar `poly` (1 = P1, 2 = P2) to a reference-triangle quadrature **order**
/// (`tri_rule` in `fem-element`).
///
/// P2 bilinear forms use at least order `5` (7-point Dunavant) so mass / stiffness are not
/// under-integrated relative to the Lagrange assembly path.
pub fn h1_tri_quad_order(poly: usize, q_hint: usize) -> u8 {
    let baseline = if poly <= 1 { 3u8 } else { 5u8 };
    let from_hint = match q_hint {
        0..=1 => 1u8,
        2..=3 => 3,
        4..=7 => 5,
        _ => 6,
    };
    baseline.max(from_hint)
}

/// Map legacy `q` hint plus scalar `poly` (1 = P1, 2 = P2) to a reference-tet quadrature **order**
/// (`tet_rule` in `fem-element`).
///
/// P2 bilinear forms use at least order `5` (10-point Grundmann-Moller, exact through degree 5).
pub fn h1_tet_quad_order(poly: usize, q_hint: usize) -> u8 {
    let baseline = if poly <= 1 { 2u8 } else { 5u8 };
    let from_hint = match q_hint {
        0..=1 => 1u8,
        2..=3 => 2,
        4..=7 => 5,
        _ => 6,
    };
    baseline.max(from_hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h1_tri_quad_order_respects_p2_baseline_and_hint_monotone() {
        assert!(h1_tri_quad_order(1, 1) >= 3);
        assert!(h1_tri_quad_order(2, 1) >= 5);
        assert!(h1_tri_quad_order(1, 7) >= h1_tri_quad_order(1, 3));
        assert!(h1_tri_quad_order(2, 8) >= h1_tri_quad_order(2, 7));
    }

    #[test]
    fn h1_tet_quad_order_respects_p2_baseline_and_hint_monotone() {
        assert!(h1_tet_quad_order(1, 1) >= 2);
        assert!(h1_tet_quad_order(2, 1) >= 5);
        assert!(h1_tet_quad_order(1, 7) >= h1_tet_quad_order(1, 3));
        assert!(h1_tet_quad_order(2, 8) >= h1_tet_quad_order(2, 7));
    }
}
