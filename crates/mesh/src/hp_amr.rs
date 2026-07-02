//! hp-Adaptive Mesh Refinement decision making.
//!
//! Combines h-refinement (element subdivision) and p-refinement (order elevation)
//! based on element-wise error estimates and a smoothness indicator.
//!
//! ## hp-Marking strategy (Houston–Süli)
//!
//! 1. Compute element error indicators η_K (e.g., via ZZ or residual estimator).
//! 2. Mark elements with η_K ≥ θ·max(η) for hp-treatment.
//! 3. For each marked element, compute a smoothness indicator κ_K:
//!    - κ_K near 0 → solution is smooth → p-refinement recommended
//!    - κ_K large → solution is singular → h-refinement recommended
//!    - Otherwise → hp-refinement (both h and p)
//! 4. Smooth the order field to limit p-jumps between adjacent elements.
//!
//! # Reference
//! P. Houston and E. Süli, "hp-Adaptive Discontinuous Galerkin Finite Element
//! Methods for First-Order Hyperbolic Problems", SIAM J. Sci. Comput., 2001.

use fem_core::ElemId;

/// Action to take for an element in hp-adaptive refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpAction {
    /// Subdivide the element (h-refinement).
    H,
    /// Elevate the polynomial order (p-refinement).
    P,
    /// Both: subdivide and elevate the children's order.
    HP,
}

/// Mark elements for hp-refinement based on error indicators and smoothness.
///
/// # Arguments
/// * `eta` — element-wise error indicators (length = n_elems)
/// * `theta_mark` — fraction of max(η) used as marking threshold (typically 0.3–0.5)
/// * `smoothness` — per-element smoothness indicator κ_K (higher = less smooth)
/// * `theta_h` — smoothness threshold for h-refinement (above this → H or HP)
/// * `theta_p` — smoothness threshold for p-refinement (below this → P)
///
/// # Returns
/// A vector of `(elem_id, action)` pairs for elements that need refinement.
///
/// # Panics
/// Panics if `eta.len() != smoothness.len()`.
pub fn hp_mark(
    eta: &[f64],
    theta_mark: f64,
    smoothness: &[f64],
    theta_h: f64,
    theta_p: f64,
) -> Vec<(ElemId, HpAction)> {
    assert_eq!(eta.len(), smoothness.len(),
        "hp_mark: eta and smoothness must have same length");

    let n_elems = eta.len();
    if n_elems == 0 {
        return Vec::new();
    }

    // 1. Compute marking threshold
    let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = theta_mark.clamp(0.0, 1.0) * max_eta;

    // 2. Mark elements above threshold
    let mut result: Vec<(ElemId, HpAction)> = Vec::new();
    for (i, (&e, &k)) in eta.iter().zip(smoothness.iter()).enumerate() {
        if e < cutoff {
            continue;
        }

        let action = if k <= theta_p {
            HpAction::P
        } else if k >= theta_h {
            HpAction::H
        } else {
            HpAction::HP
        };

        result.push((i as ElemId, action));
    }

    result
}

/// Compute a smoothness indicator for each element using the decay rate of
/// Legendre expansion coefficients (Houston–Süli type).
///
/// The indicator κ_K estimates how smooth the solution is within each element.
/// Higher values indicate less smoothness (potential singularities).
///
/// For P1 FE spaces, this is a simplified proxy using the ratio of the
/// estimated error to the element's gradient variation.
///
/// # Arguments
/// * `eta` — element-wise error indicators
/// * `elem_grad_variation` — gradient variation per element (computed from
///   recovered nodal gradients)
///
/// # Returns
/// Smoothness indicator κ_K per element. Values near 0 = smooth, > 1 = singular.
pub fn compute_smoothness_indicator(
    eta: &[f64],
    elem_grad_variation: &[f64],
) -> Vec<f64> {
    assert_eq!(eta.len(), elem_grad_variation.len(),
        "compute_smoothness_indicator: length mismatch");

    let n_elems = eta.len();
    let mut kappa = vec![0.0_f64; n_elems];

    for i in 0..n_elems {
        let gv = elem_grad_variation[i].abs();
        if gv > 1e-30 {
            // Ratio of error to gradient variation:
            // High ratio → error is large relative to gradient → likely singular
            // Low ratio → error is small relative to gradient → likely smooth
            kappa[i] = eta[i] / gv;
        } else {
            kappa[i] = 0.0; // constant solution → very smooth
        }
    }

    // Normalize to [0, 1] range
    let max_k = kappa.iter().cloned().fold(0.0_f64, f64::max);
    if max_k > 1e-30 {
        for k in &mut kappa {
            *k = (*k / max_k).min(1.0);
        }
    }

    kappa
}

/// Mark elements for p-refinement based on a smoothness indicator.
///
/// Elements with smoothness κ_K ≤ `theta_p` are recommended for p-refinement
/// (they are smooth enough that order elevation will be effective).
///
/// # Returns
/// Indices of elements recommended for p-refinement.
pub fn mark_smooth_for_p_refinement(
    smoothness: &[f64],
    theta_p: f64,
) -> Vec<ElemId> {
    smoothness.iter().enumerate()
        .filter(|(_, &k)| k <= theta_p)
        .map(|(i, _)| i as ElemId)
        .collect()
}

/// Mark elements for h-refinement based on a smoothness indicator.
///
/// Elements with smoothness κ_K ≥ `theta_h` are recommended for h-refinement
/// (they are rough/singular and subdivision will resolve the singularity).
///
/// # Returns
/// Indices of elements recommended for h-refinement.
pub fn mark_rough_for_h_refinement(
    smoothness: &[f64],
    theta_h: f64,
) -> Vec<ElemId> {
    smoothness.iter().enumerate()
        .filter(|(_, &k)| k >= theta_h)
        .map(|(i, _)| i as ElemId)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hp_mark_all_smooth_gives_p() {
        let eta = vec![1.0, 0.8, 0.6, 0.3, 0.1];
        let smoothness = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let result = hp_mark(&eta, 0.5, &smoothness, 0.7, 0.3);
        // Elements 0, 1, 2 are above threshold; all smooth → P
        for &(_, action) in &result {
            assert_eq!(action, HpAction::P, "all smooth → P");
        }
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn hp_mark_all_rough_gives_h() {
        let eta = vec![1.0, 0.8, 0.6, 0.3, 0.1];
        let smoothness = vec![0.9, 0.9, 0.9, 0.9, 0.9];
        let result = hp_mark(&eta, 0.5, &smoothness, 0.7, 0.3);
        for &(_, action) in &result {
            assert_eq!(action, HpAction::H, "all rough → H");
        }
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn hp_mark_mixed_actions() {
        let eta = vec![1.0, 0.9, 0.8, 0.1];
        // smooth, intermediate, rough
        let smoothness = vec![0.1, 0.5, 0.9, 0.1];
        let result = hp_mark(&eta, 0.5, &smoothness, 0.7, 0.3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].1, HpAction::P, "smooth → P");
        assert_eq!(result[1].1, HpAction::HP, "intermediate → HP");
        assert_eq!(result[2].1, HpAction::H, "rough → H");
    }

    #[test]
    fn hp_mark_empty_input() {
        let result = hp_mark(&[], 0.5, &[], 0.7, 0.3);
        assert!(result.is_empty());
    }

    #[test]
    fn hp_mark_below_threshold_not_marked() {
        let eta = vec![0.1, 0.1, 1.0];
        let smoothness = vec![0.1, 0.1, 0.1];
        let result = hp_mark(&eta, 0.5, &smoothness, 0.7, 0.3);
        assert_eq!(result.len(), 1, "only the highest-error element");
        assert_eq!(result[0].0, 2);
    }

    #[test]
    fn smoothness_indicator_constant_is_zero() {
        let eta = vec![0.0; 5];
        let gv = vec![0.0; 5];
        let k = compute_smoothness_indicator(&eta, &gv);
        for &ki in &k {
            assert!((ki - 0.0).abs() < 1e-15, "constant solution → κ=0");
        }
    }

    #[test]
    fn smoothness_indicator_normalized() {
        let eta = vec![1.0, 2.0];
        let gv = vec![1.0, 1.0];
        let k = compute_smoothness_indicator(&eta, &gv);
        assert!((k[0] - 0.5).abs() < 1e-15, "κ[0] should be 0.5, got {}", k[0]);
        assert!((k[1] - 1.0).abs() < 1e-15, "κ[1] should be 1.0, got {}", k[1]);
    }

    #[test]
    fn mark_smooth_works() {
        let k = vec![0.1, 0.2, 0.8, 0.9];
        let p = mark_smooth_for_p_refinement(&k, 0.3);
        assert_eq!(p, vec![0, 1], "elements with κ≤0.3");
    }

    #[test]
    fn mark_rough_works() {
        let k = vec![0.1, 0.2, 0.8, 0.9];
        let h = mark_rough_for_h_refinement(&k, 0.7);
        assert_eq!(h, vec![2, 3], "elements with κ≥0.7");
    }
}
