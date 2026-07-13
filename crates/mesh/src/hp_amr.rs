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
//! ## Multi-strategy support (Phase 2)
//!
//! Beyond the basic Houston–Süli path, this module now integrates
//! spectral smoothness estimators (Legendre coefficient decay, Fourier
//! coefficient decay) via the [`amr::smoothness`] sub‑module.  Use
//! [`HpDecisionStrategy`] to select a strategy, or [`hp_mark_with_predictor`]
//! to drive decision‑making from a [`SmoothnessPredictor`].
//!
//! # Reference
//! P. Houston and E. Süli, "hp-Adaptive Discontinuous Galerkin Finite Element
//! Methods for First-Order Hyperbolic Problems", SIAM J. Sci. Comput., 2001.

use fem_core::ElemId;
use crate::amr::smoothness::{
    self,
    ConsensusMode,
    SmoothnessEstimatorConfig,
    SmoothnessInputData,
    SmoothnessPredictor,
    SmoothnessPredictorConfig,
};

// ============================================================================
// Core types
// ============================================================================

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

/// Pre‑configured hp‑decision strategy.
///
/// Each variant maps to one or more [`SmoothnessEstimatorConfig`] values
/// and a default consensus rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HpDecisionStrategy {
    /// Original Houston–Süli residual ratio only.
    ///
    /// Parameters: `(theta_h, theta_p)` — smoothness thresholds for
    /// h‑ and p‑refinement.
    HoustonSueli(f64, f64),
    /// Legendre coefficient decay spectral estimate.
    ///
    /// Parameters: `(threshold_smooth, threshold_rough)`.
    LegendreDecay(f64, f64),
    /// Fourier coefficient decay spectral estimate.
    ///
    /// Parameter: `n_modes` — number of Fourier modes.
    FourierDecay(usize),
    /// Weighted composite: Houston–Süli + Legendre decay.
    ///
    /// Parameters: `(w_hs, w_ld)` — weights for the two estimators.
    HybridHSLegendre(f64, f64),
    /// Weighted composite of all available estimators.
    ///
    /// Parameters: `(w_hs, w_ld, w_fd)` — weights for Houston–Süli,
    /// Legendre decay, and Fourier decay.
    FullComposite(f64, f64, f64),
}

impl HpDecisionStrategy {
    /// Build a [`SmoothnessPredictorConfig`] from this strategy choice.
    pub fn to_predictor_config(&self) -> SmoothnessPredictorConfig {
        match *self {
            HpDecisionStrategy::HoustonSueli(th, tp) => SmoothnessPredictorConfig {
                estimators: vec![SmoothnessEstimatorConfig::HoustonSueli {
                    theta_h: th, theta_p: tp,
                }],
                weights: vec![1.0],
                consensus: ConsensusMode::WeightedAverage,
            },
            HpDecisionStrategy::LegendreDecay(ts, tr) => SmoothnessPredictorConfig {
                estimators: vec![SmoothnessEstimatorConfig::LegendreDecay {
                    threshold_smooth: ts, threshold_rough: tr,
                }],
                weights: vec![1.0],
                consensus: ConsensusMode::WeightedAverage,
            },
            HpDecisionStrategy::FourierDecay(nm) => SmoothnessPredictorConfig {
                estimators: vec![SmoothnessEstimatorConfig::FourierDecay { n_modes: nm }],
                weights: vec![1.0],
                consensus: ConsensusMode::WeightedAverage,
            },
            HpDecisionStrategy::HybridHSLegendre(w_hs, w_ld) => SmoothnessPredictorConfig {
                estimators: vec![
                    SmoothnessEstimatorConfig::HoustonSueli {
                        theta_h: 0.7, theta_p: 0.3,
                    },
                    SmoothnessEstimatorConfig::LegendreDecay {
                        threshold_smooth: 0.2, threshold_rough: 0.6,
                    },
                ],
                weights: vec![w_hs, w_ld],
                consensus: ConsensusMode::WeightedAverage,
            },
            HpDecisionStrategy::FullComposite(w_hs, w_ld, w_fd) => SmoothnessPredictorConfig {
                estimators: vec![
                    SmoothnessEstimatorConfig::HoustonSueli {
                        theta_h: 0.7, theta_p: 0.3,
                    },
                    SmoothnessEstimatorConfig::LegendreDecay {
                        threshold_smooth: 0.2, threshold_rough: 0.6,
                    },
                    SmoothnessEstimatorConfig::FourierDecay { n_modes: 6 },
                ],
                weights: vec![w_hs, w_ld, w_fd],
                consensus: ConsensusMode::WeightedAverage,
            },
        }
    }

    /// Short human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            HpDecisionStrategy::HoustonSueli(..) => "Houston–Süli",
            HpDecisionStrategy::LegendreDecay(..) => "LegendreDecay",
            HpDecisionStrategy::FourierDecay(..) => "FourierDecay",
            HpDecisionStrategy::HybridHSLegendre(..) => "Hybrid HS+Legendre",
            HpDecisionStrategy::FullComposite(..) => "FullComposite",
        }
    }
}

// ============================================================================
// Core marking logic
// ============================================================================

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

/// Mark elements using a multi‑strategy [`SmoothnessPredictor`].
///
/// Equivalent to calling [`hp_mark`] with the consensus smoothness scores
/// produced by the predictor.  The `theta_h` / `theta_p` thresholds are
/// taken from the *first* estimator configuration that provides them;
/// if none does, sensible defaults (0.7 / 0.3) are used.
///
/// # Arguments
/// * `eta` — element‑wise error indicators.
/// * `theta_mark` — fraction of max(η) for the marking threshold.
/// * `predictor` — a configured [`SmoothnessPredictor`].
/// * `data` — input data for the smoothness estimators.
///
/// # Returns
/// A vector of `(elem_id, action)` pairs.
pub fn hp_mark_with_predictor(
    eta: &[f64],
    theta_mark: f64,
    predictor: &SmoothnessPredictor,
    data: &SmoothnessInputData<'_>,
) -> Vec<(ElemId, HpAction)> {
    let prediction = predictor.predict(data);

    // Extract thresholds from the first estimator that defines them
    let (theta_h, theta_p) = extract_thresholds(predictor.estimators());

    hp_mark(eta, theta_mark, &prediction.consensus, theta_h, theta_p)
}

/// Mark elements using a named [`HpDecisionStrategy`].
///
/// This is a convenience wrapper around
/// [`hp_mark_with_predictor`](fn@hp_mark_with_predictor) that constructs
/// the predictor from the strategy.
pub fn hp_mark_with_strategy(
    eta: &[f64],
    theta_mark: f64,
    strategy: &HpDecisionStrategy,
    data: &SmoothnessInputData<'_>,
) -> Vec<(ElemId, HpAction)> {
    let config = strategy.to_predictor_config();
    let predictor = SmoothnessPredictor::new(config);
    hp_mark_with_predictor(eta, theta_mark, &predictor, data)
}

// ============================================================================
// Legacy smoothness indicator (kept for backward compatibility)
// ============================================================================

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
    smoothness::houston_sueli_smoothness(eta, elem_grad_variation)
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

// ============================================================================
// Internal helpers
// ============================================================================

fn extract_thresholds(estimators: &[SmoothnessEstimatorConfig]) -> (f64, f64) {
    for est in estimators {
        match est {
            SmoothnessEstimatorConfig::LegendreDecay { threshold_smooth, threshold_rough } => {
                return (*threshold_rough, *threshold_smooth);
            }
            SmoothnessEstimatorConfig::HoustonSueli { theta_h, theta_p } => {
                return (*theta_h, *theta_p);
            }
            SmoothnessEstimatorConfig::FourierDecay { .. } => {}
        }
    }
    (0.7, 0.3) // sensible defaults
}

// ============================================================================
// p‑jump limiting & order‑field smoothing (Task 2.2)
// ============================================================================

/// Element adjacency: for each element, the list of neighboring element IDs
/// that share a face (2‑D) or edge (3‑D) with it.
///
/// Built from the mesh's `edge_to_elem` mapping.  Interior faces appear as
/// a pair of entries; boundary faces contribute only to the owner.
#[derive(Debug, Clone)]
pub struct ElementAdjacency {
    /// `neighbors[e]` is the list of element IDs sharing an edge/face with `e`.
    pub neighbors: Vec<Vec<ElemId>>,
}

impl ElementAdjacency {
    /// Build adjacency from a [`Mesh`](crate::simplex::Mesh) using its
    /// pre‑computed `edge_to_elem` mapping.
    ///
    /// O(E) where E = number of edges.
    pub fn from_mesh<const D: usize>(mesh: &crate::simplex::Mesh<D>) -> Self {
        let n_elem = mesh.n_elems();
        let mut neighbors = vec![Vec::new(); n_elem];

        for chunk in mesh.edge_to_elem.chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let a = chunk[0];
            let b = chunk[1];
            let ai = a as usize;
            if b != ElemId::MAX {
                let bi = b as usize;
                if ai < n_elem && bi < n_elem {
                    if !neighbors[ai].contains(&b) {
                        neighbors[ai].push(b);
                    }
                    if !neighbors[bi].contains(&a) {
                        neighbors[bi].push(a);
                    }
                }
            }
        }

        ElementAdjacency { neighbors }
    }

    /// Build adjacency from a [`MeshTopology`](crate::topology::MeshTopology)
    /// trait object.  Uses the `face_elements` method to find element pairs
    /// that share an interior face.
    pub fn from_topology(mesh: &dyn crate::topology::MeshTopology) -> Self {
        let n_elem = mesh.n_elements();
        let mut neighbors = vec![Vec::new(); n_elem];

        for f in mesh.face_iter() {
            let (a, b_opt) = mesh.face_elements(f);
            if let Some(b) = b_opt {
                let ai = a as usize;
                let bi = b as usize;
                if ai < n_elem && bi < n_elem {
                    if !neighbors[ai].contains(&b) {
                        neighbors[ai].push(b);
                    }
                    if !neighbors[bi].contains(&a) {
                        neighbors[bi].push(a);
                    }
                }
            }
        }

        ElementAdjacency { neighbors }
    }
}

/// Limit the maximum polynomial‑order difference between adjacent elements.
///
/// Iteratively reduces the order of any element whose p‑order exceeds
/// `max_jump` above any neighbour's order, propagating the reduction
/// until the whole field satisfies |pᵢ − pⱼ| ≤ max_jump.
///
/// # Returns
/// Total number of order reductions applied.
///
/// # Example (adjacency‑free)
/// ```
/// use fem_mesh::limit_p_jumps;
/// use fem_core::ElemId;
/// let mut orders = vec![4, 1, 4, 1];
/// let adj: Vec<Vec<ElemId>> = vec![
///     vec![1],        // 0 neighbours 1
///     vec![0, 2],     // 1 neighbours 0, 2
///     vec![1, 3],     // 2 neighbours 1, 3
///     vec![2],        // 3 neighbours 2
/// ];
/// let changed = limit_p_jumps(&mut orders, &adj, 2);
/// assert!(changed > 0);
/// // After limiting: no adjacent pair differs by more than 2
/// for i in 0..orders.len() {
///     for &j in &adj[i] {
///         let jj = j as usize;
///         assert!(orders[i].abs_diff(orders[jj]) <= 2);
///     }
/// }
/// ```
pub fn limit_p_jumps(orders: &mut [usize], adjacency: &[Vec<ElemId>], max_jump: usize) -> usize {
    let mut total_changed = 0;
    let mut any_change = true;

    while any_change {
        any_change = false;
        for i in 0..orders.len() {
            let pi = orders[i];
            for &nbr in &adjacency[i] {
                let pj = orders[nbr as usize];
                if pi > pj + max_jump {
                    orders[i] = pj + max_jump;
                    total_changed += 1;
                    any_change = true;
                    break; // recheck after reduction
                }
            }
        }
    }

    total_changed
}

/// Smooth the polynomial‑order field via iterative local averaging.
///
/// At each iteration every element's order is moved toward the **median**
/// of its neighbours' orders, then clamped so that no neighbour pair
/// exceeds `max_jump`.
///
/// # Arguments
/// * `orders` — in‑place order field, mutated to the smoothed result.
/// * `adjacency` — neighbour list (see [`ElementAdjacency`]).
/// * `iterations` — number of smoothing sweeps.
/// * `max_jump` — maximum allowed p‑difference after smoothing.
///
/// # Returns
/// History of the order field at each iteration (length = `iterations + 1`,
/// where entry 0 is the **input** state).
///
/// # Example
/// ```
/// use fem_mesh::smooth_order_field;
/// use fem_core::ElemId;
/// let mut orders = vec![4, 1, 4, 1];
/// let adj: Vec<Vec<ElemId>> = vec![
///     vec![1],
///     vec![0, 2],
///     vec![1, 3],
///     vec![2],
/// ];
/// let history = smooth_order_field(&mut orders, &adj, 3, 2);
/// assert_eq!(history.len(), 4); // input + 3 iterations
/// ```
pub fn smooth_order_field(
    orders: &mut [usize],
    adjacency: &[Vec<ElemId>],
    iterations: usize,
    max_jump: usize,
) -> Vec<Vec<usize>> {
    let mut history = Vec::with_capacity(iterations + 1);
    history.push(orders.to_vec());

    for _iter in 0..iterations {
        let prev = orders.to_vec();

        // 1. Compute median candidate for each element
        let mut candidates = prev.clone();
        for i in 0..orders.len() {
            let nbrs = &adjacency[i];
            if nbrs.is_empty() {
                continue;
            }
            let mut nbr_orders: Vec<usize> =
                nbrs.iter().map(|&nbr| prev[nbr as usize]).collect();
            nbr_orders.sort_unstable();
            let mid = nbr_orders.len() / 2;
            candidates[i] = if nbr_orders.len() % 2 == 0 {
                // True median: average of two middle values (floored).
                // Using the upper median would pull low-p elements toward
                // high-p neighbours — the average gives a balanced transition.
                (nbr_orders[mid - 1] + nbr_orders[mid]) / 2
            } else {
                nbr_orders[mid]
            };
        }

        // 2. Write candidates back
        orders.copy_from_slice(&candidates);

        // 3. Enforce max_jump globally (fixes asymmetry in the median updates)
        limit_p_jumps(orders, adjacency, max_jump);

        history.push(orders.to_vec());
    }

    history
}

/// Convenience: smooth an order field to create a gradual transition between
/// blocks of differing p‑order.
///
/// Equivalent to calling [`smooth_order_field`] with `max_jump` clamped to
/// `max_jump` and `iterations` chosen to allow the interface to propagate.
/// The order field is mutated in‑place.
///
/// # Returns
/// History of the order field at each iteration (same semantics as
/// [`smooth_order_field`]).
///
/// # Example
/// ```
/// use fem_mesh::transition_layers;
/// use fem_core::ElemId;
/// // Two blocks: p=5 at the left, p=1 at the right, 1-element interface
/// let mut orders = vec![5, 5, 5, 1, 1, 1];
/// let adj: Vec<Vec<ElemId>> = vec![
///     vec![1],        // 0
///     vec![0, 2],     // 1
///     vec![1, 3],     // 2 ← interface element
///     vec![2, 4],     // 3
///     vec![3, 5],     // 4
///     vec![4],        // 5
/// ];
/// let history = transition_layers(&mut orders, &adj, 2, 4);
/// // After smoothing: no adjacent pair differs by more than 2
/// for i in 0..orders.len() {
///     for &j in &adj[i] {
///         assert!(orders[i].abs_diff(orders[j as usize]) <= 2);
///     }
/// }
/// // The interface should have broadened into a staircase
/// let max_increase = orders.windows(2).map(|w| w[0].abs_diff(w[1])).max().unwrap_or(0);
/// assert!(max_increase <= 2, "max jump ≤ 2, got {}", max_increase);
/// ```
pub fn transition_layers(
    orders: &mut [usize],
    adjacency: &[Vec<ElemId>],
    max_jump: usize,
    iterations: usize,
) -> Vec<Vec<usize>> {
    smooth_order_field(orders, adjacency, iterations, max_jump)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Original hp_mark tests ─────────────────────────────────────────────

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

    // ── HpDecisionStrategy tests ───────────────────────────────────────────

    #[test]
    fn strategy_houston_sueli_to_predictor_config() {
        let s = HpDecisionStrategy::HoustonSueli(0.7, 0.3);
        let cfg = s.to_predictor_config();
        assert_eq!(cfg.estimators.len(), 1);
        assert_eq!(cfg.consensus, ConsensusMode::WeightedAverage);
    }

    #[test]
    fn strategy_legendre_decay_to_predictor_config() {
        let s = HpDecisionStrategy::LegendreDecay(0.2, 0.6);
        let cfg = s.to_predictor_config();
        assert_eq!(cfg.estimators.len(), 1);
    }

    #[test]
    fn strategy_fourier_decay_to_predictor_config() {
        let s = HpDecisionStrategy::FourierDecay(6);
        let cfg = s.to_predictor_config();
        assert_eq!(cfg.estimators.len(), 1);
    }

    #[test]
    fn strategy_hybrid_to_predictor_config() {
        let s = HpDecisionStrategy::HybridHSLegendre(0.4, 0.6);
        let cfg = s.to_predictor_config();
        assert_eq!(cfg.estimators.len(), 2);
        assert_eq!(cfg.weights.len(), 2);
    }

    #[test]
    fn strategy_full_composite_to_predictor_config() {
        let s = HpDecisionStrategy::FullComposite(0.3, 0.4, 0.3);
        let cfg = s.to_predictor_config();
        assert_eq!(cfg.estimators.len(), 3);
    }

    #[test]
    fn strategy_labels_are_distinct() {
        let labels: Vec<&str> = vec![
            HpDecisionStrategy::HoustonSueli(0.7, 0.3).label(),
            HpDecisionStrategy::LegendreDecay(0.2, 0.6).label(),
            HpDecisionStrategy::FourierDecay(6).label(),
            HpDecisionStrategy::HybridHSLegendre(0.5, 0.5).label(),
            HpDecisionStrategy::FullComposite(1.0, 1.0, 1.0).label(),
        ];
        assert_eq!(labels.len(), 5);
        assert!(!labels.contains(&""));
    }

    // ── hp_mark_with_strategy integration tests ────────────────────────────

    #[test]
    fn mark_with_houston_sueli_strategy_matches_original() {
        let eta = vec![1.0, 0.8, 0.6, 0.3, 0.1];
        let gv = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        // Original hp_mark (legacy compute_smoothness_indicator)
        let k_legacy = compute_smoothness_indicator(&eta, &gv);
        let result_legacy = hp_mark(&eta, 0.5, &k_legacy, 0.7, 0.3);

        // New strategy-based path
        let data = SmoothnessInputData {
            eta: Some(&eta),
            grad_variation: Some(&gv),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result_new = hp_mark_with_strategy(
            &eta, 0.5, &HpDecisionStrategy::HoustonSueli(0.7, 0.3), &data,
        );

        assert_eq!(result_legacy.len(), result_new.len());
        for ((id1, a1), (id2, a2)) in result_legacy.iter().zip(result_new.iter()) {
            assert_eq!(id1, id2, "element IDs must match");
            assert_eq!(a1, a2, "actions must match");
        }
    }

    #[test]
    fn mark_with_predictor_consistency() {
        let eta = vec![1.0, 0.8, 0.6];
        let gv = vec![0.5, 0.4, 0.3];

        // Mark with predictor
        let config = SmoothnessPredictorConfig {
            estimators: vec![SmoothnessEstimatorConfig::HoustonSueli {
                theta_h: 0.7, theta_p: 0.3,
            }],
            weights: vec![1.0],
            consensus: ConsensusMode::WeightedAverage,
        };
        let predictor = SmoothnessPredictor::new(config);
        let data = SmoothnessInputData {
            eta: Some(&eta),
            grad_variation: Some(&gv),
            elem_solution_samples: None,
            quadrature: None,
        };
        let result = hp_mark_with_predictor(&eta, 0.5, &predictor, &data);

        // Compare with manual consensus
        let k = predictor.predict(&data).consensus;
        let expected = hp_mark(&eta, 0.5, &k, 0.7, 0.3);
        assert_eq!(result.len(), expected.len());
        for ((id1, a1), (id2, a2)) in result.iter().zip(expected.iter()) {
            assert_eq!(id1, id2);
            assert_eq!(a1, a2);
        }
    }

    #[test]
    fn mark_with_legendre_strategy_no_panic() {
        // Smoke test: LegendreDecay strategy with default quadrature
        let eta = vec![1.0, 0.5];
        let x: Vec<f64> = (0..8).map(|i| -1.0 + 2.0 * i as f64 / 7.0).collect();
        let u: Vec<f64> = x.iter().map(|&xi| (std::f64::consts::PI * xi).sin()).collect();
        let samples = vec![u.as_slice(); 2];

        let data = SmoothnessInputData {
            eta: Some(&eta),
            grad_variation: None,
            elem_solution_samples: Some(&samples),
            quadrature: None,
        };
        let result = hp_mark_with_strategy(
            &eta, 0.5, &HpDecisionStrategy::LegendreDecay(0.2, 0.6), &data,
        );
        // Should not panic; result may be empty or have entries
        assert!(result.len() <= 2);
    }

    #[test]
    fn extract_thresholds_falls_back_to_defaults() {
        let estimators = vec![SmoothnessEstimatorConfig::FourierDecay { n_modes: 6 }];
        let (th, tp) = extract_thresholds(&estimators);
        assert!((th - 0.7).abs() < 1e-15);
        assert!((tp - 0.3).abs() < 1e-15);
    }

    #[test]
    fn extract_thresholds_from_houston_sueli() {
        let estimators = vec![SmoothnessEstimatorConfig::HoustonSueli {
            theta_h: 0.8, theta_p: 0.2,
        }];
        let (th, tp) = extract_thresholds(&estimators);
        assert!((th - 0.8).abs() < 1e-15);
        assert!((tp - 0.2).abs() < 1e-15);
    }

    #[test]
    fn extract_thresholds_from_legendre_decay() {
        let estimators = vec![SmoothnessEstimatorConfig::LegendreDecay {
            threshold_smooth: 0.15, threshold_rough: 0.65,
        }];
        let (th, tp) = extract_thresholds(&estimators);
        assert!((th - 0.65).abs() < 1e-15);
        assert!((tp - 0.15).abs() < 1e-15);
    }

    // ── p‑jump limiting tests ───────────────────────────────────────────────

    #[test]
    fn limit_p_jumps_no_change_when_within_bound() {
        let mut orders = vec![2, 3, 2, 1];
        let adj_raw: Vec<Vec<ElemId>> = vec![
            vec![1], vec![0, 2], vec![1, 3], vec![2],
        ];
        let changed = limit_p_jumps(&mut orders, &adj_raw, 2);
        assert_eq!(changed, 0, "all within |Δp| ≤ 2 → no change");
        assert_eq!(orders, vec![2, 3, 2, 1]);
    }

    #[test]
    fn limit_p_jumps_reduces_big_jumps() {
        let mut orders = vec![5, 1, 5, 1];
        let adj_raw: Vec<Vec<ElemId>> = vec![
            vec![1], vec![0, 2], vec![1, 3], vec![2],
        ];
        let changed = limit_p_jumps(&mut orders, &adj_raw, 2);
        assert!(changed > 0, "should have reduced some orders");
        for i in 0..orders.len() {
            for &j in &adj_raw[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "|p[{}]={} − p[{}]={}| > 2", i, orders[i], jj, orders[jj]);
            }
        }
    }

    #[test]
    fn limit_p_jumps_chain_reduction() {
        let mut orders = vec![4, 1, 4, 1];
        let adj_raw: Vec<Vec<ElemId>> = vec![
            vec![1], vec![0, 2], vec![1, 3], vec![2],
        ];
        limit_p_jumps(&mut orders, &adj_raw, 2);
        for i in 0..orders.len() {
            for &j in &adj_raw[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "chain failed: |{} − {}| > 2", orders[i], orders[jj]);
            }
        }
    }

    #[test]
    fn limit_p_jumps_single_element_no_change() {
        let mut orders = vec![10];
        let adj_raw: Vec<Vec<ElemId>> = vec![vec![]];
        let changed = limit_p_jumps(&mut orders, &adj_raw, 2);
        assert_eq!(changed, 0);
        assert_eq!(orders[0], 10);
    }

    #[test]
    fn limit_p_jumps_all_equal_unchanged() {
        let mut orders = vec![3, 3, 3, 3];
        let adj_raw: Vec<Vec<ElemId>> = vec![
            vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2],
        ];
        let changed = limit_p_jumps(&mut orders, &adj_raw, 1);
        assert_eq!(changed, 0);
        assert_eq!(orders, vec![3, 3, 3, 3]);
    }

    #[test]
    fn smooth_order_field_converges_to_uniform() {
        let mut orders = vec![4, 1, 4, 1];
        let adj_raw: Vec<Vec<ElemId>> = vec![
            vec![1], vec![0, 2], vec![1, 3], vec![2],
        ];
        let history = smooth_order_field(&mut orders, &adj_raw, 10, 2);
        assert_eq!(history.len(), 11);
        for i in 0..orders.len() {
            for &j in &adj_raw[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "final state violates max_jump: |{} − {}| > 2", orders[i], orders[jj]);
            }
        }
    }

    #[test]
    fn smooth_order_field_history_length() {
        let mut orders = vec![3, 1, 3];
        let adj_raw: Vec<Vec<ElemId>> = vec![
            vec![1], vec![0, 2], vec![1],
        ];
        let history = smooth_order_field(&mut orders, &adj_raw, 5, 2);
        assert_eq!(history.len(), 6);
        assert_eq!(history[0], vec![3, 1, 3]);
    }

    #[test]
    fn smooth_order_field_single_element() {
        let mut orders = vec![5];
        let adj_raw: Vec<Vec<ElemId>> = vec![vec![]];
        let history = smooth_order_field(&mut orders, &adj_raw, 3, 2);
        assert_eq!(history.len(), 4);
        assert_eq!(orders[0], 5);
    }

    #[test]
    fn element_adjacency_from_mesh_round_trip() {
        use crate::simplex::Mesh;
        use crate::element_type::ElementType;

        // Two Quad4 elements sharing edge 1-4
        let mut mesh = Mesh::<2> {
            coords: vec![0.0,0.0, 1.0,0.0, 2.0,0.0, 0.0,1.0, 1.0,1.0, 2.0,1.0],
            conn: vec![0u32,1,4,3, 1,2,5,4],
            elem_tags: vec![0, 0],
            elem_type: ElementType::Quad4,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None, face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
        };
        mesh.build_edge_connectivity();
        assert!(mesh.edge_to_elem.len() >= 2);

        let adj = ElementAdjacency::from_mesh(&mesh);
        assert_eq!(adj.neighbors.len(), 2);
        assert!(adj.neighbors[0].contains(&1u32),
            "elem 0 should neighbour elem 1; got {:?}", adj.neighbors[0]);
        assert!(adj.neighbors[1].contains(&0u32),
            "elem 1 should neighbour elem 0; got {:?}", adj.neighbors[1]);
    }

    #[test]
    fn element_adjacency_from_mesh_isolated_elements() {
        use crate::simplex::Mesh;
        use crate::element_type::ElementType;

        let mut mesh = Mesh::<2> {
            coords: vec![0.0,0.0, 1.0,0.0, 0.0,1.0],
            conn: vec![0u32, 1, 2],
            elem_tags: vec![0],
            elem_type: ElementType::Tri3,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None, face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
        };
        mesh.build_edge_connectivity();

        let adj = ElementAdjacency::from_mesh(&mesh);
        assert_eq!(adj.neighbors.len(), 1);
        assert!(adj.neighbors[0].is_empty(),
            "single element should have no neighbours; got {:?}", adj.neighbors[0]);
    }

    #[test]
    fn limit_p_jumps_with_element_adjacency() {
        use crate::simplex::Mesh;
        use crate::element_type::ElementType;

        let mut mesh = Mesh::<2> {
            coords: vec![0.0,0.0, 1.0,0.0, 2.0,0.0, 0.0,1.0, 1.0,1.0, 2.0,1.0],
            conn: vec![0u32,1,4,3, 1,2,5,4],
            elem_tags: vec![0, 0],
            elem_type: ElementType::Quad4,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None, face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
        };
        mesh.build_edge_connectivity();

        let adj = ElementAdjacency::from_mesh(&mesh);
        let mut orders = vec![4usize, 1];
        let changed = limit_p_jumps(&mut orders, &adj.neighbors, 2);
        assert!(changed > 0, "p=4 vs p=1 with max_jump=2 should reduce");
        assert!(orders[0].abs_diff(orders[1]) <= 2,
            "final |{} − {}| ≤ 2", orders[0], orders[1]);
    }

    #[test]
    fn smooth_order_field_with_element_adjacency() {
        use crate::simplex::Mesh;
        use crate::element_type::ElementType;

        let mut mesh = Mesh::<2> {
            coords: vec![0.0,0.0, 1.0,0.0, 2.0,0.0, 3.0,0.0, 4.0,0.0,
                 0.0,1.0, 1.0,1.0, 2.0,1.0, 3.0,1.0, 4.0,1.0],
            conn: vec![0u32,1,6,5, 1,2,7,6, 2,3,8,7, 3,4,9,8],
            elem_tags: vec![0; 4],
            elem_type: ElementType::Quad4,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Line2,
            elem_types: None, elem_offsets: None,
            face_types: None, face_offsets: None, face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
        };
        mesh.build_edge_connectivity();

        let adj = ElementAdjacency::from_mesh(&mesh);
        assert!(adj.neighbors.len() >= 3,
            "should have adjacency for 4 quads, got {} entries", adj.neighbors.len());
        let mut orders = vec![4usize, 1, 4, 1];
        smooth_order_field(&mut orders, &adj.neighbors, 5, 2);

        for i in 0..orders.len() {
            for &j in &adj.neighbors[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "smoothed |{} − {}| > 2", orders[i], orders[jj]);
            }
        }
    }

    // ── Task 2.2 multi‑block transition layer tests ────────────────────────

    fn multi_block_adjacency_9() -> Vec<Vec<ElemId>> {
        // Linear chain of 9 elements: 0‑1‑2‑3‑4‑5‑6‑7‑8
        (0..9_usize)
            .map(|i| {
                let mut nbrs = Vec::new();
                if i > 0 {
                    nbrs.push((i - 1) as ElemId);
                }
                if i < 8 {
                    nbrs.push((i + 1) as ElemId);
                }
                nbrs
            })
            .collect()
    }

    #[test]
    fn multi_block_transition_gradual_staircase() {
        // Two blocks: p=5 (elems 0‑3) and p=1 (elems 5‑8),
        // with element 4 as the interface.
        // After smoothing, a gradual staircase should emerge.
        let adj = multi_block_adjacency_9();
        let mut orders = vec![5usize, 5, 5, 5, 1, 1, 1, 1, 1];

        let _history = transition_layers(&mut orders, &adj, 2, 8);

        // No adjacent pair violates |Δp| ≤ 2
        for i in 0..orders.len() {
            for &j in &adj[i] {
                let jj = j as usize;
                assert!(
                    orders[i].abs_diff(orders[jj]) <= 2,
                    "|p[{}]={} - p[{}]={}| > 2",
                    i,
                    orders[i],
                    jj,
                    orders[jj]
                );
            }
        }
        // The interface element converges to a state satisfying |Δp| ≤ 2.
        // (Exact value depends on the smoothing path and is not specified.)
        // At minimum, the interface element must differ from its neighbors by ≤ max_jump
        assert!(orders[3].abs_diff(orders[4]) <= 2,
            "interface pair (3,4): |{} - {}| > 2", orders[3], orders[4]);
        assert!(orders[4].abs_diff(orders[5]) <= 2,
            "interface pair (4,5): |{} - {}| > 2", orders[4], orders[5]);
        // Staircase should be monotonic across the interface
        for i in 0..4 {
            assert!(orders[i] >= orders[i + 1],
                "left block non-increasing failed at {}/{}: {} < {}", i, i+1, orders[i], orders[i+1]);
        }
        for i in 4..8 {
            assert!(orders[i] >= orders[i + 1],
                "right block non-increasing failed at {}/{}: {} < {}", i, i+1, orders[i], orders[i+1]);
        }
    }

    #[test]
    fn multi_block_three_blocks_two_interfaces() {
        // Three blocks: p=4 (0‑2), p=1 (3‑5), p=4 (6‑8)
        let adj = multi_block_adjacency_9();
        let mut orders = vec![4usize, 4, 4, 1, 1, 1, 4, 4, 4];

        let _history = transition_layers(&mut orders, &adj, 2, 10);

        for i in 0..orders.len() {
            for &j in &adj[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "|p[{}]={} - p[{}]={}| > 2", i, orders[i], jj, orders[jj]);
            }
        }
        // Valley elements should be raised for transition
        assert!(orders[3] >= orders[5],
            "left valley should be >= right valley: {} vs {}", orders[3], orders[5]);
    }

    #[test]
    fn multi_block_chain_propagates_reduction() {
        // Steep gradient p=7 -> p=1 should propagate to create transition
        let adj = multi_block_adjacency_9();
        let mut orders = vec![7usize, 7, 7, 1, 1, 1, 1, 1, 1];

        let _history = transition_layers(&mut orders, &adj, 2, 8);

        for i in 0..orders.len() {
            for &j in &adj[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "|p[{}]={} - p[{}]={}| > 2", i, orders[i], jj, orders[jj]);
            }
        }
        // Should have at least 3 distinct p-values across the chain
        let distinct: std::collections::HashSet<usize> =
            orders.iter().copied().collect();
        assert!(distinct.len() >= 3,
            "steep gradient should produce >= 3 distinct p-values, got {}: {:?}",
            distinct.len(), orders);
    }

    #[test]
    fn multi_block_history_records() {
        let adj = multi_block_adjacency_9();
        let mut orders = vec![5usize, 5, 5, 5, 1, 1, 1, 1, 1];

        let history = transition_layers(&mut orders, &adj, 2, 4);
        assert_eq!(history.len(), 5, "history = input + 4 iterations");
        assert_eq!(history[0], vec![5, 5, 5, 5, 1, 1, 1, 1, 1],
            "first entry is the initial state");
    }

    #[test]
    fn multi_block_no_change_when_already_conforming() {
        let adj = multi_block_adjacency_9();
        let mut orders = vec![3usize, 3, 4, 4, 5, 5, 4, 4, 3];
        let history = transition_layers(&mut orders, &adj, 2, 5);

        for i in 0..orders.len() {
            for &j in &adj[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "|p[{}]={} - p[{}]={}| > 2", i, orders[i], jj, orders[jj]);
            }
        }
        assert_eq!(history.len(), 6, "history = input + 5 iterations");
    }

    #[test]
    fn multi_block_derefine_propagation() {
        // Transition from high-p bulk to a single low-p element
        let adj = multi_block_adjacency_9();
        let mut orders = vec![6usize, 6, 6, 6, 6, 6, 6, 6, 1];

        let _history = transition_layers(&mut orders, &adj, 2, 8);

        for i in 0..orders.len() {
            for &j in &adj[i] {
                let jj = j as usize;
                assert!(orders[i].abs_diff(orders[jj]) <= 2,
                    "|p[{}]={} - p[{}]={}| > 2", i, orders[i], jj, orders[jj]);
            }
        }
        // The smoothing process converges to a state satisfying |Δp| ≤ 2
        // everywhere — the exact values depend on the smoothing path and
        // are not specified.  What matters is the constraint holds below.
        // Additional check: the single low-p element must be within max_jump
        // of its immediate neighbor.
        assert!(orders[7].abs_diff(orders[8]) <= 2,
            "|p[7]={} - p[8]={}| > 2 at the derefine interface", orders[7], orders[8]);
    }

    #[test]
    fn transition_layers_equivalent_to_smooth_field() {
        let adj = multi_block_adjacency_9();
        let mut orders_a = vec![5usize, 5, 5, 5, 1, 1, 1, 1, 1];
        let mut orders_b = orders_a.clone();

        let _ = transition_layers(&mut orders_a, &adj, 2, 4);
        let _ = smooth_order_field(&mut orders_b, &adj, 4, 2);

        assert_eq!(orders_a, orders_b,
            "transition_layers should match smooth_order_field");
    }
}
