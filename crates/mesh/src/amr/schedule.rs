//! # Transient AMR scheduling (Phase 5)
//!
//! Predict when to remesh during time-dependent simulations, supporting
//! multiple strategies: periodic, error-threshold, predictive, hybrid,
//! and zone-based multi-rate scheduling with rolling time windows.
//!
//! ## Usage
//! ```ignore
//! let mut scheduler = TimeAwareRemesher::periodic(5);
//! for step in 0..100 {
//!     // ... solve step ...
//!     if scheduler.should_remesh(step, Some(&eta)) {
//!         let eta_pred = scheduler.predict_error(&eta_history, &mesh);
//!         // refine mesh based on eta_pred ...
//!         scheduler.record_remesh(step, &eta);
//!     }
//! }
//! ```

use fem_core::ElemId;
use crate::Mesh;

// ═════════════════════════════════════════════════════════════════════════════
//  Error growth models
// ═════════════════════════════════════════════════════════════════════════════

/// Model for predicting error growth between remeshings.
#[derive(Debug, Clone)]
pub enum ErrorGrowthModel {
    /// Linear extrapolation from the last two error values.
    /// `e_{n+1} = e_n + (e_n − e_{n-1})`
    Linear,
    /// Exponential growth: `e_{n+1} = α · e_n`.
    Exponential { alpha: f64 },
}

impl ErrorGrowthModel {
    /// Predict the error at the next remesh step given the error history.
    ///
    /// `history` is the list of global error estimates at previous remesh events,
    /// ordered from oldest to newest.  Returns the predicted error.
    pub fn predict(&self, history: &[f64], n_steps_since_remesh: usize) -> f64 {
        match self {
            Self::Linear => {
                if history.len() >= 2 {
                    let last = history[history.len() - 1];
                    let prev = history[history.len() - 2];
                    let slope = (last - prev).max(0.0);
                    (last + slope * n_steps_since_remesh as f64).max(0.0)
                } else if history.len() == 1 {
                    history[0] * (1.0 + 0.1 * n_steps_since_remesh as f64)
                } else {
                    0.0
                }
            }
            Self::Exponential { alpha } => {
                if let Some(&last) = history.last() {
                    last * alpha.powi(n_steps_since_remesh as i32)
                } else {
                    0.0
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Remesh timing strategies
// ═════════════════════════════════════════════════════════════════════════════

/// Strategy for deciding when to trigger a mesh adaptation.
#[derive(Debug, Clone)]
pub enum RemeshTiming {
    /// Remesh every fixed number of steps.
    Periodic { every_n_steps: usize },
    /// Remesh when the global error estimate exceeds a threshold.
    ErrorThreshold { min_error: f64, max_error: f64 },
    /// Predict error growth and remesh preemptively before error gets too large.
    Predictive {
        /// How many steps ahead to look.
        horizon: usize,
        /// Model for error growth.
        growth_model: ErrorGrowthModel,
        /// Maximum allowed predicted error.
        max_predicted: f64,
    },
    /// Hybrid: periodic check + threshold-based triggering.
    Hybrid { period: usize, max_error: f64 },
}

// ═════════════════════════════════════════════════════════════════════════════
//  Time-aware remesher
// ═════════════════════════════════════════════════════════════════════════════

/// Configurable remeshing scheduler for transient simulations.
///
/// Tracks the history of remesh events and global error estimates, and
/// decides at each time step whether a mesh adaptation is needed.
#[derive(Debug, Clone)]
pub struct TimeAwareRemesher {
    /// Timing strategy.
    pub timing: RemeshTiming,
    /// History of global error estimates at remesh events.
    pub error_history: Vec<f64>,
    /// Step number of the last remesh.
    pub last_remesh_step: usize,
    /// Number of remesh events so far.
    pub remesh_count: usize,
    /// Whether to protect coarsening with a buffer layer.
    pub coarsening_buffer: bool,
    /// Minimum refinement level to preserve around features.
    pub min_buffer_level: u8,
}

impl TimeAwareRemesher {
    /// Create a new remesher with the given timing strategy.
    pub fn new(timing: RemeshTiming) -> Self {
        Self {
            timing,
            error_history: Vec::new(),
            last_remesh_step: 0,
            remesh_count: 0,
            coarsening_buffer: false,
            min_buffer_level: 1,
        }
    }

    /// Create a periodic remesher (every N steps).
    pub fn periodic(every_n_steps: usize) -> Self {
        Self::new(RemeshTiming::Periodic { every_n_steps })
    }

    /// Create an error-threshold remesher.
    pub fn error_threshold(max_error: f64) -> Self {
        Self::new(RemeshTiming::ErrorThreshold { min_error: max_error * 0.1, max_error })
    }

    /// Create a predictive remesher.
    pub fn predictive(horizon: usize, growth_model: ErrorGrowthModel, max_predicted: f64) -> Self {
        Self::new(RemeshTiming::Predictive { horizon, growth_model, max_predicted })
    }

    /// Create a hybrid remesher.
    pub fn hybrid(period: usize, max_error: f64) -> Self {
        Self::new(RemeshTiming::Hybrid { period, max_error })
    }

    /// Enable coarsening protection (preserve buffer refinement level).
    pub fn with_coarsening_buffer(mut self, buffer_level: u8) -> Self {
        self.coarsening_buffer = true;
        self.min_buffer_level = buffer_level;
        self
    }

    /// Decide whether the mesh should be adapted at the current step.
    ///
    /// # Arguments
    /// * `step` — current time step index.
    /// * `global_error` — optional current global error estimate.
    pub fn should_remesh(&self, step: usize, global_error: Option<f64>) -> bool {
        let steps_since = step.saturating_sub(self.last_remesh_step);

        match &self.timing {
            RemeshTiming::Periodic { every_n_steps } => {
                steps_since >= *every_n_steps
            }
            RemeshTiming::ErrorThreshold { min_error, max_error } => {
                if steps_since == 0 { return false; }
                match global_error {
                    Some(err) => err > *max_error || err < *min_error,
                    None => false,
                }
            }
            RemeshTiming::Predictive { horizon, growth_model, max_predicted } => {
                if steps_since == 0 { return false; }
                if steps_since < 2 { return false; }
                // Predict error at horizon steps from now
                let current_err = global_error.unwrap_or(
                    *self.error_history.last().unwrap_or(&0.0)
                );
                let predicted = growth_model.predict(&self.error_history, steps_since + horizon);
                predicted > *max_predicted || current_err > *max_predicted * 0.8
            }
            RemeshTiming::Hybrid { period, max_error } => {
                if steps_since == 0 { return false; }
                // Remesh if period has elapsed OR error exceeds threshold
                let period_trigger = steps_since >= *period;
                let error_trigger = match global_error {
                    Some(err) => err > *max_error,
                    None => false,
                };
                period_trigger || error_trigger
            }
        }
    }

    /// Predict the error at a future step using the configured growth model.
    pub fn predict_error(&self, n_steps_ahead: usize) -> f64 {
        match &self.timing {
            RemeshTiming::Predictive { growth_model, .. } => {
                growth_model.predict(&self.error_history, n_steps_ahead)
            }
            _ => {
                ErrorGrowthModel::Linear.predict(&self.error_history, n_steps_ahead)
            }
        }
    }

    /// Record a remesh event (call after adaptation).
    pub fn record_remesh(&mut self, step: usize, global_error: Option<f64>) {
        self.last_remesh_step = step;
        self.remesh_count += 1;
        if let Some(err) = global_error {
            self.error_history.push(err);
        }
    }

    /// Get the steps since the last remesh.
    pub fn steps_since_remesh(&self, step: usize) -> usize {
        step.saturating_sub(self.last_remesh_step)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Coarsening protection buffer (Task 5.1 Step 2)
// ═════════════════════════════════════════════════════════════════════════════

/// Configuration for protecting elements from over-coarsening.
#[derive(Debug, Clone)]
pub struct CoarseningProtection {
    /// Minimum refinement level to preserve (default: 1).
    pub min_level: u8,
    /// Buffer zone width in elements around features (default: 1).
    pub buffer_width: usize,
}

impl Default for CoarseningProtection {
    fn default() -> Self {
        Self { min_level: 1, buffer_width: 1 }
    }
}

/// Apply coarsening protection by ensuring elements near the `min_level`
/// boundary are kept at a higher level.
///
/// Modifies `derefine_candidates` in-place: elements whose removal would
/// expose a neighbour at `min_level` are removed from the candidate list.
pub fn apply_coarsening_buffer(
    derefine_candidates: &mut Vec<ElemId>,
    tree: &crate::amr::RefinementTree,
    config: &CoarseningProtection,
) {
    if derefine_candidates.is_empty() { return; }

    let candidate_set: std::collections::HashSet<ElemId> =
        derefine_candidates.iter().copied().collect();
    let mut to_remove = Vec::new();

    for &e in derefine_candidates.iter() {
        let elem_level = tree.level(e);
        if elem_level <= config.min_level {
            to_remove.push(e);
            continue;
        }

        // Find neighbours that would be exposed if this element is coarsened.
        let siblings = tree.siblings_of(e);
        for &sib in siblings.iter() {
            if sib != e && !candidate_set.contains(&sib) {
                let sib_level = tree.level(sib);
                if sib_level <= config.min_level {
                    to_remove.push(e);
                    break;
                }
            }
        }
    }

    for e in to_remove {
        derefine_candidates.retain(|&x| x != e);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Zone-based multi-rate scheduling (Task 5.1 Step 3)
// ═════════════════════════════════════════════════════════════════════════════

/// A subregion of the mesh with its own remeshing schedule.
#[derive(Debug, Clone)]
pub struct RemeshZone {
    /// Name/label for the zone.
    pub name: String,
    /// Element tags belonging to this zone.
    pub tags: Vec<i32>,
    /// Check interval (in steps) for this zone.
    pub check_interval: usize,
}

/// Zone-based scheduler that checks different mesh regions at different rates.
///
/// Example: elements near a contact region checked every 5 steps, bulk
/// material checked every 20 steps.
#[derive(Debug, Clone)]
pub struct ZoneScheduler {
    /// Global remesher (used when no zone matches or for global operations).
    pub global: TimeAwareRemesher,
    /// Per-zone remeshers.
    pub zones: Vec<RemeshZone>,
    /// Error history per zone (step → Vec<zone_error>).
    pub zone_errors: Vec<Vec<f64>>,
}

impl ZoneScheduler {
    pub fn new(global: TimeAwareRemesher) -> Self {
        Self { global, zones: Vec::new(), zone_errors: Vec::new() }
    }

    /// Add a zone with its own check interval.
    pub fn add_zone(&mut self, name: &str, tags: Vec<i32>, check_interval: usize) {
        self.zones.push(RemeshZone {
            name: name.to_string(),
            tags,
            check_interval,
        });
        self.zone_errors.push(Vec::new());
    }

    /// Check which zones should remesh at the current step.
    pub fn zones_to_remesh(&self, step: usize, _mesh: &Mesh<2>) -> Vec<usize> {
        let mut result = Vec::new();
        for (i, zone) in self.zones.iter().enumerate() {
            if step % zone.check_interval == 0 && step > 0 {
                result.push(i);
            }
        }
        result
    }

    /// Record zone-level errors after a solve.
    pub fn record_zone_error(&mut self, zone_idx: usize, error: f64) {
        if zone_idx < self.zone_errors.len() {
            self.zone_errors[zone_idx].push(error);
        }
    }

    /// Remesh count across all zones.
    pub fn total_remesh_count(&self) -> usize {
        self.global.remesh_count
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Rolling time-window AMR (Task 5.2)
// ═════════════════════════════════════════════════════════════════════════════

/// Per-element error history within a rolling time window.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    /// Window length in steps.
    pub window_length: usize,
    /// History of element error indicators: `history[step_offset][elem] = η_K`.
    /// Only the last `window_length` steps are kept.
    pub element_history: Vec<Vec<f64>>,
    /// Maximum error envelope: `max_eta[e] = max_{t in window} η_K(t)`.
    pub max_envelope: Vec<f64>,
    /// Number of elements tracked.
    n_elems: usize,
}

impl RollingWindow {
    /// Create a new rolling window of the given length.
    pub fn new(window_length: usize, n_elems: usize) -> Self {
        Self {
            window_length,
            element_history: Vec::with_capacity(window_length),
            max_envelope: vec![0.0; n_elems],
            n_elems,
        }
    }

    /// Record element error indicators for the current time step.
    ///
    /// Updates both the history and the max envelope.  When old data rolls
    /// off, the envelope is recomputed from the remaining records.
    pub fn record(&mut self, eta: &[f64]) {
        assert_eq!(eta.len(), self.n_elems, "RollingWindow: eta length mismatch");

        self.element_history.push(eta.to_vec());
        if self.element_history.len() > self.window_length {
            self.element_history.remove(0);
        }

        // Recompute envelope from current window contents
        self.max_envelope = vec![0.0; self.n_elems];
        for eta_window in &self.element_history {
            for (e, &val) in eta_window.iter().enumerate() {
                if val > self.max_envelope[e] {
                    self.max_envelope[e] = val;
                }
            }
        }
    }

    /// Get the maximum error envelope for all elements.
    pub fn envelope(&self) -> &[f64] {
        &self.max_envelope
    }

    /// Reset the window (e.g., after coarsening).
    pub fn reset(&mut self) {
        self.element_history.clear();
        self.max_envelope.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Whether the window has enough data to trigger a coarsening decision.
    pub fn is_ready(&self) -> bool {
        self.element_history.len() >= self.window_length
    }

    /// Compute coarsening candidates: elements whose current error is
    /// below `frac` of their envelope maximum throughout the window.
    ///
    /// These elements have been consistently low-error and are safe to coarsen.
    pub fn coarsening_candidates(&self, current_eta: &[f64], frac: f64) -> Vec<ElemId> {
        assert_eq!(current_eta.len(), self.n_elems);
        let mut candidates = Vec::new();
        for (e, &env) in self.max_envelope.iter().enumerate() {
            if env > 0.0 {
                // Element qualifies for coarsening if current error is small
                // relative to its peak error in the window.
                if current_eta[e] < frac * env {
                    candidates.push(e as ElemId);
                }
            }
        }
        candidates
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    // ── RemeshTiming tests ─────────────────────────────────────────────────

    #[test]
    fn periodic_remeshes_at_correct_interval() {
        let r = TimeAwareRemesher::periodic(5);
        assert!(!r.should_remesh(0, None));
        assert!(!r.should_remesh(3, None));
        assert!(r.should_remesh(5, None));  // exactly at interval
        assert!(r.should_remesh(10, None)); // multiple of interval
    }

    #[test]
    fn periodic_respects_last_remesh() {
        let mut r = TimeAwareRemesher::periodic(3);
        // Steps 0-2: no remesh yet
        assert!(!r.should_remesh(0, None));
        assert!(!r.should_remesh(1, None));
        assert!(r.should_remesh(3, None));
        r.record_remesh(3, Some(1.0));

        // Now step 4 is just 1 step after remesh
        assert!(!r.should_remesh(4, None));
        assert!(r.should_remesh(6, None)); // 3 steps after last remesh
    }

    #[test]
    fn error_threshold_triggers_on_large_error() {
        let r = TimeAwareRemesher::error_threshold(0.5);
        // After recording a remesh at step 0...
        let mut r = r;
        r.record_remesh(0, Some(0.1));

        assert!(!r.should_remesh(1, Some(0.3))); // below max
        assert!(r.should_remesh(1, Some(0.6)));  // above max
    }

    #[test]
    fn predictive_triggers_before_exceeding_threshold() {
        let model = ErrorGrowthModel::Exponential { alpha: 1.5 };
        let mut r = TimeAwareRemesher::predictive(3, model, 2.0);
        r.record_remesh(0, Some(0.5));
        r.record_remesh(5, Some(0.8));

        // After 2 data points, the exponential model predicts:
        // e_{5+3} = 0.8 * 1.5^3 = 2.7 > 2.0 → should remesh
        assert!(r.should_remesh(8, Some(1.0)));
    }

    #[test]
    fn hybrid_triggers_by_period_or_error() {
        let mut r = TimeAwareRemesher::hybrid(5, 1.0);
        r.record_remesh(0, Some(0.5));

        // Period not elapsed, error below threshold
        assert!(!r.should_remesh(3, Some(0.6)));

        // Error spikes
        assert!(r.should_remesh(4, Some(1.5)));

        // Even with low error, period eventually triggers
        let mut r2 = TimeAwareRemesher::hybrid(5, 1.0);
        r2.record_remesh(0, Some(0.5));
        assert!(r2.should_remesh(5, Some(0.3)));
    }

    // ── Error growth model tests ──────────────────────────────────────────

    #[test]
    fn linear_growth_from_two_points() {
        let model = ErrorGrowthModel::Linear;
        let history = vec![1.0, 2.0]; // slope = 1
        let predicted = model.predict(&history, 3); // 3 steps ahead
        assert!((predicted - 5.0).abs() < 1e-12,
            "Linear: 2 + 1*3 = 5, got {predicted}");
    }

    #[test]
    fn exponential_growth_doubling() {
        let model = ErrorGrowthModel::Exponential { alpha: 2.0 };
        let history = vec![1.0];
        let predicted = model.predict(&history, 3); // 1 * 2^3 = 8
        assert!((predicted - 8.0).abs() < 1e-12,
            "Exp: 1 * 2^3 = 8, got {predicted}");
    }

    // ── Remesh event tracking tests ───────────────────────────────────────

    #[test]
    fn record_remesh_tracks_count_and_error() {
        let mut r = TimeAwareRemesher::periodic(2);
        assert_eq!(r.remesh_count, 0);
        r.record_remesh(2, Some(0.5));
        assert_eq!(r.remesh_count, 1);
        assert_eq!(r.error_history.len(), 1);
        assert!((r.error_history[0] - 0.5).abs() < 1e-14);
        r.record_remesh(4, Some(0.8));
        assert_eq!(r.remesh_count, 2);
        assert_eq!(r.error_history.len(), 2);
    }

    #[test]
    fn steps_since_remesh_is_correct() {
        let mut r = TimeAwareRemesher::periodic(3);
        r.record_remesh(5, None);
        assert_eq!(r.steps_since_remesh(7), 2);
        assert_eq!(r.steps_since_remesh(5), 0);
    }

    // ── Coarsening buffer tests ────────────────────────────────────────────

    #[test]
    fn coarsening_buffer_removes_min_level() {
        let mut tree = crate::amr::RefinementTree::new();
        tree.init(4);

        let mut candidates = vec![0, 1, 2, 3];
        let config = CoarseningProtection { min_level: 1, buffer_width: 1 };
        apply_coarsening_buffer(&mut candidates, &tree, &config);

        assert!(candidates.len() < 4,
            "Buffer should remove some elements, remaining: {:?}", candidates);
    }

    // ── Rolling window tests ───────────────────────────────────────────────

    #[test]
    fn rolling_window_tracks_envelope() {
        let mut rw = RollingWindow::new(3, 2); // 2 elements

        // Step 1: eta = [1.0, 3.0]
        rw.record(&[1.0, 3.0]);
        assert_eq!(rw.envelope(), &[1.0, 3.0]);

        // Step 2: eta = [2.0, 1.0]
        rw.record(&[2.0, 1.0]);
        assert_eq!(rw.envelope(), &[2.0, 3.0]); // max per element

        // Step 3: eta = [0.5, 2.0]
        rw.record(&[0.5, 2.0]);
        assert_eq!(rw.envelope(), &[2.0, 3.0]); // max still 2.0 and 3.0
    }

    #[test]
    fn rolling_window_old_data_rolls_off() {
        let mut rw = RollingWindow::new(2, 1);

        rw.record(&[10.0]); // step 1
        assert_eq!(rw.envelope(), &[10.0]);

        rw.record(&[1.0]);  // step 2 — still within window
        assert_eq!(rw.envelope(), &[10.0]); // max = 10

        rw.record(&[2.0]);  // step 3 — step 1 rolls off (10 is gone)
        assert_eq!(rw.envelope(), &[2.0]); // max is now 2
    }

    #[test]
    fn rolling_window_coarsening_candidates() {
        let mut rw = RollingWindow::new(3, 3);
        // Element 0: high then low → coarsening candidate
        // Element 1: always medium → not candidate
        // Element 2: always high → not candidate

        rw.record(&[10.0, 5.0, 8.0]);
        rw.record(&[2.0, 6.0, 9.0]);
        rw.record(&[1.0, 4.0, 7.0]);

        // Current error: elem 0 has 1.0 vs envelope max 10.0 (1/10 = 0.1 < 0.3)
        let current = vec![1.0, 4.0, 7.0];
        let candidates = rw.coarsening_candidates(&current, 0.3);

        assert!(candidates.contains(&0), "Elem 0 should be coarsenable (1.0 < 0.3*10)");
        assert!(!candidates.contains(&1), "Elem 1 should not be coarsenable (4.0 >= 0.3*5)");
        assert!(!candidates.contains(&2), "Elem 2 should not be coarsenable (7.0 >= 0.3*8)");
    }

    #[test]
    fn rolling_window_reset_clears() {
        let mut rw = RollingWindow::new(3, 2);
        rw.record(&[1.0, 2.0]);
        rw.record(&[3.0, 4.0]);
        assert_eq!(rw.element_history.len(), 2);
        rw.reset();
        assert_eq!(rw.element_history.len(), 0);
        assert_eq!(rw.envelope(), &[0.0, 0.0]);
    }

    #[test]
    fn rolling_window_ready() {
        let mut rw = RollingWindow::new(3, 1);
        assert!(!rw.is_ready());
        rw.record(&[1.0]);
        assert!(!rw.is_ready());
        rw.record(&[2.0]);
        assert!(!rw.is_ready());
        rw.record(&[3.0]);
        assert!(rw.is_ready());
    }

    // ── Zone scheduler tests ───────────────────────────────────────────────

    #[test]
    fn zone_scheduler_checks_at_different_intervals() {
        let global = TimeAwareRemesher::periodic(10);
        let mut zs = ZoneScheduler::new(global);
        zs.add_zone("contact", vec![1], 2);
        zs.add_zone("bulk", vec![2], 5);

        let mesh = Mesh::<2>::unit_square_tri(4);

        // Step 0: no zone remesh
        assert_eq!(zs.zones_to_remesh(0, &mesh).len(), 0);

        // Step 2: zone 0 (contact, interval 2) triggers
        let zones = zs.zones_to_remesh(2, &mesh);
        assert_eq!(zones, vec![0]);

        // Step 5: zone 0 (step 5 % 2 = 1? no) and zone 1 (step 5 % 5 = 0, yes)
        let zones = zs.zones_to_remesh(5, &mesh);
        assert_eq!(zones, vec![1]);

        // Step 10: both zones trigger
        let zones = zs.zones_to_remesh(10, &mesh);
        assert_eq!(zones, vec![0, 1]);
    }

    #[test]
    fn zone_scheduler_record_and_count() {
        let global = TimeAwareRemesher::periodic(10);
        let mut zs = ZoneScheduler::new(global);
        zs.add_zone("zone_a", vec![1], 3);

        zs.record_zone_error(0, 0.5);
        zs.record_zone_error(0, 0.8);
        assert_eq!(zs.zone_errors[0], vec![0.5, 0.8]);
    }
}
