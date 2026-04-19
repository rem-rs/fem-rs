//! Utilities for composing synchronization error metrics in multiphysics runs.
//!
//! This module provides reusable trackers that measure relative variation of
//! scalar quantities or field norms across synchronization points. The trackers
//! are designed to be lightweight and cloneable so they can participate in
//! rollback-based adaptive time stepping.

use crate::multirate::{MultiRateAdaptiveConfig, MultiRateConfig, MultiRateError};

/// Common synchronization policy used by template-style multiphysics drivers.
#[derive(Debug, Clone)]
pub struct TemplateSyncPolicy {
    pub sync_error_tol: f64,
    pub max_sync_retries: usize,
    pub min_fast_dt: f64,
    pub retry_fast_dt_scale: f64,
    pub component_weights: Vec<f64>,
}

impl Default for TemplateSyncPolicy {
    fn default() -> Self {
        Self {
            sync_error_tol: 1.0,
            max_sync_retries: 2,
            min_fast_dt: 1.0e-3,
            retry_fast_dt_scale: 0.5,
            component_weights: Vec::new(),
        }
    }
}

impl TemplateSyncPolicy {
    pub fn validate(&self) -> Result<(), MultiRateError> {
        if !(self.sync_error_tol.is_finite() && self.sync_error_tol >= 0.0) {
            return Err(MultiRateError::InvalidConfig(
                "sync_error_tol must be finite and >= 0".to_string(),
            ));
        }
        if !(self.min_fast_dt.is_finite() && self.min_fast_dt > 0.0) {
            return Err(MultiRateError::InvalidConfig(
                "min_fast_dt must be finite and > 0".to_string(),
            ));
        }
        if !(self.retry_fast_dt_scale.is_finite()
            && self.retry_fast_dt_scale > 0.0
            && self.retry_fast_dt_scale < 1.0)
        {
            return Err(MultiRateError::InvalidConfig(
                "retry_fast_dt_scale must be finite and in (0, 1)".to_string(),
            ));
        }
        if self
            .component_weights
            .iter()
            .any(|w| !w.is_finite() || *w < 0.0)
        {
            return Err(MultiRateError::InvalidConfig(
                "component_weights must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }

    pub fn adaptive_config(
        &self,
        base: MultiRateConfig,
    ) -> Result<MultiRateAdaptiveConfig, MultiRateError> {
        self.validate()?;
        let cfg = MultiRateAdaptiveConfig {
            base,
            sync_error_tol: self.sync_error_tol,
            max_sync_retries: self.max_sync_retries,
            retry_fast_dt_scale: self.retry_fast_dt_scale,
            min_fast_dt: self.min_fast_dt,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn compose_error(&self, components: &[f64]) -> f64 {
        if self.component_weights.is_empty() {
            compose_sync_error(components)
        } else {
            compose_weighted_sync_error(components, &self.component_weights)
        }
    }
}

/// Track relative changes of a scalar quantity across sync points.
#[derive(Debug, Clone, Default)]
pub struct RelativeScalarTracker {
    prev: Option<f64>,
}

impl RelativeScalarTracker {
    pub fn new() -> Self {
        Self { prev: None }
    }

    /// Observe a new scalar value and return relative change.
    ///
    /// On the first observation, returns `fallback` and stores the value.
    pub fn observe(&mut self, current: f64, fallback: f64) -> f64 {
        let rel = match self.prev {
            Some(prev) if current.is_finite() => {
                (current - prev).abs() / current.abs().max(1.0e-12)
            }
            _ => fallback,
        };
        self.prev = Some(current);
        rel
    }
}

/// Track relative changes of field L2 norm across sync points.
#[derive(Debug, Clone, Default)]
pub struct RelativeL2Tracker {
    prev_l2: Option<f64>,
}

impl RelativeL2Tracker {
    pub fn new() -> Self {
        Self { prev_l2: None }
    }

    /// Observe a field and return relative change of its L2 norm.
    ///
    /// On the first observation, returns `fallback` and stores the current norm.
    pub fn observe_field(&mut self, field: &[f64], fallback: f64) -> f64 {
        let l2 = l2_norm(field);
        let rel = match self.prev_l2 {
            Some(prev) => (l2 - prev).abs() / l2.abs().max(1.0e-12),
            None => fallback,
        };
        self.prev_l2 = Some(l2);
        rel
    }
}

/// Compose one sync error value from multiple component errors.
pub fn compose_sync_error(components: &[f64]) -> f64 {
    components.iter().copied().fold(0.0_f64, f64::max)
}

/// Compose one sync error value from components with optional nonnegative weights.
///
/// Missing weights default to 1.0. Negative or non-finite weights are treated
/// as 0.0 to avoid destabilizing acceptance logic.
pub fn compose_weighted_sync_error(components: &[f64], weights: &[f64]) -> f64 {
    components
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let w = weights
                .get(i)
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(1.0);
            w * e
        })
        .fold(0.0_f64, f64::max)
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_tracker_uses_fallback_then_relative_change() {
        let mut tr = RelativeScalarTracker::new();
        let r0 = tr.observe(2.0, 0.25);
        let r1 = tr.observe(2.5, 0.25);
        assert!((r0 - 0.25).abs() < 1.0e-12);
        assert!(r1 > 0.0);
    }

    #[test]
    fn l2_tracker_uses_fallback_then_relative_change() {
        let mut tr = RelativeL2Tracker::new();
        let r0 = tr.observe_field(&[1.0, 0.0], 0.5);
        let r1 = tr.observe_field(&[2.0, 0.0], 0.5);
        assert!((r0 - 0.5).abs() < 1.0e-12);
        assert!(r1 > 0.0);
    }

    #[test]
    fn compose_sync_error_returns_component_max() {
        let e = compose_sync_error(&[1.0e-2, 1.0e-3, 5.0e-2]);
        assert!((e - 5.0e-2).abs() < 1.0e-12);
    }

    #[test]
    fn compose_weighted_sync_error_applies_weights() {
        let e = compose_weighted_sync_error(&[1.0e-2, 5.0e-3], &[2.0, 1.0]);
        assert!((e - 2.0e-2).abs() < 1.0e-12);
    }

    #[test]
    fn sync_policy_builds_adaptive_config() {
        let p = TemplateSyncPolicy::default();
        let cfg = p
            .adaptive_config(MultiRateConfig {
                t_start: 0.0,
                t_end: 1.0,
                fast_dt: 0.1,
                slow_dt: 0.2,
            })
            .unwrap();
        assert_eq!(cfg.max_sync_retries, 2);
    }
}
