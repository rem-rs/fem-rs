//! Multi-rate time stepping utilities for loosely coupled multiphysics systems.
//!
//! This module provides a lightweight sub-cycling scheduler where a fast field
//! can advance with `fast_dt` inside a slower synchronization window `slow_dt`.
//! Data exchange hooks are triggered at synchronization points.

use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct MultiRateConfig {
    pub t_start: f64,
    pub t_end: f64,
    pub fast_dt: f64,
    pub slow_dt: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MultiRateAdaptiveConfig {
    pub base: MultiRateConfig,
    pub sync_error_tol: f64,
    pub max_sync_retries: usize,
    pub retry_fast_dt_scale: f64,
    pub min_fast_dt: f64,
}

impl MultiRateConfig {
    pub fn validate(&self) -> Result<(), MultiRateError> {
        if !(self.t_end.is_finite() && self.t_start.is_finite() && self.t_end > self.t_start) {
            return Err(MultiRateError::InvalidConfig(
                "require finite t_start/t_end with t_end > t_start".to_string(),
            ));
        }
        if !(self.fast_dt.is_finite() && self.fast_dt > 0.0) {
            return Err(MultiRateError::InvalidConfig(
                "fast_dt must be finite and > 0".to_string(),
            ));
        }
        if !(self.slow_dt.is_finite() && self.slow_dt > 0.0) {
            return Err(MultiRateError::InvalidConfig(
                "slow_dt must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl MultiRateAdaptiveConfig {
    pub fn validate(&self) -> Result<(), MultiRateError> {
        self.base.validate()?;
        if !(self.sync_error_tol.is_finite() && self.sync_error_tol >= 0.0) {
            return Err(MultiRateError::InvalidConfig(
                "sync_error_tol must be finite and >= 0".to_string(),
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
        if !(self.min_fast_dt.is_finite() && self.min_fast_dt > 0.0) {
            return Err(MultiRateError::InvalidConfig(
                "min_fast_dt must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MultiRateStats {
    pub final_time: f64,
    pub sync_steps: usize,
    pub fast_substeps: usize,
    pub slow_steps: usize,
    pub sync_retries: usize,
    pub rollback_count: usize,
    pub rejected_sync_steps: usize,
    pub last_sync_error: f64,
}

#[derive(Debug, Error)]
pub enum MultiRateError {
    #[error("invalid multi-rate config: {0}")]
    InvalidConfig(String),
    #[error("sync step at t={time:.6e} did not converge after {retries} retries (error={error:.3e})")]
    SyncDidNotConverge {
        time: f64,
        retries: usize,
        error: f64,
    },
}

/// Run a multi-rate schedule with fast sub-cycles and synchronization hooks.
///
/// Per synchronization window:
/// 1) advance fast field with sub-steps of size `fast_dt` (last sub-step clipped)
/// 2) advance slow field once over the whole window
/// 3) call `on_sync(t_sync)` for data exchange/projection
pub fn run_multirate<C, Ffast, Fslow, Fsync>(
    cfg: MultiRateConfig,
    ctx: &mut C,
    mut fast_step: Ffast,
    mut slow_step: Fslow,
    mut on_sync: Fsync,
) -> Result<MultiRateStats, MultiRateError>
where
    Ffast: FnMut(&mut C, f64, f64),
    Fslow: FnMut(&mut C, f64, f64),
    Fsync: FnMut(&mut C, f64),
{
    cfg.validate()?;

    let mut stats = MultiRateStats::default();
    let mut t = cfg.t_start;

    while t < cfg.t_end - 1.0e-14 {
        let window = (cfg.t_end - t).min(cfg.slow_dt);
        let t_window_end = t + window;

        let mut tf = t;
        while tf < t_window_end - 1.0e-14 {
            let dtf = (t_window_end - tf).min(cfg.fast_dt);
            fast_step(ctx, tf, dtf);
            tf += dtf;
            stats.fast_substeps += 1;
        }

        slow_step(ctx, t, window);
        stats.slow_steps += 1;

        t = t_window_end;
        on_sync(ctx, t);
        stats.sync_steps += 1;
    }

    stats.final_time = t;
    Ok(stats)
}

/// Run a multi-rate schedule with sync-error control and rollback retry.
///
/// At each synchronization point, `on_sync` returns an error metric that must
/// satisfy `error <= sync_error_tol`. If not, the whole sync window is rolled
/// back and retried with a reduced fast time step.
pub fn run_multirate_adaptive<C, Ffast, Fslow, Fsync>(
    cfg: MultiRateAdaptiveConfig,
    ctx: &mut C,
    mut fast_step: Ffast,
    mut slow_step: Fslow,
    mut on_sync: Fsync,
) -> Result<MultiRateStats, MultiRateError>
where
    C: Clone,
    Ffast: FnMut(&mut C, f64, f64),
    Fslow: FnMut(&mut C, f64, f64),
    Fsync: FnMut(&mut C, f64) -> f64,
{
    cfg.validate()?;

    let mut stats = MultiRateStats::default();
    let mut t = cfg.base.t_start;
    let mut current_fast_dt = cfg.base.fast_dt.min(cfg.base.slow_dt).max(cfg.min_fast_dt);

    while t < cfg.base.t_end - 1.0e-14 {
        let window = (cfg.base.t_end - t).min(cfg.base.slow_dt);
        let t_window_end = t + window;

        let mut attempt = 0usize;
        loop {
            let snapshot = ctx.clone();

            let mut tf = t;
            while tf < t_window_end - 1.0e-14 {
                let dtf = (t_window_end - tf).min(current_fast_dt);
                fast_step(ctx, tf, dtf);
                tf += dtf;
                stats.fast_substeps += 1;
            }

            slow_step(ctx, t, window);
            stats.slow_steps += 1;

            let sync_error = on_sync(ctx, t_window_end);
            stats.last_sync_error = sync_error;

            if sync_error <= cfg.sync_error_tol {
                t = t_window_end;
                stats.sync_steps += 1;
                break;
            }

            if attempt >= cfg.max_sync_retries || current_fast_dt <= cfg.min_fast_dt * (1.0 + 1.0e-12)
            {
                return Err(MultiRateError::SyncDidNotConverge {
                    time: t_window_end,
                    retries: attempt,
                    error: sync_error,
                });
            }

            *ctx = snapshot;
            attempt += 1;
            stats.sync_retries += 1;
            stats.rollback_count += 1;
            stats.rejected_sync_steps += 1;
            current_fast_dt = (current_fast_dt * cfg.retry_fast_dt_scale).max(cfg.min_fast_dt);
        }
    }

    stats.final_time = t;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multirate_scheduler_counts_substeps_and_sync_points() {
        let cfg = MultiRateConfig {
            t_start: 0.0,
            t_end: 1.0,
            fast_dt: 0.1,
            slow_dt: 0.25,
        };

        let mut fast_calls = 0usize;
        let mut slow_calls = 0usize;
        let mut sync_calls = 0usize;
        let mut ctx = ();
        let st = run_multirate(
            cfg,
            &mut ctx,
            |_ctx, _t, _dt| fast_calls += 1,
            |_ctx, _t, _dt| slow_calls += 1,
            |_ctx, _t| sync_calls += 1,
        )
        .unwrap();

        assert_eq!(slow_calls, 4);
        assert_eq!(sync_calls, 4);
        assert_eq!(fast_calls, 12); // 3+3+3+3 sub-steps due to clipped windows
        assert_eq!(st.sync_steps, 4);
        assert_eq!(st.slow_steps, 4);
        assert_eq!(st.fast_substeps, 12);
        assert!((st.final_time - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn multirate_rejects_invalid_dt() {
        let bad = MultiRateConfig {
            t_start: 0.0,
            t_end: 1.0,
            fast_dt: 0.0,
            slow_dt: 0.1,
        };
        let mut ctx = ();
        assert!(run_multirate(bad, &mut ctx, |_ctx, _t, _dt| {}, |_ctx, _t, _dt| {}, |_ctx, _t| {}).is_err());
    }

    #[test]
    fn multirate_adaptive_retries_and_rolls_back() {
        #[derive(Clone)]
        struct Ctx {
            val: usize,
            accepts: usize,
        }

        let cfg = MultiRateAdaptiveConfig {
            base: MultiRateConfig {
                t_start: 0.0,
                t_end: 0.5,
                fast_dt: 0.1,
                slow_dt: 0.25,
            },
            sync_error_tol: 0.2,
            max_sync_retries: 3,
            retry_fast_dt_scale: 0.5,
            min_fast_dt: 0.01,
        };

        let mut ctx = Ctx { val: 0, accepts: 0 };
        let st = run_multirate_adaptive(
            cfg,
            &mut ctx,
            |c, _t, _dt| c.val += 1,
            |_c, _t, _dt| {},
            |c, _t| {
                // Accept only after enough fast sub-steps happened in this attempt.
                if c.val >= 4 {
                    c.accepts += 1;
                    0.1
                } else {
                    0.9
                }
            },
        )
        .unwrap();

        assert_eq!(st.sync_steps, 2);
        assert!(st.sync_retries >= 1);
        assert!(st.rollback_count >= 1);
        assert_eq!(ctx.accepts, 2);
        assert!(st.last_sync_error <= cfg.sync_error_tol);
    }

    #[test]
    fn multirate_adaptive_reports_nonconverged_sync() {
        let cfg = MultiRateAdaptiveConfig {
            base: MultiRateConfig {
                t_start: 0.0,
                t_end: 0.5,
                fast_dt: 0.1,
                slow_dt: 0.25,
            },
            sync_error_tol: 1.0e-6,
            max_sync_retries: 1,
            retry_fast_dt_scale: 0.5,
            min_fast_dt: 0.01,
        };

        let mut ctx = 0usize;
        let err = run_multirate_adaptive(
            cfg,
            &mut ctx,
            |_c, _t, _dt| {},
            |_c, _t, _dt| {},
            |_c, _t| 1.0,
        )
        .unwrap_err();

        match err {
            MultiRateError::SyncDidNotConverge { retries, .. } => {
                assert_eq!(retries, 1);
            }
            _ => panic!("unexpected error variant"),
        }
    }
}
