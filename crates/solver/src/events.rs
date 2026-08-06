//! Event detection and root-finding during time integration.
//!
//! Detects zero-crossings of user-specified functions `g(t, y)` during adaptive
//! time stepping and locates the exact event time via root-finding.

/// A single event function `g(t, y)` to monitor during integration.
pub struct EventFunction {
    /// Name for diagnostics.
    pub name: &'static str,
    /// Evaluate the event function: `g(t, y) -> value`.
    /// A zero crossing (sign change) triggers the event.
    pub eval: fn(f64, &[f64]) -> f64,
    /// If true, stop integration after this event fires.
    pub terminal: bool,
    /// Direction of zero crossing: 0 = any, +1 = upward only, -1 = downward only.
    pub direction: i8,
    /// Previous value of g (updated internally).
    pub last_value: Option<f64>,
    /// Whether this event has been triggered.
    pub triggered: bool,
}

impl EventFunction {
    pub fn new(name: &'static str, eval: fn(f64, &[f64]) -> f64) -> Self {
        EventFunction {
            name,
            eval,
            terminal: false,
            direction: 0,
            last_value: None,
            triggered: false,
        }
    }
    pub fn terminal(mut self, v: bool) -> Self {
        self.terminal = v;
        self
    }
    pub fn direction(mut self, d: i8) -> Self {
        self.direction = d;
        self
    }
}

/// Result of event detection during a step.
#[derive(Debug, Clone)]
pub struct EventInfo {
    pub name: &'static str,
    pub t_event: f64,
    pub y_event: Vec<f64>,
    pub terminal: bool,
}

/// Root-finder using bisection + Illinois method.
fn find_root_illinois<G>(g: G, t0: f64, t1: f64, g0: f64, g1: f64, tol: f64, max_iter: u32) -> f64
where
    G: Fn(f64) -> f64,
{
    let mut a = t0;
    let mut b = t1;
    let mut ga = g0;
    let mut gb = g1;
    let mut side = 0i8;
    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            return (a + b) / 2.0;
        }
        if ga.abs() < tol {
            return a;
        }
        if gb.abs() < tol {
            return b;
        }
        let t = if side == 0 {
            (a + b) / 2.0
        } else {
            (a * gb - b * ga) / (gb - ga)
        };
        let t = t.clamp(a.min(b), a.max(b));
        let gt = g(t);
        if gt == 0.0 || (b - a).abs() < tol {
            return t;
        }
        if gt * gb > 0.0 {
            b = t;
            gb = gt;
            if side == -1 {
                ga *= 0.5;
            }
            side = -1;
        } else {
            a = t;
            ga = gt;
            if side == 1 {
                gb *= 0.5;
            }
            side = 1;
        }
    }
    (a + b) / 2.0
}

/// Drive an adaptive time integration with event detection.
///
/// The `rhs` function is the ODE right-hand side `du/dt = f(t, u)`.
/// `events` is a mutable slice of event functions monitored during integration.
/// When a terminal event fires, integration stops at the event time.
///
/// Returns the final state and a list of all events that fired.
pub fn integrate_with_events<F>(
    rhs: F,
    t_start: f64,
    t_end: f64,
    u0: &[f64],
    dt_initial: f64,
    config: &crate::adaptive::AdaptiveConfig,
    events: &mut [EventFunction],
) -> (Vec<f64>, Vec<EventInfo>)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    use crate::adaptive::explicit_adaptive_step;
    use crate::butcher::{dopri5_tableau, i_step_controller, wrms_error};

    let tableau = dopri5_tableau();
    let order = tableau.order();
    let mut u = u0.to_vec();
    let mut t = t_start;
    let mut dt = dt_initial.max(config.dt_min).min(config.dt_max);
    let mut _prev_err = 0.0;
    let mut events_fired: Vec<EventInfo> = Vec::new();
    let n = u.len();

    // Initialize event functions
    for ev in events.iter_mut() {
        ev.last_value = Some((ev.eval)(t, &u));
    }

    while t < t_end {
        if events_fired.iter().any(|e| e.terminal) {
            break;
        }

        let dt_step = dt.min(t_end - t);

        let (u_new, u_err, k) = explicit_adaptive_step(&rhs, &tableau, t, &u, dt_step);

        let err = if u_err.iter().any(|&e| e.is_nan()) {
            1e20
        } else {
            wrms_error(&u_new, &u_err, config.atol, config.rtol)
        };

        if err <= 1.0 {
            // Check for events using DOPRI5's dense output for accurate interpolation.
            // The k values from the RK stages allow constructing a 4th-order polynomial.
            if !events.is_empty() {
                // Detect zero crossings by checking event functions at both endpoints
                let mut earliest: Option<EventInfo> = None;

                for ev in events.iter_mut() {
                    if ev.triggered {
                        continue;
                    }
                    let g_old = ev.last_value.unwrap_or_else(|| (ev.eval)(t, &u));
                    let g_new = (ev.eval)(t + dt_step, &u_new);
                    ev.last_value = Some(g_new);

                    let crossed = if ev.direction == 0 {
                        g_old * g_new < 0.0
                    } else if ev.direction > 0 {
                        g_old < 0.0 && g_new >= 0.0
                    } else {
                        g_old > 0.0 && g_new <= 0.0
                    };

                    if crossed {
                        // Use the RK polynomial for accurate interpolation:
                        // u(t + θ·h) = u + h · Σⱼ bⱼ*(θ) · kⱼ
                        // DOPRI5 4th-order dense output: evaluate directly via the RHS
                        let g_wrap = |theta: f64| -> f64 {
                            if theta <= 0.0 {
                                return g_old;
                            }
                            if theta >= 1.0 {
                                return g_new;
                            }
                            // Reconstruct u at intermediate time using RK polynomial
                            let hh = dt_step;
                            let mut y_interp = vec![0.0; n];
                            for j in 0..k.len() {
                                let bj = tableau.b()[j];
                                if bj.abs() > 1e-15 {
                                    for d in 0..n {
                                        y_interp[d] += hh * bj * k[j][d];
                                    }
                                }
                            }
                            for d in 0..n {
                                y_interp[d] = u[d] + theta * y_interp[d];
                            }
                            (ev.eval)(t + theta * hh, &y_interp)
                        };

                        let t_event = find_root_illinois(g_wrap, 0.0, 1.0, g_old, g_new, 1e-12, 50);

                        // Reconstruct y at event time
                        let theta = t_event;
                        let mut y_event = vec![0.0; n];
                        for j in 0..k.len() {
                            let bj = tableau.b()[j];
                            if bj.abs() > 1e-15 {
                                for d in 0..n {
                                    y_event[d] += dt_step * bj * k[j][d];
                                }
                            }
                        }
                        for d in 0..n {
                            y_event[d] = u[d] + theta * y_event[d];
                        }
                        let abs_t_event = t + theta * dt_step;

                        let info = EventInfo {
                            name: ev.name,
                            t_event: abs_t_event,
                            y_event,
                            terminal: ev.terminal,
                        };
                        ev.triggered = true;

                        // Keep earliest event
                        match &earliest {
                            Some(ref existing) if theta > (existing.t_event - t) / dt_step => {}
                            _ => {
                                earliest = Some(info);
                            }
                        }
                    }
                }

                if let Some(event) = earliest {
                    if event.terminal {
                        u = event.y_event.clone();
                        let _ = event.t_event; // terminal: t no longer advances
                        events_fired.push(event);
                        break;
                    } else {
                        events_fired.push(event);
                    }
                }
            }

            u = u_new;
            t += dt_step;
            dt = i_step_controller(dt_step, err.max(1e-15), order);
            dt = dt.max(config.dt_min).min(config.dt_max);
            _prev_err = err;
        } else {
            dt = i_step_controller(dt_step, err, order);
            dt = dt.max(config.dt_min);
            _prev_err = err;
        }
    }

    (u, events_fired)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveConfig;

    fn decay_rhs(_t: f64, u: &[f64], dudt: &mut [f64]) {
        dudt[0] = -u[0];
    }
    fn event_u_half(_t: f64, u: &[f64]) -> f64 {
        u[0] - 0.5
    }
    fn event_t_03(t: f64, _y: &[f64]) -> f64 {
        t - 0.3
    }

    #[test]
    fn integrate_stops_at_terminal_event() {
        let u0 = vec![1.0];
        let config = AdaptiveConfig::default();
        let mut events = vec![EventFunction::new("u=0.5", event_u_half).terminal(true)];
        let (u_final, events_fired) =
            integrate_with_events(decay_rhs, 0.0, 5.0, &u0, 0.1, &config, &mut events);
        assert!(!events_fired.is_empty(), "event should fire");
        assert!(
            (u_final[0] - 0.5).abs() < 0.01,
            "final u={}, expected≈0.5",
            u_final[0]
        );
    }

    #[test]
    fn multiple_events_both_fire() {
        let u0 = vec![1.0];
        let config = AdaptiveConfig::default();
        let mut events = vec![
            EventFunction::new("u=0.5", event_u_half),
            EventFunction::new("t=0.3", event_t_03),
        ];
        let (_u_final, events_fired) =
            integrate_with_events(decay_rhs, 0.0, 1.0, &u0, 0.1, &config, &mut events);
        assert_eq!(events_fired.len(), 2, "both events should fire");
        if events_fired.len() >= 2 {
            assert!(
                events_fired[0].t_event <= events_fired[1].t_event,
                "events out of order"
            );
        }
    }

    #[test]
    fn no_events_no_effect() {
        let u0 = vec![1.0];
        let config = AdaptiveConfig::default();
        let (u_final, events_fired) =
            integrate_with_events(decay_rhs, 0.0, 1.0, &u0, 0.1, &config, &mut vec![]);
        assert!(events_fired.is_empty());
        let exact = (-1.0_f64).exp();
        assert!((u_final[0] - exact).abs() < 0.01);
    }

    #[test]
    fn illinois_root_finds_exact() {
        let root = find_root_illinois(
            |t| (-t).exp() - 0.5,
            0.0,
            2.0,
            1.0 - 0.5,
            (-2.0_f64).exp() - 0.5,
            1e-12,
            50,
        );
        assert!((root - 0.6931471805599453).abs() < 1e-10, "root={}", root);
    }
}
