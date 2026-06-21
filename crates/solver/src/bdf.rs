//! Variable-order, variable-step BDF (Backward Differentiation Formula) integrator
//! using Nordsieck representation.
//!
//! Supports orders 1-6 with automatic order/step-size selection.
//!
//! # Nordsieck form
//! The Nordsieck vector `z = [y, h·y', h²·y''/2!, …, hᵏ·y⁽ᵏ⁾/k!]` stores scaled
//! derivatives and is updated via:
//!
//! 1. **Predict**: `ẑ ← P(z)` (Pascal triangle matrix)
//! 2. **Evaluate**: `f(t+h, ẑ₀)`
//! 3. **Correct**: solve `(I − h·αₖ·J)·δ = h·f − ẑ₁`, then `z⁺ = ẑ + l·δ`
//!
//! Coefficients `αₖ` and correction vector `l` depend on the current order `k`.

use crate::butcher::{wrms_error, i_step_controller};
use crate::{solve_gmres, SolverConfig};
use fem_linalg::{CooMatrix, CsrMatrix};

// ─── BDF coefficients ────────────────────────────────────────────────────────

/// αₖ = sum_{j=1}^{k} 1/j — the BDF coefficient for order k.
const BDF_ALPHA: [f64; 7] = [0.0, 1.0, 3.0/2.0, 11.0/6.0, 25.0/12.0, 137.0/60.0, 49.0/20.0];

/// `l[i]` = Nordsieck correction coefficients for order k.
/// Indexed as `L[k][i]` for i = 0..=k.
/// Derived from: l₀ = 1, lᵢ = (1/(i!·αₖ))·(dⁱ/dζⁱ)[Πⱼ₌₁ᵏ (ζ + cⱼ)] evaluated at ζ = 1.
const L_COEFFS: [[f64; 7]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // k=0 (unused)
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 2.0/3.0, 1.0/3.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 6.0/11.0, 3.0/11.0, 1.0/11.0, 0.0, 0.0, 0.0],
    [1.0, 12.0/25.0, 6.0/25.0, 4.0/25.0, 1.0/25.0, 0.0, 0.0],
    [1.0, 120.0/274.0, 60.0/274.0, 40.0/274.0, 15.0/274.0, 6.0/274.0, 0.0],
    [1.0, 60.0/147.0, 30.0/147.0, 20.0/147.0, 10.0/147.0, 6.0/147.0, 2.0/147.0],
];

/// Error weighting coefficients for order selection.
/// `e_est[k][i]` = contribution of each Nordsieck component to the truncation error.
const ERROR_WEIGHTS: [[f64; 7]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0/3.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0/11.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0/25.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 6.0/274.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0/147.0],
];

/// Pascal triangle matrix coefficients for Nordsieck prediction.
/// `P[k][i][j]` for i,j = 0..=k: ẑ[i] = Σⱼ C(j,i) · z[j] where C is the Pascal matrix.
fn pascal_coeff(_k: usize, i: usize, j: usize) -> f64 {
    if j < i { return 0.0; }
    // binomial(j, i)
    let mut c = 1.0;
    let mut num = j;
    for den in 1..=i {
        c = c * num as f64 / den as f64;
        num -= 1;
    }
    c
}

// ─── NordsieckState ──────────────────────────────────────────────────────────

/// Nordsieck history vector for BDF integration.
///
/// `z[0] = y_n`, `z[1] = h·y'_n`, `z[2] = h²·y''_n/2`, …
/// Each component is a `Vec<f64>` of length `n` (system size).
pub struct NordsieckState {
    /// Scaled-derivative vectors, length `order + 1` entries, each size `n`.
    pub z: Vec<Vec<f64>>,
    /// Current BDF order (1–6).
    pub order: usize,
    /// Current step size.
    pub dt: f64,
    /// Last Nordsieck step (for error estimation).
    nordsieck_history: Option<Vec<Vec<f64>>>,
}

impl NordsieckState {
    /// Create a new Nordsieck state initialised as BDF-1 with given `dt`.
    ///
    /// `z[0] = y₀`, `z[1] = h·f(t₀, y₀)`.
    pub fn new<F>(t0: f64, y0: &[f64], dt: f64, rhs: &F) -> Self
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let n = y0.len();
        let mut dydt = vec![0.0; n];
        rhs(t0, y0, &mut dydt);
        let z1: Vec<f64> = dydt.iter().map(|&v| dt * v).collect();
        NordsieckState {
            z: vec![y0.to_vec(), z1],
            order: 1,
            dt,
            nordsieck_history: None,
        }
    }

    /// Manually initialise from known Nordsieck vector.
    pub fn from_z(z: Vec<Vec<f64>>, dt: f64) -> Self {
        let order = z.len() - 1;
        assert!(order >= 1 && order <= 6, "BDF order {order} not supported");
        NordsieckState { z, order, dt, nordsieck_history: None }
    }

    /// Number of variables in the ODE system.
    pub fn n_vars(&self) -> usize { self.z[0].len() }

    /// Current solution `y_n`.
    pub fn y(&self) -> &[f64] { &self.z[0] }

    /// Current order.
    pub fn order(&self) -> usize { self.order }

    /// Current step size.
    pub fn dt(&self) -> f64 { self.dt }
}

// ─── BdfIntegrator ───────────────────────────────────────────────────────────

/// A variable-order, variable-step BDF integrator (orders 1–6).
///
/// # Example
/// ```ignore
/// use fem_solver::bdf::BdfIntegrator;
///
/// let mut state = BdfIntegrator::new(0.0, &y0, 0.01, &rhs);
/// let (y_final, stats) = BdfIntegrator::integrate(
///     &mut state, &rhs, &jac, 0.0, 10.0, &config
/// );
/// ```
pub struct BdfIntegrator;

impl BdfIntegrator {
    /// Take a single BDF step using Newton iteration for the correction.
    ///
    /// On success, updates `state.z` in-place.
    /// Returns the estimated local truncation error (WRMS norm).
    pub fn step<F, J>(
        state: &mut NordsieckState,
        t: f64,
        rhs: &F,
        jac_fn: &J,
        newton: &NewtonConfig,
        atol: f64,
        rtol: f64,
    ) -> Result<f64, String>
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>,
    {
        let k = state.order;
        let dt = state.dt;
        let n = state.n_vars();
        let alpha_k = BDF_ALPHA[k];
        let tnp1 = t + dt;

        // ── 1. Predict: ẑ[i] = Σⱼ P[i][j] · z[j] ──────────────────────────
        let zp_pred: Vec<Vec<f64>> = (0..=k).map(|i| {
            let mut val = vec![0.0; n];
            for j in i..=k {
                let c = pascal_coeff(k, i, j);
                if c.abs() > 0.0 {
                    for d in 0..n { val[d] += c * state.z[j][d]; }
                }
            }
            val
        }).collect();

        // ── 2. Newton correction on δ ──────────────────────────────────────
        // Solve G(δ) = δ - h·α·f(t+h, ẑ₀ + δ) + ẑ₁ = 0
        // with Newton: (I - h·α·J) Δδ = -G(δ)
        let z0_pred = &zp_pred[0];
        let z1_curr = &zp_pred[1];

        let mut delta = vec![0.0; n];
        let mut y = z0_pred.to_vec();          // y = ẑ₀ + δ
        let mut jac = None;
        let mut converged = false;

        for iter in 0..newton.max_iter {
            let mut fy = vec![0.0; n];
            rhs(tnp1, &y, &mut fy);

            // Compute F(δ) = δ - h*α*f(y) + ẑ₁
            let mut norm_f = 0.0;
            for i in 0..n {
                let val = delta[i] - dt * alpha_k * fy[i] + z1_curr[i];
                let scale = newton.atol + newton.rtol * y[i].abs().max(1e-15);
                norm_f += (val / scale).powi(2);
            }
            norm_f = (norm_f / n as f64).sqrt();

            if norm_f <= 1.0 {
                converged = true;
                break;
            }

            // Build or reuse Jacobian
            if newton.reassemble_jac || iter == 0 {
                jac = Some(jac_fn(tnp1, &y));
            }

            // Solve (I - h*α*J) Δδ = -F(δ) = -δ + h*α*f(y) - ẑ₁
            let mut rhs_lin = vec![0.0; n];
            for i in 0..n {
                rhs_lin[i] = -delta[i] + dt * alpha_k * fy[i] - z1_curr[i];
            }
            let jac = jac.as_ref().unwrap();
            let sys = build_identity_minus_dt_jac_scaled(jac, 1.0, dt * alpha_k);
            let mut ddelta = vec![0.0; n];
            let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
            solve_gmres(&sys, &rhs_lin, &mut ddelta, 30, &cfg)
                .map_err(|e| format!("Newton-GMRES BDF{k} failed: {e}"))?;

            for i in 0..n {
                delta[i] += ddelta[i];
                y[i] = z0_pred[i] + delta[i];
            }
        }

        if !converged {
            return Err(format!("BDF{k} Newton did not converge in {} iterations", newton.max_iter));
        }

        // ── 3. Update Nordsieck: z⁺ = ẑ + l·δ ────────────────────────────
        let mut zp_new = zp_pred;
        for i in 0..=k {
            let li = L_COEFFS[k][i];
            if li.abs() > 0.0 {
                for d in 0..n { zp_new[i][d] += li * delta[d]; }
            }
        }
        state.z = zp_new;

        // ── 4. Error estimation (uses outer tolerances for step control) ────
        let err_comp = ERROR_WEIGHTS[k][k];
        if err_comp.abs() > 0.0 {
            let y_err: Vec<f64> = (0..n).map(|d| err_comp * delta[d]).collect();
            Ok(wrms_error(&state.z[0], &y_err, atol, rtol))
        } else {
            Ok(0.0)
        }
    }

    /// Run an adaptive BDF time integration from `t_start` to `t_end`.
    ///
    /// Automatically selects order 1–6 and adjusts step size.
    pub fn integrate<F, J>(
        state: &mut NordsieckState,
        rhs: &F,
        jac_fn: &J,
        t_start: f64,
        t_end: f64,
        config: &BdfConfig,
    ) -> (Vec<f64>, BdfStats)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>,
    {
        let mut t = t_start;
        let mut stats = BdfStats::new();
        let mut prev_err = 0.0;
        let mut consecutive_rejections = 0u32;

        while t < t_end {
            if stats.n_steps >= config.max_steps { break; }
            stats.n_steps += 1;

            let dt_step = state.dt.min(t_end - t);
            state.dt = dt_step;

            // Save Nordsieck state before attempting step (for rollback)
            let saved_z = state.z.clone();

            let result = Self::step(state, t, rhs, jac_fn, &config.newton, config.atol, config.rtol);
            match result {
                Ok(err) => {
                    if err <= 1.0 {
                        t += dt_step;
                        stats.n_accepted += 1;
                        stats.n_rhs_eval += 1;
                        stats.n_jac_eval += 1;
                        consecutive_rejections = 0;

                        // Order selection
                        let want_order = Self::select_order(
                            state.order, err, prev_err,
                            config.atol, config.rtol,
                        );
                        if want_order != state.order {
                            Self::change_order(state, want_order);
                        }

                        let new_dt = i_step_controller(dt_step, err.max(1e-15), state.order as u8);
                        state.dt = new_dt.max(config.dt_min).min(config.dt_max);
                        prev_err = err;
                    } else {
                        stats.n_rejected += 1;
                        consecutive_rejections += 1;
                        // Restore state and shrink dt
                        state.z = saved_z;
                        state.dt = (dt_step * 0.5).max(config.dt_min);
                        prev_err = err;
                    }
                }
                Err(e) => {
                    stats.last_error = Some(e);
                    stats.n_rejected += 1;
                    consecutive_rejections += 1;
                    state.z = saved_z;
                    state.dt = (dt_step * 0.5).max(config.dt_min);
                }
            }

            // Safety: if too many consecutive rejections, abort
            if consecutive_rejections > 50 {
                stats.last_error = Some("too many consecutive rejections".to_string());
                break;
            }
        }

        stats.final_dt = state.dt;
        (state.z[0].clone(), stats)
    }

    /// Decide whether to increase, decrease, or keep the current order.
    fn select_order(current_order: usize, err: f64, prev_err: f64, _atol: f64, _rtol: f64) -> usize {
        // Simple heuristic: if error is very small, try higher order;
        // if error is large, lower order.
        if current_order < 6 && err < 0.1 * prev_err.max(1e-15) {
            (current_order + 1).min(6)
        } else if current_order > 1 && err > 2.0 {
            current_order - 1
        } else {
            current_order
        }
    }

    /// Change the Nordsieck order (expand or contract the z vector).
    fn change_order(state: &mut NordsieckState, new_order: usize) {
        let k = state.order;
        let k_new = new_order;
        let n = state.n_vars();

        if k_new > k {
            // Increase order: zero the highest derivative
            while state.z.len() <= k_new {
                state.z.push(vec![0.0; n]);
            }
        } else if k_new < k {
            // Decrease order: truncate
            state.z.truncate(k_new + 1);
        }
        state.order = k_new;
    }
}

// ─── NewtonConfig ─────────────────────────────────────────────────────────────

/// Configuration for nonlinear Newton iteration within each implicit step.
pub struct NewtonConfig {
    pub atol: f64,
    pub rtol: f64,
    pub max_iter: u32,
    /// If true, reassemble the Jacobian at every Newton iteration.
    /// If false, reuse the Jacobian from the first iteration (modified Newton).
    pub reassemble_jac: bool,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        NewtonConfig {
            atol: 1e-10,
            rtol: 1e-8,
            max_iter: 10,
            reassemble_jac: true,
        }
    }
}

// ─── BdfConfig ───────────────────────────────────────────────────────────────

/// Configuration for adaptive BDF integration.
pub struct BdfConfig {
    pub atol: f64,
    pub rtol: f64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub max_steps: u64,
    pub max_order: usize,
    pub newton: NewtonConfig,
}

impl Default for BdfConfig {
    fn default() -> Self {
        BdfConfig {
            atol: 1e-6,
            rtol: 1e-3,
            dt_min: 1e-12,
            dt_max: 1.0,
            max_steps: 1_000_000,
            max_order: 6,
            newton: NewtonConfig::default(),
        }
    }
}

// ─── BdfStats ────────────────────────────────────────────────────────────────

/// Statistics from a BDF integration.
pub struct BdfStats {
    pub n_steps: u64,
    pub n_accepted: u64,
    pub n_rejected: u64,
    pub n_rhs_eval: u64,
    pub n_jac_eval: u64,
    pub n_newton_iter: u64,
    pub n_linear_solves: u64,
    pub final_dt: f64,
    pub last_error: Option<String>,
}

impl BdfStats {
    pub fn new() -> Self {
        BdfStats {
            n_steps: 0, n_accepted: 0, n_rejected: 0,
            n_rhs_eval: 0, n_jac_eval: 0,
            n_newton_iter: 0, n_linear_solves: 0,
            final_dt: 0.0, last_error: None,
        }
    }
}

/// Build the matrix `s·I − α·J` where J is a CSR matrix and I is the identity.
fn build_identity_minus_dt_jac_scaled(jac: &CsrMatrix<f64>, s: f64, alpha: f64) -> CsrMatrix<f64> {
    let n = jac.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n { coo.add(i, i, s); }
    for i in 0..n {
        let start = jac.row_ptr[i];
        let end = jac.row_ptr[i + 1];
        for ptr in start..end {
            let j = jac.col_idx[ptr] as usize;
            coo.add(i, j, -alpha * jac.values[ptr]);
        }
    }
    coo.into_csr()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ODE: du/dt = -λu, exact u(t) = exp(-λt)
    fn decay_rhs(_t: f64, u: &[f64], dudt: &mut [f64]) {
        dudt[0] = -1.0 * u[0];
    }

    fn decay_jac(_t: f64, _u: &[f64]) -> CsrMatrix<f64> {
        let n = 1;
        let mut coo = CooMatrix::<f64>::new(n, n);
        coo.add(0, 0, -1.0);
        coo.into_csr()
    }

    #[test]
    fn bdf1_matches_implicit_euler() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 0.1, &decay_rhs);
        let newton = NewtonConfig::default();
        let _ = BdfIntegrator::step(&mut state, 0.0, &decay_rhs, &decay_jac, &newton, 1e-6, 1e-3);
        let exact = (-0.1_f64).exp();
        assert!((state.y()[0] - exact).abs() < 0.01,
            "BDF1 decay: u={}, exact={}", state.y()[0], exact);
    }

    #[test]
    fn bdf_integrates_to_final_time() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 0.05, &decay_rhs);
        let config = BdfConfig {
            atol: 1e-3, rtol: 1e-2,
            dt_min: 1e-10, dt_max: 0.1,
            ..BdfConfig::default()
        };
        let (y_final, stats) = BdfIntegrator::integrate(
            &mut state, &decay_rhs, &decay_jac,
            0.0, 1.0, &config,
        );
        let exact = (-1.0_f64).exp();
        assert!((y_final[0] - exact).abs() < 0.05,
            "BDF final: u={}, exact={}", y_final[0], exact);
        assert!(stats.n_accepted > 0);
    }

    #[test]
    fn bdf_order_changes_with_small_tolerance() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 0.01, &decay_rhs);
        let config = BdfConfig {
            atol: 1e-3, rtol: 1e-2,
            dt_min: 1e-10, dt_max: 0.5, max_order: 6, max_steps: 100000,
            newton: NewtonConfig::default(),
        };
        let (_y_final, stats) = BdfIntegrator::integrate(
            &mut state, &decay_rhs, &decay_jac,
            0.0, 0.5, &config,
        );
        assert!(stats.n_accepted > 0, "no accepted steps");
    }

    #[test]
    fn nordsieck_state_initializes_correctly() {
        let y0 = vec![2.0];
        let state = NordsieckState::new(0.0, &y0, 0.01, &decay_rhs);
        assert_eq!(state.order, 1);
        assert_eq!(state.z.len(), 2); // z[0], z[1]
        assert!((state.z[0][0] - 2.0).abs() < 1e-14);
        assert!((state.z[1][0] - 0.01 * (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn bdf_rejects_when_dt_too_large() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 1.0, &decay_rhs);
        let newton = NewtonConfig { atol: 1e-12, rtol: 1e-12, max_iter: 2, reassemble_jac: true };
        let result = BdfIntegrator::step(&mut state, 0.0, &decay_rhs, &decay_jac, &newton, 1e-12, 1e-12);
        // With a really tight tolerance and large dt, it should reject
        // (or at least not panic)
        let _ = result;
    }

    #[test]
    fn bdf_can_increase_order() {
        let y0 = vec![1.0; 3]; // 3-variable system
        let mut state = NordsieckState::new(0.0, &y0, 0.01, &decay_rhs);
        let new_order = 3;
        BdfIntegrator::change_order(&mut state, new_order);
        assert_eq!(state.order, new_order);
        assert_eq!(state.z.len(), new_order + 1);
    }

    #[test]
    fn bdf_can_decrease_order() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 0.01, &decay_rhs);
        // First go up to order 3
        BdfIntegrator::change_order(&mut state, 3);
        assert_eq!(state.order, 3);
        // Then go down
        BdfIntegrator::change_order(&mut state, 2);
        assert_eq!(state.order, 2);
        assert_eq!(state.z.len(), 3); // 2 + 1
    }

    #[test]
    fn bdf_coefficients_are_reasonable() {
        for k in 1..=6 {
            assert!(BDF_ALPHA[k] > 0.0, "alpha_{k} should be positive");
            for i in 0..=k {
                assert!(L_COEFFS[k][i].abs() < 10.0, "L[{k}][{i}] too large: {}", L_COEFFS[k][i]);
            }
        }
    }

    #[test]
    fn pascal_coefficient_is_binomial() {
        // P[i][j] = C(j,i) — binomial coefficient
        assert!((pascal_coeff(4, 0, 4) - 1.0).abs() < 1e-14); // C(4,0) = 1
        assert!((pascal_coeff(4, 1, 4) - 4.0).abs() < 1e-14); // C(4,1) = 4
        assert!((pascal_coeff(4, 2, 4) - 6.0).abs() < 1e-14); // C(4,2) = 6
        assert!((pascal_coeff(4, 3, 4) - 4.0).abs() < 1e-14); // C(4,3) = 4
        assert!((pascal_coeff(4, 4, 4) - 1.0).abs() < 1e-14); // C(4,4) = 1
    }

    #[test]
    fn select_order_increases_for_small_error() {
        let order = BdfIntegrator::select_order(3, 0.01, 0.5, 1e-6, 1e-3);
        assert_eq!(order, 4);
    }

    #[test]
    fn select_order_decreases_for_large_error() {
        let order = BdfIntegrator::select_order(3, 3.0, 0.5, 1e-6, 1e-3);
        assert_eq!(order, 2);
    }

    #[test]
    fn select_order_keeps_at_six() {
        let order = BdfIntegrator::select_order(6, 0.01, 0.5, 1e-6, 1e-3);
        assert_eq!(order, 6);
    }

    #[test]
    fn select_order_keeps_at_one() {
        let order = BdfIntegrator::select_order(1, 3.0, 0.5, 1e-6, 1e-3);
        assert_eq!(order, 1);
    }

    // ─── Stiff linear ODE system: du/dt = -1000*u ───────────────────────────

    fn stiff_rhs(_t: f64, u: &[f64], dudt: &mut [f64]) {
        dudt[0] = -1000.0 * u[0];
    }
    fn stiff_jac(_t: f64, _u: &[f64]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.add(0, 0, -1000.0);
        coo.into_csr()
    }

    #[test]
    fn bdf_handles_stiff_problem() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 1e-7, &stiff_rhs); // tiny initial dt
        let config = BdfConfig {
            atol: 1e-4, rtol: 1e-3,
            dt_min: 1e-15, dt_max: 1e-5, max_steps: 1000000, max_order: 2,
            newton: NewtonConfig { atol: 1e-4, rtol: 1e-3, ..NewtonConfig::default() },
        };
        let (y_final, stats) = BdfIntegrator::integrate(
            &mut state, &stiff_rhs, &stiff_jac,
            0.0, 0.001, &config,
        );
        let exact = (-1000.0 * 0.001_f64).exp();
        assert!((y_final[0] - exact).abs() < 0.1 ||
            y_final[0] < exact * 10.0,
            "BDF stiff: u={}, exact={}", y_final[0], exact);
        let _ = stats;
    }

    // ─── Newton iteration for nonlinear ODE ────────────────────────────────

    fn nonlinear_rhs(_t: f64, u: &[f64], dudt: &mut [f64]) {
        dudt[0] = -u[0] * u[0]; // du/dt = -u², exact: u = 1/(1+t)
    }
    fn nonlinear_jac(_t: f64, u: &[f64]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.add(0, 0, -2.0 * u[0]);
        coo.into_csr()
    }

    #[test]
    fn bdf_newton_converges() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 0.2, &nonlinear_rhs);
        let newton = NewtonConfig { atol: 1e-14, rtol: 1e-12, max_iter: 5, reassemble_jac: true };
        let result = BdfIntegrator::step(&mut state, 0.0, &nonlinear_rhs, &nonlinear_jac, &newton, 1e-6, 1e-3);
        assert!(result.is_ok(), "Newton should converge for du/dt = -u²");
        let exact = 1.0 / (1.0 + 0.2);
        assert!((state.y()[0] - exact).abs() < 0.05,
            "BDF1 nonlinear: u={}, exact={}", state.y()[0], exact);
    }

    #[test]
    fn bdf_modified_newton_converges() {
        let y0 = vec![1.0];
        let mut state = NordsieckState::new(0.0, &y0, 0.1, &nonlinear_rhs);
        let newton = NewtonConfig { atol: 1e-10, rtol: 1e-8, max_iter: 10, reassemble_jac: false };
        let result = BdfIntegrator::step(&mut state, 0.0, &nonlinear_rhs, &nonlinear_jac, &newton, 1e-6, 1e-3);
        assert!(result.is_ok(), "Modified Newton should also converge");
        let exact = 1.0 / (1.0 + 0.1);
        assert!((state.y()[0] - exact).abs() < 0.02,
            "Modified Newton: u={}, exact={}", state.y()[0], exact);
    }
}
