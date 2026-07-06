//! Explicit time integrators.
//!
//! Methods: [`ForwardEuler`] (1st order), [`Rk4`] (4th order),
//! [`Rk45`] (adaptive Dormand–Prince 4/5), [`AdamsBashforthMoulton`] (multi-step PECE).

use super::traits::TimeStepper;

// ─── Forward Euler ───────────────────────────────────────────────────────────

/// Explicit forward Euler: `u_{n+1} = uₙ + dt f(tₙ, uₙ)`.
///
/// First-order accurate.  Stability requires `dt * ρ(∂f/∂u) ≤ 2`.
pub struct ForwardEuler;

impl TimeStepper for ForwardEuler {
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let n = u.len();
        let mut dudt = vec![0.0_f64; n];
        rhs(t, u, &mut dudt);
        for i in 0..n {
            u[i] += dt * dudt[i];
        }
    }
}

// ─── RK4 ─────────────────────────────────────────────────────────────────────

/// Classic 4th-order Runge–Kutta.
///
/// `u_{n+1} = uₙ + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)`
pub struct Rk4;

impl TimeStepper for Rk4 {
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let n = u.len();
        let mut k1 = vec![0.0_f64; n];
        let mut k2 = vec![0.0_f64; n];
        let mut k3 = vec![0.0_f64; n];
        let mut k4 = vec![0.0_f64; n];
        let mut tmp = vec![0.0_f64; n];

        rhs(t, u, &mut k1);

        for i in 0..n { tmp[i] = u[i] + 0.5 * dt * k1[i]; }
        rhs(t + 0.5 * dt, &tmp, &mut k2);

        for i in 0..n { tmp[i] = u[i] + 0.5 * dt * k2[i]; }
        rhs(t + 0.5 * dt, &tmp, &mut k3);

        for i in 0..n { tmp[i] = u[i] + dt * k3[i]; }
        rhs(t + dt, &tmp, &mut k4);

        for i in 0..n {
            u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
}

// ─── RK45 (Dormand–Prince, adaptive) ─────────────────────────────────────────

/// Adaptive step-size RK45 (Dormand–Prince) integrator.
///
/// Uses a 4th-order solution for stepping and a 5th-order estimate for
/// error control.  `atol` and `rtol` control step acceptance.
pub struct Rk45 {
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
    /// Minimum allowed step size.
    pub dt_min: f64,
    /// Maximum allowed step size.
    pub dt_max: f64,
}

impl Default for Rk45 {
    fn default() -> Self {
        Rk45 { atol: 1e-6, rtol: 1e-6, dt_min: 1e-12, dt_max: 1.0 }
    }
}

// Dormand–Prince Butcher tableau coefficients
const DP_A21: f64 = 1.0/5.0;
const DP_A31: f64 = 3.0/40.0;   const DP_A32: f64 = 9.0/40.0;
const DP_A41: f64 = 44.0/45.0;  const DP_A42: f64 = -56.0/15.0; const DP_A43: f64 = 32.0/9.0;
const DP_A51: f64 = 19372.0/6561.0; const DP_A52: f64 = -25360.0/2187.0;
const DP_A53: f64 = 64448.0/6561.0; const DP_A54: f64 = -212.0/729.0;
const DP_A61: f64 = 9017.0/3168.0; const DP_A62: f64 = -355.0/33.0;
const DP_A63: f64 = 46732.0/5247.0; const DP_A64: f64 = 49.0/176.0; const DP_A65: f64 = -5103.0/18656.0;

// 4th-order weights (b)
const DP_B1: f64 = 35.0/384.0; const DP_B3: f64 = 500.0/1113.0;
const DP_B4: f64 = 125.0/192.0; const DP_B5: f64 = -2187.0/6784.0; const DP_B6: f64 = 11.0/84.0;

// Error weights (b - b*)
const DP_E1: f64 = 71.0/57600.0; const DP_E3: f64 = -71.0/16695.0;
const DP_E4: f64 = 71.0/1920.0; const DP_E5: f64 = -17253.0/339200.0;
const DP_E6: f64 = 22.0/525.0; const DP_E7: f64 = -1.0/40.0;

impl Rk45 {
    /// Advance from `t` to `t_end` starting with step `dt`, updating `u`.
    /// Returns the final time reached and final step size.
    pub fn integrate<F>(&self, t0: f64, t_end: f64, u: &mut [f64], mut dt: f64, rhs: F) -> (f64, f64)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let n = u.len();
        let mut t = t0;
        let mut k1 = vec![0.0_f64; n];
        let mut k2 = vec![0.0_f64; n];
        let mut k3 = vec![0.0_f64; n];
        let mut k4 = vec![0.0_f64; n];
        let mut k5 = vec![0.0_f64; n];
        let mut k6 = vec![0.0_f64; n];
        let mut k7 = vec![0.0_f64; n];
        let mut tmp = vec![0.0_f64; n];

        while t < t_end {
            dt = dt.min(t_end - t).max(self.dt_min);

            rhs(t, u, &mut k1);
            for i in 0..n { tmp[i] = u[i] + dt * DP_A21 * k1[i]; }
            rhs(t + dt/5.0, &tmp, &mut k2);
            for i in 0..n { tmp[i] = u[i] + dt * (DP_A31*k1[i] + DP_A32*k2[i]); }
            rhs(t + 3.0*dt/10.0, &tmp, &mut k3);
            for i in 0..n { tmp[i] = u[i] + dt * (DP_A41*k1[i] + DP_A42*k2[i] + DP_A43*k3[i]); }
            rhs(t + 4.0*dt/5.0, &tmp, &mut k4);
            for i in 0..n { tmp[i] = u[i] + dt * (DP_A51*k1[i] + DP_A52*k2[i] + DP_A53*k3[i] + DP_A54*k4[i]); }
            rhs(t + 8.0*dt/9.0, &tmp, &mut k5);
            for i in 0..n { tmp[i] = u[i] + dt * (DP_A61*k1[i] + DP_A62*k2[i] + DP_A63*k3[i] + DP_A64*k4[i] + DP_A65*k5[i]); }
            rhs(t + dt, &tmp, &mut k6);

            // 4th-order solution
            let mut u4 = u.to_vec();
            for i in 0..n {
                u4[i] += dt * (DP_B1*k1[i] + DP_B3*k3[i] + DP_B4*k4[i] + DP_B5*k5[i] + DP_B6*k6[i]);
            }
            rhs(t + dt, &u4, &mut k7);

            // Error estimate
            let err: f64 = (0..n).map(|i| {
                let e = dt * (DP_E1*k1[i] + DP_E3*k3[i] + DP_E4*k4[i] + DP_E5*k5[i] + DP_E6*k6[i] + DP_E7*k7[i]);
                let sc = self.atol + self.rtol * u[i].abs().max(u4[i].abs());
                (e / sc).powi(2)
            }).sum::<f64>().sqrt() / (n as f64).sqrt();

            if err <= 1.0 || dt <= self.dt_min {
                // Accept step
                u.copy_from_slice(&u4);
                t += dt;
            }

            // Adjust step size (PI controller with safety 0.9)
            if err > 0.0 {
                dt *= (0.9 / err).powf(0.2).clamp(0.1, 5.0);
            } else {
                dt *= 5.0;
            }
            dt = dt.min(self.dt_max).max(self.dt_min);
        }
        (t, dt)
    }
}

// ─── Adams-Bashforth-Moulton (PECE, order 2–4) ──────────────────────────────

/// State for the multi-step Adams–Bashforth–Moulton method.
///
/// Stores the RHS history `{f(t_{n-i}, u_{n-i})}` for the multi-step formula.
/// The history has length `order − 1`; before it is full the integrator uses
/// an RK4 startup procedure.
pub struct AbmState {
    /// Order of the ABM method (2, 3, or 4).
    pub order: usize,
    /// Ring buffer of previous RHS values: `buf[i] = f(t_{n-i}, u_{n-i})`.
    pub buf: Vec<Vec<f64>>,
    /// Write index (next position to fill in the ring buffer).
    pub head: usize,
    /// Number of steps taken so far (used to detect the startup phase).
    pub steps: usize,
}

impl AbmState {
    /// Create a new ABM state for the given order.
    pub fn new(order: usize) -> Self {
        let cap = order;  // need `order` entries for the AB/AM formulas
        let buf = vec![vec![0.0_f64; 1]; cap];
        AbmState { order, buf, head: 0, steps: 0 }
    }
}

/// Adams–Bashforth–Moulton PECE integrator (orders 2–4).
///
/// Predictor: Adams–Bashforth (explicit, multi-step).  
/// Corrector: Adams–Moulton (treated explicitly via PECE).  
///
/// Startup: lower-order ABM formulas + RK4 are used for the first steps until
/// the full history is available.
pub struct AdamsBashforthMoulton;

// AB coefficients: β[i] for f(t_{n-i}, u_{n-i}) at order o
const AB_COEF: [[f64; 4]; 4] = [
    [1.0,    0.0,     0.0,    0.0   ],   // order 1 (Forward Euler)
    [ 3.0/2.0, -1.0/2.0, 0.0,  0.0 ],   // order 2
    [23.0/12.0, -16.0/12.0, 5.0/12.0, 0.0], // order 3
    [55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0], // order 4
];

// AM coefficients: γ[i] for f(t_{n+1-i}, ...) at order o (γ[0] is for f_{n+1})
const AM_COEF: [[f64; 4]; 4] = [
    [1.0,    0.0,    0.0,    0.0   ],   // order 1
    [ 1.0/2.0,  1.0/2.0, 0.0,  0.0  ],   // order 2
    [ 5.0/12.0, 8.0/12.0, -1.0/12.0, 0.0], // order 3
    [ 9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0], // order 4
];

impl AdamsBashforthMoulton {
    /// Advance `u` from `t` by `dt`, using the provided `state`.
    ///
    /// Startup uses progressively lower-order ABM formulas (order 1, then 2, 3,
    /// etc.) until the full history buffer is filled.
    pub fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], state: &mut AbmState, rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]) + Clone,
    {
        let n = u.len();
        let cap = state.buf.len();

        // Resize buffer to match problem dimension on first call
        if state.steps == 0 && state.buf[0].len() != n {
            for b in &mut state.buf {
                b.resize(n, 0.0);
            }
        }

        // Evaluate f(t_n, u_n) for the current step
        let mut fn_cur = vec![0.0_f64; n];
        rhs(t, u, &mut fn_cur);

        if state.steps < cap {
            // RK4 startup: advance u without using the ABM formula,
            // then store f_{n+1} in the buffer for future ABM use.
            let rk4 = Rk4;
            rk4.step(t, dt, u, rhs.clone());

            let mut fn1 = vec![0.0_f64; n];
            rhs(t + dt, u, &mut fn1);
            state.buf[state.head] = fn1;
            state.head = (state.head + 1) % cap;
            state.steps += 1;
            return;
        }

        // ── Full ABM step ────────────────────────────────────────────────
        let effective_order = state.order;

        // 1. AB Predictor
        let ab = &AB_COEF[effective_order - 1];
        let mut u_star = u.to_vec();
        for oi in 0..effective_order {
            let bi = ab[oi];
            if bi == 0.0 { continue; }
            let idx = ((state.head as i64 - 1 - oi as i64).rem_euclid(cap as i64)) as usize;
            let fi = &state.buf[idx];
            for j in 0..n {
                u_star[j] += dt * bi * fi[j];
            }
        }

        // 2. Evaluate predictor RHS
        let mut f_star = vec![0.0_f64; n];
        rhs(t + dt, &u_star, &mut f_star);

        // 3. AM Corrector (explicit, using f_star in place of implicit f_{n+1})
        let am = &AM_COEF[effective_order - 1];
        let am_len = effective_order;  // AM coefficients are packed with γ₀ for f_{n+1}
        let mut u_new = u.to_vec();
        for j in 0..n {
            u_new[j] += dt * am[0] * f_star[j];
        }
        for oi in 1..am_len {
            let gi = am[oi];
            if gi == 0.0 { continue; }
            let idx = ((state.head as i64 - 1 - (oi as i64 - 1)).rem_euclid(cap as i64)) as usize;
            let fi = &state.buf[idx];
            for j in 0..n {
                u_new[j] += dt * gi * fi[j];
            }
        }

        // 4. Evaluate RHS at the corrected state for next step's history
        let mut f_new = vec![0.0_f64; n];
        rhs(t + dt, &u_new, &mut f_new);

        // 5. Update history buffer (store f_{n+1})
        state.buf[state.head] = f_new;
        state.head = (state.head + 1) % cap;
        state.steps += 1;

        // 6. Commit solution
        u.copy_from_slice(&u_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp_decay(lambda: f64) -> impl Fn(f64, &[f64], &mut [f64]) {
        move |_t, u, dudt| { dudt[0] = -lambda * u[0]; }
    }

    #[test]
    fn forward_euler_order1() {
        let rhs = exp_decay(1.0);
        let dt_coarse = 0.01_f64;
        let dt_fine   = 0.005_f64;
        let t_end = 1.0_f64;
        let fe = ForwardEuler;
        let mut u_c = vec![1.0_f64];
        let mut t = 0.0;
        while t < t_end - 1e-14 {
            let dt = dt_coarse.min(t_end - t);
            fe.step(t, dt, &mut u_c, &rhs);
            t += dt;
        }
        let mut u_f = vec![1.0_f64];
        t = 0.0;
        while t < t_end - 1e-14 {
            let dt = dt_fine.min(t_end - t);
            fe.step(t, dt, &mut u_f, &rhs);
            t += dt;
        }
        let exact = (-1.0_f64).exp();
        let err_c = (u_c[0] - exact).abs();
        let err_f = (u_f[0] - exact).abs();
        let ratio = err_c / err_f;
        assert!(ratio > 1.5, "FE order check: ratio={ratio:.2} (expected ~2)");
    }

    #[test]
    fn rk4_order4() {
        let rhs = exp_decay(1.0);
        let rk4 = Rk4;
        let t_end = 1.0_f64;
        let exact = (-1.0_f64).exp();
        let mut u_c = vec![1.0_f64];
        let dt_c = 0.1_f64;
        let mut t = 0.0_f64;
        while t < t_end - 1e-14 {
            let dt = dt_c.min(t_end - t);
            rk4.step(t, dt, &mut u_c, &rhs);
            t += dt;
        }
        let mut u_f = vec![1.0_f64];
        let dt_f = 0.05_f64;
        t = 0.0;
        while t < t_end - 1e-14 {
            let dt = dt_f.min(t_end - t);
            rk4.step(t, dt, &mut u_f, &rhs);
            t += dt;
        }
        let err_c = (u_c[0] - exact).abs();
        let err_f = (u_f[0] - exact).abs();
        let ratio = err_c / err_f;
        assert!(ratio > 10.0, "RK4 order check: ratio={ratio:.2} (expected ~16)");
        assert!(err_f < 1e-7, "RK4 error too large: {err_f}");
    }

    #[test]
    fn rk45_adaptive_accuracy() {
        let solver = Rk45 { atol: 1e-8, rtol: 1e-8, ..Default::default() };
        let mut u = vec![1.0_f64];
        solver.integrate(0.0, 1.0, &mut u, 0.1, exp_decay(1.0));
        let exact = (-1.0_f64).exp();
        let err = (u[0] - exact).abs();
        assert!(err < 1e-6, "RK45 error={err:.3e}");
    }

    #[test]
    fn rk4_heat_convergence() {
        let lambda = std::f64::consts::PI * std::f64::consts::PI;
        let rhs = exp_decay(lambda);
        let t_end = 0.1;
        let exact = (-lambda * t_end).exp();
        let rk4 = Rk4;
        let mut errors = vec![];
        for &dt in &[0.02_f64, 0.01, 0.005] {
            let mut u = vec![1.0_f64];
            let mut t = 0.0;
            while t < t_end - 1e-14 {
                let h = dt.min(t_end - t);
                rk4.step(t, h, &mut u, &rhs);
                t += h;
            }
            errors.push((u[0] - exact).abs());
        }
        let order = (errors[0] / errors[1]).log2();
        assert!(order > 3.5, "RK4 heat convergence order={order:.2} (expected ~4)");
    }

    #[test]
    fn forward_euler_heat_convergence() {
        let lambda = std::f64::consts::PI * std::f64::consts::PI;
        let rhs = exp_decay(lambda);
        let t_end = 0.01;
        let exact = (-lambda * t_end).exp();
        let fe = ForwardEuler;
        let mut errors = vec![];
        for &dt in &[0.0005_f64, 0.00025] {
            let mut u = vec![1.0_f64];
            let mut t = 0.0;
            while t < t_end - 1e-14 {
                let h = dt.min(t_end - t);
                fe.step(t, h, &mut u, &rhs);
                t += h;
            }
            errors.push((u[0] - exact).abs());
        }
        let order = (errors[0] / errors[1]).log2();
        assert!(order > 0.8, "Forward Euler heat convergence order={order:.2} (expected ~1)");
    }

    #[test]
    fn rk4_vanderpol_limit_cycle() {
        let mu = 1.0;
        let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1];
            dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        };
        let rk4 = Rk4;
        let mut y = vec![2.0_f64, 0.0];
        let mut t = 0.0;
        let dt = 0.01_f64;
        let t_end = 6.2832;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            rk4.step(t, h, &mut y, &rhs);
            t += h;
        }
        assert!(y[0].is_finite(), "van der Pol u diverged: {}", y[0]);
        assert!(y[1].is_finite(), "van der Pol v diverged: {}", y[1]);
        assert!(y[0].abs() < 3.0, "van der Pol u out of range: {}", y[0]);
    }

    #[test]
    fn abm2_heat_convergence() {
        let lambda = std::f64::consts::PI * std::f64::consts::PI;
        let rhs = move |_t: f64, u: &[f64], dudt: &mut [f64]| { dudt[0] = -lambda * u[0]; };
        let t_end = 0.1;
        let exact = (-lambda * t_end).exp();
        let mut errors = vec![];
        for &dt in &[0.01_f64, 0.005, 0.0025] {
            let mut u = vec![1.0_f64];
            let mut t = 0.0;
            let mut state = AbmState::new(2);
            let abm = AdamsBashforthMoulton;
            while t < t_end - 1e-14 {
                let h = dt.min(t_end - t);
                abm.step(t, h, &mut u, &mut state, rhs.clone());
                t += h;
            }
            errors.push((u[0] - exact).abs());
        }
        let order = (errors[0] / errors[1]).log2();
        assert!(order > 1.5, "ABM2 heat convergence order={order:.2} (expected ~2)");
    }

    #[test]
    fn abm4_heat_convergence() {
        let lambda = std::f64::consts::PI * std::f64::consts::PI;
        let rhs = move |_t: f64, u: &[f64], dudt: &mut [f64]| { dudt[0] = -lambda * u[0]; };
        let t_end = 0.5;
        let exact = (-lambda * t_end).exp();
        let mut errors = vec![];
        for &dt in &[0.05_f64, 0.025] {
            let mut u = vec![1.0_f64];
            let mut t = 0.0;
            let mut state = AbmState::new(4);
            let abm = AdamsBashforthMoulton;
            while t < t_end - 1e-14 {
                let h = dt.min(t_end - t);
                abm.step(t, h, &mut u, &mut state, rhs.clone());
                t += h;
            }
            errors.push((u[0] - exact).abs());
        }
        let order = (errors[0] / errors[1]).log2();
        assert!(order > 2.8, "ABM4 heat convergence order={order:.2} (expected ~4, degraded by startup)");
    }
}
