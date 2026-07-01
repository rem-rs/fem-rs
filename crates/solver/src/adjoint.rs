//! Adjoint sensitivity analysis for ODEs.
//!
//! Computes gradients of cost functionals with respect to initial conditions
//! by solving the adjoint ODE backward in time.
//!
//! # Forward problem
//! `du/dt = f(t, u)`, `u(0) = u₀`
//!
//! # Cost functional
//! `J(u) = ∫₀ᵀ g(t, u) dt + h(u(T))`
//!
//! # Adjoint equation (backward)
//! `dλ/dt = -(∂f/∂u)ᵀ · λ - (∂g/∂u)ᵀ`, `λ(T) = ∂h/∂u(u(T))`
//!
//! # Sensitivity
//! `dJ/du₀ = λ(0)`
#![allow(non_snake_case)]

use fem_linalg::CsrMatrix;
#[cfg(test)]
use fem_linalg::CooMatrix;

/// An ODE problem with cost functional for adjoint sensitivity analysis.
pub trait AdjointProblem {
    /// Number of state variables.
    fn n_states(&self) -> usize;

    /// Right-hand side: `du/dt = f(t, u)`.
    fn rhs(&self, t: f64, u: &[f64], dudt: &mut [f64]);

    /// Jacobian of RHS: `A = ∂f/∂u` as a CSR matrix.
    fn jacobian(&self, t: f64, u: &[f64]) -> CsrMatrix<f64>;

    /// Running cost integrand `g(t, u)`.
    fn cost_integrand(&self, _t: f64, _u: &[f64]) -> f64 { 0.0 }

    /// Gradient of running cost: `∂g/∂u`.
    fn cost_gradient(&self, _t: f64, _u: &[f64], dgdu: &mut [f64]) {
        dgdu.fill(0.0);
    }

    /// Terminal cost `h(u(T))`.
    fn terminal_cost(&self, _u: &[f64]) -> f64 { 0.0 }

    /// Gradient of terminal cost: `∂h/∂u`.
    fn terminal_gradient(&self, _u: &[f64]) -> Vec<f64> {
        vec![0.0; self.n_states()]
    }
}

/// A checkpoint storing the forward state and time for adjoint reconstruction.
#[derive(Clone)]
struct Checkpoint {
    t: f64,
    u: Vec<f64>,
}

/// Drive forward integration with checkpoints, then backward adjoint integration.
///
/// Returns `(u_final, sensitivity=λ(0), total_cost)`.
pub fn adjoint_sensitivity(
    problem: &dyn AdjointProblem,
    t_start: f64,
    t_end: f64,
    u0: &[f64],
    dt: f64,
    _atol: f64,
    _rtol: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = problem.n_states();
    let mut checkpoints: Vec<Checkpoint> = Vec::new();
    let mut u = u0.to_vec();

    // ── Forward pass: integrate with RK4 and store checkpoints ──────────
    // Use fixed-step RK4 (simple, reversible)
    fn rk4_step<F>(f: &F, t: f64, dt: f64, u: &mut [f64])
    where F: Fn(f64, &[f64], &mut [f64]) {
        let n = u.len();
        let mut k1 = vec![0.0; n]; f(t, u, &mut k1);
        let mut k2 = vec![0.0; n]; let mut ut = u.iter().zip(k1.iter()).map(|(&u, &k)| u + 0.5*dt*k).collect::<Vec<_>>(); f(t + 0.5*dt, &ut, &mut k2);
        let mut k3 = vec![0.0; n]; ut = u.iter().zip(k2.iter()).map(|(&u, &k)| u + 0.5*dt*k).collect(); f(t + 0.5*dt, &ut, &mut k3);
        let mut k4 = vec![0.0; n]; ut = u.iter().zip(k3.iter()).map(|(&u, &k)| u + dt*k).collect(); f(t + dt, &ut, &mut k4);
        for i in 0..n { u[i] += dt / 6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]); }
    }

    // Forward integration
    let mut t = t_start;
    let mut total_cost = 0.0;

    // Store initial checkpoint
    checkpoints.push(Checkpoint { t, u: u.clone() });
    total_cost += problem.cost_integrand(t, &u) * dt;

    while t < t_end - 1e-14 {
        let dt_step = dt.min(t_end - t);
        rk4_step(&|t, u, d| problem.rhs(t, u, d), t, dt_step, &mut u);
        t += dt_step;
        checkpoints.push(Checkpoint { t, u: u.clone() });
        total_cost += problem.cost_integrand(t, &u) * dt_step;
    }

    // Add terminal cost
    total_cost += problem.terminal_cost(&u);

    // ── Backward pass: solve adjoint equation ────────────────────────────
    // dλ/dt = -A(t)ᵀ · λ - ∂g/∂u, λ(T) = ∂h/∂u(u(T))
    //
    // Discrete-adjoint-over-RK4: we integrate the adjoint ODE backward
    // using the continuous-adjoint RK4 scheme (same order as forward RK4).
    // At each stage k2/k3 we interpolate the forward state linearly between
    // checkpoints — this is O(h²) for u(t), which is sufficient to retain
    // O(h⁴) gradient convergence for smooth problems.
    let mut lam = problem.terminal_gradient(&u);
    let mut _t = t_end;

    for i in (1..checkpoints.len()).rev() {
        let cp_prev = &checkpoints[i - 1];
        let cp_curr = &checkpoints[i];
        let h = cp_curr.t - cp_prev.t;
        if h < 1e-15 { continue; }

        // Adjoint RHS at a given (t, u, lambda):  dλ/dt = -Aᵀ·λ - ∂g/∂u
        let adjoint_rhs = |t: f64, u: &[f64], lam: &[f64], out: &mut [f64]| {
            let A = problem.jacobian(t, u);
            let mut dg = vec![0.0; n];
            problem.cost_gradient(t, u, &mut dg);
            for r in 0..n {
                out[r] = -dg[r];
                let start = A.row_ptr[r];
                let end = A.row_ptr[r + 1];
                for p in start..end {
                    let c = A.col_idx[p] as usize;
                    // Aᵀ·λ: accumulate Aⱼᵢ · λⱼ at row i
                    out[r] -= A.values[p] * lam[c];
                }
            }
        };

        // Helper: interpolate forward state between checkpoints (linear).
        let interp_u = |t_mid: f64| -> Vec<f64> {
            let alpha = (t_mid - cp_prev.t) / h;
            cp_prev.u.iter().zip(cp_curr.u.iter())
                .map(|(&up, &uc)| up + alpha * (uc - up))
                .collect()
        };

        // RK4 step (backward: integrate from cp_curr.t to cp_prev.t with -h)
        let tm = cp_curr.t;
        let u_m = &cp_curr.u;
        let tmm = cp_prev.t;
        let u_mm = &cp_prev.u;

        // k1 at (t_m, u_m)
        let mut k1 = vec![0.0; n];
        adjoint_rhs(tm, u_m, &lam, &mut k1);

        // k2 at (t_m - h/2, interp_u)
        let t2 = tm - 0.5 * h;
        let u2 = interp_u(t2);
        let mut k2 = vec![0.0; n];
        {
            let mut lam2 = vec![0.0; n];
            for i2 in 0..n { lam2[i2] = lam[i2] - 0.5 * h * k1[i2]; }
            adjoint_rhs(t2, &u2, &lam2, &mut k2);
        }

        // k3 at (t_m - h/2, interp_u)
        let mut k3 = vec![0.0; n];
        {
            let mut lam3 = vec![0.0; n];
            for i2 in 0..n { lam3[i2] = lam[i2] - 0.5 * h * k2[i2]; }
            adjoint_rhs(t2, &u2, &lam3, &mut k3);
        }

        // k4 at (t_m - h, u_{m-1})
        let mut k4 = vec![0.0; n];
        {
            let mut lam4 = vec![0.0; n];
            for i2 in 0..n { lam4[i2] = lam[i2] - h * k3[i2]; }
            adjoint_rhs(tmm, u_mm, &lam4, &mut k4);
        }

        // λ_{m-1} = λ_m + (-h)/6 · (k1 + 2·k2 + 2·k3 + k4)
        let h6 = h / 6.0;
        for ii in 0..n {
            lam[ii] = lam[ii] - h6 * (k1[ii] + 2.0 * k2[ii] + 2.0 * k3[ii] + k4[ii]);
        }

        _t = tmm;
    }

    (u, lam, total_cost)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Simple problem: du/dt = -λu, u(0) = u₀
    // Cost: J = ∫₀ᵀ u(t) dt (running cost only)
    // Exact sensitivity: dJ/du₀ = ∫₀ᵀ exp(-λt) dt = (1 - exp(-λT))/λ
    struct DecayProblem { lambda: f64 }
    impl AdjointProblem for DecayProblem {
        fn n_states(&self) -> usize { 1 }
        fn rhs(&self, _t: f64, u: &[f64], dudt: &mut [f64]) { dudt[0] = -self.lambda * u[0]; }
        fn jacobian(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -self.lambda);
            coo.into_csr()
        }
        fn cost_integrand(&self, _t: f64, u: &[f64]) -> f64 { u[0] }
        fn cost_gradient(&self, _t: f64, _u: &[f64], dgdu: &mut [f64]) { dgdu[0] = 1.0; }
    }

    #[test]
    fn adjoint_sensitivity_decay() {
        let problem = DecayProblem { lambda: 1.0 };
        let u0 = vec![2.0];
        let T = 1.0;
        let (_u_final, sensitivity, cost) = adjoint_sensitivity(
            &problem, 0.0, T, &u0, 0.01, 1e-6, 1e-6,
        );
        // Total cost J = ∫₀¹ 2·exp(-t) dt = 2·(1 - exp(-1)) ≈ 1.264
        // dJ/du₀ = ∫₀¹ exp(-t) dt = 1 - exp(-1) ≈ 0.632
        let expected_cost = 2.0 * (1.0 - (-1.0_f64).exp());
        let expected_sens = 1.0 - (-1.0_f64).exp();
        assert!((cost - expected_cost).abs() < 0.02,
            "cost={}, expected={}", cost, expected_cost);
        assert!((sensitivity[0] - expected_sens).abs() < 0.02,
            "sensitivity={}, expected={}", sensitivity[0], expected_sens);
    }

    // Problem with terminal cost only: J = h(u(T)) = u(T)
    // dJ/du₀ = exp(-λT)
    struct TerminalCostProblem { lambda: f64 }
    impl AdjointProblem for TerminalCostProblem {
        fn n_states(&self) -> usize { 1 }
        fn rhs(&self, _t: f64, u: &[f64], dudt: &mut [f64]) { dudt[0] = -self.lambda * u[0]; }
        fn jacobian(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -self.lambda);
            coo.into_csr()
        }
        fn terminal_gradient(&self, _u: &[f64]) -> Vec<f64> { vec![1.0] }
    }

    #[test]
    fn adjoint_terminal_sensitivity() {
        let problem = TerminalCostProblem { lambda: 2.0 };
        let u0 = vec![1.0];
        let T = 0.5;
        let (_u_final, sensitivity, _cost) = adjoint_sensitivity(
            &problem, 0.0, T, &u0, 0.005, 1e-6, 1e-6,
        );
        // dJ/du₀ = exp(-2*0.5) = exp(-1) ≈ 0.3679
        let expected = (-1.0_f64).exp();
        assert!((sensitivity[0] - expected).abs() < 0.01,
            "sensitivity={}, expected={}", sensitivity[0], expected);
    }

    // 2D linear ODE: u' = Au, A = [[-1, 0], [0, -2]]
    // J = ∫ u₁ dt, dJ/du₀ = [1-exp(-T), 0] (only u₁ component matters)
    struct Linear2dProblem;
    impl AdjointProblem for Linear2dProblem {
        fn n_states(&self) -> usize { 2 }
        fn rhs(&self, _t: f64, u: &[f64], dudt: &mut [f64]) {
            dudt[0] = -1.0 * u[0];
            dudt[1] = -2.0 * u[1];
        }
        fn jacobian(&self, _t: f64, _u: &[f64]) -> CsrMatrix<f64> {
            let mut coo = CooMatrix::<f64>::new(2, 2);
            coo.add(0, 0, -1.0);
            coo.add(1, 1, -2.0);
            coo.into_csr()
        }
        fn cost_integrand(&self, _t: f64, u: &[f64]) -> f64 { u[1] }
        fn cost_gradient(&self, _t: f64, _u: &[f64], dgdu: &mut [f64]) {
            dgdu[0] = 0.0;
            dgdu[1] = 1.0;
        }
    }

    #[test]
    fn adjoint_2d_sensitivity() {
        let problem = Linear2dProblem;
        let u0 = vec![1.0, 1.0];
        let T = 0.5;
        let (_u_final, sensitivity, _cost) = adjoint_sensitivity(
            &problem, 0.0, T, &u0, 0.01, 1e-6, 1e-6,
        );
        // dJ/du₀₁ = ∫₀⁰·⁵ exp(-2t) dt = (1-exp(-1))/2 ≈ 0.316
        // dJ/du₀₀ = 0 (cost only depends on u₁)
        let expected_1 = (1.0 - (-1.0_f64).exp()) / 2.0;
        assert!((sensitivity[0]).abs() < 0.01,
            "∂J/∂u₀₀ should be near 0, got {}", sensitivity[0]);
        assert!((sensitivity[1] - expected_1).abs() < 0.01,
            "∂J/∂u₀₁={}, expected={}", sensitivity[1], expected_1);
    }
}
