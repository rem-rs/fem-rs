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

use fem_linalg::{CooMatrix, CsrMatrix};

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
    let mut lam = problem.terminal_gradient(&u);
    let mut _t = t_end;

    // Backward Euler for the adjoint (stable backward in time)
    for i in (1..checkpoints.len()).rev() {
        let cp_prev = &checkpoints[i - 1];
        let cp_curr = &checkpoints[i];
        let dt_step = cp_curr.t - cp_prev.t;
        if dt_step < 1e-15 { continue; }

        // Current λ is at time t (start of this backward step)
        // We need λ at time t - dt

        // Evaluate A = ∂f/∂u at the checkpoint state
        let A = problem.jacobian(cp_curr.t, &cp_curr.u);

        // Evaluate ∂g/∂u
        let mut dgdu = vec![0.0; n];
        problem.cost_gradient(cp_curr.t, &cp_curr.u, &mut dgdu);

        // Backward Euler: λ_prev = λ + dt · (Aᵀ · λ + ∂g/∂u)
        // Solve (I - dt·Aᵀ) · λ_prev = λ + dt·∂g/∂u
        // Build I - dt·Aᵀ
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        for row in 0..n {
            let start = A.row_ptr[row];
            let end = A.row_ptr[row + 1];
            for ptr in start..end {
                let col = A.col_idx[ptr] as usize;
                let val = A.values[ptr];
                // Aᵀ: transpose → add at (col, row)
                coo.add(col, row, -dt_step * val);
            }
        }
        let sys = coo.into_csr();

        // RHS: λ + dt·∂g/∂u
        let mut rhs = vec![0.0; n];
        for i in 0..n { rhs[i] = lam[i] + dt_step * dgdu[i]; }

        let mut lam_prev = vec![0.0; n];
        let cfg = crate::SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..crate::SolverConfig::default() };
        match crate::solve_gmres(&sys, &rhs, &mut lam_prev, 30, &cfg) {
            Ok(_) => lam = lam_prev,
            Err(_e) => {
                log::warn!("Adjoint GMRES solve failed, using explicit Euler fallback: {}", _e);
                // Fallback: explicit Euler for adjoint (Aᵀ·λ)
                let mut At_lam = vec![0.0; n];
                for row in 0..n {
                    let start = A.row_ptr[row];
                    let end = A.row_ptr[row + 1];
                    for ptr in start..end {
                        let col = A.col_idx[ptr] as usize;
                        At_lam[col] += A.values[ptr] * lam[row];
                    }
                }
                for i in 0..n { lam_prev[i] = lam[i] + dt_step * (At_lam[i] + dgdu[i]); }
                lam = lam_prev;
            }
        }

        _t = cp_prev.t;
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
