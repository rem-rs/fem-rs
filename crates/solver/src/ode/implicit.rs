//! Implicit / A-stable time integrators.
//!
//! Methods: [`ImplicitEuler`] (BDF-1), [`Sdirk2`], [`Bdf2`], [`CrankNicolson`].

use crate::{SolverConfig, solve_gmres};
use super::traits::ImplicitTimeStepper;

// ─── Helper: build (sI - αJ) ─────────────────────────────────────────────────

/// Build `I − α J` as a CsrMatrix (convenience wrapper).
pub(super) fn identity_minus_dt_jac(jac: &fem_linalg::CsrMatrix<f64>, alpha: f64) -> fem_linalg::CsrMatrix<f64> {
    scaled_identity_minus_dt_jac(jac, 1.0, alpha)
}

/// Build `s I − α J` as a CsrMatrix.
pub(super) fn scaled_identity_minus_dt_jac(jac: &fem_linalg::CsrMatrix<f64>, s: f64, alpha: f64) -> fem_linalg::CsrMatrix<f64> {
    use fem_linalg::CooMatrix;
    let n = jac.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);

    // Add diagonal s I
    for i in 0..n { coo.add(i, i, s); }

    // Subtract α J
    for i in 0..n {
        for ptr in jac.row_ptr[i]..jac.row_ptr[i + 1] {
            let j = jac.col_idx[ptr] as usize;
            coo.add(i, j, -alpha * jac.values[ptr]);
        }
    }
    coo.into_csr()
}

// ─── Implicit Euler (BDF-1) ───────────────────────────────────────────────────

/// Backward (implicit) Euler: `(I - dt J) Δu = dt f(tₙ₊₁, uₙ)`
///
/// First-order, A-stable.  Each step solves a linear system.
/// For nonlinear problems this performs one fixed-point / Picard iteration.
pub struct ImplicitEuler;

impl ImplicitTimeStepper for ImplicitEuler {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let mut dudt = vec![0.0_f64; n];
        rhs(t + dt, u, &mut dudt);

        // Build system (I − dt J) Δu = dt f
        let jac = jac_fn(t + dt, u);
        let sys = identity_minus_dt_jac(&jac, dt);
        let b: Vec<f64> = dudt.iter().map(|&v| dt * v).collect();

        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let mut du = vec![0.0_f64; n];
        solve_gmres(&sys, &b, &mut du, 30, &cfg).expect("ImplicitEuler: linear solve failed");

        for i in 0..n { u[i] += du[i]; }
    }
}

// ─── SDIRK-2 ─────────────────────────────────────────────────────────────────

/// Singly Diagonally Implicit Runge–Kutta, 2nd order (Alexander, 1977).
///
/// Butcher tableau:
/// ```text
/// γ  |  γ   0
/// 1  | 1-γ  γ
/// ---|----------
///    | 1-γ  γ
/// ```
/// with γ = 1 − 1/√2 ≈ 0.2929.  Strongly S-stable (A-stable with high damping).
pub struct Sdirk2;

const SDIRK2_GAMMA: f64 = 1.0 - std::f64::consts::FRAC_1_SQRT_2; // 1 - 1/√2

impl ImplicitTimeStepper for Sdirk2 {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let g = SDIRK2_GAMMA;

        // Stage 1: Solve (I − dt γ J(t, U₁)) k₁ = f(t+γdt, U₁)
        let jac1 = jac_fn(t + g * dt, u);
        let sys1 = identity_minus_dt_jac(&jac1, dt * g);
        let mut f1 = vec![0.0_f64; n];
        rhs(t + g * dt, u, &mut f1);
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let mut k1 = vec![0.0_f64; n];
        solve_gmres(&sys1, &f1, &mut k1, 30, &cfg).expect("SDIRK2 stage 1 solve failed");

        // Stage 2: U₂ = u + dt[(1-γ) k₁ + γ k₂]
        let mut u2 = u.to_vec();
        for i in 0..n { u2[i] += dt * (1.0 - g) * k1[i]; }
        let jac2 = jac_fn(t + dt, &u2);
        let sys2 = identity_minus_dt_jac(&jac2, dt * g);
        let mut f2 = vec![0.0_f64; n];
        rhs(t + dt, &u2, &mut f2);
        let mut k2 = vec![0.0_f64; n];
        solve_gmres(&sys2, &f2, &mut k2, 30, &cfg).expect("SDIRK2 stage 2 solve failed");

        // Update: u_{n+1} = u_n + dt [(1-γ) k₁ + γ k₂]
        for i in 0..n {
            u[i] += dt * ((1.0 - g) * k1[i] + g * k2[i]);
        }
    }
}

// ─── BDF-2 ────────────────────────────────────────────────────────────────────

/// BDF-2 (2-step backward differentiation formula).
///
/// Formula: `(3/2) u_{n+1} − 2 u_n + (1/2) u_{n-1} = dt f(t_{n+1}, u_{n+1})`
///
/// Start-up step uses BDF-1 (implicit Euler) for the first step.
pub struct Bdf2;

/// State for BDF-2: holds the previous solution for the two-step formula.
pub struct Bdf2State {
    /// u at t_{n-1} (None before the first step is taken).
    pub u_prev: Option<Vec<f64>>,
}

impl Bdf2State {
    pub fn new() -> Self { Bdf2State { u_prev: None } }
}

impl Default for Bdf2State {
    fn default() -> Self { Self::new() }
}

impl Bdf2 {
    /// Advance `u` using BDF-2, updating `state` for the next call.
    pub fn step_implicit<F, J>(
        &self,
        t:       f64,
        dt:      f64,
        u:       &mut [f64],
        state:   &mut Bdf2State,
        rhs:     F,
        jac_fn:  J,
    )
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };

        match &state.u_prev {
            None => {
                // First step: use implicit Euler (BDF-1)
                let mut dudt = vec![0.0_f64; n];
                rhs(t + dt, u, &mut dudt);
                let jac = jac_fn(t + dt, u);
                let sys = identity_minus_dt_jac(&jac, dt);
                let b: Vec<f64> = dudt.iter().map(|&v| dt * v).collect();
                let mut du = vec![0.0_f64; n];
                solve_gmres(&sys, &b, &mut du, 30, &cfg).expect("BDF2 startup solve failed");
                let u_old = u.to_vec();
                for i in 0..n { u[i] += du[i]; }
                state.u_prev = Some(u_old);
            }
            Some(u_prev) => {
                // BDF-2: (3/2 I − dt J) u_{n+1} = 2 uₙ − ½ u_{n-1}
                let u_prev = u_prev.clone();
                let jac = jac_fn(t + dt, u);

                // Build (3/2 I − dt J)
                let sys = scaled_identity_minus_dt_jac(&jac, 1.5, dt);
                // RHS: 2 uₙ − ½ u_{n-1}
                let b: Vec<f64> = (0..n)
                    .map(|i| 2.0 * u[i] - 0.5 * u_prev[i])
                    .collect();

                let u_old = u.to_vec();
                let mut u_new = vec![0.0_f64; n];
                solve_gmres(&sys, &b, &mut u_new, 30, &cfg).expect("BDF2 solve failed");
                u.copy_from_slice(&u_new);
                state.u_prev = Some(u_old);
            }
        }
    }
}

// ─── Crank-Nicolson ──────────────────────────────────────────────────────────

/// Crank-Nicolson (trapezoidal) implicit time integrator.
///
/// `u_{n+1} = u_n + (Δt/2)(f(t_n, u_n) + f(t_{n+1}, u_{n+1}))`
///
/// Second-order, A-stable, symmetric.  The linearised system solved at each step
/// is `(I − Δt/2·J) Δu = Δt·f(t_n, u_n)` with J = ∂f/∂u.
pub struct CrankNicolson;

impl ImplicitTimeStepper for CrankNicolson {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };

        // f(t_n, u_n)
        let mut fn0 = vec![0.0_f64; n];
        rhs(t, u, &mut fn0);

        // (I − Δt/2·J) Δu = Δt·f_n   where J = ∂f/∂u(t_n, u_n)
        let jac = jac_fn(t, u);
        let sys = scaled_identity_minus_dt_jac(&jac, 1.0, 0.5 * dt);

        let b: Vec<f64> = fn0.iter().map(|&v| dt * v).collect();
        let mut du = vec![0.0_f64; n];
        solve_gmres(&sys, &b, &mut du, 30, &cfg).expect("CrankNicolson: linear solve failed");
        for i in 0..n { u[i] += du[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn exp_decay(lambda: f64) -> impl Fn(f64, &[f64], &mut [f64]) {
        move |_t, u, dudt| { dudt[0] = -lambda * u[0]; }
    }

    fn exp_decay_jac(n: usize, lambda: f64) -> impl Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64> {
        move |_t, _u| {
            let mut coo = CooMatrix::<f64>::new(n, n);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        }
    }

    #[test]
    fn implicit_euler_stiff_stable() {
        let lambda = 1000.0_f64;
        let rhs    = exp_decay(lambda);
        let jac    = exp_decay_jac(1, lambda);
        let ie     = ImplicitEuler;
        let t_end  = 1.0_f64;
        let dt     = 0.1_f64;
        let mut u  = vec![1.0_f64];
        let mut t  = 0.0_f64;
        while t < t_end - 1e-14 {
            let dt_act = dt.min(t_end - t);
            ie.step_implicit(t, dt_act, &mut u, &rhs, &jac);
            t += dt_act;
        }
        assert!(u[0] < 0.01, "ImplicitEuler: solution did not decay; u={:.3e}", u[0]);
        assert!(u[0] >= 0.0, "ImplicitEuler: negative solution (instability)");
    }

    #[test]
    fn sdirk2_stiff_stable() {
        let lambda = 1000.0_f64;
        let rhs    = exp_decay(lambda);
        let jac    = exp_decay_jac(1, lambda);
        let solver = Sdirk2;
        let t_end  = 1.0_f64;
        let dt     = 0.1_f64;
        let mut u  = vec![1.0_f64];
        let mut t  = 0.0_f64;
        while t < t_end - 1e-14 {
            let dt_act = dt.min(t_end - t);
            solver.step_implicit(t, dt_act, &mut u, &rhs, &jac);
            t += dt_act;
        }
        assert!(u[0] < 0.01, "SDIRK2: solution did not decay; u={:.3e}", u[0]);
        assert!(u[0] >= 0.0, "SDIRK2: negative solution (instability)");
    }

    #[test]
    fn bdf2_stiff_stable() {
        let lambda = 1000.0_f64;
        let rhs    = exp_decay(lambda);
        let jac    = exp_decay_jac(1, lambda);
        let solver = Bdf2;
        let mut state = Bdf2State::new();
        let t_end  = 1.0_f64;
        let dt     = 0.1_f64;
        let mut u  = vec![1.0_f64];
        let mut t  = 0.0_f64;
        while t < t_end - 1e-14 {
            let dt_act = dt.min(t_end - t);
            solver.step_implicit(t, dt_act, &mut u, &mut state, &rhs, &jac);
            t += dt_act;
        }
        assert!(u[0].abs() < 0.01, "BDF2: solution did not decay; u={:.3e}", u[0]);
    }

    #[test]
    fn crank_nicolson_heat_order2() {
        let lambda = std::f64::consts::PI * std::f64::consts::PI;
        let rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| { dudt[0] = -lambda * u[0]; };
        let jac = |_t: f64, _u: &[f64]| {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        };
        let cn = CrankNicolson;
        let t_end = 0.1;
        let exact = (-lambda * t_end).exp();
        let mut errors = vec![];
        for &dt in &[0.02_f64, 0.01, 0.005] {
            let mut u = vec![1.0_f64];
            let mut t = 0.0;
            while t < t_end - 1e-14 {
                let h = dt.min(t_end - t);
                cn.step_implicit(t, h, &mut u, &rhs, &jac);
                t += h;
            }
            errors.push((u[0] - exact).abs());
        }
        let order = (errors[0] / errors[1]).log2();
        assert!(order > 1.8, "CrankNicolson heat convergence order={order:.2} (expected ~2)");
        assert!(errors[2] < 5e-4, "CrankNicolson error at finest dt={e:.2e}", e=errors[2]);
    }
}
