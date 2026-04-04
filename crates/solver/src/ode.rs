//! ODE / Time integrators.
//!
//! Provides a unified `TimeStepper` trait plus several concrete integrators:
//!
//! | Method       | Type     | Order | Suitable for          |
//! |-------------|----------|-------|-----------------------|
//! | Forward Euler | Explicit | 1     | non-stiff             |
//! | RK4          | Explicit | 4     | non-stiff             |
//! | RK45 (adaptive) | Explicit | 4/5 | non-stiff, adaptive  |
//! | Implicit Euler (BDF-1) | Implicit | 1 | stiff          |
//! | SDIRK-2      | Implicit | 2     | stiff                 |
//! | BDF-2        | Implicit | 2     | stiff, multi-step     |
//!
//! # Usage
//! ```rust,ignore
//! // du/dt = -u  →  u(t) = exp(-t)
//! let rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
//!     dudt[0] = -u[0];
//! };
//! let solver = ForwardEuler::new(0.01);
//! let mut u = vec![1.0_f64];
//! solver.step(0.0, &mut u, rhs);
//! ```

use crate::{SolverConfig, solve_gmres};
use fem_linalg::{CooMatrix, CsrMatrix};

// ─── Trait ───────────────────────────────────────────────────────────────────

/// A single time-step integrator.
///
/// The RHS function has signature `rhs(t, u, dudt)` and computes
/// the time derivative `dudt = f(t, u)`.
pub trait TimeStepper: Send + Sync {
    /// Advance `u` from time `t` by step `dt`, using `rhs(t, u, dudt)`.
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]);
}

/// An implicit time-step integrator that needs to solve a nonlinear/linear system.
///
/// The `jac_fn` assembles the (approximate) Jacobian `∂f/∂u` at `(t, u)`.
/// For linear problems this is exact; for nonlinear problems it gives the Picard Jacobian.
pub trait ImplicitTimeStepper: Send + Sync {
    /// Advance `u` from time `t` by step `dt`.
    ///
    /// `rhs(t, u, dudt)` computes `f(t, u)`.
    /// `jac_fn(t, u)` returns a CSR matrix approximating `∂f/∂u`.
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>;
}

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
                dt *= (0.9 / err).powf(0.2).min(5.0).max(0.1);
            } else {
                dt *= 5.0;
            }
            dt = dt.min(self.dt_max).max(self.dt_min);
        }
        (t, dt)
    }
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
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>,
    {
        let n = u.len();
        let mut dudt = vec![0.0_f64; n];
        rhs(t + dt, u, &mut dudt);

        // Build system (I − dt J) Δu = dt f
        let jac = jac_fn(t + dt, u);
        let sys = identity_minus_dt_jac(&jac, dt);
        let b: Vec<f64> = dudt.iter().map(|&v| dt * v).collect();

        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false };
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
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>,
    {
        let n = u.len();
        let g = SDIRK2_GAMMA;

        // Stage 1: Solve (I − dt γ J(t, U₁)) k₁ = f(t+γdt, U₁)
        //   where U₁ = u + dt γ k₁  → (I − dt γ J) k₁ = f(t+γdt, u + dt γ k₁)
        //   Picard: U₁ ≈ u,  then k₁ = (I − dt γ J)⁻¹ f(t+γdt, u)
        let jac1 = jac_fn(t + g * dt, u);
        let sys1 = identity_minus_dt_jac(&jac1, dt * g);
        let mut f1 = vec![0.0_f64; n];
        rhs(t + g * dt, u, &mut f1);
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false };
        let mut k1 = vec![0.0_f64; n];
        solve_gmres(&sys1, &f1, &mut k1, 30, &cfg).expect("SDIRK2 stage 1 solve failed");

        // Stage 2: U₂ = u + dt[(1-γ) k₁ + γ k₂]
        //   Picard: U₂ ≈ u + dt(1-γ) k₁
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
/// Each step solves `(3/2 I − dt J) Δ = 2 uₙ − ½ u_{n-1} + (dt f − 3/2 uₙ)`.
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
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>,
    {
        let n = u.len();
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false };

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
                // Linearise f via Jacobian: f(u_{n+1}) ≈ J u_{n+1}
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

// ─── Newmark-β (second-order ODEs) ──────────────────────────────────────────

/// Newmark-β method for second-order ODEs: M ü + K u = f(t).
///
/// Converts the second-order system to first-order state [u, v=u̇]:
///   u_{n+1} = uₙ + dt vₙ + dt²[(½−β) aₙ + β a_{n+1}]
///   v_{n+1} = vₙ + dt[(1−γ) aₙ + γ a_{n+1}]
///
/// where aₙ = M⁻¹(fₙ − K uₙ).
///
/// Classic parameter choices:
/// - β=1/4, γ=1/2: average acceleration (unconditionally stable, 2nd order)
/// - β=0, γ=1/2: central difference (conditionally stable, 2nd order)
/// - β=1/6, γ=1/2: linear acceleration (conditionally stable, 2nd order)
pub struct Newmark {
    pub beta: f64,
    pub gamma: f64,
}

impl Default for Newmark {
    fn default() -> Self {
        // Average acceleration (trapezoidal rule) — unconditionally stable
        Newmark { beta: 0.25, gamma: 0.5 }
    }
}

/// State for the Newmark method: stores velocity and acceleration.
pub struct NewmarkState {
    pub vel: Vec<f64>,   // velocity v = du/dt
    pub acc: Vec<f64>,   // acceleration a = d²u/dt²
}

impl NewmarkState {
    pub fn new(n: usize) -> Self {
        NewmarkState { vel: vec![0.0; n], acc: vec![0.0; n] }
    }

    /// Initialize with given velocity and compute initial acceleration from M a₀ = f₀ - K u₀.
    pub fn init_from(vel: Vec<f64>, mass: &CsrMatrix<f64>, stiff: &CsrMatrix<f64>, u: &[f64], force: &[f64]) -> Self {
        let n = u.len();
        // a₀ = M⁻¹(f₀ - K u₀)
        // Solve M a₀ = f₀ - K u₀  using CG
        let mut ku = vec![0.0; n];
        stiff.spmv(u, &mut ku);
        let rhs: Vec<f64> = (0..n).map(|i| force[i] - ku[i]).collect();
        let mut acc = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500, verbose: false };
        crate::solve_cg(mass, &rhs, &mut acc, &cfg).expect("Newmark init: mass solve failed");
        NewmarkState { vel, acc }
    }
}

impl Newmark {
    /// Advance one time step for M ü + K u = f(t_{n+1}).
    ///
    /// Arguments:
    /// - `mass`: mass matrix M
    /// - `stiff`: stiffness matrix K
    /// - `force_new`: force vector f(t_{n+1})
    /// - `dt`: time step
    /// - `u`: displacement (updated in-place)
    /// - `state`: Newmark state (velocity + acceleration, updated in-place)
    /// - `bc_dofs`: Dirichlet boundary DOFs (displacement = 0)
    pub fn step(
        &self,
        mass: &CsrMatrix<f64>,
        stiff: &CsrMatrix<f64>,
        force_new: &[f64],
        dt: f64,
        u: &mut [f64],
        state: &mut NewmarkState,
        bc_dofs: &[u32],
    ) {
        let n = u.len();
        let b = self.beta;
        let g = self.gamma;

        // Predict: u_pred = u + dt*v + dt²*(0.5-β)*a
        let mut u_pred = vec![0.0; n];
        for i in 0..n {
            u_pred[i] = u[i] + dt * state.vel[i] + dt * dt * (0.5 - b) * state.acc[i];
        }

        // Solve effective system: (M + β dt² K) a_{n+1} = f_{n+1} - K u_pred
        // Build effective stiffness: S = M + β dt² K
        let coeff = b * dt * dt;
        let eff_stiff = build_effective_stiffness(mass, stiff, coeff);

        // Build effective RHS: f_{n+1} - K * u_pred
        let mut k_upred = vec![0.0; n];
        stiff.spmv(&u_pred, &mut k_upred);
        let mut rhs: Vec<f64> = (0..n).map(|i| force_new[i] - k_upred[i]).collect();

        // Apply Dirichlet BCs to the effective system
        let mut eff = eff_stiff;
        for &d in bc_dofs {
            let d = d as usize;
            eff.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs);
        }

        // Solve for a_{n+1}
        let mut a_new = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 1000, verbose: false };
        crate::solve_cg(&eff, &rhs, &mut a_new, &cfg).expect("Newmark: effective system solve failed");

        // Correct: u_{n+1} = u_pred + β dt² a_{n+1}
        for i in 0..n {
            u[i] = u_pred[i] + coeff * a_new[i];
        }

        // Update velocity: v_{n+1} = v_n + dt[(1-γ) a_n + γ a_{n+1}]
        for i in 0..n {
            state.vel[i] += dt * ((1.0 - g) * state.acc[i] + g * a_new[i]);
        }

        // Update acceleration
        state.acc.copy_from_slice(&a_new);

        // Zero BC DOFs
        for &d in bc_dofs {
            let d = d as usize;
            u[d] = 0.0;
            state.vel[d] = 0.0;
            state.acc[d] = 0.0;
        }
    }
}

/// Build S = M + α K  as a CsrMatrix.
fn build_effective_stiffness(mass: &CsrMatrix<f64>, stiff: &CsrMatrix<f64>, alpha: f64) -> CsrMatrix<f64> {
    let n = mass.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    // Add M
    for i in 0..n {
        for ptr in mass.row_ptr[i]..mass.row_ptr[i+1] {
            coo.add(i, mass.col_idx[ptr] as usize, mass.values[ptr]);
        }
    }
    // Add alpha*K
    for i in 0..n {
        for ptr in stiff.row_ptr[i]..stiff.row_ptr[i+1] {
            coo.add(i, stiff.col_idx[ptr] as usize, alpha * stiff.values[ptr]);
        }
    }
    coo.into_csr()
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build `I − α J` as a CsrMatrix.
fn identity_minus_dt_jac(jac: &CsrMatrix<f64>, alpha: f64) -> CsrMatrix<f64> {
    scaled_identity_minus_dt_jac(jac, 1.0, alpha)
}

/// Build `s I − α J` as a CsrMatrix.
fn scaled_identity_minus_dt_jac(jac: &CsrMatrix<f64>, s: f64, alpha: f64) -> CsrMatrix<f64> {
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// du/dt = -λ u  →  u(t) = exp(-λt).
    fn exp_decay(lambda: f64) -> impl Fn(f64, &[f64], &mut [f64]) {
        move |_t, u, dudt| { dudt[0] = -lambda * u[0]; }
    }

    /// Jacobian of du/dt = -λ u: J = -λ I.
    fn exp_decay_jac(n: usize, lambda: f64) -> impl Fn(f64, &[f64]) -> CsrMatrix<f64> {
        move |_t, _u| {
            let mut coo = CooMatrix::<f64>::new(n, n);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        }
    }

    #[test]
    fn forward_euler_order1() {
        // u' = -u, u(0) = 1.  Final time T = 1.  Exact: exp(-1) ≈ 0.36788.
        // Forward Euler is first-order: halving dt halves the error.
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
        // Error ratio should be ~2 for first-order method (coarse/fine)
        let ratio = err_c / err_f;
        assert!(ratio > 1.5, "FE order check: ratio={ratio:.2} (expected ~2)");
    }

    #[test]
    fn rk4_order4() {
        // u' = -u.  RK4 is 4th order: halving dt reduces error by 2^4=16.
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
        // u' = -u.  Exact: exp(-1).
        let solver = Rk45 { atol: 1e-8, rtol: 1e-8, ..Default::default() };
        let mut u = vec![1.0_f64];
        solver.integrate(0.0, 1.0, &mut u, 0.1, exp_decay(1.0));
        let exact = (-1.0_f64).exp();
        let err = (u[0] - exact).abs();
        assert!(err < 1e-6, "RK45 error={err:.3e}");
    }

    #[test]
    fn implicit_euler_stiff_stable() {
        // du/dt = -1000 u.  Explicit methods need dt < 2e-3; implicit is unconditionally stable.
        let lambda = 1000.0_f64;
        let rhs    = exp_decay(lambda);
        let jac    = exp_decay_jac(1, lambda);
        let ie     = ImplicitEuler;
        let t_end  = 1.0_f64;
        let dt     = 0.1_f64; // huge step, explicit would blow up
        let mut u  = vec![1.0_f64];
        let mut t  = 0.0_f64;
        while t < t_end - 1e-14 {
            let dt_act = dt.min(t_end - t);
            ie.step_implicit(t, dt_act, &mut u, &rhs, &jac);
            t += dt_act;
        }
        // Should have decayed to near 0 (exact is exp(-1000) ≈ 0)
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
    fn newmark_free_vibration() {
        // 1-DOF: m*a + k*u = 0, m=1, k=ω², ω=π, u(0)=1, v(0)=0
        // Exact: u(t) = cos(ωt)
        let omega = std::f64::consts::PI;
        let k = omega * omega;
        let mut mass_coo = CooMatrix::<f64>::new(1, 1);
        mass_coo.add(0, 0, 1.0);
        let mass = mass_coo.into_csr();
        let mut stiff_coo = CooMatrix::<f64>::new(1, 1);
        stiff_coo.add(0, 0, k);
        let stiff = stiff_coo.into_csr();

        let newmark = Newmark::default(); // β=1/4, γ=1/2
        let mut u = vec![1.0];
        let force = vec![0.0];
        let dt = 0.001;
        let mut state = NewmarkState::new(1);
        // Initial acc: a₀ = M⁻¹(-K u₀) = -ω²
        state.acc[0] = -k;

        let t_end = 1.0_f64;
        let n_steps = (t_end / dt).round() as usize;
        for _ in 0..n_steps {
            newmark.step(&mass, &stiff, &force, dt, &mut u, &mut state, &[]);
        }
        let exact = (omega * t_end).cos();
        let err = (u[0] - exact).abs();
        assert!(err < 0.01, "Newmark free vibration error={err:.4e} (exact={exact:.4})");
    }

    /// Heat equation: du/dt = -λ u (modal decomposition of Laplacian).
    /// Test temporal convergence order of RK4 vs exact.
    #[test]
    fn rk4_heat_convergence() {
        // Simplest: u' = -π² u (first Fourier mode of heat eq on [0,1]).
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
        // Order = log2(err[0]/err[1])
        let order = (errors[0] / errors[1]).log2();
        assert!(order > 3.5, "RK4 heat convergence order={order:.2} (expected ~4)");
    }
}
