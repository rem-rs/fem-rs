//! Structural dynamics time integrators for second-order systems.
//!
//! Methods: [`Newmark`] (Newmark-β), [`GeneralizedAlpha`] (first-order).

use crate::{solve_cg, solve_gmres, SolverConfig};

/// Build `M + α K` as a CsrMatrix.
pub(super) fn build_effective_stiffness(
    mass: &fem_linalg::CsrMatrix<f64>,
    stiff: &fem_linalg::CsrMatrix<f64>,
    alpha: f64,
) -> fem_linalg::CsrMatrix<f64> {
    use fem_linalg::CooMatrix;
    let n = mass.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    // Add M
    for i in 0..n {
        for ptr in mass.row_ptr[i]..mass.row_ptr[i + 1] {
            coo.add(i, mass.col_idx[ptr] as usize, mass.values[ptr]);
        }
    }
    // Add alpha*K
    for i in 0..n {
        for ptr in stiff.row_ptr[i]..stiff.row_ptr[i + 1] {
            coo.add(i, stiff.col_idx[ptr] as usize, alpha * stiff.values[ptr]);
        }
    }
    coo.into_csr()
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
        Newmark {
            beta: 0.25,
            gamma: 0.5,
        }
    }
}

/// State for the Newmark method: stores velocity and acceleration.
pub struct NewmarkState {
    pub vel: Vec<f64>, // velocity v = du/dt
    pub acc: Vec<f64>, // acceleration a = d²u/dt²
}

impl NewmarkState {
    pub fn new(n: usize) -> Self {
        NewmarkState {
            vel: vec![0.0; n],
            acc: vec![0.0; n],
        }
    }

    /// Initialize with given velocity and compute initial acceleration from M a₀ = f₀ - K u₀.
    pub fn init_from(
        vel: Vec<f64>,
        mass: &fem_linalg::CsrMatrix<f64>,
        stiff: &fem_linalg::CsrMatrix<f64>,
        u: &[f64],
        force: &[f64],
    ) -> Self {
        let n = u.len();
        // a₀ = M⁻¹(f₀ - K u₀)
        let mut ku = vec![0.0; n];
        stiff.spmv(u, &mut ku);
        let rhs: Vec<f64> = (0..n).map(|i| force[i] - ku[i]).collect();
        let mut acc = vec![0.0; n];
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 500,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_cg(mass, &rhs, &mut acc, &cfg).expect("Newmark init: mass solve failed");
        NewmarkState { vel, acc }
    }
}

impl Newmark {
    /// Advance one time step for M ü + K u = f(t_{n+1}).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        mass: &fem_linalg::CsrMatrix<f64>,
        stiff: &fem_linalg::CsrMatrix<f64>,
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
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 1000,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_cg(&eff, &rhs, &mut a_new, &cfg).expect("Newmark: effective system solve failed");

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

// ─── Generalized-α for second-order ODEs ─────────────────────────────────────

/// Second-order generalized-α method for `M·ü + C·u̇ + K·u = f(t)`.
///
/// Parameters derived from spectral radius at infinity `ρ_∞ ∈ [0, 1]`:
/// ```text
///   α_m = (2 - ρ_∞) / (1 + ρ_∞)
///   α_f = 1 / (1 + ρ_∞)
///   γ   = 0.5 + α_m - α_f
///   β   = 0.25 · (1 + α_m - α_f)²
/// ```
///
/// Common choices:
/// - `ρ_∞ = 1.0` → β=0.25, γ=0.5  (average acceleration / Newmark, **type 10**)
/// - `ρ_∞ = 0.0` → β=1.0,  γ=1.5  (fully implicit, **type 11**)
/// - `ρ_∞ = 0.5` → β=4/9,  γ=5/6  (L-stable, **type 12**)
pub struct GeneralizedAlpha2 {
    pub rho_inf: f64,
}

/// Second-order ODE solver selector matching MFEM's `SecondOrderODESolver::Select`.
///
/// | type | MFEM name       | `rho_inf` | Rust equivalent     |
/// |------|-----------------|-----------|---------------------|
/// | 10   | Backward Euler  | 1.0       | [`GeneralizedAlpha2`] |
/// | 11   | Trapezoidal / Newmark | 0.0  | [`GeneralizedAlpha2`] |
/// | 12   | SDIRK2 (L-stable) | 0.5     | [`GeneralizedAlpha2`] |
pub enum SecondOrderSolver {
    /// Backward Euler-like (type 10, ρ_∞=1.0, β=0.25, γ=0.5).
    BackwardEuler,
    /// Trapezoidal / Newmark (type 11, ρ_∞=0.0, β=1.0, γ=1.5).
    Trapezoidal,
    /// SDIRK2 L-stable (type 12, ρ_∞=0.5, β=4/9, γ=5/6).
    Sdirk2,
}

impl SecondOrderSolver {
    /// Create from MFEM type code (10, 11, or 12).
    /// Returns `BackwardEuler` for unknown codes.
    pub fn from_type(code: i32) -> Self {
        match code {
            10 => SecondOrderSolver::BackwardEuler,
            11 => SecondOrderSolver::Trapezoidal,
            12 => SecondOrderSolver::Sdirk2,
            _ => {
                eprintln!("SecondOrderSolver: unknown type {code}, using BackwardEuler");
                SecondOrderSolver::BackwardEuler
            }
        }
    }

    /// Return the `rho_inf` value for this solver type.
    pub fn rho_inf(&self) -> f64 {
        match self {
            SecondOrderSolver::BackwardEuler => 1.0,
            SecondOrderSolver::Trapezoidal => 0.0,
            SecondOrderSolver::Sdirk2 => 0.5,
        }
    }
}

/// State for [`GeneralizedAlpha2`]: stores velocity and acceleration.
pub struct GeneralizedAlpha2State {
    pub vel: Vec<f64>,
    pub acc: Vec<f64>,
}

impl GeneralizedAlpha2State {
    pub fn new(n: usize) -> Self {
        GeneralizedAlpha2State {
            vel: vec![0.0; n],
            acc: vec![0.0; n],
        }
    }

    pub fn init_from(
        vel: Vec<f64>,
        mass: &fem_linalg::CsrMatrix<f64>,
        stiff: &fem_linalg::CsrMatrix<f64>,
        u: &[f64],
        force: &[f64],
    ) -> Self {
        let n = u.len();
        let mut ku = vec![0.0; n];
        stiff.spmv(u, &mut ku);
        let rhs: Vec<f64> = (0..n).map(|i| force[i] - ku[i]).collect();
        let mut acc = vec![0.0; n];
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 500,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_cg(mass, &rhs, &mut acc, &cfg).expect("GeneralizedAlpha2 init: mass solve failed");
        GeneralizedAlpha2State { vel, acc }
    }
}

impl GeneralizedAlpha2 {
    pub fn new(rho_inf: f64) -> Self {
        assert!(rho_inf >= 0.0 && rho_inf <= 1.0, "ρ_∞ must be in [0, 1]");
        GeneralizedAlpha2 { rho_inf }
    }

    fn alpha_m(&self) -> f64 {
        (2.0 - self.rho_inf) / (1.0 + self.rho_inf)
    }
    fn alpha_f(&self) -> f64 {
        1.0 / (1.0 + self.rho_inf)
    }
    fn gamma(&self) -> f64 {
        0.5 + self.alpha_m() - self.alpha_f()
    }
    fn beta(&self) -> f64 {
        let g = 0.5 + self.alpha_m() - self.alpha_f();
        0.25 * (g + 0.5) * (g + 0.5) // MFEM: β = (γ+½)²/4
    }

    /// Advance one time step for M·ü + K·u = f(t).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        mass: &fem_linalg::CsrMatrix<f64>,
        stiff: &fem_linalg::CsrMatrix<f64>,
        force_new: &[f64],
        dt: f64,
        u: &mut [f64],
        state: &mut GeneralizedAlpha2State,
        bc_dofs: &[u32],
    ) {
        let n = u.len();
        let b = self.beta();
        let g = self.gamma();
        let pred_coef = self.alpha_f() / self.alpha_m();

        // Predict: u_pred = u + (α_f/α_m)·dt·v
        let mut u_pred = vec![0.0; n];
        for i in 0..n {
            u_pred[i] = u[i] + pred_coef * dt * state.vel[i];
        }

        // Solve effective system: (M + β·dt²·K)·a_{n+1} = f_{n+1} - K·u_pred
        let coeff = b * dt * dt;
        let eff_stiff = build_effective_stiffness(mass, stiff, coeff);

        let mut k_upred = vec![0.0; n];
        stiff.spmv(&u_pred, &mut k_upred);
        let mut rhs: Vec<f64> = (0..n).map(|i| force_new[i] - k_upred[i]).collect();

        let mut eff = eff_stiff;
        for &d in bc_dofs {
            let d = d as usize;
            eff.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs);
        }

        let mut a_new = vec![0.0; n];
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 1000,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_cg(&eff, &rhs, &mut a_new, &cfg)
            .expect("GeneralizedAlpha2: effective system solve failed");

        // Correct: u_{n+1} = u + dt·v + dt²·[(0.5-β)·a_n + β·a_{n+1}]
        for i in 0..n {
            u[i] += dt * state.vel[i] + dt * dt * ((0.5 - b) * state.acc[i] + b * a_new[i]);
        }

        // Update velocity: v_{n+1} = v_n + dt·[(1-γ)·a_n + γ·a_{n+1}]
        for i in 0..n {
            state.vel[i] += dt * ((1.0 - g) * state.acc[i] + g * a_new[i]);
        }

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

// ─── Generalized-α (first-order) ─────────────────────────────────────────────

/// First-order generalized-α method (Jansen, Whiting & Hulbert, 2000) for the
/// first-order system:
///
/// ```text
///   M dv/dt + K v = f(t)
/// ```
///
/// Parameter choice from spectral radius at infinity ρ_∞ ∈ [0, 1]:
/// ```text
///   α_f = 1/(1 + ρ_∞)
///   α_m = (3 - ρ_∞) / (2(1 + ρ_∞))
///   γ   = 0.5 + α_m - α_f
/// ```
///
/// Unconditionally stable, 2nd-order accurate for ρ_∞ ∈ (0, 1].
pub struct GeneralizedAlpha {
    /// Spectral radius at ω→∞.  0 = maximum algorithmic dissipation, 1 = no dissipation.
    pub rho_inf: f64,
}

impl Default for GeneralizedAlpha {
    fn default() -> Self {
        GeneralizedAlpha { rho_inf: 0.5 }
    }
}

/// State for Generalized-α: stores the previous time-derivative (v̇_{n}).
pub struct GeneralizedAlphaState {
    /// dv/dt at tₙ  (acceleration / rate-of-change).
    pub dvdt: Vec<f64>,
}

impl GeneralizedAlphaState {
    pub fn new(n: usize) -> Self {
        GeneralizedAlphaState { dvdt: vec![0.0; n] }
    }
}

impl GeneralizedAlpha {
    /// Derived parameters from ρ_∞.
    fn params(&self) -> (f64, f64, f64) {
        let r = self.rho_inf;
        let alpha_f = 1.0 / (1.0 + r);
        let alpha_m = (3.0 - r) / (2.0 * (1.0 + r));
        let gamma = 0.5 + alpha_m - alpha_f;
        (alpha_f, alpha_m, gamma)
    }

    /// Advance `v` from `t` by `dt` for the system `M dv/dt + K v = f(t)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        mass: &fem_linalg::CsrMatrix<f64>,
        stiff: &fem_linalg::CsrMatrix<f64>,
        force_fn: &dyn Fn(f64) -> Vec<f64>,
        dt: f64,
        t: f64,
        v: &mut [f64],
        state: &mut GeneralizedAlphaState,
        bc_dofs: &[u32],
    ) {
        let n = v.len();
        let (alpha_f, alpha_m, gamma) = self.params();

        // Predicted intermediate time levels
        let _t_m = t + alpha_m * dt;
        let t_f = t + alpha_f * dt;

        // Predicted v at t_f: v_f = v_n + alpha_f * gamma * dt * dvdt_n
        let v_f: Vec<f64> = (0..n)
            .map(|i| v[i] + alpha_f * gamma * dt * state.dvdt[i])
            .collect();

        // Force at t_f
        let f_f = force_fn(t_f);

        // v_f_pred = v_n  (predictor: v_f before correction)
        let v_f_pred: Vec<f64> = v.to_vec();

        // Compute K * v_f_pred
        let mut kv = vec![0.0f64; n];
        stiff.spmv(&v_f_pred, &mut kv);

        // Compute M * dvdt_n
        let mut m_dvdt = vec![0.0f64; n];
        mass.spmv(&state.dvdt, &mut m_dvdt);

        let mut kv_n = vec![0.0f64; n];
        stiff.spmv(v, &mut kv_n);
        let mut k_dvdt = vec![0.0f64; n];
        stiff.spmv(&state.dvdt, &mut k_dvdt);

        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            rhs[i] = f_f[i]
                - (1.0 - alpha_m) * m_dvdt[i]
                - kv_n[i]
                - alpha_f * (1.0 - gamma) * dt * k_dvdt[i];
        }

        // Build LHS: alpha_m * M + alpha_f * gamma * dt * K
        let mut lhs_scaled = {
            use fem_linalg::CooMatrix;
            let mut coo = CooMatrix::<f64>::new(n, n);
            for i in 0..mass.nrows {
                for ptr in mass.row_ptr[i]..mass.row_ptr[i + 1] {
                    coo.add(i, mass.col_idx[ptr] as usize, alpha_m * mass.values[ptr]);
                }
            }
            for i in 0..stiff.nrows {
                for ptr in stiff.row_ptr[i]..stiff.row_ptr[i + 1] {
                    coo.add(
                        i,
                        stiff.col_idx[ptr] as usize,
                        alpha_f * gamma * dt * stiff.values[ptr],
                    );
                }
            }
            coo.into_csr()
        };

        // Apply Dirichlet BCs
        for &d in bc_dofs {
            let d_usize = d as usize;
            lhs_scaled.apply_dirichlet_row_zeroing(d_usize, 0.0, &mut rhs);
            rhs[d_usize] = 0.0;
        }

        // Solve for dvdt_{n+1}
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 500,
            verbose: false,
            ..SolverConfig::default()
        };
        let mut dvdt_new = vec![0.0f64; n];
        solve_gmres(&lhs_scaled, &rhs, &mut dvdt_new, 30, &cfg)
            .expect("GeneralizedAlpha: linear solve failed");

        // Correct: v_{n+1} = v_n + dt * [gamma * dvdt_{n+1} + (1-gamma) * dvdt_n]
        for i in 0..n {
            v[i] += dt * (gamma * dvdt_new[i] + (1.0 - gamma) * state.dvdt[i]);
        }

        // Apply Dirichlet BCs to v
        for &d in bc_dofs {
            let d = d as usize;
            v[d] = 0.0;
            dvdt_new[d] = 0.0;
        }

        state.dvdt = dvdt_new;
        let _ = v_f; // suppress unused warning from intermediate
    }
}

// ─── Central Difference Explicit (for lumped-mass explicit dynamics) ────────

/// Explicit central difference method for second-order ODEs:
///
/// ```text
///   M·a + C·v + K·u = f_ext(t)
/// ```
///
/// With a **lumped (diagonal) mass matrix**, the acceleration solve is a
/// simple scaling — no linear system required.  This is the method used
/// by Abaqus/Explicit for crash / impact / large-deformation problems.
///
/// ## Algorithm (per step)
///
/// 1. Predict displacement:
///    ```text
///    u_{n+1} = u_n + Δt·v_n + (Δt²/2)·a_n
///    ```
///
/// 2. Compute new acceleration via lumped mass:
///    ```text
///    a_{n+1} = M⁻¹ · f_total(t_{n+1}, u_{n+1}, v_n)
///    ```
///    where `f_total = f_ext - f_int - f_damp + f_contact`.
///
/// 3. Correct velocity:
///    ```text
///    v_{n+1} = v_n + Δt·((1−γ)·a_n + γ·a_{n+1})
///    ```
///    with `γ = 0.5` → trapezoidal (2nd order).
///
/// ## CFL stability
///
/// The method is conditionally stable.  The critical time step is:
/// ```text
/// Δt_crit = 2 / ω_max
/// ```
/// where `ω_max` is the highest natural frequency.  In practice use a safety
/// factor `α ∈ [0.8, 0.98]` such that `Δt = α · Δt_crit`.
///
/// ## Abaqus equivalent
/// - Explicit dynamics with central difference integration.
/// - [`step`] is used inside an [`ExplicitDynamicsDriver`] that manages
///   contact, mass lumping, and element force computation.
pub struct CentralDifferenceExplicit {
    /// Newmark γ parameter (default 0.5 = trapezoidal, 2nd order).
    pub gamma: f64,
}

impl Default for CentralDifferenceExplicit {
    fn default() -> Self {
        CentralDifferenceExplicit { gamma: 0.5 }
    }
}

/// State for the explicit central difference method: velocity + acceleration.
pub struct ExplicitState {
    pub vel: Vec<f64>,
    pub acc: Vec<f64>,
}

impl ExplicitState {
    pub fn new(n: usize) -> Self {
        ExplicitState {
            vel: vec![0.0; n],
            acc: vec![0.0; n],
        }
    }

    /// Initialize acceleration from `M·a₀ = f_total₀`.
    /// `mass_lumped` is the diagonal of the lumped mass matrix.
    pub fn init_from(vel: Vec<f64>, mass_lumped: &[f64], force_total: &[f64]) -> Self {
        let n = vel.len();
        let mut acc = vec![0.0; n];
        for i in 0..n {
            acc[i] = if mass_lumped[i].abs() > 1e-14 {
                force_total[i] / mass_lumped[i]
            } else {
                0.0
            };
        }
        ExplicitState { vel, acc }
    }
}

impl CentralDifferenceExplicit {
    /// Advance one explicit time step.
    ///
    /// Algorithm:
    /// 1. Predict displacement: `u_{n+1} = u_n + Δt·v_n + (Δt²/2)·a_n`
    /// 2. Call `force_fn(u_pred)` to compute total force at the predicted state
    /// 3. Compute new acceleration: `a_{n+1} = M⁻¹ · force_total`
    /// 4. Correct velocity: `v_{n+1} = v_n + Δt·((1−γ)·a_n + γ·a_{n+1})`
    /// 5. Apply Dirichlet BCs
    ///
    /// # Arguments
    /// * `mass_lumped` — diagonal entries of the lumped mass matrix
    /// * `dt` — time step size
    /// * `u` — displacement (mutated in-place: `u_n → u_{n+1}`)
    /// * `state` — velocity and acceleration (mutated in-place)
    /// * `bc_dofs` — Dirichlet DOFs to zero after the step
    /// * `force_fn` — computes `f_total` at the predicted displacement `u_pred`
    ///
    /// The `force_fn` callback receives the predicted `u_pred` and must return
    /// the total force vector `f_ext(t_{n+1}) - f_int(u_pred) + f_contact(u_pred)`.
    /// This design lets the caller incorporate arbitrary forces (contact,
    /// damping, nonlinear material) without the integrator knowing about them.
    #[allow(clippy::too_many_arguments)]
    pub fn step<F>(
        &self,
        mass_lumped: &[f64],
        dt: f64,
        u: &mut [f64],
        state: &mut ExplicitState,
        bc_dofs: &[u32],
        force_fn: F,
    ) where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = u.len();
        let g = self.gamma;
        let half_dt2 = 0.5 * dt * dt;

        // 1. Predict displacement
        let mut u_pred = vec![0.0; n];
        for i in 0..n {
            u_pred[i] = u[i] + dt * state.vel[i] + half_dt2 * state.acc[i];
        }

        // Apply Dirichlet BCs to predicted displacement
        for &d in bc_dofs {
            u_pred[d as usize] = 0.0;
        }

        // 2. Compute total force at predicted state
        let force_total = force_fn(&u_pred);

        // 3. Write u = u_pred
        u.copy_from_slice(&u_pred);

        // 4. Compute new acceleration: a_{n+1} = M⁻¹ · f_total
        let mut a_new = vec![0.0; n];
        for i in 0..n {
            a_new[i] = if mass_lumped[i].abs() > 1e-14 {
                force_total[i] / mass_lumped[i]
            } else {
                0.0
            };
        }

        // 5. Correct velocity: v_{n+1} = v_n + Δt·((1-γ)·a_n + γ·a_{n+1})
        for i in 0..n {
            state.vel[i] += dt * ((1.0 - g) * state.acc[i] + g * a_new[i]);
        }

        // 6. Apply Dirichlet BCs to velocity and acceleration
        for &d in bc_dofs {
            let d = d as usize;
            state.vel[d] = 0.0;
            a_new[d] = 0.0;
        }

        state.acc.copy_from_slice(&a_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    #[test]
    fn newmark_free_vibration() {
        let omega = std::f64::consts::PI;
        let k = omega * omega;
        let mut mass_coo = CooMatrix::<f64>::new(1, 1);
        mass_coo.add(0, 0, 1.0);
        let mass = mass_coo.into_csr();
        let mut stiff_coo = CooMatrix::<f64>::new(1, 1);
        stiff_coo.add(0, 0, k);
        let stiff = stiff_coo.into_csr();
        let newmark = Newmark::default();
        let mut u = vec![1.0];
        let force = vec![0.0];
        let dt = 0.001;
        let mut state = NewmarkState::new(1);
        state.acc[0] = -k;
        let t_end = 1.0_f64;
        let n_steps = (t_end / dt).round() as usize;
        for _ in 0..n_steps {
            newmark.step(&mass, &stiff, &force, dt, &mut u, &mut state, &[]);
        }
        let exact = (omega * t_end).cos();
        let err = (u[0] - exact).abs();
        assert!(
            err < 0.01,
            "Newmark free vibration error={err:.4e} (exact={exact:.4})"
        );
    }

    #[test]
    fn generalized_alpha_exp_decay() {
        let lambda = 10.0_f64;
        let n = 1;
        let mut m_coo = CooMatrix::<f64>::new(n, n);
        m_coo.add(0, 0, 1.0);
        let mass = m_coo.into_csr();
        let mut k_coo = CooMatrix::<f64>::new(n, n);
        k_coo.add(0, 0, lambda);
        let stiff = k_coo.into_csr();
        let force_fn = |_t: f64| vec![0.0f64];
        let solver = GeneralizedAlpha::default();
        let mut v = vec![1.0_f64];
        let mut state = GeneralizedAlphaState::new(n);
        state.dvdt[0] = -lambda;
        let dt = 0.1_f64;
        let t_end = 1.0_f64;
        let mut t = 0.0_f64;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            solver.step(&mass, &stiff, &force_fn, h, t, &mut v, &mut state, &[]);
            t += h;
        }
        let exact = (-lambda * t_end).exp();
        let err = (v[0] - exact).abs();
        assert!(
            err < 0.01,
            "GeneralizedAlpha exp decay error={err:.3e} (exact={exact:.6})"
        );
    }

    #[test]
    fn generalized_alpha_stiff_stable() {
        let lambda = 1000.0_f64;
        let n = 1;
        let mut m_coo = CooMatrix::<f64>::new(n, n);
        m_coo.add(0, 0, 1.0);
        let mass = m_coo.into_csr();
        let mut k_coo = CooMatrix::<f64>::new(n, n);
        k_coo.add(0, 0, lambda);
        let stiff = k_coo.into_csr();
        let force_fn = |_t: f64| vec![0.0f64];
        let solver = GeneralizedAlpha::default();
        let mut v = vec![1.0_f64];
        let mut state = GeneralizedAlphaState::new(n);
        state.dvdt[0] = -lambda;
        let dt = 0.1_f64;
        let t_end = 1.0_f64;
        let mut t = 0.0_f64;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            solver.step(&mass, &stiff, &force_fn, h, t, &mut v, &mut state, &[]);
            t += h;
        }
        assert!(
            v[0].abs() < 0.01,
            "GeneralizedAlpha stiff: did not decay; u={:.3e}",
            v[0]
        );
    }

    // ─── Central Difference Explicit tests ────────────────────────────────

    #[test]
    fn central_difference_free_vibration() {
        // SDOF: m=1, k=π², ü + ω²·u = 0, u(0)=1, v(0)=0
        let omega = std::f64::consts::PI;
        let k = omega * omega;
        let mass_lumped = vec![1.0_f64]; // lumped mass
        let stiff_coo = {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, k);
            coo.into_csr()
        };
        let cd = CentralDifferenceExplicit::default();
        let mut u = vec![1.0_f64];
        let mut state = ExplicitState::new(1);
        // initial acceleration: a₀ = M⁻¹(f₀ - K·u₀) = -ω²
        state.acc[0] = -k;

        let dt = 0.001; // << critical
        let t_end = 1.0_f64;
        let n_steps = (t_end / dt).round() as usize;
        for _ in 0..n_steps {
            let stiff = &stiff_coo;
            cd.step(&mass_lumped, dt, &mut u, &mut state, &[], |u_pred| {
                let mut ku = vec![0.0_f64; 1];
                stiff.spmv(u_pred, &mut ku);
                vec![-ku[0]]
            });
        }
        let exact = (omega * t_end).cos();
        let err = (u[0] - exact).abs();
        // Central difference is 2nd-order accurate → small error
        assert!(
            err < 0.001,
            "Central diff free vibration error={err:.4e} (exact={exact:.4})"
        );
    }

    #[test]
    fn central_difference_explicit_vs_newmark_beta0() {
        // Compare CentralDifferenceExplicit against Newmark with β=0, γ=0.5
        // (which is the same algorithm, but Newmark does a linear solve).
        // For a lumped mass, the results should be identical.
        let omega = 10.0_f64;
        let k = omega * omega;
        let mass_lumped = vec![1.0_f64; 1];
        let mut mass_coo = CooMatrix::<f64>::new(1, 1);
        mass_coo.add(0, 0, 1.0);
        let mass = mass_coo.into_csr();
        let mut stiff_coo = CooMatrix::<f64>::new(1, 1);
        stiff_coo.add(0, 0, k);
        let stiff = stiff_coo.into_csr();

        let cd = CentralDifferenceExplicit::default();
        let newmark = Newmark {
            beta: 0.0,
            gamma: 0.5,
        };

        let dt = 0.001;
        let t_end = 0.5_f64;
        let n_steps = (t_end / dt).round() as usize;

        // Run central difference
        let mut u_cd = vec![1.0_f64];
        let mut state_cd = ExplicitState::new(1);
        state_cd.acc[0] = -k;

        for _ in 0..n_steps {
            let stiff = &stiff;
            cd.step(&mass_lumped, dt, &mut u_cd, &mut state_cd, &[], |u_pred| {
                let mut ku = vec![0.0_f64; 1];
                stiff.spmv(u_pred, &mut ku);
                vec![-ku[0]]
            });
        }

        // Run Newmark β=0 (should solve M·a = f - K·u_pred, identical with lumped M)
        let mut u_nm = vec![1.0_f64];
        let mut state_nm = NewmarkState::new(1);
        state_nm.acc[0] = -k;
        let force = vec![0.0_f64];

        for _ in 0..n_steps {
            newmark.step(&mass, &stiff, &force, dt, &mut u_nm, &mut state_nm, &[]);
        }

        let diff = (u_cd[0] - u_nm[0]).abs();
        assert!(diff < 1e-12, "Central diff vs Newmark β=0: diff={diff:.4e}");
    }

    #[test]
    fn central_difference_bc() {
        // DOF 0 is fixed (bc_dofs), DOF 1 is free.
        // System: mass = diag([1, 1]), stiffness = diag([1e6, 0]).
        // DOF 0 should stay at 0 despite initial displacement.
        let mass_lumped = vec![1.0_f64, 1.0_f64];
        let mut stiff_coo = CooMatrix::<f64>::new(2, 2);
        stiff_coo.add(0, 0, 1e6_f64);
        let stiff = stiff_coo.into_csr();

        let cd = CentralDifferenceExplicit::default();
        let mut u = vec![1.0_f64, 0.0_f64];
        let mut state = ExplicitState::new(2);

        let dt = 0.0001;
        let bc_dofs = vec![0u32];
        for _ in 0..10 {
            let stiff = &stiff;
            cd.step(&mass_lumped, dt, &mut u, &mut state, &bc_dofs, |u_pred| {
                let mut ku = vec![0.0_f64; 2];
                stiff.spmv(u_pred, &mut ku);
                vec![-ku[0], -ku[1]]
            });
        }
        assert!(
            (u[0]).abs() < 1e-14,
            "BC DOF 0 should be zero, got {:.4e}",
            u[0]
        );
        assert!((state.vel[0]).abs() < 1e-14, "BC vel[0] should be zero");
        assert!((state.acc[0]).abs() < 1e-14, "BC acc[0] should be zero");
    }
}
