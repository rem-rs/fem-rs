//! Structural dynamics time integrators for second-order systems.
//!
//! Methods: [`Newmark`] (Newmark-β), [`GeneralizedAlpha`] (first-order).

use crate::{SolverConfig, solve_cg, solve_gmres};

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
    pub fn init_from(vel: Vec<f64>, mass: &fem_linalg::CsrMatrix<f64>, stiff: &fem_linalg::CsrMatrix<f64>, u: &[f64], force: &[f64]) -> Self {
        let n = u.len();
        // a₀ = M⁻¹(f₀ - K u₀)
        let mut ku = vec![0.0; n];
        stiff.spmv(u, &mut ku);
        let rhs: Vec<f64> = (0..n).map(|i| force[i] - ku[i]).collect();
        let mut acc = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
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
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 1000, verbose: false, ..SolverConfig::default() };
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
        let gamma   = 0.5 + alpha_m - alpha_f;
        (alpha_f, alpha_m, gamma)
    }

    /// Advance `v` from `t` by `dt` for the system `M dv/dt + K v = f(t)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        mass:     &fem_linalg::CsrMatrix<f64>,
        stiff:    &fem_linalg::CsrMatrix<f64>,
        force_fn: &dyn Fn(f64) -> Vec<f64>,
        dt:       f64,
        t:        f64,
        v:        &mut [f64],
        state:    &mut GeneralizedAlphaState,
        bc_dofs:  &[u32],
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
                for ptr in mass.row_ptr[i]..mass.row_ptr[i+1] {
                    coo.add(i, mass.col_idx[ptr] as usize, alpha_m * mass.values[ptr]);
                }
            }
            for i in 0..stiff.nrows {
                for ptr in stiff.row_ptr[i]..stiff.row_ptr[i+1] {
                    coo.add(i, stiff.col_idx[ptr] as usize, alpha_f * gamma * dt * stiff.values[ptr]);
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
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
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
        assert!(err < 0.01, "Newmark free vibration error={err:.4e} (exact={exact:.4})");
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
        assert!(err < 0.01, "GeneralizedAlpha exp decay error={err:.3e} (exact={exact:.6})");
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
        assert!(v[0].abs() < 0.01, "GeneralizedAlpha stiff: did not decay; u={:.3e}", v[0]);
    }
}
