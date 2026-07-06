//! IMEX (implicit–explicit) split time integrators.
//!
//! Convenience driver: [`ImexTimeStepper`].
//! Concrete methods: [`ImexEuler`] (1st order), [`ImexSsp2`] (2nd order),
//! [`ImexRk3`] (fixed-step 3rd order), [`ImexArk3`] (adaptive 3rd order).

use crate::{SolverConfig, solve_gmres};
use super::implicit::identity_minus_dt_jac;
use super::traits::ImexOperator;

// ─── Convenience IMEX driver ─────────────────────────────────────────────────

/// Convenience IMEX driver that dispatches to existing IMEX integrators.
///
/// This avoids rewriting integration loops in examples/apps: users provide an
/// [`ImexOperator`] and choose one of the built-in methods.
pub struct ImexTimeStepper;

impl ImexTimeStepper {
    /// Integrate with first-order IMEX Euler.
    pub fn integrate_euler<O: ImexOperator>(
        &self,
        op: &O,
        t0: f64,
        t_end: f64,
        u: &mut [f64],
        dt: f64,
    ) -> f64 {
        let solver = ImexEuler;
        solver.integrate(
            t0,
            t_end,
            u,
            dt,
            |t, u, out| op.explicit(t, u, out),
            |t, u, out| op.implicit(t, u, out),
            |t, u| op.jac_implicit(t, u),
        )
    }

    /// Integrate with second-order IMEX SSP-RK2.
    pub fn integrate_ssp2<O: ImexOperator>(
        &self,
        op: &O,
        t0: f64,
        t_end: f64,
        u: &mut [f64],
        dt: f64,
    ) -> f64 {
        let solver = ImexSsp2;
        solver.integrate(
            t0,
            t_end,
            u,
            dt,
            |t, u, out| op.explicit(t, u, out),
            |t, u, out| op.implicit(t, u, out),
            |t, u| op.jac_implicit(t, u),
        )
    }

    /// Integrate with fixed-step third-order IMEX RK3.
    pub fn integrate_rk3<O: ImexOperator>(
        &self,
        op: &O,
        t0: f64,
        t_end: f64,
        u: &mut [f64],
        dt: f64,
    ) -> f64 {
        let solver = ImexRk3;
        solver.integrate(
            t0,
            t_end,
            u,
            dt,
            |t, u, out| op.explicit(t, u, out),
            |t, u, out| op.implicit(t, u, out),
            |t, u| op.jac_implicit(t, u),
        )
    }

    /// Integrate with adaptive third-order IMEX ARK3.
    /// Returns `(t_final, dt_last)`.
    pub fn integrate_ark3<O: ImexOperator>(
        &self,
        op: &O,
        t0: f64,
        t_end: f64,
        u: &mut [f64],
        dt: f64,
        solver: &ImexArk3,
    ) -> (f64, f64) {
        solver.integrate(
            t0,
            t_end,
            u,
            dt,
            |t, u, out| op.explicit(t, u, out),
            |t, u, out| op.implicit(t, u, out),
            |t, u| op.jac_implicit(t, u),
        )
    }
}

// ─── IMEX Euler (1st order) ──────────────────────────────────────────────────

/// First-order IMEX Euler (forward Euler for explicit, backward Euler for implicit).
///
/// Integrates the split system:
/// ```text
///   du/dt = f_E(t, u) + f_I(t, u)
/// ```
///
/// One step:
/// ```text
///   (I − dt J_I(t+dt)) Δu = dt [f_E(tₙ, uₙ) + f_I(t+dt, uₙ)]
///   u_{n+1} = uₙ + Δu
/// ```
pub struct ImexEuler;

impl ImexEuler {
    /// Take one IMEX Euler step advancing `u` by `dt` from time `t`.
    pub fn step<FE, FI, J>(
        &self,
        t:            f64,
        dt:           f64,
        u:            &mut [f64],
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    )
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J:  Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };

        // Explicit part at current state
        let mut fe = vec![0.0f64; n];
        rhs_explicit(t, u, &mut fe);

        // Implicit part evaluated at u^n (Picard linearisation)
        let mut fi = vec![0.0f64; n];
        rhs_implicit(t + dt, u, &mut fi);

        // Build (I − dt J_I) for implicit correction
        let jac = jac_implicit(t + dt, u);
        let lhs = identity_minus_dt_jac(&jac, dt);

        // RHS: dt * (f_E + f_I)
        let b: Vec<f64> = (0..n).map(|i| dt * (fe[i] + fi[i])).collect();

        // Solve for Δu
        let mut du = vec![0.0f64; n];
        solve_gmres(&lhs, &b, &mut du, 30, &cfg).expect("ImexEuler: linear solve failed");

        for i in 0..n { u[i] += du[i]; }
    }

    /// Integrate from `t0` to `t_end` with fixed step `dt`.
    /// Returns the final time reached.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate<FE, FI, J>(
        &self,
        t0:           f64,
        t_end:        f64,
        u:            &mut [f64],
        dt:           f64,
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    ) -> f64
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J:  Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let mut t = t0;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            self.step(t, h, u, &rhs_explicit, &rhs_implicit, &jac_implicit);
            t += h;
        }
        t
    }
}

// ─── IMEX SSP-RK2 (2nd order) ────────────────────────────────────────────────

/// Second-order IMEX SSP-RK2 (Pareschi & Russo, 2005, Scheme SI-IMEX(2,2,2)).
pub struct ImexSsp2;

const IMEX_SSP2_GAMMA: f64 = 1.0 - std::f64::consts::FRAC_1_SQRT_2; // 1 − 1/√2 ≈ 0.2929

impl ImexSsp2 {
    /// Take one IMEX SSP-RK2 step advancing `u` by `dt` from time `t`.
    pub fn step<FE, FI, J>(
        &self,
        t:            f64,
        dt:           f64,
        u:            &mut [f64],
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    )
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J:  Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let g = IMEX_SSP2_GAMMA;
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };

        // ── Stage 1 ──
        let mut ke1 = vec![0.0f64; n];
        rhs_explicit(t, u, &mut ke1);

        let t1 = t + g * dt;
        let jac1 = jac_implicit(t1, u);
        let lhs1 = identity_minus_dt_jac(&jac1, dt * g);
        let mut fi1 = vec![0.0f64; n];
        rhs_implicit(t1, u, &mut fi1);
        let mut ki1 = vec![0.0f64; n];
        solve_gmres(&lhs1, &fi1, &mut ki1, 30, &cfg).expect("ImexSsp2 stage 1 solve failed");

        // ── Stage 2 ──
        let mut u2 = u.to_vec();
        for i in 0..n {
            u2[i] += dt * g * ke1[i] + dt * (1.0 - g) * ki1[i];
        }

        let mut ke2 = vec![0.0f64; n];
        rhs_explicit(t1, &u2, &mut ke2);

        let t2 = t + dt;
        let jac2 = jac_implicit(t2, &u2);
        let lhs2 = identity_minus_dt_jac(&jac2, dt * g);
        let mut fi2 = vec![0.0f64; n];
        rhs_implicit(t2, &u2, &mut fi2);
        let mut ki2 = vec![0.0f64; n];
        solve_gmres(&lhs2, &fi2, &mut ki2, 30, &cfg).expect("ImexSsp2 stage 2 solve failed");

        // ── Update ──
        for i in 0..n {
            u[i] += dt * ((1.0 - g) * (ke1[i] + ki1[i]) + g * (ke2[i] + ki2[i]));
        }
    }

    /// Integrate from `t0` to `t_end` with fixed step `dt`.
    /// Returns the final time reached.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate<FE, FI, J>(
        &self,
        t0:           f64,
        t_end:        f64,
        u:            &mut [f64],
        dt:           f64,
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    ) -> f64
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J:  Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let mut t = t0;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            self.step(t, h, u, &rhs_explicit, &rhs_implicit, &jac_implicit);
            t += h;
        }
        t
    }
}

// ─── IMEX-ARK3(2)4L[2]SA ─────────────────────────────────────────────────────

/// IMEX-ARK3(2)4L[2]SA integrator (Kennedy & Carpenter, 2003).
///
/// Integrates the split system:
/// ```text
///   du/dt = f_E(t, u) + f_I(t, u)
/// ```
/// Butcher tables: ARK3(2)4L[2]SA, 4 stages, 3rd-order (embedded 2nd-order).
pub struct ImexArk3 {
    /// Relative tolerance for step control.
    pub rtol: f64,
    /// Absolute tolerance.
    pub atol: f64,
    /// Minimum step size.
    pub dt_min: f64,
    /// Maximum step size.
    pub dt_max: f64,
}

impl Default for ImexArk3 {
    fn default() -> Self {
        ImexArk3 { rtol: 1e-4, atol: 1e-8, dt_min: 1e-14, dt_max: 1.0 }
    }
}

// ARK3(2)4L[2]SA Butcher tableau coefficients (Kennedy & Carpenter 2003)
const ARK_GAMMA: f64 = 1767732205903.0 / 4055673282236.0;

const ARK_AI: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [ARK_GAMMA, ARK_GAMMA, 0.0, 0.0],
    [2746238789719.0 / 10658868560708.0, -640167445237.0 / 6845629431997.0, ARK_GAMMA, 0.0],
    [
        1471266399579.0 / 7840856788654.0,
        -4482444167858.0 / 7529755066697.0,
        11266239266428.0 / 11593286722821.0,
        ARK_GAMMA,
    ],
];

const ARK_BI: [f64; 4] = [
    1471266399579.0 / 7840856788654.0,
    -4482444167858.0 / 7529755066697.0,
    11266239266428.0 / 11593286722821.0,
    ARK_GAMMA,
];

const ARK_BI_HAT: [f64; 4] = [
    2756255671327.0 / 12835298489170.0,
    -10771552573575.0 / 22201958757719.0,
    9247589265047.0 / 10645013368117.0,
    2193209047091.0 / 5459859503100.0,
];

const ARK_AE: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [1767732205903.0 / 2027836641118.0, 0.0, 0.0, 0.0],
    [5535828885825.0 / 10492691773637.0, 788022342437.0 / 10882634858940.0, 0.0, 0.0],
    [
        6485989280629.0 / 16251701735622.0,
        -4246266847089.0 / 9704473918619.0,
        10755448449292.0 / 10357097424841.0,
        0.0,
    ],
];

const ARK_C: [f64; 4] = [
    0.0,
    1767732205903.0 / 2027836641118.0,
    3.0 / 5.0,
    1.0,
];

impl ImexArk3 {
    /// Integrate from `t0` to `t_end` with initial step `dt`.
    ///
    /// Returns `(t_final, dt_last)`.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate<FE, FI, J>(
        &self,
        t0:       f64,
        t_end:    f64,
        u:        &mut [f64],
        mut dt:   f64,
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    ) -> (f64, f64)
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J:  Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let n = u.len();
        let mut t = t0;
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };

        let mut ki_e = vec![vec![0.0f64; n]; 4];
        let mut ki_i = vec![vec![0.0f64; n]; 4];

        while t < t_end {
            dt = dt.min(t_end - t).max(self.dt_min);

            let mut u_stage = vec![vec![0.0f64; n]; 4];

            // Stage 0
            rhs_explicit(t + ARK_C[0] * dt, u, &mut ki_e[0]);
            rhs_implicit(t + ARK_C[0] * dt, u, &mut ki_i[0]);
            u_stage[0] = u.to_vec();

            // Stages 1..3
            for s in 1..4 {
                let mut u_s = u.to_vec();
                for j in 0..s {
                    for i in 0..n {
                        u_s[i] += dt * ARK_AE[s][j] * ki_e[j][i];
                        u_s[i] += dt * ARK_AI[s][j] * ki_i[j][i];
                    }
                }

                let t_s = t + ARK_C[s] * dt;
                let aii = ARK_AI[s][s];
                let jac_s = jac_implicit(t_s, &u_s);
                let lhs_s = identity_minus_dt_jac(&jac_s, dt * aii);
                let mut fi_s = vec![0.0f64; n];
                rhs_implicit(t_s, &u_s, &mut fi_s);

                let mut k_i_s = vec![0.0f64; n];
                solve_gmres(&lhs_s, &fi_s, &mut k_i_s, 30, &cfg)
                    .expect("ImexArk3: implicit stage solve failed");
                ki_i[s] = k_i_s;

                for i in 0..n {
                    u_s[i] += dt * aii * ki_i[s][i];
                }
                u_stage[s] = u_s.clone();

                rhs_explicit(t_s, &u_s, &mut ki_e[s]);
            }

            // 3rd-order solution
            let mut u3 = u.to_vec();
            for s in 0..4 {
                for i in 0..n {
                    u3[i] += dt * ARK_BI[s] * (ki_e[s][i] + ki_i[s][i]);
                }
            }

            // 2nd-order embedded solution for error estimate
            let mut u2 = u.to_vec();
            for s in 0..4 {
                for i in 0..n {
                    u2[i] += dt * ARK_BI_HAT[s] * (ki_e[s][i] + ki_i[s][i]);
                }
            }

            // Error norm
            let err_norm = (0..n).map(|i| {
                let e = u3[i] - u2[i];
                let sc = self.atol + self.rtol * u[i].abs().max(u3[i].abs());
                (e / sc).powi(2)
            }).sum::<f64>().sqrt() / (n as f64).sqrt();

            if err_norm <= 1.0 || dt <= self.dt_min {
                u.copy_from_slice(&u3);
                t += dt;
            }

            if err_norm > 0.0 {
                let factor = (0.9 / err_norm).powf(1.0 / 3.0);
                dt *= factor.clamp(0.1, 5.0);
            } else {
                dt *= 5.0;
            }
            dt = dt.min(self.dt_max).max(self.dt_min);
        }
        (t, dt)
    }
}

/// Fixed-step third-order IMEX RK3 integrator.
///
/// This is a convenience wrapper around [`ImexArk3`] with adaptation disabled
/// by clamping `dt_min = dt_max = dt`.
pub struct ImexRk3;

impl ImexRk3 {
    /// Take one fixed-size third-order IMEX step.
    pub fn step<FE, FI, J>(
        &self,
        t: f64,
        dt: f64,
        u: &mut [f64],
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    )
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let solver = ImexArk3 {
            rtol: 1e-14,
            atol: 1e-14,
            dt_min: dt,
            dt_max: dt,
        };
        let _ = solver.integrate(
            t,
            t + dt,
            u,
            dt,
            rhs_explicit,
            rhs_implicit,
            jac_implicit,
        );
    }

    /// Integrate with fixed step size `dt` to `t_end`.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate<FE, FI, J>(
        &self,
        t0: f64,
        t_end: f64,
        u: &mut [f64],
        dt: f64,
        rhs_explicit: FE,
        rhs_implicit: FI,
        jac_implicit: J,
    ) -> f64
    where
        FE: Fn(f64, &[f64], &mut [f64]),
        FI: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> fem_linalg::CsrMatrix<f64>,
    {
        let mut t = t0;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            self.step(t, h, u, &rhs_explicit, &rhs_implicit, &jac_implicit);
            t += h;
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    struct SplitDecayOp {
        lambda: f64,
    }

    impl ImexOperator for SplitDecayOp {
        fn explicit(&self, _t: f64, _u: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn implicit(&self, _t: f64, u: &[f64], out: &mut [f64]) {
            out[0] = -self.lambda * u[0];
        }
        fn jac_implicit(&self, _t: f64, _u: &[f64]) -> fem_linalg::CsrMatrix<f64> {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -self.lambda);
            coo.into_csr()
        }
    }

    #[test]
    fn imex_driver_euler_matches_decay() {
        let op = SplitDecayOp { lambda: 10.0 };
        let driver = ImexTimeStepper;
        let mut u = vec![1.0f64];
        let t_f = driver.integrate_euler(&op, 0.0, 0.5, &mut u, 0.01);
        let exact = (-10.0 * t_f).exp();
        assert!((u[0] - exact).abs() < 2e-2, "ImexTimeStepper Euler error too large: u={:.4e}, exact={:.4e}", u[0], exact);
    }

    #[test]
    fn imex_driver_ssp2_matches_decay() {
        let op = SplitDecayOp { lambda: 10.0 };
        let driver = ImexTimeStepper;
        let mut u = vec![1.0f64];
        let t_f = driver.integrate_ssp2(&op, 0.0, 0.5, &mut u, 0.01);
        let exact = (-10.0 * t_f).exp();
        assert!((u[0] - exact).abs() < 5e-3, "ImexTimeStepper SSP2 error too large: u={:.4e}, exact={:.4e}", u[0], exact);
    }

    #[test]
    fn imex_driver_rk3_matches_decay() {
        let op = SplitDecayOp { lambda: 10.0 };
        let driver = ImexTimeStepper;
        let mut u = vec![1.0f64];
        let t_f = driver.integrate_rk3(&op, 0.0, 0.5, &mut u, 0.02);
        let exact = (-10.0 * t_f).exp();
        assert!((u[0] - exact).abs() < 2e-3, "ImexTimeStepper RK3 error too large: u={:.4e}, exact={:.4e}", u[0], exact);
    }

    #[test]
    fn imex_ark3_non_stiff_decay() {
        let lambda = 1.0_f64;
        let n = 1;
        let f_explicit = |_t: f64, _u: &[f64], out: &mut [f64]| { out[0] = 0.0; };
        let f_implicit = move |_t: f64, u: &[f64], out: &mut [f64]| { out[0] = -lambda * u[0]; };
        let jac_implicit = move |_t: f64, _u: &[f64]| {
            let mut coo = CooMatrix::<f64>::new(n, n);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        };
        let solver = ImexArk3 { rtol: 1e-6, atol: 1e-8, ..Default::default() };
        let mut u = vec![1.0_f64];
        let (t_final, _) = solver.integrate(0.0, 1.0, &mut u, 0.1, f_explicit, f_implicit, jac_implicit);
        let exact = (-lambda * t_final).exp();
        let err = (u[0] - exact).abs();
        assert!(err < 1e-3, "ImexArk3 decay error={err:.3e} (exact={exact:.6})");
    }

    #[test]
    fn imex_ark3_adaptive_step() {
        let lambda = 100.0_f64;
        let f_e = |_t: f64, _u: &[f64], out: &mut [f64]| { out[0] = 0.0; };
        let f_i = move |_t: f64, u: &[f64], out: &mut [f64]| { out[0] = -lambda * u[0]; };
        let jac = move |_t: f64, _u: &[f64]| {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        };
        let solver = ImexArk3::default();
        let mut u = vec![1.0_f64];
        let (t_f, _) = solver.integrate(0.0, 0.1, &mut u, 0.01, f_e, f_i, jac);
        let exact = (-lambda * t_f).exp();
        assert!((u[0] - exact).abs() < 0.01, "ImexArk3 adaptive: u={:.4e}, exact={exact:.4e}", u[0]);
    }

    #[test]
    fn imex_euler_stiff_decay_stable() {
        let lambda = 100.0_f64;
        let f_e = |_t: f64, _u: &[f64], out: &mut [f64]| { out[0] = 0.0; };
        let f_i = move |_t: f64, u: &[f64], out: &mut [f64]| { out[0] = -lambda * u[0]; };
        let jac = move |_t: f64, _u: &[f64]| {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        };
        let solver = ImexEuler;
        let mut u = vec![1.0_f64];
        solver.integrate(0.0, 1.0, &mut u, 0.1, f_e, f_i, jac);
        assert!(u[0].abs() < 1e-2, "ImexEuler stiff decay unstable: u={:.3e}", u[0]);
    }

    #[test]
    fn imex_ssp2_second_order_check() {
        let lambda = 100.0_f64;
        let omega = std::f64::consts::PI;
        let f_e = move |t: f64, _u: &[f64], out: &mut [f64]| { out[0] = (omega * t).sin(); };
        let f_i = move |_t: f64, u: &[f64], out: &mut [f64]| { out[0] = -lambda * u[0]; };
        let jac = |_t: f64, _u: &[f64]| {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -lambda);
            coo.into_csr()
        };
        let exact = move |t: f64| {
            (lambda * (omega * t).sin() - omega * (omega * t).cos() + omega * (-lambda * t).exp())
                / (lambda * lambda + omega * omega)
        };
        let t_end = 1.0_f64;
        let mut u1 = vec![0.0_f64];
        let mut u2 = vec![0.0_f64];
        let solver = ImexSsp2;
        solver.integrate(0.0, t_end, &mut u1, 0.1, f_e, f_i, jac);
        solver.integrate(0.0, t_end, &mut u2, 0.05, f_e, f_i, jac);
        let e1 = (u1[0] - exact(t_end)).abs();
        let e2 = (u2[0] - exact(t_end)).abs();
        let order = (e1 / e2).log2();
        assert!(order > 1.7, "ImexSsp2 order too low: order={order:.2}, e1={e1:.3e}, e2={e2:.3e}");
    }

    #[test]
    fn imex_rk3_third_order_check() {
        let a = 0.7_f64;
        let b = 4.3_f64;
        let f_e = move |_t: f64, u: &[f64], out: &mut [f64]| { out[0] = -a * u[0]; };
        let f_i = move |_t: f64, u: &[f64], out: &mut [f64]| { out[0] = -b * u[0]; };
        let jac = |_t: f64, _u: &[f64]| {
            let mut coo = CooMatrix::<f64>::new(1, 1);
            coo.add(0, 0, -b);
            coo.into_csr()
        };
        let exact = move |t: f64| (-(a + b) * t).exp();
        let t_end = 1.0_f64;
        let solver = ImexRk3;
        let mut u1 = vec![1.0_f64];
        let mut u2 = vec![1.0_f64];
        solver.integrate(0.0, t_end, &mut u1, 0.08, f_e, f_i, jac);
        solver.integrate(0.0, t_end, &mut u2, 0.04, f_e, f_i, jac);
        let e1 = (u1[0] - exact(t_end)).abs();
        let e2 = (u2[0] - exact(t_end)).abs();
        let order = (e1 / e2).log2();
        assert!(order > 2.3, "ImexRk3 order too low: order={order:.2}, e1={e1:.3e}, e2={e2:.3e}");
    }
}
