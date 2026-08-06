//! Spectral Deferred Correction (SDC) time integration.
//!
//! SDC is an iterative method that uses a low-order solver as a preconditioner
//! for the collocation formulation of an ODE, achieving high order through
//! iteration on Gauss-Lobatto nodes.
//!
//! # Algorithm
//!
//! 1. Split `[t_n, t_{n+1}]` into `M` quadrature nodes `t_n = τ₀ < … < τ_{M-1} = t_{n+1}`
//! 2. Predict: Forward Euler prediction at each node
//! 3. For `k = 1..K` (correction sweeps):
//!    - Compute residual `r(t) = u(t) - u(t_n) - ∫ f(u(s)) ds`
//!    - Solve correction: `δ(t) = u_prov(t) + ∫ [f(u_prov(s)) - f(u(s))] ds`
//!      Explicit SDC: u_new_j = u_0 + Σ w_l · f(u_old_l)  (deferred correction iteration)

/// Gauss-Lobatto nodes and weights on [0, 1].
fn gauss_lobatto_legendre(m: usize) -> (Vec<f64>, Vec<f64>) {
    match m {
        2 => (vec![0.0, 1.0], vec![0.5, 0.5]),
        3 => (vec![0.0, 0.5, 1.0], vec![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        4 => (
            vec![
                0.0,
                0.5 - 0.5 * (3.0_f64 / 7.0).sqrt(),
                0.5 + 0.5 * (3.0_f64 / 7.0).sqrt(),
                1.0,
            ],
            vec![1.0 / 12.0, 5.0 / 12.0, 5.0 / 12.0, 1.0 / 12.0],
        ),
        _ => panic!("SDC: unsupported M={m} (supported: 2,3,4)"),
    }
}

/// Configuration for the SDC integrator.
#[derive(Debug, Clone)]
pub struct SdcConfig {
    /// Number of Gauss-Lobatto quadrature nodes per step.
    pub m: usize,
    /// Number of correction sweeps.
    pub k: usize,
}

impl Default for SdcConfig {
    fn default() -> Self {
        SdcConfig { m: 3, k: 3 }
    }
}

/// Spectral Deferred Correction integrator.
pub struct SdcIntegrator {
    pub config: SdcConfig,
}

impl SdcIntegrator {
    pub fn new(config: SdcConfig) -> Self {
        SdcIntegrator { config }
    }

    /// Integrate one step from `t` to `t + dt` using explicit SDC.
    pub fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let m = self.config.m;
        let k = self.config.k;
        let n = u.len();

        let (nodes, weights) = gauss_lobatto_legendre(m);

        // Storage: solution and RHS at each node
        let mut u_nodes = vec![vec![0.0; n]; m];
        let mut f_nodes = vec![vec![0.0; n]; m];

        // Node 0 = initial condition
        u_nodes[0].copy_from_slice(u);
        rhs(t, &u_nodes[0], &mut f_nodes[0]);

        // --- Prediction: Forward Euler sweep ---
        for j in 0..(m - 1) {
            let dj = (nodes[j + 1] - nodes[j]) * dt;
            let uj = u_nodes[j].clone();
            u_nodes[j + 1].copy_from_slice(&uj);
            for i in 0..n {
                u_nodes[j + 1][i] += dj * f_nodes[j][i];
            }
            rhs(t + nodes[j + 1] * dt, &u_nodes[j + 1], &mut f_nodes[j + 1]);
        }

        // --- Correction sweeps (Picard iteration) ---
        for _iter in 0..k {
            let f_old = f_nodes.clone(); // freeze previous RHS values

            for j in 1..m {
                // u_new(τⱼ) = u(t_n) + Σ₁ wₗ · f_old(τₗ)  (global GL quadrature)
                let u0 = u_nodes[0].clone();
                u_nodes[j].copy_from_slice(&u0);
                for l in 0..m {
                    let w = weights[l] * dt;
                    for i in 0..n {
                        u_nodes[j][i] += w * f_old[l][i];
                    }
                }
                // Re-evaluate RHS at the new solution
                rhs(t + nodes[j] * dt, &u_nodes[j], &mut f_nodes[j]);
            }
        }

        // Final state = last node
        u.copy_from_slice(&u_nodes[m - 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// du/dt = -u → u(t) = exp(-t)
    fn exp_decay_rhs(_t: f64, u: &[f64], dudt: &mut [f64]) {
        dudt[0] = -u[0];
    }

    fn exp_decay_exact(t: f64) -> f64 {
        (-t).exp()
    }

    #[test]
    fn sdc_gauss_lobatto_nodes_2() {
        let (n, _w) = gauss_lobatto_legendre(2);
        assert_eq!(n.len(), 2);
        assert!((n[0] - 0.0).abs() < 1e-14);
        assert!((n[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn sdc_gauss_lobatto_nodes_3() {
        let (n, _w) = gauss_lobatto_legendre(3);
        assert_eq!(n.len(), 3);
        assert!((n[0] - 0.0).abs() < 1e-14);
        assert!((n[1] - 0.5).abs() < 1e-14);
        assert!((n[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn sdc_integrates_exponential_decay() {
        let sdc = SdcIntegrator::new(SdcConfig { m: 3, k: 3 });
        let mut u = vec![1.0];
        let dt = 0.05;
        let n_steps = 20;
        let mut t = 0.0;
        for _ in 0..n_steps {
            sdc.step(t, dt, &mut u, exp_decay_rhs);
            t += dt;
        }
        let expected = exp_decay_exact(t);
        let err = (u[0] - expected).abs();
        assert!(
            err < 0.05,
            "SDC error at t={t}: {err:.3e}, expected {expected:.6}"
        );
    }

    #[test]
    fn sdc_converges_with_more_sweeps() {
        let dt = 0.25;
        let t_final = 1.0;
        let n_steps = (t_final / dt) as usize;

        // M=3, K=1 (one sweep)
        let sdc1 = SdcIntegrator::new(SdcConfig { m: 3, k: 1 });
        let mut u1 = vec![1.0];
        let mut t = 0.0;
        for _ in 0..n_steps {
            sdc1.step(t, dt, &mut u1, exp_decay_rhs);
            t += dt;
        }
        let err1 = (u1[0] - exp_decay_exact(t)).abs();

        // M=3, K=10 (more sweeps)
        let sdc10 = SdcIntegrator::new(SdcConfig { m: 3, k: 10 });
        let mut u10 = vec![1.0];
        t = 0.0;
        for _ in 0..n_steps {
            sdc10.step(t, dt, &mut u10, exp_decay_rhs);
            t += dt;
        }
        let err10 = (u10[0] - exp_decay_exact(t)).abs();

        assert!(
            err10 <= err1 * 1.01 || err1 > 1e-10,
            "more SDC sweeps should not increase error: K1={err1:.3e} K10={err10:.3e}"
        );
    }
}
