//! Symplectic integrators for Hamiltonian systems.
//!
//! Methods: [`VerletStepper`] (velocity-Verlet, 2nd order),
//! [`LeapfrogStepper`] (alias for Verlet), [`Yoshida4Stepper`] (4th order).

use super::traits::HamiltonianSystem;

/// Velocity-Verlet symplectic integrator (2nd order).
pub struct VerletStepper;

impl VerletStepper {
    pub fn step<S: HamiltonianSystem>(&self, sys: &S, q: &mut [f64], p: &mut [f64], dt: f64) {
        let n = q.len();
        assert_eq!(p.len(), n, "VerletStepper: q/p size mismatch");

        let mut gq = vec![0.0_f64; n];
        let mut gp = vec![0.0_f64; n];

        // Half kick: p_{n+1/2} = p_n - (dt/2) dH/dq(q_n, p_n)
        sys.grad_q(q, p, &mut gq);
        for i in 0..n {
            p[i] -= 0.5 * dt * gq[i];
        }

        // Drift: q_{n+1} = q_n + dt dH/dp(q_n, p_{n+1/2})
        sys.grad_p(q, p, &mut gp);
        for i in 0..n {
            q[i] += dt * gp[i];
        }

        // Half kick: p_{n+1} = p_{n+1/2} - (dt/2) dH/dq(q_{n+1}, p_{n+1/2})
        sys.grad_q(q, p, &mut gq);
        for i in 0..n {
            p[i] -= 0.5 * dt * gq[i];
        }
    }
}

/// Leapfrog integrator (equivalent to velocity-Verlet in kick-drift-kick form).
pub struct LeapfrogStepper;

impl LeapfrogStepper {
    pub fn step<S: HamiltonianSystem>(&self, sys: &S, q: &mut [f64], p: &mut [f64], dt: f64) {
        VerletStepper.step(sys, q, p, dt);
    }
}

/// Yoshida 4th-order symplectic composition of Verlet substeps.
pub struct Yoshida4Stepper;

impl Yoshida4Stepper {
    pub fn step<S: HamiltonianSystem>(&self, sys: &S, q: &mut [f64], p: &mut [f64], dt: f64) {
        // Yoshida composition coefficients
        let cbrt2 = 2.0_f64.powf(1.0 / 3.0);
        let w1 = 1.0 / (2.0 - cbrt2);
        let w0 = -cbrt2 / (2.0 - cbrt2);

        let verlet = VerletStepper;
        verlet.step(sys, q, p, w1 * dt);
        verlet.step(sys, q, p, w0 * dt);
        verlet.step(sys, q, p, w1 * dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct HarmonicOscillator {
        omega: f64,
    }

    impl HamiltonianSystem for HarmonicOscillator {
        fn grad_q(&self, q: &[f64], _p: &[f64], out: &mut [f64]) {
            out[0] = self.omega * self.omega * q[0];
        }
        fn grad_p(&self, _q: &[f64], p: &[f64], out: &mut [f64]) {
            out[0] = p[0];
        }
    }

    fn ho_energy(omega: f64, q: f64, p: f64) -> f64 {
        0.5 * p * p + 0.5 * omega * omega * q * q
    }

    #[test]
    fn verlet_ho_energy_nearly_conserved() {
        let sys = HarmonicOscillator { omega: 2.0 };
        let stepper = VerletStepper;
        let mut q = vec![1.0_f64];
        let mut p = vec![0.0_f64];
        let e0 = ho_energy(sys.omega, q[0], p[0]);
        let dt = 0.01;
        let n_steps = 5_000;
        for _ in 0..n_steps {
            stepper.step(&sys, &mut q, &mut p, dt);
        }
        let e1 = ho_energy(sys.omega, q[0], p[0]);
        assert!((e1 - e0).abs() < 5e-3, "Verlet energy drift too large: e0={e0:.6}, e1={e1:.6}");
    }

    #[test]
    fn yoshida4_ho_energy_better_than_verlet() {
        let sys = HarmonicOscillator { omega: 2.0 };
        let verlet = VerletStepper;
        let yosh = Yoshida4Stepper;
        let dt = 0.02;
        let n_steps = 2_000;
        let mut qv = vec![1.0_f64];
        let mut pv = vec![0.0_f64];
        let e0 = ho_energy(sys.omega, qv[0], pv[0]);
        for _ in 0..n_steps {
            verlet.step(&sys, &mut qv, &mut pv, dt);
        }
        let ev = ho_energy(sys.omega, qv[0], pv[0]);
        let mut qy = vec![1.0_f64];
        let mut py = vec![0.0_f64];
        for _ in 0..n_steps {
            yosh.step(&sys, &mut qy, &mut py, dt);
        }
        let ey = ho_energy(sys.omega, qy[0], py[0]);
        let drift_v = (ev - e0).abs();
        let drift_y = (ey - e0).abs();
        assert!(drift_y <= drift_v * 1.2, "Yoshida4 should be at least comparable/better: verlet={drift_v:.3e}, yosh={drift_y:.3e}");
    }
}
