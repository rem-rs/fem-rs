//! Symplectic integrators for Hamiltonian systems.
//!
//! The MFEM `SIAVSolver` tableau itself (orders 1–4, `ode.cpp:1109-1149`) has
//! a single implementation in this crate: [`super::mfem_ode::SiavSolver`],
//! the `TimeDependentOperator`/`P_`-operator form MFEM's electromagnetics
//! miniapps drive (ex20 pattern).  Its `(q, p)`-tuple `HamiltonianSystem`
//! twin used to live here as a second hand copy of the same tables and stage
//! loop; it had no consumers outside its own tests and was removed at D646
//! (死代码零容忍) — drive [`super::mfem_ode::SiavSolver`] directly instead.
//!
//! [`Yoshida4`] is *not* part of that duplication: it is a distinct 4th-order
//! composition (Yoshida 1990), kept for long-term energy conservation
//! experiments.
//!
//! These schemes preserve the phase-space volume of Hamiltonian systems,
//! giving excellent long-term energy conservation.

use super::traits::HamiltonianSystem;

/// 4th-order Yoshida symplectic integrator (a special composition of three
/// leapfrog steps).  Coefficients from Yoshida (1990).
pub struct Yoshida4;

impl Yoshida4 {
    /// Create a new `Yoshida4` integrator.
    pub fn new() -> Self {
        Self
    }

    /// Advance the Hamiltonian system by one time step.
    pub fn step(
        &self,
        sys: &dyn HamiltonianSystem,
        q: &mut [f64],
        p: &mut [f64],
        _t: f64,
        dt: f64,
    ) {
        // Yoshida 4th-order coefficients
        let cbrt2 = 2.0_f64.powf(1.0 / 3.0);
        let w1 = 1.0 / (2.0 - cbrt2);
        let w0 = -cbrt2 / (2.0 - cbrt2);

        // 7-stage Yoshida composition (drift-kick-drift per stage, merged)
        let n = q.len();
        let mut buf = vec![0.0_f64; n];

        // Stage 1: drift(w1/2)
        sys.grad_p(q, p, &mut buf);
        for k in 0..n {
            q[k] += (w1 / 2.0) * dt * buf[k];
        }
        // Stage 2: kick(w1)
        sys.grad_q(q, p, &mut buf);
        for k in 0..n {
            p[k] -= w1 * dt * buf[k];
        }
        // Stage 3: drift((w1+w0)/2)
        sys.grad_p(q, p, &mut buf);
        for k in 0..n {
            q[k] += ((w1 + w0) / 2.0) * dt * buf[k];
        }
        // Stage 4: kick(w0)
        sys.grad_q(q, p, &mut buf);
        for k in 0..n {
            p[k] -= w0 * dt * buf[k];
        }
        // Stage 5: drift((w1+w0)/2)
        sys.grad_p(q, p, &mut buf);
        for k in 0..n {
            q[k] += ((w1 + w0) / 2.0) * dt * buf[k];
        }
        // Stage 6: kick(w1)
        sys.grad_q(q, p, &mut buf);
        for k in 0..n {
            p[k] -= w1 * dt * buf[k];
        }
        // Stage 7: drift(w1/2)
        sys.grad_p(q, p, &mut buf);
        for k in 0..n {
            q[k] += (w1 / 2.0) * dt * buf[k];
        }
    }
}

impl Default for Yoshida4 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::traits::HamiltonianSystem;

    /// Simple harmonic oscillator: H = p²/(2m) + k·q²/2
    struct HarmonicOscillator {
        m: f64,
        k: f64,
    }

    impl HamiltonianSystem for HarmonicOscillator {
        fn grad_q(&self, q: &[f64], _p: &[f64], out: &mut [f64]) {
            out[0] = self.k * q[0];
        }
        fn grad_p(&self, _q: &[f64], p: &[f64], out: &mut [f64]) {
            out[0] = p[0] / self.m;
        }
    }

    fn harmonic_energy(q: f64, p: f64, m: f64, k: f64) -> f64 {
        0.5 * p * p / m + 0.5 * k * q * q
    }

    // D646: the former `HamiltonianSiavSolver` tests (orders 1–4 energy
    // conservation, oscillator period, coefficient tables, invalid-order
    // panic) went with the deleted duplicate; the SIAV tableau is covered by
    // `mfem_ode::SiavSolver`'s tests (`d89_ode_solvers.rs`) and ex20.  Only
    // the distinct `Yoshida4` composition keeps local coverage here.

    #[test]
    fn yoshida4_oscillator_period() {
        // Harmonic oscillator with m=1, k=1 has period 2π.
        let sys = HarmonicOscillator { m: 1.0, k: 1.0 };
        let solver = Yoshida4::new();
        let mut q = vec![1.0_f64];
        let mut p = vec![0.0_f64];
        let dt = 0.01;
        let nsteps = (2.0 * std::f64::consts::PI / dt).round() as usize;
        for _ in 0..nsteps {
            solver.step(&sys, &mut q, &mut p, 0.0, dt);
        }
        assert!(
            (q[0] - 1.0).abs() < 0.01,
            "oscillator period error too large: q={}",
            q[0]
        );
        assert!(
            p[0].abs() < 0.01,
            "oscillator period error too large: p={}",
            p[0]
        );
    }

    #[test]
    fn yoshida4_energy_conservation() {
        let sys = HarmonicOscillator { m: 1.0, k: 1.0 };
        let solver = Yoshida4::new();
        let mut q = vec![0.0_f64];
        let mut p = vec![1.0_f64];
        let e0 = harmonic_energy(q[0], p[0], 1.0, 1.0);
        let dt = 0.1;
        let nsteps = 1000;
        for _ in 0..nsteps {
            solver.step(&sys, &mut q, &mut p, 0.0, dt);
        }
        let e1 = harmonic_energy(q[0], p[0], 1.0, 1.0);
        let rel_err = (e1 - e0).abs() / e0;
        assert!(
            rel_err < 1e-4,
            "Yoshida4 energy drift too large: {rel_err:.3e}"
        );
    }
}
