//! Symplectic integrators for Hamiltonian systems.
//!
//! The [`SIAVSolver`] implements a variable-order symplectic integrator
//! (orders 1–4) using the coefficients from MFEM's `SIAV` (Symplectic
//! Integration Algorithm V) scheme.
//!
//! These schemes preserve the phase-space volume of Hamiltonian systems,
//! giving excellent long-term energy conservation.

use super::traits::HamiltonianSystem;

/// Variable-order symplectic integration algorithm (SIAV).
///
/// Supports orders 1–4 via a set of Yoshida-type composition coefficients.
/// The scheme advances `(q, p)` under Hamilton's equations:
/// ```text
///   dq/dt =  ∂H/∂p
///   dp/dt = -∂H/∂q
/// ```
///
/// Each stage updates `p` (using `-∂H/∂q`) then `q` (using `∂H/∂p`):
/// ```text
///   for i in 0..order:
///     if b[i] ≠ 0:  p += b[i] · dt · (-∂H/∂q)
///     q += a[i] · dt · (∂H/∂p)
/// ```
pub struct SIAVSolver {
    order: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl SIAVSolver {
    /// Create a new `SIAVSolver` of the given order (1–4).
    pub fn new(order: usize) -> Self {
        let (a, b) = match order {
            1 => (vec![1.0], vec![1.0]),
            2 => (vec![0.5, 0.5], vec![0.0, 1.0]),
            3 => (
                vec![2.0 / 3.0, -2.0 / 3.0, 1.0],
                vec![7.0 / 24.0, 0.75, -1.0 / 24.0],
            ),
            4 => {
                let cbrt2 = 2.0_f64.powf(1.0 / 3.0);
                let a0 = (2.0 + cbrt2 + cbrt2.recip()) / 6.0;
                let a1 = (1.0 - cbrt2 - cbrt2.recip()) / 6.0;
                (
                    vec![a0, a1, a1, a0],
                    vec![
                        0.0,
                        1.0 / (2.0 - cbrt2),
                        1.0 / (1.0 - cbrt2 * cbrt2),
                        1.0 / (2.0 - cbrt2),
                    ],
                )
            }
            o => panic!("SIAVSolver::new: unsupported order {o} (must be 1–4)"),
        };
        Self { order, a, b }
    }

    /// Order of the integrator.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Advance the Hamiltonian system by one time step `dt`.
    ///
    /// * `sys` — the Hamiltonian system (provides `grad_q` = ∂H/∂q and `grad_p` = ∂H/∂p).
    /// * `q` — generalized coordinates (updated in place).
    /// * `p` — generalized momenta (updated in place).
    /// * `dt` — time step.
    /// * `t` — current time (unused in autonomous systems but preserved for API parity).
    pub fn step(
        &self,
        sys: &dyn HamiltonianSystem,
        q: &mut [f64],
        p: &mut [f64],
        t: f64,
        dt: f64,
    ) {
        let n = q.len();
        let mut neg_grad_q = vec![0.0_f64; n];
        let mut grad_p = vec![0.0_f64; n];

        for i in 0..self.order {
            if self.b[i] != 0.0 {
                sys.grad_q(q, p, &mut neg_grad_q);
                let bi = self.b[i] * dt;
                for k in 0..n {
                    p[k] -= bi * neg_grad_q[k];  // dp/dt = -∂H/∂q
                }
            }
            sys.grad_p(q, p, &mut grad_p);
            let ai = self.a[i] * dt;
            for k in 0..n {
                q[k] += ai * grad_p[k];  // dq/dt = ∂H/∂p
            }
        }
    }
}

/// 4th-order Yoshida symplectic integrator (a special composition of three
/// leapfrog steps).  Coefficients from Yoshida (1990).
///
/// This is a fixed-order alternative to [`SIAVSolver`] when 4th-order
/// accuracy is desired without the overhead of the variable-order dispatch.
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

    fn yoshida_stage(
        &self,
        sys: &dyn HamiltonianSystem,
        q: &mut [f64],
        p: &mut [f64],
        dt: f64,
        w: f64,
    ) {
        let n = q.len();
        let mut buf = vec![0.0_f64; n];
        let c = w * dt;

        // Drift (position update)
        sys.grad_p(q, p, &mut buf);
        for k in 0..n {
            q[k] += c * buf[k];
        }
        // Kick (momentum update)
        sys.grad_q(q, p, &mut buf);
        for k in 0..n {
            p[k] -= c * buf[k];
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

    #[test]
    fn siav_order1_energy_conservation() {
        let sys = HarmonicOscillator { m: 1.0, k: 1.0 };
        let solver = SIAVSolver::new(1);
        let mut q = vec![0.0_f64];
        let mut p = vec![1.0_f64];
        let e0 = harmonic_energy(q[0], p[0], 1.0, 1.0);
        let dt = 0.01;
        let nsteps = 1000;
        for _ in 0..nsteps {
            solver.step(&sys, &mut q, &mut p, 0.0, dt);
        }
        let e1 = harmonic_energy(q[0], p[0], 1.0, 1.0);
        let rel_err = (e1 - e0).abs() / e0;
        assert!(
            rel_err < 0.1,
            "order-1 energy drift too large: {rel_err:.3e}"
        );
    }

    #[test]
    fn siav_order2_energy_conservation() {
        let sys = HarmonicOscillator { m: 1.0, k: 1.0 };
        let solver = SIAVSolver::new(2);
        let mut q = vec![0.0_f64];
        let mut p = vec![1.0_f64];
        let e0 = harmonic_energy(q[0], p[0], 1.0, 1.0);
        let dt = 0.05;
        let nsteps = 1000;
        for _ in 0..nsteps {
            solver.step(&sys, &mut q, &mut p, 0.0, dt);
        }
        let e1 = harmonic_energy(q[0], p[0], 1.0, 1.0);
        let rel_err = (e1 - e0).abs() / e0;
        assert!(
            rel_err < 1e-3,
            "order-2 energy drift too large: {rel_err:.3e}"
        );
    }

    #[test]
    fn siav_order4_energy_conservation() {
        let sys = HarmonicOscillator { m: 1.0, k: 1.0 };
        let solver = SIAVSolver::new(4);
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
            rel_err < 1e-5,
            "order-4 energy drift too large: {rel_err:.3e}"
        );
    }

    #[test]
    fn siav_oscillator_period() {
        // Harmonic oscillator with m=1, k=1 has period 2π.
        let sys = HarmonicOscillator { m: 1.0, k: 1.0 };
        let solver = SIAVSolver::new(4);
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
    fn siav_coefficients_order1() {
        let solver = SIAVSolver::new(1);
        assert_eq!(solver.a, vec![1.0]);
        assert_eq!(solver.b, vec![1.0]);
    }

    #[test]
    fn siav_coefficients_order2() {
        let solver = SIAVSolver::new(2);
        assert_eq!(solver.a, vec![0.5, 0.5]);
        assert_eq!(solver.b, vec![0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "unsupported order")]
    fn siav_invalid_order_panics() {
        SIAVSolver::new(5);
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
