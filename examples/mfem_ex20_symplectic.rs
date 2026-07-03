//! Example 20 — Symplectic integration for a harmonic oscillator
//! (analogous to MFEM ex20)
//!
//! Hamiltonian H(q,p) = p²/2 + q²/2  (unit mass, unit stiffness).
//! Uses Verlet (leapfrog) integrator to preserve energy over long times.
//!
//! Usage:
//!   cargo run --example mfem_ex20_symplectic

use fem_solver::ode::{VerletStepper, HamiltonianSystem};

struct HarmonicOscillator;

impl HamiltonianSystem for HarmonicOscillator {
    fn grad_q(&self, q: &[f64], _p: &[f64], out: &mut [f64]) {
        out[0] = q[0]; // dH/dq = q (unit stiffness)
    }
    fn grad_p(&self, _q: &[f64], p: &[f64], out: &mut [f64]) {
        out[0] = p[0]; // dH/dp = p (unit mass)
    }
}

fn energy(q: &[f64], p: &[f64]) -> f64 {
    0.5 * (p[0]*p[0] + q[0]*q[0])
}

fn main() {
    let sys = HarmonicOscillator;
    let mut q = vec![1.0_f64]; // initial position
    let mut p = vec![0.0_f64]; // initial momentum
    let dt = 0.1;
    let n_steps = 1000;
    let stepper = VerletStepper;

    let e0 = energy(&q, &p);
    for _ in 0..n_steps { stepper.step(&sys, &mut q, &mut p, dt); }
    let e1 = energy(&q, &p);
    let drift = (e1 - e0).abs();

    println!("=== ex20: Symplectic (Verlet) ===");
    println!("  Steps: {n_steps}, dt = {dt}, energy drift = {drift:.6e}");
    assert!(drift < 1e-2, "energy drift too large: {drift}");
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use fem_solver::ode::{VerletStepper, HamiltonianSystem};
    struct HO;
    impl HamiltonianSystem for HO {
        fn grad_q(&self, q: &[f64], _: &[f64], o: &mut [f64]) { o[0] = q[0]; }
        fn grad_p(&self, _: &[f64], p: &[f64], o: &mut [f64]) { o[0] = p[0]; }
    }
    #[test] fn smoke() {
        let mut q = vec![1.0]; let mut p = vec![0.0];
        VerletStepper.step(&HO, &mut q, &mut p, 0.1);
        assert!(q[0] < 1.0); // position should decrease
    }
}
