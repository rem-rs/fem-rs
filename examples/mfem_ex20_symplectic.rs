//! Example 20 鈥?Symplectic integration of Hamiltonian systems
//! **1:1 translation of MFEM ex20**
//!
//! Demonstrates variable-order symplectic ODE integration for 1D Hamiltonian
//! systems.  Hamiltonian: H(q,p,t) = T(p) + V(q,t)
//!
//! Hamilton's equations:
//!   dq/dt =  dH/dp
//!   dp/dt = -dH/dq
//!
//! Problems (selected with -p):
//!   0 鈥?Simple Harmonic Oscillator   H = (p^2/m + k路q^2)/2
//!   1 鈥?Pendulum                     H = (p^2/m + k路(1-cos(q)))/2
//!   2 鈥?Gaussian Potential Well      H = (p^2/m - k路exp(-q^2/2))/2
//!   3 鈥?Quartic Potential            H = (p^2/m + k路(1+q^2)路q^2)/2
//!   4 鈥?Negative Quartic Potential   H = (p^2/m + k路(1-q^2/8)路q^2)/2
//!
//! Usage:
//!   cargo run --example mfem_ex20_symplectic
//!   cargo run --example mfem_ex20_symplectic -- -o 2 -n 200 -dt 0.05 -p 1

use fem_solver::{HamiltonianSystem, SIAVSolver};

// 鈹€鈹€ Hamiltonian 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Parameters for the 1D Hamiltonian system.
struct Hamiltonian {
    prob: i32,   // problem type 0-4
    m: f64,      // mass
    k: f64,      // spring constant / potential strength
}

impl HamiltonianSystem for Hamiltonian {
    /// Compute dH/dq  (the force / gradient of potential).
    fn grad_q(&self, q: &[f64], _p: &[f64], out: &mut [f64]) {
        let q0 = q[0];
        out[0] = match self.prob {
            1 => self.k * q0.sin(),                              // pendulum
            2 => -self.k * q0 * (-0.5 * q0 * q0).exp(),          // Gaussian
            3 => self.k * (1.0 + 2.0 * q0 * q0) * q0,           // quartic
            4 => self.k * (1.0 - 0.25 * q0 * q0) * q0,          // negative quartic
            _ => self.k * q0,                                     // harmonic
        };
    }

    /// Compute dH/dp  (the velocity / derivative of kinetic energy).
    fn grad_p(&self, _q: &[f64], p: &[f64], out: &mut [f64]) {
        out[0] = p[0] / self.m;
    }
}

/// Hamiltonian energy H(q,p,t).
fn hamiltonian(prob: i32, m: f64, k: f64, q: f64, p: f64) -> f64 {
    let mut h = 1.0 - 0.5 / m + 0.5 * p * p / m;
    h += match prob {
        1 => k * (1.0 - q.cos()),                                 // pendulum
        2 => k * (1.0 - (-0.5 * q * q).exp()),                     // Gaussian
        3 => 0.5 * k * (1.0 + q * q) * q * q,                     // quartic
        4 => 0.5 * k * (1.0 - 0.125 * q * q) * q * q,             // negative quartic
        _ => 0.5 * k * q * q,                                       // harmonic
    };
    h
}

// 鈹€鈹€ Main 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

fn main() {
    // 1. Parse command-line options
    let mut order = 1i32;
    let mut prob = 0i32;
    let mut nsteps = 100usize;
    let mut dt = 0.1f64;
    let mut m = 1.0f64;
    let mut k = 1.0f64;

    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex20 [-o <order=1..4>] [-p <problem=0..4>] [-n <steps>] [-dt <dt>] [-m <mass>] [-k <spring>]");
                return;
            }
            "-o" | "--order" => order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p" | "--problem-type" => prob = i.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-n" | "--number-of-steps" => nsteps = i.next().and_then(|v| v.parse().ok()).unwrap_or(100),
            "-dt" | "--time-step" => dt = i.next().and_then(|v| v.parse().ok()).unwrap_or(0.1),
            "-m" | "--mass" => m = i.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-k" | "--spring-const" => k = i.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            _ => {}
        }
    }

    println!("Options used:");
    println!("   --order {order}");
    println!("   --problem-type {prob}");
    println!("   --number-of-steps {nsteps}");
    println!("   --time-step {dt}");
    println!("   --mass {m}");
    println!("   --spring-const {k}");

    // 2. Create the symplectic integrator
    let sys = Hamiltonian { prob, m, k };
    let solver = SIAVSolver::new(order);

    // 3. Initial conditions
    let mut _t = 0.0f64;
    let mut q = vec![0.0f64];
    let mut p = vec![1.0f64];
    let mut e = vec![0.0f64; nsteps + 1];

    // 4. Time-stepping
    let mut e_mean = 0.0f64;
    for i in 0..nsteps {
        // Record initial state
        if i == 0 {
            e[0] = hamiltonian(prob, m, k, q[0], p[0]);
            e_mean += e[0];
        }

        // Advance the state
        solver.step(&sys, &mut q, &mut p, dt);
        _t += dt;

        // Record energy
        e[i + 1] = hamiltonian(prob, m, k, q[0], p[0]);
        e_mean += e[i + 1];
    }

    // 5. Compute mean and standard deviation of the energy
    e_mean /= (nsteps + 1) as f64;
    let e_var: f64 = e.iter().map(|&v| (v - e_mean).powi(2)).sum::<f64>() / (nsteps + 1) as f64;
    let e_sd = e_var.sqrt();

    println!();
    println!("Mean and standard deviation of the energy");
    println!("{e_mean}\t{e_sd}");
}

