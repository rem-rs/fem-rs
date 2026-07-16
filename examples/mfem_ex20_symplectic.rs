//! Example 20 — Symplectic integration of Hamiltonian systems
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
//!   0 — Simple Harmonic Oscillator   H = (p^2/m + k·q^2)/2
//!   1 — Pendulum                     H = (p^2/m + k·(1-cos(q)))/2
//!   2 — Gaussian Potential Well      H = (p^2/m - k·exp(-q^2/2))/2
//!   3 — Quartic Potential            H = (p^2/m + k·(1+q^2)·q^2)/2
//!   4 — Negative Quartic Potential   H = (p^2/m + k·(1-q^2/8)·q^2)/2
//!
//! Usage:
//!   cargo run --example mfem_ex20_symplectic
//!   cargo run --example mfem_ex20_symplectic -- -o 2 -n 200 -dt 0.05 -p 1
//!   cargo run --example mfem_ex20_symplectic -- -p 1 -o 1 -n 120 -dt 0.1

use fem_solver::{HamiltonianSystem, SIAVSolver};
use std::fs::File;
use std::io::Write;

// ─── Hamiltonian ───────────────────────────────────────────────────────

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
            1 => self.k * q0.sin(),                             // pendulum
            2 => -self.k * q0 * (-0.5 * q0 * q0).exp(),         // Gaussian
            3 => self.k * (1.0 + 2.0 * q0 * q0) * q0,           // quartic
            4 => self.k * (1.0 - 0.25 * q0 * q0) * q0,          // negative quartic
            _ => self.k * q0,                                    // harmonic
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
        1 => k * (1.0 - q.cos()),                                // pendulum
        2 => k * (1.0 - (-0.5 * q * q).exp()),                   // Gaussian
        3 => 0.5 * k * (1.0 + q * q) * q * q,                    // quartic
        4 => 0.5 * k * (1.0 - 0.125 * q * q) * q * q,            // negative quartic
        _ => 0.5 * k * q * q,                                     // harmonic
    };
    h
}

// ─── Main ────────────────────────────────────────────────────────────

fn main() {
    // 1. Parse command-line options (matching MFEM ex20)
    let mut order = 1i32;
    let mut prob = 0i32;
    let mut nsteps = 100usize;
    let mut dt = 0.1f64;
    let mut m = 1.0f64;
    let mut k = 1.0f64;
    let mut visualization = false;
    let mut gnuplot = false;

    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex20 [-o <order=1..4>] [-p <problem=0..4>] [-n <steps>] [-dt <dt>] [-m <mass>] [-k <spring>] [-vis/-no-vis] [-gp/-no-gp]");
                return;
            }
            "-o" | "--order" => order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p" | "--problem-type" => prob = i.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-n" | "--number-of-steps" => nsteps = i.next().and_then(|v| v.parse().ok()).unwrap_or(100),
            "-dt" | "--time-step" => dt = i.next().and_then(|v| v.parse().ok()).unwrap_or(0.1),
            "-m" | "--mass" => m = i.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-k" | "--spring-const" => k = i.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-gp" | "--gnuplot" => gnuplot = true,
            "-no-gp" | "--no-gnuplot" => gnuplot = false,
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
    if visualization {
        println!("   --visualization (GLVis) — not available on this platform");
    }

    // 2. Create the symplectic integrator
    let sys = Hamiltonian { prob, m, k };
    let solver = SIAVSolver::new(order);

    // 3. Set the initial conditions
    let mut t = 0.0f64;
    let mut q = vec![0.0f64];
    let mut p = vec![1.0f64];
    let n_en = nsteps + 1;
    let mut e = vec![0.0f64; n_en];

    // 4. Open GnuPlot output file if requested
    let mut gpf: Option<File> = if gnuplot {
        Some(File::create("ex20.dat").expect("cannot create ex20.dat"))
    } else {
        None
    };
    if let Some(ref mut f) = gpf {
        writeln!(f, "{t}\t{}\t{}", q[0], p[0]).ok();
    }

    // 5. Perform time-stepping
    let mut e_mean = 0.0f64;
    for i in 0..nsteps {
        // 5a. Record initial state
        if i == 0 {
            e[0] = hamiltonian(prob, m, k, q[0], p[0]);
            e_mean += e[0];
        }

        // 5b. Advance the state
        solver.step(&sys, &mut q, &mut p, dt);
        t += dt;

        // 5c. Record the state
        e[i + 1] = hamiltonian(prob, m, k, q[0], p[0]);
        e_mean += e[i + 1];

        // 5d. GnuPlot output
        if let Some(ref mut f) = gpf {
            writeln!(f, "{t}\t{}\t{}\t{}", q[0], p[0], e[i + 1]).ok();
        }
    }

    // Finalize GnuPlot
    if let Some(_) = gpf {
        if gnuplot {
            let mut inp = File::create("gnuplot_ex20.inp").expect("cannot create gnuplot_ex20.inp");
            writeln!(inp, "plot 'ex20.dat' using 1:2 w l t 'q', \\").ok();
            writeln!(inp, "     'ex20.dat' using 1:3 w l t 'p', \\").ok();
            writeln!(inp, "     'ex20.dat' using 1:4 w l t 'H'").ok();
        }
    }

    // 6. Compute and display mean and standard deviation of the energy
    e_mean /= n_en as f64;
    let e_var: f64 = e.iter().map(|&v| (v - e_mean).powi(2)).sum::<f64>() / n_en as f64;
    let e_sd = e_var.sqrt();

    println!();
    println!("Mean and standard deviation of the energy");
    println!("{e_mean}\t{e_sd}");
}
