//! # Parallel Example 20 — Symplectic Integration  [1:1 translation of MFEM ex20p]
//!
//! Evolves a 1-D Hamiltonian system with the variable-order symplectic
//! integrator `SIAVSolver` (orders 1–4).  In parallel each rank evolves the
//! SAME Hamiltonian but from a rank-dependent initial condition
//! `(q₀, p₀) = (sin(2π·rank/np), cos(2π·rank/np))`; there is no mesh and no
//! linear solve — the only communication is the final MPI_Gather of each
//! rank's energy mean / standard deviation to rank 0.
//!
//! Hamiltonians (energy shifted to stay positive):
//! ```text
//!   0: H = 1 − 1/(2m) + p²/(2m) + k·q²/2                         (oscillator)
//!   1: H = ... + k·(1 − cos q)                                    (pendulum)
//!   2: H = ... + k·(1 − exp(−q²/2))                               (gaussian well)
//!   3: H = ... + k·(1+q²)·q²/2                                    (quartic)
//!   4: H = ... + k·(1 − q²/8)·q²/2                                (negative quartic)
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex20_parallel_symplectic -- --ranks 1 -no-vis -no-gp
//! cargo run --release --example mfem_pex20_parallel_symplectic -- --ranks 4 -p 1 -o 2 -n 120 -dt 0.1 -no-vis
//! ```

#![allow(non_snake_case)]

use std::f64::consts::PI;

use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    order: usize,
    prob: usize,
    nsteps: usize,
    dt: f64,
    mass: f64,
    spring: f64,
    ranks: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        order: 1,
        prob: 0,
        nsteps: 100,
        dt: 0.1,
        mass: 1.0,
        spring: 1.0,
        ranks: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!(
                    "Usage: ex20p [-o order] [-p prob] [-n steps] [-dt dt] [-m mass] [-k spring] [--ranks n] [-no-vis]"
                );
                std::process::exit(0);
            }
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p" | "--problem-type" => a.prob = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-n" | "--number-of-steps" => {
                a.nsteps = it.next().and_then(|v| v.parse().ok()).unwrap_or(100)
            }
            "-dt" | "--time-step" => a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.1),
            "-m" | "--mass" => a.mass = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            "-k" | "--spring-const" => {
                a.spring = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            }
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization"
            | "-gp" | "--gnuplot" | "-no-gp" | "--no-gnuplot" => {}
            _ => {}
        }
    }
    a
}

// ─── SIAVSolver tableau (MFEM linalg/ode.cpp) ────────────────────────────────

/// Variable-order symplectic integrator coefficients (orders 1–4).
fn siav_tableau(order: usize) -> (Vec<f64>, Vec<f64>) {
    match order {
        1 => (vec![1.0], vec![1.0]),
        2 => (vec![0.5, 0.5], vec![0.0, 1.0]),
        3 => (
            vec![2.0 / 3.0, -2.0 / 3.0, 1.0],
            vec![7.0 / 24.0, 0.75, -1.0 / 24.0],
        ),
        4 => {
            let cbrt2 = 2f64.powf(1.0 / 3.0);
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
        o => panic!("Unsupported order in SIAVSolver: {o}"),
    }
}

// ─── Hamiltonian and force (matching C++ ex20p) ──────────────────────────────

fn hamiltonian(q: f64, p: f64, mass: f64, spring: f64, prob: usize) -> f64 {
    let mut h = 1.0 - 0.5 / mass + 0.5 * p * p / mass;
    match prob {
        1 => h += spring * (1.0 - q.cos()),
        2 => h += spring * (1.0 - (-0.5 * q * q).exp()),
        3 => h += 0.5 * spring * (1.0 + q * q) * q * q,
        4 => h += 0.5 * spring * (1.0 - 0.125 * q * q) * q * q,
        _ => h += 0.5 * spring * q * q,
    }
    h
}

/// dq/dt = dH/dp = p/m  (GradT::Mult).
fn grad_t(p: f64, mass: f64) -> f64 {
    p / mass
}

/// dp/dt = −dH/dq  (NegGradV::Mult).
fn neg_grad_v(q: f64, spring: f64, prob: usize) -> f64 {
    match prob {
        1 => -spring * q.sin(),
        2 => -spring * q * (-0.5 * q * q).exp(),
        3 => -spring * (1.0 + 2.0 * q * q) * q,
        4 => -spring * (1.0 - 0.25 * q * q) * q,
        _ => -spring * q,
    }
}

// ─── C++ `std::cout` default-format printing (precision 6) ──────────────────

fn cpp_6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let e = x.abs().log10().floor() as i32;
    let s = if e >= -4 && e < 6 {
        let dec = (5 - e).max(0) as usize;
        format!("{:.*}", dec, x)
    } else {
        let s = format!("{:.5e}", x);
        let mut it = s.split('e');
        let mant = it.next().unwrap().to_string();
        let exp: i32 = it.next().unwrap().parse().unwrap();
        // C++ prints the exponent zero-padded after the sign (e.g. `e-05`).
        format!("{}e{}{:02}", mant, if exp < 0 { "-" } else { "+" }, exp.abs())
    };
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        let t = t.trim_end_matches('.');
        if t.is_empty() || t == "-" {
            s
        } else {
            t.to_string()
        }
    } else {
        s
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    assert!(args.prob <= 4, "problem type must be 0..4");

    let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let num_procs = comm.size();

        // Initial conditions depend on the rank.
        let mut q = (2.0 * PI * rank as f64 / num_procs as f64).sin();
        let mut p = (2.0 * PI * rank as f64 / num_procs as f64).cos();

        let (a, b) = siav_tableau(args.order);

        let mut e = Vec::with_capacity(args.nsteps + 1);
        e.push(hamiltonian(q, p, args.mass, args.spring, args.prob));

        // Time stepping (SIAVSolver::Step).
        for _ in 0..args.nsteps {
            for i in 0..args.order {
                if b[i] != 0.0 {
                    let dp = neg_grad_v(q, args.spring, args.prob);
                    p += b[i] * args.dt * dp;
                }
                let dq = grad_t(p, args.mass);
                q += a[i] * args.dt * dq;
            }
            e.push(hamiltonian(q, p, args.mass, args.spring, args.prob));
        }

        // Mean and standard deviation of the energy.
        let n = (args.nsteps + 1) as f64;
        let e_mean = e.iter().sum::<f64>() / n;
        let e_var = e.iter().map(|&x| (x - e_mean).powi(2)).sum::<f64>() / n;
        let e_sd = e_var.sqrt();

        // MPI_Gather of (mean, sd) to rank 0.
        let stats: Vec<f64> = if rank == 0 {
            let mut all = vec![e_mean, e_sd];
            for src in 1..num_procs as i32 {
                let s: Vec<f64> = comm.recv(src, 401);
                all.extend(s);
            }
            all
        } else {
            comm.send(0, 401, &[e_mean, e_sd]);
            Vec::new()
        };

        if rank == 0 {
            println!("\nMean and standard deviation of the energy for different initial conditions");
            for i in 0..num_procs {
                println!("{}: {}\t{}", i, cpp_6(stats[2 * i]), cpp_6(stats[2 * i + 1]));
            }
        }
    });
}
