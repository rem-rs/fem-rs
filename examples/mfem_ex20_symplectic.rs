//! Example 20 — Symplectic integration for Hamiltonian systems
//! (analogous to MFEM ex20.cpp)
//!
//! 5 Hamiltonian types:
//!   0 — Simple Harmonic Oscillator (mass on a spring)
//!   1 — Pendulum
//!   2 — Gaussian Potential Well
//!   3 — Quartic Potential
//!   4 — Negative Quartic Potential
//!
//! Uses SIAVSolver (variable-order symplectic integrator) to preserve energy.
//!
//! Usage:
//!   cargo run --example mfem_ex20_symplectic -- [options]
//!
//! Options:
//!   --order     <int>    Time integration order (default 1, 1..=4)
//!   --problem   <int>    Hamiltonian type 0..4 (default 0)
//!   --nsteps    <int>    Number of time steps (default 100)
//!   --dt        <float>  Time step size (default 0.1)
//!   --mass      <float>  Mass m (default 1.0)
//!   --spring    <float>  Spring constant k (default 1.0)

use fem_solver::ode::{SIAVSolver, HamiltonianSystem};

// ─── Problem parameters (mutable globals, mimicking MFEM's static globals) ─────

static mut PROB: i32 = 0;
static mut MASS: f64 = 1.0;
static mut SPRING: f64 = 1.0;

// ─── Hamiltonian ──────────────────────────────────────────────────────────────

fn hamiltonian(q: f64, p: f64) -> f64 {
    unsafe {
        // Constant shift so energy stays positive (matching MFEM)
        let h = 1.0 - 0.5 / MASS + 0.5 * p * p / MASS;
        match PROB {
            1 => h + SPRING * (1.0 - f64::cos(q)),
            2 => h + SPRING * (1.0 - f64::exp(-0.5 * q * q)),
            3 => h + 0.5 * SPRING * (1.0 + q * q) * q * q,
            4 => h + 0.5 * SPRING * (1.0 - 0.125 * q * q) * q * q,
            _ => h + 0.5 * SPRING * q * q,
        }
    }
}

// ─── System implementation ────────────────────────────────────────────────────

struct Hamiltonian;

impl HamiltonianSystem for Hamiltonian {
    fn grad_q(&self, q: &[f64], _p: &[f64], out: &mut [f64]) {
        unsafe {
            match PROB {
                1 => out[0] = SPRING * f64::sin(q[0]),
                2 => out[0] = SPRING * q[0] * f64::exp(-0.5 * q[0] * q[0]),
                3 => out[0] = SPRING * (1.0 + 2.0 * q[0] * q[0]) * q[0],
                4 => out[0] = SPRING * (1.0 - 0.25 * q[0] * q[0]) * q[0],
                _ => out[0] = SPRING * q[0],
            }
        }
    }
    fn grad_p(&self, _q: &[f64], p: &[f64], out: &mut [f64]) {
        unsafe {
            out[0] = p[0] / MASS;
        }
    }
}

// ─── CLI argument parsing ─────────────────────────────────────────────────────

struct Args {
    order: i32,
    problem: i32,
    nsteps: usize,
    dt: f64,
    mass: f64,
    spring: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        order: 1,
        problem: 0,
        nsteps: 100,
        dt: 0.1,
        mass: 1.0,
        spring: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--order" => {
                a.order = it
                    .next()
                    .expect("--order needs a value")
                    .parse()
                    .expect("--order must be an integer (1..=4)")
            }
            "--problem" => {
                a.problem = it
                    .next()
                    .expect("--problem needs a value")
                    .parse()
                    .expect("--problem must be an integer (0..=4)")
            }
            "--nsteps" => {
                a.nsteps = it
                    .next()
                    .expect("--nsteps needs a value")
                    .parse()
                    .expect("--nsteps must be an integer")
            }
            "--dt" => {
                a.dt = it
                    .next()
                    .expect("--dt needs a value")
                    .parse()
                    .expect("--dt must be a float")
            }
            "--mass" => {
                a.mass = it
                    .next()
                    .expect("--mass needs a value")
                    .parse()
                    .expect("--mass must be a float")
            }
            "--spring" => {
                a.spring = it
                    .next()
                    .expect("--spring needs a value")
                    .parse()
                    .expect("--spring must be a float")
            }
            other => eprintln!("WARNING: ignoring unknown argument {other}"),
        }
    }
    a
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    assert!(
        (0..=4).contains(&args.problem),
        "problem must be 0..=4, got {}",
        args.problem
    );
    assert!(
        (1..=4).contains(&args.order),
        "order must be 1..=4, got {}",
        args.order
    );

    // Set global parameters
    unsafe {
        PROB = args.problem;
        MASS = args.mass;
        SPRING = args.spring;
    }

    // System and solver
    let sys = Hamiltonian;
    let solver = SIAVSolver::new(args.order);

    // Initial conditions (matching MFEM ex20: q=0, p=1)
    let mut q = vec![0.0_f64];
    let mut p = vec![1.0_f64];

    // Energy history
    let mut energies = vec![0.0_f64; args.nsteps + 1];

    // Time-stepping
    for i in 0..args.nsteps {
        if i == 0 {
            energies[0] = hamiltonian(q[0], p[0]);
        }
        solver.step(&sys, &mut q, &mut p, args.dt);
        energies[i + 1] = hamiltonian(q[0], p[0]);
    }

    // Mean and standard deviation of energy
    let n = energies.len() as f64;
    let e_mean: f64 = energies.iter().sum::<f64>() / n;
    let e_var: f64 = energies.iter().map(|e| (e - e_mean).powi(2)).sum::<f64>() / n;
    let e_sd = e_var.sqrt();

    println!("=== MFEM ex20: Symplectic integration ===");
    println!("  Problem type   = {}", args.problem);
    println!("  Order          = {}", args.order);
    println!("  Steps          = {}", args.nsteps);
    println!("  dt             = {}", args.dt);
    println!("  Mass           = {}", args.mass);
    println!("  Spring const   = {}", args.spring);
    println!("  Initial        = q=0, p=1");
    println!("  Final          = q={:.6e}, p={:.6e}", q[0], p[0]);
    println!("  H mean         = {:.10e}", e_mean);
    println!("  H stddev       = {:.10e}", e_sd);
}

#[cfg(test)]
mod tests {
    use fem_solver::ode::{SIAVSolver, HamiltonianSystem};

    struct HO;
    impl HamiltonianSystem for HO {
        fn grad_q(&self, q: &[f64], _: &[f64], o: &mut [f64]) {
            o[0] = q[0];
        }
        fn grad_p(&self, _: &[f64], p: &[f64], o: &mut [f64]) {
            o[0] = p[0];
        }
    }

    #[test]
    fn smoke() {
        // Short run with default parameters; position should move from 0
        let mut q = vec![0.0];
        let mut p = vec![1.0];
        SIAVSolver::new(1).step(&HO, &mut q, &mut p, 0.1);
        assert!(q[0] > 0.0); // with p>0 and no potential, q should increase
    }

    #[test]
    fn energy_mean_stddev_computed() {
        // Run all 5 problem types and check that energy stays bounded.
        let dt = 0.1;
        let nsteps = 100;
        for prob in 0..=4 {
            // Reset globals
            unsafe {
                super::PROB = prob;
                super::MASS = 1.0;
                super::SPRING = 1.0;
            }
            let sys = super::Hamiltonian;
            let solver = SIAVSolver::new(2);
            let mut q = vec![0.0];
            let mut p = vec![1.0];
            let mut energies = vec![0.0_f64; nsteps + 1];
            for i in 0..nsteps {
                if i == 0 {
                    energies[0] = super::hamiltonian(q[0], p[0]);
                }
                solver.step(&sys, &mut q, &mut p, dt);
                energies[i + 1] = super::hamiltonian(q[0], p[0]);
            }
            let n = energies.len() as f64;
            let mean: f64 = energies.iter().sum::<f64>() / n;
            let var: f64 = energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            // Energy should be positive (shifted Hamiltonian)
            assert!(mean > 0.0, "prob {prob}: mean energy {mean} <= 0");
            // Stddev should be small (symplectic integration preserves energy)
            assert!(sd < 2.0, "prob {prob}: stddev {sd} too large");
        }
    }
}
