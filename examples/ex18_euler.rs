//! # Example 18 — 1-D Euler equations (baseline DG/FV hyperbolic path)
//!
//! A practical baseline matching MFEM ex18 intent:
//! - conservative variables `(rho, rho u, E)`
//! - approximate Riemann solvers (Lax-Friedrichs / Roe)
//! - explicit SSP-RK2 time stepping with CFL control
//! - periodic domain advection/acoustics smoke run

use fem_assembly::{HyperbolicFormIntegrator, NumericalFlux};

fn main() {
    let args = parse_args();
    println!("=== ex18_euler (baseline) ===");
    println!(
        "  n={}, tf={:.3}, cfl={:.3}, flux={}",
        args.n,
        args.tf,
        args.cfl,
        match args.flux {
            NumericalFlux::LaxFriedrichs => "lax",
            NumericalFlux::Roe => "roe",
        }
    );

    let integ = HyperbolicFormIntegrator {
        gamma: args.gamma,
        flux: args.flux,
    };

    let n = args.n.max(8);
    let dx = 1.0 / n as f64;
    let mut q = vec![[0.0; 3]; n];

    // Smooth periodic perturbation around a uniform moving state.
    for (i, qi) in q.iter_mut().enumerate() {
        let x = (i as f64 + 0.5) * dx;
        let rho = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * x).sin();
        *qi = integ.prim_to_cons(rho, 1.0, 1.0);
    }

    let mass0: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
    let mut t = 0.0;
    let mut steps = 0usize;

    while t < args.tf {
        let smax = q
            .iter()
            .map(|qi| integ.max_wave_speed_1d(qi))
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        let mut dt = args.cfl * dx / smax;
        if t + dt > args.tf {
            dt = args.tf - t;
        }
        integ.step_ssprk2_periodic(&mut q, dx, dt);
        t += dt;
        steps += 1;
    }

    let mut rho_min = f64::INFINITY;
    let mut p_min = f64::INFINITY;
    let mut rho_max = f64::NEG_INFINITY;
    for qi in &q {
        let (rho, _u, p) = integ.cons_to_prim(qi);
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
        p_min = p_min.min(p);
    }
    let mass1: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
    let mass_drift = (mass1 - mass0).abs();

    println!("  steps={}, t_final={:.4}", steps, t);
    println!(
        "  rho range: [{:.4}, {:.4}], p_min={:.4}, mass drift={:.3e}",
        rho_min, rho_max, p_min, mass_drift
    );

    assert!(rho_min > 0.0, "density became non-positive");
    assert!(p_min > 0.0, "pressure became non-positive");
    assert!(mass_drift < 5e-8, "mass conservation drift too large");

    println!("  PASS");
}

struct Args {
    n: usize,
    tf: f64,
    cfl: f64,
    gamma: f64,
    flux: NumericalFlux,
}

fn parse_args() -> Args {
    let mut out = Args {
        n: 200,
        tf: 0.2,
        cfl: 0.35,
        gamma: 1.4,
        flux: NumericalFlux::Roe,
    };

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                out.n = it.next().unwrap_or_else(|| "200".into()).parse().unwrap_or(200);
            }
            "--tf" => {
                out.tf = it.next().unwrap_or_else(|| "0.2".into()).parse().unwrap_or(0.2);
            }
            "--cfl" => {
                out.cfl = it.next().unwrap_or_else(|| "0.35".into()).parse().unwrap_or(0.35);
            }
            "--gamma" => {
                out.gamma = it.next().unwrap_or_else(|| "1.4".into()).parse().unwrap_or(1.4);
            }
            "--flux" => {
                let f = it.next().unwrap_or_else(|| "roe".into()).to_lowercase();
                out.flux = if f == "lax" || f == "lf" {
                    NumericalFlux::LaxFriedrichs
                } else {
                    NumericalFlux::Roe
                };
            }
            _ => {}
        }
    }
    out
}
