//! # Example 18 �?1-D Euler equations (baseline DG/FV hyperbolic path)
//!
//! A practical baseline matching MFEM ex18 intent:
//! - conservative variables `(rho, rho u, E)`
//! - approximate Riemann solvers (Lax-Friedrichs / Roe)
//! - explicit SSP-RK2 time stepping with CFL control
//! - periodic domain advection/acoustics smoke run

use std::f64::consts::PI;

use fem_assembly::{HyperbolicFormIntegrator, NumericalFlux};

struct RunResult {
    n: usize,
    tf: f64,
    cfl: f64,
    gamma: f64,
    rho_amplitude: f64,
    flux: NumericalFlux,
    steps: usize,
    final_time: f64,
    rho_min: f64,
    rho_max: f64,
    p_min: f64,
    mass_drift: f64,
    rho_l2_error: f64,
    velocity_l2_error: f64,
    pressure_l2_error: f64,
    density_checksum: f64,
    momentum_checksum: f64,
}

fn main() {
    let args = parse_args();
    println!("=== mfem_ex18_euler (baseline) ===");
    println!(
        "  n={}, tf={:.3}, cfl={:.3}, flux={}, amp={:.3}",
        args.n,
        args.tf,
        args.cfl,
        flux_name(args.flux),
        args.rho_amplitude,
    );

    let result = run_case(args.n, args.tf, args.cfl, args.gamma, args.flux, args.rho_amplitude);

    println!(
        "  confirmed n={}, tf={:.3}, cfl={:.3}, gamma={:.3}, flux={}, amp={:.3}",
        result.n,
        result.tf,
        result.cfl,
        result.gamma,
        flux_name(result.flux),
        result.rho_amplitude,
    );
    println!("  steps={}, t_final={:.4}", result.steps, result.final_time);
    println!(
        "  rho range: [{:.4}, {:.4}], p_min={:.4}, mass drift={:.3e}",
        result.rho_min, result.rho_max, result.p_min, result.mass_drift
    );
    println!(
        "  errors: rho={:.4e}, u={:.4e}, p={:.4e}",
        result.rho_l2_error, result.velocity_l2_error, result.pressure_l2_error
    );
    println!(
        "  checksum(rho) = {:.8e}, checksum(rho*u) = {:.8e}",
        result.density_checksum, result.momentum_checksum
    );

    assert!(result.rho_min > 0.0, "density became non-positive");
    assert!(result.p_min > 0.0, "pressure became non-positive");
    assert!(result.mass_drift < 5e-8, "mass conservation drift too large");

    println!("  PASS");
}

fn run_case(
    n: usize,
    tf: f64,
    cfl: f64,
    gamma: f64,
    flux: NumericalFlux,
    rho_amplitude: f64,
) -> RunResult {
    let integ = HyperbolicFormIntegrator { gamma, flux };

    let n = n.max(8);
    let dx = 1.0 / n as f64;
    let mut q = vec![[0.0; 3]; n];
    let base_velocity = 1.0;
    let base_pressure = 1.0;

    // Smooth periodic perturbation around a uniform moving state.
    for (i, qi) in q.iter_mut().enumerate() {
        let x = (i as f64 + 0.5) * dx;
        let rho = exact_density(x, 0.0, rho_amplitude, base_velocity);
        *qi = integ.prim_to_cons(rho, base_velocity, base_pressure);
    }

    let mass0: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
    let mut t = 0.0;
    let mut steps = 0usize;

    while t < tf {
        let smax = q
            .iter()
            .map(|qi| integ.max_wave_speed_1d(qi))
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        let mut dt = cfl * dx / smax;
        if t + dt > tf {
            dt = tf - t;
        }
        integ.step_ssprk2_periodic(&mut q, dx, dt);
        t += dt;
        steps += 1;
    }

    let mut rho_min = f64::INFINITY;
    let mut p_min = f64::INFINITY;
    let mut rho_max = f64::NEG_INFINITY;
    let mut rho_l2_error_sq = 0.0_f64;
    let mut velocity_l2_error_sq = 0.0_f64;
    let mut pressure_l2_error_sq = 0.0_f64;
    let mut density_checksum = 0.0_f64;
    let mut momentum_checksum = 0.0_f64;
    for qi in &q {
        let (rho, _u, p) = integ.cons_to_prim(qi);
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
        p_min = p_min.min(p);
    }
    for (i, qi) in q.iter().enumerate() {
        let x = (i as f64 + 0.5) * dx;
        let (rho, u, p) = integ.cons_to_prim(qi);
        let rho_exact = exact_density(x, t, rho_amplitude, base_velocity);
        rho_l2_error_sq += (rho - rho_exact).powi(2);
        velocity_l2_error_sq += (u - base_velocity).powi(2);
        pressure_l2_error_sq += (p - base_pressure).powi(2);
        density_checksum += (i as f64 + 1.0) * rho;
        momentum_checksum += (i as f64 + 1.0) * qi[1];
    }
    let mass1: f64 = q.iter().map(|qi| qi[0]).sum::<f64>() * dx;
    let mass_drift = (mass1 - mass0).abs();

    RunResult {
        n,
        tf,
        cfl,
        gamma,
        rho_amplitude,
        flux,
        steps,
        final_time: t,
        rho_min,
        rho_max,
        p_min,
        mass_drift,
        rho_l2_error: (rho_l2_error_sq / n as f64).sqrt(),
        velocity_l2_error: (velocity_l2_error_sq / n as f64).sqrt(),
        pressure_l2_error: (pressure_l2_error_sq / n as f64).sqrt(),
        density_checksum,
        momentum_checksum,
    }
}

fn exact_density(x: f64, t: f64, rho_amplitude: f64, velocity: f64) -> f64 {
    1.0 + rho_amplitude * (2.0 * PI * (x - velocity * t)).sin()
}

fn flux_name(flux: NumericalFlux) -> &'static str {
    match flux {
        NumericalFlux::LaxFriedrichs => "lax",
        NumericalFlux::Roe => "roe",
    }
}

struct Args {
    n: usize,
    tf: f64,
    cfl: f64,
    gamma: f64,
    flux: NumericalFlux,
    rho_amplitude: f64,
}

fn parse_args() -> Args {
    let mut out = Args {
        n: 200,
        tf: 0.2,
        cfl: 0.35,
        gamma: 1.4,
        flux: NumericalFlux::Roe,
        rho_amplitude: 0.2,
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
            "--rho-amplitude" | "--amp" => {
                out.rho_amplitude = it.next().unwrap_or_else(|| "0.2".into()).parse().unwrap_or(0.2);
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex18_euler_coarse_roe_case_is_positive_and_reasonably_accurate() {
        let result = run_case(50, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        assert_eq!(result.n, 50);
        assert!(result.steps > 0);
        assert!(result.rho_min > 0.75, "density floor too low: {}", result.rho_min);
        assert!(result.p_min > 0.99, "pressure floor too low: {}", result.p_min);
        assert!(result.mass_drift < 1.0e-12, "mass drift too large: {}", result.mass_drift);
        assert!(result.rho_l2_error < 2.0e-2, "rho error too large: {}", result.rho_l2_error);
        assert!(result.velocity_l2_error < 1.0e-12, "velocity should stay exact: {}", result.velocity_l2_error);
        assert!(result.pressure_l2_error < 1.0e-12, "pressure should stay exact: {}", result.pressure_l2_error);
    }

    #[test]
    fn ex18_euler_refinement_reduces_density_error() {
        let coarse = run_case(50, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        let fine = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        assert!(fine.rho_l2_error < coarse.rho_l2_error,
            "refinement should reduce density error: coarse={} fine={}",
            coarse.rho_l2_error,
            fine.rho_l2_error);
        assert!(fine.rho_l2_error < coarse.rho_l2_error * 0.75,
            "refinement gain too small: coarse={} fine={}",
            coarse.rho_l2_error,
            fine.rho_l2_error);
    }

    #[test]
    fn ex18_euler_roe_is_less_diffusive_than_lax_for_smooth_advection() {
        let roe = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        let lax = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::LaxFriedrichs, 0.2);
        assert!(roe.rho_l2_error < lax.rho_l2_error,
            "Roe should be less diffusive than Lax on this smooth case: roe={} lax={}",
            roe.rho_l2_error,
            lax.rho_l2_error);
        let roe_span = roe.rho_max - roe.rho_min;
        let lax_span = lax.rho_max - lax.rho_min;
        assert!(roe_span > lax_span,
            "Roe should preserve a larger density span: roe={} lax={}", roe_span, lax_span);
    }

    #[test]
    fn ex18_euler_zero_perturbation_preserves_constant_state() {
        let result = run_case(64, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.0);
        assert!(result.mass_drift < 1.0e-12);
        assert!(result.rho_l2_error < 1.0e-12, "rho error should vanish: {}", result.rho_l2_error);
        assert!(result.velocity_l2_error < 1.0e-12, "velocity error should vanish: {}", result.velocity_l2_error);
        assert!(result.pressure_l2_error < 1.0e-12, "pressure error should vanish: {}", result.pressure_l2_error);
        assert!((result.rho_min - 1.0).abs() < 1.0e-12);
        assert!((result.rho_max - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn ex18_euler_density_error_and_span_scale_linearly_with_amplitude() {
        let half = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.1);
        let full = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);

        assert!(half.mass_drift < 1.0e-12 && full.mass_drift < 1.0e-12);
        assert!((full.rho_l2_error / half.rho_l2_error - 2.0).abs() < 5.0e-3,
            "density error should scale linearly with amplitude: half={} full={}",
            half.rho_l2_error, full.rho_l2_error);

        let half_span = half.rho_max - half.rho_min;
        let full_span = full.rho_max - full.rho_min;
        assert!((full_span / half_span - 2.0).abs() < 5.0e-3,
            "density span should scale linearly with amplitude: half={} full={}",
            half_span, full_span);
    }

    #[test]
    fn ex18_euler_longer_advection_accumulates_more_error_and_diffusion() {
        let short = run_case(100, 0.1, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        let medium = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        let long = run_case(100, 0.4, 0.35, 1.4, NumericalFlux::Roe, 0.2);

        assert!(short.steps < medium.steps && medium.steps < long.steps,
            "expected step counts to grow with final time: short={} medium={} long={}",
            short.steps, medium.steps, long.steps);
        assert!(short.rho_l2_error < medium.rho_l2_error && medium.rho_l2_error < long.rho_l2_error,
            "density error should accumulate over longer advection: short={} medium={} long={}",
            short.rho_l2_error, medium.rho_l2_error, long.rho_l2_error);

        let short_span = short.rho_max - short.rho_min;
        let medium_span = medium.rho_max - medium.rho_min;
        let long_span = long.rho_max - long.rho_min;
        assert!(short_span > medium_span && medium_span > long_span,
            "numerical diffusion should shrink the density span over time: short={} medium={} long={}",
            short_span, medium_span, long_span);
    }

    #[test]
    fn ex18_euler_unit_background_velocity_keeps_density_and_momentum_checksums_equal() {
        let roe = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        let lax = run_case(100, 0.2, 0.35, 1.4, NumericalFlux::LaxFriedrichs, 0.2);

        assert!((roe.density_checksum - roe.momentum_checksum).abs() < 1.0e-10,
            "Roe density/momentum checksum mismatch: rho={} mom={}",
            roe.density_checksum, roe.momentum_checksum);
        assert!((lax.density_checksum - lax.momentum_checksum).abs() < 1.0e-10,
            "Lax density/momentum checksum mismatch: rho={} mom={}",
            lax.density_checksum, lax.momentum_checksum);
    }

    /// Identical inputs must produce identical density checksums (determinism).
    #[test]
    fn ex18_euler_density_checksum_is_deterministic() {
        let r1 = run_case(50, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        let r2 = run_case(50, 0.2, 0.35, 1.4, NumericalFlux::Roe, 0.2);
        assert_eq!(r1.density_checksum, r2.density_checksum,
            "density checksum not deterministic: run1={:.8e} run2={:.8e}",
            r1.density_checksum, r2.density_checksum);
        assert_eq!(r1.momentum_checksum, r2.momentum_checksum,
            "momentum checksum not deterministic: run1={:.8e} run2={:.8e}",
            r1.momentum_checksum, r2.momentum_checksum);
    }
}

