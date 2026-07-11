//! MFEM Example 18 — 2D Euler equations (DG, SSP-RK3) [1:1 translation]
//!
//! Solves compressible Euler equations on a periodic square:
//!   ∂u/∂t + ∇·F(u) = 0
//! using DG(P₁) discretization with Rusanov/Lax-Friedrichs flux and SSP-RK3.
//!
//! Problems:
//!   1: Fast moving isentropic vortex (Minf=0.5, β=1/5)
//!   3: Moving sine wave (density perturbation)
//!
//! Usage:
//! ```bash
//! cargo run --example mfem_ex18_euler -- -p 1 -r 1 -tf 2.0
//! ```
//!
//! ## Reference
//! MFEM ex18.cpp + ex18.hpp — DG hyperbolic conservation laws.

use fem_assembly::dg::dg_euler_2d::DgEuler2D;
use fem_io::mfem::read_mfem_file;
use fem_mesh::refine_uniform;

fn main() {
    let a = parse_args();
    println!("=== MFEM Example 18: 2D Euler (DG, RK3) ===");

    // 2. Read periodic square mesh
    let default_mesh = {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        p.parent().unwrap().join("data/periodic-square.mesh").to_string_lossy().to_string()
    };
    let mfem = read_mfem_file(a.mesh.as_deref().unwrap_or(&default_mesh)).expect("mesh");
    let mesh = mfem.mesh2d.expect("2D mesh");
    let mesh = if a.r > 0 { let mut m = mesh; for _ in 0..a.r { m = refine_uniform(&m); } m } else { mesh };

    // 4. DG solver (matching C++ DGHyperbolicConservationLaws)
    let order = 2;
    let mut dg = DgEuler2D::with_order(mesh, order);
    dg.periodic = true;
    dg.use_limiter = true;
    let neq = dg.n_dofs();
    println!("Number of unknowns: {neq}");
    println!("  problem: {}, order: {} (P{}), ref_levels: {}", a.p, order, order, a.r);
    println!("  t_final: {:.3}, cfl: {:.3}", a.tf, a.cfl);

    // 5. Initial condition and L2 reference (u₀ for error check)
    let init = |x: f64, y: f64| -> (f64, f64, f64, f64) {
        match a.p {
            1 | 2 => moving_vortex(x, y),
            3 => sine_wave(x, y),
            _ => (1.0, 1.0, 0.0, 1.0),
        }
    };
    let mut u = dg.project_initial(&init);
    let u0 = u.clone();

    // 6. CFL-based time step
    let h_min = dg.h_min();
    let maxcs = 2.5;
    let dt = a.cfl * h_min / maxcs / (2.0 * order as f64 + 1.0);
    println!("  h_min: {h_min:.4e}, dt: {dt:.4e}");

    // 8. SSP-RK3 time integration (matching ex18's ODE solver)
    let n_steps = (a.tf / dt).ceil() as usize;
    let mut t = 0.0;
    for ti in 1..=n_steps {
        let dta = dt.min(a.tf - t);
        dg.step_rk3(&mut u, dta);
        t += dta;
        if ti % 50 == 0 || ti == n_steps {
            println!("  step {ti:6}, t = {t:.6e}");
        }
    }

    // 9. L² error vs initial condition (matching ex18's error computation)
    let err = dg.compute_error(&u, &u0);
    println!("\n  Solution (L²) error: {err:.6e}");
    println!("Done.");
}

// ─── Problem 1 & 2: moving isentropic vortex ─────────────────────────────
fn moving_vortex(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let gamma = 1.4;
    let rad = 0.2;
    let minf = 0.5;
    let beta = 1.0 / 5.0;
    let vinf = 1.0;
    let dinf = 1.0;
    let pinf = dinf / gamma * (vinf / minf) * (vinf / minf);
    let tinf = pinf / dinf;
    let e = (-0.5 * (x * x + y * y) / (rad * rad)).exp();
    let vx = vinf * (1.0 - beta * y / rad * e);
    let vy = vinf * beta * x / rad * e;
    let si = 1.0 / (gamma - 1.0);
    let cp = gamma * si;
    let temp = tinf - 0.5 * (vinf * beta) * (vinf * beta) / cp * e;
    let dens = dinf * (temp / tinf).powf(si);
    let pres = dens * temp;
    (dens, vx, vy, pres)
}

// ─── Problem 3: moving sine wave ─────────────────────────────────────────
fn sine_wave(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let rho = 1.0 + 0.2 * (std::f64::consts::PI * (x + y)).sin();
    (rho, 0.7, 0.3, 1.0)
}

// ─── CLI ─────────────────────────────────────────────────────────────────
struct Args { mesh: Option<String>, p: i32, r: usize, tf: f64, cfl: f64 }
fn parse_args() -> Args {
    let mut a = Args { mesh: None, p: 1, r: 1, tf: 2.0, cfl: 0.3 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = it.next(),
            "-p"|"--problem" => { a.p = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-r"|"--refine" => { a.r = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-tf"|"--t-final" => { a.tf = it.next().and_then(|v| v.parse().ok()).unwrap_or(2.0); }
            "-c"|"--cfl-number" => { a.cfl = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3); }
            _ => {}
        }
    }
    a
}
