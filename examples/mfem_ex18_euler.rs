//! # Example 18 — 2D Euler equations (DG, RK3)  (analogous to MFEM ex18)
//!
//! Solves compressible Euler on a periodic square using DG(P1) + SSP-RK3.
//! Default: problem 1 (moving vortex), periodic-square.mesh.
//!
//! Usage:
//!   cargo run --example mfem_ex18_euler -- -p 1 -r 1 -tf 0.5 -c 0.3

use fem_assembly::dg::dg_euler_2d::DgEuler2D;
use fem_io::mfem::read_mfem_file;
use fem_mesh::refine_uniform;

fn main() {
    let a = parse_args();
    println!("=== Example 18: 2D Euler (DG, RK3) ===");

    // 2. Read periodic mesh
    let default_mesh = {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        p.parent().unwrap().join("data/periodic-square.mesh").to_string_lossy().to_string()
    };
    let mfem = read_mfem_file(a.mesh.as_deref().unwrap_or(&default_mesh)).expect("mesh load failed");
    let mesh = mfem.mesh2d.expect("must be 2D");
    let mesh = if a.r > 0 { let mut m = mesh; for _ in 0..a.r { m = refine_uniform(&m); } m } else { mesh };

    // 4. DG solver with configurable order
    let order = 2; // P2 (matching MFEM ex18's default accuracy)
    let mut dg = DgEuler2D::with_order(mesh, order);
    dg.periodic = true;
    dg.use_limiter = true; // prevent negative density on coarse meshes
    println!("Number of unknowns: {}", dg.n_dofs());
    println!("  problem: {}, order: {} (P{}), ref_levels: {}", a.p, order, order, a.r);
    println!("  t_final: {:.3}, cfl: {:.3}", a.tf, a.cfl);

    // 5. Initial condition
    let init = |x: f64, y: f64| -> (f64, f64, f64, f64) {
        match a.p {
            1 => moving_vortex(x, y),
            3 => { let rho = 1.0 + 0.2 * (std::f64::consts::PI * (x + y)).sin(); (rho, 0.7, 0.3, 1.0) }
            _ => (1.0, 1.0, 0.0, 1.0)
        }
    };
    let u0 = dg.project_initial(&init);
    let mut u = u0.clone();

    // 6. CFL-based dt
    let h_min = dg.h_min();
    let maxcs = 2.5;
    let dt = a.cfl * h_min / maxcs / (2.0 * 1.0 + 1.0);
    println!("  h_min: {:.4e}, dt: {:.4e}", h_min, dt);

    // 8. SSP-RK3 time integration
    let n_steps = (a.tf / dt).ceil() as usize;
    let mut t = 0.0;
    for ti in 1..=n_steps {
        let dta = dt.min(a.tf - t);
        dg.step_rk3(&mut u, dta);
        t += dta;
        if ti % 50 == 0 || ti == n_steps { println!("  step {}, t = {:.6e}", ti, t); }
    }

    // 9. L2 error vs initial condition
    let err = dg.compute_error(&u, &u0);
    println!("\n  Solution error: {:.6e}", err);
    println!("Done.");
}

fn moving_vortex(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let (gamma, minf, beta) = (1.4, 0.5, 1.0/5.0);
    let (radius, vinf, dinf) = (0.2, 1.0, 1.0);
    let pinf = dinf / gamma * (vinf / minf) * (vinf / minf);
    let tinf = pinf / dinf;
    let e = (-0.5 * (x*x + y*y) / (radius*radius)).exp();
    let si = 1.0 / (gamma - 1.0);
    let vx = vinf * (1.0 - beta * y / radius * e);
    let vy = vinf * beta * x / radius * e;
    let cp = gamma * si;
    let t = tinf - 0.5 * (vinf * beta) * (vinf * beta) / cp * e;
    let d = dinf * (t / tinf).powf(si);
    let p = d * t;
    (d, vx, vy, p)
}

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
