//! CFD Example 1: 2D steady lid-driven cavity flow (Re=100).
//!
//! Solves incompressible Navier-Stokes in a unit square with moving top lid.
//! Uses Taylor-Hood P2/P1 elements.
//!
//! ```text
//! cargo run --example cfd_cavity --release
//! cargo run --example cfd_cavity --release -- --refine 3 --Re 400
//! ```

use fem_assembly::physics::fluid_cfd::{NavierStokesProblem, NsConfig};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::SolverConfig;

struct Args {
    refine: usize,
    re: f64,
    omega: f64,
}

fn parse_args() -> Args {
    let mut a = Args { refine: 2, re: 100.0, omega: 0.7 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--refine" | "-r" => a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "--Re" | "--re" => a.re = it.next().and_then(|v| v.parse().ok()).unwrap_or(100.0),
            "--omega" => a.omega = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.7),
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    println!("=== CFD: 2D Lid-Driven Cavity ===");
    println!("  Re = {:.0}, refinements = {}", args.re, args.refine);

    let mesh = (0..args.refine)
        .fold(Mesh::<2>::make_cartesian_2d(8, 8, 1.0, 1.0), |m, _| refine_uniform(&m));
    println!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    let config = NsConfig {
        nu: 1.0 / args.re,
        rho: 1.0,
        quad_order: 4,
        nl_tol: 1e-6,
        nl_max_iter: 30,
        lin_rtol: 1e-8,
        omega: args.omega,
    };

    let ns = NavierStokesProblem::new(mesh, 2, 1, config);
    let n_vel = ns.n_vel();
    let n_pres = ns.n_pres();
    println!("  DOFs: vel={}, pres={}", n_vel, n_pres);

    let u = vec![0.0_f64; n_vel];
    let p = vec![0.0_f64; n_pres];

    let lin_cfg = SolverConfig {
        rtol: 1e-8, max_iter: 1000, verbose: false,
        ..SolverConfig::default()
    };

    let (u, _p) = ns.solve_steady(&u, &p, &lin_cfg)
        .expect("NS solve failed");

    let ke: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
    println!("  ||u|| = {:.6e}", ke);
    println!("Done.");
}
