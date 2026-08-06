//! MFEM Example 18 — 2D compressible Euler equations (DG, SSP-RK3) [1:1 translation]
//!
//! Solves the system of hyperbolic conservation laws:
//!   ∂u/∂t + ∇·F(u) = 0
//!
//! using a DG discretization with Rusanov (local Lax-Friedrichs) numerical
//! flux and explicit SSP-RK3 time integration on a periodic square mesh.
//!
//! Problems:
//!   1: Fast moving isentropic vortex (Minf=0.5, beta=1/5)
//!   2: Slow moving isentropic vortex (Minf=0.05, beta=1/50)
//!   3: Moving sine wave (density perturbation)
//!
//! Usage:
//! ```bash
//! cargo run --example mfem_ex18_euler -- -p 1 -r 1 -o 1 -s 4 -no-vis
//! ```
//!
//! ## Reference
//! MFEM ex18.cpp + ex18.hpp — DG hyperbolic conservation laws.

use fem_assembly::dg::{DgHyperbolicConservationLaws, EulerFlux, RusanovFlux};
use fem_element::lagrange::tri::{TriP1, TriP2, TriP3};
use fem_element::lagrange::QuadL2GL;
use fem_element::reference::ReferenceElement;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_mesh::element_type::ElementType;
use fem_mesh::refine_uniform;
use fem_mesh::topology::MeshTopology;
use fem_solver::ode::explicit::{ForwardEuler, Rk4};
use fem_solver::ode::traits::TimeStepper;

// ─── make_ref_elem (redefined locally since the one in dg_hyperbolic is private) ─
// Must mirror dg_hyperbolic.rs::make_ref_elem: MFEM DG_FECollection(order,
// dim, BasisType::GaussLegendre) uses the Gauss-Legendre nodal basis on
// [0,1]² — NOT the equally spaced QuadQk.  Mismatched DOF nodes make the
// initial projection and error evaluation land on the wrong physical points.
fn make_ref_elem(mesh: &dyn MeshTopology, order: u8) -> Box<dyn ReferenceElement> {
    if mesh.element_type(0) == ElementType::Quad4 {
        Box::new(QuadL2GL::new(order as usize))
    } else {
        match order {
            1 => Box::new(TriP1),
            2 => Box::new(TriP2),
            3 => Box::new(TriP3),
            _ => Box::new(TriP1),
        }
    }
}

// ─── CLI Args (C++ ex18.cpp:65-83 — matching MFEM ex18 options) ──────────────

struct Args {
    mesh: String,
    problem: i32,
    refine: usize,
    order: u8,
    ode_solver: u8,
    t_final: f64,
    dt: f64,
    cfl: f64,
    visualization: bool,
    vis_steps: usize,
}

fn default_mesh_path() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    p.parent().unwrap().parent().unwrap().join("data/periodic-square.mesh")
        .to_string_lossy().to_string()
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: default_mesh_path(),
        problem: 1,
        refine: 1,
        order: 1,
        ode_solver: 4,
        t_final: 2.0,
        dt: -0.01,
        cfl: 0.3,
        visualization: true,
        vis_steps: 50,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() {
                    a.mesh = v;
                }
            }
            "-p" | "--problem" => {
                a.problem = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-r" | "--refine" => {
                a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-s" | "--ode-solver" => {
                a.ode_solver = it.next().and_then(|v| v.parse().ok()).unwrap_or(4);
            }
            "-tf" | "--t-final" => {
                a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(2.0);
            }
            "-dt" | "--time-step" => {
                a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(-0.01);
            }
            "-c" | "--cfl-number" => {
                a.cfl = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3);
            }
            "-no-vis" => {
                a.visualization = false;
            }
            _ => {}
        }
    }
    a
}

// ─── Euler initial conditions (C++ ex18.hpp: EulerInitialCondition) ──────────

fn euler_initial_condition(problem: i32, x: &[f64], gamma: f64) -> Vec<f64> {
    // Returns [rho, rho*u, rho*v, E] — conserved variables
    match problem {
        1 => {
            // Fast moving isentropic vortex: radius=0.2, Minf=0.5, beta=1/5
            let radius = 0.2;
            let minf = 0.5;
            let beta = 1.0 / 5.0;
            let vel_inf = 1.0;
            let den_inf = 1.0;
            let pres_inf = den_inf / gamma * (vel_inf / minf) * (vel_inf / minf);
            let temp_inf = pres_inf / den_inf;

            let r2 = (x[0].powi(2) + x[1].powi(2)) / (radius * radius);
            let shrinv = 1.0 / (gamma - 1.0);
            let specific_heat = gamma * shrinv;

            let vel_x = vel_inf * (1.0 - beta * x[1] / radius * (-0.5 * r2).exp());
            let vel_y = vel_inf * beta * x[0] / radius * (-0.5 * r2).exp();
            let vel2 = vel_x * vel_x + vel_y * vel_y;
            let temp = temp_inf
                - 0.5 * (vel_inf * beta).powi(2) / specific_heat * (-r2).exp();
            let den = den_inf * (temp / temp_inf).powf(shrinv);
            let pres = den * temp;
            let energy = shrinv * pres / den + 0.5 * vel2;
            vec![den, den * vel_x, den * vel_y, den * energy]
        }
        2 => {
            // Slow moving isentropic vortex: radius=0.2, Minf=0.05, beta=1/50
            let radius = 0.2;
            let minf = 0.05;
            let beta = 1.0 / 50.0;
            let vel_inf = 1.0;
            let den_inf = 1.0;
            let pres_inf = den_inf / gamma * (vel_inf / minf) * (vel_inf / minf);
            let temp_inf = pres_inf / den_inf;

            let r2 = (x[0].powi(2) + x[1].powi(2)) / (radius * radius);
            let shrinv = 1.0 / (gamma - 1.0);
            let specific_heat = gamma * shrinv;

            let vel_x = vel_inf * (1.0 - beta * x[1] / radius * (-0.5 * r2).exp());
            let vel_y = vel_inf * beta * x[0] / radius * (-0.5 * r2).exp();
            let vel2 = vel_x * vel_x + vel_y * vel_y;
            let temp = temp_inf
                - 0.5 * (vel_inf * beta).powi(2) / specific_heat * (-r2).exp();
            let den = den_inf * (temp / temp_inf).powf(shrinv);
            let pres = den * temp;
            let energy = shrinv * pres / den + 0.5 * vel2;
            vec![den, den * vel_x, den * vel_y, den * energy]
        }
        3 => {
            // Moving sine wave
            let density =
                1.0 + 0.2 * (std::f64::consts::PI * (x[0] + x[1])).sin();
            let velocity_x = 0.7;
            let velocity_y = 0.3;
            let pressure = 1.0;
            let energy = pressure / (gamma - 1.0)
                + density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);
            vec![
                density,
                density * velocity_x,
                density * velocity_y,
                energy,
            ]
        }
        _ => panic!("Problem {problem} not supported (only 1, 2, 3)"),
    }
}

// ─── compute_h_min — CFL-based dt (C++ ex18.cpp:135-140) ─────────────────────

fn compute_h_min(mesh: &dyn MeshTopology) -> f64 {
    let mut h = f64::MAX;
    for e in 0..mesh.n_elements() as u32 {
        // Per-element geometry (MFEM GetElementSize uses the element
        // transformation; for periodic meshes the folded vertices are wrong).
        let g: Vec<[f64; 2]> = mesh.geometry_nodes(e)
            .iter()
            .map(|n| {
                let c = mesh.geom_coords_of(*n);
                [c[0], c[1]]
            })
            .collect();
        let (p0, p1, p2) = (g[0], g[1], g[2]);
        let d01 = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        let d12 = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let d20 = ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2)).sqrt();
        h = h.min(d01.min(d12.min(d20)));
    }
    h
}

// ─── L^2 projection for initial condition — MFEM: GridFunction::ProjectCoefficient

fn project_initial<F: Fn(&[f64]) -> Vec<f64>>(
    u0: &F,
    mesh: &dyn MeshTopology,
    order: u8,
    n_eq: usize,
) -> Vec<f64> {
    // MFEM GridFunction::ProjectCoefficient → FiniteElement::Project:
    // evaluate u0 at each DOF node (Gauss-Legendre nodes for L2 spaces) —
    // NOT an L² projection.
    let ref_elem = make_ref_elem(mesh, order);
    let is_quad = mesh.element_nodes(0).len() == 4;
    let dp = ref_elem.n_dofs();
    let n_elems = mesh.n_elements();
    let mut u = vec![0.0; n_elems * dp * n_eq];
    let dof_ref = ref_elem.dof_coords(); // [dof][dim], [0,1]² domain

    for e in 0..n_elems {
        let pg: Vec<[f64; 2]> = mesh.geometry_nodes(e as u32)
            .iter()
            .map(|n| {
                let c = mesh.geom_coords_of(*n);
                [c[0], c[1]]
            })
            .collect();
        let base = e * dp * n_eq;
        for i in 0..dp {
            let (xi, eta) = (dof_ref[i][0], dof_ref[i][1]);
            let (px, py) = if is_quad {
                // [0,1]² Q1 mapping (QuadL2GL reference domain), matching the
                // operator's geometry — NOT the [-1,1]² bilinear form.
                let p = [pg[0], pg[1], pg[2], pg[3]];
                let n = [(1.0-xi)*(1.0-eta), xi*(1.0-eta), xi*eta, (1.0-xi)*eta];
                (n[0]*p[0][0]+n[1]*p[1][0]+n[2]*p[2][0]+n[3]*p[3][0],
                 n[0]*p[0][1]+n[1]*p[1][1]+n[2]*p[2][1]+n[3]*p[3][1])
            } else {
                let p0 = pg[0]; let p1 = pg[1]; let p2 = pg[2];
                (p0[0] + xi*(p1[0]-p0[0]) + eta*(p2[0]-p0[0]),
                 p0[1] + xi*(p1[1]-p0[1]) + eta*(p2[1]-p0[1]))
            };
            let uv = u0(&[px, py]);
            for eq in 0..n_eq {
                u[base + i * n_eq + eq] = uv[eq];
            }
        }
    }
    u
}

// ─── L^2 error computation (C++ ex18.cpp:195 — Rust adds explicit L² error) ──

fn compute_l2_error<F: Fn(&[f64]) -> Vec<f64>>(
    sol: &[f64],
    u0: &F,
    mesh: &dyn MeshTopology,
    order: u8,
    n_eq: usize,
) -> f64 {
    let ref_elem = make_ref_elem(mesh, order);
    let is_quad = mesh.element_nodes(0).len() == 4;
    let dp = ref_elem.n_dofs();
    let q_order = 2 * order + 1;
    let qr = ref_elem.quadrature(q_order);
    let mut phi = vec![0.0; dp];
    let mut err_sq = 0.0;

    for e in 0..mesh.n_elements() {
        let base = e * dp * n_eq;
        let pg: Vec<[f64; 2]> = mesh.geometry_nodes(e as u32)
            .iter()
            .map(|n| {
                let c = mesh.geom_coords_of(*n);
                [c[0], c[1]]
            })
            .collect();

        for q in 0..qr.n_points() {
            let xi = &qr.points[q];
            let (det_j, px, py) = if is_quad {
                let p = [pg[0], pg[1], pg[2], pg[3]];
                let (xs, ys) = (xi[0], xi[1]);
                // [0,1]² Q1 mapping (QuadL2GL reference domain), matching
                // the operator's geometry — NOT the [-1,1]² bilinear form.
                let dxi = [-(1.0-ys), (1.0-ys), ys, -ys];
                let deta = [-(1.0-xs), -xs, xs, (1.0-xs)];
                let j11 = dxi[0]*p[0][0]+dxi[1]*p[1][0]+dxi[2]*p[2][0]+dxi[3]*p[3][0];
                let j12 = deta[0]*p[0][0]+deta[1]*p[1][0]+deta[2]*p[2][0]+deta[3]*p[3][0];
                let j21 = dxi[0]*p[0][1]+dxi[1]*p[1][1]+dxi[2]*p[2][1]+dxi[3]*p[3][1];
                let j22 = deta[0]*p[0][1]+deta[1]*p[1][1]+deta[2]*p[2][1]+deta[3]*p[3][1];
                let det = (j11*j22 - j12*j21).abs();
                let n = [(1.0-xs)*(1.0-ys), xs*(1.0-ys), xs*ys, (1.0-xs)*ys];
                (det, n[0]*p[0][0]+n[1]*p[1][0]+n[2]*p[2][0]+n[3]*p[3][0],
                      n[0]*p[0][1]+n[1]*p[1][1]+n[2]*p[2][1]+n[3]*p[3][1])
            } else {
                let p0 = pg[0]; let p1 = pg[1]; let p2 = pg[2];
                let det = ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])).abs();
                (det, p0[0] + xi[0]*(p1[0]-p0[0]) + xi[1]*(p2[0]-p0[0]),
                      p0[1] + xi[0]*(p1[1]-p0[1]) + xi[1]*(p2[1]-p0[1]))
            };
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);

            // Interpolate sol at QP
            let mut sol_qp = vec![0.0; n_eq];
            for eq in 0..n_eq {
                for i in 0..dp {
                    sol_qp[eq] += phi[i] * sol[base + i * n_eq + eq];
                }
            }
            let exact = u0(&[px, py]);
            for eq in 0..n_eq {
                let d = sol_qp[eq] - exact[eq];
                err_sq += w * d * d;
            }
        }
    }
    err_sq.sqrt()
}

// ─── SSP-RK2 (C++ ex18.cpp: ODESolver::Select with type 2) ───────────────────

struct SspRk2;
impl TimeStepper for SspRk2 {
    fn step<F: Fn(f64, &[f64], &mut [f64])>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F) {
        let n = u.len();
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut u1 = vec![0.0; n];
        rhs(t, u, &mut k1);
        for i in 0..n {
            u1[i] = u[i] + dt * k1[i];
        }
        rhs(t + dt, &u1, &mut k2);
        for i in 0..n {
            u[i] += 0.5 * dt * (k1[i] + k2[i]);
        }
    }
}

// ─── SSP-RK3 (C++ ex18.cpp: ODESolver::Select with type 3) ───────────────────

struct SspRk3;
impl TimeStepper for SspRk3 {
    fn step<F: Fn(f64, &[f64], &mut [f64])>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F) {
        let n = u.len();
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut u1 = vec![0.0; n];
        let mut u2 = vec![0.0; n];
        rhs(t, u, &mut k1);
        for i in 0..n {
            u1[i] = u[i] + dt * k1[i];
        }
        rhs(t + dt, &u1, &mut k2);
        for i in 0..n {
            u2[i] = 0.75 * u[i] + 0.25 * (u1[i] + dt * k2[i]);
        }
        rhs(t + 0.5 * dt, &u2, &mut k3);
        for i in 0..n {
            u[i] = u[i] / 3.0 + 2.0 / 3.0 * (u2[i] + dt * k3[i]);
        }
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // C++ ex18.cpp:107-115 — 1. Read mesh (EulerMesh or from file)
    let mfem = read_mfem_file(&args.mesh).expect("Failed to read mesh");
    let mut mesh = mfem.mesh2d.expect("2D mesh required");
    // C++ ex18.cpp:117-122 — 2. Uniform refinement
    for _ in 0..args.refine {
        mesh = refine_uniform(&mesh);
    }

    let dim = 2;
    let n_eq = dim + 2; // 4 for 2D Euler
    let order = args.order;

    // C++ ex18.cpp:124-127 — 3. Build DG operator (ex18.hpp: DGBilinearForm + HyperbolicForm)
    let flux = RusanovFlux {
        inner: EulerFlux { gamma: 1.4 },
    };
    // face-only: volume term disabled (known NaN issue with quad meshes, see handover)
    let euler_op = DgHyperbolicConservationLaws::new(
        &mesh,
        order,
        Box::new(flux),
        true, // volume term on: MFEM HyperbolicFormIntegrator (∫F·∇v)
    );
    let n_dofs = euler_op.n_dofs();
    println!("Number of unknowns: {}", n_dofs);
    println!(
        "Problem {}: {}",
        args.problem,
        match args.problem {
            1 => "Fast vortex",
            2 => "Slow vortex",
            3 => "Sine wave",
            _ => "Unknown",
        }
    );
    println!("Order: {} (P{})", order, order);
    println!("Refinement levels: {}", args.refine);
    println!("t_final: {:.3}", args.t_final);

    // Project initial condition
    let u0 = |x: &[f64]| euler_initial_condition(args.problem, x, 1.4);
    let mut sol = project_initial(&u0, &mesh, order, n_eq);

    // MFEM: mesh->Print(ofs)
    write_mfem_file("euler-mesh.mesh", &mesh)
        .expect("cannot write euler-mesh.mesh");
    // Write individual equation components in MFEM FiniteElementSpace format
    let dp = make_ref_elem(&mesh, order).n_dofs();
    let n_elems = mesh.n_elements();
    for eq in 0..n_eq {
        let mut comp = vec![0.0; n_elems * dp];
        for e in 0..n_elems {
            let base = e * dp * n_eq;
            for i in 0..dp {
                comp[e * dp + i] = sol[base + i * n_eq + eq];
            }
        }
        write_mfem_gf_file(&format!("euler-{}-init.gf", eq), 2, &comp, "H1", order, 1, 15)
            .expect("cannot write init gf");
    }
    println!("  --> saved initial mesh + solution files");

    // CFL or fixed dt
    let h_min = compute_h_min(&mesh);
    let mut dt = if args.dt > 0.0 {
        args.dt
    } else {
        let mut tmp = vec![0.0; n_dofs];
        euler_op.mult(&sol, &mut tmp);
        let max_cs = euler_op.max_char_speed();
        let cfl_dt = args.cfl * h_min / max_cs / (2.0 * order as f64 + 1.0);
        println!(
            "  h_min: {:.6e}, max_char_speed: {:.6e}, dt: {:.6e}",
            h_min, max_cs, cfl_dt
        );
        cfl_dt
    };
    println!("  initial dt: {:.6e}", dt);

    // C++ ex18.cpp:238-256 — 4. Time integration.  With CFL the time step is
    // re-computed after every step from the maximum characteristic speed
    // (ex18.cpp:249-252) — a fixed dt can de-stabilise RK4 on periodic meshes.
    let mut t = 0.0;
    let mut ti = 0usize;
    loop {
        let dta = dt.min(args.t_final - t);
        let rhs = |_: f64, u: &[f64], dudt: &mut [f64]| euler_op.mult(u, dudt);
        match args.ode_solver {
            1 => ForwardEuler.step(t, dta, &mut sol, &rhs),
            2 => SspRk2.step(t, dta, &mut sol, &rhs),
            3 => SspRk3.step(t, dta, &mut sol, &rhs),
            _ => Rk4.step(t, dta, &mut sol, &rhs),
        }
        t += dta;
        ti += 1;
        if args.dt <= 0.0 {
            // CFL: update dt from the latest max characteristic speed
            let max_cs = euler_op.max_char_speed();
            dt = args.cfl * h_min / max_cs / (2.0 * order as f64 + 1.0);
        }
        if t >= args.t_final - 1e-8 * dt {
            break;
        }
        if ti % args.vis_steps == 0 {
            println!("time step: {:6}, time: {:.6e}", ti, t);
        }
    }
    println!("time step: {:6}, time: {:.6e}", ti, t);

    // MFEM: mesh->Print(ofs)
    write_mfem_file("euler-mesh-final.mesh", &mesh)
        .expect("cannot write euler-mesh-final.mesh");
    let dp = make_ref_elem(&mesh, order).n_dofs();
    let n_elems = mesh.n_elements();
    for eq in 0..n_eq {
        let mut comp = vec![0.0; n_elems * dp];
        for e in 0..n_elems {
            let base = e * dp * n_eq;
            for i in 0..dp {
                comp[e * dp + i] = sol[base + i * n_eq + eq];
            }
        }
        write_mfem_gf_file(&format!("euler-{}-final.gf", eq), 2, &comp, "H1", order, 1, 15)
            .expect("cannot write final gf");
    }
    println!("  --> saved final mesh + solution files");

    // C++ ex18.cpp:195 — Compute L² error (Rust adds explicit error computation)
    let error = compute_l2_error(&sol, &u0, &mesh, order, n_eq);
    println!("Solution error: {:.15e}", error);
}
