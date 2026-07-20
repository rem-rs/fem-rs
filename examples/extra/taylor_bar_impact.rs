//! # Taylor bar impact benchmark (2D plane strain)
//!
//! Simulates a rectangular prism impacting a rigid frictionless wall with
//! initial velocity V₀.  Uses **finite-strain J2 plasticity** integrated with
//! **central difference explicit dynamics** — a foundation for LS-DYNA /
//! Abaqus/Explicit-class impact analysis.
//!
//! ## Physics
//! A right prism (length L, height H) moves leftward at speed V₀ toward a
//! rigid wall at x = 0.  Upon impact the bar end yields plastically and
//! "mushrooms", converting kinetic energy into plastic work.  After enough
//! time all kinetic energy is dissipated and the bar reaches a final
//! deformed shape.
//!
//! ## Reference
//! Taylor, G. I. (1948). "The use of flat-ended projectiles for determining
//! dynamic yield stress".  Proc. R. Soc. Lond. A, 194, 289–299.
//!
//! ## Usage
//! ```bash
//! # Default: steel bar L=100mm, H=20mm, V₀=200m/s
//! cargo run --example taylor_bar_impact --release -- --no-vis
//!
//! # Fine mesh, longer time
//! cargo run --example taylor_bar_impact --release -- --nx 80 --ny 20 --tf 3e-4 --no-vis
//!
//! # Custom material (OFHC copper)
//! cargo run --example taylor_bar_impact --release -- \
//!     --E 117e9 --nu 0.35 --sy 400e6 --Hhard 100e6 --rho 8960 --V 227 --no-vis
//! ```
//!
//! ## Comparison targets (LS-DYNA / Abaqus/Explicit)
//! - Final deformed length: L_final / L₀
//! - Maximum radial expansion at impact end: R_max / R₀
//! - Kinetic energy decay curve
//! - Plastic strain distribution

use fem_assembly::explicit_j2::{assemble_explicit_j2_2d, ExplicitJ2Config, ExplicitJ2QpState};
use fem_mesh::Mesh;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_solver::ode::structural::{CentralDifferenceExplicit, ExplicitState};

// ─── Command-line arguments ─────────────────────────────────────────────

struct Args {
    nx: usize,         // elements along x (length)
    ny: usize,         // elements along y (height)
    L: f64,            // bar length [m]
    H: f64,            // bar height [m]
    V0: f64,           // initial velocity [m/s] toward wall
    rho: f64,          // density [kg/m³]
    E: f64,            // Young's modulus [Pa]
    nu: f64,           // Poisson's ratio
    sigma_y: f64,      // yield stress [Pa]
    H_iso: f64,        // isotropic hardening modulus [Pa]
    t_final: f64,      // simulation time [s]
    dt: f64,           // time step (0 = auto CFL)
    vis_steps: usize,  // print every n steps
    no_vis: bool,      // suppress per-step output
    quad_order: u8,    // quadrature order
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            nx: 40,
            ny: 12,
            L: 0.1,
            H: 0.02,
            V0: 200.0,
            rho: 7800.0,
            E: 200e9,
            nu: 0.3,
            sigma_y: 250e6,
            H_iso: 500e6,
            t_final: 3e-4,
            dt: 0.0,
            vis_steps: 200,
            no_vis: false,
            quad_order: 2,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--nx" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.nx = v; }
                "--ny" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.ny = v; }
                "--L" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.L = v; }
                "--H" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.H = v; }
                "--V" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.V0 = v; }
                "--rho" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.rho = v; }
                "--E" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.E = v; }
                "--nu" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.nu = v; }
                "--sy" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.sigma_y = v; }
                "--Hhard" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.H_iso = v; }
                "--tf" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.t_final = v; }
                "--dt" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.dt = v; }
                "--vs" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.vis_steps = v; }
                "--q" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { a.quad_order = v; }
                "--no-vis" | "--no-visualization" => a.no_vis = true,
                _ => {}
            }
        }
        a
    }
}

// ─── Mesh: rectangular triangulation ────────────────────────────────────

fn make_rect_tri_mesh(nx: usize, ny: usize, L: f64, H: f64) -> Mesh<2> {
    let npx = nx + 1;
    let npy = ny + 1;
    let mut coords = Vec::with_capacity(npx * npy * 2);
    for j in 0..npy {
        for i in 0..npx {
            coords.push(i as f64 * L / nx as f64);
            coords.push(j as f64 * H / ny as f64);
        }
    }

    let nid = |i: usize, j: usize| -> u32 { (j * npx + i) as u32 };

    let mut conn = Vec::with_capacity(2 * nx * ny * 3);
    let mut elem_tags = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let n0 = nid(i, j);
            let n1 = nid(i + 1, j);
            let n2 = nid(i + 1, j + 1);
            let n3 = nid(i, j + 1);
            conn.extend_from_slice(&[n0, n1, n3]);
            elem_tags.push(1);
            conn.extend_from_slice(&[n1, n2, n3]);
            elem_tags.push(1);
        }
    }

    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    // bottom (y=0, tag=1)
    for i in 0..nx {
        face_conn.push(nid(i, 0));
        face_conn.push(nid(i + 1, 0));
        face_tags.push(1);
    }
    // right (x=L, tag=2)
    for j in 0..ny {
        face_conn.push(nid(nx, j));
        face_conn.push(nid(nx, j + 1));
        face_tags.push(2);
    }
    // top (y=H, tag=3) — reversed for outward normal
    for i in 0..nx {
        face_conn.push(nid(i + 1, ny));
        face_conn.push(nid(i, ny));
        face_tags.push(3);
    }
    // left (x=0, tag=4)
    for j in 0..ny {
        face_conn.push(nid(0, j + 1));
        face_conn.push(nid(0, j));
        face_tags.push(4);
    }

    Mesh::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        face_conn, face_tags, ElementType::Line2,
    )
}

/// Count total quadrature points in a triangular mesh.
fn count_qp_tri(mesh: &Mesh<2>, quad_order: u8) -> usize {
    use fem_element::lagrange::TriP1;
    use fem_element::ReferenceElement;
    let ref_elem = TriP1;
    let nqp = ref_elem.quadrature(quad_order).weights.len();
    mesh.n_elems() as usize * nqp
}

fn wall_contact_force(mesh: &Mesh<2>, u: &[f64], eps_n: f64, n_dofs: usize) -> Vec<f64> {
    let mut f = vec![0.0; n_dofs];
    let n_nodes = mesh.n_nodes() as usize;
    for node in 0..n_nodes {
        let x0 = mesh.node_coords(node as u32);
        let dof_x = node * 2;
        let ux = if dof_x < u.len() { u[dof_x] } else { 0.0 };
        let x_curr = x0[0] + ux;
        if x_curr < 0.0 {
            let pen = -x_curr;
            if dof_x < n_dofs {
                f[dof_x] = eps_n * pen;
            }
            // frictionless: no y-force
        }
    }
    f
}

// ─── Lumped mass (VectorH1, triangular mesh) ────────────────────────────

fn assemble_lumped_mass(mesh: &Mesh<2>, rho: f64, quad_order: u8) -> Vec<f64> {
    use fem_element::lagrange::TriP1;
    use fem_element::ReferenceElement;

    let n_nodes = mesh.n_nodes() as usize;
    let n_dofs = n_nodes * 2;
    let mut mass = vec![0.0; n_dofs];
    let ref_elem = TriP1;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let n_ldofs = nodes.len();
        let (_, det_j) = tri_jacobian(mesh, &nodes);
        let quad = ref_elem.quadrature(quad_order);
        let mut phi = vec![0.0; n_ldofs];

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j.abs();
            ref_elem.eval_basis(xi, &mut phi);
            let phi_sum: f64 = phi.iter().sum();
            for i in 0..n_ldofs {
                let m_add = w * rho * phi[i] * phi_sum;
                let node = nodes[i] as usize;
                mass[node * 2] += m_add;
                mass[node * 2 + 1] += m_add;
            }
        }
    }

    for m in mass.iter_mut() {
        if *m < 1e-30 { *m = 1e-30; }
    }
    mass
}

fn tri_jacobian(mesh: &Mesh<2>, nodes: &[u32]) -> ([[f64; 2]; 2], f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let j = [
        [x1[0] - x0[0], x2[0] - x0[0]],
        [x1[1] - x0[1], x2[1] - x0[1]],
    ];
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    (j, det)
}

// ─── CFL time step ──────────────────────────────────────────────────────

fn estimate_cfl_dt(E: f64, nu: f64, rho: f64, mesh: &Mesh<2>) -> f64 {
    let lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = E / (2.0 * (1.0 + nu));
    let c = ((lambda + 2.0 * mu) / rho).sqrt();

    let mut h_min = f64::MAX;
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let n = nodes.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let xi = mesh.node_coords(nodes[i]);
            let xj = mesh.node_coords(nodes[j]);
            let dx = xi[0] - xj[0];
            let dy = xi[1] - xj[1];
            h_min = h_min.min((dx * dx + dy * dy).sqrt().max(1e-30));
        }
    }
    0.8 * h_min / c
}

// ─── Energy diagnostics ─────────────────────────────────────────────────

fn kinetic_energy(mass: &[f64], vel: &[f64]) -> f64 {
    mass.iter().zip(vel.iter()).map(|(m, v)| 0.5 * m * v * v).sum()
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // 1. Mesh
    let mesh = make_rect_tri_mesh(args.nx, args.ny, args.L, args.H);
    let dim = 2;
    let n_nodes = mesh.n_nodes() as usize;
    let n_dofs = n_nodes * dim;

    println!("=== Taylor Bar Impact Benchmark (2D plane strain) ===");
    println!("  Mesh: {} × {} divisions, {} Tri3 nodes, {} DOFs",
             args.nx, args.ny, n_nodes, n_dofs);
    println!("  L = {} m, H = {} m, aspect = {:.1}", args.L, args.H, args.L / args.H);
    println!("  V₀ = {} m/s, ρ = {} kg/m³", args.V0, args.rho);
    println!("  E = {:.3e} Pa, ν = {}, σ_y = {:.3e} Pa, H_iso = {:.3e} Pa",
             args.E, args.nu, args.sigma_y, args.H_iso);

    // 2. Count total QPs for explicit J2 state
    let total_qp = count_qp_tri(&mesh, args.quad_order);
    let qp_states = std::cell::RefCell::new(vec![ExplicitJ2QpState::default(); total_qp]);
    let j2_cfg = ExplicitJ2Config {
        E: args.E, nu: args.nu, sigma_y: args.sigma_y, H: args.H_iso,
    };
    println!("  QPs: {}, σ_y = {:.3e} Pa", total_qp, args.sigma_y);

    // 3. Lumped mass
    let lumped = assemble_lumped_mass(&mesh, args.rho, args.quad_order);
    let total_mass = lumped.iter().sum::<f64>();
    let expected_mass = 2.0 * args.rho * args.L * args.H; // VectorH1: 2× mass (x + y DOFs)
    println!("  Mass: {:.6e} kg (expected {:.6e}, ratio {:.4})",
             total_mass, expected_mass, total_mass / expected_mass);

    // 4. Initial conditions
    let mut u = vec![0.0; n_dofs];
    let mut state = ExplicitState::new(n_dofs);

    // Identify left-wall nodes (x=0, boundary tag 4) for Dirichlet BC
    let mut wall_dofs: Vec<u32> = Vec::new();
    for f in mesh.face_iter() {
        if mesh.face_tag(f) == 4 { // left wall
            let nodes = mesh.face_nodes(f);
            for &node in nodes {
                let dof_x = node * 2; // x-component only (frictionless wall)
                if !wall_dofs.contains(&dof_x) {
                    wall_dofs.push(dof_x);
                }
            }
        }
    }
    wall_dofs.sort();
    let n_wall_dofs = wall_dofs.len();
    println!("  Wall DOFs (u_x=0): {}", n_wall_dofs);

    // Initial velocity: V0 in -x direction (except wall nodes)
    for node in 0..n_nodes {
        let dof_x = node as u32 * 2;
        if !wall_dofs.contains(&dof_x) {
            state.vel[node * 2] = -args.V0;
        }
    }

    // 5. Time step
    let dt = if args.dt > 0.0 {
        args.dt
    } else {
        let cfl_dt = estimate_cfl_dt(args.E, args.nu, args.rho, &mesh);
        println!("  CFL Δt = {:.3e} s", cfl_dt);
        cfl_dt * 0.5
    };
    let n_steps = (args.t_final / dt).ceil() as usize;
    println!("  Δt = {:.3e} s  ({} steps to t = {:.3e} s)", dt, n_steps, args.t_final);

    // 6. Central difference
    let cd = CentralDifferenceExplicit { gamma: 0.6 }; // γ>0.5 adds numerical dissipation
    let bc_dofs = wall_dofs;
    let _f_ext = vec![0.0; n_dofs];

    // 7. Energy tracking
    let init_ke = kinetic_energy(&lumped, &state.vel);
    let init_ie = 0.0; // FiniteStrainPlasticity has no elastic_energy()
    let mut cumulative_ie = 0.0;
    let mut u_prev = u.clone();
    let f_int_cell = std::cell::RefCell::new(vec![0.0; n_dofs]);

    println!("\n  {:>6}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}",
             "Step", "t [s]", "KE [J]", "IE(elas) [J]", "TE [J]", "ΔTE [J]");
    println!("  {}",
             "------  -------------  -------------  -------------  -------------  -------------");
    println!("  {:>6}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
             0, 0.0, init_ke, init_ie, init_ke + init_ie, 0.0);

    // 8. Time loop
    let mut t = 0.0;
    for step in 1..=n_steps {
        let dt_actual = dt.min(args.t_final - t);
        if dt_actual <= 0.0 {
            break;
        }

        cd.step(&lumped, dt_actual, &mut u, &mut state, &bc_dofs, |u_pred| {
            let mut r = vec![0.0; n_dofs];
            // Use rate-form explicit J2 (LS-DYNA style)
            // Compute internal force using INCREMENTAL strain: Δε = sym(∇(u_pred - u))
            let f_int = assemble_explicit_j2_2d(&mesh, u_pred, &u_prev, &mut qp_states.borrow_mut(), &j2_cfg, args.quad_order);
            r.copy_from_slice(&f_int);
            if r.iter().any(|v| v.is_nan() || v.is_infinite()) {
                eprintln!("  ❌ NaN/inf in residual at step {step}");
                return vec![0.0; n_dofs];
            }
            let mut f_total = vec![0.0; n_dofs];
            for i in 0..n_dofs {
                f_total[i] = -r[i];
            }
            // Store f_int for work computation (via captured RefCell)
            f_int_cell.replace(r);
            f_total
        });

        // Compute work increment: ΔIE = f_int · Δu  (using stored f_int)

        // Compute work increment: ΔIE = f_int · Δu  (using stored f_int)
        let r = f_int_cell.borrow().clone();
        for i in 0..n_dofs {
            let du = u[i] - u_prev[i];
            cumulative_ie += r[i] * du;
        }
        u_prev.copy_from_slice(&u);
        for i in 0..n_dofs {
            let du = u[i] - u_prev[i];
            cumulative_ie += r[i] * du;
        }
        u_prev.copy_from_slice(&u);

        t += dt_actual;

        if step % args.vis_steps == 0 || step == n_steps {
            let ke = kinetic_energy(&lumped, &state.vel);
            // Internal energy computed from cumulative work
            let te = ke + cumulative_ie;
            let dte = te - init_ke;

            if !args.no_vis {
                println!("  {:>6}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
                         step, t, ke, cumulative_ie, te, dte);
            }
        }

        if t >= args.t_final - 1e-14 {
            break;
        }
    }

    // 9. Final summary
    let final_ke = kinetic_energy(&lumped, &state.vel);
    println!("\n  === Final Results ===");
    println!("  Final KE = {:.6e} J  ({:.2}% of initial)",
             final_ke, 100.0 * final_ke / init_ke.max(1e-30));
    println!("  Plastic work = {:.6e} J", cumulative_ie);
    println!("  Energy variation   = {:.6e} J  ({:.4}%)",
             (final_ke + cumulative_ie) - init_ke,
             100.0 * ((final_ke + cumulative_ie) - init_ke) / init_ke.max(1e-30));

    // Deformed shape metrics
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = 0.0;
    for node in 0..n_nodes {
        let x0 = mesh.node_coords(node as u32);
        let x_curr = x0[0] + u[node * 2];
        let y_curr = x0[1] + u[node * 2 + 1];
        if x_curr < min_x { min_x = x_curr; }
        if x_curr > max_x { max_x = x_curr; }
        if y_curr.abs() > max_y { max_y = y_curr.abs(); }
    }
    let final_len = max_x - min_x;
    println!("\n  Final deformed length: {:.6e} m  ({:.2}% of L₀ = {:.4} m)",
             final_len, 100.0 * final_len / args.L, args.L);
    println!("  Min x (penetration)  : {:.6e} m", min_x);
    println!("  Max y (bulging)      : {:.6e} m  (×{:.4} of H/2 = {:.4} m)",
             max_y, 2.0 * max_y / args.H, args.H * 0.5);
    println!("\n  ✅ Taylor bar benchmark complete.");
}
