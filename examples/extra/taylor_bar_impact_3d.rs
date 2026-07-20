//! # Taylor bar impact benchmark (3D Hex8)
//!
//! 3D extension of the Taylor bar benchmark using Hex8 elements and
//! rate-form J2 plasticity — demonstrates the 3D explicit dynamics
//! capability for LS-DYNA / Abaqus/Explicit-class impact analysis.
//!
//! ## Usage
//! ```bash
//! cargo run --example taylor_bar_impact_3d --release -- --no-vis
//! cargo run --example taylor_bar_impact_3d --release -- --nx 16 --ny 8 --nz 8 --tf 3e-5 --no-vis
//! ```

use fem_assembly::explicit_j2::{
    assemble_explicit_j2_3d, assemble_lumped_mass_hex8,
    ExplicitJ2Config, ExplicitJ2QpState3d,
};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::ode::structural::{CentralDifferenceExplicit, ExplicitState};

fn main() {
    let (nx, ny, nz, L, H, W, V0, rho, E, nu, sigma_y, H_iso, t_final, dt_user, vis_steps, no_vis)
        = parse_args();

    let dim = 3;
    let mesh = make_hex_bar(nx, ny, nz, L, H, W);
    let n_nodes = mesh.n_nodes() as usize;
    let n_dofs = n_nodes * dim;

    println!("=== Taylor Bar 3D (Hex8) ===");
    println!("  Mesh: {}×{}×{} Hex8, {} nodes, {} DOFs", nx, ny, nz, n_nodes, n_dofs);
    println!("  L={} H={} W={}, V₀={} m/s, ρ={}", L, H, W, V0, rho);

    let total_qp = n_elems(&mesh) * 8; // HexQ1 quadrature points (order=1 → 2³=8)
    let qp_states = std::cell::RefCell::new(vec![ExplicitJ2QpState3d::default(); total_qp]);
    let j2_cfg = ExplicitJ2Config { E, nu, sigma_y, H: H_iso };

    let lumped = assemble_lumped_mass_hex8(&mesh, rho, 2);
    println!("  Total mass: {:.4e} kg (expected {:.4e})",
             lumped.iter().sum::<f64>(), 3.0 * rho * L * H * W);

    // Wall DOFs: x=0 face (tag 5)
    let mut wall_dofs: Vec<u32> = Vec::new();
    for f in mesh.face_iter() {
        if mesh.face_tag(f) == 5 {
            for &n in mesh.face_nodes(f) {
                let dof = n * 3;
                if !wall_dofs.contains(&dof) { wall_dofs.push(dof); }
            }
        }
    }
    wall_dofs.sort();
    println!("  Wall x-DOFs: {}", wall_dofs.len());

    let mut u = vec![0.0; n_dofs];
    let mut state = ExplicitState::new(n_dofs);
    // Velocity: V0 in -x direction (except wall)
    for node in 0..n_nodes {
        if !wall_dofs.contains(&(node as u32 * 3)) {
            state.vel[node * 3] = -V0;
        }
    }

    let cfl_dt = estimate_cfl_hex(&mesh, E, nu, rho);
    let dt = if dt_user > 0.0 { dt_user } else { cfl_dt * 0.5 };
    let n_steps = (t_final / dt).ceil() as usize;
    println!("  CFL Δt = {:.3e} s, using Δt = {:.3e} s ({} steps)", cfl_dt, dt, n_steps);

    let cd = CentralDifferenceExplicit { gamma: 0.5 };
    let _f_ext = vec![0.0; n_dofs];
    let init_ke = kinetic_energy_3d(&lumped, &state.vel);
    let cumulative_ie = std::cell::Cell::new(0.0f64);
    let u_prev = std::cell::RefCell::new(u.clone());

    println!("\n  {:>6}  {:>13}  {:>13}  {:>13}  {:>13}",
             "Step", "t [s]", "KE [J]", "IE [J]", "ΔTE [J]");
    println!("  {}", "------  -------------  -------------  -------------  -------------");
    println!("  {:>6}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
             0, 0.0, init_ke, 0.0, 0.0);

    let mut t = 0.0;
    for step in 1..=n_steps {
        let dt_actual = dt.min(t_final - t);
        if dt_actual <= 0.0 { break; }

        cd.step(&lumped, dt_actual, &mut u, &mut state, &wall_dofs, |u_pred| {
            let f_int = assemble_explicit_j2_3d(
                &mesh, u_pred, &u_prev.borrow(), &mut *qp_states.borrow_mut(), &j2_cfg, 2,
            );
            let mut f_total = vec![0.0; n_dofs];
            for i in 0..n_dofs { f_total[i] = -f_int[i]; }
            // Work & u_prev update (via Cell/RefCell)
            {
                let up = u_prev.borrow();
                let mut dw = 0.0;
                for i in 0..n_dofs {
                    dw += f_int[i] * (u_pred[i] - up[i]);
                }
                cumulative_ie.set(cumulative_ie.get() + dw);
            }
            u_prev.borrow_mut().copy_from_slice(u_pred);
            f_total
        });

        t += dt_actual;
        if step % vis_steps == 0 || step == n_steps {
            let ke = kinetic_energy_3d(&lumped, &state.vel);
            println!("  {:>6}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
                     step, t, ke, cumulative_ie.get(), (ke + cumulative_ie.get()) - init_ke);
        }
        if t >= t_final - 1e-14 { break; }
    }

    let final_ke = kinetic_energy_3d(&lumped, &state.vel);
    let final_ie = cumulative_ie.get();
    println!("\n  === Final ===");
    println!("  KE: {:.6e} J ({:.2}%)", final_ke, 100.0 * final_ke / init_ke.max(1e-30));
    println!("  Plastic work: {:.6e} J", final_ie);
    println!("  Energy variation: {:.6e} J ({:.4}%)",
             (final_ke + final_ie) - init_ke,
             100.0 * ((final_ke + final_ie) - init_ke) / init_ke.max(1e-30));

    // Deformation
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_yz = 0.0;
    for node in 0..n_nodes {
        let c = mesh.node_coords(node as u32);
        let x_cur = c[0] + u[node * 3];
        let y_cur = c[1] + u[node * 3 + 1];
        let z_cur = c[2] + u[node * 3 + 2];
        if x_cur < min_x { min_x = x_cur; }
        if x_cur > max_x { max_x = x_cur; }
        let r = ((y_cur - H/2.0).powi(2) + (z_cur - W/2.0).powi(2)).sqrt();
        if r > max_yz { max_yz = r; }
    }
    println!("  Length: {:.4e} m ({:.2}% of L₀={})", max_x - min_x,
             100.0 * (max_x - min_x) / L, L);
    println!("  Max radial expansion: {:.4e} m", max_yz);
    println!("  ✅ 3D Taylor bar complete.");
}

// ─── Helpers ───────────────────────────────────────────────────────────

fn parse_args() -> (usize, usize, usize, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, usize, bool) {
    let mut nx = 12; let mut ny = 6; let mut nz = 6;
    let mut L = 0.1; let mut H = 0.02; let mut W = 0.02;
    let mut V0 = 200.0; let mut rho = 7800.0;
    let mut E = 200e9; let mut nu = 0.3; let mut sy = 250e6; let mut Hh = 500e6;
    let mut tf = 3e-5; let mut dt = 0.0; let mut vs = 100; let mut nv = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() { match a.as_str() {
        "--nx" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { nx = v; }
        "--ny" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { ny = v; }
        "--nz" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { nz = v; }
        "--L" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { L = v; }
        "--H" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { H = v; }
        "--W" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { W = v; }
        "--V" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { V0 = v; }
        "--rho" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { rho = v; }
        "--E" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { E = v; }
        "--nu" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { nu = v; }
        "--sy" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { sy = v; }
        "--Hhard" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { Hh = v; }
        "--tf" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { tf = v; }
        "--dt" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { dt = v; }
        "--vs" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { vs = v; }
        "--no-vis" | "--no-visualization" => { nv = true; }
        _ => {}
    }}
    (nx, ny, nz, L, H, W, V0, rho, E, nu, sy, Hh, tf, dt, vs, nv)
}

fn make_hex_bar(nx: usize, ny: usize, nz: usize, L: f64, H: f64, W: f64) -> Mesh<3> {
    use fem_mesh::element_type::ElementType;
    let npx = nx + 1; let npy = ny + 1; let npz = nz + 1;
    let mut coords = Vec::with_capacity(npx * npy * npz * 3);
    for k in 0..npz {
        for j in 0..npy {
            for i in 0..npx {
                coords.push(i as f64 * L / nx as f64);
                coords.push(j as f64 * H / ny as f64);
                coords.push(k as f64 * W / nz as f64);
            }
        }
    }
    let nid = |i: usize, j: usize, k: usize| -> u32 { (k * npy * npx + j * npx + i) as u32 };
    let mut conn = Vec::with_capacity(nx * ny * nz * 8);
    let mut elem_tags = Vec::with_capacity(nx * ny * nz);
    for k in 0..nz { for j in 0..ny { for i in 0..nx {
        conn.extend_from_slice(&[
            nid(i,j,k), nid(i+1,j,k), nid(i+1,j+1,k), nid(i,j+1,k),
            nid(i,j,k+1), nid(i+1,j,k+1), nid(i+1,j+1,k+1), nid(i,j+1,k+1),
        ]);
        elem_tags.push(1);
    }}}
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    macro_rules! add_quad { ($a:expr,$b:expr,$c:expr,$d:expr,$t:expr) => {
        face_conn.extend_from_slice(&[$a,$b,$c,$d]); face_tags.push($t);
    }}
    // z=0 (tag 1), z=W (tag 2)
    for j in 0..ny { for i in 0..nx { add_quad!(nid(i,j,0),nid(i+1,j,0),nid(i+1,j+1,0),nid(i,j+1,0),1); }}
    for j in 0..ny { for i in 0..nx { add_quad!(nid(i,j,nz),nid(i,j+1,nz),nid(i+1,j+1,nz),nid(i+1,j,nz),2); }}
    // y=0 (tag 3), y=H (tag 4)
    for k in 0..nz { for i in 0..nx { add_quad!(nid(i,0,k),nid(i+1,0,k),nid(i+1,0,k+1),nid(i,0,k+1),3); }}
    for k in 0..nz { for i in 0..nx { add_quad!(nid(i,ny,k),nid(i,ny,k+1),nid(i+1,ny,k+1),nid(i+1,ny,k),4); }}
    // x=0 (tag 5), x=L (tag 6)
    for k in 0..nz { for j in 0..ny { add_quad!(nid(0,j,k),nid(0,j,k+1),nid(0,j+1,k+1),nid(0,j+1,k),5); }}
    for k in 0..nz { for j in 0..ny { add_quad!(nid(nx,j,k),nid(nx,j+1,k),nid(nx,j+1,k+1),nid(nx,j,k+1),6); }}
    Mesh::uniform(coords, conn, elem_tags, ElementType::Hex8,
                  face_conn, face_tags, ElementType::Quad4)
}

fn kinetic_energy_3d(mass: &[f64], vel: &[f64]) -> f64 {
    mass.iter().zip(vel.iter()).map(|(m, v)| 0.5 * m * v * v).sum()
}

fn n_elems(mesh: &Mesh<3>) -> usize {
    let mut count = 0;
    for _ in mesh.elem_iter() { count += 1; }
    count
}

fn estimate_cfl_hex(mesh: &Mesh<3>, E: f64, nu: f64, rho: f64) -> f64 {
    let mu = E / (2.0 * (1.0 + nu));
    let lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let c = ((lam + 2.0 * mu) / rho).sqrt();
    let mut h_min = f64::MAX;
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        for i in 0..8 {
            let j = (i + 1) % 8;
            let xi = mesh.node_coords(nodes[i]);
            let xj = mesh.node_coords(nodes[j]);
            let d = ((xi[0]-xj[0]).powi(2) + (xi[1]-xj[1]).powi(2) + (xi[2]-xj[2]).powi(2)).sqrt();
            if d > 0.0 { h_min = h_min.min(d); }
        }
    }
    0.8 * h_min / c.max(1e-30)
}
