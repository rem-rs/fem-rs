//! # Sod shock tube benchmark (DG Euler 2D)
//!
//! Classic 1D Riemann problem — left/right states separated at x=0.5,
//! solved with DG(P1) + SSP-RK3 on a narrow 2D triangular mesh.
//! Reflective walls on all boundaries are valid for t ≤ 0.2.
//!
//! Usage:
//!   cargo run --example dg_sod_shocktube --release -- --no-vis
//!   cargo run --example dg_sod_shocktube --release -- --nx 400 --tf 0.2 --no-vis

use fem_assembly::dg::dg_euler_2d::DgEuler2D;

fn main() {
    let (nx, ny, Lx, t_final, order, no_vis) = parse_args();
    let Ly = Lx * 0.005;

    // Build mesh + track centroids
    let (mesh, centroids) = make_tri_strip_with_centroids(nx, ny.max(1), Lx, Ly);

    let mut dg = DgEuler2D::with_order(mesh, order);
    dg.use_limiter = true; // enable limiter for shock capturing
    let dp = dg.dofs_per_elem();
    let n_elems = dg.n_dofs() / (dp * 4);

    // Sod initial condition
    let init = |x: f64, _y: f64| -> (f64, f64, f64, f64) {
        if x < 0.5 { (1.0, 0.0, 0.0, 1.0) } else { (0.125, 0.0, 0.0, 0.1) }
    };
    let mut u = dg.project_initial(&init);

    // CFL time step
    let h = dg.h_min();
    // DG CFL: dt ≤ h / ((2*order+1) * λ_max) where λ_max ≈ |u| + c
    // For Sod: λ_max ≈ 2.0. With safety factor 0.2:
    let dt = 0.2 * h / ((2.0 * order as f64 + 1.0) * 2.0);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt_actual = t_final / n_steps as f64;
    println!("=== Sod Shock Tube (DG Euler 2D) ===");
    println!("  Mesh: {}×{} Tri3(P1), {} elems, {} DOFs", nx, ny.max(1), n_elems, dg.n_dofs());
    println!("  h_min = {:.3e}, dt = {:.3e}, {} steps", h, dt_actual, n_steps);

    // SSP-RK3 time loop
    for step in 0..n_steps {
        dg.step_rk3(&mut u, dt_actual);
    }

    // Debug: print first few DOF values (always)
    for i in 0..10.min(u.len()) {
        eprintln!("  DOF[{}] = {:.6e}", i, u[i]);
    }

    // Extract element-averaged solution at centroids
    let mut results: Vec<(f64, f64, f64, f64)> = centroids.iter().map(|&(cx, _cy)| {
        // Element average = mean of nodal DOF values (P1)
        let e_idx = 0; // we iterate by centroid index
        // For each element, get the 4 conserved variables averaged over nodes
        (cx, 0.0, 0.0, 0.0)
    }).collect();

    // Actually compute element averages from DOF values
    for e in 0..n_elems {
        let mut cons_avg = [0.0; 4];
        for i in 0..dp {
            for c in 0..4 {
                cons_avg[c] += u[(e * dp + i) * 4 + c];
            }
        }
        for c in 0..4 { cons_avg[c] /= dp as f64; }
        let euler = fem_assembly::dg::dg_euler_2d::Euler2D::default();
        let (r, uvel, vvel, p) = euler.cons_to_prim(&cons_avg);
        results[e] = (centroids[e].0, r, uvel, p);
    }

    // Sort by x
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute exact Riemann solution at same x positions
    let exact: Vec<(f64, f64, f64)> = results.iter().map(|&(x, _, _, _)| {
        exact_sod_point(x, t_final)
    }).collect();

    // Print comparison
    println!("\n  x       ρ_num    ρ_exact   u_num    u_exact   p_num    p_exact");
    for (i, &(x, rn, un, pn)) in results.iter().enumerate() {
        let (re, ue, pe) = exact[i];
        if i % (results.len().max(1) / 20).max(1) == 0 {
            println!("  {:.4}  {:.6}  {:.6}  {:.6}  {:.6}  {:.6}  {:.6}",
                     x, rn, re, un, ue, pn, pe);
        }
    }

    // L² errors
    let n = results.len() as f64;
    let l2r = results.iter().zip(&exact).map(|(a, e)| (a.1 - e.0).powi(2)).sum::<f64>().sqrt() / n.sqrt();
    let l2u = results.iter().zip(&exact).map(|(a, e)| (a.2 - e.1).powi(2)).sum::<f64>().sqrt() / n.sqrt();
    let l2p = results.iter().zip(&exact).map(|(a, e)| (a.3 - e.2).powi(2)).sum::<f64>().sqrt() / n.sqrt();
    println!("\n  L² errors: ρ={:.6e}  u={:.6e}  p={:.6e}", l2r, l2u, l2p);
    println!("  ✅ Sod shock tube complete.");
}

fn make_tri_strip_with_centroids(nx: usize, ny: usize, Lx: f64, Ly: f64)
    -> (fem_mesh::Mesh<2>, Vec<(f64, f64)>)
{
    use fem_mesh::element_type::ElementType;
    use fem_mesh::Mesh;
    let npx = nx + 1; let npy = ny + 1;
    let mut coords = Vec::with_capacity(npx * npy * 2);
    for j in 0..npy { for i in 0..npx {
        coords.push(i as f64 * Lx / nx as f64);
        coords.push(j as f64 * Ly / ny as f64);
    }}
    let nid = |i: usize, j: usize| -> u32 { (j * npx + i) as u32 };
    let mut conn = Vec::with_capacity(2 * nx * ny * 3);
    let mut elem_tags = Vec::with_capacity(2 * nx * ny);
    let mut centroids = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny { for i in 0..nx {
        let e0 = nid(i,j); let e1 = nid(i+1,j);
        let e2 = nid(i+1,j+1); let e3 = nid(i,j+1);
        // Triangle 1
        conn.extend_from_slice(&[e0, e1, e3]);
        elem_tags.push(1);
        let cx1 = (coords[e0 as usize * 2] + coords[e1 as usize * 2] + coords[e3 as usize * 2]) / 3.0;
        let cy1 = (coords[e0 as usize * 2 + 1] + coords[e1 as usize * 2 + 1] + coords[e3 as usize * 2 + 1]) / 3.0;
        centroids.push((cx1, cy1));
        // Triangle 2
        conn.extend_from_slice(&[e1, e2, e3]);
        elem_tags.push(1);
        let cx2 = (coords[e1 as usize * 2] + coords[e2 as usize * 2] + coords[e3 as usize * 2]) / 3.0;
        let cy2 = (coords[e1 as usize * 2 + 1] + coords[e2 as usize * 2 + 1] + coords[e3 as usize * 2 + 1]) / 3.0;
        centroids.push((cx2, cy2));
    }}
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    for i in 0..nx {
        face_conn.extend_from_slice(&[nid(i,0), nid(i+1,0)]); face_tags.push(1);
        face_conn.extend_from_slice(&[nid(i+1,ny), nid(i,ny)]); face_tags.push(3);
    }
    for j in 0..ny {
        face_conn.extend_from_slice(&[nid(nx,j), nid(nx,j+1)]); face_tags.push(2);
        face_conn.extend_from_slice(&[nid(0,j+1), nid(0,j)]); face_tags.push(4);
    }
    let mesh = Mesh::uniform(coords, conn, elem_tags, ElementType::Tri3,
                             face_conn, face_tags, ElementType::Line2);
    (mesh, centroids)
}

fn exact_sod_point(x: f64, t: f64) -> (f64, f64, f64) {
    let gamma = 1.4;
    let (rl, ul, pl) = (1.0, 0.0, 1.0);
    let (rr, ur, pr) = (0.125, 0.0, 0.1);
    let xi = (x - 0.5) / t; // similarity variable
    let xi_max = 2.0;
    if xi <= -xi_max { return (rl, ul, pl); }
    if xi >= xi_max { return (rr, ur, pr); }

    let cl = (gamma * pl / rl).sqrt();
    let cr = (gamma * pr / rr).sqrt();

    // Solve for p* and u* using iterative method
    let p_star = solve_pstar(pl, pr, rl, rr, gamma);
    let w = solve_shock_speed(p_star, pr, rr, gamma);
    let aL = (gamma * p_star / (rl * (p_star / pl).powf(1.0/gamma))).sqrt();
    let aL_head = cl;
    let aL_tail = aL;
    let u_star = ul + (p_star - pl) / (rl * aL_head * f_m(p_star/pl, gamma).sqrt());

    // Characteristic speeds
    let s_hL = u_star - aL_tail; // left rarefaction tail
    let s_cL = ul - aL_head;     // left rarefaction head
    let s_cR = u_star;           // contact
    let s_hR = w;                // shock

    if xi <= s_cL { (rl, ul, pl) }
    else if xi < s_hL {
        // Left rarefaction fan
        let u_fan = (2.0 / (gamma + 1.0)) * (aL_head + (gamma - 1.0) * (-xi) / 2.0);
        let a_fan = (aL_head - (gamma - 1.0) * (-xi) / 2.0) * 2.0 / (gamma + 1.0);
        let r_fan = (a_fan * a_fan / gamma).powf(1.0 / (gamma - 1.0));
        let p_fan = r_fan * a_fan * a_fan / gamma;
        (r_fan, u_fan, p_fan)
    }
    else if xi <= s_cR {
        // Between left rarefaction tail and contact
        let rho_star_l = rl * (p_star / pl).powf(1.0 / gamma);
        (rho_star_l, u_star, p_star)
    }
    else if xi < s_hR {
        // Between contact and shock
        let rho_star_r = rr * (p_star / pr).powf(1.0 / gamma);
        (rho_star_r, u_star, p_star)
    }
    else {
        (rr, ur, pr)
    }
}

fn f_m(z: f64, gamma: f64) -> f64 {
    if z > 1.0 {
        (z - 1.0) / ((1.0 + (gamma + 1.0) / (2.0 * gamma) * (z - 1.0)).sqrt())
    } else {
        (gamma - 1.0) / (2.0 * gamma) * (1.0 - z) / (1.0 - z.powf((gamma - 1.0) / (2.0 * gamma)))
    }
}

fn solve_pstar(pl: f64, pr: f64, rl: f64, rr: f64, gamma: f64) -> f64 {
    let mut p = 0.5 * (pl + pr);
    for _ in 0..30 {
        let fl = f_val(p, pl, rl, gamma);
        let fr = f_val(p, pr, rr, gamma);
        let f = fl + fr - (0.0 - 0.0); // uL - uR = 0
        let df = (f_val(p * 1.001, pl, rl, gamma) - fl) / (p * 0.001) + 1e-15;
        let dp = -f / df;
        p = (p + dp).max(1e-12);
        if dp.abs() < 1e-12 * p { break; }
    }
    p
}

fn f_val(p: f64, pk: f64, rk: f64, gamma: f64) -> f64 {
    if p > pk {
        (p - pk) * ((2.0 / (gamma + 1.0)) / (p / pk + (gamma - 1.0) / (gamma + 1.0))).sqrt() * (pk * rk).sqrt()
    } else {
        (2.0 * gamma / (gamma - 1.0)) * (1.0 - (p / pk).powf((gamma - 1.0) / (2.0 * gamma))) * (gamma * pk * rk).sqrt()
    }
}

fn solve_shock_speed(p_star: f64, pr: f64, rr: f64, gamma: f64) -> f64 {
    let ratio = (gamma + 1.0) / (2.0 * gamma);
    0.0 + (gamma * pr / rr).sqrt() * (1.0 + ratio * (p_star / pr - 1.0)).sqrt()
}

fn parse_args() -> (usize, usize, f64, f64, u8, bool) {
    let mut nx = 100; let mut ny = 2; let mut Lx = 1.0;
    let mut tf = 0.2; let mut order: u8 = 1; let mut nv = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() { match a.as_str() {
        "--nx" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { nx = v; }
        "--ny" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { ny = v; }
        "--L" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { Lx = v; }
        "--tf" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { tf = v; }
        "--order" | "-o" => if let Some(v) = it.next().and_then(|v| v.parse().ok()) { order = v; }
        "--no-vis" | "--no-visualization" => { nv = true; }
        _ => {}
    }}
    (nx, ny, Lx, tf, order, nv)
}
