//! Example 38 immersed boundary baseline (toward MFEM ex38)
//!
//! Cut-cell subtriangulation on a background Tri mesh for a circular embedded domain.
//! This version adds a Nitsche-like weak Dirichlet treatment on the immersed boundary
//! using a chord-segment approximation per cut triangle.
//!
//! ## Level-set extension
//! A `LevelSetShape` abstraction allows arbitrary immersed-boundary geometry
//! described by a signed distance / level-set function ψ(x):
//!   - `Circle`    – ψ(x) = |x − c| − r  (active: ψ < 0, i.e. inside)
//!   - `Halfspace` – ψ(x) = n · x − d    (active: ψ < 0)
//! Edge crossings are found via linear interpolation of ψ, and the outward
//! normal at the interface is ∇ψ / ‖∇ψ‖.

use std::f64::consts::PI;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::solve_sparse_cholesky;
use fem_space::{constraints::apply_dirichlet, fe_space::FESpace, H1Space};

fn main() {
    let args = parse_args();
    let result = solve_embedded_problem(&args);

    println!("=== fem-rs Example 38: immersed boundary baseline ===");
    println!("  Mesh: {}x{} subdivisions, P1 elements", args.n, args.n);
    println!(
        "  Circle center: ({:.3}, {:.3}), radius = {:.3}",
        args.cx, args.cy, args.radius
    );
    println!("  Subtriangulation per cut cell: {}", args.subdiv);
    println!("  Nitsche gamma: {:.3}", args.nitsche_gamma);
    println!("  Active DOFs: {}", result.active_dofs);
    println!("  Embedded area (approx): {:.6e}", result.area_estimate);
    println!("  Exact area:             {:.6e}", result.area_exact);
    println!("  Relative area error:    {:.3e}", result.area_rel_error);
    println!("  Interface length (chord approx): {:.6e}", result.interface_length);
    println!("  Embedded L2 error:      {:.3e}", result.l2_error);
    println!("  Interface L2 error:     {:.3e}", result.boundary_l2_error);
    println!("  Value range on active set: [{:.6}, {:.6}]", result.min_u, result.max_u);
    println!();
    println!("Note: cut-cell baseline now includes a Nitsche-like weak immersed boundary treatment.");
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    radius: f64,
    cx: f64,
    cy: f64,
    alpha: f64,
    subdiv: usize,
    nitsche_gamma: f64,
    /// Level-set shape override; if None, uses the `Circle` built from cx/cy/radius.
    level_set: Option<LevelSetShape>,
}

#[derive(Debug, Clone)]
struct EmbeddedResult {
    active_dofs: usize,
    area_estimate: f64,
    area_exact: f64,
    area_rel_error: f64,
    interface_length: f64,
    l2_error: f64,
    boundary_l2_error: f64,
    min_u: f64,
    max_u: f64,
}

#[derive(Debug, Clone)]
struct Circle {
    cx: f64,
    cy: f64,
    radius: f64,
}

// ── Level-set interface geometry ─────────────────────────────────────────────

/// Generic signed-distance / level-set description of an immersed interface.
///
/// Convention: **active (interior) region = ψ(x) < 0**.
#[derive(Debug, Clone)]
enum LevelSetShape {
    /// Circular interface: ψ(x) = ‖x − c‖ − r.
    Circle(Circle),
    /// Half-space interface: ψ(x) = n · x − d  (unit outward normal `n`).
    Halfspace { normal: [f64; 2], offset: f64 },
}

impl LevelSetShape {
    /// Evaluate ψ(x).  Negative = inside (active domain).
    fn eval(&self, x: [f64; 2]) -> f64 {
        match self {
            LevelSetShape::Circle(c) => {
                let dx = x[0] - c.cx;
                let dy = x[1] - c.cy;
                (dx * dx + dy * dy).sqrt() - c.radius
            }
            LevelSetShape::Halfspace { normal, offset } => {
                normal[0] * x[0] + normal[1] * x[1] - offset
            }
        }
    }

    /// Outward unit normal at point `x` on the interface.
    fn outward_normal(&self, x: [f64; 2]) -> [f64; 2] {
        match self {
            LevelSetShape::Circle(c) => {
                let dx = x[0] - c.cx;
                let dy = x[1] - c.cy;
                let inv = 1.0 / (dx * dx + dy * dy).sqrt().max(1.0e-14);
                [dx * inv, dy * inv]
            }
            LevelSetShape::Halfspace { normal, .. } => *normal,
        }
    }

    /// True when x is in the active (interior) domain.
    fn is_active(&self, x: [f64; 2]) -> bool {
        self.eval(x) < 0.0
    }
}

/// Find the chord (midpoint, length) where the triangle (x0, x1, x2) crosses
/// the zero-level-set of `ls`, using linear interpolation of ψ along each edge.
fn triangle_level_set_chord(
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    ls: &LevelSetShape,
) -> Option<([f64; 2], f64)> {
    let mut pts = Vec::<[f64; 2]>::new();
    edge_ls_intersections(x0, x1, ls, &mut pts);
    edge_ls_intersections(x1, x2, ls, &mut pts);
    edge_ls_intersections(x2, x0, ls, &mut pts);
    dedup_points(&mut pts, 1.0e-10);
    if pts.len() < 2 {
        return None;
    }
    let p0 = pts[0];
    let p1 = pts[1];
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0e-12 {
        return None;
    }
    let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
    Some((mid, len))
}

/// Append edge–level-set intersection points to `out` using linear interpolation.
fn edge_ls_intersections(
    a: [f64; 2],
    b: [f64; 2],
    ls: &LevelSetShape,
    out: &mut Vec<[f64; 2]>,
) {
    let pa = ls.eval(a);
    let pb = ls.eval(b);
    if pa * pb < 0.0 {
        // Linear interpolation: find t ∈ (0,1) s.t. ψ(a + t(b−a)) = 0
        let t = pa / (pa - pb);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    // Exact zero at endpoint — handled by dedup
    if pa.abs() < 1.0e-14 {
        out.push(a);
    }
    if pb.abs() < 1.0e-14 {
        out.push(b);
    }
}

fn parse_args() -> Args {
    let mut args = Args {
        n: 18,
        radius: 0.30,
        cx: 0.5,
        cy: 0.5,
        alpha: 20.0,
        subdiv: 8,
        nitsche_gamma: 20.0,
        level_set: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => args.n = it.next().unwrap_or("18".into()).parse().unwrap_or(18),
            "--radius" => args.radius = it.next().unwrap_or("0.30".into()).parse().unwrap_or(0.30),
            "--cx" => args.cx = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "--cy" => args.cy = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "--alpha" => args.alpha = it.next().unwrap_or("20.0".into()).parse().unwrap_or(20.0),
            "--subdiv" => args.subdiv = it.next().unwrap_or("8".into()).parse().unwrap_or(8),
            "--nitsche-gamma" => {
                args.nitsche_gamma = it.next().unwrap_or("20.0".into()).parse().unwrap_or(20.0)
            }
            "--level-set" => {
                args.level_set = match it.next().as_deref() {
                    Some("halfspace") => Some(LevelSetShape::Halfspace {
                        normal: [0.0, 1.0],
                        offset: 0.5,
                    }),
                    _ => None, // default: use Circle from cx/cy/radius
                };
            }
            _ => {}
        }
    }
    args.radius = args.radius.clamp(0.05, 0.45);
    args.alpha = args.alpha.max(1.0e-6);
    args.subdiv = args.subdiv.max(1);
    args.nitsche_gamma = args.nitsche_gamma.max(1.0e-6);
    args
}

fn solve_embedded_problem(args: &Args) -> EmbeddedResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let ls = args.level_set.clone().unwrap_or_else(|| {
        LevelSetShape::Circle(Circle {
            cx: args.cx,
            cy: args.cy,
            radius: args.radius,
        })
    });

    let (mut mat, mut rhs, active_mask, area_estimate, interface_length) =
        assemble_embedded_system(&space, &ls, args.alpha, args.subdiv, args.nitsche_gamma);

    let inactive_dofs: Vec<u32> = active_mask
        .iter()
        .enumerate()
        .filter_map(|(i, active)| if *active { None } else { Some(i as u32) })
        .collect();
    if !inactive_dofs.is_empty() {
        apply_dirichlet(
            &mut mat,
            &mut rhs,
            &inactive_dofs,
            &vec![0.0; inactive_dofs.len()],
        );
    }

    let solution = solve_sparse_cholesky(&mat, &rhs).expect("embedded ex38 Cholesky solve failed");

    let (l2_error, boundary_l2_error, min_u, max_u) =
        embedded_solution_metrics(&space, &solution, &ls, args.subdiv);

    let area_exact = match &ls {
        LevelSetShape::Circle(c) => std::f64::consts::PI * c.radius * c.radius,
        LevelSetShape::Halfspace { normal, offset } => {
            // area of the unit square on the active side (ψ < 0)
            // For n·x = d with unit square [0,1]², the active area is
            // the integral of 1 over {x ∈ [0,1]² : n·x < d}.
            // For the canonical horizontal cut (n=[0,1], d=offset):
            let hy = offset.clamp(0.0, 1.0);
            let nx = normal[0];
            let ny = normal[1];
            // Approximate: use hy if purely vertical normal
            if nx.abs() < 1.0e-12 && ny.abs() > 0.5 {
                hy
            } else if ny.abs() < 1.0e-12 && nx.abs() > 0.5 {
                offset.clamp(0.0, 1.0)
            } else {
                // Fallback: use the area estimate itself (can't compute analytically for arbitrary n)
                area_estimate
            }
        }
    };

    let area_rel_error = ((area_estimate - area_exact) / area_exact.max(1.0e-14)).abs();
    let active_dofs = active_mask.iter().filter(|flag| **flag).count();

    EmbeddedResult {
        active_dofs,
        area_estimate,
        area_exact,
        area_rel_error,
        interface_length,
        l2_error,
        boundary_l2_error,
        min_u,
        max_u,
    }
}

fn assemble_embedded_system(
    space: &H1Space<SimplexMesh<2>>,
    ls: &LevelSetShape,
    alpha: f64,
    subdiv: usize,
    nitsche_gamma: f64,
) -> (CsrMatrix<f64>, Vec<f64>, Vec<bool>, f64, f64) {
    let mesh = space.mesh();
    let ndofs = space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(ndofs, ndofs);
    for i in 0..ndofs {
        coo.add(i, i, 0.0);
    }
    let mut rhs = vec![0.0_f64; ndofs];
    let mut active_dofs = vec![false; ndofs];
    let mut total_area = 0.0_f64;
    let mut total_interface_length = 0.0_f64;

    for elem in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(elem);
        let dofs = space.element_dofs(elem);

        let x0 = to_point(mesh.node_coords(nodes[0]));
        let x1 = to_point(mesh.node_coords(nodes[1]));
        let x2 = to_point(mesh.node_coords(nodes[2]));

        let grad = parent_gradients(x0, x1, x2);
        let area_parent =
            0.5 * ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
        let h = (2.0 * area_parent).sqrt().max(1.0e-12);

        for (sub_centroid, sub_area, phi) in subdivided_triangle_samples(x0, x1, x2, subdiv) {
            if !ls.is_active(sub_centroid) {
                continue;
            }
            total_area += sub_area;
            for &dof in dofs {
                active_dofs[dof as usize] = true;
            }

            for i in 0..3 {
                rhs[dofs[i] as usize] += alpha * sub_area * phi[i];
                for j in 0..3 {
                    let stiffness = (grad[i][0] * grad[j][0] + grad[i][1] * grad[j][1]) * sub_area;
                    let mass = alpha * sub_area * phi[i] * phi[j];
                    coo.add(dofs[i] as usize, dofs[j] as usize, stiffness + mass);
                }
            }
        }

        // Nitsche weak Dirichlet on the immersed interface (level-set chord)
        // Only apply for triangles on the ACTIVE side of the interface (ψ(centroid) < 0),
        // to avoid double-counting when the interface coincides with mesh edges.
        let centroid = [
            (x0[0] + x1[0] + x2[0]) / 3.0,
            (x0[1] + x1[1] + x2[1]) / 3.0,
        ];
        if ls.eval(centroid) < 0.0 {
            if let Some((mid, seg_len)) = triangle_level_set_chord(x0, x1, x2, ls) {
                total_interface_length += seg_len;
                let phi_mid = barycentric_shape(mid, x0, x1, x2);
                let normal = ls.outward_normal(mid);
                let penalty = nitsche_gamma / h;
                let g = 1.0_f64;

                for i in 0..3 {
                    let gi = dofs[i] as usize;
                    let dni = normal[0] * grad[i][0] + normal[1] * grad[i][1];
                    rhs[gi] += seg_len * (-dni * g + penalty * phi_mid[i] * g);

                    for j in 0..3 {
                        let gj = dofs[j] as usize;
                        let dnj = normal[0] * grad[j][0] + normal[1] * grad[j][1];
                        let aij = seg_len
                            * (-dni * phi_mid[j] - dnj * phi_mid[i]
                                + penalty * phi_mid[i] * phi_mid[j]);
                        coo.add(gi, gj, aij);
                    }
                }
            }
        }
    }

    (
        coo.into_csr(),
        rhs,
        active_dofs,
        total_area,
        total_interface_length,
    )
}

fn embedded_solution_metrics(
    space: &H1Space<SimplexMesh<2>>,
    u: &[f64],
    ls: &LevelSetShape,
    subdiv: usize,
) -> (f64, f64, f64, f64) {
    let mesh = space.mesh();
    let mut err2 = 0.0_f64;
    let mut area = 0.0_f64;
    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;

    let mut bnd_err2 = 0.0_f64;
    let mut bnd_len = 0.0_f64;

    for elem in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(elem);
        let dofs = space.element_dofs(elem);

        let x0 = to_point(mesh.node_coords(nodes[0]));
        let x1 = to_point(mesh.node_coords(nodes[1]));
        let x2 = to_point(mesh.node_coords(nodes[2]));

        for (sub_centroid, sub_area, phi) in subdivided_triangle_samples(x0, x1, x2, subdiv) {
            if !ls.is_active(sub_centroid) {
                continue;
            }
            let uh = phi[0] * u[dofs[0] as usize]
                + phi[1] * u[dofs[1] as usize]
                + phi[2] * u[dofs[2] as usize];
            err2 += sub_area * (uh - 1.0) * (uh - 1.0);
            area += sub_area;
            min_u = min_u.min(uh);
            max_u = max_u.max(uh);
        }

        if let Some((mid, seg_len)) = triangle_level_set_chord(x0, x1, x2, ls) {
            let phi_mid = barycentric_shape(mid, x0, x1, x2);
            let uh_mid = phi_mid[0] * u[dofs[0] as usize]
                + phi_mid[1] * u[dofs[1] as usize]
                + phi_mid[2] * u[dofs[2] as usize];
            bnd_err2 += seg_len * (uh_mid - 1.0) * (uh_mid - 1.0);
            bnd_len += seg_len;
        }
    }

    if !min_u.is_finite() {
        min_u = 0.0;
        max_u = 0.0;
    }

    let l2 = (err2 / area.max(1.0e-14)).sqrt();
    let bnd_l2 = (bnd_err2 / bnd_len.max(1.0e-14)).sqrt();
    (l2, bnd_l2, min_u, max_u)
}

fn subdivided_triangle_samples(
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    subdiv: usize,
) -> Vec<([f64; 2], f64, [f64; 3])> {
    let mut out = Vec::new();
    let h = 1.0 / subdiv as f64;

    for i in 0..subdiv {
        for j in 0..(subdiv - i) {
            let p00 = [i as f64 * h, j as f64 * h];
            let p10 = [(i + 1) as f64 * h, j as f64 * h];
            let p01 = [i as f64 * h, (j + 1) as f64 * h];
            add_subtriangle_sample(&mut out, x0, x1, x2, p00, p10, p01);

            if i + j + 1 < subdiv {
                let p11 = [(i + 1) as f64 * h, (j + 1) as f64 * h];
                add_subtriangle_sample(&mut out, x0, x1, x2, p10, p11, p01);
            }
        }
    }

    out
}

fn add_subtriangle_sample(
    out: &mut Vec<([f64; 2], f64, [f64; 3])>,
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
) {
    let centroid_ref = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
    let centroid_phys = map_to_phys(x0, x1, x2, centroid_ref);
    let phi = [
        1.0 - centroid_ref[0] - centroid_ref[1],
        centroid_ref[0],
        centroid_ref[1],
    ];

    let pa = map_to_phys(x0, x1, x2, a);
    let pb = map_to_phys(x0, x1, x2, b);
    let pc = map_to_phys(x0, x1, x2, c);
    let area =
        0.5 * ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])).abs();

    out.push((centroid_phys, area, phi));
}

fn parent_gradients(x0: [f64; 2], x1: [f64; 2], x2: [f64; 2]) -> [[f64; 2]; 3] {
    let two_area = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0]);
    let inv_two_area = 1.0 / two_area;
    [
        [
            (x1[1] - x2[1]) * inv_two_area,
            (x2[0] - x1[0]) * inv_two_area,
        ],
        [
            (x2[1] - x0[1]) * inv_two_area,
            (x0[0] - x2[0]) * inv_two_area,
        ],
        [
            (x0[1] - x1[1]) * inv_two_area,
            (x1[0] - x0[0]) * inv_two_area,
        ],
    ]
}

fn map_to_phys(x0: [f64; 2], x1: [f64; 2], x2: [f64; 2], xi: [f64; 2]) -> [f64; 2] {
    [
        x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
        x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
    ]
}

fn inside_circle(x: [f64; 2], circle: &Circle) -> bool {
    let dx = x[0] - circle.cx;
    let dy = x[1] - circle.cy;
    dx * dx + dy * dy <= circle.radius * circle.radius
}

fn barycentric_shape(x: [f64; 2], x0: [f64; 2], x1: [f64; 2], x2: [f64; 2]) -> [f64; 3] {
    let det = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0]);
    let l1 = ((x1[0] - x[0]) * (x2[1] - x[1]) - (x1[1] - x[1]) * (x2[0] - x[0])) / det;
    let l2 = ((x2[0] - x[0]) * (x0[1] - x[1]) - (x2[1] - x[1]) * (x0[0] - x[0])) / det;
    let l3 = 1.0 - l1 - l2;
    [l1, l2, l3]
}

fn circle_outward_normal(x: [f64; 2], circle: &Circle) -> [f64; 2] {
    let dx = x[0] - circle.cx;
    let dy = x[1] - circle.cy;
    let inv = 1.0 / (dx * dx + dy * dy).sqrt().max(1.0e-14);
    [dx * inv, dy * inv]
}

fn triangle_circle_chord(
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    circle: &Circle,
) -> Option<([f64; 2], f64)> {
    let mut pts = Vec::<[f64; 2]>::new();
    edge_circle_intersections(x0, x1, circle, &mut pts);
    edge_circle_intersections(x1, x2, circle, &mut pts);
    edge_circle_intersections(x2, x0, circle, &mut pts);
    dedup_points(&mut pts, 1.0e-10);

    if pts.len() < 2 {
        return None;
    }
    let p0 = pts[0];
    let p1 = pts[1];
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0e-12 {
        return None;
    }
    let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
    Some((mid, len))
}

fn edge_circle_intersections(a: [f64; 2], b: [f64; 2], circle: &Circle, out: &mut Vec<[f64; 2]>) {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let ax = a[0] - circle.cx;
    let ay = a[1] - circle.cy;
    let aa = vx * vx + vy * vy;
    let bb = 2.0 * (ax * vx + ay * vy);
    let cc = ax * ax + ay * ay - circle.radius * circle.radius;

    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return;
    }
    let sdisc = disc.sqrt();
    let t1 = (-bb - sdisc) / (2.0 * aa);
    let t2 = (-bb + sdisc) / (2.0 * aa);

    for t in [t1, t2] {
        if (-1.0e-12..=1.0 + 1.0e-12).contains(&t) {
            out.push([a[0] + t * vx, a[1] + t * vy]);
        }
    }
}

fn dedup_points(pts: &mut Vec<[f64; 2]>, eps: f64) {
    let mut out = Vec::<[f64; 2]>::new();
    'outer: for p in pts.iter().copied() {
        for q in &out {
            let dx = p[0] - q[0];
            let dy = p[1] - q[1];
            if dx * dx + dy * dy <= eps * eps {
                continue 'outer;
            }
        }
        out.push(p);
    }
    *pts = out;
}

fn to_point(x: &[f64]) -> [f64; 2] {
    [x[0], x[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex38_embedded_area_is_reasonable() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None,
        });
        assert!(result.area_rel_error < 3.0e-2, "area rel error = {}", result.area_rel_error);
        assert!(result.active_dofs > 0, "expected non-empty active set");
        assert!(
            result.interface_length > 1.0,
            "interface length too small: {}",
            result.interface_length
        );
    }

    #[test]
    fn ex38_embedded_solution_recovers_constant_state() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None,
        });
        assert!(result.l2_error < 6.0e-2, "embedded L2 error = {}", result.l2_error);
        assert!(
            result.boundary_l2_error < 1.0e-1,
            "embedded boundary L2 error = {}",
            result.boundary_l2_error
        );
        assert!(result.min_u > 0.75, "min_u = {}", result.min_u);
        assert!(result.max_u < 1.15, "max_u = {}", result.max_u);
    }

    #[test]
    fn ex38_finer_cut_cell_subdivision_improves_area_accuracy() {
        let coarse = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 4,
            nitsche_gamma: 20.0,
            level_set: None,
        });
        let fine = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None,
        });

        assert!(
            fine.area_rel_error < coarse.area_rel_error,
            "expected finer cut-cell subdivision to improve area accuracy: coarse={} fine={}",
            coarse.area_rel_error,
            fine.area_rel_error
        );
        assert!(fine.area_rel_error < 1.0e-3, "fine subdivision area error too large: {}", fine.area_rel_error);
    }

    #[test]
    fn ex38_nitsche_gamma_variation_preserves_constant_solution() {
        let weak = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 10.0,
            level_set: None,
        });
        let strong = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 40.0,
            level_set: None,
        });

        for (label, result) in [("weak", &weak), ("strong", &strong)] {
            assert!(result.l2_error < 1.0e-12, "{label} gamma embedded L2 error = {}", result.l2_error);
            assert!(result.boundary_l2_error < 1.0e-12, "{label} gamma boundary L2 error = {}", result.boundary_l2_error);
            assert!((result.min_u - 1.0).abs() < 1.0e-12, "{label} gamma min_u = {}", result.min_u);
            assert!((result.max_u - 1.0).abs() < 1.0e-12, "{label} gamma max_u = {}", result.max_u);
        }

        assert!(
            (weak.area_rel_error - strong.area_rel_error).abs() < 1.0e-12,
            "area estimate should be gamma-independent for fixed geometry: weak={} strong={}",
            weak.area_rel_error,
            strong.area_rel_error
        );
        assert!(
            (weak.interface_length - strong.interface_length).abs() < 1.0e-12,
            "interface length should be gamma-independent for fixed geometry: weak={} strong={}",
            weak.interface_length,
            strong.interface_length
        );
    }

    // ── Level-set halfspace tests ─────────────────────────────────────────────

    /// Horizontal cut at y = 0.5: active region = lower half [0,1] × [0, 0.5].
    /// Exact area = 0.5; the chord approximation is exact for straight interfaces.
    #[test]
    fn ex38_levelset_halfspace_area_matches_exact() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.5,
            }),
        });
        // Straight interface → area estimate should be very accurate
        assert!(
            result.area_rel_error < 1.0e-10,
            "halfspace area rel error = {}",
            result.area_rel_error
        );
        assert!(result.active_dofs > 0, "expected non-empty active set");
        // Interface is the line y=0.5 on the unit square; exact length = 1.0
        assert!(
            (result.interface_length - 1.0).abs() < 1.0e-2,
            "halfspace interface length = {}",
            result.interface_length
        );
    }

    /// The halfspace level-set should still recover u ≈ 1 in the active domain
    /// (same forcing and boundary data as the circle test).
    #[test]
    fn ex38_levelset_halfspace_recovers_constant_state() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.5,
            }),
        });
        assert!(
            result.l2_error < 1.5e-1,
            "halfspace embedded L2 error = {}",
            result.l2_error
        );
        assert!(result.min_u > 0.5, "halfspace min_u = {}", result.min_u);
        assert!(result.max_u < 1.2, "halfspace max_u = {}", result.max_u);
    }

    /// Varying γ for the halfspace should not affect the cut area estimate.
    #[test]
    fn ex38_levelset_halfspace_area_is_gamma_independent() {
        let args_base = Args {
            n: 14,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 6,
            nitsche_gamma: 10.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.4,
            }),
        };
        let weak   = solve_embedded_problem(&Args { nitsche_gamma: 10.0, ..args_base.clone() });
        let strong = solve_embedded_problem(&Args { nitsche_gamma: 50.0, ..args_base.clone() });

        assert!(
            (weak.area_estimate - strong.area_estimate).abs() < 1.0e-12,
            "halfspace area must be γ-independent: weak={} strong={}",
            weak.area_estimate,
            strong.area_estimate
        );
        assert!(
            (weak.interface_length - strong.interface_length).abs() < 1.0e-12,
            "halfspace interface length must be γ-independent: weak={} strong={}",
            weak.interface_length,
            strong.interface_length
        );
    }

    /// Identical circle geometry must give identical area and interface-length
    /// estimates on repeated calls (determinism).
    #[test]
    fn ex38_circle_embedded_results_are_deterministic() {
        let args = Args {
            n: 10, radius: 0.25, cx: 0.5, cy: 0.5,
            alpha: 20.0, subdiv: 6, nitsche_gamma: 20.0,
            level_set: None,
        };
        let r1 = solve_embedded_problem(&args);
        let r2 = solve_embedded_problem(&args);
        assert_eq!(r1.area_estimate, r2.area_estimate,
            "area estimate is not deterministic: {} vs {}", r1.area_estimate, r2.area_estimate);
        assert_eq!(r1.interface_length, r2.interface_length,
            "interface length is not deterministic: {} vs {}", r1.interface_length, r2.interface_length);
    }
}

