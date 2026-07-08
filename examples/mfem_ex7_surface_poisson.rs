//! # MFEM Example 7 — Screened Poisson on the Unit Sphere
//!
//! Solves `-Δu + u = f` on the unit sphere surface with `f = 7·x·y / r²`,
//! exact solution `u = x·y / r²`.  Demonstrates surface FEM on a 2-D manifold
//! embedded in 3-D space using `Mesh<3>` with Tri3 elements.
//!
//! Reference: `mfem/ex7.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex7_surface_poisson -- -no-vis
//! cargo run --example mfem_ex7_surface_poisson -- -r 3 -snap -no-vis
//! ```
//!
//! ## Flags
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `-r/--refine` | 2 | Uniform refinements (octahedron → subdivided N times) |
//! | `-o/--order` | 2 | FE order (only 1 supported; higher values warn) |
//! | `-e/--elem` | 0 | Element type (0 = triangles; quads not yet supported) |
//! | `-snap/--always-snap` | — | Snap nodes to sphere after each refinement |
//! | `-amr/--refine-locally` | 0 | Local refinement (not yet implemented) |
//! | `-no-vis` | — | Disable GLVis (accepted, no-op) |

use fem_assembly::boundary::surface::{
    SurfaceAssembler, SurfaceDiffusionIntegrator, SurfaceDomainSourceIntegrator,
    SurfaceMassIntegrator,
};
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_solver::{fem_to_linlvo_csr, solve_pcg, SolveResult};
use fem_space::{H1Space, fe_space::FESpace};
use linlvo::SsorPrecond;
use std::collections::HashMap;

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // ── 1. Build sphere mesh (octahedron, subdivided N times) ──────────────────
    let mut mesh = build_octahedron_mesh();
    for l in 0..=args.ref_levels {
        if l > 0 {
            mesh = subdivide_tri3_surface(&mesh);
        }
        if args.always_snap || l == args.ref_levels {
            snap_nodes(&mut mesh);
        }
    }

    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();
    eprintln!("  Mesh: {} nodes, {} triangles on unit sphere", n_nodes, n_elems);

    // ── 2. Define H¹ space (order 1) ───────────────────────────────────────────
    let order = args.order.min(1); // only order 1 supported
    let space = H1Space::new(mesh, order);
    let n_dofs = space.n_dofs();
    println!("Number of unknowns: {}", n_dofs);

    // ── 3. Assemble surface stiffness (-Δ_Γ) + mass (+u) ─────────────────────
    // SurfaceAssembler uses closure-based integrators.
    let stiffness = SurfaceAssembler::assemble_bilinear(&space, &|nodes, ke| {
        SurfaceDiffusionIntegrator.add_to_element_matrix(nodes, ke);
    });
    let mass_mat = SurfaceAssembler::assemble_bilinear(&space, &|nodes, ke| {
        SurfaceMassIntegrator.add_to_element_matrix(nodes, ke);
    });
    let mut a = stiffness;
    // a = stiffness + mass
    for i in 0..a.nrows {
        for jp in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[jp] as usize;
            a.values[jp] += mass_mat.get(i, j);
        }
    }

    // ── 4. Assemble RHS: f = 7·x·y / r² ───────────────────────────────────────
    let rhs_fn = &|x: &[f64; 3]| {
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        7.0 * x[0] * x[1] / r2
    };
    let rhs = SurfaceAssembler::assemble_linear(&space, &|nodes, fe| {
        SurfaceDomainSourceIntegrator { f: rhs_fn }.add_to_element_vector(nodes, fe);
    });

    // ── 5. Solve: PCG + SSOR(ω=1) ────────────────────────────────────────────
    let mut u = vec![0.0; n_dofs];
    let la = fem_to_linlvo_csr(&a);
    let prec = SsorPrecond::from_csr(&la, 1.0).expect("SsorPrecond::from_csr");
    let res: SolveResult = solve_pcg(&a, &rhs, &mut u, &prec, 1e-12, 5000, true)
        .expect("PCG solve failed");
    if !res.converged {
        eprintln!(
            "  WARNING: solver did not converge (iters={}, res={:.3e})",
            res.iterations, res.final_residual
        );
    }

    // ── 6. L² error: ‖u_h − u_exact‖_{L²(Γ)} via element-level quadrature ────
    // Uses 3-point rule on reference triangle (matches SurfaceMassIntegrator),
    // with surface measure dS = sqrt(det(G)) * dξ.
    let exact_fn = |x: &[f64; 3]| {
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        x[0] * x[1] / r2
    };
    let qpts = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
    let qwt = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
    let mesh_3 = space.mesh();
    let mut err2 = 0.0_f64;
    for e in 0..mesh_3.n_elems() as u32 {
        let ns = mesh_3.element_nodes(e);
        let x0 = mesh_3.node_coords(ns[0]);
        let x1 = mesh_3.node_coords(ns[1]);
        let x2 = mesh_3.node_coords(ns[2]);
        // Surface Jacobian J = [x1-x0, x2-x0]  (3×2)
        let j0 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
        let j1 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
        // Metric G = J^T * J
        let g00 = j0[0] * j0[0] + j0[1] * j0[1] + j0[2] * j0[2];
        let g01 = j0[0] * j1[0] + j0[1] * j1[1] + j0[2] * j1[2];
        let g11 = j1[0] * j1[0] + j1[1] * j1[1] + j1[2] * j1[2];
        let sqrt_det_g = (g00 * g11 - g01 * g01).sqrt().max(1e-30);
        for q in 0..3 {
            let (xi, eta) = (qpts[q][0], qpts[q][1]);
            let phi = [1.0 - xi - eta, xi, eta];
            // Physical coordinates of quadrature point
            let xp = [
                phi[0] * x0[0] + phi[1] * x1[0] + phi[2] * x2[0],
                phi[0] * x0[1] + phi[1] * x1[1] + phi[2] * x2[1],
                phi[0] * x0[2] + phi[1] * x1[2] + phi[2] * x2[2],
            ];
            let uh = phi[0] * u[ns[0] as usize]
                   + phi[1] * u[ns[1] as usize]
                   + phi[2] * u[ns[2] as usize];
            let ue = exact_fn(&xp);
            let diff = uh - ue;
            err2 += diff * diff * qwt[q] * sqrt_det_g;
        }
    }
    let l2_err = err2.sqrt();
    println!("\nL2 norm of error: {:.10e}", l2_err);

    // ── 7. Output files ───────────────────────────────────────────────────────
    {
        use fem_io::mfem::write_gf_file;
        use fem_io::mfem::write_mfem_file_3d;
        if let Err(e) = write_mfem_file_3d("sphere_refined.mesh", space.mesh()) {
            eprintln!("  Warning: could not write sphere_refined.mesh: {e}");
        }
        if let Err(e) = write_gf_file("sol.gf", 3, &u, "H1", order as u8, 1) {
            eprintln!("  Warning: could not write sol.gf: {e}");
        }
    }

    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── Sphere mesh construction ────────────────────────────────────────────────

/// Build an octahedron inscribed in the unit sphere (6 vertices, 8 triangles).
///
/// Vertices and connectivity match MFEM ex7: (±1,0,0), (0,±1,0), (0,0,±1)
/// so that the resulting mesh after refinement matches C++ exactly.
fn build_octahedron_mesh() -> Mesh<3> {
    // MFEM ex7 vertex order: X+, Y+, X-, Y-, Z+, Z-
    let coords = vec![
        1.0,  0.0,  0.0,  // 0: X+
        0.0,  1.0,  0.0,  // 1: Y+
       -1.0,  0.0,  0.0,  // 2: X-
        0.0, -1.0,  0.0,  // 3: Y-
        0.0,  0.0,  1.0,  // 4: Z+
        0.0,  0.0, -1.0,  // 5: Z-
    ];
    // MFEM ex7 triangle connectivity
    let conn = vec![
        0u32, 1, 4,  1, 2, 4,  2, 3, 4,  3, 0, 4,
        1, 0, 5,  2, 1, 5,  3, 2, 5,  0, 3, 5,
    ];

    Mesh {
        coords,
        conn,
        elem_tags: (1..=8).collect(), // attributes 1..8 (matches MFEM)
        elem_type: ElementType::Tri3,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
    }
}

/// Uniformly subdivide each Tri3 into 4. New edge-midpoint nodes are placed at
/// the linear midpoint (not yet snapped to the sphere).
fn subdivide_tri3_surface(mesh: &Mesh<3>) -> Mesh<3> {
    let old_n_tri = mesh.conn.len() / 3;

    // Build edge → midpoint map.
    let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut coords = mesh.coords.clone();
    let mut next_node = (coords.len() / 3) as u32;

    let mut new_conn = Vec::with_capacity(old_n_tri * 12);
    let mut new_tags = Vec::with_capacity(old_n_tri * 4);

    for t in 0..old_n_tri {
        let i = t * 3;
        let (a, b, c) = (mesh.conn[i], mesh.conn[i + 1], mesh.conn[i + 2]);
        let tag = mesh.elem_tags[t];

        let edge_key = |x: u32, y: u32| if x < y { (x, y) } else { (y, x) };

        let ab = *edge_map.entry(edge_key(a, b)).or_insert_with(|| {
            let j = next_node;
            next_node += 1;
            let (xa, ya, za) = (
                coords[a as usize * 3],
                coords[a as usize * 3 + 1],
                coords[a as usize * 3 + 2],
            );
            let (xb, yb, zb) = (
                coords[b as usize * 3],
                coords[b as usize * 3 + 1],
                coords[b as usize * 3 + 2],
            );
            coords.extend_from_slice(&[
                0.5 * (xa + xb),
                0.5 * (ya + yb),
                0.5 * (za + zb),
            ]);
            j
        });
        let ac = *edge_map.entry(edge_key(a, c)).or_insert_with(|| {
            let j = next_node;
            next_node += 1;
            let (xa, ya, za) = (
                coords[a as usize * 3],
                coords[a as usize * 3 + 1],
                coords[a as usize * 3 + 2],
            );
            let (xc, yc, zc) = (
                coords[c as usize * 3],
                coords[c as usize * 3 + 1],
                coords[c as usize * 3 + 2],
            );
            coords.extend_from_slice(&[
                0.5 * (xa + xc),
                0.5 * (ya + yc),
                0.5 * (za + zc),
            ]);
            j
        });
        let bc = *edge_map.entry(edge_key(b, c)).or_insert_with(|| {
            let j = next_node;
            next_node += 1;
            let (xb, yb, zb) = (
                coords[b as usize * 3],
                coords[b as usize * 3 + 1],
                coords[b as usize * 3 + 2],
            );
            let (xc, yc, zc) = (
                coords[c as usize * 3],
                coords[c as usize * 3 + 1],
                coords[c as usize * 3 + 2],
            );
            coords.extend_from_slice(&[
                0.5 * (xb + xc),
                0.5 * (yb + yc),
                0.5 * (zb + zc),
            ]);
            j
        });

        new_conn.extend_from_slice(&[a, ab, ac, b, bc, ab, c, ac, bc, ab, bc, ac]);
        new_tags.extend_from_slice(&[tag, tag, tag, tag]);
    }

    Mesh {
        coords,
        conn: new_conn,
        elem_tags: new_tags,
        elem_type: ElementType::Tri3,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
    }
}

/// Project all mesh nodes onto the unit sphere (normalize to r = 1).
fn snap_nodes(mesh: &mut Mesh<3>) {
    for n in 0..mesh.n_nodes() as u32 {
        let i = n as usize * 3;
        let (x, y, z) = (mesh.coords[i], mesh.coords[i + 1], mesh.coords[i + 2]);
        let r = (x * x + y * y + z * z).sqrt();
        mesh.coords[i] = x / r;
        mesh.coords[i + 1] = y / r;
        mesh.coords[i + 2] = z / r;
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    ref_levels: usize,
    order: u8,
    _elem_type: u8,
    always_snap: bool,
    #[allow(dead_code)]
    amr: u8,
    #[allow(dead_code)]
    no_vis: bool,
}

impl Args {
    fn parse() -> Self {
        let mut ref_levels: usize = 2;
        let mut order: u8 = 2;
        let mut elem_type: u8 = 0;
        let mut always_snap = false;
        let mut amr: u8 = 0;
        let mut no_vis = false;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-r" | "--refine" => {
                    ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(2);
                }
                "-o" | "--order" => {
                    order = it.next().and_then(|s| s.parse().ok()).unwrap_or(2);
                    if order > 1 {
                        eprintln!("  Warning: order > 1 not yet supported; using order 1");
                    }
                }
                "-e" | "--elem" => {
                    elem_type = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    if elem_type != 0 {
                        eprintln!("  Warning: only triangles (elem=0) supported; using triangles");
                    }
                }
                "-snap" | "--always-snap" => {
                    always_snap = true;
                }
                "-amr" | "--refine-locally" => {
                    amr = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    if amr > 0 {
                        eprintln!("  Warning: AMR not yet implemented for surface meshes");
                    }
                }
                "-no-vis" | "--no-visualization" => {
                    no_vis = true;
                }
                _ => {}
            }
        }

        Args {
            ref_levels,
            order,
            _elem_type: elem_type,
            always_snap,
            amr,
            no_vis,
        }
    }
}
