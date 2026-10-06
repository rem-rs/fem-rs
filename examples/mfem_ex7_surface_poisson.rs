//! # MFEM Example 7 — Screened Poisson on the Unit Sphere
//!
//! Solves `-Δu + u = f` on the unit sphere surface with `f = 7·x·y / r²`,
//! exact solution `u = x·y / r²`.  Demonstrates surface FEM on a 2-D manifold
//! embedded in 3-D space using `Mesh<3>` with Tri3/Tri6 or Quad4/Quad9 elements.
//!
//! Reference: `mfem/ex7.cpp`
//!
//! ## Usage
//! ```bash
//! # Triangles (default)
//! cargo run --example mfem_ex7_surface_poisson -- -no-vis
//! # Quads
//! cargo run --example mfem_ex7_surface_poisson -- -e 1 -snap -no-vis
//! ```
//!
//! ## Flags
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `-e/--elem` | 0 | Element type (0=tri, 1=quad) |
//! | `-r/--refine` | 2 | Uniform refinements |
//! | `-o/--order` | 2 | FE order |
//! | `-snap/--always-snap` | — | Snap after each refinement |
//! | `-amr/--refine-locally` | 0 | Not yet implemented |
//! | `-no-vis` | — | Disable GLVis (no-op) |

use fem_assembly::boundary::surface::{
    SurfaceAssembler, SurfaceBilinearIntegrator, SurfaceLinearIntegrator,
    SurfaceDiffusionIntegrator, SurfaceDomainSourceIntegrator, SurfaceMassIntegrator,
    SurfaceTri6BilinearIntegrator, SurfaceTri6LinearIntegrator,
    SurfaceQuad4Assembler, SurfaceQuad4BilinearIntegrator, SurfaceQuad4LinearIntegrator,
    SurfaceQuad4DiffusionIntegrator, SurfaceQuad4DomainSourceIntegrator,
    SurfaceQuad4MassIntegrator,
    SurfaceQuad9Assembler, SurfaceQuad9BilinearIntegrator, SurfaceQuad9LinearIntegrator,
    SurfaceQuad9DiffusionIntegrator, SurfaceQuad9MassIntegrator,
    SurfaceQuad9DomainSourceIntegrator,
};
use fem_assembly::boundary::surface_tri6::{
    SurfaceTri6Assembler,
    SurfaceTri6DiffusionIntegrator, SurfaceTri6MassIntegrator,
    SurfaceTri6DomainSourceIntegrator,
};
use fem_mesh::{
    Mesh, MeshTopology, element_type::ElementType,
    amr::{refine_at_vertex_surface, refine_uniform_surface_quad4, refine_uniform_surface_tri3},
};
use fem_linalg::CsrMatrix;
use fem_solver::{fem_to_linlvo_csr, solve_pcg, SolveResult};
use fem_space::{H1Space, fe_space::FESpace};
use fem_solver::GSSmoother;

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    // MFEM ex7 echoes the parsed options (`args.PrintOptions(cout)`, ex7.cpp:68)
    // before any other output; the echo mirrors OptionsParser::PrintOptions
    // byte-for-byte (ENABLE pair prints the long_name whose value is true, so
    // `always_snap == false` prints the false-branch name "--snap-at-the-end").
    println!("Options used:");
    println!("   --elem {}", args.elem_type);
    println!("   --order {}", args.order);
    println!("   --refine {}", args.ref_levels);
    println!("   --refine-locally {}", args.amr);
    println!("   {}", if args.no_vis { "--no-visualization" } else { "--visualization" });
    println!("   {}", if args.always_snap { "--always-snap" } else { "--snap-at-the-end" });
    let t0 = std::time::Instant::now();
    let is_quad = args.elem_type == 1;

    // ── 1. Build sphere mesh ─────────────────────────────────────────────────
    let mut mesh: Mesh<3> = if is_quad {
        build_cube_mesh()
    } else {
        build_octahedron_mesh()
    };

    // For P2 (Tri6): elevate before refinement so refinements preserve mid-edge nodes
    let use_tri6 = !is_quad && args.order >= 2;
    if use_tri6 {
        mesh = elevate_to_tri6(&mesh);
    }

    for l in 0..=args.ref_levels {
        if l > 0 {
            mesh = if is_quad {
                refine_uniform_surface_quad4(&mesh)
            } else if use_tri6 {
                refine_uniform_tri6(&mesh)
            } else {
                refine_uniform_surface_tri3(&mesh)
            };
            if args.always_snap {
                snap_nodes(&mut mesh);
            }
        }
    }
    if !args.always_snap || args.ref_levels == 0 {
        snap_nodes(&mut mesh);
    }

    // AMR: refine near north pole (MFEM ex7 -amr 1)
    for _ in 0..args.amr {
        mesh = refine_at_vertex_surface(&mesh, &[0.0, 0.0, 1.0]);
        snap_nodes(&mut mesh);
    }

    // ── 3. Define H1 space ──────────────────────────────────────────────────
    // Quad9/Tri6: order=2 — the true P2 space matching the row topology and
    // the P2 assembly bases (MFEM ex7 builds H1(order) on the same rows).
    // The old Tri6 hack (order=1 over all row nodes as "one DOF per node")
    // died when D819-A/D820-1/D832-1 corrected H1(1)-on-rows to MFEM's
    // corner-view semantics (3 dofs/element), which the 6-dof P2 assembler
    // cannot consume (round-88 Lane C found the resulting rc=101).
    let use_quad9 = is_quad && args.order >= 2;
    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();
    let elem_name = if is_quad { "quads" } else { "triangles" };
    eprintln!("  Mesh: {} nodes, {} {} on unit sphere", n_nodes, n_elems, elem_name);

    let h1_order = if use_tri6 || use_quad9 { 2 } else { args.order.min(1) };
    let mut space = H1Space::new(mesh, h1_order);
    // Snap edge-midpoint and centroid DOF coordinates to the sphere surface.
    // DofManager builds straight-line averages; for the sphere geometry we
    // need the radial projection (matching MFEM SetCurvature + snap behaviour).
    if use_quad9 {
        space.dof_manager_mut().snap_to_sphere();
    }
    let n_dofs = space.n_dofs();
    println!("Number of unknowns: {}", n_dofs);

    // ── 4. Assemble surface stiffness (-Delta_Gamma) + mass (+u) ────────────
    let rhs_fn = &|x: &[f64; 3]| {
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        7.0 * x[0] * x[1] / r2
    };

    let a: CsrMatrix<f64> = if use_quad9 {
        SurfaceQuad9Assembler::assemble_bilinear(&space, &[
            &SurfaceQuad9DiffusionIntegrator as &dyn SurfaceQuad9BilinearIntegrator,
            &SurfaceQuad9MassIntegrator,
        ])
    } else if is_quad {
        SurfaceQuad4Assembler::assemble_bilinear(&space, &[
            &SurfaceQuad4DiffusionIntegrator as &dyn SurfaceQuad4BilinearIntegrator,
            &SurfaceQuad4MassIntegrator,
        ])
    } else if use_tri6 {
        SurfaceTri6Assembler::assemble_bilinear(&space, &[
            &SurfaceTri6DiffusionIntegrator as &dyn SurfaceTri6BilinearIntegrator,
            &SurfaceTri6MassIntegrator,
        ])
    } else {
        SurfaceAssembler::assemble_bilinear(&space, &[
            &SurfaceDiffusionIntegrator as &dyn SurfaceBilinearIntegrator,
            &SurfaceMassIntegrator,
        ])
    };

    // ── 5. Assemble RHS: f = 7*x*y / r^2 ────────────────────────────────────
    let rhs: Vec<f64> = if use_quad9 {
        let src = SurfaceQuad9DomainSourceIntegrator { f: rhs_fn };
        SurfaceQuad9Assembler::assemble_linear(&space, &[
            &src as &dyn SurfaceQuad9LinearIntegrator,
        ])
    } else if is_quad {
        let src = SurfaceQuad4DomainSourceIntegrator { f: rhs_fn };
        SurfaceQuad4Assembler::assemble_linear(&space, &[
            &src as &dyn SurfaceQuad4LinearIntegrator,
        ])
    } else if use_tri6 {
        let src = SurfaceTri6DomainSourceIntegrator { f: rhs_fn };
        SurfaceTri6Assembler::assemble_linear(&space, &[
            &src as &dyn SurfaceTri6LinearIntegrator,
        ])
    } else {
        let src = SurfaceDomainSourceIntegrator { f: rhs_fn };
        SurfaceAssembler::assemble_linear(&space, &[
            &src as &dyn SurfaceLinearIntegrator,
        ])
    };

    // ── 5b. Solve: PCG + SSOR(omega=1) ────────────────────────────────────
    let mut u = vec![0.0; n_dofs];
    let la = fem_to_linlvo_csr(&a);
    let prec = GSSmoother::from_csr(&la).expect("GSSmoother");
    let res: SolveResult = solve_pcg(&a, &rhs, &mut u, &prec, 1e-12, 5000, true)
        .expect("PCG solve failed");
    if !res.converged {
        eprintln!(
            "  WARNING: solver did not converge (iters={}, res={:.3e})",
            res.iterations, res.final_residual
        );
    }

    // ── 6. L2 error via element-level quadrature ───────────────────────────
    let exact_fn = |x: &[f64; 3]| {
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        x[0] * x[1] / r2
    };
    let mesh_ptr = space.mesh();
    let l2_err = if use_quad9 {
        quad9_l2_error(mesh_ptr, &space, &u, &exact_fn).sqrt()
    } else if is_quad {
        quad4_l2_error(mesh_ptr, &u, &exact_fn).sqrt()
    } else if use_tri6 {
        // MFEM `GridFunction::ComputeL2Error` (fem/gridfunc.cpp): the default
        // integration rule is `IntRules.Get(fe.GetGeomType(), 2*fe.GetOrder()+3)`
        // — for these P2 triangles the 15-point order-7 Witherden–Vincent rule
        // (`tri_rule(7)`, D578) with the isoparametric (curved Tri6 row)
        // Jacobian at every quadrature point.  D1271: the example-local
        // 3-point/centroid-Jacobian rule was the metric deviation; the
        // measurement goes through the core MFEM-parity helper — parity probe
        // on the C++ mesh/solution snapshot: 5.43012990936432398e-3 vs C++
        // 5.43012990936427974e-3 (rel 8.1e-15, ≤2 ulp).
        let gf = fem_assembly::postproc::grid_function::GridFunction::new(&space, u.clone());
        gf.compute_l2_error(&|x: &[f64]| exact_fn(&[x[0], x[1], x[2]]), 2 * h1_order + 3)
    } else {
        tri3_l2_error(mesh_ptr, &u, &exact_fn).sqrt()
    };
    println!("\nL2 error: {:.10e}", l2_err);

    // ── 7. Output files ─────────────────────────────────────────────────────
    {
        use fem_io::mfem::write_gf_file;
        use fem_io::mfem::write_mfem_file_3d;
        if let Err(e) = write_mfem_file_3d("sphere_refined.mesh", space.mesh()) {
            eprintln!("  Warning: could not write sphere_refined.mesh: {e}");
        }
        if let Err(e) = write_gf_file("sol.gf", 3, &u, "H1", args.order, 1) {
            eprintln!("  Warning: could not write sol.gf: {e}");
        }
    }

    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mesh construction helpers
// ═══════════════════════════════════════════════════════════════════════════

fn build_octahedron_mesh() -> Mesh<3> {
    let coords = vec![
        1.0,  0.0,  0.0,
        0.0,  1.0,  0.0,
       -1.0,  0.0,  0.0,
        0.0, -1.0,  0.0,
        0.0,  0.0,  1.0,
        0.0,  0.0, -1.0,
    ];
    let conn = vec![
        0, 1, 4,  1, 2, 4,  2, 3, 4,  3, 0, 4,
        1, 0, 5,  2, 1, 5,  3, 2, 5,  0, 3, 5,
    ];
    Mesh {
        coords, conn, elem_tags: (1..=8).collect(),
        elem_type: ElementType::Tri3,
        face_conn: vec![], face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None, nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    }
}

fn build_cube_mesh() -> Mesh<3> {
    let s = 0.5773502691896257_f64; // 1/sqrt(3)
    let coords = vec![
       -s, -s, -s,   s, -s, -s,   s,  s, -s,  -s,  s, -s,
       -s, -s,  s,   s, -s,  s,   s,  s,  s,  -s,  s,  s,
    ];
    let conn = vec![
        3, 2, 1, 0,  0, 1, 5, 4,  1, 2, 6, 5,
        2, 3, 7, 6,  3, 0, 4, 7,  4, 5, 6, 7,
    ];
    Mesh {
        coords, conn, elem_tags: (1..=6).collect(),
        elem_type: ElementType::Quad4,
        face_conn: vec![], face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None, nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    }
}

fn snap_nodes(mesh: &mut Mesh<3>) {
    for n in 0..mesh.n_nodes() as u32 {
        let i = n as usize * 3;
        let (x, y, z) = (mesh.coords[i], mesh.coords[i+1], mesh.coords[i+2]);
        let r = (x*x + y*y + z*z).sqrt();
        mesh.coords[i] = x / r;
        mesh.coords[i+1] = y / r;
        mesh.coords[i+2] = z / r;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tri6 (P2 triangle) helpers
// ═══════════════════════════════════════════════════════════════════════════

fn elevate_to_tri6(mesh: &Mesh<3>) -> Mesh<3> {
    // ... unchanged from original ...
    let ne = mesh.n_elems();
    let tri3_conn = &mesh.conn;
    let n3 = mesh.n_nodes() as u32;
    let mut coords = mesh.coords.clone();
    let mut edge_map = std::collections::HashMap::<(u32, u32), u32>::new();
    let mut next_node = n3;
    let mut new_conn = Vec::with_capacity(ne * 6);

    for e in 0..ne {
        let i = e * 3;
        let a = tri3_conn[i];
        let b = tri3_conn[i + 1];
        let c = tri3_conn[i + 2];
        let key = |x: u32, y: u32| if x < y { (x, y) } else { (y, x) };

        let ab = *edge_map.entry(key(a, b)).or_insert_with(|| {
            let j = next_node; next_node += 1;
            let (xa, ya, za) = (coords[a as usize*3], coords[a as usize*3+1], coords[a as usize*3+2]);
            let (xb, yb, zb) = (coords[b as usize*3], coords[b as usize*3+1], coords[b as usize*3+2]);
            let cx = (xa + xb) / 2.0; let cy = (ya + yb) / 2.0; let cz = (za + zb) / 2.0;
            let r = (cx*cx + cy*cy + cz*cz).sqrt().max(1e-30);
            coords.extend_from_slice(&[cx/r, cy/r, cz/r]);
            j
        });
        let ac = *edge_map.entry(key(a, c)).or_insert_with(|| {
            let j = next_node; next_node += 1;
            /* same pattern */
            let (xa, ya, za) = (coords[a as usize*3], coords[a as usize*3+1], coords[a as usize*3+2]);
            let (xc, yc, zc) = (coords[c as usize*3], coords[c as usize*3+1], coords[c as usize*3+2]);
            let cx = (xa + xc) / 2.0; let cy = (ya + yc) / 2.0; let cz = (za + zc) / 2.0;
            let r = (cx*cx + cy*cy + cz*cz).sqrt().max(1e-30);
            coords.extend_from_slice(&[cx/r, cy/r, cz/r]);
            j
        });
        let bc = *edge_map.entry(key(b, c)).or_insert_with(|| {
            let j = next_node; next_node += 1;
            let (xb, yb, zb) = (coords[b as usize*3], coords[b as usize*3+1], coords[b as usize*3+2]);
            let (xc, yc, zc) = (coords[c as usize*3], coords[c as usize*3+1], coords[c as usize*3+2]);
            let cx = (xb + xc) / 2.0; let cy = (yb + yc) / 2.0; let cz = (zb + zc) / 2.0;
            let r = (cx*cx + cy*cy + cz*cz).sqrt().max(1e-30);
            coords.extend_from_slice(&[cx/r, cy/r, cz/r]);
            j
        });
        new_conn.extend_from_slice(&[a, b, c, ab, bc, ac]);
    }

    Mesh {
        coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tri6,
        face_conn: vec![], face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None, nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    }
}

fn refine_uniform_tri6(mesh: &Mesh<3>) -> Mesh<3> {
    let ne = mesh.n_elems();
    let conn6 = &mesh.conn;
    let mut coords = mesh.coords.clone();
    let mut edge_map = std::collections::HashMap::<(u32, u32), u32>::new();
    let mut next_node = mesh.n_nodes() as u32;
    let mut new_conn = Vec::with_capacity(ne * 24);

    let add_edge = |a: u32, b: u32, coords: &mut Vec<f64>, map: &mut std::collections::HashMap<(u32, u32), u32>, next: &mut u32| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        *map.entry(key).or_insert_with(|| {
            let j = *next; *next += 1;
            let (xa, ya, za) = (coords[a as usize*3], coords[a as usize*3+1], coords[a as usize*3+2]);
            let (xb, yb, zb) = (coords[b as usize*3], coords[b as usize*3+1], coords[b as usize*3+2]);
            let cx = (xa + xb) / 2.0; let cy = (ya + yb) / 2.0; let cz = (za + zb) / 2.0;
            coords.extend_from_slice(&[cx, cy, cz]);
            j
        })
    };

    for e in 0..ne {
        let i = e * 6;
        let (v0, v1, v2) = (conn6[i], conn6[i+1], conn6[i+2]);
        let (m01, m12, m20) = (conn6[i+3], conn6[i+4], conn6[i+5]);

        let a = add_edge(v0, m01, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(v0, m20, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m01, m20, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[v0, m01, m20, a, c, b]);

        let a = add_edge(v1, m12, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(v1, m01, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m12, m01, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[v1, m12, m01, a, c, b]);

        let a = add_edge(v2, m20, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(v2, m12, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m20, m12, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[v2, m20, m12, a, c, b]);

        let a = add_edge(m01, m12, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(m12, m20, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m20, m01, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[m01, m12, m20, a, b, c]);
    }

    Mesh {
        coords, conn: new_conn,
        elem_tags: vec![0; ne * 4],
        elem_type: ElementType::Tri6,
        face_conn: vec![], face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None, nc_vertex_view: None,
        vertex_parents: vec![], nc_leaf_states: None, nc_face_ids: None,
    }
}

//  L² error helpers (one per element type)
// ═══════════════════════════════════════════════════════════════════════════

fn tri3_l2_error(mesh: &Mesh<3>, u: &[f64], exact: &dyn Fn(&[f64; 3]) -> f64) -> f64 {
    
    let mut err2 = 0.0;
    let qpts = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
    let qwt = [1.0/6.0, 1.0/6.0, 1.0/6.0];
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.element_nodes(e);
        let x0 = { let c = mesh.node_coords(ns[0]); [c[0], c[1], c[2]] };
        let x1 = { let c = mesh.node_coords(ns[1]); [c[0], c[1], c[2]] };
        let x2 = { let c = mesh.node_coords(ns[2]); [c[0], c[1], c[2]] };
        let j0 = [x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]];
        let j1 = [x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]];
        let g00 = j0[0]*j0[0]+j0[1]*j0[1]+j0[2]*j0[2];
        let g01 = j0[0]*j1[0]+j0[1]*j1[1]+j0[2]*j1[2];
        let g11 = j1[0]*j1[0]+j1[1]*j1[1]+j1[2]*j1[2];
        let sqrt_det_g = (g00*g11-g01*g01).sqrt().max(1e-30);
        for q in 0..3 {
            let (xi, eta) = (qpts[q][0], qpts[q][1]);
            let phi = [1.0-xi-eta, xi, eta];
            let xp = [phi[0]*x0[0]+phi[1]*x1[0]+phi[2]*x2[0],
                      phi[0]*x0[1]+phi[1]*x1[1]+phi[2]*x2[1],
                      phi[0]*x0[2]+phi[1]*x1[2]+phi[2]*x2[2]];
            let uh = phi[0]*u[ns[0]as usize]+phi[1]*u[ns[1]as usize]+phi[2]*u[ns[2]as usize];
            let ue = exact(&xp);
            err2 += (uh - ue).powi(2) * qwt[q] * sqrt_det_g;
        }
    }
    err2
}

fn quad4_l2_error(mesh: &Mesh<3>, u: &[f64], exact: &dyn Fn(&[f64; 3]) -> f64) -> f64 {
    let mut err2 = 0.0;
    let qpts = [[-0.57735, -0.57735], [0.57735, -0.57735],
                [0.57735,  0.57735], [-0.57735,  0.57735]];
    let qwt = [1.0, 1.0, 1.0, 1.0];
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.element_nodes(e);
        let co = |i| { let c = mesh.node_coords(i); [c[0], c[1], c[2]] };
        let x: [[f64; 3]; 4] = [co(ns[0]), co(ns[1]), co(ns[2]), co(ns[3])];
        let dxi = [(-x[0][0]+x[1][0]+x[2][0]-x[3][0])/4.0,
                   (-x[0][1]+x[1][1]+x[2][1]-x[3][1])/4.0,
                   (-x[0][2]+x[1][2]+x[2][2]-x[3][2])/4.0];
        let deta = [(-x[0][0]-x[1][0]+x[2][0]+x[3][0])/4.0,
                    (-x[0][1]-x[1][1]+x[2][1]+x[3][1])/4.0,
                    (-x[0][2]-x[1][2]+x[2][2]+x[3][2])/4.0];
        let g00 = dxi[0]*dxi[0]+dxi[1]*dxi[1]+dxi[2]*dxi[2];
        let g01 = dxi[0]*deta[0]+dxi[1]*deta[1]+dxi[2]*deta[2];
        let g11 = deta[0]*deta[0]+deta[1]*deta[1]+deta[2]*deta[2];
        let sqrt_det_g = (g00*g11-g01*g01).sqrt().max(1e-30);
        for q in 0..4 {
            let (xi, eta) = (qpts[q][0], qpts[q][1]);
            let phi = [0.25*(1.0-xi)*(1.0-eta), 0.25*(1.0+xi)*(1.0-eta),
                       0.25*(1.0+xi)*(1.0+eta), 0.25*(1.0-xi)*(1.0+eta)];
            let xp = [phi[0]*x[0][0]+phi[1]*x[1][0]+phi[2]*x[2][0]+phi[3]*x[3][0],
                      phi[0]*x[0][1]+phi[1]*x[1][1]+phi[2]*x[2][1]+phi[3]*x[3][1],
                      phi[0]*x[0][2]+phi[1]*x[1][2]+phi[2]*x[2][2]+phi[3]*x[3][2]];
            let uh = phi[0]*u[ns[0]as usize]+phi[1]*u[ns[1]as usize]
                   + phi[2]*u[ns[2]as usize]+phi[3]*u[ns[3]as usize];
            let ue = exact(&xp);
            err2 += (uh-ue).powi(2) * qwt[q] * sqrt_det_g;
        }
    }
    err2
}

fn quad9_l2_error(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>, u: &[f64],
                  exact: &dyn Fn(&[f64; 3]) -> f64) -> f64 {
    use fem_assembly::boundary::surface::q2_coords_from_quad4;
    let mut err2 = 0.0;
    for e in 0..mesh.n_elems() as u32 {
        let dofs = space.element_dofs(e);
        if dofs.len() < 9 { continue; }
        let ns = mesh.element_nodes(e);
        let x = q2_coords_from_quad4(mesh, ns);
        // MFEM ComputeL2Error uses 2·GetOrder()+3 = 7th-order quadrature →
        // 4×4 Gauss-Legendre.  The 3×3 rule under-integrates (u_h − u)².
        for q in 0..16 {
            let (xi, eta) = fem_assembly::boundary::surface::q4_quad_point(q);
            let w   = fem_assembly::boundary::surface::q4_quad_weight(q);
            let jac = fem_assembly::boundary::surface::q2_jacobian_at(&x, xi, eta);
            let g00 = jac[0][0]*jac[0][0] + jac[0][1]*jac[0][1] + jac[0][2]*jac[0][2];
            let g01 = jac[0][0]*jac[1][0] + jac[0][1]*jac[1][1] + jac[0][2]*jac[1][2];
            let g11 = jac[1][0]*jac[1][0] + jac[1][1]*jac[1][1] + jac[1][2]*jac[1][2];
            let sqrt_det_g = (g00*g11 - g01*g01).sqrt().max(1e-30);
            let phi = fem_assembly::boundary::surface::q2_basis(xi, eta);
            let xp = [
                (0..9).map(|i| phi[i] * x[i][0]).sum::<f64>(),
                (0..9).map(|i| phi[i] * x[i][1]).sum::<f64>(),
                (0..9).map(|i| phi[i] * x[i][2]).sum::<f64>(),
            ];
            let uh = (0..9).map(|i| phi[i] * u[dofs[i] as usize]).sum::<f64>();
            let ue = exact(&xp);
            err2 += (uh - ue).powi(2) * w * sqrt_det_g;
        }
    }
    err2
}

// ═══════════════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════════════

struct Args {
    ref_levels: usize,
    order: u8,
    elem_type: u8,
    #[allow(dead_code)]
    always_snap: bool,
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
                }
                "-e" | "--elem" => {
                    elem_type = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                }
                "-snap" | "--always-snap" => {
                    always_snap = true;
                }
                "-amr" | "--refine-locally" => {
                    amr = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                }
                "-no-vis" | "--no-visualization" => {
                    no_vis = true;
                }
                _ => {}
            }
        }
        Args { ref_levels, order, elem_type, always_snap, amr, no_vis }
    }
}
