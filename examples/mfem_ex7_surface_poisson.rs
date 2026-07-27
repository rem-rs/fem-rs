//! # MFEM Example 7 — Screened Poisson on the Unit Sphere
//!
//! Solves `-Δu + u = f` on the unit sphere surface with `f = 7·x·y / r²`,
//! exact solution `u = x·y / r²`.  Demonstrates surface FEM on a 2-D manifold
//! embedded in 3-D space using `Mesh<3>` with Tri3 or Quad4 elements.
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
//! | `-o/--order` | 2 | FE order (only 1 supported) |
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
        }
        snap_nodes(&mut mesh);
    }

    // AMR: refine near north pole (MFEM ex7 -amr 1)
    for _ in 0..args.amr {
        mesh = refine_at_vertex_surface(&mesh, &[0.0, 0.0, 1.0]);
        snap_nodes(&mut mesh);
    }

    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();
    let elem_name = if is_quad { "quads" } else { "triangles" };
    eprintln!("  Mesh: {} nodes, {} {} on unit sphere", n_nodes, n_elems, elem_name);

    // ── 3. Define H1 space ──────────────────────────────────────────────────
    // For Tri6 mesh, use order=1 (DOFs = mesh nodes, assembly uses P2 bases)
    let h1_order = if use_tri6 { 1 } else { args.order };
    let space = H1Space::new(mesh, h1_order);
    let n_dofs = space.n_dofs();
    println!("Number of unknowns: {}", n_dofs);

    // ── 4. Assemble surface stiffness (-Delta_Gamma) + mass (+u) ────────────
    let rhs_fn = &|x: &[f64; 3]| {
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        7.0 * x[0] * x[1] / r2
    };

    let a: CsrMatrix<f64> = if is_quad {
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
    let rhs: Vec<f64> = if is_quad {
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

    // ── 5. Solve: PCG + SSOR(omega=1) ──────────────────────────────────────
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
    let mesh_3 = space.mesh();
    let mut err2 = 0.0_f64;
    if is_quad {
        let qpts = [[-0.57735, -0.57735], [0.57735, -0.57735],
                    [0.57735,  0.57735], [-0.57735,  0.57735]];
        let qwt = [1.0, 1.0, 1.0, 1.0];
        for e in 0..mesh_3.n_elems() as u32 {
            let ns = mesh_3.element_nodes(e);
            let x = [mesh_3.node_coords(ns[0]), mesh_3.node_coords(ns[1]),
                     mesh_3.node_coords(ns[2]), mesh_3.node_coords(ns[3])];
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
                let ue = exact_fn(&xp);
                let diff = uh - ue;
                err2 += diff*diff*qwt[q]*sqrt_det_g;
            }
        }
    } else if use_tri6 {
        // P2 (Tri6) L² error — 3-point quadrature, P2 basis
        for e in 0..mesh_3.n_elems() as u32 {
            let ns = mesh_3.element_nodes(e);
            let x: [[f64; 3]; 6] = {
                let mut arr = [[0.0; 3]; 6];
                for i in 0..6 { let c = mesh_3.node_coords(ns[i]); arr[i] = [c[0], c[1], c[2]]; }
                arr
            };
            let (j, sqrt_det_g, _) = fem_assembly::boundary::surface_tri6::surface_jacobian_tri6(&x);
            let _ = j;
            for q in 0..3 {
                let (xi, eta) = match q { 0 => (2./3.,1./6.), 1 => (1./6.,2./3.), _ => (1./6.,1./6.) };
                let phi = fem_assembly::boundary::surface_tri6::p2_basis_tri6(xi, eta);
                let xp = [
                    phi.iter().zip(ns.iter()).map(|(&p, &n)| p * mesh_3.node_coords(n)[0]).sum::<f64>(),
                    phi.iter().zip(ns.iter()).map(|(&p, &n)| p * mesh_3.node_coords(n)[1]).sum::<f64>(),
                    phi.iter().zip(ns.iter()).map(|(&p, &n)| p * mesh_3.node_coords(n)[2]).sum::<f64>(),
                ];
                let uh = phi.iter().zip(ns.iter()).map(|(&p, &n)| p * u[n as usize]).sum::<f64>();
                let ue = exact_fn(&xp);
                let diff = uh - ue;
                err2 += diff*diff*(1.0/6.0)*sqrt_det_g;
            }
        }
    } else {
        let qpts_tri = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];
        let qwt_tri = [1.0/6.0, 1.0/6.0, 1.0/6.0];
        for e in 0..mesh_3.n_elems() as u32 {
            let ns = mesh_3.element_nodes(e);
            let x0 = mesh_3.node_coords(ns[0]);
            let x1 = mesh_3.node_coords(ns[1]);
            let x2 = mesh_3.node_coords(ns[2]);
            let j0 = [x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]];
            let j1 = [x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]];
            let g00 = j0[0]*j0[0]+j0[1]*j0[1]+j0[2]*j0[2];
            let g01 = j0[0]*j1[0]+j0[1]*j1[1]+j0[2]*j1[2];
            let g11 = j1[0]*j1[0]+j1[1]*j1[1]+j1[2]*j1[2];
            let sqrt_det_g = (g00*g11-g01*g01).sqrt().max(1e-30);
            for q in 0..3 {
                let (xi, eta) = (qpts_tri[q][0], qpts_tri[q][1]);
                let phi = [1.0-xi-eta, xi, eta];
                let xp = [phi[0]*x0[0]+phi[1]*x1[0]+phi[2]*x2[0],
                          phi[0]*x0[1]+phi[1]*x1[1]+phi[2]*x2[1],
                          phi[0]*x0[2]+phi[1]*x1[2]+phi[2]*x2[2]];
                let uh = phi[0]*u[ns[0]as usize]+phi[1]*u[ns[1]as usize]+phi[2]*u[ns[2]as usize];
                let ue = exact_fn(&xp);
                let diff = uh - ue;
                err2 += diff*diff*qwt_tri[q]*sqrt_det_g;
            }
        }
    }
    let l2_err = err2.sqrt();
    println!("\nL2 norm of error: {:.10e}", l2_err);

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

// ─── Octahedron mesh (Tri3) ─────────────────────────────────────────────────

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
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
    }
}

// ─── Elevate Tri3 → Tri6 (add mid-edge nodes snapped to sphere) ─────────────

fn elevate_to_tri6(mesh: &Mesh<3>) -> Mesh<3> {
    let ne = mesh.n_elems();
    let tri3_conn = &mesh.conn; // 3 indices per element
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
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
    }
}

// ─── Refine Tri6 mesh (each element → 4 children) ─────────────────────────

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

        // 4 sub-triangles, each gets 6 Tri6 nodes (3 old + 3 new edge midpoints)
        // Child 0: (v0, m01, m20) → new edges: v0-m01, v0-m20, m01-m20
        let a = add_edge(v0, m01, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(v0, m20, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m01, m20, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[v0, m01, m20, a, c, b]);

        // Child 1: (v1, m12, m01) → new edges: v1-m12, v1-m01, m12-m01
        let a = add_edge(v1, m12, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(v1, m01, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m12, m01, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[v1, m12, m01, a, c, b]);

        // Child 2: (v2, m20, m12) → new edges: v2-m20, v2-m12, m20-m12
        let a = add_edge(v2, m20, &mut coords, &mut edge_map, &mut next_node);
        let b = add_edge(v2, m12, &mut coords, &mut edge_map, &mut next_node);
        let c = add_edge(m20, m12, &mut coords, &mut edge_map, &mut next_node);
        new_conn.extend_from_slice(&[v2, m20, m12, a, c, b]);

        // Child 3: (m01, m12, m20) → new edges: m01-m12, m12-m20, m20-m01
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
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
    }
}

// ─── Cube mesh (Quad4) ──────────────────────────────────────────────────────

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
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
    }
}

// ─── Snap nodes to unit sphere ──────────────────────────────────────────────

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

// ─── CLI ─────────────────────────────────────────────────────────────────────

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
