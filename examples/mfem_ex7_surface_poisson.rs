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
    SurfaceAssembler, SurfaceDiffusionIntegrator, SurfaceDomainSourceIntegrator,
    SurfaceMassIntegrator,
    SurfaceQuad4Assembler, SurfaceQuad4DiffusionIntegrator,
    SurfaceQuad4DomainSourceIntegrator, SurfaceQuad4MassIntegrator,
};
use fem_mesh::{
    Mesh, MeshTopology, element_type::ElementType,
    amr::{refine_at_vertex_surface, refine_uniform_surface_quad4, refine_uniform_surface_tri3},
};
use fem_solver::{fem_to_linlvo_csr, solve_pcg, SolveResult};
use fem_space::{H1Space, fe_space::FESpace};
use linlvo::SsorPrecond;

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
    for l in 0..=args.ref_levels {
        if l > 0 {
            mesh = if is_quad {
                refine_uniform_surface_quad4(&mesh)
            } else {
                refine_uniform_surface_tri3(&mesh)
            };
        }
        if args.always_snap || l == args.ref_levels {
            snap_nodes(&mut mesh);
        }
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

    // ── 2. Define H1 space (order 1) ─────────────────────────────────────────
    let order = args.order.min(1);
    let space = H1Space::new(mesh, order);
    let n_dofs = space.n_dofs();
    println!("Number of unknowns: {}", n_dofs);

    // ── 3. Assemble surface stiffness (-Delta_Gamma) + mass (+u) ────────────
    let rhs_fn = &|x: &[f64; 3]| {
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        7.0 * x[0] * x[1] / r2
    };

    let (mut a, mass_mat) = if is_quad {
        let k = SurfaceQuad4Assembler::assemble_bilinear(&space, &|x, ke| {
            SurfaceQuad4DiffusionIntegrator.add_to_element_matrix(x, ke);
        });
        let m = SurfaceQuad4Assembler::assemble_bilinear(&space, &|x, ke| {
            SurfaceQuad4MassIntegrator.add_to_element_matrix(x, ke);
        });
        (k, m)
    } else {
        let k = SurfaceAssembler::assemble_bilinear(&space, &|x, ke| {
            SurfaceDiffusionIntegrator.add_to_element_matrix(x, ke);
        });
        let m = SurfaceAssembler::assemble_bilinear(&space, &|x, ke| {
            SurfaceMassIntegrator.add_to_element_matrix(x, ke);
        });
        (k, m)
    };
    for i in 0..a.nrows {
        for jp in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[jp] as usize;
            a.values[jp] += mass_mat.get(i, j);
        }
    }

    // ── 4. Assemble RHS: f = 7*x*y / r^2 ────────────────────────────────────
    let rhs = if is_quad {
        SurfaceQuad4Assembler::assemble_linear(&space, &|x, fe| {
            SurfaceQuad4DomainSourceIntegrator { f: rhs_fn }.add_to_element_vector(x, fe);
        })
    } else {
        SurfaceAssembler::assemble_linear(&space, &|x, fe| {
            SurfaceDomainSourceIntegrator { f: rhs_fn }.add_to_element_vector(x, fe);
        })
    };

    // ── 5. Solve: PCG + SSOR(omega=1) ──────────────────────────────────────
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
        if let Err(e) = write_gf_file("sol.gf", 3, &u, "H1", order as u8, 1) {
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
        edge_conn: vec![], edge_to_elem: vec![],
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
        edge_conn: vec![], edge_to_elem: vec![],
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
                    if order > 1 {
                        eprintln!("  Warning: order > 1 not yet supported; using order 1");
                    }
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
