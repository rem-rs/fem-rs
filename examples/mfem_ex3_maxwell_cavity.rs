//! # Example 3 — Maxwell Electromagnetic Diffusion  (one-to-one with MFEM ex3)
//!
//! Solves the second-order definite Maxwell problem:
//!
//! ```text
//!   ∇×(∇×E) + E = f    in Ω
//!          n×E = 0    on ∂Ω
//! ```
//!
//! with a manufactured source `f = (1+κ²)·(sin(κy), sin(κx))` where `κ = π·freq`.
//! The exact solution is `E = (sin(κy), sin(κx))`.  Discretisation uses
//! Nédélec (H(curl)) edge elements.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex3_maxwell_cavity
//! cargo run --example mfem_ex3_maxwell_cavity -- -m ../data/star.mesh
//! cargo run --example mfem_ex3_maxwell_cavity -- -m ../data/star.mesh -o 2
//! cargo run --example mfem_ex3_maxwell_cavity -- -f 2.0
//! cargo run --example mfem_ex3_maxwell_cavity -- -no-vis
//! ```
//!
//! ## Output
//! Prints DOF count, linear system size, solver statistics, and L² error.
//! Writes `refined.mesh` and `sol.gf` (matching MFEM ex3 output files).

use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use fem_assembly::{
    VectorAssembler, DiscreteLinearOperator,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_element::{nedelec::TriND1, VectorReferenceElement};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_space::{
    HCurlSpace,
    fe_space::FESpace,
    constraints::{boundary_dofs_hcurl, eliminate_dirichlet},
};

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("(built-in unit square)"));
    println!("   --order {}", args.order);
    println!("   --frequency {}", args.freq);
    if args.static_cond {
        println!("   --static-condensation");
    } else {
        println!("   --no-static-condensation");
    }
    if args.visualization {
        println!("   --visualization");
    } else {
        println!("   --no-visualization");
    }

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class yet).

    // 3. Read the mesh from the given mesh file.
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(16)
    };
    let dim = 2;

    // 4. Uniform refinement: choose levels so the final mesh has ≤ 50 000 elements.
    //    (Matching MFEM ex3's refinement target.)
    let ref_levels =
        ((50000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels {
            m = refine_uniform(&m);
        }
        m
    } else {
        mesh
    };

    // 5. H(curl) Nédélec finite element space of the specified order.
    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    // 6. Essential (PEC) boundary DOFs — all external boundaries.
    //    n×E = 0  →  tangential component vanishes.
    let all_tags: Vec<i32> = space.mesh().unique_boundary_tags();
    let ess_bdr = if all_tags.is_empty() {
        vec![]
    } else {
        boundary_dofs_hcurl(space.mesh(), &space, &all_tags)
    };
    eprintln!("  Boundary DOFs: {} / {}", ess_bdr.len(), n_dofs);
    // 7. Right-hand side: b(v) = ∫ f·v dx  where
    //    f = (1+κ²)·(sin(κy), sin(κx)).
    let kappa = args.freq * PI;
    let source = MaxwellSource { kappa };
    let quad_order = args.order as u8 * 2 + 2;
    let rhs = VectorAssembler::assemble_linear(&space, &[&source], quad_order);

    // 8. Solution vector x — zero initial guess (will be set by Dirichlet below).

    // 9. Stiffness matrix: a(u, v) = ∫ (∇×u)·(∇×v) + u·v dx.
    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    let mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl, &vec_mass], quad_order,
    );
    print!("Assembling: matrix ... ");

    // 10. Form the linear system (MFEM-style elimination).
    if args.static_cond {
        eprintln!("  Warning: static condensation not yet implemented — skipping.");
    }
    println!("done.");

    // 11. Solve with PCG + GSSmoother (default) or PCG + AMS (--ams flag).
    //     Use the reduced system from eliminate_dirichlet (matches MFEM ex3 path).
    // Keep originals for AMS path (which needs the full, unmodified matrix).
    let ams_path = args.use_ams;
    let full_mat = if ams_path { Some(mat.clone()) } else { None };
    let full_rhs: Vec<f64> = if ams_path { rhs.clone() } else { Vec::new() };
    let mut x_full: Vec<f64> = Vec::new();

    let (sys_mat, sys_rhs, free_map, constrained_map, bnd_vals, linlvo_sys) = if !ess_bdr.is_empty() {
        let bv = vec![0.0_f64; ess_bdr.len()];
        let (rm, rf, fm, cm) = eliminate_dirichlet(&mat, &rhs, &ess_bdr, &bv);
        let lsys = fem_linalg::fem_to_linlvo_csr(&rm);
        (rm, rf, fm, cm, bv, lsys)
    } else {
        let free: Vec<usize> = (0..n_dofs).collect();
        let lsys = fem_linalg::fem_to_linlvo_csr(&mat);
        (mat, rhs, free, vec![], vec![], lsys)
    };
    let n_sys = sys_mat.nrows;
    println!("  Reduced system size: {n_sys}");

    let mut x_red = vec![0.0_f64; n_sys];

    if ams_path {
        use fem_solver::{solve_pcg_ams, AmsSolverConfig, AmsConfig};
        use fem_linalg::fem_to_linlvo_csr as ftl;
        let g = DiscreteLinearOperator::gradient(
            &fem_space::H1Space::new(space.mesh().clone(), 1),
            &space,
        ).expect("gradient assembly failed");
        let g_linlvo = ftl(&g);
        let mut ams_mat = full_mat.unwrap();
        for &d in &ess_bdr {
            ams_mat.eliminate_essential_bc_diag(d as usize, 1.0);
        }
        x_full.resize(n_dofs, 0.0);
        let result = solve_pcg_ams(&ams_mat, &g_linlvo, &full_rhs, &mut x_full, &AmsSolverConfig {
            inner_cfg: fem_solver::SolverConfig {
                rtol: 1e-12, atol: 1e-20, max_iter: 2000, verbose: true,
                ..fem_solver::SolverConfig::default()
            },
            ams_cfg: AmsConfig::default(),
        }).expect("PCG+AMS solve failed");
        println!("PCG+AMS: {} iters, ||r||/||b|| = {:.3e}",
            result.iterations, result.final_residual);
    } else {
        let precond = fem_solver::GSSmoother::from_csr(&linlvo_sys, 1.0)
            .expect("GSSmoother setup");
        let result = fem_solver::solve_pcg(
            &sys_mat, &sys_rhs, &mut x_red, &precond,
            1e-12, 2000, true,
        ).expect("PCG+GSSmoother solve failed");
        println!("PCG+GSSmoother: {} iters, ||r||/||b|| = {:.3e}",
            result.iterations, result.final_residual);
    }

    // Recover full solution.
    let u: Vec<f64> = if ams_path {
        x_full
    } else if !ess_bdr.is_empty() {
        fem_space::constraints::expand_from_reduced(&x_red, &free_map, &constrained_map, &bnd_vals, n_dofs)
    } else {
        x_red
    };

    // 13. Compute and print the L² norm of the error against the exact solution.
    let l2_err = fem_examples::maxwell::l2_error_hcurl_exact(
        &space, &u, |x| exact_e(x, kappa),
    );
    println!("\n|| E_h - E ||_{{L^2}} = {l2_err:.14e}\n");

    // 14. Save the refined mesh and the solution (matches MFEM ex3 output files).
    {
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, space.mesh(), None).expect("mesh write failed");
        let mut sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        for &v in &u {
            writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
        }
        eprintln!("  Wrote refined.mesh and sol.gf");
    }

    // 15. Send a nodal-projected view of the solution to GLVis.
    if args.visualization {
        match send_to_glvis(space.mesh(), &space, &u, "E") {
            Ok(_) => eprintln!("  Sent solution to GLVis (localhost:19916)"),
            Err(e) => eprintln!("  GLVis not available: {e}"),
        }
    }
}

// ─── Source term (VectorLinearIntegrator) ─────────────────────────────────────
//
//   f = (1 + κ²) · (sin(κy), sin(κx))

struct MaxwellSource {
    kappa: f64,
}

impl VectorLinearIntegrator for MaxwellSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        let coeff = 1.0 + self.kappa * self.kappa;
        let fx = coeff * (self.kappa * x[1]).sin();
        let fy = coeff * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

// ─── Exact solution ──────────────────────────────────────────────────────────
//
//   E = (sin(κy), sin(κx))

fn exact_e(x: &[f64], kappa: f64) -> [f64; 2] {
    [(kappa * x[1]).sin(), (kappa * x[0]).sin()]
}

// ─── GLVis helper ────────────────────────────────────────────────────────────
//
// H(curl) DOFs live on edges, so we project the edge solution onto mesh
// vertices by evaluating the field inside each element at its reference
// vertices and averaging at shared nodes.  The result is a nodal vector
// field that GLVis (VTK-based protocol) can display.

fn send_to_glvis(
    mesh: &Mesh<2>,
    space: &HCurlSpace<Mesh<2>>,
    u: &[f64],
    field_name: &str,
) -> std::io::Result<()> {
    let n_nodes = mesh.n_nodes() as usize;
    let ref_elem = TriND1;
    let n_ldofs = ref_elem.n_dofs();
    let dim = 2usize;

    // Reference-vertex coordinates for the Tri3 reference element.
    let ref_verts: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

    let mut sum_x = vec![0.0_f64; n_nodes];
    let mut sum_y = vec![0.0_f64; n_nodes];
    let mut count = vec![0u32; n_nodes];
    let mut ref_phi = vec![0.0_f64; n_ldofs * dim];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);

        // Affine Jacobian J = [x1-x0, x2-x0],  J^{-T} = adj(J)^T / det(J).
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = j00 * j11 - j01 * j10;
        let inv_det = 1.0 / det_j;
        let jit00 =  j11 * inv_det;
        let jit01 = -j10 * inv_det;
        let jit10 = -j01 * inv_det;
        let jit11 =  j00 * inv_det;

        // Evaluate at each of the three reference vertices.
        for vi in 0..3 {
            let xi = &ref_verts[vi];
            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            // Covariant Piola:  φ_phys = J^{-T} φ_ref.
            let mut eh_x = 0.0_f64;
            let mut eh_y = 0.0_f64;
            for i in 0..n_ldofs {
                let s = signs[i];
                let px = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                let py = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                eh_x += s * u[dofs[i]] * px;
                eh_y += s * u[dofs[i]] * py;
            }

            let nid = nodes[vi] as usize;
            sum_x[nid] += eh_x;
            sum_y[nid] += eh_y;
            count[nid] += 1;
        }
    }

    // Average contributions at shared nodes.
    let mut e_node_x = vec![0.0_f64; n_nodes];
    let mut e_node_y = vec![0.0_f64; n_nodes];
    for i in 0..n_nodes {
        if count[i] > 0 {
            let inv = 1.0 / count[i] as f64;
            e_node_x[i] = sum_x[i] * inv;
            e_node_y[i] = sum_y[i] * inv;
        }
    }

    let mut sock = fem_io::glvis::GlVisSocket::connect("localhost", 19916)?;
    sock.send_solution_2d_vector(mesh, &e_node_x, &e_node_y, field_name)?;
    Ok(())
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Args {
    mesh:          Option<String>,
    n:             usize,
    order:         u8,
    /// Static condensation (not yet implemented).
    static_cond:   bool,
    visualization: bool,
    freq:          f64,
    /// Use PCG+AMS preconditioner instead of PCG+GSSmoother.
    use_ams:       bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh:          None,
        n:             16,
        order:         1,
        static_cond:   false,
        visualization: true,
        freq:          1.0,
        use_ams:       false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next();
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1);
            }
            "-f" | "--frequency" => {
                a.freq = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1.0);
            }
            "-sc" | "--static-condensation" => {
                a.static_cond = true;
            }
            "-no-sc" | "--no-static-condensation" => {
                a.static_cond = false;
            }
            "-ams" | "--ams" => {
                a.use_ams = true;
            }
            "-vis" | "--visualization" => {
                a.visualization = true;
            }
            "-no-vis" | "--no-visualization" => {
                a.visualization = false;
            }
            _ => {}
        }
    }
    a
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_space::constraints::expand_from_reduced;

    fn solve_case(args: &Args) -> (Vec<f64>, usize, f64) {
        let mesh = if let Some(ref path) = args.mesh {
            let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
            mfem.mesh2d.expect("MFEM mesh must be 2D")
        } else {
            Mesh::<2>::unit_square_tri(args.n)
        };

        let space = HCurlSpace::new(mesh, args.order);
        let n_dofs = space.n_dofs();

        let all_tags: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess_bdr = if all_tags.is_empty() {
            vec![]
        } else {
            boundary_dofs_hcurl(space.mesh(), &space, &all_tags)
        };
        let kappa = args.freq * PI;
        let source = MaxwellSource { kappa };
        let quad_order = args.order as u8 * 2 + 2;
        let rhs = VectorAssembler::assemble_linear(&space, &[&source], quad_order);

        let curl_curl = CurlCurlIntegrator { mu: 1.0 };
        let vec_mass = VectorMassIntegrator { alpha: 1.0 };
        let mat = VectorAssembler::assemble_bilinear(
            &space, &[&curl_curl, &vec_mass], quad_order,
        );

        let (sys_mat, sys_rhs, free_map, constrained_map) = if !ess_bdr.is_empty() {
            let bv = vec![0.0_f64; ess_bdr.len()];
            eliminate_dirichlet(&mat, &rhs, &ess_bdr, &bv)
        } else {
            let free: Vec<usize> = (0..n_dofs).collect();
            (mat, rhs, free, vec![])
        };

        let bnd_vals = vec![0.0_f64; constrained_map.len()];

        let n_sys = sys_mat.nrows;
        let linlvo_sys = fem_linalg::fem_to_linlvo_csr(&sys_mat);
        let precond = fem_solver::GSSmoother::from_csr(&linlvo_sys, 1.0)
            .expect("GSSmoother setup failed");
        let mut x_red = vec![0.0_f64; n_sys];
        fem_solver::solve_pcg(&sys_mat, &sys_rhs, &mut x_red, &precond, 1e-12, 500, false)
            .expect("PCG solve failed");

        let u = if !ess_bdr.is_empty() {
            expand_from_reduced(&x_red, &free_map, &constrained_map, &bnd_vals, n_dofs)
        } else {
            x_red
        };

        let l2_err = fem_examples::maxwell::l2_error_hcurl_exact(
            &space, &u, |x| exact_e(x, kappa),
        );

        (u, n_dofs, l2_err)
    }

    fn default_args() -> Args {
        Args {
            mesh:          None,
            n:             8,
            order:         1,
            static_cond:   false,
            visualization: false,
            freq:          1.0,
            use_ams:       false,
        }
    }

    // ── Behavioural tests ────────────────────────────────────────────────

    #[test]
    fn ex3_dof_count() {
        let args = default_args();
        let (_, n_dofs, _) = solve_case(&args);
        assert_eq!(
            n_dofs, 208,
            "DOF count should be 208 for ND1 on 8×8 tri mesh"
        );
    }

    #[test]
    fn ex3_convergence_on_refinement() {
        let coarse = solve_case(&Args { n: 6, ..default_args() });
        let fine   = solve_case(&Args { n: 12, ..default_args() });

        assert!(coarse.2.is_finite() && fine.2.is_finite());
        assert!(
            fine.2 < coarse.2,
            "expected refinement to reduce L² error: coarse={:.6e} fine={:.6e}",
            coarse.2, fine.2,
        );

        let h6 = 1.0 / 6.0;
        let h12 = 1.0 / 12.0;
        let rate = f64::ln(coarse.2 / fine.2) / f64::ln(h6 / h12);
        eprintln!(
            "  [ex3] L²(6)={:.6e}  L²(12)={:.6e}  rate={:.3} (expected ~1)",
            coarse.2, fine.2, rate,
        );
        assert!(rate > 0.5, "convergence rate {:.2} too low", rate);
    }

    // ── Regression baseline ──────────────────────────────────────────────

    #[test]
    fn ex3_regression_baseline() {
        let args = Args {
            n: 8, ..default_args()
        };
        let (_, n_dofs, l2_err) = solve_case(&args);

        fem_regression::regression("mfem_ex3_maxwell_cavity")
            .check_with("l2_error", l2_err,   1e-6, 1e-10)
            .check_with("n_dofs",   n_dofs as f64, 0.0, 0.5)
            .finalize();
    }
}
