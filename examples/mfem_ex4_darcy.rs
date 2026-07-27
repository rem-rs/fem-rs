//! # Example 4 — H(div) Diffusion / Darcy  (one-to-one with MFEM ex4)
//!
//! Solves the second-order definite H(div) problem:
//!
//! ```text
//!   -∇(α ∇·F) + β F = f    in Ω
//!              F·n = 0     on ∂Ω
//! ```
//!
//! with a manufactured source derived from the exact solution
//! `F = (cos(κy)sin(κx), cos(κx)sin(κy))` where `κ = π·freq`.
//! Discretisation uses Raviart-Thomas H(div) elements.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex4_darcy
//! cargo run --example mfem_ex4_darcy -- -m ../data/star.mesh
//! cargo run --example mfem_ex4_darcy -- -m ../data/star.mesh -o 2
//! cargo run --example mfem_ex4_darcy -- -f 2.0
//! cargo run --example mfem_ex4_darcy -- -no-bc -no-vis
//! ```
//!
//! ## Output
//! Prints DOF count, linear system size, solver statistics, and L² error.
//! Writes `refined.mesh` and `sol.gf` (matching MFEM ex4 output files).

use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

use fem_assembly::{
    VectorAssembler,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    standard::{GradDivIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_solver::{SolverConfig};
use fem_space::{
    HDivSpace,
    fe_space::FESpace,
    constraints::{boundary_dofs_hdiv, form_linear_system},
};

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("../data/star.mesh"));
    println!("   --order {}", args.order);
    if args.set_bc {
        println!("   --impose-bc");
    } else {
        println!("   --dont-impose-bc");
    }
    println!("   --frequency {}", args.freq);
    if args.static_cond {
        println!("   --static-condensation");
    } else {
        println!("   --no-static-condensation");
    }
    if args.hybridization {
        println!("   --hybridization");
    } else {
        println!("   --no-hybridization");
    }
    if args.visualization {
        println!("   --visualization");
    } else {
        println!("   --no-visualization");
    }

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class yet).

    // 3. Read the mesh from the given mesh file.
    let mesh_path = args.mesh.as_deref().unwrap_or("../data/star.mesh");
    let mfem = read_mfem_file(mesh_path).expect("failed to read MFEM mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("MFEM mesh must be 2D");
    let dim = 2;

    // 4. Uniform refinement: choose levels so the final mesh has ≤ 25 000 elements.
    //    (Matching MFEM ex4's refinement target.)
    let ref_levels =
        ((25000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels {
            m = refine_uniform(&m);
        }
        m
    } else {
        mesh
    };

    // 5. H(div) Raviart-Thomas finite element space of order (args.order - 1).
    //    MFEM's RT_FECollection(order-1, dim) → RT0 for order=1, RT1 for order=2.
    let rt_order = if args.order >= 1 { args.order - 1 } else { 0 };
    let space = HDivSpace::new(mesh, rt_order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    // 6. Essential (Dirichlet) boundary DOFs — all external boundaries.
    //    BC: F·n = <projected exact normal component> (or 0 when -no-bc).
    let all_tags: Vec<i32> = space.mesh().unique_boundary_tags();
    let ess_bdr = if args.set_bc && !all_tags.is_empty() {
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags)
    } else {
        vec![]
    };
    let kappa = args.freq * PI;

    // 7. Right-hand side: b(v) = ∫ f·v dx  where
    //    f = (1+2κ²)(cos(κy)sin(κx), cos(κx)sin(κy)).
    let source = MaxwellHSource { kappa };
    let quad_order = args.order * 2 + 2;
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&source], quad_order);

    // 8. Solution vector x — zero initial guess (will be set by Dirichlet below).

    // 9. Stiffness matrix: a(u, v) = ∫ α (∇·u)(∇·v) + β u·v dx.
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    let mut mat = VectorAssembler::assemble_bilinear(
        &space, &[&grad_div, &vec_mass], quad_order,
    );
    print!("Assembling: matrix ... ");

    // 10. Form the linear system.
    if args.static_cond {
        eprintln!("  Warning: static condensation not yet implemented — skipping.");
    }
    println!("done.");

    let (u, solver_label) = if args.hybridization {
        // ── Hybridization path ─────────────────────────────────────────────
        use fem_assembly::hybridization::Hybridization;
        use fem_assembly::vector_assembler::accumulate_vector_bilinear_element;
        use fem_assembly::vector_integrator::VectorBilinearIntegrator;
        use fem_linalg::CooMatrix;

        // Project exact solution onto boundary DOFs.
        // Only boundary DOF values are used (interior DOFs set to 0).
        let x_bc_proj = if !ess_bdr.is_empty() {
            let mut xp = space.interpolate_vector(&|p| {
                let k = kappa;
                vec![(k * p[1]).cos() * (k * p[0]).sin(),
                     (k * p[0]).cos() * (k * p[1]).sin()]
            });
            // Zero out interior DOFs; keep only essential (boundary) DOF values.
            let ess_set: std::collections::HashSet<u32> = ess_bdr.iter().copied().collect();
            for i in 0..n_dofs {
                if !ess_set.contains(&(i as u32)) {
                    xp[i] = 0.0;
                }
            }
            xp
        } else {
            fem_linalg::Vector::from_vec(vec![0.0; n_dofs])
        };

        // Build hybridization.
        let mut hyb = Hybridization::new();
        hyb.init(space.mesh(), &space, &ess_bdr);

        // Build per-element matrices.
        let integrators: &[&dyn VectorBilinearIntegrator] = &[&grad_div, &vec_mass];
        let elem_dofs: Vec<&[u32]> = (0..space.mesh().n_elements() as u32)
            .map(|e| space.element_dofs(e))
            .collect();
        let n_elems = space.mesh().n_elements() as usize;
        let mut elem_rhs_list: Vec<Vec<f64>> = Vec::with_capacity(n_elems);

        // b_hat = R^T · b_rhs: signed assembled RHS for each element.
        let elem_signs: Vec<&[f64]> = (0..n_elems as u32)
            .map(|e| space.element_signs(e))
            .collect();

        for e in 0..n_elems as u32 {
            let e_dofs = space.element_dofs(e);
            let n_ldofs = e_dofs.len();
            let sgn = elem_signs[e as usize];

            // Element matrix (signed).
            let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
            accumulate_vector_bilinear_element(&space, e, integrators, quad_order, &mut coo);
            let elem_csr = coo.into_csr();
            let mut elem_mat = vec![0.0; n_ldofs * n_ldofs];
            for (li, &gi) in e_dofs.iter().enumerate() {
                let gi = gi as usize;
                for k in elem_csr.row_ptr[gi]..elem_csr.row_ptr[gi + 1] {
                    let gj = elem_csr.col_idx[k] as usize;
                    if let Some(lj) = e_dofs.iter().position(|&d| d as usize == gj) {
                        elem_mat[li * n_ldofs + lj] = elem_csr.values[k];
                    }
                }
            }
            hyb.assemble_element_matrix(e, &elem_mat);

            // b_hat = R^T · b = s_i · rhs[gi]  (signed assembled RHS).
            let mut elem_rhs = vec![0.0; n_ldofs];
            for (li, &gi) in e_dofs.iter().enumerate() {
                let s = if li < sgn.len() { sgn[li] } else { 1.0 };
                elem_rhs[li] = s * rhs[gi as usize];
            }

            // Subtract A_e · x_bc for non-homogeneous BCs.
            if !ess_bdr.is_empty() {
                for li in 0..n_ldofs {
                    for lj in 0..n_ldofs {
                        let bc_val = x_bc_proj[e_dofs[lj] as usize];
                        if bc_val != 0.0 {
                            elem_rhs[li] -= elem_mat[li * n_ldofs + lj] * bc_val;
                        }
                    }
                }
            }

            elem_rhs_list.push(elem_rhs);
        }

        hyb.finalize();

        // Solve the trace system.
        let h_mat = hyb.get_matrix().expect("H not built");
        let n_trace = h_mat.nrows;
        let elem_dofs_refs: Vec<&[u32]> = elem_dofs.iter().map(|&d| d).collect();
        let elem_rhs_refs: Vec<&[f64]> = elem_rhs_list.iter().map(|v| v.as_slice()).collect();
        let b_r = hyb.reduce_rhs(&elem_rhs_refs);

        println!("Hybridized system: {n_trace}×{n_trace}");

        let sol_r = if n_trace > 0 {
            let cfg_h = SolverConfig {
                rtol: 1e-10, max_iter: 500, verbose: false,
                ..SolverConfig::default()
            };
            let mut sr = vec![0.0; n_trace];
            let res = fem_solver::solve_pcg_gssmoother(h_mat, &b_r, &mut sr, &cfg_h)
                .expect("Hybridized PCG+GSSmoother solve");
            println!("  PCG+GSSmoother (trace): {} iters, ||r||/||b|| = {:.3e}",
                res.iterations, res.final_residual);
            sr
        } else {
            vec![]
        };

        let mut u_hyb = vec![0.0; n_dofs];
        hyb.compute_solution(&elem_rhs_refs, &sol_r, &elem_dofs_refs, &mut u_hyb);

        // Add boundary condition values to all DOFs: x = x_int + x_bc
        for i in 0..n_dofs {
            u_hyb[i] += x_bc_proj[i];
        }

        (u_hyb, "Hybridization".to_string())

    } else {
        // ── Standard path: form_linear_system + PCG ────────────────────────
        if !ess_bdr.is_empty() {
            let x_exact = space.interpolate_vector(&|p| {
                let k = kappa;
                vec![(k * p[1]).cos() * (k * p[0]).sin(),
                     (k * p[0]).cos() * (k * p[1]).sin()]
            });
            let bv: Vec<f64> = ess_bdr.iter().map(|&d| x_exact[d as usize]).collect();
            let mut x = vec![0.0_f64; n_dofs];
            form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bv);
        }
        let n_sys = n_dofs;
        println!("Size of linear system: {n_sys}");

        let mut x = vec![0.0_f64; n_dofs];
        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        let result = fem_solver::solve_pcg_gssmoother(&mat, &rhs, &mut x, &cfg)
            .expect("PCG+GSSmoother solve failed");
        println!(
            "PCG+GSSmoother: {} iterations, ||r||/||b|| = {:.3e}",
            result.iterations, result.final_residual,
        );

        (x, "PCG+GSSmoother".to_string())
    };

    println!("Solver: {solver_label}");

    // 13. Compute and print the L² norm of the error against the exact solution.
    let l2_err = fem_assembly::hdiv_error::compute_hdiv_l2_error(
        &space, &u, |x| exact_f(x, kappa),
    );
    println!("\n|| F_h - F ||_{{L^2}} = {l2_err:.14e}");

    // 14. Save the refined mesh and the solution (matches MFEM ex4 output files).
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
        match send_to_glvis(space.mesh(), &space, &u, "F") {
            Ok(_) => eprintln!("  Sent solution to GLVis (localhost:19916)"),
            Err(e) => eprintln!("  GLVis not available: {e}"),
        }
    }
}

// ─── Source term (VectorLinearIntegrator) ────────────────────────────────────
//
//   f = (1 + 2κ²) · (cos(κy)sin(κx), cos(κx)sin(κy))

struct MaxwellHSource {
    kappa: f64,
}

impl VectorLinearIntegrator for MaxwellHSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        let k = self.kappa;
        let temp = 1.0 + 2.0 * k * k;
        let fx = temp * (k * x[1]).cos() * (k * x[0]).sin();
        let fy = temp * (k * x[0]).cos() * (k * x[1]).sin();
        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

// ─── Exact solution ──────────────────────────────────────────────────────────
//
//   F = (cos(κy)sin(κx), cos(κx)sin(κy))

fn exact_f(x: &[f64], kappa: f64) -> [f64; 2] {
    let k = kappa;
    [(k * x[1]).cos() * (k * x[0]).sin(),
     (k * x[0]).cos() * (k * x[1]).sin()]
}


// ─── GLVis helper ────────────────────────────────────────────────────────────

fn send_to_glvis(
    mesh: &Mesh<2>,
    space: &HDivSpace<Mesh<2>>,
    u: &[f64],
    field_name: &str,
) -> std::io::Result<()> {
    // Project H(div) solution onto mesh nodes by evaluating the RT field
    // at element vertices and averaging at shared nodes.
    use fem_element::{raviart_thomas::TriRT0, reference::VectorReferenceElement};
    use fem_mesh::ElementTransformation;

    let n_nodes = mesh.n_nodes() as usize;
    let ref_elem = TriRT0;
    let n_ldofs = ref_elem.n_dofs();
    let dim = 2usize;

    let ref_verts: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

    let mut sum_x = vec![0.0_f64; n_nodes];
    let mut sum_y = vec![0.0_f64; n_nodes];
    let mut count = vec![0u32; n_nodes];
    let mut ref_phi = vec![0.0_f64; n_ldofs * dim];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
        let jac = tr.jacobian();
        let det_j = tr.det_j();
        let inv_det = 1.0 / det_j;

        for vi in 0..3 {
            let xi = &ref_verts[vi];
            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            // Contravariant Piola: φ_phys = (1/det(J)) · J · φ_ref
            let mut fh_x = 0.0_f64;
            let mut fh_y = 0.0_f64;
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                let px = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) * inv_det;
                let py = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) * inv_det;
                fh_x += u[dofs[i]] * px;
                fh_y += u[dofs[i]] * py;
            }

            let nid = nodes[vi] as usize;
            sum_x[nid] += fh_x;
            sum_y[nid] += fh_y;
            count[nid] += 1;
        }
    }

    let mut f_node_x = vec![0.0_f64; n_nodes];
    let mut f_node_y = vec![0.0_f64; n_nodes];
    for i in 0..n_nodes {
        if count[i] > 0 {
            let inv = 1.0 / count[i] as f64;
            f_node_x[i] = sum_x[i] * inv;
            f_node_y[i] = sum_y[i] * inv;
        }
    }

    let mut sock = fem_io::glvis::GlVisSocket::connect("localhost", 19916)?;
    sock.send_solution_2d_vector(mesh, &f_node_x, &f_node_y, field_name)?;
    Ok(())
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh:           Option<String>,
    order:          u8,
    set_bc:         bool,
    freq:           f64,
    static_cond:    bool,
    hybridization:  bool,
    visualization:  bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh:           None,
        order:          1,
        set_bc:         true,
        freq:           1.0,
        static_cond:    false,
        hybridization:  false,
        visualization:  true,
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
            "-bc" | "--impose-bc" => {
                a.set_bc = true;
            }
            "-no-bc" | "--dont-impose-bc" => {
                a.set_bc = false;
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
            "-hb" | "--hybridization" => {
                a.hybridization = true;
            }
            "-no-hb" | "--no-hybridization" => {
                a.hybridization = false;
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

    fn solve_case(args: &Args) -> (Vec<f64>, usize, f64) {
        let mesh_path = args.mesh.as_deref().unwrap_or("data/star.mesh");
        let mfem = read_mfem_file(mesh_path).unwrap_or_else(|_| {
            // Fallback: resolve relative to the examples package directory.
            let pkg_dir = env!("CARGO_MANIFEST_DIR");
            read_mfem_file(&format!("{}/../{}", pkg_dir, mesh_path))
                .expect("failed to read MFEM mesh (tried both paths)")
        });
        let mesh: Mesh<2> = mfem.mesh2d.expect("MFEM mesh must be 2D");
        let dim = 2;

        // Uniform refinement: ≤ 25 000 elements.
        let ref_levels =
            ((25000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
        let mesh = if ref_levels > 0 {
            let mut m = mesh;
            for _ in 0..ref_levels {
                m = refine_uniform(&m);
            }
            m
        } else {
            mesh
        };

        let rt_order = if args.order >= 1 { args.order - 1 } else { 0 };
        let space = HDivSpace::new(mesh, rt_order);
        let n_dofs = space.n_dofs();
        let kappa = args.freq * PI;

        let all_tags: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess_bdr = if args.set_bc && !all_tags.is_empty() {
            boundary_dofs_hdiv(space.mesh(), &space, &all_tags)
        } else {
            vec![]
        };

        let source = MaxwellHSource { kappa };
        let quad_order = args.order * 2 + 2;
        let mut rhs = VectorAssembler::assemble_linear(&space, &[&source], quad_order);

        let grad_div = GradDivIntegrator { kappa: 1.0 };
        let vec_mass = VectorMassIntegrator { alpha: 1.0 };
        let mut mat = VectorAssembler::assemble_bilinear(
            &space, &[&grad_div, &vec_mass], quad_order,
        );

        if !ess_bdr.is_empty() {
            let x_exact = space.interpolate_vector(&|p| {
                let k = kappa;
                vec![(k * p[1]).cos() * (k * p[0]).sin(),
                     (k * p[0]).cos() * (k * p[1]).sin()]
            });
            let bv: Vec<f64> = ess_bdr.iter().map(|&d| x_exact[d as usize]).collect();
            let mut x = vec![0.0_f64; n_dofs];
            form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bv);
        }

        let mut x = vec![0.0_f64; n_dofs];
        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        fem_solver::solve_pcg_gssmoother(&mat, &rhs, &mut x, &cfg)
            .expect("PCG+GSSmoother solve failed");

        let l2_err = fem_assembly::hdiv_error::compute_hdiv_l2_error(
            &space, &x, |p| exact_f(p, kappa),
        );

        (x, n_dofs, l2_err)
    }

    fn default_args() -> Args {
        Args {
            mesh:           Some("data/star.mesh".into()),
            order:          1,
            set_bc:         true,
            freq:           1.0,
            static_cond:    false,
            hybridization:  false,
            visualization:  false,
        }
    }

    #[test]
    fn ex4_solve_converges() {
        let (_, _, l2_err) = solve_case(&default_args());
        assert!(l2_err.is_finite(), "L2 error should be finite");
        assert!(l2_err > 0.0, "L2 error should be positive");
    }

    #[test]
    fn ex4_dof_count() {
        let (_, n_dofs, _) = solve_case(&default_args());
        assert_eq!(n_dofs, 41280, "RT0 on star.mesh should give 41280 DOFs");
    }

    #[test]
    fn ex4_regression_baseline() {
        let (_, n_dofs, l2_err) = solve_case(&default_args());

        fem_regression::regression("mfem_ex4_darcy")
            .check_with("l2_error", l2_err,   1e-6, 1e-10)
            .check_with("n_dofs",   n_dofs as f64, 0.0, 0.5)
            .finalize();
    }
}
