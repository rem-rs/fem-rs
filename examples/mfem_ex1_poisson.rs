//! # Example 1 — Poisson/Laplace  (one-to-one with MFEM ex1)
//!
//! Solves: `-Δu = 1` in Ω, `u = 0` on ∂Ω
//!
//! ## Usage
//!
//! ```text
//! cargo run --example mfem_ex1_poisson                          # default mesh
//! cargo run --example mfem_ex1_poisson -- -m ../data/star.mesh
//! cargo run --example mfem_ex1_poisson -- -m ../data/star.mesh -o 2
//! cargo run --example mfem_ex1_poisson -- -m ../data/star.mesh -no-vis
//! ```
//!
//! ## Output
//!
//! Prints DOF count, linear system size, solver iterations, and final residual.
//! Writes `refined.mesh` and `sol.gf` (matching MFEM ex1 output files).

use std::fs::File;
use std::io::Write;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::solve_pcg;
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{boundary_dofs, form_linear_system},
};
use fem_solver::GSSmoother;

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class).

    // 3. Read the mesh from the given mesh file.
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    let dim = 2;

    // 4. Uniform refinement: choose levels so the final mesh has ≤ 50 000 elements.
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

    // 5. H¹ finite element space of the given order.
    let space = H1Space::new(mesh.clone(), args.order);
    let n_full = space.n_dofs();

    // 6. Essential (Dirichlet) boundary DOFs — all external boundaries.
    let dm = space.dof_manager();
    let mesh_ref = space.mesh();
    let all_tags: Vec<i32> = mesh_ref.unique_boundary_tags();
    let bnd = if all_tags.is_empty() {
        vec![]
    } else {
        boundary_dofs(mesh_ref, dm, &all_tags)
    };

    // 7. Right-hand side: b(v) = ∫ 1·v dx.
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    // 8. Solution vector x — zero initial guess (built by expand_from_reduced).

    // 9. Stiffness matrix: a(u, v) = ∫ ∇u · ∇v dx.
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], args.order * 2 + 1);

    // 10. Form the linear system (MFEM FormLinearSystem — in-place, full N×N).
    let bnd_vals = vec![0.0_f64; bnd.len()];
    let mut x = vec![0.0_f64; n_full];
    form_linear_system(&mut mat, &mut rhs, &mut x, &bnd, &bnd_vals);

    // 11. Solve: PCG with symmetric Gauss-Seidel preconditoner (ω = 1 = GS).
    let n_sys = n_full;
    let linlvo_mat = fem_linalg::fem_to_linlvo_csr(&mat);
    let precond = GSSmoother::from_csr(&linlvo_mat, 1.0).expect("SSOR setup failed");
    let _result =
        solve_pcg(&mat, &rhs, &mut x, &precond, 1e-12, 500, true)
            .expect("solver failed");

    // 12. Print summary.
    println!();
    println!("Number of finite element unknowns: {}", n_full);
    println!("Size of linear system:            {}", n_sys);

    // 14. Save the refined mesh and solution (MFEM ex1 step 13).
    {
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, mesh_ref, None).expect("mesh write failed");
        let mut sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        for &v in &x {
            writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
        }
        eprintln!("  Wrote refined.mesh and sol.gf");
    }

    // 15. Send solution to GLVis (MFEM ex1 step 14).
    if args.visualization {
        match fem_io::glvis::GlVisSocket::connect("localhost", 19916) {
            Ok(mut sock) => {
                sock.send_solution_2d(mesh_ref, &x, "u").ok();
                eprintln!("  Sent solution to GLVis (localhost:19916)");
            }
            Err(e) => eprintln!("  GLVis not available: {}", e),
        }
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh:          Option<String>,
    n:             usize,
    order:         u8,
    /// Static condensation (not yet implemented).
    _static_cond:  bool,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh:          None,
        n:             16,
        order:         1,
        _static_cond:  false,
        visualization: true,
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
            "-sc" | "--static-condensation" => {
                a._static_cond = true;
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
