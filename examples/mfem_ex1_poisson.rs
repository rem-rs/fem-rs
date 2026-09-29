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
    Assembler, BilinearForm, ElimPolicy,
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
    // 1. Parse command-line options.  MFEM ex1 prints the parsed options
    //    (`args.PrintOptions(cout)`) and the device/memory configuration
    //    (`device.Print()`) as the first 10 lines of stdout.
    let args = parse_args();
    println!("Options used:");
    match args.mesh {
        Some(ref m) => println!("   --mesh {m}"),
        None => println!("   --mesh data/star.mesh"),
    }
    println!("   --order {}", args.order);
    println!("   {}", if args.static_cond { "--static-condensation" } else { "--no-static-condensation" });
    println!("   --no-partial-assembly");
    println!("   --no-full-assembly");
    println!("   --device cpu");
    println!("   {}", if args.visualization { "--visualization" } else { "--no-visualization" });
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class).

    // 3. Read the mesh from the given mesh file.
    //    MFEM ex1 defaults to `star.mesh` (20-element triangular mesh); the
    //    refinement loop below then targets ≤ 50 000 elements, matching C++.
    // 3. Read the mesh.  MFEM ex1: `Mesh mesh(mesh_file, 1, 1)` reads with
    //    `generate_edges=1` and `refine=1` — one automatic uniform refinement
    //    on load.  fem-parallel's `read_mfem_file` does not refine on load, so
    //    we apply one extra refinement to match.
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else if args.n > 0 {
        Mesh::<2>::unit_square_tri(args.n)
    } else {
        let mfem = read_mfem_file("data/star.mesh").expect("failed to read data/star.mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    };
    let mesh = refine_uniform(&mesh);  // matching C++ `Mesh(mesh_file, 1, 1)`
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

    // Print BEFORE assembly (matching C++ output order).
    println!("Number of finite element unknowns: {n_full}");

    // 6. Essential (Dirichlet) boundary DOFs.
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

    // 9. Stiffness matrix: a(u, v) = ∫ ∇u · ∇v dx.
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], args.order * 2 + 1);

    // 10-11. Form the linear system and solve (MFEM FormLinearSystem + PCG).
    //     D839-2: `-sc` mirrors MFEM ex1.cpp:219 `a.EnableStaticCondensation()`
    //     — the form-layer switch Schur-reduces the system to the trace dofs
    //     (one Q2 bubble per quad), the reduced system is solved, and
    //     `RecoverFEMSolution` back-substitutes the bubbles.  Without bubbles
    //     (order 1) the switch is a transparent no-op — same as MFEM.  The
    //     whole flow is the caller-side switch the other way round: no
    //     condensed-specific entry, just `form_linear_system` +
    //     `recover_fem_solution` on the same form.
    let x = if args.static_cond {
        let mut bf = BilinearForm::new(space.clone())
            .add_integrator(DiffusionIntegrator { kappa: 1.0 });
        bf.assemble(args.order * 2 + 1);
        bf.enable_static_condensation();
        let mut b_sc = rhs.clone();
        let x0 = vec![0.0_f64; n_full];
        let x_red = bf.form_linear_system(&bnd, &x0, &mut b_sc, ElimPolicy::DiagOne);
        println!("Size of linear system: {}", x_red.len());
        let a_red = bf.mat().expect("condensed matrix").clone();
        let linlvo_mat = fem_linalg::fem_to_linlvo_csr(&a_red);
        let precond = GSSmoother::from_csr(&linlvo_mat).expect("SSOR setup failed");
        let mut xs = vec![0.0_f64; x_red.len()];
        let _result = solve_pcg(&a_red, &b_sc[..x_red.len()], &mut xs, &precond, 1e-12, 200, true)
            .expect("solver failed");
        bf.recover_fem_solution(&xs)
    } else {
        // 10. Dirichlet BCs — in-place full N×N (MFEM FormLinearSystem).
        let bnd_vals = vec![0.0_f64; bnd.len()];
        let mut x = vec![0.0_f64; n_full];
        form_linear_system(&mut mat, &mut rhs, &mut x, &bnd, &bnd_vals);
        println!("Size of linear system: {}", n_full);

        // 11. Solve: PCG with symmetric Gauss-Seidel preconditioner (ω = 1 = GS).
        //     MFEM ex1: GSSmoother M(A); PCG(A, M, B, X, 1, 200, 1e-12, 0.0)
        let linlvo_mat = fem_linalg::fem_to_linlvo_csr(&mat);
        let precond = GSSmoother::from_csr(&linlvo_mat).expect("SSOR setup failed");
        let _result = solve_pcg(&mat, &rhs, &mut x, &precond, 1e-12, 200, true)
            .expect("solver failed");
        x
    };

    // 13. Save the refined mesh and solution (MFEM ex1 step 13).
    {
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, mesh_ref, None).expect("mesh write failed");
        let mut sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        for &v in &x {
            writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
        }
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
    /// Static condensation (MFEM `EnableStaticCondensation`, D839-2).
    static_cond:  bool,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh:          None,
        n:             0, // 0 → default to data/star.mesh (MFEM ex1 default)
        order:         1,
        static_cond:   false,
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
                a.static_cond = true;
            }
            "-n" | "--n" => {
                a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
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
