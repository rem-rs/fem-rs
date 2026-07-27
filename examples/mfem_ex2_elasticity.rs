//! # Example 2 — Linear Elasticity  (one-to-one with MFEM ex2)
//!
//! Solves a multi-material cantilever beam:
//! ```text
//!   −∇·σ(u) = 0            in Ω
//!         u = 0             on boundary attribute 1 (fixed wall)
//!   σ(u)·n = (0, −1e-2)    on boundary attribute 2 (pull down)
//! ```
//! where `σ = λ tr(ε)I + 2μ ε` and λ, μ are piecewise-constant
//! (material 1 = 50× stiffer than material 2).
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex2_elasticity
//! cargo run --example mfem_ex2_elasticity -- -m data/beam-tri.mesh
//! cargo run --example mfem_ex2_elasticity -- -m data/beam-quad.mesh -o 3
//! cargo run --example mfem_ex2_elasticity -- -m data/beam-tri.mesh -no-vis
//! ```
//!
//! ## Output
//! Prints DOF count, solver iterations, and final residual.
//! Writes `displaced.mesh` and `sol.gf` (matching MFEM ex2 output files).

use std::fs::File;
use std::io::Write;

use fem_assembly::{
    Assembler,
    assembler::face_dofs_p1,
    standard::{ElasticityIntegrator, NeumannIntegrator},
    postproc::coefficient::PWConstCoeff,
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::solve_pcg;
use fem_linalg::fem_to_linlvo_csr;
use fem_space::{
    VectorH1Space,
    fe_space::FESpace,
    constraints::{boundary_dofs, form_linear_system},
};
use fem_solver::GSSmoother;

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();
    let dim = 2usize;

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("../data/beam-tri.mesh"));
    println!("   --order {}", args.order);
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

    // 2. Read the mesh from the given mesh file.
    let mesh_path = args.mesh.as_deref().unwrap_or("../data/beam-tri.mesh");
    let mfem_file = read_mfem_file(mesh_path).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<2> = mfem_file.mesh2d.expect("MFEM mesh must be 2D");

    // Verify that the mesh has ≥2 materials and ≥2 boundary attributes.
    let n_materials = mesh.elem_tags.iter().max().copied().unwrap_or(0);
    let bnd_tags = mesh.unique_boundary_tags();
    let n_bdr_attrs = bnd_tags.iter().max().copied().unwrap_or(0);
    if n_materials < 2 || n_bdr_attrs < 2 {
        eprintln!(
            "\nInput mesh should have at least two materials and \
             two boundary attributes! (See schematic in ex2.cpp)\n"
        );
        std::process::exit(3);
    }

    // 3. NURBS degree elevation — skipped (no NURBS support yet).

    // 4. Uniform refinement: choose levels so the final mesh has ≤ 5 000 elements.
    let ref_levels =
        ((5000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }

    // 5. Vector H¹ finite element space (dim copies of scalar H1).
    let space = VectorH1Space::new(mesh, args.order, dim as u8);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();
    println!("Number of finite element unknowns: {n_dofs}");
    print!("Assembling: ");
    std::io::stdout().flush().ok();

    // 6. Essential (Dirichlet) boundary DOFs — boundary attribute 1.
    let scalar_dm = space.scalar_dof_manager();
    let mesh_ref = space.mesh();
    let bnd_scalar = boundary_dofs(mesh_ref, scalar_dm, &[1]);
    let mut clamped: Vec<u32> = Vec::with_capacity(bnd_scalar.len() * 2);
    for &d in &bnd_scalar {
        clamped.push(d);                       // x‑component DOF
        clamped.push(d + n_scalar as u32);     // y‑component DOF
    }
    let clamped_vals = vec![0.0_f64; clamped.len()];

    // 7. Right‑hand side: boundary traction on attribute 2.
    //    f = (0, −1e-2) — only the y‑component is non‑zero.
    //    We assemble a scalar Neumann integral over boundary tag 2,
    //    then place it into the y‑component block of the vector RHS.
    let quad_order = args.order as u8 * 2 + 1;
    let mut rhs = vec![0.0_f64; n_dofs];
    {
        let fdofs = face_dofs_p1(mesh_ref);
        let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
        let traction_y = Assembler::assemble_boundary_linear(
            n_scalar,
            mesh_ref,
            &fdofs,
            args.order as u8,
            &[&neumann],
            &[2],
            quad_order,
        );
        for (i, &v) in traction_y.iter().enumerate() {
            rhs[n_scalar + i] += v;
        }
    }
    print!("r.h.s. ... ");
    std::io::stdout().flush().ok();

    // 8. Solution vector x — zero initial guess.
    //    (Already satisfied by the Dirichlet values below.)

    // 9. Stiffness matrix: piecewise-constant λ and μ per element attribute.
    //    Attribute 1 (material 1): λ = 50, μ = 50  (stiff).
    //    Attribute 2 (material 2): λ =  1, μ =  1  (soft).
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);

    // 10. Form the linear system (eliminate essential BCs in‑place).
    if args.static_cond {
        eprintln!("  Warning: static condensation not yet implemented — skipping.");
    }
    print!("matrix ... ");
    std::io::stdout().flush().ok();

    println!("done.");

    // 10b. Form the linear system (MFEM FormLinearSystem — in-place, full N×N).
    let n_full = n_dofs;
    let mut x = vec![0.0_f64; n_full];
    form_linear_system(&mut mat, &mut rhs, &mut x, &clamped, &clamped_vals);

    let n_sys = n_full;
    println!("Size of linear system: {n_sys}");

    // 11. Solve the full system: PCG + GSSmoother (SSOR, ω = 1).
    let linlvo_mat = fem_to_linlvo_csr(&mat);
    let precond = GSSmoother::from_csr(&linlvo_mat, 1.0).expect("SSOR setup failed");
    let _res = solve_pcg(&mat, &rhs, &mut x, &precond, 1e-8, 500, true)
        .expect("elasticity solve failed");

    // 13. Make the mesh curved based on the FE space — skipped for simplicity.

    // 14. Save the displaced mesh and the inverted solution.
    {
        // Displace the mesh nodes by the solution.
        let mut displaced_mesh = space.mesh().clone();
        let n_nodes = displaced_mesh.n_nodes();
        for i in 0..n_nodes {
            displaced_mesh.coords[i * dim]     += x[i];             // x‑displacement
            displaced_mesh.coords[i * dim + 1] += x[n_scalar + i];  // y‑displacement
        }

        // Write the displaced mesh.
        let mut mesh_f = File::create("displaced.mesh").expect("cannot create displaced.mesh");
        write_mfem(&mut mesh_f, &displaced_mesh, None).expect("mesh write failed");

        // Write the inverted solution (x → −x, matching MFEM ex2).
        let mut sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        for &v in &x {
            writeln!(sol_f, "{:.14e}", -v).expect("sol write failed");
        }
        eprintln!("  Wrote displaced.mesh and sol.gf");
    }

    // 15. Send the solution to GLVis.
    if args.visualization {
        match fem_io::glvis::GlVisSocket::connect("localhost", 19916) {
            Ok(mut sock) => {
                // send_solution_2d_vector expects separate x/y component slices,
                // one value per node.
                sock.send_solution_2d_vector(
                    space.mesh(),
                    &x[..n_scalar],
                    &x[n_scalar..],
                    "u",
                )
                .ok();
                eprintln!("  Sent solution to GLVis (localhost:19916)");
            }
            Err(e) => eprintln!("  GLVis not available: {e}"),
        }
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh:         Option<String>,
    order:        u8,
    static_cond:  bool,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh:         None,
        order:        1,
        static_cond:  false,
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
            "-no-sc" | "--no-static-condensation" => {
                a.static_cond = false;
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
