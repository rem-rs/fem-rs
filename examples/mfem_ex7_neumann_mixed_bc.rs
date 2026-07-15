//                                MFEM Example 7
//
// 1:1 Rust translation of MFEM C++ ex7.cpp — Screened Poisson on a sphere.
//
// Compile: cargo run --example mfem_ex7_neumann_mixed_bc -- [options]
//   -e 0  : use triangles (octahedron), default
//   -e 1  : use quadrilaterals (cube)
//   -o 2  : finite element order
//   -r 4  : uniform refinement levels
//
// Description: This example code demonstrates the use of MFEM to define a
//              triangulation of a unit sphere and a simple isoparametric
//              finite element discretization of the screened Poisson problem,
//              -Delta u + u = f.
//
// Reference: mfem/examples/ex7.cpp

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator},
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::{
    H1Space,
    fe_space::FESpace,
};

// Exact solution: u = x₁·x₂ / (x₁² + x₂² + x₃²)
fn analytic_solution(x: &[f64]) -> f64 {
    let l2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    x[0] * x[1] / l2
}

// RHS: f = 7 · x₁·x₂ / (x₁² + x₂² + x₃²)
fn analytic_rhs(x: &[f64]) -> f64 {
    let l2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    7.0 * x[0] * x[1] / l2
}

fn main() {
    let args = Args::parse();
    let t0 = Instant::now();

    // ── 1. Generate initial mesh on the unit sphere ─────────────────────────
    let (mesh, elem_type_name) = if args.elem_type == 0 {
        // Inscribed octahedron: 6 vertices, 8 Tri3 faces
        (Mesh::<3>::unit_sphere_octahedron(), "Tri3")
    } else {
        // Inscribed cube: 8 vertices, 6 Quad4 faces
        (Mesh::<3>::unit_sphere_cube(), "Quad4")
    };
    eprintln!("  Mesh: {} nodes, {} elements, type = {}", mesh.n_nodes(), mesh.n_elems(), elem_type_name);

    // ── 2. Refine and snap to sphere ────────────────────────────────────────
    let mut mesh = mesh;
    let ref_levels = args.ref_levels;
    for l in 0..=ref_levels {
        if l > 0 {
            // Uniform refinement (surface mesh).
            match mesh.element_type(0) {
                fem_mesh::element_type::ElementType::Tri3 => {
                    mesh = fem_mesh::amr::refine_uniform_surface_tri3(&mesh);
                }
                _ => {
                    mesh = fem_mesh::amr::refine_uniform_surface_quad4(&mesh);
                }
            }
        }
        // Snap nodes to sphere surface (always snap after each refinement).
        mesh.snap_to_sphere();
    }
    eprintln!("  After refinement: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ── 3. Local refinement (optional, not yet implemented) ─────────────────
    // MFEM supports RefineAtVertex and RandomRefinement, which we skip here.

    // ── 4. Define H1 finite element space (isoparametric: same order as mesh) -
    let order = args.order;
    let space = H1Space::new(mesh.clone(), order);
    println!("Number of unknowns: {}", space.n_dofs());

    // ── 5. Set up linear form b(v) = ∫ f·v dS ──────────────────────────────
    let rhs_integ = DomainSourceIntegrator::new(analytic_rhs);
    let quad_rhs = (order as u8) * 2 + 1;
    let rhs = Assembler::assemble_linear(&space, &[&rhs_integ], quad_rhs);

    // ── 6. Initialize solution vector u = 0 ─────────────────────────────────
    let mut u = vec![0.0; space.n_dofs()];

    // ── 7. Set up bilinear form a(u,v) = ∫ ∇u·∇v + u·v dS ──────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mass = MassIntegrator { rho: 1.0 };
    let quad_stiff = (order as u8) * 2;
    let mat = Assembler::assemble_bilinear(
        &space,
        &[&diffusion, &mass],
        quad_stiff,
    );

    // ── 8. Assemble and solve (no Dirichlet BC — sphere is a closed surface) ─
    // MFEM: a->FormLinearSystem(empty_tdof_list, x, *b, A, X, B)
    // Since there's no Dirichlet BC, solve the full system directly.
    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_gssmoother(&mat, &rhs, &mut u, &cfg)
        .expect("PCG solve failed");

    if !res.converged {
        eprintln!("  PCG: Number of iterations: {}", res.iterations);
        eprintln!("  PCG: No convergence!");
    }

    // ── 9. Compute L² error against analytic solution ───────────────────────
    let gf = GridFunction::new(&space, u.clone());
    let l2_err = gf.compute_l2_error(&analytic_solution, (order as u8) * 2 + 2);
    println!("\nL2 norm of error: {}", l2_err);

    // ── 10. Save output (matching MFEM: sphere_refined.mesh + sol.gf) ──────
    // (Mesh<3> file output not yet supported by write_mfem.)

    eprintln!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    elem_type: usize,   // 0 = triangles (octahedron), 1 = quads (cube)
    order: u8,
    ref_levels: usize,
    #[allow(dead_code)]
    visualization: bool,
}

impl Args {
    fn parse() -> Self {
        let mut elem_type = 0usize;
        let mut order: u8 = 2;
        let mut ref_levels: usize = 4;
        let mut visualization = true;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-e" | "--elem" => {
                    elem_type = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "-o" | "--order" => {
                    order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
                }
                "-r" | "--refine" => {
                    ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(4);
                }
                "-vis" | "--visualization" => visualization = true,
                "-no-vis" | "--no-visualization" => visualization = false,
                _ => {}
            }
        }
        Args { elem_type, order, ref_levels, visualization }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use fem_regression::regression;

    use super::*;

    fn solve_sphere(elem_type: usize, order: u8, ref_levels: usize) -> f64 {
        // Same as main() but returns L² error.
        let mesh = if elem_type == 0 {
            Mesh::<3>::unit_sphere_octahedron()
        } else {
            Mesh::<3>::unit_sphere_cube()
        };

        let mut mesh = mesh;
        for l in 0..=ref_levels {
            if l > 0 {
                match mesh.element_type(0) {
                    fem_mesh::element_type::ElementType::Tri3 => {
                        mesh = fem_mesh::amr::refine_uniform_surface_tri3(&mesh);
                    }
                    _ => {
                        mesh = fem_mesh::amr::refine_uniform_surface_quad4(&mesh);
                    }
                }
            }
            mesh.snap_to_sphere();
        }

        let space = H1Space::new(mesh, order);
        let rhs_integ = DomainSourceIntegrator::new(analytic_rhs);
        let rhs = Assembler::assemble_linear(&space, &[&rhs_integ], (order as u8) * 2 + 1);

        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let mass = MassIntegrator { rho: 1.0 };
        let mat = Assembler::assemble_bilinear(&space, &[&diffusion, &mass], (order as u8) * 2);

        let mut u = vec![0.0; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-12, max_iter: 200, verbose: false, ..SolverConfig::default() };
        solve_pcg_gssmoother(&mat, &rhs, &mut u, &cfg).expect("PCG");
        let gf = GridFunction::new(&space, u);
        gf.compute_l2_error(&analytic_solution, (order as u8) * 2 + 2)
    }

    #[test]
    fn ex7_tri_sphere_converges() {
        let coarse = solve_sphere(0, 2, 2);
        let fine = solve_sphere(0, 2, 3);
        assert!(
            fine < coarse,
            "L2 error must decrease on refinement: coarse={:.6e} fine={:.6e}",
            coarse, fine
        );
        regression("mfem_ex7_tri_sphere")
            .check("l2_error_r2_o2", coarse)
            .check("l2_error_r3_o2", fine)
            .finalize();
    }

    #[test]
    fn ex7_quad_sphere_converges() {
        let coarse = solve_sphere(1, 2, 2);
        let fine = solve_sphere(1, 2, 3);
        assert!(
            fine < coarse,
            "L2 error must decrease on refinement: coarse={:.6e} fine={:.6e}",
            coarse, fine
        );
        regression("mfem_ex7_quad_sphere")
            .check("l2_error_r2_o2", coarse)
            .check("l2_error_r3_o2", fine)
            .finalize();
    }
}
