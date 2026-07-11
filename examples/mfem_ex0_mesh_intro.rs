//! # Example 0 — Mesh introduction + Poisson (analogous to MFEM ex0)
//!
//! Demonstrates the most basic usage of fem-rs: load an MFEM-format mesh, apply
//! one uniform refinement, define an H¹ finite element space, assemble and solve
//! −Δu = 1 with zero Dirichlet BCs, then write output for visualization (VTK + MFEM).
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex0_mesh_intro
//! cargo run --example mfem_ex0_mesh_intro -- -m ../data/star.mesh
//! cargo run --example mfem_ex0_mesh_intro -- -m ../data/star.mesh -o 2
//! cargo run --example mfem_ex0_mesh_intro -- -m ../data/square-disc.mesh --glvis
//! ```
//!
//! When no `-m` is given, a default unit-square triangulation is used.
//! With `--glvis`, a running GLVis server (default `localhost:19916`) will
//! display the result in real-time.

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 0: Mesh intro + Poisson (MFEM ex0) ===");

    // 1. Load mesh from an MFEM .mesh file, or generate a default unit square.
    let mesh = if let Some(ref path) = args.mesh_file {
        let mfem = fem_io::mfem::read_mfem_file(path).unwrap_or_else(|e| {
            panic!("Failed to read MFEM mesh '{}': {}", path, e)
        });
        mfem.mesh2d.expect("only 2-D meshes are supported in this example")
    } else {
        Mesh::<2>::unit_square_tri(16)
    };
    println!(
        "  Mesh: {} nodes, {} elements",
        mesh.n_nodes(),
        mesh.n_elems(),
    );

    // 2. Uniform refinement (matches MFEM ex0's `mesh.UniformRefinement()`).
    let mesh = refine_uniform(&mesh);
    println!(
        "  After refinement: {} nodes, {} elements",
        mesh.n_nodes(),
        mesh.n_elems(),
    );

    // 3. H¹ finite element space.
    let space = H1Space::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("  Number of unknowns: {}", n_dofs);

    // 4. Assemble stiffness matrix: ∫∇u·∇v
    let mat = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        args.order * 2 + 1,
    );

    // 5. Assemble RHS: ∫1·v
    let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    // 6. Homogeneous Dirichlet BCs on all boundaries.
    let dm = space.dof_manager();
    let all_tags: Vec<i32> = space.mesh().unique_boundary_tags().into_iter().collect();
    let bnd = boundary_dofs(space.mesh(), dm, &all_tags);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // 7. Solve with PCG + Jacobi preconditioner.
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged,
    );

    // 8. Write VTK output for external visualization (e.g. ParaView).
    let fname = "ex0_solution.vtu";
    let mut w = fem_io::VtkWriter::new(space.mesh());
    w.add_point_data(fem_io::DataArray::scalars("u", u.clone()));
    w.write_file(fname).expect("write VTK");
    println!("  Wrote: {}", fname);

    // Also write MFEM-format output (mesh + grid function), matching ex0.
    fem_io::mfem::write_mfem_file("ex0_mesh.mesh", space.mesh()).ok();
    fem_io::mfem::write_gf_file("ex0_solution.gf", 2, &u, "H1", args.order, 1).ok();
    println!("  Wrote: ex0_mesh.mesh, ex0_solution.gf");

    // 9. Optional GLVis visualization.
    if args.glvis {
        match fem_io::glvis::GlVisSocket::connect("localhost", 19916) {
            Ok(mut vis) => {
                vis.send_solution_2d(space.mesh(), &u, "u").ok();
                println!("  Sent to GLVis (localhost:19916)");
            }
            Err(e) => println!("  GLVis not available: {}", e),
        }
    }

    println!("Done.");
}

struct Args {
    mesh_file: Option<String>,
    order: u8,
    glvis: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: None,
        order: 1,
        glvis: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--mesh" | "-m" => a.mesh_file = it.next(),
            "--order" | "-o" => {
                a.order = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            }
            "--glvis" | "-g" => a.glvis = true,
            _ => {}
        }
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: MMS convergence (moved from the main example body)
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use fem_assembly::{
        Assembler,
        standard::{DiffusionIntegrator, DomainSourceIntegrator},
    };
    use fem_element::lagrange::TriP1;
    use fem_element::ReferenceElement;
    use fem_mesh::topology::MeshTopology;
    use fem_mesh::Mesh;
    use fem_solver::{solve_pcg_jacobi, SolverConfig};
    use fem_space::constraints::{apply_dirichlet, boundary_dofs};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;
    use std::f64::consts::PI;

    /// L² error against an exact solution on a triangulated domain.
    fn l2_error_h1<S: FESpace>(
        space: &S,
        uh: &[f64],
        u_exact: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        let mesh = space.mesh();
        let mut err2 = 0.0_f64;
        for e in 0..mesh.n_elements() as u32 {
            let re = TriP1;
            let quad = re.quadrature(5);
            let nodes = mesh.element_nodes(e);
            let gd: Vec<usize> = space
                .element_dofs(e)
                .iter()
                .map(|&d| d as usize)
                .collect();
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let det_j =
                ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
            let mut phi = vec![0.0_f64; re.n_dofs()];
            for (qi, xi) in quad.points.iter().enumerate() {
                re.eval_basis(xi, &mut phi);
                let w = quad.weights[qi] * det_j;
                let xp = [
                    x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                    x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
                ];
                let uh_qp: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
                let diff = uh_qp - u_exact(&xp);
                err2 += w * diff * diff;
            }
        }
        err2.sqrt()
    }

    #[test]
    fn ex0_poisson_converges() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let bnd_vals = vec![0.0_f64; bnd.len()];
        let mut mat = mat;
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
        let mut u = vec![0.0_f64; space.n_dofs()];
        let cfg = SolverConfig {
            rtol: 1e-12,
            ..SolverConfig::default()
        };
        let res = solve_cg(&mat, &rhs, &mut u, &cfg).expect("solver");
        assert!(res.converged);
        assert!(res.final_residual < 1e-10);

        // Verify against MMS solution u = sin(πx)sin(πy)/(2π²)
        let u_exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin() / (2.0 * PI * PI);
        let err = l2_error_h1(&space, &u, u_exact);
        assert!(
            err < 0.1,
            "L² error = {:.4e}, expected < 0.1",
            err
        );
    }
}
