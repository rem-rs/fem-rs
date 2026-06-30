//! # Example 0 — Mesh introduction + Poisson (analogous to MFEM ex0)
//!
//! Demonstrates the most basic usage of fem-rs: create/load a mesh, define a
//! finite element space, assemble, solve −Δu = 1 with zero Dirichlet BCs,
//! and write output for visualization.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex0_mesh_intro
//! cargo run --example mfem_ex0_mesh_intro -- --order 2
//! cargo run --example mfem_ex0_mesh_intro -- --order 2 --glvis
//! ```
//!
//! With `--glvis`, a running GLVis server (default `localhost:19916`) will
//! display the result in real-time.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 0: Mesh intro + Poisson ===");

    // 1. Create or load mesh
    let mesh = if let Some(ref path) = args.mesh_file {
        let msh = fem_io::read_msh_file(path).unwrap_or_else(|e| {
            panic!("Failed to read mesh file '{}': {}", path, e)
        });
        msh.into_2d().expect("mesh is not 2-D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };
    let dim = 2;
    println!("  Mesh: {} nodes, {} elements, dim={}", mesh.n_nodes(), mesh.n_elems(), dim);

    // 2. H¹ finite element space
    let space = H1Space::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("  FE space: P{}, DOFs = {}", args.order, n_dofs);

    // 3. Assemble stiffness matrix: ∫∇u·∇v
    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], args.order * 2 + 1);

    // 4. Assemble RHS: ∫1·v
    let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    // 5. Homogeneous Dirichlet BCs on all boundaries
    let dm = space.dof_manager();
    let all_tags: Vec<u16> = mesh.unique_boundary_tags().into_iter().collect();
    let bnd = boundary_dofs(space.mesh(), dm, &all_tags);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // 6. Solve with PCG + Jacobi
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");
    println!("  Solve: {} iters, residual = {:.3e}, converged = {}", res.iterations, res.final_residual, res.converged);

    // 7. Compute L² error against manufactured u = sin(πx)sin(πy)/(2π²)
    let u_exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin() / (2.0 * PI * PI);
    let l2 = l2_error_h1(&space, &u, u_exact);
    println!("  L² error = {:.4e}", l2);

    // 8. Write VTK output
    let fname = "ex0_solution.vtu";
    let mut w = fem_io::VtkWriter::new(space.mesh());
    w.add_point_data(fem_io::DataArray::scalars("u", u.clone()));
    w.write_file(fname).expect("write VTK");
    println!("  Wrote: {}", fname);

    // 9. Optional GLVis visualization
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

fn l2_error_h1<S: fem_space::fe_space::FESpace>(
    space: &S,
    uh: &[f64],
    u_exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};
    use fem_mesh::topology::MeshTopology;

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let re = TriP1;
        let quad = re.quadrature(5);
        let nodes = mesh.element_nodes(e);
        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();
        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            let uh_qp: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_qp - u_exact(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}

struct Args {
    n: usize,
    order: u8,
    mesh_file: Option<String>,
    glvis: bool,
}

fn parse_args() -> Args {
    let mut a = Args { n: 16, order: 1, mesh_file: None, glvis: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(16); }
            "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "--mesh" | "-m" => { a.mesh_file = it.next(); }
            "--glvis" | "-g" => { a.glvis = true; }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex0_poisson_converges() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];
        let mut mat = mat;
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
        let mut u = vec![0.0_f64; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-12, ..SolverConfig::default() };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver");
        assert!(res.converged);
        assert!(res.final_residual < 1e-10);
    }
}
