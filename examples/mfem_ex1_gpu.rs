//! # Example 1 (GPU) — Poisson on GPU with Jacobi-preconditioned CG.
//!
//! Demonstrates the full GPU pipeline:
//! 1. Mesh + FE space on CPU (same as ex1)
//! 2. Matrix assembly on GPU via compute shaders
//! 3. Solve on GPU with PCG + Jacobi preconditioner
//! 4. Solution readback and L² error computation
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex1_gpu --features gpu
//! ```

use std::f64::consts::PI;
use fem_assembly::{standard::{DiffusionIntegrator, DomainSourceIntegrator}, Assembler};
use fem_linalg_gpu::{GpuContext, assemble_poisson_2d_p1_gpu};
use fem_mesh::SimplexMesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::cg_gpu::PcgGpuWorkspace;
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let ctx = GpuContext::new_sync().expect("failed to create GPU context");
    println!("=== fem-rs Example 1 (GPU) ===");
    println!("  GPU f64 support: {}", ctx.features.native_f64);

    let n = 16u32;
    let order = 1u8;
    let mesh = SimplexMesh::<2>::unit_square_tri(n as usize);
    let space = H1Space::new(mesh, order);
    let n_dofs = space.n_dofs();
    let n_elem = space.mesh().n_elements() as usize;
    println!("  Mesh: {n}×{n} subdivisions, P{order} elements");
    println!("  DOFs: {n_dofs}, Elements: {n_elem}");

    // CPU-side: extract element data for GPU assembly
    let tri_nodes: Vec<f32> = (0..space.mesh().n_elements() as u32)
        .flat_map(|e| {
            let nodes = space.mesh().element_nodes(e);
            let mut coords = Vec::with_capacity(6);
            for &ni in nodes {
                let c = space.mesh().node_coords(ni);
                coords.push(c[0] as f32);
                coords.push(c[1] as f32);
            }
            coords
        })
        .collect();
    let tri_dofs: Vec<u32> = (0..space.mesh().n_elements() as u32)
        .flat_map(|e| space.element_dofs(e).iter().copied())
        .collect();

    // Step 1: GPU assembly → GpuCsrMatrix
    let gpu_mat = assemble_poisson_2d_p1_gpu(&ctx, &tri_nodes, &tri_dofs, n_elem, n_dofs);
    println!("  GPU assembly done, nnz = {}", gpu_mat.nnz);

    // CPU assembly for RHS and Dirichlet BC application
    let mut cpu_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], order * 2 + 1);
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    })], order * 2 + 1);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    apply_dirichlet(&mut cpu_mat, &mut rhs, &bnd, &vec![0.0_f64; bnd.len()]);

    // Step 2: GPU solve with Jacobi-preconditioned CG
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let mut workspace = PcgGpuWorkspace::<f64>::new(&ctx, &cpu_mat, &rhs);
    let res = workspace.solve(&ctx, &mut u, &cfg).expect("GPU PCG solver failed");
    println!("  GPU PCG: {} iterations, residual = {:.3e}", res.iterations, res.final_residual);

    // Step 3: L² error
    let exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let l2 = l2_error_h1(&space, &u, exact);
    println!("  L² error = {:.4e}", l2);
    println!("Done.");
}

fn l2_error_h1<S: FESpace>(space: &S, uh: &[f64], u_exact: impl Fn(&[f64]) -> f64) -> f64 {
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
            let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                       x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]];
            let uh_qp: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_qp - u_exact(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}
