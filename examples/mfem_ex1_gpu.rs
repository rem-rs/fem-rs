//! # Example 1 (GPU) — Poisson on GPU with Jacobi-preconditioned CG.
//!
//! Solves the scalar Poisson equation with homogeneous Dirichlet boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω
//!            u = 0    on ∂Ω
//! ```
//!
//! with `f = 1` and `κ = 1` on a unit square (or user-supplied MFEM mesh).
//! This is exactly the problem defined in MFEM's Example 1, solved on GPU.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex1_gpu --features gpu
//! cargo run --example mfem_ex1_gpu --features gpu -- --mesh ../data/star.mesh
//! ```
//!
//! ## Output
//! Prints DOF count, GPU PCG iteration count, final residual, and solution norm.

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_linalg_gpu::{GpuContext, assemble_poisson_2d_p1_gpu};
use fem_mesh::SimplexMesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::cg_gpu::PcgGpuWorkspace;
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let args = parse_args();
    let ctx = GpuContext::new_sync().expect("failed to create GPU context");

    // Load or generate mesh
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };

    let space = H1Space::new(mesh.clone(), args.order);
    let n_dofs = space.n_dofs();
    let n_elem = space.mesh().n_elements() as usize;
    println!("=== fem-rs Example 1 (GPU) ===");
    println!("  Mesh: {} elements, P{} elements", n_elem, args.order);
    println!("  GPU f64 support: {}", ctx.features.native_f64);
    println!("  DOFs: {n_dofs}");

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
    let mut cpu_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], args.order * 2 + 1);
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut cpu_mat, &mut rhs, &bnd, &bnd_vals);

    // Step 2: GPU solve with Jacobi-preconditioned CG
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let mut workspace = PcgGpuWorkspace::<f64>::new(&ctx, &cpu_mat, &rhs);
    let res = workspace.solve(&ctx, &mut u, &cfg).expect("GPU PCG solver failed");

    let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  Solve: {} PCG iterations, final residual = {:.3e}, converged = {}",
             res.iterations, res.final_residual, res.converged);
    println!("  ||u||_2 = {:.6e}", u_norm);
    println!("Done.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh:  Option<String>,
    n:     usize,
    order: u8,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 16, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh"   => { a.mesh = it.next(); }
            "--n"             => { a.n     = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--order"         => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            _ => {}
        }
    }
    a
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_solver::solve_pcg_jacobi;
    use std::f64::consts::PI;

    /// Exact solution for the MMS problem used in earlier versions of this example.
    fn exact(x: &[f64]) -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    fn l2_error_h1<S: FESpace>(space: &S, uh: &[f64], u_exact: impl Fn(&[f64]) -> f64) -> f64 {
        use fem_element::{ReferenceElement, lagrange::TriP1};
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
            let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
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
    fn mms_convergence_p1() {
        // MMS problem: -(u_xx + u_yy) = f, u = sin(πx)sin(πy) on ∂Ω
        // Source: f = 2π² sin(πx)sin(πy)
        // Reproduce the old reference solution on a tiny mesh.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let rhs_fun = DomainSourceIntegrator::new(|x: &[f64]| {
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let mut rhs = Assembler::assemble_linear(&space, &[&rhs_fun], 3);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
        let mut u = vec![0.0_f64; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let _ = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("MMS solver failed");
        let err = l2_error_h1(&space, &u, exact);
        assert!(err < 0.05, "L² error too large: {:.4e}", err);
    }
}
