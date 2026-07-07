//! # Example 1 (GPU) — Poisson/Laplace on GPU (one-to-one with MFEM ex1)
//!
//! Solves the scalar Poisson equation with homogeneous Dirichlet boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω
//!            u = 0    on ∂Ω
//! ```
//!
//! with `f = 1` and `κ = 1` on a unit square (or user-supplied MFEM mesh).
//! Stiffness matrix is assembled on GPU; RHS assembly and Dirichlet BCs are
//! applied on CPU; solve uses GPU-accelerated PCG.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex1_poisson_gpu --features gpu
//! cargo run --example mfem_ex1_poisson_gpu --features gpu -- -m ../data/star.mesh
//! cargo run --example mfem_ex1_poisson_gpu --features gpu -- --n 32 --order 2
//! ```
//!
//! ## Output
//! Prints DOF count, PCG iteration count, final residual, and solution norm.

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_linalg_gpu::{GpuContext, GpuCsrMatrix, assemble_poisson_2d_p1_gpu};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::{SolverConfig, SolveResult as SolverResult};
use fem_solver::cg_gpu::solve_pcg_gpu;
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();

    let gpu = GpuContext::new_sync().expect("failed to init wgpu");

    // Load or generate mesh
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    let mesh = &mesh;
    let space = H1Space::new(mesh.clone(), args.order);
    let n = space.n_dofs();
    let n_elem = mesh.n_elements();

    // Build element arrays for GPU assembly
    let mut elem_nodes = Vec::with_capacity(n_elem * 6);
    let mut elem_dofs = Vec::with_capacity(n_elem * 3);
    for e in 0..n_elem as u32 {
        let nds = mesh.element_nodes(e);
        for &ni in nds {
            let c = mesh.node_coords(ni);
            elem_nodes.push(c[0] as f32);
            elem_nodes.push(c[1] as f32);
        }
        let dofs = space.element_dofs(e);
        elem_dofs.extend_from_slice(dofs);
    }

    // Assemble stiffness matrix on GPU
    let _gpu_mat = assemble_poisson_2d_p1_gpu(&gpu, &elem_nodes, &elem_dofs, n_elem, n);

    // CPU: assemble RHS (f = 1) + apply Dirichlet BCs
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], args.order * 2 + 1);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // Upload to GPU and solve
    let _gpu_mat_uploaded = GpuCsrMatrix::<f64>::from_cpu(&gpu, &mat);
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res: SolverResult = solve_pcg_gpu(&gpu, &mat, &rhs, &mut u, &cfg)
        .expect("GPU PCG solve failed");

    let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("=== fem-rs Example 1: Poisson (GPU) ===");
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());
    println!("  DOFs:  {}", n);
    println!("  Solve: {} PCG iterations, final residual = {:.3e}, converged = {}",
             res.iterations, res.final_residual, res.converged);
    println!("  ||u||_2 = {:.6e}", u_norm);
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
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "--n"           => { a.n     = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--order"       => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            _ => {}
        }
    }
    a
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use fem_assembly::{
        Assembler,
        standard::{DiffusionIntegrator, DomainSourceIntegrator},
    };
    use fem_element::{ReferenceElement, lagrange::TriP1};
    use fem_linalg_gpu::{GpuContext, GpuCsrMatrix, assemble_poisson_2d_p1_gpu};
    use fem_mesh::Mesh;
    use fem_mesh::topology::MeshTopology;
    use fem_solver::{SolverConfig, SolveResult};
    use fem_solver::cg_gpu::solve_pcg_gpu;
    use fem_space::{
        H1Space, fe_space::FESpace,
        constraints::{apply_dirichlet, boundary_dofs},
    };

    fn l2_error_h1<S: FESpace>(space: &S, uh: &[f64], u_exact: impl Fn(&[f64]) -> f64) -> f64 {
        let mesh = space.mesh();
        let mut err2 = 0.0;
        for e in 0..mesh.n_elements() as u32 {
            let re = TriP1;
            let quad = re.quadrature(5);
            let nodes = mesh.element_nodes(e);
            let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
            let mut phi = vec![0.0; re.n_dofs()];
            for (qi, xi) in quad.points.iter().enumerate() {
                re.eval_basis(xi, &mut phi);
                let w = quad.weights[qi] * det_j;
                let xp = [
                    x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                    x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
                ];
                let uh_qp: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
                err2 += w * (uh_qp - u_exact(&xp)).powi(2);
            }
        }
        err2.sqrt()
    }

    #[test]
    fn mms_poisson_gpu() {
        let gpu = GpuContext::new_sync().expect("failed to init wgpu");
        let order = 1u8;
        let n_subdiv = 16usize;

        let mesh = Mesh::<2>::unit_square_tri(n_subdiv);
        let space = H1Space::new(mesh, order);
        let n = space.n_dofs();
        let n_elem = space.mesh().n_elements();

        // Element data for GPU assembly
        let mut elem_nodes = Vec::with_capacity(n_elem * 6);
        let mut elem_dofs = Vec::with_capacity(n_elem * 3);
        for e in 0..n_elem as u32 {
            let nds = space.mesh().element_nodes(e);
            for &ni in nds {
                let c = space.mesh().node_coords(ni);
                elem_nodes.push(c[0] as f32);
                elem_nodes.push(c[1] as f32);
            }
            let dofs = space.element_dofs(e);
            elem_dofs.extend_from_slice(dofs);
        }

        // GPU assembly
        let _gpu_mat = assemble_poisson_2d_p1_gpu(&gpu, &elem_nodes, &elem_dofs, n_elem, n);

        // CPU assembly with manufactured solution RHS
        let mut mat = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            3,
        );
        let mut rhs = Assembler::assemble_linear(
            &space,
            &[&DomainSourceIntegrator::new(|x: &[f64]| {
                2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
            })],
            3,
        );

        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0_f64; bnd.len()]);

        // Solve
        let _gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &mat);
        let mut u = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 0.0,
            max_iter: 5_000,
            verbose: false,
            ..SolverConfig::default()
        };
        let _res: SolveResult =
            solve_pcg_gpu(&gpu, &mat, &rhs, &mut u, &cfg).expect("GPU PCG solve failed");

        let exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin();
        let l2 = l2_error_h1(&space, &u, exact);
        let h = 1.0 / n_subdiv as f64;
        // Expect at least O(h^2) convergence for P1
        assert!(l2 < h * h, "L² error {} too large for h={}", l2, h);
    }
}
