//! # Example 1g — Poisson/Laplace on GPU (end-to-end GPU pipeline)
//!
//! Same manufactured solution as ex1 (`u = sin(πx)sin(πy)`). The stiffness
//! matrix is assembled on GPU (timed separately) and the linear system is
//! solved with Jacobi-preconditioned CG entirely on the GPU. The RHS
//! assembly and Dirichlet BCs are applied on CPU (GPU RHS integration and
//! BC modification are future work).
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex1_poisson_gpu --features gpu
//! cargo run --example mfem_ex1_poisson_gpu --features gpu -- --n 32
//! ```
//!
//! ## Output
//! Prints GPU assembly time, GPU solve time, and L² error.

use std::f64::consts::PI;
use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_linalg_gpu::{GpuContext, GpuCsrMatrix, assemble_poisson_2d_p1_gpu};
use fem_mesh::SimplexMesh;
use fem_solver::{SolverConfig, SolveResult as SolverResult};
use fem_solver::cg_gpu::solve_pcg_gpu;
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

struct GpuTiming {
    assembly: std::time::Duration,
    upload: std::time::Duration,
    solve: std::time::Duration,
}

struct Result {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: f64,
    timing: GpuTiming,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 1g: Poisson equation (GPU pipeline) ===");
    println!("  Mesh: {}×{} subdivisions, P{} elements", args.n, args.n, args.order);

    let gpu = GpuContext::new_sync().expect("failed to init wgpu");

    let result = solve_gpu(&gpu, args.n, args.order, 1.0);

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());
    println!("  DOFs:  {}", result.n_dofs);
    println!("  GPU assembly:  {:.3} ms", result.timing.assembly.as_secs_f64() * 1000.0);
    println!("  GPU upload:    {:.3} ms", result.timing.upload.as_secs_f64() * 1000.0);
    println!("  GPU solve:     {:.3} ms", result.timing.solve.as_secs_f64() * 1000.0);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  h = {:.4e},  L² error = {:.4e}", result.h, result.l2_error);
    println!("  (Expected O(h^{}) for P{} elements)", args.order + 1, args.order);
}

fn solve_gpu(gpu: &GpuContext, n_subdiv: usize, order: u8, source_scale: f64) -> Result {
    // ─── 1. Mesh & space ──────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n_subdiv);
    let space = H1Space::new(mesh, order);
    let n = space.n_dofs();
    let mesh = space.mesh();
    let n_elem = mesh.n_elements();

    // ─── 2. Build element arrays for GPU assembly ─────────────────────────
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

    // ─── 3. Assemble stiffness matrix on GPU ──────────────────────────────
    let t0 = Instant::now();
    let _gpu_mat = assemble_poisson_2d_p1_gpu(gpu, &elem_nodes, &elem_dofs, n_elem, n);
    let assembly_time = t0.elapsed();

    // ─── 4. CPU: assemble RHS + apply Dirichlet BCs ──────────────────────
    let mut cpu_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], order * 2 + 1);
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|x: &[f64]| {
        source_scale * 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    })], order * 2 + 1);
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    apply_dirichlet(&mut cpu_mat, &mut rhs, &bnd, &vec![0.0_f64; bnd.len()]);

    // ─── 5. Upload BC-applied matrix to GPU ──────────────────────────────
    let t_up = Instant::now();
    let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(gpu, &cpu_mat);
    let upload_time = t_up.elapsed();

    // ─── 6. Solve on GPU ─────────────────────────────────────────────────
    let t1 = Instant::now();
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let sr: SolverResult = solve_pcg_gpu(gpu, &cpu_mat, &rhs, &mut u, &cfg)
        .expect("GPU PCG solve failed");
    let solve_time = t1.elapsed();

    // ─── 7. L² error ─────────────────────────────────────────────────────
    let l2 = l2_error_h1(&space, &u, |x| source_scale * (PI * x[0]).sin() * (PI * x[1]).sin());

    Result {
        n_dofs: n,
        iterations: sr.iterations,
        final_residual: sr.final_residual,
        converged: sr.converged,
        h: 1.0 / n_subdiv as f64,
        l2_error: l2,
        timing: GpuTiming { assembly: assembly_time, upload: upload_time, solve: solve_time },
    }
}

fn l2_error_h1<S: FESpace>(space: &S, uh: &[f64], u_exact: impl Fn(&[f64]) -> f64) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};
    use fem_mesh::topology::MeshTopology;
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
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();
        let mut phi = vec![0.0; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            let uh_qp: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
            err2 += w * (uh_qp - u_exact(&xp)).powi(2);
        }
    }
    err2.sqrt()
}

struct Args { n: usize, order: u8 }
fn parse_args() -> Args {
    let mut a = Args { n: 16, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n     = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--order" => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            _ => {}
        }
    }
    a
}
