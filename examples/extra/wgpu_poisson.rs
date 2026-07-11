//! # Example 20 — GPU-accelerated Poisson (wgpu backend)
//!
//! Solves the Poisson problem using GPU-accelerated CG solvers, demonstrating
//! fem-rs's cross-platform GPU capability (Vulkan/Metal/DX12/WASM).
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex20_wgpu_poisson --features gpu --release
//! ```
//!
//! ## Notes
//! Requires a GPU with wgpu support. The `gpu` feature enables wgpu
//! dependencies and the GPU solver backends.



#[cfg(feature = "gpu")]
use fem_space::fe_space::FESpace;

#[cfg(feature = "gpu")]
fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 20: GPU-accelerated Poisson (wgpu) ===");

    // 1. Assemble on CPU (same as ex1)
    let mesh = Mesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
    let bnd_vals = vec![0.0_f64; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    println!("  Mesh: {}×{}, DOFs: {}", args.n, args.n, n);

    // 2. Create GPU context
    let ctx = fem_linalg_gpu::GpuContext::new_sync()
        .expect("failed to initialize wgpu GPU context");
    println!("  GPU: f64 support = {}", ctx.features.native_f64);
    println!("  Max buffer size = {} MB", ctx.features.max_buffer_size / (1024 * 1024));

    // 3. GPU CG solve
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };

    let mut u_gpu = vec![0.0_f64; n];
    let result = if ctx.features.native_f64 {
        fem_solver::cg_gpu::solve_cg_gpu(&ctx, &mat, &rhs, &mut u_gpu, &cfg)
    } else {
        // Fall back to f32
        eprintln!("  No f64 support on GPU; using f32 fallback");
        let mat_f32 = csr_f64_to_f32(&mat);
        let rhs_f32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut u_f32 = vec![0.0_f32; n];
        let r = fem_solver::cg_gpu::solve_cg_gpu_f32(&ctx, &mat_f32, &rhs_f32, &mut u_f32, &cfg)?;
        for i in 0..n { u_gpu[i] = u_f32[i] as f64; }
        r
    };
    let result = result.expect("GPU solver failed");

    // 4. CPU PCG for comparison
    let mut u_cpu = vec![0.0_f64; n];
    let cpu_cfg = SolverConfig { rtol: 1e-10, ..SolverConfig::default() };
    let cpu_result = fem_solver::solve_pcg_jacobi(&mat, &rhs, &mut u_cpu, &cpu_cfg)
        .expect("CPU solver failed");

    // 5. Compare
    let mut diff = 0.0_f64;
    for i in 0..n { diff += (u_gpu[i] - u_cpu[i]).abs(); }
    let max_diff = (0..n).map(|i| (u_gpu[i] - u_cpu[i]).abs()).fold(0.0_f64, f64::max);

    println!();
    println!("  ┌──────────────┬───────────┬───────────┐");
    println!("  │ Solver       │ Iters     │ Residual  │");
    println!("  ├──────────────┼───────────┼───────────┤");
    println!("  │ CPU (PCG)    │ {:>9} │ {:>9.3e} │", cpu_result.iterations, cpu_result.final_residual);
    println!("  │ GPU (CG)     │ {:>9} │ {:>9.3e} │", result.iterations, result.final_residual);
    println!("  └──────────────┴───────────┴───────────┘");
    println!("  │u_GPU - u_CPU│₁ = {:.3e}, max|diff| = {:.3e}", diff, max_diff);

    // 6. L² error
    let exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let l2_gpu = l2_error_p1(&space, &u_gpu, exact);
    let l2_cpu = l2_error_p1(&space, &u_cpu, exact);
    println!("  L² error (GPU): {:.4e},  (CPU): {:.4e}", l2_gpu, l2_cpu);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("=== fem-rs Example 20: GPU-accelerated Poisson ===");
    println!("  GPU feature not enabled.");
    println!("  Build with: cargo run --example mfem_ex20_wgpu_poisson --features gpu --release");
}

#[cfg(feature = "gpu")]
fn csr_f64_to_f32(src: &CsrMatrix<f64>) -> CsrMatrix<f32> {
    CsrMatrix {
        nrows: src.nrows,
        ncols: src.ncols,
        row_ptr: src.row_ptr.clone(),
        col_idx: src.col_idx.clone(),
        values: src.values.iter().map(|&v| v as f32).collect(),
    }
}

#[cfg(feature = "gpu")]
fn l2_error_p1<S: fem_space::fe_space::FESpace>(
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

#[cfg(feature = "gpu")]
struct Args { n: usize }

#[cfg(feature = "gpu")]
fn parse_args() -> Args {
    let mut a = Args { n: 32 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(32); }
            _ => {}
        }
    }
    a
}
