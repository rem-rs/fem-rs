//! # Example 1 — Poisson/Laplace  (one-to-one with MFEM ex1)
//!
//! Solves the scalar Poisson equation with homogeneous Dirichlet boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω
//!            u = 0    on ∂Ω
//! ```
//!
//! with `f = 1` and `κ = 1` on a unit square (or user-supplied MFEM mesh).
//! This is exactly the problem defined in MFEM's Example 1.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex1_poisson
//! cargo run --example mfem_ex1_poisson -- --mesh ../data/star.mesh
//! cargo run --example mfem_ex1_poisson -- -m ../data/star.mesh --order 2
//! ```
//!
//! ## Output
//! Prints DOF count, PCG iteration count, final residual, and solution norm.

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
use fem_io::mfem::read_mfem_file;

fn main() {
    let args = parse_args();

    // Load or generate mesh
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };

    let mesh = &mesh;
    let space = H1Space::new(mesh.clone(), args.order);
    let n = space.n_dofs();

    // Assemble stiffness matrix: a(u,v) = ∫∇u·∇v dx
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], args.order * 2 + 1);

    // Assemble RHS: f(v) = ∫ 1·v dx  (constant source, matching MFEM ex1)
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    // Homogeneous Dirichlet BCs
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // Solve with PCG + Jacobi
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");

    let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("=== fem-rs Example 1: Poisson  (one-to-one with MFEM ex1) ===");
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
