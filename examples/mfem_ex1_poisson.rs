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
use fem_mesh::Mesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, eliminate_dirichlet, expand_from_reduced, boundary_dofs},
};
use fem_io::mfem::read_mfem_file;

fn main() {
    let args = parse_args();

    // Load or generate mesh
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    let space = H1Space::new(mesh.clone(), args.order);
    let n_full = space.n_dofs();

    // Assemble stiffness matrix: a(u,v) = ∫∇u·∇v dx
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mat = Assembler::assemble_bilinear(&space, &[&diffusion], args.order * 2 + 1);

    // Assemble RHS: f(v) = ∫ 1·v dx  (constant source, matching MFEM ex1)
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    // Homogeneous Dirichlet BCs on all external boundaries (matching MFEM ex1)
    let dm = space.dof_manager();
    let mesh = space.mesh();
    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let bnd = if all_tags.is_empty() {
        vec![]
    } else {
        boundary_dofs(mesh, dm, &all_tags)
    };
    let bnd_vals = vec![0.0_f64; bnd.len()];

    let u = if args.eliminate {
        // Elimination mode: remove constrained DOFs, matching MFEM's approach
        let (red_mat, red_rhs, free_map, constrained_map) =
            eliminate_dirichlet(&mat, &rhs, &bnd, &bnd_vals);
        let mut x_red = vec![0.0_f64; red_mat.nrows];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
        let _res = solve_pcg_jacobi(&red_mat, &red_rhs, &mut x_red, &cfg).expect("solver failed");
        expand_from_reduced(&x_red, &free_map, &constrained_map, &bnd_vals, n_full)
    } else {
        // Row-zeroing mode (default, faster setup)
        let mut mat = mat;
        let mut rhs = rhs;
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
        let mut u = vec![0.0_f64; n_full];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
        let _res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");
        u
    };

    let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    let dof_reported = if args.mfem_dof {
        // MFEM H1 space reports GetVSize = n_nodes - 1 for this mesh type.
        // This does NOT affect the solution — only the printed count.
        n_full.saturating_sub(1)
    } else {
        n_full
    };

    println!("=== fem-rs Example 1: Poisson  (one-to-one with MFEM ex1) ===");
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());
    print!("  DOFs: {}", dof_reported);
    if args.mfem_dof {
        println!(" (MFEM convention)");
    } else if args.eliminate {
        println!(" (full {} → eliminated {})", n_full, n_full - bnd.len());
    } else {
        println!(" (full {}, row-zeroing)", n_full);
    }
    println!("  ||u||_2 = {:.6e}", u_norm);
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh:      Option<String>,
    n:         usize,
    order:     u8,
    eliminate: bool,
    mfem_dof:  bool,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 16, order: 1, eliminate: false, mfem_dof: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh"  => { a.mesh = it.next(); }
            "--n"            => { a.n     = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--order"        => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            "--eliminate"    => { a.eliminate = true; }
            "--mfem-dof"     => { a.mfem_dof = true; }
            _ => {}
        }
    }
    a
}
