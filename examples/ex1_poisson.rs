//! # Example 1 — Poisson/Laplace  (analogous to MFEM ex1)
//!
//! Solves the scalar Poisson equation with homogeneous Dirichlet boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω = [0,1]²
//!            u = 0    on ∂Ω
//! ```
//!
//! with the manufactured solution  `u(x,y) = sin(π x) sin(π y)`,  which gives
//! `f = 2 π² sin(π x) sin(π y)` and κ = 1.
//!
//! ## Usage
//! ```
//! cargo run --example ex1_poisson
//! cargo run --example ex1_poisson -- --order 2 --n 32
//! cargo run --example ex1_poisson -- --n 8   # observe h² convergence
//! cargo run --example ex1_poisson -- --n 16
//! cargo run --example ex1_poisson -- --n 32
//! ```
//!
//! ## Output
//! Prints L² error, DOF count, iteration count, and convergence rate.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    // ─── Parse CLI args ──────────────────────────────────────────────────────
    let args = parse_args();

    println!("=== fem-rs Example 1: Poisson equation ===");
    println!("  Mesh:  {}×{} subdivisions, P{} elements", args.n, args.n, args.order);

    // ─── 1. Create mesh ──────────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());

    // ─── 2. Create H¹ finite element space ──────────────────────────────────
    let space = H1Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  DOFs:  {n}");

    // ─── 3. Assemble bilinear form A = ∫ ∇u·∇v dx ───────────────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], args.order as u8 * 2 + 1);

    // ─── 4. Assemble linear form f = ∫ 2π² sin(πx)sin(πy) v dx ─────────────
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&source], args.order as u8 * 2 + 1);

    // ─── 5. Apply homogeneous Dirichlet BCs on all four walls ────────────────
    //   Tag 1 = bottom (y=0), 2 = right (x=1), 3 = top (y=1), 4 = left (x=0)
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // ─── 6. Solve K u = f with PCG + Jacobi preconditioner ──────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
        .expect("solver failed");

    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 7. L² error against exact solution u = sin(πx)sin(πy) ─────────────
    let l2 = l2_error_h1(&space, &u, |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin());
    let h = 1.0 / args.n as f64;
    println!("  h = {h:.4e},  L² error = {l2:.4e}");
    println!("  (Expected O(h^{}) for P{} elements)", args.order + 1, args.order);

    println!("\nDone. (No VTK output in this minimal example — add fem-io to enable.)");
}

// ─── L² error helper ─────────────────────────────────────────────────────────

/// Compute the L² error ‖u_h − u_exact‖_{L²(Ω)} using element quadrature.
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

        // Jacobian for the affine map from reference to physical
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();

        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;

            // Physical coords
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            // u_h at this quadrature point
            let uh_qp: f64 = phi.iter().zip(gd.iter())
                .map(|(&p, &di)| p * uh[di])
                .sum();
            let diff = uh_qp - u_exact(&xp);
            err2 += w * diff * diff;
        }
    }

    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n:     usize,
    order: u8,
}

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
