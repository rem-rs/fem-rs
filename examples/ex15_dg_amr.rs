//! # Example 15 — Poisson with mesh refinement and error estimation
//!
//! Demonstrates the AMR (Adaptive Mesh Refinement) infrastructure for the
//! scalar Poisson equation:
//!
//! ```text
//!   -Lap u = f    in Omega = [0,1]^2
//!        u = 0    on dOmega
//! ```
//!
//! Manufactured solution: `u(x,y) = sin(pi*x) sin(pi*y)`.
//!
//! Each refinement level:
//! 1. Assembles and solves the H1 system (PCG + Jacobi).
//! 2. Computes the Zienkiewicz-Zhu (ZZ) error indicators per element.
//! 3. Applies Doerfler marking to identify worst elements.
//! 4. Refines the mesh (uniform red refinement).
//! 5. Prints convergence data (L2 error, estimated error, DOF count).
//!
//! Demonstrates O(h^2) convergence for P1 elements and shows that the
//! ZZ estimator tracks the true error.
//!
//! ## Usage
//! ```
//! cargo run --example ex15_dg_amr
//! cargo run --example ex15_dg_amr -- --levels 5
//! cargo run --example ex15_dg_amr -- --n 2 --levels 7
//! cargo run --example ex15_dg_amr -- --nc --levels 4
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_mesh::amr::{refine_uniform, NCState, zz_estimator, dorfler_mark, prolongate_p1};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, apply_hanging_constraints, recover_hanging_values, boundary_dofs}};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 15: Poisson + Mesh Refinement with Error Estimation ===");
    println!("  Initial mesh: {}x{}, P1 elements", args.n0, args.n0);
    println!("  Refinement levels: {}, Doerfler theta = {}", args.levels, args.theta);
    println!("  Mode: {}\n", if args.nonconforming { "non-conforming (hanging nodes)" } else { "conforming (uniform)" });

    let u_exact = |x: &[f64]| -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    };

    let rhs_fn = |x: &[f64]| -> f64 {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    };

    println!("{:>5}  {:>8}  {:>8}  {:>12}  {:>12}  {:>8}  {:>6}  {:>8}",
             "Level", "Elems", "DOFs", "L2 error", "Est. error", "Marked", "Hang", "Ratio");
    println!("{}", "-".repeat(82));

    let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n0);
    let mut prev_l2: Option<f64> = None;
    let mut hanging_constraints = Vec::new();
    let mut nc_state = NCState::new();
    let mut prev_u: Option<Vec<f64>> = None;

    for level in 0..=args.levels {
        // ─── 1. Build H1 space on current mesh ─────────────────────────
        let space = H1Space::new(mesh.clone(), 1);
        let n = space.n_dofs();
        let n_elems = mesh.n_elems();

        // ─── 2. Assemble stiffness matrix and RHS ──────────────────────
        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);

        let source = DomainSourceIntegrator::new(&rhs_fn);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

        // ─── 3. Apply constraints ──────────────────────────────────────
        // Hanging-node constraints FIRST (before Dirichlet).
        if args.nonconforming {
            apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);
        }

        // Then Dirichlet BCs: u = 0 on all boundaries.
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // ─── 4. Solve with PCG + Jacobi ────────────────────────────────
        let mut u = prev_u.take().unwrap_or_else(|| vec![0.0_f64; n]);
        u.resize(n, 0.0);
        // Zero out constrained DOFs (recovered after solve).
        for c in &hanging_constraints { u[c.constrained] = 0.0; }
        let cfg = SolverConfig {
            rtol: 1e-10, atol: 1e-14, max_iter: 20_000, verbose: false,
            ..SolverConfig::default()
        };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg);
        match &res {
            Err(e) => {
                println!("  Solver failed at level {level}: {e:?}");
                break;
            }
            Ok(r) if !r.converged => {
                println!("  WARNING: solver did not converge at level {level} \
                          (iters={}, res={:.3e})", r.iterations, r.final_residual);
            }
            _ => {}
        }
        let _res = res.unwrap();

        // ─── 4b. Recover hanging DOF values (NC mode) ──────────────────
        if args.nonconforming {
            recover_hanging_values(&mut u, &hanging_constraints);
        }

        // ─── 5. Compute L2 error ────────────────────────────────────────
        let l2_err = h1_l2_error(&space, &u, &u_exact);

        // ─── 6. ZZ error estimator + Doerfler marking ──────────────────
        let eta = zz_estimator(&mesh, &u);
        let est_err: f64 = eta.iter().map(|e| e * e).sum::<f64>().sqrt();
        let marked = dorfler_mark(&eta, args.theta);
        let n_marked = marked.len();

        // Convergence ratio (should approach ~4 for h-refinement with P1)
        let ratio = match prev_l2 {
            Some(prev) => format!("{:.2}", prev / l2_err),
            None => "  --".to_string(),
        };
        prev_l2 = Some(l2_err);

        let n_hang = hanging_constraints.len();

        println!("{level:>5}  {n_elems:>8}  {n:>8}  {l2_err:>12.4e}  {est_err:>12.4e}  {n_marked:>8}  {n_hang:>6}  {ratio:>8}");

        if level < args.levels {
            if args.nonconforming {
                let (new_mesh, new_constraints, midpt_map) = nc_state.refine(&mesh, &marked);
                prev_u = Some(prolongate_p1(&u, new_mesh.n_nodes(), &midpt_map));
                mesh = new_mesh;
                hanging_constraints = new_constraints;
            } else {
                mesh = refine_uniform(&mesh);
                hanging_constraints.clear();
                prev_u = None; // no prolongation for uniform refinement
            }
        }
    }

    println!("\n  Expected convergence rate: O(h^2) for P1 elements (ratio -> 4.0)");
    println!("\nDone.");
}

// ─── L2 error for H1 solutions ──────────────────────────────────────────────

fn h1_l2_error<S: FESpace>(
    space: &S,
    uh: &[f64],
    exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let re  = TriP1;
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
            let uh_qp: f64 = phi.iter().zip(gd.iter())
                .map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_qp - exact(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n0:     usize,
    levels: usize,
    theta:  f64,
    nonconforming: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        n0: 4, levels: 5, theta: 0.5, nonconforming: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n0     = it.next().unwrap_or("4".into()).parse().unwrap_or(4); }
            "--levels"=> { a.levels = it.next().unwrap_or("5".into()).parse().unwrap_or(5); }
            "--theta" => { a.theta  = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5); }
            "--nc"    => { a.nonconforming = true; }
            _ => {}
        }
    }
    a
}
