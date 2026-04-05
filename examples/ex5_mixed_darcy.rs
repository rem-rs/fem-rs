//! # Example 5 — Block saddle-point system  (analogous to MFEM ex5)
//!
//! Demonstrates the block system infrastructure by solving a saddle-point
//! problem via the Uzawa (Schur complement) solver.
//!
//! The system is:
//! ```text
//!   [ A   Bᵀ ] [ u ]   [ f ]
//!   [ B  -εI ] [ p ] = [ g ]
//! ```
//!
//! where A = diffusion matrix, B = gradient coupling, and ε is a small
//! stabilization term (pressure stabilization for the P1/P1 pair).
//!
//! ## Usage
//! ```
//! cargo run --example ex5_mixed_darcy
//! cargo run --example ex5_mixed_darcy -- --n 16
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::{DivIntegrator, PressureDivIntegrator},
    standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator},
};
use fem_linalg::CooMatrix;
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 5: Saddle-point system (Uzawa) ===");
    println!("  Mesh: {}×{} subdivisions, P1/P1 elements", args.n, args.n);

    // ─── 1. Mesh and spaces ──────────────────────────────────────────────────
    let mesh_u = SimplexMesh::<2>::unit_square_tri(args.n);
    let mesh_p = SimplexMesh::<2>::unit_square_tri(args.n);
    let space_u = H1Space::new(mesh_u, 1);
    let space_p = H1Space::new(mesh_p, 1);
    let nu = space_u.n_dofs();
    let np = space_p.n_dofs();
    println!("  u-DOFs: {nu}, p-DOFs: {np}");

    // ─── 2. Assemble blocks ──────────────────────────────────────────────────
    // A = diffusion (stiffness) matrix for u
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let a_mat = Assembler::assemble_bilinear(&space_u, &[&diff], 3);

    // B, Bᵀ = gradient coupling
    let bt_mat = MixedAssembler::assemble_bilinear(&space_u, &space_p, &[&DivIntegrator], 3);
    let b_mat = MixedAssembler::assemble_bilinear(&space_p, &space_u, &[&PressureDivIntegrator], 3);

    // C = -ε M_p (pressure stabilization for P1/P1)
    let eps = 1e-2;
    let mass_p = MassIntegrator { rho: -eps };
    let c_mat = Assembler::assemble_bilinear(&space_p, &[&mass_p], 3);

    // RHS for the u-block: f_u = (some forcing)
    let source_u = DomainSourceIntegrator::new(|x: &[f64]| {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let f_u = Assembler::assemble_linear(&space_u, &[&source_u], 3);

    // g = 0 (divergence-free constraint)
    let g = vec![0.0_f64; np];

    // ─── 3. Apply Dirichlet u = 0 on boundary ───────────────────────────────
    // For Uzawa, we handle BCs by zeroing A rows/cols for boundary DOFs.
    let dm_u = space_u.dof_manager();
    let bnd_u = boundary_dofs(space_u.mesh(), dm_u, &[1, 2, 3, 4]);
    let mut a_mat = a_mat;
    let mut f_u = f_u;
    let bnd_vals = vec![0.0_f64; bnd_u.len()];
    fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bnd_u, &bnd_vals);
    println!("  Boundary u-DOFs constrained: {}", bnd_u.len());

    // ─── 4. Solve with Uzawa ─────────────────────────────────────────────────
    let sys = BlockSystem { a: a_mat, bt: bt_mat, b: b_mat, c: Some(c_mat) };
    let mut u_sol = vec![0.0_f64; nu];
    let mut p_sol = vec![0.0_f64; np];

    let cfg = SolverConfig { rtol: 1e-6, atol: 1e-10, max_iter: 2_000, verbose: false, ..SolverConfig::default() };
    let res = SchurComplementSolver::solve(&sys, &f_u, &g, &mut u_sol, &mut p_sol, &cfg)
        .expect("Uzawa solve failed");

    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 5. Report ──────────────────────────────────────────────────────────
    let u_max: f64 = u_sol.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let p_max: f64 = p_sol.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    println!("  max|u| = {u_max:.4e},  max|p| = {p_max:.4e}");

    // Verify residual: ||Au + Bᵀp - f|| + ||Bu + Cp - g||
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    sys.apply(&u_sol, &p_sol, &mut ru, &mut rp);
    let err_u: f64 = ru.iter().zip(f_u.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let err_p: f64 = rp.iter().zip(g.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    println!("  Block residual: ‖Au+Bᵀp−f‖ = {err_u:.3e},  ‖Bu+Cp−g‖ = {err_p:.3e}");

    println!("\nDone.");
}

struct Args { n: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 8 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if arg == "--n" {
            a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
        }
    }
    a
}
