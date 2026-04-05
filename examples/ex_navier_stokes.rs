//! # Navier-Stokes Example — Kovasznay Flow (Phase 44)
//!
//! Solves the steady incompressible Navier-Stokes equations using Oseen
//! linearization (Picard iteration):
//!
//! ```text
//!   −ν Δu + (w·∇)u + ∇p = f    in Ω
//!                   ∇·u = 0    in Ω
//!                     u = u_exact on ∂Ω
//! ```
//!
//! where `w` is the velocity from the previous Picard iteration.
//!
//! The benchmark is the Kovasznay flow (Re = 40) which has an analytical
//! solution on `Ω = [−0.5, 1.5] × [0, 2]`:
//!
//! ```text
//!   u_x = 1 − e^{λx} cos(2πy)
//!   u_y = λ/(2π) e^{λx} sin(2πy)
//!   p   = −e^{2λx}/2 + C
//!   λ   = Re/2 − √(Re²/4 + 4π²)
//! ```
//!
//! ## Usage
//! ```
//! cargo run --example ex_navier_stokes
//! cargo run --example ex_navier_stokes -- --n 16 --re 40
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::PressureDivIntegrator,
    standard::{VectorDiffusionIntegrator, VectorConvectionIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_space::{H1Space, VectorH1Space, fe_space::FESpace, constraints::boundary_dofs};

// ─── Kovasznay analytical solution ───────────────────────────────────────────

fn kovasznay_lambda(re: f64) -> f64 {
    re / 2.0 - (re * re / 4.0 + 4.0 * PI * PI).sqrt()
}

fn kovasznay_u(x: f64, y: f64, lam: f64) -> [f64; 2] {
    let ex = (lam * x).exp();
    [
        1.0 - ex * (2.0 * PI * y).cos(),
        lam / (2.0 * PI) * ex * (2.0 * PI * y).sin(),
    ]
}

fn kovasznay_p(x: f64, _y: f64, lam: f64) -> f64 {
    -0.5 * (2.0 * lam * x).exp()
}

// ─── Mesh generation ─────────────────────────────────────────────────────────

/// Create a triangular mesh on `[x0, x1] × [y0, y1]`.
fn rect_mesh(n: usize, x0: f64, x1: f64, y0: f64, y1: f64) -> SimplexMesh<2> {
    let mut mesh = SimplexMesh::<2>::unit_square_tri(n);
    // Scale coordinates from [0,1]² to [x0,x1] × [y0,y1]
    let nn = mesh.n_nodes();
    for i in 0..nn {
        mesh.coords[i * 2]     = x0 + mesh.coords[i * 2]     * (x1 - x0);
        mesh.coords[i * 2 + 1] = y0 + mesh.coords[i * 2 + 1] * (y1 - y0);
    }
    mesh
}

fn main() {
    let args = parse_args();
    let re = args.re;
    let nu = 1.0 / re;
    let lam = kovasznay_lambda(re);

    println!("=== fem-rs: Navier-Stokes (Kovasznay flow, Oseen/Picard) ===");
    println!("  Re = {re:.0}, ν = {nu:.4e}, λ = {lam:.6}");
    println!("  Mesh: {}×{}, P2/P1", args.n, args.n);

    // ─── 1. Mesh and spaces ──────────────────────────────────────────────────
    let mesh_u = rect_mesh(args.n, -0.5, 1.5, 0.0, 2.0);
    let mesh_p = rect_mesh(args.n, -0.5, 1.5, 0.0, 2.0);

    let space_u = VectorH1Space::new(mesh_u, 2, 2);
    let space_p = H1Space::new(mesh_p, 1);

    let n_u = space_u.n_dofs();
    let n_p = space_p.n_dofs();
    let n_scalar = space_u.n_scalar_dofs();
    println!("  velocity DOFs: {n_u} ({n_scalar} per component)");
    println!("  pressure DOFs: {n_p}");

    // ─── 2. Boundary conditions (exact solution on all boundaries) ───────────
    let scalar_dm = space_u.scalar_dof_manager();
    let mesh_u_ref = space_u.mesh();
    let bnd_all = boundary_dofs(mesh_u_ref, scalar_dm, &[1, 2, 3, 4]);

    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();
    for &d in &bnd_all {
        let coords = scalar_dm.dof_coord(d);
        let (x, y) = (coords[0], coords[1]);
        let u_ex = kovasznay_u(x, y, lam);
        bc_dofs.push(d);
        bc_vals.push(u_ex[0]);
        bc_dofs.push(d + n_scalar as u32);
        bc_vals.push(u_ex[1]);
    }
    println!("  Dirichlet: {} velocity DOFs constrained", bc_dofs.len());

    let quad_order = 5_u8;

    // ─── 3. Assemble B, B^T (constant across Picard iterations) ─────────────
    let b_mat = MixedAssembler::assemble_bilinear(
        &space_p, &space_u, &[&PressureDivIntegrator], quad_order,
    );
    let bt_mat = b_mat.transpose();

    // ─── 4. Picard iteration ─────────────────────────────────────────────────
    // Initialize with Stokes solution (w = 0)
    let mut u_sol = vec![0.0_f64; n_u];
    let mut p_sol = vec![0.0_f64; n_p];

    // Set initial guess to exact BC values on boundary
    for (i, &dof) in bc_dofs.iter().enumerate() {
        u_sol[dof as usize] = bc_vals[i];
    }

    let solver_cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };

    let picard_tol = 1e-8;
    let max_picard = 20;

    for picard in 0..max_picard {
        // Assemble A = ν K + C(w)
        let visc = VectorDiffusionIntegrator { kappa: nu };
        let conv = VectorConvectionIntegrator::new(&u_sol, n_scalar);
        let mut a_mat = Assembler::assemble_bilinear(
            &space_u, &[&visc, &conv], quad_order,
        );

        // RHS = 0 (Kovasznay has no body force when using the exact solution BCs)
        let mut f_u = vec![0.0_f64; n_u];
        let g_p = vec![0.0_f64; n_p];

        // Apply Dirichlet BCs
        fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bc_dofs, &bc_vals);

        // Pin pressure DOF 0
        let mut b_loc = b_mat.clone();
        let mut bt_loc = bt_mat.clone();
        pin_pressure_dof(&mut b_loc, &mut bt_loc, 0);

        // Solve
        let sys = BlockSystem { a: a_mat, bt: bt_loc, b: b_loc, c: None };
        let mut u_new = vec![0.0_f64; n_u];
        let mut p_new = vec![0.0_f64; n_p];

        let res = SchurComplementSolver::solve(
            &sys, &f_u, &g_p, &mut u_new, &mut p_new, &solver_cfg,
        ).expect("Oseen solve failed");

        // Compute Picard residual: ||u_new - u_old||
        let du: f64 = u_new.iter().zip(u_sol.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>().sqrt();
        let u_norm: f64 = u_new.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let rel_du = du / u_norm.max(1e-14);

        println!(
            "  Picard {}: linear iters={}, res={:.2e}, Δu/u={:.2e}",
            picard + 1, res.iterations, res.final_residual, rel_du,
        );

        u_sol = u_new;
        p_sol = p_new;

        if rel_du < picard_tol {
            println!("  Picard converged after {} iterations.", picard + 1);
            break;
        }
    }

    // ─── 5. Error analysis vs analytical solution ────────────────────────────
    let ux = &u_sol[..n_scalar];
    let uy = &u_sol[n_scalar..];

    let mut err_u_l2_sq = 0.0_f64;
    let mut u_ex_l2_sq = 0.0_f64;
    let mut err_p_l2_sq = 0.0_f64;
    let mut p_ex_l2_sq = 0.0_f64;

    // Approximate L² error via nodal evaluation
    for i in 0..n_scalar {
        let coords = scalar_dm.dof_coord(i as u32);
        let (x, y) = (coords[0], coords[1]);
        let u_ex = kovasznay_u(x, y, lam);
        err_u_l2_sq += (ux[i] - u_ex[0]).powi(2) + (uy[i] - u_ex[1]).powi(2);
        u_ex_l2_sq += u_ex[0].powi(2) + u_ex[1].powi(2);
    }

    // Pressure: compare nodal values at P1 DOFs
    let pres_dm = space_p.dof_manager();
    // Shift computed pressure to match exact at node 0
    let p0_exact = kovasznay_p(
        pres_dm.dof_coord(0)[0],
        pres_dm.dof_coord(0)[1],
        lam,
    );
    let p_shift = p0_exact - p_sol[0];
    for i in 0..n_p {
        let coords = pres_dm.dof_coord(i as u32);
        let (x, y) = (coords[0], coords[1]);
        let p_ex = kovasznay_p(x, y, lam);
        let p_h = p_sol[i] + p_shift;
        err_p_l2_sq += (p_h - p_ex).powi(2);
        p_ex_l2_sq += p_ex.powi(2);
    }

    let err_u_rel = (err_u_l2_sq / u_ex_l2_sq.max(1e-30)).sqrt();
    let err_p_rel = (err_p_l2_sq / p_ex_l2_sq.max(1e-30)).sqrt();

    println!("\n  Error vs Kovasznay analytical solution:");
    println!("    velocity L² relative error: {err_u_rel:.4e}");
    println!("    pressure L² relative error: {err_p_rel:.4e}");

    println!("\nDone.");
}

/// Pin pressure DOF `dof` to zero by zeroing its row in B and column in B^T.
fn pin_pressure_dof(
    b: &mut fem_linalg::CsrMatrix<f64>,
    bt: &mut fem_linalg::CsrMatrix<f64>,
    dof: usize,
) {
    for ptr in b.row_ptr[dof]..b.row_ptr[dof + 1] {
        b.values[ptr] = 0.0;
    }
    for i in 0..bt.nrows {
        for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
            if bt.col_idx[ptr] as usize == dof {
                bt.values[ptr] = 0.0;
            }
        }
    }
}

struct Args {
    n: usize,
    re: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 8, re: 40.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"  => { a.n  = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--re" => { a.re = it.next().unwrap_or("40".into()).parse().unwrap_or(40.0); }
            _      => {}
        }
    }
    a
}
