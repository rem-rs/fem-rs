//! # Taylor-Hood P2-P1 Stokes Example (Phase 40)
//!
//! Solves the steady Stokes equations on a lid-driven cavity:
//!
//! ```text
//!   −ν Δu + ∇p = 0    in Ω = [0,1]²
//!        ∇·u = 0    in Ω
//!          u = (1,0)  on top wall   (lid, tag 3)
//!          u = (0,0)  on other walls (tags 1,2,4)
//! ```
//!
//! Uses the inf-sup stable Taylor-Hood pair: P2 velocity / P1 pressure.
//!
//! The saddle-point system is:
//! ```text
//!   [ A   B^T ] [ u ]   [ f ]
//!   [ B    0  ] [ p ] = [ 0 ]
//! ```
//!
//! where A is the vector-Laplacian (viscous term) and B is the divergence
//! operator coupling velocity and pressure.
//!
//! ## Usage
//! ```
//! cargo run --example ex_stokes
//! cargo run --example ex_stokes -- --n 16 --nu 1.0
//! ```

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::PressureDivIntegrator,
    standard::VectorDiffusionIntegrator,
};
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_space::{H1Space, VectorH1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    let args = parse_args();
    println!("=== fem-rs: Taylor-Hood P2-P1 Stokes (lid-driven cavity) ===");
    println!("  Mesh: {}×{}, P2/P1, ν = {:.3e}", args.n, args.n, args.nu);

    // ─── 1. Mesh and spaces ──────────────────────────────────────────────────
    let mesh_u = SimplexMesh::<2>::unit_square_tri(args.n);
    let mesh_p = SimplexMesh::<2>::unit_square_tri(args.n);

    // Velocity: P2 vector, 2 components
    let space_u = VectorH1Space::new(mesh_u, 2, 2);
    // Pressure: P1 scalar
    let space_p = H1Space::new(mesh_p, 1);

    let nu = space_u.n_dofs();
    let np = space_p.n_dofs();
    let n_scalar = space_u.n_scalar_dofs();
    println!("  velocity DOFs: {nu} ({n_scalar} per component)");
    println!("  pressure DOFs: {np}");

    // ─── 2. Assemble blocks ──────────────────────────────────────────────────
    let quad_order = 5_u8; // P2 gradients need ≥ order 3

    // A = ν ∫ ∇uᵢ·∇vᵢ dx  (vector Laplacian)
    let visc = VectorDiffusionIntegrator { kappa: args.nu };
    let mut a_mat = Assembler::assemble_bilinear(&space_u, &[&visc], quad_order);

    // B: n_p × n_u  (PressureDivIntegrator: −∫ p (∇·u) dx, row=pres, col=vel)
    let b_mat = MixedAssembler::assemble_bilinear(
        &space_p, &space_u, &[&PressureDivIntegrator], quad_order,
    );
    // B^T = transpose of B
    let bt_mat = b_mat.transpose();

    // ─── 3. RHS ──────────────────────────────────────────────────────────────
    // No body force, so f = 0 initially (BCs will modify it).
    let mut f_u = vec![0.0_f64; nu];
    let g_p = vec![0.0_f64; np];

    // ─── 4. Dirichlet BCs ────────────────────────────────────────────────────
    // Boundary tags: 1=bottom, 2=right, 3=top (lid), 4=left
    let scalar_dm = space_u.scalar_dof_manager();
    let mesh_u_ref = space_u.mesh();

    // All walls: u = 0
    let bnd_all = boundary_dofs(mesh_u_ref, scalar_dm, &[1, 2, 3, 4]);
    // Lid (top wall): u_x = 1
    let bnd_lid = boundary_dofs(mesh_u_ref, scalar_dm, &[3]);

    // Build constrained DOF list + values.
    // Global DOF layout: [u_x(0..n_scalar) | u_y(0..n_scalar)]
    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();

    for &d in &bnd_all {
        // u_x DOF
        bc_dofs.push(d);
        // Lid nodes get u_x = 1, others get u_x = 0
        let is_lid = bnd_lid.contains(&d);
        bc_vals.push(if is_lid { 1.0 } else { 0.0 });
        // u_y DOF (always 0)
        bc_dofs.push(d + n_scalar as u32);
        bc_vals.push(0.0);
    }
    fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bc_dofs, &bc_vals);
    println!("  Dirichlet: {} velocity DOFs constrained", bc_dofs.len());

    // ─── 5. Pin pressure DOF 0 (remove constant nullspace) ───────────────────
    // Zero row/col 0 in B, B^T, set p₀ = 0.
    let mut b_mat = b_mat;
    let mut bt_mat = bt_mat;
    pin_pressure_dof(&mut b_mat, &mut bt_mat, 0);

    // ─── 6. Solve ────────────────────────────────────────────────────────────
    let sys = BlockSystem { a: a_mat, bt: bt_mat, b: b_mat, c: None };
    let mut u_sol = vec![0.0_f64; nu];
    let mut p_sol = vec![0.0_f64; np];

    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 1e-12,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = SchurComplementSolver::solve(&sys, &f_u, &g_p, &mut u_sol, &mut p_sol, &cfg)
        .expect("Stokes solve failed");

    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 7. Post-process ─────────────────────────────────────────────────────
    let ux = &u_sol[..n_scalar];
    let uy = &u_sol[n_scalar..];
    let ux_max = ux.iter().cloned().fold(0.0_f64, f64::max);
    let ux_min = ux.iter().cloned().fold(0.0_f64, f64::min);
    let uy_max = uy.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let p_max = p_sol.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p_min = p_sol.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  u_x range: [{ux_min:.4e}, {ux_max:.4e}]");
    println!("  max|u_y|:  {uy_max:.4e}");
    println!("  p range:   [{p_min:.4e}, {p_max:.4e}]");

    // Verify block residual: ||Au + B^Tp - f|| + ||Bu - g||
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    sys.apply(&u_sol, &p_sol, &mut ru, &mut rp);
    let err_u: f64 = ru.iter().zip(f_u.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let err_p: f64 = rp.iter().zip(g_p.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    println!("  Block residual: ‖Au+B^Tp−f‖ = {err_u:.3e},  ‖Bu−g‖ = {err_p:.3e}");

    // Divergence check: ||Bu|| should be small (incompressibility)
    println!("  Divergence: ‖∇·u‖ = {err_p:.3e}");

    println!("\nDone.");
}

/// Pin pressure DOF `dof` to zero by zeroing its row in B and column in B^T.
fn pin_pressure_dof(
    b: &mut fem_linalg::CsrMatrix<f64>,
    bt: &mut fem_linalg::CsrMatrix<f64>,
    dof: usize,
) {
    // Zero row `dof` in B (n_p × n_u)
    for ptr in b.row_ptr[dof]..b.row_ptr[dof + 1] {
        b.values[ptr] = 0.0;
    }
    // Zero column `dof` in B^T (n_u × n_p)
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
    nu: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 8, nu: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"  => { a.n  = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--nu" => { a.nu = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            _      => {}
        }
    }
    a
}
