//! # Example 2 — Linear Elasticity  (analogous to MFEM ex2)
//!
//! Solves the linear elasticity system with a body force (gravity):
//!
//! ```text
//!   −∇·σ(u) = f    in Ω = [0,1]²
//!         u = 0    on ∂Ω_D  (clamped left wall, x=0)
//!   σ(u)·n = 0    on ∂Ω_N  (traction-free elsewhere)
//! ```
//!
//! where σ = λ tr(ε) I + 2μ ε is the Cauchy stress.
//!
//! Material: steel-like (E = 200 GPa, ν = 0.3) scaled to unit dimensions.
//!
//! ## Usage
//! ```
//! cargo run --example ex2_elasticity
//! cargo run --example ex2_elasticity -- --n 16 --order 2
//! ```

use fem_assembly::{
    Assembler,
    standard::{ElasticityIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{VectorH1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 2: Linear Elasticity ===");
    println!("  Mesh: {}×{} subdivisions, P{} elements", args.n, args.n, args.order);

    // ─── Lamé parameters (E=1, ν=0.3) ───────────────────────────────────────
    let e_mod = 1.0_f64;
    let nu    = 0.3_f64;
    let lam   = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu    = e_mod / (2.0 * (1.0 + nu));
    println!("  λ = {lam:.4},  μ = {mu:.4}");

    // ─── 1. Mesh and vector space ─────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());

    let space = VectorH1Space::new(mesh, args.order, 2);
    let n = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();
    println!("  DOFs: {n}  ({n_scalar} per component)");

    // ─── 2. Assemble stiffness matrix ─────────────────────────────────────────
    let elast = ElasticityIntegrator { lambda: lam, mu };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], args.order as u8 * 2 + 1);

    // ─── 3. Gravity body force: f = (0, -ρg)  → assembled into RHS ───────────
    //  Body force in x: 0,  in y: -1
    //  VectorH1Space DOF layout: [u_x DOFs | u_y DOFs]
    //  DomainSourceIntegrator works on scalar spaces; handle manually.
    let mut rhs = vec![0.0_f64; n];
    // For the y-component load, we need ∫ (-1) v_y dx for each y-DOF.
    // In block DOF ordering, y-DOFs start at offset n_scalar.
    // Assemble a scalar mass-times-one over a temporary scalar space:
    {
        let mesh2 = SimplexMesh::<2>::unit_square_tri(args.n);
        let scalar_space = fem_space::H1Space::new(mesh2, args.order);
        let fy_integrator = DomainSourceIntegrator::new(|_x: &[f64]| -1.0_f64);
        let fy = Assembler::assemble_linear(&scalar_space, &[&fy_integrator], args.order as u8 * 2 + 1);
        // Add to y-component block of RHS (offset n_scalar)
        for (i, &v) in fy.iter().enumerate() {
            rhs[n_scalar + i] += v;
        }
    }

    // ─── 4. Dirichlet BC: clamp left wall (x=0, tag 4) ───────────────────────
    // Both u_x and u_y = 0 on the left boundary.
    // boundary_dofs uses scalar DofManager for VectorH1Space:
    let scalar_dm = space.scalar_dof_manager();
    let bnd_scalar = boundary_dofs(space.mesh(), scalar_dm, &[4]); // left wall
    // u_x DOFs (block 0) and u_y DOFs (block 1)
    let mut clamped: Vec<u32> = Vec::new();
    for &d in &bnd_scalar {
        clamped.push(d);                       // x-DOF
        clamped.push(d + n_scalar as u32);     // y-DOF
    }
    let vals = vec![0.0_f64; clamped.len()];
    apply_dirichlet(&mut mat, &mut rhs, &clamped, &vals);

    // ─── 5. Solve ─────────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
        .expect("elasticity solve failed");

    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 6. Post-process ──────────────────────────────────────────────────────
    let ux = &u[..n_scalar];
    let uy = &u[n_scalar..];
    let u_max = uy.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    let ux_max = ux.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    println!("  max|u_x| = {ux_max:.4e}");
    println!("  max|u_y| = {u_max:.4e}");
    println!("\nDone.");
}

struct Args { n: usize, order: u8 }

fn parse_args() -> Args {
    let mut a = Args { n: 8, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n     = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--order" => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            _ => {}
        }
    }
    a
}
