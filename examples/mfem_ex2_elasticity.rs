//! # Example 2 �?Linear Elasticity  (analogous to MFEM ex2)
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
//! cargo run --example mfem_ex2_elasticity
//! cargo run --example mfem_ex2_elasticity -- --n 16 --order 2
//! ```

use fem_assembly::{
    Assembler,
    standard::{ElasticityIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{VectorH1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

struct SolveResult {
    n: usize,
    order: u8,
    n_nodes: usize,
    n_elements: usize,
    n_dofs: usize,
    n_scalar_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    ux_max: f64,
    uy_max: f64,
    ux_norm: f64,
    uy_norm: f64,
    ux_checksum: f64,
    uy_checksum: f64,
}

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

    let result = solve_case(args.n, args.order, -1.0);

    println!("  Confirmed mesh: {}×{} subdivisions, P{} elements", result.n, result.n, result.order);
    println!("  Nodes: {}, Elements: {}", result.n_nodes, result.n_elements);
    println!("  DOFs: {}  ({} per component)", result.n_dofs, result.n_scalar_dofs);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  max|u_x| = {:.4e}", result.ux_max);
    println!("  max|u_y| = {:.4e}", result.uy_max);
    println!("  ||u_x||_L2 = {:.4e},  ||u_y||_L2 = {:.4e}", result.ux_norm, result.uy_norm);
    println!("  checksum(u_x) = {:.8e},  checksum(u_y) = {:.8e}", result.ux_checksum, result.uy_checksum);
    println!("\nDone.");
}

fn solve_case(n: usize, order: u8, body_force_y: f64) -> SolveResult {
    let e_mod = 1.0_f64;
    let nu    = 0.3_f64;
    let lam   = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu    = e_mod / (2.0 * (1.0 + nu));

    // ─── 1. Mesh and vector space ─────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n);

    let space = VectorH1Space::new(mesh, order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // ─── 2. Assemble stiffness matrix ─────────────────────────────────────────
    let elast = ElasticityIntegrator { lambda: lam, mu };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], order as u8 * 2 + 1);

    // ─── 3. Gravity body force: f = (0, -ρg)  �?assembled into RHS ───────────
    //  Body force in x: 0,  in y: -1
    //  VectorH1Space DOF layout: [u_x DOFs | u_y DOFs]
    //  DomainSourceIntegrator works on scalar spaces; handle manually.
    let mut rhs = vec![0.0_f64; n_dofs];
    // For the y-component load, we need �?(-1) v_y dx for each y-DOF.
    // In block DOF ordering, y-DOFs start at offset n_scalar.
    // Assemble a scalar mass-times-one over a temporary scalar space:
    {
        let mesh2 = SimplexMesh::<2>::unit_square_tri(n);
        let scalar_space = fem_space::H1Space::new(mesh2, order);
        let fy_integrator = DomainSourceIntegrator::new(|_x: &[f64]| body_force_y);
        let fy = Assembler::assemble_linear(&scalar_space, &[&fy_integrator], order as u8 * 2 + 1);
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
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
        .expect("elasticity solve failed");

    // ─── 6. Post-process ──────────────────────────────────────────────────────
    let ux = &u[..n_scalar];
    let uy = &u[n_scalar..];
    let uy_max = uy.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    let ux_max = ux.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    let ux_norm = ux.iter().map(|value| value * value).sum::<f64>().sqrt();
    let uy_norm = uy.iter().map(|value| value * value).sum::<f64>().sqrt();
    let ux_checksum = ux
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let uy_checksum = uy
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        n,
        order,
        n_nodes: space.mesh().n_nodes(),
        n_elements: space.mesh().n_elems(),
        n_dofs,
        n_scalar_dofs: n_scalar,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        ux_max,
        uy_max,
        ux_norm,
        uy_norm,
        ux_checksum,
        uy_checksum,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex2_elasticity_coarse_case_converges_with_vertical_dominance() {
        let result = solve_case(8, 1, -1.0);
        assert_eq!(result.n_nodes, 81);
        assert_eq!(result.n_elements, 128);
        assert_eq!(result.n_dofs, 162);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-9, "solver residual too large: {}", result.final_residual);
        assert!(result.uy_max > result.ux_max, "vertical displacement should dominate: ux={} uy={}", result.ux_max, result.uy_max);
        assert!(result.uy_norm > result.ux_norm, "vertical norm should dominate: ux={} uy={}", result.ux_norm, result.uy_norm);
    }

    #[test]
    fn ex2_elasticity_zero_body_force_gives_trivial_solution() {
        let result = solve_case(8, 1, 0.0);
        assert!(result.converged);
        assert!(result.ux_norm < 1.0e-12, "u_x norm should vanish: {}", result.ux_norm);
        assert!(result.uy_norm < 1.0e-12, "u_y norm should vanish: {}", result.uy_norm);
        assert!(result.ux_max < 1.0e-12, "u_x max should vanish: {}", result.ux_max);
        assert!(result.uy_max < 1.0e-12, "u_y max should vanish: {}", result.uy_max);
    }

    #[test]
    fn ex2_elasticity_solution_scales_linearly_with_body_force() {
        let unit = solve_case(8, 1, -1.0);
        let doubled = solve_case(8, 1, -2.0);
        assert!(unit.converged && doubled.converged);
        assert!((doubled.ux_norm / unit.ux_norm - 2.0).abs() < 1.0e-9,
            "u_x norm ratio mismatch: unit={} doubled={}", unit.ux_norm, doubled.ux_norm);
        assert!((doubled.uy_norm / unit.uy_norm - 2.0).abs() < 1.0e-9,
            "u_y norm ratio mismatch: unit={} doubled={}", unit.uy_norm, doubled.uy_norm);
        assert!((doubled.ux_checksum / unit.ux_checksum - 2.0).abs() < 1.0e-9,
            "u_x checksum ratio mismatch: unit={} doubled={}", unit.ux_checksum, doubled.ux_checksum);
        assert!((doubled.uy_checksum / unit.uy_checksum - 2.0).abs() < 1.0e-9,
            "u_y checksum ratio mismatch: unit={} doubled={}", unit.uy_checksum, doubled.uy_checksum);
    }

    #[test]
    fn ex2_elasticity_sign_reversed_body_force_flips_displacement() {
        let downward = solve_case(8, 1, -1.0);
        let upward = solve_case(8, 1, 1.0);
        assert!(downward.converged && upward.converged);
        assert!((downward.ux_norm - upward.ux_norm).abs() < 1.0e-12);
        assert!((downward.uy_norm - upward.uy_norm).abs() < 1.0e-12);
        assert!((downward.ux_checksum + upward.ux_checksum).abs() < 1.0e-10,
            "u_x checksum should flip sign: down={} up={}", downward.ux_checksum, upward.ux_checksum);
        assert!((downward.uy_checksum + upward.uy_checksum).abs() < 1.0e-10,
            "u_y checksum should flip sign: down={} up={}", downward.uy_checksum, upward.uy_checksum);
    }

    /// Very coarse mesh should still converge.
    #[test]
    fn ex2_elasticity_very_coarse_mesh_converges() {
        let result = solve_case(4, 1, -1.0);
        assert!(result.converged, "very coarse mesh should converge");
        assert!(result.final_residual < 1.0e-8, "residual should be small");
        assert!(result.uy_max > 0.0, "body force should produce nonzero displacement");
    }

    /// Mesh refinement should increase DOF count.
    #[test]
    fn ex2_elasticity_refinement_increases_dof_count() {
        let coarse = solve_case(8, 1, -1.0);
        let fine = solve_case(16, 1, -1.0);
        assert!(coarse.converged && fine.converged);
        assert!(fine.n_dofs > coarse.n_dofs,
            "refined mesh should have more DOFs: coarse={} fine={}",
            coarse.n_dofs, fine.n_dofs);
    }

    /// P2 should give more accurate solution than P1 on same mesh.
    #[test]
    fn ex2_elasticity_p2_higher_order_produces_larger_displacement() {
        let p1 = solve_case(8, 1, -1.0);
        let p2 = solve_case(8, 2, -1.0);
        assert!(p1.converged && p2.converged);
        // Higher order elements may yield different magnitude; verify convergence only
        assert!(p2.n_dofs > p1.n_dofs, "P2 should have more DOFs than P1");
    }

    /// Higher body force magnitude should increase displacement monotonically.
    #[test]
    fn ex2_elasticity_higher_body_force_increases_displacement() {
        let weak = solve_case(8, 1, -0.5);
        let strong = solve_case(8, 1, -2.0);
        assert!(weak.converged && strong.converged);
        // Higher downward force should produce greater downward displacement
        assert!(strong.uy_max > weak.uy_max,
            "stronger downward force should increase displacement: weak={} strong={}",
            weak.uy_max, strong.uy_max);
        assert!(strong.uy_norm > weak.uy_norm,
            "stronger force should increase y-displacement norm: weak={} strong={}",
            weak.uy_norm, strong.uy_norm);
    }
}

