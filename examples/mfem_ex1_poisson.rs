//! # Example 1 �?Poisson/Laplace  (analogous to MFEM ex1)
//!
//! Solves the scalar Poisson equation with homogeneous Dirichlet boundary conditions:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω = [0,1]²
//!            u = 0    on ∂�?
//! ```
//!
//! with the manufactured solution  `u(x,y) = sin(π x) sin(π y)`,  which gives
//! `f = 2 π² sin(π x) sin(π y)` and κ = 1.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex1_poisson
//! cargo run --example mfem_ex1_poisson -- --order 2 --n 32
//! cargo run --example mfem_ex1_poisson -- --n 8   # observe h² convergence
//! cargo run --example mfem_ex1_poisson -- --n 16
//! cargo run --example mfem_ex1_poisson -- --n 32
//! ```
//!
//! ## Output
//! Prints L² error, DOF count, iteration count, and convergence rate.

use std::f64::consts::PI;

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

struct SolveResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: f64,
    solution_l2: f64,
}

fn main() {
    // ─── Parse CLI args ──────────────────────────────────────────────────────
    let args = parse_args();
    let result = solve_case(args.n, args.order, 1.0);

    println!("=== fem-rs Example 1: Poisson equation ===");
    println!("  Mesh:  {}×{} subdivisions, P{} elements", args.n, args.n, args.order);

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());
    println!("  DOFs:  {}", result.n_dofs);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  h = {:.4e},  L² error = {:.4e}", result.h, result.l2_error);
    println!("  ||u||₂ = {:.4e}", result.solution_l2);
    println!("  (Expected O(h^{}) for P{} elements)", args.order + 1, args.order);

    println!("\nDone. (No VTK output in this minimal example �?add fem-io to enable.)");
}

fn solve_case(n_subdiv: usize, order: u8, source_scale: f64) -> SolveResult {
    // ─── 1. Create mesh ──────────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n_subdiv);

    // ─── 2. Create H¹ finite element space ──────────────────────────────────
    let space = H1Space::new(mesh, order);
    let n = space.n_dofs();

    // ─── 3. Assemble bilinear form A = ∫∇u·∇v dx ───────────────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], order * 2 + 1);

    // ─── 4. Assemble linear form f = ∫2π² sin(πx)sin(πy) v dx ─────────────
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        source_scale * 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&source], order * 2 + 1);

    // ─── 5. Apply homogeneous Dirichlet BCs on all four walls ────────────────
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    // ─── 6. Solve K u = f with PCG + Jacobi preconditioner ──────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
        .expect("solver failed");

    // ─── 7. L² error against exact solution u = sin(πx)sin(πy) ─────────────
    let l2 = l2_error_h1(&space, &u, |x: &[f64]| source_scale * (PI * x[0]).sin() * (PI * x[1]).sin());
    let solution_l2 = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    SolveResult {
        n_dofs: n,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        h: 1.0 / n_subdiv as f64,
        l2_error: l2,
        solution_l2,
    }
}

// ─── L² error helper ─────────────────────────────────────────────────────────

/// Compute the L² error ‖u_h �?u_exact‖_{L²(Ω)} using element quadrature.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex1_poisson_coarse_mesh_has_reasonable_error() {
        let result = solve_case(8, 1, 1.0);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-8, "residual = {}", result.final_residual);
        assert!(result.l2_error < 2.2e-2, "L2 error = {}", result.l2_error);
    }

    #[test]
    fn ex1_poisson_refinement_improves_p1_l2_error() {
        let coarse = solve_case(8, 1, 1.0);
        let fine = solve_case(16, 1, 1.0);

        assert!(coarse.converged && fine.converged);
        assert!(fine.l2_error < coarse.l2_error,
            "expected refinement to reduce error: coarse={} fine={}",
            coarse.l2_error, fine.l2_error);

        let observed_order = (coarse.l2_error / fine.l2_error).ln() / (coarse.h / fine.h).ln();
        assert!(observed_order > 1.5, "observed L2 order too low: {}", observed_order);
    }

    #[test]
    fn ex1_poisson_p2_is_more_accurate_than_p1_on_same_mesh() {
        let p1 = solve_case(8, 1, 1.0);
        let p2 = solve_case(8, 2, 1.0);

        assert!(p1.converged && p2.converged);
        assert!(
            p2.l2_error < p1.l2_error,
            "expected P2 to improve accuracy on the same mesh: p1={} p2={}",
            p1.l2_error,
            p2.l2_error
        );
    }

    #[test]
    fn ex1_poisson_sign_reversed_source_flips_solution() {
        let positive = solve_case(8, 1, 1.0);
        let negative = solve_case(8, 1, -1.0);

        assert!(positive.converged && negative.converged);
        let norm_rel_gap = (positive.solution_l2 - negative.solution_l2).abs()
            / positive.solution_l2.max(negative.solution_l2).max(1.0e-30);
        let error_rel_gap = (positive.l2_error - negative.l2_error).abs()
            / positive.l2_error.max(negative.l2_error).max(1.0e-30);

        assert!(norm_rel_gap < 1.0e-12, "expected solution norm invariance under sign reversal, got {}", norm_rel_gap);
        assert!(error_rel_gap < 1.0e-12, "expected L2 error invariance under sign reversal, got {}", error_rel_gap);
    }

    /// Very coarse mesh should still converge.
    #[test]
    fn ex1_poisson_very_coarse_mesh_converges() {
        let result = solve_case(4, 1, 1.0);
        assert!(result.converged, "even very coarse mesh should converge");
        assert!(result.final_residual < 1.0e-7, "residual = {}", result.final_residual);
    }

    /// P1 convergence should be observed between n=8 and n=16.
    #[test]
    fn ex1_poisson_p1_shows_h_squared_convergence() {
        let n8 = solve_case(8, 1, 1.0);
        let n16 = solve_case(16, 1, 1.0);

        assert!(n8.converged && n16.converged);
        assert!(n16.l2_error < n8.l2_error);

        // h² convergence: error ~ C·h²
        // If h₁ = 2·h₂, then error₁ ≈ 4·error₂ (ratio ≈ 4)
        let ratio = n8.l2_error / n16.l2_error;
        assert!(
            ratio > 2.5 && ratio < 5.5,
            "P1 should show O(h²) convergence: expected ratio ~4, got {:.2}",
            ratio
        );
    }

    /// P2 convergence should be better than P1 on the same mesh.
    #[test]
    fn ex1_poisson_p2_converges_faster_than_p1() {
        let n8_p1 = solve_case(8, 1, 1.0);
        let n8_p2 = solve_case(8, 2, 1.0);

        assert!(n8_p1.converged && n8_p2.converged);
        // P2 should always be more accurate than P1 on same mesh
        assert!(
            n8_p2.l2_error < n8_p1.l2_error,
            "P2 should be more accurate than P1 on same mesh: P1={:.3e} vs P2={:.3e}",
            n8_p1.l2_error, n8_p2.l2_error
        );

        // P2 should show faster convergence rate
        let n16_p1 = solve_case(16, 1, 1.0);
        let n16_p2 = solve_case(16, 2, 1.0);
        let ratio_p1 = n8_p1.l2_error / n16_p1.l2_error;
        let ratio_p2 = n8_p2.l2_error / n16_p2.l2_error;
        assert!(
            ratio_p2 > ratio_p1,
            "P2 should have higher convergence rate: P1 ratio={:.2} vs P2 ratio={:.2}",
            ratio_p1, ratio_p2
        );
    }

    /// Solution norm should scale roughly linearly with source scaling factor (superposition).
    #[test]
    fn ex1_poisson_solution_scales_linearly_with_source() {
        let scale_1 = solve_case(8, 1, 1.0);
        let scale_2 = solve_case(8, 1, 2.0);

        assert!(scale_1.converged && scale_2.converged);
        // If source f scales by 2, solution scales by 2: ||u(2f)|| ≈ 2·||u(f)||
        let ratio = scale_2.solution_l2 / scale_1.solution_l2;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "solution norm should scale linearly with source: expected ~2.0, got {:.3}",
            ratio
        );
    }
}

