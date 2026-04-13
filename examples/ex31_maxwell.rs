//! # Example 31 — Anisotropic Maxwell problem
//!
//! Solves the 2-D H(curl) problem
//!
//! ```text
//!   curl curl E + Σ E = f    in Ω = [0,1]²
//!              n×E = 0       on ∂Ω
//! ```
//!
//! with a constant anisotropic conductivity/permittivity tensor
//! `Σ = diag(σ_x, σ_y)` and the manufactured solution
//! `E = (sin(πy), sin(πx))`.

use std::f64::consts::PI;
use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const SIGMA_X: f64 = 4.0;
const SIGMA_Y: f64 = 1.5;

fn main() {
    let args = parse_args();
    let result = solve_case(args.n);

    println!("=== fem-rs Example 31: Anisotropic Maxwell ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    println!("  DOFs: {}", result.n_dofs);
    println!("  Boundary DOFs constrained: {}", result.n_boundary_dofs);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  h = {:.4e},  L² error = {:.4e}", result.h, result.l2_error);
    println!("  Σ = diag({SIGMA_X:.3}, {SIGMA_Y:.3})");
}

struct CaseResult {
    n_dofs: usize,
    n_boundary_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: f64,
}

fn solve_case(n: usize) -> CaseResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = HCurlSpace::new(mesh, 1);

    let bdr_attrs = [1, 2, 3, 4];
    let ess_bdr = [1, 1, 1, 1];
    let problem = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_anisotropic_diag(1.0, SIGMA_X, SIGMA_Y)
        .with_source_fn(source_value)
        .add_pec_zero_from_marker(&bdr_attrs, &ess_bdr)
        .build();
    let n_dofs = problem.n_dofs();
    let solved = problem.solve();

    CaseResult {
        n_dofs,
        n_boundary_dofs: solved.boundary_report.essential_dofs,
        iterations: solved.solve_result.iterations,
        final_residual: solved.solve_result.final_residual,
        converged: solved.solve_result.converged,
        h: 1.0 / n as f64,
        l2_error: l2_error_hcurl_exact(&solved.space, &solved.solution, |x| {
            [(PI * x[1]).sin(), (PI * x[0]).sin()]
        }),
    }
}

fn source_value(x: &[f64]) -> [f64; 2] {
    let fx = (PI * PI + SIGMA_X) * (PI * x[1]).sin();
    let fy = (PI * PI + SIGMA_Y) * (PI * x[0]).sin();
    [fx, fy]
}

struct Args {
    n: usize,
}

fn parse_args() -> Args {
    let mut a = Args { n: 16 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if arg.as_str() == "--n" {
            a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16);
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anisotropic_maxwell_coarse_mesh_has_reasonable_error() {
        let result = solve_case(8);
        assert!(result.converged);
        assert!(result.l2_error < 2.5e-1, "L2 error = {}", result.l2_error);
    }
}