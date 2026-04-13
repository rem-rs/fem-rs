//! # Example 33 — Maxwell with non-homogeneous tangential boundary load
//!
//! Solves the 2-D H(curl) problem
//!
//! ```text
//!   curl curl E + E = f          in Ω = [0,1]²
//!   curl E + γ (n×E) = g         on ∂Ω
//! ```
//!
//! in weak form using the H(curl) boundary terms
//!
//! ```text
//!   ∫ curl E curl v + ∫ E·v + γ ∫ (n×E)(n×v)
//!     = ∫ f·v + ∫ g (n×v)
//! ```
//!
//! with manufactured exact solution
//!
//! ```text
//!   E(x,y) = (sin(πy), sin(πx))
//! ```
//!
//! This example exercises the general non-homogeneous boundary linear form,
//! not just the special case where `curl E` vanishes on the boundary.

use std::f64::consts::PI;
use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const GAMMA: f64 = 2.0;

fn main() {
    let args = parse_args();
    let result = solve_case(args.n);

    println!("=== fem-rs Example 33: Maxwell with tangential boundary load ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    println!("  Edge DOFs: {}", result.n_dofs);
    println!("  Boundary tags: [1, 2, 3, 4], gamma = {:.3}", GAMMA);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  h = {:.4e},  L² error = {:.4e}", result.h, result.l2_error);
}

struct CaseResult {
    n_dofs: usize,
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
    let robin_bdr = [1, 1, 1, 1];
    let problem = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_isotropic_coeffs(1.0, 1.0)
        .with_source_fn(source_value)
        .add_tangential_drive_from_marker(&bdr_attrs, &robin_bdr, GAMMA, boundary_data)
        .build();
    let n_dofs = problem.n_dofs();
    let solved = problem.solve();

    CaseResult {
        n_dofs,
        iterations: solved.solve_result.iterations,
        final_residual: solved.solve_result.final_residual,
        converged: solved.solve_result.converged,
        h: 1.0 / n as f64,
        l2_error: l2_error_hcurl_exact(&solved.space, &solved.solution, exact_field),
    }
}

fn source_value(x: &[f64]) -> [f64; 2] {
    let coeff = 1.0 + PI * PI;
    [coeff * (PI * x[1]).sin(), coeff * (PI * x[0]).sin()]
}

fn exact_field(x: &[f64]) -> [f64; 2] {
    [(PI * x[1]).sin(), (PI * x[0]).sin()]
}

fn curl_exact(x: &[f64]) -> f64 {
    PI * (PI * x[0]).cos() - PI * (PI * x[1]).cos()
}

fn tangential_trace(x: &[f64], normal: &[f64]) -> f64 {
    let e = exact_field(x);
    e[0] * normal[1] - e[1] * normal[0]
}

fn boundary_data(x: &[f64], normal: &[f64]) -> f64 {
    -curl_exact(x) + GAMMA * tangential_trace(x, normal)
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
    fn tangential_drive_maxwell_coarse_mesh_has_reasonable_error() {
        let result = solve_case(8);
        assert!(result.converged);
        assert!(result.l2_error < 1.5e-1, "L2 error = {}", result.l2_error);
    }
}