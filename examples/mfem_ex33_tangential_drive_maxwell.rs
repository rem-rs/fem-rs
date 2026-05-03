//! # Example 33 �?Maxwell with non-homogeneous tangential boundary load
//!
//! Solves the 2-D H(curl) problem
//!
//! ```text
//!   curl curl E + E = f          in Ω = [0,1]²
//!   curl E + γ (n×E) = g         on ∂�?
//! ```
//!
//! in weak form using the H(curl) boundary terms
//!
//! ```text
//!   �?curl E curl v + �?E·v + γ �?(n×E)(n×v)
//!     = �?f·v + �?g (n×v)
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

const DEFAULT_GAMMA: f64 = 2.0;
const DEFAULT_SCALE: f64 = 1.0;

fn main() {
    let args = parse_args();
    let result = solve_case(args.n);

    println!("=== fem-rs Example 33: Maxwell with tangential boundary load ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    println!("  Edge DOFs: {}", result.n_dofs);
    println!("  Boundary tags: [1, 2, 3, 4], gamma = {:.3}", DEFAULT_GAMMA);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  h = {:.4e},  L² error = {:.4e}", result.h, result.l2_error);
    println!("  ||u||₂ = {:.4e}", result.solution_l2);
}

struct CaseResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: f64,
    solution_l2: f64,
}

fn solve_case(n: usize) -> CaseResult {
    solve_case_with_gamma_and_scale(n, DEFAULT_GAMMA, DEFAULT_SCALE)
}

#[cfg(test)]
fn solve_case_with_gamma(n: usize, gamma: f64) -> CaseResult {
    solve_case_with_gamma_and_scale(n, gamma, DEFAULT_SCALE)
}

fn solve_case_with_gamma_and_scale(n: usize, gamma: f64, scale: f64) -> CaseResult {
    solve_case_with_gamma_and_scale_and_field(n, gamma, scale).0
}

fn solve_case_with_gamma_and_scale_and_field(n: usize, gamma: f64, scale: f64) -> (CaseResult, Vec<f64>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = HCurlSpace::new(mesh, 1);

    let bdr_attrs = [1, 2, 3, 4];
    let robin_bdr = [1, 1, 1, 1];
    let problem = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_isotropic_coeffs(1.0, 1.0)
        .with_source_fn(move |x| source_value(x, scale))
        .add_tangential_drive_from_marker(&bdr_attrs, &robin_bdr, gamma, move |x, normal| {
            boundary_data(x, normal, gamma, scale)
        })
        .build();
    let n_dofs = problem.n_dofs();
    let solved = problem.solve();
    let solution_l2 = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();

    (
        CaseResult {
            n_dofs,
            iterations: solved.solve_result.iterations,
            final_residual: solved.solve_result.final_residual,
            converged: solved.solve_result.converged,
            h: 1.0 / n as f64,
            l2_error: l2_error_hcurl_exact(&solved.space, &solved.solution, |x| exact_field(x, scale)),
            solution_l2,
        },
        solved.solution,
    )
}

fn source_value(x: &[f64], scale: f64) -> [f64; 2] {
    let coeff = 1.0 + PI * PI;
    [scale * coeff * (PI * x[1]).sin(), scale * coeff * (PI * x[0]).sin()]
}

fn exact_field(x: &[f64], scale: f64) -> [f64; 2] {
    [scale * (PI * x[1]).sin(), scale * (PI * x[0]).sin()]
}

fn curl_exact(x: &[f64], scale: f64) -> f64 {
    scale * (PI * (PI * x[0]).cos() - PI * (PI * x[1]).cos())
}

fn tangential_trace(x: &[f64], normal: &[f64], scale: f64) -> f64 {
    let e = exact_field(x, scale);
    e[0] * normal[1] - e[1] * normal[0]
}

fn boundary_data(x: &[f64], normal: &[f64], gamma: f64, scale: f64) -> f64 {
    -curl_exact(x, scale) + gamma * tangential_trace(x, normal, scale)
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

    #[test]
    fn tangential_drive_maxwell_h_refinement_reduces_l2_error() {
        let coarse = solve_case(8);
        let fine   = solve_case(16);
        assert!(coarse.converged && fine.converged, "both solves must converge");
        assert!(fine.l2_error < coarse.l2_error,
            "finer mesh should have smaller L2 error: n=8: {:.3e}, n=16: {:.3e}",
            coarse.l2_error, fine.l2_error);
        let ratio = coarse.l2_error / fine.l2_error;
        assert!(ratio > 1.5, "expected >1.5x error reduction on mesh doubling (ND1 O(h)), got {:.2}", ratio);
    }

    #[test]
    fn tangential_drive_maxwell_converged_residual_is_small() {
        let result = solve_case(16);
        assert!(result.converged, "solver must converge at n=16");
        assert!(result.final_residual < 1e-8, "expected residual < 1e-8 after convergence, got {:.3e}", result.final_residual);
    }

    #[test]
    fn tangential_drive_maxwell_refines_monotonically_on_practical_meshes() {
        let coarse = solve_case_with_gamma(8, DEFAULT_GAMMA);
        let medium = solve_case_with_gamma(16, DEFAULT_GAMMA);

        assert!(coarse.converged && medium.converged);
        assert!(
            medium.l2_error < coarse.l2_error,
            "expected refinement to reduce error: coarse={} medium={}",
            coarse.l2_error,
            medium.l2_error
        );
    }

    #[test]
    fn tangential_drive_maxwell_remains_accurate_for_gamma_variations() {
        let weak = solve_case_with_gamma(8, 0.5);
        let strong = solve_case_with_gamma(8, 4.0);

        assert!(weak.converged && strong.converged);
        assert!(weak.final_residual < 1.0e-6, "weak-gamma residual = {}", weak.final_residual);
        assert!(strong.final_residual < 1.0e-6, "strong-gamma residual = {}", strong.final_residual);
        assert!(weak.l2_error < 1.5e-1, "weak-gamma L2 error = {}", weak.l2_error);
        assert!(strong.l2_error < 1.5e-1, "strong-gamma L2 error = {}", strong.l2_error);
    }

    #[test]
    fn tangential_drive_maxwell_solution_scales_linearly_with_boundary_drive() {
        let half = solve_case_with_gamma_and_scale(8, DEFAULT_GAMMA, 0.5);
        let full = solve_case_with_gamma_and_scale(8, DEFAULT_GAMMA, 1.0);

        assert!(half.converged && full.converged);
        let ratio = full.solution_l2 / half.solution_l2.max(1.0e-30);
        assert!(
            (ratio - 2.0).abs() < 1.0e-6,
            "expected linear response to tangential drive scaling, got ratio {}",
            ratio
        );
    }

    #[test]
    fn tangential_drive_maxwell_sign_reversed_boundary_drive_flips_solution() {
        let (positive, u_pos) = solve_case_with_gamma_and_scale_and_field(8, DEFAULT_GAMMA, 1.0);
        let (negative, u_neg) = solve_case_with_gamma_and_scale_and_field(8, DEFAULT_GAMMA, -1.0);

        assert!(positive.converged && negative.converged);
        assert_eq!(u_pos.len(), u_neg.len());

        let symmetry_err = u_pos
            .iter()
            .zip(&u_neg)
            .map(|(a, b)| (a + b).abs())
            .fold(0.0_f64, f64::max);
        let norm_rel_gap = (positive.solution_l2 - negative.solution_l2).abs()
            / positive.solution_l2.max(negative.solution_l2).max(1.0e-30);

        assert!(
            symmetry_err < 1.0e-10,
            "expected tangential-drive solution vector to flip sign under sign-reversed drive, got max symmetry error {}",
            symmetry_err
        );
        assert!(
            norm_rel_gap < 1.0e-12,
            "expected tangential-drive solution norm to remain invariant under sign reversal, got relative gap {}",
            norm_rel_gap
        );
    }

    /// Zero tangential drive must yield a trivial (zero) solution.
    #[test]
    fn tangential_drive_maxwell_zero_drive_gives_trivial_solution() {
        let result = solve_case_with_gamma_and_scale(8, DEFAULT_GAMMA, 0.0);
        assert!(result.converged);
        assert!(result.solution_l2 < 1.0e-14,
            "expected zero solution norm for zero drive, got {}", result.solution_l2);
        assert!(result.l2_error < 1.0e-14,
            "expected zero L2 error for zero drive, got {}", result.l2_error);
    }
}
