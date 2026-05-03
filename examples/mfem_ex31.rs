//! # Example 31 �?Anisotropic Maxwell problem
//!
//! Solves the 2-D H(curl) problem
//!
//! ```text
//!   curl curl E + Σ E = f    in Ω = [0,1]²
//!              n×E = 0       on ∂�?
//! ```
//!
//! with a constant anisotropic conductivity/permittivity tensor
//! `Σ = diag(σ_x, σ_y)` and the manufactured solution
//! `E = (sin(πy), sin(πx))`.

use std::f64::consts::PI;
use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const DEFAULT_SIGMA_X: f64 = 4.0;
const DEFAULT_SIGMA_Y: f64 = 1.5;
const DEFAULT_SCALE: f64 = 1.0;

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
    println!("  ||u||₂ = {:.4e}", result.solution_l2);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("  Σ = diag({DEFAULT_SIGMA_X:.3}, {DEFAULT_SIGMA_Y:.3})");
}

struct CaseResult {
    n_dofs: usize,
    n_boundary_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: f64,
    solution_l2: f64,
    solution_checksum: f64,
}

fn solve_case(n: usize) -> CaseResult {
    solve_case_with_sigma_and_scale(n, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, DEFAULT_SCALE)
}

#[cfg(test)]
fn solve_case_with_sigma(n: usize, sigma_x: f64, sigma_y: f64) -> CaseResult {
    solve_case_with_sigma_and_scale(n, sigma_x, sigma_y, DEFAULT_SCALE)
}

fn solve_case_with_sigma_and_scale(n: usize, sigma_x: f64, sigma_y: f64, scale: f64) -> CaseResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = HCurlSpace::new(mesh, 1);

    let bdr_attrs = [1, 2, 3, 4];
    let ess_bdr = [1, 1, 1, 1];
    let problem = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_anisotropic_diag(1.0, sigma_x, sigma_y)
        .with_source_fn(move |x| source_value(x, sigma_x, sigma_y, scale))
        .add_pec_zero_from_marker(&bdr_attrs, &ess_bdr)
        .build();
    let n_dofs = problem.n_dofs();
    let solved = problem.solve();
    let solution_l2 = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();
    let solution_checksum = solved.solution
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    CaseResult {
        n_dofs,
        n_boundary_dofs: solved.boundary_report.essential_dofs,
        iterations: solved.solve_result.iterations,
        final_residual: solved.solve_result.final_residual,
        converged: solved.solve_result.converged,
        h: 1.0 / n as f64,
        l2_error: l2_error_hcurl_exact(&solved.space, &solved.solution, |x| {
            [scale * (PI * x[1]).sin(), scale * (PI * x[0]).sin()]
        }),
        solution_l2,
        solution_checksum,
    }
}

fn source_value(x: &[f64], sigma_x: f64, sigma_y: f64, scale: f64) -> [f64; 2] {
    let fx = scale * (PI * PI + sigma_x) * (PI * x[1]).sin();
    let fy = scale * (PI * PI + sigma_y) * (PI * x[0]).sin();
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

    #[test]
    fn anisotropic_maxwell_exhibits_first_order_hcurl_convergence_trend() {
        let coarse = solve_case(8);
        let medium = solve_case(16);
        let fine = solve_case(32);

        assert!(coarse.converged && medium.converged && fine.converged);

        let order_1 = (coarse.l2_error / medium.l2_error).ln() / (coarse.h / medium.h).ln();
        let order_2 = (medium.l2_error / fine.l2_error).ln() / (medium.h / fine.h).ln();

        assert!(
            order_1 > 0.85,
            "coarse->medium observed order too low: order={} (errors {} -> {})",
            order_1,
            coarse.l2_error,
            medium.l2_error
        );
        assert!(
            order_2 > 0.85,
            "medium->fine observed order too low: order={} (errors {} -> {})",
            order_2,
            medium.l2_error,
            fine.l2_error
        );
    }

    #[test]
    fn anisotropic_maxwell_swapping_principal_axes_preserves_error_by_symmetry() {
        let xy = solve_case_with_sigma(12, 4.0, 1.5);
        let yx = solve_case_with_sigma(12, 1.5, 4.0);

        assert!(xy.converged && yx.converged);

        let rel_gap = (xy.l2_error - yx.l2_error).abs() / xy.l2_error.max(yx.l2_error).max(1e-30);
        assert!(
            rel_gap < 1.0e-8,
            "swapping anisotropic principal values should preserve error by symmetry: rel_gap={}",
            rel_gap
        );
    }

    #[test]
    fn anisotropic_maxwell_uniform_sigma_rescaling_preserves_solution_response() {
        let base = solve_case_with_sigma(12, 4.0, 1.5);
        let scaled = solve_case_with_sigma(12, 8.0, 3.0);

        assert!(base.converged && scaled.converged);

        let err_rel_gap = (base.l2_error - scaled.l2_error).abs()
            / base.l2_error.max(scaled.l2_error).max(1e-30);
        let sol_rel_gap = (base.solution_l2 - scaled.solution_l2).abs()
            / base.solution_l2.max(scaled.solution_l2).max(1e-30);

        assert!(
            err_rel_gap < 1.0e-3,
            "uniform sigma rescaling should preserve manufactured-solution error: rel_gap={}",
            err_rel_gap
        );
        assert!(
            sol_rel_gap < 1.0e-3,
            "uniform sigma rescaling should preserve solution norm: rel_gap={}",
            sol_rel_gap
        );
    }

    #[test]
    fn anisotropic_maxwell_solution_scales_linearly_with_source_amplitude() {
        let half = solve_case_with_sigma_and_scale(12, 4.0, 1.5, 0.5);
        let full = solve_case_with_sigma_and_scale(12, 4.0, 1.5, 1.0);

        assert!(half.converged && full.converged);
        let ratio = full.solution_l2 / half.solution_l2.max(1e-30);
        let err_ratio = full.l2_error / half.l2_error.max(1e-30);

        assert!(
            (ratio - 2.0).abs() < 1.0e-6,
            "expected anisotropic Maxwell solution norm to scale linearly, got ratio {}",
            ratio
        );
        assert!(
            (err_ratio - 2.0).abs() < 1.0e-6,
            "expected anisotropic Maxwell error to scale linearly, got ratio {}",
            err_ratio
        );
        let checksum_ratio = full.solution_checksum / half.solution_checksum.max(1e-30);
        assert!(
            (checksum_ratio - 2.0).abs() < 1.0e-6,
            "expected anisotropic Maxwell checksum to scale linearly, got ratio {}",
            checksum_ratio
        );
    }

    #[test]
    fn anisotropic_maxwell_sign_reversed_source_flips_solution() {
        let positive = solve_case_with_sigma_and_scale(12, 4.0, 1.5, 1.0);
        let negative = solve_case_with_sigma_and_scale(12, 4.0, 1.5, -1.0);

        assert!(positive.converged && negative.converged);
        assert!((positive.solution_l2 - negative.solution_l2).abs() < 1.0e-12,
            "solution norm should be sign-invariant: positive={} negative={}",
            positive.solution_l2,
            negative.solution_l2);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
        assert!((positive.l2_error - negative.l2_error).abs() < 1.0e-12,
            "manufactured-solution error should be sign-invariant: positive={} negative={}",
            positive.l2_error,
            negative.l2_error);
    }

    #[test]
    fn anisotropic_maxwell_zero_source_gives_trivial_solution() {
        let result = solve_case_with_sigma_and_scale(12, 4.0, 1.5, 0.0);
        assert!(result.converged);
        assert!(result.solution_l2 < 1.0e-14, "expected zero solution norm, got {}", result.solution_l2);
        assert!(result.solution_checksum.abs() < 1.0e-14,
            "expected zero checksum, got {}", result.solution_checksum);
        assert!(result.l2_error < 1.0e-14, "expected zero manufactured-solution error, got {}", result.l2_error);
    }

    /// Identical inputs must produce an identical checksum (determinism).
    #[test]
    fn anisotropic_maxwell_solution_is_deterministic() {
        let r1 = solve_case_with_sigma_and_scale(12, 4.0, 1.5, 1.0);
        let r2 = solve_case_with_sigma_and_scale(12, 4.0, 1.5, 1.0);
        assert_eq!(r1.solution_checksum, r2.solution_checksum,
            "anisotropic Maxwell checksum is not deterministic: {} vs {}",
            r1.solution_checksum, r2.solution_checksum);
    }
}
