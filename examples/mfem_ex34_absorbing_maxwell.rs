//! # Example 34 �?Maxwell with first-order absorbing boundary condition
//!
//! Solves the 2-D H(curl) problem
//!
//! ```text
//!   curl curl E + E = f          in Ω = [0,1]²
//!   curl E + γ_abs (n×E) = g     on ∂�?
//! ```
//!
//! interpreted as a first-order absorbing boundary closure with normalised
//! admittance `γ_abs`.

use std::f64::consts::PI;

use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const ABSORBING_GAMMA: f64 = 1.0;
const DEFAULT_SCALE: f64 = 1.0;

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 34: Maxwell with absorbing BC ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    println!("  Edge DOFs: {}", result.n_dofs);
    if args.anisotropic {
        println!(
            "  Mode: anisotropic absorbing (gamma_x={:.3}, gamma_y={:.3})",
            args.gamma_x, args.gamma_y
        );
    } else {
        println!("  Boundary tags: [1, 2, 3, 4], gamma_abs = {:.3}", ABSORBING_GAMMA);
    }
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  h = {:.4e},  L² error = {:.4e}", result.h, result.l2_error);
    println!("  ||u||₂ = {:.4e}", result.solution_l2);
    println!("  checksum = {:.8e}", result.solution_checksum);
}

struct CaseResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: f64,
    solution_l2: f64,
    solution_checksum: f64,
}

fn solve_case(args: &Args) -> CaseResult {
    solve_case_with_scale(args, DEFAULT_SCALE)
}

fn solve_case_with_scale(args: &Args, scale: f64) -> CaseResult {
    solve_case_with_scale_and_field(args, scale).0
}

fn solve_case_with_scale_and_field(args: &Args, scale: f64) -> (CaseResult, Vec<f64>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = HCurlSpace::new(mesh, 1);

    let bdr_attrs = [1, 2, 3, 4];
    let robin_bdr = [1, 1, 1, 1];

    let mut builder = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_source_fn(move |x| source_value(x, scale));

    builder = if args.anisotropic {
        let gamma_x = args.gamma_x;
        let gamma_y = args.gamma_y;
        builder.with_anisotropic_diag(1.0, 1.0, 1.0).add_absorbing_from_marker(
            &bdr_attrs,
            &robin_bdr,
            1.0,
            move |x, normal| {
                let e_tan = tangential_trace(x, normal, scale);
                let gamma_norm =
                    (gamma_x * normal[0] * normal[0] + gamma_y * normal[1] * normal[1]).sqrt();
                let gamma_eff = if gamma_norm.abs() > 1e-14 {
                    (gamma_x * normal[0].powi(2) + gamma_y * normal[1].powi(2)) / gamma_norm
                } else {
                    1.0
                };
                -curl_exact(x, scale) + gamma_eff * e_tan
            },
        )
    } else {
        builder.with_isotropic_coeffs(1.0, 1.0).add_absorbing_from_marker(
            &bdr_attrs,
            &robin_bdr,
            ABSORBING_GAMMA,
            move |x, normal| absorbing_data(x, normal, scale),
        )
    };

    let problem = builder.build();
    let n_dofs = problem.n_dofs();
    let solved = problem.solve();
    let solution_l2 = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();
    let solution_checksum = solved.solution
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let l2_error = l2_error_hcurl_exact(&solved.space, &solved.solution, |x| exact_field(x, scale));

    (
        CaseResult {
            n_dofs,
            iterations: solved.solve_result.iterations,
            final_residual: solved.solve_result.final_residual,
            converged: solved.solve_result.converged,
            h: 1.0 / args.n as f64,
            l2_error,
            solution_l2,
            solution_checksum,
        },
        solved.solution,
    )
}

fn source_value(x: &[f64], scale: f64) -> [f64; 2] {
    let coeff = 1.0 + PI * PI;
    [scale * (1.0 + coeff * (PI * x[1]).sin()), scale * (1.0 + coeff * (PI * x[0]).sin())]
}

fn exact_field(x: &[f64], scale: f64) -> [f64; 2] {
    [scale * (1.0 + (PI * x[1]).sin()), scale * (1.0 + (PI * x[0]).sin())]
}

fn curl_exact(x: &[f64], scale: f64) -> f64 {
    scale * (PI * (PI * x[0]).cos() - PI * (PI * x[1]).cos())
}

fn tangential_trace(x: &[f64], normal: &[f64], scale: f64) -> f64 {
    let e = exact_field(x, scale);
    e[0] * normal[1] - e[1] * normal[0]
}

fn absorbing_data(x: &[f64], normal: &[f64], scale: f64) -> f64 {
    -curl_exact(x, scale) + ABSORBING_GAMMA * tangential_trace(x, normal, scale)
}

struct Args {
    n: usize,
    anisotropic: bool,
    gamma_x: f64,
    gamma_y: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 16,
        anisotropic: false,
        gamma_x: 1.0,
        gamma_y: 1.5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16);
            }
            "--anisotropic" => {
                a.anisotropic = true;
            }
            "--gamma-x" => {
                a.gamma_x = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            "--gamma-y" => {
                a.gamma_y = it.next().unwrap_or("1.5".into()).parse().unwrap_or(1.5);
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorbing_maxwell_coarse_mesh_has_reasonable_error() {
        let result = solve_case(&Args {
            n: 8,
            anisotropic: false,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });
        assert!(result.converged);
        assert!(result.l2_error < 2.0e-1, "L2 error = {}", result.l2_error);
    }

    #[test]
    fn absorbing_maxwell_anisotropic_mode_has_reasonable_error() {
        let result = solve_case(&Args {
            n: 8,
            anisotropic: true,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
        assert!(result.l2_error < 2.5e-1, "L2 error = {}", result.l2_error);
    }

    #[test]
    fn absorbing_maxwell_anisotropic_mode_refines_monotonically() {
        let coarse = solve_case(&Args {
            n: 8,
            anisotropic: true,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });
        let medium = solve_case(&Args {
            n: 16,
            anisotropic: true,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });
        let fine = solve_case(&Args {
            n: 32,
            anisotropic: true,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });

        assert!(coarse.converged && medium.converged && fine.converged);
        assert!(
            medium.l2_error < coarse.l2_error,
            "expected refinement to reduce anisotropic absorbing error: coarse={} medium={}",
            coarse.l2_error,
            medium.l2_error
        );
        assert!(
            fine.l2_error < medium.l2_error,
            "expected refinement to reduce anisotropic absorbing error: medium={} fine={}",
            medium.l2_error,
            fine.l2_error
        );
        assert!(fine.l2_error < 1.5e-1, "fine-grid L2 error = {}", fine.l2_error);
    }

    #[test]
    fn absorbing_maxwell_isotropic_mode_refines_monotonically() {
        let coarse = solve_case(&Args {
            n: 8,
            anisotropic: false,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });
        let medium = solve_case(&Args {
            n: 16,
            anisotropic: false,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });

        assert!(coarse.converged && medium.converged);
        assert!(
            medium.l2_error < coarse.l2_error,
            "expected isotropic absorbing refinement to reduce error: coarse={} medium={}",
            coarse.l2_error,
            medium.l2_error
        );
	    let observed_order = (coarse.l2_error / medium.l2_error).ln() / (coarse.h / medium.h).ln();
	    assert!(observed_order > 0.85, "isotropic absorbing observed order too low: {}", observed_order);
    }

    #[test]
    fn absorbing_maxwell_anisotropic_swapped_gammas_preserve_error_by_symmetry() {
        let gx_small = solve_case(&Args {
            n: 8,
            anisotropic: true,
            gamma_x: 0.5,
            gamma_y: 3.0,
        });
        let gy_small = solve_case(&Args {
            n: 8,
            anisotropic: true,
            gamma_x: 3.0,
            gamma_y: 0.5,
        });

        assert!(gx_small.converged && gy_small.converged);
        let err_gap = (gx_small.l2_error - gy_small.l2_error).abs();
        let rel_gap = err_gap / gx_small.l2_error.max(gy_small.l2_error).max(1e-30);
        assert!(
            rel_gap < 1.0e-8,
            "swapping gamma_x/gamma_y should preserve error by symmetry: rel_gap={}",
            rel_gap
        );
    }

    #[test]
    fn absorbing_maxwell_equal_anisotropic_gammas_match_isotropic_mode() {
        let isotropic = solve_case(&Args {
            n: 10,
            anisotropic: false,
            gamma_x: 1.0,
            gamma_y: 1.0,
        });
        let equal_aniso = solve_case(&Args {
            n: 10,
            anisotropic: true,
            gamma_x: 1.0,
            gamma_y: 1.0,
        });

        assert!(isotropic.converged && equal_aniso.converged);

        let rel_gap = (isotropic.l2_error - equal_aniso.l2_error).abs()
            / isotropic.l2_error.max(equal_aniso.l2_error).max(1e-30);
        assert!(
            rel_gap < 1.0e-8,
            "equal anisotropic gammas should match isotropic absorbing mode: rel_gap={}",
            rel_gap
        );
    }

    #[test]
    fn absorbing_maxwell_solution_scales_linearly_with_source_amplitude() {
        let half = solve_case_with_scale(
            &Args {
                n: 8,
                anisotropic: false,
                gamma_x: 1.0,
                gamma_y: 1.5,
            },
            0.5,
        );
        let full = solve_case_with_scale(
            &Args {
                n: 8,
                anisotropic: false,
                gamma_x: 1.0,
                gamma_y: 1.5,
            },
            1.0,
        );

        assert!(half.converged && full.converged);

        let solution_ratio = full.solution_l2 / half.solution_l2.max(1.0e-30);
        let error_ratio = full.l2_error / half.l2_error.max(1.0e-30);
        let checksum_ratio = full.solution_checksum / half.solution_checksum.max(1.0e-30);

        assert!(
            (solution_ratio - 2.0).abs() < 1.0e-6,
            "expected absorbing solution norm to scale linearly, got ratio {}",
            solution_ratio
        );
        assert!(
            (error_ratio - 2.0).abs() < 1.0e-6,
            "expected absorbing discretization error to scale linearly, got ratio {}",
            error_ratio
        );
        assert!(
            (checksum_ratio - 2.0).abs() < 1.0e-6,
            "expected absorbing checksum to scale linearly, got ratio {}",
            checksum_ratio
        );
    }

    #[test]
    fn absorbing_maxwell_sign_reversed_drive_flips_solution() {
        let base_args = Args {
            n: 8,
            anisotropic: false,
            gamma_x: 1.0,
            gamma_y: 1.5,
        };
        let (positive, u_pos) = solve_case_with_scale_and_field(&base_args, 1.0);
        let (negative, u_neg) = solve_case_with_scale_and_field(&base_args, -1.0);

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
            "expected absorbing Maxwell solution vector to flip sign under sign-reversed drive, got max symmetry error {}",
            symmetry_err
        );
        assert!(
            norm_rel_gap < 1.0e-12,
            "expected absorbing Maxwell solution norm to remain invariant under sign reversal, got relative gap {}",
            norm_rel_gap
        );
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
    }

    #[test]
    fn absorbing_maxwell_zero_source_gives_trivial_solution() {
        let result = solve_case_with_scale(
            &Args {
                n: 8,
                anisotropic: false,
                gamma_x: 1.0,
                gamma_y: 1.5,
            },
            0.0,
        );

        assert!(result.converged);
        assert!(result.solution_l2 < 1.0e-14, "expected zero solution norm, got {}", result.solution_l2);
        assert!(result.solution_checksum.abs() < 1.0e-14,
            "expected zero checksum, got {}", result.solution_checksum);
        assert!(result.l2_error < 1.0e-14, "expected zero manufactured-solution error, got {}", result.l2_error);
    }
}
