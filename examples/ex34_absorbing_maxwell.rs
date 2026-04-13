//! # Example 34 — Maxwell with first-order absorbing boundary condition
//!
//! Solves the 2-D H(curl) problem
//!
//! ```text
//!   curl curl E + E = f          in Ω = [0,1]²
//!   curl E + γ_abs (n×E) = g     on ∂Ω
//! ```
//!
//! interpreted as a first-order absorbing boundary closure with normalised
//! admittance `γ_abs`.

use std::f64::consts::PI;

use fem_examples::maxwell::{
    StaticMaxwellBuilder,
    l2_error_hcurl_exact,
};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const ABSORBING_GAMMA: f64 = 1.0;

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
    if let Some(err) = result.l2_error {
        println!("  h = {:.4e},  L² error = {:.4e}", result.h, err);
    }
}

struct CaseResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: Option<f64>,
}

fn solve_case(args: &Args) -> CaseResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = HCurlSpace::new(mesh, 1);

    let bdr_attrs = [1, 2, 3, 4];
    let robin_bdr = [1, 1, 1, 1];

    let mut builder = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_source_fn(source_value);

    builder = if args.anisotropic {
        let gamma_x = args.gamma_x;
        let gamma_y = args.gamma_y;
        builder.with_anisotropic_diag(1.0, 1.0, 1.0).add_absorbing_from_marker(
            &bdr_attrs,
            &robin_bdr,
            1.0,
            move |x, normal| {
                let e = exact_field(x);
                let e_tan = e[0] * normal[1] - e[1] * normal[0];
                let gamma_norm = (gamma_x * normal[0] * normal[0] + gamma_y * normal[1] * normal[1]).sqrt();
                let gamma_eff = if gamma_norm.abs() > 1e-14 {
                    (gamma_x * normal[0].powi(2) + gamma_y * normal[1].powi(2)) / gamma_norm
                } else {
                    1.0
                };
                -curl_exact(x) + gamma_eff * e_tan
            },
        )
    } else {
        builder.with_isotropic_coeffs(1.0, 1.0).add_absorbing_from_marker(
            &bdr_attrs,
            &robin_bdr,
            ABSORBING_GAMMA,
            absorbing_data,
        )
    };

    let problem = builder.build();
    let n_dofs = problem.n_dofs();
    let solved = problem.solve();

    let l2_error = if args.anisotropic {
        None
    } else {
        Some(l2_error_hcurl_exact(&solved.space, &solved.solution, exact_field))
    };

    CaseResult {
        n_dofs,
        iterations: solved.solve_result.iterations,
        final_residual: solved.solve_result.final_residual,
        converged: solved.solve_result.converged,
        h: 1.0 / args.n as f64,
        l2_error,
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

fn absorbing_data(x: &[f64], normal: &[f64]) -> f64 {
    -curl_exact(x) + ABSORBING_GAMMA * tangential_trace(x, normal)
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
        let l2 = result.l2_error.expect("expected L2 error in isotropic mode");
        assert!(l2 < 1.5e-1, "L2 error = {}", l2);
    }

    #[test]
    fn absorbing_maxwell_anisotropic_mode_converges() {
        let result = solve_case(&Args {
            n: 8,
            anisotropic: true,
            gamma_x: 1.0,
            gamma_y: 1.5,
        });
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
        assert!(result.l2_error.is_none());
    }
}