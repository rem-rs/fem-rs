//! # Example 3 — Maxwell cavity  (analogous to MFEM ex3)
//!
//! Solves the vector curl-curl + mass problem on the unit square:
//!
//! ```text
//!   ∇×(∇×E) + E = f    in Ω = [0,1]²
//!          n×E = 0    on ∂Ω
//! ```
//!
//! with the manufactured solution `E = (sin(πy), sin(πx))`.
//!
//! ```text
//!   curl E = π cos(πx) − π cos(πy)  (scalar in 2-D)
//!   ∇×(curl E) = (π² sin(πy), π² sin(πx))
//!   f = ∇×∇×E + E = ((1+π²) sin(πy), (1+π²) sin(πx))
//! ```
//!
//! ## Usage
//! ```
//! cargo run --example ex3_maxwell
//! cargo run --example ex3_maxwell -- --n 8
//! cargo run --example ex3_maxwell -- --n 16
//! cargo run --example ex3_maxwell -- --n 32
//! ```

use std::f64::consts::PI;
use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 3: Maxwell cavity (curl-curl + mass) ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    if args.pml_like {
        println!(
            "  Mode: PML-like anisotropic damping (thickness={}, sigma_max={}, wx={}, wy={})",
            args.pml_thickness, args.sigma_max, args.wx, args.wy
        );
    }

    println!("  Edge DOFs: {}", result.n_dofs);
    println!("  Boundary DOFs constrained: {}", result.n_boundary_dofs);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    if let Some(err) = result.l2_error {
        println!("  h = {:.4e},  L² error = {:.4e}", result.h, err);
        println!("  (Expected O(h) for ND1 elements)");
    } else {
        println!("  h = {:.4e},  L² error = n/a (PML-like modified operator)", result.h);
    }
}

struct CaseResult {
    n_dofs: usize,
    n_boundary_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    h: f64,
    l2_error: Option<f64>,
}

fn source_value(x: &[f64]) -> [f64; 2] {
    let coeff = 1.0 + PI * PI;
    [coeff * (PI * x[1]).sin(), coeff * (PI * x[0]).sin()]
}

fn axis_sigma_1d(coord: f64, lo: f64, hi: f64, thickness: f64, sigma_max: f64) -> f64 {
    let t = thickness.max(1e-14);
    let s = if coord < lo + t {
        ((lo + t - coord) / t).clamp(0.0, 1.0)
    } else if coord > hi - t {
        ((coord - (hi - t)) / t).clamp(0.0, 1.0)
    } else {
        0.0
    };
    sigma_max * s * s
}

fn solve_case(args: &Args) -> CaseResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = HCurlSpace::new(mesh, 1);

    // MFEM-style boundary marker (`ess_bdr`): all boundary attributes are essential.
    let attrs = [1, 2, 3, 4];
    let ess_bdr = [1, 1, 1, 1];

    let mut builder = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_source_fn(source_value)
        .add_pec_zero_from_marker(&attrs, &ess_bdr);

    builder = if args.pml_like {
        let thickness = args.pml_thickness;
        let sigma_max = args.sigma_max;
        let wx = args.wx;
        let wy = args.wy;
        builder.with_anisotropic_matrix_fn(1.0, move |x| {
            let sx = wx * axis_sigma_1d(x[0], 0.0, 1.0, thickness, sigma_max);
            let sy = wy * axis_sigma_1d(x[1], 0.0, 1.0, thickness, sigma_max);
            [1.0 + sx, 0.0, 0.0, 1.0 + sy]
        })
    } else {
        builder.with_isotropic_coeffs(1.0, 1.0)
    };

    let problem = builder.build();

    let n_dofs = problem.n_dofs();
    let solved = problem.solve();
    let l2_error = if args.pml_like {
        None
    } else {
        Some(l2_error_hcurl_exact(&solved.space, &solved.solution, |x| {
            [(PI * x[1]).sin(), (PI * x[0]).sin()]
        }))
    };

    CaseResult {
        n_dofs,
        n_boundary_dofs: solved.boundary_report.essential_dofs,
        iterations: solved.solve_result.iterations,
        final_residual: solved.solve_result.final_residual,
        converged: solved.solve_result.converged,
        h: 1.0 / args.n as f64,
        l2_error,
    }
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    n: usize,
    pml_like: bool,
    pml_thickness: f64,
    sigma_max: f64,
    wx: f64,
    wy: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 16,
        pml_like: false,
        pml_thickness: 0.2,
        sigma_max: 2.0,
        wx: 1.0,
        wy: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--pml-like" => { a.pml_like = true; }
            "--pml-thickness" => {
                a.pml_thickness = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2);
            }
            "--sigma-max" => {
                a.sigma_max = it.next().unwrap_or("2.0".into()).parse().unwrap_or(2.0);
            }
            "--wx" => { a.wx = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--wy" => { a.wy = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex3_mfem_marker_path_has_reasonable_error() {
        let result = solve_case(&Args {
            n: 8,
            pml_like: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
        });
        assert!(result.converged);
        let l2 = result.l2_error.expect("expected manufactured L2 error in non-PML mode");
        assert!(l2 < 1.5e-1, "L2 error = {}", l2);
    }

    #[test]
    fn ex3_pml_like_mode_converges() {
        let result = solve_case(&Args {
            n: 8,
            pml_like: true,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.5,
        });
        assert!(result.converged);
        assert!(result.n_boundary_dofs > 0);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
        assert!(result.l2_error.is_none());
    }
}
