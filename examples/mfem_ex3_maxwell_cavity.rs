//! # Example 3 �?Maxwell cavity  (analogous to MFEM ex3)
//!
//! Solves the vector curl-curl + mass problem on the unit square:
//!
//! ```text
//!   ∇�?∇×E) + E = f    in Ω = [0,1]²
//!          n×E = 0    on ∂�?
//! ```
//!
//! with the manufactured solution `E = (sin(πy), sin(πx))`.
//!
//! ```text
//!   curl E = π cos(πx) �?π cos(πy)  (scalar in 2-D)
//!   ∇�?curl E) = (π² sin(πy), π² sin(πx))
//!   f = ∇×∇×E + E = ((1+π²) sin(πy), (1+π²) sin(πx))
//! ```
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex3
//! cargo run --example mfem_ex3 -- --n 8
//! cargo run --example mfem_ex3 -- --n 16
//! cargo run --example mfem_ex3 -- --n 32
//! ```

use std::f64::consts::PI;
use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::{
    HCurlSpace,
};

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 3: Maxwell cavity (curl-curl + mass) ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    println!("  Source scale: {}", args.source_scale);
    if args.pml_like {
        println!(
            "  Mode: PML-like anisotropic damping (thickness={}, sigma_max={}, wx={}, wy={})",
            args.pml_thickness, args.sigma_max, args.wx, args.wy
        );
    }
    if args.multi_material {
        println!(
            "  Mode: Multi-material PML (4 regions with distinct coefficients)"
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
        println!("  h = {:.4e},  L² error = n/a (PML-like/multi-material modified operator)", result.h);
        println!(
            "  ||u||�?= {:.4e}, max|u| = {:.4e}",
            result.solution_l2_norm, result.solution_max_abs
        );
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
    solution_l2_norm: f64,
    solution_max_abs: f64,
}

fn source_value(x: &[f64], source_scale: f64) -> [f64; 2] {
    let coeff = 1.0 + PI * PI;
    [
        source_scale * coeff * (PI * x[1]).sin(),
        source_scale * coeff * (PI * x[0]).sin(),
    ]
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

/// Compute anisotropic tensor [sx, 0; 0, sy] with region-dependent coefficients.
/// Divides [0,1]² into 4 quadrants: Q1 (0.5,1)², Q2 (0,0.5)², Q3 (0,0.5)×(0.5,1), Q4 (0.5,1)×(0,0.5)
/// Each quadrant gets a different (wx, wy) weight for tuned absorption.
fn multi_material_pml_tensor(
    x: &[f64],
    thickness: f64,
    sigma_max: f64,
) -> [f64; 4] {
    let (wx, wy) = if x[0] >= 0.5 && x[1] >= 0.5 {
        // Q1: high damping (1.0, 1.2)
        (1.0, 1.2)
    } else if x[0] < 0.5 && x[1] >= 0.5 {
        // Q3: moderate x, high y (0.8, 1.3)
        (0.8, 1.3)
    } else if x[0] < 0.5 && x[1] < 0.5 {
        // Q2: moderate damping (0.9, 1.1)
        (0.9, 1.1)
    } else {
        // Q4: high x, moderate y (1.2, 0.9)
        (1.2, 0.9)
    };

    let sx = wx * axis_sigma_1d(x[0], 0.0, 1.0, thickness, sigma_max);
    let sy = wy * axis_sigma_1d(x[1], 0.0, 1.0, thickness, sigma_max);
    [1.0 + sx, 0.0, 0.0, 1.0 + sy]
}

fn solve_case(args: &Args) -> CaseResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = HCurlSpace::new(mesh, 1);

    // MFEM-style boundary marker (`ess_bdr`): all boundary attributes are essential.
    let attrs = [1, 2, 3, 4];
    let ess_bdr = [1, 1, 1, 1];
    let source_scale = args.source_scale;

    let mut builder = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_source_fn(move |x| source_value(x, source_scale))
        .add_pec_zero_from_marker(&attrs, &ess_bdr);

    builder = if args.multi_material {
        let thickness = args.pml_thickness;
        let sigma_max = args.sigma_max;
        builder.with_anisotropic_matrix_fn(1.0, move |x| {
            multi_material_pml_tensor(x, thickness, sigma_max)
        })
    } else if args.pml_like {
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

    let solution_l2_norm = solved.solution.iter().map(|v| v * v).sum::<f64>().sqrt();
    let solution_max_abs = solved
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let l2_error = if args.pml_like || args.multi_material {
        None
    } else {
        Some(l2_error_hcurl_exact(&solved.space, &solved.solution, |x| {
            [
                args.source_scale * (PI * x[1]).sin(),
                args.source_scale * (PI * x[0]).sin(),
            ]
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
        solution_l2_norm,
        solution_max_abs,
    }
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    n: usize,
    pml_like: bool,
    multi_material: bool,
    pml_thickness: f64,
    sigma_max: f64,
    wx: f64,
    wy: f64,
    source_scale: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 16,
        pml_like: false,
        multi_material: false,
        pml_thickness: 0.2,
        sigma_max: 2.0,
        wx: 1.0,
        wy: 1.0,
        source_scale: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--pml-like" => { a.pml_like = true; }
            "--multi-material" => { a.multi_material = true; a.pml_like = false; }
            "--pml-thickness" => {
                a.pml_thickness = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2);
            }
            "--sigma-max" => {
                a.sigma_max = it.next().unwrap_or("2.0".into()).parse().unwrap_or(2.0);
            }
            "--wx" => { a.wx = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--wy" => { a.wy = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--source-scale" => {
                a.source_scale = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_diff(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn ex3_mfem_marker_path_has_reasonable_error() {
        let result = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
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
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.5,
            source_scale: 1.0,
        });
        assert!(result.converged);
        assert!(result.n_boundary_dofs > 0);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
        assert!(result.l2_error.is_none());
    }

    #[test]
    fn ex3_standard_mode_refinement_halves_hcurl_error() {
        let coarse = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
        });
        let medium = solve_case(&Args {
            n: 16,
            pml_like: false,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
        });

        let coarse_err = coarse.l2_error.expect("expected manufactured L2 error on coarse mesh");
        let medium_err = medium.l2_error.expect("expected manufactured L2 error on refined mesh");

        assert!(coarse.converged && medium.converged);
        assert!(medium_err < coarse_err, "expected refinement to reduce error: coarse={} medium={}", coarse_err, medium_err);
        let ratio = coarse_err / medium_err;
        assert!(ratio > 1.8, "expected roughly first-order error halving on mesh doubling, got ratio {}", ratio);
    }

    #[test]
    fn ex3_multi_material_pml_mode_converges() {
        let result = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: true,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
        });
        assert!(result.converged, "multi-material PML should converge");
        assert!(result.n_boundary_dofs > 0);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
        assert!(result.l2_error.is_none(), "multi-material mode should not compute L2 error");
        assert!(result.solution_l2_norm.is_finite());
        assert!(result.solution_max_abs.is_finite());
    }

    #[test]
    fn ex3_pml_like_stronger_sigma_reduces_solution_norm() {
        let weak = solve_case(&Args {
            n: 8,
            pml_like: true,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 0.2,
            wx: 1.0,
            wy: 1.5,
            source_scale: 1.0,
        });
        let strong = solve_case(&Args {
            n: 8,
            pml_like: true,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 4.0,
            wx: 1.0,
            wy: 1.5,
            source_scale: 1.0,
        });

        assert!(weak.converged && strong.converged);
        assert!(
            strong.solution_l2_norm < weak.solution_l2_norm,
            "expected stronger PML damping to reduce ||u||2: weak={} strong={}",
            weak.solution_l2_norm,
            strong.solution_l2_norm
        );
    }

    #[test]
    fn ex3_pml_like_swapping_axis_weights_preserves_response_by_symmetry() {
        let xy = solve_case(&Args {
            n: 8,
            pml_like: true,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.5,
            source_scale: 1.0,
        });
        let yx = solve_case(&Args {
            n: 8,
            pml_like: true,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.5,
            wy: 1.0,
            source_scale: 1.0,
        });

        assert!(xy.converged && yx.converged);
        assert!(rel_diff(xy.solution_l2_norm, yx.solution_l2_norm) < 1.0e-10,
            "expected symmetry under x/y weight swap in ||u||2: xy={} yx={}",
            xy.solution_l2_norm, yx.solution_l2_norm);
        assert!(rel_diff(xy.solution_max_abs, yx.solution_max_abs) < 1.0e-10,
            "expected symmetry under x/y weight swap in max|u|: xy={} yx={}",
            xy.solution_max_abs, yx.solution_max_abs);
    }

    #[test]
    fn ex3_standard_mode_solution_scales_linearly_with_source() {
        let half = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 0.5,
        });
        let full = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
        });

        assert!(half.converged && full.converged);
        let ratio = full.solution_l2_norm / half.solution_l2_norm.max(1.0e-30);
        assert!(
            (ratio - 2.0).abs() < 1.0e-6,
            "expected linear response to source scaling, got ratio {}",
            ratio
        );
    }

    #[test]
    fn ex3_multi_material_stronger_sigma_reduces_solution_norm() {
        let weak = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: true,
            pml_thickness: 0.2,
            sigma_max: 0.2,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
        });
        let strong = solve_case(&Args {
            n: 8,
            pml_like: false,
            multi_material: true,
            pml_thickness: 0.2,
            sigma_max: 4.0,
            wx: 1.0,
            wy: 1.0,
            source_scale: 1.0,
        });

        assert!(weak.converged && strong.converged);
        assert!(
            strong.solution_l2_norm < weak.solution_l2_norm,
            "expected stronger multi-material damping to reduce ||u||2: weak={} strong={}",
            weak.solution_l2_norm,
            strong.solution_l2_norm
        );
    }
}

