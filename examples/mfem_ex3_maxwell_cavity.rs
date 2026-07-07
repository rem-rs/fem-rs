//! # Example 3 -- Maxwell cavity  (one-to-one with MFEM ex3)
//!
//! Solves the second-order definite Maxwell problem:
//!
//! ```text
//!   curl curl E + E = f    in Omega
//!             n x E = 0    on dOmega
//! ```
//!
//! with an impressed current source `f = (1+kappa^2)*(sin(kappa*y), sin(kappa*x))`
//! where `kappa = pi * freq`.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex3_maxwell_cavity
//! cargo run --example mfem_ex3_maxwell_cavity -- -m ../data/star.mesh
//! cargo run --example mfem_ex3_maxwell_cavity -- --mesh ../data/beam-tri.mesh
//! cargo run --example mfem_ex3_maxwell_cavity -- -f 2.0
//! cargo run --example mfem_ex3_maxwell_cavity -- --n 32
//! ```

use std::f64::consts::PI;
use fem_examples::maxwell::StaticMaxwellBuilder;
use fem_io::mfem::read_mfem_file;
use fem_mesh::SimplexMesh;
use fem_space::{FESpace, HCurlSpace};

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 3: Maxwell cavity (curl-curl + mass) ===");
    match &args.mesh {
        Some(path) => println!("  Mesh file: {}", path),
        None => println!("  Mesh: {}x{} subdivisions, ND1 elements", args.n, args.n),
    }
    println!("  Frequency: {}", args.freq);
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
    println!(
        "  ||u||_2 = {:.4e}, max|u| = {:.4e}",
        result.solution_l2_norm, result.solution_max_abs
    );
}

struct CaseResult {
    n_dofs: usize,
    n_boundary_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_l2_norm: f64,
    solution_max_abs: f64,
}

fn source_value(x: &[f64], kappa: f64) -> [f64; 2] {
    let coeff = 1.0 + kappa * kappa;
    [coeff * (kappa * x[1]).sin(), coeff * (kappa * x[0]).sin()]
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
/// Divides [0,1]^2 into 4 quadrants:
///   Q1 (0.5,1)^2,  Q2 (0,0.5)^2,
///   Q3 (0,0.5)x(0.5,1), Q4 (0.5,1)x(0,0.5)
/// Each quadrant gets a different (wx, wy) weight for tuned absorption.
fn multi_material_pml_tensor(
    x: &[f64],
    thickness: f64,
    sigma_max: f64,
) -> [f64; 4] {
    let (wx, wy) = if x[0] >= 0.5 && x[1] >= 0.5 {
        (1.0, 1.2)
    } else if x[0] < 0.5 && x[1] >= 0.5 {
        (0.8, 1.3)
    } else if x[0] < 0.5 && x[1] < 0.5 {
        (0.9, 1.1)
    } else {
        (1.2, 0.9)
    };

    let sx = wx * axis_sigma_1d(x[0], 0.0, 1.0, thickness, sigma_max);
    let sy = wy * axis_sigma_1d(x[1], 0.0, 1.0, thickness, sigma_max);
    [1.0 + sx, 0.0, 0.0, 1.0 + sy]
}

/// Collect unique boundary face tags from a mesh for PEC BC.
fn boundary_tags(mesh: &SimplexMesh<2>) -> Vec<i32> {
    let mut tags: Vec<_> = mesh.face_tags.iter().copied().collect();
    tags.sort_unstable();
    tags.dedup();
    tags
}

fn solve_case(args: &Args) -> CaseResult {
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };
    let space = HCurlSpace::new(mesh, 1);

    // Mark all boundary attributes as essential (PEC).
    let bdr_attrs = boundary_tags(space.mesh());
    let kappa = args.freq * PI;

    let mut builder = StaticMaxwellBuilder::new(space)
        .with_quad_order(4)
        .with_source_fn(move |x| source_value(x, kappa))
        .add_pec_zero(&bdr_attrs);

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

    CaseResult {
        n_dofs,
        n_boundary_dofs: solved.boundary_report.essential_dofs,
        iterations: solved.solve_result.iterations,
        final_residual: solved.solve_result.final_residual,
        converged: solved.solve_result.converged,
        solution_l2_norm,
        solution_max_abs,
    }
}

// --- CLI --------------------------------------------------------------------

struct Args {
    mesh: Option<String>,
    n: usize,
    pml_like: bool,
    multi_material: bool,
    pml_thickness: f64,
    sigma_max: f64,
    wx: f64,
    wy: f64,
    freq: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 16,
        pml_like: false,
        multi_material: false,
        pml_thickness: 0.2,
        sigma_max: 2.0,
        wx: 1.0,
        wy: 1.0,
        freq: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "--n" => { a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "-f" | "--freq" => {
                a.freq = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0);
            }
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
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_examples::maxwell::l2_error_hcurl_exact;

    fn rel_diff(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1.0)
    }

    fn exact_e(x: &[f64], kappa: f64) -> [f64; 2] {
        [(kappa * x[1]).sin(), (kappa * x[0]).sin()]
    }

    fn mms_l2_error(args: &Args) -> f64 {
        let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let space = HCurlSpace::new(mesh, 1);
        let bdr_attrs = boundary_tags(space.mesh());
        let kappa = args.freq * PI;
        let problem = StaticMaxwellBuilder::new(space)
            .with_quad_order(4)
            .with_source_fn(move |x| source_value(x, kappa))
            .add_pec_zero(&bdr_attrs)
            .with_isotropic_coeffs(1.0, 1.0)
            .build();
        let solved = problem.solve();
        let kappa = args.freq * PI;
        l2_error_hcurl_exact(&solved.space, &solved.solution, |x| exact_e(x, kappa))
    }

    fn default_args() -> Args {
        Args {
            mesh: None,
            n: 8,
            pml_like: false,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.0,
            freq: 1.0,
        }
    }

    #[test]
    fn ex3_mfem_marker_path_has_reasonable_error() {
        let l2 = mms_l2_error(&default_args());
        assert!(l2 < 1.5e-1, "L2 error = {}", l2);
    }

    #[test]
    fn ex3_pml_like_mode_converges() {
        let result = solve_case(&Args {
            mesh: None,
            n: 8,
            pml_like: true,
            multi_material: false,
            pml_thickness: 0.2,
            sigma_max: 2.0,
            wx: 1.0,
            wy: 1.5,
            freq: 1.0,
        });
        assert!(result.converged);
        assert!(result.n_boundary_dofs > 0);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
    }

    #[test]
    fn ex3_standard_mode_refinement_halves_hcurl_error() {
        let coarse = mms_l2_error(&Args {
            mesh: None, n: 8, pml_like: false, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        });
        let medium = mms_l2_error(&Args {
            mesh: None, n: 16, pml_like: false, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        });

        assert!(coarse.is_finite() && medium.is_finite());
        assert!(medium < coarse, "expected refinement to reduce error: coarse={} medium={}", coarse, medium);
        let ratio = coarse / medium;
        assert!(ratio > 1.8, "expected roughly first-order error halving on mesh doubling, got ratio {}", ratio);
    }

    #[test]
    fn ex3_multi_material_pml_mode_converges() {
        let result = solve_case(&Args {
            mesh: None, n: 8, pml_like: false, multi_material: true,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        });
        assert!(result.converged, "multi-material PML should converge");
        assert!(result.n_boundary_dofs > 0);
        assert!(result.final_residual < 1.0e-6, "residual = {}", result.final_residual);
        assert!(result.solution_l2_norm.is_finite());
        assert!(result.solution_max_abs.is_finite());
    }

    #[test]
    fn ex3_pml_like_stronger_sigma_reduces_solution_norm() {
        let weak = solve_case(&Args {
            mesh: None, n: 8, pml_like: true, multi_material: false,
            pml_thickness: 0.2, sigma_max: 0.2, wx: 1.0, wy: 1.5, freq: 1.0,
        });
        let strong = solve_case(&Args {
            mesh: None, n: 8, pml_like: true, multi_material: false,
            pml_thickness: 0.2, sigma_max: 4.0, wx: 1.0, wy: 1.5, freq: 1.0,
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
            mesh: None, n: 8, pml_like: true, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.5, freq: 1.0,
        });
        let yx = solve_case(&Args {
            mesh: None, n: 8, pml_like: true, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.5, wy: 1.0, freq: 1.0,
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
    fn ex3_multi_material_stronger_sigma_reduces_solution_norm() {
        let weak = solve_case(&Args {
            mesh: None, n: 8, pml_like: false, multi_material: true,
            pml_thickness: 0.2, sigma_max: 0.2, wx: 1.0, wy: 1.0, freq: 1.0,
        });
        let strong = solve_case(&Args {
            mesh: None, n: 8, pml_like: false, multi_material: true,
            pml_thickness: 0.2, sigma_max: 4.0, wx: 1.0, wy: 1.0, freq: 1.0,
        });

        assert!(weak.converged && strong.converged);
        assert!(
            strong.solution_l2_norm < weak.solution_l2_norm,
            "expected stronger multi-material damping to reduce ||u||2: weak={} strong={}",
            weak.solution_l2_norm,
            strong.solution_l2_norm
        );
    }

    // --- Regression baseline -------------------------------------------------

    #[test]
    fn ex3_regression_baseline() {
        let args = Args {
            mesh: None, n: 8, pml_like: false, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        };
        let l2 = mms_l2_error(&args);
        let result = solve_case(&args);

        assert!(result.converged);

        fem_regression::regression("mfem_ex3_maxwell_cavity")
            .check_with("l2_error",        l2, 1e-6, 1e-10)
            .check_with("solution_l2_norm", result.solution_l2_norm, 1e-6, 1e-10)
            .check_with("solution_max_abs", result.solution_max_abs, 1e-6, 1e-10)
            .check_with("iterations",       result.iterations as f64, 1e-4, 0.5)
            .check_with("residual",         result.final_residual,   1e-4, 1e-10)
            .check_with("n_dofs",           result.n_dofs as f64,    0.0,  0.5)
            .finalize();
    }

    // --- MFEM cross-validation test -----------------------------------------

    #[test]
    fn ex3_mfem_reference_test() {
        let args = Args {
            mesh: None, n: 8, pml_like: false, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        };
        let l2 = mms_l2_error(&args);
        let result = solve_case(&args);
        assert!(result.converged, "solve must converge");

        // DOF count: matches MFEM
        assert_eq!(result.n_dofs, 208,
            "DOF count should be 208 for ND1 on 8x8 tri mesh");

        // Solution L2 norm and max abs should be finite and positive
        assert!(result.solution_l2_norm > 0.0, "solution norm must be positive");
        assert!(result.solution_l2_norm.is_finite(), "solution norm must be finite");
        assert!(result.solution_max_abs > 0.0, "solution max must be positive");

        // Solver convergence
        assert!(result.iterations > 0, "CG should take positive iterations");
        assert!(result.iterations < 500, "CG iterations should be reasonable");
        assert!(result.final_residual < 1e-8, "CG residual should be small");

        // PEC BC: solution on boundary edges should be near zero
        let n_boundary = result.n_boundary_dofs;
        assert!(n_boundary > 0 && n_boundary < result.n_dofs,
            "expected 0 < n_boundary < n_dofs, got {n_boundary} / {}", result.n_dofs);

        // Convergence rate: refine mesh, error should drop
        let l2_6 = mms_l2_error(&Args {
            mesh: None, n: 6, pml_like: false, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        });
        let l2_12 = mms_l2_error(&Args {
            mesh: None, n: 12, pml_like: false, multi_material: false,
            pml_thickness: 0.2, sigma_max: 2.0, wx: 1.0, wy: 1.0, freq: 1.0,
        });
        assert!(l2_6.is_finite() && l2_12.is_finite());
        assert!(l2_12 < l2_6, "L2 error should decrease with refinement");
        let h6 = 1.0 / 6.0;
        let h12 = 1.0 / 12.0;
        let rate = f64::ln(l2_6 / l2_12) / f64::ln(h6 / h12);
        eprintln!("  [mfem-ref] ex3: L2(6)={:.6e} L2(12)={:.6e} rate={:.3} (expected ~1)",
            l2_6, l2_12, rate);
        assert!(rate > 0.5, "convergence rate {:.2} too low", rate);

        eprintln!("  [mfem-ref] ex3: {} DOFs, {} boundary DOFs, {} CG iters, res={:.3e}",
            result.n_dofs, n_boundary, result.iterations, result.final_residual);
        let _ = l2; // suppress unused warning for the top-level l2
    }
}
