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
//!   ∫ curl E·curl v + ∫ E·v + γ ∫ (n×E)·(n×v)
//!     = ∫ f·v + ∫ g·(n×v)
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
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex33_tangential_drive_maxwell
//! cargo run --example mfem_ex33_tangential_drive_maxwell -- -m ../data/star.mesh
//! cargo run --example mfem_ex33_tangential_drive_maxwell -- --n 32
//! ```

use std::f64::consts::PI;
use fem_examples::maxwell::StaticMaxwellBuilder;
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_space::HCurlSpace;

const DEFAULT_GAMMA: f64 = 2.0;
const DEFAULT_SCALE: f64 = 1.0;

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 33: Maxwell with tangential boundary load ===");
    match &args.mesh {
        Some(path) => println!("  Mesh file: {}", path),
        None => println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n),
    }
    println!("  Edge DOFs: {}", result.n_dofs);
    println!("  Boundary tags: [1, 2, 3, 4], gamma = {:.3}", DEFAULT_GAMMA);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  ||u||₂ = {:.4e}", result.solution_l2);
}

struct CaseResult {
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_l2: f64,
}

fn solve_case(args: &Args) -> CaseResult {
    solve_case_with_gamma_and_scale(args, DEFAULT_GAMMA, DEFAULT_SCALE)
}

fn solve_case_with_gamma_and_scale(args: &Args, gamma: f64, scale: f64) -> CaseResult {
    solve_case_with_gamma_and_scale_and_field(args, gamma, scale).0
}

fn solve_case_with_gamma_and_scale_and_field(args: &Args, gamma: f64, scale: f64) -> (CaseResult, Vec<f64>) {
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
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

// --- CLI ---------------------------------------------------------------------

struct Args {
    mesh: Option<String>,
    n: usize,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 16 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "--n" => {
                a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16);
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_examples::maxwell::l2_error_hcurl_exact;

    fn default_args() -> Args {
        Args { mesh: None, n: 8 }
    }

    fn mms_l2_error(args: &Args, gamma: f64, scale: f64) -> f64 {
        let mesh = Mesh::<2>::unit_square_tri(args.n);
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
        let solved = problem.solve();
        l2_error_hcurl_exact(&solved.space, &solved.solution, |x| exact_field(x, scale))
    }

    #[test]
    fn tangential_drive_maxwell_coarse_mesh_has_reasonable_error() {
        let l2 = mms_l2_error(&default_args(), DEFAULT_GAMMA, 1.0);
        assert!(l2 < 1.5e-1, "L2 error = {}", l2);
    }

    #[test]
    fn tangential_drive_maxwell_h_refinement_reduces_l2_error() {
        let coarse_l2 = mms_l2_error(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);
        let fine_l2 = mms_l2_error(&Args { mesh: None, n: 16 }, DEFAULT_GAMMA, 1.0);
        assert!(fine_l2 < coarse_l2,
            "finer mesh should have smaller L2 error: n=8: {:.3e}, n=16: {:.3e}",
            coarse_l2, fine_l2);
        let ratio = coarse_l2 / fine_l2;
        assert!(ratio > 1.5, "expected >1.5x error reduction on mesh doubling (ND1 O(h)), got {:.2}", ratio);
    }

    #[test]
    fn tangential_drive_maxwell_converged_residual_is_small() {
        let result = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 16 }, DEFAULT_GAMMA, 1.0);
        assert!(result.converged, "solver must converge at n=16");
        assert!(result.final_residual < 1e-8, "expected residual < 1e-8 after convergence, got {:.3e}", result.final_residual);
    }

    #[test]
    fn tangential_drive_maxwell_refines_monotonically_on_practical_meshes() {
        let coarse = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);
        let medium = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 16 }, DEFAULT_GAMMA, 1.0);

        assert!(coarse.converged && medium.converged);
        let coarse_l2 = mms_l2_error(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);
        let medium_l2 = mms_l2_error(&Args { mesh: None, n: 16 }, DEFAULT_GAMMA, 1.0);
        assert!(
            medium_l2 < coarse_l2,
            "expected refinement to reduce error: coarse={} medium={}",
            coarse_l2,
            medium_l2
        );
    }

    #[test]
    fn tangential_drive_maxwell_remains_accurate_for_gamma_variations() {
        let weak = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, 0.5, 1.0);
        let strong = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, 4.0, 1.0);

        assert!(weak.converged && strong.converged);
        assert!(weak.final_residual < 1.0e-6, "weak-gamma residual = {}", weak.final_residual);
        assert!(strong.final_residual < 1.0e-6, "strong-gamma residual = {}", strong.final_residual);

        let weak_l2 = mms_l2_error(&Args { mesh: None, n: 8 }, 0.5, 1.0);
        let strong_l2 = mms_l2_error(&Args { mesh: None, n: 8 }, 4.0, 1.0);
        assert!(weak_l2 < 1.5e-1, "weak-gamma L2 error = {}", weak_l2);
        assert!(strong_l2 < 1.5e-1, "strong-gamma L2 error = {}", strong_l2);
    }

    #[test]
    fn tangential_drive_maxwell_solution_scales_linearly_with_boundary_drive() {
        let half = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 0.5);
        let full = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);

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
        let (positive, u_pos) = solve_case_with_gamma_and_scale_and_field(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);
        let (negative, u_neg) = solve_case_with_gamma_and_scale_and_field(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, -1.0);

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

    #[test]
    fn tangential_drive_maxwell_zero_drive_gives_trivial_solution() {
        let result = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 0.0);
        assert!(result.converged);
        assert!(result.solution_l2 < 1.0e-14,
            "expected zero solution norm for zero drive, got {}", result.solution_l2);
    }

    #[test]
    fn ex33_mfem_reference_test() {
        let l2 = mms_l2_error(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);
        let r = solve_case_with_gamma_and_scale(&Args { mesh: None, n: 8 }, DEFAULT_GAMMA, 1.0);
        assert!(r.converged);
        assert_eq!(r.n_dofs, 208, "ND1 on 8×8: 208 DOFs");
        assert!(r.final_residual < 1e-8);
        eprintln!("  [mfem-ref] ex33: dofs={} L2={:.6e} iter={}",
            r.n_dofs, l2, r.iterations);
        fem_regression::regression("mfem_ex33_tangential_drive_maxwell")
            .check_with("l2_error", l2, 1e-6, 1e-8)
            .check_with("solution_l2", r.solution_l2, 1e-6, 1e-8)
            .check_with("final_residual", r.final_residual, 1e-4, 1e-10)
            .finalize();
    }
}
