//! # Example 31 — Anisotropic Maxwell problem  (one-to-one with MFEM ex31)
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
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex31_anisotropic_maxwell
//! cargo run --example mfem_ex31_anisotropic_maxwell -- -m ../data/star.mesh
//! cargo run --example mfem_ex31_anisotropic_maxwell -- --n 32
//! ```

use std::f64::consts::PI;
use fem_examples::maxwell::StaticMaxwellBuilder;
use fem_io::mfem::read_mfem_file;
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const DEFAULT_SIGMA_X: f64 = 4.0;
const DEFAULT_SIGMA_Y: f64 = 1.5;
const DEFAULT_SCALE: f64 = 1.0;

fn main() {
    let args = parse_args();
    let result = solve_case(&args);

    println!("=== fem-rs Example 31: Anisotropic Maxwell ===");
    match &args.mesh {
        Some(path) => println!("  Mesh file: {}", path),
        None => println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n),
    }
    println!("  DOFs: {}", result.n_dofs);
    println!("  Boundary DOFs constrained: {}", result.n_boundary_dofs);
    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
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
    solution_l2: f64,
    solution_checksum: f64,
}

fn solve_case(args: &Args) -> CaseResult {
    solve_case_with_sigma_and_scale(args, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, DEFAULT_SCALE)
}

fn solve_case_with_sigma_and_scale(args: &Args, sigma_x: f64, sigma_y: f64, scale: f64) -> CaseResult {
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(args.n)
    };
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
        solution_l2,
        solution_checksum,
    }
}

fn source_value(x: &[f64], sigma_x: f64, sigma_y: f64, scale: f64) -> [f64; 2] {
    let fx = scale * (PI * PI + sigma_x) * (PI * x[1]).sin();
    let fy = scale * (PI * PI + sigma_y) * (PI * x[0]).sin();
    [fx, fy]
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

    fn mms_l2_error(args: &Args, sigma_x: f64, sigma_y: f64, scale: f64) -> f64 {
        let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
        let space = HCurlSpace::new(mesh, 1);
        let bdr_attrs = [1, 2, 3, 4];
        let ess_bdr = [1, 1, 1, 1];
        let problem = StaticMaxwellBuilder::new(space)
            .with_quad_order(4)
            .with_anisotropic_diag(1.0, sigma_x, sigma_y)
            .with_source_fn(move |x| source_value(x, sigma_x, sigma_y, scale))
            .add_pec_zero_from_marker(&bdr_attrs, &ess_bdr)
            .build();
        let solved = problem.solve();
        l2_error_hcurl_exact(&solved.space, &solved.solution, |x| {
            [scale * (PI * x[1]).sin(), scale * (PI * x[0]).sin()]
        })
    }

    #[test]
    fn anisotropic_maxwell_coarse_mesh_has_reasonable_error() {
        let l2 = mms_l2_error(&default_args(), DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, 1.0);
        assert!(l2 < 2.5e-1, "L2 error = {}", l2);
    }

    #[test]
    fn anisotropic_maxwell_exhibits_first_order_hcurl_convergence_trend() {
        let coarse_l2 = mms_l2_error(&Args { mesh: None, n: 8 }, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, 1.0);
        let medium_l2 = mms_l2_error(&Args { mesh: None, n: 16 }, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, 1.0);
        let fine_l2 = mms_l2_error(&Args { mesh: None, n: 32 }, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, 1.0);

        let h_c: f64 = 1.0 / 8.0;
        let h_m: f64 = 1.0 / 16.0;
        let h_f: f64 = 1.0 / 32.0;

        let order_1 = (coarse_l2 / medium_l2).ln() / (h_c / h_m).ln();
        let order_2 = (medium_l2 / fine_l2).ln() / (h_m / h_f).ln();

        assert!(
            order_1 > 0.85,
            "coarse->medium observed order too low: order={} (errors {} -> {})",
            order_1,
            coarse_l2,
            medium_l2
        );
        assert!(
            order_2 > 0.85,
            "medium->fine observed order too low: order={} (errors {} -> {})",
            order_2,
            medium_l2,
            fine_l2
        );
    }

    #[test]
    fn anisotropic_maxwell_swapping_principal_axes_preserves_error_by_symmetry() {
        let xy = mms_l2_error(&Args { mesh: None, n: 12 }, 4.0, 1.5, 1.0);
        let yx = mms_l2_error(&Args { mesh: None, n: 12 }, 1.5, 4.0, 1.0);

        let rel_gap = (xy - yx).abs() / xy.max(yx).max(1e-30);
        assert!(
            rel_gap < 1.0e-8,
            "swapping anisotropic principal values should preserve error by symmetry: rel_gap={}",
            rel_gap
        );
    }

    #[test]
    fn anisotropic_maxwell_uniform_sigma_rescaling_preserves_solution_response() {
        let base = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 1.0);
        let scaled = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 8.0, 3.0, 1.0);

        assert!(base.converged && scaled.converged);

        let sol_rel_gap = (base.solution_l2 - scaled.solution_l2).abs()
            / base.solution_l2.max(scaled.solution_l2).max(1e-30);

        assert!(
            sol_rel_gap < 1.0e-3,
            "uniform sigma rescaling should preserve solution norm: rel_gap={}",
            sol_rel_gap
        );
    }

    #[test]
    fn anisotropic_maxwell_solution_scales_linearly_with_source_amplitude() {
        let half = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 0.5);
        let full = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 1.0);

        assert!(half.converged && full.converged);
        let ratio = full.solution_l2 / half.solution_l2.max(1e-30);

        assert!(
            (ratio - 2.0).abs() < 1.0e-6,
            "expected anisotropic Maxwell solution norm to scale linearly, got ratio {}",
            ratio
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
        let positive = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 1.0);
        let negative = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, -1.0);

        assert!(positive.converged && negative.converged);
        assert!((positive.solution_l2 - negative.solution_l2).abs() < 1.0e-12,
            "solution norm should be sign-invariant: positive={} negative={}",
            positive.solution_l2,
            negative.solution_l2);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
    }

    #[test]
    fn anisotropic_maxwell_zero_source_gives_trivial_solution() {
        let result = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 0.0);
        assert!(result.converged);
        assert!(result.solution_l2 < 1.0e-14, "expected zero solution norm, got {}", result.solution_l2);
        assert!(result.solution_checksum.abs() < 1.0e-14,
            "expected zero checksum, got {}", result.solution_checksum);
    }

    #[test]
    fn anisotropic_maxwell_solution_is_deterministic() {
        let r1 = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 1.0);
        let r2 = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 12 }, 4.0, 1.5, 1.0);
        assert_eq!(r1.solution_checksum, r2.solution_checksum,
            "anisotropic Maxwell checksum is not deterministic: {} vs {}",
            r1.solution_checksum, r2.solution_checksum);
    }

    #[test]
    fn ex31_mfem_reference_test() {
        let l2 = mms_l2_error(&Args { mesh: None, n: 8 }, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, 1.0);
        let r = solve_case_with_sigma_and_scale(&Args { mesh: None, n: 8 }, DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, 1.0);
        assert!(r.converged);
        assert_eq!(r.n_dofs, 208, "ND1 on 8×8: 208 DOFs");
        assert!(r.solution_l2 > 0.0);
        assert!(r.iterations < 500);
        assert!(r.final_residual < 1e-8);

        // MFEM C++ cross-validation (verified 2026-07-05 via MSYS2 mingw64 g++ 15.2.0, MFEM 4.8-1):
        //   anisotropic curl-curl + diag(4,1.5) mass, ND1, 8×8 tri, PEC BC
        //   Manufactured: E = (sin(πy), sin(πx))
        //   Solver: PCG(GSSmoother) 5000 iters, rtol=1e-12
        //   Source: tests/mfem_references/cpp_refs/ex31_ref.cpp
        const MFEM_L2_ERROR: f64 = 0.1127764445187523_f64;
        let rel_diff = (l2 - MFEM_L2_ERROR).abs() / MFEM_L2_ERROR.max(1e-30);
        assert!(
            rel_diff < 0.02,
            "L2 error relative diff vs MFEM C++: {:.2e} (fem-rs={}, mfem={})",
            rel_diff, l2, MFEM_L2_ERROR
        );
        eprintln!(
            "  [mfem-cxx] ex31: L2 diff vs MFEM={:.2e}% (fem-rs={:.6e}, mfem={:.6e})",
            rel_diff * 100.0, l2, MFEM_L2_ERROR
        );

        fem_regression::regression("mfem_ex31_anisotropic_maxwell")
            .check_with("l2_error", l2, 1e-6, 1e-8)
            .check_with("solution_l2", r.solution_l2, 1e-6, 1e-8)
            .check_with("final_residual", r.final_residual, 1e-4, 1e-10)
            .finalize();
    }
}
