//! # Example 16 — Nonlinear Heat Equation (Newton)  (analogous to MFEM ex16)
//!
//! Solves the nonlinear heat equation with conductivity κ(u) = 1 + u²:
//!
//! ```text
//!   −∇·(κ(u) ∇u) = f    in Ω
//!              u = 0    on ∂Ω
//! ```
//!
//! Uses Newton–Raphson iteration with Picard Jacobian:
//! ```text
//!   J(uₖ) Δu = −F(uₖ),    uₖ₊₁ = uₖ + Δu
//!   F(u) = ∫κ(u) ∇u·∇v dx − ∫f v dx
//!   J(u) ≈ ∫κ(u) ∇φⱼ·∇φᵢ dx   (Picard / frozen-κ Jacobian)
//! ```
//!
//! Matches MFEM ex16 in structure: mesh from CLI, κ = 1 + α·u,
//! Newton solve with zero RHS (steady-state). MMS-based verification
//! lives under #[cfg(test)].
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex16_nonlinear_heat
//! cargo run --example mfem_ex16_nonlinear_heat -- --mesh path/to/mesh.mesh
//! cargo run --example mfem_ex16_nonlinear_heat -- --n 16 --newton-tol 1e-10
//! ```

use fem_assembly::{Assembler, physics::nonlinear::{NonlinearDiffusionForm, NewtonSolver, NewtonConfig}};
use fem_mesh::Mesh;
use fem_io::mfem::read_mfem_file;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

#[allow(dead_code)]
struct SolveResult {
    n: usize,
    newton_tol: f64,
    n_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    rms_error: f64,
    solution_norm: f64,
    solution_checksum: f64,
}

#[derive(Clone, Copy)]
struct LineSearchOptions {
    enabled: bool,
    min_alpha: f64,
    shrink: f64,
    max_backtracks: usize,
    sufficient_decrease: f64,
}

#[cfg(test)]
fn default_line_search_options() -> LineSearchOptions {
    LineSearchOptions {
        enabled: true,
        min_alpha: 1e-6,
        shrink: 0.5,
        max_backtracks: 20,
        sufficient_decrease: 1e-4,
    }
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 16: Nonlinear heat equation (Newton) ===");
    if !args.mesh_file.is_empty() {
        println!("  Mesh file: {}", args.mesh_file);
    } else {
        println!("  Mesh: {}×{} subdivisions, P1 elements", args.n, args.n);
    }
    println!("  κ(u) = 1 + {:.3}·u,  Newton tol = {:.0e}", args.alpha, args.newton_tol);
    println!(
        "  line-search: enabled={}, min_alpha={}, shrink={}, max_backtracks={}, c1={}",
        args.ls_enabled,
        args.ls_min_alpha,
        args.ls_shrink,
        args.ls_max_backtracks,
        args.ls_c1,
    );

    // Use MFEM-style zero RHS (steady nonlinear heat, analogous to MFEM ex16's
    // ConductionOperator which evolves from an initial condition).
    let result = run_main(args);
    println!("  DOFs: {}", result.n_dofs);
    if result.converged {
        println!("\n  Newton converged: {} iters, ‖F‖ = {:.3e}", result.iterations, result.final_residual);
    } else {
        println!("\n  Newton did NOT converge: {} iters, ‖F‖ = {:.3e}", result.iterations, result.final_residual);
    }
    println!("  ||u_h||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("\nDone.");
}

fn run_main(args: Args) -> SolveResult {
    let mesh = if args.mesh_file.is_empty() {
        Mesh::<2>::unit_square_tri(args.n)
    } else {
        let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    };
    let n_dofs = {
        let space = H1Space::new(mesh.clone(), 1);
        space.n_dofs()
    };
    let ls = LineSearchOptions {
        enabled: args.ls_enabled,
        min_alpha: args.ls_min_alpha,
        shrink: args.ls_shrink,
        max_backtracks: args.ls_max_backtracks,
        sufficient_decrease: args.ls_c1,
    };
    solve_nonlinear_heat(mesh, |_x: &[f64]| 0.0, args.newton_tol, ls, args.alpha, n_dofs)
}

#[cfg(test)]
fn solve_case(n: usize, newton_tol: f64, exact_scale: f64) -> SolveResult {
    let mesh = Mesh::<2>::unit_square_tri(n);
    solve_case_with_ls(mesh, newton_tol, exact_scale, default_line_search_options())
}

/// Core solver: builds the nonlinear form, assembles RHS via a user-supplied
/// source function, runs Newton, and returns diagnostics.
fn solve_nonlinear_heat(
    mesh: Mesh<2>,
    source_fn: impl Fn(&[f64]) -> f64 + Send + Sync,
    newton_tol: f64,
    ls: LineSearchOptions,
    alpha: f64,
    n_dofs: usize,
) -> SolveResult {
    let space = H1Space::new(mesh.clone(), 1);

    // Dirichlet: u = 0 on all walls
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
    let dirichlet: Vec<(usize, f64)> = bnd.iter().map(|&d| (d as usize, 0.0)).collect();

    // Assemble RHS from the provided source
    let src = fem_assembly::standard::DomainSourceIntegrator::new(source_fn);
    let rhs = Assembler::assemble_linear(&space, &[&src], 5);

    // Build nonlinear form with κ(u) = 1 + alpha·u
    let mut form = NonlinearDiffusionForm::new(
        space,
        move |u: f64| 1.0 + alpha * u,
        3,
    );
    form.set_dirichlet(dirichlet);

    // Newton solve
    let cfg = NewtonConfig {
        atol:       newton_tol,
        rtol:       newton_tol * 1e2,
        max_iter:   50,
        linear_tol: newton_tol * 0.1,
        line_search: ls.enabled,
        line_search_min_alpha: ls.min_alpha,
        line_search_shrink: ls.shrink,
        line_search_max_backtracks: ls.max_backtracks,
        line_search_sufficient_decrease: ls.sufficient_decrease,
        verbose:    true,
    };
    let solver = NewtonSolver::new(cfg);
    let mut u = vec![0.0_f64; n_dofs];

    let (converged, iterations, final_residual) = match solver.solve(&form, &rhs, &mut u) {
        Ok(r) => (true, r.iterations, r.final_residual),
        Err(r) => (false, r.iterations, r.final_residual),
    };

    let solution_norm = u.iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = u
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        n: 0,
        newton_tol,
        n_dofs,
        iterations,
        final_residual,
        converged,
        rms_error: 0.0,
        solution_norm,
        solution_checksum,
    }
}

/// MMS-based solve (test only): manufactured solution u* = sin(πx)sin(πy)
/// with the corresponding RHS f = -∇·((1+α·u*)∇u*).
#[cfg(test)]
fn solve_case_with_ls(
    mesh: Mesh<2>,
    newton_tol: f64,
    exact_scale: f64,
    ls: LineSearchOptions,
) -> SolveResult {
    use std::f64::consts::PI;
    use fem_assembly::standard::DomainSourceIntegrator;

    let space = H1Space::new(mesh.clone(), 1);
    let n_dofs = space.n_dofs();

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
    let dirichlet: Vec<(usize, f64)> = bnd.iter().map(|&d| (d as usize, 0.0)).collect();

    // Keep a separate mesh for error computation (owned clone of space's mesh)
    let err_mesh = space.mesh().clone();

    // Manufactured RHS for u* = sin(πx)sin(πy), κ(u) = 1 + u²
    let src = DomainSourceIntegrator::new(move |x: &[f64]| {
        let (sx, sy) = ((PI * x[0]).sin(), (PI * x[1]).sin());
        let (cx, cy) = ((PI * x[0]).cos(), (PI * x[1]).cos());
        let u_star = exact_scale * sx * sy;
        let kappa = 1.0 + u_star * u_star;
        let lap_u = -2.0 * PI * PI * u_star;
        let grad_kappa_dot_grad_u = 2.0 * u_star * PI * PI *
            (cx * cx * sy * sy + sx * sx * cy * cy);
        -kappa * lap_u - grad_kappa_dot_grad_u
    });
    let rhs = Assembler::assemble_linear(&space, &[&src], 5);

    let mut form = NonlinearDiffusionForm::new(
        space,
        |u: f64| 1.0 + u * u,
        3,
    );
    form.set_dirichlet(dirichlet);

    let cfg = NewtonConfig {
        atol:       newton_tol,
        rtol:       newton_tol * 1e2,
        max_iter:   50,
        linear_tol: newton_tol * 0.1,
        line_search: ls.enabled,
        line_search_min_alpha: ls.min_alpha,
        line_search_shrink: ls.shrink,
        line_search_max_backtracks: ls.max_backtracks,
        line_search_sufficient_decrease: ls.sufficient_decrease,
        verbose:    false,
    };
    let solver = NewtonSolver::new(cfg);
    let mut u = vec![0.0_f64; n_dofs];

    let (converged, iterations, final_residual) = match solver.solve(&form, &rhs, &mut u) {
        Ok(r) => (true, r.iterations, r.final_residual),
        Err(r) => (false, r.iterations, r.final_residual),
    };

    let rms_error = {
        let err_space = H1Space::new(err_mesh, 1);
        let err_dm = err_space.dof_manager();
        let mut err = 0.0_f64;
        for i in 0..n_dofs {
            let x = err_dm.dof_coord(i as u32);
            let u_ex = exact_scale * (PI * x[0]).sin() * (PI * x[1]).sin();
            err += (u[i] - u_ex).powi(2);
        }
        (err / n_dofs as f64).sqrt()
    };
    let solution_norm = u.iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = u
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        n: 0,
        newton_tol,
        n_dofs,
        iterations,
        final_residual,
        converged,
        rms_error,
        solution_norm,
        solution_checksum,
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    n: usize,
    newton_tol: f64,
    alpha: f64,
    ls_enabled: bool,
    ls_min_alpha: f64,
    ls_shrink: f64,
    ls_max_backtracks: usize,
    ls_c1: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: String::new(),
        n: 16,
        newton_tol: 1e-10,
        alpha: 1.0,
        ls_enabled: true,
        ls_min_alpha: 1e-6,
        ls_shrink: 0.5,
        ls_max_backtracks: 20,
        ls_c1: 1e-4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--mesh" | "-m" => { a.mesh_file = it.next().unwrap_or_default(); }
            "--n"           => { a.n          = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--newton-tol"  => { a.newton_tol = it.next().unwrap_or("1e-10".into()).parse().unwrap_or(1e-10); }
            "--alpha" | "-a" => { a.alpha = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--no-line-search" => { a.ls_enabled = false; }
            "--line-search" => { a.ls_enabled = true; }
            "--ls-min-alpha" => { a.ls_min_alpha = it.next().unwrap_or("1e-6".into()).parse().unwrap_or(1e-6); }
            "--ls-shrink" => { a.ls_shrink = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5); }
            "--ls-max-backtracks" => { a.ls_max_backtracks = it.next().unwrap_or("20".into()).parse().unwrap_or(20); }
            "--ls-c1" => { a.ls_c1 = it.next().unwrap_or("1e-4".into()).parse().unwrap_or(1e-4); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex16_nonlinear_heat_coarse_case_converges_with_reasonable_error() {
        let result = solve_case(8, 1e-10, 1.0);
        assert!(result.converged);
        assert_eq!(result.n_dofs, 81);
        assert!(result.iterations <= 15, "Newton took too many iterations: {}", result.iterations);
        assert!(result.final_residual < 1.0e-8, "Newton residual too large: {}", result.final_residual);
        assert!(result.rms_error < 3.5e-3, "coarse-grid RMS error too large: {}", result.rms_error);
    }

    #[test]
    fn ex16_nonlinear_heat_refinement_improves_accuracy() {
        let coarse = solve_case(8, 1e-10, 1.0);
        let fine = solve_case(16, 1e-10, 1.0);
        assert!(fine.rms_error < coarse.rms_error,
            "refinement should reduce RMS error: coarse={} fine={}", coarse.rms_error, fine.rms_error);
        assert!(fine.rms_error < 1.0e-3, "fine-grid RMS error too large: {}", fine.rms_error);
    }

    #[test]
    fn ex16_nonlinear_heat_looser_newton_tolerance_preserves_solution_accuracy() {
        let tight = solve_case(16, 1e-10, 1.0);
        let loose = solve_case(16, 1e-8, 1.0);
        assert!(tight.converged && loose.converged);
        assert!(loose.iterations <= tight.iterations,
            "looser tolerance should not need more iterations: tight={} loose={}", tight.iterations, loose.iterations);
        assert!((loose.rms_error - tight.rms_error).abs() < 1.0e-6,
            "solution accuracy drifted under looser Newton tolerance: tight={} loose={}", tight.rms_error, loose.rms_error);
    }

    #[test]
    fn ex16_nonlinear_heat_sign_reversed_manufactured_solution_flips_state() {
        let positive = solve_case(16, 1e-10, 1.0);
        let negative = solve_case(16, 1e-10, -1.0);
        assert!(positive.converged && negative.converged);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-12);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "solution checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
        assert!((positive.rms_error - negative.rms_error).abs() < 1.0e-12);
    }

    #[test]
    fn ex16_nonlinear_heat_zero_manufactured_state_gives_trivial_solution() {
        let result = solve_case(16, 1e-10, 0.0);
        assert!(result.converged, "zero-source nonlinear heat solve should converge");
        assert!(result.final_residual < 1.0e-12, "zero-source residual too large: {}", result.final_residual);
        assert!(result.rms_error < 1.0e-14, "zero manufactured state should have zero RMS error: {}", result.rms_error);
        assert!(result.solution_norm < 1.0e-14, "zero manufactured state should give zero solution norm: {}", result.solution_norm);
        assert!(result.solution_checksum.abs() < 1.0e-14,
            "zero manufactured state should give zero checksum: {}",
            result.solution_checksum);
    }

    #[test]
    fn ex16_nonlinear_heat_dof_count_matches_p1_h1_formula() {
        for &n in &[8usize, 12usize, 16usize] {
            let result = solve_case(n, 1e-10, 1.0);
            assert_eq!(result.n_dofs, (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex16_nonlinear_heat_larger_manufactured_amplitude_increases_response() {
        let half = solve_case(16, 1e-10, 0.5);
        let full = solve_case(16, 1e-10, 1.0);
        assert!(half.converged && full.converged);
        assert!(full.solution_norm > half.solution_norm,
            "larger manufactured amplitude should increase solution norm: half={} full={}",
            half.solution_norm,
            full.solution_norm);
        assert!(full.iterations >= half.iterations,
            "stronger nonlinearity should not require fewer Newton iterations: half={} full={}",
            half.iterations,
            full.iterations);
        assert!(half.rms_error.is_finite() && full.rms_error.is_finite());
        assert!(half.rms_error > 0.0 && full.rms_error > 0.0);
    }

    #[test]
    fn ex16_nonlinear_heat_tighter_tolerance_reduces_final_residual() {
        let loose = solve_case(16, 1e-8, 1.0);
        let tight = solve_case(16, 1e-10, 1.0);
        assert!(loose.converged && tight.converged);
        assert!(tight.final_residual <= loose.final_residual * 1.1,
            "tighter tolerance should not end with larger residual: loose={} tight={}",
            loose.final_residual,
            tight.final_residual);
    }

    // ─── Regression baseline ─────────────────────────────────────────────

    #[test]
    fn ex16_regression_baseline() {
        let result = solve_case(8, 1e-8, 0.01);
        assert!(result.converged);

        fem_regression::regression("mfem_ex16_nonlinear_heat")
            .check_with("rms_error",          result.rms_error,        1e-6, 1e-10)
            .check_with("solution_norm",      result.solution_norm,    1e-6, 1e-10)
            .check_with("solution_checksum",  result.solution_checksum, 1e-6, 1e-10)
            .check_with("n_dofs",             result.n_dofs as f64,    0.0,  0.5)
            .check_with("iterations",         result.iterations as f64, 1e-4, 0.5)
            .check_with("residual",           result.final_residual,   1e-4, 1e-10)
            .finalize();
    }

    #[test]
    fn ex16_mfem_reference_test() {
        let r = solve_case(8, 1e-8, 0.01);
        assert!(r.converged);
        assert_eq!(r.n_dofs, 81, "H1 P1 on 8×8: 81 DOFs");
        assert!(r.solution_norm > 0.0);
        assert!(r.rms_error > 0.0 && r.rms_error < 1.0);
        use std::time::Instant;
        let t0 = Instant::now();
        let _ = solve_case(16, 1e-8, 0.01);
        let elapsed = t0.elapsed();
        assert!(elapsed.as_secs_f64() < 30.0, "16×16 heat took {:.2}s", elapsed.as_secs_f64());
        eprintln!("  [mfem-ref] ex16: dofs={} rms={:.6e} norm={:.6e} iter={}",
            r.n_dofs, r.rms_error, r.solution_norm, r.iterations);
    }
}

