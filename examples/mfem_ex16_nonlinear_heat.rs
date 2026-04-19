//! # Example 16 �?Nonlinear Heat Equation (Newton)  (analogous to MFEM ex16)
//!
//! Solves the nonlinear heat equation with conductivity κ(u) = 1 + u²:
//!
//! ```text
//!   −∇·(κ(u) ∇u) = f    in Ω = [0,1]²
//!              u = 0    on ∂�?//! ```
//!
//! Uses Newton–Raphson iteration with Picard Jacobian:
//! ```text
//!   J(u�? Δu = −F(u�?,    uₙ₊�?= u�?+ Δu
//!   F(u) = �?κ(u) ∇u·∇v dx �?�?f v dx
//!   J(u) �?�?κ(u) ∇φⱼ·∇φᵢ dx   (Picard / frozen-κ Jacobian)
//! ```
//!
//! Manufactured solution approach: choose `u* = sin(πx)sin(πy)` and compute
//! `f = −∇·((1+u*²)∇u*)` analytically:
//!
//! For u* = sin(πx)sin(πy), κ(u*) = 1 + sin²(πx)sin²(πy):
//! ```text
//!   f = π²(2 + sin²(πx)sin²(πy)) sin(πx)sin(πy)
//!       �?2π² sin³(πx)sin(πy)cos²(πx) �?2π² sin(πx)sin³(πy)cos²(πy)
//! ```
//! (simplified below)
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex16_nonlinear_heat
//! cargo run --example mfem_ex16_nonlinear_heat -- --n 16 --newton-tol 1e-10
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, nonlinear::{NonlinearDiffusionForm, NewtonSolver, NewtonConfig}};
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

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
    println!("  Mesh: {}×{} subdivisions, P1 elements", args.n, args.n);
    println!("  κ(u) = 1 + u²,  Newton tol = {:.0e}", args.newton_tol);
    println!(
        "  line-search: enabled={}, min_alpha={}, shrink={}, max_backtracks={}, c1={}",
        args.ls_enabled,
        args.ls_min_alpha,
        args.ls_shrink,
        args.ls_max_backtracks,
        args.ls_c1,
    );

    let result = solve_case_with_ls(
        args.n,
        args.newton_tol,
        1.0,
        LineSearchOptions {
            enabled: args.ls_enabled,
            min_alpha: args.ls_min_alpha,
            shrink: args.ls_shrink,
            max_backtracks: args.ls_max_backtracks,
            sufficient_decrease: args.ls_c1,
        },
    );

    println!("  Confirmed Newton tol = {:.0e}", result.newton_tol);
    println!("  DOFs: {}", result.n_dofs);
    if result.converged {
        println!("\n  Newton converged: {} iters, ‖F‖ = {:.3e}", result.iterations, result.final_residual);
    } else {
        println!("\n  Newton did NOT converge: {} iters, ‖F‖ = {:.3e}", result.iterations, result.final_residual);
    }
    let h = 1.0 / result.n as f64;
    println!("  h = {h:.4e},  nodal RMS error = {:.4e}", result.rms_error);
    println!("  ||u_h||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);
    println!("  (Expected O(h²) for P1 manufactured solution)");
    println!("\nDone.");
}

#[cfg(test)]
fn solve_case(n: usize, newton_tol: f64, exact_scale: f64) -> SolveResult {
    solve_case_with_ls(n, newton_tol, exact_scale, default_line_search_options())
}

fn solve_case_with_ls(
    n: usize,
    newton_tol: f64,
    exact_scale: f64,
    ls: LineSearchOptions,
) -> SolveResult {
    // ─── 1. Mesh and H¹ space ─────────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    // ─── 2. Identify Dirichlet DOFs ───────────────────────────────────────────
    let dm   = space.dof_manager();
    let bnd  = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    // Dirichlet: u = 0 on all walls
    let dirichlet: Vec<(usize, f64)> = bnd.iter().map(|&d| (d as usize, 0.0)).collect();

    // ─── 3. Assemble RHS f = manufactured source ─────────────────────────────
    // f = (2 + 3sin²(πx)sin²(πy)) * π² * sin(πx)sin(πy)
    //   This comes from: -div((1+u²)∇u) where u = sin(πx)sin(πy):
    //   ∂u/∂x = π cos(πx) sin(πy),  ∂²u/∂x² = -π² sin(πx)sin(πy)
    //   κ(u) = 1 + sin²(πx)sin²(πy)
    //   -div(κ∇u) = -κ Δu - ∇κ·∇u = κ·2π²·u - ∇κ·∇u
    //   ∇�?= (2u ∂u/∂x, 2u ∂u/∂y)
    //   ∇κ·∇u = 2u(|∂u/∂x|² + |∂u/∂y|²) = 2u · π²(cos²(πx)sin²(πy) + sin²(πx)cos²(πy))
    //          = 2u · π²(cos²(πx)sin²(πy) + sin²(πx)cos²(πy))
    //   Combined: f = (1+u²)·2π²·u - 2u·π²(...)
    use fem_assembly::standard::DomainSourceIntegrator;
    let src = DomainSourceIntegrator::new(|x: &[f64]| {
        let (sx, sy) = ((PI * x[0]).sin(), (PI * x[1]).sin());
        let (cx, cy) = ((PI * x[0]).cos(), (PI * x[1]).cos());
        let u_star = exact_scale * sx * sy;
        let kappa  = 1.0 + u_star * u_star;
        let lap_u  = -2.0 * PI * PI * u_star;
        let grad_kappa_dot_grad_u = 2.0 * u_star * PI * PI *
            (cx * cx * sy * sy + sx * sx * cy * cy);
        -kappa * lap_u - grad_kappa_dot_grad_u
    });
    let mesh2 = SimplexMesh::<2>::unit_square_tri(n);
    let space2 = H1Space::new(mesh2, 1);
    let rhs = Assembler::assemble_linear(&space2, &[&src], 5);

    // ─── 4. Build nonlinear form ──────────────────────────────────────────────
    let mut form = NonlinearDiffusionForm::new(
        space,
        |u: f64| 1.0 + u * u,   // κ(u) = 1 + u²
        3,
    );
    form.set_dirichlet(dirichlet);

    // ─── 5. Newton solve ──────────────────────────────────────────────────────
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

    // ─── 6. L² error ─────────────────────────────────────────────────────────
    let dm2 = space2.dof_manager();
    let rms_error = {
        let mut err = 0.0_f64;
        for i in 0..n_dofs {
            let x = dm2.dof_coord(i as u32);
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
        n,
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
    n: usize,
    newton_tol: f64,
    ls_enabled: bool,
    ls_min_alpha: f64,
    ls_shrink: f64,
    ls_max_backtracks: usize,
    ls_c1: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 16,
        newton_tol: 1e-10,
        ls_enabled: true,
        ls_min_alpha: 1e-6,
        ls_shrink: 0.5,
        ls_max_backtracks: 20,
        ls_c1: 1e-4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"          => { a.n          = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--newton-tol" => { a.newton_tol = it.next().unwrap_or("1e-10".into()).parse().unwrap_or(1e-10); }
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
}

