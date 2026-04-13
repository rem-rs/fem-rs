//! # Example: Maxwell Cavity Eigenvalue Problem (LOBPCG)
//!
//! Computes the lowest resonant frequencies of a perfectly conducting
//! electromagnetic cavity by solving the H(curl) generalized eigenvalue problem:
//!
//! ```text
//!   curl curl E = ω² ε E    in Ω
//!         n×E = 0            on ∂Ω  (PEC boundary)
//! ```
//!
//! which becomes the discrete generalized eigenvalue problem on the **free DOFs**
//! (after eliminating boundary edge DOFs):
//!
//! ```text
//!   K_free x = λ M_free x     (λ = ω²)
//! ```
//!
//! where:
//! - `K = ∫ μ⁻¹ (curl E) · (curl v) dx`  — curl-curl stiffness
//! - `M = ∫ ε E · v dx`                   — vector mass (permittivity weighted)
//!
//! ## Analytical solution (unit square cavity, μ=ε=1)
//!
//! For the 2D vector curl-curl problem `curl curl E = ω² E` with `n×E = 0` on `∂Ω`,
//! divergence-free eigenfunctions satisfy `curl curl E = -ΔE = ω² E`.
//!
//! The lowest non-zero eigenvalues are:
//! ```text
//!   ω²₁ = π²       ≈ 9.870    E = (sin(πy), sin(πx))
//!   ω²₂ = 4π²      ≈ 39.478   E = (sin(2πy), sin(2πx))
//!   ω²₃ = 5π²      ≈ 49.348   E = (sin(πy)cos(2πx), sin(2πx)cos(πy))  etc.
//!   ω²₄ = 8π²      ≈ 78.957   E = (sin(2πy), sin(2πx)) with mixed modes
//! ```
//!
//! Note: this differs from the scalar Helmholtz eigenvalues `π²(m²+n²)` with `m,n≥1`.
//! The vector curl-curl problem admits modes where one component varies in x and the
//! other in y independently, giving smaller eigenvalues like `π²(1²+0²) = π²`.
//!
//! ## Usage
//! ```
//! cargo run --example ex_maxwell_eigenvalue
//! cargo run --example ex_maxwell_eigenvalue -- --n 16 --k 4
//! cargo run --example ex_maxwell_eigenvalue -- --n 8 --k 3
//! ```

use std::f64::consts::PI;

use fem_amg::AmgConfig;
use fem_examples::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_preconditioned_amg};
use fem_mesh::SimplexMesh;
use fem_solver::{LobpcgConfig, SolverConfig};
use fem_space::{
    H1Space,
    HCurlSpace,
    fe_space::FESpace,
};

fn main() {
    let args = parse_args();
    let result = solve_case(args.n, args.k);

    println!("=== fem-rs: Maxwell Cavity Eigenvalue (Constrained LOBPCG) ===");
    println!("  Mesh: {}×{}, seeking {} smallest physical eigenvalues", args.n, args.n, args.k);
    println!("  Edge DOFs: {}", result.n_dof);
    println!("  Free DOFs (interior edges): {}", result.n_free);
    println!(
        "  Solving sparse constrained generalized eigenproblem ({}×{}, nullity {})...",
        result.n_free,
        result.n_free,
        result.nullity
    );

    print_result(args.n, &result);
}

struct EigenCaseResult {
    n_dof: usize,
    n_free: usize,
    nullity: usize,
    eigenvalues: Vec<f64>,
    exact_eigs: Vec<f64>,
    max_rel_err: f64,
    converged: bool,
    iterations: usize,
}

fn solve_case(n: usize, k: usize) -> EigenCaseResult {
    // ─── 1. Mesh + H(curl) space ─────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(n);
    let space = HCurlSpace::new(mesh, 1);
    let n_dof = space.n_dofs();

    // MFEM-style boundary markers (`ess_bdr`): all boundary attributes are PEC.
    let bdr_attrs = [1, 2, 3, 4];
    let ess_bdr = [1, 1, 1, 1];

    // ─── 2. Build reduced generalized eigen-system from marker semantics ────
    let h1 = H1Space::new(SimplexMesh::<2>::unit_square_tri(n), 1);
    let eig_system = assemble_hcurl_eigen_system_from_marker(
        &h1,
        &space,
        &bdr_attrs,
        &ess_bdr,
        1.0,
        1.0,
        4,
    );
    let n_free = eig_system.hcurl_free_dofs.len();

    // ─── 3. Solve constrained generalized eigenproblem ───────────────────────
    let cfg = LobpcgConfig { max_iter: 800, tol: 1e-8, verbose: false };
    let inner_cfg = SolverConfig {
        rtol: 1e-2,
        atol: 1e-12,
        max_iter: 20,
        verbose: false,
        ..SolverConfig::default()
    };
    let result = solve_hcurl_eigen_preconditioned_amg(
        &eig_system,
        k,
        &cfg,
        AmgConfig::default(),
        &inner_cfg,
    ).expect("preconditioned LOBPCG failed");
    let physical_eigs = result.eigenvalues;

    let mut max_rel_err = 0.0_f64;
    let exact_eigs = analytical_eigenvalues(k);
    for (i, &lam) in physical_eigs.iter().enumerate() {
        let exact = exact_eigs.get(i).copied().unwrap_or(f64::NAN);
        let rel_err = if exact.is_finite() { (lam - exact).abs() / exact } else { f64::NAN };
        if rel_err.is_finite() { max_rel_err = max_rel_err.max(rel_err); }
    }

    EigenCaseResult {
        n_dof,
        n_free,
        nullity: eig_system.constraints.ncols(),
        eigenvalues: physical_eigs,
        exact_eigs,
        max_rel_err,
        converged: result.converged,
        iterations: result.iterations,
    }
}

fn print_result(n: usize, result: &EigenCaseResult) {
    println!("\n  Cavity resonant frequencies (ω² = λ):");
    println!("  {:>4}  {:>14}  {:>14}  {:>10}", "Mode", "Computed ω²", "Exact ω²", "Rel. err");
    println!("  {}", "-".repeat(50));

    for (i, &lam) in result.eigenvalues.iter().enumerate() {
        let exact = result.exact_eigs.get(i).copied().unwrap_or(f64::NAN);
        let rel_err = if exact.is_finite() { (lam - exact).abs() / exact } else { f64::NAN };
        println!("  {:>4}  {:>14.6}  {:>14.6}  {:>10.3e}", i + 1, lam, exact, rel_err);
    }

    if result.eigenvalues.is_empty() {
        println!("  (no physical eigenvalues found — try larger --n or smaller --k)");
        return;
    }

    let h = 1.0 / n as f64;
    println!("\n  Max relative error: {:.3e}  (h={h:.4e})", result.max_rel_err);
    println!("  Converged: {}, iterations: {}", result.converged, result.iterations);
    println!("  (Expected O(h²) convergence in ω² for ND1 elements)");

    if result.max_rel_err < 0.15 {
        println!("  ✓ Eigenvalues within 15% of exact");
    } else {
        println!("  ⚠ Use larger --n for better accuracy");
    }
}

fn analytical_eigenvalues(k: usize) -> Vec<f64> {
    let mut ev = Vec::new();
    for m in 1_i32..=10 {
        ev.push(PI * PI * (m * m) as f64);
        ev.push(PI * PI * (m * m) as f64);
    }
    for m in 1_i32..=10 {
        for n in 1_i32..=10 {
            ev.push(PI * PI * (m * m + n * n) as f64);
        }
    }
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev.into_iter().take(k).collect()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, k: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 16, k: 4 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--k" => { a.k = it.next().unwrap_or("4".into()).parse().unwrap_or(4); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxwell_eigenvalue_coarse_mesh_matches_first_modes() {
        let result = solve_case(8, 3);
        assert!(result.converged, "LOBPCG did not converge");
        assert_eq!(result.eigenvalues.len(), 3);
        assert!(result.max_rel_err < 1.0e-2, "max relative error = {}", result.max_rel_err);
        assert!(result.nullity > 0, "expected non-trivial discrete gradient nullspace");
        assert!(result.eigenvalues[0] > 9.0 && result.eigenvalues[0] < 10.5);
        assert!(result.eigenvalues[1] > 9.0 && result.eigenvalues[1] < 10.5);
        assert!(result.eigenvalues[2] > 19.0 && result.eigenvalues[2] < 20.5);
    }
}
