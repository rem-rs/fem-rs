//! # Example: Maxwell Cavity Eigenvalue Problem (LOBPCG)
//!
//! Computes the lowest resonant frequencies of a perfectly conducting
//! electromagnetic cavity by solving the H(curl) generalized eigenvalue problem:
//!
//! ```text
//!   curl curl E = ω² ε E    in Ω
//!         n×E = 0            on ∂�? (PEC boundary)
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
//! - `K = �?μ⁻�?(curl E) · (curl v) dx`  �?curl-curl stiffness
//! - `M = �?ε E · v dx`                   �?vector mass (permittivity weighted)
//!
//! ## Analytical solution (unit square cavity, μ=ε=1)
//!
//! For the 2D vector curl-curl problem `curl curl E = ω² E` with `n×E = 0` on `∂Ω`,
//! divergence-free eigenfunctions satisfy `curl curl E = -ΔE = ω² E`.
//!
//! The lowest non-zero eigenvalues are:
//! ```text
//!   ω²�?= π²       �?9.870    E = (sin(πy), sin(πx))
//!   ω²�?= 4π²      �?39.478   E = (sin(2πy), sin(2πx))
//!   ω²�?= 5π²      �?49.348   E = (sin(πy)cos(2πx), sin(2πx)cos(πy))  etc.
//!   ω²�?= 8π²      �?78.957   E = (sin(2πy), sin(2πx)) with mixed modes
//! ```
//!
//! Note: this differs from the scalar Helmholtz eigenvalues `π²(m²+n²)` with `m,n�?`.
//! The vector curl-curl problem admits modes where one component varies in x and the
//! other in y independently, giving smaller eigenvalues like `π²(1²+0²) = π²`.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex13_eigenvalue
//! cargo run --example mfem_ex13_eigenvalue -- --n 16 --k 4
//! cargo run --example mfem_ex13_eigenvalue -- --n 8 --k 3
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
        println!("  (no physical eigenvalues found �?try larger --n or smaller --k)");
        return;
    }

    let h = 1.0 / n as f64;
    println!("\n  Max relative error: {:.3e}  (h={h:.4e})", result.max_rel_err);
    println!("  Converged: {}, iterations: {}", result.converged, result.iterations);
    println!("  (Expected O(h²) convergence in ω² for ND1 elements)");

    if result.max_rel_err < 0.15 {
        println!("  �?Eigenvalues within 15% of exact");
    } else {
        println!("  �?Use larger --n for better accuracy");
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

    fn rel_err(value: f64, exact: f64) -> f64 {
        (value - exact).abs() / exact.abs().max(1e-30)
    }

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

    #[test]
    fn maxwell_eigenvalue_refinement_improves_first_modes() {
        let coarse = solve_case(8, 3);
        let fine = solve_case(12, 3);

        assert!(coarse.converged, "coarse LOBPCG did not converge");
        assert!(fine.converged, "refined LOBPCG did not converge");
        assert_eq!(coarse.eigenvalues.len(), 3);
        assert_eq!(fine.eigenvalues.len(), 3);
        assert!(coarse.max_rel_err.is_finite() && fine.max_rel_err.is_finite());

        let coarse_first_err = (coarse.eigenvalues[0] - coarse.exact_eigs[0]).abs();
        let fine_first_err = (fine.eigenvalues[0] - fine.exact_eigs[0]).abs();

        assert!(
            fine.max_rel_err < coarse.max_rel_err,
            "expected refinement to reduce max relative eigen error: coarse={} fine={}",
            coarse.max_rel_err,
            fine.max_rel_err
        );
        assert!(
            fine_first_err < coarse_first_err,
            "expected refinement to improve first eigenvalue: coarse={} fine={}",
            coarse_first_err,
            fine_first_err
        );
    }

    #[test]
    fn maxwell_eigenvalue_first_doublet_remains_nearly_degenerate() {
        let result = solve_case(10, 4);

        assert!(result.converged, "LOBPCG did not converge");
        assert!(result.eigenvalues.len() >= 2, "expected at least two physical modes");

        let lambda1 = result.eigenvalues[0];
        let lambda2 = result.eigenvalues[1];
        let exact = PI * PI;
        let split = (lambda2 - lambda1).abs() / exact;

        assert!(
            rel_err(lambda1, exact) < 1.5e-2,
            "first eigenvalue drifted too far from π²: computed={} exact={}",
            lambda1,
            exact
        );
        assert!(
            rel_err(lambda2, exact) < 1.5e-2,
            "second eigenvalue drifted too far from π²: computed={} exact={}",
            lambda2,
            exact
        );
        assert!(
            split < 1.0e-2,
            "expected first Maxwell doublet to remain nearly degenerate; relative split={}",
            split
        );
    }

    #[test]
    fn maxwell_eigenvalue_refinement_improves_fourth_mode() {
        let coarse = solve_case(8, 4);
        let fine = solve_case(12, 4);

        assert!(coarse.converged && fine.converged, "both eigen solves must converge");
        assert_eq!(coarse.eigenvalues.len(), 4);
        assert_eq!(fine.eigenvalues.len(), 4);

        let coarse_fourth_err = rel_err(coarse.eigenvalues[3], coarse.exact_eigs[3]);
        let fine_fourth_err = rel_err(fine.eigenvalues[3], fine.exact_eigs[3]);

        assert!(
            fine_fourth_err < coarse_fourth_err,
            "expected refinement to improve fourth eigenvalue: coarse={} fine={}",
            coarse_fourth_err,
            fine_fourth_err
        );
        assert!(
            fine.eigenvalues[3] > fine.eigenvalues[2],
            "expected fourth mode to stay ordered above third mode: λ3={} λ4={}",
            fine.eigenvalues[2],
            fine.eigenvalues[3]
        );
    }

    #[test]
    fn maxwell_eigenvalue_refinement_improves_first_doublet_mean_and_split() {
        let coarse = solve_case(8, 4);
        let fine = solve_case(12, 4);

        assert!(coarse.converged && fine.converged, "both eigen solves must converge");
        assert!(coarse.eigenvalues.len() >= 2 && fine.eigenvalues.len() >= 2);

        let exact = PI * PI;
        let coarse_mean = 0.5 * (coarse.eigenvalues[0] + coarse.eigenvalues[1]);
        let fine_mean = 0.5 * (fine.eigenvalues[0] + fine.eigenvalues[1]);
        let coarse_mean_err = rel_err(coarse_mean, exact);
        let fine_mean_err = rel_err(fine_mean, exact);
        let coarse_split = (coarse.eigenvalues[1] - coarse.eigenvalues[0]).abs() / exact;
        let fine_split = (fine.eigenvalues[1] - fine.eigenvalues[0]).abs() / exact;

        assert!(
            fine_mean_err < coarse_mean_err,
            "expected refinement to improve first-doublet mean: coarse={} fine={}",
            coarse_mean_err,
            fine_mean_err
        );
        assert!(
            fine_split < coarse_split,
            "expected refinement to reduce first-doublet splitting: coarse={} fine={}",
            coarse_split,
            fine_split
        );
    }

    #[test]
    fn maxwell_eigenvalue_low_modes_are_stable_when_requesting_more_pairs() {
        let base = solve_case(10, 3);
        let extended = solve_case(10, 5);

        assert!(base.converged && extended.converged, "both eigen solves must converge");
        assert_eq!(base.eigenvalues.len(), 3);
        assert!(extended.eigenvalues.len() >= 3);

        for mode in 0..3 {
            let rel_gap = (base.eigenvalues[mode] - extended.eigenvalues[mode]).abs()
                / base.eigenvalues[mode].abs().max(1e-30);
            assert!(
                rel_gap < 1.0e-10,
                "expected low Maxwell eigenmodes to remain stable when requesting more pairs: mode={} rel_gap={}",
                mode + 1,
                rel_gap
            );
        }
    }

    /// All computed Maxwell eigenvalues are positive (ω² > 0).
    #[test]
    fn maxwell_eigenvalue_all_modes_are_positive_frequencies() {
        let result = solve_case(8, 4);
        assert!(result.converged, "LOBPCG did not converge");
        for (i, &lam) in result.eigenvalues.iter().enumerate() {
            assert!(lam > 0.0,
                "expected positive eigenvalue at mode {}: got {}", i + 1, lam);
        }
    }

    /// The number of returned eigenvalues equals the requested count k.
    #[test]
    fn maxwell_eigenvalue_output_length_matches_requested_k() {
        for k in [2, 3, 4] {
            let result = solve_case(10, k);
            assert!(result.converged, "LOBPCG did not converge for k={k}");
            assert_eq!(result.eigenvalues.len(), k,
                "expected {k} eigenvalues, got {}", result.eigenvalues.len());
        }
    }
}

