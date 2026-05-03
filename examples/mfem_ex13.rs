//! # Example 13 �?Eigenvalue Problem (LOBPCG)  (analogous to MFEM ex13)
//!
//! Finds the smallest eigenvalues and eigenmodes of the Laplacian:
//!
//! ```text
//!   −Δu = λ u    in Ω = [0,1]²
//!     u = 0    on ∂�?//! ```
//!
//! In discrete form this is the generalized eigenvalue problem:
//! ```text
//!   K v = λ M v
//! ```
//! where K is the stiffness matrix and M is the mass matrix.
//!
//! The analytical eigenvalues are `λ_{m,n} = π²(m² + n²)` for m,n = 1,2,�?//! Smallest: λ₁₁ = 2π² �?19.739, λ₁₂ = λ₂₁ = 5π² �?49.348, λ₂₂ = 8π² �?78.957.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex13
//! cargo run --example mfem_ex13 -- --n 16 --k 6
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::{lobpcg, LobpcgConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

struct EigenCaseResult {
    n_dofs: usize,
    n_free: usize,
    eigenvalues: Vec<f64>,
    exact_eigs: Vec<f64>,
    max_rel_err: f64,
    converged: bool,
    iterations: usize,
}

fn main() {
    let args = parse_args();
    let result = solve_case(args.n, args.k);

    println!("=== fem-rs Example 13: Laplacian eigenvalues (LOBPCG) ===");
    println!("  Mesh: {}×{} subdivisions, {} smallest eigenpairs", args.n, args.n, args.k);
    println!("  DOFs: {}", result.n_dofs);
    println!("  Free (interior) DOFs: {}", result.n_free);

    println!("\n  Computed eigenvalues:");
    println!("  {:>4}  {:>14}  {:>14}  {:>12}", "Mode", "Computed λ", "Exact λ", "Rel. err");
    for i in 0..result.eigenvalues.len() {
        let lam = result.eigenvalues[i];
        let ex_lam = result.exact_eigs[i];
        let err = (lam - ex_lam).abs() / ex_lam.max(1.0e-30);
        println!("  {:>4}  {:>14.6}  {:>14.6}  {:>12.4e}", i + 1, lam, ex_lam, err);
    }
    println!("\n  Max relative error: {:.4e}", result.max_rel_err);
    println!("  Converged: {}, iterations: {}", result.converged, result.iterations);
    println!("\nDone.");
}

fn solve_case(n_subdiv: usize, k: usize) -> EigenCaseResult {
    // ─── 1. Mesh and H¹ space ─────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(n_subdiv);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    // ─── 2. Assemble K (stiffness) and M (mass) ───────────────────────────────
    let k_mat = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], 3
    );
    let m_mat = Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: 1.0 }], 3
    );

    // ─── 3. Apply Dirichlet BCs for eigenvalue problem ────────────────────────
    // Strategy: build reduced system restricted to free (interior) DOFs.
    // This avoids pollution from the boundary penalty modes.
    let dm   = space.dof_manager();
    let bnd  = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_set: std::collections::HashSet<u32> = bnd.iter().cloned().collect();
    let free: Vec<usize> = (0..n).filter(|&i| !bnd_set.contains(&(i as u32))).collect();
    let nf = free.len();

    // Extract the free×free submatrices using COO
    let k_free = extract_submatrix(&k_mat, &free);
    let m_free = extract_submatrix(&m_mat, &free);

    // ─── 4. Solve with LOBPCG ─────────────────────────────────────────────────
    let cfg = LobpcgConfig {
        max_iter: 500,
        tol:      1e-8,
        verbose:  false,
    };
    let result = lobpcg(&k_free, Some(&m_free), k, &cfg)
        .expect("LOBPCG failed");

    let exact_eigs = analytical_eigenvalues(k);
    let mut max_rel_err = 0.0_f64;
    for (lam, exact) in result.eigenvalues.iter().zip(exact_eigs.iter()) {
        let rel_err = (lam - exact).abs() / exact.max(1.0e-30);
        max_rel_err = max_rel_err.max(rel_err);
    }

    EigenCaseResult {
        n_dofs: n,
        n_free: nf,
        eigenvalues: result.eigenvalues,
        exact_eigs,
        max_rel_err,
        converged: result.converged,
        iterations: result.iterations,
    }
}

/// Return the k smallest analytical eigenvalues λ = π²(m²+n²), sorted.
fn analytical_eigenvalues(k: usize) -> Vec<f64> {
    let mut eigs: Vec<f64> = Vec::new();
    let max_mn = 10;
    for m in 1..=max_mn {
        for n in 1..=max_mn {
            eigs.push(PI * PI * (m * m + n * n) as f64);
        }
    }
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs.truncate(k);
    eigs
}

/// Extract the submatrix rows/cols indexed by `free_dofs` (a sorted subset).
fn extract_submatrix(a: &fem_linalg::CsrMatrix<f64>, free: &[usize]) -> fem_linalg::CsrMatrix<f64> {
    let n = free.len();
    // Build reverse map: global index �?free index (or usize::MAX if constrained)
    let global_n = a.nrows;
    let mut rev = vec![usize::MAX; global_n];
    for (fi, &gi) in free.iter().enumerate() { rev[gi] = fi; }

    let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
    for (fi, &gi) in free.iter().enumerate() {
        for ptr in a.row_ptr[gi]..a.row_ptr[gi+1] {
            let gj = a.col_idx[ptr] as usize;
            let fj = rev[gj];
            if fj != usize::MAX {
                coo.add(fi, fj, a.values[ptr]);
            }
        }
    }
    coo.into_csr()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, k: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 12, k: 6 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("12".into()).parse().unwrap_or(12); }
            "--k" => { a.k = it.next().unwrap_or("6".into()).parse().unwrap_or(6); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(value: f64, exact: f64) -> f64 {
        (value - exact).abs() / exact.abs().max(1.0e-30)
    }

    #[test]
    fn ex13_scalar_eigenvalues_coarse_mesh_matches_first_modes() {
        let result = solve_case(10, 3);

        assert!(result.converged, "LOBPCG did not converge");
        assert_eq!(result.eigenvalues.len(), 3);
        assert!(result.max_rel_err < 7.5e-2, "max relative error = {}", result.max_rel_err);
        assert!(rel_err(result.eigenvalues[0], 2.0 * PI * PI) < 3.0e-2);
        assert!(rel_err(result.eigenvalues[1], 5.0 * PI * PI) < 5.0e-2);
        assert!(rel_err(result.eigenvalues[2], 5.0 * PI * PI) < 7.0e-2);
    }

    #[test]
    fn ex13_scalar_refinement_improves_first_eigenvalue() {
        let coarse = solve_case(8, 3);
        let fine = solve_case(12, 3);

        assert!(coarse.converged && fine.converged);
        let exact = 2.0 * PI * PI;
        let coarse_err = rel_err(coarse.eigenvalues[0], exact);
        let fine_err = rel_err(fine.eigenvalues[0], exact);

        assert!(
            fine_err < coarse_err,
            "expected refinement to improve first scalar eigenvalue: coarse={} fine={}",
            coarse_err,
            fine_err
        );
    }

    #[test]
    fn ex13_scalar_first_excited_pair_remains_nearly_degenerate() {
        let result = solve_case(10, 3);

        assert!(result.converged);
        let exact = 5.0 * PI * PI;
        let split = (result.eigenvalues[2] - result.eigenvalues[1]).abs() / exact;

        assert!(rel_err(result.eigenvalues[1], exact) < 5.0e-2);
        assert!(rel_err(result.eigenvalues[2], exact) < 7.0e-2);
        assert!(split < 3.0e-2, "expected first excited pair to remain nearly degenerate, split={}", split);
    }

    #[test]
    fn ex13_scalar_low_modes_are_stable_when_requesting_more_modes() {
        let base = solve_case(10, 3);
        let extended = solve_case(10, 5);

        assert!(base.converged && extended.converged);
        for mode in 0..3 {
            let rel_gap = (base.eigenvalues[mode] - extended.eigenvalues[mode]).abs()
                / base.eigenvalues[mode].abs().max(1.0e-30);
            assert!(
                rel_gap < 1.0e-10,
                "expected low scalar eigenmodes to remain stable when requesting more modes: mode={} rel_gap={}",
                mode + 1,
                rel_gap
            );
        }
    }

    #[test]
    fn ex13_dof_count_matches_p1_h1_formula() {
        // P1 H1 on n×n tri mesh: (n+1)^2 nodes
        for n in [6usize, 8, 12] {
            let result = solve_case(n, 3);
            let expected = (n + 1) * (n + 1);
            assert_eq!(result.n_dofs, expected,
                "DOF count mismatch for n={}: got {} expected {}", n, result.n_dofs, expected);
            assert!(result.n_free < result.n_dofs,
                "free DOFs must be strictly less than total (boundary conditions applied)");
        }
    }

    #[test]
    fn ex13_eigenvalue_convergence_order_is_at_least_linear() {
        let coarse = solve_case(8, 3);
        let fine = solve_case(16, 3);
        assert!(coarse.converged && fine.converged);
        let exact = 2.0 * PI * PI;
        let coarse_err = rel_err(coarse.eigenvalues[0], exact);
        let fine_err = rel_err(fine.eigenvalues[0], exact);
        // Doubling mesh should give at least 2x improvement (h^2 expected for P1)
        assert!(fine_err < coarse_err / 2.0,
            "expected at least linear convergence: coarse_err={:.4e} fine_err={:.4e}",
            coarse_err, fine_err);
    }

    #[test]
    fn ex13_second_eigenvalue_pair_satisfies_exact_ratio() {
        // λ₂₁ = 5π², λ₁₁ = 2π², so ratio = 2.5 exactly
        let result = solve_case(12, 3);
        assert!(result.converged);
        let ratio = result.eigenvalues[1] / result.eigenvalues[0];
        assert!((ratio - 2.5).abs() < 0.05,
            "λ₂/λ₁ should be close to 2.5 (exact ratio π²·5 / π²·2): got {:.4}", ratio);
    }

    #[test]
    fn ex13_all_eigenvalues_are_positive() {
        let result = solve_case(10, 5);
        assert!(result.converged);
        for (i, &lam) in result.eigenvalues.iter().enumerate() {
            assert!(lam > 0.0, "eigenvalue {} should be positive: got {:.4e}", i+1, lam);
        }
        // Eigenvalues should be sorted in ascending order
        for w in result.eigenvalues.windows(2) {
            assert!(w[0] <= w[1] + 1e-10,
                "eigenvalues should be sorted: {} > {}", w[0], w[1]);
        }
    }
}

