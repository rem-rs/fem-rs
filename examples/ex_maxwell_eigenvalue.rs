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

use fem_assembly::{
    VectorAssembler,
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_space::{
    HCurlSpace,
    fe_space::FESpace,
    constraints::boundary_dofs_hcurl,
};

fn main() {
    let args = parse_args();

    println!("=== fem-rs: Maxwell Cavity Eigenvalue (Dense Solver) ===");
    println!("  Mesh: {}×{}, seeking {} smallest physical eigenvalues", args.n, args.n, args.k);

    // ─── 1. Mesh + H(curl) space ─────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = HCurlSpace::new(mesh, 1);
    let n_dof = space.n_dofs();
    println!("  Edge DOFs: {n_dof}");

    // ─── 2. Assemble K and M (full, unreduced) ───────────────────────────────
    let k_full = VectorAssembler::assemble_bilinear(
        &space, &[&CurlCurlIntegrator   { mu:    1.0 }], 4,
    );
    let m_full = VectorAssembler::assemble_bilinear(
        &space, &[&VectorMassIntegrator { alpha: 1.0 }], 4,
    );

    // ─── 3. Identify free DOFs (not on PEC boundary) ─────────────────────────
    let bnd_set: std::collections::HashSet<u32> = {
        boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4])
            .into_iter().collect()
    };
    let free_dofs: Vec<usize> = (0..n_dof as u32)
        .filter(|d| !bnd_set.contains(d))
        .map(|d| d as usize)
        .collect();
    let n_free = free_dofs.len();
    println!("  Free DOFs (interior edges): {n_free}");

    // ─── 4. Extract sub-matrices K_free and M_free ───────────────────────────
    let (k_free, m_free) = extract_submatrix_pair(&k_full, &m_full, &free_dofs);

    // ─── 5. Solve generalized eigenvalue problem K_free x = λ M_free x ──────
    // Use a direct dense solver: convert to dense, compute M^{-1/2} K M^{-1/2},
    // then use symmetric eigendecomposition.  Works for n ≤ ~32 (n_free ≤ ~2000).
    println!("  Solving dense generalized eigenproblem ({n_free}×{n_free})...");
    let all_eigs = dense_generalized_eig(&k_free, &m_free);

    // ─── 6. Filter null-space eigenvalues and display ─────────────────────────
    // The curl-curl null space (gradient fields with zero BC) has eigenvalue ≈ 0.
    let null_threshold = 0.5; // first physical mode at ω² = π² ≈ 9.87
    let physical_eigs: Vec<f64> = all_eigs.iter()
        .copied()
        .filter(|&lam| lam > null_threshold)
        .take(args.k)
        .collect();

    let exact_eigs = {
        // The 2D curl-curl eigenvalue problem on [0,1]² has two families of
        // divergence-free eigenfunctions:
        // 1) Single-component: (sin(mπy), 0) and (0, sin(nπx)), eigenvalue m²π² or n²π².
        // 2) Stream-function: (∂ψ/∂y, -∂ψ/∂x) with ψ = sin(mπx)sin(nπy),
        //    eigenvalue π²(m²+n²), m,n ≥ 1.
        // Sorted with multiplicities: π², π², 2π², 4π², 4π², 5π², 5π², 8π², ...
        let mut ev = Vec::new();
        // Single-component modes: m²π² (two copies for x and y).
        for m in 1_i32..=10 {
            ev.push(PI * PI * (m * m) as f64);
            ev.push(PI * PI * (m * m) as f64);
        }
        // Stream-function modes: (m²+n²)π², m,n ≥ 1.
        for m in 1_i32..=10 {
            for n in 1_i32..=10 {
                ev.push(PI * PI * (m*m + n*n) as f64);
            }
        }
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Keep duplicates — each copy represents a distinct eigenfunction.
        ev.into_iter().take(args.k).collect::<Vec<_>>()
    };

    println!("\n  Cavity resonant frequencies (ω² = λ):");
    println!("  {:>4}  {:>14}  {:>14}  {:>10}", "Mode", "Computed ω²", "Exact ω²", "Rel. err");
    println!("  {}", "-".repeat(50));

    let mut max_rel_err = 0.0_f64;
    for (i, &lam) in physical_eigs.iter().enumerate() {
        let exact = exact_eigs.get(i).copied().unwrap_or(f64::NAN);
        let rel_err = if exact.is_finite() { (lam - exact).abs() / exact } else { f64::NAN };
        if rel_err.is_finite() { max_rel_err = max_rel_err.max(rel_err); }
        println!("  {:>4}  {:>14.6}  {:>14.6}  {:>10.3e}", i+1, lam, exact, rel_err);
    }

    if physical_eigs.is_empty() {
        println!("  (no physical eigenvalues found — try larger --n or smaller --k)");
        return;
    }

    let h = 1.0 / args.n as f64;
    println!("\n  Max relative error: {max_rel_err:.3e}  (h={h:.4e})");
    println!("  (Expected O(h²) convergence in ω² for ND1 elements)");

    if max_rel_err < 0.15 {
        println!("  ✓ Eigenvalues within 15% of exact");
    } else {
        println!("  ⚠ Use larger --n for better accuracy");
    }
}

// ─── Dense generalized eigensolver: K x = λ M x ──────────────────────────────
//
// Converts to dense, computes M^{-1/2} K M^{-1/2}, then symmetric eigen.
// Returns all eigenvalues sorted ascending.
fn dense_generalized_eig(
    k: &fem_linalg::CsrMatrix<f64>,
    m: &fem_linalg::CsrMatrix<f64>,
) -> Vec<f64> {
    use nalgebra::{DMatrix, SymmetricEigen};

    let n = k.nrows;
    let k_dense = k.to_dense();
    let m_dense = m.to_dense();

    let k_mat = DMatrix::from_row_slice(n, n, &k_dense);
    let m_mat = DMatrix::from_row_slice(n, n, &m_dense);

    // Compute M^{-1/2} via eigendecomposition of M (which is SPD).
    let m_eig = SymmetricEigen::new(m_mat);
    let mut m_inv_half = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let lam = m_eig.eigenvalues[i];
        if lam > 1e-14 {
            let col = m_eig.eigenvectors.column(i);
            for r in 0..n {
                for c in 0..n {
                    m_inv_half[(r, c)] += col[r] * col[c] / lam.sqrt();
                }
            }
        }
    }

    // Symmetric transform: C = M^{-1/2} K M^{-1/2}
    let c = &m_inv_half * &k_mat * &m_inv_half;
    let eig = SymmetricEigen::new(c);

    let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

// ─── Extract K_free and M_free as a pair ─────────────────────────────────────

fn extract_submatrix_pair(
    k: &fem_linalg::CsrMatrix<f64>,
    m: &fem_linalg::CsrMatrix<f64>,
    free: &[usize],
) -> (fem_linalg::CsrMatrix<f64>, fem_linalg::CsrMatrix<f64>) {
    (extract_submatrix(k, free), extract_submatrix(m, free))
}

/// Extract the sub-matrix corresponding to `free` rows/columns.
fn extract_submatrix(
    mat:  &fem_linalg::CsrMatrix<f64>,
    free: &[usize],
) -> fem_linalg::CsrMatrix<f64> {
    use fem_linalg::CooMatrix;

    let nf = free.len();
    // Inverse map: global DOF → free-DOF index (-1 if not free).
    let mut inv = vec![usize::MAX; mat.nrows];
    for (fi, &gi) in free.iter().enumerate() {
        inv[gi] = fi;
    }

    let mut coo = CooMatrix::<f64>::new(nf, nf);
    for (fi, &gi) in free.iter().enumerate() {
        let row_start = mat.row_ptr[gi];
        let row_end   = mat.row_ptr[gi + 1];
        for idx in row_start..row_end {
            let gj = mat.col_idx[idx] as usize;
            let fj = inv[gj];
            if fj != usize::MAX {
                coo.add(fi, fj, mat.values[idx]);
            }
        }
    }
    coo.into_csr()
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
