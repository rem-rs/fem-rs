//! # Example 33 — Fractional Laplacian baseline (toward MFEM ex33)
//!
//! Solves the spectral fractional Dirichlet problem on the unit square:
//!
//! ```text
//!   (-Δ)^s u = f  in Ω = [0,1]²,
//!            u = 0  on ∂Ω,
//! ```
//!
//! using a dense generalized-eigen decomposition of the reduced FE pair
//! `(K, M)`. This is a small-scale baseline that reuses existing H¹ assembly
//! and eigen infrastructure without adding a large matrix-function backend.
//!
//! The manufactured solution is the first Dirichlet eigenmode
//! `u(x,y) = sin(πx) sin(πy)` with eigenvalue `λ₁ = 2π²`, so the right-hand side is
//! `f = λ₁^s u`.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex33_fractional_laplacian
//! cargo run --example mfem_ex33_fractional_laplacian -- --n 10 --s 0.35
//! ```

use std::collections::HashSet;
use std::f64::consts::PI;

use fem_assembly::{Assembler, GridFunction, standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, constraints::boundary_dofs, fe_space::FESpace};
use nalgebra::{DMatrix, DVector, linalg::{Cholesky, SymmetricEigen}};

fn main() {
    let args = parse_args();
    let result = solve_fractional_problem(args.n, args.s);

    println!("=== fem-rs Example 33: fractional Laplacian baseline ===");
    println!("  Mesh: {}x{} subdivisions, P1 elements", args.n, args.n);
    println!("  Fractional exponent s: {:.3}", args.s);
    println!("  Free DOFs: {}", result.free_dofs);
    println!("  First generalized eigenvalue: {:.8}", result.first_eigenvalue);
    println!("  Exact first eigenvalue:       {:.8}", result.exact_first_eigenvalue);
    println!("  Relative eigenvalue error:    {:.3e}", result.first_eigenvalue_rel_error);
    println!("  L2 error vs exact mode:       {:.3e}", result.l2_error);
    println!();
    println!("Note: this is a dense spectral baseline for ex33; scalable rational/extension-based fractional operators are still pending.");
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    s: f64,
}

#[derive(Debug, Clone)]
struct FractionalResult {
    free_dofs: usize,
    first_eigenvalue: f64,
    exact_first_eigenvalue: f64,
    first_eigenvalue_rel_error: f64,
    l2_error: f64,
}

fn parse_args() -> Args {
    let mut args = Args { n: 8, s: 0.5 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                args.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
            }
            "--s" => {
                args.s = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5);
            }
            _ => {}
        }
    }
    args.s = args.s.clamp(1.0e-6, 0.999_999);
    args
}

fn solve_fractional_problem(n: usize, s: f64) -> FractionalResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);

    let stiffness = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);

    let lambda_exact = 2.0 * PI * PI;
    let rhs_integrator = DomainSourceIntegrator::new(|x: &[f64]| lambda_exact.powf(s) * exact_mode(x));
    let rhs = Assembler::assemble_linear(&space, &[&rhs_integrator], 3);

    let dm = space.dof_manager();
    let boundary = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let boundary_set: HashSet<u32> = boundary.iter().copied().collect();
    let free: Vec<usize> = (0..space.n_dofs()).filter(|&i| !boundary_set.contains(&(i as u32))).collect();

    let k_free = extract_submatrix(&stiffness, &free);
    let m_free = extract_submatrix(&mass, &free);
    let rhs_free = extract_subvector(&rhs, &free);

    let (eigenvalues, eigenvectors_m_orthonormal) = generalized_eigendecomposition(&k_free, &m_free);

    let b = DVector::from_vec(rhs_free);
    let modal_rhs = eigenvectors_m_orthonormal.transpose() * b;
    let mut modal_solution = DVector::zeros(eigenvalues.len());
    for i in 0..eigenvalues.len() {
        modal_solution[i] = modal_rhs[i] / eigenvalues[i].powf(s);
    }
    let reduced_solution = &eigenvectors_m_orthonormal * modal_solution;

    let mut full_solution = vec![0.0_f64; space.n_dofs()];
    for (fi, &gi) in free.iter().enumerate() {
        full_solution[gi] = reduced_solution[fi];
    }

    let gf = GridFunction::new(&space, full_solution);
    let l2_error = gf.compute_l2_error(&exact_mode, 4);

    FractionalResult {
        free_dofs: free.len(),
        first_eigenvalue: eigenvalues[0],
        exact_first_eigenvalue: lambda_exact,
        first_eigenvalue_rel_error: ((eigenvalues[0] - lambda_exact) / lambda_exact).abs(),
        l2_error,
    }
}

fn exact_mode(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn extract_submatrix(a: &CsrMatrix<f64>, free: &[usize]) -> CsrMatrix<f64> {
    let n = free.len();
    let mut rev = vec![usize::MAX; a.nrows];
    for (fi, &gi) in free.iter().enumerate() {
        rev[gi] = fi;
    }

    let mut coo = CooMatrix::<f64>::new(n, n);
    for (fi, &gi) in free.iter().enumerate() {
        for ptr in a.row_ptr[gi]..a.row_ptr[gi + 1] {
            let gj = a.col_idx[ptr] as usize;
            let fj = rev[gj];
            if fj != usize::MAX {
                coo.add(fi, fj, a.values[ptr]);
            }
        }
    }
    coo.into_csr()
}

fn extract_subvector(v: &[f64], free: &[usize]) -> Vec<f64> {
    free.iter().map(|&i| v[i]).collect()
}

fn csr_to_dense(a: &CsrMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(a.nrows, a.ncols, &a.to_dense())
}

fn generalized_eigendecomposition(k: &CsrMatrix<f64>, m: &CsrMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let k_dense = csr_to_dense(k);
    let m_dense = csr_to_dense(m);

    let chol = Cholesky::new(m_dense).expect("mass matrix Cholesky failed in ex33 baseline");
    let l = chol.l();
    let l_inv = l.clone().try_inverse().expect("failed to invert Cholesky factor in ex33 baseline");

    let transformed = &l_inv * k_dense * l_inv.transpose();
    let eig = SymmetricEigen::new(transformed);

    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&i, &j| eig.eigenvalues[i].partial_cmp(&eig.eigenvalues[j]).unwrap());

    let mut eigenvalues = Vec::with_capacity(order.len());
    let mut eigenvectors = DMatrix::<f64>::zeros(order.len(), order.len());
    for (dst_col, &src_col) in order.iter().enumerate() {
        eigenvalues.push(eig.eigenvalues[src_col]);
        let col = eig.eigenvectors.column(src_col).into_owned();
        eigenvectors.set_column(dst_col, &col);
    }

    let m_orthonormal = l_inv.transpose() * eigenvectors;
    (eigenvalues, m_orthonormal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex33_fractional_first_mode_is_recovered_for_s_half() {
        let result = solve_fractional_problem(8, 0.5);
        assert!(result.first_eigenvalue_rel_error < 8.0e-2, "first eigenvalue rel error = {}", result.first_eigenvalue_rel_error);
        assert!(result.l2_error < 7.0e-2, "L2 error = {}", result.l2_error);
    }

    #[test]
    fn ex33_fractional_first_mode_is_recovered_for_small_s() {
        let result = solve_fractional_problem(8, 0.25);
        assert!(result.first_eigenvalue_rel_error < 8.0e-2, "first eigenvalue rel error = {}", result.first_eigenvalue_rel_error);
        assert!(result.l2_error < 7.0e-2, "L2 error = {}", result.l2_error);
    }
}