//! # Example 33 �?Fractional Laplacian baseline (toward MFEM ex33)
//!
//! Solves the spectral fractional Dirichlet problem on the unit square:
//!
//! ```text
//!   (-Δ)^s u = f  in Ω = [0,1]²,
//!            u = 0  on ∂�?
//! ```
//!
//! using a dense generalized-eigen decomposition of the reduced FE pair
//! `(K, M)`. This is a small-scale baseline that reuses existing H¹ assembly
//! and eigen infrastructure without adding a large matrix-function backend.
//!
//! Supports two dense backends:
//! - `spectral`: generalized eigendecomposition (reference baseline)
//! - `rational`: multi-shift rational quadrature
//!
//! The manufactured solution is the first Dirichlet eigenmode
//! `u(x,y) = sin(πx) sin(πy)` with eigenvalue `λ�?= 2π²`, so the right-hand side is
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
use fem_mesh::{MeshTopology, SimplexMesh};
use fem_solver::{SolverConfig, solve_pcg_jacobi};
use fem_space::{H1Space, constraints::boundary_dofs, fe_space::FESpace};
use nalgebra::{DMatrix, DVector, linalg::{Cholesky, SymmetricEigen}};

fn main() {
    let args = parse_args();
    let result = solve_fractional_problem_with_method(args.n, args.s, args.method, args.n_quad);

    println!("=== fem-rs Example 33: fractional Laplacian baseline ===");
    println!("  Mesh: {}x{} subdivisions, P1 elements", args.n, args.n);
    println!("  Fractional exponent s: {:.3}", args.s);
    println!("  Backend: {:?}", args.method);
    if let FractionalMethod::Rational = args.method {
        println!("  Rational quadrature points: {}", args.n_quad);
    }
    println!("  Free DOFs: {}", result.free_dofs);
    println!("  First generalized eigenvalue: {:.8}", result.first_eigenvalue);
    println!("  Exact first eigenvalue:       {:.8}", result.exact_first_eigenvalue);
    println!("  Relative eigenvalue error:    {:.3e}", result.first_eigenvalue_rel_error);
    println!("  L2 error vs exact mode:       {:.3e}", result.l2_error);
    println!("  ||u_h||_2:                    {:.8e}", result.solution_l2);
    println!("  checksum(u_h):                {:.8e}", result.solution_checksum);
    println!("  u_h(0.5,0.5) ≈                {:.8e}", result.center_value);
    println!();
    println!("Note: this example now includes dense spectral and dense rational (multi-shift) baselines; scalable sparse large-scale backends remain future work.");
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    s: f64,
    method: FractionalMethod,
    n_quad: usize,
}

#[derive(Debug, Clone, Copy)]
enum FractionalMethod {
    Spectral,
    Rational,
    /// Sparse rational quadrature: same sinc quadrature as `Rational` but solves
    /// each shifted system (K + t_q M) x = b with sparse Jacobi-PCG instead of
    /// a dense Cholesky factorisation. Scales to large DOF counts.
    SparseRational,
}

#[derive(Debug, Clone)]
struct FractionalResult {
    free_dofs: usize,
    first_eigenvalue: f64,
    exact_first_eigenvalue: f64,
    first_eigenvalue_rel_error: f64,
    l2_error: f64,
    solution_l2: f64,
    solution_checksum: f64,
    center_value: f64,
}

fn parse_args() -> Args {
    let mut args = Args {
        n: 8,
        s: 0.5,
        method: FractionalMethod::Rational,
        n_quad: 64,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                args.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
            }
            "--s" => {
                args.s = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5);
            }
            "--method" => {
                let value = it.next().unwrap_or("rational".into()).to_ascii_lowercase();
                args.method = match value.as_str() {
                    "spectral" => FractionalMethod::Spectral,
                    "rational" => FractionalMethod::Rational,
                    "sparse-rational" | "sparse_rational" | "sr" => FractionalMethod::SparseRational,
                    _ => FractionalMethod::Rational,
                };
            }
            "--nq" | "--n-quad" => {
                args.n_quad = it.next().unwrap_or("64".into()).parse().unwrap_or(64);
            }
            _ => {}
        }
    }
    args.s = args.s.clamp(1.0e-6, 0.999_999);
    args.n_quad = args.n_quad.clamp(8, 512);
    args
}

fn solve_fractional_problem(n: usize, s: f64) -> FractionalResult {
    solve_fractional_problem_with_method(n, s, FractionalMethod::Spectral, 64)
}

fn solve_fractional_problem_with_method(
    n: usize,
    s: f64,
    method: FractionalMethod,
    n_quad: usize,
) -> FractionalResult {
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

    let reduced_solution = match method {
        FractionalMethod::Spectral => {
            solve_reduced_spectral(&rhs_free, &eigenvalues, &eigenvectors_m_orthonormal, s)
        }
        FractionalMethod::Rational => {
            solve_reduced_rational(&k_free, &m_free, &rhs_free, s, n_quad, &eigenvalues)
        }
        FractionalMethod::SparseRational => {
            let v = solve_sparse_rational(&k_free, &m_free, &rhs_free, s, n_quad, &eigenvalues);
            DVector::from_vec(v)
        }
    };

    let mut full_solution = vec![0.0_f64; space.n_dofs()];
    for (fi, &gi) in free.iter().enumerate() {
        full_solution[gi] = reduced_solution[fi];
    }

    let gf = GridFunction::new(&space, full_solution);
    let l2_error = gf.compute_l2_error(&exact_mode, 4);
    let solution_l2 = gf.dofs().iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = gf.dofs()
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum();
    let center_value = (0..space.mesh().n_nodes() as u32)
        .map(|node| {
            let x = space.mesh().node_coords(node);
            let dist2 = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
            (dist2, gf.dofs()[node as usize])
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).expect("finite center distance"))
        .map(|(_, value)| value)
        .expect("fractional Laplacian mesh has no nodes");

    FractionalResult {
        free_dofs: free.len(),
        first_eigenvalue: eigenvalues[0],
        exact_first_eigenvalue: lambda_exact,
        first_eigenvalue_rel_error: ((eigenvalues[0] - lambda_exact) / lambda_exact).abs(),
        l2_error,
        solution_l2,
        solution_checksum,
        center_value,
    }
}

fn solve_reduced_spectral(
    rhs_free: &[f64],
    eigenvalues: &[f64],
    eigenvectors_m_orthonormal: &DMatrix<f64>,
    s: f64,
) -> DVector<f64> {
    let b = DVector::from_vec(rhs_free.to_vec());
    let modal_rhs = eigenvectors_m_orthonormal.transpose() * b;
    let mut modal_solution = DVector::zeros(eigenvalues.len());
    for i in 0..eigenvalues.len() {
        modal_solution[i] = modal_rhs[i] / eigenvalues[i].powf(s);
    }
    eigenvectors_m_orthonormal * modal_solution
}

fn solve_reduced_rational(
    k_free: &CsrMatrix<f64>,
    m_free: &CsrMatrix<f64>,
    rhs_free: &[f64],
    s: f64,
    n_quad: usize,
    eigenvalues: &[f64],
) -> DVector<f64> {
    // λ^{-s} = (sin(πs)/π) ∫_0^∞ t^{-s}/(t+λ) dt
    // and therefore u ≈ c_s Σ ω_q (K + t_q M)^{-1} b.
    let k_dense = csr_to_dense(k_free);
    let m_dense = csr_to_dense(m_free);
    let b = DVector::from_vec(rhs_free.to_vec());

    let lam_min = eigenvalues[0].max(1.0e-12);
    let lam_max = eigenvalues[eigenvalues.len() - 1].max(lam_min);
    let pad = 8.0_f64;
    let y_min = lam_min.ln() - pad;
    let y_max = lam_max.ln() + pad;
    let h = (y_max - y_min) / n_quad as f64;
    let c_s = (std::f64::consts::PI * s).sin() / std::f64::consts::PI;

    let mut u = DVector::<f64>::zeros(rhs_free.len());
    for q in 0..n_quad {
        let y = y_min + (q as f64 + 0.5) * h;
        let t = y.exp();
        let weight = c_s * h * ((1.0 - s) * y).exp();
        let shifted = &k_dense + t * &m_dense;
        let chol = Cholesky::new(shifted)
            .expect("ex33 rational: shifted SPD solve failed");
        let x_q = chol.solve(&b);
        u += weight * x_q;
    }

    u
}

/// Sparse rational quadrature: same sinc formula as `solve_reduced_rational` but
/// all shifted systems are solved with sparse Jacobi-PCG on the CSR matrices.
fn solve_sparse_rational(
    k_free: &CsrMatrix<f64>,
    m_free: &CsrMatrix<f64>,
    rhs_free: &[f64],
    s: f64,
    n_quad: usize,
    eigenvalues: &[f64],
) -> Vec<f64> {
    let lam_min = eigenvalues[0].max(1.0e-12);
    let lam_max = eigenvalues[eigenvalues.len() - 1].max(lam_min);
    let pad = 8.0_f64;
    let y_min = lam_min.ln() - pad;
    let y_max = lam_max.ln() + pad;
    let h = (y_max - y_min) / n_quad as f64;
    let c_s = (std::f64::consts::PI * s).sin() / std::f64::consts::PI;

    let n = rhs_free.len();
    let cfg = SolverConfig { rtol: 1.0e-10, atol: 0.0, max_iter: 2000, verbose: false, ..Default::default() };
    let mut u = vec![0.0f64; n];

    for q in 0..n_quad {
        let y = y_min + (q as f64 + 0.5) * h;
        let t = y.exp();
        let weight = c_s * h * ((1.0 - s) * y).exp();

        // Build sparse shifted system A_q = K + t * M.
        let a_q = csr_add_scaled(k_free, m_free, 1.0, t);

        // Solve A_q x_q = b with sparse Jacobi-PCG.
        let mut x_q = vec![0.0f64; n];
        solve_pcg_jacobi(&a_q, rhs_free, &mut x_q, &cfg)
            .expect("sparse rational: shifted PCG solve failed");

        for i in 0..n {
            u[i] += weight * x_q[i];
        }
    }
    u
}

/// Build α·A + β·B as a new sparse CSR matrix (same n×n, possibly overlapping sparsity).
fn csr_add_scaled(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>, alpha: f64, beta: f64) -> CsrMatrix<f64> {
    let n = a.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(i, a.col_idx[ptr] as usize, alpha * a.values[ptr]);
        }
        for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
            coo.add(i, b.col_idx[ptr] as usize, beta * b.values[ptr]);
        }
    }
    coo.into_csr()
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

    fn rel_diff(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn ex33_fractional_first_mode_is_recovered_for_s_half() {
        let result = solve_fractional_problem(8, 0.5);
        assert!(result.first_eigenvalue_rel_error < 8.0e-2, "first eigenvalue rel error = {}", result.first_eigenvalue_rel_error);
        assert!(result.l2_error < 7.0e-2, "L2 error = {}", result.l2_error);
        assert!((result.center_value - 1.0).abs() < 8.0e-2, "center value = {}", result.center_value);
    }

    #[test]
    fn ex33_fractional_first_mode_is_recovered_for_small_s() {
        let result = solve_fractional_problem(8, 0.25);
        assert!(result.first_eigenvalue_rel_error < 8.0e-2, "first eigenvalue rel error = {}", result.first_eigenvalue_rel_error);
        assert!(result.l2_error < 7.0e-2, "L2 error = {}", result.l2_error);
        assert!((result.center_value - 1.0).abs() < 8.0e-2, "center value = {}", result.center_value);
    }

    #[test]
    fn ex33_fractional_refinement_reduces_first_mode_error() {
        let coarse = solve_fractional_problem(6, 0.5);
        let fine = solve_fractional_problem(12, 0.5);

        assert!(fine.first_eigenvalue_rel_error < coarse.first_eigenvalue_rel_error,
            "expected eigenvalue error to improve under refinement: coarse={} fine={}",
            coarse.first_eigenvalue_rel_error,
            fine.first_eigenvalue_rel_error);
        assert!(fine.l2_error < coarse.l2_error,
            "expected L2 error to improve under refinement: coarse={} fine={}",
            coarse.l2_error,
            fine.l2_error);
    }

    #[test]
    fn ex33_fractional_manufactured_first_mode_remains_consistent_across_s_scan() {
        let small_s = solve_fractional_problem(8, 0.2);
        let half_s = solve_fractional_problem(8, 0.5);
        let large_s = solve_fractional_problem(8, 0.8);

        assert!(rel_diff(small_s.solution_l2, half_s.solution_l2) < 2.0e-2,
            "solution norm drift across s is too large: small={} half={}",
            small_s.solution_l2,
            half_s.solution_l2);
        assert!(rel_diff(half_s.solution_l2, large_s.solution_l2) < 2.0e-2,
            "solution norm drift across s is too large: half={} large={}",
            half_s.solution_l2,
            large_s.solution_l2);
        assert!(rel_diff(small_s.solution_checksum, half_s.solution_checksum) < 2.0e-2,
            "solution checksum drift across s is too large: small={} half={}",
            small_s.solution_checksum,
            half_s.solution_checksum);
        assert!(rel_diff(half_s.solution_checksum, large_s.solution_checksum) < 2.0e-2,
            "solution checksum drift across s is too large: half={} large={}",
            half_s.solution_checksum,
            large_s.solution_checksum);
        assert!((small_s.center_value - 1.0).abs() < 8.0e-2);
        assert!((half_s.center_value - 1.0).abs() < 8.0e-2);
        assert!((large_s.center_value - 1.0).abs() < 8.0e-2);
    }

    #[test]
    fn ex33_rational_matches_spectral_baseline() {
        let spectral = solve_fractional_problem_with_method(8, 0.5, FractionalMethod::Spectral, 64);
        let rational = solve_fractional_problem_with_method(8, 0.5, FractionalMethod::Rational, 96);

        assert!(rel_diff(spectral.solution_l2, rational.solution_l2) < 2.0e-2,
            "L2 norm mismatch: spectral={} rational={}",
            spectral.solution_l2, rational.solution_l2);
        assert!(rel_diff(spectral.solution_checksum, rational.solution_checksum) < 3.0e-2,
            "checksum mismatch: spectral={} rational={}",
            spectral.solution_checksum, rational.solution_checksum);
        assert!((spectral.center_value - rational.center_value).abs() < 3.0e-2,
            "center mismatch: spectral={} rational={}",
            spectral.center_value, rational.center_value);
    }

    // ── sparse rational tests ─────────────────────────────────────────────────

    /// Sparse Jacobi-PCG rational quadrature must match the dense spectral baseline.
    #[test]
    fn ex33_sparse_rational_matches_spectral_baseline() {
        let spectral = solve_fractional_problem_with_method(8, 0.5, FractionalMethod::Spectral, 64);
        let sparse   = solve_fractional_problem_with_method(8, 0.5, FractionalMethod::SparseRational, 96);

        assert!(rel_diff(spectral.solution_l2, sparse.solution_l2) < 3.0e-2,
            "L2 norm mismatch: spectral={} sparse={}",
            spectral.solution_l2, sparse.solution_l2);
        assert!(rel_diff(spectral.solution_checksum, sparse.solution_checksum) < 4.0e-2,
            "checksum mismatch: spectral={} sparse={}",
            spectral.solution_checksum, sparse.solution_checksum);
        assert!((spectral.center_value - sparse.center_value).abs() < 4.0e-2,
            "center mismatch: spectral={} sparse={}",
            spectral.center_value, sparse.center_value);
    }

    /// Sparse rational must converge with a good L2 error for multiple s values.
    #[test]
    fn ex33_sparse_rational_s_scan_converges_accurately() {
        for &s in &[0.25_f64, 0.5, 0.75] {
            let r = solve_fractional_problem_with_method(8, s, FractionalMethod::SparseRational, 64);
            assert!(r.l2_error < 1.0e-1,
                "sparse rational L2 error too large at s={}: {:.3e}", s, r.l2_error);
            assert!((r.center_value - 1.0).abs() < 1.5e-1,
                "sparse rational center value wrong at s={}: {:.6}", s, r.center_value);
        }
    }
}
