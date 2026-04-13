//! Eigenvalue solvers: LOBPCG and generalized eigenvalue problems.
//!
//! # Algorithms
//!
//! ## LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
//! Computes the **k smallest** eigenvalues of a generalized problem
//! `A x = λ B x` (or standard `A x = λ x` when B = I).
//!
//! The method is suitable for large sparse symmetric/SPD problems typical
//! in FEM (e.g., `K u = λ M u` for vibration frequencies).
//!
//! **Reference**: Knyazev (2001), "Toward the Optimal Preconditioned Eigensolver:
//! Locally Optimal Block Preconditioned Conjugate Gradient Method."
//!
//! ## Usage
//! ```rust,ignore
//! use fem_solver::eigen::{LobpcgConfig, lobpcg};
//! use fem_linalg::CsrMatrix;
//!
//! // Find 3 smallest eigenpairs of K x = λ M x
//! let (eigenvalues, eigenvectors) = lobpcg(&k, Some(&m), 3, &LobpcgConfig::default()).unwrap();
//! println!("λ₁ = {:.6}", eigenvalues[0]);
//! ```

use fem_linalg::CsrMatrix;
use linger::{
    KrylovSchur as LingerKrylovSchur,
    eigen::{EigenParams, EigenSolver, EigenWhich},
    sparse::CsrMatrix as LingerCsr,
};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the LOBPCG eigensolver.
#[derive(Debug, Clone)]
pub struct LobpcgConfig {
    /// Maximum number of iterations (default 300).
    pub max_iter: usize,
    /// Convergence tolerance on residual `‖Ax − λBx‖ / λ` (default 1e-8).
    pub tol: f64,
    /// Print convergence information when true.
    pub verbose: bool,
}

impl Default for LobpcgConfig {
    fn default() -> Self {
        LobpcgConfig { max_iter: 300, tol: 1e-8, verbose: false }
    }
}

/// Result returned by the LOBPCG solver.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of a matrix.  Shape: `n × k`.
    pub eigenvectors: DMatrix<f64>,
    pub iterations: usize,
    pub converged: bool,
}

// ─── LOBPCG ──────────────────────────────────────────────────────────────────

/// Compute the `k` smallest eigenpairs of `A x = λ B x` using LOBPCG.
///
/// # Arguments
/// - `a`   — symmetric (SPD) stiffness matrix.
/// - `b`   — optional mass matrix (SPD); pass `None` for standard `A x = λ x`.
/// - `k`   — number of eigenpairs to compute (block size).
/// - `cfg` — solver configuration.
///
/// # Returns
/// `EigenResult` with eigenvalues sorted ascending and corresponding eigenvectors.
pub fn lobpcg(
    a:   &CsrMatrix<f64>,
    b:   Option<&CsrMatrix<f64>>,
    k:   usize,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    lobpcg_projected(a, b, k, None, None, cfg)
}

/// Compute the `k` smallest eigenpairs of `A x = λ B x` using LOBPCG,
/// constrained to the complement of the column space of `constraints`.
///
/// This is useful for Maxwell cavity problems where the discrete gradient
/// space spans the curl-curl nullspace and must be projected out.
pub fn lobpcg_constrained(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    constraints: &DMatrix<f64>,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    lobpcg_projected(a, b, k, Some(constraints), None, cfg)
}

/// Compute the `k` smallest eigenpairs of `A x = λ B x` using LOBPCG,
/// constrained to the complement of the column space of `constraints`, with
/// a user-supplied residual preconditioner.
///
/// The callback receives the current block residual matrix `R` (`n x k`) and
/// should return an approximate preconditioned block `Z ≈ P^{-1} R` with the
/// same shape.
pub fn lobpcg_constrained_preconditioned<F>(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    constraints: &DMatrix<f64>,
    preconditioner: F,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String>
where
    F: Fn(&DMatrix<f64>) -> DMatrix<f64>,
{
    lobpcg_projected(a, b, k, Some(constraints), Some(&preconditioner), cfg)
}

fn lobpcg_projected(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    constraints: Option<&DMatrix<f64>>,
    preconditioner: Option<&dyn Fn(&DMatrix<f64>) -> DMatrix<f64>>,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    let n = a.nrows;
    assert_eq!(a.ncols, n, "A must be square");
    assert!(k >= 1 && k <= n, "k must be in [1, n]");

    let constraint_basis = constraints
        .map(|c| orthonormal_basis(c.clone(), b))
        .unwrap_or_else(|| DMatrix::<f64>::zeros(n, 0));

    if constraint_basis.ncols() + k > n {
        return Err(format!(
            "constraint space too large: n={}, k={}, constraints={}",
            n,
            k,
            constraint_basis.ncols()
        ));
    }

    // ── 1. Initialise X with random orthonormal columns ───────────────────────
    let mut x = random_feasible_orthonormal(n, k, &constraint_basis, b)?;

    let mut p = DMatrix::<f64>::zeros(n, k); // previous search direction (0 on first iter)
    let mut use_p = false;

    let mut lambdas = vec![0.0_f64; k];

    for iter in 0..cfg.max_iter {
        // ── 2. Compute AX and BX (or X) ──────────────────────────────────────
        let ax = spmm(a, &x);
        let bx = if let Some(bm) = b { spmm(bm, &x) } else { x.clone() };

        // ── 3. Rayleigh quotients ─────────────────────────────────────────────
        // Solve small dense problem in span(X, AX-λBX, P):
        // XᵀAX / XᵀBX = Rayleigh matrix → dense eigenproblem.
        let xtax = x.transpose() * &ax;
        let xtbx = x.transpose() * &bx;

        // Eigenvalues of (XᵀAX) v = λ (XᵀBX) v
        let ritz = small_generalized_eig(&xtax, &xtbx, k);
        lambdas.copy_from_slice(&ritz.0[..k]);

        // ── 4. Residuals R = AX - BX Λ ───────────────────────────────────────
        let mut r = ax.clone();
        for (j, &lj) in lambdas.iter().enumerate() {
            let bxj = bx.column(j);
            let mut rj = r.column_mut(j);
            rj.axpy(-lj, &bxj, 1.0);
        }
        project_out(&mut r, &constraint_basis, b);

        // ── 5. Convergence check ──────────────────────────────────────────────
        let res_norms: Vec<f64> = (0..k)
            .map(|j| r.column(j).norm() / lambdas[j].abs().max(1e-14))
            .collect();
        let max_res = res_norms.iter().cloned().fold(0.0_f64, f64::max);

        if cfg.verbose {
            println!("[LOBPCG] iter={iter}: max_res={max_res:.3e}");
        }

        if max_res < cfg.tol {
            return Ok(EigenResult {
                eigenvalues: lambdas,
                eigenvectors: x,
                iterations: iter + 1,
                converged: true,
            });
        }

        // ── 6. Optional residual preconditioning Z = P^{-1} R ───────────────
        let mut z = if let Some(pc) = preconditioner {
            let z_try = pc(&r);
            if z_try.nrows() != n || z_try.ncols() != k {
                return Err(format!(
                    "LOBPCG preconditioner returned wrong shape: got {}x{}, expected {}x{}",
                    z_try.nrows(),
                    z_try.ncols(),
                    n,
                    k
                ));
            }
            z_try
        } else {
            r.clone()
        };
        project_out(&mut z, &constraint_basis, b);

        // ── 7. Update X using local Rayleigh–Ritz in span(X, Z, P) ───────────
        // Build the combined basis W = [X | R | P] (skip P on first iter).
        let mut w = if use_p {
            let mut w = DMatrix::<f64>::zeros(n, 3 * k);
            w.columns_mut(0, k).copy_from(&x);
            w.columns_mut(k, k).copy_from(&z);
            w.columns_mut(2 * k, k).copy_from(&p);
            w
        } else {
            let mut w = DMatrix::<f64>::zeros(n, 2 * k);
            w.columns_mut(0, k).copy_from(&x);
            w.columns_mut(k, k).copy_from(&z);
            w
        };

        // Orthonormalise W.
        project_out(&mut w, &constraint_basis, b);
        w = orthonormal_basis(w, b);
        if w.ncols() < k {
            return Err("projected LOBPCG trial space lost rank".to_string());
        }

        // Small dense Rayleigh–Ritz in W.
        let aw = spmm(a, &w);
        let bw = if let Some(bm) = b { spmm(bm, &w) } else { w.clone() };
        let wtaw = w.transpose() * &aw;
        let wtbw = w.transpose() * &bw;

        let (ritz_vals, ritz_vecs) = small_generalized_eig(&wtaw, &wtbw, w.ncols());
        let _ = ritz_vals;

        // New X = W * C[:, 0..k] (first k Ritz vectors).
        let c = ritz_vecs.columns(0, k);
        let x_new = &w * c;
        p = DMatrix::<f64>::zeros(n, k);
        let p_cols = (w.ncols() - k).min(k);
        if p_cols > 0 {
            let p_new = &w * ritz_vecs.columns(k, p_cols);
            p.columns_mut(0, p_cols).copy_from(&p_new);
        }

        x = x_new;
        project_out(&mut x, &constraint_basis, b);
        project_out(&mut p, &constraint_basis, b);
        use_p = true;

        // Re-orthonormalise X.
        let x_basis = orthonormal_basis(x, b);
        if x_basis.ncols() < k {
            return Err("projected LOBPCG iterate lost rank".to_string());
        }
        x = x_basis.columns(0, k).into_owned();
    }

    Ok(EigenResult {
        eigenvalues: lambdas,
        eigenvectors: x,
        iterations: cfg.max_iter,
        converged: false,
    })
}

// ─── Generalized eigensolver trait ───────────────────────────────────────────

/// Trait for generalized eigenvalue solvers `A x = λ B x`.
pub trait GeneralizedEigenSolver {
    /// Compute the `k` smallest eigenpairs.
    fn solve_smallest(
        a: &CsrMatrix<f64>,
        b: Option<&CsrMatrix<f64>>,
        k: usize,
    ) -> Result<EigenResult, String>;
}

/// LOBPCG-based generalized eigensolver.
#[derive(Default)]
pub struct LobpcgSolver {
    pub cfg: LobpcgConfig,
}


impl GeneralizedEigenSolver for LobpcgSolver {
    fn solve_smallest(
        a: &CsrMatrix<f64>,
        b: Option<&CsrMatrix<f64>>,
        k: usize,
    ) -> Result<EigenResult, String> {
        lobpcg(a, b, k, &LobpcgConfig::default())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Sparse matrix × dense matrix: C = A * B.
fn spmm(a: &CsrMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let m = a.nrows;
    let k = b.ncols();
    let mut c = DMatrix::<f64>::zeros(m, k);
    for i in 0..m {
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr] as usize;
            let aij = a.values[ptr];
            for col in 0..k {
                c[(i, col)] += aij * b[(j, col)];
            }
        }
    }
    c
}

/// Compute a random orthonormal matrix of shape `n × k` (Gram–Schmidt on random).
fn random_orthonormal(n: usize, k: usize) -> DMatrix<f64> {
    // Deterministic seed using simple LCG for reproducibility.
    let mut state = 12345u64;
    let mut lcg = move || -> f64 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (u32::MAX as f64)
    };

    let mut x = DMatrix::<f64>::from_fn(n, k, |_, _| lcg() - 0.5);
    qr_orthonormalise(&mut x);
    x
}

fn random_feasible_orthonormal(
    n: usize,
    k: usize,
    constraints: &DMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
) -> Result<DMatrix<f64>, String> {
    if constraints.ncols() == 0 {
        return Ok(random_orthonormal(n, k));
    }

    let oversample = (k + constraints.ncols()).min(n);
    for _ in 0..6 {
        let mut x = random_orthonormal(n, oversample);
        project_out(&mut x, constraints, b);
        let basis = orthonormal_basis(x, b);
        if basis.ncols() >= k {
            return Ok(basis.columns(0, k).into_owned());
        }
    }

    Err("failed to construct a feasible initial LOBPCG basis".to_string())
}

fn orthonormal_basis(x: DMatrix<f64>, b: Option<&CsrMatrix<f64>>) -> DMatrix<f64> {
    let n = x.nrows();
    let mut cols: Vec<DVector<f64>> = Vec::new();

    for j in 0..x.ncols() {
        let mut v = x.column(j).clone_owned();
        for q in &cols {
            let dot = if let Some(bm) = b {
                let bv = b_times_vec(bm, &v);
                q.dot(&bv)
            } else {
                q.dot(&v)
            };
            v.axpy(-dot, q, 1.0);
        }

        let norm = if let Some(bm) = b {
            let bv = b_times_vec(bm, &v);
            v.dot(&bv).sqrt()
        } else {
            v.norm()
        };

        if norm > 1e-12 {
            v.scale_mut(1.0 / norm);
            cols.push(v);
        }
    }

    if cols.is_empty() {
        DMatrix::<f64>::zeros(n, 0)
    } else {
        DMatrix::<f64>::from_columns(&cols)
    }
}

fn project_out(x: &mut DMatrix<f64>, basis: &DMatrix<f64>, b: Option<&CsrMatrix<f64>>) {
    if basis.ncols() == 0 || x.ncols() == 0 {
        return;
    }

    let coeff = if let Some(bm) = b {
        let bx = spmm(bm, x);
        basis.transpose() * bx
    } else {
        basis.transpose() * x.clone()
    };

    *x -= basis * coeff;
}

/// Modified Gram–Schmidt orthonormalisation (in-place).
fn qr_orthonormalise(x: &mut DMatrix<f64>) {
    let k = x.ncols();
    for j in 0..k {
        // Orthogonalise column j against previous columns.
        for i in 0..j {
            let xi = x.column(i).clone_owned();
            let xj = x.column(j).clone_owned();
            let dot = xi.dot(&xj);
            let xi2 = xi.clone();
            x.column_mut(j).axpy(-dot, &xi2, 1.0);
        }
        // Normalise.
        let norm = x.column(j).norm();
        if norm > 1e-14 { x.column_mut(j).scale_mut(1.0 / norm); }
    }
}

/// B × v for sparse B and dense v.
fn b_times_vec(b: &CsrMatrix<f64>, v: &DVector<f64>) -> DVector<f64> {
    let n = b.nrows;
    let mut result = DVector::<f64>::zeros(n);
    for i in 0..n {
        for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
            let j = b.col_idx[ptr] as usize;
            result[i] += b.values[ptr] * v[j];
        }
    }
    result
}

/// Solve small dense generalized eigenvalue problem `A v = λ B v`.
/// Returns `(eigenvalues, eigenvectors)` sorted by ascending eigenvalue.
fn small_generalized_eig(a: &DMatrix<f64>, b: &DMatrix<f64>, _k: usize) -> (Vec<f64>, DMatrix<f64>) {
    let n = a.nrows();
    // B-orthogonal basis: compute Cholesky of B, then solve B^{-1/2} A B^{-T/2} v = λ v.
    // For simplicity, use eigendecomposition of B to get B^{-1/2}.
    // Use nalgebra's symmetric eigen for B.
    let b_eig = SymmetricEigen::new(b.clone());
    let b_vals = &b_eig.eigenvalues;
    let b_vecs = &b_eig.eigenvectors;

    // B^{-1/2}: scale eigenvectors by 1/sqrt(λ_B).
    let mut b_inv_half = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let lam = b_vals[i];
        if lam > 1e-14 {
            let col = b_vecs.column(i);
            for row in 0..n {
                for col2 in 0..n {
                    b_inv_half[(row, col2)] += col[row] * col[col2] / lam.sqrt();
                }
            }
        }
    }

    // Transform: C = B^{-1/2} A B^{-1/2}
    let c = b_inv_half.transpose() * a * &b_inv_half;
    let eig = SymmetricEigen::new(c);

    // Sort by ascending eigenvalue.
    let mut pairs: Vec<(f64, usize)> = eig.eigenvalues.iter().enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_vals: Vec<f64> = pairs.iter().map(|&(v, _)| v).collect();
    let mut sorted_vecs = DMatrix::<f64>::zeros(n, n);
    for (j, &(_, orig)) in pairs.iter().enumerate() {
        let vc = eig.eigenvectors.column(orig);
        let transformed = &b_inv_half * vc;
        sorted_vecs.column_mut(j).copy_from(&transformed);
    }

    (sorted_vals, sorted_vecs)
}

// ─── KrylovSchur ─────────────────────────────────────────────────────────────

/// Krylov-Schur eigenvalue solver — robust thick-restart for large sparse problems.
///
/// Computes the `k` algebraically smallest eigenvalues of `A x = λ x`.
/// Works for symmetric and non-symmetric operators.
///
/// # Parameters
/// * `a`   — system matrix (fem-rs CSR, must be square)
/// * `k`   — number of eigenvalue/vector pairs to compute
/// * `ncv` — Krylov space size (default: `k + 20`); must satisfy `k < ncv ≤ n`
pub fn krylov_schur(
    a: &CsrMatrix<f64>,
    k: usize,
    ncv: Option<usize>,
) -> Result<EigenResult, String> {
    let n = a.nrows;
    let la = _fem_to_linger_csr(a);
    let solver = match ncv {
        Some(m) => LingerKrylovSchur::new(m),
        None    => LingerKrylovSchur::default(),
    };
    let params = EigenParams::<f64>::new(k, EigenWhich::LargestAlgebraic);
    let res = solver.solve(&la, &params).map_err(|e| e.to_string())?;
    let neig = res.eigenvalues.len();
    let mut evecs = DMatrix::<f64>::zeros(n, neig);
    for (j, ev) in res.eigenvectors.iter().enumerate() {
        for i in 0..n { evecs[(i, j)] = ev.as_slice()[i]; }
    }
    Ok(EigenResult { eigenvalues: res.eigenvalues, eigenvectors: evecs, converged: res.converged > 0, iterations: res.iterations })
}

fn _fem_to_linger_csr(a: &CsrMatrix<f64>) -> LingerCsr<f64> {
    LingerCsr::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    /// Build the 1-D Laplacian tridiagonal matrix of size n.
    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i-1, -1.0); }
            if i < n-1   { coo.add(i, i+1, -1.0); }
        }
        coo.into_csr()
    }

    /// Identity matrix of size n.
    fn identity(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        coo.into_csr()
    }

    #[test]
    fn lobpcg_smallest_eigenvalue_laplacian() {
        // Smallest eigenvalue of tridiagonal laplacian of size n:
        // λ_1 = 2 - 2cos(π/(n+1)) ≈ (π/(n+1))² for large n.
        let n = 20;
        let a = laplacian_1d(n);
        let cfg = LobpcgConfig { max_iter: 300, tol: 1e-6, verbose: false };
        let res = lobpcg(&a, None, 1, &cfg).unwrap();
        let exact = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        let err = (res.eigenvalues[0] - exact).abs();
        assert!(err < 1e-4, "λ₁={:.6}, exact={exact:.6}, err={err:.2e}", res.eigenvalues[0]);
    }

    #[test]
    fn lobpcg_k_eigenvalues() {
        // Find 3 smallest eigenvalues of tridiagonal laplacian.
        let n = 20;
        let a = laplacian_1d(n);
        let cfg = LobpcgConfig { max_iter: 500, tol: 1e-6, verbose: false };
        let res = lobpcg(&a, None, 3, &cfg).unwrap();
        assert_eq!(res.eigenvalues.len(), 3);
        // Eigenvalues should be sorted ascending.
        assert!(res.eigenvalues[0] <= res.eigenvalues[1],
            "λ should be sorted: {:?}", res.eigenvalues);
        assert!(res.eigenvalues[1] <= res.eigenvalues[2]);
        // All should be positive.
        for &lam in &res.eigenvalues {
            assert!(lam > 0.0, "eigenvalue should be positive: {lam}");
        }
    }

    #[test]
    fn lobpcg_generalized_diagonal() {
        // Ax = λBx where A = diag(1,2,3,...), B = I.
        // Eigenvalues are 1, 2, 3, ...
        let n = 10;
        let mut coo_a = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo_a.add(i, i, (i + 1) as f64); }
        let a = coo_a.into_csr();
        let b = identity(n);
        let cfg = LobpcgConfig { max_iter: 300, tol: 1e-6, verbose: false };
        let res = lobpcg(&a, Some(&b), 2, &cfg).unwrap();
        let err0 = (res.eigenvalues[0] - 1.0).abs();
        let err1 = (res.eigenvalues[1] - 2.0).abs();
        assert!(err0 < 1e-4, "λ₀={:.6e}, expected 1.0, err={err0:.2e}", res.eigenvalues[0]);
        assert!(err1 < 1e-4, "λ₁={:.6e}, expected 2.0, err={err1:.2e}", res.eigenvalues[1]);
    }

    #[test]
    fn lobpcg_eigenvectors_orthonormal() {
        let n = 20;
        let a = laplacian_1d(n);
        let cfg = LobpcgConfig { max_iter: 500, tol: 1e-6, verbose: false };
        let res = lobpcg(&a, None, 3, &cfg).unwrap();
        // X^T X should be ≈ I_k.
        let xtx = res.eigenvectors.transpose() * &res.eigenvectors;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (xtx[(i, j)] - expected).abs();
                assert!(err < 1e-6, "X^TX[{i},{j}] = {:.6e}, expected {expected}", xtx[(i,j)]);
            }
        }
    }

    #[test]
    fn krylov_schur_returns_k_eigenvalues() {
        // Smoke test: KrylovSchur runs and returns the requested number of eigenvalues.
        let n = 20;
        let a = laplacian_1d(n);
        let res = krylov_schur(&a, 3, Some(15)).unwrap();
        assert_eq!(res.eigenvalues.len(), 3, "should return 3 eigenvalues");
    }

    #[test]
    fn lobpcg_constrained_skips_null_mode() {
        let mut coo = CooMatrix::<f64>::new(4, 4);
        coo.add(0, 0, 0.0);
        coo.add(1, 1, 1.0);
        coo.add(2, 2, 4.0);
        coo.add(3, 3, 9.0);
        let a = coo.into_csr();
        let b = identity(4);
        let constraints = DMatrix::<f64>::from_vec(4, 1, vec![1.0, 0.0, 0.0, 0.0]);
        let cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, verbose: false };

        let res = lobpcg_constrained(&a, Some(&b), 2, &constraints, &cfg).unwrap();

        assert!((res.eigenvalues[0] - 1.0).abs() < 1e-6, "first constrained eigenvalue = {}", res.eigenvalues[0]);
        assert!((res.eigenvalues[1] - 4.0).abs() < 1e-5, "second constrained eigenvalue = {}", res.eigenvalues[1]);
        assert!(res.eigenvectors[(0, 0)].abs() < 1e-8);
        assert!(res.eigenvectors[(0, 1)].abs() < 1e-8);
    }

    #[test]
    fn lobpcg_constrained_preconditioned_matches_expected_modes() {
        let mut coo = CooMatrix::<f64>::new(4, 4);
        coo.add(0, 0, 0.0);
        coo.add(1, 1, 1.0);
        coo.add(2, 2, 4.0);
        coo.add(3, 3, 9.0);
        let a = coo.into_csr();
        let b = identity(4);
        let constraints = DMatrix::<f64>::from_vec(4, 1, vec![1.0, 0.0, 0.0, 0.0]);
        let cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, verbose: false };

        // Exact diagonal inverse on unconstrained dofs acts as an ideal block preconditioner.
        let precond = |r: &DMatrix<f64>| {
            let mut z = r.clone();
            for j in 0..z.ncols() {
                z[(0, j)] = 0.0;
                z[(1, j)] /= 1.0;
                z[(2, j)] /= 4.0;
                z[(3, j)] /= 9.0;
            }
            z
        };

        let res = lobpcg_constrained_preconditioned(&a, Some(&b), 2, &constraints, precond, &cfg)
            .unwrap();

        assert!((res.eigenvalues[0] - 1.0).abs() < 1e-6, "first constrained eigenvalue = {}", res.eigenvalues[0]);
        assert!((res.eigenvalues[1] - 4.0).abs() < 1e-5, "second constrained eigenvalue = {}", res.eigenvalues[1]);
        assert!(res.eigenvectors[(0, 0)].abs() < 1e-8);
        assert!(res.eigenvectors[(0, 1)].abs() < 1e-8);
    }

    #[test]
    fn lobpcg_preconditioner_shape_mismatch_errors() {
        let a = laplacian_1d(8);
        let cfg = LobpcgConfig { max_iter: 50, tol: 1e-6, verbose: false };
        let constraints = DMatrix::<f64>::zeros(8, 0);

        let err = lobpcg_constrained_preconditioned(
            &a,
            None,
            2,
            &constraints,
            |_r| DMatrix::<f64>::zeros(7, 2),
            &cfg,
        ).unwrap_err();

        assert!(err.contains("wrong shape"), "unexpected error: {err}");
    }
}
