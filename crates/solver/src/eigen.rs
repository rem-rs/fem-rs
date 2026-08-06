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
//! println!("λ�?= {:.6}", eigenvalues[0]);
//! ```

use crate::solve_sparse_lu;
use fem_linalg::CsrMatrix;
use linlvo::{
    eigen::{EigenParams, EigenSolver, EigenWhich},
    sparse::CsrMatrix as linlvoCsr,
    KrylovSchur as linlvoKrylovSchur,
};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the LOBPCG eigensolver.
#[derive(Debug, Clone)]
pub struct LobpcgConfig {
    /// Maximum number of iterations (default 300).
    pub max_iter: usize,
    /// Convergence tolerance on residual `‖Ax �?λBx�?/ λ` (default 1e-8).
    pub tol: f64,
    /// Print convergence information when true.
    pub verbose: bool,
    /// When nonzero, eigenvalues below this threshold are treated as nullspace
    /// and skipped.  The Rayleigh-Ritz selects the next available non-nullspace
    /// mode instead.  Set to `0.0` (default) to disable.
    ///
    /// This is useful for curl-curl eigenvalue problems where the gradient
    /// nullspace (λ ≈ 0) must be excluded without explicit constraints.
    pub nullspace_skip: f64,
}

impl Default for LobpcgConfig {
    fn default() -> Self {
        LobpcgConfig {
            max_iter: 300,
            tol: 1e-8,
            verbose: false,
            nullspace_skip: 0.0,
        }
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

/// Build a Euclidean constraint matrix for use with [`lobpcg_constrained`].
///
/// Each column of the returned matrix is a basis vector with a single `1.0`
/// at the DOF index specified by `ess_dofs`, giving a constraint space whose
/// column span is the linear subspace of essential (Dirichlet) boundary DOFs.
/// The solver then searches **orthogonally** to this subspace, which is
/// mathematically equivalent to enforcing `u = 0` on those DOFs.
///
/// This is the recommended approach for serial LOBPCG without a preconditioner.
/// For the alternative diagonal-elimination approach (matching MFEM's
/// `EliminateEssentialBCDiag`), use [`lobpcg_essential_bc`] instead.
///
/// # Example
/// ```rust,ignore
/// let ess_dofs = collect_essential_dofs(mesh, dm, &[1, 2, 3, 4]);
/// let constraints = make_constraint_matrix(n_dofs, &ess_dofs);
/// let result = lobpcg_constrained(&a, Some(&m), k, &constraints, &cfg)?;
/// ```
pub fn make_constraint_matrix(n_dofs: usize, ess_dofs: &[usize]) -> DMatrix<f64> {
    let n_bc = ess_dofs.len();
    let mut c = DMatrix::<f64>::zeros(n_dofs, n_bc);
    for (j, &d) in ess_dofs.iter().enumerate() {
        if d < n_dofs {
            c[(d, j)] = 1.0;
        }
    }
    c
}

// ─── LOBPCG ──────────────────────────────────────────────────────────────────

/// Compute the `k` smallest eigenpairs of `A x = λ B x` using LOBPCG.
///
/// # Arguments
/// - `a`   �?symmetric (SPD) stiffness matrix.
/// - `b`   �?optional mass matrix (SPD); pass `None` for standard `A x = λ x`.
/// - `k`   �?number of eigenpairs to compute (block size).
/// - `cfg` �?solver configuration.
///
/// # Returns
/// `EigenResult` with eigenvalues sorted ascending and corresponding eigenvectors.
pub fn lobpcg(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    lobpcg_projected(a, b, k, None, None, None, cfg)
}

/// Compute the `k` smallest eigenpairs of `A x = λ B x` using LOBPCG
/// with a user-supplied residual preconditioner (no constraints).
///
/// The callback receives the current block residual matrix `R` (`n x k`) and
/// should return an approximate preconditioned block `Z ≈ P^{-1} R` with the
/// same shape.
///
/// This is the version to use when the preconditioner (e.g. AMS) handles
/// the nullspace internally — no explicit gradient constraints needed.
pub fn lobpcg_preconditioned<F>(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    preconditioner: F,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String>
where
    F: Fn(&DMatrix<f64>) -> DMatrix<f64>,
{
    lobpcg_projected(a, b, k, None, Some(&preconditioner), None, cfg)
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
    lobpcg_projected(a, b, k, Some(constraints), None, None, cfg)
}

/// Compute the `k` smallest eigenpairs of `A x = λ B x` using LOBPCG,
/// constrained to the complement of the column space of `constraints`, with
/// a user-supplied residual preconditioner.
///
/// The callback receives the current block residual matrix `R` (`n x k`) and
/// should return an approximate preconditioned block `Z �?P^{-1} R` with the
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
    lobpcg_projected(a, b, k, Some(constraints), Some(&preconditioner), None, cfg)
}

/// Like [`lobpcg_constrained_preconditioned`] but also forces a set of DOFs to
/// remain exactly zero throughout the iteration.  This is useful when the B
/// matrix has extreme diagonal values (e.g. `f64::MIN_POSITIVE` from
/// `EliminateEssentialBCDiag`) that cause B-orthonormalization underflow.
///
/// The zero DOFs are zeroed in `X`, `P`, and the preconditioned `Z` after every
/// iteration, and the residual norm is computed without those DOFs.
pub fn lobpcg_essential_bc<F>(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    constraints: &DMatrix<f64>,
    preconditioner: F,
    zero_dofs: &[usize],
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String>
where
    F: Fn(&DMatrix<f64>) -> DMatrix<f64>,
{
    lobpcg_projected(
        a,
        b,
        k,
        Some(constraints),
        Some(&preconditioner),
        Some(zero_dofs),
        cfg,
    )
}

/// Zero out specified DOFs in every column of a dense matrix.
fn zero_rows(mat: &mut DMatrix<f64>, dofs: &[usize]) {
    for &d in dofs {
        if d < mat.nrows() {
            for j in 0..mat.ncols() {
                mat[(d, j)] = 0.0;
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn lobpcg_projected(
    a: &CsrMatrix<f64>,
    b: Option<&CsrMatrix<f64>>,
    k: usize,
    constraints: Option<&DMatrix<f64>>,
    preconditioner: Option<&dyn Fn(&DMatrix<f64>) -> DMatrix<f64>>,
    zero_dofs: Option<&[usize]>,
    cfg: &LobpcgConfig,
) -> Result<EigenResult, String> {
    let n = a.nrows;
    assert_eq!(a.ncols, n, "A must be square");
    assert!(k >= 1 && k <= n, "k must be in [1, n]");

    // Constraint basis (nullspace to project out), B-orthonormalized: with
    // YᵀBY = I the projection V -= Y (Yᵀ B V) below is the exact B-projection
    // (scipy's _applyConstraints, V -= Y cho_solve(YᵀBY, BYᵀV)).  The
    // B-inner product is required: for the generalized problem the nullspace
    // (λ=0) is B-orthogonal to the sought eigenvectors, while Euclidean
    // projection leaves nullspace components that pollute the Rayleigh
    // quotient (Ritz values below the true spectrum → stalling, cf. ex32).
    // The f64::MIN_POSITIVE diagonal of M at BC DOFs does not poison the
    // B-orthonormalization: BC components contribute min()·v_bc ≈ 0 to the
    // B-inner product, so they are ignored while the column span (the full
    // gradient nullspace, BC components included) is preserved.
    let constraint_basis = constraints
        .map(|c| orthonormal_basis(c.clone(), None))
        .unwrap_or_else(|| DMatrix::<f64>::zeros(n, 0));
    let constraint_basis = orthonormal_basis(constraint_basis, b);

    if constraint_basis.ncols() + k > n {
        return Err(format!(
            "constraint space too large: n={}, k={}, constraints={}",
            n,
            k,
            constraint_basis.ncols()
        ));
    }

    // ── 1. Initialise X with random B-orthonormal (or Euclidean) columns ─────
    let mut x = random_feasible_orthonormal(n, k, &constraint_basis, b)?;
    if let Some(bc) = zero_dofs {
        zero_rows(&mut x, bc);
    }

    let mut p = DMatrix::<f64>::zeros(n, k); // previous search direction (0 on first iter)
    let mut use_p = false;

    let mut lambdas = vec![0.0_f64; k];

    for iter in 0..cfg.max_iter {
        // ── 2. Compute AX and BX (or X) ──────────────────────────────────────
        let ax = spmm(a, &x);
        let bx = if let Some(bm) = b {
            spmm(bm, &x)
        } else {
            x.clone()
        };

        // ── 3. Rayleigh quotients ─────────────────────────────────────────────
        // Solve small dense problem in span(X, AX-λBX, P):
        // XᵀAX / XᵀBX = Rayleigh matrix �?dense eigenproblem.
        let xtax = x.transpose() * &ax;
        let xtbx = x.transpose() * &bx;

        // Eigenvalues of (XᵀAX) v = λ (XᵀBX) v
        let ritz = solve_dense_generalized_eig(&xtax, &xtbx);
        lambdas.copy_from_slice(&ritz.0[..k]);

        // ── 4. Residuals R = AX - BX Λ ───────────────────────────────────────
        let mut r = ax.clone();
        for (j, &lj) in lambdas.iter().enumerate() {
            let bxj = bx.column(j);
            let mut rj = r.column_mut(j);
            rj.axpy(-lj, &bxj, 1.0);
        }
        // Project residual against the B-orthonormal constraint nullspace.
        project_out(&mut r, &constraint_basis, b);

        // ── 5. Convergence check (zero BC DOFs excluded from residual) ───────
        if let Some(bc) = zero_dofs {
            for j in 0..k {
                for &d in bc {
                    if d < n {
                        r[(d, j)] = 0.0;
                    }
                }
            }
        }
        let res_norms: Vec<f64> = (0..k)
            .map(|j| r.column(j).norm() / lambdas[j].abs().max(1e-14))
            .collect();
        let max_res = res_norms.iter().cloned().fold(0.0_f64, f64::max);

        if cfg.verbose {
            println!("[LOBPCG] iter={iter}: max_res={max_res:.3e}");
        }

        if max_res < cfg.tol {
            // When nullspace_skip is active, don't converge to nullspace modes.
            // Force more iterations until at least one eigenvalue exceeds the threshold.
            let skip = cfg.nullspace_skip > 0.0;
            let all_nullspace = skip && lambdas.iter().all(|&v| v.abs() < cfg.nullspace_skip);
            if !all_nullspace {
                return Ok(EigenResult {
                    eigenvalues: lambdas,
                    eigenvectors: x,
                    iterations: iter + 1,
                    converged: true,
                });
            }
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
        // Zero BC DOFs in Z (preconditioner may introduce components there).
        if let Some(bc) = zero_dofs {
            zero_rows(&mut z, bc);
        }
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

        // Project out the nullspace constraints (B-projection), then B-orthonormalise.
        project_out(&mut w, &constraint_basis, b);
        w = orthonormal_basis(w, b);
        if w.ncols() < k {
            return Err("projected LOBPCG trial space lost rank".to_string());
        }

        // Small dense Rayleigh–Ritz in W.
        let aw = spmm(a, &w);
        let bw = if let Some(bm) = b {
            spmm(bm, &w)
        } else {
            w.clone()
        };
        let wtaw = w.transpose() * &aw;
        let wtbw = w.transpose() * &bw;

        let (ritz_vals, ritz_vecs) = solve_dense_generalized_eig(&wtaw, &wtbw);

        // Skip nullspace modes: eigenvalues below nullspace_skip are treated as
        // zero (gradient nullspace) and excluded from the Ritz selection.
        let skip = if cfg.nullspace_skip > 0.0 {
            ritz_vals
                .iter()
                .take_while(|&&v| v.abs() < cfg.nullspace_skip)
                .count()
                .min(w.ncols().saturating_sub(k))
        } else {
            0
        };
        let n_avail = w.ncols();

        // New X = W * C[:, skip..skip+k]  (skip nullspace, take next k).
        let c = ritz_vecs.columns(skip, k);
        let x_new = &w * c;
        p = DMatrix::<f64>::zeros(n, k);
        let p_cols = (n_avail - skip - k).min(k);
        if p_cols > 0 {
            let p_new = &w * ritz_vecs.columns(skip + k, p_cols);
            p.columns_mut(0, p_cols).copy_from(&p_new);
        }

        x = x_new;
        if let Some(bc) = zero_dofs {
            zero_rows(&mut x, bc);
        }
        project_out(&mut x, &constraint_basis, b);
        if let Some(bc) = zero_dofs {
            zero_rows(&mut p, bc);
        }
        project_out(&mut p, &constraint_basis, b);
        use_p = true;

        // Re-orthonormalise X (B-inner product when B is provided).
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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        let x = random_orthonormal(n, k);
        // When B is provided, re-orthonormalize with the B-inner product so
        // the initial subspace is B-orthonormal, matching MFEM's LOBPCG.
        // This also naturally suppresses nullspace directions (e.g. BC DOFs
        // with tiny M-diagonal after EliminateEssentialBCDiag).
        if let Some(bm) = b {
            let oversample = (k + 5).min(n);
            for _ in 0..6 {
                let x2 = random_orthonormal(n, oversample);
                let basis = orthonormal_basis(x2, Some(bm));
                if basis.ncols() >= k {
                    return Ok(basis.columns(0, k).into_owned());
                }
            }
        }
        return Ok(x);
    }

    // Small random block (k+5) is enough: after projecting out the
    // constraint space (dimension n-constraints ≫ k) the columns stay
    // independent, so B-orthonormalizing yields k usable vectors.
    let oversample = (k + 5).min(n);
    for _ in 0..6 {
        let mut x = random_orthonormal(n, oversample);
        project_out(&mut x, constraints, b); // B-projection against constraints
        let basis = orthonormal_basis(x, b); // B-orthonormalise the search space
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

    // B-projection against a B-orthonormal basis (YᵀBY = I): V -= Y (Yᵀ B V).
    // With B = None this is the Euclidean projection.  B-orthonormalizing the
    // basis first makes this the exact nullspace projection for generalized
    // eigenproblems (scipy `_applyConstraints`).
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
        if norm > 1e-14 {
            x.column_mut(j).scale_mut(1.0 / norm);
        }
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
///
/// Uses eigendecomposition of B to form B^{-1/2}, then solves the
/// transformed standard symmetric eigenproblem `C y = λ y` where
/// `C = B^{-1/2} A B^{-1/2}`.  This is numerically robust even when B
/// is near-singular (eigenvalues below 1e-14 are clamped).
pub fn solve_dense_generalized_eig(a: &DMatrix<f64>, b: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
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
    let mut pairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .enumerate()
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

/// Krylov-Schur eigenvalue solver �?robust thick-restart for large sparse problems.
///
/// Computes the `k` algebraically smallest eigenvalues of `A x = λ x`.
/// Works for symmetric and non-symmetric operators.
///
/// # Parameters
/// * `a`   �?system matrix (fem-rs CSR, must be square)
/// * `k`   �?number of eigenvalue/vector pairs to compute
/// * `ncv` �?Krylov space size (default: `k + 20`); must satisfy `k < ncv �?n`
pub fn krylov_schur(
    a: &CsrMatrix<f64>,
    k: usize,
    ncv: Option<usize>,
) -> Result<EigenResult, String> {
    let n = a.nrows;
    let la = _fem_to_linlvo_csr(a);
    let solver = match ncv {
        Some(m) => linlvoKrylovSchur::new(m),
        None => linlvoKrylovSchur::default(),
    };
    let params = EigenParams::<f64>::new(k, EigenWhich::LargestAlgebraic);
    let res = solver.solve(&la, &params).map_err(|e| e.to_string())?;
    let neig = res.eigenvalues.len();
    let mut evecs = DMatrix::<f64>::zeros(n, neig);
    for (j, ev) in res.eigenvectors.iter().enumerate() {
        for i in 0..n {
            evecs[(i, j)] = ev.as_slice()[i];
        }
    }
    Ok(EigenResult {
        eigenvalues: res.eigenvalues,
        eigenvectors: evecs,
        converged: res.converged > 0,
        iterations: res.iterations,
    })
}

fn _fem_to_linlvo_csr(a: &CsrMatrix<f64>) -> linlvoCsr<f64> {
    linlvoCsr::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

// ─── ARPACK-style interface (Implicitly Restarted Arnoldi / Lanczos) ─────────

/// Eigenvalue selection mode (ARPACK-style `which` parameter).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhichEigenvalue {
    LargestMagnitude,
    SmallestMagnitude,
    LargestReal,
    SmallestReal,
    LargestImaginary,
    SmallestImaginary,
    /// Eigenvalues closest to the target σ (shift-invert mode).
    Target(f64),
}

fn which_to_linlvo(w: WhichEigenvalue) -> EigenWhich {
    match w {
        WhichEigenvalue::LargestMagnitude => EigenWhich::LargestMagnitude,
        WhichEigenvalue::SmallestMagnitude => EigenWhich::SmallestMagnitude,
        _ => EigenWhich::LargestMagnitude,
    }
}

/// ARPACK-style Implicitly Restarted Arnoldi eigensolver.
///
/// Computes `k` eigenvalues of `A x = λ x` using thick-restart Arnoldi
/// (via linlvo's Krylov-Schur).  No external libraries (no arpack-ng FFI).
///
/// # Arguments
/// * `a`     – system matrix
/// * `k`     – number of eigenvalues to compute
/// * `which` – selection criterion
/// * `ncv`   – Krylov subspace dimension (None = auto: max(k+20, 2k+1))
///
/// # Returns
/// `EigenResult` with eigenvalues matching the selection criterion.
pub fn arpack(
    a: &CsrMatrix<f64>,
    k: usize,
    which: WhichEigenvalue,
    ncv: Option<usize>,
) -> Result<EigenResult, String> {
    let n = a.nrows;
    if k >= n {
        return Err(format!("k={k} must be < n={n}"));
    }
    let krylov_dim = ncv.unwrap_or_else(|| (k + 20).max(2 * k + 1).min(n - 1));
    let la = _fem_to_linlvo_csr(a);

    match which {
        WhichEigenvalue::Target(sigma) => {
            // Shift-invert: form (A - σI), factor, solve via Krylov-Schur on the shifted system.
            // For SPD A, use Cholesky-(σI); for general, use LU.
            let mut a_shift = a.clone();
            for i in 0..n {
                for r in a_shift.row_ptr[i]..a_shift.row_ptr[i + 1] {
                    if a_shift.col_idx[r] as usize == i {
                        a_shift.values[r] -= sigma;
                        break;
                    }
                }
            }
            let la_shift = _fem_to_linlvo_csr(&a_shift);
            let solver = linlvoKrylovSchur::new(krylov_dim);
            let params = EigenParams::<f64>::new(k, EigenWhich::LargestMagnitude);
            let res = solver
                .solve(&la_shift, &params)
                .map_err(|e| e.to_string())?;
            let neig = res.eigenvalues.len();
            let mut eigenvalues = res.eigenvalues.clone();
            // Recover original eigenvalues: λ = σ + 1/θ
            for lam in &mut eigenvalues {
                *lam = sigma + 1.0 / *lam;
            }
            let mut evecs = DMatrix::<f64>::zeros(n, neig);
            for (j, ev) in res.eigenvectors.iter().enumerate() {
                for i in 0..n {
                    evecs[(i, j)] = ev.as_slice()[i];
                }
            }
            Ok(EigenResult {
                eigenvalues,
                eigenvectors: evecs,
                converged: true,
                iterations: res.iterations,
            })
        }
        _ => {
            let ew = which_to_linlvo(which);
            let solver = linlvoKrylovSchur::new(krylov_dim);
            let params = EigenParams::<f64>::new(k, ew);
            let res = solver.solve(&la, &params).map_err(|e| e.to_string())?;
            let neig = res.eigenvalues.len();
            let mut evecs = DMatrix::<f64>::zeros(n, neig);
            for (j, ev) in res.eigenvectors.iter().enumerate() {
                for i in 0..n {
                    evecs[(i, j)] = ev.as_slice()[i];
                }
            }
            Ok(EigenResult {
                eigenvalues: res.eigenvalues,
                eigenvectors: evecs,
                converged: res.converged > 0,
                iterations: res.iterations,
            })
        }
    }
}

// ─── FEAST-inspired interval eigensolver ─────────────────────────────────────

/// Configuration for the interval eigensolver.
#[derive(Debug, Clone)]
pub struct IntervalEigenConfig {
    /// Subspace size (≥ requested eigenvalues).
    pub subspace: usize,
    /// Maximum refinement iterations.
    pub max_iter: usize,
    /// Convergence tolerance on residual.
    pub tol: f64,
    pub verbose: bool,
}

impl Default for IntervalEigenConfig {
    fn default() -> Self {
        IntervalEigenConfig {
            subspace: 0,
            max_iter: 10,
            tol: 1e-8,
            verbose: false,
        }
    }
}

/// Find eigenvalues of `A x = λ x` within `[λ_min, λ_max]` using
/// a multi‑shift subspace iteration (FEAST-inspired, no complex arithmetic).
///
/// Shifts `s_q` are placed across the interval; for each shift the system
/// `(A − s_q I) y = b` is solved with the existing real‑valued sparse
/// direct solver, giving a subspace that spans the target eigenvectors.
/// Rayleigh–Ritz refinement follows.
pub fn feast_interval(
    a: &CsrMatrix<f64>,
    k: usize,
    lambda_min: f64,
    lambda_max: f64,
    cfg: &IntervalEigenConfig,
) -> Result<EigenResult, String> {
    let n = a.nrows;
    if k > n {
        return Err("k > n".into());
    }
    if lambda_max <= lambda_min {
        return Err("lambda_max must be > lambda_min".into());
    }
    let subspace = if cfg.subspace > 0 {
        cfg.subspace
    } else {
        (k + 5).max(2 * k).min(n)
    };
    let n_shifts = 4usize.min(subspace);
    let mut q = DMatrix::<f64>::zeros(n, subspace);

    // Build initial subspace from shifted solves at n_shifts points across the interval
    for s_idx in 0..n_shifts {
        let sigma = lambda_min + (lambda_max - lambda_min) * (s_idx as f64 + 0.5) / n_shifts as f64;
        // Form (A - σI)
        let mut a_shift = a.clone();
        for i in 0..n {
            for r in a_shift.row_ptr[i]..a_shift.row_ptr[i + 1] {
                if a_shift.col_idx[r] as usize == i {
                    a_shift.values[r] -= sigma;
                    break;
                }
            }
        }
        // Solve for random RHS
        let cols_per_shift = subspace / n_shifts;
        let start_col = s_idx * cols_per_shift;
        let end_col = if s_idx == n_shifts - 1 {
            subspace
        } else {
            start_col + cols_per_shift
        };
        for c in start_col..end_col {
            let mut rhs = vec![0.0; n];
            for i in 0..n {
                rhs[i] = ((i + 1) * (c + 1)).wrapping_mul(2654435761) as f64 % 100.0 / 100.0;
            }
            match solve_sparse_lu(&a_shift, &rhs) {
                Ok(y) => {
                    for i in 0..n {
                        q[(i, c)] = y[i];
                    }
                }
                Err(_) => {
                    for i in 0..n {
                        q[(i, c)] = rhs[i];
                    }
                }
            }
        }
    }

    // Orthonormalise Q
    if let Ok(q_ortho) = qr_orthonormalize(&q) {
        q = q_ortho;
    }

    // Rayleigh-Ritz: A_q = Q^T A Q, solve dense EVP
    let aq = q.transpose() * (&q_mat_mul(a, &q));
    let eig = SymmetricEigen::new(aq);
    let mut idx: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    idx.sort_by(|&i, &j| eig.eigenvalues[i].partial_cmp(&eig.eigenvalues[j]).unwrap());

    // Select eigenvalues in [λ_min, λ_max], up to k
    let mut evals = Vec::new();
    let mut evecs = Vec::new();
    for &i in &idx {
        let lam = eig.eigenvalues[i];
        if lam >= lambda_min - 1e-10 && lam <= lambda_max + 1e-10 && evals.len() < k {
            evals.push(lam);
            let mut ev = DMatrix::<f64>::zeros(n, 1);
            for j in 0..subspace {
                let c = eig.eigenvectors[(j, i)];
                for r in 0..n {
                    ev[(r, 0)] += c * q[(r, j)];
                }
            }
            // Normalise
            let norm = (0..n).map(|r| ev[(r, 0)].powi(2)).sum::<f64>().sqrt();
            if norm > 1e-14 {
                for r in 0..n {
                    ev[(r, 0)] /= norm;
                }
            }
            evecs.push(ev);
        }
    }

    let n_found = evals.len();
    let mut eigenvectors = DMatrix::<f64>::zeros(n, n_found);
    for (j, ev) in evecs.iter().enumerate() {
        for i in 0..n {
            eigenvectors[(i, j)] = ev[(i, 0)];
        }
    }

    Ok(EigenResult {
        eigenvalues: evals,
        eigenvectors,
        iterations: cfg.max_iter,
        converged: n_found >= k,
    })
}

fn q_mat_mul(a: &CsrMatrix<f64>, q: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows;
    let m = q.ncols();
    let mut aq = DMatrix::<f64>::zeros(n, m);
    for j in 0..m {
        let mut tmp = vec![0.0; n];
        a.spmv(q.column(j).as_slice(), &mut tmp);
        for i in 0..n {
            aq[(i, j)] = tmp[i];
        }
    }
    aq
}

fn qr_orthonormalize(m: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let (nrows, ncols) = m.shape();
    let mut q = m.clone();
    for j in 0..ncols {
        for i in 0..j {
            let dot: f64 = (0..nrows).map(|r| q[(r, j)] * q[(r, i)]).sum();
            for r in 0..nrows {
                q[(r, j)] -= dot * q[(r, i)];
            }
        }
        let norm: f64 = (0..nrows).map(|r| q[(r, j)].powi(2)).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for r in 0..nrows {
                q[(r, j)] /= norm;
            }
        }
    }
    Ok(q)
}

/// Configuration for the AME (Auxiliary-space Maxwell Eigensolver).
///
/// Wraps [`linlvo::eigen::AmeConfig`] for the H(curl) eigenvalue problem
/// `A x = λ M x`.  See the linlvo AME solver for details.
#[derive(Debug, Clone)]
pub struct AmeConfig {
    /// Number of eigenvalue/vector pairs to compute.
    pub nev: usize,
    /// Maximum LOBPCG iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance ‖Ax − λMx‖ / |λ| < tol (default 1e-8).
    pub tol: f64,
    /// Print convergence info when true.
    pub verbose: bool,
    /// AMS singularity regularization (default 1e-6).
    pub singularity_regularization: f64,
    /// Block oversampling (default 20).
    pub extra: usize,
}

impl Default for AmeConfig {
    fn default() -> Self {
        AmeConfig {
            nev: 5,
            max_iter: 100,
            tol: 1e-8,
            verbose: false,
            singularity_regularization: 1e-6,
            extra: 20,
        }
    }
}

/// Solve the Maxwell eigenvalue problem `A x = λ M x` using the AME solver.
///
/// The AME solver combines LOBPCG with the AMS preconditioner and discrete
/// divergence-free projection (`I − G(GᵀMG)⁻¹GᵀM`) to suppress the gradient
/// nullspace *without* explicit constraint matrices.  This matches the approach
/// used in MFEM ex13p / HYPRE AME.
///
/// # Arguments
/// * `a` — curl-curl stiffness matrix (`n × n`)
/// * `m` — mass matrix (`n × n`)
/// * `g` — discrete gradient (`n × n_nodes`), from a linear H¹ space
/// * `cfg` — solver configuration
///
/// # Returns
/// `EigenResult` with eigenvalues sorted ascending.
pub fn ame_solve(
    a: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    g: &CsrMatrix<f64>,
    cfg: &AmeConfig,
) -> Result<EigenResult, String> {
    use linlvo::eigen::AmeResult as LinlvoAmeResult;
    use linlvo::eigen::AmeSolver as LinlvoAmeSolver;

    let n = a.nrows;
    let la = _fem_to_linlvo_csr(a);
    let lm = _fem_to_linlvo_csr(m);
    let lg = _fem_to_linlvo_csr(g);

    let solver = LinlvoAmeSolver::new(cfg.nev)
        .tol(cfg.tol)
        .max_iter(cfg.max_iter)
        .verbose(cfg.verbose)
        .singularity_regularization(cfg.singularity_regularization)
        .extra(cfg.extra);

    let result: LinlvoAmeResult<f64> = solver
        .solve(&la, &lm, &lg)
        .map_err(|e| format!("AME solve failed: {e}"))?;

    let n_found = result.eigenvalues.len().min(cfg.nev);
    let mut eigenvectors = DMatrix::<f64>::zeros(n, n_found);
    for (j, ev) in result.eigenvectors.iter().enumerate().take(n_found) {
        for i in 0..n {
            eigenvectors[(i, j)] = ev.as_slice()[i];
        }
    }

    Ok(EigenResult {
        eigenvalues: result.eigenvalues[..n_found].to_vec(),
        eigenvectors,
        iterations: result.iterations,
        converged: result.converged,
    })
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
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -1.0);
            }
        }
        coo.into_csr()
    }

    /// Identity matrix of size n.
    fn identity(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 1.0);
        }
        coo.into_csr()
    }

    #[test]
    fn lobpcg_smallest_eigenvalue_laplacian() {
        // Smallest eigenvalue of tridiagonal laplacian of size n:
        // λ_1 = 2 - 2cos(π/(n+1)) �?(π/(n+1))² for large n.
        let n = 20;
        let a = laplacian_1d(n);
        let cfg = LobpcgConfig {
            max_iter: 300,
            tol: 1e-6,
            verbose: false,
            nullspace_skip: 0.0,
        };
        let res = lobpcg(&a, None, 1, &cfg).unwrap();
        let exact = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        let err = (res.eigenvalues[0] - exact).abs();
        assert!(
            err < 1e-4,
            "λ�?{:.6}, exact={exact:.6}, err={err:.2e}",
            res.eigenvalues[0]
        );
    }

    #[test]
    fn lobpcg_k_eigenvalues() {
        // Find 3 smallest eigenvalues of tridiagonal laplacian.
        let n = 20;
        let a = laplacian_1d(n);
        let cfg = LobpcgConfig {
            max_iter: 500,
            tol: 1e-6,
            verbose: false,
            nullspace_skip: 0.0,
        };
        let res = lobpcg(&a, None, 3, &cfg).unwrap();
        assert_eq!(res.eigenvalues.len(), 3);
        // Eigenvalues should be sorted ascending.
        assert!(
            res.eigenvalues[0] <= res.eigenvalues[1],
            "λ should be sorted: {:?}",
            res.eigenvalues
        );
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
        for i in 0..n {
            coo_a.add(i, i, (i + 1) as f64);
        }
        let a = coo_a.into_csr();
        let b = identity(n);
        let cfg = LobpcgConfig {
            max_iter: 300,
            tol: 1e-6,
            verbose: false,
            nullspace_skip: 0.0,
        };
        let res = lobpcg(&a, Some(&b), 2, &cfg).unwrap();
        let err0 = (res.eigenvalues[0] - 1.0).abs();
        let err1 = (res.eigenvalues[1] - 2.0).abs();
        assert!(
            err0 < 1e-4,
            "λ₀={:.6e}, expected 1.0, err={err0:.2e}",
            res.eigenvalues[0]
        );
        assert!(
            err1 < 1e-4,
            "λ�?{:.6e}, expected 2.0, err={err1:.2e}",
            res.eigenvalues[1]
        );
    }

    #[test]
    fn lobpcg_eigenvectors_orthonormal() {
        let n = 20;
        let a = laplacian_1d(n);
        let cfg = LobpcgConfig {
            max_iter: 500,
            tol: 1e-6,
            verbose: false,
            nullspace_skip: 0.0,
        };
        let res = lobpcg(&a, None, 3, &cfg).unwrap();
        // X^T X should be �?I_k.
        let xtx = res.eigenvectors.transpose() * &res.eigenvectors;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (xtx[(i, j)] - expected).abs();
                assert!(
                    err < 1e-6,
                    "X^TX[{i},{j}] = {:.6e}, expected {expected}",
                    xtx[(i, j)]
                );
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
        let cfg = LobpcgConfig {
            max_iter: 200,
            tol: 1e-8,
            verbose: false,
            nullspace_skip: 0.0,
        };

        let res = lobpcg_constrained(&a, Some(&b), 2, &constraints, &cfg).unwrap();

        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1e-6,
            "first constrained eigenvalue = {}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 4.0).abs() < 1e-5,
            "second constrained eigenvalue = {}",
            res.eigenvalues[1]
        );
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
        let cfg = LobpcgConfig {
            max_iter: 200,
            tol: 1e-8,
            verbose: false,
            nullspace_skip: 0.0,
        };

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

        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1e-6,
            "first constrained eigenvalue = {}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 4.0).abs() < 1e-5,
            "second constrained eigenvalue = {}",
            res.eigenvalues[1]
        );
        assert!(res.eigenvectors[(0, 0)].abs() < 1e-8);
        assert!(res.eigenvectors[(0, 1)].abs() < 1e-8);
    }

    #[test]
    fn lobpcg_preconditioner_shape_mismatch_errors() {
        let a = laplacian_1d(8);
        let cfg = LobpcgConfig {
            max_iter: 50,
            tol: 1e-6,
            verbose: false,
            nullspace_skip: 0.0,
        };
        let constraints = DMatrix::<f64>::zeros(8, 0);

        let err = lobpcg_constrained_preconditioned(
            &a,
            None,
            2,
            &constraints,
            |_r| DMatrix::<f64>::zeros(7, 2),
            &cfg,
        )
        .unwrap_err();

        assert!(err.contains("wrong shape"), "unexpected error: {err}");
    }

    #[test]
    fn arpack_largest_magnitude_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let res = arpack(&a, 3, WhichEigenvalue::LargestMagnitude, Some(15)).unwrap();
        assert_eq!(res.eigenvalues.len(), 3);
        // Largest magnitude of Laplacian is the LAST eigenvalue
        // For tridiagonal [-1,2,-1], λ_max ≈ 4 - 2cos(πn/(n+1)) ≈ 4
        assert!(
            res.eigenvalues[0] > 3.0,
            "largest eigenvalue should be near 4, got {}",
            res.eigenvalues[0]
        );
    }

    #[test]
    fn arpack_shift_invert_target() {
        // Shift-invert around sigma=3: should find eigenvalue closest to 3 (the middle one)
        let n = 20;
        let a = laplacian_1d(n);
        let res = arpack(&a, 1, WhichEigenvalue::Target(3.0), Some(15)).unwrap();
        assert_eq!(res.eigenvalues.len(), 1);
        // Expected eigenvalue closest to 3 is λ ≈ 3.18 (for n=20)
        assert!(
            (res.eigenvalues[0] - 3.0).abs() < 0.5,
            "shift-invert eigenvalue = {}",
            res.eigenvalues[0]
        );
    }
}
