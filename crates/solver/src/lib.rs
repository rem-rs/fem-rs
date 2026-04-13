//! # fem-solver
//!
//! Iterative and direct linear solvers backed by [`linger`].
//!
//! ## Iterative solvers
//! - [`solve_cg`]          — Conjugate Gradient (SPD systems)
//! - [`solve_cg_operator`] — Conjugate Gradient with operator callback (backend-agnostic)
//! - [`solve_gmres_operator`] — GMRES with operator callback (backend-agnostic)
//! - [`solve_bicgstab_operator`] — BiCGSTAB with operator callback (backend-agnostic)
//! - [`solve_pcg_jacobi`]  — PCG with Jacobi preconditioner
//! - [`solve_pcg_ilu0`]    — PCG with ILU(0) preconditioner
//! - [`solve_pcg_ildlt`]   — PCG with ILDLᵀ preconditioner
//! - [`solve_gmres`]       — GMRES (non-symmetric systems)
//! - [`solve_bicgstab`]    — BiCGSTAB
//! - [`solve_idrs`]        — IDR(s) (non-symmetric, short-recurrence)
//! - [`solve_tfqmr`]       — TFQMR (Transpose-Free QMR)
//!
//! ## Auxiliary-space preconditioners (Hiptmair-Xu)
//! - [`solve_pcg_ams`]     — PCG with AMS for H(curl) (Maxwell)
//! - [`solve_gmres_ams`]   — GMRES with AMS for H(curl)
//! - [`solve_pcg_ads`]     — PCG with ADS for H(div) (Darcy)
//! - [`solve_gmres_ads`]   — GMRES with ADS for H(div)
//!
//! ## Direct solvers
//! - [`solve_sparse_lu`]        — Sparse LU for general systems
//! - [`solve_sparse_cholesky`]  — Sparse Cholesky for SPD systems
//! - [`solve_sparse_ldlt`]      — Sparse LDLᵀ for symmetric indefinite systems
//!
//! All solvers operate on [`fem_linalg::CsrMatrix<T>`].

use fem_linalg::CsrMatrix as FemCsr;
use linger::{
    core::scalar::Scalar as LingerScalar,
    direct::{DirectSolver, SparseLu, SparseCholesky, SparseLdlt},
    iterative::{BiCgStab, ConjugateGradient, Fgmres, Gmres, Idrs, Tfqmr},
    precond::{AmsPrecond, AmsConfig, AdsPrecond, AdsConfig},
    sparse::CsrMatrix as LingerCsr,
    DenseVec, Ilu0Precond, IldltPrecond, JacobiPrecond, KrylovSolver, SolverParams, VerboseLevel,
};
use thiserror::Error;

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("solver did not converge in {max_iter} iterations (residual = {residual:.3e})")]
    ConvergenceFailed { max_iter: usize, residual: f64 },
    #[error("dimension mismatch: matrix is {rows}×{cols}, rhs has length {rhs}")]
    DimensionMismatch { rows: usize, cols: usize, rhs: usize },
    #[error("linger error: {0}")]
    Linger(String),
}

impl From<linger::SolverError> for SolverError {
    fn from(e: linger::SolverError) -> Self {
        match e {
            linger::SolverError::ConvergenceFailed { max_iter, residual } => {
                SolverError::ConvergenceFailed { max_iter, residual }
            }
            other => SolverError::Linger(other.to_string()),
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Outcome returned by every solver in this crate.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub converged:      bool,
    pub iterations:     usize,
    pub final_residual: f64,
}

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Verbosity level for iterative solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum PrintLevel {
    /// No output.
    #[default]
    Silent,
    /// Print summary on convergence/failure only.
    Summary,
    /// Print residual at each iteration.
    Iterations,
    /// Print residual at each iteration plus extra diagnostics.
    Debug,
}


/// Convergence parameters passed to every solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Relative tolerance (converge when ‖r‖/‖b‖ < rtol).
    pub rtol: f64,
    /// Absolute tolerance.
    pub atol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Print residual each iteration when `true`.
    pub verbose: bool,
    /// Structured verbosity level (overrides `verbose` when not Silent).
    pub print_level: PrintLevel,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 1_000, verbose: false, print_level: PrintLevel::Silent }
    }
}

impl SolverConfig {
    pub fn to_linger(&self) -> SolverParams {
        let level = match self.effective_print_level() {
            PrintLevel::Silent => VerboseLevel::Silent,
            PrintLevel::Summary => VerboseLevel::Summary,
            _ => VerboseLevel::Iterations,
        };
        SolverParams {
            rtol:           self.rtol,
            atol:           self.atol,
            max_iter:       self.max_iter,
            verbose:        level,
            check_interval: 10,
        }
    }

    /// Effective print level: uses `print_level` if set, falls back to `verbose`.
    pub fn effective_print_level(&self) -> PrintLevel {
        if self.print_level != PrintLevel::Silent {
            self.print_level
        } else if self.verbose {
            PrintLevel::Iterations
        } else {
            PrintLevel::Silent
        }
    }
}

// ─── Type conversion ─────────────────────────────────────────────────────────

/// Convert a `fem_linalg::CsrMatrix<T>` to a `linger::sparse::CsrMatrix<T>`.
///
/// The only structural difference is that fem-rs stores column indices as
/// `u32` while linger uses `usize`.
pub fn fem_to_linger_csr<T: LingerScalar>(a: &FemCsr<T>) -> LingerCsr<T> {
    LingerCsr::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

// ─── Solvers ─────────────────────────────────────────────────────────────────

/// Conjugate Gradient — for symmetric positive definite systems.
///
/// # Arguments
/// * `a`   — system matrix (fem-rs CSR)
/// * `b`   — right-hand side
/// * `x`   — initial guess on entry, solution on exit
/// * `cfg` — convergence parameters
pub fn solve_cg<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = ConjugateGradient::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Conjugate Gradient using a backend-agnostic operator callback.
///
/// This entrypoint is intended for matrix-free or foreign-backend operators
/// (e.g., reed/libCEED style) that can provide `y = A*x` without exposing a
/// concrete CSR matrix.
///
/// # Arguments
/// * `nrows`, `ncols` — operator dimensions (must be square and equal to `b.len()`).
/// * `apply`          — callback that computes `y <- A * x`.
/// * `b`              — right-hand side.
/// * `x`              — initial guess on entry, solution on exit.
/// * `cfg`            — convergence parameters.
pub fn solve_cg_operator<F>(
    nrows: usize,
    ncols: usize,
    apply: F,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    if nrows != ncols || b.len() != nrows || x.len() != ncols {
        return Err(SolverError::DimensionMismatch {
            rows: nrows,
            cols: ncols,
            rhs: b.len(),
        });
    }

    let n = nrows;
    let mut r = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];

    // r0 = b - A*x0
    apply(x, &mut ap);
    for i in 0..n {
        r[i] = b[i] - ap[i];
        p[i] = r[i];
    }

    let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let tol = cfg.atol.max(cfg.rtol * norm_b.max(1e-32));

    let mut rs_old = r.iter().map(|v| v * v).sum::<f64>();
    let mut res_norm = rs_old.sqrt();
    if res_norm <= tol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: res_norm,
        });
    }

    for iter in 0..cfg.max_iter {
        apply(&p, &mut ap);
        let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
        if p_ap.abs() < 1e-32 {
            return Err(SolverError::Linger(
                "CG operator breakdown: p^T A p is near zero".to_string(),
            ));
        }

        let alpha = rs_old / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        res_norm = rs_new.sqrt();
        if res_norm <= tol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Err(SolverError::ConvergenceFailed {
        max_iter: cfg.max_iter,
        residual: res_norm,
    })
}

/// Preconditioned CG with a Jacobi (diagonal scaling) preconditioner.
pub fn solve_pcg_jacobi<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = JacobiPrecond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Preconditioned CG with an ILU(0) preconditioner.
///
/// Requires the matrix to have a factorisation-compatible sparsity pattern.
pub fn solve_pcg_ilu0<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = Ilu0Precond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES — for general (possibly non-symmetric) systems.
///
/// `restart` controls the Krylov subspace dimension before restart (default 30).
pub fn solve_gmres<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES using a backend-agnostic operator callback.
///
/// This entrypoint is intended for matrix-free or foreign-backend operators
/// that can provide `y = A*x` without exposing a concrete CSR matrix.
pub fn solve_gmres_operator<F>(
    nrows: usize,
    ncols: usize,
    apply: F,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    if nrows != ncols || b.len() != nrows || x.len() != ncols {
        return Err(SolverError::DimensionMismatch {
            rows: nrows,
            cols: ncols,
            rhs: b.len(),
        });
    }
    if restart == 0 {
        return Err(SolverError::Linger("GMRES restart must be > 0".to_string()));
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    fn norm(v: &[f64]) -> f64 {
        dot(v, v).sqrt()
    }

    let n = nrows;
    let mut iter_total = 0usize;

    let mut ax = vec![0.0; n];
    apply(x, &mut ax);
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }

    let norm_b = norm(b);
    let tol = cfg.atol.max(cfg.rtol * norm_b.max(1e-32));
    let mut res_norm = norm(&r);
    if res_norm <= tol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: res_norm,
        });
    }

    while iter_total < cfg.max_iter {
        let beta = norm(&r);
        if beta <= tol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: beta,
            });
        }

        let mut v = vec![vec![0.0; n]; restart + 1];
        for i in 0..n {
            v[0][i] = r[i] / beta;
        }

        let mut h = vec![vec![0.0; restart]; restart + 1];
        let mut cs = vec![0.0; restart];
        let mut sn = vec![0.0; restart];
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;

        let mut inner_done = 0usize;
        let mut converged = false;

        for j in 0..restart {
            if iter_total >= cfg.max_iter {
                break;
            }

            let mut w = vec![0.0; n];
            apply(&v[j], &mut w);

            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                for k in 0..n {
                    w[k] -= h[i][j] * v[i][k];
                }
            }

            h[j + 1][j] = norm(&w);
            if h[j + 1][j] > 1e-32 {
                for k in 0..n {
                    v[j + 1][k] = w[k] / h[j + 1][j];
                }
            }

            // Apply existing Givens rotations.
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = tmp;
            }

            // Build and apply new Givens rotation.
            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-32 {
                cs[j] = h[j][j] / denom;
                sn[j] = h[j + 1][j] / denom;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }

            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            h[j + 1][j] = 0.0;

            let g_next = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            g[j + 1] = g_next;

            res_norm = g[j + 1].abs();
            iter_total += 1;
            inner_done = j + 1;

            if res_norm <= tol {
                converged = true;
                break;
            }
        }

        if inner_done == 0 {
            break;
        }

        // Back-substitution: solve upper-triangular H(0..m,0..m) * y = g(0..m)
        let m = inner_done;
        let mut y = vec![0.0; m];
        for i in (0..m).rev() {
            let mut s = g[i];
            for k in i + 1..m {
                s -= h[i][k] * y[k];
            }
            let diag = h[i][i];
            if diag.abs() < 1e-32 {
                return Err(SolverError::Linger(
                    "GMRES operator breakdown: near-singular Hessenberg diagonal".to_string(),
                ));
            }
            y[i] = s / diag;
        }

        for i in 0..m {
            for k in 0..n {
                x[k] += y[i] * v[i][k];
            }
        }

        if converged {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: res_norm,
            });
        }

        apply(x, &mut ax);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        res_norm = norm(&r);
    }

    Err(SolverError::ConvergenceFailed {
        max_iter: cfg.max_iter,
        residual: res_norm,
    })
}

/// BiCGSTAB — for non-symmetric systems; often faster than GMRES per iteration.
pub fn solve_bicgstab<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = BiCgStab::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// BiCGSTAB using a backend-agnostic operator callback.
///
/// This entrypoint is intended for matrix-free or foreign-backend operators
/// that can provide `y = A*x` without exposing a concrete CSR matrix.
pub fn solve_bicgstab_operator<F>(
    nrows: usize,
    ncols: usize,
    apply: F,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    if nrows != ncols || b.len() != nrows || x.len() != ncols {
        return Err(SolverError::DimensionMismatch {
            rows: nrows,
            cols: ncols,
            rhs: b.len(),
        });
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    fn norm(v: &[f64]) -> f64 {
        dot(v, v).sqrt()
    }

    let n = nrows;
    let mut ax = vec![0.0; n];
    apply(x, &mut ax);

    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }

    let r_hat = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];

    let norm_b = norm(b);
    let tol = cfg.atol.max(cfg.rtol * norm_b.max(1e-32));
    let mut res_norm = norm(&r);
    if res_norm <= tol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: res_norm,
        });
    }

    let mut rho_old = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;

    for iter in 0..cfg.max_iter {
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < 1e-32 {
            return Err(SolverError::Linger(
                "BiCGSTAB operator breakdown: rho is near zero".to_string(),
            ));
        }

        let beta = if iter == 0 {
            0.0
        } else {
            (rho_new / rho_old) * (alpha / omega)
        };

        for i in 0..n {
            p[i] = if iter == 0 {
                r[i]
            } else {
                r[i] + beta * (p[i] - omega * v[i])
            };
        }

        apply(&p, &mut v);
        let rhat_v = dot(&r_hat, &v);
        if rhat_v.abs() < 1e-32 {
            return Err(SolverError::Linger(
                "BiCGSTAB operator breakdown: r_hat^T v is near zero".to_string(),
            ));
        }

        alpha = rho_new / rhat_v;
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }

        let s_norm = norm(&s);
        if s_norm <= tol {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: s_norm,
            });
        }

        apply(&s, &mut t);
        let tt = dot(&t, &t);
        if tt.abs() < 1e-32 {
            return Err(SolverError::Linger(
                "BiCGSTAB operator breakdown: t^T t is near zero".to_string(),
            ));
        }

        omega = dot(&t, &s) / tt;
        if omega.abs() < 1e-32 {
            return Err(SolverError::Linger(
                "BiCGSTAB operator breakdown: omega is near zero".to_string(),
            ));
        }

        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        res_norm = norm(&r);
        if res_norm <= tol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        rho_old = rho_new;
    }

    Err(SolverError::ConvergenceFailed {
        max_iter: cfg.max_iter,
        residual: res_norm,
    })
}

/// Flexible GMRES — allows a variable preconditioner per iteration.
///
/// Unlike standard GMRES, the preconditioner may change at each Krylov step
/// (e.g. inner Krylov solve, AMG V-cycle, or any nonlinear operator).
/// With a fixed preconditioner, FGMRES produces identical iterates to
/// right-preconditioned GMRES.
///
/// `restart` controls the Krylov subspace dimension before restart (default 30).
pub fn solve_fgmres<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Fgmres::<T>::new(restart);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Flexible GMRES with Jacobi preconditioner.
pub fn solve_fgmres_jacobi<T: LingerScalar>(    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = JacobiPrecond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let solver = Fgmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// ─── ILDLt preconditioned solvers ────────────────────────────────────────────

/// Preconditioned CG with an incomplete LDLᵀ preconditioner.
///
/// Best for symmetric positive-definite systems where ILU(0) may struggle.
/// ILDLt is more robust than ILU(0) for nearly-singular or ill-conditioned
/// symmetric matrices (e.g., Poisson with extreme aspect ratios).
pub fn solve_pcg_ildlt<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IldltPrecond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES with incomplete LDLᵀ preconditioner for symmetric indefinite systems.
///
/// Ideal for saddle-point problems (Stokes, Maxwell) where Cholesky/ILU fail.
pub fn solve_gmres_ildlt<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IldltPrecond::from_csr(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── IDR(s) ──────────────────────────────────────────────────────────────────

/// IDR(s) — Induced Dimension Reduction for non-symmetric systems.
///
/// Short-recurrence method; s=4 is a good default.  Typically fewer matvecs
/// than BiCGSTAB for difficult non-symmetric problems.
pub fn solve_idrs<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    s: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Idrs::<T>::new(s);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// TFQMR — Transpose-Free Quasi-Minimal Residual for non-symmetric systems.
///
/// Does not require the transpose of A; converges smoothly on problems where
/// BiCGSTAB may stagnate.
pub fn solve_tfqmr<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = Tfqmr::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linger())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Direct solvers ───────────────────────────────────────────────────────────

/// Sparse LU direct solver for general square systems.
///
/// Exact solve (up to floating-point precision).  Use for small-to-medium
/// problems or as a reference/preconditioner.
pub fn solve_sparse_lu<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = SparseLu::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linger(e.to_string()))?;
    Ok(lx.into_vec())
}

/// Sparse Cholesky direct solver for symmetric positive-definite systems.
///
/// About 2× faster than LU for SPD matrices.
pub fn solve_sparse_cholesky<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = SparseCholesky::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linger(e.to_string()))?;
    Ok(lx.into_vec())
}

/// Sparse LDLᵀ direct solver for symmetric indefinite systems (Stokes, Maxwell saddle-point).
pub fn solve_sparse_ldlt<T: LingerScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = SparseLdlt::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linger(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linger(e.to_string()))?;
    Ok(lx.into_vec())
}

// ─── Auxiliary-space Maxwell Solver (AMS) ────────────────────────────────────

/// Configuration for AMS (Auxiliary-space Maxwell Solver) preconditioner.
///
/// AMS is the Hiptmair-Xu preconditioner for H(curl) problems (Maxwell).
/// It uses a multigrid V-cycle on the auxiliary nodal space plus
/// a stationary correction on the edge space.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct AmsSolverConfig {
    pub inner_cfg: SolverConfig,
    pub ams_cfg: AmsConfig,
}


/// Solve an H(curl) system using PCG with AMS preconditioner.
///
/// # Arguments
/// * `a`       — H(curl) stiffness matrix (edge DOFs)
/// * `g`       — Discrete gradient matrix (vertices -> edges)
/// * `b`       — right-hand side
/// * `x`       — initial guess on entry, solution on exit
/// * `cfg`     — solver configuration
///
/// # Type parameters
/// The discrete gradient `g` is passed as a linger CsrMatrix to match internal types.
/// Convert using `fem_to_linger_csr`.
pub fn solve_pcg_ams<T: LingerScalar>(
    a: &FemCsr<T>,
    g: &LingerCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AmsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ams = AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
        .map_err(|e| SolverError::Linger(e.to_string()))?;

    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&ams), &lb, &mut lx, &cfg.inner_cfg.to_linger())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve an H(curl) system using GMRES with AMS preconditioner.
///
/// Use this for non-symmetric H(curl) problems (e.g., with absorbing BCs).
pub fn solve_gmres_ams<T: LingerScalar>(
    a: &FemCsr<T>,
    g: &LingerCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &AmsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ams = AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
        .map_err(|e| SolverError::Linger(e.to_string()))?;

    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&ams), &lb, &mut lx, &cfg.inner_cfg.to_linger())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Auxiliary-space Divergence Solver (ADS) ─────────────────────────────────

/// Configuration for ADS (Auxiliary-space Divergence Solver) preconditioner.
///
/// ADS is the Hiptmair-Xu preconditioner for H(div) problems (Darcy flow).
/// It combines auxiliary-space cycles on the edge space (via curl) and
/// nodal space (via gradient) for robust H(div) preconditioning.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct AdsSolverConfig {
    pub inner_cfg: SolverConfig,
    pub ads_cfg: AdsConfig,
}


/// Solve an H(div) system using PCG with ADS preconditioner.
///
/// # Arguments
/// * `a`       — H(div) stiffness matrix (face DOFs)
/// * `c`       — Discrete curl matrix (edges -> faces)
/// * `g`       — Discrete gradient matrix (vertices -> edges)
/// * `b`       — right-hand side
/// * `x`       — initial guess on entry, solution on exit
/// * `cfg`     — solver configuration
///
/// # Notes
/// Both `c` and `g` should be converted to linger format using `fem_to_linger_csr`.
pub fn solve_pcg_ads<T: LingerScalar>(
    a: &FemCsr<T>,
    c: &LingerCsr<T>,
    g: &LingerCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AdsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linger(e.to_string()))?;

    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&ads), &lb, &mut lx, &cfg.inner_cfg.to_linger())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve an H(div) system using GMRES with ADS preconditioner.
///
/// Use this for non-symmetric H(div) problems.
pub fn solve_gmres_ads<T: LingerScalar>(
    a: &FemCsr<T>,
    c: &LingerCsr<T>,
    g: &LingerCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &AdsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linger_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linger(e.to_string()))?;

    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&ads), &lb, &mut lx, &cfg.inner_cfg.to_linger())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

fn check_dims<T>(a: &FemCsr<T>, b: &[T], x: &[T]) -> Result<(), SolverError> {
    if a.nrows != b.len() || a.ncols != x.len() {
        return Err(SolverError::DimensionMismatch {
            rows: a.nrows,
            cols: a.ncols,
            rhs:  b.len(),
        });
    }
    Ok(())
}

pub fn into_result(r: linger::SolverResult) -> SolveResult {
    SolveResult {
        converged:      r.converged,
        iterations:     r.iterations,
        final_residual: r.final_residual,
    }
}

pub mod block;
pub mod eigen;
pub mod lor;
pub mod ode;
pub use block::{BlockSystem, BlockDiagonalPrecond, BlockTriangularPrecond, SchurComplementSolver, MinresSolver};
pub use eigen::{lobpcg, lobpcg_constrained, lobpcg_constrained_preconditioned, LobpcgConfig, LobpcgSolver, EigenResult, GeneralizedEigenSolver, krylov_schur};

#[cfg(test)]
mod linger_integration_tests {
    use linger::{DenseVec, LinearOperator};
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix as NaCsr};

    #[test]
    fn nalgebra_csr_linear_operator_spmv() {
        // 2x2 matrix: [2 1; 0 3]
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 1, 3.0);
        let a: NaCsr<f64> = NaCsr::from(&coo);

        let x = DenseVec::from_vec(vec![1.0, 2.0]);
        let mut y = DenseVec::zeros(2);

        a.apply(&x, &mut y);

        let ys = y.as_slice();
        assert!((ys[0] - 4.0).abs() < 1e-12);
        assert!((ys[1] - 6.0).abs() < 1e-12);
    }
}
pub use ode::{
    TimeStepper, ImplicitTimeStepper,
    ImexOperator, ImexTimeStepper,
    HamiltonianSystem, VerletStepper, LeapfrogStepper, Yoshida4Stepper,
    ForwardEuler, Rk4, Rk45,
    ImplicitEuler, Sdirk2,
    Bdf2, Bdf2State,
    Newmark, NewmarkState,
    GeneralizedAlpha, GeneralizedAlphaState,
    ImexArk3,
    ImexEuler,
    ImexSsp2,
};
pub use lor::{LorPrecond, solve_pcg_lor, solve_gmres_lor};
pub mod sli;
pub use sli::{solve_jacobi_sli, solve_gs_sli};

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::{CooMatrix, CsrMatrix};

    /// 1-D Laplacian: tridiagonal [-1, 2, -1] of size n.
    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn cg_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_cg(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "CG failed to converge");
        // verify Ax ≈ b
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "residual too large: {err}");
    }

    #[test]
    fn pcg_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_jacobi(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged);
        assert!(res.iterations < 60, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn gmres_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn fgmres_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn fgmres_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_jacobi(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
        assert!(res.iterations < 60, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn idrs_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_idrs(&a, &b, &mut x, 4, &SolverConfig::default()).unwrap();
        assert!(res.converged, "IDR(s) failed to converge");
    }

    #[test]
    fn tfqmr_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_tfqmr(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "TFQMR failed to converge");
    }

    #[test]
    fn sparse_lu_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_lu(&a, &b).unwrap();
        // verify Ax ≈ b
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "LU residual too large: {err}");
    }

    #[test]
    fn sparse_cholesky_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_cholesky(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "Cholesky residual too large: {err}");
    }

    #[test]
    fn sparse_ldlt_direct() {        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_ldlt(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "LDLt residual too large: {err}");
    }

    #[test]
    fn pcg_ildlt_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_ildlt(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "PCG+ILDLt failed to converge");
    }

    #[test]
    fn gmres_ildlt_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_ildlt(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+ILDLt failed to converge");
    }
}
