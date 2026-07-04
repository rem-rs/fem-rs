//! # fem-solver
//!
//! Iterative and direct linear solvers backed by [`linlvo`].
//!
//! ## Iterative solvers
//! - [`solve_cg`]          �?Conjugate Gradient (SPD systems)
//! - [`solve_cg_operator`] �?Conjugate Gradient with operator callback (backend-agnostic)
//! - [`solve_gmres_operator`] �?GMRES with operator callback (backend-agnostic)
//! - [`solve_bicgstab_operator`] �?BiCGSTAB with operator callback (backend-agnostic)
//! - [`solve_pcg_jacobi`]  �?PCG with Jacobi preconditioner
//! - [`solve_pcg_ilu0`]    �?PCG with ILU(0) preconditioner
//! - [`solve_pcg_ildlt`]   �?PCG with ILDLᵀ preconditioner
//! - [`solve_gmres`]       �?GMRES (non-symmetric systems)
//! - [`solve_gmres_jacobi`] �?GMRES with Jacobi preconditioner
//! - [`solve_gmres_ilu0`]   �?GMRES with ILU(0) preconditioner
//! - [`solve_gmres_iluk`]   �?GMRES with ILU(k) preconditioner
//! - [`solve_gmres_ilut`]   �?GMRES with ILUT preconditioner
//! - [`solve_pcg_iluk`]     �?PCG with ILU(k) preconditioner
//! - [`solve_fgmres_ilut`]  �?FGMRES with ILUT preconditioner
//! - [`solve_precond_kind`] �?unified ILU-family dispatcher via [`PrecondKind`]
//! - [`solve_bicgstab`]    �?BiCGSTAB
//! - [`solve_idrs`]        �?IDR(s) (non-symmetric, short-recurrence)
//! - [`solve_tfqmr`]       �?TFQMR (Transpose-Free QMR)
//! - [`solve_fgmres_ilu0`] �?Flexible GMRES with ILU(0) preconditioner
//!
//! ## Generic preconditioner interface
//! - [`solve_pcg_precond`]    �?PCG with any type implementing [`linlvoPreconditioner`]
//! - [`solve_gmres_precond`]  �?GMRES with any type implementing [`linlvoPreconditioner`]
//! - [`solve_fgmres_precond`] �?FGMRES with any type implementing [`linlvoPreconditioner`]
//!
//! ## Auxiliary-space preconditioners (Hiptmair-Xu)
//! - [`solve_pcg_ams`]     �?PCG with AMS for H(curl) (Maxwell)
//! - [`solve_gmres_ams`]   �?GMRES with AMS for H(curl)
//! - [`solve_pcg_ads`]     �?PCG with ADS for H(div) (Darcy)
//! - [`solve_gmres_ads`]   �?GMRES with ADS for H(div)
//!
//! ## Direct solvers
//! - [`solve_sparse_lu`]        �?Sparse LU for general systems
//! - [`solve_sparse_cholesky`]  �?Sparse Cholesky for SPD systems
//! - [`solve_sparse_ldlt`]      �?Sparse LDLᵀ for symmetric indefinite systems
//! - [`solve_sparse_mumps`]     �?MUMPS-compatible direct path (baseline)
//! - [`solve_sparse_mkl`]       �?MKL-compatible direct path (baseline)
//!
//! All solvers operate on [`fem_linalg::CsrMatrix<T>`].

#![allow(clippy::needless_range_loop)]

use fem_linalg::CsrMatrix as FemCsr;
use linlvo::{
    core::scalar::Scalar as linlvoScalar,
    direct::{DirectSolver, SparseLu, SparseCholesky, SparseLdlt, MumpsSolver, MklSolver},
    iterative::{BiCgStab, ConjugateGradient, Fgmres, Gmres, Idrs, Tfqmr},
    precond::{AmsPrecond, AmsConfig, AdsPrecond, AdsConfig},
    sparse::CsrMatrix as linlvoCsr,
    DenseVec, Ilu0Precond, IldltPrecond, JacobiPrecond, KrylovSolver, Preconditioner,
};
use linlvo::precond::{IlukPrecond, IlutPrecond};

/// Re-export of linlvo's [`Preconditioner`] trait.
///
/// Implement this trait to plug any custom approximate-inverse into
/// [`solve_pcg_precond`], [`solve_gmres_precond`], or [`solve_fgmres_precond`]
/// without depending on the `linlvo` crate directly.
pub use linlvo::Preconditioner as linlvoPreconditioner;

#[cfg(feature = "gpu")]
pub mod cg_gpu;
#[cfg(feature = "gpu")]
pub mod gmres_gpu;

// ─── Re-export solver types from fem-linalg ───────────────────────────────────

pub use fem_linalg::{SolverConfig, SolverError, SolveResult, PrintLevel, fem_to_linlvo_csr, into_result};

// ─── Solvers ─────────────────────────────────────────────────────────────────

/// Conjugate Gradient �?for symmetric positive definite systems.
///
/// # Arguments
/// * `a`   �?system matrix (fem-rs CSR)
/// * `b`   �?right-hand side
/// * `x`   �?initial guess on entry, solution on exit
/// * `cfg` �?convergence parameters
pub fn solve_cg<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = ConjugateGradient::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
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
/// * `nrows`, `ncols` �?operator dimensions (must be square and equal to `b.len()`).
/// * `apply`          �?callback that computes `y <- A * x`.
/// * `b`              �?right-hand side.
/// * `x`              �?initial guess on entry, solution on exit.
/// * `cfg`            �?convergence parameters.
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
            return Err(SolverError::Linlvo(
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
pub fn solve_pcg_jacobi<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = JacobiPrecond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Preconditioned CG with an ILU(0) preconditioner.
///
/// Requires the matrix to have a factorisation-compatible sparsity pattern.
pub fn solve_pcg_ilu0<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = Ilu0Precond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES �?for general (possibly non-symmetric) systems.
///
/// `restart` controls the Krylov subspace dimension before restart (default 30).
pub fn solve_gmres<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES with Jacobi preconditioner.
pub fn solve_gmres_jacobi<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = JacobiPrecond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES with ILU(0) preconditioner.
pub fn solve_gmres_ilu0<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = Ilu0Precond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
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
        return Err(SolverError::Linlvo("GMRES restart must be > 0".to_string()));
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
            g[j] *= cs[j];
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
                return Err(SolverError::Linlvo(
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

/// BiCGSTAB �?for non-symmetric systems; often faster than GMRES per iteration.
pub fn solve_bicgstab<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = BiCgStab::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
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
            return Err(SolverError::Linlvo(
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
            return Err(SolverError::Linlvo(
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
            return Err(SolverError::Linlvo(
                "BiCGSTAB operator breakdown: t^T t is near zero".to_string(),
            ));
        }

        omega = dot(&t, &s) / tt;
        if omega.abs() < 1e-32 {
            return Err(SolverError::Linlvo(
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

/// Flexible GMRES �?allows a variable preconditioner per iteration.
///
/// Unlike standard GMRES, the preconditioner may change at each Krylov step
/// (e.g. inner Krylov solve, AMG V-cycle, or any nonlinear operator).
/// With a fixed preconditioner, FGMRES produces identical iterates to
/// right-preconditioned GMRES.
///
/// `restart` controls the Krylov subspace dimension before restart (default 30).
pub fn solve_fgmres<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Fgmres::<T>::new(restart);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Flexible GMRES with Jacobi preconditioner.
pub fn solve_fgmres_jacobi<T: linlvoScalar>(    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = JacobiPrecond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Fgmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Flexible GMRES with ILU(0) preconditioner.
pub fn solve_fgmres_ilu0<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = Ilu0Precond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Fgmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Generic preconditioner interface ────────────────────────────────────────

/// Preconditioned CG with a user-supplied preconditioner.
///
/// Accepts any type implementing [`linlvo::Preconditioner`].
/// Use [`linlvoPreconditioner`] / [`linlvo::Preconditioner`] as the trait bound
/// when building custom preconditioners.
///
/// # Example
/// ```ignore
/// use fem_solver::{solve_pcg_precond, SolverConfig};
/// use linlvo::JacobiPrecond;
/// use fem_solver::fem_to_linlvo_csr;
///
/// let la = fem_to_linlvo_csr(&a);
/// let prec = JacobiPrecond::from_csr(&la).unwrap();
/// let res = solve_pcg_precond(&a, &b, &mut x, &prec, &cfg).unwrap();
/// ```
pub fn solve_pcg_precond<T, P>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    precond: &P,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    T: linlvoScalar,
    P: Preconditioner<Vector = DenseVec<T>>,
{
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(precond as &dyn Preconditioner<Vector = DenseVec<T>>), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES with a user-supplied preconditioner.
///
/// The preconditioner type is erased at the call-site via `&dyn Preconditioner`,
/// so there is no per-preconditioner boilerplate.
pub fn solve_gmres_precond<T, P>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    precond: &P,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    T: linlvoScalar,
    P: Preconditioner<Vector = DenseVec<T>>,
{
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = Gmres::<T>::new(restart)
        .solve(&la, Some(precond as &dyn Preconditioner<Vector = DenseVec<T>>), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Flexible GMRES with a user-supplied (potentially variable) preconditioner.
///
/// Unlike standard GMRES, FGMRES tolerates preconditioners that change between
/// iterations �?inner Krylov solves, AMG V-cycles, and nonlinear operators all
/// qualify.  With a fixed preconditioner the iterates are identical to
/// right-preconditioned GMRES.
pub fn solve_fgmres_precond<T, P>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    precond: &P,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    T: linlvoScalar,
    P: Preconditioner<Vector = DenseVec<T>>,
{
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = Fgmres::<T>::new(restart)
        .solve(&la, Some(precond as &dyn Preconditioner<Vector = DenseVec<T>>), &lb, &mut lx, &cfg.to_linlvo())
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
pub fn solve_pcg_ildlt<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IldltPrecond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES with incomplete LDLᵀ preconditioner for symmetric indefinite systems.
///
/// Ideal for saddle-point problems (Stokes, Maxwell) where Cholesky/ILU fail.
pub fn solve_gmres_ildlt<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IldltPrecond::from_csr(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── ILU family (Phase 6) ────────────────────────────────────────────────────

/// ILU family preconditioner selector.
///
/// Pass one of these variants to [`solve_precond_kind`] to choose the
/// incomplete-factorisation strategy without changing the calling code.
///
/// | Variant | Fill strategy | Typical use |
/// |---------|---------------|-------------|
/// | `Ilu0`  | Sparsity of `A` | Cheap, SPD or diagonally dominant |
/// | `Iluk(k)` | Level-of-fill �?k | Better quality for moderate fill |
/// | `Ilut { tau, fill }` | Drop tolerance + fill bound | Non-symmetric, harder systems |
#[derive(Debug, Clone, Default)]
pub enum PrecondKind {
    /// ILU(0): no extra fill (fastest build, lowest quality).
    #[default]
    Ilu0,
    /// ILU(k): allow fill-in entries up to level `k`.
    /// `k = 0` equals ILU(0); larger `k` approaches exact LU.
    Iluk(usize),
    /// ILUT(τ, p): drop entries smaller than `tau × ‖row‖₂`;
    /// keep at most `fill` off-diagonal entries per row in L and U.
    Ilut {
        /// Relative drop tolerance (e.g. `0.01`).
        tau:  f64,
        /// Max off-diagonal fill per row in each factor.
        fill: usize,
    },
}

/// GMRES with ILU(k) preconditioner.
///
/// `fill_level = 0` reproduces ILU(0); increase for harder problems.
pub fn solve_gmres_iluk<T: linlvoScalar>(
    a:          &FemCsr<T>,
    b:          &[T],
    x:          &mut [T],
    restart:    usize,
    fill_level: usize,
    cfg:        &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IlukPrecond::from_csr(&la, fill_level)
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// GMRES with ILUT(τ, p) preconditioner.
///
/// * `tau`    �?relative drop tolerance (0.0 = keep all, 0.01 = aggressive)
/// * `p_fill` �?max off-diagonal fill per row in L and U
pub fn solve_gmres_ilut<T: linlvoScalar>(
    a:      &FemCsr<T>,
    b:      &[T],
    x:      &mut [T],
    restart: usize,
    tau:    f64,
    p_fill: usize,
    cfg:    &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IlutPrecond::from_csr(&la, tau, p_fill)
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// PCG with ILU(k) preconditioner (symmetric positive definite systems).
///
/// `fill_level = 0` reproduces `solve_pcg_ilu0`.
pub fn solve_pcg_iluk<T: linlvoScalar>(
    a:          &FemCsr<T>,
    b:          &[T],
    x:          &mut [T],
    fill_level: usize,
    cfg:        &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IlukPrecond::from_csr(&la, fill_level)
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Flexible GMRES (FGMRES) with ILUT preconditioner.
///
/// FGMRES tolerates a variable preconditioner; useful when the inner ILUT
/// solve is itself iterative or when the preconditioner changes between steps.
pub fn solve_fgmres_ilut<T: linlvoScalar>(
    a:      &FemCsr<T>,
    b:      &[T],
    x:      &mut [T],
    restart: usize,
    tau:    f64,
    p_fill: usize,
    cfg:    &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = IlutPrecond::from_csr(&la, tau, p_fill)
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let solver = Fgmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Unified ILU-family GMRES dispatcher.
///
/// Selects the preconditioner at runtime from a [`PrecondKind`] value.
/// Useful when the choice of preconditioner should be a configuration
/// parameter rather than a compile-time decision.
///
/// # Example
/// ```rust,ignore
/// use fem_solver::{solve_precond_kind, PrecondKind, SolverConfig};
///
/// let res = solve_precond_kind(&a, &b, &mut x, 30,
///     PrecondKind::Ilut { tau: 0.01, fill: 20 },
///     &SolverConfig::default())?;
/// ```
pub fn solve_precond_kind<T: linlvoScalar>(
    a:       &FemCsr<T>,
    b:       &[T],
    x:       &mut [T],
    restart: usize,
    kind:    PrecondKind,
    cfg:     &SolverConfig,
) -> Result<SolveResult, SolverError> {
    match kind {
        PrecondKind::Ilu0             => solve_gmres_ilu0(a, b, x, restart, cfg),
        PrecondKind::Iluk(k)          => solve_gmres_iluk(a, b, x, restart, k, cfg),
        PrecondKind::Ilut { tau, fill } => solve_gmres_ilut(a, b, x, restart, tau, fill, cfg),
    }
}

// ─── IDR(s) ──────────────────────────────────────────────────────────────────

/// IDR(s) �?Induced Dimension Reduction for non-symmetric systems.
///
/// Short-recurrence method; s=4 is a good default.  Typically fewer matvecs
/// than BiCGSTAB for difficult non-symmetric problems.
pub fn solve_idrs<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    s: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let solver = Idrs::<T>::new(s);
    let res = solver
        .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// TFQMR �?Transpose-Free Quasi-Minimal Residual for non-symmetric systems.
///
/// Does not require the transpose of A; converges smoothly on problems where
/// BiCGSTAB may stagnate.
pub fn solve_tfqmr<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let res = Tfqmr::<T>::default()
        .solve(&la, None, &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

// ─── Direct solvers ───────────────────────────────────────────────────────────

/// Sparse LU direct solver for general square systems.
///
/// Exact solve (up to floating-point precision).  Use for small-to-medium
/// problems or as a reference/preconditioner.
pub fn solve_sparse_lu<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = SparseLu::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    Ok(lx.into_vec())
}

/// Sparse Cholesky direct solver for symmetric positive-definite systems.
///
/// About 2× faster than LU for SPD matrices.
pub fn solve_sparse_cholesky<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = SparseCholesky::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    Ok(lx.into_vec())
}

/// Sparse LDLᵀ direct solver for symmetric indefinite systems (Stokes, Maxwell saddle-point).
pub fn solve_sparse_ldlt<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = SparseLdlt::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    Ok(lx.into_vec())
}

/// MUMPS-compatible direct solver baseline.
///
/// Uses `linlvo::direct::MumpsSolver`, which currently provides a stable
/// factor/solve/reuse API backed by linlvo's native multifrontal replacement
/// path rather than an external MUMPS dependency.
pub fn solve_sparse_mumps<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = MumpsSolver::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    Ok(lx.into_vec())
}

/// MKL-compatible direct solver baseline.
///
/// Uses `linlvo::direct::MklSolver`, which currently provides a stable
/// factor/solve/reuse API backed by linlvo's native multifrontal replacement
/// path rather than an external MKL dependency.
pub fn solve_sparse_mkl<T: linlvoScalar>(
    a: &FemCsr<T>,
    b: &[T],
) -> Result<Vec<T>, SolverError> {
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::zeros(b.len());
    let mut solver = MklSolver::<T>::default();
    solver.factor(&la).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    solver.solve(&lb, &mut lx).map_err(|e| SolverError::Linlvo(e.to_string()))?;
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
/// * `a`       �?H(curl) stiffness matrix (edge DOFs)
/// * `g`       �?Discrete gradient matrix (vertices -> edges)
/// * `b`       �?right-hand side
/// * `x`       �?initial guess on entry, solution on exit
/// * `cfg`     �?solver configuration
///
/// # Type parameters
/// The discrete gradient `g` is passed as a linlvo CsrMatrix to match internal types.
/// Convert using `fem_to_linlvo_csr`.
pub fn solve_pcg_ams<T: linlvoScalar>(
    a: &FemCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AmsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ams = AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&ams), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve an H(curl) system using GMRES with AMS preconditioner.
///
/// Use this for non-symmetric H(curl) problems (e.g., with absorbing BCs).
pub fn solve_gmres_ams<T: linlvoScalar>(
    a: &FemCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &AmsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ams = AmsPrecond::<T>::new(&la, g, cfg.ams_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&ams), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
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
/// * `a`       �?H(div) stiffness matrix (face DOFs)
/// * `c`       �?Discrete curl matrix (edges -> faces)
/// * `g`       �?Discrete gradient matrix (vertices -> edges)
/// * `b`       �?right-hand side
/// * `x`       �?initial guess on entry, solution on exit
/// * `cfg`     �?solver configuration
///
/// # Notes
/// Both `c` and `g` should be converted to linlvo format using `fem_to_linlvo_csr`.
pub fn solve_pcg_ads<T: linlvoScalar>(
    a: &FemCsr<T>,
    c: &linlvoCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    cfg: &AdsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let res = ConjugateGradient::<T>::default()
        .solve(&la, Some(&ads), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
        .map_err(SolverError::from)?;

    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

/// Solve an H(div) system using GMRES with ADS preconditioner.
///
/// Use this for non-symmetric H(div) problems.
pub fn solve_gmres_ads<T: linlvoScalar>(
    a: &FemCsr<T>,
    c: &linlvoCsr<T>,
    g: &linlvoCsr<T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    cfg: &AdsSolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    let ads = AdsPrecond::<T>::new(&la, c, g, cfg.ads_cfg.clone())
        .map_err(|e| SolverError::Linlvo(e.to_string()))?;

    let solver = Gmres::<T>::new(restart);
    let res = solver
        .solve(&la, Some(&ads), &lb, &mut lx, &cfg.inner_cfg.to_linlvo())
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

// ── MINRES (symmetric indefinite) ──────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// MINRES — for symmetric (possibly indefinite) linear systems.
///
/// Minimises ‖b − Ax‖₂ using Lanczos tridiagonalisation with Givens QR.
/// Storage grows as `O(n·k)` where `k ≤ cfg.max_iter` (no explicit restart).
/// For indefinite systems (e.g. Helmholtz shift, Stokes saddle-point) MINRES
/// is a robust alternative to CG (which requires SPD).
pub fn solve_minres(
    a: &FemCsr<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows;
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch { rows: n, cols: n, rhs: b.len() });
    }
    solve_minres_impl(n, |z, w| a.spmv(z, w), b, x, cfg)
}

/// MINRES using a backend-agnostic operator callback.
///
/// This entrypoint is intended for matrix-free or foreign-backend operators
/// that can provide `y = A*x` without exposing a concrete CSR matrix.
pub fn solve_minres_operator<F>(
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
        return Err(SolverError::DimensionMismatch { rows: nrows, cols: ncols, rhs: b.len() });
    }
    solve_minres_impl(nrows, apply, b, x, cfg)
}

/// Shared core: Lanczos tridiagonalisation + Givens QR.
///
/// Algorithm: Paige & Saunders (1975); vector–w update variant stores
/// two w-vectors and updates x incrementally, **not** all Lanczos vectors.
fn solve_minres_impl<F>(
    n: usize,
    apply: F,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    // r0 = b − A x0
    let mut ax = vec![0.0; n];
    apply(x, &mut ax);
    let mut r = vec![0.0; n];
    for i in 0..n { r[i] = b[i] - ax[i]; }

    let norm_b = norm(b);
    let tol = cfg.atol.max(cfg.rtol * norm_b.max(1e-32));
    let mut res_norm = norm(&r);
    if res_norm <= tol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: res_norm });
    }

    // ── Lanczos vectors V (V_0..V_{k+1}) ──────────────────────────────
    // V[0] = v_0 = 0 (placeholder)
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(cfg.max_iter + 2);
    v.push(vec![0.0; n]);                        // v₀ = 0
    {
        let mut v1 = vec![0.0; n];
        let inv = 1.0 / res_norm;
        for i in 0..n { v1[i] = r[i] * inv; }
        v.push(v1);                              // v₁
    }

    // Tricliagonal coefficients: α[1..k], β[0..k]  (β[0] = 0)
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta:  Vec<f64> = vec![0.0];          // β₀

    // QR factorisation of T̃_k:
    // R entries (three relevant diagonals of the upper-triangular factor)
    // R_{k-2,k}, R_{k-1,k}, R_{k,k} for each column k
    let mut r_sup2: Vec<f64> = Vec::new();  // R_{k-2,k}
    let mut r_sup1: Vec<f64> = Vec::new();  // R_{k-1,k}
    let mut r_diag: Vec<f64> = Vec::new();  // R_{k,k} = ρ_k

    // Givens rotation history (need TWO previous pairs for the tridiagonal QR)
    // rotation (k-2): cs_older, sn_older
    // rotation (k-1): cs_old,   sn_old
    let mut cs_old   = 1.0_f64;  // cos(θ_{k-1}) — identity when k=1
    let mut sn_old   = 0.0_f64;  // sin(θ_{k-1})
    let mut cs_older = 1.0_f64;  // cos(θ_{k-2}) — identity when k≤2
    let mut sn_older = 0.0_f64;  // sin(θ_{k-2})

    // Transformed RHS: g = Q_k^T · β₁·e₁
    // g[k-1] is the transformed entry, g[k] tracks ‖r_k‖
    let mut g = vec![res_norm];  // g[0] = β₁ initially

    for iter in 1..=cfg.max_iter {
        // ── Lanczos step ────────────────────────────────────────────────
        let mut av = vec![0.0; n];
        apply(&v[iter], &mut av);                  // A · v_iter

        let ak = dot(&v[iter], &av);                // α_k
        alpha.push(ak);

        // v_tilde = A·v_k − α_k·v_k − β_{k-1}·v_{k-1}
        for i in 0..n {
            r[i] = av[i] - ak * v[iter][i] - beta[iter - 1] * v[iter - 1][i];
        }
        let bk = norm(&r);                           // β_k
        beta.push(bk);

        if bk > 1e-32 {
            let inv = 1.0 / bk;
            let mut v_next = vec![0.0; n];
            for i in 0..n { v_next[i] = r[i] * inv; }
            v.push(v_next);                          // v_{k+1}
        } else {
            v.push(vec![0.0; n]);                    // Lanczos terminated
        }

        // ── Apply Givens QR to column k of T̃_k ────────────────────────
        // Column k has three non-zero entries:
        //   row k-1: β_{k-1}   (sub-diagonal)
        //   row k:   α_k       (diagonal)
        //   row k+1: β_k       (new sub-diagonal)
        //
        // Two previous rotations affect this column:
        //   G_{k-2,k-1} (θ_{k-2}) on rows (k-2, k-1): transforms β_{k-1}
        //   G_{k-1,k}   (θ_{k-1}) on rows (k-1, k):   transforms (result, α_k)
        // Then G_{k,k+1} (θ_k) on rows (k, k+1): zeros β_k, producing R_{k,k}.

        // Apply rotation k-2 (G_{k-2,k-1}) to (0, β_{k-1}) at rows (k-2, k-1):
        //   row k-2' = cs_older·0 + sn_older·β_{k-1} = sn_older·β  → R_{k-2,k}
        //   row k-1' = -sn_older·0 + cs_older·β_{k-1} = cs_older·β  → passed to rot k-1
        let zeta_sup2 = sn_older * beta[iter - 1];
        let zeta_sub  = cs_older * beta[iter - 1];
        r_sup2.push(zeta_sup2);  // R_{k-2,k}

        // Apply rotation k-1 (G_{k-1,k}) to (zeta_sub, α_k):
        //   R_{k-1,k}  = cs_old·zeta_sub + sn_old·α_k
        //   diag_in    = -sn_old·zeta_sub + cs_old·α_k  → passed to new rotation
        let zeta_sup1 = cs_old * zeta_sub + sn_old * ak;
        let diag_in   = -sn_old * zeta_sub + cs_old * ak;
        r_sup1.push(zeta_sup1);  // R_{k-1,k}

        // ── New Givens rotation k (G_{k,k+1}) on (diag_in, β_k) ────────
        let rk = (diag_in * diag_in + bk * bk).sqrt();
        let (csk, snk) = if rk > 1e-32 {
            (diag_in / rk, bk / rk)
        } else {
            (1.0, 0.0)
        };

        r_diag.push(rk);         // R_{k,k} = ρ_k

        // ── Update g (transformed RHS) ──────────────────────────────────
        // Apply the new rotation to (g[iter-1], 0):
        //   g[iter-1] ← csk * g[iter-1]
        //   g[iter]   ← -snk * g[iter-1]
        let g_old = g[iter - 1];
        g[iter - 1] = csk * g_old;
        g.push(-snk * g_old);

        // ‖r_k‖ = |g_{k+1}| = |g[iter]|
        res_norm = g[iter].abs();
        if res_norm <= tol {
            // ── Solve R_k y = g_{1:k} and compute x = V_k y ────────────
            let k = iter;
            let mut y = vec![0.0; k];
            // Back-substitution: R y = g_{1:k}
            // R is k×k upper triangular with up to 2 super-diagonals.
            for i in (0..k).rev() {
                let mut s = g[i];
                // R_{i,i+1} * y_{i+1}
                if i + 1 < k { s -= r_sup1[i + 1] * y[i + 1]; }
                // R_{i,i+2} * y_{i+2}
                if i + 2 < k { s -= r_sup2[i + 2] * y[i + 2]; }
                y[i] = s / r_diag[i];
            }
            // x = V_k y = sum_{j=1}^{k} y_j * v_j
            for i in 0..n { x[i] = 0.0; }
            for j in 0..k {
                for i in 0..n { x[i] += y[j] * v[j + 1][i]; }
            }
            return Ok(SolveResult { converged: true, iterations: iter, final_residual: res_norm });
        }

        // ── Shift Givens history for next iteration ────────────────────
        cs_older = cs_old;
        sn_older = sn_old;
        cs_old = csk;
        sn_old = snk;
    }

    Err(SolverError::ConvergenceFailed { max_iter: cfg.max_iter, residual: res_norm })
}

// ── GCR (generalised conjugate residual, restart) ──────────────────────────────

/// GCR(m) — for general (possibly non-symmetric) linear systems.
///
/// Generalised Conjugate Residual with restart dimension `m`.  Like GMRES it
/// minimises ‖b − Ax‖₂ over the Krylov subspace, but stores the search
/// directions explicitly (p-vectors) for orthogonalisation.
///
/// **Note:** For symmetric positive-definite systems, use CG or MINRES instead.
/// GCR's A-orthogonalisation can lose numerical orthogonality on SPD matrices,
/// leading to slow convergence.  It is primarily intended for non-symmetric
/// problems where GMRES is a valid alternative.
///
/// Storage: `O(n·m)`; choose `m` to balance memory and convergence speed.
pub fn solve_gcr(
    a: &FemCsr<f64>,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows;
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch { rows: n, cols: n, rhs: b.len() });
    }
    if restart == 0 {
        return Err(SolverError::Linlvo("GCR restart must be > 0".into()));
    }
    solve_gcr_impl(n, |z, w| a.spmv(z, w), b, x, restart, cfg)
}

/// GCR(m) using a backend-agnostic operator callback.
pub fn solve_gcr_operator<F>(
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
        return Err(SolverError::DimensionMismatch { rows: nrows, cols: ncols, rhs: b.len() });
    }
    if restart == 0 {
        return Err(SolverError::Linlvo("GCR restart must be > 0".into()));
    }
    solve_gcr_impl(nrows, apply, b, x, restart, cfg)
}

/// Shared core: restarted GCR.
fn solve_gcr_impl<F>(
    n: usize,
    apply: F,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    // Standard GCR algorithm (Eisenstat, Elman & Schultz 1983):
    //
    //   r₀ = b − Ax₀
    //   for j = 0, 1, … (outer / restart loop):
    //     for i = 0, …, m−1 (inner build):
    //       pⁱ = rⁱ … or pⁱ = rⁱ − Σ β·pˢ (full orthogonalisation)
    //       Apⁱ = A·pⁱ
    //       αⁱ = (rⁱ, Apⁱ) / (Apⁱ, Apⁱ)
    //       x += αⁱ pⁱ
    //       r^{i+1} = rⁱ − αⁱ Apⁱ
    //       if ‖r^{i+1}‖ ≤ tol: done
    //     r⁰ = r^{m}; loop

    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    apply(x, &mut ax);
    for i in 0..n { r[i] = b[i] - ax[i]; }

    let norm_b = norm(b);
    let tol = cfg.atol.max(cfg.rtol * norm_b.max(1e-32));
    let mut res_norm = norm(&r);
    if res_norm <= tol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: res_norm });
    }

    let mut total_iters = 0usize;
    let r_work = &mut r;

    loop {
        // Storage for one cycle
        let m = restart;
        let mut pp: Vec<Vec<f64>> = Vec::with_capacity(m);   // search directions
        let mut ap: Vec<Vec<f64>> = Vec::with_capacity(m);   // A·p

        // ── Inner Krylov construction ────────────────────────────────────
        for _inner in 0..m {
            if total_iters >= cfg.max_iter { break; }

            // p = r — full orthogonalisation against previous p's
            let mut p = r_work.clone();
            for j in 0..pp.len() {
                let beta = -dot(&p, &ap[j]) / dot(&ap[j], &ap[j]);
                if beta.is_finite() {
                    for i in 0..n { p[i] += beta * pp[j][i]; }
                }
            }

            // Ap = A·p
            let mut apj = vec![0.0; n];
            apply(&p, &mut apj);

            // α = (r, Ap) / (Ap, Ap)
            let ap_sq = dot(&apj, &apj);
            if ap_sq < 1e-32 {
                // Ap ≈ 0 → invariant subspace; skip this direction
                continue;
            }
            let alpha = dot(r_work, &apj) / ap_sq;

            // x += α·p
            for i in 0..n { x[i] += alpha * p[i]; }
            // r = r − α·Ap
            for i in 0..n { r_work[i] -= alpha * apj[i]; }

            pp.push(p);
            ap.push(apj);

            total_iters += 1;
            res_norm = norm(r_work);
            if res_norm <= tol {
                return Ok(SolveResult { converged: true, iterations: total_iters, final_residual: res_norm });
            }
        }

        if total_iters >= cfg.max_iter { break; }

        // ── Restart: r = b − A·x (to avoid drift from rounding) ──────────
        apply(x, &mut ax);
        for i in 0..n { r_work[i] = b[i] - ax[i]; }
        res_norm = norm(r_work);
        if res_norm <= tol {
            return Ok(SolveResult { converged: true, iterations: total_iters, final_residual: res_norm });
        }
    }

    Err(SolverError::ConvergenceFailed { max_iter: cfg.max_iter, residual: res_norm })
}


pub mod block;
pub mod block_gmres;
pub mod block_operator;
pub mod eigen;
pub mod hypre;
pub mod lor;
pub mod p_multigrid;
pub mod multirate;
pub mod multiphysics_sync;
pub mod multiphysics;
pub mod multiphysics_templates;
pub mod ode;
pub mod rom;
pub mod sdc;
pub mod butcher;
pub mod adaptive;
pub mod adjoint;
pub mod bdf;
pub mod complex_ams;
pub mod dae;
pub mod events;
pub use block_operator::{
    BlockOperator,
    BlockOpMatrix,
    SumBlockOp,
    MultiphysicsOperator,
    BlockNonlinearForm,
    BlockSolver,
    BlockDiagonalPrecondN,
    BlockTriangularPrecondN,
    build_jacobi_block_solvers,
    extract_upper_coupling,
    solve_block_precond_gmres,
    right_preconditioned_gmres,
};
pub use block::{BlockSystem, BlockDiagonalPrecond, BlockTriangularPrecond, SchurComplementSolver, MinresSolver};
pub use block_gmres::{solve_block_gmres, BlockGmresConfig};
pub use eigen::{
    lobpcg, lobpcg_constrained, lobpcg_constrained_preconditioned,
    LobpcgConfig, LobpcgSolver, EigenResult, GeneralizedEigenSolver, krylov_schur,
    arpack, WhichEigenvalue,
    feast_interval, IntervalEigenConfig,
};
pub use multiphysics::{
    CoupledProblem,
    CoupledLinearStrategy,
    CoupledNewtonConfig,
    CoupledNewtonResult,
    CoupledNewtonSolver,
    CoupledSolveError,
};
pub use multiphysics_sync::{
    RelativeL2Tracker,
    RelativeScalarTracker,
    TemplateSyncPolicy,
    compose_sync_error,
    compose_weighted_sync_error,
};
pub use multiphysics_templates::{
    BuiltinMultiphysicsTemplate,
    MultiphysicsTemplateNode,
    MultiphysicsTemplateSpec,
    TemplateCouplingStyle,
    TemplateRuntimeConfig,
    builtin_template_catalog,
    builtin_template_spec,
};
pub use multirate::{
    MultiRateAdaptiveConfig,
    MultiRateConfig,
    MultiRateError,
    MultiRateStats,
    run_multirate,
    run_multirate_adaptive,
};

#[cfg(test)]
mod linlvo_integration_tests {
    use fem_linalg::CsrMatrix;

    #[test]
    fn nalgebra_csr_linear_operator_spmv() {
        // 2x2 matrix: [2 1; 0 3]
        let mut coo = fem_linalg::CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(0, 1, 1.0);
        coo.add(1, 1, 3.0);
        let a: CsrMatrix<f64> = coo.into_csr();

        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 2];

        a.spmv(&x, &mut y);

        assert!((y[0] - 4.0).abs() < 1e-12);
        assert!((y[1] - 6.0).abs() < 1e-12);
    }
}
pub use ode::{
    TimeStepper, ImplicitTimeStepper,
    ImexOperator, ImexTimeStepper,
    HamiltonianSystem, VerletStepper, LeapfrogStepper, Yoshida4Stepper,
    ForwardEuler, Rk4, Rk45,
    ImplicitEuler, Sdirk2,
    CrankNicolson,
    AdamsBashforthMoulton, AbmState,
    Bdf2, Bdf2State,
    Newmark, NewmarkState,
    GeneralizedAlpha, GeneralizedAlphaState,
    ImexArk3,
    ImexRk3,
    ImexEuler,
    ImexSsp2,
};
pub use butcher::{
    ButcherTableau, ImexTableau,
    forward_euler_tableau, backward_euler_tableau,
    explicit_midpoint_tableau, heun_tableau,
    rk4_tableau, dopri5_tableau, fehlberg12_tableau, bs32_tableau, ck54_tableau,
    implicit_midpoint_tableau,
    sdirk2_tableau, sdirk3_tableau, sdirk4_tableau,
    imex_euler_tableau, imex_ssp2_tableau, ark3_tableau, ark5_tableau,
    wrms_error, pi_step_controller, i_step_controller,
};
pub use adaptive::{
    AdaptiveConfig, IntegratorStats, StepperState,
    integrate_adaptive, explicit_adaptive_step,
};
pub use adjoint::{
    AdjointProblem, adjoint_sensitivity,
};
pub use bdf::{
    NordsieckState, BdfIntegrator, BdfConfig, BdfStats,
    NewtonConfig,
};
pub use dae::{
    DaeState, DaeIntegrator, DaeConfig, DaeNewtonConfig,
    dae_consistent_initialization,
};
pub use events::{
    EventFunction, EventInfo,
    integrate_with_events,
};
pub use hypre::HypreBoomerAMG;
#[allow(deprecated)]
pub use hypre::{
    HypreParMatrix, HyprePrecond,
    hypre_solve_pcg, hypre_solve_gmres,
};
pub use lor::{
    LorPrecond, solve_pcg_lor, solve_gmres_lor,
    LorAmgPrecond, build_lor_operator, AmgConfig,
    solve_pcg_lor_amg, solve_gmres_lor_amg,
    GeomMGHierarchy, GeomMGPrecond, solve_vcycle_geom_mg,
};
pub use p_multigrid::{
    PmgHierarchy, PmgPrecond, solve_vcycle_pmg, fmg_solve, build_pmg_hierarchy_1d_laplacian,
};
pub mod mixed_precision;
pub mod sli;
pub mod stokes_precond;
pub mod active_set;
pub use sli::{solve_jacobi_sli, solve_gs_sli};
pub use stokes_precond::{
    StokesPrecond,
    build_pressure_mass, build_bfbt_schur,
};
pub use mixed_precision::{
    convert_csr_f64_to_f32, convert_csr_f32_to_f64,
    MixedPrecisionPrecond,
    solve_pcg_mixed, solve_cg_f32, solve_gmres_f32,
};
pub use rom::{
    Snapshots, PodBasis,
    project_system, reconstruct, relative_error,
};
pub use sdc::{
    SdcConfig, SdcIntegrator,
};

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

    /// Mildly non-symmetric 1-D convection-diffusion-like operator.
    fn nonsymmetric_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0);
            if i > 0 {
                coo.add(i, i - 1, -1.2);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -0.4);
            }
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
        // verify Ax �?b
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
    fn gmres_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_jacobi(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+Jacobi failed to converge");
        assert!(res.iterations < 60, "too many iterations: {}", res.iterations);
    }

    #[test]
    fn gmres_ilu0_nonsymmetric() {
        let n = 60;
        let a = nonsymmetric_1d(n);
        let b = vec![1.0_f64; n];
        let mut x_plain = vec![0.0_f64; n];
        let mut x_ilu = vec![0.0_f64; n];
        let plain = solve_gmres(&a, &b, &mut x_plain, 30, &SolverConfig::default()).unwrap();
        let ilu = solve_gmres_ilu0(&a, &b, &mut x_ilu, 30, &SolverConfig::default()).unwrap();
        assert!(plain.converged, "plain GMRES failed to converge");
        assert!(ilu.converged, "GMRES+ILU0 failed to converge");
        assert!(ilu.iterations <= plain.iterations,
            "GMRES+ILU0 should not need more iterations: plain={} ilu={}",
            plain.iterations, ilu.iterations);
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
    fn fgmres_ilu0_nonsymmetric() {
        let n = 60;
        let a = nonsymmetric_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_ilu0(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged, "FGMRES+ILU0 failed to converge");
    }

    // ── Generic preconditioner interface tests ────────────────────────────────

    #[test]
    fn solve_pcg_precond_jacobi() {
        // Verify the generic PCG wrapper produces the same result as solve_pcg_jacobi.
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x1 = vec![0.0_f64; n];
        let mut x2 = vec![0.0_f64; n];

        let prec = JacobiPrecond::from_csr(&fem_to_linlvo_csr(&a)).unwrap();
        let r1 = solve_pcg_precond(&a, &b, &mut x1, &prec, &SolverConfig::default()).unwrap();
        let r2 = solve_pcg_jacobi(&a, &b, &mut x2, &SolverConfig::default()).unwrap();
        assert!(r1.converged);
        assert_eq!(r1.iterations, r2.iterations);
    }

    #[test]
    fn solve_gmres_precond_ilu0() {
        let n = 60;
        let a = nonsymmetric_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let la = fem_to_linlvo_csr(&a);
        let prec = Ilu0Precond::from_csr(&la).unwrap();
        let res = solve_gmres_precond(&a, &b, &mut x, 30, &prec, &SolverConfig::default()).unwrap();
        assert!(res.converged, "generic GMRES+ILU0 failed: residual={}", res.final_residual);
    }

    #[test]
    fn solve_fgmres_precond_ildlt() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let la = fem_to_linlvo_csr(&a);
        let prec = IldltPrecond::from_csr(&la).unwrap();
        let res = solve_fgmres_precond(&a, &b, &mut x, 30, &prec, &SolverConfig::default()).unwrap();
        assert!(res.converged, "generic FGMRES+ILDLt failed: residual={}", res.final_residual);
    }

    // ── Phase 6: ILU(k) / ILUT tests ─────────────────────────────────────────

    #[test]
    fn solve_gmres_iluk0_equals_ilu0() {
        // ILU(0) and ILU(k=0) should give the same iteration count on a
        // symmetric tridiagonal (fill level 0 = no extra fill = ILU0).
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x0 = vec![0.0_f64; n];
        let mut xk = vec![0.0_f64; n];
        let cfg = SolverConfig::default();
        let r0 = solve_gmres_ilu0(&a, &b, &mut x0, 30, &cfg).unwrap();
        let rk = solve_gmres_iluk(&a, &b, &mut xk, 30, 0, &cfg).unwrap();
        assert!(r0.converged, "ILU0 did not converge");
        assert!(rk.converged, "ILU(k=0) did not converge");
    }

    #[test]
    fn solve_gmres_iluk1_converges() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_iluk(&a, &b, &mut x, 30, 1, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+ILU(1) failed: res={}", res.final_residual);
    }

    #[test]
    fn solve_gmres_iluk2_fewer_iters_than_ilu0() {
        // ILU(2) should need no more iterations than ILU(0) on Laplacian.
        let n = 80;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x0 = vec![0.0_f64; n];
        let mut x2 = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 2000, ..Default::default() };
        let r0 = solve_gmres_ilu0(&a, &b, &mut x0, 30, &cfg).unwrap();
        let r2 = solve_gmres_iluk(&a, &b, &mut x2, 30, 2, &cfg).unwrap();
        assert!(r0.converged && r2.converged);
        assert!(r2.iterations <= r0.iterations,
            "ILU(2) used more iterations ({}) than ILU(0) ({})",
            r2.iterations, r0.iterations);
    }

    #[test]
    fn solve_gmres_ilut_converges_spd() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_ilut(&a, &b, &mut x, 30, 0.01, 10, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+ILUT failed: res={}", res.final_residual);
    }

    #[test]
    fn solve_gmres_ilut_nonsym_converges() {
        // Non-symmetric banded: A[i,i]=3, A[i,i-1]=-1, A[i,i+1]=-2.
        let n = 50;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0_f64);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i + 1 < n { coo.add(i, i + 1, -2.0); }
        }
        let a = coo.into_csr();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_ilut(&a, &b, &mut x, 30, 1e-3, 15, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+ILUT (nonsym) failed: res={}", res.final_residual);
    }

    #[test]
    fn solve_pcg_iluk_converges() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_iluk(&a, &b, &mut x, 1, &SolverConfig::default()).unwrap();
        assert!(res.converged, "PCG+ILU(1) failed: res={}", res.final_residual);
    }

    #[test]
    fn solve_fgmres_ilut_converges() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_ilut(&a, &b, &mut x, 30, 0.01, 10, &SolverConfig::default()).unwrap();
        assert!(res.converged, "FGMRES+ILUT failed: res={}", res.final_residual);
    }

    #[test]
    fn solve_precond_kind_ilu0_matches_direct() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x1 = vec![0.0_f64; n];
        let mut x2 = vec![0.0_f64; n];
        let cfg = SolverConfig::default();
        solve_gmres_ilu0(&a, &b, &mut x1, 30, &cfg).unwrap();
        solve_precond_kind(&a, &b, &mut x2, 30, PrecondKind::Ilu0, &cfg).unwrap();
        for i in 0..n {
            assert!((x1[i] - x2[i]).abs() < 1e-12, "node {i} differs");
        }
    }

    #[test]
    fn solve_precond_kind_iluk_converges() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_precond_kind(&a, &b, &mut x, 30, PrecondKind::Iluk(1), &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn solve_precond_kind_ilut_converges() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let kind = PrecondKind::Ilut { tau: 0.01, fill: 10 };
        let res = solve_precond_kind(&a, &b, &mut x, 30, kind, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn ilut_solution_matches_iluk_on_spd() {
        // Both ILUT and ILU(k) should give the same (correct) solution on SPD.
        let n = 30;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut xt = vec![0.0_f64; n];
        let mut xk = vec![0.0_f64; n];
        solve_gmres_ilut(&a, &b, &mut xt, 30, 1e-12, 30, &SolverConfig { rtol: 1e-10, ..Default::default() }).unwrap();
        solve_gmres_iluk(&a, &b, &mut xk, 30, 2, &SolverConfig { rtol: 1e-10, ..Default::default() }).unwrap();
        for i in 0..n {
            assert!((xt[i] - xk[i]).abs() < 1e-8, "node {i}: ilut={:.3e} iluk={:.3e}", xt[i], xk[i]);
        }
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

    // ── MINRES tests ─────────────────────────────────────────────────────────

    /// Symmetric indefinite matrix: A = diag(-1, 1, -1, 1, …, 50×50).
    /// MINRES must converge; CG would break down.
    fn symmetric_indefinite(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            let val = if i % 2 == 0 { -1.0_f64 } else { 2.0_f64 };
            coo.add(i, i, val);
            if i > 0     { coo.add(i, i - 1, 0.5); }
            if i < n - 1 { coo.add(i, i + 1, 0.5); }
        }
        coo.into_csr()
    }

    #[test]
    fn minres_laplacian_spd_debug() {
        // Debug: 3×3 Laplacian with verification at each step
        let n = 3;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-12, max_iter: 100, verbose: false, ..Default::default() };
        let res = solve_minres(&a, &b, &mut x, &cfg).unwrap();
        eprintln!("n=3 MINRES: converged={} iters={} residual={:.6e}", res.converged, res.iterations, res.final_residual);
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        for i in 0..n { eprintln!("  x[{}] = {:.10e}  (Ax-b)[{}] = {:.6e}", i, x[i], i, (ax[i]-b[i]).abs()); }
        assert!(res.converged, "MINRES 3×3 failed");
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        eprintln!("n=3 MINRES ‖Ax−b‖ = {:.6e}", err);
        assert!(err < 1e-10, "MINRES 3×3 residual too large: {err}");
    }

    #[test]
    fn minres_residual_trace() {
        // Verify convergence for various n values.
        for &n in &[3, 4, 5, 6, 7, 8, 9, 10, 20, 50] {
            let a = laplacian_1d(n);
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let cfg = SolverConfig { rtol: 1e-10, max_iter: 2000, verbose: false, ..Default::default() };
            let res = solve_minres(&a, &b, &mut x, &cfg).unwrap_or_else(|_| panic!("n={n} solver error"));
            assert!(res.converged, "n={n} MINRES failed: iters={} res={:.3e}", res.iterations, res.final_residual);
            let mut ax = vec![0.0_f64; n];
            a.spmv(&x, &mut ax);
            let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
            assert!(err < 1e-6, "n={n} residual too large: {err}");
        }
    }

    #[test]
    fn minres_laplacian_spd() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_minres(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "MINRES (SPD) failed to converge");
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "MINRES (SPD) residual too large: {err}");
    }

    #[test]
    fn minres_indefinite() {
        let n = 50;
        let a = symmetric_indefinite(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_minres(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "MINRES (indefinite) failed to converge");
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "MINRES (indefinite) residual too large: {err}");
    }

    #[test]
    fn minres_operator_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_minres_operator(n, n, |z, w| a.spmv(z, w), &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "MINRES operator failed");
    }

    #[test]
    fn minres_converges_on_helmholtz_shift() {
        // Indefinite: Laplacian − 10·I  (negative shift → symmetric indefinite)
        let n = 50;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0_f64 - 10.0_f64);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let a = coo.into_csr();
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..Default::default() };
        let res = solve_minres(&a, &b, &mut x, &cfg).unwrap();
        assert!(res.converged, "MINRES (Helmholtz shift) failed: iters={} res={:.3e}", res.iterations, res.final_residual);
    }

    #[test]
    fn minres_residual_decreases() {
        let n = 30;
        let a = symmetric_indefinite(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 0.0, atol: 1e-12, max_iter: 1000, ..Default::default() };
        let res = solve_minres(&a, &b, &mut x, &cfg).unwrap();
        assert!(res.converged);
        assert!(res.iterations <= 50, "too many iterations: {}", res.iterations);
    }

    // ── GCR tests ────────────────────────────────────────────────────────────

    #[test]
    fn gcr_laplacian() {
        // GCR on SPD: convergence is slow due to A-orthogonalisation drift.
        // Use modest tolerance; for SPD, use CG or MINRES instead.
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-4, max_iter: 3000, ..Default::default() };
        let res = solve_gcr(&a, &b, &mut x, n, &cfg).unwrap();
        assert!(res.converged, "GCR (SPD) failed: iters={} res={:.3e}", res.iterations, res.final_residual);
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-3, "GCR residual too large: {err}");
    }

    #[test]
    fn gcr_tiny_spd() {
        // GCR converges on tiny SPD systems (n ≤ 3) at full space.
        for n in [3] {
            let a = laplacian_1d(n);
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let cfg = SolverConfig { rtol: 1e-6, max_iter: 200, ..Default::default() };
            let res = solve_gcr(&a, &b, &mut x, n, &cfg).unwrap_or_else(|e| panic!("n={n} GCR error: {e}"));
            assert!(res.converged, "n={n} GCR not converged: iters={} res={:.3e}", res.iterations, res.final_residual);
        }
    }

    #[test]
    fn gcr_converges_fewer_than_max_iters() {
        // Nonsymmetric system (convergence is reliable even with restart) 
        let n = 100;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0);
            if i > 0     { coo.add(i, i - 1, -1.2); }
            if i < n - 1 { coo.add(i, i + 1, -0.4); }
        }
        let a = coo.into_csr();
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..Default::default() };
        let res = solve_gcr(&a, &b, &mut x, 50, &cfg).unwrap();
        assert!(res.converged);
        assert!(res.iterations < 200, "GCR too many ({}): should converge <<200", res.iterations);
    }

    #[test]
    fn gcr_solution_matches_gmres() {
        let n = 40;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 4.0);
            if i > 0     { coo.add(i, i - 1, -1.5); }
            if i < n - 1 { coo.add(i, i + 1, -0.8); }
        }
        let a = coo.into_csr();
        let b: Vec<f64> = (0..n).map(|i| (i % 5 + 1) as f64).collect();
        let mut x_gcr = vec![0.0_f64; n];
        let mut x_gmres = vec![0.0_f64; n];
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 2000, ..Default::default() };
        solve_gcr(&a, &b, &mut x_gcr, 40, &cfg).unwrap();
        solve_gmres(&a, &b, &mut x_gmres, 40, &cfg).unwrap();
        for i in 0..n {
            assert!((x_gcr[i] - x_gmres[i]).abs() < 1e-6, "node {i}: gcr={:.6e} gmres={:.6e}", x_gcr[i], x_gmres[i]);
        }
    }

    #[test]
    fn gcr_zero_restart_errors() {
        let n = 10;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let r = solve_gcr(&a, &b, &mut x, 0, &SolverConfig::default());
        assert!(r.is_err(), "GCR restart=0 should error");
    }

    #[test]
    fn sparse_lu_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_lu(&a, &b).unwrap();
        // verify Ax �?b
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
    fn sparse_mumps_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_mumps(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "Mumps residual too large: {err}");
    }

    #[test]
    fn sparse_mkl_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_mkl(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "Mkl residual too large: {err}");
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

// ─── AMS / ADS integration tests ─────────────────────────────────────────────

#[cfg(test)]
mod ams_ads_tests {
    use super::*;
    use fem_assembly::{DiscreteLinearOperator, VectorAssembler};
    use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, HCurlSpace};
    use fem_space::constraints::{apply_dirichlet, boundary_dofs_hcurl};
    use fem_space::fe_space::FESpace;

    // ── AMS: H(curl) curl-curl + mass on 2-D unit square ──────────────────────

    #[test]
    fn pcg_ams_hcurl_2d_converges() {
        let n = 4;
        let mesh  = SimplexMesh::<2>::unit_square_tri(n);
        let h1    = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl)
            .expect("gradient assembly failed");

        // Apply zero Dirichlet BCs symmetrically (to keep SPD for PCG).
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 300, verbose: false, ..SolverConfig::default() },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg)
            .expect("PCG+AMS returned error");
        assert!(res.converged, "PCG+AMS did not converge in {} iters", res.iterations);
        assert!(res.final_residual < 1e-6, "residual = {}", res.final_residual);
    }

    #[test]
    fn gmres_ams_hcurl_2d_converges() {
        let n = 4;
        let mesh  = SimplexMesh::<2>::unit_square_tri(n);
        let h1    = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl)
            .expect("gradient assembly failed");

        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let vals = vec![0.0_f64; bnd.len()];
        let mut rhs = vec![1.0_f64; ndofs];
        apply_dirichlet(&mut a, &mut rhs, &bnd, &vals);

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 300, verbose: false, ..SolverConfig::default() },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_gmres_ams(&a, &g_linlvo, &rhs, &mut x, 30, &cfg)
            .expect("GMRES+AMS returned error");
        assert!(res.converged, "GMRES+AMS did not converge in {} iters", res.iterations);
        assert!(res.final_residual < 1e-6, "residual = {}", res.final_residual);
    }

    #[test]
    fn pcg_ams_solution_satisfies_ax_eq_b() {
        let n = 4;
        let mesh  = SimplexMesh::<2>::unit_square_tri(n);
        let h1    = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        // Apply zero Dirichlet BCs symmetrically (keeps SPD for PCG+AMS)
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 400, verbose: false, ..SolverConfig::default() },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg).unwrap();
        assert!(res.converged);

        // Verify Ax �?rhs
        let mut ax = vec![0.0_f64; ndofs];
        a.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(rhs.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
        assert!(err / rhs_norm < 1e-6, "relative residual = {}", err / rhs_norm);
    }

    #[test]
    fn pcg_ams_iteration_count_reasonable() {
        // AMS should converge in far fewer iterations than plain CG on H(curl)
        let n = 6;
        let mesh  = SimplexMesh::<2>::unit_square_tri(n);
        let h1    = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        // Apply zero Dirichlet BCs symmetrically (keeps SPD for PCG+AMS)
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 200, verbose: false, ..SolverConfig::default() },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg).unwrap();
        assert!(res.converged, "PCG+AMS did not converge");
        // AMS should be efficient �?converge in at most 100 iterations for this small problem
        assert!(res.iterations <= 100, "PCG+AMS took {} iters (expected �?00)", res.iterations);
    }

    // ── ADS: H(div) mass on 3-D unit cube ─────────────────────────────────────

    #[test]
    fn pcg_ads_hdiv_3d_converges() {
        use fem_space::constraints::boundary_dofs_hdiv;
        use fem_space::HDivSpace;

        let n = 2usize;
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(n);
        let h1    = H1Space::new(mesh3.clone(), 1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv  = HDivSpace::new(mesh3.clone(), 0);
        let ndofs_hdiv = hdiv.n_dofs();

        // H(div) mass matrix (SPD)
        let mut a_hdiv = VectorAssembler::assemble_bilinear(
            &hdiv,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            3,
        );

        // Discrete curl C: HCurl -> HDiv and gradient G: H1 -> HCurl
        let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv)
            .expect("curl_3d assembly failed");
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl)
            .expect("gradient assembly failed");

        // Apply zero normal-flux BCs on all boundary faces
        let bnd_hdiv = boundary_dofs_hdiv(hdiv.mesh(), &hdiv, &[1, 2, 3, 4, 5, 6]);
        let vals_hdiv = vec![0.0_f64; bnd_hdiv.len()];
        let mut rhs = vec![1.0_f64; ndofs_hdiv];
        apply_dirichlet(&mut a_hdiv, &mut rhs, &bnd_hdiv, &vals_hdiv);

        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AdsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 400, verbose: false, ..SolverConfig::default() },
            ads_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs_hdiv];
        let res = solve_pcg_ads(&a_hdiv, &c_linlvo, &g_linlvo, &rhs, &mut x, &cfg)
            .expect("PCG+ADS returned error");
        assert!(res.converged, "PCG+ADS did not converge in {} iters", res.iterations);
        assert!(res.final_residual < 1e-6, "residual = {}", res.final_residual);
    }

    #[test]
    fn gmres_ads_hdiv_3d_converges() {
        use fem_space::constraints::boundary_dofs_hdiv;
        use fem_space::HDivSpace;

        let n = 2usize;
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(n);
        let h1    = H1Space::new(mesh3.clone(), 1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv  = HDivSpace::new(mesh3.clone(), 0);
        let ndofs_hdiv = hdiv.n_dofs();

        let mut a_hdiv = VectorAssembler::assemble_bilinear(
            &hdiv,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        let bnd_hdiv = boundary_dofs_hdiv(hdiv.mesh(), &hdiv, &[1, 2, 3, 4, 5, 6]);
        let vals_hdiv = vec![0.0_f64; bnd_hdiv.len()];
        let mut rhs = vec![1.0_f64; ndofs_hdiv];
        apply_dirichlet(&mut a_hdiv, &mut rhs, &bnd_hdiv, &vals_hdiv);

        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AdsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 400, verbose: false, ..SolverConfig::default() },
            ads_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs_hdiv];
        let res = solve_gmres_ads(&a_hdiv, &c_linlvo, &g_linlvo, &rhs, &mut x, 30, &cfg)
            .expect("GMRES+ADS returned error");
        assert!(res.converged, "GMRES+ADS did not converge in {} iters", res.iterations);
        assert!(res.final_residual < 1e-6, "residual = {}", res.final_residual);
    }

    #[test]
    fn pcg_ads_solution_satisfies_ax_eq_b() {
        use fem_space::constraints::boundary_dofs_hdiv;
        use fem_space::HDivSpace;

        let n = 2usize;
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(n);
        let h1    = H1Space::new(mesh3.clone(), 1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv  = HDivSpace::new(mesh3.clone(), 0);
        let ndofs_hdiv = hdiv.n_dofs();

        let mut a_hdiv = VectorAssembler::assemble_bilinear(
            &hdiv,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        let bnd_hdiv = boundary_dofs_hdiv(hdiv.mesh(), &hdiv, &[1, 2, 3, 4, 5, 6]);
        let vals_hdiv = vec![0.0_f64; bnd_hdiv.len()];
        let mut rhs = vec![1.0_f64; ndofs_hdiv];
        apply_dirichlet(&mut a_hdiv, &mut rhs, &bnd_hdiv, &vals_hdiv);

        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AdsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() },
            ads_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs_hdiv];
        let res = solve_pcg_ads(&a_hdiv, &c_linlvo, &g_linlvo, &rhs, &mut x, &cfg).unwrap();
        assert!(res.converged);

        // Verify Ax �?rhs
        let mut ax = vec![0.0_f64; ndofs_hdiv];
        a_hdiv.spmv(&x, &mut ax);
        let err: f64 = ax.iter().zip(rhs.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
        assert!(err / rhs_norm < 1e-6, "relative residual = {}", err / rhs_norm);
    }

    #[test]
    fn pcg_ams_p1_nd2_converges() {
        // AMS with H^1 order 1 + H(curl) order 2 (ND2).
        // Tests that gradient() works with mismatched orders (h1=1, hcurl=2).
        let n = 4;
        let mesh  = SimplexMesh::<2>::unit_square_tri(n);
        let h1    = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 2); // ND2
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl)
            .expect("gradient P1->ND2 should succeed");

        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-6, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg)
            .expect("PCG+AMS (P1+ND2) returned error");
        assert!(res.converged, "PCG+AMS (P1+ND2) did not converge in {} iters", res.iterations);
        assert!(res.final_residual < 1e-6, "residual = {}", res.final_residual);
    }
}
