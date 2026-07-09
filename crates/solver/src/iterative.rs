use fem_linalg::CsrMatrix as FemCsr;
use linlvo::{
    core::scalar::Scalar as linlvoScalar,
    iterative::{BiCgStab, ConjugateGradient, Fgmres, Gmres, Idrs, Tfqmr},
    DenseVec, Ilu0Precond, IldltPrecond, JacobiPrecond, KrylovSolver, Preconditioner, SsorPrecond,
};
use linlvo::precond::{IlukPrecond, IlutPrecond};
use linlvo::{LinearOperator, Vector};
use fem_linalg::{fem_to_linlvo_csr, into_result, SolverConfig, SolverError, SolveResult};

use crate::macros::check_dims;

// ─── Macro-generated iterative solvers ──────────────────────────────────────

solve_iterative_simple!(solve_cg, ConjugateGradient<T>, "Conjugate Gradient - for symmetric positive definite systems.");

solve_precond_simple!(solve_pcg_jacobi, ConjugateGradient<T>, JacobiPrecond<T>, "Preconditioned CG with Jacobi preconditioner.");

/// PCG with symmetric Gauss-Seidel (SSOR(ω=1)) preconditioner — MFEM GSSmoother.
pub fn solve_pcg_gssmoother<T: linlvoScalar>(
    a: &FemCsr<T>, b: &[T], x: &mut [T], cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    check_dims(a, b, x)?;
    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());
    let prec = SsorPrecond::from_csr(&la, T::one()).map_err(|e| SolverError::Linlvo(e.to_string()))?;
    let res = ConjugateGradient::default()
        .solve(&la, Some(&prec), &lb, &mut lx, &cfg.to_linlvo())
        .map_err(SolverError::from)?;
    x.copy_from_slice(lx.as_slice());
    Ok(into_result(res))
}

solve_precond_simple!(solve_pcg_ilu0, ConjugateGradient<T>, Ilu0Precond<T>, "Preconditioned CG with ILU(0) preconditioner.");

solve_iterative_restart!(solve_gmres, Gmres<T>, "GMRES for general (possibly non-symmetric) systems.", restart: usize);

solve_precond_restart!(solve_gmres_jacobi, Gmres<T>, JacobiPrecond<T>, "GMRES with Jacobi preconditioner.");

solve_precond_restart!(solve_gmres_ilu0, Gmres<T>, Ilu0Precond<T>, "GMRES with ILU(0) preconditioner.");

solve_iterative_simple!(solve_bicgstab, BiCgStab<T>, "BiCGSTAB for non-symmetric systems; often faster than GMRES per iteration.");

solve_iterative_restart!(solve_fgmres, Fgmres<T>, "Flexible GMRES - allows a variable preconditioner per iteration.", restart: usize);

solve_precond_restart!(solve_fgmres_jacobi, Fgmres<T>, JacobiPrecond<T>, "Flexible GMRES with Jacobi preconditioner.");

solve_precond_restart!(solve_fgmres_ilu0, Fgmres<T>, Ilu0Precond<T>, "Flexible GMRES with ILU(0) preconditioner.");

solve_precond_simple!(solve_pcg_ildlt, ConjugateGradient<T>, IldltPrecond<T>, "Preconditioned CG with incomplete LDL^T preconditioner.");

solve_precond_restart!(solve_gmres_ildlt, Gmres<T>, IldltPrecond<T>, "GMRES with incomplete LDL^T preconditioner.");

solve_iterative_restart!(solve_idrs, Idrs<T>, "IDR(s) - Induced Dimension Reduction for non-symmetric systems.", s: usize);

solve_iterative_simple!(solve_tfqmr, Tfqmr<T>, "TFQMR - Transpose-Free Quasi-Minimal Residual for non-symmetric systems.");

// ─── CG operator ───────────────────────────────────────────────────────────

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

// ─── GMRES operator ────────────────────────────────────────────────────────

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

        let mut v = vec![0.0; (restart + 1) * n];
        for i in 0..n {
            v[i] = r[i] / beta;
        }

        let mut h = vec![0.0; (restart + 1) * restart];
        let mut cs = vec![0.0; restart];
        let mut sn = vec![0.0; restart];
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;

        let mut inner_done = 0usize;
        let mut converged = false;

        let mut w = vec![0.0; n];
        for j in 0..restart {
            if iter_total >= cfg.max_iter {
                break;
            }

            apply(&v[j * n..(j + 1) * n], &mut w);

            for i in 0..=j {
                h[i * restart + j] = dot(&w, &v[i * n..(i + 1) * n]);
                for k in 0..n {
                    w[k] -= h[i * restart + j] * v[i * n + k];
                }
            }

            h[(j + 1) * restart + j] = norm(&w);
            if h[(j + 1) * restart + j] > 1e-32 {
                for k in 0..n {
                    v[(j + 1) * n + k] = w[k] / h[(j + 1) * restart + j];
                }
            }

            // Apply existing Givens rotations.
            for i in 0..j {
                let tmp = cs[i] * h[i * restart + j] + sn[i] * h[(i + 1) * restart + j];
                h[(i + 1) * restart + j] = -sn[i] * h[i * restart + j] + cs[i] * h[(i + 1) * restart + j];
                h[i * restart + j] = tmp;
            }

            // Build and apply new Givens rotation.
            let denom = (h[j * restart + j] * h[j * restart + j] + h[(j + 1) * restart + j] * h[(j + 1) * restart + j]).sqrt();
            if denom > 1e-32 {
                cs[j] = h[j * restart + j] / denom;
                sn[j] = h[(j + 1) * restart + j] / denom;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }

            h[j * restart + j] = cs[j] * h[j * restart + j] + sn[j] * h[(j + 1) * restart + j];
            h[(j + 1) * restart + j] = 0.0;

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
                s -= h[i * restart + k] * y[k];
            }
            let diag = h[i * restart + i];
            if diag.abs() < 1e-32 {
                return Err(SolverError::Linlvo(
                    "GMRES operator breakdown: near-singular Hessenberg diagonal".to_string(),
                ));
            }
            y[i] = s / diag;
        }

        for i in 0..m {
            for k in 0..n {
                x[k] += y[i] * v[i * n + k];
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

// ─── BiCGSTAB operator ─────────────────────────────────────────────────────

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

// ─── Generic preconditioner interface ──────────────────────────────────────

/// Preconditioned conjugate gradient with energy-norm convergence.
///
/// Convergence criterion: `(M⁻¹r_k, r_k) < rtol · (M⁻¹r₀, r₀)`.
/// This checks the **preconditioned residual energy norm** — the natural
/// convergence metric for PCG — rather than the true residual `‖r‖`.
///
/// Prints per-iteration history when `verbose` is true.
pub fn solve_pcg<P>(
    a: &FemCsr<f64>,
    b: &[f64],
    x: &mut [f64],
    precond: &P,
    rtol: f64,
    max_iter: usize,
    verbose: bool,
) -> Result<SolveResult, SolverError>
where
    P: Preconditioner<Vector = DenseVec<f64>>,
{
    let n = a.nrows;
    check_dims(a, b, x)?;

    let la = fem_to_linlvo_csr(a);
    let lb = DenseVec::from_vec(b.to_vec());
    let mut lx = DenseVec::from_vec(x.to_vec());

    // r = b - A·x
    let mut r = DenseVec::zeros(n);
    la.apply(&lx, &mut r);
    for i in 0..n {
        r.as_mut_slice()[i] = lb.as_slice()[i] - r.as_slice()[i];
    }

    // z = M⁻¹·r
    let mut z = DenseVec::zeros(n);
    precond.apply_precond(&r, &mut z);

    let gamma0 = r.dot(&z);            // (B r₀, r₀)
    if gamma0 == 0.0 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let tol = rtol * gamma0;
    let mut p = z.clone();            // p₀ = z₀
    let mut gamma = gamma0;
    let mut w = DenseVec::zeros(n);

    if verbose {
        eprintln!("   Iteration :    0  (B r, r) = {:.5e}", gamma0);
    }

    for iter in 1..=max_iter {
        // w = A·p
        la.apply(&p, &mut w);

        let alpha = gamma / p.dot(&w);
        lx.axpy(alpha, &p);            // x ← x + α·p
        r.axpy(-alpha, &w);            // r ← r − α·w

        precond.apply_precond(&r, &mut z);  // z = M⁻¹·r

        let gamma_new = r.dot(&z);      // (B r_{k+1}, r_{k+1})

        if verbose {
            eprintln!("   Iteration : {:4}  (B r, r) = {:.5e}", iter, gamma_new);
        }

        if gamma_new < tol {
            x.copy_from_slice(lx.as_slice());
            let final_residual = gamma_new.sqrt();
            if verbose {
                let reduction = final_residual / gamma0.sqrt();
                let avg = reduction.powf(1.0 / iter as f64);
                eprintln!("Average reduction factor = {:.6}", avg);
            }
            return Ok(SolveResult { converged: true, iterations: iter, final_residual });
        }

        let beta = gamma_new / gamma;
        // p = z + β·p  (scale old p by β, then add z)
        p.scale(beta);
        p.axpy(1.0, &z);

        gamma = gamma_new;
    }

    x.copy_from_slice(lx.as_slice());
    Err(SolverError::ConvergenceFailed { max_iter, residual: gamma.sqrt() })
}

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
/// iterations — inner Krylov solves, AMG V-cycles, and nonlinear operators all
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

// ─── ILU family: ILU(k), ILUT, PCG-ILU(k) ─────────────────────────────────

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
/// * `tau`    — relative drop tolerance (0.0 = keep all, 0.01 = aggressive)
/// * `p_fill` — max off-diagonal fill per row in L and U
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

// ─── MINRES (symmetric indefinite) ─────────────────────────────────────────

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
/// Algorithm: Paige & Saunders (1975); Lanczos vectors stored in a contiguous
/// `Vec<f64>` (`v[iter * n + i]`) for cache locality instead of `Vec<Vec<f64>>`.
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

    // ── Lanczos vectors (contiguous: v[iter * n + i]) ──────────────────
    // V[0] = v_0 = 0 (placeholder), V[1] = v_1 = r₀ / ‖r₀‖
    let max_vecs = cfg.max_iter + 2;
    let mut v = vec![0.0_f64; max_vecs * n];  // single allocation
    // v₀ = 0 already (zero-initialised)
    {
        let inv = 1.0 / res_norm;
        for i in 0..n { v[1 * n + i] = r[i] * inv; }
    }

    // Tricliagonal coefficients: α[1..k], β[0..k]  (β[0] = 0)
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta:  Vec<f64> = vec![0.0];          // β₀

    // QR factorisation of T̃_k:
    let mut r_sup2: Vec<f64> = Vec::new();   // R_{k-2,k}
    let mut r_sup1: Vec<f64> = Vec::new();   // R_{k-1,k}
    let mut r_diag: Vec<f64> = Vec::new();   // R_{k,k} = ρ_k

    // Givens rotation history
    let mut cs_old   = 1.0_f64;
    let mut sn_old   = 0.0_f64;
    let mut cs_older = 1.0_f64;
    let mut sn_older = 0.0_f64;

    // Transformed RHS: g = Q_k^T · β₁·e₁
    let mut g = vec![res_norm];

    for iter in 1..=cfg.max_iter {
        // ── Lanczos step ────────────────────────────────────────────────
        let mut av = vec![0.0; n];
        apply(&v[iter * n..(iter + 1) * n], &mut av);

        let ak = dot(&v[iter * n..(iter + 1) * n], &av);
        alpha.push(ak);

        for i in 0..n {
            r[i] = av[i] - ak * v[iter * n + i] - beta[iter - 1] * v[(iter - 1) * n + i];
        }
        let bk = norm(&r);
        beta.push(bk);

        if bk > 1e-32 {
            let inv = 1.0 / bk;
            for i in 0..n { v[(iter + 1) * n + i] = r[i] * inv; }
        } else {
            // v_{k+1} = 0 (already zero-initialised)
        }

        // ── Apply Givens QR to column k of T̃_k ────────────────────────
        let zeta_sup2 = sn_older * beta[iter - 1];
        let zeta_sub  = cs_older * beta[iter - 1];
        r_sup2.push(zeta_sup2);

        let zeta_sup1 = cs_old * zeta_sub + sn_old * ak;
        let diag_in   = -sn_old * zeta_sub + cs_old * ak;
        r_sup1.push(zeta_sup1);

        let rk = (diag_in * diag_in + bk * bk).sqrt();
        let (csk, snk) = if rk > 1e-32 {
            (diag_in / rk, bk / rk)
        } else {
            (1.0, 0.0)
        };

        r_diag.push(rk);

        // ── Update g (transformed RHS) ──────────────────────────────────
        let g_old = g[iter - 1];
        g[iter - 1] = csk * g_old;
        g.push(-snk * g_old);

        res_norm = g[iter].abs();
        if res_norm <= tol {
            // ── Solve R_k y = g_{1:k} and compute x = V_k y ────────────
            let k = iter;
            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                let mut s = g[i];
                if i + 1 < k { s -= r_sup1[i + 1] * y[i + 1]; }
                if i + 2 < k { s -= r_sup2[i + 2] * y[i + 2]; }
                y[i] = s / r_diag[i];
            }
            // x = V_k y = Σ y_j * v_j  (contiguous access)
            for i in 0..n { x[i] = 0.0; }
            for j in 0..k {
                let vj = &v[(j + 1) * n..(j + 2) * n];
                for i in 0..n { x[i] += y[j] * vj[i]; }
            }
            return Ok(SolveResult { converged: true, iterations: iter, final_residual: res_norm });
        }

        // ── Shift Givens history ─────────────────────────────────────────
        cs_older = cs_old;
        sn_older = sn_old;
        cs_old = csk;
        sn_old = snk;
    }

    Err(SolverError::ConvergenceFailed { max_iter: cfg.max_iter, residual: res_norm })
}

/// MINRES with Jacobi (diagonal) preconditioning.
///
/// Uses split preconditioning: `A' = D^{-1/2} A D^{-1/2}` where `D = diag(A)`.
/// The algorithm is mathematically equivalent to applying MINRES to the
/// diagonally-scaled system, then unscaling the solution.
///
/// This preserves symmetry and is the most effective preconditioner for
/// diagonally-dominant matrices (mass + diffusion + stiffness combinations).
pub fn solve_minres_jacobi(
    a: &FemCsr<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows;
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch { rows: n, cols: n, rhs: b.len() });
    }

    // 1. Extract Jacobi scaling: s[i] = 1 / sqrt(|A[i,i]|)
    let mut s = vec![1.0; n];
    for i in 0..n {
        let d = a.get(i, i).abs().max(f64::MIN_POSITIVE);
        s[i] = 1.0 / d.sqrt();
    }

    // 2. Create scaled operator: A'[i,j] = s[i] * A[i,j] * s[j]
    let apply_scaled = |y: &[f64], z: &mut [f64]| {
        // tmp = A * (s ⊙ y)
        let mut tmp = vec![0.0; n];
        let mut sy = vec![0.0; n];
        for i in 0..n { sy[i] = y[i] * s[i]; }
        a.spmv(&sy, &mut tmp);
        // z = s ⊙ tmp  (i.e., z = s ⊙ (A * (s ⊙ y)))
        for i in 0..n { z[i] = s[i] * tmp[i]; }
    };

    // 3. Scale RHS: b'[i] = s[i] * b[i]
    let mut bs = vec![0.0; n];
    for i in 0..n { bs[i] = s[i] * b[i]; }

    // 4. Solve A' * y = b' using the existing MINRES implementation
    let mut y = x.to_vec();
    match solve_minres_impl(n, apply_scaled, &bs, &mut y, cfg) {
        Ok(res) => {
            // 5. Unscale: x[i] = s[i] * y[i]
            for i in 0..n { x[i] = s[i] * y[i]; }
            Ok(res)
        }
        Err(e) => Err(e),
    }
}

// ─── GCR (generalised conjugate residual, restart) ─────────────────────────

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
