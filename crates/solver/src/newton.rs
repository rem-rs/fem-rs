//! Newton-Raphson nonlinear solver with backtracking line search.
//!
//! Solves `F(u) = 0` where `F: ℝⁿ → ℝⁿ` is a nonlinear residual function.
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::{NewtonRaphsonConfig, solve_newton, SolverConfig};
//!
//! let cfg = NewtonRaphsonConfig::default();
//! let result = solve_newton(&mut u, |x| {
//!     let r = compute_residual(x);
//!     let j = compute_jacobian(x);
//!     (r, j)
//! }, &cfg, &SolverConfig::default())?;
//! ```

use fem_linalg::{CsrMatrix, SolverConfig, SolverError};

/// Configuration for the Newton-Raphson solver.
#[derive(Debug, Clone)]
pub struct NewtonRaphsonConfig {
    /// Relative tolerance on the residual norm: `‖R‖ < rtol · ‖R₀‖`.
    pub rtol: f64,
    /// Absolute tolerance on the residual norm: `‖R‖ < atol`.
    pub atol: f64,
    /// Tolerance on the update norm: `‖Δu‖ < dtol · (‖u‖ + 1)`.
    pub dtol: f64,
    /// Maximum number of Newton iterations.
    pub max_iter: usize,
    /// Maximum number of line-search backtrack steps.
    pub max_line_search_iter: usize,
    /// Backtracking factor for Armijo line search (0 < factor < 1).
    pub line_search_factor: f64,
    /// Armijo condition parameter (small, e.g. 1e-4).
    pub armijo_c: f64,
    /// Print convergence history.
    pub verbose: bool,
}

impl Default for NewtonRaphsonConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-10,
            dtol: 1e-10,
            max_iter: 50,
            max_line_search_iter: 10,
            line_search_factor: 0.5,
            armijo_c: 1e-4,
            verbose: false,
        }
    }
}

/// Result of a Newton-Raphson solve.
#[derive(Debug, Clone)]
pub struct NewtonResult {
    /// Number of Newton iterations performed.
    pub iterations: usize,
    /// Final residual norm `‖R‖`.
    pub final_residual: f64,
    /// Initial residual norm `‖R₀‖`.
    pub initial_residual: f64,
    /// Whether the solver converged.
    pub converged: bool,
    /// Reason for termination.
    pub reason: NewtonStopReason,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NewtonStopReason {
    /// Residual converged: `‖R‖ < max(rtol·‖R₀‖, atol)`.
    ResidualConverged,
    /// Update converged: `‖Δu‖ < dtol·(‖u‖ + 1)`.
    UpdateConverged,
    /// Maximum iterations reached.
    MaxIterations,
    /// Line search failed to find a valid step.
    LineSearchFailed,
    /// Linear solver failed.
    LinearSolveFailed(String),
    /// Residual or Jacobian is NaN/Inf.
    NanDetected,
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Backtracking Armijo line search.
fn line_search_armijo<F>(
    u: &[f64],
    du: &[f64],
    r0_norm: f64,
    c: f64,
    factor: f64,
    max_iter: usize,
    residual_fn: &F,
) -> Option<(f64, Vec<f64>, Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> (Vec<f64>, CsrMatrix<f64>),
{
    let mut alpha = 1.0;
    for _ in 0..max_iter {
        let u_trial: Vec<f64> = u
            .iter()
            .zip(du.iter())
            .map(|(ui, dui)| ui + alpha * dui)
            .collect();
        let (r_trial, _) = residual_fn(&u_trial);
        let r_norm = norm2(&r_trial);

        let threshold = (1.0 - 2.0 * alpha * c) * r0_norm;
        if r_norm < threshold {
            return Some((alpha, r_trial, u_trial, r_norm));
        }

        alpha *= factor;
    }
    None
}

/// Solve `F(u) = 0` using Newton-Raphson with backtracking line search
/// and PCG+Jacobi as the linear solver.
///
/// Best for SPD Jacobians (e.g. hyperelasticity).
pub fn solve_newton<F>(
    u: &mut [f64],
    residual_and_jacobian: F,
    cfg: &NewtonRaphsonConfig,
    lin_cfg: &SolverConfig,
) -> Result<NewtonResult, SolverError>
where
    F: Fn(&[f64]) -> (Vec<f64>, CsrMatrix<f64>),
{
    let n = u.len();
    let (mut r, _j) = residual_and_jacobian(u);
    let mut r_norm = norm2(&r);
    let initial_residual = r_norm;

    if cfg.verbose {
        println!("  Newton: iter=0, ‖R‖={:.6e}", r_norm);
    }

    if !r_norm.is_finite() {
        return Ok(NewtonResult {
            iterations: 0,
            final_residual: r_norm,
            initial_residual,
            converged: false,
            reason: NewtonStopReason::NanDetected,
        });
    }

    let abs_tol = (cfg.rtol * initial_residual).max(cfg.atol);
    if r_norm < abs_tol {
        return Ok(NewtonResult {
            iterations: 0,
            final_residual: r_norm,
            initial_residual,
            converged: true,
            reason: NewtonStopReason::ResidualConverged,
        });
    }

    let mut du = vec![0.0_f64; n];

    for iter in 1..=cfg.max_iter {
        let (_, jac) = residual_and_jacobian(u);
        let rhs: Vec<f64> = r.iter().map(|&v| -v).collect();

        // Solve using iterative solver
        // Solve using iterative solver
        let lin_ok = crate::solve_pcg_jacobi(&jac, &rhs, &mut du, lin_cfg).is_ok()
            || crate::solve_gmres_ilu0(&jac, &rhs, &mut du, 30, lin_cfg).is_ok();

        if !lin_ok {
            return Ok(NewtonResult {
                iterations: iter - 1,
                final_residual: r_norm,
                initial_residual,
                converged: false,
                reason: NewtonStopReason::LinearSolveFailed("PCG and GMRES both failed".into()),
            });
        }

        match line_search_armijo(
            u,
            &du,
            r_norm,
            cfg.armijo_c,
            cfg.line_search_factor,
            cfg.max_line_search_iter,
            &residual_and_jacobian,
        ) {
            Some((alpha, r_new, u_new, r_new_norm)) => {
                u.copy_from_slice(&u_new);
                r = r_new;
                r_norm = r_new_norm;

                if cfg.verbose {
                    println!(
                        "  Newton: iter={}, α={:.4e}, ‖R‖={:.6e}",
                        iter, alpha, r_norm
                    );
                }

                let abs_tol = (cfg.rtol * initial_residual).max(cfg.atol);
                if r_norm < abs_tol {
                    return Ok(NewtonResult {
                        iterations: iter,
                        final_residual: r_norm,
                        initial_residual,
                        converged: true,
                        reason: NewtonStopReason::ResidualConverged,
                    });
                }

                let du_norm = norm2(&du);
                let u_norm = norm2(u).max(1.0);
                if du_norm < cfg.dtol * u_norm {
                    return Ok(NewtonResult {
                        iterations: iter,
                        final_residual: r_norm,
                        initial_residual,
                        converged: true,
                        reason: NewtonStopReason::UpdateConverged,
                    });
                }

                if !r_norm.is_finite() {
                    return Ok(NewtonResult {
                        iterations: iter,
                        final_residual: r_norm,
                        initial_residual,
                        converged: false,
                        reason: NewtonStopReason::NanDetected,
                    });
                }
            }
            None => {
                return Ok(NewtonResult {
                    iterations: iter - 1,
                    final_residual: r_norm,
                    initial_residual,
                    converged: false,
                    reason: NewtonStopReason::LineSearchFailed,
                });
            }
        }
    }

    Ok(NewtonResult {
        iterations: cfg.max_iter,
        final_residual: r_norm,
        initial_residual,
        converged: false,
        reason: NewtonStopReason::MaxIterations,
    })
}

/// Solve `F(u) = 0` using Newton-Raphson with direct sparse LU solver.
///
/// More robust for non-SPD Jacobians than the iterative variant.
pub fn solve_newton_lu<F>(
    u: &mut [f64],
    residual_and_jacobian: F,
    cfg: &NewtonRaphsonConfig,
) -> Result<NewtonResult, SolverError>
where
    F: Fn(&[f64]) -> (Vec<f64>, CsrMatrix<f64>),
{
    let n = u.len();
    let (mut r, _j) = residual_and_jacobian(u);
    let mut r_norm = norm2(&r);
    let initial_residual = r_norm;

    if cfg.verbose {
        println!("  Newton(LU): iter=0, ‖R‖={:.6e}", r_norm);
    }

    if !r_norm.is_finite() {
        return Ok(NewtonResult {
            iterations: 0,
            final_residual: r_norm,
            initial_residual,
            converged: false,
            reason: NewtonStopReason::NanDetected,
        });
    }

    let abs_tol = (cfg.rtol * initial_residual).max(cfg.atol);
    if r_norm < abs_tol {
        return Ok(NewtonResult {
            iterations: 0,
            final_residual: r_norm,
            initial_residual,
            converged: true,
            reason: NewtonStopReason::ResidualConverged,
        });
    }

    let mut du = vec![0.0_f64; n];

    for iter in 1..=cfg.max_iter {
        let (_, jac) = residual_and_jacobian(u);
        let rhs: Vec<f64> = r.iter().map(|&v| -v).collect();

        match crate::solve_sparse_lu(&jac, &rhs) {
            Ok(du_vec) => {
                du.copy_from_slice(&du_vec);
            }
            Err(e) => {
                return Ok(NewtonResult {
                    iterations: iter - 1,
                    final_residual: r_norm,
                    initial_residual,
                    converged: false,
                    reason: NewtonStopReason::LinearSolveFailed(e.to_string()),
                });
            }
        }

        match line_search_armijo(
            u,
            &du,
            r_norm,
            cfg.armijo_c,
            cfg.line_search_factor,
            cfg.max_line_search_iter,
            &residual_and_jacobian,
        ) {
            Some((alpha, r_new, u_new, r_new_norm)) => {
                u.copy_from_slice(&u_new);
                r = r_new;
                r_norm = r_new_norm;

                if cfg.verbose {
                    println!(
                        "  Newton(LU): iter={}, α={:.4e}, ‖R‖={:.6e}",
                        iter, alpha, r_norm
                    );
                }

                let abs_tol = (cfg.rtol * initial_residual).max(cfg.atol);
                if r_norm < abs_tol {
                    return Ok(NewtonResult {
                        iterations: iter,
                        final_residual: r_norm,
                        initial_residual,
                        converged: true,
                        reason: NewtonStopReason::ResidualConverged,
                    });
                }

                let du_norm = norm2(&du);
                let u_norm = norm2(u).max(1.0);
                if du_norm < cfg.dtol * u_norm {
                    return Ok(NewtonResult {
                        iterations: iter,
                        final_residual: r_norm,
                        initial_residual,
                        converged: true,
                        reason: NewtonStopReason::UpdateConverged,
                    });
                }

                if !r_norm.is_finite() {
                    return Ok(NewtonResult {
                        iterations: iter,
                        final_residual: r_norm,
                        initial_residual,
                        converged: false,
                        reason: NewtonStopReason::NanDetected,
                    });
                }
            }
            None => {
                return Ok(NewtonResult {
                    iterations: iter - 1,
                    final_residual: r_norm,
                    initial_residual,
                    converged: false,
                    reason: NewtonStopReason::LineSearchFailed,
                });
            }
        }
    }

    Ok(NewtonResult {
        iterations: cfg.max_iter,
        final_residual: r_norm,
        initial_residual,
        converged: false,
        reason: NewtonStopReason::MaxIterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn build_1x1_csr(val: f64) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(1, 1);
        coo.add(0, 0, val);
        coo.into_csr()
    }

    fn build_2x2_csr(vals: &[f64; 4]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(2, 2);
        coo.add(0, 0, vals[0]);
        coo.add(0, 1, vals[1]);
        coo.add(1, 0, vals[2]);
        coo.add(1, 1, vals[3]);
        coo.into_csr()
    }

    #[test]
    fn newton_1d_scalar() {
        let mut u = vec![2.0_f64];
        let cfg = NewtonRaphsonConfig {
            rtol: 1e-12,
            atol: 1e-14,
            ..NewtonRaphsonConfig::default()
        };
        let lin_cfg = SolverConfig {
            rtol: 1e-14,
            max_iter: 100,
            ..SolverConfig::default()
        };

        let result = solve_newton(
            &mut u,
            |x| {
                let r = vec![x[0] * x[0] - 2.0];
                let j = build_1x1_csr(2.0 * x[0]);
                (r, j)
            },
            &cfg,
            &lin_cfg,
        )
        .unwrap();

        assert!(
            result.converged,
            "Newton should converge: {:?}",
            result.reason
        );
        assert!((u[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn newton_2d_system() {
        let mut u = vec![1.0, 0.5];
        let cfg = NewtonRaphsonConfig::default();
        let lin_cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 200,
            ..SolverConfig::default()
        };

        let result = solve_newton(
            &mut u,
            |x| {
                let r = vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
                let j = build_2x2_csr(&[2.0 * x[0], 2.0 * x[1], 1.0, -1.0]);
                (r, j)
            },
            &cfg,
            &lin_cfg,
        )
        .unwrap();

        assert!(result.converged);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((u[0] - expected).abs() < 1e-10);
        assert!((u[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn newton_poor_initial_guess() {
        let mut u = vec![100.0_f64];
        let cfg = NewtonRaphsonConfig {
            max_iter: 100,
            rtol: 1e-10,
            ..NewtonRaphsonConfig::default()
        };
        let lin_cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 100,
            ..SolverConfig::default()
        };

        let result = solve_newton(
            &mut u,
            |x| {
                let r = vec![x[0] * x[0] - 2.0];
                let j = build_1x1_csr(2.0 * x[0]);
                (r, j)
            },
            &cfg,
            &lin_cfg,
        )
        .unwrap();

        assert!(result.converged);
        assert!((u[0] - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn newton_lu_1d() {
        let mut u = vec![2.0_f64];
        let cfg = NewtonRaphsonConfig {
            rtol: 1e-12,
            ..NewtonRaphsonConfig::default()
        };

        let result = solve_newton_lu(
            &mut u,
            |x| {
                let r = vec![x[0] * x[0] - 2.0];
                let j = build_1x1_csr(2.0 * x[0]);
                (r, j)
            },
            &cfg,
        )
        .unwrap();

        assert!(result.converged);
        assert!((u[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn newton_quadratic_convergence() {
        let mut u = vec![1.5_f64];
        let cfg = NewtonRaphsonConfig {
            rtol: 1e-14,
            atol: 1e-16,
            max_iter: 20,
            ..NewtonRaphsonConfig::default()
        };
        let lin_cfg = SolverConfig {
            rtol: 1e-14,
            max_iter: 100,
            ..SolverConfig::default()
        };

        let result = solve_newton(
            &mut u,
            |x| {
                let r = vec![x[0] * x[0] - 2.0];
                let j = build_1x1_csr(2.0 * x[0]);
                (r, j)
            },
            &cfg,
            &lin_cfg,
        )
        .unwrap();

        assert!(result.converged);
        assert!(
            result.iterations <= 6,
            "Newton took {} iters",
            result.iterations
        );
    }
}
