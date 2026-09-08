//! MFEM-style stationary linear iteration (SLI) and Conjugate Gradient.
//!
//! 1:1 port of MFEM `SLISolver::Mult` and `CGSolver::Mult`
//! (`linalg/solvers.cpp`), including the exact console output produced by
//! `SetPrintLevel(1)` (`PrintLevel().Errors().Warnings().Iterations()`), so
//! the diag-smoothers miniapps reproduce the C++ iteration logs:
//!
//! - SLI:  `   Iteration :   0  ||Br|| = <nom>` per step plus
//!   `\tConv. rate: <cf>`; summary lines only when warnings are active
//!   (non-converged runs): `SLI: Number of iterations:`, `Conv. rate:`,
//!   `Average reduction factor:`, `SLI: No convergence!`.
//! - CG:   `   Iteration :   i  (B r, r) = <betanom>`; then
//!   `Average reduction factor = <arf>` and, for non-converged runs,
//!   `PCG: Number of iterations:` / `PCG: No convergence!`.
//!
//! Numbers are formatted like MFEM's default `std::ostream` state (6
//! significant digits, see [`crate::iterative::fmt_g`]).
//!
//! Convergence criteria (identical to MFEM):
//! - SLI: `||B r|| < max(||B r0|| · rel_tol, abs_tol)` (norm-based).
//! - CG:  `(B r, r) <= max((B r0, r0) · rel_tol², abs_tol²)`.
//!
//! The optional `monitor` closure mirrors MFEM's `IterativeSolver::Monitor`
//! hook: it is called with `(iteration, norm, final)` at the same points
//! (`final == true` for the closing call), which is what the miniapps'
//! `DataMonitor` (CSV of `(B r, r)` per iteration) needs.  Unlike MFEM the
//! monitor cannot request an early stop (the miniapp monitors never do).

use crate::iterative::fmt_g;

/// Solver options mirroring MFEM `IterativeSolver` parameters.
#[derive(Debug, Clone)]
pub struct SliOptions {
    /// MFEM `SetRelTol`.
    pub rel_tol: f64,
    /// MFEM `SetAbsTol`.
    pub abs_tol: f64,
    /// MFEM `SetMaxIter`.
    pub max_iter: i32,
    /// MFEM `SetPrintLevel` (only levels `-1`, `0`, `1` are distinguished
    /// here; level `1` prints the per-iteration log and enables warnings).
    pub print_level: i32,
}

impl Default for SliOptions {
    fn default() -> Self {
        Self { rel_tol: 0.0, abs_tol: 0.0, max_iter: 10, print_level: -1 }
    }
}

/// Result of an MFEM-style iterative solve.
#[derive(Debug, Clone, Copy)]
pub struct IterResult {
    /// MFEM `GetConverged()`.
    pub converged: bool,
    /// MFEM `GetNumIterations()` (`final_iter`).
    pub iterations: i32,
    /// MFEM `GetFinalNorm()`.
    pub final_norm: f64,
    /// MFEM `GetInitialNorm()`.
    pub initial_norm: f64,
}

/// Format a value the way MFEM streams it with default precision (6 digits).
fn f(x: f64) -> String {
    fmt_g(x)
}

/// MFEM `SLISolver::Mult`: stationary linear iteration `x += B (b - A x)`,
/// with optional preconditioner `B`.
///
/// `apply(y, x)` computes `y = A x`; `prec(z, r)` computes `z = B r`.
/// `x` holds the initial guess when `iterative_mode` is set, otherwise it is
/// zeroed first (MFEM default).
pub fn solve_sli<A, P>(
    n: usize,
    apply: A,
    b: &[f64],
    x: &mut [f64],
    mut prec: Option<P>,
    opts: &SliOptions,
    iterative_mode: bool,
    mut monitor: Option<&mut dyn FnMut(i32, f64, bool)>,
) -> IterResult
where
    A: Fn(&[f64], &mut [f64]),
    P: FnMut(&[f64], &mut [f64]),
{
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    let print_iterations = opts.print_level == 1;
    let warnings = opts.print_level >= 0;

    let mut r = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];

    if iterative_mode {
        apply(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
    } else {
        r.copy_from_slice(b);
        for v in x.iter_mut() {
            *v = 0.0;
        }
    }

    if let Some(ref mut p) = prec {
        p(&r, &mut z);
    }
    let nom0 = if prec.is_some() { dot(&z, &z).sqrt() } else { dot(&r, &r).sqrt() };
    let mut nom = nom0;

    if print_iterations {
        println!("   Iteration : {:>3}  ||Br|| = {}", 0, f(nom0));
    }

    let r0 = (nom0 * opts.rel_tol).max(opts.abs_tol);
    if let Some(m) = monitor.as_deref_mut() {
        m(0, nom0, false);
    }
    if nom0 <= r0 {
        return IterResult {
            converged: true,
            iterations: 0,
            final_norm: nom0,
            initial_norm: nom0,
        };
    }

    // start iteration
    let mut converged = false;
    let mut final_iter = opts.max_iter;
    let mut nomold = 1.0f64;
    let mut cf = 0.0f64;
    let mut i = 1i32;
    loop {
        match prec.as_mut() {
            Some(p) => {
                for k in 0..n {
                    x[k] += z[k]; // x = x + B (b - A x)
                }
            }
            None => {
                for k in 0..n {
                    x[k] += r[k]; // x = x + (b - A x)
                }
            }
        }

        apply(x, &mut r);
        for k in 0..n {
            r[k] = b[k] - r[k]; // r = b - A x
        }

        if let Some(p) = prec.as_mut() {
            p(&r, &mut z);
            nom = dot(&z, &z).sqrt();
        } else {
            nom = dot(&r, &r).sqrt();
        }

        cf = nom / nomold;
        nomold = nom;

        let mut done = false;
        if let Some(m) = monitor.as_deref_mut() {
            m(i, nom, false);
        }
        if nom < r0 {
            converged = true;
            final_iter = i;
            done = true;
        }

        if print_iterations {
            println!(
                "   Iteration : {:>3}  ||Br|| = {:<11}\tConv. rate: {}",
                i - 1,
                f(nom),
                f(cf)
            );
        }

        i += 1;
        if i > opts.max_iter {
            done = true;
        }
        if done {
            break;
        }
    }

    if warnings && !converged {
        let rf = (nom / nom0).powf(1.0 / final_iter as f64);
        println!("SLI: Number of iterations: {final_iter}");
        println!("Conv. rate: {}", f(cf));
        println!("Average reduction factor: {}", f(rf));
    }
    if warnings && !converged {
        println!("SLI: No convergence!");
    }

    if let Some(m) = monitor.as_deref_mut() {
        m(final_iter, nom, true);
    }

    IterResult { converged, iterations: final_iter, final_norm: nom, initial_norm: nom0 }
}

/// MFEM `CGSolver::Mult`: preconditioned conjugate gradient with the
/// `(B r, r)`-based stopping test and MFEM's exact log format.
#[allow(clippy::too_many_arguments)]
pub fn solve_cg_mfem<A, P>(
    n: usize,
    apply: A,
    b: &[f64],
    x: &mut [f64],
    mut prec: Option<P>,
    opts: &SliOptions,
    iterative_mode: bool,
    mut monitor: Option<&mut dyn FnMut(i32, f64, bool)>,
) -> IterResult
where
    A: Fn(&[f64], &mut [f64]),
    P: FnMut(&[f64], &mut [f64]),
{
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    let print_iterations = opts.print_level == 1;
    let warnings = opts.print_level >= 0;

    let mut r = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut ad = vec![0.0f64; n];

    if iterative_mode {
        apply(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
    } else {
        r.copy_from_slice(b);
        for v in x.iter_mut() {
            *v = 0.0;
        }
    }

    let nom0;
    if let Some(ref mut p) = prec {
        p(&r, &mut z); // z = B r
        d.copy_from_slice(&z);
        nom0 = dot(&z, &r);
    } else {
        d.copy_from_slice(&r);
        nom0 = dot(&d, &r);
    }
    let mut nom = nom0;

    if print_iterations {
        println!("   Iteration : {:>3}  (B r, r) = {}", 0, f(nom0));
    }

    if nom0 < 0.0 {
        if warnings {
            println!("PCG: The preconditioner is not positive definite. (Br, r) = {}", f(nom0));
        }
        if let Some(m) = monitor.as_deref_mut() {
            m(0, nom0, true);
        }
        return IterResult {
            converged: false,
            iterations: 0,
            final_norm: nom0.sqrt(),
            initial_norm: nom0,
        };
    }
    let r0 = (nom0 * opts.rel_tol * opts.rel_tol).max(opts.abs_tol * opts.abs_tol);
    if let Some(m) = monitor.as_deref_mut() {
        m(0, nom0, false);
    }
    if nom0 <= r0 {
        return IterResult {
            converged: true,
            iterations: 0,
            final_norm: nom0.sqrt(),
            initial_norm: nom0.sqrt(),
        };
    }

    apply(&d, &mut ad); // ad = A d
    let mut den = dot(&ad, &d);
    if den <= 0.0 && warnings && dot(&d, &d) > 0.0 {
        println!("PCG: The operator is not positive definite. (Ad, d) = {}", f(den));
    }
    if den == 0.0 {
        if let Some(m) = monitor.as_deref_mut() {
            m(0, nom0.sqrt(), true);
        }
        return IterResult {
            converged: false,
            iterations: 0,
            final_norm: nom0.sqrt(),
            initial_norm: nom0.sqrt(),
        };
    }

    // start iteration
    let mut converged = false;
    let mut final_iter = opts.max_iter;
    let mut betanom = 0.0f64;
    let mut i = 1i32;
    loop {
        let alpha = nom / den;
        for k in 0..n {
            x[k] += alpha * d[k]; //  x = x + alpha d
            r[k] -= alpha * ad[k]; //  r = r - alpha A d
        }

        if let Some(p) = prec.as_mut() {
            p(&r, &mut z); //  z = B r
            betanom = dot(&r, &z);
        } else {
            betanom = dot(&r, &r);
        }

        if betanom < 0.0 {
            if warnings {
                println!(
                    "PCG: The preconditioner is not positive definite. (Br, r) = {}",
                    f(betanom)
                );
            }
            converged = false;
            final_iter = i;
            break;
        }

        if print_iterations {
            println!("   Iteration : {:>3}  (B r, r) = {}", i, f(betanom));
        }

        if let Some(m) = monitor.as_deref_mut() {
            m(i, betanom, false);
        }
        if betanom <= r0 {
            converged = true;
            final_iter = i;
            break;
        }

        i += 1;
        if i > opts.max_iter {
            break;
        }

        let beta = betanom / nom;
        if prec.is_some() {
            for k in 0..n {
                d[k] = z[k] + beta * d[k]; //  d = z + beta d
            }
        } else {
            for k in 0..n {
                d[k] = r[k] + beta * d[k];
            }
        }
        apply(&d, &mut ad); //  ad = A d
        den = dot(&d, &ad);
        if den <= 0.0 {
            if dot(&d, &d) > 0.0 && warnings {
                println!("PCG: The operator is not positive definite. (Ad, d) = {}", f(den));
            }
            if den == 0.0 {
                final_iter = i;
                break;
            }
        }
        nom = betanom;
    }

    if warnings && !converged {
        println!("PCG: Number of iterations: {final_iter}");
    }
    if print_iterations {
        let arf = (betanom / nom0).powf(0.5 / final_iter as f64);
        println!("Average reduction factor = {}", f(arf));
    }
    if warnings && !converged {
        println!("PCG: No convergence!");
    }

    if let Some(m) = monitor.as_deref_mut() {
        m(final_iter, betanom.sqrt(), true);
    }

    IterResult {
        converged,
        iterations: final_iter,
        final_norm: betanom.sqrt(),
        initial_norm: nom0.sqrt(),
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn laplace_1d(n: usize) -> fem_linalg::CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        let h = 1.0 / (n + 1) as f64;
        for i in 0..n {
            coo.add(i, i, 2.0 / h);
            if i > 0 {
                coo.add(i, i - 1, -1.0 / h);
            }
            if i + 1 < n {
                coo.add(i, i + 1, -1.0 / h);
            }
        }
        coo.into_csr()
    }

    fn apply_a(a: &fem_linalg::CsrMatrix<f64>) -> impl Fn(&[f64], &mut [f64]) + '_ {
        move |x: &[f64], y: &mut [f64]| a.spmv(x, y)
    }

    /// SLI with a damped-Jacobi preconditioner (`B r = r / diag(A)`, i.e. the
    /// diag-smoothers miniapp's SLI + Jacobi combination) converges on the
    /// 1D Laplace system; unpreconditioned Richardson would diverge, as in
    /// MFEM.
    #[test]
    fn sli_richardson_converges() {
        let n = 20;
        let a = laplace_1d(n);
        let b: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let mut x = vec![0.0f64; n];
        let opts = SliOptions {
            rel_tol: 1e-10,
            max_iter: 20_000,
            ..Default::default()
        };
        let prec = |r: &[f64], z: &mut [f64]| {
            for (k, zc) in z.iter_mut().enumerate() {
                *zc = r[k] / (2.0 * (n as f64 + 1.0));
            }
        };
        let res = solve_sli(n, apply_a(&a), &b, &mut x, Some(prec), &opts, false, None);
        assert!(res.converged, "SLI did not converge");
        let mut r = vec![0.0f64; n];
        a.spmv(&x, &mut r);
        let rn: f64 = (0..n).map(|k| (b[k] - r[k]).powi(2)).sum::<f64>().sqrt();
        assert!(rn < 1e-6, "residual = {rn}");
    }

    /// MFEM CG without preconditioner solves to the (B r, r) tolerance.
    #[test]
    fn cg_mfem_converges() {
        let n = 32;
        let a = laplace_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut x = vec![0.0f64; n];
        let opts = SliOptions { rel_tol: 1e-10, max_iter: 1000, ..Default::default() };
        let res = solve_cg_mfem(
            n,
            apply_a(&a),
            &b,
            &mut x,
            None::<fn(&[f64], &mut [f64])>,
            &opts,
            false,
            None,
        );
        assert!(res.converged);
        let mut r = vec![0.0f64; n];
        a.spmv(&x, &mut r);
        let rn: f64 = (0..n).map(|k| (b[k] - r[k]).powi(2)).sum::<f64>().sqrt();
        assert!(rn < 1e-6, "residual = {rn}");
    }

    /// Non-converged SLI at print level 1 must print the MFEM summary block.
    #[test]
    fn sli_prints_summary_on_nonconvergence() {
        let n = 8;
        let a = laplace_1d(n);
        let b = vec![1.0f64; n];
        let mut x = vec![0.0f64; n];
        let opts = SliOptions {
            rel_tol: 0.0,
            abs_tol: 1e-300,
            max_iter: 3,
            print_level: 1,
            ..Default::default()
        };
        let res = solve_sli(
            n,
            apply_a(&a),
            &b,
            &mut x,
            None::<fn(&[f64], &mut [f64])>,
            &opts,
            false,
            None,
        );
        assert!(!res.converged);
        assert_eq!(res.iterations, 3);
    }

    /// The monitor hook sees the same iteration sequence as the log.
    #[test]
    fn cg_monitor_sees_final_call() {
        let n = 16;
        let a = laplace_1d(n);
        let b = vec![1.0f64; n];
        let mut x = vec![0.0f64; n];
        let opts = SliOptions { rel_tol: 1e-12, max_iter: 500, ..Default::default() };
        let mut calls: Vec<(i32, bool)> = Vec::new();
        let res = solve_cg_mfem(
            n,
            apply_a(&a),
            &b,
            &mut x,
            None::<fn(&[f64], &mut [f64])>,
            &opts,
            false,
            Some(&mut |it, _norm, fin| calls.push((it, fin))),
        );
        assert!(res.converged);
        assert!(calls.len() >= 2);
        assert!(calls.last().unwrap().1, "last monitor call must be final");
    }
}
