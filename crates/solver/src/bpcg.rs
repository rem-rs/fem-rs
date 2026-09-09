//! Bramble-Pasciak Conjugate Gradient (BPCG).
//!
//! 1:1 port of MFEM `miniapps/solvers/bramble_pasciak.{hpp,cpp}` —
//! `BPCGSolver::Mult` (the algorithm body) and the `BramblePasciakSolver`
//! block-operator construction it drives.
//!
//! Solves the saddle-point (Darcy) system
//!
//! ```text
//!     A x = [ M   Bᵀ ] [u] = [f]
//!           [ B    0 ] [p]   [g]
//! ```
//!
//! where `M` is SPD and `B` is the discrete (mixed) divergence.  The system is
//! indefinite, so plain CG does not apply.  Bramble–Pasciak rewrite it as an
//! SPD operator in a modified inner product: choose `Q` such that `M − Q` is
//! SPD (here `Q = α·λ_min·diag(M_T)` element-wise, `0 < α < 1`), set
//!
//! ```text
//!     N = [ invQ  0 ]          X = A·N − I = [ M·invQ − I    0   ]
//!         [   0   0 ]                        [ B·invQ       −I   ]
//! ```
//!
//! `X·A` is then SPD w.r.t. the inner product `(·,·)_N`.  With a block
//! preconditioner
//!
//! ```text
//!     P = [ M0     0   ]·[  I      0   ]   (cpc · tri, applied as P)
//!         [  0    M1   ] [ B·invQ  −I   ]
//! ```
//!
//! the PCG iteration is carried out implicitly on `X·A`, tracking
//! `δᵢ = (P rᵢ, rᵢ) = (t, r_red) − (r_bar, r)` with `r_bar = P·r`,
//! `r_red = N·r`, `t = A·r_bar` — this avoids ever forming `X` or `P`.
//!
//! References:
//! * P. Vassilevski, *Multilevel Block Factorization Preconditioners*
//!   (Appendix F.3), Springer, 2008.
//! * J. Bramble and J. Pasciak, *A Preconditioning Technique for Indefinite
//!   Systems Resulting From Mixed Approximations of Elliptic Problems*,
//!   Math. Comp. 50:1–17, 1988.
//!
//! # Operator contract
//! All three callbacks act on *flat* vectors of length `n = n_u + n_p`
//! (`u` block first, then `p`), exactly like MFEM's `BlockOperator` with
//! `offsets = {0, n_u, n_u + n_p}`:
//! * `apply_a` — `A` (`[[M, Bᵀ], [B, 0]]`),
//! * `apply_p` — the *particular* preconditioner `P = cpc·tri`,
//! * `apply_n` — the *incomplete* (inner-product) preconditioner
//!   `N = diag(invQ, 0)`.

use fem_linalg::{PrintLevel, SolveResult, SolverConfig, SolverError};

use crate::iterative::fmt_g;

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Run the Bramble–Pasciak CG iteration to convergence.
///
/// On entry `x` holds the initial guess; on exit the (approximate) solution.
/// Mirrors MFEM `BPCGSolver::Mult` with `iterative_mode = true` (the MFEM
/// default): the initial residual is `r₀ = b − A·x₀`.
///
/// Convergence is measured on the preconditioned quantity
/// `δ = (P r, r)`: the iteration stops when
/// `δ ≤ max(δ₀·rtol², atol²)`.
///
/// # Returns
/// `SolveResult` when the iteration converged or hit `max_iter` without
/// breakdown (see below), or `SolverError::ConvergenceFailed` when the
/// particular preconditioner `P` is not positive definite w.r.t. the
/// `N`-inner product (`δ < 0` or `γ = (Ap, p) = 0` breakdown, matching
/// MFEM's `converged = false` paths).
///
/// # Panics
/// Never.  Non-positive-definite breakdowns are reported through the
/// error result.
pub fn solve_bpcg<A, P, N>(
    n: usize,
    apply_a: A,
    apply_p: P,
    apply_n: N,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError>
where
    A: Fn(&[f64], &mut [f64]),
    P: Fn(&[f64], &mut [f64]),
    N: Fn(&[f64], &mut [f64]),
{
    if b.len() != n || x.len() != n {
        return Err(SolverError::DimensionMismatch {
            rows: n,
            cols: n,
            rhs: b.len(),
        });
    }
    let rtol = cfg.rtol;
    let atol = cfg.atol;
    let max_iter = cfg.max_iter;
    let level = cfg.effective_print_level();
    let show_each = level >= PrintLevel::Iterations;
    let show_summary = level >= PrintLevel::Summary;

    let mut r = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut r_bar = vec![0.0; n];
    let mut r_red = vec![0.0; n];
    let mut g_red = vec![0.0; n];

    // r = b − A·x   (iterative_mode = true)
    apply_a(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }

    apply_p(&r, &mut r_bar); // r_bar = P·r
    p.copy_from_slice(&r_bar); // p = r_bar
    apply_a(&p, &mut g); // g = A·p
    apply_a(&r_bar, &mut t); // t = A·r_bar
    apply_n(&r, &mut r_red); // r_red = N·r

    // delta = (t, r_red) − (r_bar, r) = (P r, r)
    let mut delta = dot(&t, &r_red) - dot(&r_bar, &r);
    let mut delta0 = delta;
    if show_each {
        println!("   Iteration : {:3}  (P r, r) = {}", 0, fmt_g(delta));
    }
    if delta < 0.0 {
        // P not positive definite — MFEM: converged = false, final_iter = 0.
        if show_summary {
            eprintln!(
                "BPCG: The preconditioner is not positive definite. (Pr, r) = {}",
                fmt_g(delta)
            );
        }
        return Err(SolverError::ConvergenceFailed {
            max_iter: 0,
            residual: (-delta).sqrt(),
        });
    }
    let del0 = (delta * rtol * rtol).max(atol * atol);
    if delta <= del0 {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: delta.sqrt(),
        });
    }

    apply_n(&g, &mut g_red); // g_red = N·g
    let mut gamma = dot(&g, &g_red) - dot(&g, &p); // gamma = (Ap, p)
    if gamma == 0.0 {
        // MFEM: converged = false, final_iter = 0 (nothing advanced yet).
        return Err(SolverError::ConvergenceFailed {
            max_iter: 0,
            residual: delta.sqrt(),
        });
    }
    // gamma < 0 (and != 0): MFEM prints a warning but keeps iterating.

    let mut converged = false;
    let mut final_iter = max_iter;
    let mut i = 1usize;
    loop {
        let alpha = delta0 / gamma;
        for k in 0..n {
            x[k] += alpha * p[k]; // x = x + alpha p
            r[k] -= alpha * g[k]; // r = r − alpha g
        }

        apply_p(&r, &mut r_bar); // r_bar = P·r
        apply_n(&r, &mut r_red); // r_red = N·r
        apply_a(&r_bar, &mut t); // t = A·r_bar
        delta = dot(&t, &r_red) - dot(&r_bar, &r);

        if delta < 0.0 {
            if show_summary {
                eprintln!(
                    "BPCG: The preconditioner is not positive definite. (Pr, r) = {}",
                    fmt_g(delta)
                );
            }
            converged = false;
            final_iter = i;
            break;
        }
        if show_each {
            println!("   Iteration : {:3}  (P r, r) = {}", i, fmt_g(delta));
        }
        if delta <= del0 {
            converged = true;
            final_iter = i;
            break;
        }
        i += 1;
        if i > max_iter {
            break;
        }

        let beta = delta / delta0;
        for k in 0..n {
            p[k] = r_bar[k] + beta * p[k]; // p = r_bar + beta p
            g[k] = t[k] + beta * g[k]; // g = t + beta g
        }
        delta0 = delta;

        apply_n(&g, &mut g_red);
        gamma = dot(&g, &g_red) - dot(&g, &p); // gamma = (Ap, p)
        if gamma <= 0.0 {
            if gamma == 0.0 {
                converged = false;
                final_iter = i;
                break;
            }
            // gamma < 0: MFEM warns and keeps iterating (negative alpha).
            if show_summary && dot(&r_bar, &r_bar) > 0.0 {
                eprintln!(
                    "BPCG: The operator is not positive definite. (Ar, r) = {}",
                    fmt_g(gamma)
                );
            }
        }
    }

    if show_summary {
        if !converged && show_each {
            println!("BPCG: Number of iterations: {}", final_iter);
        }
        let arf = (gamma / delta0).abs().powf(0.5 / final_iter.max(1) as f64);
        println!("Average reduction factor = {}", fmt_g(arf));
    }

    if !converged {
        return Err(SolverError::ConvergenceFailed {
            max_iter: final_iter,
            residual: delta.abs().sqrt(),
        });
    }
    Ok(SolveResult {
        converged: true,
        iterations: final_iter,
        final_residual: delta.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense helper: solve a small linear system by naive Gauss elimination
    /// (test-only) so we can build `M1 = (B·invQ·Bᵀ)⁻¹` analytically.
    fn dense_solve(a: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
        let n = rhs.len();
        let mut m: Vec<Vec<f64>> = a.to_vec();
        let mut b = rhs.to_vec();
        for col in 0..n {
            // partial pivot
            let mut piv = col;
            for r in col + 1..n {
                if m[r][col].abs() > m[piv][col].abs() {
                    piv = r;
                }
            }
            m.swap(col, piv);
            b.swap(col, piv);
            let d = m[col][col];
            for c in col..n {
                m[col][c] /= d;
            }
            b[col] /= d;
            for r in 0..n {
                if r != col {
                    let f = m[r][col];
                    for c in col..n {
                        m[r][c] -= f * m[col][c];
                    }
                    b[r] -= f * b[col];
                }
            }
        }
        b
    }

    /// Deterministic SPD `M` (strictly diagonally dominant tridiagonal, so
    /// `M − 0.5·diag(M)` is SPD) and full-row-rank `B`.
    fn saddle_data(n_u: usize, n_p: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut m = vec![vec![0.0; n_u]; n_u];
        for i in 0..n_u {
            m[i][i] = 2.0;
            if i > 0 {
                m[i][i - 1] = -0.5;
                m[i - 1][i] = -0.5;
            }
        }
        let mut b = vec![vec![0.0; n_u]; n_p];
        for r in 0..n_p {
            for c in 0..n_u {
                b[r][c] = ((r + 1) * (c + 1) % 5) as f64 * 0.1 + 0.01 * (r as f64 + c as f64);
            }
        }
        (m, b)
    }

    /// Flat saddle operator `[[M, Bᵀ], [B, 0]]`.
    fn apply_a(m: &[Vec<f64>], b: &[Vec<f64>], x: &[f64], y: &mut [f64]) {
        let n_u = m.len();
        let n_p = b.len();
        for i in 0..n_u {
            let mut s = 0.0;
            for (j, &mj) in m[i].iter().enumerate() {
                s += mj * x[j];
            }
            for (k, row) in b.iter().enumerate() {
                s += row[i] * x[n_u + k];
            }
            y[i] = s;
        }
        for k in 0..n_p {
            let mut s = 0.0;
            for (c, &bc) in b[k].iter().enumerate() {
                s += bc * x[c];
            }
            y[n_u + k] = s;
        }
    }

    #[test]
    fn bpcg_converges_to_exact_solution_on_saddle_system() {
        let n_u = 6;
        let n_p = 3;
        let n = n_u + n_p;
        let (m, b) = saddle_data(n_u, n_p);

        // Exact solution and rhs.
        let x_star: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64 * 0.5 - 0.2).collect();
        let mut rhs = vec![0.0; n];
        apply_a(&m, &b, &x_star, &mut rhs);

        // Q = 0.5·diag(M) (test stand-in for the elementwise λ_min scaling of
        // MFEM `ConstructMassPreconditioner`); N = diag(invQ, 0).
        let inv_q: Vec<f64> = (0..n_u).map(|i| 1.0 / (0.5 * m[i][i])).collect();
        let apply_n = |x: &[f64], y: &mut [f64]| {
            for i in 0..n_u {
                y[i] = inv_q[i] * x[i];
            }
            for k in 0..n_p {
                y[n_u + k] = 0.0;
            }
        };

        // P = cpc·tri with cpc = diag(invQ, M1), tri = [[I, 0], [B·invQ, −I]],
        // M1 = S⁻¹ with S = B·invQ·Bᵀ (test-only dense inverse).
        let binvq: Vec<Vec<f64>> = b
            .iter()
            .map(|row| (0..n_u).map(|c| row[c] * inv_q[c]).collect())
            .collect();
        let mut s = vec![vec![0.0; n_p]; n_p];
        for r in 0..n_p {
            for c in 0..n_p {
                let mut acc = 0.0;
                for j in 0..n_u {
                    acc += binvq[r][j] * b[c][j];
                }
                s[r][c] = acc;
            }
        }
        let m1_cols: Vec<Vec<f64>> = (0..n_p)
            .map(|c| {
                let mut e = vec![0.0; n_p];
                e[c] = 1.0;
                dense_solve(&s, &e)
            })
            .collect();
        let m1_mat: Vec<Vec<f64>> = (0..n_p)
            .map(|r| (0..n_p).map(|c| m1_cols[c][r]).collect())
            .collect();

        let apply_p = |x: &[f64], y: &mut [f64]| {
            // tri·x
            let mut tx = vec![0.0; n];
            tx[..n_u].copy_from_slice(&x[..n_u]);
            for k in 0..n_p {
                let mut acc = 0.0;
                for j in 0..n_u {
                    acc += binvq[k][j] * x[j];
                }
                tx[n_u + k] = acc - x[n_u + k];
            }
            // cpc·(tri·x) = (invQ·tx_u, M1·tx_p)
            for i in 0..n_u {
                y[i] = inv_q[i] * tx[i];
            }
            for r in 0..n_p {
                let mut acc = 0.0;
                for (c, &m1) in m1_mat[r].iter().enumerate() {
                    acc += m1 * tx[n_u + c];
                }
                y[n_u + r] = acc;
            }
        };

        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 1e-14,
            max_iter: 200,
            verbose: false,
            print_level: PrintLevel::Silent,
        };
        let mut x = vec![0.0; n];
        let res = solve_bpcg(
            n,
            |v, w| apply_a(&m, &b, v, w),
            &apply_p,
            &apply_n,
            &rhs,
            &mut x,
            &cfg,
        );
        let r = res.expect("BPCG should converge on this SPD-transformed saddle system");
        assert!(r.converged);
        assert!(r.iterations < 200);
        let err = x
            .iter()
            .zip(&x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "BPCG solution error too large: {err:.2e}");
    }
}
