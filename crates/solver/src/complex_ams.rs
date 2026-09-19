//! Complex impedance Maxwell solvers with AMS/ADS preconditioning.
//!
//! For time-harmonic Maxwell (`(K - ω²M + iωC) u = b`), the system is complex
//! but the discrete gradient/curl operators are real (topological).  This module
//! builds preconditioners from the real part of the system matrix and applies
//! them to both real and imaginary residual components (block-diagonal
//! real-part preconditioning).
//!
//! The AMS GMRES driver ([`solve_gmres_ams_complex`]) runs its Krylov
//! iteration **right**-preconditioned
//! ([`solve_gmres_complex_right_prec`]) so the convergence check and the
//! minimized residual are the true residual `‖b − A x‖/‖b‖`, not the
//! preconditioned one (defect D73/D406).  The ADS and BiCGSTAB drivers still
//! use the left-preconditioned `fem_linalg` kernels.
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::complex_ams::{build_ams_precond, solve_gmres_ams_complex};
//!
//! let (iters, res) = solve_gmres_ams_complex(&a_complex, &g_real,
//!     &b_re, &b_im, &mut x_re, &mut x_im, 1e-8, 200, 30,
//!     AmsConfig::default()).unwrap();
//! ```

use fem_linalg::complex_csr::ComplexCsr;
use linlvo::{
    precond::{AdsConfig, AdsPrecond, AmsConfig, AmsPrecond},
    sparse::CsrMatrix as linlvoCsr,
    DenseVec, Preconditioner,
};

/// Build a complex AMS preconditioner from the real part of A.
///
/// # Arguments
/// * `a_complex` — complex H(curl) system matrix (edge DOFs)
/// * `g`         — discrete gradient (vertices → edges, in linlvo format)
/// * `config`    — AMS configuration (coarse solver type, damping, etc.)
pub fn build_ams_precond(
    a_complex: &ComplexCsr,
    g: &linlvoCsr<f64>,
    config: AmsConfig,
) -> Result<AmsPrecond<f64>, String> {
    let a_re = real_part_csr(a_complex);
    AmsPrecond::<f64>::new(&a_re, g, config).map_err(|e| format!("AMS setup: {e}"))
}

/// Create a preconditioner closure for use with
/// [`solve_gmres_complex_with`] or [`solve_bicgstab_complex_with`].
///
/// The returned closure applies the real AMS preconditioner independently
/// to the real and imaginary part of the complex residual (block-diagonal
/// real-part preconditioning).
pub fn make_ams_closure(
    ams: &AmsPrecond<f64>,
) -> impl Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>) + '_ {
    move |r_re: &[f64], r_im: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let n_ = r_re.len();
        let mut z_re = DenseVec::zeros(n_);
        let mut z_im = DenseVec::zeros(n_);
        ams.apply_precond(&DenseVec::from_vec(r_re.to_vec()), &mut z_re);
        ams.apply_precond(&DenseVec::from_vec(r_im.to_vec()), &mut z_im);
        (z_re.into_vec(), z_im.into_vec())
    }
}

/// Create a preconditioner closure for ADS (H(div) systems).
pub fn make_ads_closure(
    ads: &AdsPrecond<f64>,
) -> impl Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>) + '_ {
    move |r_re: &[f64], r_im: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let n_ = r_re.len();
        let mut z_re = DenseVec::zeros(n_);
        let mut z_im = DenseVec::zeros(n_);
        ads.apply_precond(&DenseVec::from_vec(r_re.to_vec()), &mut z_re);
        ads.apply_precond(&DenseVec::from_vec(r_im.to_vec()), &mut z_im);
        (z_re.into_vec(), z_im.into_vec())
    }
}

/// Build a complex ADS preconditioner from the real part of A.
pub fn build_ads_precond(
    a_complex: &ComplexCsr,
    c: &linlvoCsr<f64>,
    g: &linlvoCsr<f64>,
    config: AdsConfig,
) -> Result<AdsPrecond<f64>, String> {
    let a_re = real_part_csr(a_complex);
    AdsPrecond::<f64>::new(&a_re, c, g, config).map_err(|e| format!("ADS setup: {e}"))
}

/// Solve `(A_re + i·A_im) x = b` using GMRES with AMS preconditioner.
///
/// The preconditioner is built from the real part of A with the given
/// discrete gradient matrix G.
///
/// The GMRES iteration is **right**-preconditioned (see
/// [`solve_gmres_complex_right_prec`]): the minimized and monitored residual
/// is the true residual `‖b − A x‖ / ‖b‖`.  This is the defect-D73/D406 fix —
/// a left-preconditioned in-cycle check exits every restart cycle when the
/// *preconditioned* residual crosses `tol`, which under the `hpc_default`
/// cycle underestimates the true residual by ≈2 orders of magnitude and
/// plateaus the solve at `‖r‖/‖b‖ ≈ 60·tol` regardless of budget.
#[allow(clippy::too_many_arguments)]
pub fn solve_gmres_ams_complex(
    a_complex: &ComplexCsr,
    g: &linlvoCsr<f64>,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
    ams_config: AmsConfig,
) -> Result<(usize, f64), String> {
    let ams = build_ams_precond(a_complex, g, ams_config)?;
    let prec = make_ams_closure(&ams);
    solve_gmres_complex_right_prec(
        a_complex, b_re, b_im, x_re, x_im, tol, max_iter, restart, &prec,
    )
}

/// Solve using BiCGSTAB with AMS preconditioner.
#[allow(clippy::too_many_arguments)]
pub fn solve_bicgstab_ams_complex(
    a_complex: &ComplexCsr,
    g: &linlvoCsr<f64>,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    ams_config: AmsConfig,
) -> Result<(usize, f64), String> {
    let ams = build_ams_precond(a_complex, g, ams_config)?;
    let prec = make_ams_closure(&ams);
    fem_linalg::complex_csr::solve_bicgstab_complex_with(
        a_complex, b_re, b_im, x_re, x_im, tol, max_iter, &prec,
    )
}

/// Solve using GMRES with ADS preconditioner.
#[allow(clippy::too_many_arguments)]
pub fn solve_gmres_ads_complex(
    a_complex: &ComplexCsr,
    c: &linlvoCsr<f64>,
    g: &linlvoCsr<f64>,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
    ads_config: AdsConfig,
) -> Result<(usize, f64), String> {
    let ads = build_ads_precond(a_complex, c, g, ads_config)?;
    let prec = make_ads_closure(&ads);
    fem_linalg::complex_csr::solve_gmres_complex_with(
        a_complex, b_re, b_im, x_re, x_im, tol, max_iter, restart, &prec,
    )
}

/// Solve using BiCGSTAB with ADS preconditioner.
#[allow(clippy::too_many_arguments)]
pub fn solve_bicgstab_ads_complex(
    a_complex: &ComplexCsr,
    c: &linlvoCsr<f64>,
    g: &linlvoCsr<f64>,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    ads_config: AdsConfig,
) -> Result<(usize, f64), String> {
    let ads = build_ads_precond(a_complex, c, g, ads_config)?;
    let prec = make_ads_closure(&ads);
    fem_linalg::complex_csr::solve_bicgstab_complex_with(
        a_complex, b_re, b_im, x_re, x_im, tol, max_iter, &prec,
    )
}

/// Complex Conjugate Gradient (CG) solver for complex symmetric systems.
///
/// Solves `A * x = b` where A is complex symmetric (A = A^T, not A = A^H).
/// Uses the complex inner product (u, v) = u^T v (not conjugate).
///
/// This is suitable for DPG systems which are complex coercive but not
/// necessarily Hermitian.
pub fn solve_cg_complex(
    a: &ComplexCsr,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut [f64],
    x_im: &mut [f64],
    tol: f64,
    max_iter: usize,
) -> Result<(usize, f64), String> {
    let n = a.nrows;
    if b_re.len() != n || x_re.len() != n {
        return Err(format!("Dimension mismatch: n={}, b={}, x={}", n, b_re.len(), x_re.len()));
    }

    // Helper: complex symmetric inner product (u, v) = u^T v (NOT conjugate)
    let dot = |ur: &[f64], ui: &[f64], vr: &[f64], vi: &[f64]| -> (f64, f64) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for i in 0..n {
            // (ur + i*ui) * (vr + i*vi) = (ur*vr - ui*vi) + i*(ur*vi + ui*vr)
            re += ur[i] * vr[i] - ui[i] * vi[i];
            im += ur[i] * vi[i] + ui[i] * vr[i];
        }
        (re, im)
    };

    // Helper: complex axpy y += alpha * x
    let axpy = |yr: &mut [f64], yi: &mut [f64], xr: &[f64], xi: &[f64], alpha_re: f64, alpha_im: f64| {
        for i in 0..n {
            // (alpha_re + i*alpha_im) * (xr + i*xi) = (alpha_re*xr - alpha_im*xi) + i*(alpha_re*xi + alpha_im*xr)
            let dr = alpha_re * xr[i] - alpha_im * xi[i];
            let di = alpha_re * xi[i] + alpha_im * xr[i];
            yr[i] += dr;
            yi[i] += di;
        }
    };

    // r0 = b - A*x0
    let mut r_re = vec![0.0f64; n];
    let mut r_im = vec![0.0f64; n];
    let mut ap_re = vec![0.0f64; n];
    let mut ap_im = vec![0.0f64; n];
    a.spmv_into(x_re, x_im, &mut ap_re, &mut ap_im);
    for i in 0..n {
        r_re[i] = b_re[i] - ap_re[i];
        r_im[i] = b_im[i] - ap_im[i];
    }

    let mut p_re = r_re.clone();
    let mut p_im = r_im.clone();

    let mut rs_old = dot(&r_re, &r_im, &r_re, &r_im);
    let norm_b = dot(b_re, b_im, b_re, b_im);
    let tol_sq = (tol * tol).max(1e-32) * (norm_b.0 + norm_b.1).max(1.0);

    for iter in 0..max_iter {
        // Check convergence
        let rs_norm_sq = rs_old.0 * rs_old.0 + rs_old.1 * rs_old.1;
        if rs_norm_sq <= tol_sq {
            // Copy result back
            x_re.copy_from_slice(&p_re);
            x_im.copy_from_slice(&p_im);
            return Ok((iter, rs_norm_sq.sqrt()));
        }

        // alpha_k = (r_k, r_k) / (p_k, A*p_k)
        a.spmv_into(&p_re, &p_im, &mut ap_re, &mut ap_im);
        let pap = dot(&p_re, &p_im, &ap_re, &ap_im);
        let pap_norm_sq = pap.0 * pap.0 + pap.1 * pap.1;
        if pap_norm_sq < 1e-64 {
            return Err("CG breakdown: (p, A*p) is near zero".to_string());
        }
        // alpha = rs_old / pap = (rs_old_re + i*rs_old_im) / (pap_re + i*pap_im)
        // = (rs_old_re + i*rs_old_im) * (pap_re - i*pap_im) / |pap|^2
        let alpha_re = (rs_old.0 * pap.0 + rs_old.1 * pap.1) / pap_norm_sq;
        let alpha_im = (rs_old.1 * pap.0 - rs_old.0 * pap.1) / pap_norm_sq;

        // x_{k+1} = x_k + alpha_k * p_k
        axpy(x_re, x_im, &p_re, &p_im, alpha_re, alpha_im);

        // r_{k+1} = r_k - alpha_k * A*p_k
        axpy(&mut r_re, &mut r_im, &ap_re, &ap_im, -alpha_re, -alpha_im);

        let rs_new = dot(&r_re, &r_im, &r_re, &r_im);
        let beta_norm_sq = rs_new.0 * rs_new.0 + rs_new.1 * rs_new.1;
        if beta_norm_sq < 1e-64 {
            return Ok((iter + 1, beta_norm_sq.sqrt()));
        }
        // beta_k = (r_{k+1}, r_{k+1}) / (r_k, r_k)
        let beta_re = (rs_new.0 * rs_old.0 + rs_new.1 * rs_old.1) / (rs_old.0 * rs_old.0 + rs_old.1 * rs_old.1);
        let beta_im = (rs_new.1 * rs_old.0 - rs_new.0 * rs_old.1) / (rs_old.0 * rs_old.0 + rs_old.1 * rs_old.1);

        // p_{k+1} = r_{k+1} + beta_k * p_k
        let p_old_re = p_re.clone();
        let p_old_im = p_im.clone();
        for i in 0..n {
            p_re[i] = r_re[i] + beta_re * p_old_re[i] - beta_im * p_old_im[i];
            p_im[i] = r_im[i] + beta_re * p_old_im[i] + beta_im * p_old_re[i];
        }

        rs_old = rs_new;
    }

    let rs_norm_sq = rs_old.0 * rs_old.0 + rs_old.1 * rs_old.1;
    Err(format!("CG did not converge in {} iterations (residual={:.3e})", max_iter, rs_norm_sq.sqrt()))
}

// ─── Right-preconditioned complex GMRES (true-residual driver) ────────────────

/// Restarted **right**-preconditioned GMRES for `A x = b` with a
/// caller-provided preconditioner `M ≈ A`.
///
/// `apply_prec(r_re, r_im) -> (z_re, z_im)` must compute `z = M⁻¹·r`.
///
/// The Krylov space is built on `A·M⁻¹` (`wⱼ = A·(M⁻¹·vⱼ)`) and the correction
/// is applied as `x ← x + M⁻¹·V·y`, so the minimized — and recursively
/// estimated — residual is the **true** residual `‖b − A x‖ / ‖b‖`.  With
/// *left* preconditioning (the usual `w = M⁻¹·(A·v)` form) the in-cycle
/// estimate is `‖M⁻¹(b − A x)‖`, which for auxiliary-space cycles whose
/// preconditioned residual is much smaller than the true residual (the
/// `hpc_default` Jacobi+additive cycle is nearly singular in its coarse mode)
/// crosses `tol` orders of magnitude too early: every restart cycle then exits
/// after a few iterations and the solve plateaus at `≈ 60·tol` no matter the
/// budget — the D73/D406 plateau (16×16 complex Maxwell, 6.0e-5 for tol 1e-6,
/// while tol=1e-8/1e-10/1e-12 probes land at 6.8e-7/6.4e-9/1.2e-10, i.e. the
/// plateau tracks the tolerance, not the budget).
///
/// The true residual is recomputed from scratch at the end of every restart
/// cycle, so the cheap in-cycle estimate is always confirmed against the
/// quantity the caller actually cares about.  The best iterate is returned
/// even on non-convergence (soft failure), as `(iterations, rel_res)`.
#[allow(clippy::too_many_arguments)]
fn solve_gmres_complex_right_prec<F>(
    a: &ComplexCsr,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
    apply_prec: &F,
) -> Result<(usize, f64), String>
where
    F: Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
{
    let n = a.nrows;
    assert_eq!(b_re.len(), n);
    assert_eq!(b_im.len(), n);
    if n == 0 {
        return Ok((0, 0.0));
    }
    if x_re.len() != n {
        *x_re = vec![0.0; n];
    }
    if x_im.len() != n {
        *x_im = vec![0.0; n];
    }

    let dot2 = |ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]| -> (f64, f64) {
        // (a, b) = sum conj(a_k) * b_k
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for i in 0..n {
            sr += ar[i] * br[i] + ai[i] * bi[i];
            si += ar[i] * bi[i] - ai[i] * br[i];
        }
        (sr, si)
    };

    let norm2 = |vr: &[f64], vi: &[f64]| -> f64 {
        vr.iter().zip(vi.iter()).map(|(u, v)| u * u + v * v).sum::<f64>().sqrt()
    };

    // Initial TRUE residual r = b − A x
    let mut r_re = vec![0.0_f64; n];
    let mut r_im = vec![0.0_f64; n];
    a.spmv_into(x_re, x_im, &mut r_re, &mut r_im);
    for i in 0..n {
        r_re[i] = b_re[i] - r_re[i];
        r_im[i] = b_im[i] - r_im[i];
    }

    let b_norm = norm2(b_re, b_im).max(1e-300);
    let mut res = norm2(&r_re, &r_im) / b_norm;
    if res < tol {
        return Ok((0, res));
    }

    let mut total_iter = 0usize;
    let m = restart.max(1).min(n);

    'restart: while total_iter < max_iter {
        // Arnoldi on A·M⁻¹ starting from the true residual v₀ = r/‖r‖.
        let r_norm = norm2(&r_re, &r_im);
        if r_norm <= 1e-300 {
            break 'restart;
        }
        let mut v_re: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut v_im: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v_re.push(r_re.iter().map(|&x| x / r_norm).collect());
        v_im.push(r_im.iter().map(|&x| x / r_norm).collect());
        let mut h = vec![vec![(0.0_f64, 0.0_f64); m]; m + 1]; // H[i][j]
        let mut givens: Vec<[(f64, f64); 4]> = Vec::with_capacity(m);
        let mut g = vec![(0.0_f64, 0.0_f64); m + 1];
        g[0] = (r_norm, 0.0);

        let mut k = 0usize; // number of Arnoldi columns built
        for j in 0..m {
            k = j + 1;
            total_iter += 1;

            // w = A·(M⁻¹·v_j)  — right-preconditioned operator
            let (z_re, z_im) = apply_prec(&v_re[j], &v_im[j]);
            let mut w_re = vec![0.0; n];
            let mut w_im = vec![0.0; n];
            a.spmv_into(&z_re, &z_im, &mut w_re, &mut w_im);

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let (hr, hi) = dot2(&v_re[i], &v_im[i], &w_re, &w_im);
                h[i][j] = (hr, hi);
                for kk in 0..n {
                    w_re[kk] -= hr * v_re[i][kk] - hi * v_im[i][kk];
                    w_im[kk] -= hr * v_im[i][kk] + hi * v_re[i][kk];
                }
            }
            let w_norm = norm2(&w_re, &w_im);
            h[j + 1][j] = (w_norm, 0.0);

            if w_norm > 1e-300 {
                v_re.push(w_re.iter().map(|&x| x / w_norm).collect());
                v_im.push(w_im.iter().map(|&x| x / w_norm).collect());
            } else {
                // happy breakdown: the Krylov space already spans the solution
                v_re.push(vec![0.0; n]);
                v_im.push(vec![0.0; n]);
            }

            // Apply previous Givens rotations to the new column
            for (i, gt) in givens.iter().enumerate().take(j) {
                let h1 = h[i][j];
                let h2 = h[i + 1][j];
                h[i][j] = givens_row_apply(gt[0], gt[1], h1, h2);
                h[i + 1][j] = givens_row_apply(gt[2], gt[3], h1, h2);
            }

            // New Givens rotation annihilating h[j+1][j] (same full 2×2
            // complex unitary form as the left-preconditioned driver)
            let h1 = h[j][j];
            let h2 = h[j + 1][j];
            let rot = (h1.0 * h1.0 + h1.1 * h1.1 + h2.0 * h2.0 + h2.1 * h2.1).sqrt();
            let gt = if rot < 1e-300 {
                [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
            } else {
                [
                    (h1.0 / rot, -h1.1 / rot),
                    (h2.0 / rot, -h2.1 / rot),
                    (-h2.0 / rot, -h2.1 / rot),
                    (h1.0 / rot, h1.1 / rot),
                ]
            };
            givens.push(gt);
            h[j][j] = givens_row_apply(gt[0], gt[1], h1, h2);
            h[j + 1][j] = (0.0, 0.0);

            let g1 = g[j];
            let g2 = g[j + 1];
            g[j] = givens_row_apply(gt[0], gt[1], g1, g2);
            g[j + 1] = givens_row_apply(gt[2], gt[3], g1, g2);

            // Residual estimate |g[j+1]| — for right preconditioning this IS
            // the true residual ‖b − A x_j‖ of the current partial iterate.
            res = g[j + 1].0.hypot(g[j + 1].1) / b_norm;
            if res < tol || total_iter >= max_iter {
                break;
            }
        }

        // Back-substitution: solve upper triangular H * y = g
        let mut y_re = vec![0.0_f64; k];
        let mut y_im = vec![0.0_f64; k];
        for i in (0..k).rev() {
            let (mut rr, mut ri) = (g[i].0, g[i].1);
            for jj in (i + 1)..k {
                let (hr, hi) = h[i][jj];
                rr -= hr * y_re[jj] - hi * y_im[jj];
                ri -= hr * y_im[jj] + hi * y_re[jj];
            }
            let (hr, hi) = h[i][i];
            let mag2 = hr * hr + hi * hi;
            if mag2 > 1e-300 {
                y_re[i] = (rr * hr + ri * hi) / mag2;
                y_im[i] = (ri * hr - rr * hi) / mag2;
            }
        }

        // Update x ← x + M⁻¹·V·y (the preconditioner maps Krylov corrections
        // back to solution space)
        for j in 0..k {
            let (z_re, z_im) = apply_prec(&v_re[j], &v_im[j]);
            for i in 0..n {
                x_re[i] += y_re[j] * z_re[i] - y_im[j] * z_im[i];
                x_im[i] += y_re[j] * z_im[i] + y_im[j] * z_re[i];
            }
        }

        // Refresh the true residual (recursive estimate → confirmed here)
        a.spmv_into(x_re, x_im, &mut r_re, &mut r_im);
        for i in 0..n {
            r_re[i] = b_re[i] - r_re[i];
            r_im[i] = b_im[i] - r_im[i];
        }
        res = norm2(&r_re, &r_im) / b_norm;

        if res < tol {
            break 'restart;
        }
    }

    Ok((total_iter, res))
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Apply one row `(g0, g1)` of a complex 2×2 Givens rotation to the pair
/// `(h1, h2)`: `out = g0·h1 + g1·h2` (component-wise complex arithmetic).
///
/// Local copy of the private helper in `fem_linalg::complex_csr` (which cannot
/// be re-used across crates) for [`solve_gmres_complex_right_prec`].
#[inline]
fn givens_row_apply(
    g0: (f64, f64),
    g1: (f64, f64),
    h1: (f64, f64),
    h2: (f64, f64),
) -> (f64, f64) {
    (
        g0.0 * h1.0 - g0.1 * h1.1 + g1.0 * h2.0 - g1.1 * h2.1,
        g0.0 * h1.1 + g0.1 * h1.0 + g1.0 * h2.1 + g1.1 * h2.0,
    )
}

/// Extract the real part of a `ComplexCsr` as a linlvo CSR matrix.
fn real_part_csr(c: &ComplexCsr) -> linlvoCsr<f64> {
    linlvoCsr::from_raw(
        c.nrows,
        c.ncols,
        c.row_ptr.clone(),
        c.col_idx.iter().map(|&x| x as usize).collect(),
        c.re_vals.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::complex_csr::ComplexCsr;

    #[test]
    fn ams_closure_applies() {
        let n = 3usize;
        let row_ptr = vec![0usize, 2, 4, 5];
        let col_idx = vec![0usize, 1, 0, 1, 2];
        let re_vals = vec![4.0, -1.0, -1.0, 4.0, 1.0];
        let a = linlvoCsr::from_raw(n, n, row_ptr, col_idx, re_vals);

        let g_row = vec![0usize, 1, 2, 3];
        let g_col = vec![0usize, 1, 2];
        let g_val = vec![1.0, 1.0, 1.0];
        let g = linlvoCsr::from_raw(n, n, g_row, g_col, g_val);

        let ams = AmsPrecond::<f64>::new(&a, &g, AmsConfig::default()).expect("AMS setup");
        let prec = make_ams_closure(&ams);

        let r_re = vec![1.0, 2.0, 3.0];
        let r_im = vec![0.5, 1.5, 2.5];
        let (z_re, z_im) = prec(&r_re, &r_im);
        assert_eq!(z_re.len(), 3);
        assert_eq!(z_im.len(), 3);
        assert!(z_re.iter().all(|v| v.is_finite()));
        assert!(z_im.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn complex_csr_real_part_extraction() {
        let row_ptr = vec![0usize, 1, 2, 3];
        let col_idx = vec![0u32, 1, 2];
        let re_vals = vec![2.0, 3.0, 1.0];
        let im_vals = vec![1.0, -1.0, 2.0];
        let a = ComplexCsr {
            nrows: 3,
            ncols: 3,
            row_ptr,
            col_idx,
            re_vals,
            im_vals,
        };
        let real = real_part_csr(&a);
        assert_eq!(real.nrows(), 3);
        assert_eq!(real.nnz(), 3);
    }
}
