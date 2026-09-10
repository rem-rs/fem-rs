//! Complex impedance Maxwell solvers with AMS/ADS preconditioning.
//!
//! For time-harmonic Maxwell (`(K - ω²M + iωC) u = b`), the system is complex
//! but the discrete gradient/curl operators are real (topological).  This module
//! builds preconditioners from the real part of the system matrix and applies
//! them to both real and imaginary residual components (block-diagonal
//! real-part preconditioning).
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
    fem_linalg::complex_csr::solve_gmres_complex_with(
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

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
