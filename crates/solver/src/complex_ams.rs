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
    precond::{AmsConfig, AmsPrecond, AdsConfig, AdsPrecond},
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
    AmsPrecond::<f64>::new(&a_re, g, config)
        .map_err(|e| format!("AMS setup: {e}"))
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
    AdsPrecond::<f64>::new(&a_re, c, g, config)
        .map_err(|e| format!("ADS setup: {e}"))
}

/// Solve `(A_re + i·A_im) x = b` using GMRES with AMS preconditioner.
///
/// The preconditioner is built from the real part of A with the given
/// discrete gradient matrix G.
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
        a_complex, b_re, b_im, x_re, x_im,
        tol, max_iter, restart, &prec,
    )
}

/// Solve using BiCGSTAB with AMS preconditioner.
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
        a_complex, b_re, b_im, x_re, x_im,
        tol, max_iter, &prec,
    )
}

/// Solve using GMRES with ADS preconditioner.
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
        a_complex, b_re, b_im, x_re, x_im,
        tol, max_iter, restart, &prec,
    )
}

/// Solve using BiCGSTAB with ADS preconditioner.
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
        a_complex, b_re, b_im, x_re, x_im,
        tol, max_iter, &prec,
    )
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Extract the real part of a `ComplexCsr` as a linlvo CSR matrix.
fn real_part_csr(c: &ComplexCsr) -> linlvoCsr<f64> {
    linlvoCsr::from_raw(
        c.nrows, c.ncols,
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

        let ams = AmsPrecond::<f64>::new(&a, &g, AmsConfig::default())
            .expect("AMS setup");
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
        let a = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };
        let real = real_part_csr(&a);
        assert_eq!(real.nrows(), 3);
        assert_eq!(real.nnz(), 3);
    }
}
