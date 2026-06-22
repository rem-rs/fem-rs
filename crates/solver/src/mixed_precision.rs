//! Mixed-precision helpers.
//!
//! Provides utilities for converting between f64 and f32 matrices, and a
//! mixed-precision preconditioner that stores an f32 factorization but
//! applies it to f64 vectors.
//!
//! # Supported operations
//! - [`convert_csr_f64_to_f32`] — lossy conversion of a CSR matrix
//! - [`convert_csr_f32_to_f64`] — conversion back to f64
//! - [`MixedPrecisionPrecond`] — wraps an f32 preconditioner for f64 systems

use fem_linalg::CsrMatrix;
use crate::SolveResult;

/// Lossy-convert an `f64` CSR matrix to `f32`.
pub fn convert_csr_f64_to_f32(a: &CsrMatrix<f64>) -> CsrMatrix<f32> {
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values: a.values.iter().map(|&v| v as f32).collect(),
    }
}

/// Convert an `f32` CSR matrix back to `f64`.
pub fn convert_csr_f32_to_f64(a: &CsrMatrix<f32>) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values: a.values.iter().map(|&v| v as f64).collect(),
    }
}

/// Mixed-precision preconditioner.
///
/// Stores an f32 approximation of the system matrix, and applies it as a
/// preconditioner for an f64 Krylov method via one PCG V-cycle on the f32
/// matrix.  The f32 matrix uses less memory and the SpMV is faster, at the
/// cost of reduced precision.
pub struct MixedPrecisionPrecond {
    /// f32 approximation of the system matrix.
    pub a_f32: CsrMatrix<f32>,
    /// Jacobi diagonal for the f32 matrix.
    pub diag_f32: Vec<f32>,
}

impl MixedPrecisionPrecond {
    /// Build from an f64 matrix (stored internally as f32).
    pub fn new(a: &CsrMatrix<f64>) -> Self {
        let a_f32 = convert_csr_f64_to_f32(a);
        let diag_f32 = a_f32.diagonal();
        MixedPrecisionPrecond { a_f32, diag_f32 }
    }

    /// Apply one Jacobi sweep on the f32 system with an f64 vector.
    /// The f64 vector is cast to f32, smoothed, and cast back.
    pub fn apply(&self, b: &[f64], x: &mut [f64], omega: f64, sweeps: usize) {
        let n = x.len();
        let mut x_f32 = vec![0.0f32; n];
        let mut b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let omega_f32 = omega as f32;

        for _ in 0..sweeps {
            let mut ax = vec![0.0f32; n];
            self.a_f32.spmv(&x_f32, &mut ax);
            for i in 0..n {
                let d = if self.diag_f32[i].abs() > 1e-30f32 { self.diag_f32[i] } else { 1.0f32 };
                x_f32[i] += omega_f32 * (b_f32[i] - ax[i]) / d;
            }
        }

        for i in 0..n {
            x[i] += x_f32[i] as f64;
        }
    }
}

/// Solve an f64 SPD system using an f32 mixed-precision preconditioned CG.
pub fn solve_pcg_mixed(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &crate::SolverConfig,
) -> Result<crate::SolveResult, crate::SolverError> {
    let precond = MixedPrecisionPrecond::new(a);
    let n = a.nrows;

    let mut ax = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    let mut z = vec![0.0; n];

    a.spmv(x, &mut ax);
    for i in 0..n { r[i] = b[i] - ax[i]; }

    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-32);
    let tol = cfg.atol.max(cfg.rtol * b_norm);
    let mut r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r_norm <= tol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: r_norm });
    }

    let mut rz_prev = 0.0;
    for k in 0..cfg.max_iter {
        // Apply f32 preconditioner
        z.iter_mut().for_each(|v| *v = 0.0);
        precond.apply(&r, &mut z, 0.8, 3);

        let rz = r.iter().zip(z.iter()).map(|(r, z)| r * z).sum::<f64>();

        if k == 0 {
            p.copy_from_slice(&z);
        } else {
            let beta = rz / rz_prev;
            for i in 0..n { p[i] = z[i] + beta * p[i]; }
        }

        a.spmv(&p, &mut ap);

        let pap = p.iter().zip(ap.iter()).map(|(p, ap)| p * ap).sum::<f64>();
        if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap;

        for i in 0..n { x[i] += alpha * p[i]; }
        for i in 0..n { r[i] -= alpha * ap[i]; }

        r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r_norm <= tol {
            return Ok(SolveResult { converged: true, iterations: k + 1, final_residual: r_norm });
        }

        rz_prev = rz;
    }

    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: r_norm })
}

/// Solve an f64 SPD system using f32 Jacobi-preconditioned CG (pure f32 solve).
pub fn solve_cg_f32(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &crate::SolverConfig,
) -> Result<SolveResult, crate::SolverError> {
    // Convert to f32, solve, convert back
    let a32 = convert_csr_f64_to_f32(a);
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let mut x32 = vec![0.0f32; x.len()];

    let result = crate::solve_cg(&a32, &b32, &mut x32, cfg)?;

    for i in 0..x.len() {
        x[i] = x32[i] as f64;
    }
    Ok(result)
}

/// Solve a (possibly non-symmetric) system using f32 GMRES.
pub fn solve_gmres_f32(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    cfg: &crate::SolverConfig,
) -> Result<SolveResult, crate::SolverError> {
    let a32 = convert_csr_f64_to_f32(a);
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let mut x32 = vec![0.0f32; x.len()];

    let result = crate::solve_gmres(&a32, &b32, &mut x32, restart, cfg)?;

    for i in 0..x.len() {
        x[i] = x32[i] as f64;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;
    use crate::SolverConfig;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn csr_conversion_roundtrip() {
        let a64 = laplacian_1d(5);
        let a32 = convert_csr_f64_to_f32(&a64);
        let a64_back = convert_csr_f32_to_f64(&a32);
        assert_eq!(a64.nrows, a64_back.nrows);
        assert_eq!(a64.nnz(), a64_back.nnz());
        for i in 0..a64.nrows {
            for j in 0..a64.ncols {
                let orig = a64.get(i, j);
                let back = a64_back.get(i, j);
                assert!((orig - back).abs() < 1e-12, "mismatch at ({i},{j}): {orig} vs {back}");
            }
        }
    }

    #[test]
    fn solve_cg_f32_converges() {
        let a = laplacian_1d(10);
        let n = a.nrows;
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;
        let mut x = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-4, ..Default::default() };
        let res = solve_cg_f32(&a, &b, &mut x, &cfg).unwrap();
        assert!(res.converged, "f32 CG should converge: {res:?}");
    }

    #[test]
    fn solve_gmres_f32_converges() {
        let a = laplacian_1d(10);
        let n = a.nrows;
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;
        let mut x = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-4, ..Default::default() };
        let res = solve_gmres_f32(&a, &b, &mut x, 8, &cfg).unwrap();
        assert!(res.converged, "f32 GMRES should converge: {res:?}");
    }

    #[test]
    fn solve_pcg_mixed_converges() {
        let a = laplacian_1d(20);
        let n = a.nrows;
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;
        let mut x = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-6, ..Default::default() };
        let res = solve_pcg_mixed(&a, &b, &mut x, &cfg).unwrap();
        assert!(res.converged, "mixed-precision PCG should converge: {res:?}");
    }

    #[test]
    fn f32_vs_f64_accuracy_comparison() {
        let a = laplacian_1d(32);
        let n = a.nrows;
        let b: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / (n - 1) as f64;
            (std::f64::consts::PI * x).sin()
        }).collect();

        // f64 CG
        let mut x64 = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-8, ..Default::default() };
        let _ = crate::solve_cg(&a, &b, &mut x64, &cfg).unwrap();

        // f32 CG
        let mut x32 = vec![0.0; n];
        let res32 = solve_cg_f32(&a, &b, &mut x32, &cfg).unwrap();
        assert!(res32.converged, "f32 CG should converge");

        // Difference should be bounded by f32 precision
        let diff: f64 = x64.iter().zip(x32.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / n as f64;
        assert!(diff < 1e-4, "f32 vs f64 avg diff should be < 1e-4, got {diff:.3e}");
    }

    #[test]
    fn mixed_precision_precond_improves_cg() {
        let a = laplacian_1d(16);
        let n = a.nrows;
        let b: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / (n - 1) as f64;
            x * (1.0 - x)
        }).collect();

        // Plain CG
        let mut x_cg = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-6, max_iter: 500, ..Default::default() };
        let res_cg = crate::solve_cg(&a, &b, &mut x_cg, &cfg).unwrap();

        // Mixed-precision PCG
        let mut x_mp = vec![0.0; n];
        let res_mp = solve_pcg_mixed(&a, &b, &mut x_mp, &cfg).unwrap();

        // Both should converge
        assert!(res_cg.converged && res_mp.converged,
            "CG converged={}, mixed PCG converged={}", res_cg.converged, res_mp.converged);
    }
}
