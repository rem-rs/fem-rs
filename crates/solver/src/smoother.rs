//! Diagonal stationary smoothers (MFEM `DSmoother`).
//!
//! Ported from MFEM `linalg/sparsesmoothers.hpp/.cpp`:
//! - type 0 (`Jacobi`): `D = diag(A)`.
//! - type 1 (`L1Jacobi`): `D_ii = Σ_j |A_ij|` (row l1 norm).
//! - `set_positive_diagonal(true)`: `D = |D|` (MFEM `SetPositiveDiagonal`).
//!
//! `mult(b, x)` performs `sweeps` stationary iterations `x += scale ·
//! D⁻¹(b − A x)` starting from `x = 0` unless `iterative_mode` is set —
//! exactly MFEM's `SparcSmoother::Mult` loop. This is the inner "linear
//! solver" of mesh-optimizer's `-ls 0` and the l1 preconditioner of `-ls 4`,
//! and the smoother family required by the diag-smoothers miniapps.

use fem_linalg::CsrMatrix;

/// Diagonal smoother flavor (MFEM `DSmoother(int type, ...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmootherType {
    /// `D = diag(A)`.
    Jacobi,
    /// `D_ii = Σ_j |A_ij|`.
    L1Jacobi,
}

/// Fixed-sweep diagonal smoother.
#[derive(Debug, Clone)]
pub struct DiagonalSmoother {
    pub kind: SmootherType,
    /// MFEM `SetScale` (default 1.0).
    pub scale: f64,
    /// MFEM constructor's `it` parameter.
    pub sweeps: usize,
    /// MFEM `SetPositiveDiagonal`.
    pub positive_diagonal: bool,
    /// MFEM `iterative_mode`: keep `x` as the initial guess instead of zero.
    pub iterative_mode: bool,
    d: Vec<f64>,
}

impl DiagonalSmoother {
    pub fn new(kind: SmootherType, scale: f64, sweeps: usize) -> Self {
        Self {
            kind,
            scale,
            sweeps,
            positive_diagonal: false,
            iterative_mode: false,
            d: Vec::new(),
        }
    }

    /// Construct from a precomputed diagonal (MFEM
    /// `OperatorJacobiSmoother(d, ess_tdofs)` with a caller-supplied `d`).
    ///
    /// The diag-smoothers miniapps build `d = |A| · 1` via `AbsMult`; MFEM's
    /// setup then overrides `dinv[i] = damping` at essential dofs, which
    /// corresponds to `d[ess] = 1 / damping` (damping defaults to 1).
    pub fn from_diagonal(d: Vec<f64>) -> Self {
        Self {
            kind: SmootherType::Jacobi,
            scale: 1.0,
            sweeps: 1,
            positive_diagonal: false,
            iterative_mode: false,
            d,
        }
    }

    /// Compute the (possibly positified) diagonal from `a` (MFEM `SetOperator`).
    pub fn setup(&mut self, a: &CsrMatrix<f64>) {
        let n = a.nrows;
        let mut d = vec![0.0f64; n];
        for row in 0..n {
            let (s, e) = (a.row_ptr[row], a.row_ptr[row + 1]);
            match self.kind {
                SmootherType::Jacobi => {
                    for k in s..e {
                        if a.col_idx[k] as usize == row {
                            d[row] = a.values[k];
                        }
                    }
                }
                SmootherType::L1Jacobi => {
                    let mut sum = 0.0;
                    for k in s..e {
                        sum += a.values[k].abs();
                    }
                    d[row] = sum;
                }
            }
        }
        if self.positive_diagonal {
            for v in d.iter_mut() {
                *v = v.abs();
            }
        }
        self.d = d;
    }

    /// Diagonal accessor (after `setup`).
    pub fn diagonal(&self) -> &[f64] {
        &self.d
    }

    /// MFEM `SparcSmoother::Mult`: `sweeps` stationary iterations.
    pub fn mult(&self, a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64]) {
        assert_eq!(self.d.len(), b.len(), "call setup() before mult()");
        if !self.iterative_mode {
            for v in x.iter_mut() {
                *v = 0.0;
            }
        }
        let n = b.len();
        let mut tmp = vec![0.0f64; n];
        for _ in 0..self.sweeps {
            a.spmv(x, &mut tmp);
            for i in 0..n {
                x[i] += self.scale * (b[i] - tmp[i]) / self.d[i];
            }
        }
    }
}

/// Convenience: `iters` stationary l1-Jacobi sweeps solving `A x = b` from
/// `x = 0` (mesh-optimizer `-ls 0`, `DSmoother(1, 1.0, iters)`).
pub fn solve_l1_jacobi(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], iters: usize) {
    let mut s = DiagonalSmoother::new(SmootherType::L1Jacobi, 1.0, iters);
    s.setup(a);
    s.mult(a, b, x);
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    fn diffusion_1d(n: usize) -> CsrMatrix<f64> {
        // -u'' on [0,1], Dirichlet eliminated rows kept (diag 1).
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

    #[test]
    fn jacobi_diagonal_matches_mfem_types() {
        let a = diffusion_1d(5);
        let mut ds = DiagonalSmoother::new(SmootherType::Jacobi, 1.0, 1);
        ds.setup(&a);
        assert!((ds.diagonal()[2] - 2.0 / 0.16666666666666666).abs() < 1e-12);

        let mut l1 = DiagonalSmoother::new(SmootherType::L1Jacobi, 1.0, 1);
        l1.setup(&a);
        // row l1 = |2/h| + 2·|1/h| = 4/h.
        assert!((l1.diagonal()[2] - 4.0 / 0.16666666666666666).abs() < 1e-12);
    }

    #[test]
    fn positive_diagonal_absorbs_sign() {
        // Symmetric matrix with a negative diagonal entry (e.g. mass-weighted
        // curl-curl block): Jacobi with positive_diagonal uses |A_ii|.
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, -3.0);
        coo.add(0, 1, 1.0);
        coo.add(1, 0, 1.0);
        coo.add(1, 1, 4.0);
        let a = coo.into_csr();
        let mut ds = DiagonalSmoother::new(SmootherType::Jacobi, 1.0, 1);
        ds.positive_diagonal = true;
        ds.setup(&a);
        assert_eq!(ds.diagonal()[0], 3.0);
        assert_eq!(ds.diagonal()[1], 4.0);
    }

    #[test]
    fn stationary_sweeps_solve_spd_system() {
        let n = 32;
        let a = diffusion_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut x = vec![0.0f64; n];
        let mut s = DiagonalSmoother::new(SmootherType::L1Jacobi, 1.0, 20_000);
        s.setup(&a);
        s.mult(&a, &b, &mut x);
        // l1-Jacobi on this M-matrix is the half-damped Jacobi iteration
        // (D = 2·diag(A)), spectral radius ≈ 0.9977 for n = 32; 20k sweeps
        // give ~e^-45 amplification of the initial residual.
        let mut r = vec![0.0f64; n];
        a.spmv(&x, &mut r);
        let res: f64 = (0..n).map(|i| (b[i] - r[i]).powi(2)).sum::<f64>().sqrt();
        assert!(res < 1e-3, "residual after sweeps = {}", res);
    }

    #[test]
    fn single_sweep_matches_manual_formula() {
        let a = diffusion_1d(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut s = DiagonalSmoother::new(SmootherType::L1Jacobi, 0.5, 1);
        s.setup(&a);
        let mut x = vec![0.0f64; 4];
        s.mult(&a, &b, &mut x);
        for i in 0..4 {
            let row_l1: f64 = (a.row_ptr[i]..a.row_ptr[i + 1])
                .map(|k| a.values[k].abs())
                .sum();
            assert!((x[i] - 0.5 * b[i] / row_l1).abs() < 1e-14);
        }
    }
}
