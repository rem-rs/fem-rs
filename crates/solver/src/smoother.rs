//! Diagonal and Gauss-Seidel stationary smoothers (MFEM `DSmoother`,
//! `GSSmoother`).
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
//!
//! [`GsSmoother`] is the Gauss-Seidel counterpart (`GSSmoother`) — the
//! approximate inverse used for the Schur block of the block-diagonal Darcy
//! preconditioner (ex5 / `nurbs_ex5`: `GSSmoother(S)`, `S = B·diag(M)⁻¹·Bᵀ`).

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

// ─── Gauss-Seidel smoother ───────────────────────────────────────────────────

/// MFEM `GSSmoother::GSType` (`linalg/sparsesmoothers.hpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GsType {
    /// Forward sweep, then backward (`GSSmoother::SYMMETRIC`).
    Symmetric,
    /// Forward-only sweep — the `L⁻¹` factor (`GSSmoother::FORWARD`).
    Forward,
    /// Backward-only sweep — the `U⁻¹` factor (`GSSmoother::BACKWARD`).
    Backward,
}

/// MFEM `SparseMatrix::Gauss_Seidel_forw` — the finalized (CSR) path of
/// `linalg/sparsemat.cpp`.
///
/// Rows ascend, and within a row the nonzeros are scanned in **ascending**
/// column order; `y[i] = (x[i] − Σ_{c≠i} A_ic y_c)/A_ii` uses the *current*
/// `y`, so rows already visited contribute updated values (forward GS).
/// Duplicating MFEM, a zero or missing diagonal aborts (MFEM `mfem_error`)
/// unless `x[i] == sum`.
pub fn gauss_seidel_forw(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(a.nrows, a.ncols, "Gauss_Seidel_forw: matrix must be square");
    assert!(x.len() >= a.nrows, "Gauss_Seidel_forw: x too short");
    assert_eq!(a.nrows, y.len(), "Gauss_Seidel_forw: y must match the matrix");
    for i in 0..a.nrows {
        let mut sum = 0.0;
        let mut has_diag = false;
        let mut diag = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[p] as usize;
            if c == i {
                has_diag = true;
                diag = a.values[p];
            } else {
                sum += a.values[p] * y[c];
            }
        }
        if has_diag && diag != 0.0 {
            y[i] = (x[i] - sum) / diag;
        } else if x[i] == sum {
            y[i] = sum;
        } else {
            panic!("Gauss_Seidel_forw: zero or missing diagonal at row {i}");
        }
    }
}

/// MFEM `SparseMatrix::Gauss_Seidel_back` — the finalized (CSR) path.
///
/// Same recurrence as [`gauss_seidel_forw`], but rows descend and within a row
/// the nonzeros are scanned in **descending** column order (MFEM's
/// `for (j = Ip[i+1]-1; j >= Ip[i]; j--)`), which fixes the floating-point
/// summation order.
pub fn gauss_seidel_back(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(a.nrows, a.ncols, "Gauss_Seidel_back: matrix must be square");
    assert!(x.len() >= a.nrows, "Gauss_Seidel_back: x too short");
    assert_eq!(a.nrows, y.len(), "Gauss_Seidel_back: y must match the matrix");
    for i in (0..a.nrows).rev() {
        let mut sum = 0.0;
        let mut has_diag = false;
        let mut diag = 0.0;
        for p in (a.row_ptr[i]..a.row_ptr[i + 1]).rev() {
            let c = a.col_idx[p] as usize;
            if c == i {
                has_diag = true;
                diag = a.values[p];
            } else {
                sum += a.values[p] * y[c];
            }
        }
        if has_diag && diag != 0.0 {
            y[i] = (x[i] - sum) / diag;
        } else if x[i] == sum {
            y[i] = sum;
        } else {
            panic!("Gauss_Seidel_back: zero or missing diagonal at row {i}");
        }
    }
}

/// Gauss-Seidel smoother of a sparse matrix (MFEM `GSSmoother`).
///
/// 1:1 port of `GSSmoother::Mult` (`linalg/sparsesmoothers.cpp`):
/// ```text
///   if (!iterative_mode) y = 0;
///   for (i = 0; i < iterations; i++) {
///      if (type != BACKWARD) oper->Gauss_Seidel_forw(x, y);
///      if (type != FORWARD)  oper->Gauss_Seidel_back(x, y);
///   }
/// ```
/// With the default `Symmetric` type and one iteration this is `y =
/// (UᵀD⁻¹... )` i.e. the symmetric Gauss-Seidel approximate inverse, the
/// `invS` of ex5 / `nurbs_ex5` / `nurbs_solenoidal`.
#[derive(Debug, Clone)]
pub struct GsSmoother {
    a: CsrMatrix<f64>,
    kind: GsType,
    iterations: usize,
    /// MFEM `Solver::iterative_mode` (default `false`, i.e. `y` is zeroed).
    pub iterative_mode: bool,
}

impl GsSmoother {
    /// MFEM `GSSmoother(const SparseMatrix &a, GSType t = SYMMETRIC, int it = 1)`.
    pub fn new(a: &CsrMatrix<f64>, kind: GsType, iterations: usize) -> Self {
        assert!(iterations >= 1, "GSSmoother: iterations must be >= 1");
        Self {
            a: a.clone(),
            kind,
            iterations,
            iterative_mode: false,
        }
    }

    /// The underlying matrix (MFEM `SparseSmoother::oper`).
    pub fn operator(&self) -> &CsrMatrix<f64> {
        &self.a
    }

    /// MFEM `GSSmoother::Mult`.
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        if !self.iterative_mode {
            y.fill(0.0);
        }
        for _ in 0..self.iterations {
            if self.kind != GsType::Backward {
                gauss_seidel_forw(&self.a, x, y);
            }
            if self.kind != GsType::Forward {
                gauss_seidel_back(&self.a, x, y);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    /// The SPD matrix of the MFEM reference probe
    /// (`tmp/d97/probe_d97.cpp`): `[4 -1 0 0; -1 4 -1 0; 0 -1 4 -1; 0 0 -1 3]`.
    fn small_spd() -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(4, 4);
        for (i, j, v) in [
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 3.0),
        ] {
            coo.add(i, j, v);
        }
        coo.into_csr()
    }

    /// MFEM 4.10 `GSSmoother` / `DSmoother` reference values, printed by the
    /// C++ probe on the matrix above with `b = (1, 2, 3, 4)` and
    /// `iterative_mode = false` (`x` from 0):
    /// ```text
    /// DSMOOTHER      0.25 0.5 0.75 1.3333333333333333
    /// GS_SYMMETRIC   0.47176106770833331 0.88704427083333326
    ///                1.2981770833333333 1.6302083333333333
    /// GS_FORWARD     0.25 0.5625 0.890625 1.6302083333333333
    /// GS_BACKWARD    0.44270833333333331 0.77083333333333326
    ///                1.0833333333333333 1.3333333333333333
    /// GS_SYMMETRIC_2 0.49417583147684735 0.9767033259073894
    ///                1.4350522359212241 1.7977244059244792
    /// ```
    #[test]
    fn gs_smoother_matches_mfem_reference() {
        let a = small_spd();
        let b = [1.0, 2.0, 3.0, 4.0];
        let check = |tag: &str, got: &[f64], want: &[f64]| {
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-15 * w.abs().max(1.0),
                    "{tag}[{k}]: got {g:.17e}, MFEM {w:.17e}"
                );
            }
        };

        let mut ds = DiagonalSmoother::new(SmootherType::Jacobi, 1.0, 1);
        ds.setup(&a);
        let mut x = vec![0.0; 4];
        ds.mult(&a, &b, &mut x);
        check(
            "DSmoother",
            &x,
            &[0.25, 0.5, 0.75, 1.3333333333333333],
        );

        let mut x = vec![0.0; 4];
        GsSmoother::new(&a, GsType::Symmetric, 1).mult(&b, &mut x);
        check(
            "GS_SYMMETRIC",
            &x,
            &[
                0.47176106770833331,
                0.88704427083333326,
                1.2981770833333333,
                1.6302083333333333,
            ],
        );

        let mut x = vec![0.0; 4];
        GsSmoother::new(&a, GsType::Forward, 1).mult(&b, &mut x);
        check(
            "GS_FORWARD",
            &x,
            &[0.25, 0.5625, 0.890625, 1.6302083333333333],
        );

        let mut x = vec![0.0; 4];
        GsSmoother::new(&a, GsType::Backward, 1).mult(&b, &mut x);
        check(
            "GS_BACKWARD",
            &x,
            &[
                0.44270833333333331,
                0.77083333333333326,
                1.0833333333333333,
                1.3333333333333333,
            ],
        );

        let mut x = vec![0.0; 4];
        GsSmoother::new(&a, GsType::Symmetric, 2).mult(&b, &mut x);
        check(
            "GS_SYMMETRIC_2",
            &x,
            &[
                0.49417583147684735,
                0.9767033259073894,
                1.4350522359212241,
                1.7977244059244792,
            ],
        );
    }

    /// `iterative_mode = false` (the MFEM `Solver` default) zeroes the output
    /// before sweeping, so the incoming `y` is ignored; `iterative_mode = true`
    /// keeps it as the initial guess.
    #[test]
    fn gs_smoother_iterative_mode() {
        let a = small_spd();
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut s = GsSmoother::new(&a, GsType::Symmetric, 1);

        let mut y0 = vec![7.0; 4];
        s.mult(&b, &mut y0);
        let mut y1 = vec![0.0; 4];
        s.mult(&b, &mut y1);
        assert_eq!(y0, y1, "non-iterative mode must ignore the incoming y");

        s.iterative_mode = true;
        let mut y2 = vec![0.0; 4];
        s.mult(&b, &mut y2);
        let mut y3 = vec![7.0; 4];
        s.mult(&b, &mut y3);
        assert_ne!(y2, y3, "iterative mode must use y as the initial guess");
        // gauss_seidel_forw with x = b and y = 7·1: y_i = (b_i − 7·Σ_{c≠i} A_ic)/A_ii
        let mut y4 = vec![7.0; 4];
        gauss_seidel_forw(&a, &b, &mut y4);
        assert_eq!(y4[0], (1.0 - 7.0 * -1.0) / 4.0);
    }

    /// Both sweeps must abort (MFEM `mfem_error`) on a zero/missing diagonal.
    #[test]
    #[should_panic(expected = "zero or missing diagonal")]
    fn gauss_seidel_zero_diagonal_panics() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 1, 1.0);
        coo.add(1, 1, 1.0);
        let a = coo.into_csr();
        let mut y = vec![0.0; 2];
        gauss_seidel_forw(&a, &[1.0, 1.0], &mut y);
    }

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
