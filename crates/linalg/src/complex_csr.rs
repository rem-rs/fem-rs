//! Complex-valued sparse matrix (CSR format).
//!
//! [`ComplexCsr`] stores a complex matrix `A = A_re + i·A_im` with separate
//! real and imaginary value arrays sharing the same sparsity pattern.  This
//! avoids any dependence on `num-complex` or changes to the `Scalar` trait.
//!
//! ## Layout
//!
//! ```text
//! A[i,j] = re_vals[ptr] + i * im_vals[ptr]   where ptr ∈ [row_ptr[i], row_ptr[i+1])
//! ```
//!
//! ## Operations
//! - `spmv_complex(x_re, x_im, y_re, y_im)` — y += A * x (complex multiply-accumulate)
//! - `spmv_complex_into` — y = A * x (overwrites y)
//! - `axpy_complex` — z = alpha * A * x + beta * y helper
//! - Diagonal extraction, row-scaling, transpose construction

/// Complex sparse matrix in CSR format.
///
/// Sparsity pattern (row_ptr, col_idx) is shared by both real and imaginary
/// parts.  All indices are zero-based.
#[derive(Debug, Clone)]
pub struct ComplexCsr {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointer array, length `nrows + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices, length `nnz`.
    pub col_idx: Vec<u32>,
    /// Real part of each non-zero, length `nnz`.
    pub re_vals: Vec<f64>,
    /// Imaginary part of each non-zero, length `nnz`.
    pub im_vals: Vec<f64>,
}

impl ComplexCsr {
    /// Construct from separate real/imaginary CSR matrices that share the same
    /// sparsity pattern.  Panics if dimensions or nnz differ.
    pub fn from_re_im(
        re: &crate::csr::CsrMatrix<f64>,
        im: &crate::csr::CsrMatrix<f64>,
    ) -> Self {
        assert_eq!(re.nrows, im.nrows, "row count mismatch");
        assert_eq!(re.ncols, im.ncols, "col count mismatch");
        // Build combined sparsity and values
        let n = re.nrows;
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut re_vals: Vec<f64> = Vec::new();
        let mut im_vals: Vec<f64> = Vec::new();

        for i in 0..n {
            let mut entries: std::collections::HashMap<u32, (f64, f64)> = std::collections::HashMap::new();
            for ptr in re.row_ptr[i]..re.row_ptr[i + 1] {
                let j = re.col_idx[ptr];
                entries.entry(j).or_insert((0.0, 0.0)).0 += re.values[ptr];
            }
            for ptr in im.row_ptr[i]..im.row_ptr[i + 1] {
                let j = im.col_idx[ptr];
                entries.entry(j).or_insert((0.0, 0.0)).1 += im.values[ptr];
            }
            let mut row_entries: Vec<(u32, f64, f64)> = entries
                .into_iter()
                .map(|(j, (r, m))| (j, r, m))
                .collect();
            row_entries.sort_by_key(|&(j, _, _)| j);
            for (j, r, m) in row_entries {
                col_idx.push(j);
                re_vals.push(r);
                im_vals.push(m);
            }
            row_ptr[i + 1] = col_idx.len();
        }
        ComplexCsr { nrows: n, ncols: re.ncols, row_ptr, col_idx, re_vals, im_vals }
    }

    /// Number of stored non-zeros.
    #[inline]
    pub fn nnz(&self) -> usize { self.re_vals.len() }

    /// Sparse matrix-vector multiply: `y = A * x` (complex, overwrites y).
    ///
    /// `x_re`, `x_im` are the real and imaginary parts of the input vector.
    /// `y_re`, `y_im` are overwritten with the result.
    pub fn spmv_into(&self, x_re: &[f64], x_im: &[f64], y_re: &mut [f64], y_im: &mut [f64]) {
        assert_eq!(x_re.len(), self.ncols);
        assert_eq!(x_im.len(), self.ncols);
        assert_eq!(y_re.len(), self.nrows);
        assert_eq!(y_im.len(), self.nrows);
        for i in 0..self.nrows {
            let (mut sr, mut si) = (0.0_f64, 0.0_f64);
            for ptr in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[ptr] as usize;
                let ar = self.re_vals[ptr];
                let ai = self.im_vals[ptr];
                let xr = x_re[j];
                let xi = x_im[j];
                sr += ar * xr - ai * xi;   // Re(A * x)
                si += ar * xi + ai * xr;   // Im(A * x)
            }
            y_re[i] = sr;
            y_im[i] = si;
        }
    }

    /// Extract diagonal as (re, im) pairs.
    pub fn diagonal_complex(&self) -> (Vec<f64>, Vec<f64>) {
        let mut dre = vec![0.0_f64; self.nrows];
        let mut dim = vec![0.0_f64; self.nrows];
        for i in 0..self.nrows {
            for ptr in self.row_ptr[i]..self.row_ptr[i + 1] {
                if self.col_idx[ptr] as usize == i {
                    dre[i] = self.re_vals[ptr];
                    dim[i] = self.im_vals[ptr];
                }
            }
        }
        (dre, dim)
    }

    /// Apply zero-one Dirichlet BC on row `dof` (identity row, zero off-diagonal).
    /// Modifies both re and im parts; sets rhs values.
    pub fn apply_dirichlet_row(
        &mut self,
        dof: usize,
        val_re: f64,
        val_im: f64,
        rhs_re: &mut [f64],
        rhs_im: &mut [f64],
    ) {
        for ptr in self.row_ptr[dof]..self.row_ptr[dof + 1] {
            let j = self.col_idx[ptr] as usize;
            if j == dof {
                self.re_vals[ptr] = 1.0;
                self.im_vals[ptr] = 0.0;
            } else {
                self.re_vals[ptr] = 0.0;
                self.im_vals[ptr] = 0.0;
            }
        }
        rhs_re[dof] = val_re;
        rhs_im[dof] = val_im;
    }
}

/// COO accumulator for complex sparse matrix assembly.
#[derive(Debug, Default)]
pub struct ComplexCoo {
    pub nrows: usize,
    pub ncols: usize,
    rows: Vec<u32>,
    cols: Vec<u32>,
    re_vals: Vec<f64>,
    im_vals: Vec<f64>,
}

impl ComplexCoo {
    /// Create an empty complex COO matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, rows: Vec::new(), cols: Vec::new(), re_vals: Vec::new(), im_vals: Vec::new() }
    }

    /// Add complex entry `(re + i*im)` at position `(row, col)`.
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, re: f64, im: f64) {
        self.rows.push(row as u32);
        self.cols.push(col as u32);
        self.re_vals.push(re);
        self.im_vals.push(im);
    }

    /// Add a dense complex element matrix.  `k_re` and `k_im` are row-major `k×k`.
    pub fn add_element_matrix(
        &mut self,
        dofs: &[usize],
        k_re: &[f64],
        k_im: &[f64],
    ) {
        let k = dofs.len();
        debug_assert_eq!(k_re.len(), k * k);
        debug_assert_eq!(k_im.len(), k * k);
        for i in 0..k {
            for j in 0..k {
                self.add(dofs[i], dofs[j], k_re[i * k + j], k_im[i * k + j]);
            }
        }
    }

    /// Convert to [`ComplexCsr`] by sorting and summing duplicate entries.
    pub fn into_complex_csr(mut self) -> ComplexCsr {
        let n = self.nrows;
        let m = self.ncols;
        let nnz_raw = self.rows.len();

        // Sort all entries by (row, col)
        let mut order: Vec<usize> = (0..nnz_raw).collect();
        order.sort_by_key(|&i| (self.rows[i], self.cols[i]));

        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx: Vec<u32> = Vec::with_capacity(nnz_raw);
        let mut re_vals: Vec<f64> = Vec::with_capacity(nnz_raw);
        let mut im_vals: Vec<f64> = Vec::with_capacity(nnz_raw);

        let mut prev_row: Option<usize> = None;
        let mut prev_col: Option<u32> = None;

        for &idx in &order {
            let r = self.rows[idx] as usize;
            let c = self.cols[idx];
            let rv = self.re_vals[idx];
            let iv = self.im_vals[idx];

            if prev_row == Some(r) && prev_col == Some(c) {
                // Same (row, col) — accumulate into last entry
                *re_vals.last_mut().unwrap() += rv;
                *im_vals.last_mut().unwrap() += iv;
            } else {
                // New entry
                if prev_row != Some(r) {
                    // Moved to a new row — fill in row_ptr for all skipped rows
                    let from = prev_row.map(|pr| pr + 1).unwrap_or(0);
                    for item in row_ptr.iter_mut().take(r + 1).skip(from) {
                        *item = col_idx.len();
                    }
                }
                col_idx.push(c);
                re_vals.push(rv);
                im_vals.push(iv);
                prev_row = Some(r);
                prev_col = Some(c);
            }
        }

        // Fill remaining row_ptr entries
        let from = prev_row.map(|pr| pr + 1).unwrap_or(0);
        for item in row_ptr.iter_mut().take(n + 1).skip(from) {
            *item = col_idx.len();
        }

        // Free original storage
        self.rows.clear(); self.cols.clear(); self.re_vals.clear(); self.im_vals.clear();

        ComplexCsr { nrows: n, ncols: m, row_ptr, col_idx, re_vals, im_vals }
    }
}

// ─── Complex GMRES ────────────────────────────────────────────────────────────

/// Solve `A x = b` (complex) via restarted GMRES with optional Jacobi
/// preconditioner `M ≈ diag(A)`.
///
/// Inputs and outputs are split into real/imaginary parts.
///
/// # Parameters
/// - `a`          — complex system matrix
/// - `b_re/b_im`  — RHS real/imaginary parts
/// - `x_re/x_im`  — initial guess (in) and solution (out)
/// - `tol`        — relative residual tolerance
/// - `max_iter`   — maximum GMRES iterations
/// - `restart`    — Krylov subspace size before restart (m)
/// - `precond`    — if true, apply Jacobi preconditioner using diagonal of A
///
/// Returns `(iterations, final_relative_residual)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_gmres_complex(
    a: &ComplexCsr,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
    precond: bool,
) -> Result<(usize, f64), String> {
    let (d_re, d_im) = a.diagonal_complex();
    let n = a.nrows;
    let prec: Vec<(f64, f64)> = if precond {
        d_re.iter().zip(d_im.iter()).map(|(&dr, &di)| {
            let m2 = dr * dr + di * di;
            if m2 < 1e-300 { (1.0, 0.0) } else { (dr / m2, -di / m2) }
        }).collect()
    } else {
        vec![(1.0, 0.0); n]
    };
    let jacobi = move |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut wr = vec![0.0; n]; let mut wi = vec![0.0; n];
        for i in 0..n { let (pr, pi) = prec[i];
            wr[i] = pr*vr[i] - pi*vi[i]; wi[i] = pr*vi[i] + pi*vr[i]; }
        (wr, wi)
    };
    solve_gmres_complex_with(a, b_re, b_im, x_re, x_im, tol, max_iter, restart, &jacobi)
}

/// Apply one row `(g0, g1)` of a complex 2×2 Givens rotation to the pair
/// `(h1, h2)`: `out = g0·h1 + g1·h2` (component-wise complex arithmetic).
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

/// GMRES for complex systems with a caller-provided preconditioner.
///
/// `apply_prec(r_re, r_im) -> (z_re, z_im)` should compute `z = M⁻¹·r`
/// (left preconditioning: the Krylov space is built from `M⁻¹A`).
///
/// Standard restarted complex GMRES: Arnoldi with modified Gram-Schmidt using
/// the Hermitian inner product, and full 2×2 complex Givens rotations
/// `G = (1/r)·[[h̄₁, h̄₂], [−h₂, h₁]]`, `r = √(|h₁|² + |h₂|²)`, which
/// annihilate the sub-diagonal entry exactly, followed by the complex
/// back-substitution `H y = g`.  Convergence is measured on the relative
/// residual `‖b − A x‖ / ‖b‖` (the cheap recursive estimate `|g_{j+1}|`
/// drives the in-cycle check; the true residual is recomputed at the end of
/// every restart cycle).  The best iterate is returned even on
/// non-convergence (soft failure), together with `(iterations, rel_res)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_gmres_complex_with<F>(
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
    if x_re.len() != n { *x_re = vec![0.0; n]; }
    if x_im.len() != n { *x_im = vec![0.0; n]; }

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

    // Compute initial residual r = b - A*x
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
        // Arnoldi with modified Gram-Schmidt
        let mut v_re: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut v_im: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut h = vec![vec![(0.0_f64, 0.0_f64); m]; m + 1]; // H[i][j]
        // Givens rotations stored as full 2×2 complex unitaries
        // G = (1/r)·[[conj(h1), conj(h2)], [−h2, h1]] (rows orthonormal).
        let mut givens: Vec<[(f64, f64); 4]> = Vec::with_capacity(m);
        let mut g = vec![(0.0_f64, 0.0_f64); m + 1]; // RHS of reduced system

        // Apply preconditioner to r: z0 = M^{-1} r
        let (z0_re, z0_im) = apply_prec(&r_re, &r_im);
        let beta = norm2(&z0_re, &z0_im);
        if beta <= 1e-300 {
            // Preconditioned residual vanished: x solves the system to
            // round-off.  Refresh the true residual and stop.
            a.spmv_into(x_re, x_im, &mut r_re, &mut r_im);
            for i in 0..n {
                r_re[i] = b_re[i] - r_re[i];
                r_im[i] = b_im[i] - r_im[i];
            }
            res = norm2(&r_re, &r_im) / b_norm;
            break 'restart;
        }
        g[0] = (beta, 0.0);

        // v0 = z0 / beta
        v_re.push(z0_re.iter().map(|&x| x / beta).collect());
        v_im.push(z0_im.iter().map(|&x| x / beta).collect());

        let mut k = 0usize; // number of Arnoldi columns built
        for j in 0..m {
            k = j + 1;
            total_iter += 1;

            // w = M^{-1} A v_j
            let mut av_re = vec![0.0; n];
            let mut av_im = vec![0.0; n];
            a.spmv_into(&v_re[j], &v_im[j], &mut av_re, &mut av_im);
            let (mut w_re, mut w_im) = apply_prec(&av_re, &av_im);

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let (hr, hi) = dot2(&v_re[i], &v_im[i], &w_re, &w_im);
                h[i][j] = (hr, hi);
                for k in 0..n {
                    w_re[k] -= hr * v_re[i][k] - hi * v_im[i][k];
                    w_im[k] -= hr * v_im[i][k] + hi * v_re[i][k];
                }
            }
            let w_norm = norm2(&w_re, &w_im);
            h[j + 1][j] = (w_norm, 0.0);

            // New Arnoldi vector
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

            // New Givens rotation annihilating h[j+1][j]:
            // G = (1/r)·[[conj(h1), conj(h2)], [−h2, h1]],  r = ‖(h1, h2)‖
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

            // Apply the rotation to the reduced RHS
            let g1 = g[j];
            let g2 = g[j + 1];
            g[j] = givens_row_apply(gt[0], gt[1], g1, g2);
            g[j + 1] = givens_row_apply(gt[2], gt[3], g1, g2);

            // Residual estimate |g[j+1]|
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

        // Update x = x + V * y
        for j in 0..k {
            for i in 0..n {
                x_re[i] += y_re[j] * v_re[j][i] - y_im[j] * v_im[j][i];
                x_im[i] += y_re[j] * v_im[j][i] + y_im[j] * v_re[j][i];
            }
        }

        // Update residual
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

    // Return the best iterate even on non-convergence (soft failure)
    Ok((total_iter, res))
}

// ─── Complex BiCGSTAB ────────────────────────────────────────────────────────

/// BiCGSTAB for complex systems with a caller-provided preconditioner.
///
/// `apply_prec(r_re, r_im) -> (z_re, z_im)` should compute `z = M⁻¹·r`.
#[allow(clippy::too_many_arguments)]
pub fn solve_bicgstab_complex_with<F>(
    a: &ComplexCsr,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    apply_m: &F,
) -> Result<(usize, f64), String>
where
    F: Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
{
    let n = a.nrows;
    if b_re.len() != n || b_im.len() != n {
        return Err("BiCGSTAB: dimension mismatch".into());
    }
    if x_re.len() != n { *x_re = vec![0.0; n]; }
    if x_im.len() != n { *x_im = vec![0.0; n]; }

    // ⟨a, b⟩ = Σ conj(aᵢ)·bᵢ
    let cdot = |ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]| -> (f64, f64) {
        let mut sr = 0.0f64; let mut si = 0.0f64;
        for i in 0..n { sr += ar[i]*br[i] + ai[i]*bi[i]; si += ar[i]*bi[i] - ai[i]*br[i]; }
        (sr, si)
    };
    let cnorm = |vr: &[f64], vi: &[f64]| -> f64 { let (d,_)=cdot(vr,vi,vr,vi); d.sqrt() };

    // r₀ = b - A·x₀
    let mut r_re = vec![0.0; n]; let mut r_im = vec![0.0; n];
    a.spmv_into(x_re, x_im, &mut r_re, &mut r_im);
    for i in 0..n { r_re[i] = b_re[i] - r_re[i]; r_im[i] = b_im[i] - r_im[i]; }

    let b_norm = cnorm(b_re, b_im).max(1e-300);
    let mut res = cnorm(&r_re, &r_im) / b_norm;
    if res < tol { return Ok((0, res)); }

    // Shadow residual r̂ = r₀
    let r_hat_re = r_re.clone();
    let r_hat_im = r_im.clone();

    let mut p_re = vec![0.0; n]; let mut p_im = vec![0.0; n];
    let mut v_re = vec![0.0; n]; let mut v_im = vec![0.0; n];
    let mut s_re = vec![0.0; n]; let mut s_im = vec![0.0; n];
    let mut t_re = vec![0.0; n]; let mut t_im = vec![0.0; n];

    let mut rho_old_re = 1.0; let mut rho_old_im = 0.0;
    let mut alpha_re = 1.0; let mut alpha_im = 0.0;
    let mut omega_re = 1.0; let mut omega_im = 0.0;

    for iter in 0..max_iter {
        // ρ = ⟨r̂, r⟩
        let (rho_re, rho_im) = cdot(&r_hat_re, &r_hat_im, &r_re, &r_im);
        if rho_re.hypot(rho_im) < 1e-300 {
            return Err("BiCGSTAB breakdown: ρ ≈ 0".into());
        }

        // β = (ρ/ρ_old)·(α/ω),  β=0 on first iteration
        let (beta_re, beta_im) = if iter == 0 {
            (0.0, 0.0)
        } else {
            let rho_old_m2 = rho_old_re*rho_old_re + rho_old_im*rho_old_im;
            let rho_div = ((rho_re*rho_old_re + rho_im*rho_old_im) / rho_old_m2,
                           (rho_im*rho_old_re - rho_re*rho_old_im) / rho_old_m2);
            let omega_m2 = omega_re*omega_re + omega_im*omega_im;
            let a_div_w = ((alpha_re*omega_re + alpha_im*omega_im) / omega_m2,
                           (alpha_im*omega_re - alpha_re*omega_im) / omega_m2);
            (rho_div.0*a_div_w.0 - rho_div.1*a_div_w.1,
             rho_div.0*a_div_w.1 + rho_div.1*a_div_w.0)
        };

        // p = r + β·(p - ω·v)   (iter==0: p = r)
        if iter == 0 {
            p_re.copy_from_slice(&r_re); p_im.copy_from_slice(&r_im);
        } else {
            for i in 0..n {
                let p_minus_wv_re = p_re[i] - (omega_re*v_re[i] - omega_im*v_im[i]);
                let p_minus_wv_im = p_im[i] - (omega_re*v_im[i] + omega_im*v_re[i]);
                p_re[i] = r_re[i] + beta_re*p_minus_wv_re - beta_im*p_minus_wv_im;
                p_im[i] = r_im[i] + beta_re*p_minus_wv_im + beta_im*p_minus_wv_re;
            }
        }

        // Apply M⁻¹ p → p̂, then v = A·p̂
        let (ph_re, ph_im) = apply_m(&p_re, &p_im);
        a.spmv_into(&ph_re, &ph_im, &mut v_re, &mut v_im);

        // α = ρ / ⟨r̂, v⟩
        let (rv_re, rv_im) = cdot(&r_hat_re, &r_hat_im, &v_re, &v_im);
        let rv_m2 = rv_re*rv_re + rv_im*rv_im;
        if rv_m2 < 1e-300 { return Err("BiCGSTAB breakdown: ⟨r̂,v⟩ ≈ 0".into()); }
        let inv_rv = 1.0 / rv_m2;
        alpha_re = (rho_re*rv_re + rho_im*rv_im) * inv_rv;
        alpha_im = (rho_im*rv_re - rho_re*rv_im) * inv_rv;

        // s = r - α·v
        for i in 0..n {
            s_re[i] = r_re[i] - (alpha_re*v_re[i] - alpha_im*v_im[i]);
            s_im[i] = r_im[i] - (alpha_re*v_im[i] + alpha_im*v_re[i]);
        }

        let s_norm = cnorm(&s_re, &s_im);
        if s_norm <= tol * b_norm {
            // x += α·p̂
            for i in 0..n {
                x_re[i] += alpha_re*ph_re[i] - alpha_im*ph_im[i];
                x_im[i] += alpha_re*ph_im[i] + alpha_im*ph_re[i];
            }
            return Ok((iter + 1, s_norm / b_norm));
        }

        // t = A·(M⁻¹·s)
        let (sh_re, sh_im) = apply_m(&s_re, &s_im);
        a.spmv_into(&sh_re, &sh_im, &mut t_re, &mut t_im);

        // ω = ⟨t, s⟩ / ⟨t, t⟩
        let (ts_re, ts_im) = cdot(&t_re, &t_im, &s_re, &s_im);
        let (tt_re, _) = cdot(&t_re, &t_im, &t_re, &t_im); // ⟨t,t⟩ is real
        if tt_re < 1e-300 { return Err("BiCGSTAB breakdown: ⟨t,t⟩ ≈ 0".into()); }
        omega_re = ts_re / tt_re;
        omega_im = ts_im / tt_re;

        if omega_re.hypot(omega_im) < 1e-300 {
            return Err("BiCGSTAB breakdown: ω ≈ 0".into());
        }

        // xₖ = x_{k-1} + α·p̂ + ω·ŝ
        for i in 0..n {
            x_re[i] += alpha_re*ph_re[i] - alpha_im*ph_im[i]
                     + omega_re*sh_re[i] - omega_im*sh_im[i];
            x_im[i] += alpha_re*ph_im[i] + alpha_im*ph_re[i]
                     + omega_re*sh_im[i] + omega_im*sh_re[i];
            r_re[i] = s_re[i] - (omega_re*t_re[i] - omega_im*t_im[i]);
            r_im[i] = s_im[i] - (omega_re*t_im[i] + omega_im*t_re[i]);
        }

        res = cnorm(&r_re, &r_im) / b_norm;
        if res < tol { return Ok((iter + 1, res)); }

        rho_old_re = rho_re; rho_old_im = rho_im;
    }

    Err(format!("BiCGSTAB not converged after {max_iter} iterations, residual {res:.2e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_csr_spmv_identity() {
        // A = I (3×3 real identity)
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0u32, 1, 2];
        let re_vals = vec![1.0, 1.0, 1.0];
        let im_vals = vec![0.0, 0.0, 0.0];
        let a = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };

        let x_re = vec![1.0, 2.0, 3.0];
        let x_im = vec![0.5, 1.5, 2.5];
        let mut y_re = vec![0.0; 3];
        let mut y_im = vec![0.0; 3];
        a.spmv_into(&x_re, &x_im, &mut y_re, &mut y_im);
        for i in 0..3 {
            assert!((y_re[i] - x_re[i]).abs() < 1e-14);
            assert!((y_im[i] - x_im[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn complex_csr_spmv_imaginary_shift() {
        // A = i*I (purely imaginary identity): A*x = i*x
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0u32, 1, 2];
        let re_vals = vec![0.0, 0.0, 0.0];
        let im_vals = vec![1.0, 1.0, 1.0];
        let a = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };

        let x_re = vec![1.0, 0.0, 0.0];
        let x_im = vec![0.0, 1.0, 0.0];
        let mut y_re = vec![0.0; 3];
        let mut y_im = vec![0.0; 3];
        a.spmv_into(&x_re, &x_im, &mut y_re, &mut y_im);
        // i * (1+0i) = 0+i, i * (0+i) = -1+0i
        assert!((y_re[0] - 0.0).abs() < 1e-14);
        assert!((y_im[0] - 1.0).abs() < 1e-14);
        assert!((y_re[1] + 1.0).abs() < 1e-14);
        assert!((y_im[1] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn complex_csr_diagonal() {
        let row_ptr = vec![0, 1, 2];
        let col_idx = vec![0u32, 1];
        let re_vals = vec![3.0, 5.0];
        let im_vals = vec![1.0, -2.0];
        let a = ComplexCsr { nrows: 2, ncols: 2, row_ptr, col_idx, re_vals, im_vals };
        let (dr, di) = a.diagonal_complex();
        assert_eq!(dr, vec![3.0, 5.0]);
        assert_eq!(di, vec![1.0, -2.0]);
    }

    #[test]
    fn complex_gmres_diagonal_system() {
        // Solve: (2 + i) * x = (5 + 3i)
        // x = (5+3i)/(2+i) = (5+3i)(2-i)/5 = (10-5i+6i+3)/5 = (13+i)/5 = 2.6 + 0.2i
        let n = 3;
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0u32, 1, 2];
        let re_vals = vec![2.0, 3.0, 1.0];
        let im_vals = vec![1.0, -1.0, 2.0];
        let a = ComplexCsr { nrows: n, ncols: n, row_ptr, col_idx, re_vals, im_vals };

        // b = A * x_exact where x_exact = [1+i, 2-i, 0.5+0.5i]
        // (2+i)(1+i) = 2+2i+i-1 = 1+3i
        // (3-i)(2-i) = 6-3i-2i-1 = 5-5i  → wait: (3-i)(2-i)=6-3i-2i+i²=6-5i-1=5-5i
        // (1+2i)(0.5+0.5i) = 0.5+0.5i+i+i²=-0.5+1.5i
        let b_re = vec![1.0, 5.0, -0.5];
        let b_im = vec![3.0, -5.0, 1.5];

        let mut x_re = vec![0.0; n];
        let mut x_im = vec![0.0; n];
        let (iters, res) = solve_gmres_complex(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 100, 50, true,
        ).unwrap();

        assert!(iters > 0);
        assert!(res < 1e-8, "residual too large: {}", res);
        assert!((x_re[0] - 1.0).abs() < 1e-6, "x_re[0] = {}", x_re[0]);
        assert!((x_im[0] - 1.0).abs() < 1e-6, "x_im[0] = {}", x_im[0]);
        assert!((x_re[1] - 2.0).abs() < 1e-6, "x_re[1] = {}", x_re[1]);
        assert!((x_im[1] + 1.0).abs() < 1e-6, "x_im[1] = {}", x_im[1]);
    }

    #[test]
    fn complex_gmres_spd_system() {
        // Real symmetric positive definite system stored as complex
        // (zero imaginary part): A = [[3,1,0],[1,3,1],[0,1,3]], x = [1,2,3],
        // b = A·x = [5,10,11].
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0u32, 1, 0, 1, 2, 1, 2];
        let re_vals = vec![3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0];
        let im_vals = vec![0.0; 7];
        let a = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };
        let b_re = vec![5.0, 10.0, 11.0];
        let b_im = vec![0.0; 3];

        let mut x_re = vec![0.0; 3];
        let mut x_im = vec![0.0; 3];
        let (iters, res) = solve_gmres_complex(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-13, 100, 50, true,
        ).unwrap();
        assert!(iters > 0);
        assert!(res < 1e-12, "SPD GMRES residual too large: {res:e}");
        assert!((x_re[0] - 1.0).abs() < 1e-10);
        assert!((x_re[1] - 2.0).abs() < 1e-10);
        assert!((x_re[2] - 3.0).abs() < 1e-10);
        assert!(x_im.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn complex_gmres_nonhermitian_system() {
        // Genuinely complex non-Hermitian system with known exact solution:
        // A = [[2+i, 1, 0.5], [0.3, 3−i, 1], [0, 0.4, 1+2i]],
        // x_exact = [1+i, 2−i, 0.5+0.5i]; b is formed by the spmv itself.
        let mut coo = ComplexCoo::new(3, 3);
        coo.add(0, 0, 2.0, 1.0);
        coo.add(0, 1, 1.0, 0.0);
        coo.add(0, 2, 0.5, 0.0);
        coo.add(1, 0, 0.3, 0.0);
        coo.add(1, 1, 3.0, -1.0);
        coo.add(1, 2, 1.0, 0.0);
        coo.add(2, 1, 0.4, 0.0);
        coo.add(2, 2, 1.0, 2.0);
        let a = coo.into_complex_csr();

        let x_true_re = vec![1.0, 2.0, 0.5];
        let x_true_im = vec![1.0, -1.0, 0.5];
        let mut b_re = vec![0.0; 3];
        let mut b_im = vec![0.0; 3];
        a.spmv_into(&x_true_re, &x_true_im, &mut b_re, &mut b_im);

        let mut x_re = vec![0.0; 3];
        let mut x_im = vec![0.0; 3];
        let (iters, res) = solve_gmres_complex(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-13, 100, 50, true,
        ).unwrap();
        assert!(iters > 0);
        assert!(res < 1e-12, "non-Hermitian GMRES residual too large: {res:e}");
        for i in 0..3 {
            assert!((x_re[i] - x_true_re[i]).abs() < 1e-10, "x_re[{i}] = {}", x_re[i]);
            assert!((x_im[i] - x_true_im[i]).abs() < 1e-10, "x_im[{i}] = {}", x_im[i]);
        }
    }

    #[test]
    fn complex_gmres_schrodinger_operator() {
        // Schrödinger Crank–Nicolson-type operator C = M + i·ω·A on a periodic
        // 1D P1 lattice (n = 32): M = tridiag(h/6, 4h/6, h/6),
        // A = tridiag(−1/h, 2/h, −1/h).  Unpreconditioned restarted GMRES
        // (restart 50, as in the C++ `GMRESSolver` default) must drive the
        // relative residual below 1e-12 — this operator class is what exposed
        // the old real-only Givens-rotation bug.
        use crate::CooMatrix;
        let n = 32usize;
        let h = 1.0_f64 / n as f64;
        let omega = 0.37_f64;
        let mut coo_m = CooMatrix::<f64>::new(n, n);
        let mut coo_a = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo_m.add(i, i, 4.0 * h / 6.0);
            coo_a.add(i, i, 2.0 / h);
            coo_m.add(i, (i + 1) % n, h / 6.0);
            coo_m.add(i, (i + n - 1) % n, h / 6.0);
            coo_a.add(i, (i + 1) % n, -1.0 / h);
            coo_a.add(i, (i + n - 1) % n, -1.0 / h);
        }
        let mass = coo_m.into_csr();
        let stiff = coo_a.into_csr();

        let mut coo_c = ComplexCoo::new(n, n);
        for i in 0..n {
            for p in mass.row_ptr[i]..mass.row_ptr[i + 1] {
                coo_c.add(i, mass.col_idx[p] as usize, mass.values[p], 0.0);
            }
            for p in stiff.row_ptr[i]..stiff.row_ptr[i + 1] {
                coo_c.add(i, stiff.col_idx[p] as usize, 0.0, omega * stiff.values[p]);
            }
        }
        let c = coo_c.into_complex_csr();

        let x_true_re: Vec<f64> =
            (0..n).map(|i| ((i as f64) * 0.7).sin() * (1.0 + 0.3 * (0.31 * i as f64).cos())).collect();
        let x_true_im: Vec<f64> = (0..n).map(|i| (0.53 * i as f64).cos()).collect();
        let mut b_re = vec![0.0; n];
        let mut b_im = vec![0.0; n];
        c.spmv_into(&x_true_re, &x_true_im, &mut b_re, &mut b_im);

        let mut x_re = vec![0.0; n];
        let mut x_im = vec![0.0; n];
        let (iters, res) = solve_gmres_complex(
            &c, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-13, 200, 50, false,
        ).unwrap();
        assert!(iters > 0);
        assert!(res < 1e-12, "Schrödinger GMRES residual {res:e} after {iters} iters");
        for i in 0..n {
            assert!((x_re[i] - x_true_re[i]).abs() < 1e-9, "x_re[{i}] = {}", x_re[i]);
            assert!((x_im[i] - x_true_im[i]).abs() < 1e-9, "x_im[{i}] = {}", x_im[i]);
        }
    }

    #[test]
    fn complex_coo_into_csr() {
        let mut coo = ComplexCoo::new(3, 3);
        coo.add(0, 0, 1.0, 0.5);
        coo.add(1, 1, 2.0, -1.0);
        coo.add(2, 2, 3.0, 0.0);
        coo.add(0, 0, 0.5, 0.5); // duplicate → sum
        let csr = coo.into_complex_csr();
        assert_eq!(csr.nrows, 3);
        // Row 0 should have (re=1.5, im=1.0) at col 0
        let ptr = csr.row_ptr[0];
        assert!((csr.re_vals[ptr] - 1.5).abs() < 1e-14);
        assert!((csr.im_vals[ptr] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn complex_dirichlet_bc() {
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0u32, 1, 0, 1];
        let re_vals = vec![2.0, 1.0, 1.0, 3.0];
        let im_vals = vec![0.5, 0.0, 0.0, -0.5];
        let mut a = ComplexCsr { nrows: 2, ncols: 2, row_ptr, col_idx, re_vals, im_vals };
        let mut rhs_re = vec![5.0, 7.0];
        let mut rhs_im = vec![2.0, 3.0];
        a.apply_dirichlet_row(0, 3.0, -1.0, &mut rhs_re, &mut rhs_im);
        // Row 0 should be [1+0i, 0+0i]
        assert_eq!(a.re_vals[0], 1.0);
        assert_eq!(a.im_vals[0], 0.0);
        assert_eq!(a.re_vals[1], 0.0);
        assert_eq!(a.im_vals[1], 0.0);
        assert_eq!(rhs_re[0], 3.0);
        assert_eq!(rhs_im[0], -1.0);
    }

    #[test]
    fn bicgstab_complex_diagonal_system() {
        // Solve same diagonal system as GMRES test: A·x = b
        // A = diag(2+i, 3-i, 1+2i), x_exact = [1+i, 2-i, 0.5+0.5i]
        let n = 3;
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0u32, 1, 2];
        let re_vals = vec![2.0, 3.0, 1.0];
        let im_vals = vec![1.0, -1.0, 2.0];
        let a = ComplexCsr { nrows: n, ncols: n, row_ptr, col_idx, re_vals, im_vals };
        let b_re = vec![1.0, 5.0, -0.5];
        let b_im = vec![3.0, -5.0, 1.5];

        let mut x_re = vec![0.0; n];
        let mut x_im = vec![0.0; n];

        // Jacobi preconditioner M⁻¹ ≈ diag(A)⁻¹
        let (d_re, d_im) = a.diagonal_complex();
        let prec: Vec<(f64, f64)> = d_re.iter().zip(d_im.iter()).map(|(&dr, &di)| {
            let m2 = dr * dr + di * di;
            if m2 < 1e-300 { (1.0, 0.0) } else { (dr / m2, -di / m2) }
        }).collect();
        let jacobi = |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let mut wr = vec![0.0; n]; let mut wi = vec![0.0; n];
            for i in 0..n { let (pr, pi) = prec[i];
                wr[i] = pr * vr[i] - pi * vi[i]; wi[i] = pr * vi[i] + pi * vr[i]; }
            (wr, wi)
        };
        let (iters, res) = solve_bicgstab_complex_with(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 100, &jacobi,
        ).unwrap();

        assert!(iters > 0);
        assert!(res < 1e-8, "BiCGSTAB residual too large: {res:.2e}");
        assert!((x_re[0] - 1.0).abs() < 1e-6, "x_re[0] = {}", x_re[0]);
        assert!((x_im[0] - 1.0).abs() < 1e-6, "x_im[0] = {}", x_im[0]);
        assert!((x_re[1] - 2.0).abs() < 1e-6, "x_re[1] = {}", x_re[1]);
        assert!((x_im[1] + 1.0).abs() < 1e-6, "x_im[1] = {}", x_im[1]);
    }

    #[test]
    fn bicgstab_complex_nondiagonal_system() {
        // Non-diagonal 3×3 complex system:
        // A = [[2+i, 1, 0], [1, 3-i, 1], [0, 1, 1+2i]]
        // x_exact = [1, 1+i, 1-i]
        // b = A·x computed manually
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0u32, 1, 0, 1, 2, 1, 2];
        let re_vals = vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0];
        let im_vals = vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0];
        let a = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };

        // b = A·[1, 1+i, 1-i]
        // Row 0: (2+i)·1 + 1·(1+i) = 2+i+1+i = 3+2i
        // Row 1: 1·1 + (3-i)·(1+i) + 1·(1-i) = 1 + (3+3i-i+1) + 1-i = 1+4+2i+1-i = 6+i
        // Row 2: 1·(1+i) + (1+2i)·(1-i) = 1+i + (1-i+2i+2) = 1+i+3+i = 4+2i
        let b_re = vec![3.0, 6.0, 4.0];
        let b_im = vec![2.0, 1.0, 2.0];
        let mut x_re = vec![0.0; 3];
        let mut x_im = vec![0.0; 3];

        // Jacobi preconditioner M⁻¹ ≈ diag(A)⁻¹
        let (d_re, d_im) = a.diagonal_complex();
        let prec: Vec<(f64, f64)> = d_re.iter().zip(d_im.iter()).map(|(&dr, &di)| {
            let m2 = dr * dr + di * di;
            if m2 < 1e-300 { (1.0, 0.0) } else { (dr / m2, -di / m2) }
        }).collect();
        let jacobi = |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let mut wr = vec![0.0; 3]; let mut wi = vec![0.0; 3];
            for i in 0..3 { let (pr, pi) = prec[i];
                wr[i] = pr * vr[i] - pi * vi[i]; wi[i] = pr * vi[i] + pi * vr[i]; }
            (wr, wi)
        };
        let (iters, res) = solve_bicgstab_complex_with(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 200, &jacobi,
        ).unwrap();

        assert!(iters > 0);
        assert!(res < 1e-8, "BiCGSTAB non-diag residual too large: {res:.2e}");
        assert!((x_re[0] - 1.0).abs() < 1e-6, "x_re[0] = {}", x_re[0]);
        assert!((x_im[0] - 0.0).abs() < 1e-6, "x_im[0] = {}", x_im[0]);
        assert!((x_re[1] - 1.0).abs() < 1e-6, "x_re[1] = {}", x_re[1]);
        assert!((x_im[1] - 1.0).abs() < 1e-6, "x_im[1] = {}", x_im[1]);
        assert!((x_re[2] - 1.0).abs() < 1e-6, "x_re[2] = {}", x_re[2]);
        assert!((x_im[2] + 1.0).abs() < 1e-6, "x_im[2] = {}", x_im[2]);
    }

    #[test]
    fn bicgstab_complex_spd_noprecond() {
        // Real symmetric positive definite system (imag part = 0), no precond
        // A = [[3,1,0],[1,3,1],[0,1,3]], x = [1,2,3]
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0u32, 1, 0, 1, 2, 1, 2];
        let re_vals = vec![3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0];
        let im_vals = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a = ComplexCsr { nrows: 3, ncols: 3, row_ptr, col_idx, re_vals, im_vals };
        let b_re = vec![5.0, 10.0, 11.0]; // A·[1,2,3]
        let b_im = vec![0.0; 3];

        let mut x_re = vec![0.0; 3];
        let mut x_im = vec![0.0; 3];

        // No preconditioning: identity (matches the deleted `precond: false` path)
        let identity = |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) {
            (vr.to_vec(), vi.to_vec())
        };
        let (iters, res) = solve_bicgstab_complex_with(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 100, &identity,
        ).unwrap();

        assert!(iters > 0);
        assert!(res < 1e-8, "BiCGSTAB SPD residual too large: {res:.2e}");
        assert!((x_re[0] - 1.0).abs() < 1e-6);
        assert!((x_re[1] - 2.0).abs() < 1e-6);
        assert!((x_re[2] - 3.0).abs() < 1e-6);
    }
}
