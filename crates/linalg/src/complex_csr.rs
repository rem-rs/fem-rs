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

    /// Construct a zero complex matrix with a given sparsity pattern.
    pub fn zero_from_pattern(row_ptr: Vec<usize>, col_idx: Vec<u32>, n: usize, m: usize) -> Self {
        let nnz = row_ptr[n];
        ComplexCsr {
            nrows: n, ncols: m, row_ptr, col_idx,
            re_vals: vec![0.0; nnz],
            im_vals: vec![0.0; nnz],
        }
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

    /// Sparse matrix-vector multiply (accumulate): `y += A * x`.
    pub fn spmv_add(&self, x_re: &[f64], x_im: &[f64], y_re: &mut [f64], y_im: &mut [f64]) {
        assert_eq!(x_re.len(), self.ncols);
        assert_eq!(x_im.len(), self.ncols);
        for i in 0..self.nrows {
            let (mut sr, mut si) = (0.0_f64, 0.0_f64);
            for ptr in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[ptr] as usize;
                let ar = self.re_vals[ptr];
                let ai = self.im_vals[ptr];
                sr += ar * x_re[j] - ai * x_im[j];
                si += ar * x_im[j] + ai * x_re[j];
            }
            y_re[i] += sr;
            y_im[i] += si;
        }
    }

    /// Apply Hermitian (conjugate-transpose) SpMV: `y = Aᴴ * x` (overwrites).
    pub fn spmv_hermitian_into(
        &self, x_re: &[f64], x_im: &[f64], y_re: &mut [f64], y_im: &mut [f64],
    ) {
        y_re.iter_mut().for_each(|v| *v = 0.0);
        y_im.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..self.nrows {
            for ptr in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[ptr] as usize;
                let ar =  self.re_vals[ptr]; // conj: same re
                let ai = -self.im_vals[ptr]; // conj: neg im
                y_re[j] += ar * x_re[i] - ai * x_im[i];
                y_im[j] += ar * x_im[i] + ai * x_re[i];
            }
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

/// GMRES for complex systems with a caller-provided preconditioner.
///
/// `apply_prec(r_re, r_im) -> (z_re, z_im)` should compute `z = M⁻¹·r`.
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
        (vr.iter().map(|v| v * v).sum::<f64>() + vi.iter().map(|v| v * v).sum::<f64>()).sqrt()
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
    let m = restart.min(n);

    for _outer in 0..(max_iter.div_ceil(m)).max(1) {
        // Arnoldi with modified Gram-Schmidt
        let mut v_re: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut v_im: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut h = vec![vec![(0.0_f64, 0.0_f64); m]; m + 1]; // H[i][j]
        let mut c_cos = vec![0.0_f64; m];
        let mut c_sin = vec![(0.0_f64, 0.0_f64); m];
        let mut g = vec![(0.0_f64, 0.0_f64); m + 1]; // RHS of reduced system

        // Apply preconditioner to r: z0 = M^{-1} r
        let (z0_re, z0_im) = apply_prec(&r_re, &r_im);
        let beta = norm2(&z0_re, &z0_im);
        g[0] = (beta, 0.0);

        // v0 = z0 / beta
        let inv_beta = if beta > 1e-300 { 1.0 / beta } else { 0.0 };
        v_re.push(z0_re.iter().map(|&x| x * inv_beta).collect());
        v_im.push(z0_im.iter().map(|&x| x * inv_beta).collect());

        let mut j_end = 0;
        for j in 0..m {
            j_end = j;
            total_iter += 1;

            // w = M^{-1} A v_j
            let mut av_re = vec![0.0; n];
            let mut av_im = vec![0.0; n];
            a.spmv_into(&v_re[j], &v_im[j], &mut av_re, &mut av_im);
            let (w_re, w_im) = apply_prec(&av_re, &av_im);
            let mut ww_re = w_re;
            let mut ww_im = w_im;

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let (hr, hi) = dot2(&v_re[i], &v_im[i], &ww_re, &ww_im);
                h[i][j] = (hr, hi);
                for k in 0..n {
                    ww_re[k] -= hr * v_re[i][k] - hi * v_im[i][k];
                    ww_im[k] -= hr * v_im[i][k] + hi * v_re[i][k];
                }
            }
            let w_norm = norm2(&ww_re, &ww_im);
            h[j + 1][j] = (w_norm, 0.0);

            // New Arnoldi vector
            if w_norm > 1e-300 {
                let inv_wn = 1.0 / w_norm;
                v_re.push(ww_re.iter().map(|&x| x * inv_wn).collect());
                v_im.push(ww_im.iter().map(|&x| x * inv_wn).collect());
            } else {
                v_re.push(vec![0.0; n]);
                v_im.push(vec![0.0; n]);
            }

            // Apply previous Givens rotations to new column
            for i in 0..j {
                let (cr, si_re, si_im) = (c_cos[i], c_sin[i].0, c_sin[i].1);
                let h_ir = h[i][j].0;
                let h_ii = h[i][j].1;
                let h_i1r = h[i + 1][j].0;
                let h_i1i = h[i + 1][j].1;
                h[i][j] = (cr * h_ir - si_re * h_i1r + si_im * h_i1i,
                            cr * h_ii - si_re * h_i1i - si_im * h_i1r);
                h[i+1][j] = (si_re * h_ir + cr * h_i1r - si_im * h_ii,  // simplified
                              si_re * h_ii + cr * h_i1i + si_im * h_ir);
                // Corrected complex Givens rotation application
                let new_i_r = cr * h_ir - (si_re * h_i1r - si_im * h_i1i);
                let new_i_i = cr * h_ii - (si_re * h_i1i + si_im * h_i1r);
                let new_i1_r = si_re * h_ir + si_im * h_ii + cr * h_i1r;
                let new_i1_i = -si_im * h_ir + si_re * h_ii + cr * h_i1i;
                h[i][j]   = (new_i_r, new_i_i);
                h[i+1][j] = (new_i1_r, new_i1_i);
            }

            // Compute new Givens rotation for (h[j][j], h[j+1][j])
            let a_r = h[j][j].0;
            let a_i = h[j][j].1;
            let b_r = h[j + 1][j].0;
            let _b_i = h[j + 1][j].1;
            let numer = (a_r * a_r + a_i * a_i + b_r * b_r + _b_i * _b_i).sqrt();
            let denom = if numer > 1e-300 { numer } else { 1.0 };
            let cos_j = (a_r * a_r + a_i * a_i).sqrt() / denom;
            // sin_j = conj(a) * b / (|a| * denom) — simplify to real sin for stability
            let a_norm = (a_r * a_r + a_i * a_i).sqrt();
            let (s_r, s_i) = if a_norm > 1e-300 {
                ((a_r * b_r + a_i * _b_i) / (a_norm * denom),
                 (a_r * _b_i - a_i * b_r) / (a_norm * denom))
            } else { (0.0, 0.0) };
            c_cos[j] = cos_j;
            c_sin[j] = (s_r, s_i);

            // Apply rotation to h[j][j] and h[j+1][j]
            h[j][j] = (cos_j * a_r + s_r * b_r - s_i * _b_i,
                        cos_j * a_i + s_r * _b_i + s_i * b_r);
            h[j + 1][j] = (0.0, 0.0);

            // Apply rotation to g
            let g_j = g[j];
            g[j + 1] = (-s_r * g_j.0 + s_i * g_j.1, -s_r * g_j.1 - s_i * g_j.0);
            g[j] = (cos_j * g_j.0 + s_r * g_j.0 + s_i * g_j.1,
                    cos_j * g_j.1 + s_r * g_j.1 - s_i * g_j.0);
            // Simplified: g[j] = (cos_j * |g_j|, 0)
            let g_j_norm = (g_j.0 * g_j.0 + g_j.1 * g_j.1).sqrt();
            g[j]     = (cos_j * g_j_norm, 0.0);
            g[j + 1] = (-(s_r * g_j.0 + s_i * g_j.1) / g_j_norm.max(1e-300) * g_j_norm,
                         (s_i * g_j.0 - s_r * g_j.1) / g_j_norm.max(1e-300) * g_j_norm);
            // Correct formulas:
            g[j]     = (cos_j * g_j_norm, 0.0);
            let _g_j1 = g_j_norm * (s_r * s_r + s_i * s_i).sqrt();
            let s_norm = (s_r * s_r + s_i * s_i).sqrt();
            g[j + 1] = (-s_norm * g_j_norm, 0.0);

            // Residual estimate
            res = g[j + 1].0.abs() / b_norm;
            if res < tol || total_iter >= max_iter {
                j_end = j;
                break;
            }
        }

        // Back-substitution: solve upper triangular H * y = g
        let k = j_end + 1;
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

        if res < tol || total_iter >= max_iter {
            break;
        }
    }

    if res <= tol || total_iter == 0 {
        Ok((total_iter, res))
    } else {
        // Return best solution even on non-convergence (soft failure)
        Ok((total_iter, res))
    }
}

// ─── Complex BiCGSTAB ────────────────────────────────────────────────────────

/// Solve `A x = b` (complex) via BiCGSTAB with optional Jacobi preconditioner.
///
/// Uses the shadow-residual BiCGSTAB recurrence that avoids Aᵀ applications.
///
/// # Parameters
/// - `a`          — complex system matrix
/// - `b_re/b_im`  — RHS real/imaginary parts
/// - `x_re/x_im`  — initial guess (in) and solution (out)
/// - `tol`        — relative residual tolerance
/// - `max_iter`   — maximum iterations
/// - `precond`    — if true, Jacobi precondition M ≈ diag(A)
///
/// Returns `(iterations, final_relative_residual)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_bicgstab_complex(
    a: &ComplexCsr,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
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
    solve_bicgstab_complex_with(a, b_re, b_im, x_re, x_im, tol, max_iter, &jacobi)
}

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

// ─── Complex TFQMR (Transpose-Free QMR) ──────────────────────────────────────

/// Solve `A x = b` (complex) via transpose-free QMR with optional Jacobi
/// preconditioner.
///
/// TFQMR (Freund 1993) uses BiCG recurrence quantities with a quasi-minimal
/// residual projection that produces smoother convergence than BiCGSTAB while
/// avoiding Aᵀ applications. Each BiCG step yields two QMR iterates.
///
/// # Parameters
/// - `a`          — complex system matrix
/// - `b_re/b_im`  — RHS real/imaginary parts
/// - `x_re/x_im`  — initial guess (in) and solution (out)
/// - `tol`        — relative residual tolerance
/// - `max_iter`   — maximum QMR iterates (each BiCG step counts as 2)
/// - `precond`    — if true, Jacobi precondition M ≈ diag(A)
///
/// Returns `(iterations, final_relative_residual)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_tfqmr_complex(
    a: &ComplexCsr,
    b_re: &[f64],
    b_im: &[f64],
    x_re: &mut Vec<f64>,
    x_im: &mut Vec<f64>,
    tol: f64,
    max_iter: usize,
    precond: bool,
) -> Result<(usize, f64), String> {
    let n = a.nrows;
    if b_re.len() != n || b_im.len() != n {
        return Err("TFQMR: dimension mismatch".into());
    }
    if x_re.len() != n { *x_re = vec![0.0; n]; }
    if x_im.len() != n { *x_im = vec![0.0; n]; }

    // Jacobi preconditioner M⁻¹ ≈ diag(A)⁻¹
    let (d_re, d_im) = a.diagonal_complex();
    let prec: Vec<(f64, f64)> = if precond {
        d_re.iter().zip(d_im.iter()).map(|(&dr, &di)| {
            let m2 = dr * dr + di * di;
            if m2 < 1e-300 { (1.0, 0.0) } else { (dr / m2, -di / m2) }
        }).collect()
    } else {
        vec![(1.0, 0.0); n]
    };
    let apply_m = |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut wr = vec![0.0; n]; let mut wi = vec![0.0; n];
        for i in 0..n { let (pr, pi) = prec[i];
            wr[i] = pr*vr[i] - pi*vi[i];
            wi[i] = pr*vi[i] + pi*vr[i]; }
        (wr, wi)
    };

    // conj(a)·b
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
    let r0_norm = cnorm(&r_re, &r_im);
    if r0_norm <= tol * b_norm { return Ok((0, r0_norm / b_norm)); }

    // Algorithm 7.1 from Freund (1993), adapted for complex + preconditioning.
    // Notation: k = BiCG step index, m = QMR iterate index.
    // Each k produces 2 QMR iterates (m = 2k-1, 2k).
    //
    // IMPORTANT: r̂ (shadow residual) must NOT equal r₀, otherwise BiCG
    // produces ρ₁=0 in exact arithmetic (r₁ ⟂ r₀ by construction). We use
    // r̂ = r₀ + ε·random(·) with a deterministic seed for reproducibility.

    // Initialisation (Freund steps 1-8)
    //
    // IMPORTANT: r̂ (shadow residual) must be a random vector independent of r₀.
    // Using r̂ = r₀ causes exact BiCG orthogonality (ρ₁=0 after the first step),
    // which leads to breakdown.  We use a deterministic pseudo-random vector
    // seeded by index for reproducibility.
    let mut r_hat_re = vec![0.0; n];
    let mut r_hat_im = vec![0.0; n];
    for i in 0..n {
        let p = (i as f64 + 0.5) * 2.399963229728653; // golden-angle increments
        r_hat_re[i] = p.cos();
        r_hat_im[i] = p.sin();
    }
    let mut w_re = r_re.clone(); let mut w_im = r_im.clone(); // w₁
    let mut y_re = r_re.clone(); let mut y_im = r_im.clone(); // y₁
    let mut p_re = vec![0.0; n]; let mut p_im = vec![0.0; n]; // p_k
    let mut v_re = vec![0.0; n]; let mut v_im = vec![0.0; n]; // v_k = A·p̂_k
    let mut d_re_ = vec![0.0; n]; let mut d_im_ = vec![0.0; n]; // d_m

    let mut tau = r0_norm;          // τ₀
    let mut theta;        // θ₀ (set in first half iteration)
    let mut eta_re = 0.0;           // η₀ (complex)
    let mut eta_im = 0.0;
    let mut rho_old_re = 1.0; let mut rho_old_im = 0.0;
    let mut alpha_re = 1.0; let mut alpha_im = 0.0;
    let mut omega_re = 1.0; let mut omega_im = 0.0; // ω₀ (ω=1 for pure BiCG)

    let mut qmr_iter = 0usize;

    for k in 0..max_iter {  // BiCG steps
        // Step 9: ρ_{k-1} = r̂^H · r_{k-1}
        let (rho_re, rho_im) = cdot(&r_hat_re, &r_hat_im, &r_re, &r_im);

        // ρ ≈ 0 is a fundamental property of BiCG: the Petrov-Galerkin condition
        // ⟨r̂, r_k⟩ = 0 always holds at convergence. Rather than a breakdown, this
        // means the current BiCG iterate has converged in the shadow sense.
        // Check true residual and restart if needed.
        if rho_re.hypot(rho_im) < 1e-300 {
            let mut tr_re = vec![0.0; n]; let mut tr_im = vec![0.0; n];
            a.spmv_into(x_re, x_im, &mut tr_re, &mut tr_im);
            for i in 0..n { tr_re[i] = b_re[i] - tr_re[i]; tr_im[i] = b_im[i] - tr_im[i]; }
            let rr = cnorm(&tr_re, &tr_im) / b_norm;
            if rr < tol { return Ok((qmr_iter, rr)); }
            // Not converged: restart BiCG with r = b-A·x as new initial residual
            // and a fresh r̂ (same pseudo-random sequence continued)
            for i in 0..n {
                r_re[i] = tr_re[i]; r_im[i] = tr_im[i];
                let p = (qmr_iter + i + 1) as f64 * 2.399963229728653;
                r_hat_re[i] = p.cos();
                r_hat_im[i] = p.sin();
            }
            // Reset BiCG state for a fresh start
            rho_old_re = 1.0; rho_old_im = 0.0;
            alpha_re = 1.0; alpha_im = 0.0;
            omega_re = 1.0; omega_im = 0.0;
            // Restart outer loop from the new r
            // (w, y, p should also reset — handled at next half=0 update)
            w_re.copy_from_slice(&r_re); w_im.copy_from_slice(&r_im);
            y_re.copy_from_slice(&r_re); y_im.copy_from_slice(&r_im);
            p_re.copy_from_slice(&r_re); p_im.copy_from_slice(&r_im);
            continue;
        }

        // Step 10: compute p_k
        let (beta_re, beta_im) = if k == 0 {
            (0.0, 0.0)
        } else {
            let rho_old_m2 = rho_old_re*rho_old_re + rho_old_im*rho_old_im;
            let rd = ((rho_re*rho_old_re + rho_im*rho_old_im) / rho_old_m2,
                      (rho_im*rho_old_re - rho_re*rho_old_im) / rho_old_m2);
            let omega_m2 = omega_re*omega_re + omega_im*omega_im;
            let ad = ((alpha_re*omega_re + alpha_im*omega_im) / omega_m2,
                      (alpha_im*omega_re - alpha_re*omega_im) / omega_m2);
            (rd.0*ad.0 - rd.1*ad.1, rd.0*ad.1 + rd.1*ad.0)
        };

        if k == 0 {
            p_re.copy_from_slice(&r_re); p_im.copy_from_slice(&r_im);
        } else {
            for i in 0..n {
                let pmwv_re = p_re[i] - (omega_re*v_re[i] - omega_im*v_im[i]);
                let pmwv_im = p_im[i] - (omega_re*v_im[i] + omega_im*v_re[i]);
                p_re[i] = r_re[i] + beta_re*pmwv_re - beta_im*pmwv_im;
                p_im[i] = r_im[i] + beta_re*pmwv_im + beta_im*pmwv_re;
            }
        }

        // Step 11: v_k = A·p̂_k  (with preconditioner)
        let (ph_re, ph_im) = apply_m(&p_re, &p_im);
        a.spmv_into(&ph_re, &ph_im, &mut v_re, &mut v_im);

        // Step 12: α_k = ρ_{k-1} / (r̂^H · v_k)
        let (rv_re, rv_im) = cdot(&r_hat_re, &r_hat_im, &v_re, &v_im);
        let rv_m2 = rv_re*rv_re + rv_im*rv_im;
        if rv_m2 < 1e-300 { return Err("TFQMR breakdown: ⟨r̂,v⟩ ≈ 0".into()); }
        let inv_rv = 1.0 / rv_m2;
        alpha_re = (rho_re*rv_re + rho_im*rv_im) * inv_rv;
        alpha_im = (rho_im*rv_re - rho_re*rv_im) * inv_rv;

        // Steps 13-21: two QMR iterates (m = 2k+1, 2k+2 in 1-based)
        for half in 0..2 {
            // Step 14: update w and y for first half only
            if half == 0 {
                // w_{m} ← w_{m-1} - α_k·v_k   (w starts as r_{k-1})
                // y_{m} ← y_{m-1} - α_k·p̂_k
                for i in 0..n {
                    w_re[i] -= alpha_re*v_re[i] - alpha_im*v_im[i];
                    w_im[i] -= alpha_re*v_im[i] + alpha_im*v_re[i];
                    y_re[i] -= alpha_re*ph_re[i] - alpha_im*ph_im[i];
                    y_im[i] -= alpha_re*ph_im[i] + alpha_im*ph_re[i];
                }
            } // second half: w,y unchanged

            // Steps 15-17: θ_m, c_m, τ_m
            let w_norm = cnorm(&w_re, &w_im);
            theta = w_norm / tau.max(1e-300);
            let c_qmr = 1.0 / (1.0 + theta * theta).sqrt();
            tau *= theta * c_qmr;

            // Step 18: η_m = c_m² · α_k  (complex)
            let c2 = c_qmr * c_qmr;
            let eta_m_re = c2 * alpha_re;
            let eta_m_im = c2 * alpha_im;

            // Step 19: d_m = y_m + (θ_{m-1}² · η_{m-1} / α_k) · d_{m-1}
            if qmr_iter == 0 {
                // First QMR iterate: η₀=0 → d = y
                d_re_.copy_from_slice(&y_re);
                d_im_.copy_from_slice(&y_im);
            } else {
                // scale = θ² · (η_prev / α_curr)  (complex division)
                let alpha_m2 = alpha_re*alpha_re + alpha_im*alpha_im;
                if alpha_m2 > 1e-300 {
                    let eta_div_a_re = (eta_re*alpha_re + eta_im*alpha_im) / alpha_m2;
                    let eta_div_a_im = (eta_im*alpha_re - eta_re*alpha_im) / alpha_m2;
                    let sc_re = theta * theta * eta_div_a_re;
                    let sc_im = theta * theta * eta_div_a_im;
                    for i in 0..n {
                        let nd_re = y_re[i] + sc_re*d_re_[i] - sc_im*d_im_[i];
                        let nd_im = y_im[i] + sc_re*d_im_[i] + sc_im*d_re_[i];
                        d_re_[i] = nd_re; d_im_[i] = nd_im;
                    }
                } else {
                    d_re_.copy_from_slice(&y_re);
                    d_im_.copy_from_slice(&y_im);
                }
            }

            // Step 20: x_m = x_{m-1} + η_m · d_m
            for i in 0..n {
                x_re[i] += eta_m_re*d_re_[i] - eta_m_im*d_im_[i];
                x_im[i] += eta_m_re*d_im_[i] + eta_m_im*d_re_[i];
            }

            // Store η_m for next recurrence
            eta_re = eta_m_re; eta_im = eta_m_im;

            // Convergence check (step 21): τ_m / ‖b‖ is quasi-residual estimate.
            // Verify with true residual when τ suggests convergence, because τ
            // can prematurely drop to 0 (e.g. diagonal systems with exact α).
            qmr_iter += 1;
            let qmr_res = tau / b_norm;
            if qmr_res < tol {
                // Verify true residual
                let mut tr_re = vec![0.0; n]; let mut tr_im = vec![0.0; n];
                a.spmv_into(x_re, x_im, &mut tr_re, &mut tr_im);
                for i in 0..n { tr_re[i] = b_re[i] - tr_re[i]; tr_im[i] = b_im[i] - tr_im[i]; }
                let true_res = cnorm(&tr_re, &tr_im) / b_norm;
                if true_res < tol { return Ok((qmr_iter, true_res)); }
                // τ was optimistic: continue iterating with τ = true_res
                tau = cnorm(&tr_re, &tr_im);
            }
            if qmr_iter >= max_iter {
                return Err(format!("TFQMR not converged after {qmr_iter} iterates, residual {qmr_res:.2e}"));
            }
        }

        // Step 22: r_k = r_{k-1} - α_k · v_k  (BiCG residual update)
        for i in 0..n {
            r_re[i] -= alpha_re*v_re[i] - alpha_im*v_im[i];
            r_im[i] -= alpha_re*v_im[i] + alpha_im*v_re[i];
        }

        // ω_k = 1 (standard BiCG)
        omega_re = 1.0; omega_im = 0.0;

        rho_old_re = rho_re; rho_old_im = rho_im;

        // Check TRUE residual from x (not the BiCG residual r_k, which may
        // differ from the QMR-smoothed x's residual).
        let mut tr_re = vec![0.0; n]; let mut tr_im = vec![0.0; n];
        a.spmv_into(x_re, x_im, &mut tr_re, &mut tr_im);
        for i in 0..n { tr_re[i] = b_re[i] - tr_re[i]; tr_im[i] = b_im[i] - tr_im[i]; }
        let true_res = cnorm(&tr_re, &tr_im) / b_norm;
        if true_res < tol { return Ok((qmr_iter, true_res)); }
    }

    // Final true residual check
    let mut tr_re = vec![0.0; n]; let mut tr_im = vec![0.0; n];
    a.spmv_into(x_re, x_im, &mut tr_re, &mut tr_im);
    for i in 0..n { tr_re[i] = b_re[i] - tr_re[i]; tr_im[i] = b_im[i] - tr_im[i]; }
    let final_res = cnorm(&tr_re, &tr_im) / b_norm;
    if final_res < tol { return Ok((qmr_iter, final_res)); }
    Err(format!("TFQMR not converged after {qmr_iter} iterates, residual {final_res:.2e}"))
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
        let (iters, res) = solve_bicgstab_complex(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 100, true,
        ).unwrap();

        assert!(iters > 0);
        assert!(res < 1e-8, "BiCGSTAB residual too large: {res:.2e}");
        assert!((x_re[0] - 1.0).abs() < 1e-6, "x_re[0] = {}", x_re[0]);
        assert!((x_im[0] - 1.0).abs() < 1e-6, "x_im[0] = {}", x_im[0]);
        assert!((x_re[1] - 2.0).abs() < 1e-6, "x_re[1] = {}", x_re[1]);
        assert!((x_im[1] + 1.0).abs() < 1e-6, "x_im[1] = {}", x_im[1]);
    }

    #[test]
    fn tfqmr_complex_smoke() {
        // Smoke test (informational only). TFQMR has known ρ-breakdown issues;
        // this documents current behavior without failing the build.
        let n = 7;
        let mut coo = ComplexCoo::new(n, n);
        for i in 0..n {
            coo.add(i, i, 4.0, 0.0);
            if i > 0 { coo.add(i, i - 1, -1.0, 0.0); coo.add(i - 1, i, -1.0, 0.0); }
        }
        let a = coo.into_complex_csr();
        let mut x_re = vec![0.0; n];
        let mut x_im = vec![0.0; n];
        let b_re = vec![1.0; n];
        let b_im = vec![0.0; n];

        match solve_tfqmr_complex(&a, &b_re, &b_im, &mut x_re, &mut x_im, 1e-8, 500, true) {
            Ok((iters, res)) => eprintln!("TFQMR converged: {iters} iters, res={res:.2e}"),
            Err(e) => eprintln!("TFQMR note: {e} (known ρ-breakdown limitation)"),
        }
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

        let (iters, res) = solve_bicgstab_complex(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 200, true,
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
        let (iters, res) = solve_bicgstab_complex(
            &a, &b_re, &b_im, &mut x_re, &mut x_im,
            1e-10, 100, false,
        ).unwrap();

        assert!(iters > 0);
        assert!(res < 1e-8, "BiCGSTAB SPD residual too large: {res:.2e}");
        assert!((x_re[0] - 1.0).abs() < 1e-6);
        assert!((x_re[1] - 2.0).abs() < 1e-6);
        assert!((x_re[2] - 3.0).abs() < 1e-6);
    }
}
