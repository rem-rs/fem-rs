//! Block solvers for saddle-point and block-structured systems.
//!
//! # Problem
//!
//! Saddle-point systems arise in mixed FEM (Stokes, Darcy, …):
//! ```text
//! [ A   B^T ] [ u ]   [ f ]
//! [ B   0   ] [ p ] = [ g ]
//! ```
//!
//! # Solvers provided
//!
//! | Solver | Method | Suitable for |
//! |--------|--------|--------------|
//! | [`BlockDiagonalPrecond`] | Diagonal block preconditioner | General block systems |
//! | [`BlockTriangularPrecond`] | Block upper/lower triangular | Saddle-point |
//! | [`SchurComplementSolver`] | Exact Schur complement | Small/medium systems |
//! | [`MinresSolver`] | MINRES (symmetric indefinite) | Saddle-point |
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::block::{BlockSystem, SchurComplementSolver};
//!
//! let sys = BlockSystem { a, bt, b, c: None };
//! let mut u = vec![0.0; n_u];
//! let mut p = vec![0.0; n_p];
//! SchurComplementSolver::solve(&sys, &f, &g, &mut u, &mut p, &cfg).unwrap();
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use crate::{SolverConfig, SolverError, SolveResult};

// ─── Block system ─────────────────────────────────────────────────────────────

/// A 2×2 block (saddle-point) system:
/// ```text
/// [ A   B^T ] [ u ]   [ f ]
/// [ B   C   ] [ p ] = [ g ]
/// ```
/// where `C` is typically zero or a small stabilization matrix.
pub struct BlockSystem {
    /// (1,1) block: n_u × n_u, typically symmetric positive definite.
    pub a:  CsrMatrix<f64>,
    /// (1,2) block: n_u × n_p  (B transposed).
    pub bt: CsrMatrix<f64>,
    /// (2,1) block: n_p × n_u.
    pub b:  CsrMatrix<f64>,
    /// (2,2) block: n_p × n_p (may be None → treated as zero).
    pub c:  Option<CsrMatrix<f64>>,
}

impl BlockSystem {
    pub fn n_u(&self) -> usize { self.a.nrows }
    pub fn n_p(&self) -> usize { self.b.nrows }
    pub fn n_total(&self) -> usize { self.n_u() + self.n_p() }

    /// Apply the full block matrix to `[u; p]` → `[Au + Bᵀp; Bu + Cp]`.
    pub fn apply(&self, u: &[f64], p: &[f64], ru: &mut [f64], rp: &mut [f64]) {
        // ru = A u + B^T p
        spmv_add(&self.a, u, ru);
        spmv_add(&self.bt, p, ru);
        // rp = B u + C p
        spmv_add(&self.b, u, rp);
        if let Some(c) = &self.c {
            spmv_add(c, p, rp);
        }
    }

    /// Convert to a flat `(n_u + n_p) × (n_u + n_p)` CSR matrix (for MINRES).
    pub fn to_flat_csr(&self) -> CsrMatrix<f64> {
        let n_u = self.n_u();
        let n_p = self.n_p();
        let n   = n_u + n_p;
        let mut coo = CooMatrix::<f64>::new(n, n);

        // A block
        for i in 0..n_u {
            for ptr in self.a.row_ptr[i]..self.a.row_ptr[i+1] {
                let j = self.a.col_idx[ptr] as usize;
                coo.add(i, j, self.a.values[ptr]);
            }
        }
        // B^T block (upper right)
        for i in 0..n_u {
            for ptr in self.bt.row_ptr[i]..self.bt.row_ptr[i+1] {
                let j = self.bt.col_idx[ptr] as usize;
                coo.add(i, n_u + j, self.bt.values[ptr]);
            }
        }
        // B block (lower left)
        for i in 0..n_p {
            for ptr in self.b.row_ptr[i]..self.b.row_ptr[i+1] {
                let j = self.b.col_idx[ptr] as usize;
                coo.add(n_u + i, j, self.b.values[ptr]);
            }
        }
        // C block (lower right)
        if let Some(c) = &self.c {
            for i in 0..n_p {
                for ptr in c.row_ptr[i]..c.row_ptr[i+1] {
                    let j = c.col_idx[ptr] as usize;
                    coo.add(n_u + i, n_u + j, c.values[ptr]);
                }
            }
        }
        coo.into_csr()
    }
}

// ─── Block diagonal preconditioner ───────────────────────────────────────────

/// Block-diagonal preconditioner for `[A, B^T; B, C]`:
/// applies `A^{-1}` to the first block and `S^{-1}` (or `C^{-1}`) to the second,
/// where `S = -B A^{-1} B^T + C` is the Schur complement approximation.
///
/// Here we use a diagonal scaling approximation: `A^{-1} ≈ diag(A)^{-1}`.
pub struct BlockDiagonalPrecond {
    /// Inverse diagonal of A.
    pub inv_diag_a: Vec<f64>,
    /// Inverse diagonal of S (approximated as -diag of C or identity).
    pub inv_diag_s: Vec<f64>,
}

impl BlockDiagonalPrecond {
    /// Build from block system.
    pub fn from_system(sys: &BlockSystem) -> Self {
        let n_u = sys.n_u();
        let n_p = sys.n_p();

        let inv_diag_a: Vec<f64> = (0..n_u)
            .map(|i| {
                let d = sys.a.get(i, i);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            })
            .collect();

        let inv_diag_s: Vec<f64> = if let Some(c) = &sys.c {
            (0..n_p).map(|i| {
                let d = c.get(i, i);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            }).collect()
        } else {
            vec![1.0; n_p]
        };

        BlockDiagonalPrecond { inv_diag_a, inv_diag_s }
    }

    /// Apply preconditioner: `z = P^{-1} r`.
    pub fn apply(&self, ru: &[f64], rp: &[f64], zu: &mut [f64], zp: &mut [f64]) {
        for i in 0..zu.len() { zu[i] = self.inv_diag_a[i] * ru[i]; }
        for i in 0..zp.len() { zp[i] = self.inv_diag_s[i] * rp[i]; }
    }
}

// ─── Block triangular preconditioner ────────────────────────────────────────

/// Block upper-triangular preconditioner for `[A, B^T; B, C]`.
///
/// Applies the preconditioner:
/// ```text
///   z_p = S_approx⁻¹ r_p
///   z_u = A_approx⁻¹ (r_u - B^T z_p)
/// ```
///
/// where `A_approx⁻¹ ≈ diag(A)⁻¹` and `S_approx⁻¹ ≈ diag(S)⁻¹`.
/// This is more effective than block-diagonal for saddle-point systems
/// because it captures the upper-triangular coupling.
pub struct BlockTriangularPrecond {
    /// Inverse diagonal of A.
    inv_diag_a: Vec<f64>,
    /// Inverse diagonal of S (Schur complement approximation).
    inv_diag_s: Vec<f64>,
    /// B^T matrix for coupling.
    bt: CsrMatrix<f64>,
}

impl BlockTriangularPrecond {
    /// Build from a block system.
    pub fn from_system(sys: &BlockSystem) -> Self {
        let n_u = sys.n_u();
        let n_p = sys.n_p();

        let inv_diag_a: Vec<f64> = (0..n_u)
            .map(|i| {
                let d = sys.a.get(i, i);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            })
            .collect();

        let inv_diag_s: Vec<f64> = if let Some(c) = &sys.c {
            (0..n_p).map(|i| {
                let d = c.get(i, i);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            }).collect()
        } else {
            // No C block: use identity scaling for the pressure.
            vec![1.0; n_p]
        };

        BlockTriangularPrecond {
            inv_diag_a,
            inv_diag_s,
            bt: sys.bt.clone(),
        }
    }

    /// Apply preconditioner: `z = P⁻¹ r`.
    ///
    /// 1. `z_p = diag(S)⁻¹ r_p`
    /// 2. `z_u = diag(A)⁻¹ (r_u - B^T z_p)`
    pub fn apply(&self, ru: &[f64], rp: &[f64], zu: &mut [f64], zp: &mut [f64]) {
        // Step 1: Schur block
        for i in 0..zp.len() { zp[i] = self.inv_diag_s[i] * rp[i]; }

        // Step 2: coupling + velocity block
        // tmp = B^T z_p
        let mut bt_zp = vec![0.0_f64; zu.len()];
        self.bt.spmv(zp, &mut bt_zp);
        for i in 0..zu.len() {
            zu[i] = self.inv_diag_a[i] * (ru[i] - bt_zp[i]);
        }
    }
}

// ─── Schur complement solver ─────────────────────────────────────────────────

/// Solver for saddle-point systems using the flat GMRES approach.
///
/// **Algorithm**:
/// 1. Flatten the 2×2 block system into a single sparse matrix.
/// 2. Solve with GMRES using a block-diagonal preconditioner
///    `P = diag(diag(A)⁻¹, diag(S_approx)⁻¹)`.
///
/// More robust than Uzawa for general saddle-point systems.
pub struct SchurComplementSolver;

impl SchurComplementSolver {
    /// Solve the saddle-point system.
    pub fn solve(
        sys:  &BlockSystem,
        f:    &[f64],
        g:    &[f64],
        u:    &mut [f64],
        p:    &mut [f64],
        cfg:  &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let n_u = sys.n_u();
        let n_p = sys.n_p();
        assert_eq!(u.len(), n_u); assert_eq!(p.len(), n_p);
        assert_eq!(f.len(), n_u); assert_eq!(g.len(), n_p);

        // Flatten the block system
        let flat = sys.to_flat_csr();
        let n = n_u + n_p;

        let mut rhs = vec![0.0_f64; n];
        rhs[..n_u].copy_from_slice(f);
        rhs[n_u..].copy_from_slice(g);

        // Block-diagonal preconditioner: diag(A)⁻¹ for u, diag(C)⁻¹ or identity for p
        let prec = BlockDiagonalPrecond::from_system(sys);

        // Preconditioned GMRES
        let restart = n.min(1000); // Full GMRES up to moderate size
        let mut x = vec![0.0_f64; n];

        let res = preconditioned_gmres(
            &flat, &rhs, &mut x, restart, cfg,
            |r, z| {
                // Apply block-diagonal preconditioner
                for i in 0..n_u {
                    z[i] = prec.inv_diag_a[i] * r[i];
                }
                for i in 0..n_p {
                    z[n_u + i] = prec.inv_diag_s[i] * r[n_u + i];
                }
            },
        )?;

        u.copy_from_slice(&x[..n_u]);
        p.copy_from_slice(&x[n_u..]);

        Ok(res)
    }
}

/// Right-preconditioned GMRES(m): solve `A M⁻¹ y = b`, then `x = M⁻¹ y`.
fn preconditioned_gmres(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    restart: usize,
    cfg: &SolverConfig,
    precond: impl Fn(&[f64], &mut [f64]),
) -> Result<SolveResult, SolverError> {
    let n = a.nrows;
    let b_norm = norm2(b);
    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }
    let tol = (cfg.rtol * b_norm).max(cfg.atol);

    let mut total_iters = 0;

    for _cycle in 0..((cfg.max_iter + restart - 1) / restart) {
        // r = b - A x
        let mut r = b.to_vec();
        for i in 0..n {
            for ptr in a.row_ptr[i]..a.row_ptr[i+1] {
                let j = a.col_idx[ptr] as usize;
                r[i] -= a.values[ptr] * x[j];
            }
        }
        let beta = norm2(&r);
        if beta < tol {
            return Ok(SolveResult { converged: true, iterations: total_iters, final_residual: beta });
        }

        let m = restart;
        let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; m + 1]; // Krylov basis
        let mut z: Vec<Vec<f64>> = vec![vec![0.0; n]; m];     // Preconditioned vectors
        let mut h = vec![vec![0.0_f64; m]; m + 1];            // Hessenberg
        let mut cs = vec![0.0_f64; m]; // Givens cosines
        let mut sn = vec![0.0_f64; m]; // Givens sines
        let mut e1 = vec![0.0_f64; m + 1];
        e1[0] = beta;

        for i in 0..n { v[0][i] = r[i] / beta; }

        let mut j = 0;
        while j < m && total_iters < cfg.max_iter {
            // z[j] = M⁻¹ v[j]
            precond(&v[j], &mut z[j]);

            // w = A z[j]
            let mut w = vec![0.0_f64; n];
            spmv_add(a, &z[j], &mut w);

            // Arnoldi: orthogonalize w against v[0..=j]
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                axpy_inplace(-h[i][j], &v[i], &mut w);
            }
            h[j+1][j] = norm2(&w);
            if h[j+1][j] > 1e-16 {
                for i in 0..n { v[j+1][i] = w[i] / h[j+1][j]; }
            }

            // Apply previous Givens rotations to column j of H
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i+1][j];
                h[i+1][j] = -sn[i] * h[i][j] + cs[i] * h[i+1][j];
                h[i][j] = tmp;
            }

            // Compute new Givens rotation
            let r_val = (h[j][j] * h[j][j] + h[j+1][j] * h[j+1][j]).sqrt();
            cs[j] = h[j][j] / r_val;
            sn[j] = h[j+1][j] / r_val;
            h[j][j] = r_val;
            h[j+1][j] = 0.0;

            // Apply to e1
            let tmp = cs[j] * e1[j] + sn[j] * e1[j+1];
            e1[j+1] = -sn[j] * e1[j] + cs[j] * e1[j+1];
            e1[j] = tmp;

            total_iters += 1;
            let res_norm = e1[j+1].abs();
            if cfg.verbose {
                println!("[PGMRES] iter={total_iters}: ‖r‖={res_norm:.3e}");
            }
            if res_norm < tol {
                j += 1;
                break;
            }
            j += 1;
        }

        // Back-substitute: solve H y = e1
        let k = j;
        let mut y = vec![0.0_f64; k];
        for i in (0..k).rev() {
            y[i] = e1[i];
            for jj in (i+1)..k {
                y[i] -= h[i][jj] * y[jj];
            }
            y[i] /= h[i][i];
        }

        // Update x += M⁻¹ V y = sum_j y[j] z[j]
        for jj in 0..k {
            axpy_inplace(y[jj], &z[jj], x);
        }

        // Check residual
        let mut r_check = b.to_vec();
        for i in 0..n {
            for ptr in a.row_ptr[i]..a.row_ptr[i+1] {
                let jj = a.col_idx[ptr] as usize;
                r_check[i] -= a.values[ptr] * x[jj];
            }
        }
        let final_res = norm2(&r_check);
        if final_res < tol || total_iters >= cfg.max_iter {
            return Ok(SolveResult {
                converged: final_res < tol,
                iterations: total_iters,
                final_residual: final_res,
            });
        }
    }

    Ok(SolveResult { converged: false, iterations: total_iters, final_residual: f64::NAN })
}

// ─── MINRES for symmetric indefinite systems ──────────────────────────────────

/// MINRES solver for symmetric (possibly indefinite) systems `K x = b`.
///
/// Implements the Lanczos-based MINRES algorithm from Paige & Saunders (1975),
/// following the formulation in SOL Technical Report 2011-2 by Choi & Saunders.
pub struct MinresSolver;

impl MinresSolver {
    pub fn solve(
        a:   &CsrMatrix<f64>,
        b:   &[f64],
        x:   &mut [f64],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let n = a.nrows;
        assert_eq!(b.len(), n); assert_eq!(x.len(), n);

        // r = b - A x
        let mut r = b.to_vec();
        spmv_sub_inplace(a, x, &mut r);
        let mut beta1 = norm2(&r);
        if beta1 < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: 0, final_residual: beta1 });
        }

        // Lanczos vectors
        let mut v_old = vec![0.0_f64; n];
        let mut v: Vec<f64> = r.iter().map(|&ri| ri / beta1).collect();
        let mut v_new = vec![0.0_f64; n];
        let mut beta = beta1;

        // Solution update vectors
        let mut w      = vec![0.0_f64; n];
        let mut w_bar  = vec![0.0_f64; n];

        // QR factorization scalars
        let mut delta_bar: f64;
        let mut cs = -1.0_f64;
        let mut sn = 0.0_f64;
        let mut epsilon: f64;

        let mut phi_bar = beta1;

        for iter in 0..cfg.max_iter {
            // Lanczos step
            spmv_k(&mut v_new, a, &v);
            let alpha = dot(&v_new, &v);
            axpy_inplace(-alpha, &v, &mut v_new);
            axpy_inplace(-beta, &v_old, &mut v_new);
            let beta_new = norm2(&v_new);
            if beta_new > 1e-16 {
                for vi in v_new.iter_mut() { *vi /= beta_new; }
            }

            // QR factorization: apply old Givens to get delta_bar
            let old_beta = beta;
            delta_bar = cs * old_beta + sn * alpha;
            let gamma_bar = sn * old_beta - cs * alpha;
            epsilon = sn * beta_new;

            // New Givens rotation to eliminate beta_new (via gamma_bar)
            let gamma = (gamma_bar * gamma_bar + beta_new * beta_new).sqrt();
            let cs_new;
            let sn_new;
            if gamma.abs() < 1e-30 {
                cs_new = 0.0; sn_new = 0.0;
            } else {
                cs_new = gamma_bar / gamma;
                sn_new = beta_new / gamma;
            }

            // Update solution vectors
            // w_new = (v - delta_bar * w_bar - epsilon * w) / gamma
            // Reuse: w becomes w_old, w_bar becomes w, new w_bar
            let inv_gamma = if gamma.abs() > 1e-30 { 1.0 / gamma } else { 0.0 };
            let mut w_new = vec![0.0_f64; n];
            for i in 0..n {
                w_new[i] = (v[i] - delta_bar * w_bar[i] - epsilon * w[i]) * inv_gamma;
            }

            // update x
            let phi = cs_new * phi_bar;
            phi_bar = sn_new * phi_bar;
            axpy_inplace(phi, &w_new, x);

            // Shift
            w = std::mem::replace(&mut w_bar, w_new);
            std::mem::swap(&mut v_old, &mut v);
            v.clone_from(&v_new);
            beta = beta_new;
            cs = cs_new; sn = sn_new;

            let r_norm = phi_bar.abs();
            if cfg.verbose {
                println!("[MINRES] iter={}: ‖r‖={r_norm:.3e}", iter + 1);
            }
            if r_norm < cfg.atol || r_norm < beta1 * cfg.rtol {
                return Ok(SolveResult {
                    converged: true,
                    iterations: iter + 1,
                    final_residual: r_norm,
                });
            }
        }

        Ok(SolveResult {
            converged: false,
            iterations: cfg.max_iter,
            final_residual: phi_bar.abs(),
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn spmv_add(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows {
        for ptr in a.row_ptr[i]..a.row_ptr[i+1] {
            let j = a.col_idx[ptr] as usize;
            y[i] += a.values[ptr] * x[j];
        }
    }
}

fn spmv_k(out: &mut [f64], a: &CsrMatrix<f64>, x: &[f64]) {
    out.fill(0.0);
    spmv_add(a, x, out);
}

fn spmv_sub_inplace(a: &CsrMatrix<f64>, x: &[f64], b: &mut [f64]) {
    // b = b - A x
    for i in 0..a.nrows {
        let mut s = 0.0;
        for ptr in a.row_ptr[i]..a.row_ptr[i+1] {
            let j = a.col_idx[ptr] as usize;
            s += a.values[ptr] * x[j];
        }
        b[i] -= s;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn axpy_inplace(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) { *yi += alpha * xi; }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small saddle-point system:
    /// A = [[2,0],[0,2]],  B = [[1,1]],  C = 0
    /// Exact solution: u = [1,1], p = 0
    /// f = [2,2], g = [2]  →  (B^T p = 0) → u = [1,1], B u = 2 = g.
    fn small_saddle_point() -> (BlockSystem, Vec<f64>, Vec<f64>) {
        let mut coo_a = CooMatrix::<f64>::new(2, 2);
        coo_a.add(0, 0, 2.0); coo_a.add(1, 1, 2.0);
        let a = coo_a.into_csr();

        let mut coo_bt = CooMatrix::<f64>::new(2, 1);
        coo_bt.add(0, 0, 1.0); coo_bt.add(1, 0, 1.0);
        let bt = coo_bt.into_csr();

        let mut coo_b = CooMatrix::<f64>::new(1, 2);
        coo_b.add(0, 0, 1.0); coo_b.add(0, 1, 1.0);
        let b = coo_b.into_csr();

        let sys = BlockSystem { a, bt, b, c: None };
        let f = vec![2.0_f64, 2.0];
        let g = vec![2.0_f64];
        (sys, f, g)
    }

    #[test]
    fn block_system_to_flat() {
        let (sys, _, _) = small_saddle_point();
        let flat = sys.to_flat_csr();
        assert_eq!(flat.nrows, 3);
        assert_eq!(flat.ncols, 3);
        // A block: flat[0,0]=2, flat[1,1]=2
        assert!((flat.get(0,0) - 2.0).abs() < 1e-12);
        assert!((flat.get(1,1) - 2.0).abs() < 1e-12);
        // B block: flat[2,0]=1, flat[2,1]=1
        assert!((flat.get(2,0) - 1.0).abs() < 1e-12);
        assert!((flat.get(2,1) - 1.0).abs() < 1e-12);
        // B^T block: flat[0,2]=1, flat[1,2]=1
        assert!((flat.get(0,2) - 1.0).abs() < 1e-12);
        assert!((flat.get(1,2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn schur_solver_small_system() {
        let (sys, f, g) = small_saddle_point();
        let mut u = vec![0.0_f64; 2];
        let mut p = vec![0.0_f64; 1];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };
        SchurComplementSolver::solve(&sys, &f, &g, &mut u, &mut p, &cfg).unwrap();
        // Check residuals: A u + B^T p ≈ f, B u ≈ g
        let mut ru = vec![0.0_f64; 2];
        let mut rp = vec![0.0_f64; 1];
        sys.apply(&u, &p, &mut ru, &mut rp);
        let err_u = ru.iter().zip(f.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        let err_p = rp.iter().zip(g.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        assert!(err_u < 1e-6, "residual u: {err_u:.2e}");
        assert!(err_p < 1e-6, "residual p: {err_p:.2e}");
    }

    #[test]
    fn block_diagonal_precond_apply() {
        let (sys, f, g) = small_saddle_point();
        let prec = BlockDiagonalPrecond::from_system(&sys);
        let mut zu = vec![0.0_f64; 2];
        let mut zp = vec![0.0_f64; 1];
        prec.apply(&f, &g, &mut zu, &mut zp);
        // zu[i] = f[i] / diag(A)[i] = 2/2 = 1
        assert!((zu[0] - 1.0).abs() < 1e-12);
        assert!((zu[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn block_system_apply() {
        let (sys, _, _) = small_saddle_point();
        // With u=[1,0], p=[0]: Au = [2,0], Bu = [1]
        let u = vec![1.0_f64, 0.0];
        let p = vec![0.0_f64];
        let mut ru = vec![0.0_f64; 2];
        let mut rp = vec![0.0_f64; 1];
        sys.apply(&u, &p, &mut ru, &mut rp);
        assert!((ru[0] - 2.0).abs() < 1e-12, "ru[0]={}", ru[0]);
        assert!((ru[1] - 0.0).abs() < 1e-12, "ru[1]={}", ru[1]);
        assert!((rp[0] - 1.0).abs() < 1e-12, "rp[0]={}", rp[0]);
    }

    #[test]
    fn minres_spd_identity() {
        // I * x = [1, 2, 3] → x = [1, 2, 3]
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0, 1.0); coo.add(1, 1, 1.0); coo.add(2, 2, 1.0);
        let k = coo.into_csr();
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0_f64; 3];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10, verbose: false, ..SolverConfig::default() };
        let res = MinresSolver::solve(&k, &b, &mut x, &cfg).unwrap();
        assert!(res.converged);
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 2.0).abs() < 1e-8);
        assert!((x[2] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn minres_spd_diagonal() {
        // diag(2, 3) x = [4, 9] → x = [2, 3]
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0); coo.add(1, 1, 3.0);
        let k = coo.into_csr();
        let b = vec![4.0, 9.0];
        let mut x = vec![0.0_f64; 2];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10, verbose: false, ..SolverConfig::default() };
        let res = MinresSolver::solve(&k, &b, &mut x, &cfg).unwrap();
        assert!(res.converged);
        assert!((x[0] - 2.0).abs() < 1e-8, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-8, "x[1]={}", x[1]);
    }

    #[test]
    fn minres_symmetric_indefinite() {
        // [[2, 0, 1], [0, 2, 1], [1, 1, 0]] x = [2, 2, 2] → x = [1, 1, 0]
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0, 2.0); coo.add(1, 1, 2.0);
        coo.add(0, 2, 1.0); coo.add(2, 0, 1.0);
        coo.add(1, 2, 1.0); coo.add(2, 1, 1.0);
        let k = coo.into_csr();
        let b = vec![2.0, 2.0, 2.0];
        let mut x = vec![0.0_f64; 3];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };
        let res = MinresSolver::solve(&k, &b, &mut x, &cfg).unwrap();
        // Verify actual residual
        let mut kx = vec![0.0_f64; 3];
        spmv_add(&k, &x, &mut kx);
        let res_actual = kx.iter().zip(b.iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
        assert!(res.converged, "MINRES didn't converge: iters={}, est_res={:.2e}",
                res.iterations, res.final_residual);
        assert!(res_actual < 1e-6, "MINRES actual residual = {res_actual:.2e}, x={x:?}");
    }
}
