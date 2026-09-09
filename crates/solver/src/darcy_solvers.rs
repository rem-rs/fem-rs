//! Serial port of MFEM `miniapps/solvers/darcy_solver.{hpp,cpp}` (1:1, assembled variant).
//!
//! Contains
//! * [`IterSolveParameters`] — MFEM `blocksolvers::IterSolveParameters`,
//! * [`mfem_minres`] — exact port of `MINRESSolver::Mult` (linalg/solvers.cpp),
//!   including the SPD-preconditioned path and the `||r||_B` print format,
//! * [`BdpMinresSolver`] — MFEM `blocksolvers::BDPMinresSolver`: block-diagonal
//!   preconditioned MINRES for the Darcy saddle system
//!   `[[M, Bᵀ], [B, 0]]` with `diag(M)⁻¹` and an approximate inverse of the
//!   Schur complement `S = B·diag(M)⁻¹·Bᵀ` (C++ uses hypre `BoomerAMG`; the
//!   serial port offers AMG via `fem-amg`, an exact dense inverse, or Jacobi).
//!
//! Not ported (parallel-only): `HypreParMatrix` machinery, `SetEssZeroDofs`
//! MPI bookkeeping is kept as a plain dof list.
//!
//! PA / matrix-free variants of these solvers are **not** part of this port
//! (recorded as a gap, see `div_free_solver.rs` module docs).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::amg::{AmgConfig, AmgSolver};
use fem_linalg::{CooMatrix, CsrMatrix};

/// MFEM `blocksolvers::IterSolveParameters`.
#[derive(Debug, Clone)]
pub struct IterSolveParameters {
    /// MFEM `print_level` (1 prints the per-iteration `||r||_B` MINRES lines).
    pub print_level: i32,
    pub max_iter: usize,
    pub abs_tol: f64,
    pub rel_tol: f64,
}

impl Default for IterSolveParameters {
    fn default() -> Self {
        Self {
            print_level: 0,
            max_iter: 500,
            abs_tol: 1e-12,
            rel_tol: 1e-9,
        }
    }
}

/// MFEM `SetOptions` equivalent: check the parameters once.
fn check_options(param: &IterSolveParameters) {
    assert!(param.max_iter > 0, "IterSolveParameters: max_iter must be > 0");
    assert!(param.rel_tol >= 0.0 && param.abs_tol >= 0.0);
}

/// Exact port of `MINRESSolver::Mult` (mfem/linalg/solvers.cpp, v4.10).
///
/// Based on the MINRES algorithm on p. 86, Fig. 6.9 in "Iterative Krylov
/// Methods for Large Linear Systems", by Henk A. van der Vorst, 2003,
/// extended to support an SPD preconditioner (as in MFEM).
///
/// `apply_op`  — the symmetric operator `A` (`y = A x`),
/// `apply_prec` — the SPD preconditioner `y = M⁻¹ x` (ignored if `has_prec`),
/// `x` — overwritten with the solution (MFEM `iterative_mode == false`).
///
/// Returns `(iterations, converged, final_norm)` with `final_norm = |eta|`
/// (the preconditioned residual norm estimate `||r||_B`).
pub fn mfem_minres(
    n: usize,
    apply_op: &dyn Fn(&[f64], &mut [f64]),
    apply_prec: Option<&dyn Fn(&[f64], &mut [f64])>,
    b: &[f64],
    x: &mut [f64],
    param: &IterSolveParameters,
) -> (usize, bool, f64) {
    check_options(param);
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(x.len(), n);

    let has_prec = apply_prec.is_some();

    // v1 = b (iterative_mode == false → x = 0); z = prec ? u1 : v1.
    let mut v0 = vec![0.0_f64; n];
    let mut v1 = b.to_vec();
    let mut u1 = vec![0.0_f64; n];
    let mut q = vec![0.0_f64; n];
    let mut w0 = vec![0.0_f64; n];
    let mut w1 = vec![0.0_f64; n];
    x.fill(0.0);

    let z: &[f64] = if has_prec {
        apply_prec.unwrap()(&v1, &mut u1);
        &u1
    } else {
        &v1
    };
    let dot_zv: f64 = z.iter().zip(&v1).map(|(&zi, &vi)| zi * vi).sum();
    let initial_norm = dot_zv.sqrt();
    assert!(initial_norm.is_finite(), "MINRES: eta is not finite");
    let mut eta = initial_norm;
    let mut beta = initial_norm;
    let mut gamma0 = 1.0_f64;
    let mut gamma1 = 1.0_f64;
    let mut sigma0 = 0.0_f64;
    let mut sigma1 = 0.0_f64;

    let norm_goal = (param.rel_tol * eta).max(param.abs_tol);

    if eta <= norm_goal {
        return (0, true, eta.abs());
    }
    if param.print_level >= 1 {
        println!("MINRES: iteration {:3}: ||r||_B = {}", 0, eta);
    }

    let mut it = 1_usize;
    let mut converged = false;
    while it <= param.max_iter {
        for vi in v1.iter_mut() {
            *vi /= beta;
        }
        if has_prec {
            for ui in u1.iter_mut() {
                *ui /= beta;
            }
        }
        let z: &[f64] = if has_prec { &u1 } else { &v1 };
        apply_op(z, &mut q);
        let alpha: f64 = z.iter().zip(&q).map(|(&zi, &qi)| zi * qi).sum();
        assert!(alpha.is_finite(), "MINRES: alpha is not finite");
        if it > 1 {
            // q.Add(-beta, v0)
            for (qi, &v0i) in q.iter_mut().zip(&v0) {
                *qi -= beta * v0i;
            }
        }
        // v0 = q - alpha * v1   ("add(q, -alpha, v1, v0)")
        for i in 0..n {
            v0[i] = q[i] - alpha * v1[i];
        }

        let delta = gamma1 * alpha - gamma0 * sigma1 * beta;
        let rho3 = sigma0 * beta;
        let rho2 = sigma1 * alpha + gamma0 * gamma1 * beta;
        if has_prec {
            apply_prec.unwrap()(&v0, &mut q);
            let dot_vq: f64 = v0.iter().zip(&q).map(|(&vi, &qi)| vi * qi).sum();
            beta = dot_vq.sqrt();
        } else {
            beta = v0.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();
        }
        assert!(beta.is_finite(), "MINRES: beta is not finite");
        let rho1 = delta.hypot(beta);

        if it == 1 {
            // w0 = (1/rho1) z        ((w0 == 0) and (w1 == 0))
            let inv = 1.0 / rho1;
            for i in 0..n {
                w0[i] = inv * z[i];
            }
        } else if it == 2 {
            // w0 = (1/rho1) z - (rho2/rho1) w1   ((w0 == 0))
            let inv = 1.0 / rho1;
            for i in 0..n {
                w0[i] = inv * z[i] - (rho2 / rho1) * w1[i];
            }
        } else {
            // w0 = -(rho3/rho1) w0 - (rho2/rho1) w1 + (1/rho1) z
            for i in 0..n {
                w0[i] = -(rho3 / rho1) * w0[i] - (rho2 / rho1) * w1[i] + z[i] / rho1;
            }
        }

        gamma0 = gamma1;
        gamma1 = delta / rho1;

        // x.Add(gamma1 * eta, w0)
        for (xi, &w0i) in x.iter_mut().zip(&w0) {
            *xi += gamma1 * eta * w0i;
        }

        sigma0 = sigma1;
        sigma1 = beta / rho1;

        eta = -sigma1 * eta;
        assert!(eta.is_finite(), "MINRES: eta is not finite");

        if eta.abs() <= norm_goal {
            converged = true;
            break;
        }

        if param.print_level >= 1 {
            println!("MINRES: iteration {:3}: ||r||_B = {}", it, eta.abs());
        }

        if has_prec {
            std::mem::swap(&mut u1, &mut q);
        }
        std::mem::swap(&mut v0, &mut v1);
        std::mem::swap(&mut w0, &mut w1);

        it += 1;
    }
    if !converged {
        it -= 1; // MFEM: `it--` after the exhausted loop
    }

    if param.print_level >= 1 {
        println!("MINRES: iteration {:3}: ||r||_B = {}", it, eta.abs());
        println!("MINRES: Number of iterations: {:3}", it);
    }
    (it, converged, eta.abs())
}

/// Approximate inverse used for the Schur block of [`BdpMinresSolver`].
///
/// C++ `BDPMinresSolver` always uses hypre `BoomerAMG` on `S`. The serial port
/// keeps AMG as the default (`fem-amg` smoothed aggregation) and adds an exact
/// dense inverse (used for 1:1 iteration-count comparisons against a serial
/// C++ harness) and a weak Jacobi fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurMode {
    /// fem-amg smoothed-aggregation AMG V-cycle (hypre BoomerAMG stand-in).
    Amg,
    /// Exact dense inverse of `S` (small problems only).
    Dense,
    /// `diag(S)⁻¹`.
    Diag,
}

enum SchurApprox {
    Amg(AmgSolver<f64>),
    Dense { lu: Vec<f64>, piv: Vec<usize>, n: usize },
    Diag(Vec<f64>),
}

/// Serial port of MFEM `blocksolvers::BDPMinresSolver`.
///
/// Solves the assembled Darcy saddle system `[[M, Bᵀ], [B, 0]]` with MINRES
/// preconditioned by `diag(diag(M)⁻¹, S⁻¹)`, `S = B·diag(M)⁻¹·Bᵀ`.
pub struct BdpMinresSolver {
    n_u: usize,
    n: usize,
    m: CsrMatrix<f64>,
    b: CsrMatrix<f64>,
    bt: CsrMatrix<f64>,
    inv_diag_m: Vec<f64>,
    schur: SchurApprox,
    ess_zero_dofs: Vec<usize>,
    param: IterSolveParameters,
    last_iters: AtomicUsize,
    last_converged: AtomicBool,
}

impl BdpMinresSolver {
    /// Build the solver for `M` (`n_u × n_u`, SPD) and `B` (`n_p × n_u`).
    pub fn new(m: &CsrMatrix<f64>, b: &CsrMatrix<f64>, param: IterSolveParameters, schur_mode: SchurMode) -> Self {
        assert_eq!(m.nrows, m.ncols, "BDPMinresSolver: M must be square");
        assert_eq!(m.ncols, b.ncols, "BDPMinresSolver: B must be (n_p × n_u)");
        let n_u = m.nrows;
        let n_p = b.nrows;

        // S = B · diag(M)⁻¹ · Bᵀ  (hypre: InvScaleRows on Bᵀ, ParMult, ScaleRows)
        let bt = b.transpose();
        let mut minv_bt_coo = CooMatrix::<f64>::new(n_u, n_p);
        for i in 0..n_u {
            for p in bt.row_ptr[i]..bt.row_ptr[i + 1] {
                let j = bt.col_idx[p] as usize;
                minv_bt_coo.add(i, j, bt.values[p] / m.get(i, i));
            }
        }
        let s = b.multiply(&minv_bt_coo.into_csr());

        let schur = match schur_mode {
            SchurMode::Amg => {
                // Guard zero rows (eliminated dofs) so the AMG hierarchy is SPD.
                let s_guarded = guard_zero_diagonal(&s);
                SchurApprox::Amg(AmgSolver::setup(&s_guarded, AmgConfig::default()))
            }
            SchurMode::Dense => {
                let n = n_p;
                let mut lu = vec![0.0; n * n];
                for i in 0..n {
                    for p in s.row_ptr[i]..s.row_ptr[i + 1] {
                        lu[i * n + s.col_idx[p] as usize] = s.values[p];
                    }
                }
                let mut piv = vec![0_usize; n];
                fem_linalg::dense::lu_factor(&mut lu, n, &mut piv).expect("BDPMinresSolver: S is singular");
                SchurApprox::Dense { lu, piv, n }
            }
            SchurMode::Diag => {
                let d = (0..n_p)
                    .map(|i| {
                        let dii = s.get(i, i);
                        if dii.abs() < 1e-300 { 1.0 } else { 1.0 / dii }
                    })
                    .collect();
                SchurApprox::Diag(d)
            }
        };

        Self {
            n_u,
            n: n_u + n_p,
            m: m.clone(),
            b: b.clone(),
            bt,
            inv_diag_m: (0..n_u).map(|i| 1.0 / m.get(i, i)).collect(),
            schur,
            ess_zero_dofs: Vec::new(),
            param,
            last_iters: AtomicUsize::new(0),
            last_converged: AtomicBool::new(false),
        }
    }

    /// MFEM `SetEssZeroDofs`: solution dofs forced to zero after the solve.
    pub fn set_ess_zero_dofs(&mut self, dofs: &[usize]) {
        self.ess_zero_dofs = dofs.to_vec();
    }

    /// MFEM `BDPMinresSolver::Mult`: `y ← MINRES(x)`, then zero the ess dofs.
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        let (iters, converged, _) = mfem_minres(
            self.n,
            &|v, w| self.apply_op(v, w),
            Some(&|v, w| self.apply_prec(v, w)),
            x,
            y,
            &self.param,
        );
        self.last_iters.store(iters, Ordering::Relaxed);
        self.last_converged.store(converged, Ordering::Relaxed);
        for &dof in &self.ess_zero_dofs {
            y[dof] = 0.0;
        }
    }

    /// MFEM `GetNumIterations`.
    pub fn num_iterations(&self) -> usize {
        self.last_iters.load(Ordering::Relaxed)
    }

    /// Whether the last `mult` converged.
    pub fn converged(&self) -> bool {
        self.last_converged.load(Ordering::Relaxed)
    }

    /// Block sizes `[0, n_u, n_u + n_p]` (MFEM `DarcySolver::offsets_`).
    pub fn offsets(&self) -> [usize; 3] {
        [0, self.n_u, self.n]
    }

    fn apply_op(&self, v: &[f64], w: &mut [f64]) {
        // [[M, Bᵀ], [B, 0]] · v
        self.m.spmv(&v[..self.n_u], &mut w[..self.n_u]);
        let mut bt_vp = vec![0.0; self.n_u];
        self.bt.spmv(&v[self.n_u..], &mut bt_vp);
        for (wi, bi) in w[..self.n_u].iter_mut().zip(&bt_vp) {
            *wi += bi;
        }
        self.b.spmv(&v[..self.n_u], &mut w[self.n_u..]);
    }

    fn apply_prec(&self, v: &[f64], w: &mut [f64]) {
        // diag(diag(M)⁻¹, S⁻¹) · v
        for i in 0..self.n_u {
            w[i] = self.inv_diag_m[i] * v[i];
        }
        let vp = &v[self.n_u..];
        let wp = &mut w[self.n_u..];
        match &self.schur {
            SchurApprox::Amg(amg) => {
                let z = amg.precond_apply(vp);
                wp.copy_from_slice(&z);
            }
            SchurApprox::Dense { lu, piv, n } => {
                wp.copy_from_slice(vp);
                fem_linalg::dense::lu_solve(lu, *n, piv, wp);
            }
            SchurApprox::Diag(d) => {
                for (wi, (&vi, &di)) in wp.iter_mut().zip(vp.iter().zip(d)) {
                    *wi = vi * di;
                }
            }
        }
    }
}

/// Set `A[i][i] = 1` for rows with (numerically) empty diagonal — keeps
/// downstream factorizations/AMG hierarchies non-singular.  Only affects the
/// zero rows; used as a guard for eliminated dofs (matches the intent of
/// hypre `EliminateZeroRows` + identity patches in MFEM auxiliary solvers).
pub(crate) fn guard_zero_diagonal(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        let mut has_diag = false;
        let mut row_max = 0.0_f64;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(i, a.col_idx[p] as usize, a.values[p]);
            if a.col_idx[p] as usize == i {
                has_diag = true;
            }
            row_max = row_max.max(a.values[p].abs());
        }
        if !has_diag && row_max < 1e-30 {
            coo.add(i, i, 1.0);
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::SolverConfig;

    fn lap1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -1.0);
            }
        }
        coo.into_csr()
    }

    /// MINRES port solves an SPD system to the requested tolerance
    /// (verified against a dense LU reference solve).
    #[test]
    fn minres_matches_direct_solve() {
        let a = lap1d(40);
        let b = vec![1.0; 40];
        let param = IterSolveParameters {
            print_level: 0,
            max_iter: 500,
            abs_tol: 1e-14,
            rel_tol: 1e-10,
        };
        let mut x = vec![0.0; 40];
        let (it, conv, _) = mfem_minres(40, &|v, w| a.spmv(v, w), None, &b, &mut x, &param);
        assert!(conv, "MINRES did not converge");
        // Dense reference solution.
        let mut lu = vec![0.0; 40 * 40];
        for i in 0..40 {
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                lu[i * 40 + a.col_idx[p] as usize] = a.values[p];
            }
        }
        let mut xr = b.clone();
        let mut piv = vec![0_usize; 40];
        fem_linalg::dense::lu_factor(&mut lu, 40, &mut piv).unwrap();
        fem_linalg::dense::lu_solve(&lu, 40, &piv, &mut xr);
        let diff = x.iter().zip(&xr).map(|(p, q)| (p - q).abs()).fold(0.0_f64, f64::max);
        eprintln!("[dbg] max|x−x_ref| = {diff:.3e}, iters = {it}");
        assert!(diff < 1e-7, "solution mismatch, |x−x_ref|_max = {diff:.3e}");
    }

    /// BDP-MINRES with an exact dense Schur inverse solves a small saddle
    /// system to machine precision.
    #[test]
    fn bdp_dense_schur_small_saddle() {
        // M = I₂, B = [1, 1]  →  [[1,0,1],[0,1,1],[1,1,0]] x = [2,2,2]⁻rhs
        let mut coo_m = CooMatrix::<f64>::new(2, 2);
        coo_m.add(0, 0, 1.0);
        coo_m.add(1, 1, 1.0);
        let m = coo_m.into_csr();
        let mut coo_b = CooMatrix::<f64>::new(1, 2);
        coo_b.add(0, 0, 1.0);
        coo_b.add(0, 1, 1.0);
        let b = coo_b.into_csr();

        let solver = BdpMinresSolver::new(&m, &b, IterSolveParameters::default(), SchurMode::Dense);
        // rhs: f = M u_ex + Bᵀ p_ex, g = B u_ex with u_ex = [1, -1], p_ex = 2
        // → f = [1 + 2, -1 + 2] = [3, 1], g = 0
        let rhs = [3.0, 1.0, 0.0];
        let mut sol = vec![0.0; 3];
        solver.mult(&rhs, &mut sol);
        assert!(solver.converged(), "BDP did not converge");
        assert!(solver.num_iterations() < 10, "too many iterations: {}", solver.num_iterations());
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] + 1.0).abs() < 1e-9);
        assert!((sol[2] - 2.0).abs() < 1e-9);
    }

    /// BDP with AMG Schur preconditioner converges on a 1-D Darcy system
    /// (diagonal M — `S = B·diag(M)⁻¹·Bᵀ` is the sparse 1-D graph Laplacian).
    #[test]
    fn bdp_amg_schur_converges() {
        let n_u = 30;
        let mut coo_m = CooMatrix::<f64>::new(n_u, n_u);
        for i in 0..n_u {
            coo_m.add(i, i, 2.0);
        }
        let m = coo_m.into_csr();
        // B = difference operator: (B u)_i = u_i - u_{i-1} (n_p = n_u - 1)
        let n_p = n_u - 1;
        let mut coo_b = CooMatrix::<f64>::new(n_p, n_u);
        for i in 0..n_p {
            coo_b.add(i, i, -1.0);
            coo_b.add(i, i + 1, 1.0);
        }
        let b = coo_b.into_csr();

        let solver = BdpMinresSolver::new(&m, &b, IterSolveParameters {
            rel_tol: 1e-8,
            ..IterSolveParameters::default()
        }, SchurMode::Amg);
        // Exact solution u = 1, p = 0 → f = M·1 = 2·1, g = 0.
        let mut rhs = vec![0.0; n_u + n_p];
        for r in rhs[..n_u].iter_mut() {
            *r = 2.0;
        }
        let mut sol = vec![0.0; n_u + n_p];
        solver.mult(&rhs, &mut sol);
        assert!(solver.converged(), "BDP+AMG did not converge");
        for i in 0..n_u {
            assert!((sol[i] - 1.0).abs() < 1e-5, "u[{i}] = {}", sol[i]);
        }
        assert!((sol[n_u..].iter().sum::<f64>()).abs() < 1e-4);
    }

    /// `SetEssZeroDofs` forces the listed dofs to zero in the output.
    #[test]
    fn bdp_ess_zero_dofs() {
        let mut coo_m = CooMatrix::<f64>::new(2, 2);
        coo_m.add(0, 0, 1.0);
        coo_m.add(1, 1, 1.0);
        let m = coo_m.into_csr();
        let mut coo_b = CooMatrix::<f64>::new(1, 2);
        coo_b.add(0, 0, 1.0);
        coo_b.add(0, 1, 1.0);
        let b = coo_b.into_csr();
        let mut solver = BdpMinresSolver::new(&m, &b, IterSolveParameters::default(), SchurMode::Dense);
        solver.set_ess_zero_dofs(&[0]);
        let rhs = [3.0, 1.0, 0.0];
        let mut sol = vec![1.0; 3]; // nonzero initial guess must be overwritten too
        solver.mult(&rhs, &mut sol);
        assert_eq!(sol[0], 0.0, "ess dof must be zeroed");
    }
}
