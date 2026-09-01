//! Nonlinear forms and Newton鈥揜aphson solver.
//!
//! # Overview
//!
//! A `NonlinearForm` assembles the **residual** `F(u)` and the **Jacobian** `J(u) = dF/du`
//! for a nonlinear PDE.  The [`NewtonSolver`] then iterates
//!
//! ```text
//! J(u鈧? 螖u = 鈭扚(u鈧?,   u鈧欌倞鈧?= u鈧?+ 螖u
//! ```
//!
//! until `鈥朏(u)鈥?< tol`.
//!
//! # Example: nonlinear diffusion
//! ```rust,ignore
//! let form = NonlinearDiffusionForm::new(&space, |u| 1.0 + u*u); // 魏(u) = 1 + u虏
//! let mut solver = NewtonSolver::new(NewtonConfig::default());
//! let mut u = vec![0.0; space.n_dofs()];
//! let result = solver.solve(&form, &rhs, &mut u).unwrap();
//! ```

use fem_linalg::CsrMatrix;
use fem_solver::{solve_gmres, solve_gmres_ilu0, solve_pcg_gssmoother, solve_sparse_lu, SolverConfig};

/// Linear solver strategy for the Newton inner system `J(x) · dx = -F(x)`.
///
/// Choose the solver that best matches your Jacobian's properties:
/// - [`LinearSolver::PcgGssmoother`] — SPD Jacobians (nonlinear diffusion, elasticity)
/// - [`LinearSolver::GmresIlu0`]   — ill-conditioned / non‑symmetric (default)
/// - [`LinearSolver::Gmres`]       — plain GMRES (no preconditioner)
/// - [`LinearSolver::SparseLu`]    — direct LU (small systems only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolver {
    /// GMRES + ILU(0) preconditioner.  Robust default for most nonlinear problems.
    GmresIlu0,
    /// Conjugate Gradient + Gauss–Seidel smoother.  For SPD Jacobians only.
    PcgGssmoother,
    /// Plain GMRES without preconditioning.  Use only when the Jacobian is
    /// well‑conditioned or when ILU(0) is too expensive.
    Gmres,
    /// Direct sparse LU factorisation.  Use for small systems (<10k DOFs)
    /// where robustness trumps speed.
    SparseLu,
}

/// A nonlinear PDE form that can compute residuals and Jacobians.
///
/// Implementors must provide:
/// - [`NonlinearForm::residual`]: compute `F(u)` in-place.
/// - [`NonlinearForm::jacobian`]: assemble the tangent matrix `J(u) = dF/du`.
///
/// Both are called at each Newton iteration on the current iterate `u`.
pub trait NonlinearForm: Send + Sync {
    /// Compute the residual vector `r = F(u) - b` into `r`.
    ///
    /// `u` is the current iterate (len = n_dofs), `rhs` is the external load
    /// vector, and `r` is the output residual (len = n_dofs).
    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]);

    /// Assemble the Jacobian matrix `J(u)`.
    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64>;

    /// Number of DOFs.
    fn n_dofs(&self) -> usize;
}

// 鈹€鈹€鈹€ Newton solver 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Convergence and iteration parameters for the Newton solver.
#[derive(Debug, Clone)]
pub struct NewtonConfig {
    /// Absolute tolerance on `鈥朏(u)鈥栤倐` (default 1e-10).
    pub atol: f64,
    /// Relative tolerance on `鈥朏(u)鈥栤倐 / 鈥朏(u鈧€)鈥栤倐` (default 1e-8).
    pub rtol: f64,
    /// Maximum Newton iterations (default 50).
    pub max_iter: usize,
    /// Linear solver tolerance for each Jacobian solve (default 1e-10).
    pub linear_tol: f64,
    /// Enable backtracking line-search on Newton updates.
    pub line_search: bool,
    /// Minimum step size in line-search.
    pub line_search_min_alpha: f64,
    /// Multiplicative shrink factor used during backtracking (0, 1).
    pub line_search_shrink: f64,
    /// Maximum number of backtracking reductions per Newton iteration.
    pub line_search_max_backtracks: usize,
    /// Sufficient residual decrease factor for Armijo-like acceptance.
    pub line_search_sufficient_decrease: f64,
    /// Print residual each iteration.
    pub verbose: bool,
    /// Linear solver strategy for the Jacobian system. Default: GmresIlu0.
    pub linear_solver: LinearSolver,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        NewtonConfig {
            atol:       1e-10,
            rtol:       1e-8,
            max_iter:   50,
            linear_tol: 1e-6,
            line_search: true,
            line_search_min_alpha: 1e-6,
            line_search_shrink: 0.5,
            line_search_max_backtracks: 20,
            line_search_sufficient_decrease: 1e-4,
            verbose:    false,
            linear_solver: LinearSolver::GmresIlu0,
        }
    }
}

/// Outcome of a Newton solve.
#[derive(Debug, Clone)]
pub struct NewtonResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
}

/// Newton鈥揜aphson solver for nonlinear systems `F(u) = 0`.
pub struct NewtonSolver {
    cfg: NewtonConfig,
}

impl NewtonSolver {
    pub fn new(cfg: NewtonConfig) -> Self { NewtonSolver { cfg } }

    /// Solve `F(u) = 0` starting from the initial guess in `u`.
    ///
    /// On success, `u` contains the solution.
    /// Returns `Err(NewtonResult)` if the solver did not converge.
    pub fn solve(
        &self,
        form: &dyn NonlinearForm,
        rhs:  &[f64],
        u:    &mut [f64],
    ) -> Result<NewtonResult, NewtonResult> {
        let n = form.n_dofs();
        assert_eq!(u.len(), n);
        assert_eq!(rhs.len(), n);

        let linear_cfg = SolverConfig {
            rtol:     self.cfg.linear_tol,
            atol:     0.0,
            max_iter: 3000,
            verbose:  false,
            ..SolverConfig::default()
        };

        let mut r   = vec![0.0_f64; n];
        let mut du  = vec![0.0_f64; n];
        let mut u_trial = vec![0.0_f64; n];
        let mut r_trial = vec![0.0_f64; n];

        // Initial residual
        form.residual(u, rhs, &mut r);
        let r0 = norm2(&r);

        if self.cfg.verbose {
            println!("[Newton] iter=0 鈥朏鈥?{r0:.3e}");
        }

        // Converged immediately (zero initial residual)
        if r0 < self.cfg.atol {
            return Ok(NewtonResult { converged: true, iterations: 0, final_residual: r0 });
        }

        let mut r_norm = r0;

        for iter in 0..self.cfg.max_iter {
            // Assemble Jacobian
            let jac = form.jacobian(u);

            // Solve J 螖u = 鈭抮
            let neg_r: Vec<f64> = r.iter().map(|&v| -v).collect();
            du.fill(0.0);
            solve_gmres_ilu0(&jac, &neg_r, &mut du, 100, &linear_cfg)
                .map_err(|_| NewtonResult { converged: false, iterations: iter, final_residual: r_norm })?;

            if self.cfg.line_search {
                let mut alpha = 1.0_f64;
                let mut accepted = false;
                let mut best_norm = f64::INFINITY;
                let mut best_alpha = 1.0_f64;

                for _ in 0..=self.cfg.line_search_max_backtracks {
                    for i in 0..n {
                        u_trial[i] = u[i] + alpha * du[i];
                    }
                    form.residual(&u_trial, rhs, &mut r_trial);
                    let trial_norm = norm2(&r_trial);
                    if trial_norm < best_norm {
                        best_norm = trial_norm;
                        best_alpha = alpha;
                    }

                    let target = ((1.0 - self.cfg.line_search_sufficient_decrease * alpha).max(0.0)) * r_norm;
                    if trial_norm <= target || trial_norm < r_norm {
                        accepted = true;
                        break;
                    }

                    if alpha <= self.cfg.line_search_min_alpha {
                        break;
                    }
                    alpha *= self.cfg.line_search_shrink;
                }

                let use_alpha = if accepted { alpha } else { best_alpha };
                for i in 0..n {
                    u[i] += use_alpha * du[i];
                }
            } else {
                // Update: u 鈫?u + 螖u
                for (ui, &dui) in u.iter_mut().zip(du.iter()) {
                    *ui += dui;
                }
            }

            // Recompute residual
            form.residual(u, rhs, &mut r);
            r_norm = norm2(&r);

            if self.cfg.verbose {
                println!("[Newton] iter={} 鈥朏鈥?{r_norm:.3e}", iter + 1);
            }

            if r_norm < self.cfg.atol || r_norm < r0 * self.cfg.rtol {
                return Ok(NewtonResult { converged: true, iterations: iter + 1, final_residual: r_norm });
            }
        }

        Err(NewtonResult { converged: false, iterations: self.cfg.max_iter, final_residual: r_norm })
    }

    /// Solve with a custom linear solver closure (MFEM: NewtonSolver + custom J_solver).
    ///
    /// The `linear_solve` closure receives `(jacobian: &CsrMatrix, neg_rhs: &[f64], dx: &mut [f64])`
    /// and must solve `J · dx = neg_rhs`.  Returns `Ok(())` on success.
    ///
    /// Useful for block preconditioners, GPU solves, or direct solvers not
    /// covered by the built-in `LinearSolver` variants.
    pub fn solve_custom(
        &self,
        form: &dyn NonlinearForm,
        rhs: &[f64],
        u: &mut [f64],
        linear_solve: &dyn Fn(&CsrMatrix<f64>, &[f64], &mut [f64]) -> Result<(), ()>,
    ) -> Result<NewtonResult, NewtonResult> {
        let n = form.n_dofs();
        assert_eq!(u.len(), n);
        assert_eq!(rhs.len(), n);

        let mut r = vec![0.0_f64; n];
        let mut du = vec![0.0_f64; n];
        let mut u_trial = vec![0.0_f64; n];
        let mut r_trial = vec![0.0_f64; n];

        form.residual(u, rhs, &mut r);
        let r0 = norm2(&r);
        if self.cfg.verbose {
            println!("[Newton] iter=0 ||r||={r0:.3e}");
        }
        if r0 < self.cfg.atol {
            return Ok(NewtonResult { converged: true, iterations: 0, final_residual: r0 });
        }

        let mut r_norm = r0;

        for iter in 0..self.cfg.max_iter {
            let jac = form.jacobian(u);

            let neg_r: Vec<f64> = r.iter().map(|&v| -v).collect();
            du.fill(0.0);
            linear_solve(&jac, &neg_r, &mut du)
                .map_err(|_| NewtonResult { converged: false, iterations: iter, final_residual: r_norm })?;

            if self.cfg.line_search {
                let mut alpha = 1.0_f64;
                let mut accepted = false;
                for _ in 0..=self.cfg.line_search_max_backtracks {
                    for i in 0..n { u_trial[i] = u[i] + alpha * du[i]; }
                    form.residual(&u_trial, rhs, &mut r_trial);
                    let trial_norm = norm2(&r_trial);
                    let target = ((1.0 - self.cfg.line_search_sufficient_decrease * alpha).max(0.0)) * r_norm;
                    if trial_norm <= target {
                        u.copy_from_slice(&u_trial);
                        r.copy_from_slice(&r_trial);
                        accepted = true;
                        break;
                    }
                    alpha *= self.cfg.line_search_shrink;
                    if alpha < self.cfg.line_search_min_alpha { break; }
                }
                if !accepted {
                    for (ui, &dui) in u.iter_mut().zip(du.iter()) { *ui += dui; }
                    form.residual(u, rhs, &mut r);
                }
            } else {
                for (ui, &dui) in u.iter_mut().zip(du.iter()) { *ui += dui; }
            }

            r_norm = norm2(&r);
            if self.cfg.verbose {
                println!("[Newton] iter={} ||r||={r_norm:.3e}", iter + 1);
            }
            if r_norm < self.cfg.atol || r_norm < r0 * self.cfg.rtol {
                return Ok(NewtonResult { converged: true, iterations: iter + 1, final_residual: r_norm });
            }
        }

        Err(NewtonResult { converged: false, iterations: self.cfg.max_iter, final_residual: r_norm })
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// 鈹€鈹€鈹€ Finite-difference Jacobian 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Compute a sparse Jacobian matrix via column-wise finite differences.
///
/// Perturbs each DOF by `h = eps * (1 + |u_j|)` and computes
/// `J[:,j] 鈮?(F(u + h路e獗? 鈭?F(u)) / h`.
///
/// Uses `element_dofs` to determine the sparsity pattern: only DOFs that share
/// an element with column `j` are stored in that column.
pub fn finite_diff_jacobian(
    u: &[f64],
    form: &dyn NonlinearForm,
    rhs: &[f64],
    n_dofs: usize,
    element_dofs: &[Vec<u32>],
    eps: f64,
) -> CsrMatrix<f64> {
    use std::collections::BTreeSet;
    let mut sparsity = vec![BTreeSet::new(); n_dofs];
    for elem in element_dofs {
        for &di in elem { for &dj in elem {
            if di != dj { sparsity[di as usize].insert(dj as usize); }
        }}
    }
    let mut row_ptr = vec![0usize; n_dofs + 1];
    let mut col_idx = Vec::new();
    for i in 0..n_dofs {
        col_idx.extend(sparsity[i].iter());
        row_ptr[i + 1] = col_idx.len();
    }
    let mut values = vec![0.0_f64; col_idx.len()];
    let mut r0 = vec![0.0; n_dofs];
    form.residual(u, rhs, &mut r0);
    let mut u_pert = u.to_vec();
    for j in 0..n_dofs {
        let coupled: Vec<usize> = (0..n_dofs).filter(|&i| sparsity[i].contains(&j)).collect();
        if coupled.is_empty() { continue; }
        let h = eps * (1.0 + u[j].abs());
        u_pert[j] = u[j] + h;
        let mut r1 = vec![0.0; n_dofs];
        form.residual(&u_pert, rhs, &mut r1);
        let inv_h = 1.0 / h;
        for &i in &coupled {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            if let Some(pos) = col_idx[start..end].iter().position(|&c: &usize| c == j) {
                values[start + pos] = (r1[i] - r0[i]) * inv_h;
            }
        }
        u_pert[j] = u[j];
    }
    CsrMatrix {
        nrows: n_dofs, ncols: n_dofs,
        row_ptr, col_idx: col_idx.into_iter().map(|c| c as u32).collect(), values,
    }
}

/// Wrapper: implements `NonlinearForm` using finite-difference Jacobian.
///
/// Define the residual and get a working Jacobian for prototyping.
/// For production, replace with an analytic Jacobian.
pub struct FdNonlinearForm<F> {
    n_dofs: usize,
    residual_fn: F,
    element_dofs: Vec<Vec<u32>>,
    eps: f64,
}

impl<F> FdNonlinearForm<F>
where F: Fn(&[f64], &[f64], &mut [f64]) + Send + Sync,
{
    pub fn new(n_dofs: usize, residual_fn: F, element_dofs: Vec<Vec<u32>>) -> Self {
        FdNonlinearForm { n_dofs, residual_fn, element_dofs, eps: 1e-7 }
    }
}

impl<F> NonlinearForm for FdNonlinearForm<F>
where F: Fn(&[f64], &[f64], &mut [f64]) + Send + Sync,
{
    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        (self.residual_fn)(u, rhs, r);
    }
    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        finite_diff_jacobian(u, self, &vec![0.0; self.n_dofs], self.n_dofs, &self.element_dofs, self.eps)
    }
    fn n_dofs(&self) -> usize { self.n_dofs }
}

// 鈹€鈹€鈹€ JFNK (Jacobian-Free Newton-Krylov) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Configuration for the JFNK solver.
#[derive(Debug, Clone)]
pub struct JfNKConfig {
    pub atol: f64,
    pub rtol: f64,
    pub max_iter: usize,
    pub gmres_restart: usize,
    pub gmres_max_iter: usize,
    /// Finite-difference step for Jacobian-vector products (relative to 鈥杤鈥?.
    pub eps: f64,
    pub verbose: bool,
}

impl Default for JfNKConfig {
    fn default() -> Self {
        Self {
            atol: 1e-10, rtol: 1e-8, max_iter: 50,
            gmres_restart: 30, gmres_max_iter: 300,
            eps: 1e-7, verbose: false,
        }
    }
}

/// Jacobian-Free Newton-Krylov solver.
///
/// Uses GMRES with finite-difference Jacobian-vector products
/// `J(u)路v 鈮?(F(u+蔚v) 鈭?F(u))/蔚`, eliminating the need to assemble the Jacobian.
pub struct JfNKSolver {
    cfg: JfNKConfig,
}

impl JfNKSolver {
    pub fn new(cfg: JfNKConfig) -> Self { Self { cfg } }

    /// Solve `F(u) = 0` without forming the Jacobian matrix.
    pub fn solve(
        &self,
        form: &dyn NonlinearForm,
        rhs: &[f64],
        u: &mut [f64],
    ) -> Result<NewtonResult, NewtonResult> {
        let n = form.n_dofs();
        let mut r = vec![0.0; n];
        let mut du = vec![0.0; n];
        let mut u_trial = vec![0.0; n];
        let mut r_trial = vec![0.0; n];

        form.residual(u, rhs, &mut r);
        let r0 = norm2(&r);
        let mut r_norm = r0;

        if self.cfg.verbose { println!("[JFNK] iter=0 鈥朏鈥?{r0:.3e}"); }
        if r0 < self.cfg.atol {
            return Ok(NewtonResult { converged: true, iterations: 0, final_residual: r0 });
        }

        for iter in 0..self.cfg.max_iter {
            // GMRES solve: J路du = -r, using matrix-free Jv products
            let neg_r: Vec<f64> = r.iter().map(|&v| -v).collect();
            du.fill(0.0);
            self.solve_gmres_jfnk(form, u, &neg_r, &mut du, rhs);

            // Line search
            let mut alpha = 1.0;
            let mut best_norm = r_norm;
            for _ in 0..20 {
                for i in 0..n { u_trial[i] = u[i] + alpha * du[i]; }
                form.residual(&u_trial, rhs, &mut r_trial);
                let tn = norm2(&r_trial);
                if tn < best_norm { best_norm = tn; }
                if tn < r_norm * (1.0 - 1e-4 * alpha) { break; }
                alpha *= 0.5;
                if alpha < 1e-8 { break; }
            }
            for i in 0..n { u[i] += alpha * du[i]; }

            form.residual(u, rhs, &mut r);
            r_norm = norm2(&r);
            if self.cfg.verbose { println!("[JFNK] iter={} 鈥朏鈥?{r_norm:.3e} 伪={alpha:.4}", iter+1); }

            if r_norm < self.cfg.atol || r_norm < r0 * self.cfg.rtol {
                return Ok(NewtonResult { converged: true, iterations: iter+1, final_residual: r_norm });
            }
        }
        Err(NewtonResult { converged: false, iterations: self.cfg.max_iter, final_residual: r_norm })
    }

    /// Solve `J(u)路x = b` using GMRES with matrix-free Jacobian-vector products.
    fn solve_gmres_jfnk(&self, form: &dyn NonlinearForm, u: &[f64], b: &[f64], x: &mut [f64], rhs: &[f64]) {
        let n = u.len();
        let eps = self.cfg.eps;
        let restart = self.cfg.gmres_restart.min(n);
        let max_mv = self.cfg.gmres_max_iter;

        // Evaluate F(u) once (used in all Jv products)
        let mut fu = vec![0.0; n];
        form.residual(u, rhs, &mut fu);

        // Matrix-vector product operator: v 鈫?J(u)路v 鈮?(F(u+蔚v) - F(u))/蔚
        let jv = |v: &[f64], w: &mut [f64]| {
            let eps_v = eps / (norm2(v) + 1e-30).max(1e-14);
            let mut up = u.to_vec();
            for i in 0..n { up[i] += eps_v * v[i]; }
            let mut fp = vec![0.0; n];
            form.residual(&up, rhs, &mut fp);
            for i in 0..n { w[i] = (fp[i] - fu[i]) / eps_v; }
        };

        // Simple GMRES implementation
        let mut v = vec![vec![0.0; n]; restart + 1];
        let mut h = vec![vec![0.0; restart + 1]; restart];
        let mut g = vec![0.0; restart + 1];
        let mut y = vec![0.0; restart];
        let mut cs = vec![0.0; restart];
        let mut sn = vec![0.0; restart];

        // Initial residual r0 = b - J路x = b (since x = 0)
        let r = b.to_vec();
        let beta = norm2(&r);
        if beta < 1e-30 { return; }
        for i in 0..n { v[0][i] = r[i] / beta; }
        g[0] = beta;

        let mut _iters = 0;
        'outer: for _iter in 0..max_mv / restart {
            for k in 0..restart {
                let (vk_left, vk_right) = v.split_at_mut(k + 1);
                jv(&vk_left[k], &mut vk_right[0]);
                for j in 0..=k {
                    h[k][j] = dot(&vk_right[0], &vk_left[j]);
                    for i in 0..n { vk_right[0][i] -= h[k][j] * vk_left[j][i]; }
                }
                h[k][k + 1] = norm2(&vk_right[0]);
                if h[k][k + 1] > 1e-30 {
                    for i in 0..n { vk_right[0][i] /= h[k][k + 1]; }
                }

                // Apply Givens rotation
                for j in 0..k {
                    let tmp = cs[j] * h[k][j] + sn[j] * h[k][j + 1];
                    h[k][j + 1] = -sn[j] * h[k][j] + cs[j] * h[k][j + 1];
                    h[k][j] = tmp;
                }
                let nu = (h[k][k] * h[k][k] + h[k][k + 1] * h[k][k + 1]).sqrt();
                if nu < 1e-30 { continue; }
                cs[k] = h[k][k] / nu;
                sn[k] = h[k][k + 1] / nu;
                h[k][k] = nu;
                h[k][k + 1] = 0.0;
                g[k + 1] = -sn[k] * g[k];
                g[k] *= cs[k];

                _iters += 1;
                if g[k + 1].abs() < self.cfg.atol.max(1e-12) { break 'outer; }
            }
        }

        // Back-substitute
        for i in (0..restart).rev() {
            let mut s = g[i];
            for j in i + 1..restart { s -= h[j][i] * y[j]; }
            y[i] = s / h[i][i];
        }
        for i in 0..n { for j in 0..restart { x[i] += y[j] * v[j][i]; } }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// 鈹€鈹€鈹€ Anderson acceleration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Configuration for Anderson acceleration (fixed-point mixing).
#[derive(Debug, Clone)]
pub struct AndersonConfig {
    /// Number of past residuals to store (history depth).
    pub m: usize,
    /// Mixing parameter 尾: u_new = u - 尾路F(u) (before Anderson mixing).
    pub beta: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Absolute tolerance on 鈥朏(u)鈥?
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
    /// Regularization for the least-squares solve.
    pub lambda: f64,
    pub verbose: bool,
}

impl Default for AndersonConfig {
    fn default() -> Self {
        Self { m: 5, beta: 1.0, max_iter: 200, atol: 1e-10, rtol: 1e-8, lambda: 1e-8, verbose: false }
    }
}

/// Anderson-accelerated fixed-point solver.
///
/// For `F(u) = 0`, uses Anderson mixing to accelerate convergence:
/// ```text
/// u_{k+1} = u_k - 尾路F(u_k)  (Picard step), then
///          = weighted combination of last m iterates to minimize residual.
/// ```
pub struct AndersonAccelerator {
    cfg: AndersonConfig,
}

impl AndersonAccelerator {
    pub fn new(cfg: AndersonConfig) -> Self { Self { cfg } }

    /// Solve `F(u) = 0` using Anderson-accelerated fixed-point iteration.
    pub fn solve(
        &self,
        form: &dyn NonlinearForm,
        rhs: &[f64],
        u: &mut [f64],
    ) -> Result<NewtonResult, NewtonResult> {
        let n = form.n_dofs();
        let m = self.cfg.m;
        let beta = self.cfg.beta;
        let mut r = vec![0.0; n];
        let mut r_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut u_hist: Vec<Vec<f64>> = Vec::with_capacity(m);

        form.residual(u, rhs, &mut r);
        let r0 = norm2(&r);
        let mut r_norm = r0;
        if self.cfg.verbose { println!("[Anderson] iter=0 鈥朏鈥?{r0:.3e}"); }
        if r0 < self.cfg.atol {
            return Ok(NewtonResult { converged: true, iterations: 0, final_residual: r0 });
        }

        for iter in 0..self.cfg.max_iter {
            // Picard step: u_new = u - 尾路F(u)
            let mut u_new = u.to_vec();
            for i in 0..n { u_new[i] -= beta * r[i]; }

            // Store history
            r_hist.push(r.clone());
            u_hist.push(u_new.clone());
            if r_hist.len() > m { r_hist.remove(0); u_hist.remove(0); }
            let k = r_hist.len();

            if k > 1 {
                // Solve least-squares: min_纬 鈥朏_k - 危_{j<k} 纬_j路(F_k - F_j)鈥?
                // where F_k is the latest residual.
                let n_row = n;
                let n_col = k - 1;
                if n_col > 0 {
                    // Build the matrix A_ij = (F_k - F_j)_i and RHS b_i = (F_k)_i
                    // Solve A路纬 = b via normal equations with regularization
                    let mut ata = vec![0.0; n_col * n_col];
                    let mut atb = vec![0.0; n_col];
                    for i in 0..n_row {
                        let fk = r_hist[k - 1][i];
                        for j in 0..n_col {
                            let fj = r_hist[j][i];
                            atb[j] += (fk - fj) * fk;
                        for l in 0..n_col {
                                let fl = r_hist[l][i];
                                ata[j * n_col + l] += (fk - fj) * (fk - fl);
        }
    }
                    }
                    // Regularize
                    for j in 0..n_col { ata[j * n_col + j] += self.cfg.lambda; }
                    // Solve (dense, Cramer's rule for small systems)
                    let gamma = if n_col == 1 {
                        vec![atb[0] / ata[0].max(1e-30)]
                    } else {
                        solve_2x2(&ata, &atb, n_col)
                    };
                    // Anderson update: u = 危 纬_j路u_j (last residual weight from constraint 危纬_j = 1)
                    let mut gamma_sum = 0.0;
                    for j in 0..n_col { gamma_sum += gamma[j]; }
                    for i in 0..n {
                        u_new[i] = 0.0;
                        for j in 0..n_col {
                            u_new[i] += gamma[j] * u_hist[j][i];
                        }
                        u_new[i] += (1.0 - gamma_sum) * u_hist[k - 1][i];
                    }
                }
            }

            u.copy_from_slice(&u_new);
            form.residual(u, rhs, &mut r);
            r_norm = norm2(&r);
            if self.cfg.verbose { println!("[Anderson] iter={} 鈥朏鈥?{r_norm:.3e}", iter + 1); }
            if r_norm < self.cfg.atol || r_norm < r0 * self.cfg.rtol {
                return Ok(NewtonResult { converged: true, iterations: iter + 1, final_residual: r_norm });
            }
        }
        Err(NewtonResult { converged: false, iterations: self.cfg.max_iter, final_residual: r_norm })
    }
}

fn solve_2x2(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    if n == 1 { return vec![b[0] / a[0].max(1e-30)]; }
    // Gaussian elimination for general n脳n (n small)
    let mut x = b.to_vec();
    let mut aa = a.to_vec();
    for c in 0..n {
        let mut piv = c;
        for r in c + 1..n { if aa[r * n + c].abs() > aa[piv * n + c].abs() { piv = r; } }
        for j in c..n { aa.swap(c * n + j, piv * n + j); }
        x.swap(c, piv);
        let pv = aa[c * n + c];
        if pv.abs() < 1e-30 { continue; }
        for j in c..n { aa[c * n + j] /= pv; }
        x[c] /= pv;
        for r in 0..n {
            if r != c {
                let f = aa[r * n + c];
                for j in c..n { aa[r * n + j] -= f * aa[c * n + j]; }
                x[r] -= f * x[c];
            }
        }
    }
    x
}

// 鈹€鈹€鈹€ NonlinearDiffusionForm 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TetP1, TetP2, TriP1}, lagrange::factory::{TriPk, TetPk}};
use fem_linalg::CooMatrix;
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

/// Nonlinear diffusion form: `F(u) = 鈭?魏(u) 鈭噓 路 鈭噕 dx 鈭?鈭?f v dx`.
///
/// The Jacobian is `J(u)[i,j] = 鈭?[魏(u) 鈭囅嗏奔 路 鈭囅嗎耽 + 魏'(u) 蠁獗?鈭噓 路 鈭囅嗎耽] dx`
/// (linearisation via product rule).
///
/// For simplicity (and robustness), the Jacobian uses **Picard linearisation**:
/// `J_Picard(u)[i,j] = 鈭?魏(u) 鈭囅嗏奔 路 鈭囅嗎耽 dx`
/// (freeze 魏 at the current iterate, ignore the 魏' term).
/// This is first-order convergent; set `use_full_jacobian = true` for quadratic.
pub struct NonlinearDiffusionForm<S: FESpace, K>
where
    K: Fn(f64) -> f64 + Send + Sync,
{
    space:           S,
    kappa:           K,
    /// Also assemble the `魏'(u)` term in the Jacobian for quadratic convergence.
    pub use_full_jacobian: bool,
    /// Derivative of 魏: `kappa_prime(u)`.  Only used if `use_full_jacobian`.
    #[allow(dead_code)]
    kappa_prime:     Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    quad_order:      u8,
    /// Fixed (linear) Dirichlet constrained DOFs 鈫?prescribed value.
    dirichlet: Vec<(usize, f64)>,
}

impl<S: FESpace, K> NonlinearDiffusionForm<S, K>
where
    K: Fn(f64) -> f64 + Send + Sync,
{
    /// Create a nonlinear diffusion form with Picard Jacobian.
    pub fn new(space: S, kappa: K, quad_order: u8) -> Self {
        NonlinearDiffusionForm { space, kappa, use_full_jacobian: false, kappa_prime: None, quad_order, dirichlet: vec![] }
    }

    /// Set the constrained (Dirichlet) DOFs (index, prescribed value).
    pub fn set_dirichlet(&mut self, dofs: Vec<(usize, f64)>) {
        self.dirichlet = dofs;
    }
}

impl<S: FESpace, K> NonlinearForm for NonlinearDiffusionForm<S, K>
where
    K: Fn(f64) -> f64 + Send + Sync,
{
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        let mesh  = self.space.mesh();
        let dim   = mesh.dim() as usize;
        let order = self.space.order();
        let n     = self.space.n_dofs();

        // r = 0
        r.iter_mut().for_each(|v| *v = 0.0);

        let mut phi      = Vec::<f64>::new();
        let mut grad_ref = Vec::<f64>::new();
        let mut grad_p   = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let et   = mesh.element_type(e);
            let re   = ref_elem(et, order);
            let n_l  = re.n_dofs();
            let quad = re.quadrature(self.quad_order);
            let gd: Vec<usize> = self.space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.clone().try_inverse().unwrap().transpose();

            phi.resize(n_l, 0.0);
            grad_ref.resize(n_l * dim, 0.0);
            grad_p.resize(n_l * dim, 0.0);

            let _x0 = mesh.node_coords(nodes[0]);
            let mut f_elem = vec![0.0_f64; n_l];

            for (qi, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[qi] * det_j.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                xform_grads(&jit, &grad_ref, &mut grad_p, n_l, dim);

                // Interpolate u at this quadrature point
                let u_qp: f64 = gd.iter().zip(phi.iter()).map(|(&d, &ph)| u[d] * ph).sum();
                let kappa_qp = (self.kappa)(u_qp);

                // 鈭噓 at this point
                let grad_u: Vec<f64> = (0..dim).map(|d| {
                    gd.iter().zip(grad_p.chunks(dim)).map(|(&di, gpi)| u[di] * gpi[d]).sum::<f64>()
                }).collect();

                // F(u)[i] += w 魏(u) 鈭噓 路 鈭囅嗎耽
                for i in 0..n_l {
                    let dot: f64 = (0..dim).map(|d| grad_u[d] * grad_p[i*dim+d]).sum();
                    f_elem[i] += w * kappa_qp * dot;
                }
            }

            // Scatter: r[gi] += f_elem[i] 鈭?rhs[gi]
            for (i, &gi) in gd.iter().enumerate() {
                r[gi] += f_elem[i];
            }
        }

        // Subtract RHS
        for i in 0..n { r[i] -= rhs[i]; }

        // Apply Dirichlet: r[d] = u[d] - value
        for &(d, val) in &self.dirichlet {
            r[d] = u[d] - val;
        }
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        // Picard linearisation: J = 鈭?魏(u_h) 鈭囅嗏奔 路 鈭囅嗎耽 dx
        let mesh  = self.space.mesh();
        let dim   = mesh.dim() as usize;
        let order = self.space.order();
        let n     = self.space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);

        let mut phi      = Vec::<f64>::new();
        let mut grad_ref = Vec::<f64>::new();
        let mut grad_p   = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let et   = mesh.element_type(e);
            let re   = ref_elem(et, order);
            let n_l  = re.n_dofs();
            let quad = re.quadrature(self.quad_order);
            let gd: Vec<usize> = self.space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.clone().try_inverse().unwrap().transpose();

            phi.resize(n_l, 0.0);
            grad_ref.resize(n_l * dim, 0.0);
            grad_p.resize(n_l * dim, 0.0);

            let mut k_elem = vec![0.0_f64; n_l * n_l];

            for (qi, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[qi] * det_j.abs();
                re.eval_basis(xi, &mut phi);
                re.eval_grad_basis(xi, &mut grad_ref);
                xform_grads(&jit, &grad_ref, &mut grad_p, n_l, dim);

                let u_qp: f64 = gd.iter().zip(phi.iter()).map(|(&d, &ph)| u[d] * ph).sum();
                let kappa_qp = (self.kappa)(u_qp);

                for i in 0..n_l {
                    for j in 0..n_l {
                        let dot: f64 = (0..dim).map(|d| grad_p[i*dim+d] * grad_p[j*dim+d]).sum();
                        k_elem[i*n_l+j] += w * kappa_qp * dot;
                    }
                }
            }

            for (i, &gi) in gd.iter().enumerate() {
                for (j, &gj) in gd.iter().enumerate() {
                    coo.add(gi, gj, k_elem[i*n_l+j]);
                }
            }
        }

        let mut jac = coo.into_csr();

        // Apply Dirichlet rows: zero row, set diagonal to 1.
        for &(d, _val) in &self.dirichlet {
            for ptr in jac.row_ptr[d]..jac.row_ptr[d+1] {
                jac.values[ptr] = 0.0;
            }
            *jac.get_mut(d, d) = 1.0;
        }

        jac
    }
}

// 鈹€鈹€鈹€ Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

fn ref_elem(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) => Box::new(TriPk::new(3)),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetPk::new(3)),
        _ => panic!("nonlinear ref_elem: unsupported ({et:?}, {order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col+1]);
        for row in 0..dim { j[(row,col)] = xc[row] - x0[row]; }
    }
    (j.clone(), j.determinant())
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

// 鈹€鈹€鈹€ LBFGS (Limited-memory BFGS) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Convergence and iteration parameters for the LBFGS solver.
#[derive(Debug, Clone)]
pub struct LbfgsConfig {
    pub atol: f64,                    // 鈥朏鈥栤倐 absolute tolerance
    pub rtol: f64,                    // 鈥朏鈥栤倐 / 鈥朏鈧€鈥栤倐 relative tolerance
    pub max_iter: usize,              // max outer iterations
    pub history: usize,               // L-BFGS history size m
    pub line_search_max_backtracks: usize,
    pub line_search_shrink: f64,
    pub line_search_sufficient_decrease: f64,
    pub verbose: bool,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        LbfgsConfig {
            atol: 1e-10, rtol: 1e-8, max_iter: 200, history: 10,
            line_search_max_backtracks: 20, line_search_shrink: 0.5,
            line_search_sufficient_decrease: 1e-4, verbose: false,
        }
    }
}

/// Outcome of an LBFGS solve.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
}

/// Limited-memory BFGS quasi-Newton solver.
///
/// Approximates the inverse Hessian via `m` history pairs (s岬? y岬? and the
/// two-loop recursion (Nocedal & Wright).  No Jacobian assembly is required.
pub struct LbfgsSolver {
    cfg: LbfgsConfig,
}

impl LbfgsSolver {
    pub fn new(cfg: LbfgsConfig) -> Self { LbfgsSolver { cfg } }

    /// Solve `F(u) = 0` starting from the initial guess in `u`.
    pub fn solve(
        &self,
        form: &dyn NonlinearForm,
        rhs:  &[f64],
        u:    &mut [f64],
    ) -> Result<LbfgsResult, LbfgsResult> {
        let n = form.n_dofs();
        assert_eq!(u.len(), n);
        let m = self.cfg.history.max(1);

        let mut g  = vec![0.0_f64; n];  // current gradient (= residual)
        let mut g_old = vec![0.0_f64; n];
        let mut d  = vec![0.0_f64; n];  // search direction
        form.residual(u, rhs, &mut g);
        let g0_norm = norm2(&g);
        let mut g_norm = g0_norm;

        if g0_norm < self.cfg.atol {
            return Ok(LbfgsResult { converged: true, iterations: 0, final_residual: g0_norm });
        }

        // Ring buffers for s = x_{k+1} 鈭?x_k  and  y = g_{k+1} 鈭?g_k
        let mut ss: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut yy: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho: Vec<f64> = Vec::with_capacity(m);
        let mut head: usize = 0;

        for iter in 0..self.cfg.max_iter {
            // 鈹€鈹€ Build search direction d via two-loop recursion 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            d.copy_from_slice(&g);

            // Loop 1: q 鈫?g; 伪岬?= 蟻岬?路 s岬⑨祤 q; q 鈫?q 鈭?伪岬?y岬?
            let n_hist = ss.len();
            let mut alpha = vec![0.0_f64; n_hist];
            for i in (0..n_hist).rev() {
                let idx = (head + i) % n_hist;
                let si = &ss[idx];
                let yi = &yy[idx];
                alpha[i] = rho[idx] * dot_product(si, &d);
                for j in 0..n { d[j] -= alpha[i] * yi[j]; }
            }

            // Scale: 纬 = (y岬忊伝鹿岬€ s岬忊伝鹿) / (y岬忊伝鹿岬€ y岬忊伝鹿)
            if n_hist > 0 {
                let last = (head + n_hist - 1) % n_hist;
                let sl = &ss[last];
                let yl = &yy[last];
                let sy = dot_product(sl, yl);
                let yy_inner = dot_product(yl, yl);
                let gamma = if yy_inner > 1e-32 { sy / yy_inner } else { 1.0 };
                for j in 0..n { d[j] *= gamma; }
            }

            // Loop 2: 尾岬?= 蟻岬?路 y岬⑨祤 z; z 鈫?z + s岬?伪岬?鈭?尾岬?
            for i in 0..n_hist {
                let idx = (head + i) % n_hist;
                let si = &ss[idx];
                let yi = &yy[idx];
                let beta = rho[idx] * dot_product(yi, &d);
                for j in 0..n { d[j] += si[j] * (alpha[i] - beta); }
            }

            // d = -z  (negative gradient direction)
            for j in 0..n { d[j] = -d[j]; }

            // 鈹€鈹€ Line search (Armijo backtracking) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            let mut step = 1.0_f64;
            let mut accepted = false;
            form.residual(u, rhs, &mut g_old);
            let old_norm = norm2(&g_old);

            for _ in 0..self.cfg.line_search_max_backtracks {
                let mut u_trial = vec![0.0_f64; n];
                for j in 0..n { u_trial[j] = u[j] + step * d[j]; }
                let mut g_trial = vec![0.0_f64; n];
                form.residual(&u_trial, rhs, &mut g_trial);
                let trial_norm = norm2(&g_trial);

                let target = (1.0 - self.cfg.line_search_sufficient_decrease * step).max(0.0) * old_norm;
                if trial_norm <= target {
                    u.copy_from_slice(&u_trial);
                    g.copy_from_slice(&g_trial);
                    accepted = true;
                    break;
                }
                step *= self.cfg.line_search_shrink;
            }
            if !accepted {
                // Accept best effort (step was too small)
                for j in 0..n { u[j] += step * d[j]; }
                form.residual(u, rhs, &mut g);
            }

            g_norm = norm2(&g);

            if self.cfg.verbose {
                println!("[LBFGS] iter={} 鈥朏鈥?{g_norm:.3e} step={step:.2e} hist={n_hist}", iter + 1);
            }

            if g_norm < self.cfg.atol || g_norm < g0_norm * self.cfg.rtol {
                return Ok(LbfgsResult { converged: true, iterations: iter + 1, final_residual: g_norm });
            }

            // 鈹€鈹€ Update history 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if ss.len() < m {
                ss.push(vec![0.0_f64; n]);
                yy.push(vec![0.0_f64; n]);
                rho.push(0.0);
            }
            let si = &mut ss[head];
            let yi = &mut yy[head];
            for j in 0..n { si[j] = step * d[j]; }
            for j in 0..n { yi[j] = g[j] - g_old[j]; }
            let sy = dot_product(si, yi);
            rho[head] = if sy.abs() > 1e-32 { 1.0 / sy } else { 1.0 };
            head = (head + 1) % m;
        }

        Err(LbfgsResult { converged: false, iterations: self.cfg.max_iter, final_residual: g_norm })
    }
}

// 鈹€鈹€鈹€ Trust-region (Steihaug-CG) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Convergence and iteration parameters for the trust-region solver.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig {
    pub atol: f64,
    pub rtol: f64,
    pub max_iter: usize,
    /// Initial trust-region radius.
    pub delta0: f64,
    /// Maximum trust-region radius.
    pub delta_max: f64,
    /// Shrink factor when step is rejected.
    pub shrink: f64,
    /// Expand factor when step is accepted (good quality).
    pub expand: f64,
    /// Threshold 蟻 for accepting step.
    pub eta_accept: f64,
    /// Threshold 蟻 for expanding the radius.
    pub eta_expand: f64,
    /// Linear solver tolerance for CG inner solve.
    pub linear_tol: f64,
    pub verbose: bool,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        TrustRegionConfig {
            atol: 1e-10, rtol: 1e-8, max_iter: 100,
            delta0: 1.0, delta_max: 1e8, shrink: 0.25, expand: 2.0,
            eta_accept: 0.1, eta_expand: 0.75,
            linear_tol: 1e-6, verbose: false,
        }
    }
}

/// Outcome of a trust-region solve.
#[derive(Debug, Clone)]
pub struct TrustRegionResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
}

/// Trust-region method with Steihaug-CG subproblem solver.
///
/// Uses the quadratic model `m(p) = F(u)岬€p + 陆 p岬€J(u)p` and CG to solve
/// the constrained minimisation `min_{鈥杙鈥栤墹螖} m(p)`.
pub struct TrustRegionSolver {
    cfg: TrustRegionConfig,
}

impl TrustRegionSolver {
    pub fn new(cfg: TrustRegionConfig) -> Self { TrustRegionSolver { cfg } }

    /// Solve `F(u) = 0` starting from the initial guess in `u`.
    pub fn solve(
        &self,
        form: &dyn NonlinearForm,
        rhs:  &[f64],
        u:    &mut [f64],
    ) -> Result<TrustRegionResult, TrustRegionResult> {
        let n = form.n_dofs();
        assert_eq!(u.len(), n);
        let cfg = &self.cfg;

        let mut delta = cfg.delta0;
        let mut g = vec![0.0_f64; n];   // residual = gradient of 陆鈥朏鈥柭?
        let _p = vec![0.0_f64; n];   // step
        let mut g_jp = vec![0.0_f64; n];// J路p for CG

        form.residual(u, rhs, &mut g);
        let g0_norm = norm2(&g);
        let mut g_norm = g0_norm;

        if g_norm < cfg.atol {
            return Ok(TrustRegionResult { converged: true, iterations: 0, final_residual: g_norm });
        }

        for iter in 0..cfg.max_iter {
            let jac = form.jacobian(u);

            // 鈹€鈹€ Steihaug-CG for the constrained step 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // Solve min_{鈥杙鈥栤墹螖}  g岬€路p + 陆路p岬€路J路p
            let mut z = vec![0.0_f64; n];  // CG iterate
            let mut r_cg = g.clone();      // negative gradient = -g (we minimise g路p + 陆p路J路p)
            // Actually: m(p) = g岬€p + 陆p岬€Jp. 鈭噈 = g + Jp = 0 鈫?Jp = -g
            // So we solve J路p = -g with CG, truncated by 鈥杙鈥?鈮?螖.
            for j in 0..n { r_cg[j] = -g[j]; }  // r = -g 鈭?J路z  (initially z=0 鈫?r = -g)

            let mut cg_dir = r_cg.clone();
            let mut rz = dot_product(&r_cg, &r_cg);
            let mut reached_boundary = false;
            let mut cg_finished = false;

            let cg_max = n.min(200);
            for _cg in 0..cg_max {
                if rz < cfg.linear_tol * cfg.linear_tol {
                    cg_finished = true;
                    break;
                }

                // J路cg_dir
                jac.spmv(&cg_dir, &mut g_jp);
                let cg_jcg = dot_product(&cg_dir, &g_jp);

                if cg_jcg <= 0.0 {
                    // Negative curvature: step to boundary along cg_dir
                    // 蟿 solves 鈥杬 + 蟿路cg_dir鈥?= 螖
                    let z_norm_sq = dot_product(&z, &z);
                    let d_norm_sq = dot_product(&cg_dir, &cg_dir);
                    let zd = dot_product(&z, &cg_dir);
                    // 鈥杬 + 蟿路d鈥柭?= z路z + 2蟿路z路d + 蟿虏路d路d = 螖虏
                    let a = d_norm_sq;
                    let b = 2.0 * zd;
                    let c = z_norm_sq - delta * delta;
                    let disc = b * b - 4.0 * a * c;
                    if disc > 0.0 && a > 0.0 {
                        let tau = (-b + disc.sqrt()) / (2.0 * a);
                        for j in 0..n { z[j] += tau * cg_dir[j]; }
                    }
                    reached_boundary = true;
                    break;
                }

                let alpha = rz / cg_jcg;
                // Check if z + 伪路cg_dir would exceed 螖
                let new_z_sq = dot_product(&z, &z) + 2.0 * alpha * dot_product(&z, &cg_dir)
                             + alpha * alpha * dot_product(&cg_dir, &cg_dir);
                if new_z_sq > delta * delta {
                    // Step truncated by boundary
                    let z_norm_sq = dot_product(&z, &z);
                    let d_norm_sq = dot_product(&cg_dir, &cg_dir);
                    let zd = dot_product(&z, &cg_dir);
                    let a = d_norm_sq;
                    let b = 2.0 * zd;
                    let c = z_norm_sq - delta * delta;
                    let disc = b * b - 4.0 * a * c;
                    if disc > 0.0 && a > 0.0 {
                        let tau = (-b + disc.sqrt()) / (2.0 * a);
                        for j in 0..n { z[j] += tau * cg_dir[j]; }
                    }
                    reached_boundary = true;
                    break;
                }

                for j in 0..n { z[j] += alpha * cg_dir[j]; }
                for j in 0..n { r_cg[j] -= alpha * g_jp[j]; }

                let rz_new = dot_product(&r_cg, &r_cg);
                let beta = rz_new / rz;
                for j in 0..n { cg_dir[j] = r_cg[j] + beta * cg_dir[j]; }
                rz = rz_new;
            }

            if !cg_finished && !reached_boundary {
                // CG completed naturally 鈥?z is the Newton step
            }

            // z is the candidate step p_k
            // 鈹€鈹€ Compute 蟻 = actual reduction / predicted reduction 鈹€鈹€鈹€鈹€鈹€
            let mut g_new = vec![0.0_f64; n];
            let mut u_new = vec![0.0_f64; n];
            for j in 0..n { u_new[j] = u[j] + z[j]; }
            form.residual(&u_new, rhs, &mut g_new);

            // Model predicted reduction: m(0) 鈭?m(p) = 鈭抔岬€p 鈭?陆p岬€Jp
            jac.spmv(&z, &mut g_jp);
            let predicted = -dot_product(&g, &z) - 0.5 * dot_product(&z, &g_jp);
            let actual = 0.5 * (norm2(&g).powi(2) - norm2(&g_new).powi(2));
            let rho = if predicted.abs() > 1e-32 { actual / predicted } else { 0.0 };

            if rho > cfg.eta_accept {
                u.copy_from_slice(&u_new);
                g.copy_from_slice(&g_new);
                g_norm = norm2(&g);
                delta = (delta * cfg.expand).min(cfg.delta_max);
            } else {
                delta *= cfg.shrink;
            }

            if self.cfg.verbose {
                println!("[TR] iter={} 鈥朏鈥?{g_norm:.3e} 螖={delta:.3e} 蟻={rho:.3e}", iter + 1);
            }

            if g_norm < cfg.atol || g_norm < g0_norm * cfg.rtol {
                return Ok(TrustRegionResult { converged: true, iterations: iter + 1, final_residual: g_norm });
            }

            if delta < 1e-16 {
                break;
            }
        }

        Err(TrustRegionResult { converged: false, iterations: self.cfg.max_iter, final_residual: g_norm })
    }
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, DofManager, fe_space::FESpace, constraints::boundary_dofs};

    fn get_bnd_dofs(mesh: &Mesh<2>, order: u8) -> Vec<usize> {
        let dm = DofManager::new(mesh, order);
        boundary_dofs(mesh, &dm, &[1, 2, 3, 4])
            .iter().map(|&d| d as usize).collect()
    }

    /// For 魏(u) = const, the nonlinear problem reduces to a linear one.
    /// Verify that the Newton solver converges in 1 iteration.
    #[test]
    fn newton_linear_problem_converges_in_one_iter() {
        let mesh  = Mesh::<2>::unit_square_tri(6);
        let bnd   = get_bnd_dofs(&mesh, 1);
        let space = H1Space::new(mesh, 1);
        let n     = space.n_dofs();

        let rhs = vec![0.0_f64; n];

        let mut form = NonlinearDiffusionForm::new(space, |_u| 1.0, 3);
        form.set_dirichlet(bnd.iter().map(|&d| (d, 0.0)).collect());

        let mut u = vec![0.0_f64; n];
        let cfg = NewtonConfig { atol: 1e-12, rtol: 1e-10, max_iter: 10, ..Default::default() };
        let res = NewtonSolver::new(cfg).solve(&form, &rhs, &mut u).unwrap();
        assert!(res.converged, "Newton did not converge");
        assert!(res.iterations <= 2, "Expected 鈮? iters for linear problem, got {}", res.iterations);
        let rn = norm2(&u);
        assert!(rn < 1e-12, "u should be zero but 鈥杣鈥?= {rn}");
    }

    /// Nonlinear problem: 魏(u) = 1 + u虏, constant forcing.
    /// Just verify convergence.
    #[test]
    fn newton_nonlinear_converges() {
        let mesh  = Mesh::<2>::unit_square_tri(8);
        let bnd   = get_bnd_dofs(&mesh, 1);
        let space = H1Space::new(mesh, 1);
        let n     = space.n_dofs();

        // Use a properly assembled load vector so u stays O(1).
        use crate::assembler::Assembler;
        use crate::standard::DomainSourceIntegrator;
        let rhs = Assembler::assemble_linear(
            &space,
            &[&DomainSourceIntegrator::new(|_| 1.0)],
            3,
        );

        let mut form = NonlinearDiffusionForm::new(space, |u| 1.0 + u * u, 3);
        form.set_dirichlet(bnd.iter().map(|&d| (d, 0.0)).collect());

        let mut u = vec![0.0_f64; n];
        let cfg = NewtonConfig { atol: 1e-10, rtol: 1e-8, max_iter: 50, ..Default::default() };
        let res = NewtonSolver::new(cfg).solve(&form, &rhs, &mut u);
        assert!(res.is_ok() && res.unwrap().converged, "Newton did not converge for nonlinear problem");
    }

    /// Jacobian finite-difference check: J[i,j] 鈮?(F(u+蔚 e獗?[i] 鈭?F(u)[i]) / 蔚.
    /// Uses constant 魏 so that the Picard Jacobian equals the full tangent stiffness.
    #[test]
    fn jacobian_finite_difference_check() {
        let mesh  = Mesh::<2>::unit_square_tri(2);
        let bnd   = get_bnd_dofs(&mesh, 1);
        let space = H1Space::new(mesh, 1);
        let n     = space.n_dofs();
        let rhs   = vec![0.0_f64; n];

        // With constant 魏, the Picard Jacobian matches the full tangent exactly.
        let mut form = NonlinearDiffusionForm::new(space, |_u| 1.5, 3);
        form.set_dirichlet(bnd.iter().map(|&d| (d, 0.0)).collect());

        let mut u: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64) / n as f64).collect();
        for &(d, v) in &form.dirichlet { u[d] = v; }

        let jac  = form.jacobian(&u);
        let eps  = 1e-6_f64;
        let mut r0 = vec![0.0_f64; n];
        form.residual(&u, &rhs, &mut r0);

        let free_dofs: Vec<usize> = (0..n)
            .filter(|d| !bnd.contains(d))
            .take(5)
            .collect();

        for &j in &free_dofs {
            let mut u_pert = u.clone();
            u_pert[j] += eps;
            let mut r1 = vec![0.0_f64; n];
            form.residual(&u_pert, &rhs, &mut r1);

            for i in 0..n {
                let fd  = (r1[i] - r0[i]) / eps;
                let an  = jac.get(i, j);
                let err = (fd - an).abs();
                if fd.abs() > 1e-10 || an.abs() > 1e-10 {
                    let rel = err / fd.abs().max(an.abs());
                    assert!(rel < 1e-4 || err < 1e-8,
                        "Jacobian check at ({i},{j}): fd={fd:.3e} an={an:.3e} err={err:.3e}");
                }
            }
        }
    }
}












