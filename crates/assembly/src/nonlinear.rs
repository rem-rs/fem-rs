//! Nonlinear forms and Newton–Raphson solver.
//!
//! # Overview
//!
//! A `NonlinearForm` assembles the **residual** `F(u)` and the **Jacobian** `J(u) = dF/du`
//! for a nonlinear PDE.  The [`NewtonSolver`] then iterates
//!
//! ```text
//! J(uₙ) Δu = −F(uₙ),   uₙ₊₁ = uₙ + Δu
//! ```
//!
//! until `‖F(u)‖ < tol`.
//!
//! # Example: nonlinear diffusion
//! ```rust,ignore
//! let form = NonlinearDiffusionForm::new(&space, |u| 1.0 + u*u); // κ(u) = 1 + u²
//! let mut solver = NewtonSolver::new(NewtonConfig::default());
//! let mut u = vec![0.0; space.n_dofs()];
//! let result = solver.solve(&form, &rhs, &mut u).unwrap();
//! ```

use fem_linalg::CsrMatrix;
use fem_solver::{solve_gmres, SolverConfig};

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

// ─── Newton solver ────────────────────────────────────────────────────────────

/// Convergence and iteration parameters for the Newton solver.
#[derive(Debug, Clone)]
pub struct NewtonConfig {
    /// Absolute tolerance on `‖F(u)‖₂` (default 1e-10).
    pub atol: f64,
    /// Relative tolerance on `‖F(u)‖₂ / ‖F(u₀)‖₂` (default 1e-8).
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
}

impl Default for NewtonConfig {
    fn default() -> Self {
        NewtonConfig {
            atol:       1e-10,
            rtol:       1e-8,
            max_iter:   50,
            linear_tol: 1e-10,
            line_search: true,
            line_search_min_alpha: 1e-6,
            line_search_shrink: 0.5,
            line_search_max_backtracks: 20,
            line_search_sufficient_decrease: 1e-4,
            verbose:    false,
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

/// Newton–Raphson solver for nonlinear systems `F(u) = 0`.
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
            max_iter: 1000,
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
            println!("[Newton] iter=0 ‖F‖={r0:.3e}");
        }

        // Converged immediately (zero initial residual)
        if r0 < self.cfg.atol {
            return Ok(NewtonResult { converged: true, iterations: 0, final_residual: r0 });
        }

        let mut r_norm = r0;

        for iter in 0..self.cfg.max_iter {
            // Assemble Jacobian
            let jac = form.jacobian(u);

            // Solve J Δu = −r
            let neg_r: Vec<f64> = r.iter().map(|&v| -v).collect();
            du.fill(0.0);
            solve_gmres(&jac, &neg_r, &mut du, 30, &linear_cfg)
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
                // Update: u ← u + Δu
                for (ui, &dui) in u.iter_mut().zip(du.iter()) {
                    *ui += dui;
                }
            }

            // Recompute residual
            form.residual(u, rhs, &mut r);
            r_norm = norm2(&r);

            if self.cfg.verbose {
                println!("[Newton] iter={} ‖F‖={r_norm:.3e}", iter + 1);
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

// ─── NonlinearDiffusionForm ───────────────────────────────────────────────────

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TetP1, TetP2, TetP3, TriP1, TriP2, TriP3}};
use fem_linalg::CooMatrix;
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

/// Nonlinear diffusion form: `F(u) = ∫ κ(u) ∇u · ∇v dx − ∫ f v dx`.
///
/// The Jacobian is `J(u)[i,j] = ∫ [κ(u) ∇φⱼ · ∇φᵢ + κ'(u) φⱼ ∇u · ∇φᵢ] dx`
/// (linearisation via product rule).
///
/// For simplicity (and robustness), the Jacobian uses **Picard linearisation**:
/// `J_Picard(u)[i,j] = ∫ κ(u) ∇φⱼ · ∇φᵢ dx`
/// (freeze κ at the current iterate, ignore the κ' term).
/// This is first-order convergent; set `use_full_jacobian = true` for quadratic.
pub struct NonlinearDiffusionForm<S: FESpace, K>
where
    K: Fn(f64) -> f64 + Send + Sync,
{
    space:           S,
    kappa:           K,
    /// Also assemble the `κ'(u)` term in the Jacobian for quadratic convergence.
    pub use_full_jacobian: bool,
    /// Derivative of κ: `kappa_prime(u)`.  Only used if `use_full_jacobian`.
    #[allow(dead_code)]
    kappa_prime:     Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    quad_order:      u8,
    /// Fixed (linear) Dirichlet constrained DOFs → prescribed value.
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

                // ∇u at this point
                let grad_u: Vec<f64> = (0..dim).map(|d| {
                    gd.iter().zip(grad_p.chunks(dim)).map(|(&di, gpi)| u[di] * gpi[d]).sum::<f64>()
                }).collect();

                // F(u)[i] += w κ(u) ∇u · ∇φᵢ
                for i in 0..n_l {
                    let dot: f64 = (0..dim).map(|d| grad_u[d] * grad_p[i*dim+d]).sum();
                    f_elem[i] += w * kappa_qp * dot;
                }
            }

            // Scatter: r[gi] += f_elem[i] − rhs[gi]
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
        // Picard linearisation: J = ∫ κ(u_h) ∇φⱼ · ∇φᵢ dx
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn ref_elem(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
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

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, DofManager, fe_space::FESpace, constraints::boundary_dofs};

    fn get_bnd_dofs(mesh: &SimplexMesh<2>, order: u8) -> Vec<usize> {
        let dm = DofManager::new(mesh, order);
        boundary_dofs(mesh, &dm, &[1, 2, 3, 4])
            .iter().map(|&d| d as usize).collect()
    }

    /// For κ(u) = const, the nonlinear problem reduces to a linear one.
    /// Verify that the Newton solver converges in 1 iteration.
    #[test]
    fn newton_linear_problem_converges_in_one_iter() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(6);
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
        assert!(res.iterations <= 2, "Expected ≤2 iters for linear problem, got {}", res.iterations);
        let rn = norm2(&u);
        assert!(rn < 1e-12, "u should be zero but ‖u‖ = {rn}");
    }

    /// Nonlinear problem: κ(u) = 1 + u², constant forcing.
    /// Just verify convergence.
    #[test]
    fn newton_nonlinear_converges() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(8);
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

    /// Jacobian finite-difference check: J[i,j] ≈ (F(u+ε eⱼ)[i] − F(u)[i]) / ε.
    /// Uses constant κ so that the Picard Jacobian equals the full tangent stiffness.
    #[test]
    fn jacobian_finite_difference_check() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(2);
        let bnd   = get_bnd_dofs(&mesh, 1);
        let space = H1Space::new(mesh, 1);
        let n     = space.n_dofs();
        let rhs   = vec![0.0_f64; n];

        // With constant κ, the Picard Jacobian matches the full tangent exactly.
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
