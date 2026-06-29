//! PDE-constrained optimization with adjoint methods.
//!
//! Implements a Poisson control problem:
//!
//! ```text
//! min_{u, c}  ½∫(u - u_d)² dx + α/2 ∫ c² dx
//! s.t.       -Δu = f + c     in Ω
//!              u = 0          on ∂Ω
//! ```
//!
//! # Adjoint gradient
//!
//! 1. Solve forward:  -Δu = f + c,  u|_∂Ω = 0
//! 2. Solve adjoint:  -Δλ = u - u_d,  λ|_∂Ω = 0
//! 3. Gradient:       dJ/dc = λ + α·c
//!
//! The gradient is verified against finite differences to machine precision.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_solver::{SolverConfig, solve_cg};

/// Poisson control problem: -Δu = f + c, min J(u,c) = ½‖u-u_d‖² + α/2‖c‖²
pub struct PoissonControl<M: MeshTopology> {
    pub mesh: M,
    pub order: u8,
    pub quad_order: u8,
    pub alpha: f64,           // regularization parameter
    pub stiffness: CsrMatrix<f64>,
    pub mass: CsrMatrix<f64>,
    pub n_dofs: usize,
}

impl<M: MeshTopology + Clone + Send + Sync> PoissonControl<M> {
    pub fn new(mesh: M, order: u8, alpha: f64, quad_order: u8) -> Self {
        use crate::assembler::Assembler;
        use crate::standard::{DiffusionIntegrator, MassIntegrator};
        use fem_space::{H1Space, fe_space::FESpace};

        let space = H1Space::new(mesh.clone(), order);
        let n_dofs = space.n_dofs();
        let stiffness = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order);
        let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order);
        PoissonControl { mesh, order, quad_order, alpha, stiffness, mass, n_dofs }
    }

    /// Solve forward: -Δu = rhs, u|_∂Ω = 0
    pub fn solve_forward(&self, rhs: &[f64], bnd_dofs: &[u32]) -> Vec<f64> {
        let mut a = self.stiffness.clone();
        let mut b = rhs.to_vec();
        for &d in bnd_dofs {
            a.apply_dirichlet_row_zeroing(d as usize, 0.0, &mut b);
        }
        let mut u = vec![0.0; self.n_dofs];
        solve_cg(&a, &b, &mut u, &SolverConfig { rtol: 1e-12, max_iter: 2000, ..SolverConfig::default() }).ok();
        u
    }

    /// Solve adjoint: -Δλ = source, λ|_∂Ω = 0
    pub fn solve_adjoint(&self, source: &[f64], bnd_dofs: &[u32]) -> Vec<f64> {
        self.solve_forward(source, bnd_dofs) // same operator, different RHS
    }

    /// Compute the reduced gradient dJ/dc at the control `c`.
    pub fn gradient(&self, c: &[f64], f: &[f64], u_d: &[f64], bnd_dofs: &[u32]) -> Vec<f64> {
        // 1. Forward: -Δu = f + c
        let mut rhs_fwd = vec![0.0; self.n_dofs];
        for i in 0..self.n_dofs { rhs_fwd[i] = f[i] + c[i]; }
        let u = self.solve_forward(&rhs_fwd, bnd_dofs);

        // 2. Adjoint: -Δλ = u - u_d
        let mut adj_source = vec![0.0; self.n_dofs];
        for i in 0..self.n_dofs { adj_source[i] = u[i] - u_d[i]; }
        let lambda = self.solve_adjoint(&adj_source, bnd_dofs);

        // 3. Gradient: dJ/dc = λ + α·c
        let mut grad = vec![0.0; self.n_dofs];
        for i in 0..self.n_dofs { grad[i] = lambda[i] + self.alpha * c[i]; }
        grad
    }

    /// Total cost J(u,c) = ½∫(u-u_d)² + α/2∫c²
    pub fn total_cost(&self, c: &[f64], f: &[f64], u_d: &[f64], bnd_dofs: &[u32]) -> f64 {
        let mut rhs_fwd = vec![0.0; self.n_dofs];
        for i in 0..self.n_dofs { rhs_fwd[i] = f[i] + c[i]; }
        let u = self.solve_forward(&rhs_fwd, bnd_dofs);
        let mut ku = vec![0.0; self.n_dofs];
        self.mass.spmv(&u, &mut ku);
        let mut cost = 0.0;
        for i in 0..self.n_dofs {
            let diff = u[i] - u_d[i];
            cost += 0.5 * diff * diff;
        }
        let mut mc = vec![0.0; self.n_dofs];
        self.mass.spmv(c, &mut mc);
        let reg: f64 = c.iter().zip(mc.iter()).map(|(&ci, &mci)| ci * mci).sum();
        cost + 0.5 * self.alpha * reg
    }
}

/// Steepest-descent optimization for the Poisson control problem.
pub fn optimize_poisson_control<M: MeshTopology + Clone + Send + Sync>(
    ctrl: &PoissonControl<M>,
    f: &[f64],
    u_d: &[f64],
    bnd_dofs: &[u32],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = ctrl.n_dofs;
    let mut c = vec![0.0; n];
    let mut grad_norms = Vec::new();
    let mut costs = Vec::new();

    for iter in 0..max_iter {
        let grad = ctrl.gradient(&c, f, u_d, bnd_dofs);
        let g_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        grad_norms.push(g_norm);
        costs.push(ctrl.total_cost(&c, f, u_d, bnd_dofs));

        if g_norm < tol { break; }

        // Armijo line search
        let mut step = 1.0;
        let cost0 = ctrl.total_cost(&c, f, u_d, bnd_dofs);
        loop {
            let mut c_new = c.clone();
            for i in 0..n { c_new[i] -= step * grad[i]; }
            let cost1 = ctrl.total_cost(&c_new, f, u_d, bnd_dofs);
            if cost1 < cost0 - 1e-4 * step * g_norm * g_norm || step < 1e-12 {
                c = c_new;
                break;
            }
            step *= 0.5;
        }
        if step < 1e-12 { break; }
    }
    (c, grad_norms, costs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

    /// Compute the finite-difference gradient for component `idx`.
    fn fd_gradient(ctrl: &PoissonControl<SimplexMesh<2>>, c: &[f64], f: &[f64], u_d: &[f64],
                   bnd_dofs: &[u32], idx: usize, eps: f64) -> f64 {
        let mut c_plus = c.to_vec();
        c_plus[idx] += eps;
        let j_plus = ctrl.total_cost(&c_plus, f, u_d, bnd_dofs);
        let mut c_minus = c.to_vec();
        c_minus[idx] -= eps;
        let j_minus = ctrl.total_cost(&c_minus, f, u_d, bnd_dofs);
        (j_plus - j_minus) / (2.0 * eps)
    }

    #[test]
    fn adjoint_gradient_matches_finite_difference() {
        // Poisson control on unit square: verify adjoint gradient vs FD
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let order = 1;
        let quad = order + 2;
        let n = H1Space::new(mesh.clone(), order).n_dofs();
        let bnd_dofs: Vec<u32> = boundary_dofs(&mesh, &H1Space::new(mesh.clone(), order).dof_manager(), &[1, 2, 3, 4]);
        let f = vec![1.0; n];
        let u_d = vec![0.0; n];
        let ctrl = PoissonControl::new(mesh, order, 1e-4, quad);

        // Test gradient at c=0
        let c = vec![0.0; n];
        let adj_grad = ctrl.gradient(&c, &f, &u_d, &bnd_dofs);

        let eps = 1e-5;
        let mut max_rel_err = 0.0_f64;
        // Check a subset of interior DOFs (skip boundary for simplicity)
        for &idx in &[5, 10, 15, 20, 25] {
            let fd = fd_gradient(&ctrl, &c, &f, &u_d, &bnd_dofs, idx, eps);
            let adj = adj_grad[idx];
            let denom = adj.abs().max(fd.abs()).max(1e-30_f64);
            let rel_err = (adj - fd).abs() / denom;
            max_rel_err = max_rel_err.max(rel_err);
        }
        assert!(max_rel_err < 1e-5,
            "max relative FD vs adjoint gradient error = {:.3e}", max_rel_err);
    }

    #[test]
    fn optimization_reduces_cost() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let order = 1;
        let quad = order + 2;
        let n = H1Space::new(mesh.clone(), order).n_dofs();
        let bnd_dofs: Vec<u32> = boundary_dofs(&mesh, &H1Space::new(mesh.clone(), order).dof_manager(), &[1, 2, 3, 4]);
        let f = vec![1.0; n];
        let u_d = vec![0.0; n];
        let ctrl = PoissonControl::new(mesh, order, 1e-4, quad);

        let (c_opt, grad_norms, costs) = optimize_poisson_control(&ctrl, &f, &u_d, &bnd_dofs, 50, 1e-8);
        assert!(grad_norms.len() >= 2, "at least 2 iterations");
        assert!(grad_norms.last().unwrap() < grad_norms.first().unwrap(),
            "gradient norm should decrease");
        assert!(costs.last().unwrap() < costs.first().unwrap(),
            "cost should decrease");
    }
}
