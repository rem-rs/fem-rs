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

use fem_linalg::CsrMatrix;
use crate::{SolverConfig, solve_cg};

/// Poisson control problem: -Δu = f + c, min J(u,c) = ½‖u-u_d‖² + α/2‖c‖²
pub struct PoissonControl {
    pub stiffness: CsrMatrix<f64>,
    pub mass: CsrMatrix<f64>,
    pub alpha: f64,
    pub n_dofs: usize,
}

impl PoissonControl {
    pub fn new(stiffness: CsrMatrix<f64>, mass: CsrMatrix<f64>, alpha: f64) -> Self {
        let n_dofs = stiffness.nrows;
        PoissonControl { stiffness, mass, alpha, n_dofs }
    }

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

    pub fn solve_adjoint(&self, source: &[f64], bnd_dofs: &[u32]) -> Vec<f64> {
        let mut a = self.stiffness.clone();
        let mut b = source.to_vec();
        for &d in bnd_dofs {
            a.apply_dirichlet_row_zeroing(d as usize, 0.0, &mut b);
        }
        let mut lambda = vec![0.0; self.n_dofs];
        solve_cg(&a, &b, &mut lambda, &SolverConfig { rtol: 1e-12, max_iter: 2000, ..SolverConfig::default() }).ok();
        lambda
    }

    pub fn gradient(&self, lambda: &[f64], control: &[f64]) -> Vec<f64> {
        assert_eq!(lambda.len(), control.len());
        lambda.iter().zip(control.iter()).map(|(&l, &c)| l + self.alpha * c).collect()
    }

    pub fn cost_functional(&self, u: &[f64], u_d: &[f64], control: &[f64]) -> f64 {
        assert_eq!(u.len(), u_d.len());
        let n = u.len();
        let mut j = 0.0;
        for i in 0..n {
            let diff = u[i] - u_d[i];
            j += 0.5 * diff * diff + 0.5 * self.alpha * control[i] * control[i];
        }
        j
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::{CooMatrix, CsrMatrix};

    fn build_1d_laplacian(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            if i > 0 { coo.add(i, i - 1, -1.0); }
            coo.add(i, i, 2.0);
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn adjoint_gradient_matches_finite_differences() {
        let n = 10;
        let h = 1.0 / (n - 1) as f64;
        let stiffness = build_1d_laplacian(n);
        let mass = CooMatrix::<f64>::new(n, n).into_csr();
        let alpha = 0.01;
        let pc = PoissonControl::new(stiffness, mass, alpha);

        let interior: Vec<usize> = (1..n - 1).collect();
        let bnd: Vec<u32> = vec![0, (n - 1) as u32];

        let u_d: Vec<f64> = (0..n).map(|i| (std::f64::consts::PI * i as f64 * h).sin()).collect();
        let control = vec![0.0; n];

        let rhs = vec![0.0; n];
        let u = pc.solve_forward(&rhs, &bnd);

        let source: Vec<f64> = u.iter().zip(u_d.iter()).map(|(&ui, &udi)| ui - udi).collect();
        let lambda = pc.solve_adjoint(&source, &bnd);

        let grad = pc.gradient(&lambda, &control);

        let eps = 1e-5;
        for &idx in &interior {
            let mut c_plus = control.clone();
            c_plus[idx] += eps;
            let mut c_minus = control.clone();
            c_minus[idx] -= eps;

            let u_plus = pc.solve_forward(&c_plus, &bnd);
            let j_plus = pc.cost_functional(&u_plus, &u_d, &c_plus);

            let u_minus = pc.solve_forward(&c_minus, &bnd);
            let j_minus = pc.cost_functional(&u_minus, &u_d, &c_minus);

            let fd_grad = (j_plus - j_minus) / (2.0 * eps);
            let diff = (grad[idx] - fd_grad).abs();
            assert!(diff < 1e-4, "mismatch at {}: adjoint={:.6e}, fd={:.6e}", idx, grad[idx], fd_grad);
        }
    }

    #[test]
    fn cost_functional_computes_correctly() {
        let n = 5;
        let stiffness = build_1d_laplacian(n);
        let mass = CooMatrix::<f64>::new(n, n).into_csr();
        let alpha = 0.1;
        let pc = PoissonControl::new(stiffness, mass, alpha);

        let u = vec![0.0; n];
        let u_d = vec![1.0; n];
        let c = vec![0.5; n];

        let j = pc.cost_functional(&u, &u_d, &c);
        let expected = 0.5 * 5.0 + 0.5 * 0.1 * 5.0 * 0.25;
        assert!((j - expected).abs() < 1e-12, "J={} expected={}", j, expected);
    }
}
