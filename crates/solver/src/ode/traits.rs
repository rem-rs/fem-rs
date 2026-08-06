//! Traits for ODE time integration: [`TimeStepper`], [`ImplicitTimeStepper`],
//! [`HamiltonianSystem`], [`ImexOperator`].

use fem_linalg::CsrMatrix;

/// A single time-step integrator.
///
/// The RHS function has signature `rhs(t, u, dudt)` and computes
/// the time derivative `dudt = f(t, u)`.
pub trait TimeStepper: Send + Sync {
    /// Advance `u` from time `t` by step `dt`, using `rhs(t, u, dudt)`.
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where
        F: Fn(f64, &[f64], &mut [f64]);
}

/// An implicit time-step integrator that needs to solve a nonlinear/linear system.
///
/// The `jac_fn` assembles the (approximate) Jacobian `∂f/∂u` at `(t, u)`.
/// For linear problems this is exact; for nonlinear problems it gives the Picard Jacobian.
pub trait ImplicitTimeStepper: Send + Sync {
    /// Advance `u` from time `t` by step `dt`.
    ///
    /// `rhs(t, u, dudt)` computes `f(t, u)`.
    /// `jac_fn(t, u)` returns a CSR matrix approximating `∂f/∂u`.
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>;
}

/// Hamiltonian system in canonical form:
///
/// dq/dt =  ∂H/∂p
/// dp/dt = -∂H/∂q
pub trait HamiltonianSystem: Send + Sync {
    /// Compute `dH/dq`.
    fn grad_q(&self, q: &[f64], p: &[f64], out: &mut [f64]);
    /// Compute `dH/dp`.
    fn grad_p(&self, q: &[f64], p: &[f64], out: &mut [f64]);
}

/// Split operator interface for IMEX methods.
///
/// Represents systems of the form:
/// `du/dt = f_E(t, u) + f_I(t, u)`
/// where `f_E` is treated explicitly and `f_I` implicitly.
pub trait ImexOperator: Send + Sync {
    /// Compute explicit (non-stiff) part `f_E(t, u)`.
    fn explicit(&self, t: f64, u: &[f64], out: &mut [f64]);

    /// Compute implicit (stiff) part `f_I(t, u)`.
    fn implicit(&self, t: f64, u: &[f64], out: &mut [f64]);

    /// Jacobian of implicit part: `J_I = ∂f_I/∂u`.
    fn jac_implicit(&self, t: f64, u: &[f64]) -> CsrMatrix<f64>;

    /// Solve the implicit-stage system `k = g(u, dt)` in MFEM's
    /// `ImplicitSolve(dt, x, k)` semantics: for `f_I(u) = -M⁻¹ S u` this is
    /// `(M + dt·S) k = -S·x`.  The default implementation uses the Jacobian:
    /// `(I − dt·J_I(u)) k = f_I(u)` solved with GMRES.
    fn implicit_solve(&self, dt: f64, x: &[f64], k: &mut [f64]) {
        let n = x.len();
        let jac = self.jac_implicit(0.0, x);
        let lhs = crate::ode::implicit::identity_minus_dt_jac(&jac, dt);
        let mut b = vec![0.0f64; n];
        self.implicit(0.0, x, &mut b);
        let cfg = crate::SolverConfig {
            rtol: 1e-9,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..crate::SolverConfig::default()
        };
        crate::solve_gmres(&lhs, &b, k, 30, &cfg).expect("ImexOperator::implicit_solve failed");
    }
}
