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

/// Second-order ODE system: `M · d²u/dt² = f(t, u, du/dt)`.
///
/// For wave-equation-type systems of the form
/// `M · ü + C · u̇ + K · u = f(t)`.
///
/// MFEM equivalent: `SecondOrderTimeDependentOperator`.
pub trait SecondOrderTimeDependentOperator: Send + Sync {
    /// Compute `d²u/dt²` for the explicit scheme:
    /// `M · d²u/dt² = f(t, u, du/dt)`.
    fn mult(&self, t: f64, u: &[f64], dudt: &[f64], d2udt2: &mut [f64]);

    /// Solve the backward-Euler-type implicit system:
    /// `(M + γ·dt·C + β·dt²·K) · d²u/dt² = f(t, u + dt·du/dt + ...)`.
    ///
    /// The default calls `mult` (suitable for undamped systems where M
    /// is the mass matrix and `d²u/dt² = M^{-1} · (-K·u)`).
    fn implicit_solve(&self, dt: f64, t: f64, u: &[f64], dudt: &[f64], d2udt2: &mut [f64]) {
        let _ = dt;
        self.mult(t, u, dudt, d2udt2);
    }
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
}
