//! ODE / Time integrators.
//!
//! Provides a unified [`TimeStepper`] trait plus several concrete integrators:
//!
//! | Method               | Type     | Order | Suitable for            |
//! |----------------------|----------|-------|-------------------------|
//! | Forward Euler        | Explicit | 1     | non-stiff               |
//! | RK4                  | Explicit | 4     | non-stiff               |
//! | RK45 (adaptive)      | Explicit | 4/5   | non-stiff, adaptive     |
//! | Implicit Euler (BDF-1)| Implicit | 1   | stiff                   |
//! | SDIRK-2              | Implicit | 2     | stiff                   |
//! | BDF-2                | Implicit | 2     | stiff, multi-step       |
//! | Crank-Nicolson       | Implicit | 2     | stiff, symmetric        |
//! | ABM (PECE)           | Explicit | 2–4   | multi-step              |
//! | Verlet / Leapfrog    | Symplectic | 2  | Hamiltonian systems     |
//! | Yoshida 4            | Symplectic | 4  | Hamiltonian systems     |
//! | Newmark-β            | Structural | 2  | 2nd-order ODEs (Mü+Ku=f)|
//! | Generalized-α        | Structural | 2  | 1st-order (Mv̇+Kv=f)   |
//! | IMEX Euler           | IMEX     | 1     | split stiff/non-stiff   |
//! | IMEX SSP-RK2         | IMEX     | 2     | split stiff/non-stiff   |
//! | IMEX RK3 / ARK3      | IMEX     | 3     | split stiff/non-stiff   |
//!
//! # Usage
//! ```rust,ignore
//! // du/dt = -u  →  u(t) = exp(-t)
//! let rhs = |_t: f64, u: &[f64], dudt: &mut [f64]| {
//!     dudt[0] = -u[0];
//! };
//! let solver = ForwardEuler::new(0.01);
//! let mut u = vec![1.0_f64];
//! solver.step(0.0, &mut u, rhs);
//! ```

pub mod explicit;
pub mod imex;
pub mod implicit;
pub mod structural;
pub mod symplectic;
pub mod traits;

pub use explicit::{AbmState, AdamsBashforthMoulton, ForwardEuler, Rk4, Rk45};
pub use imex::{
    ImexArk3, ImexDirkRk3, ImexEuler, ImexExpImplEuler, ImexRk2_222, ImexRk2_232, ImexRk3,
    ImexSsp2, ImexTimeStepper,
};
pub use implicit::{Bdf2, Bdf2State, CrankNicolson, ImplicitEuler, Sdirk2};
pub use structural::{
    CentralDifferenceExplicit, ExplicitState, GeneralizedAlpha, GeneralizedAlpha2,
    GeneralizedAlpha2State, GeneralizedAlphaState, Newmark, NewmarkState, SecondOrderSolver,
};
pub use symplectic::{LeapfrogStepper, SIAVSolver, VerletStepper, Yoshida4Stepper};
pub use traits::{HamiltonianSystem, ImexOperator, ImplicitTimeStepper, TimeStepper};
