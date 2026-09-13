//!
//! Provides a unified [`TimeStepper`] trait plus concrete integrators:
//!
//! | Method          | Type  | Order | Suitable for        |
//! |-----------------|-------|-------|---------------------|
//! | Forward Euler   | Explicit | 1  | non-stiff           |
//! | RK2 family      | Explicit | 2  | non-stiff           |
//! | RK4             | Explicit | 4  | non-stiff           |
//! | RK6 / RK8       | Explicit | 6/8 | high accuracy      |
//! | AB5             | Explicit | 5  | non-stiff, multistep|
//! | Implicit Euler  | Implicit | 1  | stiff               |
//! | SDIRK-2         | Implicit | 2  | stiff               |
//! | Implicit midpoint | Implicit | 2 | stiff, symplectic   |
//! | SDIRK-34 / SDIRK-33 | Implicit | 4/3 | stiff          |
//! | ESDIRK-32 / ESDIRK-33 | Implicit | 2/3 | stiff        |
//! | BDF-2           | Implicit | 2  | stiff, multi-step   |
//! | IMEX Euler      | IMEX  | 1     | split stiff/non-stiff |
//! | IMEX SSP-RK2    | IMEX  | 2     | split stiff/non-stiff |
//! | IMEX RK3 / ARK3 | IMEX  | 3     | split stiff/non-stiff |
//! | Complex CN      | Implicit | 2     | Schrödinger-type    |

pub mod explicit;
pub mod high_order;
pub mod imex;
pub mod implicit;
pub mod mfem_ode;
pub mod structural;
pub mod symplectic;
pub mod traits;
pub mod complex_cn;

pub use explicit::{ForwardEuler, Rk3Ssp, Rk4};
pub use high_order::{
    Ab5, Ab5State, Esdirk32, Esdirk33, ImplicitMidpoint, Rk2, Rk6, Rk8, Sdirk33, Sdirk34,
};
pub use imex::{
    ImexArk3, ImexDirkRk3, ImexEuler, ImexExpImplEuler, ImexRk2_222, ImexRk2_232, ImexRk3,
    ImexSsp2, ImexTimeStepper,
};
pub use implicit::{Bdf2, Bdf2State, CrankNicolson, ImplicitEuler, Sdirk2};
pub use structural::{Newmark, NewmarkState, GeneralizedAlpha, GeneralizedAlphaState};
pub use symplectic::{SIAVSolver, Yoshida4};
pub use mfem_ode::{
    compute_slope_from_state, BackwardEulerSolver, DenseOperator, ImplicitMidpointSolver,
    LinearOp, OdeSolver, Sdirk23Gamma, Sdirk23Solver, Sdirk33Solver, Sdirk34Solver, SiaState,
    SiavSolver, TimeDependentOperator,
};
pub use traits::{ImplicitTimeStepper, ImexOperator, TimeStepper};
pub use complex_cn::{build_complex_hamiltonian, build_complex_hamiltonian_real,
                     ComplexCrankNicolson};
