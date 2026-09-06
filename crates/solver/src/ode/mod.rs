//!
//! Provides a unified [`TimeStepper`] trait plus concrete integrators:
//!
//! | Method          | Type  | Order | Suitable for        |
//! |-----------------|-------|-------|---------------------|
//! | Forward Euler   | Explicit | 1  | non-stiff           |
//! | RK4             | Explicit | 4  | non-stiff           |
//! | Implicit Euler  | Implicit | 1  | stiff               |
//! | SDIRK-2         | Implicit | 2  | stiff               |
//! | BDF-2           | Implicit | 2  | stiff, multi-step   |
//! | IMEX Euler      | IMEX  | 1     | split stiff/non-stiff |
//! | IMEX SSP-RK2    | IMEX  | 2     | split stiff/non-stiff |
//! | IMEX RK3 / ARK3 | IMEX  | 3     | split stiff/non-stiff |
//! | Complex CN      | Implicit | 2     | Schrödinger-type    |

pub mod explicit;
pub mod imex;
pub mod implicit;
pub mod symplectic;
pub mod traits;
pub mod complex_cn;

pub use explicit::{ForwardEuler, Rk4};
pub use imex::{
    ImexArk3, ImexDirkRk3, ImexEuler, ImexExpImplEuler, ImexRk2_222, ImexRk2_232, ImexRk3,
    ImexSsp2, ImexTimeStepper,
};
pub use implicit::{Bdf2, Bdf2State, CrankNicolson, ImplicitEuler, Sdirk2};
pub use symplectic::{SIAVSolver, Yoshida4};
pub use traits::{ImplicitTimeStepper, ImexOperator, TimeStepper};
pub use complex_cn::{build_complex_hamiltonian, build_complex_hamiltonian_real,
                     ComplexCrankNicolson};
