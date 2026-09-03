//! ODE / Time integrators.
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

pub mod explicit;
pub mod imex;
pub mod implicit;
pub mod traits;

pub use explicit::{ForwardEuler, Rk4};
pub use imex::{
    ImexArk3, ImexDirkRk3, ImexEuler, ImexExpImplEuler, ImexRk2_222, ImexRk2_232, ImexRk3,
    ImexSsp2, ImexTimeStepper,
};
pub use implicit::{Bdf2, Bdf2State, ImplicitEuler, Sdirk2};
pub use traits::ImplicitTimeStepper;
pub use traits::{ImexOperator, TimeStepper};
