pub mod dg_base;
pub mod dg;
pub mod dg_advection;
pub mod dg_elasticity;
pub mod dg_hyperbolic;
pub mod dg_imex;

pub use dg::*;
pub use dg_advection::*;
pub use dg_elasticity::*;
pub use dg_hyperbolic::*;
pub use dg_imex::*;
