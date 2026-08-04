//! Cut-element integration rules (implicit interface / subdomain quadrature).
//!
//! 1:1 port of MFEM `fem/intrules_cut.cpp` — [`MomentFitting`] constructs
//! quadrature rules on elements cut by a level set via moment-fitting
//! (Mueller–Kummer–Oberlack 2013).

pub mod div_free_3d_data;
pub mod moment_fitting;

pub use moment_fitting::rule_npts;
pub use moment_fitting::{CutGeom, CutRule, MomentFitting};
