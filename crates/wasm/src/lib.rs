//! # fem-wasm
//!
//! WebAssembly bindings for fem-rs.  Exposes a JS-friendly [`WasmSolver`]
//! API for solving the Poisson equation on a unit-square mesh entirely in the
//! browser.
//!
//! ## Feature flags
//!
//! | Feature  | Effect |
//! |----------|--------|
//! | `wasm`   | Enable `wasm-bindgen` attribute macros; required when building with `wasm-pack`. |
//! | *(none)* | Compile as a plain `rlib` — all logic is testable with `cargo test`. |
//!
//! ## Quick start (native test)
//! ```rust
//! use fem_wasm::WasmSolver;
//! use std::f64::consts::PI;
//!
//! let mut solver = WasmSolver::new(8);
//! let n = solver.n_dofs() as usize;
//! let rhs = solver.assemble_constant_rhs(2.0 * PI * PI);
//! let u   = solver.solve(&rhs).unwrap();
//! assert_eq!(u.len(), n);
//! ```

mod solver;

pub use solver::WasmSolver;
