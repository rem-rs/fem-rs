//! `fem-ceed` — bridge between fem-rs mesh types and the [reed] discretization
//! library.
//!
//! ## Overview
//!
//! This crate connects:
//!
//! * **fem-rs** — finite-element mesh + linear-algebra types
//! * **reed** — a libCEED-inspired operator decomposition framework
//!
//! The `Eᵀ Bᵀ D B E` operator pattern from reed maps cleanly onto the
//! fem-rs mesh:
//!
//! | Symbol | Component | fem-ceed module |
//! |--------|-----------|-----------------|
//! | E, Eᵀ | Element restriction (mesh connectivity) | [`restriction`] |
//! | B, Bᵀ | Basis evaluation (shape functions) | reed-cpu `SimplexBasis` |
//! | D      | Quadrature data (geometry factors) | [`context`] |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use fem_ceed::FemCeed;
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(8);
//! let ceed = FemCeed::new();
//!
//! // Apply scalar mass matrix: output = M * ones
//! let n_nodes = mesh.n_nodes();
//! let ones = vec![1.0_f64; n_nodes];
//! let mass_ones = ceed.apply_mass_2d(&mesh, 1, 3, &ones).unwrap();
//! // sum(mass_ones) ≈ area of unit square = 1.0
//!
//! // Apply scalar stiffness matrix: K * u
//! let stiffness_u = ceed.apply_poisson_2d(&mesh, 1, 3, &ones).unwrap();
//! // sum(stiffness_u) ≈ 0 (integral of Laplacian of constants vanishes)
//! ```

pub mod context;
pub mod qfunction;
pub mod restriction;

pub use context::{CeedBackend, FemCeed, FemCeedError};
pub use restriction::{mesh_to_elem_restriction, qdata_elem_restriction};
