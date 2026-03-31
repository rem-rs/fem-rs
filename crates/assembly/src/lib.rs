//! # fem-assembly
//!
//! Bilinear/linear form assembly: [`Assembler`], [`BilinearIntegrator`],
//! [`LinearIntegrator`], and standard integrators (diffusion, mass, source,
//! Neumann).
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
//! use fem_mesh::SimplexMesh;
//! use fem_space::H1Space;
//!
//! let mesh  = SimplexMesh::<2>::unit_square_tri(16);
//! let space = H1Space::new(mesh, 1);
//!
//! // Assemble K and f for -Δu = 2π² sin(πx)sin(πy)
//! let stiffness = Assembler::assemble_bilinear(
//!     &space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
//! let rhs = Assembler::assemble_linear(
//!     &space, &[&DomainSourceIntegrator::new(|x| {
//!         use std::f64::consts::PI;
//!         2.0 * PI * PI * (PI*x[0]).sin() * (PI*x[1]).sin()
//!     })], 3);
//! ```
//!
//! Enable the `parallel` feature for rayon-parallel element loops (not yet implemented).

pub mod assembler;
pub mod integrator;
pub mod standard;

pub use assembler::{Assembler, face_dofs_p1, face_dofs_p2};
pub use integrator::{
    BdQpData, BilinearIntegrator, BoundaryLinearIntegrator,
    LinearIntegrator, QpData,
};
