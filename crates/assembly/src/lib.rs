//! # fem-assembly
//!
//! Bilinear/linear form assembly: [`Assembler`], [`BilinearIntegrator`],
//! [`LinearIntegrator`], and standard integrators (diffusion, mass, source,
//! Neumann, elasticity).
//!
//! Also provides:
//! - [`MixedAssembler`] — rectangular assembly for mixed bilinear forms.
//! - [`DgAssembler`] — interior-penalty DG assembly (Phase 14).
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

pub mod assembler;
pub mod coefficient;
pub mod integrator;
pub mod standard;
pub mod mixed;
pub mod interior_faces;
pub mod dg;
pub mod nonlinear;
pub mod partial;
pub mod vector_integrator;
pub mod vector_assembler;
pub mod grid_function;
pub mod postprocess;
pub mod discrete_op;

pub use assembler::{Assembler, face_dofs_p1, face_dofs_p2};
pub use discrete_op::DiscreteLinearOperator;
pub use integrator::{
    BdQpData, BilinearIntegrator, BoundaryBilinearIntegrator, BoundaryLinearIntegrator,
    LinearIntegrator, QpData,
};
pub use vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
pub use vector_assembler::VectorAssembler;
pub use mixed::{MixedAssembler, MixedBilinearIntegrator, DivIntegrator, PressureDivIntegrator};
pub use dg::{DgAssembler};
pub use interior_faces::InteriorFaceList;
pub use nonlinear::{NonlinearForm, NewtonSolver, NewtonConfig, NewtonResult};
pub use partial::{MatFreeOperator, PAMassOperator, PADiffusionOperator, LumpedMassOperator};
pub use grid_function::GridFunction;
pub use postprocess::{compute_element_gradients, compute_h1_error, recover_gradient_nodal};
