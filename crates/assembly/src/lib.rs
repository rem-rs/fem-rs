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
//!
//! ## Feature flags
//!
//! - **`parallel`** — Rayon-parallel volume assembly when `n_elements` meets
//!   `assembly_parallel_min_elems()` (default `64`; override env
//!   `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS`), and enables `fem-linalg/parallel` for
//!   threaded SpMV on large local matrices.
//! - **`reed`** — libCEED-style partial assembly helpers backed by the
//!   workspace-pinned [`reed`](https://github.com/rem-rs/reed) crates (`fem_assembly::reed`).

pub mod assembler;
pub mod backend;
pub mod coefficient;
pub mod complex;
pub mod integrator;
pub mod standard;
pub mod mixed;
pub mod interior_faces;
pub mod dg;
pub mod dg_elasticity;
pub mod hyperbolic;
pub mod nonlinear;
pub mod partial;
pub mod vector_integrator;
pub mod vector_assembler;
pub mod vector_boundary;
pub mod grid_function;
pub mod postprocess;
pub mod discrete_op;
pub mod transfer;
pub mod static_cond;
pub mod iga;

#[cfg(feature = "reed")]
pub mod reed;

pub use assembler::{Assembler, face_dofs_p1, face_dofs_p2};
#[cfg(feature = "parallel")]
pub use assembler::{assembly_parallel_min_elems, FEM_ASSEMBLY_PARALLEL_MIN_ELEMS};
pub use backend::{CsrLinearOperator, LinearOperator, OperatorBackend};
pub use complex::{ComplexAssembler, ComplexGridFunction, ComplexLinearForm, ComplexSystem};
pub use discrete_op::{DiscreteLinearOperator, DiscreteOpError};
pub use integrator::{
    BdQpData, BilinearIntegrator, BoundaryBilinearIntegrator, BoundaryLinearIntegrator,
    LinearIntegrator, QpData,
};
pub use vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
pub use vector_assembler::VectorAssembler;
pub use vector_boundary::{
    VectorBoundaryAssembler, VectorBoundaryBilinearIntegrator, VectorBoundaryLinearIntegrator,
    VectorBdQpData, TangentialMassIntegrator,
};
pub use mixed::{MixedAssembler, MixedBilinearIntegrator, DivIntegrator, PressureDivIntegrator};
pub use dg::{DgAssembler};
pub use dg_elasticity::DgElasticityAssembler;
pub use hyperbolic::{HyperbolicFormIntegrator, NumericalFlux};
pub use interior_faces::InteriorFaceList;
pub use nonlinear::{NonlinearForm, NewtonSolver, NewtonConfig, NewtonResult};
pub use partial::{MatFreeOperator, PAMassOperator, PADiffusionOperator, LumpedMassOperator};
pub use grid_function::GridFunction;
pub use postprocess::{compute_element_gradients, compute_h1_error, compute_kelly_indicators, recover_gradient_nodal};
pub use transfer::{
    net_boundary_flux_h1_p1_2d,
    transfer_h1_p1_nonmatching,
    transfer_h1_p1_nonmatching_3d,
    transfer_h1_p1_nonmatching_l2_projection,
    transfer_h1_p1_nonmatching_l2_projection_3d,
    transfer_h1_p1_nonmatching_l2_projection_conservative,
    transfer_h1_p1_nonmatching_l2_projection_conservative_3d,
    ConservativeTransferReport,
    TransferError,
    TransferStats,
};
pub use static_cond::{StaticCondensation, GlobalBacksolve, condense_global};
