//! # fem-assembly
//!
//! Bilinear/linear form assembly: [`Assembler`], [`BilinearIntegrator`],
//! [`LinearIntegrator`], and standard integrators (diffusion, mass, source,
//! Neumann, elasticity).
//!
//! - **IGA FESpace:** for [`IgaFESpace1D`](fem_space::IgaFESpace1D) / [`IgaFESpace2D`](fem_space::IgaFESpace2D),
//!   use [`Assembler::assemble_bilinear_iga_1d`](Assembler::assemble_bilinear_iga_1d) /
//!   [`Assembler::assemble_bilinear_iga_2d`](Assembler::assemble_bilinear_iga_2d) (or [`iga_assembler`](iga_assembler) directly),
//!   not the generic [`Assembler::assemble_bilinear`](Assembler::assemble_bilinear) loop.
//!   1D physical L² and Laplacian w.r.t. an isogeometric map `x(u) = Σ c_i R_i` are
//!   [`assemble_bilinear_mass_iga_1d_physical`](iga_assembler::assemble_bilinear_mass_iga_1d_physical) /
//!   [`assemble_bilinear_diffusion_iga_1d_physical`](iga_assembler::assemble_bilinear_diffusion_iga_1d_physical)
//!   (or [`Assembler::assemble_bilinear_iga_1d_mass_physical`](Assembler::assemble_bilinear_iga_1d_mass_physical) / [`assemble_bilinear_iga_1d_physical`](Assembler::assemble_bilinear_iga_1d_physical));
//!   Parametric 1D / 2D Helmholtz: [`assemble_bilinear_helmholtz_iga_1d`](iga_assembler::assemble_bilinear_helmholtz_iga_1d) /
//!   [`assemble_bilinear_helmholtz_iga_2d`](iga_assembler::assemble_bilinear_helmholtz_iga_2d);
//!   Physical 1D: [`assemble_bilinear_helmholtz_iga_1d_physical`](iga_assembler::assemble_bilinear_helmholtz_iga_1d_physical) /
//!   [`Assembler::assemble_bilinear_iga_1d_helmholtz_physical`](Assembler::assemble_bilinear_iga_1d_helmholtz_physical);
//!   [`Assembler::assemble_bilinear_iga_1d_helmholtz`](Assembler::assemble_bilinear_iga_1d_helmholtz) and
//!   [`Assembler::assemble_bilinear_iga_2d_helmholtz`](Assembler::assemble_bilinear_iga_2d_helmholtz) are thin wrappers;
//!   `assemble_bilinear_iga_1d/2d` fuses a single diffusion + single mass item into one pass.
//!   1D parametric stiffness and mass in `u` remain [`assemble_bilinear_diffusion_iga_1d`](iga_assembler::assemble_bilinear_diffusion_iga_1d) /
//!   [`assemble_bilinear_mass_iga_1d`](iga_assembler::assemble_bilinear_mass_iga_1d).
//!
//! Also provides:
//! - [`MixedAssembler`] — rectangular assembly for mixed bilinear forms.
//! - [`DgAssembler`] — interior-penalty DG assembly (Phase 14).
//! - [`h1_quad_order_hint`] — map legacy quadrature hints to triangle/tet rule orders for H¹ assembly.
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
//! - **`parallel`** — Rayon-parallel **volume** assembly for [`Assembler`],
//!   [`VectorAssembler`], [`MixedAssembler`], and DG volume / face loops when
//!   counts meet `assembly_parallel_min_elems()` (adaptive default: `64` on a
//!   single worker down to `8` on 8+ workers; env
//!   `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS` forces a fixed threshold), plus
//!   `fem-linalg/parallel` for threaded SpMV on large local matrices.
//! - **`reed`** — libCEED-style QFunctions plus coordinated FEM entry points in the `reed`
//!   submodule (`FemCeed`, scalar H¹ `apply_mass_{2d,3d}` / `apply_poisson_{2d,3d}`, …) using the same
//!   [`Assembler`] + [`H1Space`](fem_space::H1Space) kernels as default builds.  With this feature,
//!   `cache_*` / `cache_h1_scalar_ops_{2d,3d}` wrappers and discrete helpers (`assemble_mass_h1_2d`,
//!   curl pairing, …) are also **re-exported at the crate root**.  Quadrature hint mapping
//!   (`h1_tri_quad_order`, `h1_tet_quad_order`) lives in [`h1_quad_order_hint`] for every build.
//!   Curl CSR and
//!   `assemble_curl_hdiv_pairing_2d_nd2_rt2` align with [`DiscreteLinearOperator`] /
//!   [`VectorAssembler`].  Backed by the workspace-pinned [`reed`](https://github.com/rem-rs/reed)
//!   crates.

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
pub mod nonlinear_hyperelasticity;
pub mod partial;
pub mod vector_integrator;
pub mod vector_assembler;
pub mod vector_boundary;
pub mod grid_function;
pub mod iga_assembler;
mod assembler_iga_fespace;
pub mod postprocess;
pub mod discrete_op;
pub mod transfer;
pub mod static_cond;
pub mod iga;
pub mod dg_advection;
pub mod dpg;
pub mod dg_cdr;
pub mod navier_stokes;
pub use navier_stokes::{
    assemble_convection_matrix, assemble_divergence_matrix,
    assemble_oseen_block, assemble_pressure_mass,
    solve_oseen_step, solve_ns_picard,
    assemble_ale_convection_matrix, assemble_ale_oseen_block,
};
pub mod error_estimate;
pub mod h1_quad_order_hint;
pub mod phasefield;
pub mod fsi;
pub mod thermoelastic;

#[cfg(feature = "reed")]
pub mod reed;

#[cfg(feature = "reed")]
pub use reed::{
    assemble_curl_hdiv_pairing_2d_nd2_rt2, assemble_mass_h1_2d, assemble_mass_h1_3d,
    assemble_poisson_h1_2d, assemble_poisson_h1_3d, CachedH1Mass2d, CachedH1Mass3d,
    CachedH1Poisson2d, CachedH1Poisson3d, CachedH1ScalarOps2d, CachedH1ScalarOps3d, CeedBackend,
    FemCeed, FemCeedError,
};

pub use h1_quad_order_hint::{h1_tet_quad_order, h1_tri_quad_order};

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
pub use vector_assembler::{VectorAssembler, TRI_ND2_RT2_MIXED_QUAD_ORDER};
pub use vector_boundary::{
    VectorBoundaryAssembler, VectorBoundaryBilinearIntegrator, VectorBoundaryLinearIntegrator,
    VectorBdQpData, TangentialMassIntegrator,
};
pub use mixed::{MixedAssembler, MixedBilinearIntegrator, DivIntegrator, PressureDivIntegrator};
pub use dg::{DgAssembler};
pub use dpg::{
    solve_dpg_convection_diffusion_1d,
    solve_galerkin_convection_diffusion_1d,
};
pub use dg_advection::{DGAdvectionIntegrator, DgAdvectionRhs, DgFaceIntegrator, DgFaceQpData, assemble_dg_interior_faces, assemble_advection_boundary};
pub use dg_cdr::DgCdrSystem;
pub use dg_elasticity::DgElasticityAssembler;
pub use hyperbolic::{HyperbolicFormIntegrator, NumericalFlux};
pub use interior_faces::InteriorFaceList;
pub use nonlinear::{NonlinearForm, NewtonSolver, NewtonConfig, NewtonResult};
pub use nonlinear_hyperelasticity::HyperelasticityForm;
pub use partial::{MatFreeOperator, PAMassOperator, PADiffusionOperator, LumpedMassOperator,
                  HcurlMatrixFreeOperator, solve_hcurl_matrix_free,
                  solve_hcurl_eigen_preconditioned_amg};
#[cfg(feature = "reed")]
pub use reed::HcurlReedOperator;
pub use grid_function::GridFunction;
pub use grid_function::project_coefficient;
pub use iga_assembler::{
    assemble_bilinear_diffusion_iga_1d, assemble_bilinear_diffusion_iga_1d_physical,
    assemble_bilinear_helmholtz_iga_1d, assemble_bilinear_helmholtz_iga_1d_physical,
    assemble_bilinear_mass_iga_1d, assemble_bilinear_mass_iga_1d_physical,
    assemble_linear_source_iga_1d, assemble_linear_source_iga_1d_physical,
    assemble_bilinear_diffusion_iga_2d, assemble_bilinear_helmholtz_iga_2d,
    assemble_bilinear_mass_iga_2d, assemble_linear_source_iga_2d,
};
pub use assembler_iga_fespace::{Iga1dBilinearItem, Iga2dBilinearItem};
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
pub use phasefield::{
    assemble_degraded_stiffness, assemble_phase_field_system,
    build_elem_dof_cache, compute_elastic_energy, update_history_field,
    miehe_split_2d, MieheSplit2d,
    assemble_miehe_stiffness_and_force, compute_psi_plus,
};
pub use fsi::{
    assemble_mesh_stiffness, solve_mesh_movement_laplacian,
    fsi_interface_faces, fsi_interface_nodes,
    nodal_displacement_to_dofs,
    assemble_fluid_traction_to_struct,
    FsiConfig, FsiReport, fsi_couple_step, fsi_partitioned_solve,
};
pub use thermoelastic::{
    assemble_thermal_expansion_rhs, assemble_heat_system,
    solve_thermoelastic_staggered,
};
pub use static_cond::{StaticCondensation, GlobalBacksolve, condense_global};
