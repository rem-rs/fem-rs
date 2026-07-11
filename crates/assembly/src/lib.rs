//! # fem-assembly
//!
//! Bilinear/linear form assembly: [`Assembler`], [`BilinearIntegrator`],
//! [`LinearIntegrator`], and standard integrators (diffusion, mass, source,
//! Neumann, elasticity).
#![allow(non_snake_case, unreachable_patterns, unused_imports, unused_variables, unused_assignments, clippy::needless_range_loop)]
//!
//! - **IGA FESpace:** for [`IgaFESpace1D`](fem_space::IgaFESpace1D) / [`IgaFESpace2D`](fem_space::IgaFESpace2D),
//!   use [`Assembler::assemble_bilinear_iga_1d`](Assembler::assemble_bilinear_iga_1d) /
//!   [`Assembler::assemble_bilinear_iga_2d`](Assembler::assemble_bilinear_iga_2d) (or [`iga_assembler`](iga_assembler) directly),
//!   not the generic [`Assembler::assemble_bilinear`](Assembler::assemble_bilinear) loop.
//!   1D physical L虏 and Laplacian w.r.t. an isogeometric map `x(u) = 危 c_i R_i` are
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
//! - [`MixedAssembler`] 鈥?rectangular assembly for mixed bilinear forms.
//! - [`DgAssembler`] 鈥?interior-penalty DG assembly (Phase 14).
//! - [`h1_quad_order_hint`] 鈥?map legacy quadrature hints to triangle/tet rule orders for H鹿 assembly.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
//! use fem_mesh::Mesh;
//! use fem_space::H1Space;
//!
//! let mesh  = Mesh::<2>::unit_square_tri(16);
//! let space = H1Space::new(mesh, 1);
//!
//! // Assemble K and f for -螖u = 2蟺虏 sin(蟺x)sin(蟺y)
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
//! - **`parallel`** 鈥?Rayon-parallel **volume** assembly for [`Assembler`],
//!   [`VectorAssembler`], [`MixedAssembler`], and DG volume / face loops when
//!   counts meet `assembly_parallel_min_elems()` (adaptive default: `64` on a
//!   single worker down to `8` on 8+ workers; env
//!   `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS` forces a fixed threshold), plus
//!   `fem-linalg/parallel` for threaded SpMV on large local matrices.
//! - **`reed`** 鈥?libCEED-style QFunctions plus coordinated FEM entry points in the `reed`
//!   submodule (`FemCeed`, scalar H鹿 `apply_mass_{2d,3d}` / `apply_poisson_{2d,3d}`, 鈥? using the same
//!   [`Assembler`] + [`H1Space`](fem_space::H1Space) kernels as default builds.  With this feature,
//!   `cache_*` / `cache_h1_scalar_ops_{2d,3d}` wrappers and discrete helpers (`assemble_mass_h1_2d`,
//!   curl pairing, 鈥? are also **re-exported at the crate root**.  Quadrature hint mapping
//!   (`h1_tri_quad_order`, `h1_tet_quad_order`) lives in [`h1_quad_order_hint`] for every build.
//!   Curl CSR and
//!   `assemble_curl_hdiv_pairing_2d_nd2_rt2` align with [`DiscreteLinearOperator`] /
//!   [`VectorAssembler`].  Backed by the workspace-pinned [`reed`](https://github.com/rem-rs/reed)
//!   crates.

pub mod assembler;
pub mod backend;
pub mod complex;
pub mod integrator;
pub mod standard;
pub mod block_assembler;
pub mod mixed;
pub mod interior_faces;
pub mod partial;
pub mod vector_integrator;
pub mod vector_assembler;
mod assembler_iga_fespace;
pub mod discrete_op;
pub mod transfer;
/// High-level Form abstractions (BilinearForm / LinearForm).
pub mod form;
pub mod ams_solver;
pub mod hdiv_error;
pub mod hybridization;
pub mod static_cond;
pub mod h1_quad_order_hint;
pub mod lor_factory;

pub use physics::navier_stokes::{
    assemble_convection_matrix, assemble_divergence_matrix,
    assemble_oseen_block, assemble_pressure_mass,
    solve_oseen_step, solve_ns_picard,
    assemble_ale_convection_matrix, assemble_ale_oseen_block,
};

// 鈹€鈹€ Reorganised subdirectory modules 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
pub mod physics;
pub mod boundary;
pub mod postproc;

// 鈹€鈹€ Method-family subdirectories 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// Discontinuous Galerkin (DG) interior-penalty assembly and related solvers.
pub mod dg;
pub use dg::*;

/// Discontinuous Petrov-Galerkin (DPG) methods.
pub mod dpg;
pub use dpg::*;

/// Hybridizable Discontinuous Galerkin (HDG) methods.
pub mod hdg;
pub use hdg::*;

/// Weak Galerkin (WG) methods.
pub mod wg;
pub use wg::*;

/// eXtended Finite Element Method (XFEM).
pub mod xfem;
pub use xfem::*;

/// Contact mechanics (Signorini, friction, mortar, Nitsche).
pub mod contact;
pub use contact::*;

/// Isogeometric Analysis (IGA).
pub mod iga;
pub use iga::*;

/// Plasticity (J2, Drucker-Prager, crystal plasticity).
pub mod plasticity;
pub use plasticity::*;

/// Phase field fracture (brittle fracture, Cahn-Hilliard, damage).
pub mod phasefield;
pub use phasefield::*;

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
    BoundaryMassIntegrator, RobinLFIntegrator,
    LinearIntegrator, QpData,
};
pub use vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
pub use vector_assembler::{VectorAssembler, TRI_ND2_RT2_MIXED_QUAD_ORDER, geo_ref_elem_from_mesh, isoparametric_jacobian};
pub use boundary::vector_boundary::{
    VectorBoundaryAssembler, VectorBoundaryBilinearIntegrator, VectorBoundaryLinearIntegrator,
    VectorBdQpData, TangentialMassIntegrator, HdivNormalFluxIntegrator,
};
pub use mixed::{MixedAssembler, MixedBilinearIntegrator, DivIntegrator, PressureDivIntegrator, HDivL2ScaledDiv};
pub use block_assembler::{
    assemble_mixed_block, assemble_diagonal_block, assemble_system_2x2,
};
pub use physics::hyperbolic::{HyperbolicFormIntegrator, NumericalFlux, HyperbolicConservationLaw, EulerConservationLaw, minmod, limiter_minmod_tet_p1};
pub use interior_faces::InteriorFaceList;
pub use physics::nonlinear::{LinearSolver, NonlinearForm, NewtonSolver, NewtonConfig, NewtonResult, JfNKConfig, JfNKSolver, AndersonConfig, AndersonAccelerator, finite_diff_jacobian, FdNonlinearForm, LbfgsConfig, LbfgsResult, LbfgsSolver, TrustRegionConfig, TrustRegionResult, TrustRegionSolver};
pub use physics::nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel};
pub use partial::{MatFreeOperator, PAMassOperator, PADiffusionOperator, LumpedMassOperator,
                  HcurlMatrixFreeOperator, solve_hcurl_matrix_free,
                  solve_hcurl_eigen_preconditioned_amg};
#[cfg(feature = "reed")]
pub use reed::HcurlReedOperator;
pub use postproc::grid_function::GridFunction;
pub use postproc::grid_function::{project_coefficient, project_hcurl_coefficient, project_hcurl_coefficient_2d};
pub use iga::iga_assembler::{
    assemble_bilinear_diffusion_iga_1d, assemble_bilinear_diffusion_iga_1d_physical,
    assemble_bilinear_helmholtz_iga_1d, assemble_bilinear_helmholtz_iga_1d_physical,
    assemble_bilinear_mass_iga_1d, assemble_bilinear_mass_iga_1d_physical,
    assemble_linear_source_iga_1d, assemble_linear_source_iga_1d_physical,
    assemble_bilinear_diffusion_iga_2d, assemble_bilinear_helmholtz_iga_2d,
    assemble_bilinear_mass_iga_2d, assemble_linear_source_iga_2d,
};
pub use assembler_iga_fespace::{Iga1dBilinearItem, Iga2dBilinearItem};
pub use postproc::postprocess::{compute_element_gradients, compute_h1_error, compute_kelly_indicators, recover_gradient_nodal};
pub use transfer::{
    build_prolongation_h1,
    build_prolongation_h1_3d,
    build_prolongation_hcurl,
    build_prolongation_hdiv,
    get_prolongation_h1,
    get_prolongation_h1_3d,
    get_prolongation_hcurl,
    get_prolongation_hdiv,
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
pub use physics::fsi::{
    assemble_mesh_stiffness, solve_mesh_movement_laplacian,
    fsi_interface_faces, fsi_interface_nodes,
    nodal_displacement_to_dofs,
    assemble_fluid_traction_to_struct,
    FsiConfig, FsiReport, fsi_couple_step, fsi_partitioned_solve,
};
pub use physics::thermoelastic::{
    assemble_thermal_expansion_rhs, assemble_heat_system,
    solve_thermoelastic_staggered,
};
pub use static_cond::{StaticCondensation, GlobalBacksolve, condense_global};
pub use lor_factory::{build_lor_amg_h1, build_lor_amg_h1_3d};
pub use form::{BilinearForm, LinearForm, VectorBilinearForm, VectorLinearForm, form_linear_system, recover_fem_solution};
pub use postproc::coefficient::{ConstantMatrixCoeff, FnMatrixCoeff, PwMatrixCoeff, ScalarMatrixCoeff, MeshDependentCoefficient};

// 鈹€鈹€ Re-export sub-modules from reorganized directories 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
pub use physics::*;
pub use boundary::*;
pub use postproc::*;

