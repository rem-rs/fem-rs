//! Transient incompressible Navier–Stokes solver, split scheme formulation.
//!
//! 1:1 port of MFEM's `miniapps/fluids/navier/navier_solver.{hpp,cpp}`
//! (MFEM 4.10, `mfem::navier::NavierSolver`).  The coupled momentum and
//! incompressibility equations are decoupled with the split scheme of
//! Tomboulides/Lee/Orszag (1997) — see Franco/Camier/Andrej/Pazner (2020),
//! section 4.2 — which requires three solves per time step:
//!
//! 1. **Extrapolation step** for the explicitly treated nonlinear terms
//!    (`Mv⁻¹`, CG + Jacobi) — an EXTk (Adams–Bashforth) extrapolation of
//!    `N(u) = -∫(u·∇u)·v` combined with the BDFk solution history.
//! 2. **Pressure Poisson solve** `Sp p = ∇·F̃ + boundary terms` (CG with
//!    `OrthoSolver(GSSmoother)` in the pure-Neumann case).
//! 3. **Helmholtz solve** `H u = …` with `H = (bd0/dt)·Mv + ν·K_v` and the
//!    velocity Dirichlet DOFs eliminated (CG + Jacobi).
//!
//! BDF/EXT coefficients (`SetTimeIntegrationCoefficients`) bootstrap with
//! BDF1 and raise the order each step up to `max_bdf_order` (3 by default),
//! with the variable-time-step ratios `rho1 = dt(tₙ)/dt(tₙ₋₁)` and
//! `rho2 = dt(tₙ₋₁)/dt(tₙ₋₂)`.
//!
//! # Scope
//!
//! `fem-solver` cannot depend on `fem-assembly` (the latter depends on the
//! former), so — exactly like the C++ class, which owns the `ParMesh` and all
//! the forms — the driver owns only the *time-stepping algebra* and receives
//! the discretization through the [`NavierDiscretization`] trait.  The miniapp
//! supplies the `[H¹]^d × H¹` forms (`Mv`, `Sp`, `D`, `G`, `H`, `N`), the
//! curl-curl post-processing, the boundary projections and the CFL.
//!
//! The port follows the **full-assembly, non-numerical-integration** path of
//! the C++ class (equivalent to the C++ `-no-pa -no-ni` configuration):
//! partial assembly (the C++ default) is not implemented in fem-rs, and the
//! `OperatorJacobiSmoother` / LOR-AMG preconditioners of the PA path have no
//! assembled analogue here.  The serial C++ reference run
//! (`tmp/navier_gen_serial_harness.py`) uses the same configuration and the
//! same preconditioners (`DSmoother` Jacobi for `Mv`/`H`, `GSSmoother`
//! wrapped in `OrthoSolver` for `Sp`), so every number produced here is
//! directly comparable with it.
//!
//! # Differences from the C++ full-assembly path
//!
//! * `HInvPC` is rebuilt in C++ only for partial assembly; the full-assembly
//!   path builds the Jacobi diagonal once in `Setup` and reuses it (stale
//!   after the BDF order increases).  Reproduced here on purpose: `h_diag` is
//!   snapshotted in [`NavierSolver::setup`].
//! * The `PrintInfo` banner omits the `MFEM version` / `MFEM GIT` lines.
//! * [`NavierConfig::convection_stabilization`] =
//!   [`ConvectionStabilization::DeferredUpwind`] is a fem-rs-only extension
//!   (D1125): the step's `H` gains `beta·D_up(u_lag)`, the lagged upwind
//!   defect.  The default `Off` is byte-identical to the C++ path.
//! * [`NavierConfig::pressure_mode`] =
//!   [`PressureMode::Incremental`] is a fem-rs-only extension (D952-H2): the
//!   van Kan / Guermond–Minev–Shen split sequencing (pressure as a state,
//!   increment Poisson, mass-form projection) replaces the fused TLO step.
//!   [`PressureMode::IncrementalSelfConsistent`] (D952-H2', round-135)
//!   repairs its measured instability with the projection-adjoint Poisson
//!   left end `−D·Mv⁻¹·G` (nested Krylov).  The default `Classical` is
//!   byte-identical to the C++ path.
//!
//! # The `D`/`G` pairing and its boundary term
//!
//! The split scheme uses *both* mixed operators, and they are not transposes
//! of one another.  With `D[i,(k,c)] = ∫ φᵢ ∂_c φ_k` and
//! `G[(k,c),i] = ∫ φ_k ∂_c φᵢ`,
//!
//! ```text
//! ∫_Ω φᵢ ∂_c φ_k = -∫_Ω φ_k ∂_c φᵢ + ∫_Γ φᵢ φ_k n_c ds
//!   ⇒  Gᵀ = -D + B ,   B[i,(k,c)] = ∫_Γ φᵢ φ_k n_c ds ,
//! ```
//!
//! and `B` is precisely the flux represented by the two boundary functionals
//! of the scheme: `resp = -D·FText + FText_bdr - (bd0/dt)·g_bdr`, where
//! `FText_bdr = ∫_Γ (FText·n) q` and `g_bdr = Σ ∫_Γ (u_D·n) q`.  Equivalently
//! the pair satisfies the divergence theorem `D·u + Gᵀ·u = ∫_Γ (u·n) φ`,
//! *not* `D = Gᵀ`.  Replacing either matrix by the transpose of the other —
//! or dropping `B` from the right-hand side — silently changes the pressure
//! Poisson problem; both must be assembled from their own MFEM integrator
//! (`VectorDivergenceIntegrator`, `GradientIntegrator`).
//!
//! # Kernel status (D46)
//!
//! The boundary functionals are available in `fem-assembly` as
//! `standard::VectorBoundaryNormalLFIntegrator`
//! (`∫_Γ (v(x)·n) φᵢ ds`, MFEM's `BoundaryNormalLFIntegrator(VectorCoefficient&)`)
//! assembled by `Assembler::assemble_boundary_linear` with the order-generic
//! face element of `assembler::ref_elem_face` and the face DOF list from
//! `assembler::face_dofs_h1`.  Note that fem-rs also exports a *different*
//! type of the same name (`standard::BoundaryNormalLFIntegrator`) computing
//! the scalar `∫_Γ g(x) φᵢ ds`.
//!
//! The miniapps still evaluate the `FText` functional with a small local face
//! loop: its coefficient is a *velocity grid function*, and evaluating it on a
//! face needs the owning element's DOF list, which the assembler's
//! [`BdQpData`](https://docs.rs/fem-assembly) payload does not carry (it only
//! exposes `phi`, not `elem_dofs`).  `g_bdr` — an analytic coefficient — is
//! assembled by the kernel.

use std::time::Instant;

use fem_linalg::CsrMatrix;

use crate::sli::{solve_cg_mfem, SliOptions};

/// MFEM `NAVIER_VERSION`.
pub const NAVIER_VERSION: &str = "0.1";

// ─── Solver configuration (navier_solver.hpp members) ────────────────────────

/// Convection treatment of the split scheme
/// ([`NavierConfig::convection_stabilization`]).
///
/// MFEM's `navier::NavierSolver` evaluates the nonlinear terms with the
/// centered Galerkin `VectorConvectionNLFIntegrator`
/// (navier_solver.cpp:124-126, `nlcoeff.constant = -1`) extrapolated
/// explicitly in time (EXTk, navier_solver.cpp:411-427).  That is the
/// [`ConvectionStabilization::Off`] default, and the solver stays
/// item-for-item identical to the C++ miniapp with it.
///
/// The remaining variant is a fem-rs extension (D1125) for cell-Re regimes
/// where the centered + explicit combination has a time-step stability
/// ceiling (round-109/110: re1000 @ dt=5e-3 diverges, cell-Re ~ 15.6).
///
/// # Why the upwind defect rides the implicit operator
///
/// The textbook deferred-correction blend puts `beta·(N_upwind − N_gal)` in
/// the *residual*; evaluated through this scheme's EXTk extrapolation it is
/// an explicit diffusion, and the exact BDF2/EXT2 characteristic
/// `1.5g² + (−2 + 2(iθ + x))g + (0.5 − (iθ + x)) = 0` with
/// `x = beta·ν_up·k²·dt` has `|g| > 1` at the grid scale for **every**
/// `beta > 0` (measured stability table in `tmp/d114nav/REPORT.md`; a seeded
/// n=16/dt=5e-2 cavity run diverged to 3.5e4 with beta = 0.3).  The blend
/// therefore adds the — lagged, linear, SPD — upwind defect to the
/// *implicit* Helmholtz operator instead:
/// `H = (bd0/dt)·Mv + ν·Kv + beta·D_up(u_lag)`.  That damps every mode
/// unconditionally, keeps the three-solve projection structure (no nonlinear
/// iteration: `u_lag` is the EXTk-extrapolated velocity the step already
/// forms — Picard-style lagged defect correction), and `beta → 1` is the
/// full upwind.  Everything else — residual, pressure solve, curl-curl,
/// boundary terms — stays on the MFEM path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvectionStabilization {
    /// Centered Galerkin convection, plain Helmholtz operator
    /// (MFEM-identical default).
    Off,
    /// Deferred-correction upwind blend (D1125): the Helmholtz operator of
    /// the step gains `beta * D_up(u_lag)` with
    ///
    /// ```text
    /// D_up(u)_{i,j} = ∫ ν_up ∇φ_j · ∇φ_i dx ,  ν_up = |u_h| · h_e / 2 ,
    /// ```
    ///
    /// the FE form of first-order upwinding (`beta → 0` recovers `Off` exactly,
    /// `beta = 1` is the full upwind).  `u_lag` is the extrapolated velocity
    /// `ab1·un + ab2·unm1 + ab3·unm2` of the current step, so the added
    /// operator is linear and lagged.
    DeferredUpwind {
        /// Blend factor `beta` (0 = Off-equivalent, 1 = full upwind).
        beta: f64,
    },
}

/// Pressure formulation family of the split step
/// ([`NavierConfig::pressure_mode`], D952-H2).
///
/// [`PressureMode::Classical`] is the MFEM-identical non-incremental scheme:
/// every step solves the pressure Poisson system for the ABSOLUTE pressure
/// from scratch (the previous `pn` only warm-starts the CG; the right-hand
/// side never consumes it as state), which leaves an O(dt) — in the viscous
/// coupling O(ν·dt) — splitting error in the pressure that regenerates each
/// step and caps the transient velocity order (measured on the decay rig,
/// round-132/133: coarse-pair order 1.59; ν-scan 1.59@ν=0.05 → 1.71@ν=0.005).
///
/// [`PressureMode::Incremental`] is the incremental pressure re-assembly of
/// the van Kan / Guermond–Minev–Shen pressure-correction family (J. van Kan,
/// SIAM J. Sci. Stat. Comput. 7(3), 1986; Guermond, Minev & Shen, SIAM J.
/// Sci. Comput. 28(2), 2006, §2.2/§3.1): the pressure becomes a STATE — the
/// step solves the Poisson system for the O(dt) INCREMENT `δp` and
/// re-assembles `p^{n+1} = p_ext + δp` with `p_ext = ab1·pⁿ + ab2·p^{n−1} +
/// ab3·p^{n−2}` the EXTk pressure extrapolation (the same coefficients the
/// velocity extrapolation uses, so BDF1 bootstraps with `p_ext = pⁿ`).  The
/// increment right-hand side substitutes the pressure-EXTRAPOLATED split
/// velocity into the classical data functional,
/// `resp_incr = resp − (bd0/dt)·Gᵀ·H⁻¹·(G·p_ext)`, i.e. the textbook
/// `-Δp_ext` increment term carried through the dissipative Helmholtz
/// inverse (a raw constraint-operator term bypassing `H⁻¹` was measured to
/// blow up ~1.5×/step — round-134 variant scan).  Because the splitting
/// defect of the mass-matrix surrogate now multiplies the O(dt) increment
/// instead of the O(1) absolute pressure, the O(ν·dt) term no longer
/// regenerates per step — this is the lever the round-133 Timmermans patch
/// (which only reshaped the RHS data) measurably could not supply.
///
/// The [`NavierConfig::rotational`] flag stays orthogonal: with both active
/// the step realizes the rotational-incremental family (`p^{n+1} = p_ext +
/// δp − ν·Π(∇·ũ)`, GMS 2006 §3.2).
///
/// [`PressureMode::IncrementalSelfConsistent`] (D952-H2', round-135) is the
/// self-consistent repair of the round-134 finding: the raw increment
/// Poisson (assembled `Sp` left end) is not the exact discrete adjoint of
/// the mass-form projection under consistent mass, and the measured
/// instability of the `Incremental` split sequencing traces to that
/// mismatch compounded with the projection composition (see
/// [`NavierConfig::increment_projection_absolute`]).  The self-consistent
/// mode keeps the exact same split sequencing and the exact same increment
/// data functional `b1 = (bd0/dt)·(Gᵀ·u* − g_bdr)` but solves the
/// CONSTRAINT equation with the exact adjoint of the projection as the
/// left end,
///
/// ```text
///   (−D·Mv⁻¹·G)·δp = b1                                   (increment projection)
///   (−D·Mv⁻¹·G)·δp = b1 + (D·Mv⁻¹·G)·p_ext                (absolute projection)
/// ```
///
/// so the end-of-step velocity `u⁺ = u* − (dt/bd0)·Mv⁻¹·G·target` is
/// discretely divergence-free to the outer CG tolerance BY CONSTRUCTION
/// (the O(1) divergence residue of the raw-`Sp` form — the explicit-
/// convection feedback channel — vanishes identically).  The operator
/// `−Q`, `Q = D·Mv⁻¹·G`, is the negative semidefinite bulk adjoint
/// (`−Q = Gᵀ·Mv⁻¹·G` up to the boundary-flux term, i.e. the mass-form
/// mirror of `Sp`); applying it costs one inner `Mv⁻¹` CG per outer
/// iteration (a nested Krylov operator — see
/// [`NavierConfig::sc_inner_rtol`] for the inner tolerance contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureMode {
    /// MFEM-identical classical non-incremental path (the default — every
    /// step of the default configuration is bit-identical to the pre-D952-H2
    /// kernel).
    Classical,
    /// Incremental pressure re-assembly (pressure as state; see the type
    /// docs).  Provisional steps are rejected loudly (they overwrite `pn`
    /// without rotating the history, which would corrupt the state).
    Incremental,
    /// D952-H2' self-consistent incremental re-assembly: the `Incremental`
    /// split sequencing with the projection-adjoint Poisson left end
    /// `−D·Mv⁻¹·G` (nested Krylov; see the type docs).  Provisional steps
    /// are rejected loudly, exactly like [`PressureMode::Incremental`].
    IncrementalSelfConsistent,
}

/// Relative tolerances / iteration caps / print levels of the three solves.
///
/// Mirrors the double-precision `NavierSolver` member defaults of
/// `navier_solver.hpp` and the hard-coded `SetMaxIter(200)` in `Setup`.
#[derive(Debug, Clone)]
pub struct NavierConfig {
    /// `max_bdf_order` (MFEM default 3).
    pub max_bdf_order: i32,
    /// `rtol_mvsolve` (1e-12).
    pub rtol_mvsolve: f64,
    /// `rtol_spsolve` (1e-6).
    pub rtol_spsolve: f64,
    /// `rtol_hsolve` (1e-8).
    pub rtol_hsolve: f64,
    /// `pl_mvsolve`.
    pub pl_mvsolve: i32,
    /// `pl_spsolve`.
    pub pl_spsolve: i32,
    /// `pl_hsolve`.
    pub pl_hsolve: i32,
    /// `verbose` (the `Setup` banner, the per-step iteration table and
    /// [`NavierSolver::print_info`]).
    pub verbose: bool,
    /// Use the AMG pressure preconditioner — the analogue of MFEM's
    /// `SpInvPC = HypreBoomerAMG` (navier_solver.cpp:267-282) for builds with
    /// `MFEM_USE_HYPRE`, wrapped in `OrthoSolver` exactly like the C++
    /// `SpInvOrthoPC` when `pres_dbcs` is empty.
    ///
    /// `false` (the default) keeps the no-hypre serial fallback
    /// `OrthoSolver(GSSmoother)`, which is the gear the recorded serial-mirror
    /// verifications of the navier miniapps were produced with (the smoother
    /// cannot converge the 26k-dof pure-Neumann pressure within the 200
    /// iteration cap — MFEM 4.10's own serial mirror stagnates identically).
    pub pressure_amg: bool,
    /// Convection treatment ([`ConvectionStabilization::Off`] = the
    /// MFEM-identical default; D1125 for the stabilized variants).
    pub convection_stabilization: ConvectionStabilization,
    /// D952-H2 pressure formulation family ([`PressureMode`]).  The default
    /// [`PressureMode::Classical`] keeps every step bit-identical to the
    /// MFEM path.
    pub pressure_mode: PressureMode,
    /// D952-H2/H2' projection composition of the incremental families
    /// (`pressure_mode != Classical`): which pressure the step's mass-form
    /// projection subtracts,
    ///
    /// ```text
    ///   true  (default): u⁺ = u* − (dt/bd0)·Mv⁻¹·G·p^{n+1}   (round-134 form)
    ///   false          : u⁺ = u* − (dt/bd0)·Mv⁻¹·G·δp        (GMS 2006 §2.2)
    /// ```
    ///
    /// The default keeps every `Incremental` step bit-identical to the
    /// round-134 kernel.  `false` realizes the textbook van Kan /
    /// Guermond–Minev–Shen increment-only projection; with
    /// [`PressureMode::IncrementalSelfConsistent`] the two values isolate
    /// the projection-composition factor of the round-134 instability
    /// experimentally (the {raw `Sp`, `−Q`} × {absolute, increment} 2×2
    /// ladder of the round-135 design).  Incompatible with `rotational`
    /// (the Timmermans correction edits the ABSOLUTE pressure state — an
    /// increment-only projection would silently drop it): that combination
    /// aborts loudly at the step entry.
    pub increment_projection_absolute: bool,
    /// D952-H2' inner `Mv⁻¹` CG relative tolerance of the nested
    /// self-consistent operator (`pressure_mode ==
    /// IncrementalSelfConsistent`).  The divergence constraint does NOT
    /// depend on this tolerance — the outer CG and the projection apply the
    /// IDENTICAL inexact operator, so `D·u⁺ = O(outer rtol)` regardless;
    /// the inner tolerance only trades Poisson consistency vs cost (one
    /// inner `Mv⁻¹` solve per outer iteration).  Default `1e-8`.
    pub sc_inner_rtol: f64,
    /// D952 Timmermans **rotational pressure correction** (fem-rs extension,
    /// NOT an MFEM feature — the C++ `NavierSolver` is the classical
    /// non-rotational scheme).  `false` (the default) keeps every step
    /// bit-identical to the MFEM path.  When `true`, the step inserts one
    /// extra Helmholtz pre-solve that realizes the Timmermans, Minev & Van
    /// De Vosse (1995) sequencing: the PURE VISCOUS split velocity `ũ` (the
    /// momentum BDF step without the pressure drive) supplies the rotational
    /// term
    ///
    /// ```text
    ///   p^{n+1} += −ν·Π(∇·ũ^{n+1}) ,
    /// ```
    ///
    /// with `Π` the discretization's pressure-space representation of the
    /// divergence ([`NavierDiscretization::rotational_divergence_pressure`]);
    /// the projection solve then consumes the CORRECTED pressure, so the
    /// term enters the dynamics (a post-step pressure patch alone would be
    /// dynamics-dead, because the next step's pressure Poisson solve never
    /// consumes `pn` as state).  Essential pressure DOFs (Dirichlet data)
    /// are skipped, and on the pure-Neumann path `MeanZero` is re-applied,
    /// so both pressure gauge conventions stay those of the flag-off steps.
    pub rotational: bool,
}

impl Default for NavierConfig {
    fn default() -> Self {
        NavierConfig {
            max_bdf_order: 3,
            rtol_mvsolve: 1e-12,
            rtol_spsolve: 1e-6,
            rtol_hsolve: 1e-8,
            pl_mvsolve: 0,
            pl_spsolve: 0,
            pl_hsolve: 0,
            verbose: true,
            pressure_amg: false,
            convection_stabilization: ConvectionStabilization::Off,
            pressure_mode: PressureMode::Classical,
            increment_projection_absolute: true,
            sc_inner_rtol: 1.0e-8,
            rotational: false,
        }
    }
}

// ─── Discretization interface ────────────────────────────────────────────────

/// The forms, boundary data and post-processing the split scheme needs.
///
/// Every method corresponds to one `ParBilinearForm` / `ParLinearForm` /
/// post-processing call of the C++ `NavierSolver`; the miniapp implements them
/// with `fem-assembly` on the equal-order `[H¹]^d × H¹` spaces.
///
/// Matrices are indexed by (true-)DOF; for a serial conforming space the true
/// DOFs are the space DOFs.  Velocity DOFs use the `[H¹]^d` block layout of
/// `fem_space::VectorH1Space` (component `c`, scalar dof `s` →
/// `c * n_scalar + s`), matching MFEM's `Ordering::byNODES` global layout.
pub trait NavierDiscretization {
    /// `vfes->GetVSize()` (velocity true DOFs).
    fn n_vel(&self) -> usize;
    /// `pfes->GetVSize()` (pressure true DOFs).
    fn n_pres(&self) -> usize;
    /// `vel_ess_tdof` — essential velocity DOFs (all Dirichlet boundaries).
    fn vel_ess_dofs(&self) -> &[usize];
    /// `pres_ess_tdof` — essential pressure DOFs (empty for pure Neumann).
    fn pres_ess_dofs(&self) -> &[usize];

    /// `Mv_form` — `∫ u·v dx` in `[H¹]^d`.
    fn assemble_mass_velocity(&self) -> CsrMatrix<f64>;
    /// `Sp_form` — `∫ ∇p·∇q dx` in `H¹`.
    fn assemble_pressure_laplace(&self) -> CsrMatrix<f64>;
    /// `D_form` — `D[i,(k,c)] = ∫ φ_i ∂φ_k/∂x_c dx` (MFEM
    /// `VectorDivergenceIntegrator`, `n_pres × n_vel`).
    ///
    /// **`D` is not the exact transpose of [`Self::assemble_gradient`]**, even
    /// though both discretize the same pair of spaces.  Element-wise
    /// integration by parts gives
    ///
    /// ```text
    /// ∫_Ω φ_i ∂_c φ_k dx = -∫_Ω φ_k ∂_c φ_i dx + ∫_Γ φ_i φ_k n_c ds ,
    /// ```
    ///
    /// so `D` and `Gᵀ` differ by the boundary term `B[i,(k,c)] = ∫_Γ φ_i φ_k n_c ds`
    /// — the term the split scheme carries explicitly through the
    /// `FText_bdr`/`g_bdr` functionals.  Both matrices must therefore be
    /// assembled independently, and the identity to test against is the
    /// *divergence theorem*
    ///
    /// ```text
    /// D·u + Gᵀ·u = ∫_Γ (u·n) φ  —  i.e. -∫∇p·u + ∫ p ∇·u = ∫_Γ p (u·n) ,
    /// ```
    ///
    /// not `D = Gᵀ` (`navier_kovasznay`'s `divergence_theorem_identity`).
    /// Filling in `G = Dᵀ` silently drops the boundary flux and changes the
    /// pressure Poisson right-hand side.
    fn assemble_divergence(&self) -> CsrMatrix<f64>;
    /// `G_form` — `G[(k,c),i] = ∫ φ_k ∂φ_i/∂x_c dx` (MFEM
    /// `GradientIntegrator`, `n_vel × n_pres`).
    ///
    /// See [`Self::assemble_divergence`]: `G ≠ Dᵀ`; the two differ by the
    /// boundary term `∫_Γ φ_k φ_i n_c ds`, which is exactly the flux the
    /// `FText_bdr`/`g_bdr` boundary functionals carry.  In the split-scheme
    /// algebra only `G·p` is used (`resu = Mv·Fext - G·p`), and it must be the
    /// `GradientIntegrator` matrix, not `Dᵀ`.
    fn assemble_gradient(&self) -> CsrMatrix<f64>;
    /// `H_form` — `mass_coeff·Mv + visc_coeff·K_v` with `K_v` the vector
    /// Laplacian (`VectorDiffusionIntegrator`).  The C++ code reassembles
    /// this form with the current `bd0/dt` on every step.
    fn assemble_helmholtz(&self, mass_coeff: f64, visc_coeff: f64) -> CsrMatrix<f64>;
    /// `N->Mult(u, Nu)` — the convection residual
    /// `Nu_i = ∫ (u·∇u)·φ_i dx`, i.e. MFEM's `VectorConvectionNLFIntegrator`
    /// evaluated with `Q = 1` (the driver applies the `nlcoeff = -1` factor of
    /// the C++ form).
    fn convection_residual(&self, u: &[f64], out: &mut [f64]);

    /// The upwind-defect operator of
    /// [`ConvectionStabilization::DeferredUpwind`]
    /// ([`NavierConfig::convection_stabilization`]): the SPD block
    ///
    /// ```text
    /// D_up(u)_{i,j} = ∫ ν_up ∇φ_j · ∇φ_i dx ,  ν_up = |u_h| · h_e / 2 ,
    /// ```
    ///
    /// evaluated with the lagged velocity `u` (the step's EXTk-extrapolated
    /// velocity).  Never called with the default
    /// [`ConvectionStabilization::Off`]; the default implementation aborts
    /// loudly so a discretization that does not implement the stabilized
    /// operator cannot silently run unstabilized.
    fn assemble_upwind_defect(&self, u: &[f64]) -> CsrMatrix<f64> {
        let _ = u;
        panic!(
            "ConvectionStabilization::DeferredUpwind requires the discretization \
             to implement `assemble_upwind_defect`"
        );
    }

    /// `∇×∇×u` at the velocity DOFs, i.e. the C++ `ComputeCurl2D` applied
    /// twice (without the `kin_vis` factor, which the driver applies).
    /// In 3-D (`navier_tgv`) this is `ComputeCurl3D` applied twice, matching
    /// the `dim == 3` branch of `NavierSolver::Step`.
    fn curl_curl(&self, u: &[f64]) -> Vec<f64>;

    /// `ComputeCurl2D(Lext_gf, curlu_gf)` followed by
    /// `ComputeCurl2D(curlu_gf, curlcurlu_gf, true)` (the `dim == 3` branch
    /// calls `ComputeCurl3D` twice) — the two curls `NavierSolver::Step`
    /// computes on the extrapolated velocity
    /// `Lext = ab1·un + ab2·unm1 + ab3·unm2`, returned as
    /// `(curlu_gf, curlcurlu_gf)`.
    ///
    /// The first of the two — MFEM's `GetCurrentVorticity()` — is what the
    /// `navier_bifurcation` particle solver interpolates onto the particles
    /// (`GridFunction &w_gf = *flow_solver.GetCurrentVorticity()`), so it has
    /// to come out of the *step*, not from a re-run of the curl on the
    /// accepted velocity.
    ///
    /// The default implementation delegates to [`Self::curl_curl`] and reports
    /// an empty vorticity: the four miniapps that never call
    /// `GetCurrentVorticity()` (`navier_mms`, `navier_kovasznay`,
    /// `navier_kovasznay_vs`, `navier_shear`) then leave
    /// [`NavierSolver::vorticity`] empty, exactly as before.
    fn curl_curl_and_vorticity(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (Vec::new(), self.curl_curl(u))
    }
    /// `ComputeCurl3D(u, cu)` — one application of MFEM's 3-D curl.
    ///
    /// Like `ComputeCurl2D` this is **not** a DG weak form but a nodal point
    /// evaluation: for every element, the curl of the interpolation of `u`
    /// (`grad = (loc_dataᵀ·dshape)·J⁻¹` evaluated at the element's nodal
    /// points with the element's *own* per-element periodic geometry) is
    /// accumulated into the shared DOFs and divided by the zone count
    /// (`cu(ldof) += vals[j]; zones_per_vdof[ldof]++; … cu /= nz` — the
    /// serial form of the `GroupCommunicator` reduce/bcast pair).
    ///
    /// Only 3-D discretizations implement it; the default aborts, as calling
    /// it on a 2-D space is a programming error (the C++ class compiles the
    /// 2-D/3-D choice out of `Step` via `pmesh->Dimension()`).
    fn compute_curl_3d(&self, _u: &[f64]) -> Vec<f64> {
        unimplemented!("ComputeCurl3D requires a 3-D discretization");
    }
    /// The pressure-space representation of `∇·u` consumed by the D952
    /// rotational pressure correction ([`NavierConfig::rotational`]): given
    /// the end-of-step velocity `u`, return `n_pres()` values `δp` with
    /// `δp ≈ ∇·u` **in pressure DOF space**.  The discretization owns both
    /// the divergence convention (which weak-divergence form / matrix) and
    /// the projection choice (e.g. the L² projection `M_p⁻¹·(D·u)`); the
    /// kernel only scales by `−ν` and adds the result to the step pressure
    /// (skipping the essential pressure DOFs, re-applying `MeanZero` on the
    /// pure-Neumann path).
    ///
    /// Never called with [`NavierConfig::rotational`] = `false`; the default
    /// aborts loudly so a discretization that does not implement the
    /// rotational-correction representation cannot silently run a no-op
    /// (the [`Self::assemble_upwind_defect`] contract pattern).
    fn rotational_divergence_pressure(&self, _u: &[f64]) -> Vec<f64> {
        panic!(
            "NavierConfig::rotational requires the discretization to implement \
             `rotational_divergence_pressure`"
        );
    }
    /// `un_next_gf.ProjectBdrCoefficient(coeff, attr)` — overwrite the
    /// velocity Dirichlet DOFs of `out` with the data at time `t` (interior
    /// entries untouched).
    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]);
    /// `pn_gf.ProjectBdrCoefficient(coeff, attr)` — pressure Dirichlet data.
    ///
    /// Only called when [`Self::pres_ess_dofs`] is non-empty.
    fn project_pressure_bdr(&self, t: f64, out: &mut [f64]) {
        let _ = (t, out);
    }
    /// `FText_bdr_form` — `∫_Γ (FText·n) φ ds` on the velocity Dirichlet
    /// boundaries (`BoundaryNormalLFIntegrator` with the `FText` GF).
    fn assemble_ftext_bdr(&self, ftext: &[f64]) -> Vec<f64>;
    /// `g_bdr_form` — `Σ ∫_Γ (u_D·n) φ ds` over the velocity Dirichlet BCs.
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64>;
    /// `f_form` — the acceleration (`f^{n+1}`) body force at time `t`
    /// (MFEM `VectorDomainLFIntegrator`); zero when no accel term was added.
    fn assemble_accel(&self, t: f64) -> Vec<f64> {
        let _ = t;
        vec![0.0; self.n_vel()]
    }
    /// `MeanZero(v)` — subtract the mass-weighted mean `∫v dx / vol(Ω)`.
    fn mean_zero(&self, v: &mut [f64]);
    /// `ComputeCFL(u, dt)`.
    fn compute_cfl(&self, u: &[f64], dt: f64) -> f64;
    /// `FormSystemMatrix` (matrix part) + `FormLinearSystem` (RHS part) with
    /// MFEM's default `DIAG_KEEP` policy: zero the off-diagonal entries of the
    /// essential rows/columns, keep the diagonal, and
    /// `rhs[other] -= A[other,rc]·x[rc]`, `rhs[rc] = A[rc,rc]·x[rc]`.
    ///
    /// `ess` is [`Self::vel_ess_dofs`] or [`Self::pres_ess_dofs`], and
    /// `values[k]` is the current solution entry of `ess[k]`.
    fn eliminate_bc(
        &self,
        mat: &mut CsrMatrix<f64>,
        rhs: &mut [f64],
        ess: &[usize],
        values: &[f64],
    );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Diagonal of `a` (MFEM `SparseMatrix::GetDiag`).
fn diagonal(a: &CsrMatrix<f64>) -> Vec<f64> {
    let n = a.nrows;
    let mut d = vec![0.0_f64; n];
    for row in 0..n {
        for k in a.row_ptr[row]..a.row_ptr[row + 1] {
            if a.col_idx[k] as usize == row {
                d[row] = a.values[k];
            }
        }
    }
    d
}

/// MFEM `SparseMatrix::Gauss_Seidel_forw`: a forward sweep updating `y` in
/// place, reading the *current* `y` entries for the not-yet-updated rows.
///
/// MFEM's `GSSmoother::Mult` zeroes `y` first unless `iterative_mode` is set;
/// `OrthoSolver` propagates *its own* `iterative_mode`, which stays `false`
/// (`OrthoSolver() : Solver(0, false)`), so the sweeps of the pressure
/// preconditioner always start from `y = 0`.
fn gs_forward(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows {
        let mut sum = 0.0;
        let mut diag = None;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[k] as usize;
            if c == i {
                diag = Some(a.values[k]);
            } else {
                sum += a.values[k] * y[c];
            }
        }
        match diag {
            Some(d) if d != 0.0 => y[i] = (x[i] - sum) / d,
            _ if x[i] == sum => y[i] = sum,
            _ => panic!("SparseMatrix::Gauss_Seidel_forw: zero diagonal at row {i}"),
        }
    }
}

/// MFEM `SparseMatrix::Gauss_Seidel_back` (same in-place semantics).
fn gs_backward(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in (0..a.nrows).rev() {
        let mut sum = 0.0;
        let mut diag = None;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[k] as usize;
            if c == i {
                diag = Some(a.values[k]);
            } else {
                sum += a.values[k] * y[c];
            }
        }
        match diag {
            Some(d) if d != 0.0 => y[i] = (x[i] - sum) / d,
            _ if x[i] == sum => y[i] = sum,
            _ => panic!("SparseMatrix::Gauss_Seidel_back: zero diagonal at row {i}"),
        }
    }
}

/// MFEM `OrthoSolver::Orthogonalize` — arithmetic mean over the true DOFs.
fn orthogonalize(v: &mut [f64]) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

/// Format `v` like C++ `std::scientific` with `prec` decimals (MFEM's
/// `mfem::out`) or like `printf("%.*E")` in upper-case mode — the exponent
/// always carries a sign and at least two digits.
pub fn fmt_sci(v: f64, prec: usize, upper: bool) -> String {
    let s = format!("{v:.prec$e}", prec = prec);
    let (mant, exp) = s.split_once('e').expect("fmt_sci: no exponent");
    let e: i32 = exp.parse().expect("fmt_sci: bad exponent");
    format!(
        "{}{}{}{:02}",
        mant,
        if upper { 'E' } else { 'e' },
        if e < 0 { '-' } else { '+' },
        e.abs()
    )
}

// ─── The solver ─────────────────────────────────────────────────────────────

/// Velocity-space state vectors (MFEM `un`, `un_next`, `unm1`, `unm2`).
#[derive(Default, Clone)]
struct VelState {
    un: Vec<f64>,
    un_next: Vec<f64>,
    unm1: Vec<f64>,
    unm2: Vec<f64>,
}

impl VelState {
    fn new(n: usize) -> Self {
        VelState {
            un: vec![0.0; n],
            un_next: vec![0.0; n],
            unm1: vec![0.0; n],
            unm2: vec![0.0; n],
        }
    }
}

/// MFEM `navier::NavierSolver` (split-scheme transient incompressible NS).
pub struct NavierSolver<D: NavierDiscretization> {
    disc: D,
    cfg: NavierConfig,
    /// Kinematic viscosity (dimensionless).
    kin_vis: f64,

    // `OperatorHandle`s assembled once by `Setup`.
    mv: CsrMatrix<f64>,
    sp: CsrMatrix<f64>,
    d: CsrMatrix<f64>,
    g: CsrMatrix<f64>,
    /// `MvInvPC` / `HInvPC` (`DSmoother`-equivalent Jacobi diagonals),
    /// snapshotted in `Setup` exactly like the C++ full-assembly path.
    mv_diag: Vec<f64>,
    h_diag: Vec<f64>,
    /// `SpInvPC` — the [`crate::amg::AmgSolver`] analogue of MFEM's
    /// `HypreBoomerAMG` pressure preconditioner (navier_solver.cpp:270/278),
    /// built in `Setup` when [`NavierConfig::pressure_amg`] is set.
    sp_amg: Option<crate::amg::AmgSolver<f64>>,

    vel: VelState,
    pn: Vec<f64>,
    /// Pressure history of [`PressureMode::Incremental`] (D952-H2):
    /// `pnm1 = pⁿ`, `pnm2 = p^{n−1}` at the time `step` runs (invariant after
    /// each accepted step: `pn = p^{n+1}`).  The classical path never reads
    /// them; the shift happens inside the incremental branch BEFORE the solve
    /// overwrites `pn` (and never on provisional steps — rejected loudly).
    pnm1: Vec<f64>,
    pnm2: Vec<f64>,
    /// D952-H2' bookkeeping: total inner `Mv⁻¹` CG iterations spent by the
    /// nested self-consistent Poisson operator of the last step (outer
    /// iterations × 1, plus the constant-data applies — `Q·p_ext` of the
    /// absolute-projection arm and the essential reactions).  `0` unless
    /// `pressure_mode == IncrementalSelfConsistent`.
    sc_inner_iters: i32,
    /// `curlu_gf` — `∇×Lext` of the last step
    /// ([`NavierDiscretization::curl_curl_and_vorticity`]); empty when the
    /// discretization does not implement the vorticity output.
    curlu: Vec<f64>,

    // BDFk/EXTk bookkeeping.
    cur_step: i32,
    dthist: [f64; 3],
    bd0: f64,
    bd1: f64,
    bd2: f64,
    bd3: f64,
    ab1: f64,
    ab2: f64,
    ab3: f64,

    // Iteration counts / residuals of the last step.
    iter_mvsolve: i32,
    iter_spsolve: i32,
    iter_hsolve: i32,
    res_mvsolve: f64,
    res_spsolve: f64,
    res_hsolve: f64,

    // Timers: `sw_setup`, `sw_step`, `sw_extrap`, `sw_curlcurl`, `sw_spsolve`,
    // `sw_hsolve` in seconds.
    rt_setup: f64,
    rt_step: f64,
    rt_extrap: f64,
    rt_curlcurl: f64,
    rt_spsolve: f64,
    rt_hsolve: f64,
}

impl<D: NavierDiscretization> NavierSolver<D> {
    /// MFEM `NavierSolver::NavierSolver(mesh, order, kin_vis)`: allocate the
    /// state vectors and print the version banner.  `disc` plays the role of
    /// the `(mesh, order)` pair, `kin_vis` is the kinematic viscosity.
    pub fn new(disc: D, kin_vis: f64, cfg: NavierConfig) -> Self {
        let nv = disc.n_vel();
        let np = disc.n_pres();
        let solver = NavierSolver {
            disc,
            cfg,
            kin_vis,
            mv: CsrMatrix::new_empty(0, 0),
            sp: CsrMatrix::new_empty(0, 0),
            d: CsrMatrix::new_empty(0, 0),
            g: CsrMatrix::new_empty(0, 0),
            mv_diag: Vec::new(),
            h_diag: Vec::new(),
            sp_amg: None,
            vel: VelState::new(nv),
            pn: vec![0.0; np],
            pnm1: vec![0.0; np],
            pnm2: vec![0.0; np],
            sc_inner_iters: 0,
            curlu: Vec::new(),
            cur_step: 0,
            dthist: [0.0; 3],
            bd0: 0.0,
            bd1: 0.0,
            bd2: 0.0,
            bd3: 0.0,
            ab1: 0.0,
            ab2: 0.0,
            ab3: 0.0,
            iter_mvsolve: 0,
            iter_spsolve: 0,
            iter_hsolve: 0,
            res_mvsolve: 0.0,
            res_spsolve: 0.0,
            res_hsolve: 0.0,
            rt_setup: 0.0,
            rt_step: 0.0,
            rt_extrap: 0.0,
            rt_curlcurl: 0.0,
            rt_spsolve: 0.0,
            rt_hsolve: 0.0,
        };
        if solver.cfg.verbose {
            solver.print_info();
        }
        solver
    }

    /// MFEM `NavierSolver::PrintInfo`.
    pub fn print_info(&self) {
        println!("NAVIER version: {NAVIER_VERSION}");
        println!("Velocity #DOFs: {}", self.disc.n_vel());
        println!("Pressure #DOFs: {}", self.disc.n_pres());
    }

    /// MFEM `NavierSolver::Setup(dt)`: assemble the time-independent
    /// operators, build the preconditioners and initialise the history.
    pub fn setup(&mut self, dt: f64) {
        let t_start = Instant::now();
        if self.cfg.verbose {
            println!("Setup");
            println!("Using Full Assembly");
        }

        self.mv = self.disc.assemble_mass_velocity();
        self.sp = self.disc.assemble_pressure_laplace();
        self.d = self.disc.assemble_divergence();
        self.g = self.disc.assemble_gradient();

        // `SpInvPC`: the BoomerAMG analogue, built once for the constant `Sp`
        // (MFEM builds `HypreBoomerAMG` on the LOR/assembled matrix in Setup).
        if self.cfg.pressure_amg {
            self.sp_amg = Some(crate::amg::AmgSolver::setup(
                &self.sp,
                crate::amg::boomeramg_config(),
            ));
        }

        // `MvInvPC = DSmoother(Mv)` and `HInvPC = DSmoother(H)` with the
        // initial `H_bdfcoeff = 1/dt`; both are built once, in Setup.
        self.mv_diag = diagonal(&self.mv);
        let h = self.disc.assemble_helmholtz(1.0 / dt, self.kin_vis);
        self.h_diag = diagonal(&h);

        // "If the initial condition was set, it has to be aligned with
        // dependent Vectors and GridFunctions": un_next = un.
        self.vel.un_next.copy_from_slice(&self.vel.un);

        // Set initial time step in the history array.
        self.dthist = [dt, 0.0, 0.0];

        self.rt_setup = t_start.elapsed().as_secs_f64();
    }

    /// MFEM `NavierSolver::UpdateTimestepHistory(dt)`.
    pub fn update_timestep_history(&mut self, dt: f64) {
        self.dthist[2] = self.dthist[1];
        self.dthist[1] = self.dthist[0];
        self.dthist[0] = dt;

        // The nonlinear extrapolation history (Nun, Nunm1, Nunm2) is rotated
        // in C++ as well, but every entry is recomputed at the beginning of
        // each `Step`, so it carries no state.
        self.vel.unm2.copy_from_slice(&self.vel.unm1);
        self.vel.unm1.copy_from_slice(&self.vel.un);

        // un_next_gf.GetTrueDofs(un_next); un = un_next; un_gf = un.
        self.vel.un.copy_from_slice(&self.vel.un_next);
    }

    /// MFEM `NavierSolver::SetTimeIntegrationCoefficients(step)`.
    pub fn set_time_integration_coefficients(&mut self, step: i32) {
        // Maximum BDF order to use at current time step:
        // step + 1 <= order <= max_bdf_order.
        let bdf_order = (step + 1).min(self.cfg.max_bdf_order);

        // Ratio of time step history dt(t_n)/dt(t_{n-1}) and
        // dt(t_{n-1})/dt(t_{n-2}).
        let rho1 = self.dthist[0] / self.dthist[1];
        let rho2 = if bdf_order == 3 {
            self.dthist[1] / self.dthist[2]
        } else {
            0.0
        };

        if step == 0 && bdf_order == 1 {
            self.bd0 = 1.0;
            self.bd1 = -1.0;
            self.bd2 = 0.0;
            self.bd3 = 0.0;
            self.ab1 = 1.0;
            self.ab2 = 0.0;
            self.ab3 = 0.0;
        } else if step >= 1 && bdf_order == 2 {
            self.bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
            self.bd1 = -(1.0 + rho1);
            self.bd2 = rho1.powi(2) / (1.0 + rho1);
            self.bd3 = 0.0;
            self.ab1 = 1.0 + rho1;
            self.ab2 = -rho1;
            self.ab3 = 0.0;
        } else if step >= 2 && bdf_order == 3 {
            self.bd0 = 1.0 + rho1 / (1.0 + rho1) + (rho2 * rho1) / (1.0 + rho2 * (1.0 + rho1));
            self.bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
            self.bd2 = rho1.powi(2) * (rho2 + 1.0 / (1.0 + rho1));
            self.bd3 = -(rho2.powi(3) * rho1.powi(2) * (1.0 + rho1))
                / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
            self.ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
            self.ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
            self.ab3 = (rho2.powi(2) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
        }
    }

    /// MFEM `NavierSolver::Orthogonalize(Vector&)`.
    pub fn orthogonalize_vec(&self, v: &mut [f64]) {
        orthogonalize(v);
    }

    /// MFEM `NavierSolver::ComputeCFL(u, dt)`.
    pub fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
        self.disc.compute_cfl(u, dt)
    }

    /// `GetCurrentVelocity()`.
    pub fn velocity(&self) -> &[f64] {
        &self.vel.un
    }
    /// `GetProvisionalVelocity()`.
    pub fn provisional_velocity(&self) -> &[f64] {
        &self.vel.un_next
    }
    /// `GetCurrentPressure()`.
    pub fn pressure(&self) -> &[f64] {
        &self.pn
    }
    /// `GetCurrentVorticity()` — `∇×Lext` of the last [`Self::step`], empty
    /// unless the discretization implements
    /// [`NavierDiscretization::curl_curl_and_vorticity`] with a non-empty
    /// vorticity (only `navier_bifurcation`, which feeds it to its particle
    /// solver, does).
    pub fn vorticity(&self) -> &[f64] {
        &self.curlu
    }
    /// Mutable current velocity — the initial condition
    /// (`GetCurrentVelocity()->ProjectCoefficient`).
    pub fn velocity_mut(&mut self) -> &mut [f64] {
        &mut self.vel.un
    }
    /// The discretization (the C++ `vfes`/`pfes` owner), for post-processing
    /// that lives outside the solver (L² errors, exact-solution projections).
    pub fn discretization(&self) -> &D {
        &self.disc
    }
    /// `SetMaxBDFOrder`.
    pub fn set_max_bdf_order(&mut self, order: i32) {
        self.cfg.max_bdf_order = order;
    }
    /// The public `dthist` member of the C++ class: the time-step history
    /// `[dt(tₙ), dt(tₙ₋₁), dt(tₙ₋₂)]` that the variable-time-step miniapps
    /// (`navier_kovasznay_vs`) queue through
    /// [`Self::update_timestep_history`].
    pub fn dthist(&self) -> [f64; 3] {
        self.dthist
    }
    /// `MvInv->GetNumIterations()` of the last step.
    pub fn iter_mvsolve(&self) -> i32 {
        self.iter_mvsolve
    }
    /// `SpInv->GetNumIterations()` of the last step.
    pub fn iter_spsolve(&self) -> i32 {
        self.iter_spsolve
    }
    /// Total inner `Mv⁻¹` CG iterations of the nested self-consistent
    /// operator during the last step (`pressure_mode ==
    /// IncrementalSelfConsistent`): one inner solve per outer Poisson
    /// iteration plus the constant-data applies (`Q·p_ext` of the
    /// absolute-projection arm, the essential reactions).  `0` otherwise.
    /// This is the per-step cost KPI of the D952-H2' nested Krylov.
    pub fn iter_sc_inner(&self) -> i32 {
        self.sc_inner_iters
    }
    /// `HInv->GetNumIterations()` of the last step.
    pub fn iter_hsolve(&self) -> i32 {
        self.iter_hsolve
    }
    /// `MvInv->GetFinalNorm()` of the last step: the final preconditioned
    /// residual norm `√((B r, r))` (MFEM's CG stopping norm, B = the
    /// Setup-time Jacobi diagonal of `Mv`) of the CG solve of the momentum
    /// system `Mv·x = F(uⁿ) + f^{n+1}` with a zero initial guess; `0.0`
    /// before the first [`Self::step`].
    pub fn res_mvsolve(&self) -> f64 {
        self.res_mvsolve
    }
    /// `SpInv->GetFinalNorm()` of the last step: the final preconditioned
    /// residual norm `√((B r, r))` (B = `OrthoSolver(GSSmoother)` or the
    /// Setup-time AMG V-cycle, as selected by the pressure BCs) of the CG
    /// solve of the pressure Poisson system `Sp·p = B1` warm-started from
    /// the previous pressure; `0.0` before the first [`Self::step`].
    pub fn res_spsolve(&self) -> f64 {
        self.res_spsolve
    }
    /// `HInv->GetFinalNorm()` of the last step: the final preconditioned
    /// residual norm `√((B r, r))` (B = the Setup-time Jacobi diagonal of
    /// the BDF-scaled Helmholtz matrix `H`) of the CG solve of the viscous
    /// system `H·u^{n+1} = B2` warm-started from the projected boundary
    /// data; `0.0` before the first [`Self::step`].
    pub fn res_hsolve(&self) -> f64 {
        self.res_hsolve
    }

    /// MFEM `NavierSolver::Step(time, dt, cur_step, provisional = false)`.
    ///
    /// With `provisional = false` the computed step is accepted immediately:
    /// `UpdateTimestepHistory(dt)` is called and `time += dt`.
    pub fn step(&mut self, time: &mut f64, dt: f64, cur_step: i32, provisional: bool) {
        // D952-H2/H2': a provisional step overwrites `pn` without rotating
        // the velocity/pressure history (MFEM semantics — `X1` aliases
        // `pn_gf`), which would leave the incremental pressure state (`pn`
        // consumed as pⁿ by the next `p_ext`) polluted by the rejected
        // trial.  No caller in the workspace steps provisionally; the modes
        // demand the contract loudly rather than corrupting the state
        // silently.
        if provisional && self.cfg.pressure_mode != PressureMode::Classical {
            panic!(
                "pressure_mode keeps the pressure as accumulated state \
                 (p^{{n+1}} = p_ext + δp with the history shifted per \
                 accepted step); provisional steps are not supported"
            );
        }
        // D952-H2' composition guard: the Timmermans rotational correction
        // edits the ABSOLUTE pressure state (`p += −ν·Π(∇·ũ)`) between the
        // Poisson solve and the projection.  The round-134 absolute-
        // projection composition consumes that corrected state naturally;
        // the increment-only projection would project the UNCORRECTED
        // increment and silently drop the rotational term.  The combination
        // is rejected instead of miscomposed.
        if self.cfg.rotational && !self.cfg.increment_projection_absolute {
            panic!(
                "rotational + increment-only projection is not composed: the \
                 Timmermans correction edits the absolute pressure, which an \
                 increment-only mass projection would drop; use the default \
                 increment_projection_absolute = true (or turn rotational off)"
            );
        }
        let t_step = Instant::now();
        let mut t_sub = Instant::now();
        self.set_time_integration_coefficients(cur_step);
        self.cur_step = cur_step;

        let nv = self.disc.n_vel();
        let npl = self.disc.n_pres();
        let t_now = *time + dt;
        let vel_ess: Vec<usize> = self.disc.vel_ess_dofs().to_vec();
        let pres_ess: Vec<usize> = self.disc.pres_ess_dofs().to_vec();

        // H = (bd0/dt) Mv + ν Kv, reassembled with the current BDF coefficient
        // (`H_form->Update(); H_form->Assemble(); H_form->FormSystemMatrix()`).
        let mut h = self.disc.assemble_helmholtz(self.bd0 / dt, self.kin_vis);

        // D1125: the lagged velocity `u_lag = ab1·un + ab2·unm1 + ab3·unm2`
        // (the extrapolated velocity whose curl feeds the pressure step
        // below) drives the deferred-correction upwind defect; the SPD block
        // joins the IMPLICIT operator (`beta → 0` keeps `H` bit-identical).
        let mut lext = vec![0.0_f64; nv];
        for i in 0..nv {
            lext[i] = self.ab1 * self.vel.un[i]
                + self.ab2 * self.vel.unm1[i]
                + self.ab3 * self.vel.unm2[i];
        }
        if let ConvectionStabilization::DeferredUpwind { beta } =
            self.cfg.convection_stabilization
        {
            if beta != 0.0 {
                let dup = self.disc.assemble_upwind_defect(&lext);
                h = h.add(&dup);
            }
        }

        // Extrapolated f^{n+1}: the acceleration coefficient time is set to
        // t + dt before the linear form is reassembled.
        let accel = self.disc.assemble_accel(t_now);

        // Nonlinear extrapolated terms: N(u) = -C(u)·u on un, unm1, unm2 —
        // always the MFEM path (`N->Mult` on the three history levels,
        // navier_solver.cpp:411-413); the D1125 stabilization never touches
        // the residual.
        let mut nun = vec![0.0_f64; nv];
        let mut nunm1 = vec![0.0_f64; nv];
        let mut nunm2 = vec![0.0_f64; nv];
        self.disc.convection_residual(&self.vel.un, &mut nun);
        self.disc.convection_residual(&self.vel.unm1, &mut nunm1);
        self.disc.convection_residual(&self.vel.unm2, &mut nunm2);

        // Fext = ab1·N(un) + ab2·N(unm1) + ab3·N(unm2) + fn
        // (the `-1` is the `nlcoeff` of `VectorConvectionNLFIntegrator`).
        let mut fext = vec![0.0_f64; nv];
        for i in 0..nv {
            fext[i] = -(self.ab1 * nun[i] + self.ab2 * nunm1[i] + self.ab3 * nunm2[i])
                + accel[i];
        }
        self.rt_extrap = t_sub.elapsed().as_secs_f64();

        // Fext = Mv⁻¹ (F(u^n) + f^{n+1}) — `MvInv->Mult` with
        // `iterative_mode = false` (zero initial guess).
        let mut tmp1 = vec![0.0_f64; nv];
        {
            let mv = &self.mv;
            let diag = &self.mv_diag;
            let apply = |x: &[f64], y: &mut [f64]| mv.spmv(x, y);
            let jac = |r: &[f64], z: &mut [f64]| {
                for i in 0..r.len() {
                    z[i] = r[i] / diag[i];
                }
            };
            let res = solve_cg_mfem(
                nv,
                apply,
                &fext,
                &mut tmp1,
                Some(jac),
                &SliOptions {
                    rel_tol: self.cfg.rtol_mvsolve,
                    abs_tol: 0.0,
                    max_iter: 200,
                    print_level: self.cfg.pl_mvsolve,
                },
                false,
                None,
            );
            self.iter_mvsolve = res.iterations;
            self.res_mvsolve = res.final_norm;
        }
        fext.copy_from_slice(&tmp1);

        // Compute BDF terms: Fext += (-bd1·un - bd2·unm1 - bd3·unm2)/dt.
        for i in 0..nv {
            fext[i] += (-self.bd1 * self.vel.un[i] - self.bd2 * self.vel.unm1[i]
                - self.bd3 * self.vel.unm2[i])
                / dt;
        }

        // Pressure Poisson: Lext = ab1·un + ab2·unm1 + ab3·unm2 (already
        // formed above, where the D1125 defect reads it), then Lext *= ν
        // after the two curl applications (`ComputeCurl2D`).
        t_sub = Instant::now();
        let (curlu, cc) = self.disc.curl_curl_and_vorticity(&lext);
        self.curlu = curlu;
        for i in 0..nv {
            lext[i] = self.kin_vis * cc[i];
        }
        self.rt_curlcurl = t_sub.elapsed().as_secs_f64();

        // FText = Fext - ν CurlCurl(u).
        let mut ftext = fext.clone();
        for i in 0..nv {
            ftext[i] -= lext[i];
        }

        // p_r = ∇·FText (D·FText, negated).
        let mut resp = vec![0.0_f64; npl];
        self.d.spmv(&ftext, &mut resp);
        for v in resp.iter_mut() {
            *v = -*v;
        }

        // Add boundary terms: resp += FText_bdr - (bd0/dt)·g_bdr.
        let ftext_bdr = self.disc.assemble_ftext_bdr(&ftext);
        let g_bdr = self.disc.assemble_g_bdr(t_now);
        for i in 0..npl {
            resp[i] += ftext_bdr[i] - (self.bd0 / dt) * g_bdr[i];
        }

        if pres_ess.is_empty() {
            // Pure Neumann: remove the nullspace of the right-hand side.
            orthogonalize(&mut resp);
        }

        // `X1` aliases `pn_gf` in the C++ `FormLinearSystem`.
        let mut pn = std::mem::take(&mut self.pn);
        self.disc.project_pressure_bdr(t_now, &mut pn);
        let mut b1 = resp;
        // D952-H2 incremental re-assembly (`PressureMode::Incremental`): the
        // pressure becomes a STATE.  The Poisson system is solved for the
        // O(dt) INCREMENT `δp` and the absolute pressure is re-assembled as
        // `p^{n+1} = p_ext + δp` with `p_ext` the EXTk pressure
        // extrapolation (van Kan 1986; Guermond–Minev–Shen 2006 §2.2/§3.1).
        // The increment RHS substitutes the pressure-EXTRAPOLATED split
        // velocity into the classical data functional (see the long comment
        // inside the branch).  The default (`Classical`) never executes this
        // branch — the step stays bit-identical to the MFEM path.
        let mut p_ext: Option<Vec<f64>> = None;
        // The split velocity `u* = H⁻¹·(Mv·fext − G·p_ext)` of the
        // incremental modes (consumed by the split projection below; `None`
        // on the classical path).
        let mut split_vel: Option<Vec<f64>> = None;
        if self.cfg.pressure_mode != PressureMode::Classical {
            // ── D952-H2 split sequencing (van Kan 1986; Guermond–Minev–Shen
            // 2006 §2.2/§3.1) ──
            //
            // The FUSED step (classical) applies the full Helmholtz inverse
            // to the WHOLE pressure drive: u⁺ = H⁻¹·(Mv·fext − G·p^{n+1}).
            // The incremental scheme's second-order mechanism lives exactly
            // where that differs from the split form: the momentum sees only
            // the extrapolated pressure through the full Helmholtz, and the
            // increment's gradient is subtracted in MASS form,
            //
            //   u* = H⁻¹·(Mv·fext − G·p_ext)
            //   Sp·δp = (bd0/dt)·(Gᵀ·u* − g_bdr)      [raw increment Poisson;
            //           ∂_nδp = (bd0/dt)(u*·n − u_D·n) — vanishes on Γ_D]
            //   p^{n+1} = p_ext + δp
            //   u⁺ = u* − (dt/bd0)·Mv⁻¹·(G·p^{n+1}) ,  u⁺|_Γ = u_D .
            //
            // Reshaping the RHS of the fused absolute solve instead was
            // MEASURED to be structurally wrong (any kernel surrogate of the
            // textbook `-Δp_ext` term either cancels the composition — the
            // consistent `−Sp·p_ext` choice, a no-op — or leaves an O(1)
            // operator mismatch `(Sp − Q)·p_ext`, Q = GᵀMv⁻¹G, which drove
            // an oscillating absolute pressure and an energy-growing
            // trajectory; round-134 variant scan + trajectory diagnostic).
            //
            // EXTk pressure extrapolation from the pressure history (the
            // same coefficients the velocity extrapolation uses).  The k-th
            // order extrapolation needs k REAL history levels: the kernel
            // has no pressure initial condition (the classical path never
            // consumes one), so until the history is mature the
            // extrapolation stays EXT1 (`p_ext = pⁿ`) — extrapolating with
            // a fake `p⁰ = 0` kicks the trajectory.
            let mature = cur_step >= 2;
            let pe: Vec<f64> = if mature {
                (0..npl)
                    .map(|i| {
                        self.ab1 * pn[i] + self.ab2 * self.pnm1[i] + self.ab3 * self.pnm2[i]
                    })
                    .collect()
            } else {
                pn.clone()
            };
            // History shift BEFORE the solves overwrite `pn`: pnm2 ← p^{n−1},
            // pnm1 ← pⁿ (the composition below restores `pn = p^{n+1}`).
            // Provisional steps never reach this point (the loud guard at the
            // step entry rejects them: they overwrite `pn` without rotating
            // the history).
            self.pnm2.copy_from_slice(&self.pnm1);
            self.pnm1.copy_from_slice(&pn);
            // Split velocity: one Helmholtz solve with the extrapolated
            // pressure drive.  This is the step's Helmholtz solve (the fused
            // main solve below is skipped); it eliminates `h` ONCE with the
            // projected boundary data.
            let mut gpe = vec![0.0_f64; nv];
            self.g.spmv(&pe, &mut gpe);
            let mut resu_ext = vec![0.0_f64; nv];
            self.mv.spmv(&fext, &mut resu_ext);
            for i in 0..nv {
                resu_ext[i] -= gpe[i];
            }
            self.disc.project_velocity_bdr(t_now, &mut self.vel.un_next);
            let mut b2ext = resu_ext;
            if !vel_ess.is_empty() {
                let vals: Vec<f64> = vel_ess.iter().map(|&d| self.vel.un_next[d]).collect();
                self.disc.eliminate_bc(&mut h, &mut b2ext, &vel_ess, &vals);
            }
            let mut us = self.vel.un_next.clone();
            {
                let hd = &self.h_diag;
                let apply = |x: &[f64], y: &mut [f64]| h.spmv(x, y);
                let jac = |r: &[f64], z: &mut [f64]| {
                    for i in 0..r.len() {
                        z[i] = r[i] / hd[i];
                    }
                };
                let t_solve = Instant::now();
                let res = solve_cg_mfem(
                    nv,
                    apply,
                    &b2ext,
                    &mut us,
                    Some(jac),
                    &SliOptions {
                        rel_tol: self.cfg.rtol_hsolve,
                        abs_tol: 0.0,
                        max_iter: 200,
                        print_level: self.cfg.pl_hsolve,
                    },
                    true,
                    None,
                );
                self.rt_hsolve = t_solve.elapsed().as_secs_f64();
                self.iter_hsolve = res.iterations;
                self.res_hsolve = res.final_norm;
            }
            // Raw increment data functional — IDENTICAL for both incremental
            // modes (`Incremental` and `IncrementalSelfConsistent`): the
            // classical data pattern (`Gᵀ·(data) − (bd0/dt)·g_bdr`) fed with
            // the split velocity's acceleration `(bd0/dt)·u*`.  `Gᵀ·w` rides
            // the exact identity `Gᵀ = B − D` (`assemble_ftext_bdr(w) =
            // B·w`).  `g_bdr` is the boundary functional assembled above for
            // `resp`.  The modes differ ONLY downstream: the Poisson LEFT
            // end (assembled `Sp` vs the projection-adjoint `−D·Mv⁻¹·G`) and
            // the projection target (see the solve block below).
            b1 = vec![0.0_f64; npl];
            let ftb = self.disc.assemble_ftext_bdr(&us);
            let mut dus = vec![0.0_f64; npl];
            self.d.spmv(&us, &mut dus);
            for i in 0..npl {
                b1[i] = (self.bd0 / dt) * (ftb[i] - dus[i] - g_bdr[i]);
            }
            if pres_ess.is_empty() {
                // Keep the RHS in the range of the pure-Neumann `Sp`.
                orthogonalize(&mut b1);
            }
            split_vel = Some(us);
            p_ext = Some(pe);
        }
        // Essential data of the Poisson system, hoisted for both elimination
        // styles (the classical in-place `eliminate_bc` and the D952-H2'
        // matrix-free bookkeeping inside the SC solve).
        let ess_vals: Option<Vec<f64>> = if pres_ess.is_empty() {
            None
        } else {
            let vals: Vec<f64> = match &p_ext {
                // Incremental: the increment's essential data is
                // `p_D(t^{n+1}) − p_ext|_Γ` — the Dirichlet data enters
                // through the composition `pn = p_ext + δp` below.
                Some(pe) => pres_ess.iter().map(|&d| pn[d] - pe[d]).collect(),
                None => pres_ess.iter().map(|&d| pn[d]).collect(),
            };
            Some(vals)
        };
        let sc_mode = self.cfg.pressure_mode == PressureMode::IncrementalSelfConsistent;
        if let Some(vals) = &ess_vals {
            if sc_mode {
                // The SC left end is matrix-free — its essential elimination
                // rides the SC solve below (essential values pinned into the
                // outer CG iterate, reactions via one extra operator apply).
                // `sp` still takes its (idempotent) DIAG_KEEP elimination so
                // the preconditioner sweeps run on the same matrix shape as
                // the classical path; the throwaway scratch absorbs the
                // Sp-based reactions the SC operator replaces with its own.
                let mut scratch = b1.clone();
                self.disc
                    .eliminate_bc(&mut self.sp, &mut scratch, &pres_ess, vals);
            } else {
                self.disc
                    .eliminate_bc(&mut self.sp, &mut b1, &pres_ess, vals);
            }
        }

        // SpInv->Mult(B1, X1): with no pressure Dirichlet BCs the
        // preconditioner is `OrthoSolver(GSSmoother)`, otherwise the bare
        // smoother.  Classical solves for the absolute pressure (warm start
        // from `pⁿ`, `iterative_mode = true`); Incremental solves for the
        // increment (zero initial guess — the classical warm start would
        // seed the O(dt) increment with an O(p) value).  The D952-H2'
        // self-consistent mode solves the SAME increment system with the
        // projection-adjoint left end (nested Krylov, see the SC arm).
        //
        // `delta_p` carries the solved increment to the split projection
        // when the projection composes with the INCREMENT
        // (`increment_projection_absolute = false`); the absolute arms
        // project `pn` and leave it `None`.
        let mut delta_p: Option<Vec<f64>> = None;
        {
            let sp = &self.sp;
            let amg = self.sp_amg.as_ref();
            let apply = |x: &[f64], y: &mut [f64]| sp.spmv(x, y);
            let ortho = pres_ess.is_empty();
            let mut r_ortho = vec![0.0_f64; npl];
            let mut z_ortho = vec![0.0_f64; npl];
            let prec = |r: &[f64], out: &mut [f64]| {
                let rin: &[f64] = if ortho {
                    r_ortho.copy_from_slice(r);
                    orthogonalize(&mut r_ortho);
                    &r_ortho
                } else {
                    r
                };
                if let Some(amg) = amg {
                    // `SpInvPC->Mult`: the BoomerAMG analogue V-cycle.
                    out.copy_from_slice(&amg.precond_apply(rin));
                } else {
                    // `GSSmoother::Mult` zeroes its output (`iterative_mode`
                    // is false, as propagated by `OrthoSolver`), then runs one
                    // forward and one backward sweep.
                    out.fill(0.0);
                    gs_forward(sp, rin, out);
                    gs_backward(sp, rin, out);
                }
                if ortho {
                    z_ortho.copy_from_slice(out);
                    orthogonalize(&mut z_ortho);
                    out.copy_from_slice(&z_ortho);
                }
            };
            let t_solve = Instant::now();
            // D952-H2' inner-solve bookkeeping (see the SC arm).
            let inner = std::cell::Cell::new(0_i32);
            let res = match &p_ext {
                None => solve_cg_mfem(
                    npl,
                    apply,
                    &b1,
                    &mut pn,
                    Some(prec),
                    &SliOptions {
                        rel_tol: self.cfg.rtol_spsolve,
                        abs_tol: 0.0,
                        max_iter: 200,
                        print_level: self.cfg.pl_spsolve,
                    },
                    true,
                    None,
                ),
                Some(pe) => {
                    if !sc_mode {
                        let mut pinc = vec![0.0_f64; npl];
                        let res = solve_cg_mfem(
                            npl,
                            apply,
                            &b1,
                            &mut pinc,
                            Some(prec),
                            &SliOptions {
                                rel_tol: self.cfg.rtol_spsolve,
                                abs_tol: 0.0,
                                max_iter: 200,
                                print_level: self.cfg.pl_spsolve,
                            },
                            false,
                            None,
                        );
                        // Re-assemble the absolute pressure `p^{n+1} = p_ext +
                        // δp`; essential DOFs keep the projected Dirichlet data
                        // (`pn` holds it — the increment's own essential entries
                        // carry `p_D − p_ext` and must not be added on top).
                        let mut ess = pres_ess.iter();
                        let mut next_ess = ess.next();
                        for i in 0..npl {
                            if next_ess.is_some_and(|e| *e == i) {
                                next_ess = ess.next();
                                continue;
                            }
                            pn[i] = pe[i] + pinc[i];
                        }
                        if !self.cfg.increment_projection_absolute {
                            delta_p = Some(pinc);
                        }
                        res
                    } else {
                        // ── D952-H2' self-consistent increment Poisson ──
                        //
                        // Left end: the exact discrete adjoint of the
                        // mass-form projection, `A_sc = −D·Mv⁻¹·G` (the
                        // negative-semidefinite bulk mirror of `Sp`:
                        // `−Q = Gᵀ·Mv⁻¹·G` up to the boundary-flux term;
                        // SPD-dominant, CG-convergent).  Data (b1 = the
                        // raw increment functional assembled above):
                        //
                        //   increment projection:  A_sc·δp = b1
                        //   absolute projection:   A_sc·δp = b1 + Q·p_ext
                        //
                        // because zero end-of-step divergence
                        // `D·u⁺ = D·u* − (dt/bd0)·Q·target = 0` demands
                        // `Q·target = (bd0/dt)·D·u* = −b1 + O(boundary)`,
                        // with `target = δp` (textbook GMS §2.2) or
                        // `p_ext + δp` (round-134 composition).  The outer
                        // CG and the projection apply the IDENTICAL inexact
                        // operator, so the constraint closes to the OUTER
                        // tolerance regardless of the inner `Mv⁻¹`
                        // tolerance (`sc_inner_rtol` only trades Poisson
                        // consistency vs cost).
                        let mv = &self.mv;
                        let mv_diag = &self.mv_diag;
                        let dm = &self.d;
                        let gm = &self.g;
                        let inner_rtol = self.cfg.sc_inner_rtol;
                        let absolute = self.cfg.increment_projection_absolute;
                        let pres_ess_ref: &[usize] = &pres_ess;
                        let inner_sc = &inner;
                        let apply_sc = move |x: &[f64], y: &mut [f64]| {
                            let mut gx = vec![0.0_f64; nv];
                            gm.spmv(x, &mut gx);
                            let mut w = vec![0.0_f64; nv];
                            let r_in = solve_cg_mfem(
                                nv,
                                |xx: &[f64], yy: &mut [f64]| mv.spmv(xx, yy),
                                &gx,
                                &mut w,
                                Some(move |r: &[f64], z: &mut [f64]| {
                                    for (zz, (rr, &dg)) in
                                        z.iter_mut().zip(r.iter().zip(mv_diag.iter()))
                                    {
                                        *zz = rr / dg;
                                    }
                                }),
                                &SliOptions {
                                    rel_tol: inner_rtol,
                                    abs_tol: 0.0,
                                    max_iter: 200,
                                    print_level: 0,
                                },
                                false,
                                None,
                            );
                            inner_sc.set(inner_sc.get() + r_in.iterations);
                            if r_in.iterations >= 200 && !r_in.converged {
                                eprintln!(
                                    "D952-H2' inner Mv-CG hit the iteration cap \
                                     (final norm {:.3e}) — the nested operator is \
                                     degraded",
                                    r_in.final_norm
                                );
                            }
                            dm.spmv(&w, y);
                            for v in y.iter_mut() {
                                *v = -*v;
                            }
                            // Essential-row closure: `y[ess] = x[ess]` (the
                            // outer iterate keeps the essential values via
                            // its pinned initial guess below; the closure
                            // makes the essential residual — and hence every
                            // preconditioned direction — vanish there).
                            for &dof in pres_ess_ref {
                                y[dof] = x[dof];
                            }
                        };
                        // Data: b1 (+ Q·p_ext on the absolute arm — note the
                        // sign: `A_sc·δp = b1 + Q·p_ext` with `A_sc = −Q`).
                        let mut b_sc = b1;
                        if absolute {
                            let mut qpe = vec![0.0_f64; npl];
                            apply_sc(pe, &mut qpe);
                            for i in 0..npl {
                                b_sc[i] -= qpe[i];
                            }
                        }
                        if ortho {
                            orthogonalize(&mut b_sc);
                        }
                        // Matrix-free essential bookkeeping: the reaction
                        // `A_sc[free,ess]·vals` rides the RHS (one extra
                        // apply); the essential entries of the RHS carry the
                        // pinned values themselves.
                        if let Some(vals) = &ess_vals {
                            let mut pad = vec![0.0_f64; npl];
                            for (k, &d) in pres_ess.iter().enumerate() {
                                pad[d] = vals[k];
                            }
                            let mut react = vec![0.0_f64; npl];
                            apply_sc(&pad, &mut react);
                            for (k, &d) in pres_ess.iter().enumerate() {
                                b_sc[d] = vals[k];
                                react[d] = 0.0;
                            }
                            for i in 0..npl {
                                b_sc[i] -= react[i];
                            }
                        }
                        // Zero initial guess on the free block; the
                        // essential entries start PINNED at their data (with
                        // the closure above this keeps every Krylov
                        // direction's essential block at zero, so the outer
                        // CG sees a linear operator — the matrix-free
                        // analogue of MFEM's DIAG_KEEP elimination).
                        let mut pinc = vec![0.0_f64; npl];
                        if let Some(vals) = &ess_vals {
                            for (k, &d) in pres_ess.iter().enumerate() {
                                pinc[d] = vals[k];
                            }
                        }
                        let res = solve_cg_mfem(
                            npl,
                            apply_sc,
                            &b_sc,
                            &mut pinc,
                            Some(prec),
                            &SliOptions {
                                rel_tol: self.cfg.rtol_spsolve,
                                abs_tol: 0.0,
                                max_iter: 200,
                                print_level: self.cfg.pl_spsolve,
                            },
                            true,
                            None,
                        );
                        // Compose `p^{n+1} = p_ext + δp` (free DOFs; the
                        // essential DOFs keep the projected Dirichlet data).
                        let mut ess = pres_ess.iter();
                        let mut next_ess = ess.next();
                        for i in 0..npl {
                            if next_ess.is_some_and(|e| *e == i) {
                                next_ess = ess.next();
                                continue;
                            }
                            pn[i] = pe[i] + pinc[i];
                        }
                        if !absolute {
                            delta_p = Some(pinc);
                        }
                        res
                    }
                }
            };
            self.sc_inner_iters = inner.get();
            self.rt_spsolve = t_solve.elapsed().as_secs_f64();
            self.iter_spsolve = res.iterations;
            self.res_spsolve = res.final_norm;
        }

        if pres_ess.is_empty() {
            // MeanZero(pn_gf): remove the pressure nullspace.
            self.disc.mean_zero(&mut pn);
        }
        self.pn = pn;

        // D952 Timmermans rotational pressure correction (opt-in,
        // `NavierConfig::rotational`): the classical scheme's pressure comes
        // from the *extrapolated* velocity's curl-curl alone, which leaves an
        // O(dt) splitting term in the end-of-step velocity; the rotational
        // split adds `p^{n+1} += −ν·∇·ũ^{n+1}` (Timmermans, Minev & Van De
        // Vosse 1995) with ũ^{n+1} the PURE VISCOUS split velocity.  This
        // branch realizes that sequencing on top of the step above: one extra
        // Helmholtz solve without the pressure drive gives ũ, its pressure-
        // space divergence forms the correction, and `pn` is REPLACED by the
        // corrected `pc` — the projection solve below then consumes exactly
        // the rotational-corrected pressure (a post-step pressure patch alone
        // would be dynamics-dead: the next step's Poisson solve never consumes
        // `pn` as state — measured, round-133 lane 3).  Essential pressure
        // DOFs keep their Dirichlet data; `MeanZero` is re-applied on the
        // pure-Neumann path.  The default (`rotational = false`) never
        // executes this branch — the step stays bit-identical to the MFEM
        // path.
        if self.cfg.rotational {
            // Viscous split: ũ = H⁻¹·(Mv·fext) — the momentum BDF step with
            // NO pressure drive (the same `h` the projection solve below
            // uses; essential velocity rows keep their Dirichlet values).
            let mut resu_visc = vec![0.0_f64; nv];
            self.mv.spmv(&fext, &mut resu_visc);
            self.disc.project_velocity_bdr(t_now, &mut self.vel.un_next);
            let mut b_visc = resu_visc;
            if !vel_ess.is_empty() {
                let vals: Vec<f64> = vel_ess.iter().map(|&d| self.vel.un_next[d]).collect();
                self.disc.eliminate_bc(&mut h, &mut b_visc, &vel_ess, &vals);
            }
            {
                let hd = &self.h_diag;
                let apply = |x: &[f64], y: &mut [f64]| h.spmv(x, y);
                let jac = |r: &[f64], z: &mut [f64]| {
                    for i in 0..r.len() {
                        z[i] = r[i] / hd[i];
                    }
                };
                let res = solve_cg_mfem(
                    nv,
                    apply,
                    &b_visc,
                    &mut self.vel.un_next,
                    Some(jac),
                    &SliOptions {
                        rel_tol: self.cfg.rtol_hsolve,
                        abs_tol: 0.0,
                        max_iter: 200,
                        print_level: self.cfg.pl_hsolve,
                    },
                    true,
                    None,
                );
                let _ = (res.iterations, res.final_norm); // the viscous pre-solve is not the step's HELM record
            }
            // Rotational term: pc = pn − ν·Π(∇·ũ).
            let div_u = self.disc.rotational_divergence_pressure(&self.vel.un_next);
            assert_eq!(
                div_u.len(),
                npl,
                "rotational_divergence_pressure must return n_pres values"
            );
            let mut pc = self.pn.clone();
            let mut ess = pres_ess.iter();
            let mut next_ess = ess.next();
            for (i, d) in div_u.iter().enumerate() {
                if next_ess.is_some_and(|e| *e == i) {
                    next_ess = ess.next();
                    continue;
                }
                pc[i] -= self.kin_vis * d;
            }
            if pres_ess.is_empty() {
                // MeanZero(pc): the pure-Neumann gauge convention of every
                // completed step (same as after the Poisson solve above).
                self.disc.mean_zero(&mut pc);
            }
            self.pn = pc;
        }

        if let Some(us) = split_vel {
            // ── D952-H2/H2' split projection: u⁺ = u* − (dt/bd0)·Mv⁻¹·G·target
            // (the pressure gradient rides the MASS matrix — the fused
            // Helmholtz projection below is exactly what the incremental
            // schemes must NOT do).  The projection TARGET carries the
            // composition factor of the D952-H2' experiment: `p^{n+1}` on
            // the absolute arms (the round-134 default form) or the solved
            // increment `δp` on the textbook GMS §2.2 arms.  Essential
            // velocity dofs keep the Dirichlet data (u* already carries it).
            let proj_target = match &delta_p {
                Some(dp) => dp.clone(),
                None => self.pn.clone(),
            };
            let mut gp = vec![0.0_f64; nv];
            self.g.spmv(&proj_target, &mut gp);
            let mut w = vec![0.0_f64; nv];
            {
                let mv = &self.mv;
                let diag = &self.mv_diag;
                let apply = |x: &[f64], y: &mut [f64]| mv.spmv(x, y);
                let jac = |r: &[f64], z: &mut [f64]| {
                    for i in 0..r.len() {
                        z[i] = r[i] / diag[i];
                    }
                };
                let res = solve_cg_mfem(
                    nv,
                    apply,
                    &gp,
                    &mut w,
                    Some(jac),
                    &SliOptions {
                        rel_tol: self.cfg.rtol_mvsolve,
                        abs_tol: 0.0,
                        max_iter: 200,
                        print_level: self.cfg.pl_mvsolve,
                    },
                    false,
                    None,
                );
                let _ = (res.iterations, res.final_norm); // the split projection is not the step's MV record
            }
            let scale = dt / self.bd0;
            self.vel.un_next.copy_from_slice(&us);
            let mut ess = vel_ess.iter();
            let mut next_ess = ess.next();
            for i in 0..nv {
                if next_ess.is_some_and(|e| *e == i) {
                    next_ess = ess.next();
                    continue;
                }
                self.vel.un_next[i] -= scale * w[i];
            }
        } else {
            // Project velocity: resu = -G·pn + Mv·Fext.
            let mut resu = vec![0.0_f64; nv];
            self.g.spmv(&self.pn, &mut resu);
            for v in resu.iter_mut() {
                *v = -*v;
            }
            {
                let mut mv_fext = vec![0.0_f64; nv];
                self.mv.spmv(&fext, &mut mv_fext);
                for i in 0..nv {
                    resu[i] += mv_fext[i];
                }
            }

            // un_next_gf.ProjectBdrCoefficient(vel_dbcs).
            self.disc.project_velocity_bdr(t_now, &mut self.vel.un_next);

            // H_form->FormLinearSystem(vel_ess_tdof, un_next_gf, resu_gf, …):
            // `X2` aliases `un_next_gf` and the RHS is the eliminated `resu_gf`.
            let mut b2 = resu;
            if !vel_ess.is_empty() {
                let vals: Vec<f64> = vel_ess.iter().map(|&d| self.vel.un_next[d]).collect();
                self.disc.eliminate_bc(&mut h, &mut b2, &vel_ess, &vals);
            }

            // HInv->Mult(B2, X2) — `iterative_mode = true` (initial guess
            // `un_next_gf`) with the Setup-time Jacobi preconditioner.
            {
                let hd = &self.h_diag;
                let apply = |x: &[f64], y: &mut [f64]| h.spmv(x, y);
                let jac = |r: &[f64], z: &mut [f64]| {
                    for i in 0..r.len() {
                        z[i] = r[i] / hd[i];
                    }
                };
                let t_solve = Instant::now();
                let res = solve_cg_mfem(
                    nv,
                    apply,
                    &b2,
                    &mut self.vel.un_next,
                    Some(jac),
                    &SliOptions {
                        rel_tol: self.cfg.rtol_hsolve,
                        abs_tol: 0.0,
                        max_iter: 200,
                        print_level: self.cfg.pl_hsolve,
                    },
                    true,
                    None,
                );
                self.rt_hsolve = t_solve.elapsed().as_secs_f64();
                self.iter_hsolve = res.iterations;
                self.res_hsolve = res.final_norm;
            }
        }

        if !provisional {
            self.update_timestep_history(dt);
            *time += dt;
        }

        self.rt_step = t_step.elapsed().as_secs_f64();

        if self.cfg.verbose {
            self.print_step_iterations();
        }
    }

    /// The `verbose` block at the end of `NavierSolver::Step`.
    fn print_step_iterations(&self) {
        println!("{:>7}{:>3}{:>8}{:>12}", "", "It", "Resid", "Reltol");
        println!(
            "MVIN {:>5}   {}   {}",
            self.iter_mvsolve,
            fmt_sci(self.res_mvsolve, 2, false),
            fmt_sci(self.cfg.rtol_mvsolve, 2, false)
        );
        println!(
            "PRES {:>5}   {}   {}",
            self.iter_spsolve,
            fmt_sci(self.res_spsolve, 2, false),
            fmt_sci(self.cfg.rtol_spsolve, 2, false)
        );
        println!(
            "HELM {:>5}   {}   {}",
            self.iter_hsolve,
            fmt_sci(self.res_hsolve, 2, false),
            fmt_sci(self.cfg.rtol_hsolve, 2, false)
        );
    }

    /// MFEM `NavierSolver::PrintTimingData`.
    pub fn print_timing_data(&self) {
        let rt = [
            self.rt_setup,
            self.rt_step,
            self.rt_extrap,
            self.rt_curlcurl,
            self.rt_spsolve,
            self.rt_hsolve,
        ];
        let hdr = ["SETUP", "STEP", "EXTRAP", "CURLCURL", "PSOLVE", "HSOLVE"];
        println!(
            "{}",
            hdr.iter()
                .map(|h| format!("{h:>10}"))
                .collect::<String>()
        );
        println!(
            "{}",
            rt.iter()
                .map(|v| format!("{:>10}", fmt_sci(*v, 3, false)))
                .collect::<String>()
        );
        println!(
            "{:>10}{}",
            " ",
            rt[1..]
                .iter()
                .map(|v| format!("{:>10}", fmt_sci(*v / rt[1], 3, false)))
                .collect::<String>()
        );
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The BDFk/EXTk coefficients must reproduce the C++
    /// `SetTimeIntegrationCoefficients` branch by branch.
    #[test]
    fn bdf_ext_coefficients_match_mfem() {
        let disc = ToyDisc::new(2, 1, true);
        let mut s = NavierSolver::new(disc, 1.0, NavierConfig { verbose: false, ..Default::default() });

        // Step 0, uniform dt: BDF1/EXT1.
        s.setup(0.1);
        s.set_time_integration_coefficients(0);
        assert_eq!((s.bd0, s.bd1, s.bd2, s.bd3), (1.0, -1.0, 0.0, 0.0));
        assert_eq!((s.ab1, s.ab2, s.ab3), (1.0, 0.0, 0.0));

        // Step 1, uniform dt: BDF2/EXT2 with rho1 = 1.
        s.update_timestep_history(0.1);
        s.set_time_integration_coefficients(1);
        assert!((s.bd0 - 1.5).abs() < 1e-15);
        assert!((s.bd1 + 2.0).abs() < 1e-15);
        assert!((s.bd2 - 0.5).abs() < 1e-15);
        assert_eq!(s.bd3, 0.0);
        assert!((s.ab1 - 2.0).abs() < 1e-15);
        assert!((s.ab2 + 1.0).abs() < 1e-15);

        // Step 2, uniform dt: BDF3/EXT3 with rho1 = rho2 = 1.
        s.update_timestep_history(0.1);
        s.set_time_integration_coefficients(2);
        assert!((s.bd0 - 11.0 / 6.0).abs() < 1e-14, "bd0 = {}", s.bd0);
        assert!((s.bd1 + 3.0).abs() < 1e-14, "bd1 = {}", s.bd1);
        assert!((s.bd2 - 1.5).abs() < 1e-14, "bd2 = {}", s.bd2);
        assert!((s.bd3 + 1.0 / 3.0).abs() < 1e-14, "bd3 = {}", s.bd3);
        assert!((s.ab1 - 3.0).abs() < 1e-14, "ab1 = {}", s.ab1);
        assert!((s.ab2 + 3.0).abs() < 1e-14, "ab2 = {}", s.ab2);
        assert!((s.ab3 - 1.0).abs() < 1e-14, "ab3 = {}", s.ab3);

        // Variable time steps: rho1 = 2, rho2 = 0.5 (dt = 0.2, 0.1, 0.2).
        let mut s2 = NavierSolver::new(
            ToyDisc::new(2, 1, true),
            1.0,
            NavierConfig { verbose: false, ..Default::default() },
        );
        s2.setup(0.2);
        s2.update_timestep_history(0.1);
        s2.update_timestep_history(0.2);
        s2.set_time_integration_coefficients(2);
        let rho1 = 2.0_f64;
        let rho2 = 0.5_f64;
        let bd0 = 1.0 + rho1 / (1.0 + rho1) + (rho2 * rho1) / (1.0 + rho2 * (1.0 + rho1));
        let bd3 = -(rho2.powi(3) * rho1.powi(2) * (1.0 + rho1))
            / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
        assert!((s2.bd0 - bd0).abs() < 1e-14);
        assert!((s2.bd3 - bd3).abs() < 1e-14);

        // `SetMaxBDFOrder(1)` clamps the order.
        let mut s3 = NavierSolver::new(
            ToyDisc::new(2, 1, true),
            1.0,
            NavierConfig { verbose: false, ..Default::default() },
        );
        s3.setup(0.1);
        s3.set_max_bdf_order(1);
        s3.update_timestep_history(0.1);
        s3.set_time_integration_coefficients(1);
        // bdf_order = 1 but step >= 1: no branch matches, as in C++ (the
        // coefficients keep their previous value).
        assert_eq!(s3.bd0, 0.0);
    }

    /// The `ComputeCurl3D` trait surface: 2-D discretizations keep the
    /// default, which aborts with a message (the C++ class only calls it when
    /// `pmesh->Dimension() == 3`).
    #[test]
    #[should_panic(expected = "ComputeCurl3D requires a 3-D discretization")]
    fn compute_curl_3d_default_panics() {
        let disc = ToyDisc::new(2, 1, true);
        let _ = disc.compute_curl_3d(&[1.0, 0.0]);
    }

    /// Gauss-Seidel sweeps reproduce MFEM's in-place forward/backward sweeps.
    #[test]
    fn gauss_seidel_sweeps() {
        // A = [[4, 1], [1, 3]], b = [1, 2].
        let mut a = CsrMatrix::new_empty(2, 2);
        a.row_ptr = vec![0, 2, 4];
        a.col_idx = vec![0, 1, 0, 1];
        a.values = vec![4.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        let mut y = vec![0.0, 0.0];
        // Forward sweep from zero: y0 = 1/4, y1 = (2 - y0)/3.
        gs_forward(&a, &b, &mut y);
        assert!((y[0] - 0.25).abs() < 1e-15);
        assert!((y[1] - (1.75 / 3.0)).abs() < 1e-15);
        // Backward sweep continues in place: y1 = (2 - y0)/3, then
        // y0 = (1 - y1)/4.
        gs_backward(&a, &b, &mut y);
        let y1 = (2.0 - 0.25) / 3.0;
        assert!((y[1] - y1).abs() < 1e-15);
        assert!((y[0] - (1.0 - y1) / 4.0).abs() < 1e-15);

        // Warm start: MFEM's `GSSmoother` does not zero `y` in
        // `iterative_mode`, so the incoming `y[1]` enters the first update.
        let mut y_warm = vec![7.0, -3.0];
        gs_forward(&a, &b, &mut y_warm);
        assert!((y_warm[0] - (1.0 + 3.0) / 4.0).abs() < 1e-15);
        assert!((y_warm[1] - (2.0 - y_warm[0]) / 3.0).abs() < 1e-15);
    }

    /// `Orthogonalize` / `MeanZero`-style mean removal.
    #[test]
    fn orthogonalize_removes_mean() {
        let mut v = vec![1.0, 2.0, 3.0, 6.0];
        orthogonalize(&mut v);
        assert!(v.iter().sum::<f64>().abs() < 1e-15);
        assert!((v[3] - v[0] - 5.0).abs() < 1e-15);
    }

    /// C++-compatible scientific formatting (`%.2E` / `std::scientific`).
    #[test]
    fn scientific_formatting() {
        assert_eq!(fmt_sci(6.23e-2, 2, true), "6.23E-02");
        assert_eq!(fmt_sci(1.0e-3, 2, true), "1.00E-03");
        assert_eq!(fmt_sci(6.57565e-7, 5, true), "6.57565E-07");
        assert_eq!(fmt_sci(9.98e-13, 2, false), "9.98e-13");
        assert_eq!(fmt_sci(1.0e-12, 2, false), "1.00e-12");
        assert_eq!(fmt_sci(1.0, 3, false), "1.000e+00");
        assert_eq!(fmt_sci(-1.0, 3, false), "-1.000e+00");
        assert_eq!(fmt_sci(0.0, 3, false), "0.000e+00");
    }

    // ─── A tiny hand-checkable discretization ────────────────────────────────

    /// Two velocity DOFs, one pressure DOF, diagonal operators — the driver's
    /// algebra can be verified by hand.
    struct ToyDisc {
        nv: usize,
        np: usize,
        vel_ess: Vec<usize>,
        pres_ess: Vec<usize>,
        /// `assemble_helmholtz` records the coefficients it was called with.
        h_calls: std::cell::RefCell<Vec<(f64, f64)>>,
        /// `eliminate_bc` records the essential DOF sets it saw.
        elim_calls: std::cell::RefCell<Vec<Vec<usize>>>,
        cv: std::cell::Cell<f64>,
        /// `rotational_divergence_pressure` override (the D952 toy channel):
        /// `None` reproduces the loud default (the disc does not support the
        /// rotational correction); `Some(v)` returns the hand-set `v` so the
        /// kernel's `−ν·δp` update is checkable by hand.
        rot_div: std::cell::RefCell<Option<Vec<f64>>>,
    }

    impl ToyDisc {
        fn new(nv: usize, np: usize, dirichlet: bool) -> Self {
            ToyDisc {
                nv,
                np,
                vel_ess: if dirichlet { vec![0] } else { vec![] },
                pres_ess: if dirichlet { vec![] } else { vec![0] },
                h_calls: std::cell::RefCell::new(Vec::new()),
                elim_calls: std::cell::RefCell::new(Vec::new()),
                cv: std::cell::Cell::new(0.0),
                rot_div: std::cell::RefCell::new(None),
            }
        }

        fn eye(n: usize, scale: f64) -> CsrMatrix<f64> {
            let mut m = CsrMatrix::new_empty(n, n);
            m.row_ptr = (0..=n).collect();
            m.col_idx = (0..n as u32).collect();
            m.values = vec![scale; n];
            m
        }
    }

    impl NavierDiscretization for ToyDisc {
        fn n_vel(&self) -> usize {
            self.nv
        }
        fn n_pres(&self) -> usize {
            self.np
        }
        fn vel_ess_dofs(&self) -> &[usize] {
            &self.vel_ess
        }
        fn pres_ess_dofs(&self) -> &[usize] {
            &self.pres_ess
        }
        fn assemble_mass_velocity(&self) -> CsrMatrix<f64> {
            Self::eye(self.nv, 2.0)
        }
        fn assemble_pressure_laplace(&self) -> CsrMatrix<f64> {
            Self::eye(self.np, 1.0)
        }
        fn assemble_divergence(&self) -> CsrMatrix<f64> {
            // n_pres x n_vel, all ones.
            let mut m = CsrMatrix::new_empty(self.np, self.nv);
            m.row_ptr = (0..=self.np).map(|r| r * self.nv).collect();
            m.col_idx = (0..self.np * self.nv).map(|c| (c % self.nv) as u32).collect();
            m.values = vec![1.0; self.np * self.nv];
            m
        }
        fn assemble_gradient(&self) -> CsrMatrix<f64> {
            // n_vel x n_pres = Dᵀ.
            let mut m = CsrMatrix::new_empty(self.nv, self.np);
            m.row_ptr = (0..=self.nv).map(|r| r * self.np).collect();
            m.col_idx = (0..self.nv * self.np).map(|c| (c % self.np) as u32).collect();
            m.values = vec![1.0; self.nv * self.np];
            m
        }
        fn assemble_helmholtz(&self, mass_coeff: f64, visc_coeff: f64) -> CsrMatrix<f64> {
            self.h_calls.borrow_mut().push((mass_coeff, visc_coeff));
            // 2*mass + visc on the diagonal, 1 on the off-diagonal.
            let mut m = CsrMatrix::new_empty(self.nv, self.nv);
            let mut rp = vec![0usize];
            let mut ci = Vec::new();
            let mut va = Vec::new();
            for r in 0..self.nv {
                for c in 0..self.nv {
                    ci.push(c as u32);
                    va.push(if r == c { 2.0 * mass_coeff + visc_coeff } else { 1.0 });
                }
                rp.push(ci.len());
            }
            m.row_ptr = rp;
            m.col_idx = ci;
            m.values = va;
            m
        }
        fn convection_residual(&self, u: &[f64], out: &mut [f64]) {
            // Toy: `∫ (u·∇u)·φ ≈ 0.5 u` component-wise.
            for (o, u) in out.iter_mut().zip(u.iter()) {
                *o = 0.5 * u;
            }
        }
        fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
            // A fixed symmetric "curl curl": (u0 + u1, u0 + u1, …)/2.
            let s: f64 = u.iter().sum::<f64>() / u.len() as f64;
            vec![s; u.len()]
        }
        fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
            for &d in &self.vel_ess {
                out[d] = 1.0 + t;
            }
        }
        fn project_pressure_bdr(&self, t: f64, out: &mut [f64]) {
            for &d in &self.pres_ess {
                out[d] = 2.0 + t;
            }
        }
        fn assemble_ftext_bdr(&self, ftext: &[f64]) -> Vec<f64> {
            vec![ftext[0] * 0.25; self.np]
        }
        fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
            vec![0.5 + t; self.np]
        }
        fn assemble_accel(&self, t: f64) -> Vec<f64> {
            vec![t; self.nv]
        }
        fn mean_zero(&self, v: &mut [f64]) {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            for x in v.iter_mut() {
                *x -= mean;
            }
        }
        fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
            self.cv.set(dt * u.iter().fold(0.0_f64, |m, &v| m.max(v.abs())));
            self.cv.get()
        }
        fn rotational_divergence_pressure(&self, _u: &[f64]) -> Vec<f64> {
            match self.rot_div.borrow().as_ref() {
                Some(v) => v.clone(),
                None => panic!(
                    "ToyDisc: rotational_divergence_pressure override not set — \
                     the loud-default path (NavierConfig::rotational on a disc \
                     without the representation) aborts here"
                ),
            }
        }
        fn eliminate_bc(
            &self,
            mat: &mut CsrMatrix<f64>,
            rhs: &mut [f64],
            ess: &[usize],
            values: &[f64],
        ) {
            self.elim_calls.borrow_mut().push(ess.to_vec());
            for (k, &d) in ess.iter().enumerate() {
                let mut diag = 0.0;
                for j in mat.row_ptr[d]..mat.row_ptr[d + 1] {
                    let c = mat.col_idx[j] as usize;
                    if c == d {
                        diag = mat.values[j];
                    } else {
                        // Reaction from the TRUE column entry `A[c,d]` only
                        // (MFEM DIAG_KEEP `EliminateRowCol`, cf.
                        // `apply_dirichlet_keep_diag`, D409/D433) — not from
                        // the pivot-row entry `A[d,c]`.
                        if let Some(p) = mat.find_entry(c, d) {
                            let a_cd = mat.values[p];
                            if a_cd != 0.0 {
                                rhs[c] -= a_cd * values[k];
                                mat.values[p] = 0.0;
                            }
                        }
                    }
                }
                for j in mat.row_ptr[d]..mat.row_ptr[d + 1] {
                    if mat.col_idx[j] as usize != d {
                        mat.values[j] = 0.0;
                    }
                }
                rhs[d] = diag * values[k];
            }
        }
    }

    /// The driver runs one step of the pure-Neumann (no pressure BC) path and
    /// records the C++ calling convention of the forms.
    #[test]
    fn toy_driver_pure_neumann_step() {
        let disc = ToyDisc::new(2, 1, true);
        let mut s = NavierSolver::new(
            disc,
            0.5,
            NavierConfig { verbose: false, ..Default::default() },
        );
        s.velocity_mut().copy_from_slice(&[1.0, 0.0]);
        s.setup(0.1);
        // `assemble_helmholtz(1/dt, kin_vis)` at Setup.
        assert_eq!(s.disc.h_calls.borrow().len(), 1);
        assert_eq!(s.disc.h_calls.borrow()[0], (1.0 / 0.1, 0.5));

        let mut t = 0.0;
        s.step(&mut t, 0.1, 0, false);
        assert_eq!(t, 0.1);
        assert!(s.velocity().iter().all(|v| v.is_finite()));
        assert!(s.pressure().iter().all(|v| v.is_finite()));
        // Step 0 uses BDF1, so the reassembly must use (bd0/dt, kin_vis).
        assert_eq!(s.disc.h_calls.borrow().len(), 2);
        assert_eq!(s.disc.h_calls.borrow()[1], (1.0 / 0.1, 0.5));
        // The velocity Dirichlet DOF is eliminated with the projected value.
        assert_eq!(s.disc.elim_calls.borrow().len(), 1);
        assert_eq!(s.disc.elim_calls.borrow()[0], vec![0]);
        // The essential entry of the provisional velocity equals the data.
        assert!((s.provisional_velocity()[0] - 1.1).abs() < 1e-12);
        // CFL uses the accepted velocity and the time step.
        let cfl = s.compute_cfl(s.velocity(), 0.1);
        assert!(cfl > 0.0);
        assert!(s.iter_mvsolve() > 0);
        assert!(s.iter_hsolve() > 0);
    }

    /// The `res_*` accessors pair with the `iter_*` ones: they expose the
    /// final preconditioned residual norm `√((B r, r))` each sub-solve of
    /// the last `step` recorded in `res_mvsolve`/`res_spsolve`/`res_hsolve`
    /// (the MFEM `MvInv/SpInv/HInv->GetFinalNorm()` triple).  `0.0` on a
    /// fresh solver, overwritten by every sub-solve; on the toy diagonal
    /// operators the CG can land on an exactly-zero residual (e.g. the
    /// 1-DOF pure-Neumann pressure RHS vanishes under `Orthogonalize`, and
    /// `Mv = 2·I` with the matching Jacobi preconditioner solves exactly),
    /// so only finiteness/non-negativity is asserted here — strict
    /// positivity of the norms is covered by the real-operator cavity
    /// driver on the pro-fluid side (`step_monitors_over_one_cavity_step`).
    #[test]
    fn res_accessors_track_last_step_norms() {
        // Fresh solver: no sub-solve has ever run.
        let mut s = NavierSolver::new(
            ToyDisc::new(2, 1, true),
            0.5,
            NavierConfig { verbose: false, ..Default::default() },
        );
        assert_eq!(s.res_mvsolve(), 0.0);
        assert_eq!(s.res_spsolve(), 0.0);
        assert_eq!(s.res_hsolve(), 0.0);

        s.velocity_mut().copy_from_slice(&[1.0, 0.0]);
        s.setup(0.1);
        let mut t = 0.0;
        s.step(&mut t, 0.1, 0, false);
        for (name, res) in [
            ("mvsolve", s.res_mvsolve()),
            ("spsolve", s.res_spsolve()),
            ("hsolve", s.res_hsolve()),
        ] {
            assert!(res.is_finite(), "{name} final norm not finite: {res}");
            assert!(res >= 0.0, "{name} final norm negative: {res}");
        }
    }

    /// D433 regression: the `eliminate_bc` test double must take reactions
    /// from the **true column entries** `A[other, dof]` — MFEM DIAG_KEEP
    /// `SparseMatrix::EliminateRowCol` semantics, cf. `CsrMatrix::
    /// apply_dirichlet_keep_diag` (D409) — not from the pivot-row entries
    /// `A[dof, other]`, which coincide only for numerically symmetric
    /// matrices.  Failed on the pre-D433 double (row-driven reaction); the
    /// red run is archived under `tmp/d410/d433_red.txt`.
    #[test]
    fn eliminate_bc_double_uses_true_column_reaction() {
        let disc = ToyDisc::new(2, 1, true);
        // Structurally symmetric, numerically **antisymmetric** 2×2:
        // A[0,1] = +3, A[1,0] = −3 (saddle-coupling style).
        let mut a = CsrMatrix::new_empty(2, 2);
        a.row_ptr = vec![0, 2, 4];
        a.col_idx = vec![0, 1, 0, 1];
        a.values = vec![4.0, 3.0, -3.0, 5.0];
        let mut rhs = [10.0_f64, 20.0];
        disc.eliminate_bc(&mut a, &mut rhs, &[0], &[2.0]);
        // DIAG_KEEP: rhs[0] = A[0,0]·2 = 8; reaction from the TRUE column
        // entry A[1,0] = −3: rhs[1] = 20 − (−3)·2 = 26.  The pre-D433
        // row-driven variant used A[0,1] = +3 and produced 20 − 6 = 14.
        assert!((rhs[0] - 8.0).abs() < 1e-14, "rhs[0] = {} want 8", rhs[0]);
        assert!(
            (rhs[1] - 26.0).abs() < 1e-14,
            "rhs[1] = {} want 26 (reaction must come from A[1,0] = -3, not A[0,1] = +3)",
            rhs[1]
        );
        assert!(a.get(0, 1).abs() < 1e-14, "row off-diag not eliminated");
        assert!(a.get(1, 0).abs() < 1e-14, "column off-diag not eliminated");
    }

    /// The *fully periodic* configuration (both essential DOF lists empty, as
    /// for a mesh without boundary elements — MFEM's `periodic-square.mesh`):
    /// no `FormSystemMatrix`/`FormLinearSystem` elimination at all, the
    /// pressure solve runs with `OrthoSolver(GSSmoother)`, `resp` is
    /// orthogonalised and `pn` is mean-zeroed.  This is the path
    /// `navier_shear` exercises.
    #[test]
    fn toy_driver_fully_periodic_step() {
        let mut disc = ToyDisc::new(2, 2, true);
        disc.vel_ess.clear();
        disc.pres_ess.clear();
        let mut s = NavierSolver::new(
            disc,
            0.25,
            NavierConfig { verbose: false, ..Default::default() },
        );
        s.velocity_mut().copy_from_slice(&[1.0, -0.5]);
        s.setup(0.2);
        let mut t = 0.0;
        s.step(&mut t, 0.2, 0, false);
        assert_eq!(t, 0.2);
        assert!(s.velocity().iter().all(|v| v.is_finite()));
        assert!(s.pressure().iter().all(|v| v.is_finite()));
        // No essential DOF was ever eliminated.
        assert!(s.disc.elim_calls.borrow().is_empty());
        // Both `H` and `Sp` keep their diagonal (no `DIAG_KEEP` zeroing) and
        // the pressure is mean-zeroed (`MeanZero`).
        let pmean = s.pressure().iter().sum::<f64>() / s.pressure().len() as f64;
        assert!(pmean.abs() < 1e-15, "mean p = {pmean}");
        // `H` is reassembled with (bd0/dt, kin_vis) = (1/0.2, 0.25).
        assert_eq!(s.disc.h_calls.borrow().len(), 2);
        assert_eq!(s.disc.h_calls.borrow()[1], (1.0 / 0.2, 0.25));
    }

    /// The pressure-Dirichlet path (project + eliminate + no mean zero) is
    /// exercised, and a provisional step must not advance the time.
    #[test]
    fn toy_driver_pressure_bc_and_provisional() {
        let disc = ToyDisc::new(2, 2, false);
        let mut s = NavierSolver::new(
            disc,
            1.0,
            NavierConfig { verbose: false, ..Default::default() },
        );
        s.velocity_mut().copy_from_slice(&[0.5, -0.25]);
        s.setup(0.05);
        let mut t = 0.0;
        s.step(&mut t, 0.05, 0, true);
        // Provisional: time is untouched and the history is not rotated.
        assert_eq!(t, 0.0);
        assert_eq!(s.dthist, [0.05, 0.0, 0.0]);
        // The pressure essential DOF was eliminated and the boundary data
        // projected to 2.0 + t_now = 2.05, so the identity system keeps it.
        assert_eq!(s.disc.pres_ess, vec![0]);
        assert!(
            (s.pressure()[0] - 2.05).abs() < 1e-9,
            "p0 = {}",
            s.pressure()[0]
        );
        // A second, accepted step rotates the history.
        s.step(&mut t, 0.05, 1, false);
        assert_eq!(t, 0.05);
        assert_eq!(s.dthist, [0.05, 0.05, 0.0]);
    }

    /// D952 rotational-correction wiring (kernel level, hand-checked on the
    /// toy disc): with `NavierConfig::rotational` the step pressure gains
    /// exactly `−ν·δp` (the disc's `rotational_divergence_pressure` output),
    /// essential pressure DOFs keep their Dirichlet value, and the
    /// pure-Neumann path re-applies `MeanZero` — i.e. the correction rides
    /// on top of an otherwise identical flag-off step.
    #[test]
    fn rotational_correction_updates_pressure_by_minus_nu_div_u() {
        let nu = 0.37;
        let div = vec![3.0_f64, -1.0];
        let run = |rotational: bool, dirichlet: bool| -> Vec<f64> {
            let disc = ToyDisc::new(2, 2, dirichlet);
            disc.rot_div.replace(Some(div.clone()));
            let mut s = NavierSolver::new(
                disc,
                nu,
                NavierConfig { verbose: false, rotational, ..Default::default() },
            );
            s.velocity_mut().copy_from_slice(&[0.5, -0.25]);
            s.setup(0.1);
            let mut t = 0.0;
            s.step(&mut t, 0.1, 0, false);
            s.pressure().to_vec()
        };
        // Pure-Neumann rig (vel_ess = [0], no pressure BC): the flag-off
        // step is mean-zero, and the correction shifts by −ν·(δp − mean δp).
        let base = run(false, true);
        let rot = run(true, true);
        let mean_div = (div[0] + div[1]) / 2.0;
        for i in 0..2 {
            let expect = base[i] - nu * (div[i] - mean_div);
            assert!(
                (rot[i] - expect).abs() < 1e-12,
                "pure-Neumann pn[{i}]: {} vs expected {expect}",
                rot[i]
            );
        }
        // Pressure-Dirichlet rig (pres_ess = [0]): the essential dof keeps
        // the projected BC value; the free dof takes the raw `−ν·δp` with no
        // mean shift.
        let base_d = run(false, false);
        let rot_d = run(true, false);
        assert!(
            (rot_d[0] - base_d[0]).abs() < 1e-12,
            "essential pressure dof moved: {} vs {}",
            rot_d[0],
            base_d[0]
        );
        assert!(
            (rot_d[1] - (base_d[1] - nu * div[1])).abs() < 1e-12,
            "free pressure dof: {} vs expected {}",
            rot_d[1],
            base_d[1] - nu * div[1]
        );
    }

    /// The loud default: `NavierConfig::rotational` on a discretization
    /// without `rotational_divergence_pressure` support must abort, not
    /// silently run the classical scheme as if the flag were off.
    #[test]
    #[should_panic(expected = "rotational_divergence_pressure")]
    fn rotational_flag_without_disc_support_aborts_loudly() {
        let disc = ToyDisc::new(2, 2, true); // rot_div stays None
        let mut s = NavierSolver::new(
            disc,
            0.1,
            NavierConfig {
                verbose: false,
                rotational: true,
                ..Default::default()
            },
        );
        s.setup(0.1);
        let mut t = 0.0;
        s.step(&mut t, 0.1, 0, false);
    }

    /// D952-H2 split-sequencing wiring (kernel level, hand-checked on the
    /// toy disc): one accepted step of `PressureMode::Incremental` on
    /// `ToyDisc(2, 2, dirichlet=false)` (ν = 1, dt = 0.05, u₀ = (0.5, −0.25),
    /// BDF1 bootstrap, so `p_ext = pⁿ` with the essential dof already at its
    /// projected data `p_ext = (p_D, 0) = (2.05, 0)`) must produce, by hand
    /// expansion of every stage the step runs,
    ///
    /// ```text
    ///   fext = Mv⁻¹(−N(u⁰) + f) + BDF/dt  = (9.9, −4.9125)
    ///   u*   = H⁻¹·(Mv·fext − G·p_ext)    = (5917, −4037)/13440
    ///          (b2ext = (71/4, −95/8); H diag 41, off 1, det 1680)
    ///   b1   = (bd0/dt)·(Gᵀu* − g_bdr) = (0, −31171/2688)
    ///          (Gᵀ = B − D; essential data p_D − p_ext|_Γ = 0)
    ///   p⁺   = (p_D, p_ext[1] + δp[1])    = (2.05, −11.5963541667)
    ///   u⁺   = u* − (dt/bd0)·Mv⁻¹·(G·p⁺)  = (0.6789118304, −0.0617116815)
    /// ```
    ///
    /// Every stage of the split sequencing (extrapolated-pressure Helmholtz
    /// drive, raw increment Poisson, mass-form projection, essential-dof
    /// composition) is pinned — a wiring loss or sign flip in any of them
    /// moves these numbers and the pin goes red.
    #[test]
    fn incremental_mode_split_step_hand_checked() {
        let disc = ToyDisc::new(2, 2, false); // vel_ess = [], pres_ess = [0]
        let mut s = NavierSolver::new(
            disc,
            1.0,
            NavierConfig {
                verbose: false,
                pressure_mode: PressureMode::Incremental,
                rtol_mvsolve: 1.0e-12,
                rtol_spsolve: 1.0e-12,
                rtol_hsolve: 1.0e-12,
                ..Default::default()
            },
        );
        s.velocity_mut().copy_from_slice(&[0.5, -0.25]);
        s.setup(0.05);
        let mut t = 0.0;
        s.step(&mut t, 0.05, 0, false);
        assert_eq!(t, 0.05);
        let p = s.pressure();
        let u = s.velocity();
        let want_p = [2.05_f64, -31171.0 / 2688.0];
        let want_u = [0.6789118303571429, -33177.0 / 537600.0];
        for k in 0..2 {
            assert!(
                (p[k] - want_p[k]).abs() < 1e-9,
                "pressure[{}] = {} vs hand-computed {}",
                k,
                p[k],
                want_p[k]
            );
            assert!(
                (u[k] - want_u[k]).abs() < 1e-9,
                "velocity[{}] = {} vs hand-computed {}",
                k,
                u[k],
                want_u[k]
            );
        }
    }

    /// The incremental composition keeps the pressure-Dirichlet convention:
    /// the essential dof holds the projected BC data bit-exactly (the
    /// increment's own essential entries carry `p_D − p_ext` and are NOT
    /// added on top), the free dof is the composed `p_ext + δp` and differs
    /// from the classical free dof (mode-active discriminator).
    #[test]
    fn incremental_mode_pressure_dirichlet_composes_bc_and_increment() {
        let run = |mode: PressureMode| -> Vec<f64> {
            let disc = ToyDisc::new(2, 2, false); // pres_ess = [0]
            let mut s = NavierSolver::new(
                disc,
                1.0,
                NavierConfig {
                    verbose: false,
                    pressure_mode: mode,
                    rtol_spsolve: 1.0e-12,
                    ..Default::default()
                },
            );
            s.velocity_mut().copy_from_slice(&[0.5, -0.25]);
            s.setup(0.05);
            let mut t = 0.0;
            s.step(&mut t, 0.05, 0, false);
            s.step(&mut t, 0.05, 1, false);
            s.pressure().to_vec()
        };
        let base = run(PressureMode::Classical);
        let incr = run(PressureMode::Incremental);
        // t after two accepted steps = 0.1; the essential dof holds 2.0 + 0.1.
        assert!(
            (incr[0] - (2.0 + 0.1)).abs() < 1e-12,
            "essential pressure dof moved off the BC data: {}",
            incr[0]
        );
        assert!(
            (incr[1] - base[1]).abs() > 1e-6,
            "incremental free dof {} identical to classical {} — mode is a no-op",
            incr[1],
            base[1]
        );
    }

    /// The loud contract: `pressure_mode` incremental families keep the
    /// pressure as accumulated state, and a provisional step (which
    /// overwrites `pn` without rotating the history — MFEM `X1` aliasing
    /// semantics) would corrupt the state on a retry.  It must abort, not
    /// silently pollute.  Both incremental modes share the guard.
    #[test]
    fn incremental_mode_rejects_provisional_steps() {
        for mode in [PressureMode::Incremental, PressureMode::IncrementalSelfConsistent] {
            let aborted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let disc = ToyDisc::new(2, 2, true);
                let mut s = NavierSolver::new(
                    disc,
                    0.1,
                    NavierConfig {
                        verbose: false,
                        pressure_mode: mode,
                        ..Default::default()
                    },
                );
                s.setup(0.1);
                let mut t = 0.0;
                s.step(&mut t, 0.1, 0, true);
            }))
            .is_err();
            assert!(
                aborted,
                "{mode:?}: a provisional step did not abort — the accumulated \
                 pressure state would be corrupted on a retry"
            );
        }
    }

    /// The D952-H2' composition guard: the Timmermans rotational correction
    /// edits the ABSOLUTE pressure state between the Poisson solve and the
    /// projection; an increment-only mass projection would project the
    /// UNCORRECTED increment and silently drop the rotational term.  The
    /// combination aborts loudly instead.
    #[test]
    #[should_panic(expected = "rotational + increment-only projection is not composed")]
    fn sc_increment_projection_rejects_rotational_mix() {
        let disc = ToyDisc::new(2, 2, true);
        let mut s = NavierSolver::new(
            disc,
            0.1,
            NavierConfig {
                verbose: false,
                pressure_mode: PressureMode::IncrementalSelfConsistent,
                increment_projection_absolute: false,
                rotational: true,
                ..Default::default()
            },
        );
        s.setup(0.1);
        let mut t = 0.0;
        s.step(&mut t, 0.1, 0, false);
    }

    /// D952-H2' self-consistent sequencing, TEXTBOOK arm (kernel level,
    /// hand-checked on the toy disc): one accepted step of
    /// `IncrementalSelfConsistent` with the GMS §2.2 increment-only
    /// projection on `ToyDisc(2, 2, dirichlet=false)` (ν = 1, dt = 0.05,
    /// u₀ = (0.5, −0.25), BDF1 bootstrap, `p_ext = (p_D, 0) = (2.05, 0)`).
    /// The shared stages match `incremental_mode_split_step_hand_checked`
    /// (`u* = (5917, −4037)/13440`, raw data `b1 = (0, −31171/2688)`); the
    /// SC left end solves the constraint equation
    /// `(−D·Mv⁻¹·G)·δp = b1` — with the toy `Q = D·Mv⁻¹·G = ones(2×2)` and
    /// the essential reaction `A_sc[·,ess]·vals = 0` (vals = p_D − p_ext =
    /// 0) the eliminated free equation is `−δp₁ = −31171/2688`, so
    ///
    /// ```text
    ///   δp   = (0, +31171/2688)              (rr134 raw form: −31171/2688)
    ///   p⁺   = p_ext + δp  = (41/20, 31171/2688)
    ///   u⁺   = u* − (dt/bd0)·Mv⁻¹·(G·δp) = (16165, −63467)/107520 .
    /// ```
    ///
    /// The projection target is the INCREMENT — the textbook composition the
    /// round-134 kernel deviated from.  Any wiring loss (operator sign, data
    /// functional, projection target, essential bookkeeping) moves these
    /// numbers and the pin goes red.  Cost KPI: the outer Poisson CG
    /// converges on the toy's 1-free-dof eliminated system in one iteration
    /// (two `apply_sc` calls: the zero initial residual costs no inner
    /// work, the direction apply one inner `Mv⁻¹` CG) — `iter_sc_inner = 1`.
    #[test]
    fn incremental_sc_increment_projection_hand_checked() {
        let disc = ToyDisc::new(2, 2, false); // vel_ess = [], pres_ess = [0]
        let mut s = NavierSolver::new(
            disc,
            1.0,
            NavierConfig {
                verbose: false,
                pressure_mode: PressureMode::IncrementalSelfConsistent,
                increment_projection_absolute: false,
                rtol_mvsolve: 1.0e-12,
                rtol_spsolve: 1.0e-12,
                rtol_hsolve: 1.0e-12,
                ..Default::default()
            },
        );
        s.velocity_mut().copy_from_slice(&[0.5, -0.25]);
        s.setup(0.05);
        let mut t = 0.0;
        s.step(&mut t, 0.05, 0, false);
        assert_eq!(t, 0.05);
        let p = s.pressure();
        let u = s.velocity();
        let want_p = [2.05_f64, 31171.0 / 2688.0];
        let want_u = [16165.0 / 107520.0, -63467.0 / 107520.0];
        for k in 0..2 {
            assert!(
                (p[k] - want_p[k]).abs() < 1e-9,
                "pressure[{}] = {} vs hand-computed {}",
                k,
                p[k],
                want_p[k]
            );
            assert!(
                (u[k] - want_u[k]).abs() < 1e-9,
                "velocity[{}] = {} vs hand-computed {}",
                k,
                u[k],
                want_u[k]
            );
        }
        assert_eq!(s.iter_spsolve(), 1, "outer SC CG iteration count");
        assert_eq!(s.iter_sc_inner(), 1, "nested inner Mv-CG KPI");
    }

    /// D952-H2' self-consistent sequencing, ABSOLUTE arm (the round-134
    /// composition kept, only the left end replaced — the registered H2'
    /// form): same stage as the textbook pin, but the constraint equation
    /// carries the projection's absolute target,
    /// `(−D·Mv⁻¹·G)·δp = b1 + Q·p_ext` (`Q·p_ext` free row = `−2.05` on the
    /// toy), so `δp₁ = 31171/2688 − 41/20 = 128303/13440` and the composed
    /// absolute pressure satisfies the same constraint manifold
    /// (`Σp⁺ = 41/20 + 128303/13440 = 31171/2688 = −b1₁`):
    ///
    /// ```text
    ///   p⁺   = (p_D, 128303/13440) = (41/20, 128303/13440)
    ///   u⁺   = u* − (dt/bd0)·Mv⁻¹·(G·p⁺) = (80825, −317335)/537600 .
    /// ```
    ///
    /// (On the toy's degenerate all-ones `G` the velocity coincides with
    /// the textbook arm's — `G·v = (Σv)(1,1)` and both targets share
    /// `Σ = 31171/2688`; the DISCRIMINATOR here is the free pressure dof:
    /// 128303/13440 vs the textbook arm's 31171/2688 vs rr134's
    /// −31171/2688.)  Cost KPI: one extra nonzero operator apply for
    /// `Q·p_ext` (the zero-valued essential reaction costs no inner work)
    /// plus the single outer iteration — `iter_sc_inner = 2`.
    #[test]
    fn incremental_sc_absolute_projection_hand_checked() {
        let disc = ToyDisc::new(2, 2, false); // vel_ess = [], pres_ess = [0]
        let mut s = NavierSolver::new(
            disc,
            1.0,
            NavierConfig {
                verbose: false,
                pressure_mode: PressureMode::IncrementalSelfConsistent,
                increment_projection_absolute: true,
                rtol_mvsolve: 1.0e-12,
                rtol_spsolve: 1.0e-12,
                rtol_hsolve: 1.0e-12,
                ..Default::default()
            },
        );
        s.velocity_mut().copy_from_slice(&[0.5, -0.25]);
        s.setup(0.05);
        let mut t = 0.0;
        s.step(&mut t, 0.05, 0, false);
        let p = s.pressure();
        let u = s.velocity();
        let want_p = [2.05_f64, 128303.0 / 13440.0];
        let want_u = [80825.0 / 537600.0, -317335.0 / 537600.0];
        for k in 0..2 {
            assert!(
                (p[k] - want_p[k]).abs() < 1e-9,
                "pressure[{}] = {} vs hand-computed {}",
                k,
                p[k],
                want_p[k]
            );
            assert!(
                (u[k] - want_u[k]).abs() < 1e-9,
                "velocity[{}] = {} vs hand-computed {}",
                k,
                u[k],
                want_u[k]
            );
        }
        assert_eq!(s.iter_sc_inner(), 2, "nested inner Mv-CG KPI");
    }
}
