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

use std::time::Instant;

use fem_linalg::CsrMatrix;

use crate::sli::{solve_cg_mfem, SliOptions};

/// MFEM `NAVIER_VERSION`.
pub const NAVIER_VERSION: &str = "0.1";

// ─── Solver configuration (navier_solver.hpp members) ────────────────────────

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
    fn assemble_divergence(&self) -> CsrMatrix<f64>;
    /// `G_form` — `G[(k,c),i] = ∫ φ_k ∂φ_i/∂x_c dx` (MFEM
    /// `GradientIntegrator`, `n_vel × n_pres`; `G = Dᵀ` on the same spaces).
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

    /// `∇×∇×u` at the velocity DOFs, i.e. the C++ `ComputeCurl2D` applied
    /// twice (without the `kin_vis` factor, which the driver applies).
    fn curl_curl(&self, u: &[f64]) -> Vec<f64>;
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

    vel: VelState,
    pn: Vec<f64>,

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
            vel: VelState::new(nv),
            pn: vec![0.0; np],
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
    /// `MvInv->GetNumIterations()` of the last step.
    pub fn iter_mvsolve(&self) -> i32 {
        self.iter_mvsolve
    }
    /// `SpInv->GetNumIterations()` of the last step.
    pub fn iter_spsolve(&self) -> i32 {
        self.iter_spsolve
    }
    /// `HInv->GetNumIterations()` of the last step.
    pub fn iter_hsolve(&self) -> i32 {
        self.iter_hsolve
    }

    /// MFEM `NavierSolver::Step(time, dt, cur_step, provisional = false)`.
    ///
    /// With `provisional = false` the computed step is accepted immediately:
    /// `UpdateTimestepHistory(dt)` is called and `time += dt`.
    pub fn step(&mut self, time: &mut f64, dt: f64, cur_step: i32, provisional: bool) {
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

        // Extrapolated f^{n+1}: the acceleration coefficient time is set to
        // t + dt before the linear form is reassembled.
        let accel = self.disc.assemble_accel(t_now);

        // Nonlinear extrapolated terms: N(u) = -C(u)·u on un, unm1, unm2.
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

        // Pressure Poisson: Lext = ab1·un + ab2·unm1 + ab3·unm2, then
        // Lext *= ν after the two curl applications (`ComputeCurl2D`).
        t_sub = Instant::now();
        let mut lext = vec![0.0_f64; nv];
        for i in 0..nv {
            lext[i] = self.ab1 * self.vel.un[i]
                + self.ab2 * self.vel.unm1[i]
                + self.ab3 * self.vel.unm2[i];
        }
        let cc = self.disc.curl_curl(&lext);
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
        if !pres_ess.is_empty() {
            let vals: Vec<f64> = pres_ess.iter().map(|&d| pn[d]).collect();
            self.disc
                .eliminate_bc(&mut self.sp, &mut b1, &pres_ess, &vals);
        }

        // SpInv->Mult(B1, X1): `iterative_mode = true`; with no pressure
        // Dirichlet BCs the preconditioner is `OrthoSolver(GSSmoother)`,
        // otherwise the bare smoother.
        {
            let sp = &self.sp;
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
                // `GSSmoother::Mult` zeroes its output (`iterative_mode` is
                // false, as propagated by `OrthoSolver`), then runs one
                // forward and one backward sweep.
                out.fill(0.0);
                gs_forward(sp, rin, out);
                gs_backward(sp, rin, out);
                if ortho {
                    z_ortho.copy_from_slice(out);
                    orthogonalize(&mut z_ortho);
                    out.copy_from_slice(&z_ortho);
                }
            };
            let t_solve = Instant::now();
            let res = solve_cg_mfem(
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
            );
            self.rt_spsolve = t_solve.elapsed().as_secs_f64();
            self.iter_spsolve = res.iterations;
            self.res_spsolve = res.final_norm;
        }

        if pres_ess.is_empty() {
            // MeanZero(pn_gf): remove the pressure nullspace.
            self.disc.mean_zero(&mut pn);
        }
        self.pn = pn;

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
                        let a = mat.values[j];
                        if a != 0.0 {
                            rhs[c] -= a * values[k];
                            if let Some(p) = mat.find_entry(c, d) {
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
}
