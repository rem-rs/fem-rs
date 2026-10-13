//! # Miniapp Joule — transient magnetics and Joule heating
//!
//! 1:1 port of MFEM `miniapps/electromagnetics/joule.cpp` +
//! `joule_solver.{hpp,cpp}` (MFEM 4.10), serial rank count.
//!
//! The C++ solves a time-dependent eddy-current problem with Joule heating
//! (joule.cpp:16-31):
//!
//! ```text
//!   Div sigma Grad Phi = 0
//!   sigma E = Curl B/mu - sigma grad Phi
//!   dB/dt = - Curl E
//!   F = -k Grad T
//!   c dT/dt = -Div(F) + sigma E.E
//! ```
//!
//! The state is one `BlockVector` (`F`, joule.cpp:479-491) holding six fields
//! in five FE spaces, advanced by `MagneticDiffusionEOperator::ImplicitSolve`
//! (joule_solver.cpp:401-640) through MFEM's `ODESolver` family
//! (`-s 1|2|3|22|23|34`).
//!
//! ## Round 150 — the coupled operator + Backward-Euler time loop
//!
//! `MagneticDiffusionEOperator` is ported and the run goes through the full
//! time loop (the stdout `step … dot(E, J) = …` lines are reproduced
//! **digit-for-digit** at the C++ print precision): on `-m
//! data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5` the port prints
//! `1.78064984` / `5.12554673`, exactly the MFEM 4.10 MPI oracle (np 1).
//! The C++ `ImplicitSolve` writes the advanced P/E/F/W state directly
//! (MakeRef'd views) while returning B/T slopes, so the step loop unfolds
//! the `BackwardEulerSolver::Step` semantics inline instead of using the
//! fem-solver ODE trait (whose slope-vs-state dichotomy cannot express the
//! mixed convention).
//!
//! Solver mapping (tol 1e-9 relative, max 1000 iterations both sides):
//! A0 PCG+AMG (`fem_solver::amg::solve_amg_cg`), A1 PCG+AMS
//! (`solve_pcg_ams`, P1 discrete gradient — hypre's AMS convention), A2 /
//! M3 PCG+diag (`solve_pcg_dsmoother`).  **Registered preconditioner gap**:
//! C++ preconditions A2 with `HypreADS`; fem-rs has no ADS, so the converged
//! solutions are the parity target, not the iteration paths.  Quadrature
//! orders follow MFEM's per-integrator defaults (a0 `OrderGrad+GetOrder=5`,
//! A1 mass `OrderW+2·GetOrder=6` via the D680 hook, curl-curl `2·GetOrder=4`,
//! A2 mass 6 / div-div 4, M3 mass 2, weakCurl `2+2−1=3`, weakDiv 4).
//!
//! Structural fast pins (stderr, not part of the C++ stdout): the projected
//! divergence `|<φ, div B>|` of the RT curl of the ND field stays at
//! round-off (~1e-16, discrete div∘curl = 0), and the BE energy ledger
//! `Δ(c·∫T) = Σ dt·(∫W − ∫ div F)` closes to PCG tolerance.
//!
//! ## Round 106 (D1046 segment) — registration audit + `-visit` cycle 0
//!
//! * **The round-102 P0 registration for this file is misattributed**: the
//!   audit line reads "joule `-vp`/`-nbcs`/AMR(`-maxit>1`) 缺", but MFEM 4.10
//!   `joule.cpp` has **no** `-vp`, `-nbcs` or `-maxit` options at all (its full
//!   option set is `-m -rs -rp -o -s -tf -dt -mu -cnd -f -vis -visit -vs -k
//!   -print -amr -sc -debug -hl -p`).  Those three options belong to
//!   **`volta.cpp`** (`-vp` voltaic-pile params, `-nbcs` Neumann-BC surfaces,
//!   `-maxit` max-AMR-iterations) and are registered as `miniapp_volta`'s
//!   exit(3) gaps there.  Nothing in this file corresponds to them; the
//!   genuine joule residuals are the coupled time loop (below) and the
//!   `-amr 1`/`-sc 1`/`-debug 1`/`-print 1`/`-vis` paths.
//! * **`-visit` (VisItDataCollection) cycle 0 is now written** (was refused):
//!   C++ saves cycle 0 *before* the time loop (joule.cpp:573-575, all six
//!   fields zero).  The port writes the byte-verified file set
//!   `Joule_000000/{mesh,Phi,E,B,T,w,F}.000000` + `Joule_000000.mfem_root`
//!   (`--ranks 1` only): the mesh at the dc precision 6, the GF files with
//!   MFEM's `DofTransformation::TransformPrimal` **negative-zero** pattern
//!   (`+0 × (−1) → "-0"`, from the ND/RT orientation signs), and the
//!   `.mfem_root` JSON.  fem-rs's ND/RT global dof numbering is
//!   slot-for-slot MFEM's on this mesh (D120), so the -0 pattern is
//!   reproducible; the parity target is the MFEM 4.10 MPI oracle
//!   (`$HOME/mfem410_mpi`, `mpirun -np 1`).
//! * **stdout purity**: the D120/D121 instrumentation moved off stdout (it is
//!   not part of the C++ output), so the comparable stdout region — banner,
//!   options dump, skin depths, the five dof lines — is byte-identical to the
//!   oracle and the run then stops with status 3 before the time loop.
//!
//! ## Port boundary (this file)
//!
//! Done 1:1 here, verified against the C++ binary:
//! * the `display_banner` ASCII art (joule.cpp:779-788) and MFEM's
//!   `OptionsParser` dump (joule.cpp:160-216) in registration order, including
//!   joule's **duplicate `-p` registration** (problem at joule.cpp:201,
//!   send-port at joule.cpp:203 — `-p rod` still binds to the string option);
//! * the two `Skin depth …` lines (joule.cpp:222-227) with MFEM's default
//!   `operator<<` formatting (`%.6g`);
//! * the mesh read + `-rs` serial uniform refinements (joule.cpp:283, 377-380,
//!   with the `-rp` parallel refinements left to `-rp`/`-no-rp`);
//! * the four FE spaces with the C++ orders (joule.cpp:432-448):
//!   `L2(order-1)`, `ND(order)`, `RT(order-1)`, `H1(order)`;
//! * the five `Number of … unknowns` lines (joule.cpp:456-463) from
//!   `GlobalTrueVSize()`;
//! * the `true_offset` / `BlockVector` six-field layout and the six
//!   `ParGridFunction::MakeRef` views (joule.cpp:479-500) — in fem-rs the
//!   layout is over `n_local_dofs()` (= MFEM `GetVSize()`, the ghost-inclusive
//!   length) and the views are `GridFunction::make_ref`;
//! * `oper.Init(F)` (joule_solver.cpp:101-144) for the all-zero default initial
//!   condition: the C++ projects `Zero_vec`/`Zero` onto all six fields, which
//!   is exactly zero;
//! * the four material maps (joule.cpp:238-277 `sigmaMap`, `InvTcondMap`,
//!   `TcapMap`, `InvTcapMap`) as `MeshDependentCoefficient { map, scale }`
//!   (joule_solver.cpp:908-947) — the coefficient is `map[elem attribute] *
//!   scale`, and `-cnd`/`-mu`/`-f` are folded into `wj_`/`mj_`/`sj_`
//!   (joule.cpp:218-220).
//!
//! Not ported here (each is listed with the reason; every gate below still
//! exits with status 3 and a message — the coupled solves themselves **were
//! ported in round 150**, see that section above):
//! * ~~`MagneticDiffusionEOperator`'s four coupled solves~~ — **closed (round
//!   150)**: A0 PCG+AMG, A1 PCG+AMS, A2/M3 PCG+diag, the weakCurl/weakDiv/
   //! weakDivC mixed forms and the BE time loop all live in `main` now.  Note
//!   the C++ **stdout itself is not reproducible** past the dof banner: hypre
//!   prints its `BoomerAMG SETUP PARAMETERS:` block and its operator tables
//!   unconditionally (`-hl 0` does not silence them), so a byte comparison of
//!   a full joule stdout is impossible for any non-hypre implementation.  The
//!   comparable regions are the banner + options + skin depths + dof counts
//!   (byte-exact since D1046) and the
//!   `step      1,	t =  0.500,	dot(E, J) = 1.78064984` /
//!   `step      2,	t =  1.000,	dot(E, J) = 5.12554673` lines — both now
//!   reproduced digit-for-digit (round 150).  `dot(E, J)` is `ElectricLosses
//!   = m1->ParInnerProduct(E_gf, E_gf) = ∫ σ E·E`, and it depends **only** on
//!   the electromagnetic half of the operator
//!   (`P → grad P → weakCurlᵀB → M1 + dt·S1 solve → E ← E − grad P`), never on
//!   `W`, `F` or `T`;
//! * the H¹→H(curl) discrete gradient `grad` (`ParDiscreteLinearOperator` +
//!   `GradientInterpolator`, joule_solver.cpp:787-796): **landed in D110** and
//!   **wired into the time loop in round 150** (`grad` — H¹(P2) → ND2 on 3-D
//!   hexahedra, `crates/assembly/src/discrete_op.rs`, tests
//!   `crates/assembly/tests/d110_p2_nd2_gradient_3d_hex.rs`);
//! * ~~the H(curl)→H(div) discrete curl at order 2 on hexahedra~~ — **closed in
//!   D120** and wired in round 150: `DiscreteLinearOperator::curl_3d(ND2 →
//!   RT1)` covers hexahedra (was tet-only, panicking with "tet face must have
//!   an interpolation anchor"); joule's `curl->Mult(E, dB)`
//!   (joule_solver.cpp:585) uses it;
//! * ~~`GetJouleHeating`'s L2 projection of `σ|E|²`~~ — **closed in D121** and
//!   wired in round 150: `project_coefficient_element` is the trilinear-map-
//!   aware, element-aware entry (`GridFunction::evaluate_vector_at_element`
//!   goes through the geometry table, so the Hex8 corner-Jacobian caveat below
//!   no longer applies);
//! * the parallel Nédélec `ND2` / `RT1` DOF partition at ≥ 2 ranks —
//!   **closed in D412** (round 48): the H(curl)/H(div) partitions now key
//!   3-D face DOFs by canonical face geometry (3 smallest global vertex ids,
//!   position read from the min-global-id adjacent element) and edge DOFs by
//!   the global-min endpoint, so `exchange_ghost_interior_ids` no longer
//!   emits sentinel GIDs for ghost face/interior DOFs and
//!   `GhostExchange::from_partition` no longer panics; the multi-rank form of
//!   the D110 parallel test is un-`#[ignore]`d and green at 2/4 ranks
//!   (`crates/parallel/tests/d412_nd2_rt1_face_dof_partition_3d_par.rs`,
//!   `crates/parallel/tests/d110_p2_nd2_gradient_3d_hex_par.rs`);
//! * `-sc 1` static condensation (`ParBilinearForm::EnableStaticCondensation`),
//!   `-amr 1` (`GeneralRefinement` + `Rebalance`), `-debug 1`
//!   (`hypre_ParCSRMatrixPrint`), `-gfprint 1`, `-vis`, and `-visit` with
//!   `--ranks > 1` (the cycle-0 serial `-visit` save is now written — see the
//!   round-106 section; the GLVis/VisIt *display* layers are not ported);
//! * the `.gen` sample meshes (`cylinder-hex-q2.gen`, `coil.gen`, joule.cpp:89-93)
//!   need MFEM's NetCDF mesh reader; `fem_io::mfem::read_mfem_file` only reads
//!   the `.mesh` v1.0 format, so the `.mesh` fixtures
//!   (`cylinder-hex.mesh`, `cylinder-tet.mesh`) are the runnable ones.
//!
//! ## Verification
//!
//! Reference: `mpicxx … joule.cpp joule_solver.cpp ../common/pfem_extras.cpp`
//! against MFEM 4.10 (`$HOME/mfem410_mpi`), 1 rank,
//! `-m data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5 -no-vis -no-visit`
//! (the defaults `-o 2`, `-mu 1`, `-cnd 2*pi*10`, `-f 1/60`).
//!
//! Round 106: the byte-identical region now covers the whole stdout prefix —
//! banner, the whole `Options used:` dump, the blank line + the two
//! `Skin depth` lines, and the five `Number of … unknowns` lines
//! (`6456 / 2016 / 6882 / 6456 / 2443`) — on the default run, the `-rs 1`
//! refinement run (`51684 / 16128 / 55050 / 51684 / 19531`) and the `-visit`
//! run.  The `-visit` cycle-0 file set is byte-identical too: `mesh.000000`
//! (19,721 B), the six GF slices `Phi/E/B/T/w/F.000000` (including the
//! `DofTransformation` `-0` pattern of E/B/F: 2430/2616/2616 negative zeros)
//! and `Joule_000000.mfem_root` (1,771 B).
//!
//! One known stray stdout line is gone (round 107, D1049 closed): the mesh
//! reader no longer prints `Elements with wrong orientation: 70 / 252 (not
//! fixed)` — `check_element_orientation` now checks wedge/pyramid/hex
//! through the MFEM center trilinear Jacobian in the linear case too
//! (`crates/mesh/src/simplex.rs`).  The full 36-line stdout prefix compares
//! byte-identical to the C++ oracle **unfiltered** (default / `-rs 1` /
//! `-visit`); the d106 byte pin's filter for that exact string is now a
//! no-op (it lives in `examples/tests/`, outside this lane's territory).
//! The only other intended difference is the `--mesh` path string itself.
//!
//! Round-150 end-to-end numbers (MFEM 4.10 MPI oracle, np 1,
//! `-m data/cylinder-hex.mesh -p rod -no-vis -no-visit`):
//! * `-tf 1.0 -dt 0.5`: both `dot(E, J)` lines digit-for-digit
//!   (`1.78064984` / `5.12554673`);
//! * default (`-tf 100`): 200 steps, 177 of the 200 `dot(E, J)` lines
//!   digit-for-digit at the C++ 8-decimal print precision, the other 23 off
//!   by ±1 in the last digit (worst relative difference 3.0e-9 — the PCG
//!   tolerance itself; the preconditioner paths differ, see the round-150
//!   section);
//! * structural pins (stderr): `max |<φ, div B>| = 1.5e-15` over 200 steps
//!   and the BE energy ledger `Δ(c·∫T)` vs `Σ dt·(∫W − ∫ div F)` closes to
//!   3.9e-15 relative (323.532492337816052 / 323.532492337814801).
//! Evidence: `tmp/rr150lane1/` (oracle + Rust stdout/steps for both configs).
//!
//! Usage:
//!   cargo run --release --example miniapp_joule -- -m data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5 -no-vis -no-visit
//!   cargo run --release --example miniapp_joule -- -m data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5 -no-vis -visit

use std::sync::Arc;

use fem_assembly::mixed::{
    assemble_hcurl_hdiv_weak_curl, assemble_hdiv_l2_mixed, HDivL2DivIntegrator,
    HDivL2ScaledDiv,
};
use fem_assembly::standard::{
    CurlCurlIntegrator, DiffusionIntegrator, DivDivIntegrator, MassIntegrator,
    VectorMassIntegrator,
};
use fem_assembly::postproc::coefficient::{CoeffCtx, PWConstCoeff, ScalarCoeff};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{ParallelFESpace, WorkerConfig};
use fem_space::constraints::{
    boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv, form_linear_system_vdofs,
};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

// ─── Banner (joule.cpp:779-788) ─────────────────────────────────────────────

/// `display_banner`, verbatim (note the trailing spaces of each line: the C++
/// is `os << "    |    | ____  __ __|  |   ____     " << endl` etc., so the
/// lines end with a run of spaces before the newline).
const BANNER: [&str; 6] = [
    "     ____.            .__             ",
    "    |    | ____  __ __|  |   ____     ",
    "    |    |/  _ \\|  |  \\  | _/ __ \\ ",
    "/\\__|    (  <_> )  |  /  |_\\  ___/  ",
    "\\________|\\____/|____/|____/\\___  >",
    "                                \\/   ",
];

// ─── Formatting parity ──────────────────────────────────────────────────────

/// C++ ostream default `<<` for `double` (`%.6g`): 6 significant digits, the
/// trailing zeros and the decimal point stripped, exponent form outside
/// `[1e-4, 1e6)`.  Matches `OptionsParser::PrintOptions` and the skin-depth
/// line, both of which use the stream's default precision.
fn g6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= 6 {
        let s = format!("{:.*e}", 5, x);
        let (mant, e) = s.split_once('e').expect("scientific form");
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        let e: i32 = e.parse().expect("exponent");
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, x);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

// ─── Options (`OptionsParser`, joule.cpp:138-216) ───────────────────────────

/// joule's options with the C++ defaults (joule.cpp:138-158) in registration
/// order, which is also the order of the `Options used:` dump.
struct Opts {
    mesh: String,
    ser_ref_levels: u32,
    par_ref_levels: u32,
    order: u32,
    ode_solver_type: i32,
    t_final: f64,
    dt: f64,
    mu: f64,
    sigma: f64,
    freq: f64,
    visualization: bool,
    visit: bool,
    vis_steps: u32,
    gfprint: i32,
    basename: String,
    amr: i32,
    static_cond: i32,
    debug: i32,
    solver_print_level: i32,
    problem: String,
    visport: u32,
    /// Not a C++ option: the fem-rs parallel launcher's rank count.  The C++
    /// reference is `mpirun -np <ranks>`, so `--ranks 1` is the serial 1:1
    /// comparison and the default.
    ranks: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            mesh: "cylinder-hex.mesh".to_string(),
            ser_ref_levels: 0,
            par_ref_levels: 0,
            order: 2,
            ode_solver_type: 1,
            t_final: 100.0,
            dt: 0.5,
            mu: 1.0,
            sigma: 2.0 * std::f64::consts::PI * 10.0,
            freq: 1.0 / 60.0,
            visualization: true,
            visit: true,
            vis_steps: 1,
            gfprint: 0,
            basename: "Joule".to_string(),
            amr: 0,
            static_cond: 0,
            debug: 0,
            solver_print_level: 0,
            problem: "rod".to_string(),
            visport: 19916,
            ranks: 1,
        }
    }
}

impl Opts {
    /// `OptionsParser::Parse` for joule's option set.
    ///
    /// joule registers `-p` twice (joule.cpp:201 problem, joule.cpp:203
    /// send-port); MFEM resolves `-p <string>` to the *string* option, so a
    /// non-numeric value after `-p` sets `problem` and a numeric one sets
    /// `visport` — the C++ dump of `-p rod` shows `--problem rod` with
    /// `--send-port` left at its default.
    fn parse(args: &[String]) -> Self {
        let mut o = Opts::default();
        let mut i = 0;
        while i < args.len() {
            let a = args[i].as_str();
            let val = |i: &mut usize| -> String {
                *i += 1;
                args.get(*i).cloned().unwrap_or_default()
            };
            match a {
                "-m" | "--mesh" => o.mesh = val(&mut i),
                "-rs" | "--refine-serial" => {
                    o.ser_ref_levels = val(&mut i).parse().expect("bad -rs")
                }
                "-rp" | "--refine-parallel" => {
                    o.par_ref_levels = val(&mut i).parse().expect("bad -rp")
                }
                "-o" | "--order" => o.order = val(&mut i).parse().expect("bad -o"),
                "-s" | "--ode-solver" => {
                    o.ode_solver_type = val(&mut i).parse().expect("bad -s")
                }
                "-tf" | "--t-final" => o.t_final = val(&mut i).parse().expect("bad -tf"),
                "-dt" | "--time-step" => o.dt = val(&mut i).parse().expect("bad -dt"),
                "-mu" | "--permeability" => {
                    o.mu = val(&mut i).parse().expect("bad -mu")
                }
                "-cnd" | "--sigma" => o.sigma = val(&mut i).parse().expect("bad -cnd"),
                "-f" | "--frequency" => o.freq = val(&mut i).parse().expect("bad -f"),
                "-vis" | "--visualization" => o.visualization = true,
                "-no-vis" | "--no-visualization" => o.visualization = false,
                "-visit" | "--visit" => o.visit = true,
                "-no-visit" | "--no-visit" => o.visit = false,
                "-vs" | "--visualization-steps" => {
                    o.vis_steps = val(&mut i).parse().expect("bad -vs")
                }
                "-k" | "--outputfilename" => o.basename = val(&mut i),
                "-print" | "--print" => o.gfprint = val(&mut i).parse().expect("bad -print"),
                "-amr" | "--amr" => o.amr = val(&mut i).parse().expect("bad -amr"),
                "-sc" | "--static-condensation" => {
                    o.static_cond = val(&mut i).parse().expect("bad -sc")
                }
                "-debug" | "--debug" => o.debug = val(&mut i).parse().expect("bad -debug"),
                "-hl" | "--hypre-print-level" => {
                    o.solver_print_level = val(&mut i).parse().expect("bad -hl")
                }
                "-p" => {
                    let v = val(&mut i);
                    // The duplicate registration: an integer binds to the
                    // send-port option, anything else to the problem name.
                    if let Ok(port) = v.parse::<u32>() {
                        o.visport = port;
                    } else {
                        o.problem = v;
                    }
                }
                "--problem" => o.problem = val(&mut i),
                "--send-port" => o.visport = val(&mut i).parse().expect("bad --send-port"),
                "--ranks" => o.ranks = val(&mut i).parse().expect("bad --ranks"),
                _ => {}
            }
            i += 1;
        }
        o
    }

    /// `OptionsParser::PrintOptions` — same lines, same order, same `%.6g`
    /// value formatting as the C++ stream.
    fn print_options(&self) {
        println!("Options used:");
        println!("   --mesh {}", self.mesh);
        println!("   --refine-serial {}", self.ser_ref_levels);
        println!("   --refine-parallel {}", self.par_ref_levels);
        println!("   --order {}", self.order);
        println!("   --ode-solver {}", self.ode_solver_type);
        println!("   --t-final {}", g6(self.t_final));
        println!("   --time-step {}", g6(self.dt));
        println!("   --permeability {}", g6(self.mu));
        println!("   --sigma {}", g6(self.sigma));
        println!("   --frequency {}", g6(self.freq));
        println!(
            "   {}",
            if self.visualization { "--visualization" } else { "--no-visualization" }
        );
        println!("   {}", if self.visit { "--visit" } else { "--no-visit" });
        println!("   --visualization-steps {}", self.vis_steps);
        println!("   --outputfilename {}", self.basename);
        println!("   --print {}", self.gfprint);
        println!("   --amr {}", self.amr);
        println!("   --static-condensation {}", self.static_cond);
        println!("   --debug {}", self.debug);
        println!("   --hypre-print-level {}", self.solver_print_level);
        println!("   --problem {}", self.problem);
        println!("   --send-port {}", self.visport);
    }
}

// ─── MeshDependentCoefficient (joule_solver.cpp:47-62, 908-947) ─────────────

/// A material property looked up by mesh attribute and scaled.
///
/// `MeshDependentCoefficient::Eval` is `materialMap->find(T.Attribute)` times
/// `scaleFactor` (and a hard error when the attribute is missing).  fem-rs's
/// [`PWConstCoeff`] already does the tag lookup with a configurable default;
/// this wrapper adds MFEM's `scaleFactor` (`SetScaleFactor`, used by
/// `buildA2` to fold `dt` into `InvTcap`).
#[derive(Clone)]
struct MeshDependentCoefficient {
    map: PWConstCoeff,
    scale: f64,
    /// The attribute list of the map, kept for the missing-attribute error.
    attrs: Vec<i32>,
}

impl MeshDependentCoefficient {
    /// The four joule maps cover attributes 1..=3 (joule.cpp:257-271).
    fn new(entries: [(i32, f64); 3], scale: f64) -> Self {
        MeshDependentCoefficient {
            map: PWConstCoeff::new(entries),
            scale,
            attrs: entries.iter().map(|&(t, _)| t).collect(),
        }
    }

    /// `SetScaleFactor(scale)` (joule_solver.cpp:57).
    fn set_scale_factor(&mut self, scale: f64) {
        self.scale = scale;
    }

    /// `MeshDependentCoefficient::Eval` at an element attribute: the map value
    /// times the scale factor, erroring out on a missing attribute
    /// (joule_solver.cpp:933-944).
    fn eval(&self, attr: i32) -> f64 {
        if !self.attrs.contains(&attr) {
            panic!("MeshDependentCoefficient attribute {attr} not found");
        }
        // PWConstCoeff::eval only reads `ctx.elem_tag`.
        let ctx = CoeffCtx::from_qp(&[0.0; 3], 3, 0, attr, None, None);
        self.scale * self.map.eval(&ctx)
    }
}

// ─── VisItDataCollection cycle-0 writer (joule.cpp:562-576) ─────────────────

/// C++ `ostream << double` at stream precision 6 (`%.6g` semantics, including
/// `-0` for negative zero — `Mesh::Print` has no `ZeroSubnormal`).
fn g6_mesh(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        let s = format!("{:.*e}", 5, x);
        let (mant, e) = s.split_once('e').expect("scientific form");
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        let e: i32 = e.parse().expect("exponent");
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, x);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// `VisItDataCollection::Save` mesh slice for a straight (uncurved) 3-D hex
/// mesh at the dc precision (the MFEM 4.10 oracle prints the vertices at the
/// stream default 6).  The crate writer (`fem_io::mfem::write_mfem_file*`) is
/// pinned to `Mesh::Save`'s precision 16, so the v1.0 text is written here
/// directly, following **`ParMesh::Print`** (pmesh.cpp:4856): its geometry
/// comment block stops at PRISM (no PYRAMID line), and the boundary block is
/// the boundary faces in **NCMesh face order** — `joule.cpp:349` calls
/// `mesh->EnsureNCMesh()`, and the ParNCMesh-backed ParMesh rebuilds its
/// boundary in face-discovery order (scan elements in order, MFEM
/// `Hexahedron::faces` local order), each printed with the canonical local
/// face rotation (`CheckBdrElementOrientation` alignment).  The *attributes*
/// come from the file's own `boundary` section (keyed by the sorted vertex
/// set); `fem_mesh`'s `face_nodes` canonicalizes rotations, so both the order
/// and the rotations are rebuilt here rather than read from `MeshTopology`.
fn write_dc_mesh_text(mesh: &fem_mesh::Mesh<3>, bdr_attrs: &BdrAttrMap) -> String {
    use fem_mesh::topology::MeshTopology;

    let ne = mesh.n_elems();
    let nv = mesh.n_nodes();
    let mut s = String::with_capacity(48 * 1024);
    s.push_str("MFEM mesh v1.0\n");
    s.push_str("\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n");
    s.push_str("# POINT       = 0\n");
    s.push_str("# SEGMENT     = 1\n");
    s.push_str("# TRIANGLE    = 2\n");
    s.push_str("# SQUARE      = 3\n");
    s.push_str("# TETRAHEDRON = 4\n");
    s.push_str("# CUBE        = 5\n");
    s.push_str("# PRISM       = 6\n");
    s.push_str("#\n");
    s.push_str("\ndimension\n3");
    s.push_str("\n\nelements\n");
    s.push_str(&format!("{ne}\n"));
    for e in 0..ne as u32 {
        s.push_str(&format!("{}", mesh.element_tag(e)));
        s.push_str(" 5"); // Geometry::CUBE
        for &n in mesh.elem_nodes(e) {
            s.push_str(&format!(" {n}"));
        }
        s.push('\n');
    }
    s.push_str("\nboundary\n");
    s.push_str(&format!("{}\n", bdr_attrs.len()));
    for e in 0..ne as u32 {
        let nodes = mesh.elem_nodes(e);
        for face in HEX_FACES {
            let quad = [
                nodes[face[0]] as usize,
                nodes[face[1]] as usize,
                nodes[face[2]] as usize,
                nodes[face[3]] as usize,
            ];
            let mut key = quad;
            key.sort_unstable();
            if let Some(&attr) = bdr_attrs.get(&key) {
                s.push_str(&format!(
                    "{attr} 3 {} {} {} {}\n",
                    quad[0], quad[1], quad[2], quad[3]
                ));
            }
        }
    }
    s.push_str("\nvertices\n");
    s.push_str(&format!("{nv}\n3\n"));
    for v in 0..nv as u32 {
        let c = mesh.coords_of(v);
        s.push_str(&format!(
            "{} {} {}\n",
            g6_mesh(c[0]),
            g6_mesh(c[1]),
            g6_mesh(c[2])
        ));
    }
    s
}

/// MFEM `Hexahedron::faces` (mesh/hexahedron.cpp): local faces 0..5.
const HEX_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

/// Boundary attribute keyed by the sorted vertex set of the face.
type BdrAttrMap = std::collections::HashMap<[usize; 4], i32>;

/// Parse a mesh file's `boundary` section into `(sorted vertex set → attr)`.
fn read_bdr_attrs(path: &str) -> Option<BdrAttrMap> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    loop {
        match lines.next() {
            Some(l) if l.trim() == "boundary" => break,
            Some(_) => continue,
            None => return None,
        }
    }
    let count: usize = lines.next()?.trim().parse().ok()?;
    let mut map = BdrAttrMap::with_capacity(count);
    for _ in 0..count {
        let toks: Vec<&str> = lines.next()?.split_whitespace().collect();
        if toks.len() != 6 {
            return None; // not a quad bdr entry
        }
        let attr: i32 = toks[0].parse().ok()?;
        if toks[1] != "3" {
            return None; // not a SQUARE
        }
        let mut key = [
            toks[2].parse::<usize>().ok()?,
            toks[3].parse::<usize>().ok()?,
            toks[4].parse::<usize>().ok()?,
            toks[5].parse::<usize>().ok()?,
        ];
        key.sort_unstable();
        map.insert(key, attr);
    }
    Some(map)
}

/// One GF slice file: `GridFunction::Save` header (the FESpace carries vdim 1
/// even for the vector ND/RT elements — the file's `VDim:` line is 1) plus the
/// values one per line (`Ordering: 0` → `Vector::Print(os, 1)`).
///
/// `signs` reproduces MFEM's `DofTransformation::TransformPrimal` on the all
/// +0 IC projection: value[k] = `+0 × sign` → `"0"` / `"-0"`.  The grid loop
/// is element-by-element in order with overwrite, so a dof shared between
/// elements keeps the sign of the **last** writer — exactly C++
/// `SetSubVector(vdofs, vals)` semantics.
fn write_dc_field_file(
    dir: &std::path::Path,
    name: &str,
    basis: &str,
    n_values: usize,
    signed: Option<(&[Vec<f64>], &[Vec<u32>])>,
) -> std::io::Result<()> {
    let mut s = String::with_capacity(4 * n_values + 128);
    s.push_str("FiniteElementSpace\n");
    s.push_str(&format!("FiniteElementCollection: {basis}\n"));
    s.push_str("VDim: 1\n");
    s.push_str("Ordering: 0\n\n");
    match signed {
        None => {
            for _ in 0..n_values {
                s.push_str("0\n");
            }
        }
        Some((signs, dofs)) => {
            let mut vals = vec![0.0f64; n_values];
            for (e, gd) in dofs.iter().enumerate() {
                for (k, &g) in gd.iter().enumerate() {
                    vals[g as usize] = 0.0 * signs[e][k];
                }
            }
            for v in &vals {
                s.push_str(if v.is_sign_negative() { "-0\n" } else { "0\n" });
            }
        }
    }
    std::fs::write(dir.join(format!("{name}.000000")), s)
}

/// The `.mfem_root` body — `VisItDataCollection::GetVisItRootString` (fields
/// sorted by name, `comps` = `GridFunction::VectorDim()` which counts the
/// vector-FE components even though the GF file prints `VDim: 1`).
fn dc_root_json(
    dir_name: &str,
    fields: &[(&str, &str, u32, u32, u32)], // (name, basis, comps, lod, order)
) -> String {
    let mut sorted: Vec<&(&str, &str, u32, u32, u32)> = fields.iter().collect();
    sorted.sort_by_key(|f| f.0);
    let mut s = String::new();
    s.push_str("{\n  \"dsets\": {\n    \"main\": {\n");
    s.push_str("      \"cycle\": 0,\n");
    s.push_str("      \"domains\": 1,\n");
    s.push_str("      \"fields\": {\n");
    for (i, f) in sorted.iter().enumerate() {
        s.push_str(&format!("        \"{}\": {{\n", f.0));
        s.push_str(&format!("          \"path\": \"{dir_name}/{}.%06d\",\n", f.0));
        s.push_str("          \"tags\": {\n");
        s.push_str("            \"assoc\": \"nodes\",\n");
        s.push_str(&format!("            \"basis\": \"{}\",\n", f.1));
        s.push_str(&format!("            \"comps\": \"{}\",\n", f.2));
        s.push_str(&format!("            \"lod\": \"{}\",\n", f.3));
        s.push_str(&format!("            \"order\": \"{}\"\n", f.4));
        s.push_str("          }\n        }");
        s.push_str(if i + 1 < sorted.len() { ",\n" } else { "\n" });
    }
    s.push_str("      },\n");
    s.push_str("      \"mesh\": {\n");
    s.push_str("        \"format\": \"0\",\n");
    s.push_str(&format!("        \"path\": \"{dir_name}/mesh.%06d\",\n"));
    s.push_str("        \"tags\": {\n");
    s.push_str("          \"max_lods\": \"32\",\n");
    s.push_str("          \"spatial_dim\": \"3\",\n");
    s.push_str("          \"topo_dim\": \"3\"\n");
    s.push_str("        }\n      },\n");
    s.push_str("      \"time\": 0,\n");
    s.push_str("      \"time_step\": 0\n");
    s.push_str("    }\n  }\n}\n");
    s
}

// ─── main (joule.cpp) ───────────────────────────────────────────────────────

/// `Tcapacity` (joule.cpp:147) — a local, not an option.
const T_CAPACITY: f64 = 1.0;
/// `Tconductivity` (joule.cpp:148) — a local, not an option.
const T_CONDUCTIVITY: f64 = 0.01;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = Opts::parse(&args);

    for line in BANNER {
        println!("{line}");
    }
    opts.print_options();

    // ── clipped / unported options (joule.cpp has no equivalent guard) ──────
    if opts.visualization {
        eprintln!(
            "mfem_miniapp_joule: -vis (GLVis socket display) is not ported; pass -no-vis"
        );
        std::process::exit(3);
    }
    if opts.gfprint == 1 {
        eprintln!(
            "mfem_miniapp_joule: -print 1 (grid-function dumps) is not ported; pass -print 0"
        );
        std::process::exit(3);
    }
    if opts.amr == 1 {
        eprintln!(
            "mfem_miniapp_joule: -amr 1 (non-conforming refinement + Rebalance) is not ported; pass -amr 0"
        );
        std::process::exit(3);
    }
    if opts.static_cond == 1 {
        eprintln!(
            "mfem_miniapp_joule: -sc 1 (ParBilinearForm static condensation) is not ported; pass -sc 0"
        );
        std::process::exit(3);
    }
    if opts.debug != 0 {
        eprintln!("mfem_miniapp_joule: -debug 1 (hypre matrix dumps) is not ported");
        std::process::exit(3);
    }
    if opts.visit && opts.ranks > 1 {
        eprintln!(
            "mfem_miniapp_joule: -visit with --ranks > 1 (per-rank dc slice files) \
             is not ported; pass --ranks 1"
        );
        std::process::exit(3);
    }
    // The round-150 coupled operator + BE time loop is assembled on the local
    // partition (C++ `ParBilinearForm` per-rank assembly); the serial 1:1
    // reference is `--ranks 1`, so refuse a multi-rank run rather than
    // silently diverging from the oracle on the ghost-dof exchange paths the
    // mixed forms do not implement.
    if opts.ranks != 1 {
        eprintln!(
            "mfem_miniapp_joule: the coupled time loop assembles on the local \
             partition; pass --ranks 1 for the 1:1 serial run"
        );
        std::process::exit(3);
    }
    // The dc mesh boundary block is rebuilt in NCMesh face order (see
    // `write_dc_mesh_text`) with the attributes from the file's own boundary
    // section.  Through `-rs` uniform refinement the refined boundary faces
    // are generated in fem-mesh's own order/rotation — MFEM's refined-bdr
    // parity is a kernel gap (D1048), so refined `-visit` runs are refused.
    let raw_bdr = Arc::new(if opts.visit {
        match read_bdr_attrs(&opts.mesh) {
            Some(b) if opts.ser_ref_levels == 0 => b,
            _ if opts.ser_ref_levels != 0 => {
                eprintln!(
                    "mfem_miniapp_joule: -visit with -rs > 0 (refined boundary order/rotation \
                     through UniformRefinement is not byte-parity in fem-mesh, D1048); \
                     pass -rs 0"
                );
                std::process::exit(3);
            }
            _ => {
                eprintln!(
                    "mfem_miniapp_joule: -visit needs an MFEM v1.0 .mesh with a quad \
                     boundary section"
                );
                std::process::exit(3);
            }
        }
    } else {
        BdrAttrMap::new()
    });
    if opts.par_ref_levels != 0 {
        eprintln!(
            "mfem_miniapp_joule: -rp (parallel refinement) is applied to the fem-rs \n\
             partition, not to the serial mesh as in MFEM; pass -rp 0 for the 1:1 run"
        );
        std::process::exit(3);
    }

    // joule.cpp:354-372 — the ODE solver factory.
    match opts.ode_solver_type {
        1 | 2 | 3 | 22 | 23 | 34 => {}
        other => {
            println!("Unknown ODE solver type: {other}");
            std::process::exit(3);
        }
    }

    // joule.cpp:242-277 — the material maps.  `sigmaAir`, `TcondAir` and
    // `TcapAir` are the "air" values; the rod problem is attributes 1 (rod)
    // and 2 (air), the coil problem adds 3 (also air).
    if opts.problem != "rod" && opts.problem != "coil" {
        eprintln!(
            "Problem {} not recognized",
            opts.problem
        );
        std::process::exit(3);
    }
    let sigma_air = 1.0e-6 * opts.sigma;
    let tcond_air = 1.0e6 * T_CONDUCTIVITY;
    let tcap_air = T_CAPACITY;
    let sigma = MeshDependentCoefficient::new(
        [(1, opts.sigma), (2, sigma_air), (3, sigma_air)],
        1.0,
    );
    let inv_tcond = MeshDependentCoefficient::new(
        [
            (1, 1.0 / T_CONDUCTIVITY),
            (2, 1.0 / tcond_air),
            (3, 1.0 / tcond_air),
        ],
        1.0,
    );
    let tcap = MeshDependentCoefficient::new(
        [(1, T_CAPACITY), (2, tcap_air), (3, tcap_air)],
        1.0,
    );
    let mut inv_tcap = MeshDependentCoefficient::new(
        [
            (1, 1.0 / T_CAPACITY),
            (2, 1.0 / tcap_air),
            (3, 1.0 / tcap_air),
        ],
        1.0,
    );
    // `buildA2` folds dt into InvTcap via SetScaleFactor
    // (joule_solver.cpp:687).
    inv_tcap.set_scale_factor(opts.dt);

    // Real check of the four maps against joule.cpp:257-271 (no stdout: the
    // C++ prints nothing here either, so the byte comparison stays valid).
    assert_eq!(sigma.eval(1), opts.sigma);
    assert_eq!(sigma.eval(2), 1.0e-6 * opts.sigma);
    assert_eq!(sigma.eval(3), 1.0e-6 * opts.sigma);
    assert_eq!(inv_tcond.eval(1), 1.0 / T_CONDUCTIVITY);
    // TcondAir = 1e6 * Tconductivity = 1e4.
    assert_eq!(inv_tcond.eval(2), 1.0 / (1.0e6 * T_CONDUCTIVITY));
    assert_eq!(tcap.eval(1), T_CAPACITY);
    assert_eq!(inv_tcap.eval(1), opts.dt);
    assert_eq!(inv_tcap.eval(2), opts.dt);

    // `mj_`, `sj_`, `wj_` (joule.cpp:218-220) and the skin-depth lines
    // (joule.cpp:222-227), `cout << "\nSkin depth … = " << sqrt(…) << "\n…" << endl`.
    let wj = 2.0 * std::f64::consts::PI * opts.freq;
    println!();
    println!(
        "Skin depth sqrt(2.0/(wj*mj*sj)) = {}",
        g6((2.0 / (wj * opts.mu * opts.sigma)).sqrt())
    );
    println!(
        "Skin depth sqrt(2.0*dt/(mj*sj)) = {}",
        g6((2.0 * opts.dt / (opts.mu * opts.sigma)).sqrt())
    );

    // joule.cpp:283 — `Mesh(mesh_file, 1, 1)`; joule.cpp:349
    // `mesh->EnsureNCMesh()` is a no-op for the conforming `.mesh` fixtures.
    let mfem = fem_io::mfem::read_mfem_file(&opts.mesh).unwrap_or_else(|e| {
        eprintln!("mfem_miniapp_joule: failed to read mesh {}: {e}", opts.mesh);
        std::process::exit(3);
    });
    let mut mesh0 = match mfem.mesh3d {
        Some(m) => m,
        None => {
            eprintln!(
                "mfem_miniapp_joule: {} is not a 3-D volume mesh (`.gen` NetCDF meshes are not supported by the .mesh reader)",
                opts.mesh
            );
            std::process::exit(3);
        }
    };
    // joule.cpp:377-380.
    for _ in 0..opts.ser_ref_levels {
        mesh0 = fem_mesh::amr::refine_uniform_3d(&mesh0);
    }
    let mesh0 = Arc::new(mesh0);

    let opts = Arc::new(opts);

    let opts_rank = Arc::clone(&opts);
    let raw_bdr_rank = Arc::clone(&raw_bdr);
    let launcher = ThreadLauncher::new(WorkerConfig::new(opts.ranks));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let opts = &*opts_rank;
        let par_mesh = partition_mesh(&mesh0, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        // joule.cpp:432-448 — the four FE collections' orders.
        let l2 = ParallelFESpace::new(
            L2Space::new(local_mesh.clone(), opts.order.saturating_sub(1) as u8),
            &par_mesh,
            comm.clone(),
        );
        let nd = ParallelFESpace::new(
            HCurlSpace::new(local_mesh.clone(), opts.order as u8),
            &par_mesh,
            comm.clone(),
        );
        let rt = ParallelFESpace::new(
            HDivSpace::new(local_mesh.clone(), opts.order.saturating_sub(1) as u8),
            &par_mesh,
            comm.clone(),
        );
        let h1 = {
            // D109 fixed the two defects this constructor used to work around:
            // `DofPartition::from_dof_manager` used to classify the six face DOFs
            // of every Q2 hex as element-interior DOFs (3097 instead of 2443), and
            // `ParallelFESpace::new`'s H¹ arm used to fall back to a node-only P1
            // partition (364 at order 2).  Both now give MFEM's 2443, verified
            // against `mpirun` on this mesh, so the plain constructor is used and
            // `n_global_dofs()` is the true `GlobalTrueVSize`.
            ParallelFESpace::new(
                H1Space::new(local_mesh.clone(), opts.order as u8),
                &par_mesh,
                comm.clone(),
            )
        };

        // joule.cpp:451-463 — `GlobalTrueVSize()` per space.  Note the C++
        // prints the *H(div)* true size twice (temperature flux and magnetic
        // field share the RT space).
        let h1_true = h1.n_global_dofs();
        if rank == 0 {
            println!("Number of Temperature Flux unknowns:  {}", rt.n_global_dofs());
            println!("Number of Temperature unknowns:       {}", l2.n_global_dofs());
            println!("Number of Electric Field unknowns:    {}", nd.n_global_dofs());
            println!("Number of Magnetic Field unknowns:    {}", rt.n_global_dofs());
            println!("Number of Electrostatic unknowns:     {h1_true}");
        }

        // joule.cpp:465-468 + 479-491 — `true_offset` is built from
        // `GetVSize()` and the `BlockVector` holds the six fields.  The
        // partition lengths agree with the spaces for L²/H(curl)/H(div); H¹
        // uses the space length (the P1-only partition would truncate it).
        let v_l2 = l2.n_local_dofs();
        let v_nd = nd.n_local_dofs();
        let v_rt = rt.n_local_dofs();
        let v_h1 = h1_true;
        assert_eq!(v_l2, l2.local_space().n_dofs());
        assert_eq!(v_nd, nd.local_space().n_dofs());
        assert_eq!(v_rt, rt.local_space().n_dofs());
        let true_offset: [usize; 7] = [
            0,
            v_l2,
            v_l2 + v_rt,
            v_l2 + v_rt + v_h1,
            v_l2 + v_rt + v_h1 + v_nd,
            v_l2 + v_rt + v_h1 + v_nd + v_rt,
            v_l2 + v_rt + v_h1 + v_nd + v_rt + v_l2,
        ];
        let mut f = fem_linalg::BlockVector::from_offsets(&true_offset);

        // joule.cpp:492-500 (the `main` views) and joule_solver.cpp:129-136
        // (`Init`'s views): T, F, P, E, B, w in block order.
        {
            let mut views = f.views_mut().into_iter();
            let mut t = GridFunction::make_ref(l2.local_space(), views.next().unwrap());
            let mut tf = GridFunction::make_ref(rt.local_space(), views.next().unwrap());
            let mut p = GridFunction::make_ref(h1.local_space(), views.next().unwrap());
            let mut e = GridFunction::make_ref(nd.local_space(), views.next().unwrap());
            let mut b = GridFunction::make_ref(rt.local_space(), views.next().unwrap());
            let mut w = GridFunction::make_ref(l2.local_space(), views.next().unwrap());
            // `oper.Init(F)` (joule_solver.cpp:138-143) with the identically
            // zero `Zero_vec`/`Zero` coefficients: every field is zero.
            fn zero<'a, S: fem_space::fe_space::FESpace>(gf: &mut GridFunction<'a, S>) {
                for v in gf.dofs_mut() {
                    *v = 0.0;
                }
            }
            zero(&mut t);
            zero(&mut tf);
            zero(&mut p);
            zero(&mut e);
            zero(&mut b);
            zero(&mut w);
            // `GetJouleHeating` writes into the last (L2) field.
            assert!(w.dofs_mut().iter().all(|&v| v == 0.0));
        }
        assert_eq!(f.len(), true_offset[6], "BlockVector length = true_offset[6]");

        // joule.cpp:562-576 — the VisIt data collection: cycle 0 is saved
        // *before* the time loop, with all six fields at the Init values.
        if opts.visit {
            let dc_dir = format!("{}_{:06}", opts.basename, 0);
            let _ = std::fs::remove_dir_all(&dc_dir);
            let dir = std::path::Path::new(&dc_dir);
            std::fs::create_dir_all(dir).expect("create dc dir");
            let mesh_txt = write_dc_mesh_text(&mesh0, &raw_bdr_rank);
            std::fs::write(dir.join("mesh.000000"), mesh_txt).expect("write dc mesh");
            // Field sizes: the FESpace VSize per collection (GetVSize == the
            // local size at one rank).  E/B/F carry the DofTransformation
            // negative-zero pattern; the scalar H1/L2 fields print plain "0".
            let nd_l = nd.local_space();
            let rt_l = rt.local_space();
            let n_elems = mesh0.n_elems() as usize;
            let nd_signs: Vec<Vec<f64>> =
                (0..n_elems as u32).map(|e| nd_l.element_signs(e).to_vec()).collect();
            let nd_dofs: Vec<Vec<u32>> =
                (0..n_elems as u32).map(|e| nd_l.element_dofs(e).to_vec()).collect();
            let rt_signs: Vec<Vec<f64>> =
                (0..n_elems as u32).map(|e| rt_l.element_signs(e).to_vec()).collect();
            let rt_dofs: Vec<Vec<u32>> =
                (0..n_elems as u32).map(|e| rt_l.element_dofs(e).to_vec()).collect();
            // Registration order (joule.cpp:566-571): E, B, T, w, Phi, F.
            write_dc_field_file(dir, "E", "ND_3D_P2", v_nd,
                Some((&nd_signs, &nd_dofs))).expect("write E");
            write_dc_field_file(dir, "B", "RT_3D_P1", v_rt,
                Some((&rt_signs, &rt_dofs))).expect("write B");
            write_dc_field_file(dir, "T", "L2_3D_P1", v_l2, None).expect("write T");
            write_dc_field_file(dir, "w", "L2_3D_P1", v_l2, None).expect("write w");
            write_dc_field_file(dir, "Phi", "H1_3D_P2", v_h1, None).expect("write Phi");
            write_dc_field_file(dir, "F", "RT_3D_P1", v_rt,
                Some((&rt_signs, &rt_dofs))).expect("write F");
            let fields: Vec<(&str, &str, u32, u32, u32)> = vec![
                // (name, basis, comps=VectorDim, lod=max(1,FE order), order):
                // the RT_3D_P1 collection's FE carries order 2 (p+1 in the
                // normal direction), so its tags are 2/2 per the MFEM 4.10
                // oracle (pmesh.cpp RegisterField → GetOrder()).
                ("E", "ND_3D_P2", 3, 2, 2),
                ("B", "RT_3D_P1", 3, 2, 2),
                ("T", "L2_3D_P1", 1, 1, 1),
                ("w", "L2_3D_P1", 1, 1, 1),
                ("Phi", "H1_3D_P2", 1, 2, 2),
                ("F", "RT_3D_P1", 3, 2, 2),
            ];
            std::fs::write(
                format!("{dc_dir}.mfem_root"),
                dc_root_json(&dc_dir, &fields),
            )
            .expect("write dc root");
        }

        // joule.cpp:287-346 — the boundary-condition attribute masks.  The rod
        // problem fixes attributes 1..=3 for E and 1..=2 for the thermal flux
        // and the potential; the coil problem differs (joule.cpp:290-315).
        //
        // ── D120/D121: the EM half's local targets, assembled on this mesh ──
        // The ND2/RT1/L2 global DOF numbering is slot-for-slot MFEM's on this
        // mesh (element-0 tables + counts verified against the MFEM 4.10
        // probes, tmp/d120/doforder_{mfem,femrs}.txt), so every quantity below
        // is directly comparable with tmp/d120/d121_emhalf_probe.cpp.
        let nd_local = nd.local_space();
        let rt_local = rt.local_space();
        let h1_local = h1.local_space();
        let l2_local = l2.local_space();

        // curl_3d(ND2 → RT1) — D120 (was tet-only: "tet face must have an
        // interpolation anchor" panic on this mesh).
        let grad = fem_assembly::discrete_op::DiscreteLinearOperator::gradient(h1_local, nd_local)
            .expect("D110 gradient(P2 -> ND2) on hexes");
        let curl = fem_assembly::discrete_op::DiscreteLinearOperator::curl_3d(nd_local, rt_local)
            .expect("D120 curl_3d(ND2 -> RT1) on hexes");

        // Order-2 de Rham closure on this mesh: curl(grad p) = 0.
        let mut cg_max = 0.0_f64;
        for j in 0..h1_local.n_dofs() {
            let mut unit = vec![0.0; h1_local.n_dofs()];
            unit[j] = 1.0;
            let mut g = vec![0.0; nd_local.n_dofs()];
            grad.spmv(&unit, &mut g);
            let mut c = vec![0.0; rt_local.n_dofs()];
            curl.spmv(&g, &mut c);
            for v in c {
                cg_max = cg_max.max(v.abs());
            }
        }

        // ElectricLosses machinery: M1 = ND mass with the rod/air sigma map,
        // el = E^T M1 E for the deterministic dof vector (MFEM probe:
        // el = 7715.3007859716654).
        let m1 = fem_assembly::vector_assembler::VectorAssembler::assemble_bilinear(
            nd_local,
            &[&fem_assembly::standard::VectorMassIntegrator {
                alpha: sigma.map.clone(),
            }],
            4,
        );
        let e_vec: Vec<f64> = (0..nd_local.n_dofs())
            .map(|i| {
                1.0 + (0.7 * (i as f64 + 1.0)).sin()
                    + 0.25 * (1.3 * (2.0 * i as f64 + 1.0)).cos()
            })
            .collect();
        let mut m1e = vec![0.0; nd_local.n_dofs()];
        m1.spmv(&e_vec, &mut m1e);
        let el: f64 = e_vec.iter().zip(m1e.iter()).map(|(a, b)| a * b).sum();

        // GetJouleHeating: sigma |E|^2 projected onto L2(1) — the D121
        // element-aware, trilinear-aware projection entry.
        let e_gf = GridFunction::new(nd_local, e_vec);
        let w = fem_assembly::postproc::project_coefficient_element(l2_local, &|elem, xi, x| {
            let ev = e_gf.evaluate_vector_at_element(elem, xi);
            let tag = local_mesh.element_tag(elem);
            let ctx =
                fem_assembly::postproc::coefficient::CoeffCtx::from_qp(x, 3, elem, tag, None, None);
            sigma.scale * sigma.map.eval(&ctx) * (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2])
        });
        let w_sum: f64 = w.iter().sum();
        let w_max = w.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

        if rank == 0 {
            // D1046 (round 106): this instrumentation is NOT part of the C++
            // stdout — it moved from `println!` to `eprintln!` so the
            // comparable stdout region (banner → dof banner) stays byte-exact.
            eprintln!("D120/D121 EM-half local targets (cylinder-hex, -o 2):");
            eprintln!(
                "  curl_3d(ND2->RT1): {} x {}, nnz = {}",
                curl.nrows,
                curl.ncols,
                curl.nnz()
            );
            eprintln!("  max |curl(grad P2)| = {cg_max:.3e}   (order-2 de Rham closure)");
            eprintln!("  dot(E, J) machinery, el = E^T M1 E = {el:.17e}");
            eprintln!("  joule heating W (L2 projection of sigma |E|^2): sum = {w_sum:.17e}, max = {w_max:.17e}");
            eprintln!("  MFEM 4.10 probe: el = 7715.3007859716654e0, sum = 7936759.0844848659e0, max = 59455.579501421256e0");
        }

        // ── round-150: MagneticDiffusionEOperator + the coupled BE time loop ──
        // 1:1 with joule_solver.cpp.  The C++ `ImplicitSolve` (lines 401-640)
        // writes the advanced P/E/F/W state **directly** (they are MakeRef'd
        // views of the state vector) while returning B/T *slopes* in dX_dt;
        // MFEM's `BackwardEulerSolver::Step` then does `x += dt·k` (the zero
        // slopes of P/E/F/W leave the written state untouched).  fem-solver's
        // ODE trait expresses slope-vs-state as a per-operator dichotomy, so
        // the step below unfolds the C++ semantics literally instead.
        let n_l2 = v_l2;
        let n_rt = v_rt;
        let n_nd = v_nd;
        let n_h1 = v_h1;
        let (o_t, o_f, o_p, o_e, o_b, o_w) = (
            true_offset[0], true_offset[1], true_offset[2],
            true_offset[3], true_offset[4], true_offset[5],
        );

        // buildA0 (joule_solver.cpp:700-706): H1 stiffness `∫ σ ∇u·∇v`.
        // MFEM `DiffusionIntegrator` order = Trans.OrderGrad + GetOrder = 3+2
        // (eltrans.cpp:509 Qk `k(d-1)+(l-1)` with geom H1Q1, FE H1Q2, + 2).
        let a0 = fem_assembly::Assembler::assemble_bilinear(
            h1_local,
            &[&DiffusionIntegrator { kappa: sigma.map.clone() }],
            5,
        );
        // buildA1 (709-721): `∫ σ u·v + dt/μ (∇×u)·(∇×v)` on ND — the curl-curl
        // coefficient is `dtMuInv = dt·(1/μ)` (joule_solver.cpp:712).  MFEM
        // picks the rules per integrator: `VectorFEMassIntegrator` =
        // `OrderW + 2·GetOrder = 2 + 2·2` (D680 hook) and `CurlCurlIntegrator`
        // = `2·GetOrder = 4` (Qk tensor, curl_curl.rs hook); the quad_order
        // argument only feeds the hook-less ones.
        let a1 = fem_assembly::vector_assembler::VectorAssembler::assemble_bilinear(
            nd_local,
            &[
                &VectorMassIntegrator { alpha: sigma.map.clone() },
                &CurlCurlIntegrator { mu: opts.dt / opts.mu },
            ],
            4,
        );
        // buildA2 (724-740): `∫ (1/k) u·v + dt·(1/c) (∇·u)(∇·v)` on RT.  The
        // dt rides `InvTcap::SetScaleFactor(dt)` (joule.cpp:800-802), so it is
        // folded into the PW table here.
        let inv_tcap_dt = MeshDependentCoefficient::new(
            [
                (1, opts.dt / T_CAPACITY),
                (2, opts.dt / tcap_air),
                (3, opts.dt / tcap_air),
            ],
            1.0,
        );
        let a2 = fem_assembly::vector_assembler::VectorAssembler::assemble_bilinear(
            rt_local,
            &[
                &VectorMassIntegrator { alpha: inv_tcond.map.clone() },
                &DivDivIntegrator { kappa: inv_tcap_dt.map.clone() },
            ],
            4, // MFEM DivDivIntegrator: 2·GetOrder - 2 + 2 (Qk) = 4
        );
        // buildM1 (743-752): ND mass with σ — `ElectricLosses` (845-848) and
        // the `M1 ∇P` term of the A1 right-hand side.
        let m1 = fem_assembly::vector_assembler::VectorAssembler::assemble_bilinear(
            nd_local,
            &[&VectorMassIntegrator { alpha: sigma.map.clone() }],
            4,
        );
        // buildM3 (755-765): L2 mass with c; plus the unit mass behind
        // `ParLinearForm temp_lf` (`DomainLFIntegrator(Wcoeff)` = ∫ W φ).
        let m3 = fem_assembly::Assembler::assemble_bilinear(
            l2_local,
            &[&MassIntegrator { rho: T_CAPACITY }],
            2,
        );
        let m3_unit = fem_assembly::Assembler::assemble_bilinear(
            l2_local,
            &[&MassIntegrator { rho: 1.0 }],
            2,
        );
        // buildCurl (790-810): the discrete curl interpolator is the D120 one
        // assembled above; the weak curl `∫ (1/μ)(∇×u)·v` (RT rows × ND cols;
        // order = GetOrder(ND2) + GetOrder(RT1) - 1 = 3) is new.  Both are
        // applied to grid-function-sized vectors.
        let weak_curl = assemble_hcurl_hdiv_weak_curl(nd_local, rt_local, 3, 1.0 / opts.mu);
        let wc_t = weak_curl.transpose();
        // buildDiv (812-831): `weakDivC = ∫ (1/c)(∇·u) φ` (InvTcap, dt-folded)
        // and `weakDiv = ∫ (∇·u) φ` (L2 rows × RT cols, order 4).
        let weak_div_c = assemble_hdiv_l2_mixed(
            l2_local,
            rt_local,
            &[&HDivL2ScaledDiv { alpha: inv_tcap_dt.map.clone() }],
            4,
        );
        let weak_div =
            assemble_hdiv_l2_mixed(l2_local, rt_local, &[&HDivL2DivIntegrator], 4);
        let wd_t = weak_div.transpose();
        let wdc_t = weak_div_c.transpose();
        // buildGrad (834-843): the H1 → ND interpolator is the D110 one
        // assembled above; AMS additionally gets MFEM's P1 discrete gradient
        // (the hypre AMS convention, ex3's calibration).
        let g_p1 = fem_assembly::discrete_op::DiscreteLinearOperator::gradient(
            &fem_space::H1Space::new(local_mesh.clone(), 1),
            nd_local,
        )
        .expect("P1 gradient for AMS");
        let g_linlvo = fem_linalg::fem_to_linlvo_csr(&g_p1);

        // Essential true dofs (joule.cpp:1046-1058 mask values): ND attrs
        // 1..=3 for the E field, RT attrs 1..=2 (zero normal flux) and H1
        // attrs 1..=2 for the potential.
        let ess_nd = boundary_dofs_hcurl(&local_mesh, nd_local, &[1, 2, 3]);
        let ess_rt = boundary_dofs_hdiv(&local_mesh, rt_local, &[1, 2]);
        let ess_h1 = boundary_dofs(&local_mesh, h1_local.dof_manager(), &[1, 2]);

        // HyprePCG settings (joule_solver.hpp:35-36): tol 1e-9 relative,
        // max 1000 iterations, print level 0.
        let cfg = fem_solver::SolverConfig {
            rtol: 1.0e-9,
            atol: 0.0,
            max_iter: 1000,
            verbose: false,
            ..fem_solver::SolverConfig::default()
        };

        // joule.cpp:587-602 — the BE time loop (with the state-overwrite
        // semantics of `ImplicitSolve` unfolded, see the block header).
        let mut t: f64 = 0.0;
        let mut ti: u64 = 0;
        let mut last_step = false;
        // Fast-pin accumulators (structural identities of the discretisation,
        // reported on stderr): max |<φ, div B>| (Gauss law for the RT curl of
        // the ND field — div ∘ curl = 0 exactly, so the projection must be
        // zero) and the BE energy ledger `Δ(c∫T) = Σ dt·(∫W - ∫ div F)`.
        let mut div_b_max = 0.0_f64;
        let mut ledger_rhs = 0.0_f64;
        let mut ledger_lhs = 0.0_f64;
        while !last_step {
            ti += 1;
            if t + opts.dt >= opts.t_final - opts.dt / 2.0 {
                last_step = true;
            }
            // BackwardEulerSolver::Step: `SetTime(t + dt)` then ImplicitSolve.
            t += opts.dt;
            let wj = 2.0 * std::f64::consts::PI * opts.freq;
            let (t_cur, b_cur) = {
                let s = f.as_slice();
                (s[o_t..o_f].to_vec(), s[o_b..o_w].to_vec())
            };

            // 1. A0 dP = b0 (joule_solver.cpp:464-496): project p_bc = ±cos(ωt)
            //    onto H1 (nodal interpolation), form the system with the
            //    poisson attrs 1..=2 essential, PCG + AMG.
            let volt = (wj * t).cos();
            let mut x0 = fem_assembly::postproc::project_coefficient_element(
                h1_local,
                &|_e: u32, _xi: &[f64], x: &[f64]| {
                    if x[2] < 0.0 { volt } else { -volt }
                },
            );
            let mut v0 = vec![0.0_f64; n_h1];
            let mut a0e = a0.clone();
            form_linear_system_vdofs(&mut a0e, &mut v0, &mut x0, &ess_h1, false);
            let mut p_new = x0;
            fem_solver::amg::solve_amg_cg(
                &a0e,
                &v0,
                &mut p_new,
                &fem_solver::amg::boomeramg_config(),
                &cfg,
            )
            .expect("A0 PCG+AMG solve");

            // 2. v1 = weakCurlᵀ B + M1 ∇P (509-518).  C++ uses the **just
            //    solved** dP here (`grad->Mult(P,E)` runs after
            //    `a0->RecoverFEMSolution(*X0,*v0,P)` overwrote the P state).
            let mut v1 = vec![0.0_f64; n_nd];
            wc_t.spmv(&b_cur, &mut v1);
            let mut gp = vec![0.0_f64; n_nd];
            grad.spmv(&p_new, &mut gp);
            m1.spmv_add(1.0, &gp, 1.0, &mut v1);

            // 3. A1 dE = v1 (526-560).  The E essential data comes from
            //    `ProjectBdrCoefficientTangent(edot_bc)`, and `edot_bc` is
            //    identically zero (joule.cpp:732-735) — the elimination values
            //    are all zero, which is the initial X1.  The iteration budget
            //    is widened against SOLVER_MAX_IT: fem-rs's AMS is weaker than
            //    hypre's (no scaling/cycling knobs), so the `-rs 1` 55k-dof
            //    system converges past 1000 iterations where hypreAMS lands
            //    inside them — the 1e-9 relative stopping test (hence the
            //    converged solution) is unchanged.
            let mut x1 = vec![0.0_f64; n_nd];
            let mut a1e = a1.clone();
            form_linear_system_vdofs(&mut a1e, &mut v1, &mut x1, &ess_nd, false);
            let mut e_tot = x1;
            fem_solver::solve_pcg_ams(
                &a1e,
                &g_linlvo,
                &v1,
                &mut e_tot,
                &fem_solver::AmsSolverConfig {
                    inner_cfg: fem_solver::SolverConfig {
                        max_iter: 10_000,
                        ..cfg.clone()
                    },
                    ams_cfg: fem_solver::AmsConfig::default(),
                },
            )
            .expect("A1 PCG+AMS solve");

            // 4. E ← E_ind − ∇P (567-572).
            for (e, g) in e_tot.iter_mut().zip(gp.iter()) {
                *e -= *g;
            }

            // 5. dB/dt = −curl E (575-578).
            let mut db = vec![0.0_f64; n_rt];
            curl.spmv(&e_tot, &mut db);
            for v in db.iter_mut() {
                *v = -*v;
            }

            // 6. W = GetJouleHeating(E) (581-582) — the D121 element-aware L2
            //    projection of σ|E|².
            let e_gf = GridFunction::new(nd_local, e_tot.clone());
            let w_new = fem_assembly::postproc::project_coefficient_element(
                l2_local,
                &|elem: u32, xi: &[f64], x: &[f64]| {
                    let ev = e_gf.evaluate_vector_at_element(elem, xi);
                    let tag = local_mesh.element_tag(elem);
                    let ctx = CoeffCtx::from_qp(x, 3, elem, tag, None, None);
                    sigma.scale * sigma.map.eval(&ctx)
                        * (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2])
                },
            );

            // 7. v2 = dt·weakDivCᵀ W + weakDivᵀ T (585-590).  `v2` is an
            //    H(Div)-sized temporary in the C++ (`ParGridFunction(&HDivFESpace)`).
            let mut v2 = vec![0.0_f64; n_rt];
            wdc_t.spmv(&w_new, &mut v2);
            for v in v2.iter_mut() {
                *v *= opts.dt;
            }
            wd_t.spmv_add(1.0, &t_cur, 1.0, &mut v2);

            // 8. A2 dF = v2 (594-633).  The F essential data comes from
            //    `ProjectBdrCoefficientNormal(Zero_vec)` — identically zero.
            //    C++ preconditions with HypreADS; fem-rs has no ADS, so the
            //    diag-smoother PCG stands in (registered preconditioner gap —
            //    the tolerance-converged solution is the target, not the
            //    iteration path).
            let mut x2 = vec![0.0_f64; n_rt];
            let mut a2e = a2.clone();
            form_linear_system_vdofs(&mut a2e, &mut v2, &mut x2, &ess_rt, false);
            let mut f_new = x2;
            fem_solver::solve_pcg_dsmoother(&a2e, &v2, &mut f_new, &cfg)
                .expect("A2 PCG+diag solve (ADS gap)");

            // 9. M3 dT = <W,·> − weakDiv·F (638-662); no essential BCs.
            let mut lf = vec![0.0_f64; n_l2];
            m3_unit.spmv(&w_new, &mut lf);
            weak_div.spmv_add(-1.0, &f_new, 1.0, &mut lf);
            let mut d_t = vec![0.0_f64; n_l2];
            fem_solver::solve_pcg_dsmoother(&m3, &lf, &mut d_t, &cfg)
                .expect("M3 PCG+diag solve");

            // 10. State update: P/E/F/W are the written advanced state, B and
            //     T accumulate the returned slopes (`x += dt·k`).
            let s = f.as_slice_mut();
            for (x, &d) in s[o_t..o_f].iter_mut().zip(d_t.iter()) {
                *x += opts.dt * d;
            }
            s[o_f..o_p].copy_from_slice(&f_new);
            s[o_p..o_e].copy_from_slice(&p_new);
            s[o_e..o_b].copy_from_slice(&e_tot);
            for (b, &d) in s[o_b..o_w].iter_mut().zip(db.iter()) {
                *b += opts.dt * d;
            }
            s[o_w..].copy_from_slice(&w_new);

            // Fast pins.  Gauss: the projected divergence of the RT curl of an
            // ND field vanishes exactly (discrete identity), so |weakDiv·B| is
            // round-off-level at every step.  Ledger: summing `M3 dT = lf`
            // over the L2 partition of unity gives `Δ(c∫T) = Σ dt·(∫W−∫divF)`
            // — closed to PCG tolerance on both sides.
            let s = f.as_slice();
            let b_blk = &s[o_b..o_w];
            let mut div_b = vec![0.0_f64; n_l2];
            weak_div.spmv(b_blk, &mut div_b);
            let this_db = div_b.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            div_b_max = div_b_max.max(this_db);
            ledger_rhs += opts.dt * lf.iter().sum::<f64>();
            let mut m3_t = vec![0.0_f64; n_l2];
            m3.spmv(&s[o_t..o_f], &mut m3_t);
            ledger_lhs = m3_t.iter().sum::<f64>();

            // joule.cpp:606-614 — ElectricLosses + the step line
            // (`cout << fixed`, setw(6) ti, setw(6) setprecision(3) t,
            // setprecision(8) el; vis_steps 1 prints every step).
            if last_step || (ti % opts.vis_steps as u64) == 0 {
                let e_blk = &s[o_e..o_b];
                let mut m1e = vec![0.0_f64; n_nd];
                m1.spmv(e_blk, &mut m1e);
                let el: f64 = e_blk.iter().zip(m1e.iter()).map(|(a, b)| a * b).sum();
                if rank == 0 {
                    println!("step {:>6},\tt = {:>6.3},\tdot(E, J) = {:.8}", ti, t, el);
                }
            }
        }
        if rank == 0 {
            // Round-150 fast pins (stderr — not part of the C++ stdout).
            eprintln!("round-150 fast pins (cylinder-hex -o 2 -s 1):");
            eprintln!("  max |<phi, div B>| over steps = {div_b_max:.3e}   (discrete div-curl = 0)");
            eprintln!("  energy ledger: lhs c*int(T) = {ledger_lhs:.17e}, rhs sum dt*(int W - int div F) = {ledger_rhs:.17e}");
            eprintln!("  oracle (mfem410_mpi, np 1): step 1 el = 1.78064984, step 2 el = 5.12554673");
        }
    });
}
