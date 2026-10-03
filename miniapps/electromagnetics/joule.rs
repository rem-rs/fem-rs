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
//! Not ported here (each is listed with the reason; the run stops after the dof
//! banner with status 3 and a message):
//! * `MagneticDiffusionEOperator`'s four coupled solves (the a0/a1/a2/m1/m2/m3
//!   systems, `HyprePCG` + `HypreBoomerAMG`/`HypreAMS`/`HypreADS`).  fem-rs has
//!   PCG + AMG (`fem_parallel::par_amg`, AMS via `ParAmsPrecond`) and the
//!   SIAV/ODE family (D89/D90), so the solvers are no longer the blocker; the
//!   blockers are the two discrete operators and the projection listed below.
//!   Note also that the C++ **stdout itself is not reproducible** past the dof
//!   banner: hypre prints its `BoomerAMG SETUP PARAMETERS:` block and its
//!   operator tables unconditionally (`-hl 0` does not silence them), so a byte
//!   comparison of a full joule run is impossible for any non-hypre
//!   implementation.  The comparable region is exactly the banner + options +
//!   skin depths + dof counts implemented below (C++ lines 1-106 for the
//!   reference run below), after which the C++ emits ~66 lines of hypre
//!   diagnostics and only then joule's own
//!   `step      1,	t =  0.500,	dot(E, J) = 1.78064984` /
//!   `step      2,	t =  1.000,	dot(E, J) = 5.12554673` (lines 107-108).  Those
//!   two lines are the honest end-to-end target for this miniapp: `dot(E, J)`
//!   is `ElectricLosses = m1->ParInnerProduct(E_gf, E_gf) = ∫ σ E·E`, and — see
//!   below — it depends **only** on the electromagnetic half of the operator
//!   (`P → grad P → weakCurlᵀB → M1 + dt·S1 solve → E ← E − grad P`), never on
//!   `W`, `F` or `T`;
//! * the H¹→H(curl) discrete gradient `grad` (`ParDiscreteLinearOperator` +
//!   `GradientInterpolator`, joule_solver.cpp:787-796): **landed in D110** —
//!   `DiscreteLinearOperator::gradient` now covers `H¹(P2) → ND2` on 3-D
//!   hexahedra (`crates/assembly/src/discrete_op.rs`, tests
//!   `crates/assembly/tests/d110_p2_nd2_gradient_3d_hex.rs`).  It is *not*
//!   wired below yet because of the next two bullets;
//! * the H(curl)→H(div) **discrete curl at order 2 on hexahedra**.
//!   `DiscreteLinearOperator::curl_3d(ND2 → RT1)` is tetrahedron-only: on a hex
//!   mesh it panics with "tet face must have an interpolation anchor"
//!   (`crates/assembly/src/discrete_op.rs`), and joule's `curl->Mult(E, dB)`
//!   (joule_solver.cpp:585) needs exactly that operator.  The lowest-order pair
//!   `(ND1 → RT0)` *is* topological and does cover hexes, so this is an
//!   order-2 gap, not a hex gap;
//! * `GetJouleHeating`'s L2 projection of `σ|E|²` (joule_solver.cpp:805-815):
//!   it is a `GridFunctionCoefficient`-valued projection, i.e. the integrand
//!   reads `E_gf.GetVectorValue(T, ip)`, so it must be assembled
//!   element-by-element with the H(curl) basis pushed through the element map.
//!   `fem_assembly::postproc::project_coefficient` takes a *physical point*
//!   closure and `GridFunction::evaluate_vector_at_element` builds its element
//!   Jacobian through `simplex_jacobian`, which for `Hex8` is the constant
//!   corner-based (affine) Jacobian — wrong for the trilinear/curved hexes of
//!   `cylinder-hex.mesh`.  Needs a trilinear-map-aware entry point;
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
//! Usage:
//!   cargo run --release --example miniapp_joule -- -m data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5 -no-vis -no-visit
//!   cargo run --release --example miniapp_joule -- -m data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5 -no-vis -visit

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use fem_assembly::postproc::coefficient::{CoeffCtx, PWConstCoeff, ScalarCoeff};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{ParallelFESpace, WorkerConfig};
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

    let unported = Arc::new(AtomicBool::new(false));
    let unported_rank = Arc::clone(&unported);
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
        // `bdr_attributes.Max()` is the largest boundary attribute of the mesh
        // (fem-rs has no `Array<int> bdr_attributes`; the maximum over the
        // boundary-face tags is the same number).
        //
        // D785: the max runs over the **global** mesh (`mesh0`), not the rank
        // partition — C++'s `bdr_attributes.Max()` is a whole-mesh quantity,
        // and on np≥2 a rank whose partition misses the largest-tag boundary
        // face would build a shorter (wrong) ess mask.
        let n_bdr = (0..mesh0.n_boundary_faces() as u32)
            .map(|fc| mesh0.face_tag(fc) as usize)
            .max()
            .unwrap_or(0);
        let (ess_bdr, thermal_ess_bdr, poisson_ess_bdr) = if opts.problem == "coil" {
            (vec![1; n_bdr], {
                let mut v = vec![0; n_bdr];
                v[2] = 1;
                v
            }, {
                let mut v = vec![0; n_bdr];
                v[0] = 1;
                v[1] = 1;
                v
            })
        } else {
            (vec![1; n_bdr], {
                let mut v = vec![0; n_bdr];
                v[0] = 1;
                v[1] = 1;
                v
            }, {
                let mut v = vec![0; n_bdr];
                v[0] = 1;
                v[1] = 1;
                v
            })
        };
        assert!(ess_bdr.iter().all(|&v| v == 1));

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

        // ── Not ported: the operator + time loop ────────────────────────────
        if rank == 0 {
            eprintln!(
                "mfem_miniapp_joule: the coupled solves + time loop are not wired yet.  The run\n\
                 writes the byte-exact stdout prefix (banner, options dump, skin depths, the\n\
                 five dof lines), saves the -visit cycle-0 data collection (D1046), then stops\n\
                 with status 3.\n\
                 Remaining gaps, in dependency order:\n\
                 1. weakCurl/weakDiv/weakDivC mixed operators (H(curl)->H(div) weak forms)\n\
                 and the A0 = Div sigma Grad solve (PCG+AMG), A1 = M1 + dt S1 solve (PCG+AMS),\n\
                 A2 = M2 + dt S2 solve (PCG+ADS), M2/M3 solves, the ODE driver and the time\n\
                 loop of MagneticDiffusionEOperator::ImplicitSolve (joule_solver.cpp:401-640).\n\
                 Past the dof banner the C++ stdout is hypre diagnostics; the end-to-end\n\
                 target remains the C++ run's last two lines, 'step 1/2, t = 0.5/1.0,\n\
                 dot(E, J) = 1.78064984 / 5.12554673'.\n\
                 2. -amr 1 needs MFEM's 3-D nonconforming refinement of the attr-1 (rod)\n\
                 elements + Rebalance -- fem-rs has no 3-D NCMesh; partial refinement cannot\n\
                 be emulated with the uniform `refine_uniform_3d` (the C++ refines ONLY the\n\
                 metal region, so the dof counts would differ).\n\
                 3. -sc 1 / -debug 1 / -print 1 / -vis / -visit with --ranks > 1 and the .gen\n\
                 NetCDF meshes remain unported.\n\
                 CLOSED: curl_3d(ND2 -> RT1) hex (D120), GetJouleHeating projection (D121),\n\
                 the >=2-ranks ND2/RT1 partition (D412), the H1(P2) -> ND2 gradient (D110).\n\
                 Ported and checked here: banner, options dump, skin depths, the four FE\n\
                 spaces with their orders, the five GlobalTrueVSize lines, the six-field\n\
                 BlockVector layout with its make_ref views, the four material maps, the\n\
                 -visit cycle-0 file set (D1046), and the D120/D121 EM-half local targets\n\
                 on stderr above.\n\
                 Requested: -o {} -s {} -tf {} -dt {} n_bdr={n_bdr}\n\
                 local block sizes [L2,RT,H1,ND] = [{},{},{},{}]  block_len={}\n\
                 H1 true dofs = {h1_true}\n\
                 ess_bdr={ess_bdr:?} thermal_ess_bdr={thermal_ess_bdr:?} poisson_ess_bdr={poisson_ess_bdr:?}",
                opts.order,
                opts.ode_solver_type,
                g6(opts.t_final),
                g6(opts.dt),
                v_l2,
                v_rt,
                v_h1,
                v_nd,
                f.len(),
            );
            unported_rank.store(true, Ordering::Relaxed);
        }
    });

    if unported.load(Ordering::Relaxed) {
        std::process::exit(3);
    }
}
