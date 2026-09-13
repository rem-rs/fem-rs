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
//!   PCG + diagonal (and the SIAV/ODE family from D89/D90), but no AMG/AMS/ADS,
//!   and the C++ **stdout itself is not reproducible** past this point: hypre
//!   prints its `BoomerAMG SETUP PARAMETERS:` block and its operator tables
//!   unconditionally, so a byte comparison of a full joule run is impossible
//!   for any non-hypre implementation.  The comparable region is exactly the
//!   banner + options + skin depths + dof counts implemented below;
//! * the `Mult` path (joule_solver.cpp:162-375) and `GetJouleHeating`'s L2
//!   projection of `sigma |E|^2` (joule_solver.cpp:805-815), which needs a
//!   `GridFunctionCoefficient`-valued projection (the coefficient reads
//!   `E_gf.GetVectorValue(T, ip)`);
//! * the H1→H(curl) discrete gradient `grad` (`ParDiscreteLinearOperator` +
//!   `GradientInterpolator`, joule_solver.cpp:787-796) — fem-rs has the
//!   H(curl)→H(div) discrete curl (`ParDiscreteLinearOperator::curl_3d`) but no
//!   H¹→H(curl) gradient;
//! * `-sc 1` static condensation (`ParBilinearForm::EnableStaticCondensation`),
//!   `-amr 1` (`GeneralRefinement` + `Rebalance`), `-debug 1`
//!   (`hypre_ParCSRMatrixPrint`), `-gfprint 1`, `-vis`, `-visit` (no
//!   visualization layer is ported);
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
//! Byte-identical prefix (everything fem-rs prints before it stops): the
//! banner, the whole `Options used:` dump, the blank line + the two
//! `Skin depth` lines, and the five `Number of … unknowns` lines —
//! `6456 / 2016 / 6882 / 6456 / 2443`.  The only intended difference is the
//! `--mesh` path string itself.
//!
//! Usage:
//!   cargo run --release --example miniapp_joule -- -m data/cylinder-hex.mesh -p rod -tf 1.0 -dt 0.5 -no-vis -no-visit

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use fem_assembly::postproc::coefficient::{CoeffCtx, PWConstCoeff, ScalarCoeff};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{ParallelFESpace, WorkerConfig};
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
    if opts.visit {
        eprintln!(
            "mfem_miniapp_joule: -visit (VisIt DataCollection output) is not ported; pass -no-visit"
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

        // joule.cpp:287-346 — the boundary-condition attribute masks.  The rod
        // problem fixes attributes 1..=3 for E and 1..=2 for the thermal flux
        // and the potential; the coil problem differs (joule.cpp:290-315).
        //
        // `bdr_attributes.Max()` is the largest boundary attribute of the mesh
        // (fem-rs has no `Array<int> bdr_attributes`; the maximum over the
        // boundary-face tags is the same number).
        let n_bdr = (0..local_mesh.n_boundary_faces() as u32)
            .map(|fc| local_mesh.face_tag(fc) as usize)
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

        // ── Not ported: the operator + time loop ────────────────────────────
        if rank == 0 {
            eprintln!(
                "mfem_miniapp_joule: MagneticDiffusionEOperator (the four coupled solves\n\
                 a0/a1/a2/m1/m2/m3 with HyprePCG + BoomerAMG/AMS/ADS) is not ported yet —\n\
                 fem-rs has no AMG/AMS/ADS, and hypre prints its own stdout, so a full-run\n\
                 byte comparison is impossible for any non-hypre implementation.  The run\n\
                 stops after the dof banner (verified byte-exact against the C++ binary).\n\
                 Ported and checked here: banner, options dump, skin depths, the four FE\n\
                 spaces with their orders, the five GlobalTrueVSize lines, the six-field\n\
                 BlockVector layout with its make_ref views, the four material maps.\n\
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
