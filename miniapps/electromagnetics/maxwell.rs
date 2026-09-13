//! # Miniapp Maxwell — simple full-wave electromagnetic simulation
//!
//! 1:1 port of MFEM `miniapps/electromagnetics/maxwell.cpp` +
//! `maxwell_solver.{hpp,cpp}` (MFEM 4.10), serial rank count.
//!
//! Solves the coupled first-order Maxwell system
//!
//! ```text
//!   epsilon dE/dt = Curl 1/mu B - sigma E - J
//!           dB/dt = - Curl E
//! ```
//!
//! with `E` in H(curl) (Nedelec edge elements) and `B` in H(div)
//! (Raviart-Thomas face elements), advanced by the symplectic integrator
//! `SIAVSolver` (`linalg/ode.cpp`, orders 1–4) — taken from the library
//! (`fem_solver::SiavSolver`, D90) rather than from a local copy, with the
//! `maxwell` rank-local vectors supplied through `SiaState`.
//!
//! ## Port boundary (what is 1:1 and what is not)
//!
//! 1:1 (verified against the C++ binary, see the notes at the end of this
//! header):
//! * option set/defaults and the `Options used:` dump shape (only the
//!   parsed-value formatting of `OptionsParser::PrintOptions`);
//! * mesh read + `-rs` uniform refinements; the two FE spaces (ND order `p`,
//!   RT order `p-1` as in `common/pfem_extras.cpp`) and both printed dof
//!   counts;
//! * the whole `MaxwellSolver` operator set for the lossless path: H(div)
//!   mass with `1/mu`, the weak curl coupling `(1/mu curl(B), v)`, the
//!   discrete curl `-Curl`, the H(curl) mass with `epsilon`, the dipole
//!   current source `jd_`, initial fields (`EFieldFunc`/`BFieldFunc` are
//!   identically zero in the C++ miniapp, so the dof vectors are exactly 0 —
//!   the projection is not needed to reproduce them), the energy functional
//!   `0.5 (E^T M1 E + B^T M2 B)`, `GetMaximumTimeStep`'s power iteration,
//!   `SnapTimeStep` and the `Energy(<t>ns):  <e>J` time-step table;
//! * homogeneous `-dbcs` essential dofs (the C++ `dEdtBCFunc` used by
//!   `main` returns zero, so `FormLinearSystem`'s essential values are 0 and
//!   `apply_dirichlet_par(p, 0.0, ..)` is the exact equivalent);
//! * the SIAV step sequence, including the `max(t+dt, ti+ts*it)` snapshot
//!   targets and the `(int)(dtScale*dt/dtMax)` time-slice key of the
//!   implicit operator.
//!
//! Clipped (each exits with status 3 and a message):
//! * `-vis` (GLVis socket display) and `-visit` (VisIt DataCollection output)
//!   — no visualization layer is ported, so `-no-vis -no-visit` is required;
//! * `-cs`, `-abcs` — the loss operator `M1(sigma) + M1(eta_inv)|_ABC` and
//!   the implicit (`A1[dt] = M1(eps) + 0.5 dt L`) solve are not ported yet;
//!   note the lossy branch is exactly the `lossy_` path of
//!   `MaxwellSolver::implicitSolve` and needs only the two mass matrices
//!   above (the boundary piece is `TangentialMassIntegrator`, which exists);
//! * NURBS meshes (the C++ `mesh->NURBSext` projection: one
//!   `UniformRefinement()` + `SetCurvature(2)`) and 2-D meshes (the ported
//!   `curl_3d`/mixed assembly chain is 3-D).
//!
//! Two implementation notes where the C++ uses a library facility that has
//! no fem-rs counterpart in `crates/` (deliberately *not* worked around
//! silently, see the report):
//! * `HypreParVector::Randomize(1234)` (hypre's RNG) seeds the power
//!   iteration; a deterministic LCG vector is substituted, so the printed
//!   `Maximum Time Step` differs (C++ 0.141749 ns vs 0.145761 ns for the
//!   reference run — the power-method eigenvalue estimate to the C++ loop's
//!   own 1e-3 `ptol`, not a porting error).  Both land in the same
//!   `SnapTimeStep` bucket, so `Number of Time Steps` / `Time Step Size` and
//!   every subsequent number are unaffected;
//! * the discrete curl `-Curl` (`ParDiscreteLinearOperator::curl_3d`) is still
//!   assembled with H(div) rows and applied transposed in the power iteration
//!   (`maximum_time_step`, where `curl_t = self.neg_curl.transpose()` maps
//!   H(div) → H(curl)); unlike `weakCurlMuInv_` that matrix is used only for
//!   the `dtMax` power method, and the transpose only needs the *rows* of the
//!   H(curl) space there.  `weakCurlMuInv_` itself used to be built the same
//!   way and now uses D88's owned-H(curl)-row assembly (below).
//!
//! ## D99: `weakCurlMuInv_` via the owned-H(curl)-row assembler
//!
//! `MaxwellSolver` builds `weakCurlMuInv_` as
//! `ParMixedBilinearForm(HDiv, HCurl)` + `MixedVectorWeakCurlIntegrator`, i.e.
//! an operator whose **rows are H(curl) true DOFs** and whose columns are
//! H(div) DOFs.  Earlier rounds assembled the H(div)-row matrix and applied
//! `.transpose()` at the call site.  That is exact when one rank owns every
//! DOF, but for several ranks the H(div)-row path has already truncated its
//! rows to the owned H(div) DOFs, so its transpose exposes a row for every
//! *local* H(curl) DOF — ghost rows carrying only this rank's element
//! contributions.  The call site now uses
//! `ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff` (D88), which
//! assembles the local matrix in the H(curl)-row orientation, permutes by the
//! **H(curl)** partition and keeps only the owned H(curl) rows.  For one rank
//! the two are bit-for-bit the same matrix: the serial kernel
//! `assemble_hdiv_hcurl_weak_curl` is literally
//! `assemble_hcurl_hdiv_weak_curl(...).transpose()`, and at one rank the
//! permutation is the identity and there are no ghost rows.  At 2 ranks the
//! shape changes (H(curl) owned rows instead of all-local H(curl) rows) and
//! the ghost rows are gone; see the report for the multi-rank numbers.
//!
//! ## Verification (serial, `--ranks 1`)
//!
//! Reference: `mpicxx … maxwell.cpp maxwell_solver.cpp` against MFEM 4.10
//! (`$HOME/mfem410_mpi`), 1 rank.  Command (both sides):
//! `-m <fichera.mesh> -rs 3 -ts 0.25 -tf 10 -dp "-0.5 -0.5 0.0 -0.5 -0.5 1.0
//! 0.1 1 .5 1" -dbcs -1 -no-vis -no-visit`.
//!
//! * `Number of H(Curl) unknowns: 12336` / `Number of H(Div)  unknowns: 11520`
//!   — equal;
//! * the whole stdout (banner, `Options used:` dump, ctor progress lines, dof
//!   counts, `Number of Time Steps: 100`, `Time Step Size: 0.1ns`, and all 40
//!   `Energy(<t>ns):  <e>J` lines of the 100-step run) is **byte-identical**
//!   apart from the mesh path string and the `Maximum Time Step` line above.
//!   After the D99 swap the run is byte-identical to the pre-D99 run as well
//!   (`diff` of the two `--ranks 1` logs is empty);
//! * the E/B split of the energy agrees too: the first-step invariants
//!   `J^T M1^-1 J = 3.193661332363493e8` and `|M1^-1 J| = 2.942452391632176e10`
//!   are equal to the last digit against an independent C++ probe built from
//!   public MFEM API.
//!
//! Usage:
//!   cargo run --release --example miniapp_maxwell -- -m data/fichera.mesh -rs 3 -ts 0.25 -tf 10 -dp "-0.5 -0.5 0.0 -0.5 -0.5 1.0 0.1 1 .5 1" -dbcs -1 -no-vis -no-visit
//!   cargo run --release --example miniapp_maxwell -- -m data/fichera.mesh -no-vis -no-visit

use std::sync::Arc;

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::postproc::coefficient::FnCoeff;
use fem_assembly::standard::{VectorDomainLFIntegrator, VectorMassIntegrator};
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_discrete_operator::ParDiscreteLinearOperator;
use fem_parallel::par_mesh::ParallelMesh;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{
    Comm, ParCsrMatrix, ParMixedAssembler, ParVector, ParVectorAssembler, ParallelFESpace,
    WorkerConfig,
};
use fem_solver::{SolverConfig, SiavSolver, SiaState, TimeDependentOperator};
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::{HCurlSpace, HDivSpace};

// ─── Physical constants (electromagnetics.hpp) ───────────────────────────────

/// Permittivity of free space (F/m) — `epsilon0_`.
const EPSILON0: f64 = 8.854_187_8176e-12;
/// Permeability of free space (H/m) — `mu0_`.
const MU0: f64 = 4.0e-7 * std::f64::consts::PI;
/// Scale factor between input time units and seconds — `tScale_`.
const T_SCALE: f64 = 1.0e-9;
/// Scale used to convert dt to an integer key — `MaxwellSolver::dtScale_`.
const DT_SCALE: f64 = 1.0e6;
/// `MaxwellSolver::logging_` default.
const LOGGING: i32 = 1;
/// `HyprePCG` tolerance / iteration cap of `setupSolver`.
const SOLVER_TOL: f64 = 1.0e-12;
const SOLVER_MAX_IT: u32 = 200;

/// `display_banner` (maxwell.cpp), each line padded to the 54 columns of the
/// C++ string literals.
const BANNER: [&str; 7] = [
    "     ___    ____",
    "    /   |  /   /                           __   __",
    "   /    |_/ _ /__  ___  _____  _  __ ____ |  | |  |",
    "  /         \\__  \\ \\  \\/  /\\ \\/ \\/ // __ \\|  | |  |",
    " /   /|_/   // __ \\_>    <  \\     /\\  ___/|  |_|  |__",
    "/___/  /_  /(____  /__/\\_ \\  \\/\\_/  \\___  >____/____/",
    "         \\/       \\/      \\/             \\/",
];

// ─── C++ `ostream` default float formatting (`%g`, precision 6) ──────────────

/// MFEM prints every scalar of this miniapp through `std::ostream` with its
/// default precision 6, i.e. `printf("%g")`.  Rust's `{}` prints the shortest
/// round-trip form instead, so the energy table would not line up.
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

// ─── Option parsing (`OptionsParser` parity) ────────────────────────────────

struct Opts {
    mesh: String,
    s_order: u32,
    t_order: u32,
    serial_ref_levels: u32,
    parallel_ref_levels: u32,
    dt_safety_factor: f64,
    ti: f64,
    tf: f64,
    ts: f64,
    ds_params: Vec<f64>,
    ms_params: Vec<f64>,
    cs_params: Vec<f64>,
    dp_params: Vec<f64>,
    abcs: Vec<i32>,
    dbcs: Vec<i32>,
    visualization: bool,
    visit: bool,
    visport: u32,
    /// Not a C++ option: the fem-rs parallel launcher's rank count.  The C++
    /// reference is `mpirun -np <ranks>`, so `--ranks 1` is the serial 1:1
    /// comparison and the default.
    ranks: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            mesh: "data/ball-nurbs.mesh".to_string(),
            s_order: 1,
            t_order: 1,
            serial_ref_levels: 0,
            parallel_ref_levels: 0,
            dt_safety_factor: 0.95,
            ti: 0.0,
            tf: 40.0,
            ts: 1.0,
            ds_params: Vec::new(),
            ms_params: Vec::new(),
            cs_params: Vec::new(),
            dp_params: Vec::new(),
            abcs: Vec::new(),
            dbcs: Vec::new(),
            visualization: true,
            visit: true,
            visport: 19916,
            ranks: 1,
        }
    }
}

impl Opts {
    /// Same options, same defaults as `maxwell.cpp`'s `OptionsParser`.
    ///
    /// A token is consumed as an option value while it parses as a number, so
    /// `-dbcs -1` (all boundary attributes) works like MFEM's parser, and a
    /// following `-flag` stops the list.
    fn parse(args: &[String]) -> Self {
        let mut o = Opts::default();
        let mut i = 0;
        while i < args.len() {
            let a = args[i].as_str();
            let nums = |i: &mut usize| -> Vec<String> {
                // MFEM's OptionsParser accepts one quoted, whitespace-separated
                // value (e.g. -dp "-0.5 -0.5 0 ...") as well as several tokens,
                // and stops at the next `-flag`.
                let mut out = Vec::new();
                while *i + 1 < args.len() {
                    let pieces: Vec<&str> =
                        args[*i + 1].split_whitespace().collect();
                    if pieces.is_empty()
                        || !pieces.iter().all(|p| p.parse::<f64>().is_ok())
                    {
                        break;
                    }
                    out.extend(pieces.iter().map(|s| s.to_string()));
                    *i += 1;
                }
                out
            };
            match a {
                "-m" | "--mesh" => {
                    o.mesh = args.get(i + 1).cloned().unwrap_or_default();
                    i += 1;
                }
                "-so" | "--spatial-order" => {
                    o.s_order = args[i + 1].parse().expect("bad -so");
                    i += 1;
                }
                "-to" | "--temporal-order" => {
                    o.t_order = args[i + 1].parse().expect("bad -to");
                    i += 1;
                }
                "-rs" | "--serial-ref-levels" => {
                    o.serial_ref_levels = args[i + 1].parse().expect("bad -rs");
                    i += 1;
                }
                "-rp" | "--parallel-ref-levels" => {
                    o.parallel_ref_levels = args[i + 1].parse().expect("bad -rp");
                    i += 1;
                }
                "-sf" | "--dt-safety-factor" => {
                    o.dt_safety_factor = args[i + 1].parse().expect("bad -sf");
                    i += 1;
                }
                "-ti" | "--initial-time" => {
                    o.ti = args[i + 1].parse().expect("bad -ti");
                    i += 1;
                }
                "-tf" | "--final-time" => {
                    o.tf = args[i + 1].parse().expect("bad -tf");
                    i += 1;
                }
                "-ts" | "--snapshot-time" => {
                    o.ts = args[i + 1].parse().expect("bad -ts");
                    i += 1;
                }
                "-ds" | "--dielectric-sphere-params" => {
                    o.ds_params = nums(&mut i).iter().map(|s| s.parse().unwrap()).collect();
                }
                "-ms" | "--magnetic-shell-params" => {
                    o.ms_params = nums(&mut i).iter().map(|s| s.parse().unwrap()).collect();
                }
                "-cs" | "--conductive-sphere-params" => {
                    o.cs_params = nums(&mut i).iter().map(|s| s.parse().unwrap()).collect();
                }
                "-dp" | "--dipole-pulse-params" => {
                    o.dp_params = nums(&mut i).iter().map(|s| s.parse().unwrap()).collect();
                }
                "-abcs" | "--absorbing-bc-surf" => {
                    o.abcs = nums(&mut i).iter().map(|s| s.parse().unwrap()).collect();
                }
                "-dbcs" | "--dirichlet-bc-surf" => {
                    o.dbcs = nums(&mut i).iter().map(|s| s.parse().unwrap()).collect();
                }
                "-vis" | "--visualization" => o.visualization = true,
                "-no-vis" | "--no-visualization" => o.visualization = false,
                "-visit" | "--visit" => o.visit = true,
                "-no-visit" => o.visit = false,
                "-p" | "--send-port" => {
                    o.visport = args[i + 1].parse().expect("bad -p");
                    i += 1;
                }
                "--ranks" => {
                    o.ranks = args[i + 1].parse().expect("bad --ranks");
                    i += 1;
                }
                other => panic!("unknown option {other}"),
            }
            i += 1;
        }
        o
    }

    /// `args.PrintOptions(cout)`: the long name and the current value.
    fn print_options(&self) {
        let arr = |v: &[f64]| {
            if v.is_empty() {
                "''".to_string()
            } else {
                format!("'{}'", v.iter().map(|&x| g6(x)).collect::<Vec<_>>().join(" "))
            }
        };
        let iarr = |v: &[i32]| {
            if v.is_empty() {
                "''".to_string()
            } else {
                format!(
                    "'{}'",
                    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
                )
            }
        };
        println!("Options used:");
        println!("   --mesh {}", self.mesh);
        println!("   --spatial-order {}", self.s_order);
        println!("   --temporal-order {}", self.t_order);
        println!("   --serial-ref-levels {}", self.serial_ref_levels);
        println!("   --parallel-ref-levels {}", self.parallel_ref_levels);
        println!("   --dt-safety-factor {}", g6(self.dt_safety_factor));
        println!("   --initial-time {}", g6(self.ti));
        println!("   --final-time {}", g6(self.tf));
        println!("   --snapshot-time {}", g6(self.ts));
        println!("   --dielectric-sphere-params {}", arr(&self.ds_params));
        println!("   --magnetic-shell-params {}", arr(&self.ms_params));
        println!("   --conductive-sphere-params {}", arr(&self.cs_params));
        println!("   --dipole-pulse-params {}", arr(&self.dp_params));
        println!("   --absorbing-bc-surf {}", iarr(&self.abcs));
        println!("   --dirichlet-bc-surf {}", iarr(&self.dbcs));
        println!(
            "   {}",
            if self.visualization {
                "--visualization"
            } else {
                "--no-visualization"
            }
        );
        println!(
            "   {}",
            if self.visit {
                "--visit"
            } else {
                "--no-visualization"
            }
        );
        println!("   --send-port {}", self.visport);
    }
}

// ─── Coefficient functions (maxwell.cpp, verbatim) ──────────────────────────

/// `dielectric_sphere`: ε = εr·ε0 inside the sphere, ε0 outside.
/// Params: `[cx cy cz R eps_r]`.
fn dielectric_sphere(ds: &[f64], x: &[f64]) -> f64 {
    let r2: f64 = x.iter().zip(ds).map(|(a, b)| (a - b) * (a - b)).sum();
    if r2.sqrt() <= ds[x.len()] {
        ds[x.len() + 1] * EPSILON0
    } else {
        EPSILON0
    }
}

/// `magnetic_shell`: μ = μr·μ0 inside the shell, μ0 outside.
/// Params: `[cx cy cz R_in R_out mu_r]`.
fn magnetic_shell(ms: &[f64], x: &[f64]) -> f64 {
    let r2: f64 = x.iter().zip(ms).map(|(a, b)| (a - b) * (a - b)).sum();
    let r = r2.sqrt();
    if r >= ms[x.len()] && r <= ms[x.len() + 1] {
        MU0 * ms[x.len() + 2]
    } else {
        MU0
    }
}

/// `conductive_sphere`: σ = σ0 inside the sphere, 0 outside.
/// Params: `[cx cy cz R sigma]`.
fn conductive_sphere(cs: &[f64], x: &[f64]) -> f64 {
    let r2: f64 = x.iter().zip(cs).map(|(a, b)| (a - b) * (a - b)).sum();
    if r2.sqrt() <= cs[x.len()] {
        cs[x.len() + 1]
    } else {
        0.0
    }
}

/// `dipole_pulse`: cylindrical rod of current with a derivative-of-Gaussian
/// time dependence.  Params: `[axis start (3), axis end (3), radius,
/// amplitude, pulse center (ns), pulse width (ns)]`.
fn dipole_pulse(dp: &[f64], x: &[f64], t: f64) -> [f64; 3] {
    let dim = x.len();
    let mut v = [0.0; 3];
    let mut h = 0.0;
    for i in 0..dim {
        v[i] = dp[dim + i] - dp[i];
        h += v[i] * v[i];
    }
    h = h.sqrt();
    if h == 0.0 {
        return [0.0; 3];
    }
    for i in 0..dim {
        v[i] /= h;
    }

    let r = dp[2 * dim];
    let a = dp[2 * dim + 1] * T_SCALE;
    let b = dp[2 * dim + 2] * T_SCALE;
    let c = dp[2 * dim + 3] * T_SCALE;

    // xv = xu·v with xu = x - (axis start); then the perpendicular part is
    // xu -= xv·v (MFEM `xu.Add(-xv, v)` — the full xv is used, so the two
    // loops cannot be merged).
    let mut xv = 0.0;
    for i in 0..dim {
        xv += (x[i] - dp[i]) * v[i];
    }
    let mut xp2 = 0.0;
    for i in 0..dim {
        let perp = x[i] - dp[i] - xv * v[i];
        xp2 += perp * perp;
    }
    let xp = xp2.sqrt();

    let mut j = [0.0; 3];
    if xv >= 0.0 && xv <= h && xp <= r {
        j.copy_from_slice(&v);
    }
    let scale = a * (t - b) * (-0.5 * ((t - b) / c).powi(2)).exp() / (c * c);
    for d in 0..dim {
        j[d] *= scale;
    }
    j
}

/// `SnapTimeStep`: round `dt` down so that `tmax` is an integer multiple.
fn snap_time_step(tmax: f64, dtmax: f64, dt: &mut f64) -> i32 {
    let dsteps = tmax / dtmax;
    let mut nsteps = 10.0_f64.powf(dsteps.log10().ceil()) as i32;
    for i in 1..=5 {
        let a = (dsteps / 5.0_f64.powi(i)).log10().ceil() as i32;
        let nstepsi = 5.0_f64.powi(i) as i32 * std::cmp::max(1, 10.0_f64.powi(a) as i32);
        nsteps = std::cmp::min(nsteps, nstepsi);
    }
    *dt = tmax / nsteps as f64;
    nsteps
}

// ─── Symplectic integrator (`linalg/ode.cpp`: SIASolver / SIAVSolver) ───────
//
// The tables and the stage sequence now live in the library
// (`fem_solver::mfem_ode::SiavSolver` — a 1:1 port of `SIAVSolver::Step`) and
// are shared with `joule`.  What stays here is the rank-local wiring that MFEM
// expresses with raw pointers (`SIASolver::Init(Operator&, TimeDependentOperator&)`):
//
// * [`SiaRankVector`] — `SiaState` for a `ParVector`, i.e. ghost-inclusive
//   values plus the `update_ghosts()` MFEM spells out after each `Add`;
// * [`MaxwellF`] — the `F_` side, which is the `MaxwellSolver` operator
//   (`Type::EXPLICIT` in `maxwell.cpp`: `main` never passes `Type::IMPLICIT`).

/// `SiaState` for a rank-local `ParVector`.
struct SiaRankVector<'a>(&'a mut ParVector);

impl SiaState for SiaRankVector<'_> {
    fn len(&self) -> usize { self.0.as_slice().len() }
    fn slice(&self) -> &[f64] { self.0.as_slice() }
    fn slice_mut(&mut self) -> &mut [f64] { self.0.as_slice_mut() }
    /// `ParVector::update_ghosts()` — the C++ `Run` loop's explicit halo
    /// exchange after each accumulation.
    fn sync(&mut self) { self.0.update_ghosts(); }
}

/// `F_` of the SIAV system: the `MaxwellSolver` operator.
///
/// `MaxwellSolver` is `Type::EXPLICIT` (`maxwell.cpp` builds it without a
/// `Type::IMPLICIT` argument), so `SIAVSolver::Step` calls `Mult`.  For the
/// lossless operator `Mult` and `ImplicitSolve(0, ·)` are the same map — the
/// matrix does not depend on `dt` — which is what `Maxwell::implicit_solve`
/// computes with `dt = 0`.
struct MaxwellF<'a> {
    m: &'a Maxwell,
    t: f64,
}

impl MaxwellF<'_> {
    /// One `Maxwell::implicit_solve` with the given stage time and step, with
    /// the rank-local vectors copied in and out of the solver's scratch (MFEM
    /// passes the `HypreParVector`s straight through).
    ///
    /// Note the **two different spaces**: `MaxwellSolver::Mult` maps the H(div)
    /// `B` field to the H(curl) `dE/dt` (`q` is a rank-local `ParVector` of the
    /// RT space and `dp_` one of the ND space — MFEM hides this behind raw
    /// `Vector`s, `ParVector` checks the length).
    fn solve(&self, dt: f64, x: &[f64], out: &mut [f64]) {
        let mut xv = ParVector::zeros(&self.m.rt);
        let n_in = x.len().min(xv.as_slice().len());
        xv.as_slice_mut()[..n_in].copy_from_slice(&x[..n_in]);
        let mut ov = ParVector::zeros(&self.m.nd);
        self.m.implicit_solve(self.t, dt, &mut xv, &mut ov);
        let n_out = out.len().min(ov.as_slice().len());
        out[..n_out].copy_from_slice(&ov.as_slice()[..n_out]);
    }
}

impl TimeDependentOperator for MaxwellF<'_> {
    fn size(&self) -> usize { self.m.nd.dof_partition().n_owned_dofs }

    fn set_time(&mut self, t: f64) { self.t = t; }

    fn mult(&self, x: &[f64], out: &mut [f64]) { self.solve(0.0, x, out); }

    fn implicit_solve(&mut self, dt: f64, x: &[f64], out: &mut [f64]) {
        self.solve(dt, x, out);
    }

    fn is_explicit(&self) -> bool { true }
}

// ─── MaxwellSolver (maxwell_solver.cpp) ────────────────────────────────────

/// Per-rank `MaxwellSolver`: the lossless operator set, the state vectors and
/// the cached implicit operator `A1[0]`.
struct Maxwell {
    nd: ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
    rt: ParallelFESpace<HDivSpace<fem_mesh::Mesh<3>>>,
    quad_order: u8,
    /// `A1_[0] = M1(eps)` — H(curl) mass, eliminated in place by the
    /// `FormLinearSystem` equivalent.
    m1: ParCsrMatrix,
    /// `M2MuInv_` — H(div) mass with `1/mu`.
    m2: ParCsrMatrix,
    /// `WeakCurlMuInv_` — H(div) dofs → H(curl) dofs, assembled directly in
    /// the H(curl)-row orientation (`ParMixedAssembler::
    /// assemble_hdiv_hcurl_curl_with_coeff`, D88).
    weak_curl: fem_linalg::CsrMatrix<f64>,
    /// `NegCurl_` — the discrete curl, negated: H(curl) → H(div).
    neg_curl: fem_linalg::CsrMatrix<f64>,
    /// `jd_` source parameters (`-dp`), empty when absent.
    dp_params: Vec<f64>,
    /// Essential true dofs, partition-local owned indices (`dbc_dofs_`).
    dbc_dofs: Vec<usize>,
    /// `GetMaximumTimeStep()` result (cached, `dtMax_`).
    dt_max: f64,
}

impl Maxwell {
    /// Constructor, same order of construction *and* of the `logging_ > 0`
    /// progress lines as `MaxwellSolver::MaxwellSolver`.
    fn new(
        par_mesh: &ParallelMesh<fem_mesh::Mesh<3>>,
        comm: &Comm,
        opts: &Opts,
        eps: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
        mu_inv: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    ) -> Self {
        let rank = comm.rank();
        let log = |msg: &str| {
            if rank == 0 && LOGGING > 0 {
                println!("{msg}");
            }
        };

        let local_mesh = par_mesh.local_mesh().clone();
        // ND_ParFESpace(order, dim) / RT_ParFESpace(order, dim)
        // = ND_FECollection(p) / RT_FECollection(p-1) (common/pfem_extras.cpp).
        let nd = ParallelFESpace::new(
            HCurlSpace::new(local_mesh.clone(), opts.s_order as u8),
            par_mesh,
            comm.clone(),
        );
        let rt = ParallelFESpace::new(
            HDivSpace::new(local_mesh.clone(), opts.s_order.saturating_sub(1) as u8),
            par_mesh,
            comm.clone(),
        );
        let quad_order = (2 * opts.s_order + 1) as u8;

        // lossy_ = abcs.Size() > 0 || sigma_ != NULL — clipped in this port.
        if !opts.ds_params.is_empty() {
            log("Creating Permittivity Coefficient");
        }
        if !opts.ms_params.is_empty() {
            log("Creating Permeability Coefficient");
        }

        // Dirichlet boundary condition surfaces (homogeneous dEdt BC).
        let mut dbc_dofs = Vec::new();
        if !opts.dbcs.is_empty() {
            log("Configuring Dirichlet BC");
            let attrs = resolve_tags(&local_mesh, &opts.dbcs);
            for &attr in &attrs {
                for d in boundary_dofs_hcurl(&local_mesh, nd.local_space(), &[attr]) {
                    let p = nd.dof_partition().permute_dof(d) as usize;
                    if p < nd.dof_partition().n_owned_dofs {
                        dbc_dofs.push(p);
                    }
                }
            }
            dbc_dofs.sort_unstable();
            dbc_dofs.dedup();
        }

        // Bilinear forms.
        log("Creating H(Div) Mass Operator");
        let eps_c = |x: &[f64]| eps(x);
        let mu_inv_c = |x: &[f64]| mu_inv(x);
        let m1 = ParVectorAssembler::assemble_bilinear(
            &nd,
            &[&VectorMassIntegrator {
                alpha: FnCoeff(&eps_c),
            }],
            quad_order,
        );
        let m2 = ParVectorAssembler::assemble_bilinear(
            &rt,
            &[&VectorMassIntegrator {
                alpha: FnCoeff(&mu_inv_c),
            }],
            quad_order,
        );

        log("Creating Weak Curl Operator");
        // MFEM's WeakCurlMuInv_ is `ParMixedBilinearForm(HDiv, HCurl)` i.e.
        // rows = H(curl) true dofs.  D88's owned-H(curl)-row entry point
        // assembles the local matrix in that orientation, permutes it with the
        // H(curl) partition as the row partition and keeps only the owned
        // H(curl) rows — so no `.transpose()` at the call site (that would
        // transpose the already row-truncated H(div)-row matrix and expose
        // rows for every local H(curl) DOF, ghosts included).
        let weak_curl = ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff(
            &nd,
            &rt,
            quad_order,
            FnCoeff(&mu_inv_c),
        );

        log("Creating discrete curl operator");
        let mut neg_curl = ParDiscreteLinearOperator::curl_3d(&nd, &rt);
        for v in neg_curl.values.iter_mut() {
            *v = -*v;
        }

        // Build grid functions / linear forms (maxwell_solver.cpp ctor).
        if !opts.dp_params.is_empty() {
            log("Creating Current Source");
        }

        let mut s = Maxwell {
            nd,
            rt,
            quad_order,
            m1,
            m2,
            weak_curl,
            neg_curl,
            dp_params: opts.dp_params.clone(),
            dbc_dofs,
            dt_max: -1.0,
        };
        // `dtMax_ = GetMaximumTimeStep();` is the last statement of the ctor
        // (it prints "Creating implicit operator for dt = 0" via setupSolver).
        s.dt_max = s.maximum_time_step(comm, opts);
        s
    }

    /// `PrintSizes()`.
    fn print_sizes(&self, comm: &Comm) {
        if comm.rank() == 0 {
            println!("Number of H(Curl) unknowns: {}", self.nd.n_global_dofs());
            println!("Number of H(Div)  unknowns: {}", self.rt.n_global_dofs());
        }
    }

    /// `GetEnergy()`: `0.5 (E^T A1 E + B^T M2 B)`.
    fn energy(&self, e: &mut ParVector, b: &mut ParVector) -> f64 {
        let mut m1e = ParVector::zeros_like(e);
        let mut m2b = ParVector::zeros_like(b);
        self.m1.spmv(e, &mut m1e);
        self.m2.spmv(b, &mut m2b);
        0.5 * (e.global_dot(&m1e) + b.global_dot(&m2b))
    }

    /// `GetMaximumTimeStep()`: power iteration on `-Curl^T M2 -Curl`.
    ///
    /// Seeded with a deterministic vector in place of
    /// `HypreParVector::Randomize(1234)` (hypre RNG); the loop, its `ptol`
    /// and the 20-iteration cap are the C++ ones.
    fn maximum_time_step(
        &mut self,
        comm: &Comm,
        opts: &Opts,
    ) -> f64 {
        if comm.rank() == 0 && LOGGING > 0 {
            // `setupSolver(0, 0.0)` logs this before creating A1[0].
            println!("Creating implicit operator for dt = {}", g6(0.0));
        }
        let mut v0 = ParVector::zeros(&self.nd);
        {
            // Deterministic LCG (splitmix-style) fill of the true dofs.
            let mut state = 1234u64;
            for x in v0.as_slice_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *x = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            }
        }
        v0.update_ghosts();
        let mut v1 = ParVector::zeros(&self.nd);
        let mut u0 = ParVector::zeros(&self.rt);
        let mut hd = ParVector::zeros(&self.rt);
        let mut rhs = ParVector::zeros(&self.nd);
        let n_rt_rows = self.neg_curl.nrows;
        let curl_t = self.neg_curl.transpose();

        // `setupSolver(0, 0.0)` — creates A1[0] and the PCG/(Jacobi) solver.
        let mut a1 = self.m1.clone_vec();
        self.form_linear_system(&mut a1, &mut rhs);

        let cfg = SolverConfig {
            rtol: SOLVER_TOL,
            max_iter: SOLVER_MAX_IT as usize,
            verbose: false,
            ..SolverConfig::default()
        };

        let mut dt0 = 1.0;
        let mut change = 1.0;
        let mut iter = 0;
        while iter < 20 && change > 0.001 {
            let norm_v0 = v0.global_dot(&v0);
            let scale = 1.0 / norm_v0.sqrt();
            for x in v0.as_slice_mut() {
                *x *= scale;
            }

            self.neg_curl
                .spmv(v0.as_slice(), &mut u0.as_slice_mut()[..n_rt_rows]);
            self.m2.spmv(&mut u0, &mut hd);
            let n_nd_rows = curl_t.nrows;
            curl_t.spmv(hd.as_slice(), &mut rhs.as_slice_mut()[..n_nd_rows]);

            par_solve_pcg_jacobi(&a1, &rhs, &mut v1, &cfg)
                .unwrap_or_else(|e| panic!("maxwell: power-method solve failed: {e}"));

            let lambda = v0.global_dot(&v1);
            let dt1 = 2.0 / lambda.sqrt();
            change = ((dt1 - dt0) / dt0).abs();
            dt0 = dt1;
            if comm.rank() == 0 && LOGGING > 1 {
                println!("{iter}:  {dt0} {change}");
            }
            std::mem::swap(&mut v0, &mut v1);
            iter += 1;
        }
        if opts.t_order > 4 {
            panic!("Unsupported order in SIAVSolver");
        }
        dt0
    }

    /// `a1_[idt]->FormLinearSystem(dbc_dofs_, dedt_, rhs_, A1_[idt], ..)`:
    /// the essential values are zero (`dEdtBCFunc`), so this is the
    /// symmetric row/column elimination with a unit diagonal.
    fn form_linear_system(&self, a1: &mut ParCsrMatrix, rhs: &mut ParVector) {
        for &p in &self.dbc_dofs {
            a1.apply_dirichlet_par(p, 0.0, rhs);
        }
    }

    /// `implicitSolve(dt, B, dEdt)` for the lossless path: the current
    /// source is re-assembled at time `t` and the H(curl) mass system is
    /// solved with PCG + diagonal preconditioning.
    fn implicit_solve(
        &self,
        t: f64,
        dt: f64,
        b_vec: &mut ParVector,
        out: &mut ParVector,
    ) {
        let mut rhs = ParVector::zeros(&self.nd);
        let n_rows = self.weak_curl.nrows;
        self.weak_curl
            .spmv(b_vec.as_slice(), &mut rhs.as_slice_mut()[..n_rows]);

        if !self.dp_params.is_empty() {
            let jd = self.assemble_current(t);
            for (r, j) in rhs.as_slice_mut().iter_mut().zip(jd.as_slice().iter()) {
                *r -= *j;
            }
        }

        // setupSolver(idt, dt): only `idt == 0` exists on the lossless path.
        let _ = dt;
        let mut a1 = self.m1.clone_vec();
        self.form_linear_system(&mut a1, &mut rhs);

        let cfg = SolverConfig {
            rtol: SOLVER_TOL,
            max_iter: SOLVER_MAX_IT as usize,
            verbose: false,
            ..SolverConfig::default()
        };
        for v in out.as_slice_mut() {
            *v = 0.0;
        }
        par_solve_pcg_jacobi(&a1, &rhs, out, &cfg)
            .unwrap_or_else(|e| panic!("maxwell: implicit solve failed: {e}"));
        out.update_ghosts();
    }

    /// `jd_->Assemble()` with `jCoef_->SetTime(t)`.
    fn assemble_current(&self, t: f64) -> ParVector {
        let dp = self.dp_params.clone();
        let f = FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
            let j = dipole_pulse(&dp, x, t);
            out[..3].copy_from_slice(&j[..3]);
        });
        ParVectorAssembler::assemble_linear(
            &self.nd,
            &[&VectorDomainLFIntegrator { f }],
            self.quad_order,
        )
    }
}

/// MFEM's `AttrToMarker` convention: a negative entry marks every boundary
/// attribute; otherwise the listed attributes are used.
fn resolve_tags(mesh: &fem_mesh::Mesh<3>, tags: &[i32]) -> Vec<i32> {
    if tags.iter().any(|&t| t < 0) {
        let mut all: Vec<i32> = (0..mesh.n_boundary_faces() as u32)
            .map(|f| mesh.face_tag(f))
            .collect();
        all.sort_unstable();
        all.dedup();
        all
    } else {
        tags.to_vec()
    }
}

// ─── main (maxwell.cpp) ────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = Opts::parse(&args);

    for line in BANNER {
        println!("{line:<54}");
    }
    opts.print_options();

    if opts.visualization {
        eprintln!(
            "mfem_miniapp_maxwell: -vis (GLVis socket display) is not ported; pass -no-vis"
        );
        std::process::exit(3);
    }
    if opts.visit {
        eprintln!(
            "mfem_miniapp_maxwell: -visit (VisIt DataCollection output) is not ported; pass -no-visit"
        );
        std::process::exit(3);
    }
    if !opts.abcs.is_empty() {
        eprintln!(
            "mfem_miniapp_maxwell: -abcs (absorbing boundary + implicit loss solve) is not ported yet"
        );
        std::process::exit(3);
    }
    if !opts.cs_params.is_empty() {
        eprintln!(
            "mfem_miniapp_maxwell: -cs (conductive sphere + implicit loss solve) is not ported yet"
        );
        std::process::exit(3);
    }
    if opts.t_order < 1 || opts.t_order > 4 {
        eprintln!(
            "mfem_miniapp_maxwell: -to {} is not supported by SIAVSolver (1..4)",
            opts.t_order
        );
        std::process::exit(3);
    }

    let header = std::fs::read_to_string(&opts.mesh)
        .unwrap_or_else(|_| String::new());
    if !header.starts_with("MFEM mesh") {
        if header.starts_with("MFEM NURBS mesh") {
            eprintln!(
                "mfem_miniapp_maxwell: NURBS meshes need the C++ projection \
                 (UniformRefinement + SetCurvature(2)); not ported"
            );
        } else {
            eprintln!("\nCan not open mesh file: {}\n", opts.mesh);
        }
        std::process::exit(if header.starts_with("MFEM NURBS mesh") { 3 } else { 2 });
    }

    let mfem = fem_io::mfem::read_mfem_file(&opts.mesh).unwrap_or_else(|e| {
        eprintln!("failed to read mesh {}: {e}", opts.mesh);
        std::process::exit(2);
    });
    let mut mesh0 = match mfem.mesh3d {
        Some(m) => m,
        None => {
            eprintln!(
                "mfem_miniapp_maxwell: 2-D meshes are not ported (the discrete curl chain is 3-D)"
            );
            std::process::exit(3);
        }
    };
    for _ in 0..opts.serial_ref_levels {
        mesh0 = fem_mesh::amr::refine_uniform_3d(&mesh0);
    }
    let mesh0 = Arc::new(mesh0);

    // Coefficient assignment, as in main(): NULL when the option is absent.
    let ds = opts.ds_params.clone();
    let ms = opts.ms_params.clone();
    let eps = move |x: &[f64]| -> f64 {
        if ds.is_empty() {
            EPSILON0
        } else {
            dielectric_sphere(&ds, x)
        }
    };
    let mu_inv = move |x: &[f64]| -> f64 {
        if ms.is_empty() {
            1.0 / MU0
        } else {
            1.0 / magnetic_shell(&ms, x)
        }
    };

    // Maxwell.PrintSizes() happens after the ctor; ti*T_SCALE and the time
    // loop mirror maxwell.cpp's main().
    let ti_s = opts.ti * T_SCALE;
    let tf_s = opts.tf * T_SCALE;
    let ts_s = opts.ts * T_SCALE;
    let order = opts.t_order;

    let n_workers = opts.ranks;
    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    let opts_rank = Arc::new(opts);
    launcher.launch(move |comm| {
        let par_mesh = partition_mesh(&mesh0, &comm);
        let maxwell = Maxwell::new(&par_mesh, &comm, &opts_rank, &eps, &mu_inv);
        maxwell.print_sizes(&comm);

        // SetInitialEField/SetInitialBField with the identically zero
        // EFieldFunc/BFieldFunc.
        let mut e = ParVector::zeros(&maxwell.nd);
        let mut b = ParVector::zeros(&maxwell.rt);

        if comm.rank() == 0 {
            println!(
                "Energy({}ns):  {}J",
                g6(opts_rank.ti),
                g6(maxwell.energy(&mut e, &mut b))
            );
            println!("Maximum Time Step:     {}ns", g6(maxwell.dt_max / T_SCALE));
        }

        let mut dt = 1.0e-12;
        let nsteps = snap_time_step(tf_s - ti_s, opts_rank.dt_safety_factor * maxwell.dt_max, &mut dt);
        if comm.rank() == 0 {
            println!("Number of Time Steps:  {nsteps}");
            println!("Time Step Size:        {}ns", g6(dt / T_SCALE));
        }

        // SIAVSolver siaSolver(tOrder); siaSolver.Init(NegCurl, Maxwell).
        let siav = SiavSolver::new(order as usize);
        let mut t = ti_s;
        let mut it = 1_i64;
        let mut cur = ParVector::zeros(&maxwell.nd);
        let mut db = ParVector::zeros(&maxwell.rt);
        while t < tf_s {
            // siaSolver.Run(B, E, t, dt, max(t + dt, ti + ts*it))
            // (the SIAV step itself advances `t` — like `SIASolver::Run`).
            let t_stop = f64::max(t + dt, ti_s + ts_s * it as f64);
            while t < t_stop {
                maxwell.solve_step(&mut b, &mut e, &mut cur, &mut db, &siav, &mut t, dt);
            }
            if comm.rank() == 0 {
                println!(
                    "Energy({}ns):  {}J",
                    g6(t / T_SCALE),
                    g6(maxwell.energy(&mut e, &mut b))
                );
            }
            it += 1;
        }
    });
}

impl Maxwell {
    /// One `SIAVSolver::Step(q = B, p = E)` — the library implementation of
    /// `linalg/ode.cpp:1151` (D90), wired to this miniapp's rank-local vectors:
    /// `P_` is `NegCurl` (`dq = -Curl E`), `F_` is [`MaxwellF`], and `dp`/`dq`
    /// are the scratch vectors `SIASolver::Init` sizes (`cur`, `db`).
    fn solve_step(
        &self,
        b: &mut ParVector,
        e: &mut ParVector,
        cur: &mut ParVector,
        db: &mut ParVector,
        siav: &SiavSolver,
        t: &mut f64,
        dt: f64,
    ) {
        let n_rows = self.neg_curl.nrows;
        let mut f = MaxwellF { m: self, t: *t };
        siav.step(
            &mut f,
            |x: &[f64], y: &mut [f64]| self.neg_curl.spmv(x, &mut y[..n_rows]),
            &mut SiaRankVector(b),
            &mut SiaRankVector(e),
            &mut SiaRankVector(cur),
            &mut SiaRankVector(db),
            t,
            dt,
        );
    }
}
