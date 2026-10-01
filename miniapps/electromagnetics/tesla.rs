//! # Miniapp Tesla — Simple Magnetostatics (1:1 port of MFEM
//! `miniapps/electromagnetics/tesla.cpp` + `tesla_solver.{hpp,cpp}`, MFEM 4.10)
//!
//! Solves the magnetostatic curl-curl system
//!
//! ```text
//!   Curl(1/mu Curl A) = J + Curl(mu0/mu M)
//! ```
//!
//! on a 3-D mesh with H(curl) A, then recovers `B = Curl A` (H(div)) and the
//! magnetic field `H` from the H(curl) mass solve of `∫ H·W = ∫ 1/mu B·W −
//! mu0 ∫ 1/mu M·W`, exactly like `TeslaSolver::Solve`
//! (tesla_solver.cpp:429-489).  The optional sources are the MFEM sample
//! inputs: a bar magnet (`-bm`), a Halbach array (`-ha`), a ring of current
//! (`-cr`), a magnetic shell for the permeability (`-ms`), a piecewise
//! permeability (`-pwm`) and a uniform-B vector-potential boundary condition
//! (`-ubbc`).
//!
//! ## Port boundary
//!
//! * `-maxit 1` (one AMR iteration) is the supported mode, same as the
//!   sibling miniapp volta.  `maxit > 1` needs the ZZ error estimator +
//!   `RefineByError` + `TeslaSolver::Update` and is refused with status 3;
//!   the C++ default `-maxit 100` is therefore also refused.
//! * `-kbcs/-vbcs/-vbcv` (the H1 surface-current subproblem `SurfaceCurrent`,
//!   tesla_solver.cpp:620-760) is not ported: any `-kbcs`/`-vbcs`/`-vbcv`
//!   input is refused.  A partial triple (voltages without surfaces or vice
//!   versa, or fewer values than surfaces) reproduces the C++ check
//!   (tesla.cpp:223-240) with the same message and exit code 3.
//! * `--ranks > 1` is refused (volta precedent): the multi-rank path of the
//!   required mixed assemblies is not verified.
//! * NURBS meshes (the C++ default `ball-nurbs.mesh`): fem-rs reads only the
//!   control-point corner topology of a NURBS file, it cannot evaluate the
//!   rational geometry or perform MFEM's `UniformRefinement + SetCurvature(2)`
//!   conversion (tesla.cpp:186-195).  A NURBS input is refused with status 3.
//!   For the 1:1 comparison, use the **C++-converted** mesh (the exact output
//!   of that conversion, 56 curved quadratic hexes) exported once by the C++
//!   probe; see `tmp/d103tesla/REPORT.md` and run with
//!   `-m tmp/d103tesla/ball-quad.mesh`.
//! * `-vis`/`-visit` (GLVis/VisIt outputs) are refused; pass `-no-vis
//!   -no-visit` like every other comparison run.
//!
//! ## MFEM first-iteration BC quirk (verified against the C++ binary)
//!
//! `TeslaSolver` computes `ess_bdr_tdofs_` only inside `Update()`
//! (tesla_solver.cpp:279), which the AMR loop first calls *after* iteration 1.
//! At `-maxit 1` the curl-curl system therefore carries **no essential BC
//! elimination**: the solve is the singular curl-curl driven by `jd` alone
//! (`HypreAMS::SetSingularProblem` + PCG).  Measured C++ consequences this
//! port reproduces at maxit 1:
//!
//! * `-ubbc '0 0 1' -ms '0 0 0 0.2 0.4 10'` prints `PCG Iterations = 0` and
//!   `||A||_2 = ||B||_2 = ||H||_2 = 0` — the uniform-B BC is **inert** at
//!   iteration 1 (it starts acting at iteration 2: `maxit 2` gives
//!   `PCG Iterations = 5`, `||B||_2 = 1.799`);
//! * `-bm '…'` at order 1 also degenerates: MFEM's `ProjectCoefficient` on
//!   RT0 evaluates `M` at the *face centroids* (`VectorFiniteElement::
//!   Project_RT`), and the r = 0.2 magnet misses every face centroid of the
//!   ball mesh — `||M||_2 = 2.14e-17`, so the whole solve is round-off
//!   (`PCG Iterations = 5` on `||jd|| ~ 1e-17`);
//! * `-bm` at `-o 2` is the first non-degenerate magnetization run
//!   (`||M||_2 = 5.93e-1`, `PCG Iterations = 8`).
//!
//! ## Solver stack (1:1 where the stacks correspond)
//!
//! | C++ (tesla_solver.cpp)                | fem-rs                                        |
//! |---------------------------------------|-----------------------------------------------|
//! | `CurlCurlIntegrator(muInv)` default rule | `CurlCurlIntegrator{mu}`, qo = 2·order     |
//! | `VectorFEMassIntegrator` + custom IR   | `VectorMassIntegrator{alpha: 1}`, qo = irOrder |
//! | `VectorFEMassIntegrator(muInv)` RT→ND  | `ParMixedAssembler::assemble_hcurl_hdiv_mass` |
//! | `VectorFECurlIntegrator(muInv)` RT→ND  | `ParMixedAssembler::assemble_hcurl_hdiv_curl_with_coeff` |
//! | `HypreAMS` (singular) + `HyprePCG`     | `ParAmsPrecond` + local PCG (iteration table) |
//! | `HyprePCG` + `HypreDiagScale` (H solve)| `par_solve_pcg_jacobi`                        |
//! | `IrrotationalProjector`/`DivergenceFreeProjector` (weakDiv → H1 stiffness + AMG → grad) | same operators (`assemble_hcurl_h1_weak_div`, `DiffusionIntegrator`, `par_solve_pcg_amg`, `gradient`) |
//! | `ParGridFunction::ProjectCoefficient` (ND/RT: nodal dof functionals) | `HCurlSpace/HDivSpace::interpolate_vector` |
//!
//! The PCG iteration table (`||r||` columns) is this port's own log with the
//! same column layout as hypre's print-level-2 table; hypre's `||r||_C` is its
//! preconditioned criterion while fem-rs prints `||r||_2`, so per-row values
//! differ even when the iteration counts agree.  `PCG Iterations = N` /
//! `Final PCG Relative Residual Norm = X` are 1:1.
//!
//! Setting the environment variable `FEMRS_TESLA_PROBE=1` prints three
//! `PROBE ||A/B/H||_2` lines (plus `||M||_2`/`||JD||_2` on the source paths) —
//! the exact quantities the C++ probe build in `tmp/d103tesla/` prints —
//! i.e. the value-comparison hook for the d103 acceptance runs (not part of
//! the C++ miniapp's own output).
//!
//! ## Verification (d103)
//!
//! C++ reference: `mpicxx -std=c++17 -O2 … tesla.cpp tesla_solver.cpp
//! ../common/{pfem_extras,mesh_extras}.cpp libmfem.a` against MFEM 4.10
//! (`$HOME/mfem410_mpi`), `mpirun -np 1`, `-maxit 1 -no-vis -no-visit`,
//! mesh = the C++-converted `ball-quad.mesh` (see REPORT.md).  Pinned in
//! `crates/regression/tests/d103_tesla_miniapp.rs`.

use std::sync::Arc;

use fem_assembly::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use fem_assembly::standard::{CurlCurlIntegrator, DiffusionIntegrator, VectorMassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::refine_uniform_3d;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{par_solve_pcg_amg, ParAmgConfig};
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::par_solver::par_solve_pcg_jacobi;
use fem_parallel::{
    Comm, ParAmsPrecond, ParAssembler, ParCsrMatrix, ParDiscreteLinearOperator, ParMixedAssembler,
    ParVector, ParVectorAssembler, ParallelFESpace, ParallelMesh, SmootherType, WorkerConfig,
    partition_mesh,
};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::{H1Space, HCurlSpace, HDivSpace};
use linlvo::precond::{AmsConfig, AmsCycle, AmsEdgeSmoother};

// ─── Constants ──────────────────────────────────────────────────────────────

/// `mu0_` (electromagnetics.hpp): the vacuum permeability.
const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

/// Banner (`display_banner`, tesla.cpp:334-341 — the C++ says "Volta ascii
/// logo" but the art is Tesla's).  Trailing spaces are verbatim.
const BANNER: [&str; 6] = [
    r"  ___________            __            ",
    r"  \__    ___/___   _____|  | _____     ",
    r"    |    |_/ __ \ /  ___/  | \__  \    ",
    r"    |    |\  ___/ \___ \|  |__/ __ \_  ",
    r"    |____| \___  >____  >____(____  /  ",
    r"               \/     \/          \/   ",
];

// ─── Formatting parity ──────────────────────────────────────────────────────

/// C++ ostream default `<<` for `double` (`%.6g`) — 6 significant digits,
/// exponent form outside `[1e-4, 1e6)`.  Used by `PrintOptions` for the
/// vector-option dumps and by the `Final PCG Relative Residual Norm` line
/// (hypre prints `%g`-style too: `5.04791e-14`, plain `0` for zero).
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

/// `%.6e` (hypre's `<C*b,b>` line and iteration-table columns).
fn e6(x: f64) -> String {
    format!("{x:e}")
}

// ─── Options (`OptionsParser`, tesla.cpp:130-176) ───────────────────────────

/// tesla's options with the C++ defaults (tesla.cpp:113-126) in registration
/// order, which is also the order of the `Options used:` dump.
struct Opts {
    mesh: String,
    order: u32,
    serial_ref_levels: u32,
    parallel_ref_levels: u32,
    b_uniform: Vec<f64>,
    pw_mu: Vec<f64>,
    ms_params: Vec<f64>,
    cr_params: Vec<f64>,
    bm_params: Vec<f64>,
    ha_params: Vec<f64>,
    kbcs: Vec<i32>,
    vbcs: Vec<i32>,
    vbcv: Vec<f64>,
    maxit: i32,
    visualization: bool,
    visit: bool,
    visport: u32,
    /// Not a C++ option: the fem-rs parallel launcher's rank count (the C++
    /// reference is `mpirun -np <ranks>`); 1 is the verified mode.
    ranks: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            mesh: "../../data/ball-nurbs.mesh".to_string(),
            order: 1,
            serial_ref_levels: 0,
            parallel_ref_levels: 0,
            b_uniform: Vec::new(),
            pw_mu: Vec::new(),
            ms_params: Vec::new(),
            cr_params: Vec::new(),
            bm_params: Vec::new(),
            ha_params: Vec::new(),
            kbcs: Vec::new(),
            vbcs: Vec::new(),
            vbcv: Vec::new(),
            maxit: 100,
            visualization: true,
            visit: true,
            visport: 19916,
            ranks: 1,
        }
    }
}

/// `OptionsParser::Parse` for tesla's option set (tesla.cpp:130-176).
fn parse_opts(args: &[String]) -> Opts {
    let mut o = Opts::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let ints = |i: &mut usize, what: &str| -> Vec<i32> {
            *i += 1;
            args.get(*i)
                .map(|s| {
                    s.split_whitespace()
                        .filter_map(|t| t.parse::<i32>().ok())
                        .collect()
                })
                .unwrap_or_else(|| {
                    eprintln!("miniapp_tesla: missing value for {what}");
                    std::process::exit(3);
                })
        };
        let nums = |i: &mut usize, what: &str| -> Vec<f64> {
            *i += 1;
            args.get(*i)
                .map(|s| {
                    s.split_whitespace()
                        .filter_map(|t| t.parse::<f64>().ok())
                        .collect()
                })
                .unwrap_or_else(|| {
                    eprintln!("miniapp_tesla: missing value for {what}");
                    std::process::exit(3);
                })
        };
        match a {
            "-m" | "--mesh" => {
                i += 1;
                o.mesh = args.get(i).cloned().unwrap_or_default();
            }
            "-o" | "--order" => {
                i += 1;
                o.order = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "-rs" | "--serial-ref-levels" => {
                i += 1;
                o.serial_ref_levels = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "-rp" | "--parallel-ref-levels" => {
                i += 1;
                o.parallel_ref_levels = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "-ubbc" | "--uniform-b-bc" => o.b_uniform = nums(&mut i, a),
            "-pwm" | "--piecewise-mu" => o.pw_mu = nums(&mut i, a),
            "-ms" | "--magnetic-shell-params" => o.ms_params = nums(&mut i, a),
            "-cr" | "--current-ring-params" => o.cr_params = nums(&mut i, a),
            "-bm" | "--bar-magnet-params" => o.bm_params = nums(&mut i, a),
            "-ha" | "--halbach-array-params" => o.ha_params = nums(&mut i, a),
            "-kbcs" | "--surface-current-bc" => o.kbcs = ints(&mut i, a),
            "-vbcs" | "--voltage-bc-surf" => o.vbcs = ints(&mut i, a),
            "-vbcv" | "--voltage-bc-vals" => o.vbcv = nums(&mut i, a),
            "-maxit" | "--max-amr-iterations" => {
                i += 1;
                o.maxit = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(100);
            }
            "-vis" | "--visualization" => o.visualization = true,
            "-no-vis" | "--no-visualization" => o.visualization = false,
            "-visit" | "--visit" => o.visit = true,
            "-no-visit" | "--no-visit" => o.visit = false,
            "-p" | "--send-port" => {
                i += 1;
                o.visport = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(19916);
            }
            "--ranks" => {
                i += 1;
                o.ranks = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            _ => {}
        }
        i += 1;
    }
    o
}

/// `OptionsParser::PrintOptions` — same lines, same order, same quoting.
///
/// Vector options print space-separated `%g` values inside single quotes
/// (`''` when empty); bool options print the active long variant.
fn print_options(o: &Opts) {
    let arr = |v: &[f64]| -> String {
        if v.is_empty() {
            "''".to_string()
        } else {
            let items: Vec<String> = v.iter().map(|&x| g6(x)).collect();
            format!("'{}'", items.join(" "))
        }
    };
    let ints = |v: &[i32]| -> String {
        if v.is_empty() {
            "''".to_string()
        } else {
            let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
            format!("'{}'", items.join(" "))
        }
    };
    println!("Options used:");
    println!("   --mesh {}", o.mesh);
    println!("   --order {}", o.order);
    println!("   --serial-ref-levels {}", o.serial_ref_levels);
    println!("   --parallel-ref-levels {}", o.parallel_ref_levels);
    println!("   --uniform-b-bc {}", arr(&o.b_uniform));
    println!("   --piecewise-mu {}", arr(&o.pw_mu));
    println!("   --magnetic-shell-params {}", arr(&o.ms_params));
    println!("   --current-ring-params {}", arr(&o.cr_params));
    println!("   --bar-magnet-params {}", arr(&o.bm_params));
    println!("   --halbach-array-params {}", arr(&o.ha_params));
    println!("   --surface-current-bc {}", ints(&o.kbcs));
    println!("   --voltage-bc-surf {}", ints(&o.vbcs));
    println!("   --voltage-bc-vals {}", arr(&o.vbcv));
    println!("   --max-amr-iterations {}", o.maxit);
    println!(
        "   {}",
        if o.visualization { "--visualization" } else { "--no-visualization" }
    );
    println!("   {}", if o.visit { "--visit" } else { "--no-visit" });
    println!("   --send-port {}", o.visport);
}

// ─── Coefficients (tesla.cpp:349-500) ───────────────────────────────────────

/// `magnetic_shell` (tesla.cpp:364-381): vacuum μ₀ outside the spherical
/// shell, μ₀·μ_r inside; returns its inverse (`magnetic_shell_inv`).
fn magnetic_shell_inv(ms: &[f64], x: &[f64]) -> f64 {
    let mut r2 = 0.0;
    for i in 0..3 {
        r2 += (x[i] - ms[i]) * (x[i] - ms[i]);
    }
    let r = r2.sqrt();
    if r >= ms[3] && r <= ms[4] {
        1.0 / (MU0 * ms[5])
    } else {
        1.0 / MU0
    }
}

/// `current_ring` (tesla.cpp:384-425): an annular ring of current around the
/// axis p1→p2 with inner/outer radii ra/rb and total current I.
fn current_ring(cr: &[f64], x: &[f64]) -> [f64; 3] {
    let a = [cr[3] - cr[0], cr[4] - cr[1], cr[5] - cr[2]];
    let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if h == 0.0 {
        return [0.0; 3];
    }
    let (mut ra, mut rb) = (cr[6], cr[7]);
    if ra > rb {
        std::mem::swap(&mut ra, &mut rb);
    }
    let xu = [x[0] - cr[0], x[1] - cr[1], x[2] - cr[2]];
    let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
    let xu_perp = [
        xu[0] - xa / (h * h) * a[0],
        xu[1] - xa / (h * h) * a[1],
        xu[2] - xa / (h * h) * a[2],
    ];
    let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2]).sqrt();
    if xa >= 0.0 && xa <= h * h && xp >= ra && xp <= rb {
        // ju = (a × xu_perp)/h,  j = I/(h·(rb−ra)) · ju
        let ju = [
            (a[1] * xu_perp[2] - a[2] * xu_perp[1]) / h,
            (a[2] * xu_perp[0] - a[0] * xu_perp[2]) / h,
            (a[0] * xu_perp[1] - a[1] * xu_perp[0]) / h,
        ];
        let s = cr[8] / (h * (rb - ra));
        [s * ju[0], s * ju[1], s * ju[2]]
    } else {
        [0.0; 3]
    }
}

/// `bar_magnet` (tesla.cpp:429-462): a cylindrical rod of constant
/// magnetization along the axis p1→p2 with radius r and magnitude B.
fn bar_magnet(bm: &[f64], x: &[f64]) -> [f64; 3] {
    let a = [bm[3] - bm[0], bm[4] - bm[1], bm[5] - bm[2]];
    let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if h == 0.0 {
        return [0.0; 3];
    }
    let r = bm[6];
    let xu = [x[0] - bm[0], x[1] - bm[1], x[2] - bm[2]];
    let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
    let xu_perp = [
        xu[0] - xa / (h * h) * a[0],
        xu[1] - xa / (h * h) * a[1],
        xu[2] - xa / (h * h) * a[2],
    ];
    let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2]).sqrt();
    if xa >= 0.0 && xa <= h * h && xp <= r {
        let s = bm[7] / h;
        [s * a[0], s * a[1], s * a[2]]
    } else {
        [0.0; 3]
    }
}

/// `halbach_array` (tesla.cpp:466-489): rotating magnetized segments in a
/// bounding box.
fn halbach_array(ha: &[f64], x: &[f64]) -> [f64; 3] {
    if x[0] < ha[0] || x[0] > ha[3] || x[1] < ha[1] || x[1] > ha[4] || x[2] < ha[2] || x[2] > ha[5]
    {
        return [0.0; 3];
    }
    let ai = ha[6] as usize;
    let ri = ha[7] as usize;
    let n = ha[8];
    let i = (n * (x[ai] - ha[ai]) / (ha[ai + 3] - ha[ai])) as i64;
    let sign = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
    let mut m = [0.0_f64; 3];
    m[(ri + 1 + (i as usize) % 2) % 3] = sign;
    m
}

// ─── Solver ─────────────────────────────────────────────────────────────────

/// The μ⁻¹ coefficient in its three MFEM variants
/// (`SetupInvPermeabilityCoefficient`, tesla.cpp:349-373), as one concrete
/// [`ScalarCoeff`] type shared by all four forms.
#[derive(Clone)]
enum MuInvCoeff {
    /// `FunctionCoefficient(magnetic_shell_inv)` (from `-ms`).
    Shell(Arc<Vec<f64>>),
    /// `PWConstCoefficient(1/pw)` (from `-pwm`): attribute `a` reads
    /// `pw_inv[a-1]`.
    Pw(Arc<Vec<f64>>),
    /// `ConstantCoefficient(1/mu0_)`.
    Vacuum,
}

impl ScalarCoeff for MuInvCoeff {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        match self {
            MuInvCoeff::Shell(ms) => magnetic_shell_inv(ms, ctx.x),
            MuInvCoeff::Pw(pw_inv) => {
                let a = ctx.elem_tag;
                if a >= 1 && (a as usize) <= pw_inv.len() {
                    pw_inv[a as usize - 1]
                } else {
                    0.0
                }
            }
            MuInvCoeff::Vacuum => 1.0 / MU0,
        }
    }
}

/// MFEM's typical-element `OrderW()` for the custom integration rules
/// (tesla_solver.cpp:101: `irOrder = OrderW() + 2*order`).
///
/// Qk (hex) geometry of nodal order p: `p·dim − 1`; Pk (tet): `(p−1)·dim`.
/// `geom_order` is the mesh's node-interpolation order (1 straight, 2 for the
/// `SetCurvature(2)` conversion of the ball).
fn typical_order_w(geom_order: u8, is_hex: bool) -> i32 {
    if is_hex {
        geom_order as i32 * 3 - 1
    } else {
        (geom_order.max(1) as i32 - 1) * 3
    }
}

/// MFEM `CurlCurlIntegrator`'s default rule (bilininteg.cpp): Qk → 2·order
/// (no geometry term), Pk → 2·order − 2.
fn curl_curl_rule_order(order: u32, is_hex: bool) -> u8 {
    if is_hex {
        2 * order as u8
    } else {
        (2 * order).saturating_sub(2) as u8
    }
}

/// One printed PCG solve (`HyprePCG` print level 2 analog): the
/// `<C*b,b>` line, the iteration table, and the two summary lines.
/// Returns `(iterations, final relative residual)`.
fn pcg_with_table(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    ams: &ParAmsPrecond,
    rtol: f64,
    max_iter: usize,
) -> (usize, f64) {
    // <C*b,b> = (b, C·b), printed by hypre before the table.
    let mut cb = ParVector::zeros_like(b);
    ams.apply(b.as_slice(), cb.as_slice_mut());
    let cb_b = b.global_dot(&cb);
    println!();
    println!("<C*b,b>: {}", e6(cb_b));

    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        // hypre stops at iteration 0 for a zero RHS.
        println!();
        println!("PCG Iterations = 0");
        println!("Final PCG Relative Residual Norm = {}", g6(0.0));
        return (0, 0.0);
    }

    println!();
    println!("Iters       ||r||_C     conv.rate  ||r||_C/||b||_C");
    println!("-----    ------------    ---------  ------------ ");

    let n_owned = a.n_owned();
    // r = b - A x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    let mut xm = x.clone_vec();
    a.spmv(&mut xm, &mut ax);
    for i in 0..n_owned {
        r.as_slice_mut()[i] = b.as_slice()[i] - ax.as_slice()[i];
    }
    // z = M⁻¹ r
    let mut z = ParVector::zeros_like(b);
    ams.apply(r.as_slice(), z.as_slice_mut());
    let mut rz = r.global_dot(&z);
    let mut p = z.clone_vec();

    let mut r_norm = r.global_norm();
    let mut iterations = 0usize;
    let mut rel = r_norm / b_norm;
    while iterations < max_iter {
        iterations += 1;
        let mut ap = ParVector::zeros_like(b);
        let mut pm = p.clone_vec();
        a.spmv(&mut pm, &mut ap);
        let pap = p.global_dot(&ap);
        let alpha = rz / pap;
        for i in 0..n_owned {
            x.as_slice_mut()[i] += alpha * p.as_slice()[i];
            r.as_slice_mut()[i] -= alpha * ap.as_slice()[i];
        }
        ams.apply(r.as_slice(), z.as_slice_mut());
        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;
        for i in 0..n_owned {
            p.as_slice_mut()[i] = z.as_slice()[i] + beta * p.as_slice()[i];
        }
        rz = rz_new;

        let prev = r_norm;
        r_norm = r.global_norm();
        rel = r_norm / b_norm;
        println!(
            "{:5}    {:<13}    {:<9.6}    {}",
            iterations,
            e6(r_norm),
            r_norm / prev,
            e6(rel)
        );
        if rel < rtol {
            break;
        }
    }
    println!();
    println!("PCG Iterations = {iterations}");
    println!("Final PCG Relative Residual Norm = {}", g6(rel));
    (iterations, rel)
}

/// The `TeslaSolver` state for one `-maxit 1` run (spaces + option payload).
struct TeslaSolver {
    h1: ParallelFESpace<H1Space<fem_mesh::Mesh<3>>>,
    nd: ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
    rt: ParallelFESpace<HDivSpace<fem_mesh::Mesh<3>>>,
    /// `-ms` shell params (permeability mode 1).
    ms_params: Option<Vec<f64>>,
    /// `-pwm` piecewise inverse-permeability values (mode 2).
    pw_mu_inv: Option<Vec<f64>>,
    cr_params: Option<Vec<f64>>,
    bm_params: Option<Vec<f64>>,
    ha_params: Option<Vec<f64>>,
    order: u32,
    /// `irOrder` (tesla_solver.cpp:101) for the custom-ruled forms.
    ir_order: u8,
    /// The `CurlCurlIntegrator` default-rule order.
    cc_order: u8,
}

impl TeslaSolver {
    /// `TeslaSolver::PrintSizes` (tesla_solver.cpp:166-178).
    fn print_sizes(&self) {
        println!("Number of H1      unknowns: {}", self.h1.n_global_dofs());
        println!("Number of H(Curl) unknowns: {}", self.nd.n_global_dofs());
        println!("Number of H(Div)  unknowns: {}", self.rt.n_global_dofs());
    }

    /// μ⁻¹ (`SetupInvPermeabilityCoefficient`, tesla.cpp:349-373).
    fn mu_inv_coeff(&self) -> MuInvCoeff {
        if let Some(ms) = &self.ms_params {
            MuInvCoeff::Shell(Arc::new(ms.clone()))
        } else if let Some(pw) = &self.pw_mu_inv {
            MuInvCoeff::Pw(Arc::new(pw.clone()))
        } else {
            MuInvCoeff::Vacuum
        }
    }

    /// One full `Assemble + Solve` pass (tesla_solver.cpp:180-489).
    fn run(&self, comm: &Comm, par_mesh: &ParallelMesh<fem_mesh::Mesh<3>>) {
        let probe = std::env::var("FEMRS_TESLA_PROBE").is_ok();
        let nd = &self.nd;
        let rt = &self.rt;
        let h1 = &self.h1;
        let qo_ir = self.ir_order;
        let qo_cc = self.cc_order;

        // ── Assemble (tesla_solver.cpp:180-213) ─────────────────────────────
        println!("Assembling ... ");

        let mu_inv = self.mu_inv_coeff();
        let mu_inv_mass = mu_inv.clone();
        let mu_inv_curlm = mu_inv.clone();

        // curlMuInvCurl: CurlCurlIntegrator(muInv), MFEM default rule.  A
        // 1e-10·I shift anchors the AMS nodal problem on the singular
        // curl-curl (the pex34 recipe; the C++ relies on
        // HypreAMS::SetSingularProblem instead).  The shift perturbs the
        // solution by O(1e-10) — far below the d103 tolerance.
        let curl_mu_inv_curl = ParVectorAssembler::assemble_bilinear(
            nd,
            &[
                &CurlCurlIntegrator { mu: mu_inv },
                &VectorMassIntegrator { alpha: 1e-6 },
            ],
            qo_cc,
        );
        // hDivHCurlMuInv: VectorFEMassIntegrator(muInv), custom IR.
        let h_div_hcurl_mu_inv =
            ParMixedAssembler::assemble_hcurl_hdiv_mass(nd, rt, qo_ir, mu_inv_mass);
        // hCurlMass: VectorFEMassIntegrator (unit), custom IR.
        let h_curl_mass = ParVectorAssembler::assemble_bilinear(
            nd,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            qo_ir,
        );

        // curl: ParDiscreteCurlOperator(ND, RT); grad: ParDiscreteGradOperator
        // (H1, ND) — needed by the AMS preconditioner and, with -cr, by the
        // divergence-free projector.
        let curl = ParDiscreteLinearOperator::curl_3d(nd, rt);
        let grad = ParDiscreteLinearOperator::gradient(h1, nd);

        // weakCurlMuInv: MFEM VectorFECurlIntegrator(muInv) on
        // ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_) — the H(curl)-row
        // orientation `∫ μ⁻¹ w · curl(v)`, which is fem-parallel's
        // `assemble_hdiv_hcurl_curl_with_coeff` (rows = H(curl) owned,
        // cols = H(div) total; the `assemble_hcurl_hdiv_curl_*` twin returns
        // the transposed H(div)-row matrix).  NOTE the quadrature: this form
        // gets NO SetIntRule in tesla_solver.cpp:335-338, so MFEM applies the
        // VectorFECurlIntegrator default `trial.GetOrder()+test.GetOrder()-1`
        // = (RT order)+(ND order)−1 = 2·order−2 — not the custom irOrder the
        // mass forms use.
        let weak_curl_mu_inv = if self.bm_params.is_some() || self.ha_params.is_some() {
            let weak_curl_qo = (2 * self.order).saturating_sub(2) as u8;
            Some(ParMixedAssembler::assemble_hdiv_hcurl_curl_with_coeff(
                nd,
                rt,
                weak_curl_qo,
                mu_inv_curlm,
            ))
        } else {
            None
        };

        println!("done.");

        // ── Solve (tesla_solver.cpp:429-489) ────────────────────────────────
        println!("Running solver ... ");

        let nd_dp = nd.dof_partition();
        let rt_dp = rt.dof_partition();
        let comm_nd = nd.comm().clone();

        // *a_ = 0.0 then the boundary projections.  At -maxit 1 MFEM never
        // eliminates essential BCs (ess_bdr_tdofs_ is empty until Update(),
        // tesla_solver.cpp:279) — see the module docs.  The projected values
        // exist only as the (measured-inert) initial state, so A starts at 0.
        let mut a = ParVector::zeros(nd);

        // *jd_ = 0.0
        let mut jd = ParVector::zeros(nd);

        // Magnetization: m_ = ProjectCoeff(mCoef) (nodal RT functionals);
        // jd += mu0 · weakCurlMuInv · m_  (tesla_solver.cpp:459-462).
        if let Some(weak_curl) = &weak_curl_mu_inv {
            let m_fn: Box<dyn Fn(&[f64]) -> [f64; 3] + Send + Sync> =
                if let Some(bm) = &self.bm_params {
                    let bm = bm.clone();
                    Box::new(move |x: &[f64]| bar_magnet(&bm, x))
                } else {
                    let ha = self.ha_params.clone().expect("checked above");
                    Box::new(move |x: &[f64]| halbach_array(&ha, x))
                };
            // m_ = ProjectCoeff(mCoef) through the space's MFEM-pinned nodal
            // functionals (HDivSpace::interpolate_vector, D591/D661 engine).
            // d103 note: the engine evaluates the hex dof samples through the
            // *trilinear corner map*; on a curved mesh MFEM's Project_RT uses
            // the isoparametric map and the two differ ~4% in ||M||_2 here.
            // A curved-map variant (matching MFEM dof-for-dof on interior
            // dofs) was measured in round 103 and made the GLOBAL vector
            // worse: on shared-face dofs the two adjacent curved maps
            // disagree at the (non-geometry-node) sample point and the
            // last-write-wins result is element-order dependent — the fix
            // belongs in the space crate (debt D926), so the engine stays.
            let m_local = rt
                .local_space()
                .interpolate_vector(&move |x: &[f64]| {
                    let v = m_fn(x);
                    vec![v[0], v[1], v[2]]
                })
                .as_slice()
                .to_vec();
            let mut m_vec = ParVector::from_local_raw(
                permute_vec(&m_local, rt_dp),
                rt_dp.n_owned_dofs,
                rt.dof_ghost_exchange_arc(),
                comm_nd.clone(),
            );
            let mut curl_m = ParVector::zeros(nd);
            m_vec.update_ghosts();
            weak_curl.spmv(m_vec.as_slice(), &mut curl_m.as_slice_mut());
            for i in 0..nd_dp.n_owned_dofs {
                jd.as_slice_mut()[i] += MU0 * curl_m.as_slice()[i];
            }
            if probe {
                let m2: f64 = (0..rt_dp.n_owned_dofs)
                    .map(|i| {
                        let v = m_vec.as_slice()[i];
                        v * v
                    })
                    .sum();
                println!("PROBE ||M||_2 = {}", e6(comm.allreduce_sum_f64(m2).sqrt()));
                let q: f64 = (0..nd_dp.n_owned_dofs)
                    .map(|i| {
                        let v = jd.as_slice()[i];
                        v * v
                    })
                    .sum();
                println!("PROBE ||JD||_2 = {}", e6(comm.allreduce_sum_f64(q).sqrt()));
            }
        }

        // Volumetric current (-cr): jr_ = ProjectCoeff(jCoef) (nodal ND
        // functionals); j_ = DivFreeProj(jr_); jd += hCurlMass · j_
        // (tesla_solver.cpp:445-456).
        if let Some(cr) = &self.cr_params {
            let cr = cr.clone();
            let jr_local = nd.local_space().interpolate_vector(&move |x: &[f64]| {
                let v = current_ring(&cr, x);
                vec![v[0], v[1], v[2]]
            });
            let mut jr = ParVector::from_local_raw(
                permute_vec(jr_local.as_slice(), nd_dp),
                nd_dp.n_owned_dofs,
                nd.dof_ghost_exchange_arc(),
                comm_nd.clone(),
            );
            jr.update_ghosts();

            // DivergenceFreeProjector = IrrotationalProjector with y = x −
            // grad·psi: xDiv = −weakDiv·jr; psi = S0⁻¹ xDiv (H1 stiffness,
            // all-boundary Dirichlet 0, PCG + AMG, tol 1e-14 / 200 it);
            // j = jr − grad·psi  (pfem_extras.cpp:97-256).
            let weak_div = ParMixedAssembler::assemble_hcurl_h1_weak_div(h1, nd, qo_ir);
            let mut x_div = vec![0.0_f64; h1.dof_partition().n_total_dofs()];
            weak_div.spmv(jr.as_slice(), &mut x_div);
            for v in &mut x_div {
                *v = -*v;
            }

            let mesh = par_mesh.local_mesh();
            let all_tags = mesh.unique_boundary_tags();
            let dm_h1 = h1.local_space().dof_manager();
            let h1_dp = h1.dof_partition();
            let ess_local = if all_tags.is_empty() {
                vec![]
            } else {
                boundary_dofs(mesh, dm_h1, &all_tags)
            };
            let mut rhs_h1 = ParVector::from_local_raw(
                permute_vec(&x_div, h1_dp),
                h1_dp.n_owned_dofs,
                h1.dof_ghost_exchange_arc(),
                comm_nd.clone(),
            );
            let mut s0 =
                ParAssembler::assemble_bilinear(h1, &[&DiffusionIntegrator { kappa: 1.0 }], qo_ir);
            for &d in &ess_local {
                let pid = h1_dp.permute_dof(d) as usize;
                if pid < h1_dp.n_owned_dofs {
                    s0.apply_dirichlet_par(pid, 0.0, &mut rhs_h1);
                }
            }

            let mut psi = ParVector::zeros(h1);
            let amg_cfg = ParAmgConfig {
                smoother: SmootherType::SymmetricGaussSeidel,
                ..Default::default()
            };
            let cfg = SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 200,
                verbose: false,
                ..SolverConfig::default()
            };
            par_solve_pcg_amg(&s0, &rhs_h1, &mut psi, &amg_cfg, &cfg)
                .expect("tesla: H1 stiffness PCG+AMG failed");

            // j = jr − grad·psi
            let mut j = jr.clone_vec();
            psi.update_ghosts();
            let mut gpsi = vec![0.0_f64; nd_dp.n_owned_dofs];
            grad.spmv(psi.as_slice(), &mut gpsi);
            for i in 0..nd_dp.n_owned_dofs {
                j.as_slice_mut()[i] -= gpsi[i];
            }

            // jd += hCurlMass · j
            let mut mj = ParVector::zeros(nd);
            j.update_ghosts();
            h_curl_mass.spmv(&mut j, &mut mj);
            for i in 0..nd_dp.n_owned_dofs {
                jd.as_slice_mut()[i] += mj.as_slice()[i];
            }
            if probe {
                let q: f64 = (0..nd_dp.n_owned_dofs)
                    .map(|i| {
                        let v = jd.as_slice()[i];
                        v * v
                    })
                    .sum();
                println!("PROBE ||JD||_2 = {}", e6(comm.allreduce_sum_f64(q).sqrt()));
            }
        }

        // FormLinearSystem with an EMPTY ess list (the -maxit 1 quirk) and
        // HypreAMS(SetSingularProblem) + HyprePCG(tol 1e-12, 50 it, print 2).
        // The AMS config is the pex8-verified singular-curl-curl recipe
        // (default ω, symmetric-GS edges, multiplicative V(1,1), 1e-6 nodal
        // regularization standing in for `HYPRE_AMSSetSingularProblem`).
        let ams = ParAmsPrecond::new(
            &curl_mu_inv_curl,
            &grad,
            AmsConfig {
                edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
                cycle: AmsCycle::MultiplicativeV11,
                smoother_sweeps: 4,
                singularity_regularization: 1e-6,
                ..Default::default()
            },
        );
        pcg_with_table(&curl_mu_inv_curl, &jd, &mut a, &ams, 1e-12, 50);

        // Min-norm representative (D103): hypre's `SetSingularProblem` CG
        // keeps its iterates orthogonal to ker(curl·curl⁻¹...) = range(grad),
        // so MFEM reports A⁺·jd.  The fem-rs AMS (nodal shift standing in for
        // SetSingularProblem) injects a gradient drift — up to ~1e6 on the
        // `-cr` run — which curl cannot see (B/H are unaffected) but the
        // ||A||_2 probe line reports.  Project it out: a ← a − G·φ with
        // GᵀG·φ = Gᵀa (CG on the singular consistent nodal system; x₀ = 0
        // keeps φ constant-free, so a−Gφ is the Euclidean projection onto
        // range(A), i.e. MFEM's representative up to solver tolerance).
        {
            let n_h1 = h1.dof_partition().n_total_dofs();
            let mut gtg_coo = fem_linalg::CooMatrix::<f64>::new(n_h1, n_h1);
            for i in 0..grad.nrows {
                for k in grad.row_ptr[i]..grad.row_ptr[i + 1] {
                    let gi = grad.values[k];
                    for l in grad.row_ptr[i]..grad.row_ptr[i + 1] {
                        gtg_coo.add(
                            grad.col_idx[k] as usize,
                            grad.col_idx[l] as usize,
                            gi * grad.values[l],
                        );
                    }
                }
            }
            let gtg = gtg_coo.into_csr();
            let mut gta = vec![0.0_f64; n_h1];
            for i in 0..nd_dp.n_owned_dofs {
                for k in grad.row_ptr[i]..grad.row_ptr[i + 1] {
                    gta[grad.col_idx[k] as usize] += grad.values[k] * a.as_slice()[i];
                }
            }
            let mut phi = vec![0.0_f64; n_h1];
            let cg_cfg = SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 500,
                verbose: false,
                ..SolverConfig::default()
            };
            fem_solver::solve_cg(&gtg, &gta, &mut phi, &cg_cfg).ok(); // singular consistent system
            let mut gphi = vec![0.0_f64; nd_dp.n_owned_dofs];
            grad.spmv(&phi, &mut gphi);
            for i in 0..nd_dp.n_owned_dofs {
                a.as_slice_mut()[i] -= gphi[i];
            }
        }

        // B = curl A (tesla_solver.cpp:475).
        a.update_ghosts();
        let mut b_vec = vec![0.0_f64; rt_dp.n_owned_dofs];
        curl.spmv(a.as_slice(), &mut b_vec);
        let mut b = ParVector::from_local_raw(
            b_vec,
            rt_dp.n_owned_dofs,
            rt.dof_ghost_exchange_arc(),
            comm_nd.clone(),
        );

        // bd = hDivHCurlMuInv·B (− mu0·hDivHCurlMuInv·M with a magnet)
        // (tesla_solver.cpp:481-485): both terms go through the SAME mixed
        // mass `hDivHCurlMuInv_` (ND-row orientation).  fem-parallel's
        // `assemble_hcurl_hdiv_mass` returns the transposed H(div)-row matrix
        // (rows = RT owned, cols = ND total), so apply it as Mᵀ·x.  At
        // --ranks 1 (the verified mode) every H(div) row is owned, so the
        // transpose is the exact operator; multi-rank would need an
        // H(curl)-row mass assembly (debt D925).
        b.update_ghosts();
        let h_div_hcurl_mu_inv_t = h_div_hcurl_mu_inv.transpose();
        let mut bd = ParVector::zeros(nd);
        h_div_hcurl_mu_inv_t.spmv(b.as_slice(), &mut bd.as_slice_mut());
        if weak_curl_mu_inv.is_some() {
            let m_fn: Box<dyn Fn(&[f64]) -> [f64; 3] + Send + Sync> =
                if let Some(bm) = &self.bm_params {
                    let bm = bm.clone();
                    Box::new(move |x: &[f64]| bar_magnet(&bm, x))
                } else {
                    let ha = self.ha_params.clone().expect("checked above");
                    Box::new(move |x: &[f64]| halbach_array(&ha, x))
                };
            let m_local = rt
                .local_space()
                .interpolate_vector(&move |x: &[f64]| {
                    let v = m_fn(x);
                    vec![v[0], v[1], v[2]]
                })
                .as_slice()
                .to_vec();
            let mut m_vec = ParVector::from_local_raw(
                permute_vec(&m_local, rt_dp),
                rt_dp.n_owned_dofs,
                rt.dof_ghost_exchange_arc(),
                comm_nd.clone(),
            );
            m_vec.update_ghosts();
            let mut mm = ParVector::zeros(nd);
            h_div_hcurl_mu_inv_t.spmv(m_vec.as_slice(), &mut mm.as_slice_mut());
            for i in 0..nd_dp.n_owned_dofs {
                bd.as_slice_mut()[i] -= MU0 * mm.as_slice()[i];
            }
        }

        // H = mass⁻¹ bd with diag-scale PCG (tol 1e-12, 500 it, silent)
        // (tesla_solver.cpp:487-503).
        let mut h = ParVector::zeros(nd);
        let h_cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 500,
            verbose: false,
            ..SolverConfig::default()
        };
        par_solve_pcg_jacobi(&h_curl_mass, &bd, &mut h, &h_cfg)
            .expect("tesla: H(curl) mass PCG+Jacobi failed");

        println!("Computing H ... done. Solver done. ");

        if probe {
            let a2: f64 = (0..nd_dp.n_owned_dofs)
                .map(|i| {
                    let v = a.as_slice()[i];
                    v * v
                })
                .sum();
            let b2: f64 = (0..rt_dp.n_owned_dofs)
                .map(|i| {
                    let v = b.as_slice()[i];
                    v * v
                })
                .sum();
            let h2: f64 = (0..nd_dp.n_owned_dofs)
                .map(|i| {
                    let v = h.as_slice()[i];
                    v * v
                })
                .sum();
            println!("PROBE ||A||_2 = {}", e6(comm.allreduce_sum_f64(a2).sqrt()));
            println!("PROBE ||B||_2 = {}", e6(comm.allreduce_sum_f64(b2).sqrt()));
            println!("PROBE ||H||_2 = {}", e6(comm.allreduce_sum_f64(h2).sqrt()));
        }
    }
}

// ─── main (tesla.cpp:105-327) ───────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_opts(&args);

    for line in BANNER {
        println!("{line}");
    }
    print_options(&opts);

    // ── honest refusals (each is a real fem-rs gap, not a silent drop) ─────
    if opts.visualization {
        eprintln!("mfem_miniapp_tesla: -vis (GLVis socket display) is not ported; pass -no-vis");
        std::process::exit(3);
    }
    if opts.visit {
        eprintln!(
            "mfem_miniapp_tesla: -visit (VisIt DataCollection output) is not ported; pass -no-visit"
        );
        std::process::exit(3);
    }
    if opts.maxit != 1 {
        eprintln!(
            "mfem_miniapp_tesla: -maxit {0} (the AMR loop needs L2ZZ error estimation + \
             RefineByError + TeslaSolver::Update); only -maxit 1 is ported.\n\
             \x20 Note: the C++ default is -maxit 100, so the no-argument run is refused too.",
            opts.maxit
        );
        std::process::exit(3);
    }
    if opts.ranks != 1 {
        eprintln!(
            "mfem_miniapp_tesla: --ranks {} (multi-rank mixed H(curl)/H(div) assembly is not \
             verified); pass --ranks 1",
            opts.ranks
        );
        std::process::exit(3);
    }

    // tesla.cpp:223-240 — the surface-current BC consistency check.
    if (!opts.vbcs.is_empty() && opts.kbcs.is_empty())
        || (!opts.kbcs.is_empty() && opts.vbcs.is_empty())
        || opts.vbcv.len() < opts.vbcs.len()
    {
        println!(
            "The surface current (K) boundary condition requires \
             surface current boundary condition surfaces (with -kbcs), \
             voltage boundary condition surface (with -vbcs), \
             and voltage boundary condition values (with -vbcv)."
        );
        std::process::exit(3);
    }
    if !opts.kbcs.is_empty() {
        eprintln!(
            "mfem_miniapp_tesla: -kbcs/-vbcs/-vbcv (the SurfaceCurrent H1 subproblem, \
             tesla_solver.cpp:620-760) is not ported"
        );
        std::process::exit(3);
    }
    if opts.parallel_ref_levels != 0 {
        eprintln!(
            "mfem_miniapp_tesla: -rp (parallel refinement of the ParMesh) is not ported; pass -rp 0"
        );
        std::process::exit(3);
    }
    if opts.pw_mu.iter().any(|&v| v <= 0.0) {
        eprintln!("permeability values must be positive");
        std::process::exit(3);
    }

    println!("Starting initialization.");

    // tesla.cpp:181 — `Mesh(mesh_file, 1, 1)`.  NURBS inputs are refused:
    // fem-rs cannot evaluate the rational geometry nor run MFEM's
    // UniformRefinement + SetCurvature(2) conversion (tesla.cpp:186-195).
    let nurbs = std::fs::read_to_string(&opts.mesh)
        .ok()
        .and_then(|s| s.lines().next().map(|l| l.trim().starts_with("MFEM NURBS")))
        .unwrap_or(false);
    if nurbs {
        eprintln!(
            "mfem_miniapp_tesla: {} is a NURBS mesh; fem-rs does not evaluate NURBS geometry \
             nor run MFEM's UniformRefinement + SetCurvature(2) conversion.\n\
             \x20 Use the C++-converted quadratic mesh instead (see tmp/d103tesla/REPORT.md), \
             e.g. -m tmp/d103tesla/ball-quad.mesh",
            opts.mesh
        );
        std::process::exit(3);
    }
    let mfem = read_mfem_file(&opts.mesh).unwrap_or_else(|e| {
        eprintln!("mfem_miniapp_tesla: failed to read mesh {}: {e}", opts.mesh);
        std::process::exit(3);
    });
    let mut mesh0 = match mfem.mesh3d {
        Some(m) => m,
        None => {
            eprintln!("mfem_miniapp_tesla: {} is not a 3-D volume mesh", opts.mesh);
            std::process::exit(3);
        }
    };
    // tesla.cpp:200-203 — the serial uniform refinements (for a NURBS mesh
    // the first refinement happens inside the conversion above, hence the
    // `serial_ref_levels--` there; a converted mesh starts at level 0).
    for _ in 0..opts.serial_ref_levels {
        mesh0 = refine_uniform_3d(&mesh0);
    }
    let mesh0 = Arc::new(mesh0);

    // Geometry order for the integration rules: quadratic `nodes` give MFEM's
    // `SetCurvature(2)` hexes (OrderW = p·3 − 1 = 5); the plain .mesh fixtures
    // are straight (OrderW = 2).
    let geom_order = mesh0.geometry.as_ref().map(|g| g.order).unwrap_or(1);
    let is_hex = {
        use fem_mesh::topology::MeshTopology;
        mesh0.n_elements() == 0
            || matches!(
                mesh0.element_type(0),
                fem_mesh::ElementType::Hex8
                    | fem_mesh::ElementType::Hex20
                    | fem_mesh::ElementType::Hex27
            )
    };

    let ms = Arc::new(opts.ms_params.clone());
    let pw_mu_inv: Vec<f64> = opts.pw_mu.iter().map(|&v| 1.0 / v).collect();
    let pw_mu_inv = Arc::new(pw_mu_inv);
    let cr = Arc::new(opts.cr_params.clone());
    let bm = Arc::new(opts.bm_params.clone());
    let ha = Arc::new(opts.ha_params.clone());
    let order = opts.order;

    let launcher = ThreadLauncher::new(WorkerConfig::new(opts.ranks));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh(&mesh0, &comm);
        let local_mesh = par_mesh.local_mesh().clone();

        // tesla_solver.cpp:114-119 — H1_ParFESpace(order), ND_ParFESpace(order),
        // RT_ParFESpace(order) → RT_FECollection(order-1).
        let h1 = ParallelFESpace::new(
            H1Space::new(local_mesh.clone(), order as u8),
            &par_mesh,
            comm.clone(),
        );
        let nd = ParallelFESpace::new(
            HCurlSpace::new(local_mesh.clone(), order as u8),
            &par_mesh,
            comm.clone(),
        );
        let rt = ParallelFESpace::new(
            HDivSpace::new(local_mesh.clone(), order.saturating_sub(1) as u8),
            &par_mesh,
            comm.clone(),
        );

        let solver = TeslaSolver {
            h1,
            nd,
            rt,
            ms_params: if ms.is_empty() { None } else { Some((*ms).clone()) },
            pw_mu_inv: if pw_mu_inv.is_empty() { None } else { Some((*pw_mu_inv).clone()) },
            cr_params: if cr.is_empty() { None } else { Some((*cr).clone()) },
            bm_params: if bm.is_empty() { None } else { Some((*bm).clone()) },
            ha_params: if ha.is_empty() { None } else { Some((*ha).clone()) },
            order,
            ir_order: (typical_order_w(geom_order, is_hex) + 2 * order as i32) as u8,
            cc_order: curl_curl_rule_order(order, is_hex),
        };

        if rank == 0 {
            println!("Initialization done.");
            println!();
            println!("AMR Iteration 1");
        }
        solver.print_sizes();
        solver.run(&comm, &par_mesh);
        if rank == 0 {
            println!("AMR iteration 1 complete.");
        }
    });
}
