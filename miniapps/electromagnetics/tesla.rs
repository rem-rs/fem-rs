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
//! The curl-curl PCG (tesla_solver.cpp:355-365 → `pcg_with_table` below) is
//! the hypre `hypre_PCGSolve` port including its convergence semantics: MFEM
//! sets no `SetUseTwoNorm`, so hypre runs `two_norm = 0` — the iteration
//! table's `||r||_C` column is the preconditioned norm `sqrt((r_k, C·r_k))`
//! and the stopping test is `gamma = <C*r,r>/<C*b,b> < tol²` (hypre
//! `krylov/pcg.c:278`) — NOT the plain `||r||₂/||b||₂` test.  `PCG
//! Iterations = N` / `Final PCG Relative Residual Norm = X` are 1:1.
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
use fem_parallel::par_mixed_assembler::permute_rect_csr;
use fem_parallel::par_solver::par_solve_pcg_jacobi;
use fem_parallel::{
    Comm, ParAssembler, ParCsrMatrix, ParDiscreteLinearOperator, ParMixedAssembler, ParVector,
    ParVectorAssembler, ParallelFESpace, ParallelMesh, SmootherType, WorkerConfig, partition_mesh,
};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};
use linlvo::amg::{AmgConfig, SmootherType as AuxSmoother};
use linlvo::precond::{AmsConfig, AmsCycle, AmsEdgeSmoother, AmsPrecond, AuxSpaceSolver};
use linlvo::{DenseVec, Preconditioner};

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
    if x == 0.0 {
        // C `%e` of zero (hypre's `<C*b,b>` line, the C++ probe's PROBE lines).
        return "0.000000e+00".to_string();
    }
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

// ─── AMS face (curl) auxiliary space — the D957 mechanism ───────────────────

/// The three `id_ND` interpolation blocks `Pi_x/Pi_y/Pi_z` MFEM's `HypreAMS`
/// hands to hypre (`HypreAMS::MakeGradientAndInterpolation`,
/// `linalg/hypre.cpp`): the identity interpolator
/// `id_ND : [H¹]^sdim → H(curl)` (`IdentityInterpolator`,
/// `fem/bilininteg.hpp:4064`) with per-component blocks
/// `ran_fe.Project(dom_fe, Trans, elmat)` — the MFEM `Project_ND` contract
/// (`fe_base.cpp:1404`): row `k` is the point functional
///
/// ```text
///   π_d(k, j) = φ_j(x_k) · (J(x_k)·t_k)_d
/// ```
///
/// evaluated at the ND dof slot (`FE::Nodes` reference point `x_k` with the
/// `dof2tk` tangent `t_k`).  This is the mechanism missing from the fem-rs
/// AMS in round 104 (debt D957): hypre runs its block-Pi multiplicative cycle
/// (`cycle_type = 13`) with `B_Pi_d = AMG(Pi_dᵀ·A·Pi_d)` on exactly these
/// blocks, which is where the C++ tesla PCG iteration counts come from.
///
/// The slot layout, tangents and orientation signs are the space crate's
/// MFEM-pinned `HexNDk` engine (the same tables
/// `HCurlSpace::interpolate_vector`'s D926 curved branch validates bitwise
/// against MFEM).  Hex meshes only (the d103 acceptance meshes
/// ball-quad/inline-hex are hexes); other cell types exit(3).
fn assemble_pi_blocks(
    h1: &H1Space<fem_mesh::Mesh<3>>,
    nd: &HCurlSpace<fem_mesh::Mesh<3>>,
) -> Result<([fem_linalg::CsrMatrix<f64>; 3], Vec<[f64; 3]>), String> {
    use fem_element::nedelec::HexNDk;
    use fem_mesh::element_type::ElementType;

    let mesh = h1.mesh_topology();
    let hnd = HexNDk::new(nd.order() as usize);
    // `(reference slot x_k, reference tangent t_k)` per local dof slot — the
    // MFEM `FE::Nodes`/`tk` tables (same slot order as `nd.element_dofs`).
    let anchors = hnd.dof_anchors();
    let n_nd = nd.n_dofs();
    let n_h1 = h1.n_dofs();
    let p_h1 = h1.get_order();

    // A point-value ND functional is a **global** dof evaluated ONCE (MFEM
    // scatters per element with the vdofs orientation table — assignment, not
    // accumulation; `HCurlSpace::interpolate_vector` likewise loops dofs, not
    // elements).  Pick one host element per dof: edge dofs are shared by up to
    // 4 hexes, but the slot point, `J·t_k` and the H¹ trace on the shared edge
    // agree across the hosts, so any host yields the same row.
    let mut host_of: Vec<(u32, usize)> = vec![(u32::MAX, usize::MAX); n_nd];
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        if !matches!(et, ElementType::Hex8 | ElementType::Hex20) {
            return Err(format!(
                "tesla: the AMS face-space (id_ND Pi blocks) is implemented for hex meshes \
                 only; element {e} is {et:?} (the d103 acceptance meshes are hexes)"
            ));
        }
        for (m, &gdof) in nd.element_dofs(e).iter().enumerate() {
            host_of[gdof as usize] = (e, m);
        }
    }

    let mut coo = [
        fem_linalg::CooMatrix::<f64>::new(n_nd, n_h1),
        fem_linalg::CooMatrix::<f64>::new(n_nd, n_h1),
        fem_linalg::CooMatrix::<f64>::new(n_nd, n_h1),
    ];
    // Physical slot point of every row dof (the Project_ND evaluation point —
    // the canonical key for cross-implementation dumps).
    let mut row_key: Vec<[f64; 3]> = vec![[f64::NAN; 3]; n_nd];
    for (dof, &(e, m)) in host_of.iter().enumerate() {
        if m == usize::MAX {
            return Err(format!(
                "tesla: ND dof {dof} has no host element (space/mesh mismatch)"
            ));
        }
        let et = mesh.element_type(e);
        let h1_ref = fem_space::ref_elem::h1_field_element(et, p_h1, h1.pyramid_basis());
        let h1_dofs = h1.element_dofs_u32(e);
        let signs = nd.element_signs(e);
        let (xi, tk) = &anchors[m];
        let (jac, x) = fem_mesh::element_jacobian_at(mesh, e, xi, 3);
        row_key[dof] = [x[0], x[1], x[2]];
        let j = [
            [jac[(0, 0)], jac[(0, 1)], jac[(0, 2)]],
            [jac[(1, 0)], jac[(1, 1)], jac[(1, 2)]],
            [jac[(2, 0)], jac[(2, 1)], jac[(2, 2)]],
        ];
        let mut shape = vec![0.0_f64; h1_ref.n_dofs()];
        h1_ref.eval_basis(xi, &mut shape);
        // t = J·t_k (MFEM Project_ND: `Trans.Jacobian().InnerProduct(tk, vk)`).
        let mut t = [0.0_f64; 3];
        for (r, t_r) in t.iter_mut().enumerate() {
            *t_r = j[r][0] * tk[0] + j[r][1] * tk[1] + j[r][2] * tk[2];
        }
        let s = signs[m];
        for (&gdof, &phij) in h1_dofs.iter().zip(shape.iter()) {
            let v = s * phij;
            coo[0].add(dof, gdof as usize, v * t[0]);
            coo[1].add(dof, gdof as usize, v * t[1]);
            coo[2].add(dof, gdof as usize, v * t[2]);
        }
    }
    Ok((
        [
            std::mem::replace(&mut coo[0], fem_linalg::CooMatrix::new(0, 0)).into_csr(),
            std::mem::replace(&mut coo[1], fem_linalg::CooMatrix::new(0, 0)).into_csr(),
            std::mem::replace(&mut coo[2], fem_linalg::CooMatrix::new(0, 0)).into_csr(),
        ],
        row_key,
    ))
}

/// Keep the owned rows of a partition-permuted rectangular local matrix (the
/// `ParDiscreteLinearOperator::gradient` recipe's final step, inlined here
/// because the parallel crate ships the permutation but not the extraction).
fn keep_owned_rows(mat: &fem_linalg::CsrMatrix<f64>, n_owned: usize) -> fem_linalg::CsrMatrix<f64> {
    let mut coo = fem_linalg::CooMatrix::<f64>::new(n_owned, mat.ncols);
    for row in 0..n_owned.min(mat.nrows) {
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            coo.add(row, mat.col_idx[k] as usize, mat.values[k]);
        }
    }
    coo.into_csr()
}

/// Row/column permutation of a linlvo CSR matrix: `out[perm[r], c'] = m[r, c]`
/// per permuted axis (`None` = identity).  Pattern-preserving — explicit zeros
/// are carried over, so the sparsity the AMG sees only moves, never shrinks.
fn permute_linlvo_csr(
    m: &linlvo::sparse::CsrMatrix<f64>,
    row_perm: Option<&[usize]>,
    col_perm: Option<&[usize]>,
) -> linlvo::sparse::CsrMatrix<f64> {
    let mut coo = linlvo::sparse::CooMatrix::<f64>::new(m.nrows(), m.ncols());
    for r in 0..m.nrows() {
        let nr = match row_perm {
            Some(p) => p[r],
            None => r,
        };
        for k in m.row_ptr()[r]..m.row_ptr()[r + 1] {
            let c = m.col_idx()[k];
            let nc = match col_perm {
                Some(p) => p[c],
                None => c,
            };
            coo.push(nr, nc, m.values()[k]);
        }
    }
    linlvo::sparse::CsrMatrix::from_coo(&coo)
}

/// The `-cr` weak-divergence action `xDiv = −W·jr` (the `IrrotationalProjector`
/// input, `pfem_extras.cpp:180-181`) runs through the fem-parallel kernel entry
/// `ParMixedAssembler::assemble_hcurl_h1_weak_div` (MFEM
/// `VectorFEWeakDivergenceIntegrator`, bilininteg.cpp:1852-1941, at the custom
/// `irOrder` rule).
///
/// D1041 closed (round 107): that entry used to route through the frozen
/// `fem_assembly` kernel, which assembles the RAW element-local trial shapes —
/// its D58 face-block canonicalization is a no-op on hexes and the H(curl)
/// orientation signs never reached the columns, so `W·jr` on a discretely
/// divergence-free field was O(‖jr‖) instead of 0 (measured d106 on ball-quad
/// `-cr`: ‖W·jr‖ = 7.4e-1 vs the C++ probe's 7.6e-17).  Round 107 added the
/// signed kernel `fem_parallel::par_mixed_assembler::
/// assemble_hcurl_h1_weak_div_signed` (the frozen twin lives in the
/// D1041-forbidden `crates/assembly`) behind the same entry, and the round-106
/// local `assemble_weak_div_matrix` bypass was retired.  Pin:
/// `crates/parallel/tests/d107_d1041_weak_div_signs.rs`.
///
/// Single-rank AMS preconditioner wrapping linlvo's
/// [`AmsPrecond::with_pi`](linlvo::precond::AmsPrecond::with_pi) — the MFEM
/// `HypreAMS(SetSingularProblem)` cycle (block-Pi multiplicative `0345430`).
///
/// tesla needs the hypre AMS *cycle semantics*, which live in linlvo's serial
/// `AmsPrecond`; `fem_parallel::ParAmsPrecond` remains the block-Jacobi
/// wrapper used by the other consumers and does not accept Pi blocks (its
/// `crates/parallel` file is outside this lane's territory).
struct TeslaAms {
    inner: AmsPrecond<f64>,
    n_owned: usize,
    /// D1095 serial parity map: partition dof id → canonical (space) dof id.
    /// Empty on the multi-rank path (identity — hypre's parallel orderings are
    /// its own and no serial-parity target exists there).
    perm: Vec<usize>,
}

impl TeslaAms {
    /// Apply the preconditioner: `z = M⁻¹ r` (rank-local, single rank).
    ///
    /// With `perm` set the wrapper maps the partition-ordered residual into
    /// the canonical order the inner `AmsPrecond` was built in (the C++/hypre
    /// ordering) and maps the correction back; the PCG outer loop stays in
    /// partition order and is permutation-invariant.
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        if self.perm.is_empty() {
            let lr = DenseVec::from_vec(r.to_vec());
            let mut lz = DenseVec::zeros(self.n_owned);
            self.inner.apply_precond(&lr, &mut lz);
            z.copy_from_slice(lz.as_slice());
        } else {
            let mut rc = vec![0.0_f64; self.n_owned];
            for (p, &c) in self.perm.iter().enumerate() {
                rc[c] = r[p];
            }
            let lr = DenseVec::from_vec(rc);
            let mut lz = DenseVec::zeros(self.n_owned);
            self.inner.apply_precond(&lr, &mut lz);
            let lz = lz.as_slice();
            for (p, &c) in self.perm.iter().enumerate() {
                z[p] = lz[c];
            }
        }
    }
}

/// One printed PCG solve — the exact `HyprePCG` (hypre `hypre_PCGSolve`,
/// print level 2) port for the curl-curl solve (tesla_solver.cpp:355-365:
/// tol 1e-12, maxit 50, print 2).  MFEM sets no `SetUseTwoNorm`, so hypre
/// runs `two_norm = 0`: the printed `||r||_C` column is the
/// **preconditioned** residual norm `sqrt((r_k, C·r_k))` and the stopping
/// test is the energy-norm one (pcg.c:278)
///
/// ```text
/// gamma = <C*r, r> / <C*b, b>  <  eps = tol²
/// ```
///
/// — not the plain `||r||₂/||b||₂` test.  Returns `(iterations, final
/// relative residual)` (the two `HyprePCG` getters MFEM prints).
#[allow(clippy::too_many_arguments)]
fn pcg_with_table(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    ams: &TeslaAms,
    rtol: f64,
    max_iter: usize,
) -> (usize, f64) {
    // Pre-loop: p = C·b; bi_prod = <C*b,b> (pcg.c:357-378), printed first.
    let mut cb = ParVector::zeros_like(b);
    ams.apply(b.as_slice(), cb.as_slice_mut());
    let bi_prod = b.global_dot(&cb);
    println!();
    println!("<C*b,b>: {}", e6(bi_prod));

    if bi_prod <= 0.0 {
        // pcg.c:417-431: bi_prod == 0 (zero rhs) → x = 0 and return with 0
        // iterations, relative residual 0.
        for v in x.as_slice_mut().iter_mut() {
            *v = 0.0;
        }
        println!();
        println!("PCG Iterations = 0");
        println!("Final PCG Relative Residual Norm = {}", g6(0.0));
        return (0, 0.0);
    }
    if let Some(dir) = std::env::var("FEMRS_TESLA_CB_DUMP").ok() {
        // D114A probe: dump b and C·b (partition order) so the first AMS
        // action can be replayed offline against the A dumps.
        let mut out = String::with_capacity(2 * b.as_slice().len() * 26);
        for (i, &v) in b.as_slice().iter().enumerate() {
            out.push_str(&format!("b {i} {v:.17e}\n"));
        }
        for (i, &v) in cb.as_slice().iter().enumerate() {
            out.push_str(&format!("cb {i} {v:.17e}\n"));
        }
        let _ = std::fs::write(dir, out);
    }
    let eps = rtol * rtol; // pcg.c:406: eps = r_tol * r_tol

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
    // p = C·r; gamma = <r, C·r>; norms[0] = sqrt(gamma) — the conv.rate
    // denominator of iteration 1 (pcg.c:451-459, 483-493).
    let mut z = ParVector::zeros_like(b);
    ams.apply(r.as_slice(), z.as_slice_mut());
    let mut rz = r.global_dot(&z);
    let mut p = z.clone_vec();

    let mut r_norm = rz.sqrt();
    let mut iterations = 0usize;
    let mut rel = (rz / bi_prod).sqrt();
    while iterations < max_iter {
        iterations += 1;
        let mut ap = ParVector::zeros_like(b);
        let mut pm = p.clone_vec();
        a.spmv(&mut pm, &mut ap);
        let sdotp = p.global_dot(&ap);
        if sdotp == 0.0 {
            // pcg.c:528-534: "Zero sdotp value in PCG" → break, iteration
            // counted.
            break;
        }
        let alpha = rz / sdotp;
        for i in 0..n_owned {
            x.as_slice_mut()[i] += alpha * p.as_slice()[i];
            r.as_slice_mut()[i] -= alpha * ap.as_slice()[i];
        }
        // s = C·r; gamma = <r, s> (pcg.c:589-591).
        ams.apply(r.as_slice(), z.as_slice_mut());
        let rz_new = r.global_dot(&z);

        let prev = r_norm;
        r_norm = rz_new.sqrt();
        rel = (rz_new / bi_prod).sqrt();
        println!(
            "{:5}    {:<13}    {:<9.6}    {}",
            iterations,
            e6(r_norm),
            r_norm / prev,
            e6(rel)
        );

        // The basic convergence test (pcg.c:677): i_prod/bi_prod < eps.
        if rz_new / bi_prod < eps {
            break;
        }
        // Subnormal gamma guard (pcg.c:703-707): no hope of further progress.
        if !(rz_new > f64::MIN_POSITIVE) {
            break;
        }
        let beta = rz_new / rz;
        for i in 0..n_owned {
            p.as_slice_mut()[i] = z.as_slice()[i] + beta * p.as_slice()[i];
        }
        rz = rz_new;
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
        print!("Assembling ... ");

        let mu_inv = self.mu_inv_coeff();
        let mu_inv_mass = mu_inv.clone();
        let mu_inv_curlm = mu_inv.clone();

        // curlMuInvCurl: CurlCurlIntegrator(muInv), MFEM default rule.  The
        // earlier 1e-6·(mass) shift (pex34 recipe) is GONE: the D957 AMS
        // handles the singular curl-curl internally (singular_problem =>
        // hypre `SetSingularProblem` semantics), and PCG keeps its iterates
        // in range(A) for a consistent rhs — the C++ solves the same
        // unshifted singular system.
        let curl_mu_inv_curl = ParVectorAssembler::assemble_bilinear(
            nd,
            &[&CurlCurlIntegrator { mu: mu_inv }],
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
            if probe {
                let q: f64 = (0..nd_dp.n_owned_dofs)
                    .map(|i| {
                        let v = jr.as_slice()[i];
                        v * v
                    })
                    .sum();
                println!("PROBE ||JR||_2 = {}", e6(comm.allreduce_sum_f64(q).sqrt()));
            }

            // DivergenceFreeProjector = IrrotationalProjector with y = x −
            // grad·psi: xDiv = −weakDiv·jr; psi = S0⁻¹ xDiv (H1 stiffness,
            // all-boundary Dirichlet 0, PCG + AMG, tol 1e-14 / 200 it);
            // j = jr − grad·psi  (pfem_extras.cpp:97-256).  D1041 (round 107):
            // the kernel entry carries the H(curl) orientation signs, so the
            // round-106 local signed-W bypass is gone.  The kernel returns
            // owned H¹ rows in partition order and consumes the ND columns in
            // partition order too — jr is permuted (dual/sign transform) into
            // that layout first.
            let w_div = ParMixedAssembler::assemble_hcurl_h1_weak_div(&h1, &nd, qo_ir);
            let jr_par = permute_vec(jr_local.as_slice(), nd_dp);
            let mut x_div = vec![0.0_f64; w_div.nrows];
            w_div.spmv(&jr_par, &mut x_div);
            if probe {
                let q: f64 = x_div.iter().map(|v| v * v).sum();
                println!("PROBE ||XDIV||_2 = {}", e6(comm.allreduce_sum_f64(q).sqrt()));
            }
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
            // x_div is already the kernel's owned H¹ rows in partition order.
            let mut rhs_h1 = ParVector::zeros(h1);
            rhs_h1.as_slice_mut()[..x_div.len()].copy_from_slice(&x_div);
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
            if probe {
                let q: f64 = (0..nd_dp.n_owned_dofs)
                    .map(|i| {
                        let v = j.as_slice()[i];
                        v * v
                    })
                    .sum();
                println!("PROBE ||J||_2 = {}", e6(comm.allreduce_sum_f64(q).sqrt()));
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
        //
        // D957: the AMS is the MFEM `HypreAMS` face-space mechanism — the
        // id_ND interpolation blocks Pi_x/Pi_y/Pi_z (assembled by
        // `assemble_pi_blocks` above, the `HYPRE_AMSSetInterpolations`
        // payload) plus the singular-problem cycle (`0345430`: the nodal
        // gradient arm is dropped, hypre 2.28 ams.c:3688).  The options mirror
        // MFEM `HypreAMS::MakeSolver` defaults: symmetric-GS edge smoothing
        // (`rlx_type 2` — which in hypre is the *l1-scaled* SGS: for relax
        // types 1-4 `hypre_AMSSetup` computes the row l1 norms of A
        // (ams.c:3041-3053) and the hybrid-SOR relax divides by them), one
        // relax sweep per level, multiplicative V(1,1).
        let h1_dp = h1.dof_partition();
        let (pi_canonical, pi_row_key) =
            match assemble_pi_blocks(h1.local_space(), nd.local_space()) {
                Ok(pi) => pi,
                Err(e) => {
                    eprintln!("mfem_miniapp_tesla: {e}");
                    std::process::exit(3);
                }
            };
        let pi_local: Vec<fem_linalg::CsrMatrix<f64>> = pi_canonical
            .iter()
            .map(|m| keep_owned_rows(&permute_rect_csr(m, nd_dp, h1_dp), nd_dp.n_owned_dofs))
            .collect();
        if let Ok(true) = std::env::var("FEMRS_TESLA_PI_DUMP").map(|v| v == "1") {
            // Canonical-key dump (matches tmp/d105ams/pi_probe.cpp): each
            // triplet keyed by the row's ND slot point and the column's H¹
            // dof coordinate, so the two numberings can be matched.
            let mut col_key: Vec<[f64; 3]> = vec![[f64::NAN; 3]; pi_canonical[0].ncols];
            let dm_h1 = h1.local_space().dof_manager();
            for dof in 0..pi_canonical[0].ncols as u32 {
                let c = dm_h1.dof_coord(dof);
                col_key[dof as usize] = [c[0], c[1], c[2]];
            }
            for (name, m) in ["pi_x", "pi_y", "pi_z"].iter().zip(pi_canonical.iter()) {
                let path = format!("tmp/d105ams/rs_{name}.txt");
                let mut out = String::new();
                out.push_str(&format!("{} {} {}\n", m.nrows, m.ncols, m.nnz()));
                for r in 0..m.nrows {
                    for k in m.row_ptr[r]..m.row_ptr[r + 1] {
                        let c = m.col_idx[k] as usize;
                        out.push_str(&format!(
                            "{:.15e} {:.15e} {:.15e} {:.15e} {:.15e} {:.15e} {:.17e}\n",
                            pi_row_key[r][0], pi_row_key[r][1], pi_row_key[r][2],
                            col_key[c][0], col_key[c][1], col_key[c][2], m.values[k]
                        ));
                    }
                }
                let _ = std::fs::write(&path, out);
            }
        }
        let la = fem_linalg::fem_to_linlvo_csr(curl_mu_inv_curl.diag_block());
        let lg = fem_linalg::fem_to_linlvo_csr(&grad);
        let lpi: Vec<_> = pi_local.iter().map(fem_linalg::fem_to_linlvo_csr).collect();
        // D1095 serial parity: C++ hands hypre the canonical (space) dof
        // numbering — a one-rank `ParFiniteElementSpace` is unpermuted — while
        // the fem-parallel solver view renames the same dofs into the
        // partition layout (owned edge dofs sorted by global vertex pair,
        // owned faces by face key; `DofPartition`).  Both layouts carry
        // bitwise-identical values (measured d110a: the Pi-block row scalars
        // match C++ 1460/1460 bitwise), but BoomerAMG's index-ordered steps
        // (HMIS greedy C-point scan, symmetric-GS sweeps, Multipass pass
        // order) make the preconditioner ORDER-dependent — the partition
        // layout converged one PCG step faster than C++ (7/6/4 vs 8/7/5)
        // purely through tie-breaks.  At one rank, permute the linlvo handoff
        // back to the canonical order (`DofPartition::unpermute_dof`; the H¹
        // partition is already identity there) and map the wrapper's vectors
        // in/out, exactly reproducing the C++→hypre handoff.
        let nd_canon: Vec<usize> = if comm.size() == 1 {
            (0..nd_dp.n_total_dofs())
                .map(|p| nd_dp.unpermute_dof(p as u32) as usize)
                .collect()
        } else {
            Vec::new()
        };
        // The H¹ partition permutes the same way (owned edge dofs by vertex
        // pair, faces by face key); measured d110a: our A_Pi edge columns ran
        // in sorted-pair order while C++ runs first-encounter — 3188/3188
        // edge and 2064/2064 face couplings of the vertex rows re-verified
        // under exactly that rule.  Unpermute the G/Pi columns so the AMG's
        // H¹-space index order is C++'s too.
        let h1_canon: Vec<usize> = if comm.size() == 1 {
            (0..h1_dp.n_total_dofs())
                .map(|p| h1_dp.unpermute_dof(p as u32) as usize)
                .collect()
        } else {
            Vec::new()
        };
        let (la, lg, lpi) = if nd_canon.is_empty() {
            (la, lg, lpi)
        } else {
            let rp = Some(&nd_canon[..]);
            let cp = Some(&h1_canon[..]);
            (
                permute_linlvo_csr(&la, rp, rp),
                permute_linlvo_csr(&lg, rp, cp),
                lpi.iter()
                    .map(|m| permute_linlvo_csr(m, rp, cp))
                    .collect(),
            )
        };
        if let Ok(path) = std::env::var("FEMRS_TESLA_A_DUMP") {
            // Canonical-order curl-curl A triplet dump (`row col %.17e`,
            // row-major over the stored CSR pattern) — the numbering and the
            // quantity of the C++ probe's `Aop` dump (`tmp/d110a/cpp_A.txt`,
            // 141448 entries on ball-quad o2).  The D1115 bitwise-alignment
            // instrument: any residual assembly ulp shows up here before it
            // can reach the HMIS strength ties.
            let mut out = String::with_capacity(la.nnz() * 32);
            for r in 0..la.nrows() {
                for k in la.row_ptr()[r]..la.row_ptr()[r + 1] {
                    out.push_str(&format!(
                        "{} {} {:.17e}\n",
                        r,
                        la.col_idx()[k],
                        la.values()[k]
                    ));
                }
            }
            let _ = std::fs::write(&path, out);
        }
        if let Ok(true) = std::env::var("FEMRS_TESLA_PI_DUMP").map(|v| v == "2") {
            // π·v keyed by the row slot point (v = x-coordinate field) — the
            // numbering-free action comparison against tmp/d105ams/pi_probe.
            let vx = h1.local_space().interpolate(&|x: &[f64]| x[0]);
            let sv: f64 = vx.as_slice().iter().sum();
            println!("SUM v = {sv:.17e}");
            let mut sum_piv = 0.0_f64;
            let mut out = String::new();
            for (r, key) in pi_row_key.iter().enumerate() {
                let mut yr = 0.0_f64;
                for k in pi_canonical[0].row_ptr[r]..pi_canonical[0].row_ptr[r + 1] {
                    let c = pi_canonical[0].col_idx[k] as usize;
                    yr += pi_canonical[0].values[k] * vx.as_slice()[c];
                }
                sum_piv += yr;
                out.push_str(&format!(
                    "{:.15e} {:.15e} {:.15e} {:.17e}\n",
                    key[0], key[1], key[2], yr
                ));
            }
            println!("SUM piv = {sum_piv:.17e}");
            let _ = std::fs::write("tmp/d105ams/rs_piv.txt", out);
        }
        if let Ok(true) = std::env::var("FEMRS_TESLA_PI_CHECK").map(|v| v == "1") {
            // Project_ND invariant: π_d·(component dofs of v) must reproduce the
            // space's own `interpolate_vector` point functionals (both are the
            // MFEM `Project_ND` contract — slot layout cross-check).  The field
            // must be one its H¹ interpolation reproduces EXACTLY at the ND
            // slots — the geometry itself (x ↦ x) — any other field carries an
            // O(h²) interpolation error on a curved mesh that would mask the
            // comparison.
            let field = |x: &[f64]| vec![x[0], x[1], x[2]];
            let v_local: Vec<fem_linalg::Vector<f64>> = (0..3)
                .map(|c| {
                    let comp = move |x: &[f64]| field(x)[c];
                    h1.local_space().interpolate(&comp)
                })
                .collect();
            let ref_dofs = nd.local_space().interpolate_vector(&field);
            let mut sum = vec![0.0_f64; pi_canonical[0].nrows];
            for (m, v) in pi_canonical.iter().zip(v_local.iter()) {
                let mut out = vec![0.0_f64; m.nrows];
                m.spmv(v.as_slice(), &mut out);
                for (s, o) in sum.iter_mut().zip(out.iter()) {
                    *s += o;
                }
            }
            let mut max_dev = 0.0_f64;
            let mut worst = 0usize;
            for (i, s) in sum.iter().enumerate() {
                let d = (s - ref_dofs.as_slice()[i]).abs();
                if d > max_dev {
                    max_dev = d;
                    worst = i;
                }
            }
            println!(
                "PI-CHECK max|π·v − interpolate_vector(v)| = {max_dev:.3e} (row {worst}: π·v = {}, ref = {})",
                sum[worst], ref_dofs.as_slice()[worst]
            );
            {
                // Per-host breakdown for the worst row: all (elem, slot) hosts
                // of that dof, their sign/J·t/φ evaluations, vs the ref value.
                let mut n_dev = 0usize;
                for (i, s) in sum.iter().enumerate() {
                    if (s - ref_dofs.as_slice()[i]).abs() > 1e-10 {
                        n_dev += 1;
                    }
                }
                println!("PI-CHECK rows deviating > 1e-10: {n_dev} / {}", sum.len());
            }
        }
        let inner = match AmsPrecond::<f64>::with_pi(
            &la,
            &lg,
            &lpi,
            AmsConfig {
                edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
                cycle: AmsCycle::MultiplicativeV11,
                singular_problem: true,
                // B_Pi/B_G analogues: hypre BoomerAMG(V(1,1), l1-sym-GS
                // smoothers (relax_type 8), relaxation-based coarsest solve
                // (SetCycleRelaxType(·, 3)) — linlvo's `with_pi` enforces the
                // coarsest-solve part.
                node_solver: AuxSpaceSolver::Amg(AmgConfig {
                    smoother: AuxSmoother::L1SymmetricGaussSeidel,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("mfem_miniapp_tesla: AMS setup failed: {e}");
                std::process::exit(3);
            }
        };
        let mut ams = TeslaAms {
            inner,
            n_owned: nd_dp.n_owned_dofs,
            perm: nd_canon.clone(),
        };
        if std::env::var("FEMRS_TESLA_M_CHECK").is_ok() {
            // M symmetry/positivity probe: uᵀ(Mv) vs vᵀ(Mu), uᵀ(Mu).
            let n = nd_dp.n_owned_dofs;
            let mut u = vec![0.0_f64; n];
            let mut v = vec![0.0_f64; n];
            let mut seed = 12345u64;
            let mut rng = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f64) / (u32::MAX as f64) - 0.5
            };
            for x in u.iter_mut() {
                *x = rng();
            }
            for x in v.iter_mut() {
                *x = rng();
            }
            let mut mu = vec![0.0_f64; n];
            let mut mv = vec![0.0_f64; n];
            let mut mv2 = vec![0.0_f64; n];
            ams.apply(&u, &mut mu);
            ams.apply(&v, &mut mv);
            ams.apply(&v, &mut mv2);
            let umv: f64 = u.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
            let vmu: f64 = v.iter().zip(mu.iter()).map(|(a, b)| a * b).sum();
            let umu: f64 = u.iter().zip(mu.iter()).map(|(a, b)| a * b).sum();
            let vmv: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
            let mv_sym: f64 = mv.iter().zip(mv2.iter()).map(|(a, b)| (a - b).abs()).sum();
            println!("M-CHECK uMv = {umv:.6e}  vMu = {vmu:.6e}  uMu = {umu:.6e}  vMv = {vmv:.6e}  det-apply = {mv_sym:.3e}");
        }
        let (pcg_its, pcg_rel) = pcg_with_table(&curl_mu_inv_curl, &jd, &mut a, &ams, 1e-12, 50);
        if pcg_rel >= 1e-12 && pcg_its >= 50 {
            // D972 fallback: retry once with the three-space cycle (nodal arm
            // kept).  The strict `SetSingularProblem` cycle's B_Pi (linlvo
            // AMG, not BoomerAMG-grade) leaves PCG's search directions in
            // ker(A) on smooth right-hand sides.
            let inner = match AmsPrecond::<f64>::with_pi(
                &la,
                &lg,
                &lpi,
                AmsConfig {
                    edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
                    cycle: AmsCycle::MultiplicativeV11,
                    singular_problem: false,
                    node_solver: AuxSpaceSolver::Amg(AmgConfig {
                        smoother: AuxSmoother::L1SymmetricGaussSeidel,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ) {
                Ok(p) => p,
                Err(_) => unreachable!("identical setup succeeded above"),
            };
            ams = TeslaAms {
                inner,
                n_owned: nd_dp.n_owned_dofs,
                perm: nd_canon.clone(),
            };
            println!("AMS retry with the three-space cycle (034515430)");
            pcg_with_table(&curl_mu_inv_curl, &jd, &mut a, &ams, 1e-12, 50);
        }

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
