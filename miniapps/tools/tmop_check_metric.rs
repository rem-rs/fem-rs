//! TMOP metric checker — partial delivery (exit 3) for MFEM
//! `miniapps/tools/tmop-check-metric.cpp` (MFEM 4.10).
//!
//! ## What the C++ program is
//!
//! `tmop-check-metric -mid <id>` checks **one** metric on a real mesh
//! (`MakeCartesian2D(1,1,QUADRILATERAL)` / `MakeCartesian3D(1,1,1,HEXAHEDRON)`,
//! H¹ order 2, `TargetConstructor::IDEAL_SHAPE_UNIT_SIZE`, `TMOP_Integrator`)
//! and prints three lines (measured, `-mid 360`):
//!
//! ```text
//! --- EvalW:     0 errors out of 522 comparisons with det(T) > 0.
//! --- EvalP:     avg rate of convergence (should be 2): 2.00061
//! --- AssembleH: avg rate of convergence (should be 2): 1.97716
//! ```
//!
//! * `EvalW` — 1000 random `T` (`T_vec.Randomize(i)`, MFEM's RNG), compared as
//!   `|EvalW(T) - EvalWMatrixForm(T)| / |EvalWMatrixForm(T)|` against the
//!   geom-to-perf Jacobian of the 1-element mesh.
//! * `EvalP` — finite-difference convergence of `integ->AssembleElementVector`
//!   (the *analytic* element gradient) versus `integ->GetElementEnergy`.
//! * `AssembleH` — finite-difference convergence of `integ->AssembleElementGrad`
//!   (the analytic element Hessian) versus `AssembleElementVector`.
//!
//! ## Gap list (round 32, D129) — why this is not that program
//!
//! 1. `fem_mesh::tmop::TmopIntegrator2D/3D` exposes only
//!    `compute_element_energy` + **finite-difference** `compute_element_gradient_fd` /
//!    `compute_element_hessian_fd`; there is no analytic `AssembleElementVector` /
//!    `AssembleElementGrad` and no `Transformation`/`FiniteElement`-based element
//!    energy, so the `EvalP`/`AssembleH` convergence rates cannot be reproduced.
//! 2. The `-mid` zoo is incomplete: the C++ switch accepts 40 ids
//!    (2-D `1 2 7 9 14 22 50 55 56 58 77 80 85 90 94 98`, 3-D
//!    `301 302 303 304 313 315 316 318 321 322 323 328 332 333 334 338 342 347 360`,
//!    A-metrics `11 36 51 107 126`); `fem_mesh::tmop` implements 21 of them.
//!    Missing: **80, 85, 90, 94, 98, 313, 322, 328, 332, 333, 334, 338, 342, 347**
//!    (T-metrics) and **11, 36, 51, 107, 126** (A-metrics).
//! 3. The previous revision of this file was a *different* program: a fixed
//!    21-metric self-check that silently ignored `-mid`, printed
//!    `EvalW errors: 0/0` for every 2-D metric (the 2-D random `T` generator in
//!    `fem_mesh::tmop::check` keeps `det(T) <= 0` for all 100 samples, so the
//!    comparison loop body never runs) and exited 0.  It is removed.
//!
//! Unknown ids print `Unknown metric_id: <id>` and exit with code 3, exactly
//! like MFEM's `default:` branch.
//!
//! Usage: cargo run --release --example tmop_check_metric -- -mid 360

use std::process::exit;

/// Metric ids accepted by the C++ switch in
/// `miniapps/tools/tmop-check-metric.cpp`.
const CPP_IDS: &[i32] = &[
    // T-metrics, 2-D
    1, 2, 7, 9, 14, 22, 50, 55, 56, 58, 77, 80, 85, 90, 94, 98,
    // T-metrics, 3-D
    301, 302, 303, 304, 313, 315, 316, 318, 321, 322, 323, 328, 332, 333, 334, 338, 342, 347, 360,
    // A-metrics
    11, 36, 51, 107, 126,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut metric_id: i32 = 2;
    let mut a_metric_version = false;
    let mut verbose = false;
    let mut convergence_iter: i32 = 10;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-mid" | "--metric-id" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { metric_id = val; } }
            }
            "-A" | "-Ametric" => a_metric_version = true,
            "-no-A" | "--no-Ametric" => a_metric_version = false,
            "-v" | "-verbose" => verbose = true,
            "-no-v" | "--no-verbose" => verbose = false,
            "-i" | "--iterations" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { convergence_iter = val; } }
            }
            // C++ `args.ParseCheck()`: OptionsParser rejects anything it does not
            // know (`Unrecognized option: <opt>` + usage + exit 1) rather than
            // ignoring it — measured rc=1 on the C++ binary for `-bogus`.
            other => {
                eprintln!("Unrecognized option: {other}");
                std::process::exit(1);
            }
        }
    }

    // C++ `args.PrintOptions(cout)`.
    println!("Options used:");
    println!("   --metric-id {metric_id}");
    println!("   --{}", if a_metric_version { "Ametric" } else { "no-Ametric" });
    println!("   --{}", if verbose { "verbose" } else { "no-verbose" });
    println!("   --iterations {convergence_iter}");

    if !CPP_IDS.contains(&metric_id) {
        // C++ `default: cout << "Unknown metric_id: " << metric_id << endl; return 3;`
        println!("Unknown metric_id: {metric_id}");
        exit(3);
    }

    eprintln!(
        "tmop-check-metric (Rust port): partial delivery, exit 3. The C++ program checks metric \
         {metric_id} through the mesh/FE-space `TMOP_Integrator` with the *analytic* \
         `AssembleElementVector` / `AssembleElementGrad` (EvalP/AssembleH convergence rates) and \
         1000 random `T` from MFEM's RNG. `fem_mesh::tmop` provides only a simplified element \
         energy plus finite-difference gradient/Hessian, and implements 21 of the 40 metric ids \
         the C++ switch accepts (missing: 80, 85, 90, 94, 98, 313, 322, 328, 332, 333, 334, 338, \
         342, 347 and the A-metrics 11, 36, 51, 107, 126). No check is run — the previous \
         self-check was a different program (and reported 0/0 comparisons in 2-D)."
    );
    exit(3);
}
