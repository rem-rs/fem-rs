//! Parallel ultraweak DPG solver for the acoustics (Helmholtz) problem — MFEM
//! `miniapps/dpg/pacoustics.cpp` (MFEM 4.10).
//!
//! **STATUS: NOT PORTED — declarative `exit(3)` with the gap list below.**
//! The C++ miniapp parses the same command line and this port prints the
//! identical option banner, then refuses to run rather than emit a wrong
//! number.
//!
//! # What the C++ miniapp solves
//!
//! ```text
//!     -Δ p - ω² p = f̃   in Ω,      p = p₀   on ∂Ω
//! ```
//!
//! through the first-order system `∇p + iωαu = 0`, `∇·u + iωβp = f`
//! (`α = JᵀJ/|J|`, `β = |J|`) with the traces `p̂ ∈ H^{1/2}`, `û ∈ H^{-1/2}`:
//!
//! ```text
//! |   |     p       |     u        |    p̂     |   û     |  RHS   |
//! | v | -(p, ∇·v)   | iω(αu, v)    | <p̂, v·n> |         |        |
//! | q | -(u, ∇q)    | iω(βp, q)    |          | <û, q>  |  (f,q) |
//! ```
//!
//! with the adjoint-graph test norm, complex valued
//! (`ComplexDPGWeakForm` / `ParComplexDPGWeakForm`).
//!
//! # Gap list (what fem-rs is missing)
//!
//! 1. **No parallel complex DPG.** MFEM uses `ParComplexDPGWeakForm`
//!    (`util/pcomplexweakform.hpp`). fem-rs has the serial
//!    `ComplexDPGWeakForm` (`crates/assembly/src/complex_dpg_weakform.rs`,
//!    exercised by `miniapps/dpg/dpg_acoustics_2d.rs` /
//!    `dpg_acoustics_3d.rs`) and the real-valued `ParDpgWeakForm`
//!    ([`fem_parallel::par_dpg_weakform`]), but **not** the combination: the
//!    complex parallel trace numbering, the real-doubled `Pᵀ A P` and the
//!    complex essential-DOF elimination are missing.
//! 2. **3-D H1-trace parallel numbering.** `ParDpgWeakForm` numbers 2-D
//!    `H1_Trace_FECollection` vertex DOFs (the shared skeleton vertices) and
//!    face-discontinuous trace DOFs; the 3-D H1 trace additionally has DOFs on
//!    the skeleton *edges* (shared by every face meeting at the edge) and on
//!    the face interiors, which is not implemented (the code panics with a
//!    clear message instead of guessing).
//! 3. `-pmg` and `-pref > 0` — same gaps as `pdiffusion`.
//! 4. The `meshes/scatter.mesh` / plane-wave problem cases need the
//!    `MFEM_USE_GSLIB` point-source machinery of the C++ miniapp, which is not
//!    ported.
//!
//! # Available today
//!
//! * `pdiffusion` — verified 1:1 against the C++ MPI reference for `-np 1` and
//!   `-np 2` (dofs, L2 error, DPG residual).
//! * Serial complex DPG: `dpg_acoustics_2d`, `dpg_acoustics_3d`.
//!
//! Usage:
//!   cargo run --release --example pacoustics -- --ranks 2
//! (exits 3 with the gap list above)

use std::process::exit;

const GAPS: &[&str] = &[
    "no parallel COMPLEX DPG: MFEM uses ParComplexDPGWeakForm \
     (miniapps/dpg/util/pcomplexweakform.{hpp,cpp}); fem-rs has the serial \
     ComplexDPGWeakForm and the real ParDpgWeakForm, but no complex parallel \
     trace numbering / real-doubled P^T A P / complex essential-DOF elimination",
    "3-D H1-trace parallel numbering is missing: the 3-D H1 trace has DOFs on the \
     skeleton EDGES (shared by every incident face) as well as on face interiors, \
     while ParDpgWeakForm numbers only 2-D H1-trace (shared vertices) and \
     face-discontinuous traces (crates/parallel/src/par_dpg_weakform.rs panics for \
     that combination today)",
    "`-pmg` (PRefinementMultigrid) is not ported (no p-prolongation for DPG blocks)",
    "`-pref > 0` (parallel AMR + ParDPGWeakForm::Update) is not ported: \
     fem_parallel::par_refine_marked* rebuilds the partition with compact node ids, \
     which breaks the identity node numbering the DPG trace numbering depends on",
    "the scatter.mesh / plane-wave point-source problem cases need the GSLIB \
     point-source machinery of the C++ miniapp (not ported)",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let prob: i32 = get("-prob")
        .or_else(|| get("--problem"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mesh = get("-m")
        .or_else(|| get("--mesh"))
        .unwrap_or_else(|| "../../data/inline-quad.mesh".to_string());
    let order = get("-o").or_else(|| get("--order")).unwrap_or_else(|| "1".into());
    let delta_order = get("-do")
        .or_else(|| get("--delta-order"))
        .unwrap_or_else(|| "1".into());
    let rnum = get("-rnum")
        .or_else(|| get("--number-of-wavelengths"))
        .unwrap_or_else(|| "1".into());
    let sref = get("-sref")
        .or_else(|| get("--serial-ref"))
        .unwrap_or_else(|| "0".into());
    let pref = get("-pref")
        .or_else(|| get("--parallel-ref"))
        .unwrap_or_else(|| "0".into());
    let ranks = get("--ranks").unwrap_or_else(|| "2".into());

    println!("=== fem-rs pacoustics: parallel ultraweak DPG for the Helmholtz problem ===");
    println!("Options used:");
    println!("   --mesh {mesh}");
    println!("   --order {order}");
    println!("   --delta-order {delta_order}");
    println!("   --number-of-wavelengths {rnum}");
    println!("   --problem {prob}");
    println!("   --serial-ref {sref}");
    println!("   --parallel-ref {pref}");
    println!("   --ranks {ranks}");
    println!("   --no-visualization");
    println!();
    eprintln!(
        "pacoustics: GAP — this miniapp is a complex-valued parallel DPG problem and \
         fem-rs cannot assemble it yet:"
    );
    for (i, g) in GAPS.iter().enumerate() {
        eprintln!("  {}. {g}", i + 1);
    }
    eprintln!(
        "Refusing to print a convergence table: a wrong L2 error / iteration count must \
         never be reported as success.  Run `pdiffusion` for the verified parallel DPG \
         path, or `dpg_acoustics_2d` / `dpg_acoustics_3d` for the serial complex DPG."
    );
    exit(3);
}
