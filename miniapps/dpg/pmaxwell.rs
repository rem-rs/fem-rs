//! Parallel ultraweak DPG solver for Maxwell's equations — MFEM
//! `miniapps/dpg/pmaxwell.cpp` (MFEM 4.10).
//!
//! **STATUS: NOT PORTED — declarative `exit(3)` with the gap list below.**
//! The C++ miniapp parses the same command line and this port prints the
//! identical option banner, then refuses to run rather than emit a wrong
//! number.
//!
//! # What the C++ miniapp solves
//!
//! ```text
//!     ∇×∇×E - ω²εμ E = -iωμ J    in Ω,
//! ```
//!
//! through the first-order system with the traces
//! `Ê = n×H`, `Ĥ = n×E` (2-D: `Ê ∈ RT-trace`, `Ĥ ∈ H¹-trace`; 3-D:
//! `Ê, Ĥ ∈ ND-trace`), test spaces `F ∈ ND/H¹(q)`, `G ∈ ND(q)`, and — for the
//! three PML problem cases — a `CartesianPML` layer with matrix coefficients.
//!
//! # Gap list (what fem-rs is missing)
//!
//! 1. **No parallel complex DPG.** MFEM uses `ParComplexDPGWeakForm`
//!    (`util/pcomplexweakform.hpp`). fem-rs has the serial
//!    `ComplexDPGWeakForm` (exercised by `miniapps/dpg/dpg_maxwell_2d.rs` /
//!    `dpg_maxwell_3d.rs`) and the real-valued `ParDpgWeakForm`
//!    ([`fem_parallel::par_dpg_weakform`]), but not the combination.
//! 2. **ND-trace parallel numbering.** `ParDpgWeakForm` does not number
//!    `ND_Trace_FECollection` blocks (the 3-D `Ê, Ĥ` and the 2-D `Ê` need
//!    edge-shared H(curl) traces with MFEM's orientation signs); the code
//!    panics with a clear message instead of guessing.
//! 3. **3-D H1-trace parallel numbering** — as in `pacoustics` (edge-shared
//!    DOFs on the skeleton).
//! 4. **The PML**. `CartesianPML`, `PmlCoefficient`, `PmlMatrixCoefficient`
//!    and `RestrictedCoefficient` (`miniapps/dpg/util/pml.{hpp,cpp}`) are not
//!    ported at all, so the three `pml_*` problem cases (`-prob 2`, `3`, `4`)
//!    have no fem-rs counterpart. These coefficients are also spatially varying
//!    matrix coefficients, which the `Dpg*Integrator` family only supports for
//!    constant `q: f64` / constant matrices.
//! 5. `-pmg` and `-pref > 0` — same gaps as `pdiffusion`.
//!
//! # Available today
//!
//! * `pdiffusion` — verified 1:1 against the C++ MPI reference for `-np 1` and
//!   `-np 2` (dofs, L2 error, DPG residual).
//! * Serial complex DPG: `dpg_maxwell_2d`, `dpg_maxwell_3d`.
//!
//! Usage:
//!   cargo run --release --example pmaxwell -- --ranks 2
//! (exits 3 with the gap list above)

use std::process::exit;

const GAPS: &[&str] = &[
    "no parallel COMPLEX DPG: MFEM uses ParComplexDPGWeakForm \
     (miniapps/dpg/util/pcomplexweakform.{hpp,cpp}); fem-rs has the serial \
     ComplexDPGWeakForm and the real ParDpgWeakForm, but no complex parallel \
     trace numbering / real-doubled P^T A P / complex essential-DOF elimination",
    "ND-trace (H(curl) skeleton) parallel numbering is missing: ParDpgWeakForm \
     numbers 2-D H1-trace and face-discontinuous trace blocks only; the 3-D E-hat/H-hat \
     and the 2-D E-hat are ND_Trace_FECollection blocks with MFEM edge-orientation \
     signs and panics today",
    "3-D H1-trace parallel numbering is missing (edge-shared DOFs on the skeleton)",
    "the PML is not ported: CartesianPML, PmlCoefficient, PmlMatrixCoefficient and \
     RestrictedCoefficient (miniapps/dpg/util/pml.{hpp,cpp}) have no fem-rs \
     counterpart, so `-prob 2/3/4` (pml_general, pml_plane_wave_scatter, \
     pml_pointsource) cannot be assembled; they also need spatially varying MATRIX \
     coefficients, which the Dpg*Integrator family only accepts as constant `q: f64`",
    "`-pmg` (PRefinementMultigrid) is not ported (no p-prolongation for DPG blocks)",
    "`-pref > 0` (parallel AMR + ParDPGWeakForm::Update) is not ported: \
     fem_parallel::par_refine_marked* rebuilds the partition with compact node ids, \
     which breaks the identity node numbering the DPG trace numbering depends on",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let names = [
        "plane_wave",
        "fichera_oven",
        "pml_general",
        "pml_plane_wave_scatter",
        "pml_pointsource",
    ];
    let iprob: usize = get("-prob")
        .or_else(|| get("--problem"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(names.len() - 1);
    let mesh = get("-m").or_else(|| get("--mesh")).unwrap_or_else(|| {
        match iprob {
            1 => "meshes/fichera-waveguide.mesh".to_string(),
            3 | 4 => "meshes/scatter.mesh".to_string(),
            _ => "../../data/inline-quad.mesh".to_string(),
        }
    });
    let order = get("-o").or_else(|| get("--order")).unwrap_or_else(|| "1".into());
    let delta_order = get("-do")
        .or_else(|| get("--delta-order"))
        .unwrap_or_else(|| "1".into());
    let rnum = get("-rnum")
        .or_else(|| get("--number-of-wavelengths"))
        .unwrap_or_else(|| "1".into());
    let sr = get("-sref").or_else(|| get("--serial-ref")).unwrap_or_else(|| "0".into());
    let pr = get("-pref").or_else(|| get("--parallel-ref")).unwrap_or_else(|| "1".into());
    let ranks = get("--ranks").unwrap_or_else(|| "2".into());

    println!("=== fem-rs pmaxwell: parallel ultraweak DPG for Maxwell's equations ===");
    println!("Options used:");
    println!("   --mesh {mesh}");
    println!("   --order {order}");
    println!("   --delta-order {delta_order}");
    println!("   --number-of-wavelengths {rnum}");
    println!("   --problem {iprob} ({})", names[iprob]);
    println!("   --serial-ref {sr}");
    println!("   --parallel-ref {pr}");
    println!("   --ranks {ranks}");
    println!("   --no-visualization");
    println!();
    eprintln!(
        "pmaxwell: GAP — this miniapp is a complex-valued parallel DPG Maxwell problem \
         with PML and fem-rs cannot assemble it yet:"
    );
    for (i, g) in GAPS.iter().enumerate() {
        eprintln!("  {}. {g}", i + 1);
    }
    eprintln!(
        "Refusing to print a convergence table: a wrong L2 error / iteration count must \
         never be reported as success.  Run `pdiffusion` for the verified parallel DPG \
         path, or `dpg_maxwell_2d` / `dpg_maxwell_3d` for the serial complex DPG."
    );
    exit(3);
}
