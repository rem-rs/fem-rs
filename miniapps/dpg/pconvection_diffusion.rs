//! Parallel ultraweak DPG solver for the convection–diffusion problem — MFEM
//! `miniapps/dpg/pconvection-diffusion.cpp` (MFEM 4.10).
//!
//! **STATUS: NOT PORTED — declarative `exit(3)` with the gap list below.**
//! The C++ miniapp parses the same command line and this port prints the
//! identical option banner, then refuses to run rather than emit a wrong
//! number.
//!
//! # What the C++ miniapp solves
//!
//! ```text
//!     -ε Δu + β·∇u = f   in Ω,     u = u₀   on ∂Ω
//! ```
//!
//! through the first-order system `σ = ε ∇u`, `∇·σ − β·∇u = ...` with
//! skeleton traces `û ∈ H^{1/2}`, `f̂ ∈ H^{-1/2}` and the block table
//! (`util/weakform.hpp` `DPGWeakForm`):
//!
//! ```text
//! |   |     u      |     σ      |    û     |    f̂     |  RHS   |
//! | v | -(βu,∇v)   |  (σ,∇v)    |          | <f̂,v>    |  (f,v) |
//! | τ | -(u,∇·τ)   | 1/ε(σ,τ)   | <û,τ·n>  |          |        |
//! ```
//!
//! with the *coefficient-weighted* test norm
//! `c₁(v,δv) + ε(∇v,∇δv) + (β·∇v,β·∇δv) + c₂(τ,δτ) + (∇·τ,∇·δτ)`, where the
//! L²(P0) coefficient fields `c₁, c₂` are computed **per element** by
//! `setup_test_norm_coeffs` (`util/preconditioners.cpp`) from a local
//! eigenvalue/optimisation analysis.
//!
//! # Gap list (what fem-rs is missing)
//!
//! The parallel machinery itself is *not* the blocker — the real-valued
//! `ParDPGWeakForm` equivalent
//! ([`fem_parallel::par_dpg_weakform::ParDpgWeakForm`]) exists and is verified
//! by `miniapps/dpg/pdiffusion.rs`.  What is missing sits in
//! `crates/assembly` (owned by other routes) and has no serial counterpart to
//! build on either (`miniapps/dpg/` has no `convection_diffusion` port):
//!
//! 1. **Spatially varying coefficients.** `DpgMixedScalarWeakDivergenceIntegrator`,
//!    `DpgMassIntegrator`, `DpgDiffusionIntegrator`, `DpgVectorFEMassIntegrator`
//!    all carry a *constant* `q: f64` (or a constant matrix `Vec<Vec<f64>>`).
//!    pconvection-diffusion needs:
//!    * a **vector** coefficient `β(x)` (`MixedScalarWeakDivergenceIntegrator(betacoeff)`),
//!    * a **tensor** coefficient `ββᵀ` (`DiffusionIntegrator(bbtcoeff)`) plus the
//!      constant `ε` variant,
//!    * two **L²(P0) GridFunction coefficients** `c₁, c₂` on the test spaces.
//! 2. **`setup_test_norm_coeffs`** (`util/preconditioners.cpp`): the per-element
//!    computation of the test-norm weights `c₁, c₂`.  Without it the test norm —
//!    and therefore the DPG solution — is a different method, so the C++
//!    convergence table could not be reproduced.
//! 3. `-pmg` (`PRefinementMultigrid`) and `-pref > 0` (parallel AMR) share the
//!    same gaps as `pdiffusion`.
//!
//! # Available today
//!
//! * `--ranks {1,2}`: distributed broken volume + trace spaces, global trace
//!   numbering, `Pᵀ A P` assembly, essential-DOF elimination on global ids,
//!   static condensation, residual and L²-error reduction (see `pdiffusion`).
//! * Constant-coefficient versions of every integrator above.
//!
//! Usage:
//!   cargo run --release --example pconvection_diffusion -- --ranks 2
//! (exits 3 with the gap list above)

use std::process::exit;

const GAPS: &[&str] = &[
    "no DPG integrator with a spatially varying / vector / tensor coefficient: \
     fem-rs `DpgMixedScalarWeakDivergenceIntegrator`, `DpgMassIntegrator`, \
     `DpgDiffusionIntegrator` and `DpgVectorFEMassIntegrator` all take a constant \
     `q: f64`; the C++ miniapp needs beta(x) (vector), beta*beta^T (tensor) and the \
     L2(P0) coefficient fields c1, c2 (crates/assembly, owned by another route)",
    "`setup_test_norm_coeffs` (miniapps/dpg/util/preconditioners.cpp) — the per-element \
     test-norm weights c1, c2 — is not ported; without it the test norm differs and the \
     DPG solution is a different discretisation (no C++ number could be matched)",
    "`-pmg` (PRefinementMultigrid) is not ported (no p-prolongation for DPG blocks)",
    "`-pref > 0` (parallel AMR + ParDPGWeakForm::Update) is not ported: \
     fem_parallel::par_refine_marked* rebuilds the partition with compact node ids, \
     which breaks the identity node numbering the DPG trace numbering depends on",
];

fn main() {
    // Same command line as the C++ miniapp (`OptionsParser`).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let mesh = get("-m").or_else(|| get("--mesh")).unwrap_or_else(|| {
        let prob: i32 = get("-prob").and_then(|v| v.parse().ok()).unwrap_or(0);
        // C++: prob EJ / curved_streamlines / bdr_layer force inline-quad.
        if (1..=3).contains(&prob) {
            "../../data/inline-quad.mesh".to_string()
        } else {
            "../../data/inline-quad.mesh".to_string()
        }
    });
    let order = get("-o").or_else(|| get("--order")).unwrap_or_else(|| "1".into());
    let delta_order = get("-do")
        .or_else(|| get("--delta-order"))
        .unwrap_or_else(|| "1".into());
    let sref = get("-sref")
        .or_else(|| get("--serial-ref"))
        .unwrap_or_else(|| "0".into());
    let pref = get("-pref")
        .or_else(|| get("--parallel-ref"))
        .unwrap_or_else(|| "0".into());
    let prob = get("-prob").or_else(|| get("--problem")).unwrap_or_else(|| "0".into());
    let ranks = get("--ranks").unwrap_or_else(|| "2".into());

    let names = ["sinusoidal", "EJ", "curved_streamlines", "bdr_layer"];
    let ip: usize = prob.parse::<usize>().unwrap_or(0).min(names.len() - 1);

    println!("=== fem-rs pconvection-diffusion: parallel ultraweak DPG for \
              convection-diffusion ===");
    println!("Options used:");
    println!("   --mesh {mesh}");
    println!("   --order {order}");
    println!("   --delta_order {delta_order}");
    println!("   --num-serial-refinements {sref}");
    println!("   --num-parallel-refinements {pref}");
    println!("   --problem {prob} ({})", names[ip]);
    println!("   --ranks {ranks}");
    println!("   --no-visualization");
    println!();
    eprintln!(
        "pconvection-diffusion: GAP — the parallel DPG machinery \
         (ParDpgWeakForm = distributed broken + trace spaces, P^T A P, essential-DOF \
         elimination on global ids) IS implemented in fem-rs and verified by \
         miniapps/dpg/pdiffusion.rs, but this miniapp cannot be finished because:"
    );
    for (i, g) in GAPS.iter().enumerate() {
        eprintln!("  {}. {g}", i + 1);
    }
    eprintln!(
        "Refusing to print a convergence table: a wrong L2 error / residual must never \
         be reported as success.  Run `pdiffusion` for the verified parallel DPG path."
    );
    exit(3);
}
