//! Miniapp: MFEM `nurbs_ex24` — mixed spaces on a NURBS mesh.
//!
//! **Status: partially ported (round 26).**  This file is *not* a complete 1:1
//! port of `miniapps/nurbs/nurbs_ex24.cpp`; the stages below are, and
//! everything from the coefficient projection onwards exits with status 3
//! rather than printing numbers this port cannot produce.
//!
//! MFEM's example has three variants of the mixed de Rham projectors:
//! `-p 0` `(grad p, u)` for `p ∈ H¹` tested against `u ∈ H(curl)`,
//! `-p 1` `(curl v, u)` for `v ∈ H(curl)` tested against `u ∈ H(div)` (3-D only),
//! `-p 2` `(div v, q)` for `v ∈ H(div)` tested against `q ∈ L_2` — with
//! `trial_fec`/`test_fec` built from `NURBSFECollection`,
//! `NURBS_HCurlFECollection(order, dim)` and `NURBS_HDivFECollection(order, dim)`
//! on a single `NURBSExtension(mesh->NURBSext, order)` (stolen by the test
//! space), then `ProjectCoefficient(coeff, ProjectType::DEFAULT)`, a mass
//! matrix + the matching mixed integrator, PCG (`rtol 1e-12`, `max_iter 1000`)
//! with `DSmoother` and two `ComputeL2Error` lines.
//!
//! Ported 1:1 (verified against the C++ binary, MFEM 4.10, with
//! `cube-nurbs.mesh -o 1 -r 1`):
//!
//! | variant | `trial_size` / `test_size` | MFEM's printed lines |
//! |---|---|---|
//! | `-p 0` | 27 H¹, 144 H(curl) | `Number of HCurl finite element unknowns: 144` / `Number of H1 finite element unknowns: 27` |
//! | `-p 1` | 144 H(curl), 108 H(div) | `Number of HCurl finite element unknowns: 144` / `Number of HDiv finite element unknowns: 108` |
//! | `-p 2` | 108 H(div), 27 L₂ | `Number of HDiv finite element unknowns: 108` / `Number of L2 finite element unknowns: 27` |
//!
//! (the 3-D default `-r 1` grid has 8 elements; the `-o 1` analysis spaces are
//! `NurbsFESpace` = 27, `NurbsHCurlSpace` = 144 and `NurbsHDivSpace` = 108 DOFs).
//!
//! **Not ported**: steps 6-11, i.e. `GridFunction::ProjectCoefficient(coeff,
//! `ProjectType::DEFAULT)` on the *trial* space followed by
//! `MixedBilinearForm::SpMat().Mult` with `MixedVectorGradientIntegrator` /
//! `MixedVectorCurlIntegrator` / `VectorFEDivergenceIntegrator` between two
//! *different* NURBS spaces, the `MassIntegrator`/`VectorFEMassIntegrator`
//! system, PCG with `DSmoother`, and the two `ComputeL2Error` lines.
//! `MixedVectorGradientIntegrator` (H¹ scalar → H(curl)) and
//! `MixedVectorCurlIntegrator` (H(curl) → H(div)) are cross-space NURBS forms
//! that fem-rs does not implement; `VectorFEDivergenceIntegrator` (H(div) →
//! H¹ scalar) *is* available as
//! `NurbsHDivSpace::assemble_mixed_divergence` but the rest of the `-p 2`
//! pipeline is not, so all three variants stop at the same place.  MFEM's
//! iteration logs (`(B r, r)` per step and `Average reduction factor`) and the
//! two error lines per variant are therefore not reproducible here.
//! (Round 28 added `NurbsHDivSpace::assemble_vector_boundary_flux`, the D106
//! boundary-element assembly path — `nurbs_ex24` itself has no boundary form,
//! so it does not advance this file.)
//!
//! Also not ported: `refined.mesh` / `sol.gf` (no NURBS mesh writer, see
//! `nurbs_ex1`) and the GLVis socket.

use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHCurlSpace, NurbsHDivSpace};

struct Args {
    mesh: String,
    order: usize,
    ref_levels: i64,
    prob: usize,
    nurbs: bool,
}

fn parse_args() -> Args {
    let mut a =
        Args { mesh: "data/cube-nurbs.mesh".to_string(), order: 1, ref_levels: -1, prob: 0, nurbs: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_else(|| a.mesh.clone()),
            "-o" | "--order" => {
                a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "-p" | "--problem-type" => {
                a.prob = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "-n" | "--nurbs" => a.nurbs = true,
            "-nn" | "--no-nurbs" => a.nurbs = false,
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // 3. `Mesh *mesh = new Mesh(mesh_file, 1, 1); dim = mesh->Dimension();`
    let geo = fem_space::NurbsExtension::from_mesh_str(&text).expect("NURBS mesh");
    let dim = geo.dim();
    if args.prob == 1 && dim != 3 {
        eprintln!("nurbs_ex24: MFEM_ABORT — the curl problem is only defined in 3D.");
        std::process::exit(3);
    }
    // MFEM 4.10's serial `nurbs_ex24` has no NURBS-aware fallback here: the
    // `else` branch (standard `H1/ND/RT/L2_FECollection`) is *not* a NURBS
    // space and fem-rs's NURBS path is the only one this miniapp can build.
    if !(args.nurbs && geo.n_patches() == 1) {
        eprintln!(
            "nurbs_ex24: only the NURBS branch (`-n`, single patch) is ported; the standard \
             finite-element branch is not."
        );
        std::process::exit(3);
    }

    // 4. `ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim)`.
    let n_elems = geo.n_elements();
    let ref_levels = if args.ref_levels < 0 {
        ((50000.0_f64 / n_elems as f64).ln() / std::f64::consts::LN_2 / dim as f64).floor() as i64
    } else {
        args.ref_levels
    } as usize;

    // 5. `NURBSext = new NURBSExtension(mesh->NURBSext, order);` plus the
    //    `{trial,test}_fec` pair, `mfem::out << "Create NURBS finite element"`.
    println!("Create NURBS finite element");
    let n_h1 = NurbsFESpace::from_mesh_str(&text, ref_levels, &[args.order])
        .expect("H1 NURBS space")
        .n_dofs();
    let n_curl = NurbsHCurlSpace::from_mesh_str(&text, ref_levels, args.order)
        .expect("H(curl) NURBS space")
        .n_dofs();
    let n_div = NurbsHDivSpace::from_mesh_str(&text, ref_levels, args.order)
        .expect("H(div) NURBS space")
        .n_dofs();

    // `trial_size = trial_fes.GetTrueVSize(); test_size = test_fes.GetTrueVSize();`
    // printed per variant (MFEM's `NURBSFECollection` slots are H¹ spaces, not
    // L₂ — the `else` branch's `L2_FECollection` is the non-NURBS path).
    match args.prob {
        0 => {
            println!("Number of HCurl finite element unknowns: {n_curl}");
            println!("Number of H1 finite element unknowns: {n_h1}");
        }
        1 => {
            println!("Number of HCurl finite element unknowns: {n_curl}");
            println!("Number of HDiv finite element unknowns: {n_div}");
        }
        _ => {
            println!("Number of HDiv finite element unknowns: {n_div}");
            println!("Number of L2 finite element unknowns: {n_h1}");
        }
    }

    // 6.-11. projection, the mixed assembly, the PCG solve and the two
    // `ComputeL2Error` lines.
    eprintln!(
        "nurbs_ex24: stages 6-11 are not ported (the cross-space NURBS mixed integrators \
         MixedVectorGradientIntegrator / MixedVectorCurlIntegrator and the DSmoother-backed \
         PCG are missing from fem-rs); see the module docs. Run MFEM's nurbs_ex24 for the \
         iteration log and the error lines."
    );
    std::process::exit(3);
}
