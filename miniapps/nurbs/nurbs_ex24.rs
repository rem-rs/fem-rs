//! Miniapp: MFEM `nurbs_ex24` — mixed spaces on a NURBS mesh.
//!
//! 1:1 port of `miniapps/nurbs/nurbs_ex24.cpp` (MFEM 4.10) for the NURBS
//! branch (`-n`, single patch) and all three de Rham variants:
//!
//! | `-p` | trial space | test space | `a_mixed` integrator | `a` integrator |
//! |---|---|---|---|---|
//! | 0 | `NURBSFECollection(order)` (H¹) | `NURBS_HCurlFECollection` | `MixedVectorGradientIntegrator(one)` | `VectorFEMassIntegrator(one)` |
//! | 1 | `NURBS_HCurlFECollection` | `NURBS_HDivFECollection` (3-D) | `MixedVectorCurlIntegrator(one)` | `VectorFEMassIntegrator(one)` |
//! | 2 | `NURBS_HDivFECollection` | `NURBSFECollection(order)` | `VectorFEDivergenceIntegrator(one)` | `MassIntegrator(one)` |
//!
//! on a single `NURBSExtension(mesh->NURBSext, order)` stolen by the test
//! space, then: `gftrial.ProjectCoefficient(coeff, ProjectType::DEFAULT)`
//! (which for a NURBS space is the element-local L² projection followed by a
//! least-squares fit back onto the NURBS basis), `mixed.SpMat().Mult(gftrial,
//! x)`, PCG (`rtol 1e-12`, `max_iter 1000`, `print_level 1`) with
//! `DSmoother(a.SpMat())`, `exact_proj.ProjectCoefficient(exact, DEFAULT)` and
//! the two `ComputeL2Error` lines.
//!
//! Verified against the C++ binary (MFEM 4.10) with
//! `cube-nurbs.mesh -r 1 -p 0/1/2 -no-vis`.  Reproduced **byte-identically**:
//! the `Create NURBS finite element` banner, the two `Number of … unknowns`
//! lines and both `|| … ||_{L_2} = …` error lines of every variant (the
//! options/device banners printed before the first line are not reproduced, as
//! in the other `nurbs_*` ports).  The PCG iteration block is **not**
//! byte-identical — an honest gap, not a missing piece:
//!
//! | variant | unknowns | C++ PCG | here | C++ errors | here |
//! |---|---|---|---|---|---|
//! | `-p 0` | 144 H(curl), 27 H¹ | 24 it, ARF `0.283378` | 24 it, ARF `0.287008` | `0.0224956`, `0.0039157` | same |
//! | `-p 1` | 144 H(curl), 108 H(div) | 3 it, ARF `5.6814e-08` | 3 it, ARF `5.84589e-08` | `0.488496`, `0.51453` | same |
//! | `-p 2` | 108 H(div), 27 L₂ | 11 it, ARF `0.0547909` | 12 it, ARF `0.0514464` | `0.00271413`, `0.00260626` | same |
//!
//! The assembled system, the projections and the error norms agree with MFEM
//! to 1e-15…1e-12 relative (`crates/space/tests/nurbs_ex24_mixed.rs` pins
//! them); the first ~15 CG iterates print identically, and the Krylov path
//! only splits once `(B r, r)` has fallen to ~1e-11, where a 1-ulp difference
//! in the matrix entries becomes visible at 6 significant digits.  The 1-ulp
//! difference comes from the NURBS **geometry**: MFEM evaluates the
//! knot-inserted control net over the refined spans
//! (`NURBSPatch::UniformRefinement` + `KnotInsert`), while
//! `NurbsFESpace::geometry` deliberately evaluates the *original* control net
//! over the refined parameter intervals — mathematically the same map,
//! rounded differently.  Closing this would mean porting MFEM's control-net
//! knot insertion, not adding a missing integrator.
//!
//! Still not ported (no fem-rs writer): `refined.mesh` / `sol.gf` (see
//! `nurbs_ex1`) and the GLVis socket — steps 12/13 of the C++ file.  The
//! standard (non-NURBS) branch of step 5 (`-nn` on a NURBS mesh, i.e. the
//! `H1/ND/RT/L2_FECollection` path) is not ported either; it aborts with
//! status 3 rather than printing numbers this port cannot produce.

use fem_linalg::CsrMatrix;
use fem_solver::{fmt_g, solve_pcg_dsmoother, SolverConfig, SolverError};
use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHCurlSpace, NurbsHDivSpace};

/// `freq = 1.0; kappa = freq*M_PI;` (file scope in the C++).
const FREQ: f64 = 1.0;

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

// ─── the exact solution of `nurbs_ex24.cpp` (`dim` is the mesh dimension) ────

fn p_exact(x: &[f64], dim: usize) -> f64 {
    if dim == 3 {
        x[0].sin() * x[1].sin() * x[2].sin()
    } else if dim == 2 {
        x[0].sin() * x[1].sin()
    } else {
        0.0
    }
}

fn gradp_exact(x: &[f64], dim: usize) -> Vec<f64> {
    if dim == 3 {
        vec![
            x[0].cos() * x[1].sin() * x[2].sin(),
            x[0].sin() * x[1].cos() * x[2].sin(),
            x[0].sin() * x[1].sin() * x[2].cos(),
        ]
    } else {
        vec![x[0].cos() * x[1].sin(), x[0].sin() * x[1].cos()]
    }
}

fn div_gradp_exact(x: &[f64], dim: usize) -> f64 {
    if dim == 3 {
        -3.0 * x[0].sin() * x[1].sin() * x[2].sin()
    } else if dim == 2 {
        -2.0 * x[0].sin() * x[1].sin()
    } else {
        0.0
    }
}

fn v_exact(x: &[f64], dim: usize) -> Vec<f64> {
    let kappa = FREQ * std::f64::consts::PI;
    if dim == 3 {
        vec![(kappa * x[1]).sin(), (kappa * x[2]).sin(), (kappa * x[0]).sin()]
    } else {
        vec![(kappa * x[1]).sin(), (kappa * x[0]).sin()]
    }
}

fn curlv_exact(x: &[f64], dim: usize) -> Vec<f64> {
    let kappa = FREQ * std::f64::consts::PI;
    if dim == 3 {
        vec![
            -kappa * (kappa * x[2]).cos(),
            -kappa * (kappa * x[0]).cos(),
            -kappa * (kappa * x[1]).cos(),
        ]
    } else {
        vec![0.0; dim]
    }
}

/// Step 9 of the C++: `CGSolver` with `rtol 1e-12`, `max_iter 1000`,
/// `SetPrintLevel(1)` and `DSmoother Jacobi(a.SpMat())`, starting from zero
/// (`rhs = x; x = 0.0;` in C++).
fn solve_pcg(a: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 1000,
        verbose: true,
        ..SolverConfig::default()
    };
    let mut x = vec![0.0_f64; rhs.len()];
    if let Err(e) = solve_pcg_dsmoother(a, rhs, &mut x, &cfg) {
        // MFEM's miniapp ignores the non-convergence; only unexpected errors
        // abort.
        let SolverError::ConvergenceFailed { .. } = e else {
            panic!("PCG failed: {e}");
        };
    }
    x
}

/// `GridFunction::ComputeL2Error(VectorCoefficient)` uses
/// `intorder = 2*fe->GetOrder() + 3` per element; every element of the H(div)
/// NURBS space shares the analysis order, so one rule covers the mesh.
fn hdiv_l2_order(space: &NurbsHDivSpace) -> u8 {
    (2 * space.element_fe(0).order() + 3) as u8
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
    let h1 = NurbsFESpace::from_mesh_str(&text, ref_levels, &[args.order])
        .expect("H1 NURBS space");
    let curl = NurbsHCurlSpace::from_mesh_str(&text, ref_levels, args.order)
        .expect("H(curl) NURBS space");
    let div = NurbsHDivSpace::from_mesh_str(&text, ref_levels, args.order)
        .expect("H(div) NURBS space");

    // `trial_size = trial_fes.GetTrueVSize(); test_size = test_fes.GetTrueVSize();`
    // printed per variant (MFEM's `NURBSFECollection` slots are H¹ spaces, not
    // L₂ — the `else` branch's `L2_FECollection` is the non-NURBS path).
    match args.prob {
        0 => {
            println!("Number of HCurl finite element unknowns: {}", curl.n_dofs());
            println!("Number of H1 finite element unknowns: {}", h1.n_dofs());
        }
        1 => {
            println!("Number of HCurl finite element unknowns: {}", curl.n_dofs());
            println!("Number of HDiv finite element unknowns: {}", div.n_dofs());
        }
        _ => {
            println!("Number of HDiv finite element unknowns: {}", div.n_dofs());
            println!("Number of L2 finite element unknowns: {}", h1.n_dofs());
        }
    }

    // 6.-11. `gftrial.ProjectCoefficient(·, ProjectType::DEFAULT)` (the NURBS
    // element-L² projection), the mass form on the test space, the matching
    // mixed form on `(trial, test)`, `mixed.SpMat().Mult(gftrial, x)`, the PCG
    // solve with `DSmoother`, `exact_proj.ProjectCoefficient(·, DEFAULT)` and
    // the two `ComputeL2Error` lines.
    // (`gftrial.SetTrueVector()/SetFromTrueVector()` are no-ops here: no
    // conforming constraints appear on a conforming NURBS mesh.)
    match args.prob {
        0 => {
            let gftrial = h1.project_coefficient_element_l2(&|x| p_exact(x, dim));
            let a = curl.assemble_mass(1.0);
            let mixed = h1.assemble_mixed_gradient(&curl);
            let mut rhs = vec![0.0_f64; curl.n_dofs()];
            mixed.spmv(&gftrial, &mut rhs);
            let x = solve_pcg(&a, &rhs);

            let exact_proj = curl.project_coefficient_element_l2(&|x| gradp_exact(x, dim));
            let err_sol = curl.compute_l2_error(&x, &|x| gradp_exact(x, dim));
            let err_proj = curl.compute_l2_error(&exact_proj, &|x| gradp_exact(x, dim));

            println!(
                "\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl): \
                 || E_h - grad p ||_{{L_2}} = {}",
                fmt_g(err_sol)
            );
            println!();
            println!(
                " Projection E_h of exact grad p in H(curl): || E_h - grad p ||_{{L_2}} = {}",
                fmt_g(err_proj)
            );
            println!();
        }
        1 => {
            let gftrial = curl.project_coefficient_element_l2(&|x| v_exact(x, dim));
            let a = div.assemble_mass(1.0);
            let mixed = curl.assemble_mixed_curl(&div);
            let mut rhs = vec![0.0_f64; div.n_dofs()];
            mixed.spmv(&gftrial, &mut rhs);
            let x = solve_pcg(&a, &rhs);

            let exact_proj = div.project_coefficient_element_l2(&|x| curlv_exact(x, dim));
            let order_quad = hdiv_l2_order(&div);
            let err_sol = div.compute_l2_error(&x, &|x| curlv_exact(x, dim), order_quad);
            let err_proj = div.compute_l2_error(&exact_proj, &|x| curlv_exact(x, dim), order_quad);

            println!(
                "\n Solution of (E_h,w) = (curl v_h,w) for E_h and w in H(div): \
                 || E_h - curl v ||_{{L_2}} = {}",
                fmt_g(err_sol)
            );
            println!();
            println!(
                " Projection E_h of exact curl v in H(div): || E_h - curl v ||_{{L_2}} = {}",
                fmt_g(err_proj)
            );
            println!();
        }
        _ => {
            let gftrial = div.project_coefficient_element_l2(&|x| gradp_exact(x, dim));
            let a = h1.assemble_mass(1.0);
            let mixed = div.assemble_mixed_divergence(&h1);
            let mut rhs = vec![0.0_f64; h1.n_dofs()];
            mixed.spmv(&gftrial, &mut rhs);
            let x = solve_pcg(&a, &rhs);

            let exact_proj = h1.project_coefficient_element_l2(&|x| div_gradp_exact(x, dim));
            // `int order_quad = max(3, 2*order+1);` with `irs[i] =
            // &IntRules.Get(i, order_quad)` for every geometry.
            let order_quad = std::cmp::max(3, 2 * args.order + 1) as u8;
            let err_sol = h1.compute_l2_error(&x, &|x| div_gradp_exact(x, dim), order_quad);
            let err_proj =
                h1.compute_l2_error(&exact_proj, &|x| div_gradp_exact(x, dim), order_quad);

            println!(
                "\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: \
                 || f_h - div v ||_{{L_2}} = {}",
                fmt_g(err_sol)
            );
            println!();
            println!(
                " Projection f_h of exact div v in L_2: || f_h - div v ||_{{L_2}} = {}",
                fmt_g(err_proj)
            );
            println!();
        }
    }

    // 12.-13. `refined.mesh` / `sol.gf` and the GLVis socket: fem-rs has no
    // writer for NURBS meshes (see `nurbs_ex1`) and no socket either, so
    // nothing is emitted here.
}
