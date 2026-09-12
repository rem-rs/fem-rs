//! Miniapp: NURBS Example 3 — Electromagnetic diffusion with H(curl) NURBS.
//! 1:1 port of MFEM nurbs_ex3.cpp: `curl muinv curl E + sigma E = f` with
//! tangential (essential) boundary data, discretized with the
//! `NURBS_HCurlFECollection` + `NURBSExtension` H(curl) NURBS space
//! ([`NurbsHCurlSpace`]).
//!
//! The exact solution `E = (sin(kappa y), sin(kappa x))` drives both the
//! right-hand side `f = (1 + kappa^2) E` (`VectorFEDomainLFIntegrator`) and
//! the initial/essential data via `x.ProjectCoefficient(E)` — MFEM's
//! `NURBS_HCurl*FiniteElement::Project` Botella-point interpolation.  All
//! boundary attributes are essential (`ess_bdr = 1`), so `FormLinearSystem`
//! imposes the exact tangential trace and PCG starts from the projected
//! field.
//!
//! Port notes:
//! * The solve block (PCG + `GSSmoother`, `rtol 1e-12`, `max_it 500`) and the
//!   L2-error line are byte-identical to the C++ binary.
//! * Not ported: `refined.mesh` / `sol.gf` (no NURBS mesh writer, see
//!   `nurbs_ex1`), the GLVis socket and the VisIt data collection.
//! * `GetCurlExtension` (and hence this example) requires single-patch NURBS
//!   meshes, as in MFEM.

use std::f64::consts::{LN_2, PI};

use fem_linalg::fem_to_linlvo_csr;
use fem_solver::{fmt_g, solve_pcg, GSSmoother};
use fem_space::constraints::form_linear_system;
use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsHCurlSpace;

/// `freq = 1.0` and `kappa = freq * M_PI` (nurbs_ex3.cpp globals).
fn kappa(freq: f64) -> f64 {
    freq * PI
}

/// `E_exact` (nurbs_ex3.cpp): `dim`-component exact field.
fn exact_e(x: &[f64], kap: f64, dim: usize) -> Vec<f64> {
    if dim == 3 {
        vec![(kap * x[1]).sin(), (kap * x[2]).sin(), (kap * x[0]).sin()]
    } else {
        let mut e = vec![(kap * x[1]).sin(), (kap * x[0]).sin()];
        if x.len() == 3 {
            e.push(0.0);
        }
        e
    }
}

/// `f_exact` (nurbs_ex3.cpp): `(1 + kappa^2) E`.
fn exact_f(x: &[f64], kap: f64, dim: usize) -> Vec<f64> {
    let k2 = 1.0 + kap * kap;
    if dim == 3 {
        vec![k2 * (kap * x[1]).sin(), k2 * (kap * x[2]).sin(), k2 * (kap * x[0]).sin()]
    } else {
        let mut f = vec![k2 * (kap * x[1]).sin(), k2 * (kap * x[0]).sin()];
        if x.len() == 3 {
            f.push(0.0);
        }
        f
    }
}

struct Args {
    mesh: String,
    order: i32,
    ref_levels: i32,
    freq: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/square-nurbs.mesh".to_string(),
        order: 1,
        ref_levels: -1,
        freq: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next().unwrap_or_else(|| a.mesh.clone()); }
            "-o" | "--order" => { a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1); }
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "-f" | "--frequency" => {
                a.freq = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let kap = kappa(args.freq);
    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // 3. `Mesh *mesh = new Mesh(mesh_file, 1, 1); dim = mesh->Dimension();`
    let mesh_ext = NurbsExtension::from_mesh_str(&text).expect("failed to parse the NURBS mesh");
    let dim = mesh_ext.dim();
    let n_elems = mesh_ext.n_elements();

    // 4. `ref_levels = floor(log(50000./NE)/log(2.)/dim)` when `-r` is not
    //    given, then that many uniform refinements.
    let ref_levels = if args.ref_levels < 0 {
        ((50000.0_f64 / n_elems as f64).ln() / LN_2 / dim as f64).floor() as i32
    } else {
        args.ref_levels
    } as usize;

    // 5. `fec = new NURBS_HCurlFECollection(order, dim);
    //     NURBSext = new NURBSExtension(mesh->NURBSext, order);`
    //     mfem::out << "Create NURBS fec and ext" << std::endl;
    println!("Create NURBS fec and ext");
    let space = NurbsHCurlSpace::from_mesh_str(&text, ref_levels, args.order as usize)
        .expect("failed to build the H(curl) NURBS space");
    println!("Number of finite element unknowns: {}", space.n_dofs());

    // 6. `ess_bdr = 1` (every boundary attribute essential) ->
    //    `GetEssentialTrueDofs(ess_bdr, ess_tdof_list)`.
    let ess_tdof_list = if mesh_ext.max_bdr_attribute() > 0 {
        space.essential_dofs()
    } else {
        Vec::new()
    };
    println!("Number of knowns in essential BCs: {}", ess_tdof_list.len());

    // 7. `b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));`
    let b = space.assemble_vector_domain_lf(&|x: &[f64]| exact_f(x, kap, dim));

    // 8. `x.ProjectCoefficient(E);` — the projected field provides both the
    //    essential (tangential) data and PCG's initial guess.
    let mut x = space.project_coefficient(&|x: &[f64]| exact_e(x, kap, dim));
    // `ess_tdof_list` membership mask (MFEM excludes the interior of `X`).
    let mut ess_mask = vec![false; x.len()];
    for &d in &ess_tdof_list {
        ess_mask[d as usize] = true;
    }

    // 9.-10. `a->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
    //         a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));`
    let mut a = space.assemble_system(1.0, 1.0);
    let ess_vals: Vec<f64> = ess_tdof_list.iter().map(|&d| x[d as usize]).collect();
    // `a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B)` with MFEM's default
    // `copy_interior = 0`, i.e. `X.SetSubVectorComplement(ess_tdof_list, 0.0)`:
    // only the essential values of the projected field survive as PCG's initial
    // guess, the interior part of `x` is discarded.
    for (i, v) in x.iter_mut().enumerate() {
        if !ess_mask[i] {
            *v = 0.0;
        }
    }
    let mut rhs = b;
    form_linear_system(&mut a, &mut rhs, &mut x, &ess_tdof_list, &ess_vals);

    println!("Size of linear system: {}", a.nrows);

    // 11. `GSSmoother M((SparseMatrix &)(*A));
    //      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);`
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a)).expect("GS smoother");
    solve_pcg(&a, &rhs, &mut x, &gs, 1e-12, 500, true).expect("PCG failed");

    // 13. `cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n'
    //      << endl;`
    let err = space.compute_l2_error(&x, &|x: &[f64]| exact_e(x, kap, dim));
    println!();
    println!("|| E_h - E ||_{{L^2}} = {}", fmt_g(err));
    println!();
}
