//! D337 regression: hex RT `-o >= 2` in ex24 `-p 2` (div → L²).
//!
//! After D329 un-panicked the hex mixed-divergence consumer, the example
//! `examples/mfem_ex24_discrete_ops.rs -m data/inline-hex.mesh -p 2` still
//! disagreed with MFEM 4.10 ex24 for every RT order ≥ 1.  A layer-by-layer
//! ladder (`tmp/d342/D337_report.md`) pinned **two** independent defects, both
//! covered here:
//!
//! 1. `postproc::grid_function::compute_l2_error_l2` reconstructed the field
//!    with `ElementType::ref_elem(order)` — the **H1** Gauss-Lobatto
//!    `HexQk`/`QuadQk` in topological order — for a default (GaussLegendre,
//!    lexicographic) L² space.  Every order ≥ 1 therefore reported a bogus
//!    error (ex24 `-p 2 -o 3`: 0.0166 instead of 1.16e-7).  Fixed by
//!    evaluating with `assembler::ref_elem_vol_for_space`, the same reference
//!    element the assembly uses.
//! 2. `project_hdiv_coefficient_{2,3}d` claimed to mirror MFEM's
//!    `GridFunction::ProjectCoefficient` but computed an **L² projection**;
//!    the MFEM call is `Project_RT`, the nodal/dual *interpolant*, the only
//!    operator satisfying `div ∘ I = P_L2 ∘ div` (which is why C++'s
//!    `errSol == errProj`).  Fixed by delegating to
//!    `HDivSpace::interpolate_vector`, the D289-verified `Project_RT` engine.
//!
//! Ground truth (`tmp/d342/ex24_div_prec.cpp` — MFEM 4.10 ex24, div branch,
//! `cout.precision(17)`; `-n N -r R` = `N³` Cartesian hexes refined `R` times,
//! which is exactly `data/inline-hex.mesh` for `N = 4`):
//!
//! ```text
//!   elements    -o  errSol (=errInterp)      errProj
//!        64      1  8.6694338617974723e-02    8.6728938847397560e-02
//!        64      2  1.7151055588173131e-03    1.7155139344217150e-03
//!        64      3  5.9527359640757348e-05    5.9528154477954811e-05
//!       512      1  4.3538652740310962e-02    4.3543059725352967e-02
//!       512      2  4.3031834340324931e-04    4.3034334022083812e-04
//!       512      3  7.4353255363526030e-06    7.4353505700396996e-06
//!     32768      1  1.0899634115392687e-02    1.0899703375005593e-02
//!     32768      2  2.6924835113406903e-05    2.6924932121628810e-05
//!     32768      3  1.1614989573674600e-07    1.1614992024991857e-07
//!     32768      4  1.3998768047217274e-10    1.3998749928750382e-10
//! ```
//!
//! fem-rs reproduces `errSol`/`errInterp` and the interpolant form of line (c)
//! to 10–15 digits at every size above.  Two residual gaps, both outside this
//! crate's D337 scope, are recorded as debts in `tmp/d342/D337_report.md`: the
//! example's inline L² form of line (c) (D343) and hex RT3 / `-o 4`, which
//! panics in `HDivSpace::new` (D342).
//!
//! Run the default (fast) set via
//!   `cargo test -p fem-assembly --test d337_ex24_div_ladder -- --nocapture`
//! and the ex24-sized acceptance leg via
//!   `cargo test --release -p fem-assembly --test d337_ex24_div_ladder -- --ignored --nocapture`

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::postproc::grid_function::{
    compute_l2_error_l2, project_hdiv_coefficient_3d,
};
use fem_assembly::standard::{DomainSourceIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_mesh::Mesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{fe_space::FESpace, HDivSpace, L2Space};

fn div_gradp_exact(x: &[f64]) -> f64 {
    -3.0 * x[0].sin() * x[1].sin() * x[2].sin()
}

fn gradp_exact(x: &[f64]) -> Vec<f64> {
    vec![
        x[0].cos() * x[1].sin() * x[2].sin(),
        x[0].sin() * x[1].cos() * x[2].sin(),
        x[0].sin() * x[1].sin() * x[2].cos(),
    ]
}

/// ex24 `-p 2` on `n³` Cartesian hexes refined `ref_levels` times, replicating
/// `examples/mfem_ex24_discrete_ops.rs::solve_div_3d` (`cli_order` = `-o`).
///
/// Returns `(errSol, errInterp, errProj_interp, errProj_l2)`:
/// * `errSol`/`errInterp` — the two RT legs of the example;
/// * `errProj_interp` — line (c) as **MFEM** computes it: on a nodal L² space
///   `GridFunction::ProjectCoefficient` is the nodal interpolant
///   (`L2Space::interpolate` = `f(dof_coords)`);
/// * `errProj_l2` — line (c) as the shipped example computes it, an inline L²
///   mass solve.  The two differ by `O(h^{p+1})` in the L² error (D343).
///
/// All four use C++'s error rule `max(2, 2*order+1)` so they are comparable
/// with MFEM digit for digit.
fn ex24_div_pipeline(n: usize, ref_levels: usize, cli_order: u8) -> (f64, f64, f64, f64) {
    let mut mesh = Mesh::<3>::unit_cube_hex(n);
    for _ in 0..ref_levels {
        mesh = fem_mesh::refine_uniform_3d(&mesh);
    }
    let qo = (2 * cli_order + 1).max(3) as u8;
    let err_qo = (2 * cli_order + 1).max(2) as u8; // C++ `order_quad`
    let rt_order = cli_order - 1;
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    let l2 = L2Space::new(mesh.clone(), rt_order);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    // v_h: MFEM `ProjectCoefficient` == `Project_RT` (the example's helper).
    let v = project_hdiv_coefficient_3d(&rt, &|x: &[f64], out: &mut [f64]| {
        let g = gradp_exact(x);
        out[..3].copy_from_slice(&g[..3]);
    }, qo);

    let d = assemble_hdiv_l2_mixed(&l2, &rt, &[&HDivL2DivIntegrator], qo);
    let mass = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], qo);
    let mut rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut rhs);

    let mut f_sol = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs, &mut f_sol, &cfg).expect("PCG sol");
    let mut f_interp = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs, &mut f_interp, &cfg).expect("PCG interp");
    let rhs_ex =
        Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(div_gradp_exact)], qo);
    let mut f_ex = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut f_ex, &cfg).expect("PCG ex");
    let f_ex_interp = l2.interpolate(&div_gradp_exact).into_vec();

    (
        compute_l2_error_l2(&f_sol, &l2, &div_gradp_exact, err_qo, None),
        compute_l2_error_l2(&f_interp, &l2, &div_gradp_exact, err_qo, None),
        compute_l2_error_l2(&f_ex_interp, &l2, &div_gradp_exact, err_qo, None),
        compute_l2_error_l2(&f_ex, &l2, &div_gradp_exact, err_qo, None),
    )
}

/// MFEM 4.10 ex24 `-p 2` ground truth for the two small ladders (see module
/// docs).  `(elements, cli_order, errSol, errProj)`.
const MFEM_GOLDEN: &[(usize, u8, f64, f64)] = &[
    (64, 1, 8.6694338617974723e-02, 8.6728938847397560e-02),
    (64, 2, 1.7151055588173131e-03, 1.7155139344217150e-03),
    (64, 3, 5.9527359640757348e-05, 5.9528154477954811e-05),
    (512, 1, 4.3538652740310962e-02, 4.3543059725352967e-02),
    (512, 2, 4.3031834340324931e-04, 4.3034334022083812e-04),
    (512, 3, 7.4353255363526030e-06, 7.4353505700396996e-06),
];

/// Every ex24 `-p 2` leg must reproduce MFEM digit for digit on the small
/// ladders — this is the regression that D337's two fixes establish.  Line (c)
/// is pinned through its *interpolant* form (MFEM's `ProjectCoefficient`
/// semantics); the L²-projection form the shipped example uses inline is
/// checked for its documented `O(h^{p+1})` offset (D343).
#[test]
fn ex24_div_pipeline_matches_mfem_small_meshes() {
    for &(elems, cli_order, want_sol, want_proj) in MFEM_GOLDEN {
        // 64 elements = 2³ hexes refined once; 512 = 2³ refined twice.
        let (n, ref_levels) = if elems == 64 { (2usize, 1usize) } else { (2, 2) };
        let (e1, e2, e3_interp, e3_l2) = ex24_div_pipeline(n, ref_levels, cli_order);
        println!(
            "  {elems} elems o={cli_order}: sol={e1:.17e} interp={e2:.17e} \
             proj(interp)={e3_interp:.17e} proj(L2, example)={e3_l2:.17e} \
             (mfem {want_sol:.17e} / {want_proj:.17e})"
        );
        for (tag, got, want) in [
            ("errSol", e1, want_sol),
            ("errInterp", e2, want_sol),
            ("errProj (interpolant)", e3_interp, want_proj),
        ] {
            assert!(
                (got - want).abs() <= 1e-11 * want.abs(),
                "{elems} elems o={cli_order} {tag}: {got:.17e} vs mfem {want:.17e}"
            );
        }
        // D343: the example's inline L²-projection line (c) sits within
        // O(h^{cli_order}) of MFEM's interpolant line — at 64 elements
        // (h = 1/4) that is 7.2e-4 relative, at 32³ ~1e-5.
        assert!(
            (e3_l2 - want_proj).abs() <= 1e-2 * want_proj.abs(),
            "{elems} elems o={cli_order} errProj (L², example): {e3_l2:.17e} vs mfem {want_proj:.17e}"
        );
    }
}

/// The RT field the example builds must be MFEM's `Project_RT` interpolant, not
/// an L² projection — reintroducing the mass-matrix path changes ex24's `-p 2`
/// result by ~1e-3 relative (RT1) / 2.4e-2 (RT2).
#[test]
fn project_hdiv_coefficient_3d_is_the_project_rt_interpolant() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    for order in 0..=2u8 {
        let rt = HDivSpace::new(mesh.clone(), order);
        let qo = (2 * (order as usize + 1) + 1).max(3) as u8;
        let got = project_hdiv_coefficient_3d(&rt, &|x: &[f64], out: &mut [f64]| {
            let g = gradp_exact(x);
            out[..3].copy_from_slice(&g[..3]);
        }, qo);
        let want = rt.interpolate_vector(&gradp_exact).into_vec();
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert_eq!(
                got[i], want[i],
                "RT{order} dof {i}: projection helper {} != Project_RT {}",
                got[i], want[i]
            );
        }
    }
}

/// D337 defect 1 in isolation: the L² error routine must reconstruct the space
/// its dofs belong to.  A polynomial of degree `p` lies in `L2_p` exactly, so
/// after the projection its error must be round-off; with the old H1 basis it
/// was 4.3e-1 (p=1) / 1.6e-3 (p=2) on this 8-element mesh.
#[test]
fn l2_error_routine_uses_the_space_basis() {
    for l2_p in 0..=2u8 {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let l2 = L2Space::new(mesh.clone(), l2_p);
        let coeff: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = if l2_p == 0 {
            &|_: &[f64]| 3.0
        } else {
            &|x: &[f64]| 1.0 + 2.0 * x[0] + 3.0 * x[1] - 4.0 * x[2]
        };
        let qo = (2 * (l2_p + 1) + 1).max(3) as u8;
        let b = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(coeff)], qo);
        let m = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], qo);
        let mut c = vec![0.0; l2.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-14, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
        solve_pcg_jacobi(&m, &b, &mut c, &cfg).expect("pcg");

        // The nodal dofs themselves (a check that the projection is right, so a
        // failure below can only be the error routine).
        let coords = l2.dof_coords();
        let mut maxdev = 0.0_f64;
        for (i, ch) in coords.chunks_exact(3).enumerate() {
            maxdev = maxdev.max((c[i] - coeff(ch)).abs());
        }
        assert!(maxdev < 1e-12, "L2_{l2_p} dofs are not the nodal values: {maxdev:.3e}");

        let err_qo = (2 * (l2_p + 1) + 6).max(7) as u8;
        let err = compute_l2_error_l2(&c, &l2, &coeff, err_qo, None);
        assert!(err < 1e-12, "L2_{l2_p} exact-polynomial L² error = {err:.3e}");
    }
}

/// The L² error of a smooth field must shrink with the L² order (it used to
/// *grow*: 0.1698 → 0.1354 → 0.2784 on this mesh).
#[test]
fn l2_error_of_smooth_field_shrinks_with_order() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let mut prev = f64::INFINITY;
    for l2_p in 0..=2u8 {
        let l2 = L2Space::new(mesh.clone(), l2_p);
        let qo = (2 * (l2_p + 1) + 1).max(3) as u8;
        let err_qo = (2 * (l2_p + 1) + 6).max(7) as u8;
        let b = Assembler::assemble_linear(
            &l2,
            &[&DomainSourceIntegrator::new(div_gradp_exact)],
            qo,
        );
        let m = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], qo);
        let mut c = vec![0.0; l2.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-14, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
        solve_pcg_jacobi(&m, &b, &mut c, &cfg).expect("pcg");
        let err = compute_l2_error_l2(&c, &l2, &div_gradp_exact, err_qo, None);
        println!("  L2_{l2_p}: {err:.6e}");
        assert!(err < prev, "L2_{l2_p} error {err:.6e} did not improve on {prev:.6e}");
        prev = err;
    }
}

/// Acceptance at the shipped ex24 size: `data/inline-hex.mesh` (= 4³ hexes)
/// with ex24's `ref_levels` = 3 → 32³ hexes, exactly the C++ run.  The two RT
/// legs and line (c) in its interpolant form must hit MFEM's numbers; the
/// example's inline L² form of line (c) is checked for its documented offset
/// (D343).
/// `cargo test --release ... -- --ignored --nocapture`
#[test]
#[ignore = "32^3 mesh with RT2 (2.7M dofs), ~60 s in release"]
fn ex24_div_pipeline_32cubed_matches_mfem() {
    let cases: [(u8, f64, f64); 3] = [
        (1u8, 1.0899634115392687e-02, 1.0899703375005593e-02),
        (2u8, 2.6924835113406903e-05, 2.6924932121628810e-05),
        (3u8, 1.1614989573674600e-07, 1.1614992024991857e-07),
    ];
    for (cli_order, want_sol, want_proj) in cases {
        let (e1, e2, e3_interp, e3_l2) = ex24_div_pipeline(4, 3, cli_order);
        println!(
            "  32^3 -o {cli_order}: sol={e1:.17e} interp={e2:.17e} proj(interp)={e3_interp:.17e} \
             proj(L2, example)={e3_l2:.17e} (mfem {want_sol:.17e} / {want_proj:.17e})"
        );
        assert!(
            (e1 - want_sol).abs() <= 1e-8 * want_sol.abs(),
            "-o {cli_order} errSol: {e1:.17e} vs mfem {want_sol:.17e}"
        );
        assert!(
            (e3_interp - want_proj).abs() <= 1e-8 * want_proj.abs(),
            "-o {cli_order} errProj (interpolant): {e3_interp:.17e} vs mfem {want_proj:.17e}"
        );
        assert!(
            (e3_l2 - want_proj).abs() <= 1e-3 * want_proj.abs(),
            "-o {cli_order} errProj (L², example): {e3_l2:.17e} vs mfem {want_proj:.17e}"
        );
    }
}
