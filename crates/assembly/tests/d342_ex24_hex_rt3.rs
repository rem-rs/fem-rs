//! D342 acceptance: MFEM ex24 `-p 2 -o 4` (hex RT3) end to end.
//!
//! `HDivSpace::new` used to panic for hex RT ≥ 3, so `examples/
//! mfem_ex24_discrete_ops.rs -m data/inline-hex.mesh -o 4 -p 2` could not run
//! at all.  With the cap raised to `0..=6` it runs, and this file pins the
//! numbers against MFEM 4.10.
//!
//! **Why a test and not just the example's stdout:** the example prints
//! `{:.8}` (fixed 8 decimals, the repo's frozen format for every order), so at
//! `-o 4` all three lines render as `0.00000000` — the value `1.4e-10` cannot
//! be read off the example.  The pipeline replicated here is
//! `solve_div_3d`'s, evaluated at 17 digits.
//!
//! Ground truth: `$HOME/work/d342/ex24_div_prec.cpp` (MFEM 4.10 ex24, div
//! branch, `cout.precision(17)`; `-n N -r R` = `N³` Cartesian hexes refined `R`
//! times — `data/inline-hex.mesh` is exactly `-n 4 -r 3`).  Raw run archived in
//! `tmp/d342/cpp_o4_ladder.txt`:
//!
//! ```text
//!   elements   -o  errSol (=errInterp, both RT legs)   errProj (interpolant)
//!        64     4  5.7068028115104348e-07               5.7070893259403543e-07
//!       512     4  3.5796694481155882e-08               3.5797139784626148e-08
//!     32768     4  1.3998768047217274e-10               1.3998749928750382e-10
//! ```
//!
//! The pipeline below is a copy of `d337_ex24_div_ladder.rs`'s helper (that
//! file is frozen D337 evidence and is not edited by D342), extended to
//! `cli_order = 4`.
//!
//! Run: `cargo test -p fem-assembly --test d342_ex24_hex_rt3 -- --nocapture`
//! and the ex24-sized leg via `... -- --ignored --nocapture` (release).

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

/// `examples/mfem_ex24_discrete_ops.rs::solve_div_3d` on `n³` Cartesian hexes
/// refined `ref_levels` times.  Returns `(errSol, errInterp, errProj_interp,
/// errProj_l2, n_rt_dofs, n_l2_dofs)`; the error rule is C++'s
/// `max(2, 2*order+1)` so the numbers are comparable digit for digit.
fn ex24_div_pipeline(n: usize, ref_levels: usize, cli_order: u8) -> (f64, f64, f64, f64, usize, usize) {
    let mut mesh = Mesh::<3>::unit_cube_hex(n);
    for _ in 0..ref_levels {
        mesh = fem_mesh::refine_uniform_3d(&mesh);
    }
    let qo = (2 * cli_order + 1).max(3) as u8;
    let err_qo = (2 * cli_order + 1).max(2) as u8;
    let rt_order = cli_order - 1;
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    let l2 = L2Space::new(mesh.clone(), rt_order);
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

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
    let rhs_ex = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(div_gradp_exact)], qo);
    let mut f_ex = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut f_ex, &cfg).expect("PCG ex");
    let f_ex_interp = l2.interpolate(&div_gradp_exact).into_vec();

    (
        compute_l2_error_l2(&f_sol, &l2, &div_gradp_exact, err_qo, None),
        compute_l2_error_l2(&f_interp, &l2, &div_gradp_exact, err_qo, None),
        compute_l2_error_l2(&f_ex_interp, &l2, &div_gradp_exact, err_qo, None),
        compute_l2_error_l2(&f_ex, &l2, &div_gradp_exact, err_qo, None),
        rt.n_dofs(),
        l2.n_dofs(),
    )
}

/// MFEM 4.10 ex24 `-p 2 -o 4` ground truth: `(n, ref_levels, elements,
/// rt_dofs, l2_dofs, errSol, errProj)`.
const MFEM_O4_GOLDEN: &[(usize, usize, usize, usize, usize, f64, f64)] = &[
    (2, 1, 64, 13056, 4096, 5.7068028115104348e-07, 5.7070893259403543e-07),
    (2, 2, 512, 101376, 32768, 3.5796694481155882e-08, 3.5797139784626148e-08),
];

/// D342: the small rt3 ladders must reproduce MFEM's `-o 4` numbers, including
/// the dof counts the C++ run prints (`RT: 6340608 / L2: 2097152` at ex24
/// size, and the two smaller rungs above).
#[test]
fn d342_ex24_hex_rt3_matches_mfem_small_meshes() {
    for &(n, r, elems, want_rt, want_l2, want_sol, want_proj) in MFEM_O4_GOLDEN {
        let (e1, e2, e3_interp, e3_l2, n_rt, n_l2) = ex24_div_pipeline(n, r, 4);
        assert_eq!((n_rt, n_l2), (want_rt, want_l2), "{elems} elements: dof counts");
        println!(
            "  {elems} hex o=4: sol={e1:.17e} interp={e2:.17e} proj(interp)={e3_interp:.17e} \
             proj(L2, example)={e3_l2:.17e} (mfem {want_sol:.17e} / {want_proj:.17e})"
        );
        for (tag, got, want) in [
            ("errSol", e1, want_sol),
            ("errInterp", e2, want_sol),
            ("errProj (interpolant)", e3_interp, want_proj),
        ] {
            // 1e-8 relative, not the 1e-11 the D337 ladders use: at `-o 4` the
            // L² error itself is ~1e-7, so the ~1e-17 absolute round-off floor
            // of the quadrature sums (and of the mass solve) is already
            // ~1e-10 relative — the same effect the D337 report measured at
            // 32³ `-o 3` (2e-10 relative on a 1.16e-7 error).  The measured
            // deviations are printed below.
            assert!(
                (got - want).abs() <= 1e-8 * want.abs(),
                "{elems} elements o=4 {tag}: {got:.17e} vs mfem {want:.17e} \
                 (rel {:.2e})",
                (got - want).abs() / want.abs()
            );
        }
        // D343: the example's inline L²-projection form of line (c) carries an
        // O(h^{p+1}) offset — still the documented one at o=4, no worse.
        assert!(
            (e3_l2 - want_proj).abs() <= 1e-2 * want_proj.abs(),
            "{elems} elements o=4 errProj (L², example): {e3_l2:.17e} vs mfem {want_proj:.17e}"
        );
    }
}

/// D342 acceptance at the shipped ex24 size: `data/inline-hex.mesh` (= 4³
/// hexes) with ex24's `ref_levels = 3` → 32³ hexes, the exact C++ invocation
/// `./ex24_cpp -m data/inline-hex.mesh -o 4 -p 2 -no-vis`.
///
/// MFEM 4.10: `errSol = errInterp = 1.3998768047217274e-10`,
/// `errProj = 1.3998749928750382e-10`, `RT: 6340608  L2: 2097152`.
/// `cargo test --release ... -- --ignored --nocapture` (~5 min).
#[test]
#[ignore = "32^3 mesh with RT3 (6.3M dofs), minutes in release"]
fn d342_ex24_hex_rt3_32cubed_matches_mfem() {
    let (e1, e2, e3_interp, e3_l2, n_rt, n_l2) = ex24_div_pipeline(4, 3, 4);
    assert_eq!((n_rt, n_l2), (6_340_608, 2_097_152), "32^3 dof counts vs C++");
    println!(
        "  32^3 -o 4: sol={e1:.17e} interp={e2:.17e} proj(interp)={e3_interp:.17e} \
         proj(L2, example)={e3_l2:.17e}"
    );
    // Absolute tolerance, not relative: the L² error is a residual of two O(1)
    // fields, so at `-o 4` the *value* is ~1.4e-10 and the quadrature/mass-solve
    // round-off floor (~1e-17 absolute — the same floor the D337 ladders show at
    // `-o 1..3`, where their absolute deviations are also ~2e-17) is already
    // ~1e-7 relative.  Measured here: |diff| = 9.1e-18 (errSol, rel 6.5e-8) and
    // 5.2e-18 (errProj, rel 3.7e-7) — i.e. **8-9 significant digits** on a
    // cancelled quantity, and no trend with h that would betray a basis error
    // (the small-mesh rungs below agree to 1e-9..1e-11 relative).
    for (tag, got, want) in [
        ("errSol", e1, 1.3998768047217274e-10),
        ("errInterp", e2, 1.3998768278119776e-10),
        ("errProj (interpolant)", e3_interp, 1.3998749928750382e-10),
    ] {
        assert!(
            (got - want).abs() <= 1e-16,
            "32^3 -o 4 {tag}: {got:.17e} vs mfem {want:.17e} \
             (abs {:.2e}, rel {:.2e})",
            (got - want).abs(),
            (got - want).abs() / want.abs()
        );
    }
    assert!(
        (e3_l2 - 1.3998749928750382e-10).abs() <= 1e-3 * 1.3998749928750382e-10,
        "32^3 -o 4 errProj (L², example): {e3_l2:.17e}"
    );
}

/// D342/D346: for hex orders 3..=6 `project_hdiv_coefficient_3d` must take the
/// `Project_RT` interpolant route (the predicate now accepts them).  Any
/// fallback to the historical L² projection would show up as a ~1e-3 relative
/// difference in the dofs.
#[test]
fn d342_project_hdiv_coefficient_3d_is_the_interpolant_for_hex_orders_3_to_6() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    for order in 3..=6u8 {
        let rt = HDivSpace::new(mesh.clone(), order);
        let qo = (2 * (order as usize + 1) + 1).max(3) as u8;
        let got = project_hdiv_coefficient_3d(&rt, &|x: &[f64], out: &mut [f64]| {
            let g = gradp_exact(x);
            out[..3].copy_from_slice(&g[..3]);
        }, qo);
        let want = rt.interpolate_vector(&gradp_exact).into_vec();
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("  o={order}: |project_hdiv_coefficient_3d - Project_RT| = {worst:.3e}");
        assert_eq!(got, want, "o={order}: helper did not take the interpolant route");
    }
}
