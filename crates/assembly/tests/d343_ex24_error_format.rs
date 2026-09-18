//! D343: ex24 `-p 2` — the printed error format and the L²-error quadrature
//! order.
//!
//! MFEM 4.10 `examples/ex24.cpp` prints the three error lines with
//! `cout << err` at the **default** stream precision (6 significant digits,
//! exactly C's `%g`); the `precision(8)` calls at `ex24.cpp:355/358/367` apply
//! only to `mesh_ofs`/`sol_ofs`/`sol_sock`.  fem-rs used `{:.8}` (fixed 8
//! decimals), which is both a different format and blind at `-o 4`
//! (`0.00000000` where C++ prints `1.39988e-10`).
//!
//! Separately, ex24's `-p 2` error norms are evaluated on
//! `order_quad = max(2, 2*order+1)` (`ex24.cpp:334`), while the example used
//! `(2*order+6).max(7)`.  The L² error of a *piecewise-constant* L² field
//! against the non-polynomial `div v` is quadrature-sensitive at the 1e-5
//! relative level, so that divergence is visible in the printed digits — it is
//! the real cause of D343's "o=1 last digit" gap, not round-off.
//!
//! Ground truth (`tmp/d343/cpp_o1234_raw.txt`, the round-45 prebuilt MFEM 4.10
//! ex24 binary; `-m $HOME/mfem410_ser/data/inline-hex.mesh -o 1 -p 2 -no-vis`):
//!
//! ```text
//!   Solution ... = 0.0108996
//!   Divergence interpolant ... = 0.0108996
//!   Projection ... = 0.0108997
//! ```
//!
//! and at 17 digits (`tmp/d342/ex24_div_prec.cpp -n 4 -r 3 -o 1 -p 2`):
//! `errSol == errInterp == 1.08996341153926868e-02`,
//! `errProj (ProjectCoefficient interpolant) = 1.0899703375005593e-02`.
//!
//! Run: `cargo test --release -p fem-assembly --test d343_ex24_error_format -- --nocapture`

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::postproc::grid_function::{compute_l2_error_l2, project_hdiv_coefficient_3d};
use fem_assembly::standard::{DomainSourceIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_mesh::Mesh;
use fem_solver::{fmt_g, solve_pcg_jacobi, SolverConfig};
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

/// ex24 `-p 2` on `n³` hexes refined `ref_levels` times, returning the four
/// error norms (see `d337_ex24_div_ladder.rs::ex24_div_pipeline`) **plus** the
/// two legs re-evaluated with the example's own quadrature order.
fn pipeline(n: usize, ref_levels: usize, cli_order: u8) -> Ex24Div {
    let mut mesh = Mesh::<3>::unit_cube_hex(n);
    for _ in 0..ref_levels {
        mesh = fem_mesh::refine_uniform_3d(&mesh);
    }
    let qo = (2 * cli_order + 1).max(3) as u8;
    let rt_order = cli_order - 1;
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    let l2 = L2Space::new(mesh.clone(), rt_order);
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        ..SolverConfig::default()
    };

    let v = project_hdiv_coefficient_3d(
        &rt,
        &|x: &[f64], out: &mut [f64]| {
            let g = gradp_exact(x);
            out[..3].copy_from_slice(&g[..3]);
        },
        qo,
    );

    let d = assemble_hdiv_l2_mixed(&l2, &rt, &[&HDivL2DivIntegrator], qo);
    let mass = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], qo);
    let mut rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut rhs);

    let mut f_sol = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs, &mut f_sol, &cfg).expect("PCG sol");
    let mut f_interp = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs, &mut f_interp, &cfg).expect("PCG interp");

    // Line (c) as MFEM computes it: `exact_proj.ProjectCoefficient(divgradp_coef)`
    // — on a nodal L² space that is the nodal interpolant.
    let f_ex_interp = l2.interpolate(&div_gradp_exact).into_vec();
    // Line (c) as the shipped example computes it: an inline L² mass solve.
    let rhs_ex =
        Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(div_gradp_exact)], qo);
    let mut f_ex = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut f_ex, &cfg).expect("PCG ex");

    let cpp_qo = (2 * cli_order + 1).max(2) as u8; // ex24.cpp:334 `order_quad`
    let ex_qo = (2 * cli_order + 6).max(7) as u8; // the example's quadrature
    let err = |f: &[f64], q: u8| compute_l2_error_l2(f, &l2, &div_gradp_exact, q, None);

    Ex24Div {
        n_dofs_rt: rt.n_dofs(),
        n_dofs_l2: l2.n_dofs(),
        cpp_qo,
        ex_qo,
        cpp: [err(&f_sol, cpp_qo), err(&f_interp, cpp_qo), err(&f_ex_interp, cpp_qo)],
        example: [err(&f_sol, ex_qo), err(&f_interp, ex_qo), err(&f_ex, ex_qo)],
        proj_l2_cpp_qo: err(&f_ex, cpp_qo),
    }
}

struct Ex24Div {
    n_dofs_rt: usize,
    n_dofs_l2: usize,
    cpp_qo: u8,
    ex_qo: u8,
    /// C++ semantics: RT sol, RT interpolant, `ProjectCoefficient` interpolant —
    /// all on `max(2, 2*order+1)`.
    cpp: [f64; 3],
    /// The shipped example: RT sol, RT interpolant, inline L² mass solve — all on
    /// `(2*order+6).max(7)`.
    example: [f64; 3],
    proj_l2_cpp_qo: f64,
}

/// The `-o 1` end-to-end case the debt is about: exactly the example's mesh
/// (`data/inline-hex.mesh` = 16³ hexes, `ref_levels = 1` -> 32³ = 32768 hexes,
/// `tmp/d342` "`-n 4 -r 3` = 32768"), at C++'s quadrature order.
#[test]
fn d343_ex24_div_o1_matches_mfem_printed_text() {
    let r = pipeline(16, 1, 1);
    assert_eq!((r.n_dofs_rt, r.n_dofs_l2), (101376, 32768));
    println!(
        "  o=1 32768 hexes  q(cpp)={} q(example)={}\n  \
         cpp    : sol={:.17e} interp={:.17e} proj_interp={:.17e}\n  \
         example: sol={:.17e} interp={:.17e} proj_L2={:.17e}\n  \
         printed with fmt_g: cpp {} / {} / {}   example {} / {} / {}",
        r.cpp_qo,
        r.ex_qo,
        r.cpp[0],
        r.cpp[1],
        r.cpp[2],
        r.example[0],
        r.example[1],
        r.example[2],
        fmt_g(r.cpp[0]),
        fmt_g(r.cpp[1]),
        fmt_g(r.cpp[2]),
        fmt_g(r.example[0]),
        fmt_g(r.example[1]),
        fmt_g(r.example[2]),
    );

    // MFEM 4.10, 17 digits (`tmp/d342/ex24_div_prec.cpp -n 4 -r 3 -o 1 -p 2`).
    let mfem_sol = 1.0899634115392687e-02_f64;
    let mfem_proj = 1.0899703375005593e-02_f64;

    // Lines 1 and 2 at C++'s quadrature order reproduce MFEM.
    for (tag, got) in [("errSol", r.cpp[0]), ("errInterp", r.cpp[1])] {
        assert!(
            (got - mfem_sol).abs() <= 1e-11 * mfem_sol.abs(),
            "{tag} at order_quad={}: {got:.17e} vs mfem {mfem_sol:.17e}",
            r.cpp_qo
        );
    }
    // Line 3 is MFEM's `ProjectCoefficient` (nodal interpolant) form.
    assert!(
        (r.cpp[2] - mfem_proj).abs() <= 1e-11 * mfem_proj.abs(),
        "errProj(interpolant): {:.17e} vs mfem {mfem_proj:.17e}",
        r.cpp[2]
    );

    // The printed text at 6 significant digits (C++ stdout, `tmp/d343`).
    assert_eq!(fmt_g(r.cpp[0]), "0.0108996", "line (a) printed form");
    assert_eq!(fmt_g(r.cpp[1]), "0.0108996", "line (b) printed form");
    assert_eq!(fmt_g(r.cpp[2]), "0.0108997", "line (c) printed form");

    // The example's over-integration is *not* neutral: it moves lines (a)/(b)
    // by ~8e-6 absolute (~7.7e-6 relative).  This is the D343 mechanism.
    let shift = (r.example[0] - r.cpp[0]).abs() / r.cpp[0].abs();
    println!("  example vs C++ quadrature shift: {shift:.3e} relative");
    assert!(shift > 1e-7, "over-integration is expected to move the printed digit");
}

/// Line (c): MFEM's `ProjectCoefficient` into a nodal L² space is the nodal
/// interpolant, not an L² mass solve.  On the ex24-sized mesh the two differ by
/// ~2e-5 relative, which is exactly the `0.01089949` vs `0.0108997` gap D343
/// recorded.
#[test]
fn d343_ex24_line_c_operator_is_nodal_interpolation() {
    let r = pipeline(16, 1, 1);
    let mfem_proj = 1.0899703375005593e-02_f64;
    let rel_interp = (r.cpp[2] - mfem_proj).abs() / mfem_proj;
    let rel_l2 = (r.proj_l2_cpp_qo - mfem_proj).abs() / mfem_proj;
    println!(
        "  line (c) at order_quad={}: interpolant={:.17e} (rel {rel_interp:.3e})  \
         L² mass solve={:.17e} (rel {rel_l2:.3e})",
        r.cpp_qo, r.cpp[2], r.proj_l2_cpp_qo
    );
    assert!(rel_interp < 1e-10, "the interpolant is MFEM's operator");
    assert!(rel_l2 > 1e-6, "the L² mass solve is measurably different");
}
