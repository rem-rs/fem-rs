//! D1221 regression pins: the 3-D acoustics DPG **face** quadrature is sized
//! per trace integrator the MFEM way (`test_fe.GetOrder() +
//! trial_face_fe.GetOrder()`, bilininteg.cpp TraceIntegrator :4429 /
//! NormalTraceIntegrator :4480 / TangentTraceIntegrator :4582), not with one
//! fixed order.
//!
//! Defect (round 115): the serial complex weak form defaulted to a single
//! face rule of order 4 (exact to degree 5 on a quad).  At `-o 3 -do 1` the
//! NormalTrace pair (hatp 3 × RT 3) and the TraceIntegrator pair (RT-trace 2 ×
//! H1 4) both need degree 6 — the under-integrated B blocks corrupted the
//! normal equations (L2 7.070e-3 vs C++ 5.793e-03 on the 4×4×4 hex cube; on
//! the tet mesh `-o 3` was off by 17x: 5.335e-1 vs 3.079e-02).  Every hex
//! order row with `p_face + test_order ≤ 5` had always matched, which is why
//! `-o 2` (4), `-o 3 -do 0` (5), `-o 1 -do 3` (5) and `-o 2 -do 2` (5) never
//! showed it.
//!
//! Reference values: MFEM 4.10 harness `~/work/d115b/adump.cpp` — the
//! round-101 Maxwell harness (`~/work/mx3/cdump.cpp`) pattern applied to
//! `miniapps/dpg/acoustics.cpp` (`prob = 0`, no BCs): `ComplexDPGWeakForm::
//! BlockMat_r()/BlockMat_i()` on `MakeCartesian3D(1,1,1)` = fem-rs
//! `unit_cube_hex(1)` (dof numbering provably coincides on one element).
//! With MFEM's stock Gauss-Legendre RT-trace gauge, every block NOT involving
//! the û trial space (index 3) matches fem-rs at ≤ 2.4e-14 relative; the û
//! blocks differ only by fem-rs's documented equispaced RT-trace face gauge
//! (identical span, solution-invariant — verified: re-running the harness
//! with `BasisType::ClosedUniform` makes ALL 16 blocks match fem-rs at
//! ≤ 1.4e-14, all 47524 nonzeros at identical positions).
//!
//! The pinned quantities are the permutation- and gauge-invariant per-block
//! Frobenius norms and traces; tolerances `1e-12` relative (quadrature-order
//! mistakes move these by percent-level).

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDivDivIntegrator, DpgDiffusionIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedVectorGradientIntegrator,
    DpgMixedVectorWeakDivergenceIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTVectorFEMassIntegrator, DpgTraceIntegrator, DpgVectorFEDivergenceIntegrator,
    DpgVectorFEMassIntegrator,
};
use fem_mesh::Mesh;

const DIM: usize = 3;
const OMEGA: f64 = 2.0 * std::f64::consts::PI;

type Inv = (usize, usize, f64, f64);

/// 1:1 with `miniapps/dpg/dpg_acoustics_3d.rs` `solve_level`: trial blocks
/// `p ∈ L2(o−1)`, `u ∈ L2(o−1)³`, `p̂ ∈ H1-trace(o)`, `û ∈ RT-trace(o−1)`;
/// test blocks `q ∈ H1(o+δ)`, `v ∈ RT(o+δ−1)`; adjoint graph norm.
fn build(mesh: &Mesh<3>, order: u8, delta_order: u8) -> ComplexDPGWeakForm<Mesh<3>> {
    let p = order;
    let test_order = order + delta_order;
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());

    let ps = a.add_trial_scalar_space(p - 1);
    let us = a.add_trial_vector_space(p - 1, DIM);
    let hatp = a.add_trial_trace_space_h1(p);
    let hatu = a.add_trial_trace_space(p - 1);
    let q = a.add_test_space(VolKind::Scalar, test_order);
    let v = a.add_test_space(VolKind::HDiv, test_order - 1);

    // i ω (p, q)
    a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: OMEGA })), ps, q);
    // -(u, ∇q)
    a.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us, q);
    // -(p, ∇·v)
    a.add_trial_integrator(
        Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })),
        None,
        ps,
        v,
    );
    // i ω (u, v)
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: OMEGA })),
        us,
        v,
    );
    // <p̂, v·n>
    a.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp, v);
    // <û, q>
    a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu, q);

    // Adjoint graph norm (test integrators)
    a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q, q);
    a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
    a.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v, v);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v, v);
    // -iω (∇q, δv) → G[v, q]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorGradientIntegrator {
            q: vec![
                vec![-OMEGA, 0.0, 0.0],
                vec![0.0, -OMEGA, 0.0],
                vec![0.0, 0.0, -OMEGA],
            ],
        })),
        v,
        q,
    );
    // iω (v, ∇δq) → G[q, v]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
            q: vec![
                vec![-OMEGA, 0.0, 0.0],
                vec![0.0, -OMEGA, 0.0],
                vec![0.0, 0.0, -OMEGA],
            ],
        })),
        q,
        v,
    );
    // ω² (v, δv), ω² (q, δq)
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: OMEGA * OMEGA })),
        None,
        v,
        v,
    );
    a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: OMEGA * OMEGA })), None, q, q);
    // -iω (∇·v, δq) → G[q, v]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -OMEGA })),
        q,
        v,
    );
    // iω (q, ∇·v) → G[v, q]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -OMEGA })),
        v,
        q,
    );

    a.store_matrices(true);
    a.assemble();
    a
}

/// Per-block `(frobenius, trace)` of the real part, keyed `(bi, bj)`.
/// The "trace" follows the harness definition: the sum of entries on the
/// block's local (rectangular) diagonal `row − off[bi] == col − off[bj]`
/// (deterministic, gauge-invariant; for diagonal blocks it is the trace).
fn block_invariants(a: &ComplexDPGWeakForm<Mesh<3>>) -> Vec<Inv> {
    let m = a.block_mat_r();
    let offs = a.trial_offsets();
    let blk = |g: usize| offs.iter().rposition(|&o| o <= g).unwrap();
    let mut acc = std::collections::HashMap::<(usize, usize), (f64, f64)>::new();
    for i in 0..m.nrows {
        for p in m.row_ptr[i]..m.row_ptr[i + 1] {
            let j = m.col_idx[p] as usize;
            let bi = blk(i);
            let bj = blk(j);
            let e = acc.entry((bi, bj)).or_insert((0.0, 0.0));
            e.0 += m.values[p] * m.values[p];
            if i - offs[bi] == j - offs[bj] {
                e.1 += m.values[p];
            }
        }
    }
    let mut v: Vec<Inv> = acc
        .into_iter()
        .map(|((bi, bj), (f, t))| (bi, bj, f.sqrt(), t))
        .collect();
    v.sort_by_key(|x| (x.0, x.1));
    v
}

fn check_invariants(got: &[Inv], want: &[Inv], tag: &str) {
    let find = |bi: usize, bj: usize| got.iter().find(|g| (g.0, g.1) == (bi, bj));
    for &(bi, bj, wf, wt) in want {
        let g = find(bi, bj)
            .unwrap_or_else(|| panic!("{tag}: block ({bi},{bj}) missing from the assembled matrix"));
        for (k, (gv, wv)) in [("frob", (g.2, wf)), ("trace", (g.3, wt))] {
            assert!(
                (gv - wv).abs() <= 1e-12 * wv.abs().max(1.0),
                "{tag}: block ({bi},{bj}) {k}: got {gv:.17e}, MFEM harness {wv:.17e} (|Δ| = {:.3e})",
                (gv - wv).abs()
            );
        }
    }
}

/// The D1221 defect case: `-o 3 -do 1` on one hex.  All six non-û blocks are
/// pinned against the stock-MFEM harness (the û gauge differs by design; the
/// û blocks are pinned against the gauge-matched harness in the second test).
#[test]
fn one_hex_o3do1_non_hatu_blocks_match_mfem_harness() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let a = build(&mesh, 3, 1);
    assert_eq!(a.trial_block_sizes(), vec![27, 81, 56, 54]);

    // MFEM harness `adump 3 1` (stock Gauss-Legendre RT-trace gauge): every
    // block that does not touch trial space 3 (û) matches fem-rs.
    let want: Vec<Inv> = vec![
        // (bi, bj, frobenius_r, trace_r)
        (0, 0, 2.01698638672247521e-01, 9.73843672357821877e-01),
        (0, 2, 1.09761395622245531e-01, 5.06615153349623815e-03),
        (1, 1, 3.52378349998299256e-01, 2.93053046974952647e+00),
        (2, 0, 1.09761395622245531e-01, 5.06615153349623815e-03),
        (2, 2, 1.71940355652477894e+00, 8.98810101081628154e+00),
    ];
    check_invariants(&block_invariants(&a), &want, "1-hex o3do1 (non-û, stock gauge)");

    // Sensitivity: the pre-D1221 behaviour — one fixed order-4 face rule —
    // must move the NormalTrace-only block (2,2) well past the pin tolerance
    // (the face integrand needs degree 6; order 4 integrates exactly 5).
    let mut broken = build(&mesh, 3, 1);
    broken.set_face_quad_order(4);
    broken.assemble();
    let got = block_invariants(&broken);
    let f22 = got.iter().find(|g| (g.0, g.1) == (2, 2)).unwrap().2;
    let w22 = want.iter().find(|w| (w.0, w.1) == (2, 2)).unwrap().2;
    assert!(
        (f22 - w22).abs() > 1e-3 * w22,
        "order-4 face rule should reproduce the D1221 under-integration on (2,2): \
         got {f22:.17e}, harness {w22:.17e}"
    );
}

/// Same case with the û blocks included: fem-rs's equispaced RT-trace face
/// gauge is the harness rebuilt with `BasisType::ClosedUniform` — ALL 16
/// blocks then agree (verified entry-by-entry, 47524/47524 nonzeros), so the
/// full real-part invariant table is pinned here.
#[test]
fn one_hex_o3do1_full_table_matches_equispaced_gauge_harness() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let a = build(&mesh, 3, 1);

    // MFEM harness `adump_u 3 1` (`RT_Trace_FECollection(o-1, 3, INTEGRAL,
    // BasisType::ClosedUniform)`).
    let want: Vec<Inv> = vec![
        (0, 0, 2.01698638672247521e-01, 9.73843672357821877e-01),
        (0, 2, 1.09761395622245531e-01, 5.06615153349623815e-03),
        (1, 1, 3.52378349998299256e-01, 2.93053046974952647e+00),
        (1, 3, 1.19248425143684861e-01, 8.03935976839068207e-03),
        (2, 0, 1.09761395622245531e-01, 5.06615153349623815e-03),
        (2, 2, 1.71940355652477894e+00, 8.98810101081628154e+00),
        (3, 1, 1.19248425143684861e-01, 8.03935976839068207e-03),
        (3, 3, 1.90064553136531433e+00, 8.66899585072222578e+00),
    ];
    check_invariants(&block_invariants(&a), &want, "1-hex o3do1 (equispaced gauge, real)");

    // Imaginary part — includes the two TraceIntegrator channels (0,3)/(3,0)
    // and the graph-norm cross (2,3)/(3,2).  Same block-local trace as above.
    let m = a.block_mat_i();
    let offs = a.trial_offsets();
    let blk = |g: usize| offs.iter().rposition(|&o| o <= g).unwrap();
    let mut acc = std::collections::HashMap::<(usize, usize), (f64, f64)>::new();
    for i in 0..m.nrows {
        for p in m.row_ptr[i]..m.row_ptr[i + 1] {
            let j = m.col_idx[p] as usize;
            let bi = blk(i);
            let bj = blk(j);
            let e = acc.entry((bi, bj)).or_insert((0.0, 0.0));
            e.0 += m.values[p] * m.values[p];
            if i - offs[bi] == j - offs[bj] {
                e.1 += m.values[p];
            }
        }
    }
    let mut got: Vec<(usize, usize, f64, f64)> = acc
        .into_iter()
        .map(|((bi, bj), (f, t))| (bi, bj, f.sqrt(), t))
        .collect();
    got.sort_by_key(|x| (x.0, x.1));
    let want_i: Vec<(usize, usize, f64, f64)> = vec![
        (0, 1, 5.29272143274295368e-03, -5.08836133714217966e-17),
        (0, 3, 1.02892743460783850e-01, -7.37104938688215444e-02),
        (1, 0, 5.29272143274295368e-03, 5.08836133714217966e-17),
        (1, 2, 9.88595649108686830e-02, 1.05283061167926462e-02),
        (2, 1, 9.88595649108686830e-02, -1.05283061167926462e-02),
        (2, 3, 1.10260177576916618e+00, 9.96604443815494456e-02),
        (3, 0, 1.02892743460783850e-01, 7.37104938688215444e-02),
        (3, 2, 1.10260177576916618e+00, -9.96604443815494456e-02),
    ];
    for &(bi, bj, wf, wt) in &want_i {
        let g = got
            .iter()
            .find(|g| (g.0, g.1) == (bi, bj))
            .unwrap_or_else(|| panic!("o3do1 imag: block ({bi},{bj}) missing"));
        for (k, (gv, wv)) in [("frob", (g.2, wf)), ("trace", (g.3, wt))] {
            assert!(
                (gv - wv).abs() <= 1e-12 * wv.abs().max(1.0),
                "1-hex o3do1 imag block ({bi},{bj}) {k}: got {gv:.17e}, harness {wv:.17e}"
            );
        }
    }
}
