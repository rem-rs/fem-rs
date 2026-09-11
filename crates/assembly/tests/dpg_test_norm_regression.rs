//! DPG test-norm (`G`) coverage: the assembled **normal-equation** blocks of
//! the complex 3-D Maxwell weak form, pinned against an MFEM 4.10 harness.
//!
//! Motivation (D36): the four *cross* blocks of the adjoint graph norm
//! (`-i ω μ (F, ∇×δG)`, `-i ω ε (∇×F, δG)`, `i ω ε (∇×G, δF)`,
//! `i ω μ (G, ∇×δF)`) are the only assembly step of the DPG stack that had no
//! regression coverage at all — every existing test invoked the weak form in a
//! way that cancels them (`A x = BᵀG⁻¹B x = BᵀG⁻¹f` holds for *any* `G`, and
//! the per-element "identity" tests use manufactured tuples that satisfy the
//! element equations directly).  A silently wrong `G` therefore stayed
//! invisible.
//!
//! Reference values: MFEM 4.10 harness `$HOME/work/mx3/cdump.cpp`, which
//! dumps `ComplexDPGWeakForm::BlockMat_r()/BlockMat_i()` — the assembled
//! normal equations `A = BᵀG⁻¹B` (i.e. `G` **and** `B` contracted together,
//! which is the basis-independent form of the same data).  Verified equal to
//! this crate's `block_mat_r()/block_mat_i()` entry by entry on the one-hex
//! meshes (`max|Δ| = 1.9e-15` at `-o 1`, `6.9e-15` at `-o 2`, all 900 /
//! 20734 nonzeros) and, on the `2×2×2` hex mesh at `-o 2`, up to the
//! mesh-numbering permutation (identical per-block Frobenius norms and
//! traces, `|Δ| ≤ 1e-13`).
//!
//! The pinned quantities below are permutation- and (trial-)basis-invariant
//! scalars of each block — Frobenius norm and trace — plus a handful of
//! individual entries for the one-hex meshes (whose dof numbering provably
//! coincides with MFEM's: one element, MFEM's entity-traversal numbering).
//!
//! Tolerances: `1e-12` relative, i.e. far tighter than any of the defects this
//! test is meant to catch (the smallest pinned cross-block norm is `8e-4`,
//! and quadrature-order mistakes move these numbers by percent-level).

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl3dPairingIntegrator, DpgCurlCurlIntegrator, DpgMixedVectorCurlIntegrator,
    DpgMixedVectorWeakCurlIntegrator, DpgTangentTraceIntegrator3D, DpgTVectorFEMassIntegrator,
    DpgVectorFEMassIntegrator,
};
use fem_mesh::Mesh;

const MU: f64 = 1.0;
const EPS: f64 = 1.0;
const OMEGA: f64 = 2.0 * std::f64::consts::PI;

/// 1:1 with `miniapps/dpg/dpg_maxwell_3d.rs` (`solve_level`): trial blocks
/// `E, H ∈ (L2(p−1))³`, `Ê, Ĥ ∈ ND_Trace(p)`, test blocks
/// `F, G ∈ ND(order+δ)`, adjoint graph norm, `StoreMatrices` omitted (the
/// normal-equation blocks are what this test pins).
fn build(mesh: &Mesh<3>, p: u8, delta_order: u8) -> ComplexDPGWeakForm<Mesh<3>> {
    let test_order = p + delta_order;
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());
    a.set_quad_order((2 * test_order).min(10));
    let es = a.add_trial_vector_space(p - 1, 3);
    let hs = a.add_trial_vector_space(p - 1, 3);
    let hate = a.add_trial_trace_space_nd(p);
    let hath = a.add_trial_trace_space_nd(p);
    let f = a.add_test_space(VolKind::HCurl, test_order);
    let g = a.add_test_space(VolKind::HCurl, test_order);

    a.add_trial_integrator(
        Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })),
        None,
        es,
        f,
    );
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: -EPS * OMEGA })),
        es,
        g,
    );
    a.add_trial_integrator(
        Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })),
        None,
        hs,
        g,
    );
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: MU * OMEGA })),
        hs,
        f,
    );
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);

    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: MU * MU * OMEGA * OMEGA })),
        None,
        f,
        f,
    );
    // The four graph-norm cross blocks (D36): their `G` entries are the only
    // ones whose (contracted) effect is not symmetric-invariant.
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -MU * OMEGA })),
        g,
        f,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorCurlIntegrator { q: -EPS * OMEGA })),
        g,
        f,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorCurlIntegrator { q: EPS * OMEGA })),
        f,
        g,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: MU * OMEGA })),
        f,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator {
            q: EPS * EPS * OMEGA * OMEGA,
        })),
        None,
        g,
        g,
    );
    a.assemble();
    a
}

type Inv = (usize, usize, f64, f64, f64, f64);

/// `(bi, bj, ‖·‖_F(real), tr(real), ‖·‖_F(imag), tr(imag))` for every block of
/// the assembled normal-equation matrices with at least one entry.
fn block_invariants(a: &ComplexDPGWeakForm<Mesh<3>>) -> Vec<Inv> {
    let offs = a.trial_offsets();
    let nb = offs.len() - 1;
    let blk = |i: usize| -> usize {
        (0..nb).find(|&k| i >= offs[k] && i < offs[k + 1]).unwrap_or(usize::MAX)
    };
    // key = (bi, bj), value = [frob_r, trace_r, frob_i, trace_i]
    let mut acc: std::collections::BTreeMap<(usize, usize), [f64; 4]> =
        std::collections::BTreeMap::new();
    for (part, m) in [(0usize, a.block_mat_r()), (1usize, a.block_mat_i())] {
        for i in 0..m.nrows {
            for p in m.row_ptr[i]..m.row_ptr[i + 1] {
                let j = m.col_idx[p] as usize;
                let v = m.values[p];
                let key = (blk(i), blk(j));
                let e = acc.entry(key).or_insert([0.0; 4]);
                e[2 * part] += v * v;
                if i == j {
                    e[2 * part + 1] += v;
                }
            }
        }
    }
    acc.into_iter()
        .map(|((bi, bj), e)| (bi, bj, e[0].sqrt(), e[1], e[2].sqrt(), e[3]))
        .collect()
}

/// Compare against a reference table of `(bi, bj, frob_r, trace_r, frob_i,
/// trace_i)`.  Every listed block must be present with matching invariants;
/// any *additional* assembled block must be (numerically) empty — the
/// reference may omit blocks whose whole block is exactly zero there.
fn check_invariants(got: &[Inv], want: &[Inv], tag: &str) {
    let find = |bi: usize, bj: usize| got.iter().find(|g| (g.0, g.1) == (bi, bj));
    for w in want {
        let g = find(w.0, w.1).unwrap_or_else(|| {
            panic!("{tag}: block ({},{}) missing from the assembled matrix", w.0, w.1)
        });
        for (k, (gv, wv)) in [
            ("frob_r", (g.2, w.2)),
            ("trace_r", (g.3, w.3)),
            ("frob_i", (g.4, w.4)),
            ("trace_i", (g.5, w.5)),
        ] {
            let tol = 1e-12 * wv.abs().max(1.0);
            assert!(
                (gv - wv).abs() <= tol,
                "{tag}: block ({},{}) {k}: got {gv:.17e}, MFEM harness {wv:.17e} (|Δ| = {:.3e})",
                w.0,
                w.1,
                (gv - wv).abs()
            );
        }
    }
    for g in got {
        if find(g.0, g.1).is_none() {
            let m = g.2.max(g.3.abs()).max(g.4).max(g.5.abs());
            assert!(
                m < 1e-10,
                "{tag}: unexpected non-empty assembled block ({},{}) (max invariant {m:.3e})",
                g.0,
                g.1
            );
        }
    }
}

/// Real-part entry of the assembled normal-equation matrix.
fn entry_r(a: &ComplexDPGWeakForm<Mesh<3>>, i: usize, j: usize) -> f64 {
    let m = a.block_mat_r();
    (m.row_ptr[i]..m.row_ptr[i + 1])
        .find(|&p| m.col_idx[p] as usize == j)
        .map(|p| m.values[p])
        .unwrap_or(0.0)
}

/// Imaginary-part entry of the assembled normal-equation matrix.
fn entry_i(a: &ComplexDPGWeakForm<Mesh<3>>, i: usize, j: usize) -> f64 {
    let m = a.block_mat_i();
    (m.row_ptr[i]..m.row_ptr[i + 1])
        .find(|&p| m.col_idx[p] as usize == j)
        .map(|p| m.values[p])
        .unwrap_or(0.0)
}

#[test]
fn one_hex_o1_normal_equations_match_mfem_harness() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let a = build(&mesh, 1, 1);
    assert_eq!(a.trial_block_sizes(), vec![3, 3, 12, 12]);

    // MFEM harness `cdump 1 1 1 1.0` (MakeCartesian3D(1,1,1), ω = 2π, μ = ε = 1).
    let want: Vec<Inv> = vec![
        (0, 0, 1.69100236381615, 2.92890200984865, 8.45399081805865e-17, 0.0),
        (0, 1, 0.0, 0.0, 8.45399081805865e-17, 0.0),
        (0, 2, 0.0352373796685099, 0.0, 0.0, 0.0),
        (0, 3, 0.0, 0.0, 0.382261240090763, 0.0),
        (1, 0, 0.0, 0.0, 8.45399081805865e-17, 0.0),
        (1, 1, 1.69100236381615, 2.92890200984865, 0.0, 0.0),
        (1, 2, 0.0, 0.0, 0.382261240090763, 0.0),
        (1, 3, 0.0352373796685101, 0.0, 0.0, 0.0),
        (2, 0, 0.0352373796685099, 0.0, 0.0, 0.0),
        (2, 1, 0.0, 0.0, 0.382261240090763, 0.0),
        (2, 2, 0.826828210287329, 2.35249865723545, 0.0, 0.0),
        (2, 3, 0.0, 0.0, 0.0585004631635964, 0.0),
        (3, 0, 0.0, 0.0, 0.382261240090763, 0.0),
        (3, 1, 0.0352373796685101, 0.0, 0.0, 0.0),
        (3, 2, 0.0, 0.0, 0.0585004631635964, 0.0),
        (3, 3, 0.826828210287329, 2.35249865723545, 0.0, 0.0),
    ];
    check_invariants(&block_invariants(&a), &want, "1-hex o1");

    // Individual entries (MFEM harness, real part, global dof indices).
    for (i, j, want_r) in [
        (1, 1, 0.97630066994954967),
        (6, 6, 0.19604155476962093),
        (2, 16, -0.010172155318575848),
        (16, 2, -0.010172155318575848),
        (10, 10, 0.19604155476962085),
        (25, 25, 0.19604155476962071),
    ] {
        let got = entry_r(&a, i, j);
        assert!(
            (got - want_r).abs() <= 1e-12 * want_r.abs().max(1.0),
            "1-hex o1 A_r[{i},{j}] = {got:.17e}, MFEM harness {want_r:.17e}"
        );
    }
    // Imaginary part, including the two graph-norm cross blocks
    // `E–Ĥ` (block (0,3)) and `Ê–Ĥ` (block (2,3)).
    for (i, j, want_i) in [
        (1, 18, -0.078028748888825245),
        (18, 1, 0.078028748888825245),
        (27, 12, -0.0084438145388050882),
        (12, 27, 0.0084438145388050882),
        (4, 16, 0.078028748888825245),
        (2, 3, -5.3061651974104637e-17),
    ] {
        let got = entry_i(&a, i, j);
        assert!(
            (got - want_i).abs() <= 1e-12 * want_i.abs().max(1.0),
            "1-hex o1 A_i[{i},{j}] = {got:.17e}, MFEM harness {want_i:.17e}"
        );
    }
}

#[test]
fn one_hex_o2_normal_equations_match_mfem_harness() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let a = build(&mesh, 2, 1);
    assert_eq!(a.trial_block_sizes(), vec![24, 24, 48, 48]);

    // MFEM harness `cdump 1 2 1 1.0`.
    let want: Vec<Inv> = vec![
        (0, 0, 0.597681942541029, 2.92798360332087, 0.0, 0.0),
        (0, 1, 0.0, 0.0, 0.00862080532989227, 0.0),
        (0, 2, 0.15929551313477, 0.0, 0.0, 0.0),
        (0, 3, 0.0, 0.0, 0.251814373164005, 0.0),
        (1, 0, 0.0, 0.0, 0.00862080532989227, 0.0),
        (1, 1, 0.597681942541029, 2.92798360332087, 0.0, 0.0),
        (1, 2, 0.0, 0.0, 0.251814373164005, 0.0),
        (1, 3, 0.15929551313477, 0.0, 0.0, 0.0),
        (2, 0, 0.15929551313477, 0.0, 0.0, 0.0),
        (2, 1, 0.0, 0.0, 0.251814373164005, 0.0),
        (2, 2, 1.94566086796437, 10.0199544526696, 0.0, 0.0),
        (2, 3, 0.0, 0.0, 0.864642440927566, 0.0),
        (3, 0, 0.0, 0.0, 0.251814373164005, 0.0),
        (3, 1, 0.15929551313477, 0.0, 0.0, 0.0),
        (3, 2, 0.0, 0.0, 0.864642440927566, 0.0),
        (3, 3, 1.94566086796437, 10.0199544526696, 0.0, 0.0),
    ];
    check_invariants(&block_invariants(&a), &want, "1-hex o2");

    for (i, j, want_r) in [
        (137, 137, 0.32252863521974834),
        (48, 0, -0.014208757786427138),
        (0, 48, -0.014208757786427138),
        (50, 9, -0.014208757786427152),
        (14, 5, 0.00037006956491150537),
        (10, 10, 0.12199931680503623),
        (25, 25, 0.12199931680503627),
        (1, 1, 0.12199931680503615),
        (6, 6, 0.1219993168050362),
    ] {
        let got = entry_r(&a, i, j);
        assert!(
            (got - want_r).abs() <= 1e-12 * want_r.abs().max(1.0),
            "1-hex o2 A_r[{i},{j}] = {got:.17e}, MFEM harness {want_r:.17e}"
        );
    }
    // Graph-norm cross-block entries (imaginary part).
    for (i, j, want_i) in [
        (52, 138, -0.044708323468767699),
        (8, 28, 0.0012385797987062332),
        (74, 26, -0.028745941226694043),
        (122, 2, 0.028745941226693862),
    ] {
        let got = entry_i(&a, i, j);
        assert!(
            (got - want_i).abs() <= 1e-12 * want_i.abs().max(1.0),
            "1-hex o2 A_i[{i},{j}] = {got:.17e}, MFEM harness {want_i:.17e}"
        );
    }
}

/// `2×2×2` hex mesh at `-o 2`: pins the *shared-face* trace couplings and the
/// whole `G` contraction on a multi-element mesh.  The mesh numbering of
/// `Mesh::unit_cube_hex(2)` differs from MFEM's `MakeCartesian3D(2,2,2)`
/// (the harness comparison is then only defined up to that dof permutation,
/// which is exactly what the invariants below are invariant under); both
/// sides also share the same block sparsity pattern.
#[test]
fn eight_hex_o2_normal_equations_match_mfem_harness() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let a = build(&mesh, 2, 1);
    assert_eq!(a.trial_block_sizes(), vec![192, 192, 252, 252]);

    // MFEM harness `cdump 2 2 1 1.0`.
    let want: Vec<Inv> = vec![
        (0, 0, 0.214608150357157, 2.9736658659946, 0.0, 0.0),
        (0, 1, 0.0, 0.0, 0.000819655375185175, 0.0),
        (0, 2, 0.200300688455315, 0.0, 0.0, 0.0),
        (0, 3, 0.0, 0.0, 0.13359579531959, 0.0),
        (1, 0, 0.0, 0.0, 0.000819655375185176, 0.0),
        (1, 1, 0.214608150357157, 2.9736658659946, 0.0, 0.0),
        (1, 2, 0.0, 0.0, 0.13359579531959, 0.0),
        (1, 3, 0.200300688455314, 0.0, 0.0, 0.0),
        (2, 0, 0.200300688455314, 0.0, 0.0, 0.0),
        (2, 1, 0.0, 0.0, 0.133595795319589, 0.0),
        (2, 2, 24.8050834688067, 290.662534128532, 0.0, 0.0),
        (2, 3, 0.0, 0.0, 10.847612992251, 0.0),
        (3, 0, 0.0, 0.0, 0.133595795319589, 0.0),
        (3, 1, 0.200300688455314, 0.0, 0.0, 0.0),
        (3, 2, 0.0, 0.0, 10.847612992251, 0.0),
        (3, 3, 24.8050834688067, 290.662534128532, 0.0, 0.0),
    ];
    check_invariants(&block_invariants(&a), &want, "8-hex o2");
}
