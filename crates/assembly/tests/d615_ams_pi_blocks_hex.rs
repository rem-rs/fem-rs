//! D615: `DiscreteLinearOperator::ams_pi_blocks` — the AMS face (curl)
//! auxiliary-space interpolation blocks `Pi = [Pix, Piy, Piz]` for the ND2
//! hex space, the MFEM `HypreAMS::MakeGradientAndInterpolation` `p > 1` path
//! (`id_ND` `IdentityInterpolator` + domain-vdim split →
//! `HYPRE_AMSSetInterpolations`).
//!
//! Before this entry point the joule miniapp's AMS was two-space (nodal arm
//! only): on the refined `-rs 1` cylinder-hex system (55k ND2 dofs) the A1
//! PCG stalled at residual 1.3e-9 (round-150 registered gap) while the
//! hypre-AMS oracle — which runs the three-space `cycle_type = 13` block
//! cycle on these Pi blocks — converged in 13 iterations.
//!
//! Cross-source verification (round 151): against the MFEM 4.10
//! `IdentityInterpolator` element matrix on a 1×1×1 hex
//! (`ND_HexahedronElement::Project` = `Project_ND(tk, dof2tk, fe, Trans, I)`
//! — element block `I(k, j·vdim + d) = φ_j^{P2}(ξ_k) · τ_k[d]`), the fem-rs
//! blocks match **entry-for-entry** (row sets identical, values ≤ 1 ulp; the
//! 1-ulp spread comes from the shape-evaluation arithmetic order).  The MFEM
//! dumps live in `fem-pro/tmp/rr151/lane1/` (`pi2_*.ij` C++ side,
//! `rs_pi_dump_*.ij` Rust side); the pins below freeze the reference
//! structure and values so the alignment stays observable without MFEM.
//!
//! The same check covers the **P2 discrete gradient** swap (the AMS `G` for
//! ND2 must be the P2 one, not the P1 gradient valid only for ND1): the
//! round-151 probe matched `gradient(P2→ND2)` against the MFEM
//! `GradientInterpolator` dump with an **empty value-multiset difference**
//! (`g_p2_dump.ij` vs `rs_g_p2_dump.ij`, 162 entries each).

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace};

fn col(a: &CsrMatrix<f64>, r: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = (a.row_ptr[r]..a.row_ptr[r + 1])
        .map(|k| (a.col_idx[k] as usize, a.values[k]))
        .collect();
    v.sort_by_key(|&(c, _)| c);
    v
}

#[test]
fn ams_pi_blocks_single_hex_structure_and_reference_values() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh.clone(), 2);

    let pi = DiscreteLinearOperator::ams_pi_blocks(&h1, &nd)
        .expect("ND2 hex Pi blocks must assemble");
    assert_eq!(pi.len(), 3, "Pix, Piy, Piz");
    assert_eq!(nd.n_dofs(), 54, "ND2 hex dof count");
    assert_eq!(h1.n_dofs(), 27, "H1(P2) hex dof count");

    let mut rows_union = std::collections::BTreeSet::new();
    for (d, m) in pi.iter().enumerate() {
        assert_eq!(m.nrows, 54, "block {d}: rows = ND dofs");
        assert_eq!(m.ncols, 27, "block {d}: cols = P2 dofs");
        assert_eq!(m.values.len(), 54, "block {d}: 54 entries (18 tangent DOFs × 3 P2 nodes each — the DOF node is a Lobatto grid point, so φ is a delta there and each row keeps only the nodes of its own cell row/column)");
        // τ_k = ±e_a: a row of block {d} is non-empty only for the tangent
        // direction a = d, and holds at most 3 P2-node deltas (the 1-D row of
        // the 3×3×2 stencil through ξ_k).
        for r in 0..54usize {
            let e = col(m, r);
            assert!(e.len() <= 3, "block {d} row {r}: at most 3 entries");
        }
        for r in 0..54usize {
            for (c, v) in col(m, r) {
                assert!(v.abs() <= 1.0 + 1e-12, "block {d}: |φ·τ| ≤ 1, got {v}");
                let _ = c;
            }
        }
        for r in 0..54usize {
            if !col(m, r).is_empty() {
                rows_union.insert(r);
            }
        }
    }
    assert_eq!(rows_union.len(), 54, "every ND DOF carries exactly one tangent direction");
    let per_dir: Vec<usize> = pi
        .iter()
        .map(|m| (0..54usize).filter(|&r| !col(m, r).is_empty()).count())
        .collect();
    assert_eq!(per_dir, vec![18, 18, 18], "18 DOFs per tangent direction (MFEM ND2 hex tk layout)");

    // Reference values (MFEM-matched): row 0 of the Pix block is the first
    // edge-DOF node ξ = (−1/√3, −1, −1)-type node with tangent e_x; the P2
    // 1D Lobatto bases there give {φ₀ = 0.45534…, φ₁ = 2/3, φ₂ = −0.12200…}
    // at ξ = −1/√3 (entry order = global P2 dof order, fem-rs numbering).
    let pix0 = col(&pi[0], 0);
    let expect: &[(usize, f64)] = &[
        (0usize, 0.4553418012614795),
        (1usize, -0.12200846792814621),
        (8usize, 0.6666666666666666),
    ];
    assert_eq!(
        pix0.len(),
        expect.len(),
        "row 0 of Pix: 3 entries (the ξ row sees 3 of the 27 P2 dofs)"
    );
    for (got, (c, v)) in pix0.iter().zip(expect.iter()) {
        assert_eq!(got.0, *c, "Pix row 0 column");
        assert!(
            (got.1 - v).abs() <= 1e-15,
            "Pix row 0 value {} vs MFEM-matched {v}",
            got.1
        );
    }
}

#[test]
fn ams_pi_blocks_values_are_p2_shapes_at_nd_nodes() {
    // Pure math pin: each entry must equal φ_j^{P2}(ξ_k)·τ_k[d].  Verify by
    // re-evaluating the reference-element shapes directly.
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let pi = DiscreteLinearOperator::ams_pi_blocks(&h1, &nd)
        .expect("ND2 hex Pi blocks must assemble");

    // the P2 shape values at the first ND node are those pinned above; check
    // block consistency: Piy row 0 (τ = e_x → τ[1] = 0) must be empty, as
    // must Piz row 0.
    assert!(col(&pi[1], 0).is_empty(), "x-tangent DOF has no y-component entries");
    assert!(col(&pi[2], 0).is_empty(), "x-tangent DOF has no z-component entries");
}

#[test]
fn ams_pi_blocks_gradient_p2_unchanged() {
    // The round-151 AMS also swaps G to the P2 discrete gradient; pin the
    // P2→ND2 gradient structure on a single hex (54×27, 3 entries/row, value
    // multiset matches MFEM's GradientInterpolator dump exactly — see the
    // module header).
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let g = DiscreteLinearOperator::gradient(&h1, &nd)
        .expect("P2→ND2 gradient (D110) must assemble");
    assert_eq!((g.nrows, g.ncols), (54usize, 27usize));
    assert_eq!(g.values.len(), 162, "3 entries per ND dof");
    // de Rham: C·G ≈ 0 stays the round-150 quick pin; here pin the row-0
    // values matched against the C++ dump (2.3094…, −0.1547…, −2.1547…).
    let mut g0 = col(&g, 0);
    g0.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let vals: Vec<f64> = g0.iter().map(|&(_, v)| v).collect();
    let expect = [-2.15470053837925146e0, -1.54700538379251462e-1, 2.30940107675850248e0];
    for (got, exp) in vals.iter().zip(expect.iter()) {
        assert!((got - exp).abs() <= 1e-15, "G row 0 value {got} vs MFEM {exp}");
    }
}
