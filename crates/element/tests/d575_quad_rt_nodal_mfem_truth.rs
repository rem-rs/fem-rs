//! D575 — `mfem_quad_nodal_dofs` (the whole-table flip fix) pinned against
//! MFEM 4.10 ground truth.
//!
//! Truth source: the C++ probe `tmp/d575/d575_probe.cpp` (built against
//! `$HOME/mfem410_ser`, output archived as `tmp/d575/d575_probe.out`,
//! `QUAD_RT p=0..3` blocks): MFEM `RT_QuadrilateralElement(p)` dof nodes
//! (`FE::Nodes`, public `GetNodes()`) and per-dof reference normals, the
//! latter recovered with the public API by evaluating the nodal
//! (GaussLegendre) variant's `CalcVShape` at each dof's own node — row idx is
//! the unit axis vector `nk[dof2nk[idx]]` because the nodal dual matrix is
//! the identity.  The probe's rebuilt `dof_map` was verified against MFEM's
//! public `GetDofMap()` for p = 0..=4 (MAPCHECK lines, all zero).
//!
//! Provenance of the encoded rule (MFEM 4.10 `fem/fe/fe_rt.cpp`):
//! `RT_QuadrilateralElement` constructor lines 26-140; orientation flips
//! lines 81-106 (x block: closed index `i <= p/2` on every open row 82-88,
//! odd-p supplement `i = p/2+1` on `j > p/2` 89-93; y block transposed
//! 94-106); `nk[8]` table line 23; node/normal assignment 108-140.  Axes
//! below use MFEM's own nk indices: 0 = (0,-1), 1 = (1,0), 2 = (0,1),
//! 3 = (-1,0).
//!
//! The old fem-rs table judged the interior flips by the *open* grid index;
//! that coincides with MFEM only for k <= 1 and inverts 6 of 24 interior
//! rows at k = 2 and 8 of 40 at k = 3 (see `tmp/d575/flip_diff.md`).  The
//! normal comparison below is exact, so this test fails against the old
//! table.

use fem_element::raviart_thomas::quad_rt1::mfem_quad_nodal_dofs;

/// `(flat node xy pairs, nk axis indices)` per order k = 0..=3, transcribed
/// from the probe dump.
const TRUTH: [(&[f64], &[u8]); 4] = [
    // k=0 — 4 dofs: probe `QUAD_RT p=0` rows 0..3.
    (
        &[
            0.5, 0.0, 1.0, 0.5,
            0.5, 1.0, 0.0, 0.5,
        ],
        &[
            0, 1, 2, 3,
        ],
    ),
    // k=1 — 12 dofs: probe `QUAD_RT p=1` rows 0..11.
    (
        &[
            0.2113248654051871, 0.0, 0.7886751345948129, 0.0,
            1.0, 0.2113248654051871, 1.0, 0.7886751345948129,
            0.7886751345948129, 1.0, 0.2113248654051871, 1.0,
            0.0, 0.7886751345948129, 0.0, 0.2113248654051871,
            0.5, 0.2113248654051871, 0.5, 0.7886751345948129,
            0.2113248654051871, 0.5, 0.7886751345948129, 0.5,
        ],
        &[
            0, 0, 1, 1, 2, 2, 3, 3, 1, 3, 0, 2,
        ],
    ),
    // k=2 — 24 dofs: probe `QUAD_RT p=2` rows 0..23.
    (
        &[
            0.11270166537925831, 0.0, 0.5, 0.0,
            0.8872983346207417, 0.0, 1.0, 0.11270166537925831,
            1.0, 0.5, 1.0, 0.8872983346207417,
            0.8872983346207417, 1.0, 0.5, 1.0,
            0.11270166537925831, 1.0, 0.0, 0.8872983346207417,
            0.0, 0.5, 0.0, 0.11270166537925831,
            0.27639320225002106, 0.11270166537925831, 0.7236067977499789, 0.11270166537925831,
            0.27639320225002106, 0.5, 0.7236067977499789, 0.5,
            0.27639320225002106, 0.8872983346207417, 0.7236067977499789, 0.8872983346207417,
            0.11270166537925831, 0.27639320225002106, 0.5, 0.27639320225002106,
            0.8872983346207417, 0.27639320225002106, 0.11270166537925831, 0.7236067977499789,
            0.5, 0.7236067977499789, 0.8872983346207417, 0.7236067977499789,
        ],
        &[
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 3, 1, 3, 1, 0, 0,
            0, 2, 2, 2,
        ],
    ),
    // k=3 — 40 dofs: probe `QUAD_RT p=3` rows 0..39.
    (
        &[
            0.0694318442029737, 0.0, 0.33000947820757187, 0.0,
            0.6699905217924281, 0.0, 0.9305681557970263, 0.0,
            1.0, 0.0694318442029737, 1.0, 0.33000947820757187,
            1.0, 0.6699905217924281, 1.0, 0.9305681557970263,
            0.9305681557970263, 1.0, 0.6699905217924281, 1.0,
            0.33000947820757187, 1.0, 0.0694318442029737, 1.0,
            0.0, 0.9305681557970263, 0.0, 0.6699905217924281,
            0.0, 0.33000947820757187, 0.0, 0.0694318442029737,
            0.17267316464601146, 0.0694318442029737, 0.5, 0.0694318442029737,
            0.8273268353539885, 0.0694318442029737, 0.17267316464601146, 0.33000947820757187,
            0.5, 0.33000947820757187, 0.8273268353539885, 0.33000947820757187,
            0.17267316464601146, 0.6699905217924281, 0.5, 0.6699905217924281,
            0.8273268353539885, 0.6699905217924281, 0.17267316464601146, 0.9305681557970263,
            0.5, 0.9305681557970263, 0.8273268353539885, 0.9305681557970263,
            0.0694318442029737, 0.17267316464601146, 0.33000947820757187, 0.17267316464601146,
            0.6699905217924281, 0.17267316464601146, 0.9305681557970263, 0.17267316464601146,
            0.0694318442029737, 0.5, 0.33000947820757187, 0.5,
            0.6699905217924281, 0.5, 0.9305681557970263, 0.5,
            0.0694318442029737, 0.8273268353539885, 0.33000947820757187, 0.8273268353539885,
            0.6699905217924281, 0.8273268353539885, 0.9305681557970263, 0.8273268353539885,
        ],
        &[
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 3,
            1, 1, 3, 3, 1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
        ],
    ),
];

#[test]
fn nodal_table_matches_mfem_probe_k0_to_k3() {
    for (k, &(nodes, axes)) in TRUTH.iter().enumerate() {
        let (pts, nks) = mfem_quad_nodal_dofs(k);
        assert_eq!(pts.len(), axes.len(), "k={k}: row count");
        assert_eq!(nodes.len(), 2 * axes.len(), "k={k}: node table size");
        for i in 0..pts.len() {
            let (x, y) = (nodes[2 * i], nodes[2 * i + 1]);
            assert!(
                (pts[i][0] - x).abs() < 1e-15 && (pts[i][1] - y).abs() < 1e-15,
                "k={k} row {i}: node ({}, {}) vs probe ({x}, {y})",
                pts[i][0],
                pts[i][1],
            );
            let want = match axes[i] {
                0 => [0.0, -1.0],
                1 => [1.0, 0.0],
                2 => [0.0, 1.0],
                _ => [-1.0, 0.0],
            };
            assert!(
                nks[i][0] == want[0] && nks[i][1] == want[1],
                "k={k} row {i}: nk ({}, {}) vs probe axis {} {:?}",
                nks[i][0],
                nks[i][1],
                axes[i],
                want,
            );
        }
    }
}

/// D575 guard: the fix must leave the k <= 1 tables untouched — those are
/// the orders the MFEM-exact transfer prolongation consumes (the gate in
/// `crates/assembly/src/transfer.rs` serves RT orders 0/1 only), and the
/// two flip judgements coincide there, so any drift would mean the
/// node/normal sources changed.
#[test]
fn k0_k1_tables_unchanged() {
    let (pts1, nks1) = mfem_quad_nodal_dofs(1);
    assert_eq!(pts1.len(), 12);
    // Interior rows of k = 1: node positions and normals pinned exactly
    // (face rows 0..8 are covered by the truth table above).
    let expect = [
        [0.5, 0.21132486540518711775],
        [0.5, 0.78867513459481288225],
        [0.21132486540518711775, 0.5],
        [0.78867513459481288225, 0.5],
    ];
    for (i, e) in expect.iter().enumerate() {
        assert_eq!(pts1[8 + i], *e, "k=1 interior row {i}");
    }
    assert_eq!(nks1[8], [1.0, 0.0]);
    assert_eq!(nks1[9], [-1.0, 0.0]);
    assert_eq!(nks1[10], [0.0, -1.0]);
    assert_eq!(nks1[11], [0.0, 1.0]);
}
