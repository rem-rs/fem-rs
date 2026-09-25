//! D797-1 (round 74) — `mixed::HCurlH1WeakDiv` on **non-identity** affine
//! transformations.
//!
//! Round 73's `d772r73_tri_nd_pairing.rs` golden is assembled at `J = I`
//! (single unit triangle), where MFEM's form
//! `M[j,i] = -ip.weight·(Q/detJ)·ûᵀ adj(J) adj(J)ᵀ ∇̂v`
//! (`VectorFEWeakDivergenceIntegrator`, `fem/bilininteg.cpp:1852`; ND
//! physical value `J⁻ᵀû` per `fe_base.cpp:1168 CalcVShape_ND`) reduces to the
//! raw reference-space form and the defective integrand — `grad_phys` already
//! `J⁻ᵀ∇̂v` transformed *again* by `J⁻ᵀ`, times `1/detJ` — is invisible.
//! Under a uniform scaling `x ↦ c·x` the defect scaled the whole element
//! matrix by `c⁻³` (measured, pre-fix: `K(0,7) = 0.375 / 0.046875 / 3.0` for
//! `c = 1 / 2 / 0.5`), while MFEM's matrix is exactly invariant in 2D.
//!
//! Goldens: `tmp/d797/d797_weakdiv.cpp` (WSL, MFEM 4.10 serial) →
//! `tmp/d797/red_mfem_probe.txt`; real `IsoparametricTransformation`s with
//! `J = [v1−v0 | v2−v0]` — the same parametrization convention as
//! `ElementTransformation::from_simplex_nodes`.  The `unit` block equals the
//! round-73 `MFEM_WEAKDIV_ND2_H1` entry for entry, cross-validating the probe.
//! Red evidence: `tmp/d797/{red_femrs_dump.txt, red_entrywise.txt}`
//! (pre-fix ratios 0.125 = 2⁻³ at `c = 2`, 8 at `c = ½`, and O(1) garbage on
//! the non-diagonal `sh1`/`sh2` shears); green: `green_entrywise.txt`
//! (max abs diff 4.2e-16 over all 5×24 entries).

use fem_assembly::mixed::{assemble_hcurl_h1_weak_div, HCurlH1WeakDiv};
use fem_mesh::{ElementType, Mesh};
use fem_space::{FESpace, HCurlSpace, H1Space};

/// Affine triangle geometries `(v0, v1, v2)` and their Jacobians:
/// `unit` J=I, `s2` J=2I, `s05` J=½I, `sh1` J=[[1,1],[0,1]] (det 1),
/// `sh2` J=[[1,1],[0,2]] (det 2).
const GEOS: [(&str, [f64; 6]); 5] = [
    ("unit", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    ("s2", [0.0, 0.0, 2.0, 0.0, 0.0, 2.0]),
    ("s05", [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
    ("sh1", [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
    ("sh2", [0.0, 0.0, 1.0, 0.0, 1.0, 2.0]),
];

/// MFEM 4.10 `VectorFEWeakDivergenceIntegrator::AssembleElementMatrix2`
/// element matrices (H¹ order-1 rows × ND order-2 cols, `IntRules.Get(
/// TRIANGLE, 4)`), dumped by `tmp/d797/d797_weakdiv.cpp` per geometry.
const MFEM_GOLDEN: [(&str, &[[f64; 8]; 3]); 5] = [
    (
        "unit",
        &[
            [0.026415608175648388, 0.09858439182435158, 0.072168783648703133, -0.072168783648703244, -0.098584391824351622, -0.026415608175648232, 0.375, 0.37500000000000033],
            [-0.041666666666666644, -0.041666666666666671, -0.015251058491018212, 0.056917725157684972, 0.056917725157684944, -0.015251058491018347, -0.37499999999999994, -1.700029006457271e-16],
            [0.015251058491018264, -0.056917725157684909, -0.056917725157684923, 0.015251058491018274, 0.041666666666666671, 0.041666666666666588, 1.2251668156751069e-17, -0.37500000000000017],
        ],
    ),
    (
        "s2",
        &[
            [0.026415608175648388, 0.09858439182435158, 0.072168783648703133, -0.072168783648703244, -0.098584391824351622, -0.026415608175648232, 0.375, 0.37500000000000033],
            [-0.041666666666666644, -0.041666666666666671, -0.015251058491018212, 0.056917725157684972, 0.056917725157684944, -0.015251058491018347, -0.37499999999999994, -1.700029006457271e-16],
            [0.015251058491018264, -0.056917725157684909, -0.056917725157684923, 0.015251058491018274, 0.041666666666666671, 0.041666666666666588, 1.2251668156751069e-17, -0.37500000000000017],
        ],
    ),
    (
        "s05",
        &[
            [0.026415608175648388, 0.09858439182435158, 0.072168783648703133, -0.072168783648703244, -0.098584391824351622, -0.026415608175648232, 0.375, 0.37500000000000033],
            [-0.041666666666666644, -0.041666666666666671, -0.015251058491018212, 0.056917725157684972, 0.056917725157684944, -0.015251058491018347, -0.37499999999999994, -1.700029006457271e-16],
            [0.015251058491018264, -0.056917725157684909, -0.056917725157684923, 0.015251058491018274, 0.041666666666666671, 0.041666666666666588, 1.2251668156751069e-17, -0.37500000000000017],
        ],
    ),
    (
        "sh1",
        &[
            [0.041666666666666644, 0.041666666666666671, 0.015251058491018212, -0.056917725157684972, -0.056917725157684944, 0.015251058491018347, 0.37499999999999994, 1.700029006457271e-16],
            [-0.098584391824351553, -0.026415608175648427, 0.026415608175648489, 0.098584391824351678, 0.072168783648703216, -0.072168783648703272, -0.75, 0.37499999999999983],
            [0.056917725157684902, -0.015251058491018243, -0.041666666666666706, -0.041666666666666706, -0.015251058491018279, 0.056917725157684937, 0.375, -0.375],
        ],
    ),
    (
        "sh2",
        &[
            [0.083333333333333287, 0.083333333333333343, 0.030502116982036424, -0.11383545031536994, -0.11383545031536989, 0.030502116982036694, 0.74999999999999989, 3.4000580129145419e-16],
            [-0.11179219591217576, -0.07570780408782421, -0.0096687836487030776, 0.13466878364870333, 0.12146097956087902, -0.058960979560879159, -0.93749999999999989, 0.18749999999999967],
            [0.028458862578842451, -0.0076255292455091215, -0.020833333333333353, -0.020833333333333353, -0.0076255292455091397, 0.028458862578842468, 0.1875, -0.1875],
        ],
    ),
];

fn triangle(v: [f64; 6]) -> Mesh<2> {
    Mesh::<2>::uniform(
        v.to_vec(),
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1i32, 2, 3],
        ElementType::Line2,
    )
}

/// The assembled `HCurlH1WeakDiv` element matrix (H¹ rows × ND cols, quad
/// order 4 — the rule the goldens use) on the given affine triangle.
fn element_matrix(v: [f64; 6]) -> [[f64; 8]; 3] {
    let h1 = H1Space::new(triangle(v), 1);
    let nd = HCurlSpace::new(triangle(v), 2);
    let integ = HCurlH1WeakDiv::new(1.0_f64);
    let m = assemble_hcurl_h1_weak_div(&h1, &nd, &[&integ], 4);
    let rows = h1.element_dofs(0);
    let cols = nd.element_dofs(0);
    assert_eq!(rows.len(), 3);
    assert_eq!(cols.len(), 8);
    let mut out = [[0.0_f64; 8]; 3];
    for (i, &gr) in rows.iter().enumerate() {
        for (j, &gc) in cols.iter().enumerate() {
            out[i][j] = m.get(gr as usize, gc as usize);
        }
    }
    out
}

/// MFEM tet ground truth (`tmp/d797/d797_weakdiv_tet.cpp`, ND order 1 × H¹
/// order 1, `IntRules.Get(TETRAHEDRON, 3)`): 3D adjudicates both the
/// dimension-generic fix and the c¹ scaling (`K(2I-tet) = 2·K(unit-tet)`,
/// since `-∫∇v·u dx` scales as length in 3D).
const MFEM_TET_GOLDEN: [&[[f64; 6]; 4]; 2] = [
    // geo unit (J = I, det 1):
    &[
        [0.16666666666666669, 0.16666666666666671, 0.16666666666666671, 1.7347234759768071e-18, 1.7347234759768071e-18, 1.7347234759768071e-18],
        [-0.083333333333333343, -0.041666666666666664, -0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.0],
        [-0.041666666666666664, -0.083333333333333343, -0.041666666666666664, -0.041666666666666664, 0.0, 0.041666666666666664],
        [-0.041666666666666664, -0.041666666666666664, -0.083333333333333329, 0.0, -0.041666666666666664, -0.041666666666666664],
    ],
    // geo s2 (J = 2I, det 8):
    &[
        [0.33333333333333337, 0.33333333333333343, 0.33333333333333343, 3.4694469519536142e-18, 3.4694469519536142e-18, 3.4694469519536142e-18],
        [-0.16666666666666669, -0.083333333333333329, -0.083333333333333329, 0.083333333333333329, 0.083333333333333329, 0.0],
        [-0.083333333333333329, -0.16666666666666669, -0.083333333333333329, -0.083333333333333329, 0.0, 0.083333333333333329],
        [-0.083333333333333329, -0.083333333333333329, -0.16666666666666666, 0.0, -0.083333333333333329, -0.083333333333333329],
    ],
];

/// Pin (D797-1, 3D): the tet weak-divergence element matrix is MFEM's entry
/// by entry on the unit tet *and* the uniformly doubled tet (which also pins
/// the correct 3D scaling `K ↦ 2·K`).
#[test]
fn weak_div_tet_matches_mfem_and_scales_as_length() {
    for (s, golden) in [(1.0_f64, MFEM_TET_GOLDEN[0]), (2.0, MFEM_TET_GOLDEN[1])] {
        let mesh = Mesh::<3>::uniform(
            vec![0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s],
            vec![0, 1, 2, 3],
            vec![1],
            ElementType::Tet4,
            vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2],
            vec![1i32, 2, 3, 4],
            ElementType::Tri3,
        );
        let h1 = H1Space::new(mesh, 1);
        let nd = HCurlSpace::new(
            Mesh::<3>::uniform(
                vec![0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s],
                vec![0, 1, 2, 3],
                vec![1],
                ElementType::Tet4,
                vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2],
                vec![1i32, 2, 3, 4],
                ElementType::Tri3,
            ),
            1,
        );
        let integ = HCurlH1WeakDiv::new(1.0_f64);
        let m = assemble_hcurl_h1_weak_div(&h1, &nd, &[&integ], 3);
        for (i, &gr) in h1.element_dofs(0).iter().enumerate() {
            for (j, &gc) in nd.element_dofs(0).iter().enumerate() {
                let got = m.get(gr as usize, gc as usize);
                let want = golden[i][j];
                assert!(
                    (got - want).abs() < 1e-12,
                    "tet s={s} K({i},{j}) = {got:.17e} != MFEM {want:.17e}"
                );
            }
        }
    }
}

/// Pin (D797-1): on real non-identity transformations the assembled weak-div
/// element matrix is MFEM's, entry by entry, for uniform scalings *and* non-
/// diagonal shears.  Red before the fix (c⁻³ scaling + O(1) shear garbage).
#[test]
fn weak_div_matches_mfem_on_affine_transforms() {
    for (name, v) in GEOS {
        let golden = MFEM_GOLDEN.iter().find(|(g, _)| *g == name).unwrap().1;
        let m = element_matrix(v);
        for (i, row) in m.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = golden[i][j];
                assert!(
                    (got - want).abs() < 1e-12,
                    "geo {name} K({i},{j}) = {got:.17e} != MFEM {want:.17e}"
                );
            }
        }
    }
}

/// Pin (D797-1): the 2D weak-divergence element matrix is invariant under
/// uniform scaling `x ↦ c·x` (MFEM reference integrand `ûᵀ adj(J) adj(J)ᵀ ∇̂v`
/// with weight `ip.weight` carries `det(J)·c⁻·c⁻ = c⁰` for J = cI).  Red
/// before the fix: every entry scaled as `c⁻³`.
#[test]
fn weak_div_is_scale_invariant() {
    let (_, unit) = GEOS[0];
    let base = element_matrix(unit);
    for (name, v) in &GEOS[1..3] {
        let m = element_matrix(*v);
        for (i, row) in m.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = base[i][j];
                let tol = 1e-14_f64.max(want.abs() * 1e-14);
                assert!(
                    (got - want).abs() < tol,
                    "geo {name} K({i},{j}) = {got:.17e} != c-invariant {want:.17e}"
                );
            }
        }
    }
}
