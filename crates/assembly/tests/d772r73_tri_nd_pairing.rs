//! D772 / D773 (round 73) — HCurl **triangle** arm: the space / assembler /
//! evaluator pairing, adjudicated the way D765 did for the quad.
//!
//! # Adjudication (`tmp/d777/tri_slot_adjudication.md`)
//!
//! MFEM `ND_TriangleElement(p)` (`fem/fe/fe_nd.cpp:1108`) enumerates its DOFs
//! as `p` Gauss-Legendre points per edge — `eop = OpenPoints(p-1)`, edge
//! `(0,1)` at `(eop[i], 0)`, edge `(1,2)` at `(eop[p-1-i], eop[i])`, edge
//! `(2,0)` at `(0, eop[p-1-i])` — plus the barycentric interior block, with the
//! functional tangents `tk = {1,0, -1,1, 0,-1, 0,1}`.  fem-rs has **two**
//! elements for that one MFEM element: the explicit order-2 specialization
//! [`TriND2`] and the order-generic [`TriNDk`]`::new(p)` port.  They are not
//! two families: both are the nodal dual basis of the *same* functionals on the
//! same space `N_p = P_{p-1}² ⊕ x^⊥ P̃_{p-1}`, hence the same basis (measured
//! here: `3.6e-15`, i.e. round-off).  So the tri arm's defect was **not** a
//! QuadNDk-style family mismatch but a *missing arm*:
//!
//! * D772 — `mixed::ref_elem_vec` had only `(HCurl, Tri3|Tri6, 1)`, so every
//!   mixed tri form at order ≥ 2 returned `Err` (and the space's tri tables are
//!   `TRI_EDGES_MFEM` from order 2 on, `hcurl.rs:52`);
//! * D773 — `dpg_basis::hcurl_ref_elem(Tri3, 2)` named `TriNDk::new(2)` where
//!   `vector_assembler`'s chooser names `TriND2`; the two differ by one ulp in
//!   `dof_coords` (slots 3/5, `1 − GL2[0]` evaluated two ways), nothing more.
//!
//! Both are fixed by making the tri arm name the *same* element as
//! `vector_assembler::paired_vector_reference_element`; the pins below hold the
//! three-way agreement, the MFEM goldens, and the no-family-mismatch verdict.
//!
//! MFEM reference dumps: `tmp/d777/d777_tri_nd.cpp` → `$HOME/work/d777/tri_slots_p{1..4}.txt`
//! (`FE::Nodes` + `CalcVShape` at the nodes) and `tmp/d777/d777_mixed.cpp` →
//! `$HOME/work/d777/mixed_nd{2,3}_h11_unit.txt`
//! (`VectorFEWeakDivergenceIntegrator::AssembleElementMatrix2`, H¹ test × ND
//! trial, explicit `IntRules.Get(TRIANGLE, 4)`).

use fem_assembly::mixed::{assemble_hcurl_h1_weak_div, ref_elem_vec, HCurlH1WeakDiv};
use fem_assembly::vector_assembler::paired_vector_reference_element;
use fem_element::nedelec::{TriND2, TriNDk};
use fem_element::reference::VectorReferenceElement;
use fem_mesh::ElementType;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, H1Space, SpaceType};

/// MFEM 4.10 `ND_TriangleElement(p)::Nodes` (`fe_nd.cpp:1133`), one array per
/// order 1..4, dumped by `d777_tri_nd <p>`.
const MFEM_TRI_NODES: [&[(f64, f64)]; 5] = [
    &[],
    &[(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
    &[
        (0.21132486540518711, 0.0),
        (0.78867513459481287, 0.0),
        (0.78867513459481287, 0.21132486540518711),
        (0.21132486540518711, 0.78867513459481287),
        (0.0, 0.78867513459481287),
        (0.0, 0.21132486540518711),
        (0.33333333333333331, 0.33333333333333331),
        (0.33333333333333331, 0.33333333333333331),
    ],
    &[
        (0.11270166537925831, 0.0),
        (0.5, 0.0),
        (0.8872983346207417, 0.0),
        (0.8872983346207417, 0.11270166537925831),
        (0.5, 0.5),
        (0.11270166537925831, 0.8872983346207417),
        (0.0, 0.8872983346207417),
        (0.0, 0.5),
        (0.0, 0.11270166537925831),
        (0.17445763018700944, 0.17445763018700944),
        (0.17445763018700944, 0.17445763018700944),
        (0.65108473962598112, 0.17445763018700944),
        (0.65108473962598112, 0.17445763018700944),
        (0.17445763018700944, 0.65108473962598112),
        (0.17445763018700944, 0.65108473962598112),
    ],
    &[
        (0.0694318442029737, 0.0),
        (0.33000947820757187, 0.0),
        (0.66999052179242813, 0.0),
        (0.93056815579702634, 0.0),
        (0.93056815579702634, 0.0694318442029737),
        (0.66999052179242813, 0.33000947820757187),
        (0.33000947820757187, 0.66999052179242813),
        (0.0694318442029737, 0.93056815579702634),
        (0.0, 0.93056815579702634),
        (0.0, 0.66999052179242813),
        (0.0, 0.33000947820757187),
        (0.0, 0.0694318442029737),
        (0.10128650732345634, 0.10128650732345634),
        (0.10128650732345634, 0.10128650732345634),
        (0.44935674633827183, 0.10128650732345634),
        (0.44935674633827183, 0.10128650732345634),
        (0.79742698535308731, 0.10128650732345634),
        (0.79742698535308731, 0.10128650732345634),
        (0.10128650732345634, 0.44935674633827183),
        (0.10128650732345634, 0.44935674633827183),
        (0.44935674633827183, 0.44935674633827183),
        (0.44935674633827183, 0.44935674633827183),
        (0.10128650732345634, 0.79742698535308731),
        (0.10128650732345634, 0.79742698535308731),
    ],
];

/// MFEM `ND_TriangleElement::tk[8] = { 1.,0., -1.,1., 0.,-1., 0.,1. }`
/// (`fe_nd.cpp:1103`) — the *unnormalized* reference tangents of, in order, the
/// `(0,1)`, `(1,2)`, `(2,0)` edge slots and the two interior slots.
const MFEM_TRI_TK: [(f64, f64); 4] = [(1.0, 0.0), (-1.0, 1.0), (0.0, -1.0), (0.0, 1.0)];

/// MFEM tangent slot per DOF: `p` per edge (slots `0..3p`, one `dof2tk` index
/// per edge), then the interior points get `(1,0)` / `(0,1)` alternately
/// (`fe_nd.cpp:1150`: `dof2tk[n++] = 0; … dof2tk[n++] = 3;`).
fn mfem_tangent(p: usize, i: usize) -> (f64, f64) {
    if i < 3 * p {
        return MFEM_TRI_TK[i / p];
    }
    if (i - 3 * p) % 2 == 0 {
        MFEM_TRI_TK[0]
    } else {
        MFEM_TRI_TK[3]
    }
}

/// MFEM `VectorFEWeakDivergenceIntegrator::AssembleElementMatrix2` on the unit
/// right triangle `(0,0),(1,0),(0,1)` **with the identity transformation**
/// (`T.SetIdentityTransformation(Geometry::TRIANGLE)`; MFEM's own
/// `GetElementTransformation` for a hand-built triangle carries a rotated
/// reference parameterization, `J = [[-1,-1],[1,0]]`, which is not the
/// reference-space form the Rust lane assembles), H¹ test order 1 × ND trial
/// order 2, explicit `IntRules.Get(TRIANGLE, 4)` (6 points) — row-major
/// `(i,j)` with `i` the H¹ test DOF and `j` the ND trial DOF.
const MFEM_WEAKDIV_ND2_H1: &[&[f64]] = &[
    &[
        0.026415608175648388,
        0.09858439182435158,
        0.072168783648703133,
        -0.072168783648703244,
        -0.098584391824351622,
        -0.026415608175648232,
        0.375,
        0.37500000000000033,
    ],
    &[
        -0.041666666666666644,
        -0.041666666666666671,
        -0.015251058491018212,
        0.056917725157684972,
        0.056917725157684944,
        -0.015251058491018347,
        -0.37499999999999994,
        -1.700029006457271e-16,
    ],
    &[
        0.015251058491018264,
        -0.056917725157684909,
        -0.056917725157684923,
        0.015251058491018274,
        0.041666666666666671,
        0.041666666666666588,
        1.2251668156751069e-17,
        -0.37500000000000017,
    ],
];

/// The same, ND trial order 3 (`IntRules.Get(TRIANGLE, 4)`, 6 points).
const MFEM_WEAKDIV_ND3_H1: &[&[f64]] = &[
    &[
        0.015815278696580781,
        0.0079437851573162018,
        0.0079396659568811388,
        -0.0078756127396998714,
        -9.7144514654701197e-17,
        0.0078756127396997604,
        -0.0079396659568809792,
        -0.0079437851573161203,
        -0.015815278696580951,
        0.13861645496846331,
        0.13861645496846281,
        0.13861645496846289,
        0.19106836025229595,
        0.19106836025229584,
        0.13861645496846328,
    ],
    &[
        -0.0079183148844872552,
        -0.0052958567715441392,
        -0.0079183148844873368,
        0.0078969638120935954,
        0.0026479283857721398,
        2.1351072393752633e-05,
        2.1351072393699724e-05,
        0.0026479283857720128,
        0.0078969638120936526,
        -0.15610042339640748,
        0.017483968427944432,
        -0.13861645496846287,
        -0.01748396842794438,
        -0.17358439182435159,
        -1.6826817716975029e-16,
    ],
    &[
        -0.0078969638120935312,
        -0.0026479283857720557,
        -2.1351072393784726e-05,
        -2.1351072393722276e-05,
        -0.0026479283857720436,
        -0.0078969638120935173,
        0.007918314884487283,
        0.0052958567715441071,
        0.0079183148844873055,
        0.017483968427944148,
        -0.15610042339640728,
        8.3006861912470667e-18,
        -0.17358439182435159,
        -0.017483968427944276,
        -0.13861645496846312,
    ],
];

/// The three-way pairing: MFEM element name, the assembler's chooser, the
/// evaluator, and the mixed lane must all name the same reference element.
fn basis_max_diff(a: &dyn VectorReferenceElement, b: &dyn VectorReferenceElement) -> f64 {
    assert_eq!(a.n_dofs(), b.n_dofs());
    let n = a.n_dofs();
    let d = a.dim() as usize;
    let (mut va, mut vb) = (vec![0.0; n * d], vec![0.0; n * d]);
    let mut worst = 0.0f64;
    for i in 0..=16 {
        for j in 0..=(16 - i) {
            let xi = [i as f64 / 16.0, j as f64 / 16.0];
            a.eval_basis_vec(&xi, &mut va);
            b.eval_basis_vec(&xi, &mut vb);
            for (x, y) in va.iter().zip(vb.iter()) {
                worst = worst.max((x - y).abs());
            }
        }
    }
    worst
}

/// Pin: the tri Nédélec elements place their DOFs exactly where MFEM's
/// `ND_TriangleElement(p)` does — `FE::Nodes`, order by order, DOF by DOF.
/// `TriNDk::new(2)` and `TriND2` are checked against the same `p = 2` table,
/// which is the "is p = 2 a second family?" question of D772, answered *no*.
#[test]
fn tri_nd_nodes_match_mfem() {
    for p in 1..=4usize {
        let elem = TriNDk::new(p);
        let coords = elem.dof_coords();
        let want = MFEM_TRI_NODES[p];
        assert_eq!(coords.len(), want.len(), "p={p}");
        for (i, (c, w)) in coords.iter().zip(want.iter()).enumerate() {
            assert!(
                (c[0] - w.0).abs() < 1e-15 && (c[1] - w.1).abs() < 1e-15,
                "p={p} ND_TriangleElement node {i}: fem-rs ({}, {}) vs MFEM ({}, {})",
                c[0],
                c[1],
                w.0,
                w.1
            );
        }
        if p == 2 {
            let coords = TriND2.dof_coords();
            for (i, (c, w)) in coords.iter().zip(want.iter()).enumerate() {
                assert!(
                    (c[0] - w.0).abs() < 1e-15 && (c[1] - w.1).abs() < 1e-15,
                    "p=2 TriND2 node {i}: fem-rs ({}, {}) vs MFEM ({}, {})",
                    c[0],
                    c[1],
                    w.0,
                    w.1
                );
            }
        }
    }
}

/// fem-rs's own tri **order-1** tangent convention: the third edge is
/// `(0,2)` (`hcurl.rs::TRI_EDGES_ND1`, the historical round-13 baseline the
/// space's order-1 slot/sign table is built for), not MFEM's `(2,0)`.  Only
/// the third edge's *direction* differs, i.e. at `p = 1` the third functional
/// is the negative of MFEM's (`tmp/d777/tri_slot_adjudication.md` §1); from
/// `p = 2` on the two agree exactly (`TRI_EDGES_MFEM`).
const FEMRS_TRI_TK_P1: [(f64, f64); 3] = [(1.0, 0.0), (-1.0, 1.0), (0.0, 1.0)];

/// The tangent whose functional the element at order `p` implements: MFEM's
/// `tk` for `p >= 2`, fem-rs's order-1 convention at `p = 1`.
fn element_tangent(p: usize, i: usize) -> (f64, f64) {
    if p == 1 {
        FEMRS_TRI_TK_P1[i]
    } else {
        mfem_tangent(p, i)
    }
}

/// Pin: the tri Nédélec basis *is* MFEM's nodal dual basis — with MFEM's own
/// `(Nodes, tk)` functionals `σ_i(Φ) = Φ(x_i)·t_i`, `σ_i(Φ_j) = δ_ij` to
/// round-off, for every order (and for `TriND2` at `p = 2`, i.e. the two names
/// describe one element).  At `p = 1` the third edge's functional is fem-rs's
/// own `(0,2)`-direction one, which is exactly `-σ` of MFEM's `(2,0)` one.
#[test]
fn tri_nd_is_nodal_at_mfem_functionals() {
    for p in 1..=4usize {
        let boxes: Vec<Box<dyn VectorReferenceElement>> = if p == 2 {
            vec![Box::new(TriNDk::new(2)), Box::new(TriND2)]
        } else {
            vec![Box::new(TriNDk::new(p))]
        };
        for elem in boxes {
            let n = elem.n_dofs();
            let nodes = MFEM_TRI_NODES[p];
            assert_eq!(nodes.len(), n, "p={p}");
            let mut vals = vec![0.0; n * 2];
            for i in 0..n {
                let (tx, ty) = element_tangent(p, i);
                let xi = [nodes[i].0, nodes[i].1];
                elem.eval_basis_vec(&xi, &mut vals);
                for j in 0..n {
                    let sigma = vals[j * 2] * tx + vals[j * 2 + 1] * ty;
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sigma - expect).abs() < 1e-10,
                        "p={p}: sigma_{i}(Phi_{j}) = {sigma}, expected {expect}"
                    );
                }
            }
        }
    }
}

/// Pin (the D772 verdict): `TriND2` and `TriNDk::new(2)` implement the same
/// functionals and therefore the same basis — they are *not* two families.
/// (Measured `3.6e-15`; the only real difference is one ulp in `dof_coords`
/// where `1 − GL2[0]` is evaluated two ways, which is what D773 aligned.)
#[test]
fn tri_nd2_and_tri_ndk_p2_are_one_element() {
    let (a, b) = (TriND2, TriNDk::new(2));
    assert_eq!(a.order(), b.order());
    assert_eq!(a.n_dofs(), b.n_dofs());
    let diff = basis_max_diff(&a, &b);
    assert!(diff < 1e-13, "TriND2 vs TriNDk::new(2): {diff:.3e}");
    let (ca, cb) = (a.dof_coords(), b.dof_coords());
    for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
        assert!(
            (x[0] - y[0]).abs() <= f64::EPSILON && (x[1] - y[1]).abs() <= f64::EPSILON,
            "slot {i}: TriND2 {x:?} vs TriNDk::new(2) {y:?} (one ulp expected)"
        );
    }
}

/// Pin (D772): the mixed lane's tri H(curl) arm names the element the vector
/// assembler's chooser names — for every order, and for both tri element types.
/// Red before the fix: `ref_elem_vec(HCurl, Tri3, 2)` returned `Err`
/// (the arm only had order 1) and `Tri6` likewise.
#[test]
fn mixed_tri_hcurl_arm_pairs_with_assembler() {
    for o in 1..=4u8 {
        for et in [ElementType::Tri3, ElementType::Tri6] {
            let mixed = ref_elem_vec(et, o, SpaceType::HCurl)
                .unwrap_or_else(|e| panic!("{et:?} order {o}: ref_elem_vec: {e}"));
            let chooser = paired_vector_reference_element(SpaceType::HCurl, et, 2, o);
            assert_eq!(mixed.n_dofs(), chooser.n_dofs(), "{et:?} o={o}");
            assert_eq!(mixed.order(), chooser.order(), "{et:?} o={o}");
            let (ca, cb) = (mixed.dof_coords(), chooser.dof_coords());
            for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                assert_eq!(x, y, "{et:?} o={o} slot {i}: mixed vs assembler");
            }
            let diff = basis_max_diff(&*mixed, &*chooser);
            assert!(diff < 1e-15, "{et:?} o={o}: basis diff {diff:.3e}");
        }
    }
    // Unsupported orders keep the old `Err` contract (the chooser panics there).
    assert!(ref_elem_vec(ElementType::Tri3, 0, SpaceType::HCurl).is_err());
}

/// Pin (D773): the DPG evaluator's tri arm names the element the assembler
/// names.  Red before the fix at `p = 2`, where it named `TriNDk::new(2)`
/// against the assembler's `TriND2` (one ulp in `dof_coords`, `3.6e-15` in the
/// basis) — a naming inconsistency, not a different family.
#[test]
fn dpg_tri_hcurl_arm_pairs_with_assembler() {
    use fem_assembly::dpg::dpg_basis::hcurl_ref_elem;
    for p in 1..=4u8 {
        for et in [ElementType::Tri3, ElementType::Tri6] {
            let dpg = hcurl_ref_elem(et, p);
            let chooser = paired_vector_reference_element(SpaceType::HCurl, et, 2, p);
            assert_eq!(dpg.n_dofs(), chooser.n_dofs(), "{et:?} p={p}");
            let (ca, cb) = (dpg.dof_coords(), chooser.dof_coords());
            for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                assert_eq!(x, y, "{et:?} p={p} slot {i}: dpg vs assembler");
            }
            let diff = basis_max_diff(&*dpg, &*chooser);
            assert!(diff < 1e-15, "{et:?} p={p}: basis diff {diff:.3e}");
        }
    }
}

/// The unit right triangle `(0,0),(1,0),(0,1)` as a one-element `Tri3` mesh —
/// `J = I`, so `VectorFEWeakDivergenceIntegrator`'s `adj(J)` is the identity
/// and the element matrix is the raw reference-space form MFEM dumps.
fn unit_triangle_mesh() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1i32, 2, 3],
        ElementType::Line2,
    )
}

/// Pin (numeric, MFEM golden ≤ 1e-12): with the fixed tri arm, the mixed
/// weak-divergence form assembles *MFEM's* element matrix —
/// `VectorFEWeakDivergenceIntegrator`, H¹ order 1 × ND order 2/3, explicit
/// `IntRules.Get(TRIANGLE, 4)` and MFEM's identity transformation (see the
/// golden's doc comment).  Red before the fix: the arm returned `Err` and the
/// assembly panicked.
#[test]
fn mixed_tri_weak_div_element_matrix_matches_mfem() {
    for (p, golden) in [(2u8, MFEM_WEAKDIV_ND2_H1), (3, MFEM_WEAKDIV_ND3_H1)] {
        let mesh = unit_triangle_mesh();
        let h1 = H1Space::new(mesh, 1);
        let nd = HCurlSpace::new(unit_triangle_mesh(), p);
        let m = assemble_hcurl_h1_weak_div(
            &h1,
            &nd,
            &[&HCurlH1WeakDiv::new(1.0_f64)],
            4,
        );
        let rows = h1.element_dofs(0);
        let cols = nd.element_dofs(0);
        assert_eq!(rows.len(), golden.len(), "p={p}: H1 row count");
        assert_eq!(cols.len(), golden[0].len(), "p={p}: ND column count");
        for (i, &gr) in rows.iter().enumerate() {
            for (j, &gc) in cols.iter().enumerate() {
                let got = m.get(gr as usize, gc as usize);
                let want = golden[i][j];
                assert!(
                    (got - want).abs() < 1e-12,
                    "p={p} K({i},{j}) = {got:.17e} != MFEM {want:.17e}"
                );
            }
        }
    }
}

