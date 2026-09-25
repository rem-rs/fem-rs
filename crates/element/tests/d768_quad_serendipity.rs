//! D768 — the 2-D serendipity element is on MFEM's `[0,1]²` square.
//!
//! `QuadSerendipityPk` was the last `[-1,1]²` reference element of
//! `crates/element/src/serendipity.rs`: its lattice, monomial frame (shifted by
//! `+1`, i.e. `[0,2]`) and quadrature (`quad_rule`) all belonged to the
//! historical `[-1,1]²` convention while every consumer of the crate's 2-D
//! reference elements pairs them with a `[0,1]²` rule (`quad_rule_01`, the
//! D729/D738 PA rule) and a `[0,1]²` geometry map.  That is the D743 hex
//! lesson in 2-D, and it was silent, not loud:
//!
//! | quantity | before ([-1,1]² frame) | after (this file pins it) | MFEM 4.10 |
//! |---|---|---|---|
//! | `dof_coords()` of p = 1 | (-1,-1), (0,-1), (-1,0), (0,0) | (0,0), (1,0), (0,1), (1,1) | `BiLinear2DFiniteElement` nodes (0,0),(1,0),(1,1),(0,1) |
//! | own rule points | `quad_rule` (∈ [-1,1]²) | `quad_rule_01` (∈ [0,1]²) | `IntRules.Get(SQUARE, o)` |
//! | p = 1 mass entry `M00` (crate rule) | **1/144** | **1/9** | 1/9 (4/36) |
//!
//! i.e. the old element's basis is the bilinear basis scaled by 1/4 per
//! dimension, so a consumer integrating it with the crate's `[0,1]²` rule got
//! 1/16 of MFEM's mass entries (the hex arm's 1/64 was the same factor
//! squared, 3-D).  Red evidence on the pre-fix source: this file run inside a
//! `c1986cab` worktree (tmp/d768/).
//!
//! **Audit note (updated by D809).**  MFEM's only serendipity class is the
//! *2-D* `H1Ser_QuadrilateralElement` (`fe_ser.cpp:25`), dumped by
//! `tmp/d768/d768_probe.cpp` into `tmp/d768/mfem_2d_dump.txt`.  D768 recorded
//! that this crate's arm was **not** that element (equispaced lattice, `4p`
//! nodal DOFs, truncated tensor space, vs MFEM's Gauss-Lobatto lattice,
//! `(p²+3p+6)/2` DOFs and the genuine serendipity space `S_p`).  **D809 ported
//! MFEM's construction**, so the two now agree entry for entry for `p ≥ 2`;
//! the equality is pinned in `d809_mfem_h1ser_port.rs` (against
//! `tests/data/d809_mfem_h1ser_truth.txt`) and this file keeps the D768
//! properties that still hold:
//!
//! * the `[0,1]²` **frame** (rule = `quad_rule_01`, all nodes inside the unit
//!   square) — the reason D768 existed;
//! * `p = 1` bit-for-bit MFEM `BiLinear2DFiniteElement` (values, gradients and
//!   the mass matrix) — MFEM's `H1Ser(1)` is degenerate (5 DOFs, an identically
//!   zero fifth shape), so the bilinear is the faithful order-1 member;
//! * nodality for `p ≤ 3` (now on the **Gauss-Lobatto** lattice) and the
//!   documented **non**-nodality from `p = 4` (MFEM's interior Legendre
//!   bubbles).

use fem_element::quadrature::quad_rule_01;
use fem_element::serendipity::QuadSerendipityPk;
use fem_element::ReferenceElement;

/// MFEM 4.10 `BiLinear2DFiniteElement` (`fe_fixed_order.cpp:115-131`) evaluated
/// at the probe's points — dumped verbatim into `tmp/d768/mfem_2d_dump.txt`.
/// Slots are MFEM's vertex order (0,0), (1,0), (1,1), (0,1).
const MFEM_BILIN_PTS: [[f64; 2]; 8] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [0.25, 0.75],
    [0.10000000000000001, 0.90000000000000002],
    [0.33333333333333331, 0.66666666666666674],
];
const MFEM_BILIN_VAL: [[f64; 4]; 8] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.25, 0.25, 0.25, 0.25],
    [0.1875, 0.0625, 0.1875, 0.5625],
    [
        0.089999999999999983,
        0.0099999999999999985,
        0.090000000000000011,
        0.81000000000000005,
    ],
    [
        0.22222222222222221,
        0.11111111111111108,
        0.22222222222222224,
        0.44444444444444453,
    ],
];
const MFEM_BILIN_GRAD: [[f64; 8]; 8] = [
    [-1.0, -1.0, 1.0, -0.0, 0.0, 0.0, -0.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -0.0, 0.0],
    [0.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0, 0.0],
    [0.0, -1.0, 0.0, -0.0, 1.0, 0.0, -1.0, 1.0],
    [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5],
    [-0.25, -0.75, 0.25, -0.25, 0.75, 0.25, -0.75, 0.75],
    [
        -0.099999999999999978,
        -0.90000000000000002,
        0.099999999999999978,
        -0.10000000000000001,
        0.90000000000000002,
        0.10000000000000001,
        -0.90000000000000002,
        0.90000000000000002,
    ],
    [
        -0.33333333333333326,
        -0.66666666666666674,
        0.33333333333333326,
        -0.33333333333333331,
        0.66666666666666674,
        0.33333333333333331,
        -0.66666666666666674,
        0.66666666666666674,
    ],
];

/// `QuadSerendipityPk(p)`'s lattice slot order is row-major over
/// `(i, j) ∈ boundary lattice`, i.e. slot `s` carries the node
/// `(s & 1, (s >> 1) & 1) · (1/p)` at `p = 1` and, at `p = 2`, the lattice
/// points `(0,0), (½,0), (1,0), (0,½), (1,½), (0,1), (½,1), (1,1)`.  MFEM's
/// vertex/edge order is (0,0), (1,0), (1,1), (0,1), south, east, north, west.
/// The crate's own lattice order for `p = 1` vs MFEM's vertex order.
const P1_SLOT_TO_MFEM_VERTEX: [usize; 4] = [0, 1, 3, 2];

fn integ(fe: &dyn ReferenceElement, order: u8) -> [f64; 4] {
    let q = fe.quadrature(order);
    let n = fe.n_dofs();
    let mut phi = vec![0.0; n];
    let mut int = vec![0.0; n];
    for (qi, pt) in q.points.iter().enumerate() {
        fe.eval_basis(pt, &mut phi);
        for i in 0..n {
            int[i] += q.weights[qi] * phi[i];
        }
    }
    let mut out = [0.0; 4];
    out[..n].copy_from_slice(&int);
    out
}

/// p = 1 is MFEM's `BiLinear2DFiniteElement`, bit-for-bit (values and
/// gradients, including its `-1.+y` spellings and `-0.0` signs).
#[test]
fn d768_p1_is_mfem_bilinear_bitwise() {
    let fe = QuadSerendipityPk::new(1);
    let mut v = vec![0.0; 4];
    let mut g = vec![0.0; 8];
    for (pi, pt) in MFEM_BILIN_PTS.iter().enumerate() {
        fe.eval_basis(pt, &mut v);
        fe.eval_grad_basis(pt, &mut g);
        for s in 0..4 {
            let m = P1_SLOT_TO_MFEM_VERTEX[s];
            assert_eq!(
                v[s].to_bits(),
                MFEM_BILIN_VAL[pi][m].to_bits(),
                "p=1 shape {s} at {:?}: {} vs MFEM {}",
                pt,
                v[s],
                MFEM_BILIN_VAL[pi][m]
            );
            for d in 0..2 {
                assert_eq!(
                    g[2 * s + d].to_bits(),
                    MFEM_BILIN_GRAD[pi][2 * m + d].to_bits(),
                    "p=1 dshape {s},{d} at {:?}: {} vs MFEM {}",
                    pt,
                    g[2 * s + d],
                    MFEM_BILIN_GRAD[pi][2 * m + d]
                );
            }
        }
    }
}

/// The frame quantities the consumers rely on: nodes and rule points inside
/// `[0,1]²` (red before the fix: `(-1,-1)`-style nodes and a `[-1,1]²` rule),
/// the rule being `quad_rule_01`, and `p = 1` at the unit-square corners.
#[test]
fn d768_frame_is_the_unit_square() {
    for p in 1..=4 {
        let fe = QuadSerendipityPk::new(p);
        let nodes = fe.dof_coords();
        // D809: the count is MFEM's `(p²+3p+6)/2` (4p only for p ≤ 3; p = 4
        // carries one interior bubble DOF, p = 5 three).
        assert_eq!(nodes.len(), fe.n_dofs());
        for c in &nodes {
            assert!(
                (0.0..=1.0).contains(&c[0]) && (0.0..=1.0).contains(&c[1]),
                "p={p}: node {c:?} outside [0,1]²"
            );
        }
        for o in 2..=6u8 {
            let q = fe.quadrature(o);
            assert_eq!(q.points.len(), quad_rule_01(o).points.len());
            assert_eq!(q.weights, quad_rule_01(o).weights);
            for pt in &q.points {
                assert!(
                    (0.0..=1.0).contains(&pt[0]) && (0.0..=1.0).contains(&pt[1]),
                    "p={p} o={o}: rule point {pt:?} outside [0,1]²"
                );
            }
        }
    }
    assert_eq!(
        QuadSerendipityPk::new(1).dof_coords(),
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0]
        ]
    );
}

/// The bilinear mass matrix of a unit-square element — the D743-style red
/// number.  `M = (1/36)·[[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]` (CCW vertex
/// order); using the element's own (now `[0,1]²`) rule this must come out exact
/// to round-off.  With the old `[-1,1]²` basis paired with the crate's
/// `[0,1]²` rule every entry was 1/16 of these (M00 = 1/144 instead of 1/9).
#[test]
fn d768_p1_mass_matrix_matches_mfem_bilinear() {
    let fe = QuadSerendipityPk::new(1);
    let q = fe.quadrature(4);
    let mut phi = vec![0.0; 4];
    let mut m = [[0.0_f64; 4]; 4];
    for (qi, pt) in q.points.iter().enumerate() {
        fe.eval_basis(pt, &mut phi);
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] += q.weights[qi] * phi[i] * phi[j];
            }
        }
    }
    // crate slot order (0,0), (1,0), (0,1), (1,1) → the standard CCW matrix.
    // (0,0)/(1,0)/(1,1)/(0,1) rows: [4,2,1,2], [2,4,2,1], [1,2,4,2], [2,1,2,4];
    // e.g. M(0,0),(0,1) = ∫(1-x)²·∫y(1-y) = (1/3)(1/6) = 1/18 = 2/36.
    let expect_ccw = [
        [4.0_f64 / 36.0, 2.0 / 36.0, 1.0 / 36.0, 2.0 / 36.0],
        [2.0 / 36.0, 4.0 / 36.0, 2.0 / 36.0, 1.0 / 36.0],
        [1.0 / 36.0, 2.0 / 36.0, 4.0 / 36.0, 2.0 / 36.0],
        [2.0 / 36.0, 1.0 / 36.0, 2.0 / 36.0, 4.0 / 36.0],
    ];
    let perm = P1_SLOT_TO_MFEM_VERTEX;
    for i in 0..4 {
        for j in 0..4 {
            let e = expect_ccw[perm[i]][perm[j]];
            assert!(
                (m[i][j] - e).abs() < 1e-15,
                "M[{i}][{j}] = {:e}, expected {e:e} (1/16 of it before D768)",
                m[i][j]
            );
        }
    }
}

/// Nodality follows MFEM (D809): the element is nodal on the **Gauss-Lobatto**
/// lattice for `p ≤ 3` (MFEM's own Kronecker residual is `0.000e+00` there),
/// and **not** nodal from `p = 4`, where MFEM's interior Legendre bubbles are
/// non-interpolatory — the partition-of-unity residual at the sample points is
/// `6.25e-2` in MFEM's own dump, reproduced here rather than "fixed".
#[test]
fn d768_nodal_on_the_gauss_lobatto_lattice() {
    // p = 1: the bilinear is nodal on the unit-square corners.
    let fe1 = QuadSerendipityPk::new(1);
    let mut phi = vec![0.0; 4];
    for (i, c) in fe1.dof_coords().iter().enumerate() {
        fe1.eval_basis(c, &mut phi);
        for j in 0..4 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((phi[j] - want).abs() < 1e-12, "p=1: φ_{j}({c:?})");
        }
    }
    // ∫φ = 1/4 at p = 1 (corner subcells), summing to 1.
    let int = integ(&fe1, 4);
    for (i, v) in int.iter().enumerate() {
        assert!((v - 0.25).abs() < 1e-15, "∫φ_{i} = {v}");
    }
    let total: f64 = int.iter().sum();
    assert!((total - 1.0).abs() < 1e-15, "Σ∫φ = {total}");

    // p = 2, 3: nodal on the GLL lattice, and a partition of unity.
    for p in 2..=3usize {
        let fe = QuadSerendipityPk::new(p);
        let nodes = fe.dof_coords();
        let n = fe.n_dofs();
        let mut phi = vec![0.0; n];
        for (i, c) in nodes.iter().enumerate() {
            fe.eval_basis(c, &mut phi);
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - want).abs() < 1e-12,
                    "p={p}: φ_{j}({c:?}) = {} (want {want})",
                    phi[j]
                );
            }
        }
        let q = fe.quadrature(2 * p as u8 + 1);
        for pt in &q.points {
            fe.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "p={p}: POU = {s}");
        }
    }

    // p = 4: NOT nodal, and NOT a partition of unity — MFEM's own behaviour
    // (D809-1; `Σ∫φ = 1.0278`, POU residual 6.25e-2 at the sample points).
    // The vertices are still nodal (the edge corrections vanish there); it is
    // the *interior* bubble slot that is not interpolatory.
    let fe4 = QuadSerendipityPk::new(4);
    let nodes4 = fe4.dof_coords();
    let n4 = fe4.n_dofs();
    let mut p4 = vec![0.0; n4];
    let interior = nodes4[n4 - 1].clone();
    fe4.eval_basis(&interior, &mut p4);
    let self_dev = (p4[n4 - 1] - 1.0).abs();
    assert!(
        self_dev > 1e-3,
        "p=4's interior bubble is expected to be non-nodal (MFEM quirk), \
         φ_last(node_last) deviates by only {self_dev:e}"
    );
    // And the partition of unity fails at generic points, as in MFEM's dump.
    let q4 = fe4.quadrature(4);
    assert!(
        q4.points.iter().any(|pt| {
            fe4.eval_basis(pt, &mut p4);
            (p4.iter().sum::<f64>() - 1.0).abs() > 1e-3
        }),
        "p=4 POU residual should be O(1e-2), not machine zero (MFEM quirk)"
    );
}
