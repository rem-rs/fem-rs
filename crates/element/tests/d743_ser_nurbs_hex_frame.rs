//! D743 — the serendipity and NURBS **hexahedral** arms live on MFEM's
//! `[0,1]³` reference cube.
//!
//! Round 70's D721 re-based the whole hex family (`hex_rule`, `HexQ1`, `HexQk`,
//! the geometry kernel, the PA kernels) on MFEM's natural `[0,1]³` and left two
//! consumers behind, registered as D743: "they still consume `hex_rule`, which
//! changed, so they must be flipped before they are exercised".
//!
//! * [`fem_element::HexSerendipityPk`] built its basis from monomials of
//!   `u = ξ + 1` on the historical `[-1,1]³` lattice (nodes `-1 + 2i/p`) while
//!   its `quadrature()` already returned the new `[0,1]³` rule — so every
//!   quadrature point was evaluated *outside* the element's own reference cell
//!   (points of `[0,1]³` were read as `[-1,1]³` coordinates), shrinking every
//!   mass entry of the p = 1 member by 2⁻⁶ and corrupting p ≥ 2.
//! * The NURBS hex arm (`nurbs.rs` / `nurbs_vector.rs`) evaluates its basis on
//!   the knot span, i.e. on `[0,1]³` (a clamped uniform knot vector spans the
//!   unit interval), so the new rule is the correct one — the arm needed its
//!   *frame pinned*, not its rule replaced.  Its 2-D sibling still consumed the
//!   `[-1,1]²` rule for a `[0,1]²` knot domain (registered as D768).
//!
//! Truth is `tests/data/d743_ser_nurbs_hex_mfem.txt`: a verbatim MFEM 4.10
//! dump produced by `tmp/d743/probe_d743.cpp` —
//! `IntRules.Get(Geometry::CUBE/SQUARE, order)`, `TriLinear3DFiniteElement::
//! CalcShape/CalcDShape`, the identity-Jacobian reference mass and stiffness of
//! the trilinear hex on the unit cube, and the `NURBS3DFiniteElement` /
//! `NURBS2DFiniteElement` reference mass matrices of a single clamped span with
//! unit weights (the exact object the NURBS arm integrates).
//!
//! MFEM has **no hex serendipity element** (`fe_ser.hpp` carries only
//! `H1Ser_QuadrilateralElement`, and `H1_FECollection` maps every hexahedral
//! cell type to `H1_HexahedronElement(p)`), so the p ≥ 2 members of the fem-rs
//! family (`8 + 12(p-1) + 6(p-1)²` DOFs, the truncated tensor space) have no C++
//! counterpart; their acceptance here is the frame itself (unit-cube lattice
//! nodes, Kronecker deltas on that lattice, gradients against finite
//! differences) plus the p = 1 member, which *is* MFEM's trilinear hex and is
//! compared bit-for-bit.

use fem_element::nurbs::{KnotVector, NurbsPatch2D, NurbsPatch3D};
use fem_element::nurbs_vector::{NurbsHCurl3D, NurbsHDiv3D};
use fem_element::quadrature::{hex_rule, quad_rule_01};
use fem_element::reference::VectorReferenceElement;
use fem_element::{HexQ1, HexSerendipityPk, ReferenceElement};

const FIXTURE: &str = include_str!("data/d743_ser_nurbs_hex_mfem.txt");

/// MFEM `TriLinear3DFiniteElement` node order (`fe_fixed_order.cpp`), the same
/// slot order as [`HexQ1`].  The serendipity family enumerates the same lattice
/// with `i` fastest (`nd3`), so the two slot orders differ by this permutation.
const MFEM_HEX8_NODES: [(f64, f64, f64); 8] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
];

// ─── fixture ────────────────────────────────────────────────────────────────

/// One `HEADER` block of the dump: the header text plus its data lines.
struct Block {
    header: String,
    lines: Vec<String>,
}

/// The parsed MFEM dump, in file order.
struct Dump {
    blocks: Vec<Block>,
}

impl Dump {
    fn parse(text: &str) -> Self {
        let mut blocks: Vec<Block> = Vec::new();
        // Data lines are `P …`, `S …`, `G …` or a bare number.  (The `S ` test
        // needs the trailing space: `SER2 p 2 ndof 8` is a header.)
        let is_header = |line: &str| {
            !(line.starts_with("P ")
                || line.starts_with("S ")
                || line.starts_with("G ")
                || line.parse::<f64>().is_ok())
        };
        for line in text.lines() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if is_header(line) {
                blocks.push(Block {
                    header: line.to_string(),
                    lines: Vec::new(),
                });
            } else {
                let b = blocks.last_mut().expect("data line before any header");
                b.lines.push(line.to_string());
            }
        }
        Dump { blocks }
    }

    /// All blocks whose header starts with `prefix`, in file order.
    fn blocks_of(&self, prefix: &str) -> Vec<&Block> {
        self.blocks
            .iter()
            .filter(|b| b.header.starts_with(prefix))
            .collect()
    }

    /// The single block with this exact header.
    fn block(&self, header: &str) -> &Block {
        let hits: Vec<&Block> = self.blocks.iter().filter(|b| b.header == header).collect();
        assert_eq!(
            hits.len(),
            1,
            "fixture block {header:?} (found {})",
            hits.len()
        );
        hits[0]
    }

    /// The `idx`-th `RULE <tag> <npts>` block as `(points, weights)`.
    fn rule(&self, tag: &str, idx: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let blocks = self.blocks_of(&format!("RULE {tag}"));
        let b = blocks[idx];
        let mut pts = Vec::new();
        let mut wts = Vec::new();
        for l in &b.lines {
            let v: Vec<f64> = l[2..]
                .split_whitespace()
                .map(|t| t.parse().expect("rule value"))
                .collect();
            wts.push(*v.last().unwrap());
            pts.push(v[..v.len() - 1].to_vec());
        }
        (pts, wts)
    }

    /// The `idx`-th `TRILIN order <o>` block as `(shapes, grads)`, one entry per
    /// rule point (`shapes[q]` has 8 values, `grads[q]` 24, row-major).
    fn trilinear(&self, idx: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let b = self.blocks_of("TRILIN order ")[idx];
        let (mut s, mut g) = (Vec::new(), Vec::new());
        for (i, l) in b.lines.iter().enumerate() {
            let vals: Vec<f64> = l[2..]
                .split_whitespace()
                .map(|t| t.parse().expect("trilin value"))
                .collect();
            if i % 2 == 0 {
                s.push(vals);
            } else {
                g.push(vals);
            }
        }
        (s, g)
    }

    /// A bare-number block (`TRILIN_MASS_CUBE4`, `NURBS3D_MASS p 1`, …) as a
    /// flat vector in file order.
    fn flat(&self, header: &str) -> Vec<f64> {
        self.block(header)
            .lines
            .iter()
            .map(|l| l.parse().expect("flat value"))
            .collect()
    }
}

fn dump() -> Dump {
    Dump::parse(FIXTURE)
}

fn bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Slot permutation from the serendipity lattice order (`i` fastest) to MFEM's
/// `TriLinear3DFiniteElement` / [`HexQ1`] order, matched by node coordinate.
fn ser_to_mfem_perm(nodes: &[Vec<f64>]) -> Vec<usize> {
    nodes
        .iter()
        .map(|c| {
            MFEM_HEX8_NODES
                .iter()
                .position(|m| {
                    let mm = [m.0, m.1, m.2];
                    (0..3).all(|d| (c[d] - mm[d]).abs() < 1e-12)
                })
                .unwrap_or_else(|| panic!("serendipity node {c:?} is not a unit-cube corner"))
        })
        .collect()
}

/// Reference mass and stiffness of a scalar element on the identity-mapped unit
/// cube: `M_ij = Σ_q w_q φ_i(q) φ_j(q)`, `K_ij = Σ_q w_q ∇φ_i·∇φ_j`, summed in
/// MFEM's order (points outer, then `i`, then `j`) so the p = 1 comparison can
/// be bit-for-bit.
fn unit_cube_mass_stiffness(
    e: &dyn ReferenceElement,
    qr: &fem_element::QuadratureRule,
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = e.n_dofs();
    let mut m = vec![0.0; n * n];
    let mut k = vec![0.0; n * n];
    let mut v = vec![0.0; n];
    let mut g = vec![0.0; n * dim];
    for q in 0..qr.points.len() {
        e.eval_basis(&qr.points[q], &mut v);
        e.eval_grad_basis(&qr.points[q], &mut g);
        let w = qr.weights[q];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] += w * v[i] * v[j];
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += g[i * dim + d] * g[j * dim + d];
                }
                k[i * n + j] += w * dot;
            }
        }
    }
    (m, k)
}

fn max_rel_dev(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "matrix sizes");
    let scale = b.iter().fold(0.0_f64, |s, &x| s.max(x.abs()));
    a.iter()
        .zip(b)
        .fold(0.0_f64, |s, (&x, &y)| s.max((x - y).abs()))
        / scale.max(1e-300)
}

// ─── the rule the arms consume ───────────────────────────────────────────────

/// `hex_rule` is MFEM's `IntRules.Get(Geometry::CUBE, order)` bit-for-bit —
/// the frame every arm here has to share (D721 pinned the rule; this test
/// re-anchors it next to its consumers).
#[test]
fn d743_mfem_cube_rule_is_hex_rule_bitwise() {
    let d = dump();
    for (idx, order) in (2..=5u8).enumerate() {
        let (mp, mw) = d.rule("CUBE", idx);
        let r = hex_rule(order);
        assert_eq!(r.points.len(), mp.len(), "CUBE order {order}: point count");
        for q in 0..mp.len() {
            for dd in 0..3 {
                assert!(
                    bits_eq(r.points[q][dd], mp[q][dd]),
                    "CUBE order {order} point {q}: fem-rs {:.17e} vs MFEM {:.17e}",
                    r.points[q][dd],
                    mp[q][dd]
                );
                assert!(
                    (0.0..=1.0).contains(&r.points[q][dd]),
                    "CUBE order {order} point {q} outside [0,1]³"
                );
            }
            assert!(
                bits_eq(r.weights[q], mw[q]),
                "CUBE order {order} weight {q}: fem-rs {:.17e} vs MFEM {:.17e}",
                r.weights[q],
                mw[q]
            );
        }
    }
}

/// The SQUARE rule this file anchors the NURBS 2-D arm with, checked the same
/// way (the fixture's first SQUARE block is `IntRules.Get(SQUARE, 4)`).
#[test]
fn d743_mfem_square_rule_is_quad_rule_01_bitwise() {
    let d = dump();
    let (mp, mw) = d.rule("SQUARE", 0);
    let r = quad_rule_01(4);
    assert_eq!(r.points.len(), mp.len(), "SQUARE order 4: point count");
    for q in 0..mp.len() {
        for dd in 0..2 {
            assert!(
                bits_eq(r.points[q][dd], mp[q][dd]),
                "SQUARE order 4 point {q}: fem-rs {:.17e} vs MFEM {:.17e}",
                r.points[q][dd],
                mp[q][dd]
            );
        }
        assert!(bits_eq(r.weights[q], mw[q]), "SQUARE order 4 weight {q}");
    }
}

// ─── serendipity hex arm ────────────────────────────────────────────────────

/// p = 1 of the serendipity family *is* MFEM's `TriLinear3DFiniteElement`
/// (D721's [`HexQ1`]): values and 3-D gradients bit-for-bit at every point of
/// the MFEM CUBE order-4 rule, and the DOF coordinates are the unit-cube
/// corners.
#[test]
fn d743_serendipity_hex_q1_is_mfems_trilinear_bitwise() {
    let d = dump();
    let (mf_s, mf_g) = d.trilinear(2); // TRILIN order 4
    let ser = HexSerendipityPk::new(1);
    let perm = ser_to_mfem_perm(&ser.dof_coords());
    let qr = ser.quadrature(4);
    assert_eq!(qr.points.len(), mf_s.len(), "p=1: rule point count");
    let mut v = vec![0.0; 8];
    let mut g = vec![0.0; 24];
    let mut q1 = vec![0.0; 8];
    for q in 0..qr.points.len() {
        ser.eval_basis(&qr.points[q], &mut v);
        ser.eval_grad_basis(&qr.points[q], &mut g);
        HexQ1.eval_basis(&qr.points[q], &mut q1);
        for i in 0..8 {
            let j = perm[i];
            assert!(
                bits_eq(v[i], mf_s[q][j]),
                "p=1 shape q={q} slot {i}(→MFEM {j}) at {:?}: fem-rs {:.17e} vs MFEM {:.17e}",
                qr.points[q],
                v[i],
                mf_s[q][j]
            );
            assert!(
                bits_eq(v[i], q1[j]),
                "p=1 shape q={q} slot {i}(→HexQ1 {j}): {:.17e} vs {:.17e}",
                v[i],
                q1[j]
            );
            for dd in 0..3 {
                assert!(
                    bits_eq(g[i * 3 + dd], mf_g[q][j * 3 + dd]),
                    "p=1 grad q={q} slot {i}(→MFEM {j}) d={dd}: fem-rs {:.17e} vs MFEM {:.17e}",
                    g[i * 3 + dd],
                    mf_g[q][j * 3 + dd]
                );
            }
        }
    }
}

/// The p = 1 reference mass and stiffness on the identity unit cube equal
/// MFEM's `TriLinear3DFiniteElement` numbers bit-for-bit (rule, basis values
/// and summation order all agree).
#[test]
fn d743_serendipity_hex_q1_unit_cube_mass_stiffness_match_mfem_bitwise() {
    let d = dump();
    let mfem_m = d.flat("TRILIN_MASS_CUBE4");
    let mfem_k = d.flat("TRILIN_STIFF_CUBE4");
    let ser = HexSerendipityPk::new(1);
    let qr = ser.quadrature(4);
    let (m, k) = unit_cube_mass_stiffness(&ser, &qr, 3);
    let perm = ser_to_mfem_perm(&ser.dof_coords());
    for i in 0..8 {
        for j in 0..8 {
            let (mi, mj) = (perm[i], perm[j]);
            assert!(
                bits_eq(m[i * 8 + j], mfem_m[mi * 8 + mj]),
                "p=1 mass [{i},{j}]→MFEM[{mi},{mj}]: fem-rs {:.17e} vs MFEM {:.17e}",
                m[i * 8 + j],
                mfem_m[mi * 8 + mj]
            );
            assert!(
                bits_eq(k[i * 8 + j], mfem_k[mi * 8 + mj]),
                "p=1 stiffness [{i},{j}]→MFEM[{mi},{mj}]: fem-rs {:.17e} vs MFEM {:.17e}",
                k[i * 8 + j],
                mfem_k[mi * 8 + mj]
            );
        }
    }
}

/// Orders 2 and 3: nodal on the unit-cube lattice (`i/p`), partition of unity
/// and finite-difference gradients on `[0,1]³`.
#[test]
fn d743_serendipity_hex_higher_orders_are_nodal_on_the_unit_cube() {
    for p in [2usize, 3] {
        let e = HexSerendipityPk::new(p);
        let n = e.n_dofs();
        let coords = e.dof_coords();
        assert_eq!(coords.len(), n);
        for c in &coords {
            assert_eq!(c.len(), 3);
            for &x in c {
                assert!(
                    (0.0..=1.0).contains(&x),
                    "p={p} dof coord {c:?} outside the unit cube"
                );
                assert!(
                    (x * p as f64 - (x * p as f64).round()).abs() < 1e-12,
                    "p={p} dof coord {c:?} is not on the i/p lattice"
                );
            }
        }
        // Kronecker deltas at those nodes.  The p ≥ 2 basis is the LU-solved
        // monomial interpolant (no MFEM counterpart), so the residual is the
        // monomial Vandermonde's conditioning (~1e-12 at p = 3).
        let mut v = vec![0.0; n];
        for (i, c) in coords.iter().enumerate() {
            e.eval_basis(c, &mut v);
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v[j] - want).abs() < 1e-10,
                    "p={p} node {i}: φ_{j} = {} (want {want})",
                    v[j]
                );
            }
        }
        // Partition of unity + gradients vs central finite differences on
        // [0,1]³ (the p ≥ 2 monomial path extrapolates smoothly, so the probe
        // point may step outside the cube).
        let qr = e.quadrature(4);
        let mut g = vec![0.0; n * 3];
        let h = 1e-5;
        for q in 0..qr.points.len() {
            let x = &qr.points[q];
            e.eval_basis(x, &mut v);
            let sum: f64 = v.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "p={p} POU = {sum}");
            e.eval_grad_basis(x, &mut g);
            for dd in 0..3 {
                let (mut xp, mut xm) = (x.clone(), x.clone());
                xp[dd] += h;
                xm[dd] -= h;
                let (mut vp, mut vm) = (vec![0.0; n], vec![0.0; n]);
                e.eval_basis(&xp, &mut vp);
                e.eval_basis(&xm, &mut vm);
                for i in 0..n {
                    let fd = (vp[i] - vm[i]) / (2.0 * h);
                    assert!(
                        (g[i * 3 + dd] - fd).abs() < 1e-8,
                        "p={p} q={q} dof {i} axis {dd}: grad {:.12e} vs FD {fd:.12e}",
                        g[i * 3 + dd]
                    );
                }
            }
        }
    }
}

// ─── NURBS arms ─────────────────────────────────────────────────────────────

/// The NURBS hex arm's rule is the `[0,1]³` rule (the knot span of a clamped
/// uniform NURBS patch) and its mass matrix equals MFEM's
/// `NURBS3DFiniteElement` reference numbers.
#[test]
fn d743_nurbs_hex_arm_is_unit_cube_and_matches_mfem() {
    let d = dump();
    for p in 1..=2usize {
        let kv = KnotVector::uniform(p, 1); // single clamped span on [0,1]
        let patch = NurbsPatch3D::uniform(kv.clone(), kv.clone(), kv.clone());
        let order = (2 * p + 1) as u8;
        let qr = patch.quadrature(order);
        let hr = hex_rule(order);
        assert_eq!(qr.points.len(), hr.points.len());
        for q in 0..hr.points.len() {
            for dd in 0..3 {
                assert!(
                    bits_eq(qr.points[q][dd], hr.points[q][dd]),
                    "NURBS3D p={p} rule point {q} axis {dd}: {:.17e} vs {:.17e}",
                    qr.points[q][dd],
                    hr.points[q][dd]
                );
                assert!(
                    (0.0..=1.0).contains(&qr.points[q][dd]),
                    "NURBS3D p={p} rule point {q} outside [0,1]³: {:?}",
                    qr.points[q]
                );
            }
            assert!(bits_eq(qr.weights[q], hr.weights[q]), "NURBS3D p={p} weight {q}");
        }
        let (m, _k) = unit_cube_mass_stiffness(&patch, &qr, 3);
        let mfem = d.flat(&format!("NURBS3D_MASS p {p}"));
        let dev = max_rel_dev(&m, &mfem);
        // Measured: 0 (p = 1, the span-local bilinear basis is MFEM-identical)
        // and 5.42e-17 (p = 2).
        assert!(
            dev <= 1e-15,
            "NURBS3D p={p} mass vs MFEM: max relative deviation {dev:.3e}"
        );
    }
}

/// The 2-D NURBS sibling: same span-local convention, `[0,1]²` rule (D768 —
/// before the D743 flip this arm consumed the `[-1,1]²` rule for a `[0,1]²`
/// knot domain, sampling three quarters of its points outside the patch).
#[test]
fn d743_nurbs_quad_arm_is_unit_square_and_matches_mfem() {
    let d = dump();
    for p in 1..=2usize {
        let kv = KnotVector::uniform(p, 1);
        let patch = NurbsPatch2D::uniform(kv.clone(), kv.clone());
        let order = (2 * p + 1) as u8;
        let qr = patch.quadrature(order);
        let tgt = quad_rule_01(order);
        assert_eq!(qr.points.len(), tgt.points.len());
        for q in 0..tgt.points.len() {
            for dd in 0..2 {
                assert!(
                    bits_eq(qr.points[q][dd], tgt.points[q][dd]),
                    "NURBS2D p={p} rule point {q} axis {dd}: {:.17e} vs {:.17e}",
                    qr.points[q][dd],
                    tgt.points[q][dd]
                );
            }
            assert!(bits_eq(qr.weights[q], tgt.weights[q]), "NURBS2D p={p} weight {q}");
            assert!(
                qr.points[q].iter().all(|&x| (0.0..=1.0).contains(&x)),
                "NURBS2D p={p} point {q} outside the knot domain: {:?}",
                qr.points[q]
            );
        }
        let (m, _k) = unit_cube_mass_stiffness(&patch, &qr, 2);
        let mfem = d.flat(&format!("NURBS2D_MASS p {p}"));
        let dev = max_rel_dev(&m, &mfem);
        // Measured: <= 1e-17 for p = 1 and p = 2.
        assert!(
            dev <= 1e-15,
            "NURBS2D p={p} mass vs MFEM: max relative deviation {dev:.3e}"
        );
    }
}

/// The NURBS **vector** hex arms share the scalar arm's frame: `[0,1]³` rule,
/// every point inside the knot domain.
#[test]
fn d743_nurbs_vector_hex_arms_use_the_unit_cube_rule() {
    // The vector elements take the pure knot-sequence type (`iga::KnotVector`,
    // `[0,0,1,1]` = degree 1, one clamped span on [0,1]).
    let kv = fem_element::iga::KnotVector::new_clamped(vec![0.0, 0.0, 1.0, 1.0])
        .expect("KnotVector::new_clamped");
    let hdiv = NurbsHDiv3D::from_knot_vectors(kv.clone(), kv.clone(), kv.clone())
        .expect("NurbsHDiv3D::from_knot_vectors");
    let hcurl = NurbsHCurl3D::from_knot_vectors(kv.clone(), kv.clone(), kv.clone())
        .expect("NurbsHCurl3D::from_knot_vectors");
    let hr = hex_rule(4);
    for (name, qr) in [
        ("HDiv3D", hdiv.quadrature(4)),
        ("HCurl3D", hcurl.quadrature(4)),
    ] {
        assert_eq!(qr.points.len(), hr.points.len(), "{name} rule size");
        for q in 0..hr.points.len() {
            for dd in 0..3 {
                assert!(
                    bits_eq(qr.points[q][dd], hr.points[q][dd]),
                    "{name} rule point {q} axis {dd}: {:.17e} vs {:.17e}",
                    qr.points[q][dd],
                    hr.points[q][dd]
                );
                assert!(
                    (0.0..=1.0).contains(&qr.points[q][dd]),
                    "{name} rule point {q} outside [0,1]³: {:?}",
                    qr.points[q]
                );
            }
            assert!(bits_eq(qr.weights[q], hr.weights[q]), "{name} weight {q}");
        }
    }
}
