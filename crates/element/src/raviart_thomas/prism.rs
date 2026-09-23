//! Raviart-Thomas H(div) element on the reference prism — MFEM
//! `RT_WedgeElement(p)` slot layout (D444/D436) with the k = 0 basis
//! normalized to `RT0WdgFiniteElement` (D572 — see *DOF functionals*).
//!
//! Reference prism (MFEM `Geometry::PRISM` vertices, fem-rs axes): bottom
//! tri {0,1,2} at `xi = 0`, top tri {3,4,5} at `xi = 1`; the triangle plane
//! is `(eta, zeta)` (fem-rs `(eta,zeta)` = MFEM `(x,y)`, fem-rs `xi` = MFEM
//! `z`), MFEM wedge ref coords `z ∈ [0,1]` = fem-rs `xi ∈ [0,1]`.
//!
//! # Slot layout (D444)
//!
//! Slot order = MFEM `RT_WedgeElement::RT_WedgeElement` construction order
//! (`fem/fe/fe_rt.cpp:1079-1201`, probe `tmp/d444/probe_d444.out`):
//!
//! ```text
//!   [ bottom tri ((p+1)(p+2)/2), top tri ((p+1)(p+2)/2),
//!     q0 = zeta=0 quad ((p+1)^2, MFEM FaceVert (0,1,4,3)),
//!     q1 = diagonal quad ((p+1)^2, (1,2,5,4)),
//!     q2 = eta=0 quad   ((p+1)^2, (2,0,3,5)),
//!     interior p(p+1)(3p+4)/2 ]
//! ```
//!
//! The quad faces follow the MFEM canonical frames (`u` along `c0→c1`,
//! `v` along `c0→c3` of the `FaceVert` row; dof slot `n = v*(p+1) + u`,
//! probe-verified against `RT_WedgeElement(1)` nodes 6..18): q0 frame
//! `(eta, xi)`, q1 frame `(zeta, xi)` (u runs along the diagonal
//! `zeta: 0→1`, so `eta = 1-u`), q2 frame `(1-zeta, xi)` (u runs `c0→c1` =
//! vert 2→0, `zeta: 1→0`).
//!
//! The two triangular faces use the fem-rs standard barycentric grid
//! convention (slot `(j,i)` at barycentric weights `(v1: j, v2: i)`, i.e.
//! `eta ∝ j`, `zeta ∝ i`, moment poly `eta^j zeta^i`) on **both** faces.
//! MFEM's own reference element enumerates its *top* face transposed
//! (`fe_rt.cpp:1129-1136` fetches the `L2TriangleFE` node
//! `j + i(2p+3-i)/2`, the transposed index function; probe nodes 3..5 vs
//! 0..2) — a reference-element-internal slot quirk that fem-rs normalises
//! to the standard grid (documented deviation: the dof *set*, vsize, ess
//! and all global dof maps are unaffected; only the within-top-block slot
//! permutation differs from MFEM's raw Nodes table).
//!
//! # Dimension / interior (D436)
//!
//! `dim = (p+1)(3p^2+12p+10)/2` — probe: `RT_WedgeElement(1) dof=25`,
//! `(2) dof=69` (`tmp/d444/probe_d444.out`); interior `p(p+1)(3p+4)/2`
//! (`RT_dof[PRISM]`, `fe_coll.cpp:2581-2583`), laid out MFEM-tensor-fashion:
//! horizontal interior `p(p+1)^2` (`RT_TriangleElement(p)` interior ⊗
//! `L2Segment(p)`) then vertical interior `p(p+1)(p+2)/2`
//! (`L2Triangle(p)` ⊗ `H1Segment(p+1)`, segment indices `2..=p+1`).
//!
//! # DOF functionals
//!
//! The `k = 0` basis is the hard-coded MFEM **`RT0WdgFiniteElement`** tensor
//! basis (D572 adjudication: fem-rs HDiv k=0 follows the collection MFEM's
//! `RT0_3DFECollection` serves, `fe_coll.hpp:1470-1476`).  Relative to the
//! generic `RT_WedgeElement(0)` tensor product, `RT0WdgFiniteElement::
//! CalcVShape` (`fe_fixed_order.cpp:6403`) doubles the two *triangular-face*
//! basis functions (slots 0,1) and leaves the three quadrilateral-face ones
//! untouched; its `nk` table (`fe_fixed_order.cpp:6439`) halves the same two
//! rows to `n̂|F| = (±1/2, 0, 0)` (probe `tmp/d572/d572_families.txt`: NK
//! wedge_fix, VSHAPE wedge_fix — tri slots ×2, quads identical).  Both
//! scalings flow from this one basis: `HDivSpace::interpolate_vector`'s
//! reference dual `W = φ·nk` becomes `diag(2,2,1,1,1)` against its unchanged
//! `n̂-axis` sample rows, so the stored dof is `f·adj(J)·n̂|F|` on every face
//! (tri dofs halve, quad dofs unchanged — probe
//! `tmp/d572/d572_prism_mass.txt` PROJ: tri `∓0.5`, quads equal across
//! collections) while `W = φ·nk` stays diagonal; the reconstructed field
//! `Σ dof·φ` and the transfer/prolongation rows are invariant on
//! axis-aligned frames (`d482` stays green).  For `k >= 1` the basis is the
//! moment-dual construction of this crate: the dof functionals are the
//! *exact* normal-flux moments `∫_face (Φ·n̂) q ds` (`q` over the face
//! monomials of degree ≤ p in the frame above) plus volume moments
//! `∫_V Φ_c w dx` for the interior (weights mirroring the MFEM tensor index
//! ranges), evaluated with exact-enough quadrature; the basis is the
//! pseudo-dual of those functionals over the MFEM wedge polynomial space
//! (horizontal comps `RT_tri(p) ⊗ P_p(xi)`-shaped monomials, vertical comp
//! `P_p(tri) ⊗ P_{p+1}(xi)`).  The span is exactly MFEM's wedge RT space;
//! dof *values* of a projected field are moment-based, not MFEM's nodal
//! point samples (that residual gap is recorded in the D444 debt notes).
//! k ≥ 1 keeps the generic `RT_WedgeElement` alignment, which is what
//! `RT_FECollection(p, 3)` serves (`fe_coll.cpp:2581`).

use crate::quadrature::{
    gauss_lobatto_01, gauss_legendre_01, prism_rule, quad_rule_01, tri_rule,
};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Monomial helpers ───────────────────────────────────────────────────────

/// Monomial `xi^a · eta^b · zeta^c` of vector component `comp`
/// (0 = xi/layer, 1 = eta, 2 = zeta).
#[derive(Clone)]
struct Mono {
    comp: u8,
    a: usize,
    b: usize,
    c: usize,
}

fn eval_mono(m: &Mono, xi: f64, eta: f64, zeta: f64) -> f64 {
    xi.powi(m.a as i32) * eta.powi(m.b as i32) * zeta.powi(m.c as i32)
}

/// MFEM-shaped wedge monomial span (D444): horizontal components (eta, zeta)
/// carry `RT_TriangleElement(p) ⊗ L2Segment(p)`-shaped monomials —
/// triangle-plane degree `b + c ≤ p + 1`, layer degree `a ≤ p`; the vertical
/// component carries `L2Triangle(p) ⊗ H1Segment(p+1)`-shaped monomials —
/// `b + c ≤ p`, `a ≤ p + 1`.  Column count `(p+1)(p+2)(3p+8)/2 ≥ n_dof` for
/// every `p` (difference `(p+1)(p+3) > 0`), so the dual system is never
/// rank-deficient, and the span equals MFEM's wedge RT space.
fn prism_monos(p: usize) -> Vec<Mono> {
    let mut m = Vec::new();
    // vertical comp (0 = xi): tri degree ≤ p, xi degree ≤ p+1
    for b in 0..=p {
        for c in 0..=(p - b) {
            for a in 0..=(p + 1) {
                m.push(Mono { comp: 0, a, b, c });
            }
        }
    }
    // horizontal comps (1, 2): tri degree ≤ p+1, xi degree ≤ p
    for comp in [1u8, 2] {
        for b in 0..=(p + 1) {
            for c in 0..=(p + 1 - b) {
                for a in 0..=p {
                    m.push(Mono { comp, a, b, c });
                }
            }
        }
    }
    m
}

// ─── Face definitions (MFEM slot order, D444) ───────────────────────────────

/// Face slot order = MFEM `RT_WedgeElement` / `Geometry::PRISM` FaceVert
/// order: 0 = bottom tri (xi=0, n̂=(−1,0,0)), 1 = top tri (xi=1, n̂=(1,0,0)),
/// 2 = q0 quad (zeta=0, n̂=(0,0,−1), FaceVert (0,1,4,3)), 3 = q1 quad
/// (diagonal eta+zeta=1, n̂=(0,1,1)/√2, ds=√2, (1,2,5,4)), 4 = q2 quad
/// (eta=0, n̂=(0,−1,0), (2,0,3,5)).
const FACE_DEFS: [(u8, [f64; 3]); 5] = [
    (0, [-1.0, 0.0, 0.0]),                              // tri, xi=0
    (0, [1.0, 0.0, 0.0]),                               // tri, xi=1
    (1, [0.0, 0.0, -1.0]),                              // quad, zeta=0
    (1, [0.0, 0.7071067811865475, 0.7071067811865475]), // quad diagonal
    (1, [0.0, -1.0, 0.0]),                              // quad, eta=0
];

/// Face-monospace sizes: `(tri_dofs, quad_dofs)` of order `p`.
fn face_block_sizes(p: usize) -> (usize, usize) {
    ((p + 1) * (p + 2) / 2, (p + 1) * (p + 1))
}

/// Interior dof count of MFEM `RT_WedgeElement(p)`:
/// `p(p+1)(3p+4)/2` (`RT_dof[PRISM]`, `fe_coll.cpp:2581-2583`;
/// `fe_rt.cpp:1183-1201` loops: horizontal `p(p+1)^2` + vertical
/// `p(p+1)(p+2)/2`).
fn wedge_interior_dofs(p: usize) -> usize {
    p * (p + 1) * (3 * p + 4) / 2
}

/// Total dof count of MFEM `RT_WedgeElement(p)`:
/// `(p+1)(3p²+12p+10)/2` (probe: 25 @ p=1, 69 @ p=2).
fn prism_rtk_dim(p: usize) -> usize {
    let (tri, quad) = face_block_sizes(p);
    2 * tri + 3 * quad + wedge_interior_dofs(p)
}

// ─── DOF functionals ────────────────────────────────────────────────────────

/// One face moment functional: `∫_face (Φ·n̂) eta^b·zeta^c ds` on a tri face
/// (standard barycentric convention: the poly exponents follow the grid
/// weights `(v1: b, v2: c)`), or `∫_face (Φ·n̂) u^i v^j ds` on a quad face
/// in its MFEM canonical frame (exact quadrature).
fn face_dof_value(m: &Mono, face: usize, b: usize, c: usize) -> f64 {
    let (_ftype, n) = FACE_DEFS[face];
    let dot = match m.comp {
        0 => n[0],
        1 => n[1],
        _ => n[2],
    };
    if face < 2 {
        // Tri face xi = const: moment poly eta^b zeta^c.  Order ≥ 14 covers
        // the monomial span's total degree (≤ 2p+1 at p = 3) plus the poly
        // degree — exact functionals (an under-integrated functional
        // perturbs the dual and leaks flux onto foreign faces).
        let xi_val = if face == 0 { 0.0 } else { 1.0 };
        let rule = tri_rule(((b + c) * 2 + 14).min(20) as u8);
        let mut sum = 0.0;
        for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
            let (eta, zeta) = (pt[0], pt[1]);
            let mv = eval_mono(m, xi_val, eta, zeta);
            let poly = eta.powi(b as i32) * zeta.powi(c as i32);
            sum += w * dot * mv * poly;
        }
        sum
    } else {
        // Quad faces in their MFEM canonical frames (D444):
        //   q0 (zeta=0, FaceVert (0,1,4,3)): u=eta, v=xi
        //   q1 (diag,   FaceVert (1,2,5,4)): u=zeta (eta = 1-u), v=xi
        //   q2 (eta=0,  FaceVert (2,0,3,5)): u=1-zeta,         v=xi
        // dof slot n = v*(p+1) + u enumerates (i=u, j=v); ds as noted.
        let (i, j) = (b, c);
        let rule = quad_rule_01(((i + j) * 2 + 14).min(20) as u8);
        let ds = if face == 3 { std::f64::consts::SQRT_2 } else { 1.0 };
        let mut sum = 0.0;
        for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
            let (u, v) = (pt[0], pt[1]);
            let (xi, eta, zeta) = match face {
                2 => (v, u, 0.0),
                3 => (v, 1.0 - u, u),
                _ => (v, 0.0, 1.0 - u),
            };
            let mv = eval_mono(m, xi, eta, zeta);
            let poly = u.powi(i as i32) * v.powi(j as i32);
            sum += w * ds * dot * mv * poly;
        }
        sum
    }
}

/// Interior volume moment weight `∫_V mono · xi^a eta^b zeta^c dV`
/// (exact quadrature over the reference prism; the caller restricts the
/// monomial's component to the functional's target component).
fn interior_dof_value(rule: &QuadratureRule, m: &Mono, a: usize, b: usize, c: usize) -> f64 {
    let mut sum = 0.0;
    for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
        let mv = eval_mono(m, pt[0], pt[1], pt[2]);
        let poly = pt[0].powi(a as i32) * pt[1].powi(b as i32) * pt[2].powi(c as i32);
        sum += w * mv * poly;
    }
    sum
}

// ─── Vandermonde construction ───────────────────────────────────────────────

/// Pseudo-dual of the Vandermonde rows via `V·Vᵀ` Gaussian elimination;
/// panics if rank deficient (the dof functionals must be independent — a
/// silent pseudo inverse would paper over construction bugs).
fn solve_normal_eq(v: &[Vec<f64>], n: usize, m: usize) -> Vec<f64> {
    let mut vvt = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for col in 0..m {
                s += v[i][col] * v[j][col];
            }
            vvt[i][j] = s;
        }
    }
    let mut a = vvt.clone();
    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }
    for c in 0..n {
        let mut best = c;
        let mut bv = a[c][c].abs();
        for r in (c + 1)..n {
            if a[r][c].abs() > bv {
                bv = a[r][c].abs();
                best = r;
            }
        }
        assert!(
            bv > 1e-25,
            "PrismRTk: dual Gram matrix rank-deficient at pivot {c} (|piv| = {bv})"
        );
        a.swap(c, best);
        inv.swap(c, best);
        let ip = 1.0 / a[c][c];
        for j in 0..n {
            a[c][j] *= ip;
            inv[c][j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r][c];
            for j in 0..n {
                a[r][j] -= f * a[c][j];
                inv[r][j] -= f * inv[c][j];
            }
        }
    }
    let mut coeff = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..n {
                s += v[k][j] * inv[k][i];
            }
            coeff[i * m + j] = s;
        }
    }
    coeff
}

/// Build the Vandermonde rows in MFEM slot order:
/// `[bottom tri, top tri, q0, q1, q2, interior]` (D444).
fn build_prism_rtk(p: usize) -> (Vec<f64>, usize) {
    let n = prism_rtk_dim(p);
    let monos = prism_monos(p);
    let m = monos.len();
    let mut vand = vec![vec![0.0_f64; m]; n];
    let mut row = 0;

    // Faces 0..5 in MFEM slot order.  Tri faces: standard barycentric grid
    // (j = eta exponent outer, i = zeta exponent inner).  Quad faces:
    // canonical frames, slot n = v*(p+1) + u (u = frame coord 1 inner).
    for face in 0..5 {
        if face < 2 {
            for j in 0..=p {
                for i in 0..=(p - j) {
                    for (col, mo) in monos.iter().enumerate() {
                        vand[row][col] = face_dof_value(mo, face, j, i);
                    }
                    row += 1;
                }
            }
        } else {
            for j in 0..=p {
                for i in 0..=p {
                    for (col, mo) in monos.iter().enumerate() {
                        vand[row][col] = face_dof_value(mo, face, i, j);
                    }
                    row += 1;
                }
            }
        }
    }

    // Interior (k >= 1), MFEM tensor structure in moment form (D436):
    // horizontal `∫Φ_eta w`, `∫Φ_zeta w` for w = xi^m · (tri monomial of
    // degree ≤ p−1), m = 0..=p → p(p+1)² rows; then vertical `∫Φ_xi w` for
    // w = xi^m · (tri monomial of degree ≤ p), m = 2..=p+1
    // → p(p+1)(p+2)/2 rows.
    if p >= 1 {
        let order = (4 * p + 8).min(40) as u8;
        let rule = prism_rule(order);
        for comp in [1u8, 2] {
            for mm in 0..=p {
                for bb in 0..p {
                    for cc in 0..=(p - 1 - bb) {
                        for (col, mo) in monos.iter().enumerate() {
                            if mo.comp == comp {
                                vand[row][col] = interior_dof_value(&rule, mo, mm, bb, cc);
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
        for mm in 2..=(p + 1) {
            for bb in 0..=p {
                for cc in 0..=(p - bb) {
                    for (col, mo) in monos.iter().enumerate() {
                        if mo.comp == 0 {
                            vand[row][col] = interior_dof_value(&rule, mo, mm, bb, cc);
                        }
                    }
                    row += 1;
                }
            }
        }
    }

    assert_eq!(row, n, "row count {row} != dimension {n} for p={p}");
    (solve_normal_eq(&vand, n, m), m)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Raviart-Thomas H(div) element on the reference prism — MFEM
/// `RT_WedgeElement(p)` slot layout (D444).  Supports `p ≤ 3` (the moment
/// dual is verified 0..=3; higher orders wait on a nodal MFEM port).
pub struct PrismRTk {
    p: usize,
    coeff: Vec<f64>,
    n: usize,
    m: usize,
    monos: Vec<Mono>,
}

/// Order-0 element (alias for `PrismRTk::new(0)`, kept for backward compat).
pub type PrismRT0 = PrismRTk;

/// MFEM `RT0WdgFiniteElement` dof node/normal rows — `(point, nk)` per dof in
/// the element's slot order (D597 single source, the D541 pyramid precedent):
/// the space engine's dual rows (`HDivSpace::interp_rows(Prism6)`) and the
/// prolongation builder's slot table (`hdiv_rt_slot_rows(Prism, 0)`) both
/// consume this one definition instead of keeping hand-copied 5-row tables.
///
/// Frame: axes = (xi, eta, zeta) with xi = layer, (eta, zeta) = triangle
/// plane; slot order = bottom tri, top tri, q0 (zeta = 0), q1 (diagonal
/// eta+zeta = 1), q2 (eta = 0).  The triangular-face normals carry the
/// `RT0WdgFiniteElement` nk `n̂|F| = ±½` (`fe_fixed_order.cpp:6439`) — the
/// quad rows coincide with the generic `RT_WedgeElement` table — matching the
/// D572 basis (tri slots doubled) and the stored dofs (dof = f·adj(J)·n̂|F|);
/// the reference dual is the identity, so the prolongation rows P = B are
/// MFEM's RT0Wdg `GetLocalInterpolation` (`fe_fixed_order.cpp:6442`) directly
/// (D584).  Only order 0 has a nodal table (the collection
/// `RT0_3DFECollection` serves); orders ≥ 1 are moment-based in this crate
/// and stay off the nodal path.
pub fn mfem_nodal_rows() -> Vec<([f64; 3], [f64; 3])> {
    vec![
        ([0.0, 1.0 / 3.0, 1.0 / 3.0], [-0.5, 0.0, 0.0]),
        ([1.0, 1.0 / 3.0, 1.0 / 3.0], [0.5, 0.0, 0.0]),
        ([0.5, 0.5, 0.0], [0.0, 0.0, -1.0]),
        ([0.5, 0.5, 0.5], [0.0, 1.0, 1.0]),
        ([0.5, 0.0, 0.5], [0.0, -1.0, 0.0]),
    ]
}

impl PrismRTk {
    pub fn new(order: usize) -> Self {
        assert!(
            order <= 3,
            "PrismRTk: order {order} exceeds the verified moment-dual cap 3 \
             (MFEM RT_WedgeElement is order-generic; a nodal port is future work)"
        );
        let (coeff, m) = build_prism_rtk(order);
        let n = prism_rtk_dim(order);
        PrismRTk {
            p: order,
            coeff,
            n,
            m,
            monos: prism_monos(order),
        }
    }

    /// Half-open slot range of face `face` (0..5; 5 = interior) in the
    /// element's slot layout — `[bottom, top, q0, q1, q2, interior]`.
    /// Public for MFEM-layout parity tests.
    pub fn slot_range(&self, face: usize) -> std::ops::Range<usize> {
        let (tri, quad) = face_block_sizes(self.p);
        let start = match face {
            0 => 0,
            1 => tri,
            2 => 2 * tri,
            3 => 2 * tri + quad,
            4 => 2 * tri + 2 * quad,
            _ => 2 * tri + 3 * quad,
        };
        let len = match face {
            0 | 1 => tri,
            2 | 3 | 4 => quad,
            _ => wedge_interior_dofs(self.p),
        };
        start..start + len
    }
}

/// 1-D Gauss (open) points, `n` of them on [0,1] — MFEM `OpenPoints(p)`
/// (Gauss-Legendre), used by the `L2TriangleFE` lattice / `L2SegmentFE` /
/// `RTTriangleFE` edge nodes.
fn open_points(n: usize) -> Vec<f64> {
    gauss_legendre_01(n).0
}

/// 1-D Gauss-Lobatto (closed) points, `n` of them on [0,1] — MFEM
/// `ClosedPoints` for `H1SegmentFE`.
fn closed_points(n: usize) -> Vec<f64> {
    gauss_lobatto_01(n).0
}

/// MFEM `H1SegmentFE(p+1)` node index → `ClosedPoints(p+1)` position:
/// `Nodes[0]=cp[0], Nodes[1]=cp[p+1], Nodes[i+1]=cp[i]` (`fe_h1.cpp:21-35`).
fn h1_seg_node(m: usize, p: usize) -> f64 {
    let cp = closed_points(p + 2);
    match m {
        0 => cp[0],
        1 => cp[p + 1],
        i => cp[i - 1],
    }
}

impl VectorReferenceElement for PrismRTk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        self.n
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        if self.p == 0 {
            // MFEM RT0WdgFiniteElement tensor product (CalcVShape in
            // fe_fixed_order.cpp:6403; D572): the generic RT_WedgeElement(0)
            // product with the two TRIANGULAR-face functions doubled
            // (shape(0,2) = 2z−2, shape(1,2) = 2z in MFEM's z-layer axis).
            // Reference axes are xi = layer, (eta,zeta) = triangle plane.
            // DOF order = wedge face order (bottom/top tri, then 3 quads
            // with t_dof = 2D RT0-triangle edges 0,1,2):
            //   Φ₀(bottom) = (2(xi−1), 0, 0)     Φ₁(top) = (2xi, 0, 0)
            //   Φ₂ = (0, η, ζ−1)  Φ₃ = (0, η, ζ)  Φ₄ = (0, η−1, ζ)
            let (a, b, c) = (xi[0], xi[1], xi[2]);
            values.fill(0.0);
            values[0] = 2.0 * (a - 1.0);
            values[3] = 2.0 * a;
            values[7] = b;
            values[8] = c - 1.0;
            values[10] = b;
            values[11] = c;
            values[13] = b - 1.0;
            values[14] = c;
            return;
        }
        let mut mv = vec![0.0_f64; self.monos.len()];
        for (j, m) in self.monos.iter().enumerate() {
            mv[j] = eval_mono(m, xi[0], xi[1], xi[2]);
        }
        values.fill(0.0);
        for i in 0..self.n {
            for j in 0..self.m {
                let idx = i * self.m + j;
                if idx < self.coeff.len() {
                    let c = self.coeff[idx];
                    if c != 0.0 {
                        values[i * 3 + self.monos[j].comp as usize] += c * mv[j];
                    }
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let h = 1e-6;
        let n3 = self.n * 3;
        let mut vp = vec![0.0; n3];
        let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0] + h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0] - h, xi[1], xi[2]], &mut vm);
            let dfy_dx = (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
            let dfz_dx = (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1] + h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1] - h, xi[2]], &mut vm);
            let dfx_dy = (vp[i * 3] - vm[i * 3]) / (2.0 * h);
            let dfz_dy = (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] + h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] - h], &mut vm);
            let dfx_dz = (vp[i * 3] - vm[i * 3]) / (2.0 * h);
            let dfy_dz = (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
            curl_vals[i * 3] = dfz_dy - dfy_dz;
            curl_vals[i * 3 + 1] = dfx_dz - dfz_dx;
            curl_vals[i * 3 + 2] = dfy_dx - dfx_dy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        if self.p == 0 {
            // div of MFEM RT0WdgFiniteElement (CalcDivShape,
            // fe_fixed_order.cpp:6429): every dof carries div 2 — the
            // doubled tri-face functions (∂/∂xi of 2(xi±1)) and the quad
            // dofs (∂/∂η + ∂/∂ζ of the 2D RT0 edge) = 1 + 1 = 2 alike.
            div_vals[0] = 2.0;
            div_vals[1] = 2.0;
            div_vals[2] = 2.0;
            div_vals[3] = 2.0;
            div_vals[4] = 2.0;
            return;
        }
        let h = 1e-6;
        let n3 = self.n * 3;
        let mut vp = vec![0.0; n3];
        let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0] + h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0] - h, xi[1], xi[2]], &mut vm);
            let dfx = (vp[i * 3] - vm[i * 3]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1] + h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1] - h, xi[2]], &mut vm);
            let dfy = (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] + h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] - h], &mut vm);
            let dfz = (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
            div_vals[i] = dfx + dfy + dfz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        prism_rule(order)
    }

    /// MFEM `RT_WedgeElement` node positions (`tmp/d444/probe_d444.out`,
    /// `RT_WedgeElement::RT_WedgeElement` Nodes table) in fem-rs axes
    /// (`(eta,zeta)` = MFEM `(x,y)`, `xi` = MFEM `z`).  The tri *face*
    /// blocks use the standard barycentric enumeration (see module docs for
    /// the documented top-face deviation from MFEM's raw internal order);
    /// the interior blocks mirror MFEM's own enumeration verbatim.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.p;
        let op = open_points(p + 1); // Gauss (L2 tri lattice / RT tri edges / L2 seg)
        let mut coords = Vec::with_capacity(self.n);

        // Standard-grid tri node (j, i) — barycentric weights (v1: j, v2: i)
        // on the OpenPoints lattice: (eta, zeta) = (op[j]/w, op[i]/w).
        let tri_face_node = |j: usize, i: usize| -> (f64, f64) {
            let w = op[j] + op[i] + op[p - j - i];
            (op[j] / w, op[i] / w)
        };
        // MFEM `L2Triangle(p)` node l (enumeration j' outer, i' inner):
        // (x, y) = (op[i']/w, op[j']/w) → fem-rs (eta, zeta) (fe_l2.cpp:570).
        let l2_tri_node = |j: usize, i: usize| -> (f64, f64) {
            let w = op[i] + op[j] + op[p - i - j];
            (op[i] / w, op[j] / w)
        };

        // Faces 0/1: bottom/top tri (xi = 0 / xi = 1), standard grid order.
        for &xi_val in &[0.0, 1.0] {
            for j in 0..=p {
                for i in 0..=(p - j) {
                    let (eta, zeta) = tri_face_node(j, i);
                    coords.push(vec![xi_val, eta, zeta]);
                }
            }
        }
        // Faces 2..5: quads, slot n = v*(p+1)+u, (u,v) = canonical frames.
        for j in 0..=p {
            for i in 0..=p {
                // q0 (zeta=0): u=eta, v=xi
                coords.push(vec![op[j], op[i], 0.0]);
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                // q1 (diag): u=zeta, eta = 1-u, v=xi
                coords.push(vec![op[j], 1.0 - op[i], op[i]]);
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                // q2 (eta=0): u=1-zeta, v=xi
                coords.push(vec![op[j], 0.0, 1.0 - op[i]]);
            }
        }
        // Interior.  Horizontal: p(p+1)^2 dofs — for each L2Seg point m
        // (xi = op[m]) and each RTTri(p) interior node (OpenPoints(p-1)
        // lattice, (x,y) = (iop[i]/w, iop[j]/w)), the (eta, zeta) component
        // pair at one point.  Vertical: p(p+1)(p+2)/2 — L2Tri nodes at
        // H1Seg indices 2..=p+1.
        if p >= 1 {
            let iop = open_points(p);
            for mm in 0..=p {
                for j in 0..p {
                    for i in 0..(p - j) {
                        let w = iop[i] + iop[j] + iop[p - 1 - i - j];
                        let eta = iop[i] / w;
                        let zeta = iop[j] / w;
                        coords.push(vec![op[mm], eta, zeta]); // eta-comp dof
                        coords.push(vec![op[mm], eta, zeta]); // zeta-comp dof
                    }
                }
            }
            for m in 2..=(p + 1) {
                for j in 0..=p {
                    for i in 0..=(p - j) {
                        let (eta, zeta) = l2_tri_node(j, i);
                        coords.push(vec![h1_seg_node(m, p), eta, zeta]);
                    }
                }
            }
        }
        coords
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// MFEM `RT_WedgeElement(p)` dof counts — probe
    /// `tmp/d444/probe_d444.out` (`dof=25` @ p=1, `dof=69` @ p=2) and the
    /// closed formula `(p+1)(3p²+12p+10)/2`.
    #[test]
    fn prism_rtk_dim_matches_mfem_rt_wedge() {
        assert_eq!(prism_rtk_dim(0), 5);
        assert_eq!(prism_rtk_dim(1), 25);
        assert_eq!(prism_rtk_dim(2), 69);
        assert_eq!(prism_rtk_dim(3), 146);
        assert_eq!(PrismRTk::new(1).n_dofs(), 25);
        assert_eq!(PrismRTk::new(2).n_dofs(), 69);
    }

    /// MFEM `RT_WedgeElement` interior count `p(p+1)(3p+4)/2`
    /// (`fe_coll.cpp:2581-2583`): 0, 7, 30, 78 at p = 0..3.
    #[test]
    fn wedge_interior_counts_match_mfem() {
        assert_eq!(wedge_interior_dofs(0), 0);
        assert_eq!(wedge_interior_dofs(1), 7);
        assert_eq!(wedge_interior_dofs(2), 30);
        assert_eq!(wedge_interior_dofs(3), 78);
    }

    /// Point on reference face `h` with canonical frame coords (u, v) and
    /// the outward normal / surface element of that face.
    fn face_point(h: usize, u: f64, v: f64) -> ([f64; 3], [f64; 3], f64) {
        match h {
            0 => ([0.0, u, v], [-1.0, 0.0, 0.0], 1.0),
            1 => ([1.0, u, v], [1.0, 0.0, 0.0], 1.0),
            2 => ([u, v, 0.0], [0.0, 0.0, -1.0], 1.0),
            3 => ([u, 1.0 - v, v], [0.0, 1.0, 1.0], std::f64::consts::SQRT_2),
            _ => ([u, 0.0, v], [0.0, -1.0, 0.0], 1.0),
        }
    }

    /// Exact normal-flux moments `∫_h (φ_slot·n̂_h)·q dA` of basis function
    /// `slot` over reference face `h`; `q` runs over the monomials {1, u, v}
    /// (the degree-≤1 face frame — exactly the RT1 dof functional set) and
    /// additionally {u², uv, v²} so the own-face check sees any nonzero
    /// trace of degree ≤ 2.
    fn face_flux_moments(e: &PrismRTk, slot: usize, h: usize) -> Vec<f64> {
        let mut phi = vec![0.0_f64; e.n_dofs() * 3];
        let mut mom = vec![0.0_f64; 6];
        let tests: [fn(f64, f64) -> f64; 6] = [
            |_, _| 1.0,
            |u, _| u,
            |_, v| v,
            |u, _| u * u,
            |u, v| u * v,
            |_, v| v * v,
        ];
        let rule = if h < 2 {
            tri_rule(12)
        } else {
            quad_rule_01(12)
        };
        for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
            let (x, n, ds) = face_point(h, pt[0], pt[1]);
            e.eval_basis_vec(&x, &mut phi);
            for (t, tc) in tests.iter().enumerate() {
                let q = tc(pt[0], pt[1]);
                for c in 0..3 {
                    mom[t] += w * ds * q * phi[slot * 3 + c] * n[c];
                }
            }
        }
        mom
    }

    /// D444 core property: every face group's basis functions carry
    /// vanishing normal-flux moments on all *other* faces and a nonzero
    /// own-face moment; interior basis functions vanish on all five faces.
    /// This is the slot ↔ face agreement the RT1 prism assembly pairing
    /// needs (the original D444 lesion had quad slots 2..4 moments on the
    /// wrong faces).
    #[test]
    fn prism_rtk1_slot_groups_are_face_conforming() {
        let e = PrismRTk::new(1);
        assert_eq!(e.n_dofs(), 25);
        for g in 0..5 {
            let range = e.slot_range(g);
            assert_eq!(range.len(), if g < 2 { 3 } else { 4 });
            for slot in range.clone() {
                for h in 0..5 {
                    let mom = face_flux_moments(&e, slot, h);
                    if h == g {
                        let max = mom.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                        assert!(
                            max > 1e-8,
                            "slot {slot} (face {g}): own-face moments vanish: {mom:?}"
                        );
                    } else {
                        // Foreign-face vanishing needs only the RT1 dof
                        // functional set {1, u, v} (moments 0..3); the dual
                        // solve leaves ~1e-8-level residuals (a structurally
                        // wrong slot↔face assignment gives O(1e-1..1)).
                        for (t, &mv) in mom.iter().take(3).enumerate() {
                            assert!(
                                mv.abs() < 1e-6,
                                "slot {slot} (face {g}): moment {t} on foreign face {h} = {mv}"
                            );
                        }
                    }
                }
            }
        }
        // Interior group: vanishing flux moments (RT1 functional set
        // {1, u, v}) on every face — the higher-degree residual of the
        // trace is unconstrained by the dual construction.
        for slot in e.slot_range(5) {
            for h in 0..5 {
                let mom = face_flux_moments(&e, slot, h);
                for (t, &mv) in mom.iter().take(3).enumerate() {
                    assert!(
                        mv.abs() < 1e-6,
                        "interior slot {slot}: moment {t} on face {h} = {mv}"
                    );
                }
            }
        }
    }

    /// `dof_coords` reproduce the MFEM `RT_WedgeElement(1)` Nodes table
    /// (probe `tmp/d444/probe_d444.out`) per slot group (sorted set
    /// comparison — the top tri face's internal enumeration is normalised,
    /// see module docs).
    #[test]
    fn prism_rtk1_dof_coords_match_mfem_probe() {
        const A: f64 = 0.211324865405187107; // Gauss(1) low
        const B: f64 = 0.788675134594812866; // Gauss(1) high
        const C: f64 = 0.174457630187009438; // L2Tri(1) lattice low
        const D: f64 = 0.651084739625981124; // L2Tri(1) lattice high
        let e = PrismRTk::new(1);
        let coords = e.dof_coords();
        assert_eq!(coords.len(), 25);
        let mut sorted: Vec<[f64; 3]> = coords.iter().map(|c| [c[0], c[1], c[2]]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected: Vec<[f64; 3]> = vec![
            // bottom tri (xi = 0)
            [0.0, C, C],
            [0.0, C, D],
            [0.0, D, C],
            // top tri (xi = 1)
            [1.0, C, C],
            [1.0, C, D],
            [1.0, D, C],
            // q0 (zeta = 0): (xi, eta) Gauss tensor
            [A, A, 0.0],
            [A, B, 0.0],
            [B, A, 0.0],
            [B, B, 0.0],
            // q1 (diagonal eta + zeta = 1): (xi, zeta) Gauss tensor
            [A, B, A],
            [A, A, B],
            [B, B, A],
            [B, A, B],
            // q2 (eta = 0): (xi, 1-zeta) Gauss tensor
            [A, 0.0, B],
            [A, 0.0, A],
            [B, 0.0, B],
            [B, 0.0, A],
            // interior horizontal: RTTri(1) interior node (1/3, 1/3) x L2Seg
            [A, 1.0 / 3.0, 1.0 / 3.0],
            [A, 1.0 / 3.0, 1.0 / 3.0],
            [B, 1.0 / 3.0, 1.0 / 3.0],
            [B, 1.0 / 3.0, 1.0 / 3.0],
            // interior vertical: L2Tri(1) nodes at H1Seg(2) node 2 (xi = 0.5)
            [0.5, C, C],
            [0.5, D, C],
            [0.5, C, D],
        ];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, (g, w)) in sorted.iter().zip(expected.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (g[c] - w[c]).abs() < 1e-14,
                    "coord {i} comp {c}: got {} want {} (got {g:?} want {w:?})",
                    g[c],
                    w[c]
                );
            }
        }
    }

    /// D597: the single-source RT0Wdg row table — consumed by the space
    /// engine (`HDivSpace::interp_rows(Prism6)`) and the prolongation builder
    /// (`hdiv_rt_slot_rows(Prism, 0)`) in place of their former hand copies —
    /// equals the MFEM probe values (face centroids/centres, tri nk halved to
    /// `n̂|F| = ±½`, quad nk the unnormalised generic table;
    /// `fe_fixed_order.cpp:6439`, probes `tmp/d572` / `tmp/d584`).
    #[test]
    fn mfem_nodal_rows_rt0_single_source() {
        let rows = mfem_nodal_rows();
        assert_eq!(rows.len(), PrismRTk::new(0).n_dofs());
        let want: Vec<([f64; 3], [f64; 3])> = vec![
            ([0.0, 1.0 / 3.0, 1.0 / 3.0], [-0.5, 0.0, 0.0]),
            ([1.0, 1.0 / 3.0, 1.0 / 3.0], [0.5, 0.0, 0.0]),
            ([0.5, 0.5, 0.0], [0.0, 0.0, -1.0]),
            ([0.5, 0.5, 0.5], [0.0, 1.0, 1.0]),
            ([0.5, 0.0, 0.5], [0.0, -1.0, 0.0]),
        ];
        assert_eq!(rows.len(), want.len());
        for (i, ((xi, nk), (wxi, wnk))) in rows.iter().zip(want.iter()).enumerate() {
            for d in 0..3 {
                assert!(
                    (xi[d] - wxi[d]).abs() < 1e-15 && (nk[d] - wnk[d]).abs() < 1e-15,
                    "row {i} comp {d}: ({xi:?}, {nk:?}) vs ({wxi:?}, {wnk:?})"
                );
            }
        }
    }

    /// Order-2/3 constructions stay finite (MFEM probe counts 69/146 pinned
    /// in [`prism_rtk_dim_matches_mfem_rt_wedge`]).
    #[test]
    fn prism_rtk_higher_orders_finite() {
        for p in 2..=3 {
            let e = PrismRTk::new(p);
            let mut v = vec![0.0; e.n_dofs() * 3];
            for pt in &e.quadrature(6).points {
                e.eval_basis_vec(pt, &mut v);
                assert!(v.iter().all(|x| x.is_finite()), "p={p}: non-finite basis");
            }
        }
    }

    /// k=0 carries MFEM `RT0WdgFiniteElement` (D572): the generic
    /// `RT_WedgeElement(0)` tensor basis with the two triangular-face slots
    /// doubled (`CalcVShape`, fe_fixed_order.cpp:6403) and div 2 on every
    /// dof (`CalcDivShape`, :6429).  Sample at (0.3, 0.2, 0.5): the old
    /// generic values (−0.7, 0.3 | 0.2,−0.5 | 0.2,0.5 | −0.8,0.5) double on
    /// slots 0,1.
    #[test]
    fn prism_rt0_is_mfem_rt0wdg() {
        let e = PrismRTk::new(0);
        assert_eq!(e.n_dofs(), 5);
        let mut v = vec![0.0; 15];
        e.eval_basis_vec(&[0.3, 0.2, 0.5], &mut v);
        assert_eq!(
            v,
            vec![
                -1.4, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.5, 0.0, 0.2, 0.5, 0.0, -0.8, 0.5
            ]
        );
        let mut d = vec![0.0; 5];
        e.eval_div(&[0.3, 0.2, 0.5], &mut d);
        assert_eq!(d, vec![2.0, 2.0, 2.0, 2.0, 2.0]);
    }
}
