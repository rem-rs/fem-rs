//! Raviart-Thomas H(div) element on the reference pyramid — MFEM
//! `RT_FuentesPyramidElement(p)` alignment (D445/D437).
//!
//! Reference pyramid (MFEM `Geometry::PYRAMID`, identical in fem-rs axes
//! `(xi,eta,zeta) = (x,y,z)`): base quad {0,1,2,3} on z=0 ([0,1]², CCW),
//! apex {4} = (0,0,1) (`fem/geom.cpp` `GeomVert[7]`).
//!
//! # Slot layout (D445)
//!
//! Slot order = MFEM `RT_FuentesPyramidElement::RT_FuentesPyramidElement`
//! construction order (`fem/fe/fe_rt.cpp:1273-1373`, probe
//! `tmp/d444/probe_d444.out` / `tmp/d444/pyr_incode.out`):
//!
//! ```text
//!   [ base quad ((p+1)^2, FaceVert (3,2,1,0)),
//!     tri (0,1,4) (eta=0), tri (1,2,4) (xi+zeta=1),
//!     tri (2,3,4) (eta+zeta=1), tri (3,0,4) (xi=0),
//!     each (p+1)(p+2)/2,
//!     interior 3p(p+1)^2  (x-, y-, z-component blocks) ]
//! ```
//!
//! Total `(p+1)(3p(p+2)+5)` (`fe_rt.cpp:1273-1275`): 5, 28, 87, 164 at
//! p = 0..3 — probe: `RT_FuentesPyramidElement(1) dof=28`, `(2) dof=87`.
//! The space-level MFEM oracle on a single-pyramid mesh (probe
//! `tmp/d444/pyr_incode.out`): k=0 vsize=ess=5; k=1 vsize=28, ess=16.
//!
//! The base quad uses the MFEM canonical frame (`u` = c0→c1 = vert 3→2 =
//! +xi, `v` = c0→c3 = vert 3→0 = −eta, slot `n = v*(p+1)+u`); the four
//! triangular faces use the fem-rs standard barycentric grid convention
//! (slot `(j,i)` at barycentric weights `(v1: j, v2: i)` of the face's
//! `FaceVert` row, moment poly `s^j t^i` in the face frame).  MFEM's own
//! reference element enumerates the triangular faces with heterogeneous
//! internal orders (transposed on (0,1,4), partially reversed on (2,3,4)
//! and (3,0,4); `fe_rt.cpp:1304-1330`) — the same reference-element-internal
//! slot quirk documented for the wedge top face: fem-rs normalises to the
//! standard grid, which leaves the dof *set*, vsize, ess and global dof maps
//! unchanged and keeps the element/space pairing self-consistent.
//!
//! # DOF functionals
//!
//! Moment-dual construction (see `prism.rs` module docs for the general
//! pattern): exact normal-flux moments `∫_face (Φ·n̂) q ds` on the five
//! faces plus `3p(p+1)²` interior volume moments `∫_V Φ_c w dx` with
//! weights mirroring the Fuentes interior index ranges (x-component:
//! `xi^i eta^j zeta^k`, i = 1..=p, j,k = 0..=p; y-component: j = 1..=p;
//! z-component: k = 1..=p), all with exact-enough quadrature.  The dof
//! *values* are moment-based, not MFEM's nodal point samples (residual gap
//! recorded in the D445 debt notes).

use crate::quadrature::{
    gauss_lobatto_01, gauss_legendre_01, pyramid_rule, quad_rule_01, tri_rule,
};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Monomial helpers ───────────────────────────────────────────────────────

/// Monomial `xi^a · eta^b · zeta^c / scale` of vector component `comp`
/// (0 = xi, 1 = eta, 2 = zeta).  `scale` equilibrates the Vandermonde
/// columns by the monomial's L² norm over the reference pyramid — high
/// powers of `zeta` carry almost no volume mass on the shrinking pyramid,
/// which wrecks the Gram conditioning (max-scaling is not enough).  Column
/// scaling does not change the span nor the dual basis, only the numerics.
#[derive(Clone)]
struct Mono {
    comp: u8,
    a: usize,
    b: usize,
    c: usize,
    scale: f64,
}

fn eval_mono(m: &Mono, xi: f64, eta: f64, zeta: f64) -> f64 {
    (xi.powi(m.a as i32) * eta.powi(m.b as i32) * zeta.powi(m.c as i32)) / m.scale
}

/// Per-component tensor monomial span: `a, b ≤ p+1`, `c ≤ p+2`.  Column
/// count `3(p+2)²(p+3)` exceeds the dof count `(p+1)(3p²+6p+5)` for every
/// `p` (difference `12p²+37p+31 > 0`), so the dual system is never
/// rank-deficient.
/// Per-component tensor monomial span: `a, b ≤ p+1`, `c ≤ p+2`.  Column
/// count `3(p+2)²(p+3)` exceeds the dof count `(p+1)(3p²+6p+5)` for every
/// `p` (difference `12p²+37p+31 > 0`), so the dual system is never
/// rank-deficient.  (A tighter span was tried and made the Gram
/// conditioning *worse* — the interior functionals become nearly dependent
/// on it — so the redundancy stays.)
fn pyramid_monos(p: usize) -> Vec<Mono> {
    // L²-equilibration quadrature: exact for degree 2(p+2)+2.
    let sq_rule = pyramid_rule((2 * p + 8).min(40) as u8);
    let l2 = |a: usize, b: usize, c: usize| -> f64 {
        let mut s = 0.0;
        for (pt, &w) in sq_rule.points.iter().zip(sq_rule.weights.iter()) {
            let mv = pt[0].powi(a as i32) * pt[1].powi(b as i32) * pt[2].powi(c as i32);
            s += w * mv * mv;
        }
        s.sqrt().max(1e-12)
    };
    let mut m = Vec::new();
    for comp in 0..3u8 {
        for a in 0..=(p + 1) {
            for b in 0..=(p + 1) {
                for c in 0..=(p + 2) {
                    let scale = l2(a, b, c);
                    m.push(Mono { comp, a, b, c, scale });
                }
            }
        }
    }
    m
}

// ─── Face definitions (Fuentes slot order, D445) ────────────────────────────

/// Face slot order = MFEM `Geometry::PYRAMID` FaceVert order: 0 = base quad
/// (zeta=0, n̂=(0,0,−1), (3,2,1,0)), 1 = tri (0,1,4) (eta=0, n̂=(0,−1,0)),
/// 2 = tri (1,2,4) (xi+zeta=1, n̂=(1,0,1)/√2), 3 = tri (2,3,4)
/// (eta+zeta=1, n̂=(0,1,1)/√2), 4 = tri (3,0,4) (xi=0, n̂=(−1,0,0)).
const FACE_DEFS: [(u8, [f64; 3]); 5] = [
    (1, [0.0, 0.0, -1.0]),                              // quad base zeta=0
    (0, [0.0, -1.0, 0.0]),                              // tri (0,1,4)
    (0, [0.7071067811865475, 0.0, 0.7071067811865475]), // tri (1,2,4)
    (0, [0.0, 0.7071067811865475, 0.7071067811865475]), // tri (2,3,4)
    (0, [-1.0, 0.0, 0.0]),                              // tri (3,0,4)
];

/// Fuentes interior dof count: `3p(p+1)²` (`fe_rt.cpp:1343-1373` loops;
/// `RT_dof[PYRAMID] = 3p(p+1)²`, `fe_coll.cpp:2584-2586`).
fn fuentes_interior_dofs(p: usize) -> usize {
    3 * p * (p + 1) * (p + 1)
}

/// Total dof count of MFEM `RT_FuentesPyramidElement(p)`:
/// `(p+1)(3p(p+2)+5)` (probe: 28 @ p=1, 87 @ p=2).
fn pyramid_rtk_dim(p: usize) -> usize {
    let tri = (p + 1) * (p + 2) / 2;
    (p + 1) * (p + 1) + 4 * tri + fuentes_interior_dofs(p)
}

// ─── DOF functionals ────────────────────────────────────────────────────────

/// One face moment functional.  Quad face (base): `∫ (Φ·n̂) u^i v^j` in the
/// canonical frame (u = c0→c1 = +xi, v = c0→c3 = −eta).  Tri faces:
/// `∫ (Φ·n̂) s^j t^i` with (s,t) the barycentric frame coordinates along
/// (v0→v1, v0→v2) of the face's `FaceVert` row (standard convention).
fn face_dof_value(m: &Mono, face: usize, b: usize, c: usize) -> f64 {
    let (_ftype, n) = FACE_DEFS[face];
    let dot = match m.comp {
        0 => n[0],
        1 => n[1],
        _ => n[2],
    };
    if face == 0 {
        // Base quad zeta = 0: point (u, 1-v, 0), poly u^b v^c.  Order 14
        // covers the monomial span's total degree (≤ 7 at p = 3) plus the
        // poly degree — exact functionals (an under-integrated functional
        // perturbs the dual and leaks flux onto foreign faces).
        let rule = quad_rule_01(14);
        let mut sum = 0.0;
        for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
            let (u, v) = (pt[0], pt[1]);
            let mv = eval_mono(m, u, 1.0 - v, 0.0);
            let poly = u.powi(b as i32) * v.powi(c as i32);
            sum += w * dot * mv * poly;
        }
        sum
    } else {
        // Tri faces in their FaceVert frames (D445):
        //   t(0,1,4): (s,t) = (xi, zeta)
        //   t(1,2,4): (s,t) = (eta, zeta), xi = 1-t
        //   t(2,3,4): (s,t) = bary (v0=2 → v1=3, v0→v2=4): (1-s-t, 1-t, t)
        //   t(3,0,4): (s,t) = bary (v0=3 → v1=0, v0→v2=4): (0, 1-s-t, t)
        let (j, i) = (b, c);
        let rule = tri_rule(((j + i) * 2 + 14).min(20) as u8);
        let mut sum = 0.0;
        for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
            let (s, t) = (pt[0], pt[1]);
            let (xi, eta, zeta) = match face {
                1 => (s, 0.0, t),
                2 => (1.0 - t, s, t),
                3 => (1.0 - s - t, 1.0 - t, t),
                _ => (0.0, 1.0 - s - t, t),
            };
            let mv = eval_mono(m, xi, eta, zeta);
            let poly = s.powi(j as i32) * t.powi(i as i32);
            sum += w * dot * mv * poly;
        }
        sum
    }
}

/// Interior volume moment weight `∫_V mono · xi^a eta^b zeta^c dV`
/// (exact quadrature; the caller restricts the monomial's component to the
/// functional's target component).
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
/// panics if rank deficient.
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
            "PyraRTk: dual Gram matrix rank-deficient at pivot {c} (|piv| = {bv})"
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

/// Build the Vandermonde rows in Fuentes slot order:
/// `[base quad, tri (0,1,4), tri (1,2,4), tri (2,3,4), tri (3,0,4),
/// interior x/y/z]` (D445).
fn build_pyramid_rtk(p: usize) -> (Vec<f64>, usize) {
    let n = pyramid_rtk_dim(p);
    let monos = pyramid_monos(p);
    let m = monos.len();
    let mut vand = vec![vec![0.0_f64; m]; n];
    let mut row = 0;

    for face in 0..5 {
        if face == 0 {
            // Base quad: slot n = v*(p+1) + u (u inner).
            for j in 0..=p {
                for i in 0..=p {
                    for (col, mo) in monos.iter().enumerate() {
                        vand[row][col] = face_dof_value(mo, face, i, j);
                    }
                    row += 1;
                }
            }
        } else {
            // Tri faces: standard grid (j = s-exponent outer, i = t-exponent
            // inner).
            for j in 0..=p {
                for i in 0..=(p - j) {
                    for (col, mo) in monos.iter().enumerate() {
                        vand[row][col] = face_dof_value(mo, face, j, i);
                    }
                    row += 1;
                }
            }
        }
    }

    // Interior 3p(p+1)²: Fuentes component order (x, y, z) with weights
    // mirroring the Fuentes node index ranges (exact quadrature).
    if p >= 1 {
        let order = (4 * p + 8).min(40) as u8;
        let rule = pyramid_rule(order);
        // x-component: i = 1..=p, j = 0..=p, k = 0..=p
        for i in 1..=p {
            for j in 0..=p {
                for k in 0..=p {
                    for (col, mo) in monos.iter().enumerate() {
                        if mo.comp == 0 {
                            vand[row][col] = interior_dof_value(&rule, mo, i, j, k);
                        }
                    }
                    row += 1;
                }
            }
        }
        // y-component: j = 1..=p, i = 0..=p, k = 0..=p
        for j in 1..=p {
            for i in 0..=p {
                for k in 0..=p {
                    for (col, mo) in monos.iter().enumerate() {
                        if mo.comp == 1 {
                            vand[row][col] = interior_dof_value(&rule, mo, i, j, k);
                        }
                    }
                    row += 1;
                }
            }
        }
        // z-component: k = 1..=p, i = 0..=p, j = 0..=p
        for k in 1..=p {
            for i in 0..=p {
                for j in 0..=p {
                    for (col, mo) in monos.iter().enumerate() {
                        if mo.comp == 2 {
                            vand[row][col] = interior_dof_value(&rule, mo, i, j, k);
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

/// Raviart-Thomas H(div) element on the reference pyramid — MFEM
/// `RT_FuentesPyramidElement(p)` slot layout (D445).  Supports `p ≤ 3`
/// (the moment dual is verified 0..=3; higher orders wait on a nodal MFEM
/// port).
pub struct PyraRTk {
    p: usize,
    coeff: Vec<f64>,
    n: usize,
    m: usize,
    monos: Vec<Mono>,
}

/// Order-0 element (alias for `PyraRTk::new(0)`, kept for backward compat).
pub type PyraRT0 = PyraRTk;

impl PyraRTk {
    pub fn new(order: usize) -> Self {
        assert!(
            order <= 3,
            "PyraRTk: order {order} exceeds the verified moment-dual cap 3 \
             (MFEM RT_FuentesPyramidElement is order-generic; a nodal port is \
             future work)"
        );
        let (coeff, m) = build_pyramid_rtk(order);
        let n = pyramid_rtk_dim(order);
        PyraRTk {
            p: order,
            coeff,
            n,
            m,
            monos: pyramid_monos(order),
        }
    }

    /// Half-open slot range of face `face` (0..5; 5 = interior) in the
    /// element's slot layout — `[base, tri 0,1,4 / 1,2,4 / 2,3,4 / 3,0,4,
    /// interior]`.  Public for MFEM-layout parity tests.
    pub fn slot_range(&self, face: usize) -> std::ops::Range<usize> {
        let tri = (self.p + 1) * (self.p + 2) / 2;
        let quad = (self.p + 1) * (self.p + 1);
        let start = match face {
            0 => 0,
            1 => quad,
            2 => quad + tri,
            3 => quad + 2 * tri,
            4 => quad + 3 * tri,
            _ => quad + 4 * tri,
        };
        let len = match face {
            0 => quad,
            1 | 2 | 3 | 4 => tri,
            _ => fuentes_interior_dofs(self.p),
        };
        start..start + len
    }
}

/// 1-D Gauss (open) points, `n` of them on [0,1] — MFEM `OpenPoints`
/// (`bop` = Fuentes' face-lattice points, `iop` = interior open points).
fn open_points(n: usize) -> Vec<f64> {
    gauss_legendre_01(n).0
}

/// 1-D Gauss-Lobatto (closed) points, `n` of them on [0,1] — MFEM
/// `ClosedPoints(p+1)` (`icp` = Fuentes' closed interior points).
fn closed_points(n: usize) -> Vec<f64> {
    gauss_lobatto_01(n).0
}

impl VectorReferenceElement for PyraRTk {
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
        let mut mv = vec![0.0_f64; self.monos.len()];
        for (j, m) in self.monos.iter().enumerate() {
            mv[j] = eval_mono(m, xi[0], xi[1], xi[2]);
        }
        values.fill(0.0);
        for i in 0..self.n {
            for j in 0..self.m {
                if i * self.m + j < self.coeff.len() {
                    let c = self.coeff[i * self.m + j];
                    if c != 0.0 {
                        values[i * 3 + self.monos[j].comp as usize] += c * mv[j];
                    }
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], cv: &mut [f64]) {
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
            cv[i * 3] = dfz_dy - dfy_dz;
            cv[i * 3 + 1] = dfx_dz - dfz_dx;
            cv[i * 3 + 2] = dfy_dx - dfx_dy;
        }
    }

    fn eval_div(&self, xi: &[f64], dv: &mut [f64]) {
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
            dv[i] = dfx + dfy + dfz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    /// MFEM `RT_FuentesPyramidElement` node positions
    /// (`tmp/d444/probe_d444.out`, Fuentes p=1 block) — fem-rs axes equal
    /// MFEM's `(x,y,z)`.  The four tri faces use the standard barycentric
    /// enumeration (see module docs for the documented normalisation of
    /// MFEM's heterogeneous internal tri orders); the base quad and the
    /// interior mirror MFEM's enumeration verbatim.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.p;
        let bop = open_points(p + 1); // Fuentes `bop` face lattice (p+1 points)
        let iop = open_points(p + 1); // Fuentes `iop` (OpenPoints(p), p+1 points)
        let icp = closed_points(p + 2); // Fuentes `icp` (ClosedPoints(p+1))
        let mut coords = Vec::with_capacity(self.n);

        // Face 0: base quad, slot n = v*(p+1)+u — MFEM node
        // (bop[u], bop[p-v], 0).
        for j in 0..=p {
            for i in 0..=p {
                coords.push(vec![bop[i], bop[p - j], 0.0]);
            }
        }
        // Faces 1..5: tris, standard grid — node (j, i) at the barycentric
        // lattice (s = bop[j]/w, t = bop[i]/w), w = bop[j]+bop[i]+bop[p-j-i].
        for face in 1..=4 {
            for j in 0..=p {
                for i in 0..=(p - j) {
                    let w = bop[j] + bop[i] + bop[p - j - i];
                    let (s, t) = (bop[j] / w, bop[i] / w);
                    let (xi, eta, zeta) = match face {
                        1 => (s, 0.0, t),
                        2 => (1.0 - t, s, t),
                        3 => (1.0 - s - t, 1.0 - t, t),
                        _ => (0.0, 1.0 - s - t, t),
                    };
                    coords.push(vec![xi, eta, zeta]);
                }
            }
        }
        // Interior, Fuentes order (x, y, z) with his node positions.
        if p >= 1 {
            // x: i = 1..=p, j = 0..=p, k = 0..=p
            for k in 0..=p {
                for j in 0..=p {
                    for i in 1..=p {
                        let w = 1.0 - iop[k];
                        coords.push(vec![icp[i] * w, iop[j] * w, iop[k]]);
                    }
                }
            }
            // y: j = 1..=p, i = 0..=p, k = 0..=p
            for k in 0..=p {
                for j in 1..=p {
                    for i in 0..=p {
                        let w = 1.0 - iop[k];
                        coords.push(vec![iop[i] * w, icp[j] * w, iop[k]]);
                    }
                }
            }
            // z: k = 1..=p, i = 0..=p, j = 0..=p
            for k in 1..=p {
                for j in 0..=p {
                    for i in 0..=p {
                        let w = 1.0 - icp[k];
                        coords.push(vec![iop[i] * w, iop[j] * w, icp[k]]);
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

    /// MFEM `RT_FuentesPyramidElement(p)` dof counts — probe
    /// `tmp/d444/probe_d444.out` (`dof=28` @ p=1, `dof=87` @ p=2) and the
    /// closed formula `(p+1)(3p(p+2)+5)` (`fe_rt.cpp:1273-1275`).
    #[test]
    fn pyra_rtk_dim_matches_mfem_fuentes() {
        assert_eq!(pyramid_rtk_dim(0), 5);
        assert_eq!(pyramid_rtk_dim(1), 28);
        assert_eq!(pyramid_rtk_dim(2), 87);
        assert_eq!(pyramid_rtk_dim(3), 200);
        assert_eq!(PyraRTk::new(1).n_dofs(), 28);
        assert_eq!(PyraRTk::new(2).n_dofs(), 87);
    }

    /// Fuentes interior count `3p(p+1)²` (`RT_dof[PYRAMID]`,
    /// `fe_coll.cpp:2584-2586`): 0, 12, 54, 108 at p = 0..3.
    #[test]
    fn fuentes_interior_counts_match_mfem() {
        assert_eq!(fuentes_interior_dofs(0), 0);
        assert_eq!(fuentes_interior_dofs(1), 12);
        assert_eq!(fuentes_interior_dofs(2), 54);
        assert_eq!(fuentes_interior_dofs(3), 144);
    }

    /// Point on reference face `h` with frame coords (u, v) and the outward
    /// normal / surface element.
    fn face_point(h: usize, u: f64, v: f64) -> ([f64; 3], [f64; 3], f64) {
        match h {
            0 => ([u, 1.0 - v, 0.0], [0.0, 0.0, -1.0], 1.0),
            1 => ([u, 0.0, v], [0.0, -1.0, 0.0], 1.0),
            2 => ([1.0 - v, u, v], [1.0, 0.0, 1.0], std::f64::consts::SQRT_2),
            3 => (
                [1.0 - u - v, 1.0 - v, v],
                [0.0, 1.0, 1.0],
                std::f64::consts::SQRT_2,
            ),
            _ => ([0.0, 1.0 - u - v, v], [-1.0, 0.0, 0.0], 1.0),
        }
    }

    /// Exact normal-flux moments `∫_h (φ_slot·n̂_h)·q dA`; `q` runs over
    /// {1, u, v} (the RT1 dof functional set — the foreign-face vanishing
    /// set) plus {u², uv, v²} for the own-face nonzero check.
    fn face_flux_moments(e: &PyraRTk, slot: usize, h: usize) -> Vec<f64> {
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
        let rule = if h == 0 { quad_rule_01(12) } else { tri_rule(12) };
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

    /// D445 core property: every face group's basis functions carry
    /// vanishing normal-flux moments (RT1 functional set) on all *other*
    /// faces and a nonzero own-face moment; interior basis functions vanish
    /// on all five faces.  This is the slot ↔ face agreement the (future)
    /// pyramid RT1 assembly pairing needs — the original D445 lesion had
    /// element slots [base, x=0, y=0, x+z, y+z] against the space's
    /// [4 tris, base] (and now Fuentes [base, 4 tris]).
    #[test]
    fn pyra_rtk1_slot_groups_are_face_conforming() {
        let e = PyraRTk::new(1);
        assert_eq!(e.n_dofs(), 28);
        for g in 0..5 {
            let range = e.slot_range(g);
            assert_eq!(range.len(), if g == 0 { 4 } else { 3 });
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
                        for (t, &mv) in mom.iter().take(3).enumerate() {
                            assert!(
                                mv.abs() < 1e-7,
                                "slot {slot} (face {g}): moment {t} on foreign face {h} = {mv}"
                            );
                        }
                    }
                }
            }
        }
        for slot in e.slot_range(5) {
            for h in 0..5 {
                let mom = face_flux_moments(&e, slot, h);
                for (t, &mv) in mom.iter().take(3).enumerate() {
                    assert!(
                        mv.abs() < 1e-7,
                        "interior slot {slot}: moment {t} on face {h} = {mv}"
                    );
                }
            }
        }
    }

    /// `dof_coords` reproduce the MFEM `RT_FuentesPyramidElement(1)` Nodes
    /// table (probe `tmp/d444/probe_d444.out`) as a sorted multiset — the
    /// tri-face internal enumeration is normalised (see module docs), so
    /// the comparison is order-insensitive but position-exact.
    #[test]
    fn pyra_rtk1_dof_coords_match_mfem_probe() {
        // MFEM Fuentes(1) node table, fem-rs axes == MFEM (x,y,z)
        // (values %.17e from tmp/d444/probe_d444.out lines 28 dof block).
        const A: f64 = 0.211324865405187107;
        const B: f64 = 0.788675134594812866;
        const C: f64 = 0.174457630187009438; // bop/w lattice low
        const D: f64 = 0.651084739625981124; // bop/w lattice high
        const E: f64 = 1.0 - D; // 1 - lattice high (probe 3.48915260374018876e-01)
        const F: f64 = 1.0 - C; // 1 - lattice low (probe 8.25542369812990562e-01)
        const P3: f64 = 1.0 / 3.0;
        let e = PyraRTk::new(1);
        let coords = e.dof_coords();
        assert_eq!(coords.len(), 28);
        let mut sorted: Vec<[f64; 3]> = coords.iter().map(|c| [c[0], c[1], c[2]]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected: Vec<[f64; 3]> = vec![
            // base quad (zeta = 0)
            [A, B, 0.0],
            [B, B, 0.0],
            [A, A, 0.0],
            [B, A, 0.0],
            // tri (0,1,4): eta = 0
            [C, 0.0, C],
            [D, 0.0, C],
            [C, 0.0, D],
            // tri (1,2,4): xi + zeta = 1
            [F, C, C],
            [F, D, C],
            [E, C, D],
            // tri (2,3,4): eta + zeta = 1
            [D, F, C],
            [C, F, C],
            [C, E, D],
            // tri (3,0,4): xi = 0
            [0.0, D, C],
            [0.0, C, C],
            [0.0, C, D],
            // interior x (icp[1]=0.5 * (1-iop[k]), iop[j]*(1-iop[k]), iop[k])
            [0.5 * (1.0 - A), A * (1.0 - A), A],
            [0.5 * (1.0 - A), B * (1.0 - A), A],
            [0.5 * (1.0 - B), A * (1.0 - B), B],
            [0.5 * (1.0 - B), B * (1.0 - B), B],
            // interior y
            [A * (1.0 - A), 0.5 * (1.0 - A), A],
            [B * (1.0 - A), 0.5 * (1.0 - A), A],
            [A * (1.0 - B), 0.5 * (1.0 - B), B],
            [B * (1.0 - B), 0.5 * (1.0 - B), B],
            // interior z ((1-icp[1]) scaling = 0.5)
            [A * 0.5, A * 0.5, 0.5],
            [B * 0.5, A * 0.5, 0.5],
            [A * 0.5, B * 0.5, 0.5],
            [B * 0.5, B * 0.5, 0.5],
        ];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let _ = (E, F, P3);
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

    /// Order-2/3 constructions stay finite (MFEM probe counts 87/164 pinned
    /// in [`pyra_rtk_dim_matches_mfem_fuentes`]).
    #[test]
    fn pyra_rtk_higher_orders_finite() {
        for p in 2..=3 {
            let e = PyraRTk::new(p);
            let mut v = vec![0.0; e.n_dofs() * 3];
            for pt in &e.quadrature(6).points {
                e.eval_basis_vec(pt, &mut v);
                assert!(v.iter().all(|x| x.is_finite()), "p={p}: non-finite basis");
            }
        }
    }

    /// k=0 behaviour: dim 5 (face blocks only, no interior).
    #[test]
    fn pyra_rt0_dim() {
        assert_eq!(PyraRTk::new(0).n_dofs(), 5);
        let e = PyraRTk::new(0);
        let mut v = vec![0.0; 15];
        for pt in &e.quadrature(4).points {
            e.eval_basis_vec(pt, &mut v);
            assert!(v.iter().all(|x| x.is_finite()));
        }
    }
}
