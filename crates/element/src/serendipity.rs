use crate::quadrature::{hex_rule, quad_rule_01};
use crate::reference::{QuadratureRule, ReferenceElement};

/// Equispaced nodal lattice on `[0,1]` — the frame of **both** arms of this
/// module.
///
/// The hexahedral family of this crate lives on MFEM's `[0,1]³` reference cube
/// (D721 flipped `hex_rule`, `HexQ1` and `HexQk`; the straight-geometry mesh
/// frame is the unit cube), and D768 moved the 2-D serendipity arm onto the
/// matching `[0,1]²` frame — it was the last `[-1,1]²` lattice of this module.
/// Both MFEM 2-D elements this arm is measured against live on
/// `Geometry::SQUARE` = `[0,1]²`: `fe_ser.cpp`'s `H1Ser_QuadrilateralElement`
/// and `fe_fixed_order.cpp`'s `BiLinear2DFiniteElement` (the p = 1 member)
/// evaluate their shape functions directly in `[0,1]²`.  A `[-1,1]²` lattice
/// here would put every point of the rule the consumers pair with a reference
/// element (`quad_rule_01`, the D729/D738 PA/SumFact rule) outside the
/// element, scale the basis by 1/4 per dimension, and — exactly as in the D743
/// hex lesson — produce silently wrong values (1/16 of MFEM's bilinear mass
/// entries) instead of an error.
fn nd_01(p: usize) -> Vec<f64> {
    (0..=p).map(|i| i as f64 / p as f64).collect()
}

// Quad serendipity: monomials {x^i y^j : i=0 or i=p or j=0 or j=p}
fn mono_ij(p: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for j in 0..=p {
        for i in 0..=p {
            if i == 0 || i == p || j == 0 || j == p {
                v.push((i, j));
            }
        }
    }
    v
}
fn nodes_2d(p: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for j in 0..=p {
        for i in 0..=p {
            if i == 0 || i == p || j == 0 || j == p {
                v.push((i, j));
            }
        }
    }
    v
}
fn pow(x: f64, e: usize) -> f64 {
    if e == 0 {
        1.
    } else {
        x.powi(e as i32)
    }
}

fn build_coef2d(p: usize) -> Vec<f64> {
    let n = 4 * p;
    let xv = nd_01(p);
    let mi = mono_ij(p);
    let nds = nodes_2d(p);
    let mut v = vec![0.; n * n];
    for r in 0..n {
        let (ni, nj) = nds[r];
        // D768: monomials are evaluated in the element's own `[0,1]²`
        // coordinates (the old code shifted them into `[0,2]`, the `[-1,1]²`
        // arm's frame).
        let xi = xv[ni];
        let et = xv[nj];
        for c in 0..n {
            let (mi, mj) = mi[c];
            v[r * n + c] = pow(xi, mi) * pow(et, mj);
        }
    }
    let mut inv = vec![0.; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.;
    }
    let mut a = v;
    for c in 0..n {
        let mut mr = c;
        let mut mv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let x = a[r * n + c].abs();
            if x > mv {
                mv = x;
                mr = r
            }
        }
        for j in 0..n {
            a.swap(c * n + j, mr * n + j);
            inv.swap(c * n + j, mr * n + j);
        }
        let pv = a[c * n + c];
        let ip = 1. / pv;
        for j in 0..n {
            a[c * n + j] *= ip;
            inv[c * n + j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= f * a[c * n + j];
                inv[r * n + j] -= f * inv[c * n + j];
            }
        }
    }
    let mut c = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = inv[j * n + i];
        }
    }
    c
}

/// Serendipity quadrilateral of order `p` on MFEM's **`[0,1]²` unit square**
/// (D768; the 2-D sibling of [`HexSerendipityPk`]): `4p` DOFs on the equispaced
/// `i/p` lattice, interpolating the truncated tensor space
/// `{x^i y^j : i ∈ {0,p} ∨ j ∈ {0,p}}`.  `p = 1` is MFEM's
/// `BiLinear2DFiniteElement` (`fe_fixed_order.cpp:115`, bit-for-bit, via the
/// same factorised formulas).
///
/// **Audit note (D768)** — this element is *not* MFEM's
/// `H1Ser_QuadrilateralElement` (`fe_ser.cpp:25`), the only serendipity class
/// MFEM ships: MFEM's element uses the **Gauss-Lobatto** lattice,
/// `(p² + 3p + 6)/2` DOFs (interior *bubble* — non-nodal — functions appear at
/// `p ≥ 4`) and the genuine serendipity space `S_p = {x^i y^j : sd(i,j) ≤ p}`
/// (`sd` = superlinear degree; `xy` ∈ `S_2`, `x²y²` ∉).  This crate's element
/// is the `[0,1]²` analogue of the D743 hex arm: equispaced lattice, `4p`
/// nodal DOFs, superlinear degree `2p`-truncated tensor space.  The two agree
/// in DOF count and node positions only at `p ≤ 2`, and their spans are
/// different spaces from `p = 2` on, so no pointwise ≤1e-12 match with MFEM's
/// serendipity class is possible for `p ≥ 2` — measured in
/// `crates/element/tests/d768_quad_serendipity.rs`, which pins the p = 1
/// bit-for-bit equality that *is* available.
pub struct QuadSerendipityPk {
    p: usize,
    co: Vec<f64>,
    mi: Vec<(usize, usize)>,
    nds: Vec<(usize, usize)>,
}

impl QuadSerendipityPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        Self {
            p,
            co: build_coef2d(p),
            mi: mono_ij(p),
            nds: nodes_2d(p),
        }
    }
}

impl ReferenceElement for QuadSerendipityPk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        4 * self.p
    }
    fn eval_basis(&self, xi: &[f64], vals: &mut [f64]) {
        let n = self.n_dofs();
        // p = 1 *is* MFEM's `BiLinear2DFiniteElement` (`fe_fixed_order.cpp:115`,
        // `Geometry::SQUARE` = `[0,1]²`): the four factorised products are used
        // directly so the p = 1 member is bit-for-bit the MFEM element instead
        // of the LU-solved monomial interpolant of it (the D743 hex lesson).
        // The monomial machinery below is only needed from p = 2 on.
        if self.p == 1 {
            let (x, y) = (xi[0], xi[1]);
            let (ox, oy) = (1.0 - x, 1.0 - y);
            // Lattice slot `s` is the corner `(s & 1, (s >> 1) & 1)`:
            // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1).
            vals[0] = ox * oy;
            vals[1] = x * oy;
            vals[2] = ox * y;
            vals[3] = x * y;
            return;
        }
        for i in 0..n {
            let mut s = 0.;
            for j in 0..n {
                let (mi, mj) = self.mi[j];
                s += self.co[i * n + j] * pow(xi[0], mi) * pow(xi[1], mj);
            }
            vals[i] = s;
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let n = self.n_dofs();
        // Same p = 1 specialisation as `eval_basis`: MFEM
        // `BiLinear2DFiniteElement::CalcDShape` (fe_fixed_order.cpp:124),
        // verbatim per vertex — including its `-1.+y` / `1.-y` spellings, which
        // is what keeps the products bit-identical — permuted into the lattice
        // slot order (MFEM's vertex order is (0,0),(1,0),(1,1),(0,1)).
        if self.p == 1 {
            let (x, y) = (xi[0], xi[1]);
            let dx = [-1. + y, 1. - y, -y, y];
            let dy = [-1. + x, -x, 1. - x, x];
            for s in 0..4 {
                grads[2 * s] = dx[s];
                grads[2 * s + 1] = dy[s];
            }
            return;
        }
        let u = xi[0];
        let v = xi[1];
        for i in 0..n {
            let mut sx = 0.;
            let mut sy = 0.;
            for j in 0..n {
                let (mi, mj) = self.mi[j];
                let mx = if mi == 0 {
                    0.
                } else {
                    mi as f64 * pow(u, mi - 1) * pow(v, mj)
                };
                let my = if mj == 0 {
                    0.
                } else {
                    mj as f64 * pow(u, mi) * pow(v, mj - 1)
                };
                sx += self.co[i * n + j] * mx;
                sy += self.co[i * n + j] * my;
            }
            grads[i * 2] = sx;
            grads[i * 2 + 1] = sy;
        }
    }
    /// MFEM's `IntRules.Get(Geometry::SQUARE, order)` — the `[0,1]²` rule the
    /// element's own frame requires (D768; `quad_rule_01`'s order → point-count
    /// mapping is MFEM's, D738).
    fn quadrature(&self, o: u8) -> QuadratureRule {
        quad_rule_01(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let xv = nd_01(self.p);
        let mut c = Vec::new();
        for &(i, j) in &self.nds {
            c.push(vec![xv[i], xv[j]]);
        }
        c
    }
}

// Hex serendipity on MFEM's `[0,1]³` (D743)
fn mi3(p: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for k in 0..=p {
        for j in 0..=p {
            for i in 0..=p {
                if i == 0 || i == p || j == 0 || j == p || k == 0 || k == p {
                    v.push((i, j, k));
                }
            }
        }
    }
    v
}
fn nd3(p: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for k in 0..=p {
        for j in 0..=p {
            for i in 0..=p {
                if i == 0 || i == p || j == 0 || j == p || k == 0 || k == p {
                    v.push((i, j, k));
                }
            }
        }
    }
    v
}

fn build_hex(p: usize) -> Vec<f64> {
    let n = {
        let e = p.saturating_sub(1);
        8 + 12 * e + 6 * e * e
    };
    let xv = nd_01(p);
    let m = mi3(p);
    let d = nd3(p);
    let mut v = vec![0.; n * n];
    for r in 0..n {
        let (ni, nj, nk) = d[r];
        let (xi, et, zt) = (xv[ni], xv[nj], xv[nk]);
        for c in 0..n {
            let (mi, mj, mk) = m[c];
            v[r * n + c] = pow(xi, mi) * pow(et, mj) * pow(zt, mk);
        }
    }
    let mut inv = vec![0.; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.;
    }
    let mut a = v;
    for c in 0..n {
        let mut mr = c;
        let mut mv = a[c * n + c].abs();
        for r in (c + 1)..n {
            let x = a[r * n + c].abs();
            if x > mv {
                mv = x;
                mr = r
            }
        }
        for j in 0..n {
            a.swap(c * n + j, mr * n + j);
            inv.swap(c * n + j, mr * n + j);
        }
        let pv = a[c * n + c];
        if pv.abs() < 1e-14 {
            continue;
        }
        let ip = 1. / pv;
        for j in 0..n {
            a[c * n + j] *= ip;
            inv[c * n + j] *= ip;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            for j in 0..n {
                a[r * n + j] -= f * a[c * n + j];
                inv[r * n + j] -= f * inv[c * n + j];
            }
        }
    }
    let mut cc = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            cc[i * n + j] = inv[j * n + i];
        }
    }
    cc
}

/// Serendipity hexahedron of order `p` on **MFEM's `[0,1]³` unit cube**
/// (D743; the round-70 D721 flip re-based `hex_rule`/`HexQ1`/`HexQk` on that
/// frame, and this arm follows them): `8 + 12(p-1) + 6(p-1)²` DOFs on the
/// equispaced `i/p` lattice, interpolating the truncated tensor space
/// `{x^i y^j z^k : i ∈ {0,p} ∨ j ∈ {0,p} ∨ k ∈ {0,p}}`.  `p = 1` is MFEM's
/// `TriLinear3DFiniteElement` (bit-for-bit, via the same factorised formulas as
/// `HexQ1`); `p ≥ 2` has no MFEM counterpart (MFEM's `fe_ser.hpp` carries only
/// the 2-D `H1Ser_QuadrilateralElement`, and its `H1_FECollection` maps every
/// hexahedral cell type to `H1_HexahedronElement(p)`).
pub struct HexSerendipityPk {
    p: usize,
    co: Vec<f64>,
    mi: Vec<(usize, usize, usize)>,
    nds: Vec<(usize, usize, usize)>,
}
impl HexSerendipityPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        Self {
            p,
            co: build_hex(p),
            mi: mi3(p),
            nds: nd3(p),
        }
    }
    fn n(&self) -> usize {
        let e = self.p.saturating_sub(1);
        8 + 12 * e + 6 * e * e
    }
}

impl ReferenceElement for HexSerendipityPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        self.n()
    }
    fn eval_basis(&self, xi: &[f64], vals: &mut [f64]) {
        let n = self.n();
        // p = 1 *is* MFEM's `TriLinear3DFiniteElement` (`fe_fixed_order.cpp`),
        // which the hex family already carries verbatim as `HexQ1` (D721): the
        // eight factorised products are used directly so the p = 1 member is
        // bit-for-bit the MFEM element instead of the LU-solved monomial
        // interpolant of it.  The monomial machinery below is only needed from
        // p = 2 on, where the truncated tensor space has no MFEM counterpart.
        if self.p == 1 {
            let (x, y, z) = (xi[0], xi[1], xi[2]);
            let (ox, oy, oz) = (1.0 - x, 1.0 - y, 1.0 - z);
            // Slot `s` is the lattice corner `(s & 1, (s >> 1) & 1, (s >> 2) & 1)`;
            // each value is the same left-associative product of the three
            // 1-D factors MFEM's `CalcShape` forms.
            let fx = [ox, x];
            let fy = [oy, y];
            let fz = [oz, z];
            for s in 0..8 {
                let (i, j, k) = (s & 1, (s >> 1) & 1, (s >> 2) & 1);
                vals[s] = fx[i] * fy[j] * fz[k];
            }
            return;
        }
        let u = xi[0];
        let v = xi[1];
        let w = xi[2];
        for i in 0..n {
            let mut s = 0.;
            for j in 0..n {
                let (mi, mj, mk) = self.mi[j];
                s += self.co[i * n + j] * pow(u, mi) * pow(v, mj) * pow(w, mk);
            }
            vals[i] = s;
        }
    }
    fn eval_grad_basis(&self, xi: &[f64], g: &mut [f64]) {
        let n = self.n();
        // Same p = 1 specialisation as `eval_basis`: MFEM
        // `TriLinear3DFiniteElement::CalcDShape`, verbatim (D721's `HexQ1`),
        // permuted into the serendipity lattice slot order.
        if self.p == 1 {
            let (x, y, z) = (xi[0], xi[1], xi[2]);
            let (ox, oy, oz) = (1.0 - x, 1.0 - y, 1.0 - z);
            let fx = [ox, x];
            let fy = [oy, y];
            let fz = [oz, z];
            let dfx = [-1.0_f64, 1.0];
            let dfy = [-1.0_f64, 1.0];
            let dfz = [-1.0_f64, 1.0];
            for s in 0..8 {
                let (i, j, k) = (s & 1, (s >> 1) & 1, (s >> 2) & 1);
                let (px, py, pz) = (fx[i], fy[j], fz[k]);
                g[3 * s] = dfx[i] * py * pz;
                g[3 * s + 1] = px * dfy[j] * pz;
                g[3 * s + 2] = px * py * dfz[k];
            }
            return;
        }
        let u = xi[0];
        let v = xi[1];
        let w = xi[2];
        for i in 0..n {
            let mut sx = 0.;
            let mut sy = 0.;
            let mut sz = 0.;
            for j in 0..n {
                let (mi, mj, mk) = self.mi[j];
                let mx = if mi == 0 {
                    0.
                } else {
                    (mi as f64) * pow(u, mi - 1) * pow(v, mj) * pow(w, mk)
                };
                let my = if mj == 0 {
                    0.
                } else {
                    (mj as f64) * pow(u, mi) * pow(v, mj - 1) * pow(w, mk)
                };
                let mz = if mk == 0 {
                    0.
                } else {
                    (mk as f64) * pow(u, mi) * pow(v, mj) * pow(w, mk - 1)
                };
                sx += self.co[i * n + j] * mx;
                sy += self.co[i * n + j] * my;
                sz += self.co[i * n + j] * mz;
            }
            g[i * 3] = sx;
            g[i * 3 + 1] = sy;
            g[i * 3 + 2] = sz;
        }
    }
    /// MFEM's `IntRules.Get(Geometry::CUBE, order)` — the `[0,1]³` rule the
    /// whole hex family consumes since D721, on the element's own frame.
    fn quadrature(&self, o: u8) -> QuadratureRule {
        hex_rule(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let xv = nd_01(self.p);
        let mut c = Vec::new();
        for &(i, j, k) in &self.nds {
            c.push(vec![xv[i], xv[j], xv[k]]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pou(e: &dyn ReferenceElement) {
        let q = e.quadrature(4);
        let mut p = vec![0.; e.n_dofs()];
        for pt in &q.points {
            // D768: the module's frame is `[0,1]^dim` (both arms) — a
            // `[-1,1]`-frame rule would evaluate the basis outside the element.
            for c in pt.iter() {
                assert!(
                    (-1e-15..=1.0 + 1e-15).contains(c),
                    "quadrature point outside the reference element: {pt:?}"
                );
            }
            e.eval_basis(pt, &mut p);
            let s: f64 = p.iter().sum();
            assert!((s - 1.).abs() < 1e-12, "POU={s}");
        }
    }
    fn interp(e: &dyn ReferenceElement) {
        let c = e.dof_coords();
        let n = e.n_dofs();
        let mut p = vec![0.; n];
        for (i, cc) in c.iter().enumerate() {
            e.eval_basis(cc, &mut p);
            for j in 0..n {
                let exp = if i == j { 1. } else { 0. };
                assert!((p[j] - exp).abs() < 1e-12);
            }
        }
    }
    fn grad(e: &dyn ReferenceElement, pts: &[[f64; 2]]) {
        let h = 1e-7;
        let n = e.n_dofs();
        let (mut vc, mut vx, mut vy, mut g) =
            (vec![0.; n], vec![0.; n], vec![0.; n], vec![0.; n * 2]);
        for p in pts {
            let (x, y) = (p[0], p[1]);
            e.eval_basis(&[x, y], &mut vc);
            e.eval_basis(&[x + h, y], &mut vx);
            e.eval_basis(&[x, y + h], &mut vy);
            e.eval_grad_basis(&[x, y], &mut g);
            for i in 0..n {
                assert!((g[i * 2] - (vx[i] - vc[i]) / h).abs() < 1e-5);
                assert!((g[i * 2 + 1] - (vy[i] - vc[i]) / h).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn q1() {
        assert_eq!(QuadSerendipityPk::new(1).n_dofs(), 4);
        pou(&QuadSerendipityPk::new(1));
        interp(&QuadSerendipityPk::new(1));
    }
    #[test]
    fn q2() {
        assert_eq!(QuadSerendipityPk::new(2).n_dofs(), 8);
        pou(&QuadSerendipityPk::new(2));
        interp(&QuadSerendipityPk::new(2));
        grad(&QuadSerendipityPk::new(2), &[[0.2, 0.3], [0.7, 0.9]]);
    }
    #[test]
    fn q3() {
        assert_eq!(QuadSerendipityPk::new(3).n_dofs(), 12);
        pou(&QuadSerendipityPk::new(3));
        interp(&QuadSerendipityPk::new(3));
    }
    #[test]
    fn q4() {
        assert_eq!(QuadSerendipityPk::new(4).n_dofs(), 16);
        pou(&QuadSerendipityPk::new(4));
        interp(&QuadSerendipityPk::new(4));
    }
    /// D768 — the 2-D arm's frame is `[0,1]²`: nodes, quadrature points and the
    /// p = 1 corner labels all sit inside the unit square.  Red before the
    /// migration, where `QuadSerendipityPk::new(1).dof_coords()` was
    /// `[(-1,-1), (0,-1), (-1,0), (0,0)]` and the rule was `quad_rule`.
    #[test]
    fn q_frame_is_unit_square() {
        use crate::quadrature::{quad_rule, quad_rule_01};
        for p in 1..=4 {
            let e = QuadSerendipityPk::new(p);
            for c in e.dof_coords() {
                assert!(
                    (0.0..=1.0).contains(&c[0]) && (0.0..=1.0).contains(&c[1]),
                    "p={p}: dof coordinate {c:?} outside [0,1]²"
                );
            }
            for o in 2..=6u8 {
                let q = e.quadrature(o);
                for pt in &q.points {
                    assert!(
                        (0.0..=1.0).contains(&pt[0]) && (0.0..=1.0).contains(&pt[1]),
                        "p={p} o={o}: rule point {pt:?} outside [0,1]²"
                    );
                }
                // The rule is MFEM's SQUARE rule (D738's order → point count),
                // not the `[-1,1]²` `quad_rule`.
                assert_eq!(q.points.len(), quad_rule_01(o).points.len());
                assert_ne!(q.weights[0], quad_rule(o).weights[0]);
            }
        }
        // p = 1: the four unit-square corners in the module's lattice order.
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
    /// D768 — `∫_{[0,1]²} φ_i` with the element's own rule.  p = 1 gives 1/4
    /// per vertex (the bilinear corner subcell); before the migration the same
    /// integral over the crate's `[0,1]²` rule was 1/16 of that, because the
    /// basis carried the `[-1,1]²` frame's 1/4-per-dimension scaling.
    #[test]
    fn q1_mass_integrals_on_unit_square() {
        let e = QuadSerendipityPk::new(1);
        let q = e.quadrature(3);
        let mut phi = vec![0.0; 4];
        let mut int = [0.0_f64; 4];
        for (qi, pt) in q.points.iter().enumerate() {
            e.eval_basis(pt, &mut phi);
            for i in 0..4 {
                int[i] += q.weights[qi] * phi[i];
            }
        }
        for (i, v) in int.iter().enumerate() {
            assert!(
                (v - 0.25).abs() < 1e-15,
                "∫φ_{i} over [0,1]² = {v}, expected 1/4 (bilinear corner subcell)"
            );
        }
        let total: f64 = int.iter().sum();
        assert!((total - 1.0).abs() < 1e-15, "Σ∫φ = {total}");
    }
    #[test]
    fn h1() {
        assert_eq!(HexSerendipityPk::new(1).n_dofs(), 8);
        pou(&HexSerendipityPk::new(1));
        interp(&HexSerendipityPk::new(1));
    }
    #[test]
    fn h2() {
        assert_eq!(HexSerendipityPk::new(2).n_dofs(), 26);
        pou(&HexSerendipityPk::new(2));
        interp(&HexSerendipityPk::new(2));
    }
    #[test]
    fn h3() {
        assert_eq!(HexSerendipityPk::new(3).n_dofs(), 56);
        pou(&HexSerendipityPk::new(3));
    }
}
