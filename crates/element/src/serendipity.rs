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

fn pow(x: f64, e: usize) -> f64 {
    if e == 0 {
        1.
    } else {
        x.powi(e as i32)
    }
}

/// Serendipity quadrilateral of order `p` on MFEM's **`[0,1]²` unit square**
/// — a faithful port of MFEM 4.10's `H1Ser_QuadrilateralElement`
/// (`fem/fe/fe_ser.cpp:25`, the only 2-D serendipity class MFEM ships; it is
/// reached through `H1_FECollection(p, 2, BasisType::Serendipity)`,
/// `fe_coll.cpp:1844`).
///
/// **D809 (round 75)** — the D768 audit below had established that this arm
/// was *not* MFEM's element: it used the equispaced `i/p` lattice, `4p` nodal
/// DOFs and the truncated tensor span `{xⁱyʲ : i ∈ {0,p} ∨ j ∈ {0,p}}`.  MFEM's
/// class is a genuinely different construction, now reproduced here:
///
/// * **DOF count** `(p² + 3p + 6)/2` — `5/8/12/17/23` for `p = 1..5`
///   (`(pm3·pm2)/2` *interior* DOFs plus the `4p` boundary ones,
///   `fe_coll.cpp:1850`).  The old `4p` matches this only for `p = 2, 3`;
///   from `p = 4` the serendipity space carries interior **bubble** DOFs
///   (`p = 4` → 1, `p = 5` → 3).
/// * **Nodes** `poly1d.ClosedPoints(p, GaussLobatto)` — the *Gauss-Lobatto*
///   closed lattice mapped onto `[0,1]`.  For `n = p+1` points this is
///   `{0, 1}` for `p = 1`, `{0, ½, 1}` for `p = 2` (the GLL interior node of
///   the 3-point rule *is* the midpoint, so `p = 2` coincides with the old
///   equispaced lattice) and genuinely clustered from `p = 3`
///   (`0.2764/0.7236` at `p = 3`).
/// * **Span** the real serendipity space `S_p = {xⁱyʲ : sd(i,j) ≤ p}` (`sd` =
///   superlinear degree): `xy ∈ S_2` but `x²y² ∉`.  The old span had it
///   backwards (`x²y²` in, `xy` out).
/// * **Slot order** vertices `(0,0),(1,0),(1,1),(0,1)`, then south → east →
///   north → west edge DOFs, then the interior bubbles — MFEM's order, not the
///   old row-major lattice order.
/// * **Nodal?** `p ≤ 3` only.  From `p = 4` the interior functions are the
///   Legendre-product bubbles `P_k(x)P_{j-4-k}(y)·x(1-x)y(1-y)`, which are
///   *not* nodal (`fe_ser.cpp:110`).
///
/// * **`p = 1` is kept as the 4-DOF bilinear** (`BiLinear2DFiniteElement`,
///   bit-for-bit).  MFEM's own `H1Ser_QuadrilateralElement(1)` is degenerate:
///   the DOF formula gives 5, the constructor's edge loop is empty, so the
///   fifth shape function is **identically zero** — measured, see
///   `tests/d809_mfem_h1ser_port.rs` (`shape 1 4 = 0` at every sample point,
///   and `kron p=1` = 1.0, i.e. not even nodal).  MFEM's ordinary
///   `H1_FECollection(1)` — what a `Quad4`/`Quad8` mesh is actually built
///   with — uses the bilinear, which is the useful member and the one D768
///   pinned.  The old docs' `4p` claim therefore survives at `p = 1` and `p = 2`
///   only by coincidence of the count formula.
pub struct QuadSerendipityPk {
    p: usize,
    /// Gauss-Lobatto closed points on `[0,1]` — MFEM
    /// `poly1d.ClosedPoints(p, GaussLobatto)`.
    cp: Vec<f64>,
    /// The 1-D GLL nodal basis on `[-1,1]` — MFEM
    /// `poly1d.GetBasis(p, GaussLobatto)`.  Evaluated at `2x−1` for a point
    /// `x ∈ [0,1]`.
    lag: crate::lagrange::factory::Lagrange1D,
    nd: usize,
}

impl QuadSerendipityPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        // GLL closed points on [-1,1] → [0,1] (MFEM's Poly_1D lives on [0,1]).
        let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
        let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
        let lag = crate::lagrange::factory::Lagrange1D::new(p);
        let nd = if p == 1 { 4 } else { (p * p + 3 * p + 6) / 2 };
        Self { p, cp, lag, nd }
    }

    /// MFEM `Poly_1D::CalcLegendre(p, x, u)`: shifted Legendre `P_0..P_p` at
    /// `x ∈ [0,1]` (the recursion runs on `z = 2x−1`), `fe_base.cpp:2343`.
    fn legendre(&self, p: usize, x: f64) -> Vec<f64> {
        let mut u = vec![0.0_f64; p + 1];
        u[0] = 1.0;
        if p == 0 {
            return u;
        }
        let z = 2.0 * x - 1.0;
        u[1] = z;
        for n in 1..p {
            let (nf, zf) = (n as f64, z);
            u[n + 1] = ((2.0 * nf + 1.0) * zf * u[n] - nf * u[n - 1]) / (nf + 1.0);
        }
        u
    }

    /// MFEM `Poly_1D::CalcLegendre(p, x, u, d)`: values and **d/dx**
    /// derivatives (note `d[1] = 2`, the `dz/dx` factor), `fe_base.cpp:2357`.
    fn legendre_d(&self, p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
        let mut u = vec![0.0_f64; p + 1];
        let mut d = vec![0.0_f64; p + 1];
        u[0] = 1.0;
        if p == 0 {
            return (u, d);
        }
        let z = 2.0 * x - 1.0;
        u[1] = z;
        d[1] = 2.0;
        for n in 1..p {
            let (nf, zf) = (n as f64, z);
            u[n + 1] = ((2.0 * nf + 1.0) * zf * u[n] - nf * u[n - 1]) / (nf + 1.0);
            d[n + 1] = (4.0 * nf + 2.0) * u[n] + d[n - 1];
        }
        (u, d)
    }

    /// The interior bubble count and slot base — MFEM's
    /// `4 + 4·(p−1) + interior_total` (`fe_ser.cpp:161`).
    fn interior_base(&self) -> usize {
        4 + 4 * (self.p - 1)
    }

    /// MFEM's interior slot list in fill order: `(j, k)` with
    /// `j ∈ 4..=p`, `k ∈ 0..j−3` (`fe_ser.cpp:162`).
    fn interior_slots(&self) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        if self.p > 3 {
            for j in 4..=self.p {
                for k in 0..(j - 3) {
                    v.push((j, k));
                }
            }
        }
        v
    }

    /// MFEM `H1Ser_QuadrilateralElement`'s node table: the tensor
    /// `Sr_DOF_MAP` (`fe_base.cpp:2515`) reduced to the serendipity DOFs —
    /// vertices, then south/east/north/west edge DOFs, then the interior
    /// tensor slots in `j`-major order (`fe_ser.cpp:38-53`).
    fn nodes(&self) -> Vec<[f64; 2]> {
        let p = self.p;
        if p == 1 {
            // The 4-DOF bilinear member: MFEM `BiLinear2DFiniteElement` vertex
            // order, which is this arm's historical slot order.
            return vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        }
        let cp = &self.cp;
        let mut n = Vec::with_capacity(self.nd);
        n.push([cp[0], cp[0]]);
        n.push([cp[p], cp[0]]);
        n.push([cp[p], cp[p]]);
        n.push([cp[0], cp[p]]);
        for i in 1..p {
            n.push([cp[i], cp[0]]); // south
        }
        for i in 1..p {
            n.push([cp[p], cp[i]]); // east
        }
        for i in 1..p {
            n.push([cp[p - i], cp[p]]); // north
        }
        for i in 1..p {
            n.push([cp[0], cp[p - i]]); // west
        }
        // Interior: the first `nd - 4p` tensor slots of the `j`-major interior
        // loop, i.e. `(cp[i], cp[1])` for `i = 1, 2, …` (Sr_DOF_MAP).
        let extra = self.nd - 4 * p;
        for t in 0..extra {
            n.push([cp[1 + t], cp[1]]);
        }
        n
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
        self.nd
    }
    fn eval_basis(&self, xi: &[f64], vals: &mut [f64]) {
        // p = 1 *is* MFEM's `BiLinear2DFiniteElement` (`fe_fixed_order.cpp:115`,
        // `Geometry::SQUARE` = `[0,1]²`): the four factorised products are used
        // directly so the p = 1 member is bit-for-bit the MFEM element.
        // (MFEM's `H1Ser_QuadrilateralElement(1)` is degenerate — 5 DOFs with
        // an identically zero fifth shape — so the bilinear is the faithful
        // choice for a `Quad4`/`Quad8` order-1 reference element.)
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
        let p = self.p;
        let (x, y) = (xi[0], xi[1]);
        // MFEM `poly1d.GetBasis(p, GaussLobatto).Eval` — the 1-D GLL nodal
        // basis, evaluated at the `[0,1]` point through `z = 2x−1`.
        let nx = self.lag.val(2.0 * x - 1.0);
        let ny = self.lag.val(2.0 * y - 1.0);
        // Edge DOFs: a nodal interpolant on the edge, weighted by the linear
        // function that vanishes on the opposite edge (`fe_ser.cpp:66-75`).
        for i in 0..p - 1 {
            vals[4 + i] = nx[i + 1] * (1.0 - y); // south (y = 0)
            vals[4 + (p - 1) + i] = ny[i + 1] * x; // east  (x = 1)
            vals[4 + 3 * (p - 1) - i - 1] = nx[i + 1] * y; // north (y = 1)
            vals[4 + 4 * (p - 1) - i - 1] = ny[i + 1] * (1.0 - x); // west (x = 0)
        }
        // Interior bubbles (p ≥ 4), `fe_ser.cpp:110-127`.
        let slots = self.interior_slots();
        if !slots.is_empty() {
            let leg_x = self.legendre(p - 2, x);
            let leg_y = self.legendre(p - 2, y);
            let base = self.interior_base();
            let w = x * (1.0 - x) * y * (1.0 - y);
            for (t, &(j, k)) in slots.iter().enumerate() {
                vals[base + t] = leg_x[k] * leg_y[j - 4 - k] * w;
            }
        }
        // Vertex DOFs: the bilinear minus the incident edge corrections
        // (`fe_ser.cpp:88-105`).  MFEM's `BiLinear2DFiniteElement` order is
        // (0,0), (1,0), (1,1), (0,1); the `1 − edgePts[i+1]` weights are the
        // bilinear of *that* vertex at the (slot-reversed) edge node, which is
        // why the north/west reads look permuted.
        let bil = [(1.0 - x) * (1.0 - y), x * (1.0 - y), x * y, (1.0 - x) * y];
        let (mut f0, mut f1, mut f2, mut f3) = (0.0_f64, 0.0, 0.0, 0.0);
        for i in 0..p - 1 {
            let w = 1.0 - self.cp[i + 1];
            f0 += w * (vals[4 + i] + vals[4 + 4 * (p - 1) - i - 1]);
            f1 += w * (vals[4 + (p - 1) + i] + vals[4 + (p - 2) - i]);
            f2 += w * (vals[4 + 2 * (p - 1) + i] + vals[1 + 2 * p - i]);
            f3 += w * (vals[4 + 3 * (p - 1) + i] + vals[3 * p - i]);
        }
        vals[0] = bil[0] - f0;
        vals[1] = bil[1] - f1;
        vals[2] = bil[2] - f2;
        vals[3] = bil[3] - f3;
    }
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
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
        let p = self.p;
        let (x, y) = (xi[0], xi[1]);
        let (nx, dnx) = self.lag.val_d(2.0 * x - 1.0);
        let (ny, dny) = self.lag.val_d(2.0 * y - 1.0);
        // `lag.val_d` differentiates w.r.t. its own ([-1,1]) argument, so the
        // chain rule contributes a factor 2 (`fe_ser.cpp:141-155` uses
        // `edgeNodalBasis.Eval(x, nodalX, DnodalX)` directly at `x ∈ [0,1]`,
        // which is the same value).
        for i in 0..p - 1 {
            let (vx, dx_, vy, dy_) = (nx[i + 1], 2.0 * dnx[i + 1], ny[i + 1], 2.0 * dny[i + 1]);
            let s = 4 + i;
            grads[2 * s] = dx_ * (1.0 - y);
            grads[2 * s + 1] = -vx;
            let e = 4 + (p - 1) + i;
            grads[2 * e] = vy;
            grads[2 * e + 1] = dy_ * x;
            let nn = 4 + 3 * (p - 1) - i - 1;
            grads[2 * nn] = dx_ * y;
            grads[2 * nn + 1] = vx;
            let w = 4 + 4 * (p - 1) - i - 1;
            grads[2 * w] = -vy;
            grads[2 * w + 1] = dy_ * (1.0 - x);
        }
        // Interior bubbles (`fe_ser.cpp:180-206`).
        let slots = self.interior_slots();
        if !slots.is_empty() {
            let (leg_x, dleg_x) = self.legendre_d(p - 2, x);
            let (leg_y, dleg_y) = self.legendre_d(p - 2, y);
            let base = self.interior_base();
            for (t, &(j, k)) in slots.iter().enumerate() {
                let (kx, ky) = (leg_x[k], leg_y[j - 4 - k]);
                let (dkx, dky) = (dleg_x[k], dleg_y[j - 4 - k]);
                grads[2 * (base + t)] =
                    ky * y * (1.0 - y) * (dkx * x * (1.0 - x) + kx * (1.0 - 2.0 * x));
                grads[2 * (base + t) + 1] =
                    kx * x * (1.0 - x) * (dky * y * (1.0 - y) + ky * (1.0 - 2.0 * y));
            }
        }
        // Vertex gradients: bilinear minus the incident edge corrections
        // (`fe_ser.cpp:157-186`).
        let dbil = [
            [-(1.0 - y), -(1.0 - x)],
            [1.0 - y, -x],
            [y, x],
            [-y, 1.0 - x],
        ];
        for k in 0..4 {
            grads[2 * k] = dbil[k][0];
            grads[2 * k + 1] = dbil[k][1];
        }
        for i in 0..p - 1 {
            let w = 1.0 - self.cp[i + 1];
            let pairs = [
                (4 + i, 4 + 4 * (p - 1) - i - 1),
                (4 + (p - 1) + i, 4 + (p - 2) - i),
                (4 + 2 * (p - 1) + i, 1 + 2 * p - i),
                (4 + 3 * (p - 1) + i, 3 * p - i),
            ];
            for (v, &(a, b)) in pairs.iter().enumerate() {
                grads[2 * v] -= w * (grads[2 * a] + grads[2 * b]);
                grads[2 * v + 1] -= w * (grads[2 * a + 1] + grads[2 * b + 1]);
            }
        }
    }
    /// MFEM's `IntRules.Get(Geometry::SQUARE, order)` — the `[0,1]²` rule the
    /// element's own frame requires (D768; `quad_rule_01`'s order → point-count
    /// mapping is MFEM's, D738).
    fn quadrature(&self, o: u8) -> QuadratureRule {
        quad_rule_01(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes().into_iter().map(|n| n.to_vec()).collect()
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
        // D809: MFEM's `(p²+3p+6)/2` — 17 at p = 4 (16 boundary + 1 interior
        // bubble DOF).  The partition of unity and nodality checks do *not*
        // apply from p = 4 on: MFEM's own `H1Ser_QuadrilateralElement` loses
        // both (`Σ∫φ = 1.0278`, POU residual 6.25e-2 — D809-1), and this arm
        // reproduces that faithfully.  Both are pinned per-order in
        // `tests/d809_mfem_h1ser_port.rs`, against the MFEM dump.
        assert_eq!(QuadSerendipityPk::new(4).n_dofs(), 17);
        grad(&QuadSerendipityPk::new(4), &[[0.2, 0.3], [0.7, 0.9]]);
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
