//! MFEM 4.10's `L2_FuentesPyramidElement` — the **default** L2/DG pyramid
//! element (`ScalarPyramid::DefaultType = 1`, `fem/fe/fe_pyramid.hpp:23`,
//! selected by `L2_FECollection`'s `pyr_type`, `fe_coll.cpp:2340-2351`).
//!
//! 1:1 port of `fem/fe/fe_l2.cpp:926-1076` plus the shared
//! `FuentesPyramid` helpers it calls:
//!
//! * nodes — `fe_l2.cpp:963-973`:
//!   `op = Poly_1D::OpenPoints(p, VerifyOpen(btype))` and
//!   `node_o = (op[i](1 − a·op[k]), op[j](1 − a·op[k]), a·op[k])` for
//!   `o = k(p+1)² + j(p+1) + i`.  `a = 1` for an open `btype` (the
//!   `L2_FECollection` default `BasisType::GaussLegendre`); for a **closed**
//!   `btype` (`VerifyOpen` accepts `GaussLobatto` too —
//!   `Quadrature1D::CheckOpen` returns every type, `intrules.cpp:1137-1151`)
//!   `a` is the largest Gauss-Legendre node of the order-`p` rule
//!   (`fe_l2.cpp:944-952`), which is what keeps the point set open in `z`:
//!   MFEM's comment there states the basis is *not independent* on closed
//!   interpolation points for `p ≥ 1`.
//! * basis — `fe_l2.cpp:1002-1039`: the raw tensor expansion
//!   `u_o = sca_x[i]·sca_y[j]·sca_z[k]` with
//!   `sca_x = CalcHomogenizedScaLegendre(p, mu0(z,xy,1), mu1(z,xy,1))`,
//!   `sca_y` likewise with `ab = 2`, `sca_z = CalcHomogenizedScaLegendre(p,
//!   mu0(z), mu1(z))`; the nodal basis is `φ_m = Σ_o T⁻¹(m,o) u_o` with
//!   `T(o,m) = u_o(node_m)` (`Ti.Factor(T)`, `fe_l2.cpp:990`).
//! * gradients — `fe_l2.cpp:1041-1076`, evaluated *directly* (not through
//!   `CalcScaledLegendre`) because the `1/(1−z)` and `1/(1−z)²` chain factors
//!   are explicit there.
//!
//! DOF count is `(p+1)³` (1, 8, 27, 64, 125 for `p = 0..4`), unlike either
//! Bergot arm (`(p+1)(p+2)(2p+3)/6` = 5, 14, 30) — see the D306 note on
//! [`super::pyramid::PyramidPk`], which is *neither* MFEM L2 arm.
//!
//! Reference pyramid: the fem-rs/MFEM reference pyramid
//! `(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1)` with `x ∈ [0,1−z]`,
//! `y ∈ [0,1−z]`, so MFEM's values are reproduced **without any rescaling**
//! (measured, not assumed — `crates/element/tests/d325_l2_fuentes_pyramid.rs`
//! checks the raw quotient of every shape value against the MFEM dump).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::pyramid::calc_legendre_d;
use super::pyramid_fuentes::{calc_scaled_legendre, mu0_xy, mu0_z, mu1_xy, mu1_z};
use crate::quadrature::pyramid_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

/// DOF count of MFEM's `L2_FuentesPyramidElement(p)`: `(p+1)³`
/// (`fe_l2.cpp:927`).
pub const fn l2_fuentes_pyramid_n_dofs(p: usize) -> usize {
    (p + 1) * (p + 1) * (p + 1)
}

/// `FuentesPyramid::CalcHomogenizedScaLegendre(p, s0, s1, u)` (value-only
/// overload, `fe_pyramid.cpp:447-451`) = `CalcScaledLegendre(p, s1, s0 + s1, u)`.
fn homogenized_sca_legendre(p: usize, s0: f64, s1: f64) -> Vec<f64> {
    calc_scaled_legendre(p, s1, s0 + s1).0
}

/// Gauss-Legendre nodes on `[0,1]`, ascending — MFEM `Poly_1D::OpenPoints(p)`
/// (`fe_base.hpp:1193` → `Poly_1D::GetPoints`, `fe_base.cpp:2440-2468` →
/// `QuadratureFunctions1D::GaussLegendre`).
fn open_points(p: usize) -> Vec<f64> {
    crate::quadrature::gauss_legendre_01_arbitrary(p + 1).0
}

/// Gauss-Lobatto nodes on `[0,1]`, ascending — the closed counterpart MFEM
/// reaches through `Poly_1D::GetPoints(p, BasisType::GaussLobatto)`.
///
/// `p = 0` is the single-point case (`n = 1`), where MFEM's
/// `QuadratureFunctions1D::GivePolyPoints(1, ·, GaussLobatto)` yields the
/// midpoint `0.5` — the same point as the open table (measured: the probe's
/// `ORDER p=0 ... closed=1` block has `OP 1 / 0.5`).
fn closed_points(p: usize) -> Vec<f64> {
    if p == 0 {
        return vec![0.5];
    }
    let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
    g.iter().map(|&x| 0.5 * (x + 1.0)).collect()
}

/// Node table of MFEM `L2_FuentesPyramidElement(p, btype)` in DOF order
/// (`fe_l2.cpp:963-973`): `o = k(p+1)² + j(p+1) + i` and
/// `node_o = (op[i](1 − a·op[k]), op[j](1 − a·op[k]), a·op[k])`.
///
/// `closed` selects `btype`: `false` is `BasisType::GaussLegendre` (the
/// `L2_FECollection` default), `true` is a closed `btype` such as
/// `BasisType::GaussLobatto`; see [`l2_fuentes_a_factor`].
pub fn l2_fuentes_pyramid_nodes(p: usize, closed: bool) -> Vec<[f64; 3]> {
    let op = if closed { closed_points(p) } else { open_points(p) };
    let a = l2_fuentes_a_factor(p, closed);
    let mut n = Vec::with_capacity(l2_fuentes_pyramid_n_dofs(p));
    for k in 0..=p {
        let zk = a * op[k];
        let s = 1.0 - zk;
        for j in 0..=p {
            for i in 0..=p {
                n.push([op[i] * s, op[j] * s, zk]);
            }
        }
    }
    n
}

/// MFEM's `z`-collapse factor `a` (`fe_l2.cpp:944-952`): `1.0` for an open
/// `btype`, else the largest Gauss-Legendre node of the order-`p` rule when
/// `p > 0`.
///
/// MFEM's comment there is the reason the closed arm exists at all: the basis
/// is *not independent* on closed interpolation points for `p ≥ 1`, so a
/// closed request is forced open in `z` (`a < 1`), which is a limitation of
/// this element family rather than of the request.
pub fn l2_fuentes_a_factor(p: usize, closed: bool) -> f64 {
    if closed && p > 0 {
        open_points(p)[p]
    } else {
        1.0
    }
}

/// Raw (`pre-Vandermonde`) expansion `u_o` of the L2 Fuentes pyramid at
/// `(x, y, z)` — `fe_l2.cpp:1002-1032` — in the tensor DOF order
/// `o = k(p+1)² + j(p+1) + i`.
///
/// The `z == 1` guard is MFEM's: the `x`/`y` blocks collapse to `e₀` where the
/// collapsed coordinates are singular (the apex).  The `z` block is always
/// evaluated through the homogenized Legendre expansion.
///
/// Public for the same reason [`super::pyramid_fuentes::fuentes_raw_basis`] is:
/// the fixture's Vandermonde is exactly this function sampled at the element's
/// own nodes, so the parity test can rebuild it without reaching into private
/// state.
pub fn l2_fuentes_raw_basis(p: usize, x: f64, y: f64, z: f64, u: &mut [f64]) {
    let xy = [x, y];
    let (sx, sy) = if z < 1.0 {
        (
            homogenized_sca_legendre(p, mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)),
            homogenized_sca_legendre(p, mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)),
        )
    } else {
        let mut ex = vec![0.0; p + 1];
        let mut ey = vec![0.0; p + 1];
        ex[0] = 1.0;
        ey[0] = 1.0;
        (ex, ey)
    };
    let sz = homogenized_sca_legendre(p, mu0_z(z), mu1_z(z));
    let mut o = 0;
    for k in 0..=p {
        for j in 0..=p {
            for i in 0..=p {
                u[o] = sx[i] * sy[j] * sz[k];
                o += 1;
            }
        }
    }
}

struct L2FuentesPyramidPkInner {
    order: usize,
    closed: bool,
    nodes: Vec<[f64; 3]>,
    /// `φ_m = Σ_o ti[m·n + o] · u_o(x)` — row-major `T⁻¹`
    /// (`fe_l2.cpp:990`, `Ti.Factor(T)`).
    ti: Vec<f64>,
}

/// MFEM `L2_FuentesPyramidElement(p, btype)` — the default L2 pyramid element
/// of `L2_FECollection` (`pyr_type = ScalarPyramid::DefaultType = 1`),
/// `(p+1)³` DOFs.
///
/// Two constructions, matching MFEM's `btype` split:
///
/// * [`L2FuentesPyramidPk::new`] — `btype = BasisType::GaussLegendre`, the
///   `L2_FECollection` default: **open** Gauss-Legendre nodes in every
///   direction, `a = 1`.
/// * [`L2FuentesPyramidPk::new_gauss_lobatto`] — `btype =
///   BasisType::GaussLobatto`: Gauss-Lobatto `op` in `x`/`y` and the collapsed
///   `z` factor `a =` the largest order-`p` GL node, so the point set stays
///   **open in `z`** (MFEM's documented limitation, `fe_l2.cpp:937-943`).
///
/// The DOF order is MFEM's `L2_DOF_MAP` tensor order
/// (`o = k(p+1)² + j(p+1) + i`, `i` fastest) — the same order
/// `L2Space` numbers its per-element DOFs in.
pub struct L2FuentesPyramidPk {
    inner: Arc<L2FuentesPyramidPkInner>,
}

impl L2FuentesPyramidPk {
    /// MFEM `L2_FuentesPyramidElement(p, BasisType::GaussLegendre)`.
    pub fn new(p: usize) -> Self {
        Self::fetch(p, false)
    }

    /// MFEM `L2_FuentesPyramidElement(p, BasisType::GaussLobatto)`.
    pub fn new_gauss_lobatto(p: usize) -> Self {
        Self::fetch(p, true)
    }

    /// Whether this element uses the closed (`GaussLobatto`) `btype`.
    pub fn is_closed(&self) -> bool {
        self.inner.closed
    }

    /// MFEM's node table for the given order and `btype` closure, in DOF order.
    pub fn nodes(p: usize, closed: bool) -> Vec<[f64; 3]> {
        l2_fuentes_pyramid_nodes(p, closed)
    }

    fn fetch(p: usize, closed: bool) -> Self {
        static CACHE: OnceLock<Mutex<HashMap<(usize, bool), Arc<L2FuentesPyramidPkInner>>>> =
            OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let inner = {
            let mut m = cache
                .lock()
                .expect("L2FuentesPyramidPk cache poisoned");
            m.entry((p, closed))
                .or_insert_with(|| Arc::new(l2_fuentes_pyramid_pk_build(p, closed)))
                .clone()
        };
        Self { inner }
    }
}

fn l2_fuentes_pyramid_pk_build(p: usize, closed: bool) -> L2FuentesPyramidPkInner {
    let nodes = l2_fuentes_pyramid_nodes(p, closed);
    let n = nodes.len();
    let mut u = vec![0.0; n];
    // MFEM builds `T(o, m) = u_o(node_m)` column-by-column over the nodes
    // (`fe_l2.cpp:970-989`) and factorizes the dense matrix (`Ti.Factor(T)`).
    let mut t = nalgebra::DMatrix::<f64>::zeros(n, n);
    for (m, node) in nodes.iter().enumerate() {
        l2_fuentes_raw_basis(p, node[0], node[1], node[2], &mut u);
        for (o, &v) in u.iter().enumerate() {
            t[(o, m)] = v;
        }
    }
    let ti_m = t
        .try_inverse()
        .expect("L2FuentesPyramidPk: singular Vandermonde matrix");
    let mut ti = vec![0.0; n * n];
    for m in 0..n {
        for o in 0..n {
            ti[m * n + o] = ti_m[(m, o)];
        }
    }
    L2FuentesPyramidPkInner { order: p, closed, nodes, ti }
}

impl ReferenceElement for L2FuentesPyramidPk {
    fn dim(&self) -> u8 {
        3
    }

    fn order(&self) -> u8 {
        self.inner.order as u8
    }

    fn n_dofs(&self) -> usize {
        self.inner.nodes.len()
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.inner.order;
        let n = self.inner.nodes.len();
        let mut u = vec![0.0; n];
        l2_fuentes_raw_basis(p, xi[0], xi[1], xi[2], &mut u);
        for m in 0..n {
            let mut acc = 0.0;
            for o in 0..n {
                acc += self.inner.ti[m * n + o] * u[o];
            }
            values[m] = acc;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.inner.order;
        let n = self.inner.nodes.len();
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        // `fe_l2.cpp:1050-1054`: the derivatives are taken from the *direct*
        // shifted-Legendre evaluations at `x/(1−z)`, `y/(1−z)` and `z`, not
        // from `CalcScaledLegendre` — MFEM hard-codes the chain factors.
        let (sx, dsx) = calc_legendre_d(p, x / (1.0 - z));
        let (sy, dsy) = calc_legendre_d(p, y / (1.0 - z));
        let (sz, dsz) = calc_legendre_d(p, z);
        let omz = 1.0 - z;
        let mut du = vec![[0.0_f64; 3]; n];
        let mut o = 0;
        for k in 0..=p {
            for j in 0..=p {
                for i in 0..=p {
                    du[o] = [
                        dsx[i] * sy[j] * sz[k] / omz,
                        sx[i] * dsy[j] * sz[k] / omz,
                        sx[i] * sy[j] * dsz[k]
                            + (x * dsx[i] * sy[j] + y * sx[i] * dsy[j]) * sz[k]
                                / (omz * omz),
                    ];
                    o += 1;
                }
            }
        }
        for m in 0..n {
            let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
            for o in 0..n {
                let c = self.inner.ti[m * n + o];
                gx += c * du[o][0];
                gy += c * du[o][1];
                gz += c * du[o][2];
            }
            grads[m * 3] = gx;
            grads[m * 3 + 1] = gy;
            grads[m * 3 + 2] = gz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.inner.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}
