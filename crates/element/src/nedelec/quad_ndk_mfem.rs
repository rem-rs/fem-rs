//! Nédélec H(curl) element on the reference quadrilateral `[0,1]²` — a 1:1 port
//! of MFEM's `ND_QuadrilateralElement(p, GaussLobatto, ob)` (`fem/fe/fe_nd.cpp`,
//! the element behind `ND_FECollection(p, 2)`).
//!
//! # Tensor-product structure
//!
//! `ND_p = Q_{p-1,p} × Q_{p,p-1}`, `dof = 2p(p+1)`:
//!
//! - **x components**: `o_i(x)·c_j(y)`, `i = 0..p` (open), `j = 0..p` (closed)
//! - **y components**: `c_i(x)·o_j(y)`, `i = 0..p` (closed), `j = 0..p` (open)
//!
//! with the closed 1-D basis `c` the degree-`p` Gauss-Lobatto-Legendre nodal
//! basis at the `p+1` points `ClosedPoints(p, GaussLobatto)` and the open basis
//! `o` the `p` modes of the `ob` basis type ([`QNdOpen`]):
//!
//! - [`QNdOpen::GaussLegendre`] — MFEM `BasisType::GaussLegendre`, the
//!   `ND_FECollection(p, dim)` default: the degree-`p-1` nodal Lagrange modes at
//!   the `p` open Gauss-Legendre points `OpenPoints(p-1, GaussLegendre)`;
//! - [`QNdOpen::IntegratedGLL`] — MFEM `BasisType::IntegratedGLL`, the open half
//!   of the `(GaussLobatto, IntegratedGLL)` basis pair MFEM documents for LOR
//!   discretizations (`fem/lor/lor.hpp`): the integrated (Gerritsma) edge
//!   functions `o_i = −Σ_{j≤i} c'_j` built from the degree-`p` GLL closed
//!   basis — MFEM's `Poly_1D::Basis(p−1, Integrated)` attaches an auxiliary
//!   GLL basis of degree `(p−1)+1 = p` — with `ScaleIntegrated(false)`.
//!
//! The 1-D helpers are shared with the hex NDk element
//! ([`crate::gll_basis::ClosedBasis`] / [`crate::gll_basis::open_basis`]), so a
//! quad and a hex agree bit-for-bit on every 1-D factor.
//!
//! # Local DOF order (`dof_map`)
//!
//! Exactly MFEM's constructor order (see [`slot_table`]):
//!
//! ```text
//! 0 .. p        bottom edge (y = 0),  +x   slot (i, 0)
//! p .. 2p       right edge  (x = 1),  +y   slot (p, j)
//! 2p .. 3p      top edge    (y = 1),  −x   slot (p−1−i, p)   (sign −1)
//! 3p .. 4p      left edge   (x = 0),  −y   slot (0, p−1−j)   (sign −1)
//! 4p .. 4p+p(p−1)          interior x components (sign +1)
//! 4p+p(p−1) .. 2p(p+1)     interior y components (sign +1)
//! ```
//!
//! The top/left edges are **enumerated in reverse** of their coordinate order
//! with MFEM's `−1 − idx` flip, so the local slot index of every edge increases
//! from the edge's first to its second local vertex — which is exactly what
//! `HCurlSpace`'s signed anti-diagonal edge pairing assumes, and why the local
//! numbering lines up slot-for-slot with `HCurlSpace`'s combinatorial quad
//! layout (edges in `QUAD_EDGES` order, then interiors).  Edge DOFs are the
//! nodal point-value functionals `σ(Φ) = Φ(x_i)·t̂_i` with `t̂_i = ±e_a`
//! (MFEM `dof2tk`), so `dof_coords()[i]` is the node of `σ_i` and the sign of
//! the functional is the sign baked into the basis function.
//!
//! # VectorElement semantics
//!
//! For `p = 1` the element degenerates to the Whitney 1-form
//! (`Q_{0,1} × Q_{1,0}`), identical to MFEM's `ND_QuadrilateralElement(1)`.

use crate::gll_basis::ClosedBasis;
use crate::gll_basis::{gl_nodes, gll_nodes};
use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};
use std::sync::OnceLock;

/// Open (tangential) `p`-point Gauss-Legendre nodal modes of degree `p-1` on
/// `[0,1]` — the same 1-D factor MFEM uses for both tensor element families
/// (`Poly_1D::GetBasis`, keyed by degree and basis type only), evaluated
/// un-scaled because the quad reference square is MFEM's `[0,1]²`.

/// The 1-D open-basis kind of the tensor ND basis — MFEM's `ob_type` argument
/// of `ND_QuadrilateralElement(p, cb_type, ob_type)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QNdOpen {
    /// MFEM `BasisType::GaussLegendre` — the `ND_FECollection(p, 2)` default
    /// (what [`QuadND::new`] builds): nodal open modes at the Gauss-Legendre
    /// points, so every DOF is a point-value functional.
    GaussLegendre,
    /// MFEM `BasisType::IntegratedGLL` — the LOR-compatible open basis (see the
    /// module docs).  Same DOF count/layout/positions; only the open modes
    /// differ (`is_nodal = false` in MFEM).
    IntegratedGLL,
}

// ─── Slot layout (MFEM ND_QuadrilateralElement dof_map + CalcVShape) ────────

// ─── Slot layout (MFEM ND_QuadrilateralElement dof_map + CalcVShape) ────────

/// One local DOF of the tensor ND element: the 1-D modes it is built from and
/// the tensor cell they occupy.
///
/// - `xfam`: `true` for the x-family functions `o(x)·c(y)` (component direction
///   x), `false` for the y family `c(x)·o(y)`.
/// - `oo`, `oc`: the tensor cell — `oo` is the **open** mode index (along the
///   component direction: x for the x family, y for the y family) and `oc` the
///   **closed** mode index (across it).  `open_modes(p, ·)` has `p` entries
///   (`oo ∈ 0..p`), `ClosedBasis::new(p)` has `p + 1` (`oc ∈ 0..=p`).
/// - `ci`, `cj`: the indices of the two 1-D mode factors in their own arrays —
///   for the x family `(op[ci], cp[cj])` is the DOF node, for the y family
///   `(cp[ci], op[cj])`.  Kept explicit (rather than re-derived from `(oo,
///   oc)`) because the tensor-block slot number and the node indices are two
///   different bookkeepings of the same DOF.
/// - `slot`: the MFEM constructor's tensor-slot number (`i + j·p` for the x
///   family, `p(p+1) + i + j·(p+1)` for the y family) — distinct per DOF, used
///   for the bijection checks.
/// - `sign`: MFEM's `dof_map` orientation flip (`−1 − idx`), baked into the
///   basis function exactly as `CalcVShape` multiplies by `s`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QNdSlot {
    pub xfam: bool,
    pub oo: usize,
    pub oc: usize,
    pub ci: usize,
    pub cj: usize,
    pub slot: usize,
    pub sign: f64,
}

/// The local DOFs in MFEM's `Nodes` order — a 1:1 transcription of the
/// `ND_QuadrilateralElement` constructor (`fem/fe/fe_nd.cpp`): build the
/// signed `dof_map` (tensor slot -> ±local index) in the constructor's write
/// order, then invert it.
///
/// The resulting local DOF order is
///
/// ```text
/// 0 .. p          bottom edge (0,1), y = 0   x family, open i ascending,  sign +
/// p .. 2p         right edge  (1,2), x = 1   y family, open j ascending,  sign +
/// 2p .. 3p        top edge    (2,3), y = 1   x family, open i descending, sign −
/// 3p .. 4p        left edge   (3,0), x = 0   y family, open j descending, sign −
/// 4p .. 4p+p(p−1)     interior x components (closed mode j = 1..p−1 outer)
/// 4p+p(p−1) .. 2p(p+1) interior y components (closed mode i = 1..p−1 inner)
/// ```
///
/// which is exactly `HCurlSpace`'s combinatorial quad layout (the `QUAD_EDGES`
/// cycle bottom/right/top/left, each edge's slots along its local vertex
/// direction, then the interiors), and MFEM's own `FE::Nodes` numbering.
/// (There is **no** single sort key that reproduces this for all `p` — the x
/// tensor slots stride by `p` while the y slots stride by `p+1` — hence the
/// literal `dof_map` transcription.)
fn slot_table(p: usize) -> Vec<QNdSlot> {
    let dof2 = p * (p + 1);
    // MFEM constructor `dof_map`: tensor slot -> signed local dof index.
    let mut dof_map = vec![0_i32; 2 * dof2];
    let mut o: i32 = 0;
    // edges
    for i in 0..p {
        // (0,1): x slot `i + 0*p`.
        dof_map[i] = o;
        o += 1;
    }
    for j in 0..p {
        // (1,2): y slot `p + j*(p+1)`.
        dof_map[dof2 + p + j * (p + 1)] = o;
        o += 1;
    }
    for i in 0..p {
        // (2,3): x slot `(p-1-i) + p*p`, sign flip.
        dof_map[(p - 1 - i) + p * p] = -1 - o;
        o += 1;
    }
    for j in 0..p {
        // (3,0): y slot `0 + (p-1-j)*(p+1)`, sign flip.
        dof_map[dof2 + (p - 1 - j) * (p + 1)] = -1 - o;
        o += 1;
    }
    // interior: x-components (j outer, i inner), then y-components.
    for j in 1..p {
        for i in 0..p {
            dof_map[i + j * p] = o;
            o += 1;
        }
    }
    for j in 0..p {
        for i in 1..p {
            dof_map[dof2 + i + j * (p + 1)] = o;
            o += 1;
        }
    }
    debug_assert_eq!(o as usize, 2 * dof2);
    // Invert: local slot `idx` = the tensor cell whose dof_map entry is ±idx.
    let mut out: Vec<Option<QNdSlot>> = vec![None; 2 * dof2];
    for (tslot, &d) in dof_map.iter().enumerate() {
        let (idx, sign) = if d < 0 {
            ((-1 - d) as usize, -1.0)
        } else {
            (d as usize, 1.0)
        };
        // x slots are `i + j*p` (stride `p`), y slots `i + j*(p+1)` (stride
        // `p+1`) offset by the x-block size `dof2`.
        let (xfam, ci, cj) = if tslot < dof2 {
            (true, tslot % p, tslot / p)
        } else {
            let s = tslot - dof2;
            (false, s % (p + 1), s / (p + 1))
        };
        // x family: open index along x = `ci`, closed along y = `cj`; y family
        // the reverse.
        let (oo, oc) = if xfam { (ci, cj) } else { (cj, ci) };
        out[idx] = Some(QNdSlot { xfam, oo, oc, ci, cj, slot: tslot, sign });
    }
    let out: Vec<QNdSlot> = out
        .into_iter()
        .map(|s| s.expect("ND quad dof_map is a bijection"))
        .collect();
    debug_assert_eq!(out.len(), 2 * p * (p + 1));
    out
}


// ─── Element data ────────────────────────────────────────────────────────────

struct QuadNDData {
    p: usize,
    dof: usize,
    slots: Vec<QNdSlot>,
}

fn nd_data(p: usize) -> &'static QuadNDData {
    const MAX_P: usize = 8;
    static CACHE: [OnceLock<QuadNDData>; MAX_P + 1] = {
        #[allow(clippy::declare_interior_mutable_const)]
        const NEW: OnceLock<QuadNDData> = OnceLock::new();
        [NEW; MAX_P + 1]
    };
    assert!(p <= MAX_P, "QuadND: order {p} > {MAX_P} unsupported");
    CACHE[p].get_or_init(|| QuadNDData {
        p,
        dof: 2 * p * (p + 1),
        slots: slot_table(p),
    })
}

/// The open 1-D modes along one axis at `x` on `[0,1]` (`p` of them) and their
/// derivatives — MFEM's `obasis1d.Eval` / `EvalIntegrated` on the reference
/// square.
fn open_modes(open: QNdOpen, p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    match open {
        // MFEM `BasisType::GaussLegendre`: degree-`p-1` nodal modes at the `p`
        // open Gauss-Legendre points of `[0,1]`.  NB the shared
        // `open_basis_scaled` helper is parameterised by the `[-1,1]` nodes
        // (`crate::gll_basis::gl_nodes`), so the quad builds its own `[0,1]`
        // barycentric evaluation.
        QNdOpen::GaussLegendre => open_basis_01(p, x),
        // MFEM `BasisType::IntegratedGLL`: `o_i = -sum_{j<=i} c'_j` (Gerritsma
        // edge functions).  MFEM builds `obasis1d = GetBasis(p-1, IntegratedGLL)`
        // (`VectorTensorFiniteElement`, fe_base.cpp), whose `Integrated` basis
        // carries an *auxiliary* GLL barycentric basis of degree
        // `(p-1)+1 = p` — the same `p+1` points as the closed basis — and
        // `EvalIntegrated` accumulates the auxiliary derivatives:
        // `u[0] = -d[0], u[j] = u[j-1] - d[j]`.  The auxiliary derivative is
        // taken w.r.t. the `[0,1]` reference coordinate (MFEM's `Poly_1D`
        // bases are `[0,1]`-parameterised), which is `2 x d/dz` of the
        // `[-1,1]`-parameterised `ClosedBasis`.  `ScaleIntegrated(false)`
        // leaves the modes un-scaled by the subcell widths.
        QNdOpen::IntegratedGLL => {
            let cb = ClosedBasis::new(p);
            let v = cb.eval(2.0 * x - 1.0);
            let mut o = vec![0.0_f64; p];
            let mut od = vec![0.0_f64; p];
            for i in 0..p {
                o[i] = if i == 0 { -2.0 * v.dc[0] } else { o[i - 1] - 2.0 * v.dc[i] };
                od[i] = if i == 0 { -4.0 * v.d2c[0] } else { od[i - 1] - 4.0 * v.d2c[i] };
            }
            (o, od)
        }
    }
}

/// The `p` open Gauss-Legendre points of `[0,1]` (MFEM `OpenPoints(p-1,
/// GaussLegendre)`), ascending.
#[inline]
fn gl_points_01(p: usize) -> Vec<f64> {
    // `gll_basis::gl_nodes` is the `[-1,1]` table; map it onto `[0,1]`.
    gl_nodes(p).iter().map(|&x| 0.5 * (x + 1.0)).collect()
}

/// Degree-`p-1` barycentric Lagrange values/derivatives at `x ∈ [0,1]` on the
/// `p` open Gauss-Legendre points — the `[0,1]` counterpart of
/// `hex_ndk::open_basis_scaled` (which is hard-wired to the `[-1,1]` nodes).
fn open_basis_01(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let nodes = gl_points_01(p);
    let n = nodes.len();
    let mut w = vec![1.0_f64; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                w[j] /= nodes[j] - nodes[k];
            }
        }
    }
    let mut c = vec![0.0_f64; n];
    let mut dc = vec![0.0_f64; n];
    if let Some(m) = (0..n).find(|&m| x == nodes[m]) {
        c[m] = 1.0;
        let a0: f64 = (0..n)
            .filter(|&k| k != m)
            .map(|k| 1.0 / (nodes[m] - nodes[k]))
            .sum();
        dc[m] = a0;
        for j in 0..n {
            if j != m {
                dc[j] = (w[j] / w[m]) / (nodes[m] - nodes[j]);
            }
        }
    } else {
        let mut lam = vec![0.0_f64; n];
        let mut dlam = vec![0.0_f64; n];
        let (mut s, mut ds) = (0.0_f64, 0.0_f64);
        for j in 0..n {
            let t = x - nodes[j];
            lam[j] = w[j] / t;
            dlam[j] = -w[j] / (t * t);
            s += lam[j];
            ds += dlam[j];
        }
        for j in 0..n {
            c[j] = lam[j] / s;
            dc[j] = (dlam[j] - c[j] * ds) / s;
        }
    }
    (c, dc)
}

/// Nédélec element on the reference quadrilateral — MFEM
/// `ND_QuadrilateralElement(p, GaussLobatto, ob)`.
pub struct QuadND {
    order: usize,
    open: QNdOpen,
}

impl QuadND {
    /// MFEM `ND_QuadrilateralElement(p, GaussLobatto, GaussLegendre)` — the
    /// `ND_FECollection(p, 2)` default (nodal point-value DOFs).
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "QuadND requires order >= 1");
        QuadND { order: p, open: QNdOpen::GaussLegendre }
    }

    /// MFEM `ND_QuadrilateralElement(p, GaussLobatto, IntegratedGLL)` — the
    /// LOR-compatible basis pair (`fem/lor/lor.hpp`).  Same DOF count, layout
    /// and node positions as [`QuadND::new`]; only the open modes change.
    pub fn new_integrated_gll(p: usize) -> Self {
        assert!(p >= 1, "QuadND requires order >= 1");
        QuadND { order: p, open: QNdOpen::IntegratedGLL }
    }

    /// The open-basis kind of this element.
    pub fn open_basis_kind(&self) -> QNdOpen {
        self.open
    }

    /// Reference tangents of the local DOFs, same order as
    /// [`VectorReferenceElement::dof_coords`]:
    ///
    /// ```text
    /// [ +1, 0 ]  x family (bottom / top edges, interior x block)
    /// [  0,+1 ]  y family (right / left edges, interior y block)
    /// ```
    ///
    /// with the *negative* entries of MFEM's `dof_map` (`dof2tk = 2/3`) already
    /// baked into the basis functions, so a DOF's tangent is the fixed
    /// `±e_a` direction of its functional.
    pub fn dof_tangents(&self) -> Vec<[f64; 2]> {
        // The DOF functionals are `sigma(Phi) = Phi(x_i) . t_i` with `t_i` the
        // *edge direction* of the DOF's edge (MFEM's `dof2tk` table, whose sign
        // the basis function already carries).  Interior DOFs use the component
        // direction.
        nd_data(self.order)
            .slots
            .iter()
            .map(|&sl| {
                let p = self.order;
                if sl.xfam {
                    if sl.oc == p {
                        [-1.0, 0.0] // top edge (2,3), −x (MFEM dof2tk = 2)
                    } else {
                        [1.0, 0.0] // bottom edge + interior x block (tk = 0)
                    }
                } else if sl.oc == 0 {
                    [0.0, -1.0] // left edge (3,0), −y (MFEM dof2tk = 3)
                } else {
                    [0.0, 1.0] // right edge + interior y block (tk = 1)
                }
            })
            .collect()
    }
}

impl VectorReferenceElement for QuadND {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        2 * self.order * (self.order + 1)
    }

    /// MFEM `CalcVShape`: every DOF is the tensor product of its two 1-D mode
    /// factors, written into the component selected by its family.
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let d = nd_data(self.order);
        let p = d.p;
        // `ClosedBasis` evaluates the `[-1,1]` GLL basis; map the `[0,1]`
        // reference coordinate into it.  (The covered direction/closed 
        // derivative picks up the chain factor 2, which cancels in `curl`.)
        let (cx, _dcx) = {
            let v = ClosedBasis::new(p).eval(2.0 * xi[0] - 1.0);
            (v.c, v.dc)
        };
        let (cy, _dcy) = {
            let v = ClosedBasis::new(p).eval(2.0 * xi[1] - 1.0);
            (v.c, v.dc)
        };
        let (ox, _dox) = open_modes(self.open, p, xi[0]);
        let (oy, _doy) = open_modes(self.open, p, xi[1]);

        values.fill(0.0);
        for (n, sl) in d.slots.iter().enumerate() {
            // x family `o_ci(x)·c_cj(y)`; y family `c_ci(x)·o_cj(y)`.
            let (fx, fy) = if sl.xfam {
                (ox[sl.ci], cy[sl.cj])
            } else {
                (cx[sl.ci], oy[sl.cj])
            };
            let v = sl.sign * fx * fy;
            if sl.xfam {
                values[n * 2] = v;
            } else {
                values[n * 2 + 1] = v;
            }
        }
    }

    /// MFEM `CalcCurlShape`: in 2-D the scalar curl is `∂Φ_y/∂x − ∂Φ_x/∂y`, and
    /// only the *closed* factor of each tensor function is differentiated (the
    /// open factor runs along the function's own component direction), so the
    /// closed factor is always the one carrying `ci`/`cj`'s derivative.
    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let d = nd_data(self.order);
        let p = d.p;
        // `ClosedBasis` evaluates on `[-1,1]`; map the `[0,1]` reference
        // coordinate in, and keep the chain factor 2 so both derivatives are
        // w.r.t. `xi`.
        let (_cx, dcx) = {
            let v = ClosedBasis::new(p).eval(2.0 * xi[0] - 1.0);
            (v.c, v.dc.iter().map(|&d| 2.0 * d).collect::<Vec<_>>())
        };
        let (_cy, dcy) = {
            let v = ClosedBasis::new(p).eval(2.0 * xi[1] - 1.0);
            (v.c, v.dc.iter().map(|&d| 2.0 * d).collect::<Vec<_>>())
        };
        let (ox, _dox) = open_modes(self.open, p, xi[0]);
        let (oy, _doy) = open_modes(self.open, p, xi[1]);

        curl_vals.fill(0.0);
        for (n, sl) in d.slots.iter().enumerate() {
            if sl.xfam {
                // Φ_x = o_ci(x)·c_cj(y)  ->  −∂Φ_x/∂y = −o_ci·c'_cj
                curl_vals[n] = -sl.sign * ox[sl.ci] * dcy[sl.cj];
            } else {
                // Φ_y = c_ci(x)·o_cj(y)  ->  ∂Φ_y/∂x = c'_ci·o_cj
                curl_vals[n] = sl.sign * dcx[sl.ci] * oy[sl.cj];
            }
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    /// MFEM `FE::Nodes`: the x-family DOF sits at `(op[ci], cp[cj])`, the
    /// y-family one at `(cp[ci], op[cj])` — the same index pair the evaluator
    /// above uses as the tensor cell.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let d = nd_data(self.order);
        let p = d.p;
        let op = gl_points_01(p);
        // `gll_basis::gll_nodes` lives on `[-1,1]`; MFEM closed points are on
        // `[0,1]`.
        let cp: Vec<f64> = gll_nodes(p).iter().map(|&x| 0.5 * (x + 1.0)).collect();
        d.slots
            .iter()
            .map(|sl| {
                if sl.xfam {
                    vec![op[sl.ci], cp[sl.cj]]
                } else {
                    vec![cp[sl.ci], op[sl.cj]]
                }
            })
            .collect()
    }
}


#[cfg(test)]
mod mfem_quad_dump;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_dofs() {
        assert_eq!(QuadND::new(1).n_dofs(), 4);
        assert_eq!(QuadND::new(2).n_dofs(), 12);
        assert_eq!(QuadND::new(3).n_dofs(), 24);
        assert_eq!(QuadND::new(4).n_dofs(), 40);
    }

    #[test]
    fn finite() {
        for k in 1..=4 {
            let e = QuadND::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 2];
            let mut c = vec![0.0; n];
            for pt in &[(0.31, 0.73), (0.5, 0.5), (0.13, 0.62)] {
                e.eval_basis_vec(&[pt.0, pt.1], &mut v);
                e.eval_curl(&[pt.0, pt.1], &mut c);
                assert!(v.iter().all(|x| x.is_finite()), "k={k} phi");
                assert!(c.iter().all(|x| x.is_finite()), "k={k} curl");
            }
        }
    }

    /// Every local DOF owns a distinct tensor slot, and the local order is
    /// MFEM's constructor order: the four edges (bottom, right, top, left)
    /// first, then the interior x block, then the interior y block.
    #[test]
    fn slot_table_is_a_bijection() {
        for p in 1..=6usize {
            let slots = slot_table(p);
            let n_dof = 2 * p * (p + 1);
            assert_eq!(slots.len(), n_dof);
            // Exactly p(p+1) dofs per family (the Q_{p-1,p} x Q_{p,p-1} split).
            assert_eq!(slots.iter().filter(|sl| sl.xfam).count(), p * (p + 1));
            // Mode indices stay inside their 1-D arrays, signs are ±1, every
            // (family, open, closed) triple is used exactly once, and the MFEM
            // tensor-slot numbers are distinct.
            let mut seen: Vec<(bool, usize, usize)> = Vec::new();
            let mut tensor_slots: Vec<usize> = Vec::new();
            for sl in slots.iter() {
                assert!(sl.oo < p, "p={p}: open index {} out of range", sl.oo);
                assert!(sl.oc <= p, "p={p}: closed index {} out of range", sl.oc);
                assert!(sl.sign == 1.0 || sl.sign == -1.0);
                let key = (sl.xfam, sl.oo, sl.oc);
                assert!(!seen.contains(&key), "p={p}: mode triple {key:?} reused");
                seen.push(key);
                assert!(
                    !tensor_slots.contains(&sl.slot),
                    "p={p}: tensor slot {} reused",
                    sl.slot
                );
                tensor_slots.push(sl.slot);
            }
            assert_eq!(seen.len(), n_dof);
            // Edge blocks in constructor order, each along its local vertex
            // direction: bottom (+), right (+), top (−), left (−).
            for (m, sl) in slots.iter().enumerate() {
                match m / p {
                    0 => {
                        assert!(sl.xfam && sl.oc == 0 && sl.sign == 1.0, "p={p}: bottom block @{m}");
                    }
                    1 => {
                        assert!(!sl.xfam && sl.oc == p && sl.sign == 1.0, "p={p}: right block @{m}");
                    }
                    2 => {
                        assert!(sl.xfam && sl.oc == p && sl.sign == -1.0, "p={p}: top block @{m}");
                    }
                    _ if m < 4 * p => {
                        assert!(!sl.xfam && sl.oc == 0 && sl.sign == -1.0, "p={p}: left block @{m}");
                    }
                    _ => {}
                }
            }
            // Interior x block then interior y block, closed mode outer.
            let int = &slots[4 * p..];
            for (m, sl) in int.iter().enumerate() {
                if m < p * (p - 1) {
                    assert!(sl.xfam, "p={p}: interior x block @{m}");
                } else {
                    assert!(!sl.xfam, "p={p}: interior y block @{m}");
                }
            }
        }
    }

    /// Every edge DOF of the slot table must be the point-value functional
    /// `σ(Φ) = Φ(x_i)·t̂_i` (MFEM nodal semantics): this pins the top/left edge
    /// reversal-plus-sign-flip and the edge half of `dof_coords`.
    #[test]
    fn edge_slots_are_nodal() {
        for p in 1..=4usize {
            let e = QuadND::new(p);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let tks = e.dof_tangents();
            let mut v = vec![0.0_f64; n * 2];
            for i in 0..4 * p {
                let xi = [coords[i][0], coords[i][1]];
                e.eval_basis_vec(&xi, &mut v);
                for j in 0..4 * p {
                    let d = v[j * 2] * tks[i][0] + v[j * 2 + 1] * tks[i][1];
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (d - want).abs() < 1e-11,
                        "p={p}: edge σ_{i}(φ_{j}) = {d} (want {want})",
                    );
                }
            }
        }
    }

    /// Per-DOF match against MFEM's `ND_QuadrilateralElement(p, GaussLobatto,
    /// GaussLegendre)` dump: the node table and the `CalcVShape` /
    /// `CalcCurlShape` values at three sample points, **slot for slot** — the
    /// local DOF numbering is MFEM's constructor order (see [`slot_table`]).
    #[test]
    fn gauss_legendre_matches_mfem_dump() {
        for p in 1..=4usize {
            let e = QuadND::new(p);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let mfem_nodes: &[[f64; 2]] = match p {
                1 => &mfem_quad_dump::NODES_1,
                2 => &mfem_quad_dump::NODES_2,
                3 => &mfem_quad_dump::NODES_3,
                _ => &mfem_quad_dump::NODES_4,
            };
            assert_eq!(coords.len(), n);
            for i in 0..n {
                assert!(
                    (coords[i][0] - mfem_nodes[i][0]).abs() < 1e-14
                        && (coords[i][1] - mfem_nodes[i][1]).abs() < 1e-14,
                    "p={p}: node {i} {:?} != mfem {:?}",
                    coords[i],
                    mfem_nodes[i]
                );
            }

            let pts = [[0.137, 0.621], [0.71, 0.22], [0.5, 0.5]];
            let mut v = vec![0.0_f64; n * 2];
            let mut c = vec![0.0_f64; n];
            for (q, pt) in pts.iter().enumerate() {
                let mfem: &[[f64; 3]] = match (p, q) {
                    (1, 0) => &mfem_quad_dump::VC_1_0_0,
                    (1, 1) => &mfem_quad_dump::VC_1_0_1,
                    (1, _) => &mfem_quad_dump::VC_1_0_2,
                    (2, 0) => &mfem_quad_dump::VC_2_0_0,
                    (2, 1) => &mfem_quad_dump::VC_2_0_1,
                    (2, _) => &mfem_quad_dump::VC_2_0_2,
                    (3, 0) => &mfem_quad_dump::VC_3_0_0,
                    (3, 1) => &mfem_quad_dump::VC_3_0_1,
                    (3, _) => &mfem_quad_dump::VC_3_0_2,
                    (4, 0) => &mfem_quad_dump::VC_4_0_0,
                    (4, 1) => &mfem_quad_dump::VC_4_0_1,
                    _ => &mfem_quad_dump::VC_4_0_2,
                };
                e.eval_basis_vec(pt, &mut v);
                e.eval_curl(pt, &mut c);
                for i in 0..n {
                    assert!(
                        (v[i * 2] - mfem[i][0]).abs() < 1e-12,
                        "p={p} q={q} phi[{i}]_x: {} vs mfem[i] {}",
                        v[i * 2],
                        mfem[i][0]
                    );
                    assert!(
                        (v[i * 2 + 1] - mfem[i][1]).abs() < 1e-12,
                        "p={p} q={q} phi[{i}]_y: {} vs mfem[i] {}",
                        v[i * 2 + 1],
                        mfem[i][1]
                    );
                    assert!(
                        (c[i] - mfem[i][2]).abs() < 1e-12,
                        "p={p} q={q} curl[{i}]: {} vs mfem[i] {}",
                        c[i],
                        mfem[i][2]
                    );
                }
            }
        }
    }

    /// Per-DOF match against MFEM's `ND_QuadrilateralElement(p, GaussLobatto,
    /// IntegratedGLL)` dump — the LOR-compatible basis pair.  Same node table
    /// as the Gauss-Legendre variant; only the open modes differ.  For `p = 1`
    /// the integrated mode is the constant `−c'_0 = 1`, so both variants
    /// coincide with the Whitney element (the dumps are identical).
    #[test]
    fn integrated_gll_matches_mfem_dump() {
        for p in 1..=4usize {
            let e = QuadND::new_integrated_gll(p);
            let n = e.n_dofs();
            let pts = [[0.137, 0.621], [0.71, 0.22], [0.5, 0.5]];
            let mut v = vec![0.0_f64; n * 2];
            let mut c = vec![0.0_f64; n];
            for (q, pt) in pts.iter().enumerate() {
                let mfem: &[[f64; 3]] = match (p, q) {
                    (1, 0) => &mfem_quad_dump::VC_1_6_0,
                    (1, 1) => &mfem_quad_dump::VC_1_6_1,
                    (1, _) => &mfem_quad_dump::VC_1_6_2,
                    (2, 0) => &mfem_quad_dump::VC_2_6_0,
                    (2, 1) => &mfem_quad_dump::VC_2_6_1,
                    (2, _) => &mfem_quad_dump::VC_2_6_2,
                    (3, 0) => &mfem_quad_dump::VC_3_6_0,
                    (3, 1) => &mfem_quad_dump::VC_3_6_1,
                    (3, _) => &mfem_quad_dump::VC_3_6_2,
                    (4, 0) => &mfem_quad_dump::VC_4_6_0,
                    (4, 1) => &mfem_quad_dump::VC_4_6_1,
                    _ => &mfem_quad_dump::VC_4_6_2,
                };
                e.eval_basis_vec(pt, &mut v);
                e.eval_curl(pt, &mut c);
                for i in 0..n {
                    assert!(
                        (v[i * 2] - mfem[i][0]).abs() < 1e-12,
                        "p={p} q={q} phi[{i}]_x: {} vs {}",
                        v[i * 2],
                        mfem[i][0]
                    );
                    assert!(
                        (v[i * 2 + 1] - mfem[i][1]).abs() < 1e-12,
                        "p={p} q={q} phi[{i}]_y: {} vs {}",
                        v[i * 2 + 1],
                        mfem[i][1]
                    );
                    assert!(
                        (c[i] - mfem[i][2]).abs() < 1e-12,
                        "p={p} q={q} curl[{i}]: {} vs {}",
                        c[i],
                        mfem[i][2]
                    );
                }
            }
        }
    }

    /// The Gauss-Legendre variant's DOFs are point-value functionals: the
    /// pairing `σ_i(φ_j) = φ_j(x_i)·t̂_i` is the identity, and every edge DOF
    /// vanishes tangentially on the other three edges.
    #[test]
    fn gauss_legendre_dofs_are_nodal() {
        for p in 1..=3usize {
            let e = QuadND::new(p);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let tks = e.dof_tangents();
            let mut v = vec![0.0_f64; n * 2];
            for i in 0..n {
                let xi = [coords[i][0], coords[i][1]];
                e.eval_basis_vec(&xi, &mut v);
                for j in 0..n {
                    let d = v[j * 2] * tks[i][0] + v[j * 2 + 1] * tks[i][1];
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (d - want).abs() < 1e-11,
                        "p={p}: σ_{i}(φ_{j}) = {d} (want {want})",
                    );
                }
            }
        }
    }

    /// Edge DOFs are *tangentially* supported on their own edge: on `y = 0`
    /// the **tangential (x) component** of every basis function not anchored on
    /// the bottom edge vanishes (the closed GLL factor is nodal at `y = 0`),
    /// and on `x = 1` the tangential (y) component of every function not on the
    /// right edge vanishes.  (Normal traces are unconstrained for H(curl), so
    /// the *normal* components are generically nonzero on the edge.)
    #[test]
    fn edge_dofs_have_local_trace() {
        for p in 2..=3usize {
            let e = QuadND::new(p);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let tks = e.dof_tangents();
            let mut v = vec![0.0_f64; n * 2];
            // y = 0: only x-DOFs anchored on the bottom edge have an x-component.
            e.eval_basis_vec(&[0.37, 0.0], &mut v);
            for o in 0..n {
                let on_bottom = tks[o][1] == 0.0 && coords[o][1] == 0.0;
                let expect_zero = !on_bottom;
                assert!(
                    !expect_zero || v[o * 2].abs() < 1e-13,
                    "p={p}: slot {o} leaks tangentially on y=0: {}",
                    v[o * 2],
                );
                // Bottom slots themselves are the surviving tangential modes.
                if on_bottom {
                    assert!(v[o * 2].abs() > 1e-8, "p={p}: bottom slot {o} degenerate");
                }
            }
            // x = 1: only y-DOFs anchored on the right edge have a y-component.
            e.eval_basis_vec(&[1.0, 0.41], &mut v);
            for o in 0..n {
                let on_right = tks[o][0] == 0.0 && coords[o][0] == 1.0;
                assert!(
                    on_right || v[o * 2 + 1].abs() < 1e-13,
                    "p={p}: slot {o} leaks tangentially on x=1: {}",
                    v[o * 2 + 1],
                );
            }
        }
    }

    /// p = 1 must be the Whitney 1-form of MFEM's `ND_QuadrilateralElement(1)`:
    /// in `Nodes` order `(1−y, 0)`, `(0, x)`, `(−y, 0)`, `(0, x−1)` at the
    /// point `(0.3, 0.7)`, all with scalar curl 1.
    #[test]
    fn p1_is_whitney() {
        let e = QuadND::new(1);
        let mut v = vec![0.0_f64; 8];
        let mut c = vec![0.0_f64; 4];
        e.eval_basis_vec(&[0.3, 0.7], &mut v);
        e.eval_curl(&[0.3, 0.7], &mut c);
        // MFEM `Nodes` order: bottom edge `(1−y, 0)`, right edge `(0, x)`,
        // top edge `(−y, 0)` (its local direction runs from the top-right
        // corner leftwards, so the sign is baked negative) and left edge
        // `(0, −(1−x))`; all four have scalar curl 1.
        assert!((v[0] - 0.3).abs() < 1e-14, "phi0_x = {}", v[0]); // 1−y
        assert!((v[1] - 0.0).abs() < 1e-14);
        assert!((v[2] - 0.0).abs() < 1e-14, "phi1_x = {}", v[2]);
        assert!((v[3] - 0.3).abs() < 1e-14, "phi1_y = {}", v[3]); // x
        assert!((v[4] - (-0.7)).abs() < 1e-14, "phi2_x = {}", v[4]); // −y
        assert!((v[5] - 0.0).abs() < 1e-14);
        assert!((v[6] - 0.0).abs() < 1e-14, "phi3_x = {}", v[6]);
        assert!((v[7] - (-0.7)).abs() < 1e-14, "phi3_y = {}", v[7]); // x−1
        for i in 0..4 {
            assert!((c[i] - 1.0).abs() < 1e-14, "curl[{i}] = {}", c[i]);
        }
    }

    /// The IntegratedGLL open modes are the Gerritsma edge functions, yet the
    /// x-family still spans constants: `Σ_j c_j(y) = 1` (closed GLL partition
    /// of unity) and `1 ∈ span{o_i(x)}` (the `p` edge functions span
    /// `P_{p-1}`), so some combination of the x-family tensor modes is the
    /// constant field `u ≡ (1, 0)`.
    #[test]
    fn integrated_gll_span_contains_constant() {
        for p in 1..=3usize {
            let e = QuadND::new_integrated_gll(p);
            let n = e.n_dofs();
            // 1-D: represent 1 in the open-mode basis (collocate at GL points).
            let gl = gl_points_01(p);
            let mut b = nalgebra::DMatrix::<f64>::zeros(p, p);
            let rhs = nalgebra::DVector::from_element(p, 1.0);
            for r in 0..p {
                let (o, _) = open_modes(QNdOpen::IntegratedGLL, p, gl[r]);
                for c in 0..p {
                    b[(r, c)] = o[c];
                }
            }
            let sol = b.lu().solve(&rhs).expect("open-mode collocation singular");
            // x-family dof (i, j) carries coefficient `sign·b_i` for every j
            // (the top-edge dof's baked-in sign has to be undone to represent
            // the constant +1).
            let xcoef: Vec<f64> = (0..n)
                .map(|i| {
                    let sl = &nd_data(p).slots[i];
                    if sl.xfam { sl.sign * sol[sl.oo] } else { 0.0 }
                })
                .collect();
            let pts = [
                [0.1f64, 0.2f64],
                [0.5, 0.1],
                [0.9, 0.6],
                [0.3, 0.8],
                [0.7, 0.4],
                [0.2, 0.5],
            ];
            let mut vals = vec![0.0; n * 2];
            let mut max_err = 0.0f64;
            for pt in &pts {
                e.eval_basis_vec(pt, &mut vals);
                let ux: f64 = (0..n).map(|i| xcoef[i] * vals[i * 2]).sum();
                max_err = max_err.max((ux - 1.0).abs());
            }
            assert!(max_err < 1e-11, "p={p}: constant x-comp not representable");
        }
    }

    /// Finite-difference curl consistency for both open-basis variants.
    #[test]
    fn curl_matches_finite_difference() {
        for p in 1..=3usize {
            for e in [QuadND::new(p), QuadND::new_integrated_gll(p)] {
                let n = e.n_dofs();
                let eps = 1e-6;
                let pt = [0.137, 0.621];
                let mut vp = vec![0.0; n * 2];
                let mut vm = vec![0.0; n * 2];
                let mut fd = vec![0.0; n];
                for d in 0..2 {
                    let mut xp = pt;
                    let mut xm = pt;
                    xp[d] += eps;
                    xm[d] -= eps;
                    e.eval_basis_vec(&xp, &mut vp);
                    e.eval_basis_vec(&xm, &mut vm);
                    for i in 0..n {
                        let dvy = (vp[i * 2 + 1] - vm[i * 2 + 1]) / (2.0 * eps);
                        let dvx = (vp[i * 2] - vm[i * 2]) / (2.0 * eps);
                        fd[i] += if d == 0 { dvy } else { -dvx };
                    }
                }
                let mut an = vec![0.0; n];
                e.eval_curl(&pt, &mut an);
                for i in 0..n {
                    assert!(
                        (an[i] - fd[i]).abs() < 1e-5 * (1.0 + an[i].abs()),
                        "p={p}: curl[{i}] {} vs fd {}",
                        an[i],
                        fd[i]
                    );
                }
            }
        }
    }
}





















