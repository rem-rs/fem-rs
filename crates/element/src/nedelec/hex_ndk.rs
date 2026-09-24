//! Nédélec hexahedral element `ND_k` on the reference hex `[0,1]^3`.
//!
//! 1:1 port of MFEM `ND_HexahedronElement(p, GaussLobatto, GaussLegendre)` —
//! the element `ND_FECollection(p, dim)` builds by default, i.e. the ND hex
//! element every miniapp/example actually uses (`dpg_maxwell_3d`, the LOR
//! and Maxwell drivers): the **nodal** variant whose DOFs are the point-value
//! functionals `σ_i(Φ) = Φ(x_i)·t̂_i` at MFEM's `FE::Nodes` (Gauss-Legendre
//! open points along the component direction × GLL closed points across it)
//! with the *unnormalized* reference tangents `t̂_i = ±e_a` (MFEM `dof2tk`).
//!
//! **Native `[0,1]³` frame (D721).**  The basis functions are MFEM's own,
//! evaluated directly on `[0,1]`: the closed GLL factor through
//! `ClosedBasis::new_01` (`[0,1]` nodes, `[0,1]` derivatives) and the open
//! Gauss-Legendre nodal modes through [`open_basis`] (`[0,1]` open points,
//! no scale).  Before D721 every open mode carried a `1/2` and was evaluated
//! at `ξ̃ = (ξ+1)/2`, the pull-back that made `J^-T_ref·Φ_ref = Φ_MFEM` on the
//! unit cube despite `J = J_MFEM/2`; with the frame flipped that factor is
//! gone and the physical basis is MFEM's bit for bit.
//!
//! Tensor structure per component (`c` = closed GLL nodal mode of degree `k`
//! on `[0,1]`, `o` = open Gauss-Legendre nodal mode of degree `k-1`, `±`
//! hats are the endpoint GLL modes `c_0`/`c_k`):
//!
//! ```text
//!     x-dofs: o(x)·c(y)·c(z)   y-dofs: c(x)·o(y)·c(z)   z-dofs: c(x)·c(y)·o(z)
//! ```
//!
//! Local DOF order — D225: the exact MFEM `dof_map` enumeration (see
//! [`nd_slot_table`]):
//! - `12k` edge dofs in MFEM `Geometry::CUBE::Edges` order (==
//!   `HCurlSpace::HEX_EDGES`), `k` open modes per edge along the local edge
//!   direction, all positive;
//! - `6·2k(k-1)` face dofs in MFEM `CUBE::FaceVert` order (z−, y−, x+,
//!   y+, x−, z+), each face a first-tangent block then a second-tangent
//!   block with MFEM's intra-face reversals and its `-1-(o++)` negative
//!   orientation signs on the z− y-block, the y+ x-block and the x− y-block;
//! - `3k(k-1)^2` interior dofs (x-, y-, z-blocks, MFEM loop order, positive).
//!
//! The reference orientation signs are baked into the basis exactly as in
//! MFEM (`shape(idx,·) = s·tensor`); the flipped slots' dual tangents
//! (`dof_tangents`) flip with them, and `HCurlSpace`'s geometric orientation
//! encoding composes on top.

use crate::gll_basis::ClosedBasis;
use crate::reference::VectorReferenceElement;

/// The 1-D open-factor kind of the tensor ND basis — the `ob_type` argument of
/// MFEM `ND_HexahedronElement(p, cb_type, ob_type)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NdOpenBasis {
    /// MFEM `BasisType::GaussLegendre` — the `ND_FECollection(p, dim)` default
    /// (what [`HexNDk::new`] builds).  Open modes are the degree-`p-1`
    /// Gauss-Legendre point-value Lagrange polynomials anchored at the FE's
    /// `Nodes` (the dof functionals are point values, `is_nodal = true`).
    GaussLegendre,
    /// MFEM `BasisType::IntegratedGLL` — the open half of the
    /// `(GaussLobatto, IntegratedGLL)` basis pair that MFEM documents for LOR
    /// discretizations (`fem/lor/lor.hpp`: "the high-order finite element
    /// space should use ... basis pair (GaussLobatto, IntegratedGLL) for
    /// Nedelec and Raviart-Thomas elements").  Open modes are the integrated
    /// (Gerritsma) edge functions `-Σ_{j<=i} c'_j` built from the degree-`p`
    /// GLL closed basis (`Poly_1D::Basis::EvalIntegrated`, `is_nodal = false`);
    /// the dof positions are unchanged (the same Gauss-Legendre points).
    IntegratedGLL,
}

pub struct HexNDk {
    order: usize,
    open: NdOpenBasis,
    /// The MFEM `dof_map` slot table (see [`nd_slot_table`]), precomputed so
    /// the hot evaluation paths never allocate.
    slots: Vec<NdSlot>,
}

/// One element-local DOF of the MFEM dof_map: `(block, i, j, k, flip)` —
/// `block` is the component direction (0/1/2 = x/y/z), `(i, j, k)` the tensor
/// indices (the block's own index is the open factor along `block`, the other
/// two are closed GLL indices), and `flip` is MFEM's `-1-(o++)` reference
/// orientation sign (`dof2tk` negative tangent).
type NdSlot = (u8, usize, usize, usize, bool);

/// The dof enumeration of MFEM `ND_HexahedronElement::ND_HexahedronElement`
/// (`fem/fe/fe_nd.cpp`), i.e. the slot order `CalcVShape` writes.
///
/// Edges in `Geometry::Constants<CUBE>::Edges` order (all positive), then the
/// six faces in `CUBE::FaceVert` order — bottom z−, front y−, right x+,
/// back y+, left x−, top z+ — each as a first-tangent block then a
/// second-tangent block, then the three interior blocks.
///
/// Per face (the faces whose local frame is a reflection of the increasing-
/// axis frame enumerate one index in reverse, and three blocks carry the
/// negative `-1-(o++)` sign):
/// - bottom z−: x-block closed-y descending, +; y-block open-y descending
///   outer / closed-x inner, all **negative**;
/// - front y−: x-block closed-z ascending, +; z-block open-z ascending, +;
/// - right x+: y-block closed-z ascending, +; z-block open-z ascending, +;
/// - back y+: x-block closed-z ascending / open-x descending inner, all
///   **negative**; z-block open-z ascending / closed-x descending inner, +;
/// - left x−: y-block closed-z ascending / open-y descending inner, all
///   **negative**; z-block open-z ascending / closed-y descending inner, +;
/// - top z+: x-block closed-y ascending, +; y-block open-y ascending, +.
///
/// Interiors: x-block (k,z outer, j,y middle, i,x-open inner); y-block
/// (k,z outer, j,y-open middle, i,x inner); z-block (k,z-open outer,
/// j,y middle, i,x inner) — all positive.
fn nd_slot_table(p: usize) -> Vec<NdSlot> {
    let mut t: Vec<NdSlot> = Vec::with_capacity(3 * p * (p + 1) * (p + 1));
    // edges (0,1),(1,2),(3,2),(0,3),(4,5),(5,6),(7,6),(4,7),(0,4),(1,5),(2,6),(3,7)
    for i in 0..p {
        t.push((0, i, 0, 0, false));
    }
    for j in 0..p {
        t.push((1, p, j, 0, false));
    }
    for i in 0..p {
        t.push((0, i, p, 0, false));
    }
    for j in 0..p {
        t.push((1, 0, j, 0, false));
    }
    for i in 0..p {
        t.push((0, i, 0, p, false));
    }
    for j in 0..p {
        t.push((1, p, j, p, false));
    }
    for i in 0..p {
        t.push((0, i, p, p, false));
    }
    for j in 0..p {
        t.push((1, 0, j, p, false));
    }
    for k in 0..p {
        t.push((2, 0, 0, k, false));
    }
    for k in 0..p {
        t.push((2, p, 0, k, false));
    }
    for k in 0..p {
        t.push((2, p, p, k, false));
    }
    for k in 0..p {
        t.push((2, 0, p, k, false));
    }
    // bottom (3,2,1,0) z=-1
    for j in (1..p).rev() {
        for i in 0..p {
            t.push((0, i, j, 0, false));
        }
    }
    for j in (0..p).rev() {
        for i in 1..p {
            t.push((1, i, j, 0, true));
        }
    }
    // front (0,1,5,4) y=-1
    for k in 1..p {
        for i in 0..p {
            t.push((0, i, 0, k, false));
        }
    }
    for k in 0..p {
        for i in 1..p {
            t.push((2, i, 0, k, false));
        }
    }
    // right (1,2,6,5) x=+1
    for k in 1..p {
        for j in 0..p {
            t.push((1, p, j, k, false));
        }
    }
    for k in 0..p {
        for j in 1..p {
            t.push((2, p, j, k, false));
        }
    }
    // back (2,3,7,6) y=+1
    for k in 1..p {
        for i in (0..p).rev() {
            t.push((0, i, p, k, true));
        }
    }
    for k in 0..p {
        for i in (1..p).rev() {
            t.push((2, i, p, k, false));
        }
    }
    // left (3,0,4,7) x=-1
    for k in 1..p {
        for j in (0..p).rev() {
            t.push((1, 0, j, k, true));
        }
    }
    for k in 0..p {
        for j in (1..p).rev() {
            t.push((2, 0, j, k, false));
        }
    }
    // top (4,5,6,7) z=+1
    for j in 1..p {
        for i in 0..p {
            t.push((0, i, j, p, false));
        }
    }
    for j in 0..p {
        for i in 1..p {
            t.push((1, i, j, p, false));
        }
    }
    // interior
    for k in 1..p {
        for j in 1..p {
            for i in 0..p {
                t.push((0, i, j, k, false));
            }
        }
    }
    for k in 1..p {
        for j in 0..p {
            for i in 1..p {
                t.push((1, i, j, k, false));
            }
        }
    }
    for k in 0..p {
        for j in 1..p {
            for i in 1..p {
                t.push((2, i, j, k, false));
            }
        }
    }
    t
}

impl HexNDk {
    /// MFEM `ND_HexahedronElement(p, GaussLobatto, GaussLegendre)` — the
    /// default nodal element (D32/D36 semantics).
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "HexNDk requires order >= 1");
        HexNDk { order: p, open: NdOpenBasis::GaussLegendre, slots: nd_slot_table(p) }
    }

    /// MFEM `ND_HexahedronElement(p, GaussLobatto, IntegratedGLL)` — the
    /// LOR-compatible basis pair.  Same dof count/layout and dof positions as
    /// [`HexNDk::new`]; only the open modes differ.
    pub fn new_integrated_gll(p: usize) -> Self {
        assert!(p >= 1, "HexNDk requires order >= 1");
        HexNDk {
            order: p,
            open: NdOpenBasis::IntegratedGLL,
            slots: nd_slot_table(p),
        }
    }

    /// The `p` open 1-D modes along one axis at `x` (reference `[0,1]`).
    fn open_modes(&self, x: f64) -> Vec<f64> {
        match self.open {
            NdOpenBasis::GaussLegendre => open_basis(self.order, x).0,
            NdOpenBasis::IntegratedGLL => {
                let cb = ClosedBasis::new_01(self.order);
                cb.integrated(&cb.eval_mfem(x))
            }
        }
    }
}

impl VectorReferenceElement for HexNDk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        3 * self.order * (self.order + 1) * (self.order + 1)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        // MFEM `CalcVShape` uses the value-only closed-basis overload
        // (`val_mfem`); the derivative overload (`eval_mfem`) is reserved for
        // the div/curl paths, exactly as in `CalcDivShape`/`CalcCurlShape`.
        let vx = ClosedBasis::new_01(p).val_mfem(xi[0]);
        let vy = ClosedBasis::new_01(p).val_mfem(xi[1]);
        let vz = ClosedBasis::new_01(p).val_mfem(xi[2]);
        let (cx, cy, cz) = (&vx, &vy, &vz);
        let oxs = self.open_modes(xi[0]);
        let oys = self.open_modes(xi[1]);
        let ozs = self.open_modes(xi[2]);
        values.fill(0.0);

        // D225: slot order = MFEM `dof_map` enumeration (`nd_slot_table`);
        // `flip` is MFEM's `-1-(o++)` reference orientation sign.
        for (slot, &(block, i, j, k, flip)) in self.slots.iter().enumerate() {
            let s = if flip { -1.0 } else { 1.0 };
            let v = match block {
                0 => s * oxs[i] * cy[j] * cz[k],
                1 => s * cx[i] * oys[j] * cz[k],
                _ => s * cx[i] * cy[j] * ozs[k],
            };
            values[slot * 3 + block as usize] = v;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let p = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        curl_vals.fill(0.0);

        let vx = ClosedBasis::new_01(p).eval_mfem(x);
        let vy = ClosedBasis::new_01(p).eval_mfem(y);
        let vz = ClosedBasis::new_01(p).eval_mfem(z);
        let (cx, cy, cz) = (&vx.c, &vy.c, &vz.c);
        let (dcx, dcy, dcz) = (&vx.dc, &vy.dc, &vz.dc);
        let oxs = self.open_modes(x);
        let oys = self.open_modes(y);
        let ozs = self.open_modes(z);

        // For a dofs-along-d tensor function o(t_d)·C·D the curl only
        // differentiates the two CLOSED factors (the open direction is the
        // component direction itself):
        //   x: curl = s·(0, o·C·D', -o·C'·D)
        //   y: curl = s·(-C·o·D', 0, C'·o·D)
        //   z: curl = s·(C·D'·o, -C'·D·o, 0)
        // D225: slot order/signs from `nd_slot_table` (MFEM dof_map).
        for (slot, &(block, i, j, k, flip)) in self.slots.iter().enumerate() {
            let s = if flip { -1.0 } else { 1.0 };
            let (c0, c1, c2) = match block {
                0 => (s * oxs[i] * cy[j] * dcz[k], s * oxs[i] * dcy[j] * cz[k], 0.0),
                1 => (s * cx[i] * oys[j] * dcz[k], 0.0, s * dcx[i] * oys[j] * cz[k]),
                _ => (0.0, s * cx[i] * dcy[j] * ozs[k], s * dcx[i] * cy[j] * ozs[k]),
            };
            let d = slot * 3;
            match block {
                0 => {
                    curl_vals[d + 1] = c0;
                    curl_vals[d + 2] = -c1;
                }
                1 => {
                    curl_vals[d] = -c0;
                    curl_vals[d + 2] = c2;
                }
                _ => {
                    curl_vals[d] = c1;
                    curl_vals[d + 1] = -c2;
                }
            }
        }
    }

    fn eval_div(&self, _: &[f64], dv: &mut [f64]) {
        for v in dv.iter_mut() {
            *v = 0.0;
        }
    }
    fn quadrature(&self, o: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::hex_rule(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.dof_layout().into_iter().map(|(x, _)| x.to_vec()).collect()
    }
}

impl HexNDk {
    /// Reference tangents of the local DOFs, same order as
    /// [`VectorReferenceElement::dof_coords`].
    ///
    /// These are the reference images of the DOF functionals
    /// `σ_i(Φ) = Φ(x_i)·t̂_i`: `t̂_i = ±e_a` for the component direction `a` of
    /// DOF `i` — MFEM's *unnormalized* `tk` in its own `[0,1]³` frame (D721;
    /// the historical magnitude 2 was the `[-1,1]` pull-back `J = J_MFEM/2`).
    /// The dual used by `HCurlSpace::interpolate_vector` and MFEM
    /// `VectorFiniteElement::Project_ND` is `J·t̂` in the same frame.
    pub fn dof_tangents(&self) -> Vec<[f64; 3]> {
        self.dof_layout().into_iter().map(|(_, t)| t).collect()
    }

    /// `(node, reference tangent)` of every local DOF — the element-local
    /// anchor table behind [`dof_coords`](Self::dof_coords) and
    /// [`dof_tangents`](Self::dof_tangents) (the two always agree slotwise).
    pub fn dof_anchors(&self) -> Vec<([f64; 3], [f64; 3])> {
        self.dof_layout()
    }

    /// `(node, reference tangent)` of every local DOF in the element's local
    /// order — the single source of truth behind [`dof_coords`](Self::dof_coords)
    /// and [`dof_tangents`](Self::dof_tangents), driven by the MFEM dof_map
    /// slot table (node = the open factor's Gauss-Legendre point along the
    /// *component* direction, GLL points across it; the tangent carries the
    /// sign of negatively oriented slots, MFEM `dof2tk`).
    fn dof_layout(&self) -> Vec<([f64; 3], [f64; 3])> {
        let p = self.order;
        let gl = crate::gll_basis::gl_nodes_01(p);
        let gll = crate::gll_basis::gll_nodes_01(p);
        self.slots
            .iter()
            .map(|&(block, i, j, k, flip)| {
                let s = if flip { -1.0 } else { 1.0 };
                let node = match block {
                    0 => [gl[i], gll[j], gll[k]],
                    1 => [gll[i], gl[j], gll[k]],
                    _ => [gll[i], gll[j], gl[k]],
                };
                let mut tau = [0.0_f64; 3];
                tau[block as usize] = s;
                (node, tau)
            })
            .collect()
    }
}

/// Open (tangential) `p`-point Gauss-Legendre nodal modes of degree `p-1` on
/// `[0,1]`.
///
/// These are MFEM's tensor open 1-D functions (`poly1d.GetBasis(p-1,
/// BasisType::GaussLegendre)`, evaluated by `Poly_1D::Basis::Eval` as the
/// degree-`p-1` Lagrange interpolation at the `OpenPoints(p-1, GaussLegendre)`
/// nodes) — the *point-value* duals of the element's `FE::Nodes` open
/// coordinates, and the shared 1-D factor of the hex NDk element and of the
/// quad `QuadND` element (which is exactly MFEM's sharing: `Poly_1D` is
/// basis-type/degree-keyed, independent of the element shape).
///
/// D721: MFEM's open points are `[0,1]` values and the modes carry **no**
/// normalization factor — the historical `1/2` here was the `[0,1] → [-1,1]`
/// pull-back (`J = J_MFEM/2`, covariant `J^-T`); with the hex reference frame
/// flipped, the physical basis `V_phys = J^-T·Φ_ref` is MFEM's directly.  For
/// `p = 1` the single mode is the constant `1`.
///
/// Returns `(values, derivatives)`.
pub(crate) fn open_basis(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(p >= 1, "open_basis requires p >= 1");
    let nodes = crate::gll_basis::gl_nodes_01(p);
    // MFEM `Poly_1D::Basis::Eval(y, u)` (value-only, stable centre, division
    // form) for the values — the overload `CalcVShape`/`CalcDivShape`/
    // `CalcCurlShape` all use for the open factors — and `Eval(y, u, d)`
    // (reciprocal form) for the derivatives.
    let c = crate::lagrange::factory::mfem_bary_val_01(&nodes, x);
    let (_v, dc) = crate::lagrange::factory::mfem_bary_1d_01(&nodes, x);
    (c, dc)
}

#[cfg(test)]
mod mfem_nodal_dump;

#[cfg(test)]
mod mfem_integrated_dump;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_dofs() {
        assert_eq!(HexNDk::new(1).n_dofs(), 12);
        assert_eq!(HexNDk::new(2).n_dofs(), 54);
        assert_eq!(HexNDk::new(3).n_dofs(), 144);
        assert_eq!(HexNDk::new(4).n_dofs(), 300);
    }
    #[test]
    fn finite() {
        for k in 1..=4 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            for p in &[(0.3, -0.5, 0.7), (0.0, 0.0, 0.0), (-0.8, 0.2, 0.9)] {
                e.eval_basis_vec(&[p.0, p.1, p.2], &mut v);
                for &x in &v {
                    assert!(x.is_finite(), "eval_basis k={k} at {p:?}");
                }
            }
        }
    }
    #[test]
    fn curl_finite() {
        for k in 1..=4 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let mut c = vec![0.0; n * 3];
            for p in &[(0.3, -0.5, 0.7), (0.0, 0.0, 0.0), (-0.8, 0.2, 0.9)] {
                e.eval_curl(&[p.0, p.1, p.2], &mut c);
                for &x in &c {
                    assert!(x.is_finite(), "eval_curl k={k} at {p:?}");
                }
            }
        }
    }
    #[test]
    fn dof_coords_count() {
        for k in 1..=4 {
            assert_eq!(HexNDk::new(k).dof_coords().len(), HexNDk::new(k).n_dofs());
        }
    }

    /// ND1 must be the Whitney form with unit reference edge integral: the
    /// reference basis on edge e0 (y=z=0) is `o_0(x)·hat·hat` with
    /// `o_0 = 1` (both the Gauss-Legendre nodal degree-0 mode and the
    /// integrated mode evaluate to exactly 1 in MFEM's `[0,1]` normalisation,
    /// D721), so the reference line integral along the unit edge
    /// (∫_0^1 phi_ref dx) is exactly 1 — the dual used by
    /// `HCurlSpace::interpolate_vector` and the tangential BC projection.
    #[test]
    fn nd1_unit_edge_integral() {
        let e = HexNDk::new(1);
        let mut v = vec![0.0; e.n_dofs() * 3];
        e.eval_basis_vec(&[0.5, 0.0, 0.0], &mut v);
        // o_0(0.5) = 1, hats = 1.
        assert!((v[0] - 1.0).abs() < 1e-14, "ND1 e0 value {}", v[0]);
        // ∫_0^1 o_0 dx = 1 (5-pt Gauss is exact).
        let (xs, ws) = crate::quadrature::gauss_legendre_01(5);
        let mut acc = 0.0;
        for (q, &x) in xs.iter().enumerate() {
            acc += ws[q] * open_basis(1, x).0[0];
        }
        assert!((acc - 1.0).abs() < 1e-14, "∫ o_0 = {acc}");
    }

    /// MFEM `ND_HexahedronElement(1, GaussLobatto, IntegratedGLL)` reference
    /// values (unit-cube coords) at (x,y,z) = (0.137, -0.413, 0.621), dumped
    /// from mfem-4.9 (tmp harness).  For `p = 1` the integrated and the
    /// Gauss-Legendre nodal open bases coincide (both are the constant 1 in
    /// MFEM's `[0,1]` normalisation), so this table pins the ND1 path
    /// bit-for-bit after the D36 nodal rework.  D721: the fem-rs reference
    /// frame *is* MFEM's `[0,1]³`, so the comparison is direct — no `×2`
    /// (V) / `×4` (curl) physical conversion.
    #[test]
    fn nd1_matches_mfem_reference() {
        let e = HexNDk::new(1);
        let xi = [0.137, -0.413, 0.621];
        let n = e.n_dofs();
        let mut v = vec![0.0; n * 3];
        let mut c = vec![0.0; n * 3];
        e.eval_basis_vec(&xi, &mut v);
        e.eval_curl(&xi, &mut c);
        // MFEM dump (V p=1 q=0): dof | vx vy vz | cx cy cz
        let mfem: [[f64; 6]; 12] = crate::testsupport::mfem_nd1_q0();
        for i in 0..n {
            for d in 0..3 {
                let vr = v[i * 3 + d];
                assert!(
                    (vr - mfem[i][d]).abs() < 5e-14,
                    "V[{i}][{d}]: rust {vr} vs mfem {}",
                    mfem[i][d]
                );
                let cr = c[i * 3 + d];
                assert!(
                    (cr - mfem[i][3 + d]).abs() < 5e-14,
                    "curl[{i}][{d}]: rust {cr} vs mfem {}",
                    mfem[i][3 + d]
                );
            }
        }
    }

    /// Every local DOF functional is the point value `σ_i(Φ) = Φ(x_i)·t̂_i`
    /// with `(x_i, t̂_i) = (dof_coords[i], dof_tangents[i])`: the pairing
    /// matrix `σ_i(φ_j)` must be the identity.  This is the "nodal D32
    /// semantics" that `HCurlSpace::interpolate_vector`, the tangential BC
    /// projection and MFEM's `Project_ND` all rely on — and it also pins the
    /// `dof_coords`/`dof_tangents` layout against `eval_basis_vec`.
    #[test]
    fn dof_functionals_are_point_values() {
        for k in 1..=3 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let tangents = e.dof_tangents();
            assert_eq!(coords.len(), n);
            assert_eq!(tangents.len(), n);
            let mut v = vec![0.0_f64; n * 3];
            for i in 0..n {
                let xi = [coords[i][0], coords[i][1], coords[i][2]];
                let t = tangents[i];
                e.eval_basis_vec(&xi, &mut v);
                for j in 0..n {
                    let d = v[j * 3] * t[0] + v[j * 3 + 1] * t[1] + v[j * 3 + 2] * t[2];
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (d - want).abs() < 1e-12,
                        "k={k}: σ_{i}(φ_{j}) = {d} (want {want})",
                    );
                }
            }
        }
    }

    /// Per-DOF match against the MFEM `ND_HexahedronElement(p, GaussLobatto,
    /// GaussLegendre)` dump (the element `ND_FECollection(p, dim)` builds by
    /// default — what every MFEM miniapp/example uses): for every fem-rs DOF
    /// there is exactly one MFEM DOF carrying the *same function* up to sign
    /// at both sample points, and the match is a bijection.  D721: both sides
    /// live on `[0,1]³`, so the historical `V_femrs = V_mfem/2`,
    /// `curl_femrs = curl_mfem/4` conversions are gone.
    ///
    /// The sign is allowed because MFEM bakes the reference face/edge
    /// orientations into its `dof_map` (`dof2tk` negatives), while fem-rs
    /// keeps the element-local frame and applies the orientation encoding in
    /// `HCurlSpace` — a convention difference, not a different basis.
    #[test]
    fn ndk_matches_mfem_nodal_dump() {
        for k in 1..=3 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            // MFEM unit-cube sample points (the dump's own coordinates).
            let pts = [[0.5685, 0.2935, 0.8105], [0.855, 0.61, 0.235]];
            let mut rust: Vec<Vec<[f64; 6]>> = Vec::new();
            for pt in &pts {
                let mut v = vec![0.0_f64; n * 3];
                let mut c = vec![0.0_f64; n * 3];
                e.eval_basis_vec(pt, &mut v);
                e.eval_curl(pt, &mut c);
                rust.push(
                    (0..n)
                        .map(|i| {
                            [
                                v[i * 3],
                                v[i * 3 + 1],
                                v[i * 3 + 2],
                                c[i * 3],
                                c[i * 3 + 1],
                                c[i * 3 + 2],
                            ]
                        })
                        .collect(),
                );
            }
            let mfem: Vec<Vec<[f64; 6]>> =
                (0..2).map(|q| mfem_nodal_dump::vc(k, q)).collect();
            let mut seen = vec![usize::MAX; n];
            for i in 0..n {
                let mut hits = Vec::new();
                for (j, row) in mfem[0].iter().enumerate() {
                    for sgn in [1.0_f64, -1.0] {
                        let ok = (0..6).all(|d| (rust[0][i][d] - sgn * row[d]).abs() < 1e-12)
                            && (0..6)
                                .all(|d| (rust[1][i][d] - sgn * mfem[1][j][d]).abs() < 1e-12);
                        if ok {
                            hits.push((j, sgn));
                        }
                    }
                }
                assert_eq!(
                    hits.len(),
                    1,
                    "k={k}: fem-rs dof {i} matches {hits:?} MFEM dofs (expected exactly one)",
                );
                seen[i] = hits[0].0;
                if hits[0].1 < 0.0 {
                    // MFEM's orientation sign on this DOF.
                    assert!(hits[0].1 == -1.0);
                }
            }
            let mut sorted = seen.clone();
            sorted.sort_unstable();
            let want: Vec<usize> = (0..n).collect();
            assert_eq!(sorted, want, "k={k}: the DOF match must be a bijection");
        }
    }

    /// Raw reference moments `[∫F_i, ∫curl F_i]` over the reference hex must
    /// match MFEM's dump through the same per-DOF bijection — after D721 the
    /// two reference domains coincide, so the ratio is exactly **1** for both
    /// the values and the curls (the historical `4`/`2` conversions were the
    /// `[-1,1]` pull-back).  This pins the per-DOF magnitudes *by integration*
    /// (independent of the point-value sample above).
    #[test]
    fn ndk_moments_match_mfem_nodal_dump() {
        for k in 1..=3 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let qr = e.quadrature((2 * k + 6) as u8);
            let mut v = vec![0.0_f64; n * 3];
            let mut c = vec![0.0_f64; n * 3];
            let mut m = vec![[0.0_f64; 6]; n];
            for (q, xi) in qr.points.iter().enumerate() {
                let w = qr.weights[q];
                e.eval_basis_vec(xi, &mut v);
                e.eval_curl(xi, &mut c);
                for i in 0..n {
                    for d in 0..3 {
                        m[i][d] += w * v[i * 3 + d];
                        m[i][3 + d] += w * c[i * 3 + d];
                    }
                }
            }
            // Same DOF bijection as the point-value test (recomputed here so
            // the moment check stands alone), at MFEM's own sample coordinates.
            let pts = [[0.5685, 0.2935, 0.8105], [0.855, 0.61, 0.235]];
            let mut rust_vc: Vec<Vec<[f64; 6]>> = Vec::new();
            for pt in &pts {
                let mut vv = vec![0.0_f64; n * 3];
                let mut cc = vec![0.0_f64; n * 3];
                e.eval_basis_vec(pt, &mut vv);
                e.eval_curl(pt, &mut cc);
                rust_vc.push(
                    (0..n)
                        .map(|i| {
                            [
                                vv[i * 3],
                                vv[i * 3 + 1],
                                vv[i * 3 + 2],
                                cc[i * 3],
                                cc[i * 3 + 1],
                                cc[i * 3 + 2],
                            ]
                        })
                        .collect(),
                );
            }
            let mfem_vc: Vec<Vec<[f64; 6]>> =
                (0..2).map(|q| mfem_nodal_dump::vc(k, q)).collect();
            let mm = mfem_nodal_dump::moments(k);
            for i in 0..n {
                let j = (0..n)
                    .find(|&j| {
                        (0..6).all(|d| (rust_vc[0][i][d] - mfem_vc[0][j][d]).abs() < 1e-12)
                            || (0..6).all(|d| (rust_vc[0][i][d] + mfem_vc[0][j][d]).abs() < 1e-12)
                    })
                    .unwrap_or_else(|| panic!("k={k}: no MFEM counterpart for dof {i}"));
                let s = if (0..6).all(|d| (rust_vc[0][i][d] - mfem_vc[0][j][d]).abs() < 1e-12) {
                    1.0
                } else {
                    -1.0
                };
                for d in 0..6 {
                    let want = s * mm[j][d];
                    assert!(
                        (m[i][d] - want).abs() < 1e-12,
                        "k={k} dof {i} (mfem {j}) moment[{d}]: rust {} vs {want}",
                        m[i][d],
                    );
                }
            }
        }
    }

    /// Per-DOF match against the MFEM `ND_HexahedronElement(p, GaussLobatto,
    /// IntegratedGLL)` dump (the LOR-compatible basis pair of MFEM
    /// `fem/lor/lor.hpp`): for every fem-rs DOF there is exactly one MFEM DOF
    /// carrying the *same function* up to sign at both sample points, and the
    /// match is a bijection.  D721: both sides live on `[0,1]³`, so the
    /// historical `V_femrs = V_mfem/2`, `curl_femrs = curl_mfem/4` conversions
    /// are gone and the values are compared directly.  For `p = 1` the
    /// integrated and the Gauss-Legendre nodal open bases coincide (both are
    /// the constant `1` in MFEM's normalisation), so this table also pins the
    /// ND1 path.
    #[test]
    fn ndk_integrated_gll_matches_mfem_dump() {
        for k in 1..=3 {
            let e = HexNDk::new_integrated_gll(k);
            let n = e.n_dofs();
            // MFEM unit-cube sample points (the dump's own coordinates).
            let pts = [[0.5685, 0.2935, 0.8105], [0.855, 0.61, 0.235]];
            let mut rust: Vec<Vec<[f64; 6]>> = Vec::new();
            for pt in &pts {
                let mut v = vec![0.0_f64; n * 3];
                let mut c = vec![0.0_f64; n * 3];
                e.eval_basis_vec(pt, &mut v);
                e.eval_curl(pt, &mut c);
                rust.push(
                    (0..n)
                        .map(|i| {
                            [
                                v[i * 3],
                                v[i * 3 + 1],
                                v[i * 3 + 2],
                                c[i * 3],
                                c[i * 3 + 1],
                                c[i * 3 + 2],
                            ]
                        })
                        .collect(),
                );
            }
            let mfem: Vec<Vec<[f64; 6]>> =
                (0..2).map(|q| mfem_integrated_dump::vc(k, q)).collect();
            let mut seen = vec![usize::MAX; n];
            for i in 0..n {
                let mut hits = Vec::new();
                for (j, row) in mfem[0].iter().enumerate() {
                    for sgn in [1.0_f64, -1.0] {
                        let ok = (0..6).all(|d| (rust[0][i][d] - sgn * row[d]).abs() < 1e-12)
                            && (0..6)
                                .all(|d| (rust[1][i][d] - sgn * mfem[1][j][d]).abs() < 1e-12);
                        if ok {
                            hits.push(j);
                        }
                    }
                }
                assert_eq!(hits.len(), 1, "k={k}: fem-rs dof {i} matched {hits:?} MFEM dofs");
                seen[i] = hits[0];
            }
            let mut sorted = seen.clone();
            sorted.sort_unstable();
            let want: Vec<usize> = (0..n).collect();
            assert_eq!(sorted, want, "k={k}: the DOF match must be a bijection");
        }
    }

    /// The IntegratedGLL open 1-D modes are the Gerritsma edge functions with
    /// the subcell-integral dual property: `∫_{[ξ_a, ξ_{a+1}]} o_j dx = δ_{a,j}`
    /// over the degree-`p` GLL subcells (MFEM `Poly_1D::Basis::EvalIntegrated`
    /// with `scale_integrated = false`), hence unit total integral.
    #[test]
    fn integrated_gll_open_modes_unit_subcell_integrals() {
        for p in 1..=4usize {
            let nodes = crate::gll_basis::gll_nodes(p);
            for j in 0..p {
                let mut total = 0.0;
                for a in 0..p {
                    let (xs, ws) = crate::quadrature::gauss_legendre_arbitrary(8);
                    let mut acc = 0.0;
                    for (q, &x) in xs.iter().enumerate() {
                        // Map the [-1,1] rule onto the subcell [nodes[a], nodes[a+1]].
                        let mid = 0.5 * (nodes[a] + nodes[a + 1]);
                        let half = 0.5 * (nodes[a + 1] - nodes[a]);
                        let cb = ClosedBasis::new(p);
                        let o = cb.integrated(&cb.eval(mid + half * x));
                        acc += ws[q] * half * o[j];
                    }
                    let want = if a == j { 1.0 } else { 0.0 };
                    assert!(
                        (acc - want).abs() < 1e-12,
                        "p={p}: ∫ subcell {a} o_{j} = {acc} (want {want})",
                    );
                    total += acc;
                }
                assert!((total - 1.0).abs() < 1e-12, "p={p}: ∫ o_{j} = {total}");
            }
        }
    }

    /// Curl consistency: the analytic curl must match a central finite
    /// difference of the basis values.
    #[test]
    fn curl_matches_finite_difference() {
        for k in 1..=3 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let eps = 1e-6;
            let pt = [0.137, -0.413, 0.621];
            let eval = |xi: &[f64], out: &mut Vec<f64>| {
                out.clear();
                out.resize(n * 3, 0.0);
                e.eval_basis_vec(xi, out);
            };
            let mut vp = Vec::new();
            let mut vm = Vec::new();
            let mut fd = vec![0.0_f64; n * 3];
            for d in 0..3 {
                let mut xp = pt;
                let mut xm = pt;
                xp[d] += eps;
                xm[d] -= eps;
                eval(&xp, &mut vp);
                eval(&xm, &mut vm);
                for i in 0..n {
                    // curl_x = dv_z/dy - dv_y/dz, etc.
                    fd[i * 3] += (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * eps)
                        * if d == 1 { 1.0 } else { 0.0 };
                    fd[i * 3] -= (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * eps)
                        * if d == 2 { 1.0 } else { 0.0 };
                    fd[i * 3 + 1] += (vp[i * 3] - vm[i * 3]) / (2.0 * eps)
                        * if d == 2 { 1.0 } else { 0.0 };
                    fd[i * 3 + 1] -= (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * eps)
                        * if d == 0 { 1.0 } else { 0.0 };
                    fd[i * 3 + 2] += (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * eps)
                        * if d == 0 { 1.0 } else { 0.0 };
                    fd[i * 3 + 2] -= (vp[i * 3] - vm[i * 3]) / (2.0 * eps)
                        * if d == 1 { 1.0 } else { 0.0 };
                }
            }
            let mut cc = vec![0.0; n * 3];
            e.eval_curl(&pt, &mut cc);
            for i in 0..n * 3 {
                let scale = 1.0 + cc[i].abs();
                assert!(
                    (cc[i] - fd[i]).abs() < 1e-5 * scale,
                    "k={k} curl[{i}]: analytic {} vs fd {}",
                    cc[i],
                    fd[i]
                );
            }
        }
    }
}
