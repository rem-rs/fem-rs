//! Nédélec hexahedral element `ND_k` on the reference hex `[-1,1]^3`.
//!
//! 1:1 port of MFEM `ND_HexahedronElement(p, GaussLobatto, GaussLegendre)` —
//! the element `ND_FECollection(p, dim)` builds by default, i.e. the ND hex
//! element every miniapp/example actually uses (`dpg_maxwell_3d`, the LOR
//! and Maxwell drivers): the **nodal** variant whose DOFs are the point-value
//! functionals `σ_i(Φ) = Φ(x_i)·t̂_i` at MFEM's `FE::Nodes` (Gauss-Legendre
//! open points along the component direction × GLL closed points across it)
//! with the *unnormalized* reference tangents `t̂_i = ±e_a` (MFEM `dof2tk`).
//!
//! Pulled back from MFEM's natural interval `[0,1]` to `[-1,1]`: the physical
//! basis functions on a given hex are *identical* to MFEM's.  The Jacobian
//! factor 2 of the `[-1,1]` map (`J = J_MFEM/2`, covariant transform `J^-T`)
//! is cancelled by the factor 1/2 of every pulled-back open mode — each ND
//! tensor function carries exactly one open factor, along its component
//! direction:
//!
//! ```text
//!     o_a(ξ)  = (1/2)·ℓ_a(ξ̃),   ξ̃ = (ξ+1)/2,
//! ```
//!
//! with `ℓ_a` MFEM's Gauss-Legendre nodal Lagrange polynomial of degree `k-1`
//! at the `k` open points (the very points of `FE::Nodes`), so that
//! `J^-T_ref · Φ_ref = Φ_MFEM` on the unit cube exactly as it did for the
//! (now replaced) integrated-Gerritsma open modes `-Σ_{j<=a} c'_j` — see the
//! round-15 D36 rework.  The previous `IntegratedGLL` basis spanned the same
//! tensor space (so DPG's normal equations were unaffected) but its per-DOF
//! functions were *not* the point-value duals that `HCurlSpace` and MFEM's
//! `Project_ND` assume.
//!
//! Tensor structure per component (`c` = closed GLL nodal mode of degree `k`
//! on `[-1,1]`, `o` = open Gauss-Legendre nodal mode of degree `k-1`, `±`
//! hats are the endpoint GLL modes `c_0`/`c_k`):
//!
//! ```text
//!     x-dofs: o(x)·c(y)·c(z)   y-dofs: c(x)·o(y)·c(z)   z-dofs: c(x)·c(y)·o(z)
//! ```
//!
//! Local DOF order (matches `HCurlSpace`'s element-DOF layout):
//! - `12k` edge dofs in `HCurlSpace::HEX_EDGES` order (MFEM
//!   `Geometry::CUBE::Edges`), `k` open modes per edge along the local edge
//!   direction,
//! - `6·2k(k-1)` face dofs in `HCurlSpace::HEX_QUAD_FACES` order
//!   (z−, z+, y−, y+, x−, x+); each face holds a first-tangent block then a
//!   second-tangent block of `k(k-1)` modes (interior closed index outer,
//!   open index inner),
//! - `3k(k-1)^2` interior dofs (x-, y-, z-blocks; closed indices interior,
//!   open index innermost).
//!
//! Cross-element orientation is *not* baked into the basis (fem-rs orders the
//! face dofs in the element-local frame and `HCurlSpace` applies the
//! orientation encoding); per-dof *magnitudes* therefore match MFEM exactly
//! and MFEM's negatively-oriented dofs differ only by that sign.

use crate::gll_basis::{gl_nodes, ClosedBasis};
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
}

impl HexNDk {
    /// MFEM `ND_HexahedronElement(p, GaussLobatto, GaussLegendre)` — the
    /// default nodal element (D32/D36 semantics).
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "HexNDk requires order >= 1");
        HexNDk { order: p, open: NdOpenBasis::GaussLegendre }
    }

    /// MFEM `ND_HexahedronElement(p, GaussLobatto, IntegratedGLL)` — the
    /// LOR-compatible basis pair.  Same dof count/layout and dof positions as
    /// [`HexNDk::new`]; only the open modes differ.
    pub fn new_integrated_gll(p: usize) -> Self {
        assert!(p >= 1, "HexNDk requires order >= 1");
        HexNDk { order: p, open: NdOpenBasis::IntegratedGLL }
    }

    /// The `p` open 1-D modes along one axis at `x` (reference `[-1,1]`).
    fn open_modes(&self, x: f64) -> Vec<f64> {
        match self.open {
            NdOpenBasis::GaussLegendre => open_basis(self.order, x).0,
            NdOpenBasis::IntegratedGLL => {
                let cb = ClosedBasis::new(self.order);
                cb.integrated(&cb.eval(x))
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
        let vx = ClosedBasis::new(p).eval(xi[0]);
        let vy = ClosedBasis::new(p).eval(xi[1]);
        let vz = ClosedBasis::new(p).eval(xi[2]);
        let (cx, cy, cz) = (&vx.c, &vy.c, &vz.c);
        let oxs = self.open_modes(xi[0]);
        let oys = self.open_modes(xi[1]);
        let ozs = self.open_modes(xi[2]);
        values.fill(0.0);

        // Edge basis in MFEM `Geometry::Constants<Geometry::CUBE>::Edges`
        // order (matches HCurlSpace::HEX_EDGES):
        //   e0 (0,1) x y=-1 z=-1; e1 (1,2) y x=+1 z=-1; e2 (3,2) x y=+1 z=-1;
        //   e3 (0,3) y x=-1 z=-1; e4 (4,5) x y=-1 z=+1; e5 (5,6) y x=+1 z=+1;
        //   e6 (7,6) x y=+1 z=+1; e7 (4,7) y x=-1 z=+1;
        //   e8..e11 z-edges (0,4),(1,5),(2,6),(3,7) = (x,y) (-1,-1),(1,-1),(1,1),(-1,1).
        // For each edge the mode index `j` (0..p) runs the open 1-D modes
        // along the local edge direction (v0 -> v1).
        // Endpoint closed-mode indices: c_0 on the -1 side, c_p on the +1
        // side (== the linear hats only when p = 1).
        let e0 = |s: f64| if s < 0.0 { 0usize } else { p };
        let x_edges = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // (y,z)
        for (ei, &(y0, z0)) in x_edges.iter().enumerate() {
            let hy = cy[e0(y0)];
            let hz = cz[e0(z0)];
            let e = [0usize, 2, 4, 6][ei];
            for j in 0..p {
                values[(e * p + j) * 3] = oxs[j] * hy * hz;
            }
        }
        let y_edges = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,z)
        for (ei, &(x0, z0)) in y_edges.iter().enumerate() {
            let hx = cx[e0(x0)];
            let hz = cz[e0(z0)];
            let e = [1usize, 3, 5, 7][ei];
            for j in 0..p {
                values[(e * p + j) * 3 + 1] = oys[j] * hx * hz;
            }
        }
        let z_edges = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,y)
        for (ei, &(x0, y0)) in z_edges.iter().enumerate() {
            let hx = cx[e0(x0)];
            let hy = cy[e0(y0)];
            let e = 8 + ei;
            for j in 0..p {
                values[(e * p + j) * 3 + 2] = ozs[j] * hx * hy;
            }
        }

        // Face + interior bubbles (k >= 2).
        if p >= 2 {
            let mut off = 12 * p;
            // Faces z=-1 / z=+1: x-tangent block o_j(x)·c_i(y)·hat, then
            //                     y-tangent block o_j(y)·c_i(x)·hat.
            for &zs in &[-1.0, 1.0] {
                let hz = cz[e0(zs)];
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3] = oxs[j] * cy[i] * hz;
                        off += 1;
                    }
                }
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3 + 1] = oys[j] * cx[i] * hz;
                        off += 1;
                    }
                }
            }
            // Faces y=-1 / y=+1: x-tangent block o_j(x)·hat·c_i(z), then
            //                     z-tangent block o_j(z)·c_i(x)·hat.
            for &ys in &[-1.0, 1.0] {
                let hy = cy[e0(ys)];
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3] = oxs[j] * hy * cz[i];
                        off += 1;
                    }
                }
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3 + 2] = ozs[j] * cx[i] * hy;
                        off += 1;
                    }
                }
            }
            // Faces x=-1 / x=+1: y-tangent block o_j(y)·c_i(z)·hat, then
            //                     z-tangent block o_j(z)·c_i(y)·hat.
            for &xs in &[-1.0, 1.0] {
                let hx = cx[e0(xs)];
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3 + 1] = oys[j] * cz[i] * hx;
                        off += 1;
                    }
                }
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3 + 2] = ozs[j] * cy[i] * hx;
                        off += 1;
                    }
                }
            }

            // Interior: 3k(k-1)^2 curl-conforming bubbles, vanishing traces on
            // all faces (both closed factors are interior GLL modes).
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3] = oxs[j] * cy[i] * cz[l];
                        off += 1;
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3 + 1] = oys[j] * cx[i] * cz[l];
                        off += 1;
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        values[off * 3 + 2] = ozs[j] * cx[i] * cy[l];
                        off += 1;
                    }
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let p = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        curl_vals.fill(0.0);

        let vx = ClosedBasis::new(p).eval(x);
        let vy = ClosedBasis::new(p).eval(y);
        let vz = ClosedBasis::new(p).eval(z);
        let (cx, cy, cz) = (&vx.c, &vy.c, &vz.c);
        let (dcx, dcy, dcz) = (&vx.dc, &vy.dc, &vz.dc);
        let oxs = self.open_modes(x);
        let oys = self.open_modes(y);
        let ozs = self.open_modes(z);

        // For a dofs-along-d tensor function o(t_d)·C·D the curl only
        // differentiates the two CLOSED factors (the open direction is the
        // component direction itself):
        //   x: curl = (0, o·C·D', -o·C'·D)
        //   y: curl = (-C·o·D', 0, C'·o·D)
        //   z: curl = (C·D'·o, -C'·D·o, 0)

        // Edge curls in MFEM CUBE edge order (matches eval_basis_vec).
        let e0 = |s: f64| if s < 0.0 { 0usize } else { p };
        let x_edges = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // (y,z)
        for (ei, &(y0, z0)) in x_edges.iter().enumerate() {
            let hy = cy[e0(y0)];
            let hz = cz[e0(z0)];
            let dhy = dcy[e0(y0)];
            let dhz = dcz[e0(z0)];
            let e = [0usize, 2, 4, 6][ei];
            for j in 0..p {
                let d = e * p + j;
                curl_vals[d * 3 + 1] = oxs[j] * hy * dhz;
                curl_vals[d * 3 + 2] = -oxs[j] * dhy * hz;
            }
        }
        let y_edges = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,z)
        for (ei, &(x0, z0)) in y_edges.iter().enumerate() {
            let hx = cx[e0(x0)];
            let hz = cz[e0(z0)];
            let dhx = dcx[e0(x0)];
            let dhz = dcz[e0(z0)];
            let e = [1usize, 3, 5, 7][ei];
            for j in 0..p {
                let d = e * p + j;
                curl_vals[d * 3] = -oys[j] * hx * dhz;
                curl_vals[d * 3 + 2] = oys[j] * dhx * hz;
            }
        }
        let z_edges = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,y)
        for (ei, &(x0, y0)) in z_edges.iter().enumerate() {
            let hx = cx[e0(x0)];
            let hy = cy[e0(y0)];
            let dhx = dcx[e0(x0)];
            let dhy = dcy[e0(y0)];
            let e = 8 + ei;
            for j in 0..p {
                let d = e * p + j;
                curl_vals[d * 3] = ozs[j] * hx * dhy;
                curl_vals[d * 3 + 1] = -ozs[j] * dhx * hy;
            }
        }

        // Face curls (k >= 2); index order mirrors eval_basis_vec.
        if p >= 2 {
            let mut off = 12 * p;
            for &zs in &[-1.0, 1.0] {
                let hz = cz[e0(zs)];
                let dhz = dcz[e0(zs)];
                // x-tangent: Phi = (o_j·c_i·hz, 0, 0)
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3 + 1] = oxs[j] * cy[i] * dhz;
                        curl_vals[off * 3 + 2] = -oxs[j] * dcy[i] * hz;
                        off += 1;
                    }
                }
                // y-tangent: Phi = (0, o_j(y)·c_i(x)·hz, 0)
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3] = -oys[j] * cx[i] * dhz;
                        curl_vals[off * 3 + 2] = dcx[i] * oys[j] * hz;
                        off += 1;
                    }
                }
            }
            for &ys in &[-1.0, 1.0] {
                let hy = cy[e0(ys)];
                let dhy = dcy[e0(ys)];
                // x-tangent: Phi = (o_j·hy·c_i, 0, 0)
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3 + 1] = oxs[j] * hy * dcz[i];
                        curl_vals[off * 3 + 2] = -oxs[j] * dhy * cz[i];
                        off += 1;
                    }
                }
                // z-tangent: Phi = (0, 0, o_j·c_i·hy)
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3] = cx[i] * dhy * ozs[j];
                        curl_vals[off * 3 + 1] = -dcx[i] * hy * ozs[j];
                        off += 1;
                    }
                }
            }
            for &xs in &[-1.0, 1.0] {
                let hx = cx[e0(xs)];
                let dhx = dcx[e0(xs)];
                // y-tangent: Phi = (0, o_j·c_i(z)·hx, 0)
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3] = -hx * dcz[i] * oys[j];
                        curl_vals[off * 3 + 2] = dhx * cz[i] * oys[j];
                        off += 1;
                    }
                }
                // z-tangent: Phi = (0, 0, o_j·c_i·hx)
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3] = hx * dcy[i] * ozs[j];
                        curl_vals[off * 3 + 1] = -dhx * cy[i] * ozs[j];
                        off += 1;
                    }
                }
            }

            // Interior curls (k >= 2); index order mirrors eval_basis_vec.
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3 + 1] = oxs[j] * cy[i] * dcz[l];
                        curl_vals[off * 3 + 2] = -oxs[j] * dcy[i] * cz[l];
                        off += 1;
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3] = -cx[i] * oys[j] * dcz[l];
                        curl_vals[off * 3 + 2] = dcx[i] * oys[j] * cz[l];
                        off += 1;
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        curl_vals[off * 3] = cx[i] * dcy[l] * ozs[j];
                        curl_vals[off * 3 + 1] = -dcx[i] * cy[l] * ozs[j];
                        off += 1;
                    }
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
    /// `σ_i(Φ) = Φ(x_i)·t̂_i`: `t̂_i = 2·e_a` for the component direction `a` of
    /// DOF `i`.  The magnitude 2 is the `[-1,1]` pull-back of MFEM's unit
    /// `tk` (`J = J_MFEM/2`, so `J·t̂ = J_MFEM·tk` is the physical tangent —
    /// the *unnormalized* edge/face vector), which is exactly the dual used by
    /// `HCurlSpace::interpolate_vector` and MFEM
    /// `VectorFiniteElement::Project_ND`.
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
    /// and [`dof_tangents`](Self::dof_tangents), mirroring the tensor blocks
    /// of `eval_basis_vec` exactly (node = the open factor's Gauss-Legendre
    /// point along the *component* direction, GLL points across it).
    fn dof_layout(&self) -> Vec<([f64; 3], [f64; 3])> {
        let p = self.order;
        let gl = gl_nodes(p);
        let gll = crate::gll_basis::gll_nodes(p);
        let n = self.n_dofs();
        let tx = [2.0, 0.0, 0.0];
        let ty = [0.0, 2.0, 0.0];
        let tz = [0.0, 0.0, 2.0];
        let mut c: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(n);
        // Edge DOFs in MFEM CUBE edge order e0..e11 (matching `eval_basis_vec`
        // and `HCurlSpace::HEX_EDGES`).  Each edge carries p open modes
        // anchored at the Gauss-Legendre points (MFEM's ND dof positions).
        let x_edges = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // (y,z)
        let y_edges = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,z)
        let z_edges = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,y)
        for ei in 0..4 {
            let (y0, z0) = x_edges[ei];
            for j in 0..p {
                c.push(([gl[j], y0, z0], tx));
            }
            let (x0, z0) = y_edges[ei];
            for j in 0..p {
                c.push(([x0, gl[j], z0], ty));
            }
        }
        for &(x0, y0) in &z_edges {
            for j in 0..p {
                c.push(([x0, y0, gl[j]], tz));
            }
        }
        if p >= 2 {
            // Face DOFs: (p-1) interior closed anchors x p open anchors, two
            // tangent blocks per face, HEX_QUAD_FACES order (z-, z+, y-, y+,
            // x-, x+).  Matches the eval_basis_vec blocks.
            for &zs in &[-1.0, 1.0] {
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gl[j], gll[i], zs], tx));
                    }
                }
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gll[i], gl[j], zs], ty));
                    }
                }
            }
            for &ys in &[-1.0, 1.0] {
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gl[j], ys, gll[i]], tx));
                    }
                }
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gll[i], ys, gl[j]], tz));
                    }
                }
            }
            for &xs in &[-1.0, 1.0] {
                for i in 1..p {
                    for j in 0..p {
                        c.push(([xs, gl[j], gll[i]], ty));
                    }
                }
                for i in 1..p {
                    for j in 0..p {
                        c.push(([xs, gll[i], gl[j]], tz));
                    }
                }
            }
            // Interior DOFs: 3 components, mirroring eval_basis_vec blocks.
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gl[j], gll[i], gll[l]], tx));
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gll[i], gl[j], gll[l]], ty));
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        c.push(([gll[i], gll[l], gl[j]], tz));
                    }
                }
            }
        }
        while c.len() < n {
            c.push(([0.0; 3], [0.0; 3]));
        }
        c
    }
}

/// Open (tangential) `p`-point Gauss-Legendre nodal modes of degree `p-1` on
/// `[0,1]`, scaled by `scale`.
///
/// These are MFEM's tensor open 1-D functions (`poly1d.GetBasis(p-1,
/// BasisType::GaussLegendre)`, evaluated by `Poly_1D::Basis::Eval` as the
/// degree-`p-1` Lagrange interpolation at the `OpenPoints(p-1, GaussLegendre)`
/// nodes) — the *point-value* duals of the element's `FE::Nodes` open
/// coordinates.  The function is the shared 1-D factor of the hex NDk element
/// (`[-1,1]` reference, `scale = 0.5`) and of the quad `QuadND` element
/// (`[0,1]` reference, `scale = 1.0`), which is exactly MFEM's sharing
/// (`Poly_1D` is basis-type/degree-keyed, independent of the element shape).
///
/// The hex's extra `1/2` is the pull-back normalization of the `[0,1] ->
/// [-1,1]` reference change: with `J = J_MFEM/2` and the covariant map `J^-T`,
/// halving every open factor makes the physical basis (`V_phys = 2·V_ref` on
/// the unit cube) equal to MFEM's, exactly as the former integrated-Gerritsma
/// modes were normalized (unit integral over `[-1,1]` ≡ half of MFEM's `u_a`).
/// For `p = 1` the single hex mode is the constant `1/2`, bit-identical to the
/// old Gerritsma mode `-c'_0`, so the ND1 paths are unchanged; the quad's is
/// the constant `1`.
///
/// Returns `(values, derivatives)`.
pub(crate) fn open_basis_scaled(p: usize, x: f64, scale: f64) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(p >= 1, "open_basis requires p >= 1");
    let nodes = gl_nodes(p);
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
    // Exact evaluation at a node (the barycentric formula is singular there).
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
    // Pull-back normalization to the element's reference interval.
    for v in c.iter_mut() {
        *v *= scale;
    }
    for v in dc.iter_mut() {
        *v *= scale;
    }
    (c, dc)
}

/// Hex flavour of [`open_basis_scaled`]: the `[-1,1]` reference interval
/// (`scale = 0.5`).
#[inline]
pub(crate) fn open_basis(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    open_basis_scaled(p, x, 0.5)
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
    /// reference basis on edge e0 (y=z=-1) is `o_0(x)·hat·hat` with
    /// `o_0 = 1/2` (both the Gauss-Legendre nodal degree-0 mode and the old
    /// integrated mode evaluate to exactly 1/2), so the physical line
    /// integral along a unit edge (∫_{-1}^{1} phi_ref dx, see the Piola math
    /// in the module docs) is exactly 1 — the dual used by
    /// `HCurlSpace::interpolate_vector` and the tangential BC projection.
    #[test]
    fn nd1_unit_edge_integral() {
        let e = HexNDk::new(1);
        let mut v = vec![0.0; e.n_dofs() * 3];
        e.eval_basis_vec(&[0.0, -1.0, -1.0], &mut v);
        // o_0(0) = 1/2, hats = 1.
        assert!((v[0] - 0.5).abs() < 1e-14, "ND1 e0 value {}", v[0]);
        // ∫_{-1}^{1} o_0 dx = 1 (5-pt Gauss is exact).
        let (xs, ws) = crate::quadrature::gauss_legendre_arbitrary(5);
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
    /// MFEM's `[0,1]` normalisation), so this table still pins the ND1 path
    /// bit-for-bit after the D36 nodal rework.  The fem-rs reference lives on
    /// [-1,1]; the *physical* basis on the unit cube (J = diag(1/2), so
    /// V_phys = 2·V_ref and curl_phys = 4·curl_ref) must equal MFEM's.
    #[test]
    fn nd1_matches_mfem_reference() {
        let e = HexNDk::new(1);
        let xi = [2.0 * 0.137 - 1.0, 2.0 * (-0.413) - 1.0, 2.0 * 0.621 - 1.0];
        let n = e.n_dofs();
        let mut v = vec![0.0; n * 3];
        let mut c = vec![0.0; n * 3];
        e.eval_basis_vec(&xi, &mut v);
        e.eval_curl(&xi, &mut c);
        // MFEM dump (V p=1 q=0): dof | vx vy vz | cx cy cz
        let mfem: [[f64; 6]; 12] = crate::testsupport::mfem_nd1_q0();
        for i in 0..n {
            for d in 0..3 {
                let vr = 2.0 * v[i * 3 + d];
                assert!(
                    (vr - mfem[i][d]).abs() < 5e-14,
                    "V[{i}][{d}]: rust {vr} vs mfem {}",
                    mfem[i][d]
                );
                let cr = 4.0 * c[i * 3 + d];
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
    /// (`V_femrs = V_mfem/2`, `curl_femrs = curl_mfem/4` on the unit cube) at
    /// both sample points, and the match is a bijection.
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
            // fem-rs reference points (0.137,-0.413,0.621) and (0.71,0.22,-0.53).
            let pts = [[0.137, -0.413, 0.621], [0.71, 0.22, -0.53]];
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
                                2.0 * v[i * 3],
                                2.0 * v[i * 3 + 1],
                                2.0 * v[i * 3 + 2],
                                4.0 * c[i * 3],
                                4.0 * c[i * 3 + 1],
                                4.0 * c[i * 3 + 2],
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
    /// match MFEM's dump through the same per-DOF bijection
    /// (`∫F_femrs = 4·∫F_mfem`, `∫curl F_femrs = 2·∫curl F_mfem`), which
    /// pins the per-DOF magnitudes *by integration* (independent of the
    /// point-value sample above).
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
            // the moment check stands alone).
            let pts = [[0.137, -0.413, 0.621], [0.71, 0.22, -0.53]];
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
                                2.0 * vv[i * 3],
                                2.0 * vv[i * 3 + 1],
                                2.0 * vv[i * 3 + 2],
                                4.0 * cc[i * 3],
                                4.0 * cc[i * 3 + 1],
                                4.0 * cc[i * 3 + 2],
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
                let scale = [4.0, 4.0, 4.0, 2.0, 2.0, 2.0];
                for d in 0..6 {
                    let want = s * scale[d] * mm[j][d];
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
    /// carrying the *same function* up to sign (`V_femrs = V_mfem/2`,
    /// `curl_femrs = curl_mfem/4` on the unit cube) at both sample points,
    /// and the match is a bijection.  For `p = 1` the integrated and the
    /// Gauss-Legendre nodal open bases coincide (both are the constant `1` in
    /// MFEM's normalisation), so this table also pins the ND1 path.
    #[test]
    fn ndk_integrated_gll_matches_mfem_dump() {
        for k in 1..=3 {
            let e = HexNDk::new_integrated_gll(k);
            let n = e.n_dofs();
            let pts = [[0.137, -0.413, 0.621], [0.71, 0.22, -0.53]];
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
                                2.0 * v[i * 3],
                                2.0 * v[i * 3 + 1],
                                2.0 * v[i * 3 + 2],
                                4.0 * c[i * 3],
                                4.0 * c[i * 3 + 1],
                                4.0 * c[i * 3 + 2],
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
