//! Nédélec hexahedral element `ND_k` on the reference hex `[-1,1]^3`.
//!
//! 1:1 port of MFEM `ND_HexahedronElement(p, GaussLobatto, IntegratedGLL)`
//! (the basis pair MFEM documents for LOR-compatible ND spaces), pulled back
//! from MFEM's natural interval `[0,1]` to `[-1,1]`.  The physical basis
//! functions on a given hex are *identical* to MFEM's: the Jacobian factor 2
//! of the `[-1,1]` map (`J = J_MFEM/2`, covariant transform `J^-T`) cancels
//! the factor 1/2 of the pulled-back integrated open modes — every ND tensor
//! function carries exactly one open factor along its component direction.
//!
//! Tensor structure per component (`c` = closed GLL nodal mode of degree `k`
//! on `[-1,1]`, `o` = open integrated-Gerritsma mode with unit integral over
//! `[-1,1]`, `±` hats are the endpoint GLL modes `c_0`/`c_k`):
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
//! Unlike MFEM, no per-face orientation signs are folded into the basis: the
//! fem-rs `HCurlSpace` orders face dofs in the element-local frame (consistent
//! across neighbours on conforming meshes) and applies edge-orientation signs
//! at the space level.  Per-dof *magnitudes* therefore match MFEM exactly;
//! MFEM's negatively-oriented dofs differ only by the sign MFEM bakes in for
//! its canonical face orientation.

use crate::gll_basis::{gl_nodes, ClosedBasis};
use crate::reference::VectorReferenceElement;

pub struct HexNDk {
    order: usize,
}

impl HexNDk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "HexNDk requires order >= 1");
        HexNDk { order: p }
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
        let oxs = partial_open(&vx.dc);
        let oys = partial_open(&vy.dc);
        let ozs = partial_open(&vz.dc);
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
        let oxs = partial_open(&vx.dc);
        let oys = partial_open(&vy.dc);
        let ozs = partial_open(&vz.dc);

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
        let p = self.order;
        let gl = gl_nodes(p);
        let gll = crate::gll_basis::gll_nodes(p);
        let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        // Edge DOFs in MFEM CUBE edge order e0..e11 (matching `eval_basis_vec`
        // and `HCurlSpace::HEX_EDGES`).  Each edge carries p open modes
        // anchored at the Gauss-Legendre points (MFEM's ND dof positions).
        let x_edges = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // (y,z)
        let y_edges = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,z)
        let z_edges = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,y)
        for ei in 0..4 {
            let (y0, z0) = x_edges[ei];
            for j in 0..p {
                c.push(vec![gl[j], y0, z0]);
            }
            let (x0, z0) = y_edges[ei];
            for j in 0..p {
                c.push(vec![x0, gl[j], z0]);
            }
        }
        for &(x0, y0) in &z_edges {
            for j in 0..p {
                c.push(vec![x0, y0, gl[j]]);
            }
        }
        if p >= 2 {
            // Face DOFs: (p-1) interior closed anchors x p open anchors, two
            // tangent blocks per face, HEX_QUAD_FACES order (z-, z+, y-, y+,
            // x-, x+).  Matches the eval_basis_vec blocks.
            for &zs in &[-1.0, 1.0] {
                for _ in 0..2 {
                    for i in 1..p {
                        for j in 0..p {
                            c.push(vec![gl[j], gll[i], zs]);
                        }
                    }
                }
            }
            for &ys in &[-1.0, 1.0] {
                for _ in 0..2 {
                    for i in 1..p {
                        for j in 0..p {
                            c.push(vec![gl[j], ys, gll[i]]);
                        }
                    }
                }
            }
            for &xs in &[-1.0, 1.0] {
                for _ in 0..2 {
                    for i in 1..p {
                        for j in 0..p {
                            c.push(vec![xs, gll[i], gl[j]]);
                        }
                    }
                }
            }
            // Interior DOFs: 3 components, mirroring eval_basis_vec blocks.
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        c.push(vec![gl[j], gll[i], gll[l]]);
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        c.push(vec![gll[i], gl[j], gll[l]]);
                    }
                }
            }
            for l in 1..p {
                for i in 1..p {
                    for j in 0..p {
                        c.push(vec![gll[i], gll[l], gl[j]]);
                    }
                }
            }
        }
        while c.len() < n {
            c.push(vec![0.0; 3]);
        }
        c
    }
}

/// Integrated (Gerritsma) open modes `o_i = -Σ_{j<=i} c'_j` from the closed
/// basis derivatives (unit integral over [-1,1]; see [`crate::gll_basis`]).
fn partial_open(dc: &[f64]) -> Vec<f64> {
    let n = dc.len() - 1;
    let mut o = vec![0.0_f64; n];
    if n == 0 {
        return o;
    }
    o[0] = -dc[0];
    for i in 1..n {
        o[i] = o[i - 1] - dc[i];
    }
    o
}

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

    /// ND1 must be the integrated (unit edge integral) Whitney form:
    /// the reference basis on edge e0 (y=z=-1) is `o_0(x)·hat·hat` with
    /// `o_0 = 1/2`, so the physical line integral along a unit edge
    /// (∫_{-1}^{1} phi_ref dx, see the Piola math in the module docs) is
    /// exactly 1 — the dual used by `HCurlSpace::interpolate_vector` and the
    /// tangential BC projection.
    #[test]
    fn nd1_unit_edge_integral() {
        let e = HexNDk::new(1);
        let mut v = vec![0.0; e.n_dofs() * 3];
        e.eval_basis_vec(&[0.0, -1.0, -1.0], &mut v);
        // o_0(0) = 1/2, hats = 1.
        assert!((v[0] - 0.5).abs() < 1e-14, "ND1 e0 value {}", v[0]);
        // Unit integral: ∫_{-1}^{1} o_0 dx = 1 (5-pt Gauss is exact).
        let (xs, ws) = crate::quadrature::gauss_legendre_arbitrary(5);
        let cb = ClosedBasis::new(1);
        let mut acc = 0.0;
        for (q, &x) in xs.iter().enumerate() {
            acc += ws[q] * partial_open(&cb.eval(x).dc)[0];
        }
        assert!((acc - 1.0).abs() < 1e-14, "∫ o_0 = {acc}");
    }

    /// MFEM `ND_HexahedronElement(1, GaussLobatto, IntegratedGLL)` reference
    /// values (unit-cube coords) at (x,y,z) = (0.137, -0.413, 0.621), dumped
    /// from mfem-4.9 (tmp harness).  The fem-rs reference lives on [-1,1];
    /// the *physical* basis on the unit cube (J = diag(1/2), so
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

    /// MFEM `ND_HexahedronElement(2, GaussLobatto, IntegratedGLL)` reference
    /// (unit-cube coords) at (0.137, -0.413, 0.621).  MFEM's local dof order
    /// differs from fem-rs's by the face-orientation permutation/sign
    /// machinery fem-rs handles at the space level, so the comparison is on
    /// the sorted magnitude spectrum of the V and curl columns: it locks the
    /// basis *functions* (per-dof scales) without freezing the orientation
    /// convention.
    #[test]
    fn nd2_matches_mfem_reference_spectrum() {
        let e = HexNDk::new(2);
        let xi = [2.0 * 0.137 - 1.0, 2.0 * (-0.413) - 1.0, 2.0 * 0.621 - 1.0];
        let n = e.n_dofs();
        let mut v = vec![0.0; n * 3];
        let mut c = vec![0.0; n * 3];
        e.eval_basis_vec(&xi, &mut v);
        e.eval_curl(&xi, &mut c);
        let mfem = crate::testsupport::mfem_nd2_q0();
        let mut vs: Vec<f64> = (0..n * 3).map(|i| (2.0 * v[i]).abs()).collect();
        let mut cs: Vec<f64> = (0..n * 3).map(|i| (4.0 * c[i]).abs()).collect();
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mv: Vec<f64> = mfem.iter().map(|r| r[0]).chain(mfem.iter().map(|r| r[1]))
            .chain(mfem.iter().map(|r| r[2])).map(f64::abs).collect();
        let mc: Vec<f64> = mfem.iter().map(|r| r[3]).chain(mfem.iter().map(|r| r[4]))
            .chain(mfem.iter().map(|r| r[5])).map(f64::abs).collect();
        let mut mv = mv; let mut mc = mc;
        mv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, (&a, &b)) in vs.iter().zip(mv.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "V spectrum[{i}]: rust {a} vs mfem {b}");
        }
        for (i, (&a, &b)) in cs.iter().zip(mc.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "curl spectrum[{i}]: rust {a} vs mfem {b}");
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
