//! Raviart-Thomas hexahedral element `RT_k` on the reference hex `[-1,1]^3`.
//!
//! 1:1 port of MFEM `RT_HexahedronElement(p, GaussLobatto, IntegratedGLL)`
//! (MFEM order `p`; `RT_k` with `k = p`), the basis pair MFEM documents for
//! LOR-compatible RT spaces, pulled back from MFEM's natural interval `[0,1]`
//! to `[-1,1]`.  The physical basis functions on a given hex are *identical*
//! to MFEM's: the contravariant transform on `[-1,1]` carries `det(J)^-1·J`
//! with `J = J_MFEM/2` (factor 4 per component), cancelling the factor 1/4 of
//! the two pulled-back integrated open modes every RT tensor function has.
//!
//! Tensor structure per component (`c` = closed GLL nodal mode of degree
//! `k+1` on `[-1,1]`, `o` = open integrated-Gerritsma mode with unit integral
//! over `[-1,1]`):
//!
//! ```text
//!     x-dofs: c(x)·o(y)·o(z)   y-dofs: o(x)·c(y)·o(z)   z-dofs: o(x)·o(y)·c(z)
//! ```
//!
//! Local DOF order (matches `HDivSpace::build_3d_hex`):
//! - six face blocks of `(k+1)^2` dofs in `HDivSpace::HEX_FACES` order —
//!   MFEM `Geometry::CUBE::FaceVert`: bottom z−, front y−, right x+, back y+,
//!   left x−, top z+ — so the positional (basis i ↔ element_dofs[i]) pairing
//!   used by the vector assembler is geometrically consistent across
//!   neighbouring hexes;
//! - three interior blocks of `k(k+1)^2` dofs (x/y/z components, closed
//!   interior index `1..=k` innermost per MFEM's loop order).
//!
//! Each face-block function carries the sign of its outward normal
//! (`s = -1` on the −x/−y/−z faces, `+1` on +x/+y/+z) so that the nominal
//! face mode has unit outward flux `∫_F φ·n dA = 1`; this is MFEM's `nk`
//! orientation convention for the positively oriented face dofs and makes
//! the flux dofs exact duals of the basis (matching
//! `HDivSpace`'s flux-dof interpretation).
//!
//! Within each face block the two free (open) indices run (i outer, j inner);
//! the closed factor is anchored at the face's endpoint GLL node.

use crate::gll_basis::{gl_nodes, ClosedBasis};
use crate::reference::VectorReferenceElement;

pub struct HexRTk {
    order: usize,
}

impl HexRTk {
    pub fn new(p: usize) -> Self {
        HexRTk { order: p }
    }
}

impl VectorReferenceElement for HexRTk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        3 * (self.order + 1) * (self.order + 1) * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        values.fill(0.0);

        // Closed basis degree k+1 (k+2 GLL modes); open modes integrated from
        // its derivatives (k+1 modes).
        let vx = ClosedBasis::new(k + 1).eval(x);
        let vy = ClosedBasis::new(k + 1).eval(y);
        let vz = ClosedBasis::new(k + 1).eval(z);
        let (cx, cy, cz) = (&vx.c, &vy.c, &vz.c);
        let ox = partial_open(&vx.dc);
        let oy = partial_open(&vy.dc);
        let oz = partial_open(&vz.dc);
        let m = k + 1;

        let mut off = 0usize;
        // Face blocks in HDivSpace::HEX_FACES order (bottom z-, front y-,
        // right x+, back y+, left x-, top z+); entries are
        // (component axis, face coordinate, outward-normal sign).  The closed
        // mode is anchored at the face endpoint node (index 0 or k+1).
        let faces = [
            (2, -1.0, -1.0),
            (1, -1.0, -1.0),
            (0, 1.0, 1.0),
            (1, 1.0, 1.0),
            (0, -1.0, -1.0),
            (2, 1.0, 1.0),
        ];
        for &(axis, fc, s) in &faces {
            let idx = if fc < 0.0 { 0 } else { k + 1 };
            match axis {
                0 => {
                    // x-comp: s·c_idx(x)·o_i(y)·o_j(z)
                    for j in 0..m {
                        for i in 0..m {
                            values[off * 3] = s * cx[idx] * oy[i] * oz[j];
                            off += 1;
                        }
                    }
                }
                1 => {
                    // y-comp: s·o_i(x)·c_idx(y)·o_j(z)
                    for j in 0..m {
                        for i in 0..m {
                            values[off * 3 + 1] = s * ox[i] * cy[idx] * oz[j];
                            off += 1;
                        }
                    }
                }
                _ => {
                    // z-comp: s·o_i(x)·o_j(y)·c_idx(z)
                    for j in 0..m {
                        for i in 0..m {
                            values[off * 3 + 2] = s * ox[i] * oy[j] * cz[idx];
                            off += 1;
                        }
                    }
                }
            }
        }
        debug_assert_eq!(off, 6 * m * m);

        // Interior blocks (k >= 1), MFEM loop order (closed-interior index
        // innermost): x: c_i(x)·o_j(y)·o_l(z), i = 1..=k.
        if k >= 1 {
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        values[off * 3] = cx[i] * oy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // y: o_i(x)·c_j(y)·o_l(z), j = 1..=k.
            for l in 0..m {
                for j in 1..=k {
                    for i in 0..m {
                        values[off * 3 + 1] = ox[i] * cy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // z: o_i(x)·o_j(y)·c_l(z), l = 1..=k.
            for l in 1..=k {
                for j in 0..m {
                    for i in 0..m {
                        values[off * 3 + 2] = ox[i] * oy[j] * cz[l];
                        off += 1;
                    }
                }
            }
        }
        debug_assert_eq!(off, self.n_dofs());
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        div_vals.fill(0.0);

        let vx = ClosedBasis::new(k + 1).eval(x);
        let vy = ClosedBasis::new(k + 1).eval(y);
        let vz = ClosedBasis::new(k + 1).eval(z);
        let (dcx, dcy, dcz) = (&vx.dc, &vy.dc, &vz.dc);
        let ox = partial_open(&vx.dc);
        let oy = partial_open(&vy.dc);
        let oz = partial_open(&vz.dc);
        let m = k + 1;

        // The divergence only differentiates the single CLOSED factor.
        // Face blocks first, same (axis, coordinate, sign) order as
        // eval_basis_vec.
        let mut off = 0usize;
        let faces = [
            (2, -1.0, -1.0),
            (1, -1.0, -1.0),
            (0, 1.0, 1.0),
            (1, 1.0, 1.0),
            (0, -1.0, -1.0),
            (2, 1.0, 1.0),
        ];
        for &(axis, fc, s) in &faces {
            let idx = if fc < 0.0 { 0 } else { k + 1 };
            match axis {
                0 => {
                    // div = s·c'_idx(x)·o_i(y)·o_j(z)
                    for j in 0..m {
                        for i in 0..m {
                            div_vals[off] = s * dcx[idx] * oy[i] * oz[j];
                            off += 1;
                        }
                    }
                }
                1 => {
                    // div = s·o_i(x)·c'_idx(y)·o_j(z)
                    for j in 0..m {
                        for i in 0..m {
                            div_vals[off] = s * ox[i] * dcy[idx] * oz[j];
                            off += 1;
                        }
                    }
                }
                _ => {
                    // div = s·o_i(x)·o_j(y)·c'_idx(z)
                    for j in 0..m {
                        for i in 0..m {
                            div_vals[off] = s * ox[i] * oy[j] * dcz[idx];
                            off += 1;
                        }
                    }
                }
            }
        }
        if k >= 1 {
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        div_vals[off] = dcx[i] * oy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            for l in 0..m {
                for j in 1..=k {
                    for i in 0..m {
                        div_vals[off] = ox[i] * dcy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            for l in 1..=k {
                for j in 0..m {
                    for i in 0..m {
                        div_vals[off] = ox[i] * oy[j] * dcz[l];
                        off += 1;
                    }
                }
            }
        }
        debug_assert_eq!(off, self.n_dofs());
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        curl_vals.fill(0.0);

        let vx = ClosedBasis::new(k + 1).eval(x);
        let vy = ClosedBasis::new(k + 1).eval(y);
        let vz = ClosedBasis::new(k + 1).eval(z);
        let (cx, cy, cz) = (&vx.c, &vy.c, &vz.c);
        // The curl of an RT tensor function differentiates its two OPEN
        // factors: o'_i = -Σ_{t<=i} c''_t.
        let ox = partial_open(&vx.dc);
        let oy = partial_open(&vy.dc);
        let oz = partial_open(&vz.dc);
        let dox = partial_open(&vx.d2c);
        let doy = partial_open(&vy.d2c);
        let doz = partial_open(&vz.d2c);
        let m = k + 1;

        let mut off = 0usize;
        // Face dofs, same (axis, coordinate, sign) order as eval_basis_vec.
        let faces = [
            (2, -1.0, -1.0),
            (1, -1.0, -1.0),
            (0, 1.0, 1.0),
            (1, 1.0, 1.0),
            (0, -1.0, -1.0),
            (2, 1.0, 1.0),
        ];
        for &(axis, fc, s) in &faces {
            let idx = if fc < 0.0 { 0 } else { k + 1 };
            match axis {
                // x-comp: Phi = s·(c_idx·o_i·o_j, 0, 0)
                //   curl = (0, s·c_idx·o_i·o'_j, -s·c_idx·o'_i·o_j)
                0 => {
                    for j in 0..m {
                        for i in 0..m {
                            curl_vals[off * 3 + 1] = s * cx[idx] * oy[i] * doz[j];
                            curl_vals[off * 3 + 2] = -s * cx[idx] * doy[i] * oz[j];
                            off += 1;
                        }
                    }
                }
                // y-comp: Phi = s·(0, o_i·c_idx·o_j, 0)
                //   curl = (-s·o_i·c_idx·o'_j, 0, s·o'_i·c_idx·o_j)
                1 => {
                    for j in 0..m {
                        for i in 0..m {
                            curl_vals[off * 3] = -s * ox[i] * cy[idx] * doz[j];
                            curl_vals[off * 3 + 2] = s * dox[i] * cy[idx] * oz[j];
                            off += 1;
                        }
                    }
                }
                // z-comp: Phi = s·(0, 0, o_i·o_j·c_idx)
                //   curl = (s·o_i·o'_j·c_idx, -s·o'_i·o_j·c_idx, 0)
                _ => {
                    for j in 0..m {
                        for i in 0..m {
                            curl_vals[off * 3] = s * ox[i] * doy[j] * cz[idx];
                            curl_vals[off * 3 + 1] = -s * dox[i] * oy[j] * cz[idx];
                            off += 1;
                        }
                    }
                }
            }
        }
        if k >= 1 {
            // x-comp interior: Phi = (c_i(x)·o_j(y)·o_l(z), 0, 0)
            //   curl = (0, c_i·o_j·o'_l, -c_i·o'_j·o_l)
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        curl_vals[off * 3 + 1] = cx[i] * oy[j] * doz[l];
                        curl_vals[off * 3 + 2] = -cx[i] * doy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // y-comp interior: Phi = (0, o_i(x)·c_j(y)·o_l(z), 0)
            //   curl = (-o_i·c_j·o'_l, 0, o'_i·c_j·o_l)
            for l in 0..m {
                for j in 1..=k {
                    for i in 0..m {
                        curl_vals[off * 3] = -ox[i] * cy[j] * doz[l];
                        curl_vals[off * 3 + 2] = dox[i] * cy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // z-comp interior: Phi = (0, 0, o_i(x)·o_j(y)·c_l(z))
            //   curl = (o_i·o'_j·c_l, -o'_i·o_j·c_l, 0)
            for l in 1..=k {
                for j in 0..m {
                    for i in 0..m {
                        curl_vals[off * 3] = ox[i] * doy[j] * cz[l];
                        curl_vals[off * 3 + 1] = -dox[i] * oy[j] * cz[l];
                        off += 1;
                    }
                }
            }
        }
        debug_assert_eq!(off, self.n_dofs());
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;
        let m = k + 1;
        if k == 0 {
            return vec![
                vec![0.0, 0.0, -1.0], // z=-1
                vec![0.0, -1.0, 0.0], // y=-1
                vec![1.0, 0.0, 0.0],  // x=+1
                vec![0.0, 1.0, 0.0],  // y=+1
                vec![-1.0, 0.0, 0.0], // x=-1
                vec![0.0, 0.0, 1.0],  // z=+1
            ];
        }
        let gl = gl_nodes(m);
        let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        // Face DOFs: (k+1)^2 per face in HEX_FACES order (bottom z-, front
        // y-, right x+, back y+, left x-, top z+); the normal coordinate sits
        // on the face, free coordinates at the GL points.
        for &(sx, sy, sz) in &[
            (0.0_f64, 0.0, -1.0), // z=-1 (free x, free y)
            (0.0, -1.0, 0.0),     // y=-1 (free x, free z)
            (1.0, 0.0, 0.0),      // x=+1 (free y, free z)
            (0.0, 1.0, 0.0),      // y=+1
            (-1.0, 0.0, 0.0),     // x=-1
            (0.0, 0.0, 1.0),      // z=+1
        ] {
            for j in 0..m {
                for i in 0..m {
                    if sz != 0.0 {
                        c.push(vec![gl[i], gl[j], sz]);
                    } else if sy != 0.0 {
                        c.push(vec![gl[i], sy, gl[j]]);
                    } else {
                        c.push(vec![sx, gl[i], gl[j]]);
                    }
                }
            }
        }
        // Interior DOFs (k >= 1), mirroring the eval blocks.
        for _ in 0..3 * k * m * m {
            c.push(vec![0.0, 0.0, 0.0]);
        }
        while c.len() < n {
            c.push(vec![0.0, 0.0, 0.0]);
        }
        c
    }
}

/// Integrated (Gerritsma) open modes `o_i = -Σ_{j<=i} c'_j` from the closed
/// basis derivative array `d` (unit integral over [-1,1]).
fn partial_open(d: &[f64]) -> Vec<f64> {
    let n = d.len() - 1;
    let mut o = vec![0.0_f64; n];
    if n == 0 {
        return o;
    }
    o[0] = -d[0];
    for i in 1..n {
        o[i] = o[i - 1] - d[i];
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_dofs() {
        assert_eq!(HexRTk::new(0).n_dofs(), 6);
        assert_eq!(HexRTk::new(1).n_dofs(), 36);
        assert_eq!(HexRTk::new(2).n_dofs(), 108);
    }

    #[test]
    fn finite() {
        for k in 0..=2 {
            let e = HexRTk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            let mut d = vec![0.0; n];
            let mut c = vec![0.0; n * 3];
            for p in &[(0.0, 0.0, 0.0), (0.3, -0.5, 0.7), (-0.2, 0.4, -0.6)] {
                e.eval_basis_vec(&[p.0, p.1, p.2], &mut v);
                e.eval_div(&[p.0, p.1, p.2], &mut d);
                e.eval_curl(&[p.0, p.1, p.2], &mut c);
                for val in v.iter().chain(d.iter()).chain(c.iter()) {
                    assert!(val.is_finite());
                }
            }
        }
    }

    /// RT0 must be the unit-flux Whitney form: each face mode has exactly
    /// unit outward flux through its own face and zero flux through the other
    /// five (MFEM RT0 with the IntegratedGLL open basis), with constant
    /// reference divergence 1/8 (physical divergence 1).
    #[test]
    fn rt0_unit_face_fluxes() {
        let e = HexRTk::new(0);
        assert_eq!(e.n_dofs(), 6);
        let (g1, w1) = crate::quadrature::gauss_legendre_arbitrary(3);
        // Faces in HEX_FACES order: bottom z-, front y-, right x+, back y+,
        // left x-, top z+.
        let faces: [(f64, usize); 6] = [
            (-1.0, 2),
            (-1.0, 1),
            (1.0, 0),
            (1.0, 1),
            (-1.0, 0),
            (1.0, 2),
        ];
        let normals: [[f64; 3]; 6] = [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut flux = vec![vec![0.0_f64; 6]; 6];
        for (f, &(fc, axis)) in faces.iter().enumerate() {
            for (fg, &ga) in g1.iter().enumerate() {
                for (fh, &gb) in g1.iter().enumerate() {
                    let xi = match axis {
                        0 => [fc, ga, gb],
                        1 => [ga, fc, gb],
                        _ => [ga, gb, fc],
                    };
                    let wq = w1[fg] * w1[fh];
                    let n = e.n_dofs();
                    let mut v = vec![0.0; n * 3];
                    e.eval_basis_vec(&xi, &mut v);
                    for dof in 0..n {
                        flux[dof][f] += wq
                            * (v[dof * 3] * normals[f][0]
                                + v[dof * 3 + 1] * normals[f][1]
                                + v[dof * 3 + 2] * normals[f][2]);
                    }
                }
            }
        }
        for dof in 0..6 {
            for f in 0..6 {
                let want = if dof == f { 1.0 } else { 0.0 };
                assert!(
                    (flux[dof][f] - want).abs() < 1e-13,
                    "dof {dof} flux through face {f}: {} (want {want})",
                    flux[dof][f]
                );
            }
        }
        let mut d = vec![0.0; 6];
        e.eval_div(&[0.13, -0.42, 0.57], &mut d);
        for di in &d {
            assert!((di - 0.125).abs() < 1e-14);
        }
    }

    /// Divergence consistency: analytic div vs central finite difference.
    #[test]
    fn div_matches_finite_difference() {
        for k in 0..=2 {
            let e = HexRTk::new(k);
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
            let mut fd = vec![0.0_f64; n];
            for d in 0..3 {
                let mut xp = pt;
                let mut xm = pt;
                xp[d] += eps;
                xm[d] -= eps;
                eval(&xp, &mut vp);
                eval(&xm, &mut vm);
                for i in 0..n {
                    fd[i] += (vp[i * 3 + d] - vm[i * 3 + d]) / (2.0 * eps);
                }
            }
            let mut dd = vec![0.0; n];
            e.eval_div(&pt, &mut dd);
            for i in 0..n {
                let scale = 1.0 + dd[i].abs();
                assert!(
                    (dd[i] - fd[i]).abs() < 1e-5 * scale,
                    "k={k} div[{i}]: analytic {} vs fd {}",
                    dd[i],
                    fd[i]
                );
            }
        }
    }

    /// Curl consistency: analytic curl vs central finite difference.
    #[test]
    fn curl_matches_finite_difference() {
        for k in 0..=2 {
            let e = HexRTk::new(k);
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
