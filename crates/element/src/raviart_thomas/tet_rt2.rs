//! Raviart-Thomas RT2 element on the reference tetrahedron.
//!
//! # Space: RT_2 = [P_2]^3 + (x,y,z)*P~2
//! dim = 30 + 6 = 36 DOFs.
//!
//! # DOFs (36 total)
//! - 6 normal-flux face moments per face x 4 faces = 24 face DOFs
//! - 12 interior DOFs (integral u.w dV for w in [P_1]^3)
//!
//! Basis evaluation delegates to the generic TetRTk::new(2) implementation.

use crate::quadrature::tet_rule;
use crate::raviart_thomas::TetRTk;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Raviart-Thomas RT2 H(div) element on the reference tetrahedron - 36 DOFs, order 2.
pub struct TetRT2;

impl VectorReferenceElement for TetRT2 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        36
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        TetRTk::new(2).eval_basis_vec(xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        TetRTk::new(2).eval_div(xi, div_vals);
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        // RT2 on reference tetrahedron: 6 face moments per face x 4 faces = 24
        // face DOFs, plus 12 interior DOFs. We assign each DOF a physical-space
        // location: face DOFs sit on the open face (strictly inside the 2-simplex),
        // interior DOFs scatter within the tet. These coordinates are intended for
        // visualization, nodal identification, and DOF-graph layout - not for
        // basis evaluation (which goes through TetRTk).
        let face_lattice: [(f64, f64, f64); 6] = [
            (0.6, 0.2, 0.2),
            (0.2, 0.6, 0.2),
            (0.2, 0.2, 0.6),
            (0.4, 0.4, 0.2),
            (0.4, 0.2, 0.4),
            (0.2, 0.4, 0.4),
        ];
        let mut c = Vec::with_capacity(36);
        // Face 0 (x + y + z = 1):  (a, b, c)
        for &(a, b, cc) in &face_lattice {
            c.push(vec![a, b, cc]);
        }
        // Face 1 (x = 0):  (0, b, c)
        for &(_a, b, cc) in &face_lattice {
            c.push(vec![0.0, b, cc]);
        }
        // Face 2 (y = 0):  (b, 0, c)
        for &(_a, b, cc) in &face_lattice {
            c.push(vec![b, 0.0, cc]);
        }
        // Face 3 (z = 0):  (b, c, 0)
        for &(_a, b, cc) in &face_lattice {
            c.push(vec![b, cc, 0.0]);
        }
        // Interior 12 DOFs - hand-picked symmetric scatter inside the unit tet.
        let interior: [[f64; 3]; 12] = [
            [0.40, 0.20, 0.20],
            [0.20, 0.40, 0.20],
            [0.20, 0.20, 0.40],
            [0.20, 0.20, 0.20],
            [0.30, 0.30, 0.20],
            [0.30, 0.20, 0.30],
            [0.20, 0.30, 0.30],
            [0.25, 0.25, 0.25],
            [0.15, 0.25, 0.35],
            [0.35, 0.15, 0.25],
            [0.25, 0.35, 0.15],
            [0.10, 0.30, 0.30],
        ];
        for &p in &interior {
            c.push(vec![p[0], p[1], p[2]]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_rt2_n_dofs() {
        assert_eq!(TetRT2.n_dofs(), 36);
    }

    #[test]
    fn tet_rt2_basis_finite() {
        let elem = TetRT2;
        let mut v = vec![0.0; 36 * 3];
        for xi in &[
            vec![0., 0., 0.],
            vec![1., 0., 0.],
            vec![0., 1., 0.],
            vec![0., 0., 1.],
            vec![0.25, 0.25, 0.25],
            vec![0.0, 0.5, 0.5],
        ] {
            elem.eval_basis_vec(xi, &mut v);
            for &val in &v {
                assert!(val.is_finite(), "non-finite at {xi:?}: {val}");
            }
        }
    }

    #[test]
    fn tet_rt2_div_finite() {
        let elem = TetRT2;
        let mut div = vec![0.0; 36];
        let qr = elem.quadrature(3);
        for xi in &qr.points {
            elem.eval_div(xi, &mut div);
            for &d in &div {
                assert!(d.is_finite());
            }
        }
    }

    #[test]
    fn tet_rt2_dof_coords_layout() {
        let elem = TetRT2;
        let coords = elem.dof_coords();
        assert_eq!(coords.len(), 36, "RT2 has 36 DOFs");

        // Face DOFs: first 24 must lie on the corresponding face plane.
        // Face 0: x + y + z = 1
        for i in 0..6 {
            let p = &coords[i];
            let s = p[0] + p[1] + p[2];
            assert!((s - 1.0).abs() < 1e-12, "face 0 dof {i}: x+y+z = {s}");
        }
        // Face 1: x = 0
        for i in 6..12 {
            assert!(
                coords[i][0].abs() < 1e-12,
                "face 1 dof {i}: x = {}",
                coords[i][0]
            );
        }
        // Face 2: y = 0
        for i in 12..18 {
            assert!(
                coords[i][1].abs() < 1e-12,
                "face 2 dof {i}: y = {}",
                coords[i][1]
            );
        }
        // Face 3: z = 0
        for i in 18..24 {
            assert!(
                coords[i][2].abs() < 1e-12,
                "face 3 dof {i}: z = {}",
                coords[i][2]
            );
        }
        // Interior DOFs strictly inside the unit tet
        for i in 24..36 {
            let p = &coords[i];
            assert!(
                p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0,
                "interior dof {i} has non-positive coord: {p:?}"
            );
            assert!(
                p[0] + p[1] + p[2] < 1.0 - 1e-12,
                "interior dof {i} not strictly inside: x+y+z = {}",
                p[0] + p[1] + p[2]
            );
        }

        // All face DOFs must be distinct (no two-fold duplicates)
        for i in 0..24 {
            for j in (i + 1)..24 {
                let d = (coords[i][0] - coords[j][0]).abs()
                    + (coords[i][1] - coords[j][1]).abs()
                    + (coords[i][2] - coords[j][2]).abs();
                assert!(d > 1e-9, "face DOFs {i} and {j} coincide");
            }
        }
    }
}
