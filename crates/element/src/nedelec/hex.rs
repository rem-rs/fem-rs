//! Nedelec-I lowest-order element on the reference hexahedron `[0,1]^3`.
//!
//! Local edge ordering:
//! - e0..e3: bottom face perimeter (z=0)
//! - e4..e7: top face perimeter (z=1)
//! - e8..e11: vertical edges (z direction)
//!
//! D721: the reference frame is MFEM's `[0,1]³` (the basis below is MFEM's
//! `ND_HexahedronElement(1)` Whitney form on the unit cube, with the physical
//! covariant Piola map `J^-T` making it a unit edge integral on a unit hex).

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Lowest-order H(curl) element on reference hex, 12 edge DOFs.
pub struct HexND1;

impl VectorReferenceElement for HexND1 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 12 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        // x-directed edges: e0..e3 (against MFEM's `-e_x` for the two
        // descending-x edges, as in MFEM `dof2tk`).
        values[0] = (1.0 - y) * (1.0 - z);
        values[1] = 0.0;
        values[2] = 0.0;

        values[3] = -y * (1.0 - z);
        values[4] = 0.0;
        values[5] = 0.0;

        values[6] = (1.0 - y) * z;
        values[7] = 0.0;
        values[8] = 0.0;

        values[9] = -y * z;
        values[10] = 0.0;
        values[11] = 0.0;

        // y-directed edges: e4..e7
        values[12] = 0.0;
        values[13] = x * (1.0 - z);
        values[14] = 0.0;

        values[15] = 0.0;
        values[16] = (1.0 - x) * (1.0 - z);
        values[17] = 0.0;

        values[18] = 0.0;
        values[19] = x * z;
        values[20] = 0.0;

        values[21] = 0.0;
        values[22] = (1.0 - x) * z;
        values[23] = 0.0;

        // z-directed edges: e8..e11
        values[24] = 0.0;
        values[25] = 0.0;
        values[26] = (1.0 - x) * (1.0 - y);

        values[27] = 0.0;
        values[28] = 0.0;
        values[29] = x * (1.0 - y);

        values[30] = 0.0;
        values[31] = 0.0;
        values[32] = x * y;

        values[33] = 0.0;
        values[34] = 0.0;
        values[35] = (1.0 - x) * y;
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        // True curl of the `[0,1]³` basis above (D721 frame; the old
        // `[-1,1]³` block was the same with each `(1±u)` mapped and the
        // chain factor 2 folded in).
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let (ox, oy, oz) = (1.0 - x, 1.0 - y, 1.0 - z);
        let zero = 0.0_f64;
        let curl: [[f64; 3]; 12] = [
            [zero, -oy, oz],    // e0:  (1-y)(1-z) e_x
            [zero, y, oz],      // e1: -y(1-z) e_x
            [zero, oy, z],      // e2:  (1-y)z e_x
            [zero, -y, z],      // e3: -y z e_x
            [x, zero, oz],      // e4:  x(1-z) e_y
            [ox, zero, -oz],    // e5:  (1-x)(1-z) e_y
            [-x, zero, z],      // e6:  x z e_y
            [-ox, zero, -z],    // e7:  (1-x)z e_y
            [-ox, oy, zero],    // e8:  (1-x)(1-y) e_z
            [-x, -oy, zero],    // e9:  x(1-y) e_z
            [x, -y, zero],      // e10: x y e_z
            [ox, y, zero],      // e11: (1-x)y e_z
        ];
        for (i, c) in curl.iter().enumerate() {
            curl_vals[i * 3] = c[0];
            curl_vals[i * 3 + 1] = c[1];
            curl_vals[i * 3 + 2] = c[2];
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0, 0.0],
            vec![0.5, 1.0, 0.0],
            vec![0.5, 0.0, 1.0],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 0.5, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![1.0, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.0, 0.5],
            vec![1.0, 0.0, 0.5],
            vec![1.0, 1.0, 0.5],
            vec![0.0, 1.0, 0.5],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd1_hex_edge_moments_are_nodal() {
        let elem = HexND1;
        let mut vals = vec![0.0_f64; elem.n_dofs() * 3];

        let tangents = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let edge_len = [1.0_f64; 12];

        for (j, (mid, (t, l))) in elem
            .dof_coords()
            .iter()
            .zip(tangents.iter().zip(edge_len.iter()))
            .enumerate()
        {
            elem.eval_basis_vec(mid, &mut vals);
            for i in 0..elem.n_dofs() {
                let dof = (vals[i * 3] * t[0] + vals[i * 3 + 1] * t[1] + vals[i * 3 + 2] * t[2])
                    * l;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }
}
