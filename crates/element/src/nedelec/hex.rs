//! Nedelec-I lowest-order element on the reference hexahedron `[-1,1]^3`.
//!
//! Local DOF ordering MUST match `HCurlSpace::HEX_EDGES` (crates/space/hcurl.rs):
//! - e0..e3: x-directed edges, (y,z) = (-1,-1),(1,-1),(1,1),(-1,1), all +x
//! - e4..e7: y-directed edges, (x,z) = (-1,-1),(1,-1),(1,1),(-1,1), all +y
//! - e8..e11: z-directed edges, (x,y) = (-1,-1),(1,-1),(1,1),(-1,1), all +z

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

        // x-directed edges: e0..e3 (all +x)
        values[0] = 0.125 * (1.0 - y) * (1.0 - z);
        values[1] = 0.0;
        values[2] = 0.0;

        values[3] = 0.125 * (1.0 + y) * (1.0 - z);
        values[4] = 0.0;
        values[5] = 0.0;

        values[6] = 0.125 * (1.0 + y) * (1.0 + z);
        values[7] = 0.0;
        values[8] = 0.0;

        values[9] = 0.125 * (1.0 - y) * (1.0 + z);
        values[10] = 0.0;
        values[11] = 0.0;

        // y-directed edges: e4..e7 (all +y)
        values[12] = 0.0;
        values[13] = 0.125 * (1.0 - x) * (1.0 - z);
        values[14] = 0.0;

        values[15] = 0.0;
        values[16] = 0.125 * (1.0 + x) * (1.0 - z);
        values[17] = 0.0;

        values[18] = 0.0;
        values[19] = 0.125 * (1.0 + x) * (1.0 + z);
        values[20] = 0.0;

        values[21] = 0.0;
        values[22] = 0.125 * (1.0 - x) * (1.0 + z);
        values[23] = 0.0;

        // z-directed edges: e8..e11 (all +z)
        values[24] = 0.0;
        values[25] = 0.0;
        values[26] = 0.125 * (1.0 - x) * (1.0 - y);

        values[27] = 0.0;
        values[28] = 0.0;
        values[29] = 0.125 * (1.0 + x) * (1.0 - y);

        values[30] = 0.0;
        values[31] = 0.0;
        values[32] = 0.125 * (1.0 + x) * (1.0 + y);

        values[33] = 0.0;
        values[34] = 0.0;
        values[35] = 0.125 * (1.0 - x) * (1.0 + y);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        // curl of (f,0,0) = (0, -∂f/∂z, ∂f/∂y); f = 0.125(1±y)(1±z)
        // MFEM hex ND1 convention: x-directed dofs carry the OPPOSITE curl
        // sign (0, +∂f/∂z, -∂f/∂y) — matches MFEM CalcCurlShape, needed for
        // curl-curl matrices to agree bit-for-bit.
        // e0 (y,z)=(-1,-1)
        curl_vals[0] = 0.0;
        curl_vals[1] = -0.125 * (1.0 - y);
        curl_vals[2] = 0.125 * (1.0 - z);
        // e1 (y,z)=(1,-1)
        curl_vals[3] = 0.0;
        curl_vals[4] = -0.125 * (1.0 + y);
        curl_vals[5] = -0.125 * (1.0 - z);
        // e2 (y,z)=(1,1)
        curl_vals[6] = 0.0;
        curl_vals[7] = 0.125 * (1.0 + y);
        curl_vals[8] = -0.125 * (1.0 + z);
        // e3 (y,z)=(-1,1)
        curl_vals[9] = 0.0;
        curl_vals[10] = 0.125 * (1.0 - y);
        curl_vals[11] = 0.125 * (1.0 + z);

        // curl of (0,f,0) = (∂f/∂z·? no: (-∂f/∂z, 0, ∂f/∂x))
        // e4 (x,z)=(-1,-1)
        curl_vals[12] = 0.125 * (1.0 - x);
        curl_vals[13] = 0.0;
        curl_vals[14] = -0.125 * (1.0 - z);
        // e5 (x,z)=(1,-1)
        curl_vals[15] = 0.125 * (1.0 + x);
        curl_vals[16] = 0.0;
        curl_vals[17] = 0.125 * (1.0 - z);
        // e6 (x,z)=(1,1)
        curl_vals[18] = -0.125 * (1.0 + x);
        curl_vals[19] = 0.0;
        curl_vals[20] = 0.125 * (1.0 + z);
        // e7 (x,z)=(-1,1)
        curl_vals[21] = -0.125 * (1.0 - x);
        curl_vals[22] = 0.0;
        curl_vals[23] = -0.125 * (1.0 + z);

        // curl of (0,0,f) = (∂f/∂y, -∂f/∂x, 0)
        // e8 (x,y)=(-1,-1)
        curl_vals[24] = -0.125 * (1.0 - x);
        curl_vals[25] = 0.125 * (1.0 - y);
        curl_vals[26] = 0.0;
        // e9 (x,y)=(1,-1)
        curl_vals[27] = -0.125 * (1.0 + x);
        curl_vals[28] = -0.125 * (1.0 - y);
        curl_vals[29] = 0.0;
        // e10 (x,y)=(1,1)
        curl_vals[30] = 0.125 * (1.0 + x);
        curl_vals[31] = -0.125 * (1.0 + y);
        curl_vals[32] = 0.0;
        // e11 (x,y)=(-1,1)
        curl_vals[33] = 0.125 * (1.0 - x);
        curl_vals[34] = 0.125 * (1.0 + y);
        curl_vals[35] = 0.0;
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
            // x-edges: (y,z) = (-1,-1),(1,-1),(1,1),(-1,1)
            vec![0.0, -1.0, -1.0],
            vec![0.0, 1.0, -1.0],
            vec![0.0, 1.0, 1.0],
            vec![0.0, -1.0, 1.0],
            // y-edges: (x,z) = (-1,-1),(1,-1),(1,1),(-1,1)
            vec![-1.0, 0.0, -1.0],
            vec![1.0, 0.0, -1.0],
            vec![1.0, 0.0, 1.0],
            vec![-1.0, 0.0, 1.0],
            // z-edges: (x,y) = (-1,-1),(1,-1),(1,1),(-1,1)
            vec![-1.0, -1.0, 0.0],
            vec![1.0, -1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![-1.0, 1.0, 0.0],
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
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let edge_len = [2.0_f64; 12];

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
