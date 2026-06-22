//! Lagrange elements on the reference hexahedron `[-1,1]³`.

use crate::quadrature::hex_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Q1 ───────────────────────────────────────────────────────────────────────

/// Trilinear Lagrange element on the reference hex `[-1,1]³` — 8 DOFs.
///
/// Node ordering: bottom face (z=−1) then top face (z=+1), each as a
/// counter-clockwise quad starting from (−1,−1).
///
/// | Index | (ξ, η, ζ)      |
/// |-------|----------------|
/// | 0     | (−1, −1, −1)   |
/// | 1     | (+1, −1, −1)   |
/// | 2     | (+1, +1, −1)   |
/// | 3     | (−1, +1, −1)   |
/// | 4     | (−1, −1, +1)   |
/// | 5     | (+1, −1, +1)   |
/// | 6     | (+1, +1, +1)   |
/// | 7     | (−1, +1, +1)   |
///
/// Basis: φᵢ = (1 + ξᵢ ξ)(1 + ηᵢ η)(1 + ζᵢ ζ) / 8
pub struct HexQ1;

const Q1_NODES: [(f64, f64, f64); 8] = [
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    ( 1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
    ( 1.0, -1.0,  1.0),
    ( 1.0,  1.0,  1.0),
    (-1.0,  1.0,  1.0),
];

impl ReferenceElement for HexQ1 {
    fn dim(&self)    -> u8    { 3 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 8 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        for (i, &(xi_i, eta_i, zeta_i)) in Q1_NODES.iter().enumerate() {
            values[i] = 0.125
                * (1.0 + xi_i   * x)
                * (1.0 + eta_i  * y)
                * (1.0 + zeta_i * z);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        for (i, &(xi_i, eta_i, zeta_i)) in Q1_NODES.iter().enumerate() {
            let f_xi   = 1.0 + xi_i   * x;
            let f_eta  = 1.0 + eta_i  * y;
            let f_zeta = 1.0 + zeta_i * z;
            grads[i * 3]     = 0.125 * xi_i   * f_eta  * f_zeta;
            grads[i * 3 + 1] = 0.125 * eta_i  * f_xi   * f_zeta;
            grads[i * 3 + 2] = 0.125 * zeta_i * f_xi   * f_eta;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { hex_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q1_NODES.iter().map(|&(x, y, z)| vec![x, y, z]).collect()
    }
}

// ─── Q2 ───────────────────────────────────────────────────────────────────────

/// Biquadratic Lagrange element on the reference hex `[-1,1]³` — 27 DOFs.
///
/// Node ordering (MFEM-compatible tensorial):
/// - 0..7:  8 vertices
/// - 8..19: 12 edge midpoints (4 on bottom z=-1, 4 on top z=+1, 4 vertical)
/// - 20..25: 6 face-centre DOFs (□0..□5: ζ=-1, ζ=+1, η=-1, η=+1, ξ=-1, ξ=+1)
/// - 26:    volume-centre DOF (0,0,0)
///
/// Basis: φᵢ = L_ix(ξᵢ)(ξ) · L_iy(ηᵢ)(η) · L_iz(ζᵢ)(ζ)
/// where L(-1), L(0), L(+1) are the quadratic 1-D Lagrange polynomials.
pub struct HexQ2;

const Q2_NODES_HEX: [(f64, f64, f64); 27] = {
    let mut n = [(0.0, 0.0, 0.0); 27];
    // vertices
    n[0] = (-1.0, -1.0, -1.0);
    n[1] = ( 1.0, -1.0, -1.0);
    n[2] = ( 1.0,  1.0, -1.0);
    n[3] = (-1.0,  1.0, -1.0);
    n[4] = (-1.0, -1.0,  1.0);
    n[5] = ( 1.0, -1.0,  1.0);
    n[6] = ( 1.0,  1.0,  1.0);
    n[7] = (-1.0,  1.0,  1.0);
    // edges: bottom z=-1 (0→1, 1→2, 2→3, 3→0)
    n[8]  = ( 0.0, -1.0, -1.0);
    n[9]  = ( 1.0,  0.0, -1.0);
    n[10] = ( 0.0,  1.0, -1.0);
    n[11] = (-1.0,  0.0, -1.0);
    // edges: top z=+1 (4→5, 5→6, 6→7, 7→4)
    n[12] = ( 0.0, -1.0,  1.0);
    n[13] = ( 1.0,  0.0,  1.0);
    n[14] = ( 0.0,  1.0,  1.0);
    n[15] = (-1.0,  0.0,  1.0);
    // edges: vertical (0→4, 1→5, 2→6, 3→7)
    n[16] = (-1.0, -1.0,  0.0);
    n[17] = ( 1.0, -1.0,  0.0);
    n[18] = ( 1.0,  1.0,  0.0);
    n[19] = (-1.0,  1.0,  0.0);
    // face centres: ζ=-1, ζ=+1, η=-1, η=+1, ξ=-1, ξ=+1
    n[20] = ( 0.0,  0.0, -1.0);
    n[21] = ( 0.0,  0.0,  1.0);
    n[22] = ( 0.0, -1.0,  0.0);
    n[23] = ( 0.0,  1.0,  0.0);
    n[24] = (-1.0,  0.0,  0.0);
    n[25] = ( 1.0,  0.0,  0.0);
    // volume centre
    n[26] = ( 0.0,  0.0,  0.0);
    n
};

fn hex_q2_1d(x: f64) -> ([f64; 3], [f64; 3]) {
    let vals = [
        0.5 * x * (x - 1.0),
        1.0 - x * x,
        0.5 * x * (x + 1.0),
    ];
    let ders = [
        0.5 * (2.0 * x - 1.0),
        -2.0 * x,
        0.5 * (2.0 * x + 1.0),
    ];
    (vals, ders)
}

fn coord_to_q2_idx(c: f64) -> usize {
    if c < -0.5 { 0 } else if c > 0.5 { 2 } else { 1 }
}

impl ReferenceElement for HexQ2 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 27 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (lx, _) = hex_q2_1d(x);
        let (ly, _) = hex_q2_1d(y);
        let (lz, _) = hex_q2_1d(z);
        for (i, &(xi_i, eta_i, zeta_i)) in Q2_NODES_HEX.iter().enumerate() {
            let ix = coord_to_q2_idx(xi_i);
            let iy = coord_to_q2_idx(eta_i);
            let iz = coord_to_q2_idx(zeta_i);
            values[i] = lx[ix] * ly[iy] * lz[iz];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (lx, dlx) = hex_q2_1d(x);
        let (ly, dly) = hex_q2_1d(y);
        let (lz, dlz) = hex_q2_1d(z);
        for (i, &(xi_i, eta_i, zeta_i)) in Q2_NODES_HEX.iter().enumerate() {
            let ix = coord_to_q2_idx(xi_i);
            let iy = coord_to_q2_idx(eta_i);
            let iz = coord_to_q2_idx(zeta_i);
            grads[i * 3]     = dlx[ix] * ly[iy]  * lz[iz];
            grads[i * 3 + 1] = lx[ix]  * dly[iy] * lz[iz];
            grads[i * 3 + 2] = lx[ix]  * ly[iy]  * dlz[iz];
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { hex_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q2_NODES_HEX.iter().map(|&(x, y, z)| vec![x, y, z]).collect()
    }
}

// ─── Q3 ───────────────────────────────────────────────────────────────────────

/// Bicubic Lagrange element on the reference hex `[-1,1]³` — 64 DOFs.
///
/// Tensor-product of degree-3 Lagrange polynomials through nodes
/// at ξ ∈ {-1, -1/3, 1/3, 1}.
///
/// Node ordering: vertices → edges → faces → volume (MFEM-compatible).
pub struct HexQ3;

const Q3_NODES_HEX: [(f64, f64, f64); 64] = {
    const N1D: [f64; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut n = [(0.0, 0.0, 0.0); 64];
    // vertices (8)
    n[0] = (N1D[0], N1D[0], N1D[0]);
    n[1] = (N1D[3], N1D[0], N1D[0]);
    n[2] = (N1D[3], N1D[3], N1D[0]);
    n[3] = (N1D[0], N1D[3], N1D[0]);
    n[4] = (N1D[0], N1D[0], N1D[3]);
    n[5] = (N1D[3], N1D[0], N1D[3]);
    n[6] = (N1D[3], N1D[3], N1D[3]);
    n[7] = (N1D[0], N1D[3], N1D[3]);
    // edges (12 × 2 = 24 DOFs on edges)
    n[8]  = (N1D[1], N1D[0], N1D[0]); n[9]  = (N1D[2], N1D[0], N1D[0]);   // e0: 0→1
    n[10] = (N1D[3], N1D[1], N1D[0]); n[11] = (N1D[3], N1D[2], N1D[0]);   // e1: 1→2
    n[12] = (N1D[2], N1D[3], N1D[0]); n[13] = (N1D[1], N1D[3], N1D[0]);   // e2: 2→3 (reversed)
    n[14] = (N1D[0], N1D[2], N1D[0]); n[15] = (N1D[0], N1D[1], N1D[0]);   // e3: 3→0 (reversed)
    n[16] = (N1D[1], N1D[0], N1D[3]); n[17] = (N1D[2], N1D[0], N1D[3]);   // e4: 4→5
    n[18] = (N1D[3], N1D[1], N1D[3]); n[19] = (N1D[3], N1D[2], N1D[3]);   // e5: 5→6
    n[20] = (N1D[2], N1D[3], N1D[3]); n[21] = (N1D[1], N1D[3], N1D[3]);   // e6: 6→7 (reversed)
    n[22] = (N1D[0], N1D[2], N1D[3]); n[23] = (N1D[0], N1D[1], N1D[3]);   // e7: 7→4 (reversed)
    n[24] = (N1D[0], N1D[0], N1D[1]); n[25] = (N1D[0], N1D[0], N1D[2]);   // e8: 0→4
    n[26] = (N1D[3], N1D[0], N1D[1]); n[27] = (N1D[3], N1D[0], N1D[2]);   // e9: 1→5
    n[28] = (N1D[3], N1D[3], N1D[1]); n[29] = (N1D[3], N1D[3], N1D[2]);   // e10: 2→6
    n[30] = (N1D[0], N1D[3], N1D[1]); n[31] = (N1D[0], N1D[3], N1D[2]);   // e11: 3→7
    // faces (6 × (3-1)² = 24 DOFs)
    // Face 0: z=min
    n[32] = (N1D[1], N1D[1], N1D[0]); n[33] = (N1D[2], N1D[1], N1D[0]);
    n[34] = (N1D[1], N1D[2], N1D[0]); n[35] = (N1D[2], N1D[2], N1D[0]);
    // Face 1: z=max
    n[36] = (N1D[1], N1D[1], N1D[3]); n[37] = (N1D[2], N1D[1], N1D[3]);
    n[38] = (N1D[1], N1D[2], N1D[3]); n[39] = (N1D[2], N1D[2], N1D[3]);
    // Face 2: y=min
    n[40] = (N1D[1], N1D[0], N1D[1]); n[41] = (N1D[2], N1D[0], N1D[1]);
    n[42] = (N1D[1], N1D[0], N1D[2]); n[43] = (N1D[2], N1D[0], N1D[2]);
    // Face 3: y=max
    n[44] = (N1D[1], N1D[3], N1D[1]); n[45] = (N1D[2], N1D[3], N1D[1]);
    n[46] = (N1D[1], N1D[3], N1D[2]); n[47] = (N1D[2], N1D[3], N1D[2]);
    // Face 4: x=min
    n[48] = (N1D[0], N1D[1], N1D[1]); n[49] = (N1D[0], N1D[2], N1D[1]);
    n[50] = (N1D[0], N1D[1], N1D[2]); n[51] = (N1D[0], N1D[2], N1D[2]);
    // Face 5: x=max
    n[52] = (N1D[3], N1D[1], N1D[1]); n[53] = (N1D[3], N1D[2], N1D[1]);
    n[54] = (N1D[3], N1D[1], N1D[2]); n[55] = (N1D[3], N1D[2], N1D[2]);
    // Volume interior ((3-1)³ = 8 DOFs)
    n[56] = (N1D[1], N1D[1], N1D[1]); n[57] = (N1D[2], N1D[1], N1D[1]);
    n[58] = (N1D[1], N1D[2], N1D[1]); n[59] = (N1D[2], N1D[2], N1D[1]);
    n[60] = (N1D[1], N1D[1], N1D[2]); n[61] = (N1D[2], N1D[1], N1D[2]);
    n[62] = (N1D[1], N1D[2], N1D[2]); n[63] = (N1D[2], N1D[2], N1D[2]);
    n
};

fn hex_q3_lagrange_1d(x: f64) -> ([f64; 4], [f64; 4]) {
    const NODES: [f64; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];

    let mut vals = [1.0_f64; 4];
    for i in 0..4 {
        for j in 0..4 {
            if j != i {
                vals[i] *= (x - NODES[j]) / (NODES[i] - NODES[j]);
            }
        }
    }

    let mut ders = [0.0_f64; 4];
    for i in 0..4 {
        let mut sum = 0.0;
        for m in 0..4 {
            if m == i { continue; }
            let mut term = 1.0 / (NODES[i] - NODES[m]);
            for j in 0..4 {
                if j != i && j != m {
                    term *= (x - NODES[j]) / (NODES[i] - NODES[j]);
                }
            }
            sum += term;
        }
        ders[i] = sum;
    }
    (vals, ders)
}

fn coord_to_q3_idx(c: f64) -> usize {
    const NODES: [f64; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut best = 0usize;
    let mut best_d = f64::MAX;
    for (i, &n) in NODES.iter().enumerate() {
        let d = (c - n).abs();
        if d < best_d { best_d = d; best = i; }
    }
    best
}

impl ReferenceElement for HexQ3 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 3 }
    fn n_dofs(&self) -> usize { 64 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (lx, _) = hex_q3_lagrange_1d(x);
        let (ly, _) = hex_q3_lagrange_1d(y);
        let (lz, _) = hex_q3_lagrange_1d(z);
        for (i, &(xi_i, eta_i, zeta_i)) in Q3_NODES_HEX.iter().enumerate() {
            let ix = coord_to_q3_idx(xi_i);
            let iy = coord_to_q3_idx(eta_i);
            let iz = coord_to_q3_idx(zeta_i);
            values[i] = lx[ix] * ly[iy] * lz[iz];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let (lx, dlx) = hex_q3_lagrange_1d(x);
        let (ly, dly) = hex_q3_lagrange_1d(y);
        let (lz, dlz) = hex_q3_lagrange_1d(z);
        for (i, &(xi_i, eta_i, zeta_i)) in Q3_NODES_HEX.iter().enumerate() {
            let ix = coord_to_q3_idx(xi_i);
            let iy = coord_to_q3_idx(eta_i);
            let iz = coord_to_q3_idx(zeta_i);
            grads[i * 3]     = dlx[ix] * ly[iy]  * lz[iz];
            grads[i * 3 + 1] = lx[ix]  * dly[iy] * lz[iz];
            grads[i * 3 + 2] = lx[ix]  * ly[iy]  * dlz[iz];
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { hex_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q3_NODES_HEX.iter().map(|&(x, y, z)| vec![x, y, z]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(4);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(4);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-11, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn hex_q1_pou()       { check_pou(&HexQ1); }
    #[test] fn hex_q1_grad_zero() { check_grad_zero(&HexQ1); }

    #[test]
    fn hex_q1_node_dofs() {
        let mut phi = vec![0.0; 8];
        for (i, &(x, y, z)) in Q1_NODES.iter().enumerate() {
            HexQ1.eval_basis(&[x, y, z], &mut phi);
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    // ── HexQ2 ─────────────────────────────────────────────────────────────

    #[test] fn hex_q2_pou()       { check_pou(&HexQ2); }
    #[test] fn hex_q2_grad_zero() { check_grad_zero(&HexQ2); }
    #[test] fn hex_q2_n_dofs()    { assert_eq!(HexQ2.n_dofs(), 27); }

    #[test]
    fn hex_q2_node_dofs() {
        let mut phi = vec![0.0; 27];
        for (i, &(x, y, z)) in Q2_NODES_HEX.iter().enumerate() {
            HexQ2.eval_basis(&[x, y, z], &mut phi);
            for j in 0..27 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn hex_q2_n_dofs_matches_hex_qk() {
        use crate::lagrange::factory::HexQk;
        assert_eq!(HexQ2.n_dofs(), HexQk::new(2).n_dofs());
    }

    // ── HexQ3 ─────────────────────────────────────────────────────────────

    #[test] fn hex_q3_pou()       { check_pou(&HexQ3); }
    #[test] fn hex_q3_grad_zero() { check_grad_zero(&HexQ3); }
    #[test] fn hex_q3_n_dofs()    { assert_eq!(HexQ3.n_dofs(), 64); }

    #[test]
    fn hex_q3_node_dofs() {
        let mut phi = vec![0.0; 64];
        for (i, &(x, y, z)) in Q3_NODES_HEX.iter().enumerate() {
            HexQ3.eval_basis(&[x, y, z], &mut phi);
            for j in 0..64 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test]
    fn hex_q3_n_dofs_matches_hex_qk() {
        use crate::lagrange::factory::HexQk;
        assert_eq!(HexQ3.n_dofs(), HexQk::new(3).n_dofs());
    }

    #[test]
    fn hex_q3_gradient_fd() {
        let h = 1e-7;
        let n = 64;
        let elem = HexQ3;
        let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
            vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n*3]
        );
        for &(x, y, z) in &[(0.3, -0.5, 0.7), (-0.1, 0.2, -0.3)] {
            elem.eval_basis(&[x, y, z], &mut vc);
            elem.eval_basis(&[x+h, y, z], &mut vx);
            elem.eval_basis(&[x, y+h, z], &mut vy);
            elem.eval_basis(&[x, y, z+h], &mut vz);
            elem.eval_grad_basis(&[x, y, z], &mut grads);
            for i in 0..n {
                let fd_x = (vx[i] - vc[i]) / h;
                let fd_y = (vy[i] - vc[i]) / h;
                let fd_z = (vz[i] - vc[i]) / h;
                assert!((grads[i*3] - fd_x).abs() < 1e-5, "({x},{y},{z}) i={i} gx");
                assert!((grads[i*3+1] - fd_y).abs() < 1e-5, "({x},{y},{z}) i={i} gy");
                assert!((grads[i*3+2] - fd_z).abs() < 1e-5, "({x},{y},{z}) i={i} gz");
            }
        }
    }
}
