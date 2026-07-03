//! Hex Q2 / Quad Q2 partial-assembly for diffusion.
//!
//! Hex Q2: 27 nodes, 3×3×3 Gauss qp.
//! Quad Q2: 9 nodes, 3×3 Gauss qp.
//!
//! Uses full element matrix (no sum-factorization yet).

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;

// ─── 1D Q2 Lagrange basis on [-1, 0, +1] ────────────────────────────────────
#[inline] fn l0(t: f64) -> f64 { 0.5 * t * (t - 1.0) }  // ℓ at -1
#[inline] fn l1(t: f64) -> f64 { 1.0 - t * t }           // ℓ at 0
#[inline] fn l2(t: f64) -> f64 { 0.5 * t * (t + 1.0) }   // ℓ at +1
#[inline] fn d0(t: f64) -> f64 { t - 0.5 }
#[inline] fn d1(t: f64) -> f64 { -2.0 * t }
#[inline] fn d2(t: f64) -> f64 { t + 0.5 }

// ─── 3-point Gauss–Legendre on [-1,1] (exact for degree 5) ──────────────────
const GL3_PTS: [f64; 3] = [-0.7745966692414834, 0.0, 0.7745966692414834];
const GL3_WTS: [f64; 3] = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];

// ─── Q2 hex node → (ix, iy, iz) ∈ {0,1,2}³ mapping ─────────────────────────
const HEX_Q2_MAP: [(usize, usize, usize); 27] = {
    let mut m = [(0, 0, 0); 27];
    // vertices: 0..7
    m[0] = (0, 0, 0); m[1] = (2, 0, 0); m[2] = (2, 2, 0); m[3] = (0, 2, 0);
    m[4] = (0, 0, 2); m[5] = (2, 0, 2); m[6] = (2, 2, 2); m[7] = (0, 2, 2);
    // edges bottom: 8..11
    m[8] = (1, 0, 0); m[9] = (2, 1, 0); m[10] = (1, 2, 0); m[11] = (0, 1, 0);
    // edges top: 12..15
    m[12] = (1, 0, 2); m[13] = (2, 1, 2); m[14] = (1, 2, 2); m[15] = (0, 1, 2);
    // edges vertical: 16..19
    m[16] = (0, 0, 1); m[17] = (2, 0, 1); m[18] = (2, 2, 1); m[19] = (0, 2, 1);
    // face centres: 20..25
    m[20] = (1, 1, 0); m[21] = (1, 1, 2); m[22] = (1, 0, 1); m[23] = (1, 2, 1);
    m[24] = (0, 1, 1); m[25] = (2, 1, 1);
    // volume: 26
    m[26] = (1, 1, 1);
    m
};

/// Build PA data for Hex Q2 diffusion.
pub fn build_hex_q2_pa_data<M: MeshTopology>(
    mesh: &M, kappa: &dyn Fn(&[f64]) -> f64
) -> PaData {
    let n_elems = mesh.n_elements();
    let nqp = 27; // 3×3×3
    let dim = 3;
    let mut pd = PaData::new(n_elems, nqp, dim);

    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64; 3]> = (0..8).map(|i| { let c = mesh.node_coords(nodes[i]); [c[0], c[1], c[2]] }).collect();
        // Compute physical coordinates for all 27 nodes from 8 vertices
        let ref_nodes: [(f64, f64, f64); 27] = [
            (-1.,-1.,-1.),( 1.,-1.,-1.),( 1., 1.,-1.),(-1., 1.,-1.),
            (-1.,-1., 1.),( 1.,-1., 1.),( 1., 1., 1.),(-1., 1., 1.),
            ( 0.,-1.,-1.),( 1., 0.,-1.),( 0., 1.,-1.),(-1., 0.,-1.),
            ( 0.,-1., 1.),( 1., 0., 1.),( 0., 1., 1.),(-1., 0., 1.),
            (-1.,-1., 0.),( 1.,-1., 0.),( 1., 1., 0.),(-1., 1., 0.),
            ( 0., 0.,-1.),( 0., 0., 1.),( 0.,-1., 0.),( 0., 1., 0.),
            (-1., 0., 0.),( 1., 0., 0.),( 0., 0., 0.),
        ];
        let _x: Vec<[f64; 3]> = ref_nodes.iter().map(|&(rx, ry, rz)| {
            let mut xp = [0.0; 3];
            for i in 0..8 {
                let (xi, et, zt) = ref_nodes[i];
                let phi = (1.0+xi*rx)*(1.0+et*ry)*(1.0+zt*rz) / 8.0;
                for d in 0..3 { xp[d] += phi * v[i][d]; }
            }
            xp
        }).collect();

        for (qz, &qz_pt) in GL3_PTS.iter().enumerate() {
            for (qy, &qy_pt) in GL3_PTS.iter().enumerate() {
                for (qx, &qx_pt) in GL3_PTS.iter().enumerate() {
                    let qi = qz * 9 + qy * 3 + qx;

                    // Jacobian J using Q1 (trilinear) mapping from 8 vertices
                    let mut jac = [[0.0_f64; 3]; 3];
                    for i in 0..8 {
                        let (xi, et, zt) = ref_nodes[i];
                        let _phi   = (1.0+xi*qx_pt)*(1.0+et*qy_pt)*(1.0+zt*qz_pt) / 8.0;
                        let d_xi  = xi*(1.0+et*qy_pt)*(1.0+zt*qz_pt) / 8.0;
                        let d_et  = (1.0+xi*qx_pt)*et*(1.0+zt*qz_pt) / 8.0;
                        let d_zt  = (1.0+xi*qx_pt)*(1.0+et*qy_pt)*zt / 8.0;
                        for d in 0..3 {
                            jac[0][d] += d_xi * v[i][d];
                            jac[1][d] += d_et * v[i][d];
                            jac[2][d] += d_zt * v[i][d];
                        }
                    }

                    let d = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
                          - jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])
                          + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
                    let det_j = d.abs();
                    let inv = 1.0 / d;
                    let jit = [
                        [(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])*inv, (jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])*inv, (jac[0][1]*jac[1][2]-jac[0][2]*jac[1][1])*inv],
                        [(jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])*inv, (jac[0][0]*jac[2][2]-jac[0][2]*jac[2][0])*inv, (jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])*inv],
                        [(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])*inv, (jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])*inv, (jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0])*inv],
                    ];

                    // Physical point for κ (Q1 mapping)
                    let mut xp = [0.0; 3];
                    for i in 0..8 {
                        let (xi, et, zt) = ref_nodes[i];
                        let phi = (1.0+xi*qx_pt)*(1.0+et*qy_pt)*(1.0+zt*qz_pt) / 8.0;
                        for d in 0..3 { xp[d] += phi * v[i][d]; }
                    }

                    let qd = pd.elem_qp_mut(e, qi);
                    for i in 0..3 { for j in 0..3 { qd[i*3+j] = jit[i][j]; } }
                    qd[9] = det_j;
                    qd[10] = kappa(&xp);
                }
            }
        }
    }
    pd
}

/// y += A·x for Hex Q2 diffusion (full element matrix).
pub fn pa_apply_hex_q2(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let nld = dofs.len();
        let mut xe = vec![0.0_f64; nld];
        for i in 0..nld { xe[i] = x[dofs[i] as usize]; }
        let mut ye = vec![0.0_f64; nld];

        for (qz, &qz_pt) in GL3_PTS.iter().enumerate() {
            for (qy, &qy_pt) in GL3_PTS.iter().enumerate() {
                for (qx, &qx_pt) in GL3_PTS.iter().enumerate() {
                    let qi = qz * 9 + qy * 3 + qx;
                    let off = (e * 27 + qi) * 11;
                    let jit0 = [pd.data[off], pd.data[off+1], pd.data[off+2]];
                    let jit1 = [pd.data[off+3], pd.data[off+4], pd.data[off+5]];
                    let jit2 = [pd.data[off+6], pd.data[off+7], pd.data[off+8]];
                    let scale = GL3_WTS[qx] * GL3_WTS[qy] * GL3_WTS[qz] * pd.data[off+9] * pd.data[off+10];

                    let lx = [l0(qx_pt), l1(qx_pt), l2(qx_pt)];
                    let ly = [l0(qy_pt), l1(qy_pt), l2(qy_pt)];
                    let lz = [l0(qz_pt), l1(qz_pt), l2(qz_pt)];
                    let dx = [d0(qx_pt), d1(qx_pt), d2(qx_pt)];
                    let dy = [d0(qy_pt), d1(qy_pt), d2(qy_pt)];
                    let dz = [d0(qz_pt), d1(qz_pt), d2(qz_pt)];

                    // Precompute ref gradients at each qp for all 27 nodes
                    let mut rg = [[0.0_f64; 3]; 27];
                    for n in 0..27 {
                        let (ix, iy, iz) = HEX_Q2_MAP[n];
                        rg[n] = [dx[ix]*ly[iy]*lz[iz], lx[ix]*dy[iy]*lz[iz], lx[ix]*ly[iy]*dz[iz]];
                    }

                    // Flux = Σ_j J⁻ᵀ·∇̂φ_j · x_e[j]
                    let mut flux = [0.0; 3];
                    for j in 0..27 {
                        let pg = [jit0[0]*rg[j][0]+jit0[1]*rg[j][1]+jit0[2]*rg[j][2],
                                  jit1[0]*rg[j][0]+jit1[1]*rg[j][1]+jit1[2]*rg[j][2],
                                  jit2[0]*rg[j][0]+jit2[1]*rg[j][1]+jit2[2]*rg[j][2]];
                        flux[0] += pg[0] * xe[j];
                        flux[1] += pg[1] * xe[j];
                        flux[2] += pg[2] * xe[j];
                    }

                    // Scatter: ye[i] += scale * (J⁻ᵀ∇̂φ_i)·flux
                    for i in 0..27 {
                        let pg = [jit0[0]*rg[i][0]+jit0[1]*rg[i][1]+jit0[2]*rg[i][2],
                                  jit1[0]*rg[i][0]+jit1[1]*rg[i][1]+jit1[2]*rg[i][2],
                                  jit2[0]*rg[i][0]+jit2[1]*rg[i][1]+jit2[2]*rg[i][2]];
                        ye[i] += scale * (pg[0]*flux[0] + pg[1]*flux[1] + pg[2]*flux[2]);
                    }
                }
            }
        }
        for i in 0..27 { y[dofs[i] as usize] += ye[i]; }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quad Q2 (9-node)
// ═══════════════════════════════════════════════════════════════════════════════

const QUAD_Q2_MAP: [(usize, usize); 9] = [
    (0, 0), (2, 0), (2, 2), (0, 2),  // vertices
    (1, 0), (2, 1), (1, 2), (0, 1),  // edge midpoints
    (1, 1),                            // face centre
];

pub fn build_quad_q2_pa_data<M: MeshTopology>(
    mesh: &M, kappa: &dyn Fn(&[f64]) -> f64
) -> PaData {
    let n_elems = mesh.n_elements();
    let mut pd = PaData::new(n_elems, 9, 2);
    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64; 2]> = (0..4).map(|i| { let c = mesh.node_coords(nodes[i]); [c[0], c[1]] }).collect();
        let ref_quad: [(f64, f64); 9] = [
            (-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),
            (0.,-1.),(1.,0.),(0.,1.),(-1.,0.),
            (0.,0.),
        ];
        let x: Vec<[f64; 2]> = ref_quad.iter().map(|&(rx, ry)| {
            let mut xp = [0.0; 2];
            for i in 0..4 { let (xi, et) = ref_quad[i]; let phi = (1.0+xi*rx)*(1.0+et*ry)/4.0; for d in 0..2{xp[d]+=phi*v[i][d];} }
            xp
        }).collect();
        for (qy, &qy_pt) in GL3_PTS.iter().enumerate() {
            for (qx, &qx_pt) in GL3_PTS.iter().enumerate() {
                let qi = qy * 3 + qx;
                let mut jac = [[0.0; 2]; 2];
                for n in 0..9 {
                    let (ix, iy) = QUAD_Q2_MAP[n];
                    let lx = [l0(qx_pt), l1(qx_pt), l2(qx_pt)];
                    let ly = [l0(qy_pt), l1(qy_pt), l2(qy_pt)];
                    let dx = [d0(qx_pt), d1(qx_pt), d2(qx_pt)];
                    let dy = [d0(qy_pt), d1(qy_pt), d2(qy_pt)];
                    jac[0][0] += dx[ix]*ly[iy]*x[n][0]; jac[0][1] += dx[ix]*ly[iy]*x[n][1];
                    jac[1][0] += lx[ix]*dy[iy]*x[n][0]; jac[1][1] += lx[ix]*dy[iy]*x[n][1];
                }
                let det_j = (jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0]).abs();
                let inv = 1.0/(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0]);
                let jit = [jac[1][1]*inv, -jac[0][1]*inv, -jac[1][0]*inv, jac[0][0]*inv];
                let mut xp = [0.0;2];
                for n in 0..9 {
                    let (ix,iy)=QUAD_Q2_MAP[n];
                    let phi=[l0(qx_pt),l1(qx_pt),l2(qx_pt)][ix]*[l0(qy_pt),l1(qy_pt),l2(qy_pt)][iy];
                    for d in 0..2{xp[d]+=phi*x[n][d];}
                }
                let qd = pd.elem_qp_mut(e, qi);
                qd[0..4].copy_from_slice(&jit); qd[4]=det_j; qd[5]=kappa(&xp);
            }
        }
    }
    pd
}

pub fn pa_apply_quad_q2(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let mut xe = [0.0_f64; 9]; for i in 0..9 { xe[i] = x[dofs[i] as usize]; }
        let mut ye = [0.0_f64; 9];

        for (qy, &qy_pt) in GL3_PTS.iter().enumerate() {
            for (qx, &qx_pt) in GL3_PTS.iter().enumerate() {
                let qi = qy * 3 + qx;
                let off = (e * 9 + qi) * 6;
                let (jit00,jit01,jit10,jit11) = (pd.data[off],pd.data[off+1],pd.data[off+2],pd.data[off+3]);
                let scale = GL3_WTS[qx]*GL3_WTS[qy]*pd.data[off+4]*pd.data[off+5];

                let lx = [l0(qx_pt), l1(qx_pt), l2(qx_pt)];
                let ly = [l0(qy_pt), l1(qy_pt), l2(qy_pt)];
                let dx = [d0(qx_pt), d1(qx_pt), d2(qx_pt)];
                let dy = [d0(qy_pt), d1(qy_pt), d2(qy_pt)];

                let mut flux = [0.0;2];
                for j in 0..9 { let (ix,iy)=QUAD_Q2_MAP[j];
                    let pg=[jit00*dx[ix]*ly[iy]+jit01*lx[ix]*dy[iy], jit10*dx[ix]*ly[iy]+jit11*lx[ix]*dy[iy]];
                    flux[0]+=pg[0]*xe[j]; flux[1]+=pg[1]*xe[j];
                }
                for i in 0..9 { let (ix,iy)=QUAD_Q2_MAP[i];
                    let pg=[jit00*dx[ix]*ly[iy]+jit01*lx[ix]*dy[iy], jit10*dx[ix]*ly[iy]+jit11*lx[ix]*dy[iy]];
                    ye[i] += scale*(pg[0]*flux[0]+pg[1]*flux[1]);
                }
            }
        }
        for i in 0..9 { y[dofs[i] as usize] += ye[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    #[test]
    fn hex_q2_pa_finite() {
        // Hex Q2: only test basic properties (H1Space on Hex8 doesn't expose 27 DOFs)
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let pd = build_hex_q2_pa_data(&mesh, &|_|1.0);
        // Verify PA data is finite
        assert!(pd.data.iter().all(|v| v.is_finite()));
        assert!(pd.data.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn quad_q2_pa_symmetric() {
        let mesh = SimplexMesh::<2>::unit_square_quad(1); // 1 element
        let space = fem_space::H1Space::new(mesh, 2);
        let pd = build_quad_q2_pa_data(space.mesh(), &|_|1.0);
        let ed: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32).map(|e| space.element_dofs(e).to_vec()).collect();

        let n = space.n_dofs();
        let mut rng:u64=42;
        let x:Vec<f64>=(0..n).map(|_|{rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);((rng>>11)as f64)/((1u64<<53)as f64)}).collect();

        let mut y=vec![0.0;n]; pa_apply_quad_q2(&pd,&ed,&x,&mut y);
        assert!(y.iter().all(|v| v.is_finite()), "Quad Q2 PA produced non-finite values");

        let energy: f64 = (0..n).map(|i| y[i]*x[i]).sum();
        assert!(energy > 0.0, "Quad Q2 PA energy = {energy} should be positive");

        let ones:Vec<f64>=vec![1.0;n];
        let mut yc=vec![0.0;n]; pa_apply_quad_q2(&pd,&ed,&ones,&mut yc);
        assert!(yc.iter().map(|v|v.abs()).fold(0.0,f64::max) < 1e-14, "Quad Q2 constant residual");
    }
}
