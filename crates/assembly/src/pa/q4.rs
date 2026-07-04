//! Hex Q4 partial-assembly for diffusion.
//!
//! Uses HexQk generic from fem-element (works for any order).
//! Q4: 125 nodes (5×5×5), 5×5×5 Gauss quadrature.

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;

// ─── 5-point Gauss–Legendre on [-1, 1] (exact for degree 9) ─────────────────
const GL5_PTS: [f64; 5] = [-0.906_179_845_938_664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906_179_845_938_664];
const GL5_WTS: [f64; 5] = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];

// ─── 1D Q4 Lagrange on [-1, -0.5, 0, 0.5, 1] at quadrature points ──────────
const Q4_NODES_1D: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

fn build_1d_basis() -> ([[f64; 5]; 5], [[f64; 5]; 5]) {
    let mut b = [[0.0_f64; 5]; 5];
    let mut d = [[0.0_f64; 5]; 5];
    for q in 0..5 {
        let t = GL5_PTS[q];
        let at_node = Q4_NODES_1D.iter().position(|&n| (t - n).abs() < 1e-15);
        for i in 0..5 {
            if let Some(k) = at_node {
                b[q][i] = if i == k { 1.0 } else { 0.0 };
                if i != k {
                    d[q][i] = 0.0;
                } else {
                    let mut s = 0.0;
                    for j in 0..5 { if j != i { s += 1.0 / (Q4_NODES_1D[i] - Q4_NODES_1D[j]); } }
                    d[q][i] = s;
                }
            } else {
                let mut val = 1.0;
                let mut der = 0.0;
                for j in 0..5 { if j == i { continue; }
                    let denom = Q4_NODES_1D[i] - Q4_NODES_1D[j];
                    val *= (t - Q4_NODES_1D[j]) / denom;
                    der += 1.0 / (t - Q4_NODES_1D[j]);
                }
                der *= val;
                b[q][i] = val; d[q][i] = der;
            }
        }
    }
    (b, d)
}

/// Build PA data for Hex Q4 diffusion via HexQk generic.
pub fn build_hex_q4_pa_data<M: MeshTopology>(mesh: &M, kappa: &dyn Fn(&[f64]) -> f64) -> PaData {
    let n_elems = mesh.n_elements();
    let mut pd = PaData::new(n_elems, 125, 3);
    let hex8_ref: [(f64,f64,f64);8] = [(-1.,-1.,-1.),(1.,-1.,-1.),(1.,1.,-1.),(-1.,1.,-1.),(-1.,-1.,1.),(1.,-1.,1.),(1.,1.,1.),(-1.,1.,1.)];

    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64;3]> = (0..8).map(|i|{let c=mesh.node_coords(nodes[i]);[c[0],c[1],c[2]]}).collect();
        for (qz,&qz_pt) in GL5_PTS.iter().enumerate() { for (qy,&qy_pt) in GL5_PTS.iter().enumerate() { for (qx,&qx_pt) in GL5_PTS.iter().enumerate() {
            let qi = qz*25 + qy*5 + qx;
            let mut jac=[[0.0;3];3];
            for i in 0..8{let(xi,et,zt)=hex8_ref[i];let d_xi=xi*(1.0+et*qy_pt)*(1.0+zt*qz_pt)/8.0;let d_et=(1.0+xi*qx_pt)*et*(1.0+zt*qz_pt)/8.0;let d_zt=(1.0+xi*qx_pt)*(1.0+et*qy_pt)*zt/8.0;for d in 0..3{jac[0][d]+=d_xi*v[i][d];jac[1][d]+=d_et*v[i][d];jac[2][d]+=d_zt*v[i][d];}}
            let d=jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
            let det_j=d.abs();let inv=1.0/d;
            let jit=|i:usize,j:usize|->f64{match(i,j){(0,0)=>(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])*inv,(0,1)=>(jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])*inv,(0,2)=>(jac[0][1]*jac[1][2]-jac[0][2]*jac[1][1])*inv,(1,0)=>(jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])*inv,(1,1)=>(jac[0][0]*jac[2][2]-jac[0][2]*jac[2][0])*inv,(1,2)=>(jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])*inv,(2,0)=>(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])*inv,(2,1)=>(jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])*inv,(2,2)=>(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0])*inv,_=>0.0}};
            let mut xp=[0.0;3];for i in 0..8{let(xi,et,zt)=hex8_ref[i];let phi=(1.0+xi*qx_pt)*(1.0+et*qy_pt)*(1.0+zt*qz_pt)/8.0;for d in 0..3{xp[d]+=phi*v[i][d];}}
            let qd=pd.elem_qp_mut(e,qi);
            for a in 0..3{for b in 0..3{qd[a*3+b]=jit(a,b);}}qd[9]=det_j;qd[10]=kappa(&xp);
        }}}
    }
    pd
}

/// y += A·x for Hex Q4 diffusion (sum-factorized).
pub fn pa_apply_hex_q4(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let (b, d) = build_1d_basis();
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < 125 { continue; }
        let mut xe = [[[0.0_f64; 5]; 5]; 5];
        for iz in 0..5 { for iy in 0..5 { for ix in 0..5 {
            xe[ix][iy][iz] = x[dofs[ix + iy*5 + iz*25] as usize];
        }}}
        let mut ye = [[[0.0_f64; 5]; 5]; 5];

        for qz in 0..5 { for qy in 0..5 { for qx in 0..5 {
            let off = (e*125 + qz*25 + qy*5 + qx) * 11;
            let (jit00,jit01,jit02) = (pd.data[off],pd.data[off+1],pd.data[off+2]);
            let (jit10,jit11,jit12) = (pd.data[off+3],pd.data[off+4],pd.data[off+5]);
            let (jit20,jit21,jit22) = (pd.data[off+6],pd.data[off+7],pd.data[off+8]);
            let sc = GL5_WTS[qx]*GL5_WTS[qy]*GL5_WTS[qz]*pd.data[off+9]*pd.data[off+10];

            let (bq,dq) = (b[qx],d[qx]); let (bqy,dqy) = (b[qy],d[qy]); let (bqz,dqz) = (b[qz],d[qz]);

            // Tensor-contracted flux
            let contract = |op_ξ:&[f64;5], op_η:&[f64;5], op_ζ:&[f64;5]| -> f64 {
                let mut s = 0.0;
                for iz in 0..5{let opz=op_ζ[iz];for iy in 0..5{let opy=op_η[iy]*opz;for ix in 0..5{s+=op_ξ[ix]*opy*xe[ix][iy][iz];}}}
                s
            };
            let (c00,c01,c02) = (contract(&dq,&bqy,&bqz), contract(&bq,&dqy,&bqz), contract(&bq,&bqy,&dqz));
            let (f0,f1,f2) = (jit00*c00+jit01*c01+jit02*c02, jit10*c00+jit11*c01+jit12*c02, jit20*c00+jit21*c01+jit22*c02);

            for iz in 0..5{for iy in 0..5{for ix in 0..5{
                let pg0=jit00*dq[ix]*bqy[iy]*bqz[iz]+jit01*bq[ix]*dqy[iy]*bqz[iz]+jit02*bq[ix]*bqy[iy]*dqz[iz];
                let pg1=jit10*dq[ix]*bqy[iy]*bqz[iz]+jit11*bq[ix]*dqy[iy]*bqz[iz]+jit12*bq[ix]*bqy[iy]*dqz[iz];
                let pg2=jit20*dq[ix]*bqy[iy]*bqz[iz]+jit21*bq[ix]*dqy[iy]*bqz[iz]+jit22*bq[ix]*bqy[iy]*dqz[iz];
                ye[ix][iy][iz] += sc*(pg0*f0+pg1*f1+pg2*f2);
            }}}
        }}}

        for iz in 0..5{for iy in 0..5{for ix in 0..5{
            y[dofs[ix+iy*5+iz*25] as usize] += ye[ix][iy][iz];
        }}}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn hex_q4_pa_finite() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let pd = build_hex_q4_pa_data(&mesh, &|_|1.0);
        assert!(pd.data.iter().all(|v| v.is_finite()));
        assert!(pd.data.iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn hex_q4_sf_self_consistent() {
        // First verify the 1D basis
        let (b, d) = build_1d_basis();
        for q in 0..5 {
            let sum_b: f64 = b[q].iter().sum();
            assert!((sum_b-1.0).abs() < 1e-12, "Q4 POU failed at qp {q}: {sum_b}");
            let sum_d: f64 = d[q].iter().sum();
            assert!(sum_d.abs() < 1e-12, "Q4 grad sum not zero at qp {q}: {sum_d}");
            for i in 0..5 { assert!(b[q][i].is_finite() && d[q][i].is_finite(), "non-finite basis at qp {q}, i={i}"); }
        }

        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let pd = build_hex_q4_pa_data(&mesh, &|_|1.0);
        let n = 125;
        let ed: Vec<Vec<u32>> = vec![(0..125).map(|i| i as u32).collect()];
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n).map(|_|{rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);((rng>>11)as f64)/((1u64<<53)as f64)}).collect();
        let mut y = vec![0.0; n]; pa_apply_hex_q4(&pd, &ed, &x, &mut y);
        assert!(y.iter().all(|v| v.is_finite()), "Q4 PA produced non-finite values");
        // Energy positive
        let e: f64 = (0..n).map(|i| y[i]*x[i]).sum();
        assert!(e > 0.0, "Q4 PA energy {e} should be positive");
        // Constant field gives zero
        let ones: Vec<f64> = vec![1.0; n];
        let mut yc = vec![0.0; n]; pa_apply_hex_q4(&pd, &ed, &ones, &mut yc);
        assert!(yc.iter().map(|v|v.abs()).fold(0.0,f64::max) < 1e-14, "Q4 constant residual");
    }
}
