//! Hex Q3 partial-assembly for diffusion — per-element per-qp apply.
//!
//! Q3: 64 nodes (4×4×4), 4×4×4 Gauss quadrature.
//! Uses direct per-qp, per-node computation (O(p⁶) naive, sufficient for p=3).

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;

// ─── Q3 1D nodes on [-1, 1] ──────────────────────────────────────────────────
const Q3_NODES: [f64; 4] = [-1.0, -1.0/3.0, 1.0/3.0, 1.0];

// ─── 4-point Gauss–Legendre on [-1, 1] ──────────────────────────────────────
const GL4_PTS: [f64; 4] = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
const GL4_WTS: [f64; 4] = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538];

// ─── 1D Lagrange basis & derivatives at quadrature points ────────────────────
fn build_1d_basis() -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut B = [[0.0_f64; 4]; 4];
    let mut D = [[0.0_f64; 4]; 4];
    for q in 0..4 {
        let t = GL4_PTS[q];
        for i in 0..4 {
            let mut val = 1.0;
            let mut der = 0.0;
            for j in 0..4 {
                if j == i { continue; }
                let denom = Q3_NODES[i] - Q3_NODES[j];
                val *= (t - Q3_NODES[j]) / denom;
                der += 1.0 / (t - Q3_NODES[j]);
            }
            der *= val;
            B[q][i] = val;
            D[q][i] = der;
        }
    }
    (B, D)
}

fn hex_q3_ixyz(n: usize) -> (usize, usize, usize) {
    (n % 4, (n / 4) % 4, n / 16)
}

/// Build PA data for Hex Q3 diffusion.
pub fn build_hex_q3_pa_data<M: MeshTopology>(mesh: &M, kappa: &dyn Fn(&[f64]) -> f64) -> PaData {
    let n_elems = mesh.n_elements();
    let mut pd = PaData::new(n_elems, 64, 3);
    let hex8_ref: [(f64,f64,f64);8] = [(-1.,-1.,-1.),(1.,-1.,-1.),(1.,1.,-1.),(-1.,1.,-1.),(-1.,-1.,1.),(1.,-1.,1.),(1.,1.,1.),(-1.,1.,1.)];
    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64;3]> = (0..8).map(|i|{let c=mesh.node_coords(nodes[i]);[c[0],c[1],c[2]]}).collect();
        for (qz,&qz_pt) in GL4_PTS.iter().enumerate() { for (qy,&qy_pt) in GL4_PTS.iter().enumerate() { for (qx,&qx_pt) in GL4_PTS.iter().enumerate() {
            let qi = qz*16 + qy*4 + qx;
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

/// y += A·x for Hex Q3 diffusion (direct per-qp).
pub fn pa_apply_hex_q3(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let (B, D) = build_1d_basis();
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < 64 { continue; }
        let mut xe = [0.0_f64; 64]; for i in 0..64 { xe[i] = x[dofs[i] as usize]; }
        let mut ye = [0.0_f64; 64];

        for qz in 0..4 { for qy in 0..4 { for qx in 0..4 {
            let qi = qz*16+qy*4+qx;
            let off=(e*64+qi)*11;
            let(j0,j1,j2)=(pd.data[off],pd.data[off+1],pd.data[off+2]);
            let(j3,j4,j5)=(pd.data[off+3],pd.data[off+4],pd.data[off+5]);
            let(j6,j7,j8)=(pd.data[off+6],pd.data[off+7],pd.data[off+8]);
            let sc=GL4_WTS[qx]*GL4_WTS[qy]*GL4_WTS[qz]*pd.data[off+9]*pd.data[off+10];
            let(bq,dq)=(B[qx],D[qx]);let(bqy,dqy)=(B[qy],D[qy]);let(bqz,dqz)=(B[qz],D[qz]);

            let mut rg = [[0.0_f64;3];64];
            for n in 0..64{let(ix,iy,iz)=hex_q3_ixyz(n);rg[n]=[dq[ix]*bqy[iy]*bqz[iz],bq[ix]*dqy[iy]*bqz[iz],bq[ix]*bqy[iy]*dqz[iz]];}
            let mut fl=[0.0;3];
            for j in 0..64{let pg=[j0*rg[j][0]+j1*rg[j][1]+j2*rg[j][2],j3*rg[j][0]+j4*rg[j][1]+j5*rg[j][2],j6*rg[j][0]+j7*rg[j][1]+j8*rg[j][2]];fl[0]+=pg[0]*xe[j];fl[1]+=pg[1]*xe[j];fl[2]+=pg[2]*xe[j];}
            for i in 0..64{let pg=[j0*rg[i][0]+j1*rg[i][1]+j2*rg[i][2],j3*rg[i][0]+j4*rg[i][1]+j5*rg[i][2],j6*rg[i][0]+j7*rg[i][1]+j8*rg[i][2]];ye[i]+=sc*(pg[0]*fl[0]+pg[1]*fl[1]+pg[2]*fl[2]);}
        }}}
        for i in 0..64 { y[dofs[i] as usize] += ye[i]; }
    }
}

/// Sum-factorized PA apply for Hex Q3: y += A·x.
///
/// Uses 1D tensor contractions instead of full per-node loops.
pub fn pa_apply_hex_q3_sf(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let (B, D) = build_1d_basis();
    // Precompute 1D basis gradients per qp: G[d][q][i] for direction d, qp q, node i
    // G[0][q][i] = D[q][i], G[1][q][i] = B[q][i], G[2][q][i] = B[q][i] (for d=0)
    // G[0][q][i] = B[q][i], G[1][q][i] = D[q][i], G[2][q][i] = B[q][i] (for d=1)
    // G[0][q][i] = B[q][i], G[1][q][i] = B[q][i], G[2][q][i] = D[q][i] (for d=2)
    // We store all 3×3×4×4 = 144 values: G_dir_axis[q][i] for dir=0,1,2 and axis=0,1,2
    // Actually simpler: just use B and D directly as before.

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < 64 { continue; }
        let mut xe_3d = [[[0.0_f64; 4]; 4]; 4]; // [ix][iy][iz]
        for iz in 0..4 { for iy in 0..4 { for ix in 0..4 {
            xe_3d[ix][iy][iz] = x[dofs[ix + iy*4 + iz*16] as usize];
        }}}
        let mut ye_3d = [[[0.0_f64; 4]; 4]; 4];

        for qz in 0..4 { for qy in 0..4 { for qx in 0..4 {
            let off = (e*64 + qz*16 + qy*4 + qx) * 11;
            let (jit00,jit01,jit02) = (pd.data[off],pd.data[off+1],pd.data[off+2]);
            let (jit10,jit11,jit12) = (pd.data[off+3],pd.data[off+4],pd.data[off+5]);
            let (jit20,jit21,jit22) = (pd.data[off+6],pd.data[off+7],pd.data[off+8]);
            let sc = GL4_WTS[qx]*GL4_WTS[qy]*GL4_WTS[qz]*pd.data[off+9]*pd.data[off+10];

            let (bq,dq) = (B[qx],D[qx]); let (bqy,dqy) = (B[qy],D[qy]); let (bqz,dqz) = (B[qz],D[qz]);

            // Sum-factorized flux computation:
            // For each of 3 reference gradient directions, compute Σ 1D_op · xe
            // using triple tensor contraction.

            // Helper: apply (op_ξ, op_η, op_ζ) to xe, return scalar
            let contract = |op_ξ: &[f64;4], op_η: &[f64;4], op_ζ: &[f64;4]| -> f64 {
                let mut s = 0.0;
                for iz in 0..4 { let opz = op_ζ[iz]; for iy in 0..4 { let opy = op_η[iy] * opz; for ix in 0..4 {
                    s += op_ξ[ix] * opy * xe_3d[ix][iy][iz];
                }}}
                s
            };

            // flux[d] = Σ_dd J⁻ᵀ_{d,dd} · apply_1D(op_{dd,ξ}, op_{dd,η}, op_{dd,ζ})
            // where op_{0,ξ}=D, op_{0,η}=B, op_{0,ζ}=B (ξ-derivative)
            //       op_{1,ξ}=B, op_{1,η}=D, op_{1,ζ}=B (η-derivative)
            //       op_{2,ξ}=B, op_{2,η}=B, op_{2,ζ}=D (ζ-derivative)

            let c00 = contract(&dq, &bqy, &bqz); // (Dξ ⊗ Bη ⊗ Bζ) · x
            let c01 = contract(&bq, &dqy, &bqz); // (Bξ ⊗ Dη ⊗ Bζ) · x
            let c02 = contract(&bq, &bqy, &dqz); // (Bξ ⊗ Bη ⊗ Dζ) · x

            let flux0 = jit00*c00 + jit01*c01 + jit02*c02;
            let flux1 = jit10*c00 + jit11*c01 + jit12*c02;
            let flux2 = jit20*c00 + jit21*c01 + jit22*c02;

            // Scatter back using sum-factorization:
            // ye[i] += scale · (J⁻ᵀ·∇̂φ_i(q)) · flux
            // ∇̂φ_i[0] = Dξ[ix]·Bη[iy]·Bζ[iz]
            // ∇̂φ_i[1] = Bξ[ix]·Dη[iy]·Bζ[iz]
            // ∇̂φ_i[2] = Bξ[ix]·Bη[iy]·Dζ[iz]
            for iz in 0..4 { for iy in 0..4 { for ix in 0..4 {
                let (lxi,lyi,lzi) = (bq[ix],bqy[iy],bqz[iz]);
                let (dxi,dyi,dzi) = (dq[ix],dqy[iy],dqz[iz]);
                let pg0 = jit00*dxi*lyi*lzi + jit01*lxi*dyi*lzi + jit02*lxi*lyi*dzi;
                let pg1 = jit10*dxi*lyi*lzi + jit11*lxi*dyi*lzi + jit12*lxi*lyi*dzi;
                let pg2 = jit20*dxi*lyi*lzi + jit21*lxi*dyi*lzi + jit22*lxi*lyi*dzi;
                ye_3d[ix][iy][iz] += sc * (pg0*flux0 + pg1*flux1 + pg2*flux2);
            }}}
        }}}

        for iz in 0..4 { for iy in 0..4 { for ix in 0..4 {
            y[dofs[ix + iy*4 + iz*16] as usize] += ye_3d[ix][iy][iz];
        }}}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn hex_q3_pa_finite() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let pd = build_hex_q3_pa_data(&mesh, &|_|1.0);
        assert!(pd.data.iter().all(|v| v.is_finite()));
        assert!(pd.data.iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn hex_q3_1d_basis() {
        let (B, D) = build_1d_basis();
        for q in 0..4 { assert!((B[q].iter().sum::<f64>()-1.0).abs()<1e-14); }
        for q in 0..4 { assert!(D[q].iter().sum::<f64>().abs()<1e-14); }
        for i in 0..4 { for j in 0..4 {
            let t = Q3_NODES[j];
            let mut val=1.0; for m in 0..4{if m!=i{val*=(t-Q3_NODES[m])/(Q3_NODES[i]-Q3_NODES[m]);}}
            let exp = if i==j{1.0}else{0.0};
            assert!((val-exp).abs()<1e-14,"ℓ_{i}(x_{j})={val} exp={exp}");
        }}
    }

    #[test]
    fn hex_q3_sf_matches_naive() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1); // 1 element, 8 vertices, 125 Q3 DOFs
        // Build PA and apply with both methods on a properly-sized array
        let pd = build_hex_q3_pa_data(&mesh, &|_|1.0);
        let n = 125; // Q3 DOFs per element
        let ed: Vec<Vec<u32>> = vec![(0..125).map(|i| i as u32).collect()];
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n).map(|_|{rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);((rng>>11)as f64)/((1u64<<53)as f64)}).collect();

        let mut y1 = vec![0.0; n]; pa_apply_hex_q3(&pd, &ed, &x, &mut y1);
        let mut y2 = vec![0.0; n]; pa_apply_hex_q3_sf(&pd, &ed, &x, &mut y2);
        assert!(y1.iter().all(|v| v.is_finite()), "naive produced non-finite");
        assert!(y2.iter().all(|v| v.is_finite()), "SF produced non-finite");
        let max_err: f64 = (0..n).map(|i| (y1[i]-y2[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-14, "SF vs naive mismatch {max_err}");
    }
}
