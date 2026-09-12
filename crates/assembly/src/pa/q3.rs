//! Hex Q3 partial-assembly for diffusion — per-element per-qp apply.
//!
//! Q3: 64 nodes (4×4×4), 4×4×4 Gauss quadrature.
//! Uses direct per-qp, per-node computation (O(p⁶) naive, sufficient for p=3).
//!
//! The 1-D nodes and the element-local slot order are taken from
//! [`fem_element::lagrange::hex::HexQ3`] (D77): GLL(4) nodes in MFEM's
//! `H1_HexahedronElement(3)` order, which is what `DofManager::build_pk_hex`
//! numbers `H1Space` element DOFs in.  The pre-D77 kernel used an equispaced
//! 4-node table in lexicographic order — a second, silent divergence from the
//! assembly (and from MFEM's `H1_FECollection`).

use crate::pa::hex_layout::{hex_slots, tensor_slots};
use crate::pa::types::PaData;
use fem_element::lagrange::hex::{HexQ1, HexQ3};
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;

// ─── 4-point Gauss–Legendre on [-1, 1] ──────────────────────────────────────
const GL4_PTS: [f64; 4] = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
const GL4_WTS: [f64; 4] = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538];

/// `(1-D GLL nodes, slot → tensor index)` of the 64-slot Q3 hex kernel.
fn hex_q3_slots() -> (Vec<f64>, Vec<[usize; 3]>) {
    hex_slots(&HexQ3)
}

/// The 8 trilinear (Q1) geometry nodes, in the mesh's corner node order.
fn hex_vertices() -> Vec<[f64; 3]> {
    HexQ1
        .dof_coords()
        .into_iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

// ─── 1D Lagrange basis & derivatives at quadrature points ────────────────────
fn build_1d_basis(nodes: &[f64; 4]) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut b = [[0.0_f64; 4]; 4];
    let mut d = [[0.0_f64; 4]; 4];
    for q in 0..4 {
        let t = GL4_PTS[q];
        for i in 0..4 {
            let mut val = 1.0;
            let mut der = 0.0;
            for j in 0..4 {
                if j == i { continue; }
                let denom = nodes[i] - nodes[j];
                val *= (t - nodes[j]) / denom;
                der += 1.0 / (t - nodes[j]);
            }
            der *= val;
            b[q][i] = val;
            d[q][i] = der;
        }
    }
    (b, d)
}

/// Build PA data for Hex Q3 diffusion.
pub fn build_hex_q3_pa_data<M: MeshTopology>(mesh: &M, kappa: &dyn Fn(&[f64]) -> f64) -> PaData {
    let n_elems = mesh.n_elements();
    let mut pd = PaData::new(n_elems, 64, 3);
    let hex8_ref = hex_vertices();
    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64;3]> = (0..8).map(|i|{let c=mesh.node_coords(nodes[i]);[c[0],c[1],c[2]]}).collect();
        for (qz,&qz_pt) in GL4_PTS.iter().enumerate() { for (qy,&qy_pt) in GL4_PTS.iter().enumerate() { for (qx,&qx_pt) in GL4_PTS.iter().enumerate() {
            let qi = qz*16 + qy*4 + qx;
            let mut jac=[[0.0;3];3];
            for i in 0..8{let[xi,et,zt]=hex8_ref[i];let d_xi=xi*(1.0+et*qy_pt)*(1.0+zt*qz_pt)/8.0;let d_et=(1.0+xi*qx_pt)*et*(1.0+zt*qz_pt)/8.0;let d_zt=(1.0+xi*qx_pt)*(1.0+et*qy_pt)*zt/8.0;for d in 0..3{jac[0][d]+=d_xi*v[i][d];jac[1][d]+=d_et*v[i][d];jac[2][d]+=d_zt*v[i][d];}}
            let d=jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])-jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])+jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
            let det_j=d.abs();let inv=1.0/d;
            let jit=|i:usize,j:usize|->f64{match(i,j){(0,0)=>(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])*inv,(0,1)=>(jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])*inv,(0,2)=>(jac[0][1]*jac[1][2]-jac[0][2]*jac[1][1])*inv,(1,0)=>(jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])*inv,(1,1)=>(jac[0][0]*jac[2][2]-jac[0][2]*jac[2][0])*inv,(1,2)=>(jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])*inv,(2,0)=>(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])*inv,(2,1)=>(jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])*inv,(2,2)=>(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0])*inv,_=>0.0}};
            let mut xp=[0.0;3];for i in 0..8{let[xi,et,zt]=hex8_ref[i];let phi=(1.0+xi*qx_pt)*(1.0+et*qy_pt)*(1.0+zt*qz_pt)/8.0;for d in 0..3{xp[d]+=phi*v[i][d];}}
            let qd=pd.elem_qp_mut(e,qi);
            for a in 0..3{for b in 0..3{qd[a*3+b]=jit(a,b);}}qd[9]=det_j;qd[10]=kappa(&xp);
        }}}
    }
    pd
}

/// y += A·x for Hex Q3 diffusion (direct per-qp).
pub fn pa_apply_hex_q3(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let (nodes, slots) = hex_q3_slots();
    let (b, d) = build_1d_basis(&[nodes[0], nodes[1], nodes[2], nodes[3]]);
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
            let(bq,dq)=(b[qx],d[qx]);let(bqy,dqy)=(b[qy],d[qy]);let(bqz,dqz)=(b[qz],d[qz]);

            let mut rg = [[0.0_f64;3];64];
            for n in 0..64{let t=slots[n];let(ix,iy,iz)=(t[0],t[1],t[2]);rg[n]=[dq[ix]*bqy[iy]*bqz[iz],bq[ix]*dqy[iy]*bqz[iz],bq[ix]*bqy[iy]*dqz[iz]];}
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
    let (nodes, slots) = hex_q3_slots();
    let (b, d) = build_1d_basis(&[nodes[0], nodes[1], nodes[2], nodes[3]]);
    // Tensor grid → element-local slot (the sum-factorized contraction runs on
    // the tensor grid, so it needs the inverse of the slot map).
    let inv = tensor_slots(&slots, 4);

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < 64 { continue; }
        let mut xe_3d = [[[0.0_f64; 4]; 4]; 4]; // [ix][iy][iz]
        for iz in 0..4 { for iy in 0..4 { for ix in 0..4 {
            xe_3d[ix][iy][iz] = x[dofs[inv[ix][iy][iz]] as usize];
        }}}
        let mut ye_3d = [[[0.0_f64; 4]; 4]; 4];

        for qz in 0..4 { for qy in 0..4 { for qx in 0..4 {
            let off = (e*64 + qz*16 + qy*4 + qx) * 11;
            let (jit00,jit01,jit02) = (pd.data[off],pd.data[off+1],pd.data[off+2]);
            let (jit10,jit11,jit12) = (pd.data[off+3],pd.data[off+4],pd.data[off+5]);
            let (jit20,jit21,jit22) = (pd.data[off+6],pd.data[off+7],pd.data[off+8]);
            let sc = GL4_WTS[qx]*GL4_WTS[qy]*GL4_WTS[qz]*pd.data[off+9]*pd.data[off+10];

            let (bq,dq) = (b[qx],d[qx]); let (bqy,dqy) = (b[qy],d[qy]); let (bqz,dqz) = (b[qz],d[qz]);

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
            y[dofs[inv[ix][iy][iz]] as usize] += ye_3d[ix][iy][iz];
        }}}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::ReferenceElement;
    use fem_mesh::Mesh;

    /// D77 pin: the kernel's slot → tensor map is bit-identical to `HexQ3`'s
    /// (== `HexQk::new(3)`'s == `DofManager::build_pk_hex`'s) `dof_coords()`,
    /// and the 1-D nodes are its GLL(4) nodes — not the pre-D77 equispaced
    /// lexicographic table.
    #[test]
    fn hex_q3_pa_slots_match_element() {
        let (nodes, slots) = hex_q3_slots();
        let coords = HexQ3.dof_coords();
        assert_eq!(slots.len(), 64);
        for (slot, t) in slots.iter().enumerate() {
            for d in 0..3 {
                assert_eq!(
                    nodes[t[d]], coords[slot][d],
                    "slot {slot} axis {d}: PA kernel must reproduce HexQ3::dof_coords bit-exactly"
                );
            }
        }
        // Pre-D77 the kernel loaded `dofs[ix + iy*4 + iz*16]` (lexicographic
        // tensor order), a different permutation of the same 64 nodes.
        let lex: Vec<[usize; 3]> = (0..64).map(|n| [n % 4, (n / 4) % 4, n / 16]).collect();
        assert_ne!(slots, lex, "pre-D77 lexicographic tensor order");
    }

    #[test]
    fn hex_q3_pa_finite() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let pd = build_hex_q3_pa_data(&mesh, &|_|1.0);
        assert!(pd.data.iter().all(|v| v.is_finite()));
        assert!(pd.data.iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn hex_q3_1d_basis() {
        let (nodes, _) = hex_q3_slots();
        let nodes = [nodes[0], nodes[1], nodes[2], nodes[3]];
        let (b, d) = build_1d_basis(&nodes);
        for q in 0..4 { assert!((b[q].iter().sum::<f64>()-1.0).abs()<1e-14); }
        for q in 0..4 { assert!(d[q].iter().sum::<f64>().abs()<1e-14); }
        for i in 0..4 { for j in 0..4 {
            let t = nodes[j];
            let mut val=1.0; for m in 0..4{if m!=i{val*=(t-nodes[m])/(nodes[i]-nodes[m]);}}
            let exp = if i==j{1.0}else{0.0};
            assert!((val-exp).abs()<1e-14,"ℓ_{i}(x_{j})={val} exp={exp}");
        }}
    }

    #[test]
    fn hex_q3_sf_matches_naive() {
        let mesh = Mesh::<3>::unit_cube_hex(1); // 1 element, 8 vertices, 125 Q3 DOFs
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
