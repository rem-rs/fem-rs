//! Quad Q1 (bilinear) partial-assembly apply for diffusion.
//! 4 nodes, 2×2=4 Gauss–Legendre points.

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;

const GL_PTS: [f64; 2] = [-0.57735_02691_89626, 0.57735_02691_89626];
const GL_WTS: [f64; 2] = [1.0, 1.0];

#[inline] fn l0(t: f64) -> f64 { 0.5 * (1.0 - t) }
#[inline] fn l1(t: f64) -> f64 { 0.5 * (1.0 + t) }
#[inline] fn d0(_: f64) -> f64 { -0.5 }
#[inline] fn d1(_: f64) -> f64 {  0.5 }

/// Quad node ordering (CCW): 0:(-1,-1) 1:(+1,-1) 2:(+1,+1) 3:(-1,+1)
fn quad_ab(n: usize) -> (usize, usize) {
    let a = (n & 1) ^ ((n >> 1) & 1);
    let b = n >> 1;
    (a, b)
}

pub fn build_quad_q1_pa_data<M: MeshTopology>(mesh: &M, kappa: &dyn Fn(&[f64]) -> f64) -> PaData {
    let n_elems = mesh.n_elements();
    let mut pd = PaData::new(n_elems, 4, 2);
    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let x: Vec<[f64; 2]> = (0..4).map(|i| { let c = mesh.node_coords(nodes[i]); [c[0], c[1]] }).collect();
        for (qy, &qy_pt) in GL_PTS.iter().enumerate() {
            for (qx, &qx_pt) in GL_PTS.iter().enumerate() {
                let qi = qy * 2 + qx;
                let mut jac = [[0.0; 2]; 2];
                for n in 0..4 {
                    let (a,b) = quad_ab(n);
                    let (pa,pb) = (if a==0{l0(qx_pt)}else{l1(qx_pt)}, if b==0{l0(qy_pt)}else{l1(qy_pt)});
                    let (da,db) = (if a==0{d0(qx_pt)}else{d1(qx_pt)}, if b==0{d0(qy_pt)}else{d1(qy_pt)});
                    jac[0][0] += da*pb*x[n][0]; jac[0][1] += da*pb*x[n][1];
                    jac[1][0] += pa*db*x[n][0]; jac[1][1] += pa*db*x[n][1];
                }
                let det_j = (jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0]).abs();
                let inv = 1.0/(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0]);
                let jit = [jac[1][1]*inv, -jac[0][1]*inv, -jac[1][0]*inv, jac[0][0]*inv];
                let mut xp = [0.0;2];
                for n in 0..4 {
                    let (a,b)=quad_ab(n); let phi=(if a==0{l0(qx_pt)}else{l1(qx_pt)})*(if b==0{l0(qy_pt)}else{l1(qy_pt)});
                    for d in 0..2{xp[d]+=phi*x[n][d];}
                }
                let qd = pd.elem_qp_mut(e, qi);
                qd[0..4].copy_from_slice(&jit); qd[4]=det_j; qd[5]=kappa(&xp);
            }
        }
    }
    pd
}

pub fn pa_apply_quad_q1(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let mut xe = [0.0_f64; 4]; for i in 0..4 { xe[i] = x[dofs[i] as usize]; }
        let mut ye = [0.0_f64; 4];

        for qy in 0..2 { for qx in 0..2 {
            let off = (e*4+qy*2+qx)*6;
            let (jit00,jit01,jit10,jit11) = (pd.data[off],pd.data[off+1],pd.data[off+2],pd.data[off+3]);
            let scale = GL_WTS[qx]*GL_WTS[qy]*pd.data[off+4]*pd.data[off+5];

            let (l0x,l1x,d0x,d1x)=(l0(GL_PTS[qx]),l1(GL_PTS[qx]),d0(GL_PTS[qx]),d1(GL_PTS[qx]));
            let (l0y,l1y,d0y,d1y)=(l0(GL_PTS[qy]),l1(GL_PTS[qy]),d0(GL_PTS[qy]),d1(GL_PTS[qy]));

            let mut flux = [0.0;2];
            for j in 0..4 { let (a,b)=quad_ab(j);
                let (pa,pb)=(if a==0{l0x}else{l1x},if b==0{l0y}else{l1y});
                let (da,db)=(if a==0{d0x}else{d1x},if b==0{d0y}else{d1y});
                let pg=[jit00*da*pb+jit01*pa*db, jit10*da*pb+jit11*pa*db];
                flux[0] += pg[0]*xe[j]; flux[1] += pg[1]*xe[j];
            }
            for i in 0..4 { let (a,b)=quad_ab(i);
                let (pa,pb)=(if a==0{l0x}else{l1x},if b==0{l0y}else{l1y});
                let (da,db)=(if a==0{d0x}else{d1x},if b==0{d0y}else{d1y});
                let pg=[jit00*da*pb+jit01*pa*db, jit10*da*pb+jit11*pa*db];
                ye[i] += scale*(pg[0]*flux[0]+pg[1]*flux[1]);
            }
        }}
        for i in 0..4 { y[dofs[i] as usize] += ye[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace};
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;

    fn quad_elem_dofs(space: &H1Space<SimplexMesh<2>>) -> Vec<Vec<u32>> {
        let mesh = space.mesh();
        (0..mesh.n_elements() as u32).map(|e| space.element_dofs(e).to_vec()).collect()
    }

    #[test]
    fn quad_q1_pa_matches_assembled() {
        let mesh = SimplexMesh::<2>::unit_square_quad(4);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);

        let mesh2 = SimplexMesh::<2>::unit_square_quad(4);
        let space2 = H1Space::new(mesh2, 1);
        let pd = build_quad_q1_pa_data(space2.mesh(), &|_| 1.0);
        let elem_dofs = quad_elem_dofs(&space2);

        let n = space.n_dofs();
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n).map(|_| { rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng>>11)as f64)/((1u64<<53)as f64) }).collect();

        let mut y_ref = vec![0.0; n]; mat.spmv(&x, &mut y_ref);
        let mut y_pa = vec![0.0; n]; pa_apply_quad_q1(&pd, &elem_dofs, &x, &mut y_pa);

        let max_err: f64 = (0..n).map(|i| (y_pa[i]-y_ref[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-12, "Quad Q1 PA max error {max_err}");
    }
}
