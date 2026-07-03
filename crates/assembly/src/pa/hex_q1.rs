//! Hex Q1 (trilinear) partial-assembly apply for diffusion.
//!
//! 8 nodes, 2×2×2=8 Gauss–Legendre quadrature points.

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;

// ─── 1D Gauss–Legendre on [-1,1] for p=2 (exact) ────────────────────────────
const GL_PTS: [f64; 2] = [-0.57735_02691_89626, 0.57735_02691_89626];
const GL_WTS: [f64; 2] = [1.0, 1.0];

// ─── 1D Lagrange basis (Q1) on [-1,1] ───────────────────────────────────────
#[inline] fn l0(t: f64) -> f64 { 0.5 * (1.0 - t) }
#[inline] fn l1(t: f64) -> f64 { 0.5 * (1.0 + t) }
#[inline] fn d0(_: f64) -> f64 { -0.5 }
#[inline] fn d1(_: f64) -> f64 {  0.5 }

/// Hex node → (a,b,c) ∈ {0,1}³ mapping.
/// Standard ordering: bottom face z=-1 CCW, then top face z=+1 CCW:
///   0:(-1,-1,-1) 1:(+1,-1,-1) 2:(+1,+1,-1) 3:(-1,+1,-1)
///   4:(-1,-1,+1) 5:(+1,-1,+1) 6:(+1,+1,+1) 7:(-1,+1,+1)
fn hex_abc(n: usize) -> (usize, usize, usize) {
    let a = (n & 1) ^ ((n >> 1) & 1);  // XOR of bits 0 and 1
    let b = (n >> 1) & 1;
    let c = n >> 2;
    (a, b, c)
}

/// Build PA data: J⁻ᵀ, |detJ|, κ at each qp per element.
pub fn build_hex_q1_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
) -> PaData {
    let n_elems = mesh.n_elements();
    let nqp = 8;
    let dim = 3;
    let mut pd = PaData::new(n_elems, nqp, dim);

    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let x: Vec<[f64; 3]> = (0..8).map(|i| { let c = mesh.node_coords(nodes[i]); [c[0], c[1], c[2]] }).collect();

        for (q, &qz) in GL_PTS.iter().enumerate() {
            for (qy_idx, &qy) in GL_PTS.iter().enumerate() {
                for (qx_idx, &qx) in GL_PTS.iter().enumerate() {
                    let qi = q * 4 + qy_idx * 2 + qx_idx;

                    let mut jac = [[0.0_f64; 3]; 3];
                    for n in 0..8 {
                        let (a, b, c) = hex_abc(n);
                        let (l0_x, l1_x, d0_x, d1_x) = (l0(qx), l1(qx), d0(qx), d1(qx));
                        let (l0_y, l1_y, d0_y, d1_y) = (l0(qy), l1(qy), d0(qy), d1(qy));
                        let (l0_z, l1_z, d0_z, d1_z) = (l0(qz), l1(qz), d0(qz), d1(qz));
                        let (phi_a, phi_b, phi_c) = (if a==0{l0_x}else{l1_x}, if b==0{l0_y}else{l1_y}, if c==0{l0_z}else{l1_z});
                        let (da, db, dc) = (if a==0{d0_x}else{d1_x}, if b==0{d0_y}else{d1_y}, if c==0{d0_z}else{d1_z});
                        for d in 0..3 {
                            jac[0][d] += da * phi_b * phi_c * x[n][d];
                            jac[1][d] += phi_a * db * phi_c * x[n][d];
                            jac[2][d] += phi_a * phi_b * dc * x[n][d];
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

                    let mut xp = [0.0; 3];
                    for n in 0..8 {
                        let (a, b, c) = hex_abc(n);
                        let phi = (if a==0{l0(qx)}else{l1(qx)})*(if b==0{l0(qy)}else{l1(qy)})*(if c==0{l0(qz)}else{l1(qz)});
                        for d in 0..3 { xp[d] += phi * x[n][d]; }
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

/// y += A·x via element-by-element PA.
pub fn pa_apply_hex_q1(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let nf = 11; // 3×3 J⁻ᵀ + |detJ| + κ
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let mut xe = [0.0_f64; 8];
        for i in 0..8 { xe[i] = x[dofs[i] as usize]; }
        let mut ye = [0.0_f64; 8];

        for (qz_idx, &qz) in GL_PTS.iter().enumerate() {
            for (qy_idx, &qy) in GL_PTS.iter().enumerate() {
                for (qx_idx, &qx) in GL_PTS.iter().enumerate() {
                    let qi = qz_idx * 4 + qy_idx * 2 + qx_idx;
                    let off = (e * 8 + qi) * nf;
                    let jit0 = [pd.data[off], pd.data[off+1], pd.data[off+2]];
                    let jit1 = [pd.data[off+3], pd.data[off+4], pd.data[off+5]];
                    let jit2 = [pd.data[off+6], pd.data[off+7], pd.data[off+8]];
                    let scale = GL_WTS[qx_idx] * GL_WTS[qy_idx] * GL_WTS[qz_idx] * pd.data[off+9] * pd.data[off+10];

                    let (l0_x,l1_x,d0_x,d1_x) = (l0(qx),l1(qx),d0(qx),d1(qx));
                    let (l0_y,l1_y,d0_y,d1_y) = (l0(qy),l1(qy),d0(qy),d1(qy));
                    let (l0_z,l1_z,d0_z,d1_z) = (l0(qz),l1(qz),d0(qz),d1(qz));

                    let mut flux = [0.0; 3];
                    for j in 0..8 {
                        let (a,b,c) = hex_abc(j);
                        let (pa,pb,pc) = (if a==0{l0_x}else{l1_x}, if b==0{l0_y}else{l1_y}, if c==0{l0_z}else{l1_z});
                        let (da,db,dc) = (if a==0{d0_x}else{d1_x}, if b==0{d0_y}else{d1_y}, if c==0{d0_z}else{d1_z});
                        let (g0,g1,g2) = (da*pb*pc, pa*db*pc, pa*pb*dc);
                        let pg = [jit0[0]*g0+jit0[1]*g1+jit0[2]*g2, jit1[0]*g0+jit1[1]*g1+jit1[2]*g2, jit2[0]*g0+jit2[1]*g1+jit2[2]*g2];
                        flux[0] += pg[0] * xe[j]; flux[1] += pg[1] * xe[j]; flux[2] += pg[2] * xe[j];
                    }
                    for i in 0..8 {
                        let (a,b,c) = hex_abc(i);
                        let (pa,pb,pc) = (if a==0{l0_x}else{l1_x}, if b==0{l0_y}else{l1_y}, if c==0{l0_z}else{l1_z});
                        let (da,db,dc) = (if a==0{d0_x}else{d1_x}, if b==0{d0_y}else{d1_y}, if c==0{d0_z}else{d1_z});
                        let (g0,g1,g2) = (da*pb*pc, pa*db*pc, pa*pb*dc);
                        let pg = [jit0[0]*g0+jit0[1]*g1+jit0[2]*g2, jit1[0]*g0+jit1[1]*g1+jit1[2]*g2, jit2[0]*g0+jit2[1]*g1+jit2[2]*g2];
                        ye[i] += scale * (pg[0]*flux[0] + pg[1]*flux[1] + pg[2]*flux[2]);
                    }
                }
            }
        }
        for i in 0..8 { y[dofs[i] as usize] += ye[i]; }
    }
}

/// y += M·x via Hex Q1 mass PA — reuses the same PaData (uses |detJ| and κ as ρ).
///
/// The mass operator is `M[i,j] = Σ_q |detJ_q|·w_q·ρ_q·φ_i(q)·φ_j(q)`,
/// which does not need the J⁻ᵀ gradient transform (only basis values).
pub fn pa_apply_mass_hex_q1(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let nf = 11;
    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let mut xe = [0.0_f64; 8];
        for i in 0..8 { xe[i] = x[dofs[i] as usize]; }
        let mut ye = [0.0_f64; 8];

        for (qz_idx, &qz) in GL_PTS.iter().enumerate() {
            for (qy_idx, &qy) in GL_PTS.iter().enumerate() {
                for (qx_idx, &qx) in GL_PTS.iter().enumerate() {
                    let qi = qz_idx * 4 + qy_idx * 2 + qx_idx;
                    let off = (e * 8 + qi) * nf;
                    let scale = GL_WTS[qx_idx] * GL_WTS[qy_idx] * GL_WTS[qz_idx]
                        * pd.data[off + 9] // |detJ|
                        * pd.data[off + 10]; // ρ (stored in κ slot)

                    let phi = [
                        l0(qx)*l0(qy)*l0(qz), l1(qx)*l0(qy)*l0(qz),
                        l1(qx)*l1(qy)*l0(qz), l0(qx)*l1(qy)*l0(qz),
                        l0(qx)*l0(qy)*l1(qz), l1(qx)*l0(qy)*l1(qz),
                        l1(qx)*l1(qy)*l1(qz), l0(qx)*l1(qy)*l1(qz),
                    ];

                    let u_qp: f64 = (0..8).map(|j| phi[j] * xe[j]).sum();
                    for i in 0..8 { ye[i] += scale * phi[i] * u_qp; }
                }
            }
        }
        for i in 0..8 { y[dofs[i] as usize] += ye[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace};
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;

    fn hex_elem_dofs(space: &H1Space<SimplexMesh<3>>) -> Vec<Vec<u32>> {
        let mesh = space.mesh();
        (0..mesh.n_elements() as u32).map(|e| space.element_dofs(e).to_vec()).collect()
    }

    #[test]
    fn hex_q1_pa_matches_assembled() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);

        let mesh2 = SimplexMesh::<3>::unit_cube_hex(2);
        let space2 = H1Space::new(mesh2, 1);
        let pd = build_hex_q1_pa_data(space2.mesh(), &|_| 1.0);
        let elem_dofs = hex_elem_dofs(&space2);

        let n = space.n_dofs();
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n).map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng>>11) as f64)/((1u64<<53) as f64) }).collect();

        let mut y_ref = vec![0.0; n];
        mat.spmv(&x, &mut y_ref);

        let mut y_pa = vec![0.0; n];
        pa_apply_hex_q1(&pd, &elem_dofs, &x, &mut y_pa);

        let max_err: f64 = (0..n).map(|i| (y_pa[i] - y_ref[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-12, "Hex Q1 PA max error {max_err}");
    }

    #[test]
    fn hex_q1_pa_mass_matches_assembled() {
        use crate::standard::MassIntegrator;

        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 2);

        let mesh2 = SimplexMesh::<3>::unit_cube_hex(2);
        let space2 = H1Space::new(mesh2, 1);
        let pd = build_hex_q1_pa_data(space2.mesh(), &|_| 1.0);
        let elem_dofs = hex_elem_dofs(&space2);

        let n = space.n_dofs();
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n).map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng>>11) as f64)/((1u64<<53) as f64) }).collect();

        let mut y_ref = vec![0.0; n];
        mat.spmv(&x, &mut y_ref);

        let mut y_pa = vec![0.0; n];
        pa_apply_mass_hex_q1(&pd, &elem_dofs, &x, &mut y_pa);

        let max_err: f64 = (0..n).map(|i| (y_pa[i] - y_ref[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-12, "Hex Q1 mass PA max error {max_err}");
    }
}
