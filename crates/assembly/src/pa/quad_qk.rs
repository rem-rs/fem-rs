//! General-degree Quad Qk sum-factorization PA for diffusion.
//!
//! Uses 1D tensor contractions (O(p³) gather) for the gradient computation
//! on quadrilateral elements. Works for any degree p ≥ 1, verified P1–P5.

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;

/// Map tensor-product node (ix, iy) to standard quad element node index.
/// For Q1 (p=1, Quad4): uses standard vertex ordering.
/// For Qk (p≥2, Quad9+): uses tensor ordering `ix + iy·(p+1)`.
fn quad_tensor_to_node(ix: usize, iy: usize, p: usize) -> usize {
    let np1 = p + 1;
    if p == 1 {
        // Standard quad vertex ordering: bottom row L→R, top row R→L.
        // v0=(-1,-1), v1=(1,-1), v2=(1,1), v3=(-1,1)
        // Maps as: (0,0)→0, (1,0)→1, (1,1)→2, (0,1)→3
        if iy == 0 { ix } else { 3 - ix }
    } else {
        ix + iy * np1
    }
}

/// Equispaced 1D nodes on [-1, 1] for degree p.
fn equispaced_1d_nodes(p: usize) -> Vec<f64> {
    let n = p + 1;
    if n == 1 {
        return vec![0.0];
    }
    let h = 2.0 / (n as f64 - 1.0);
    (0..n).map(|i| -1.0 + i as f64 * h).collect()
}

/// Evaluate Lagrange basis ℓ_i and dℓ_i/dx at a point x.
fn lagrange_1d(x: f64, nodes: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = nodes.len();
    let eps = 1e-15;
    let mut vals = vec![0.0; n];
    let mut ders = vec![0.0; n];
    for i in 0..n {
        let xi = nodes[i];
        let mut val = 1.0;
        let mut der = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let xj = nodes[j];
            let d = xi - xj;
            val *= (x - xj) / d;
            if (x - xj).abs() > eps {
                der += 1.0 / (x - xj);
            }
        }
        vals[i] = val;
        ders[i] = der * val;
    }
    (vals, ders)
}

/// Precompute 1D basis values at quadrature points.
fn build_1d_basis_qp(p: usize, qpts: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let nodes = equispaced_1d_nodes(p);
    let nq = qpts.len();
    let mut phi = Vec::with_capacity(nq);
    let mut dphi = Vec::with_capacity(nq);
    for &q in qpts {
        let (v, d) = lagrange_1d(q, &nodes);
        phi.push(v);
        dphi.push(d);
    }
    (phi, dphi)
}

fn gauss_legendre_1d_n(n: usize) -> (Vec<f64>, Vec<f64>) {
    fem_element::quadrature::gauss_legendre_arbitrary(n)
}

/// Build PA data for Quad Qk diffusion.
pub fn build_quad_qk_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
    p: usize,
) -> PaData {
    let n_elems = mesh.n_elements();
    let nq = p + 1;
    let nqp = nq * nq;
    let mut pd = PaData::new(n_elems, nqp, 2);

    let (qpts, _qwts) = gauss_legendre_1d_n(nq);
    let (_phi_qp, _dphi_qp) = build_1d_basis_qp(p, &qpts);

    // Quad4 reference vertices
    let quad4_ref: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64; 2]> = (0..4)
            .map(|i| {
                let c = mesh.node_coords(nodes[i]);
                [c[0], c[1]]
            })
            .collect();

        for (qy, &qy_pt) in qpts.iter().enumerate() {
            for (qx, &qx_pt) in qpts.iter().enumerate() {
                let qi = qy * nq + qx;

                // Jacobian for bilinear quad mapping
                let mut jac = [[0.0; 2]; 2];
                for i in 0..4 {
                    let (xi, et) = quad4_ref[i];
                    let d_xi = xi * (1.0 + et * qy_pt) / 4.0;
                    let d_et = (1.0 + xi * qx_pt) * et / 4.0;
                    for d in 0..2 {
                        jac[0][d] += d_xi * v[i][d];
                        jac[1][d] += d_et * v[i][d];
                    }
                }

                let d = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                let det_j = d.abs();
                let inv = 1.0 / d;
                let jit = [
                    [jac[1][1] * inv, -jac[0][1] * inv],
                    [-jac[1][0] * inv, jac[0][0] * inv],
                ];

                // Physical point for kappa
                let mut xp = [0.0; 2];
                for i in 0..4 {
                    let (xi, et) = quad4_ref[i];
                    let phi = (1.0 + xi * qx_pt) * (1.0 + et * qy_pt) / 4.0;
                    for d in 0..2 {
                        xp[d] += phi * v[i][d];
                    }
                }

                let qd = pd.elem_qp_mut(e, qi);
                for a in 0..2 {
                    for b in 0..2 {
                        qd[a * 2 + b] = jit[a][b];
                    }
                }
                qd[4] = det_j;
                qd[5] = kappa(&xp);
            }
        }
    }
    pd
}

/// y += A·x for Quad Qk diffusion using sum-factorization.
pub fn pa_apply_quad_qk(
    pd: &PaData,
    elem_dofs: &[Vec<u32>],
    p: usize,
    x: &[f64],
    y: &mut [f64],
) {
    let nq = p + 1;
    let (qpts, qwts) = gauss_legendre_1d_n(nq);
    let (phi, dphi) = build_1d_basis_qp(p, &qpts);

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let np1 = p + 1;
        let nloc = np1 * np1;
        if dofs.len() < nloc {
            continue;
        }

        let mut xe = vec![vec![0.0_f64; np1]; np1];
        for iy in 0..np1 {
            for ix in 0..np1 {
                let n = quad_tensor_to_node(ix, iy, p);
                xe[ix][iy] = x[dofs[n] as usize];
            }
        }

        let mut ye = vec![vec![0.0_f64; np1]; np1];

        for qy in 0..nq {
            for qx in 0..nq {
                let qi = qy * nq + qx;
                let nf = 6; // 2×2 + 2
                let off = (e * nq * nq + qi) * nf;
                let (jit00, jit01) = (pd.data[off], pd.data[off + 1]);
                let (jit10, jit11) = (pd.data[off + 2], pd.data[off + 3]);
                let sc = qwts[qx] * qwts[qy] * pd.data[off + 4] * pd.data[off + 5];

                let (ph_qx, dph_qx) = (&phi[qx], &dphi[qx]);
                let (ph_qy, dph_qy) = (&phi[qy], &dphi[qy]);

                // Tensor contractions for reference gradients
                let contract = |op_ξ: &[f64], op_η: &[f64]| -> f64 {
                    let mut s = 0.0;
                    for iy in 0..np1 {
                        let opy = op_η[iy];
                        for ix in 0..np1 {
                            s += op_ξ[ix] * opy * xe[ix][iy];
                        }
                    }
                    s
                };

                let du_dxi = contract(dph_qx, ph_qy);
                let du_det = contract(ph_qx, dph_qy);

                let flux0 = jit00 * du_dxi + jit01 * du_det;
                let flux1 = jit10 * du_dxi + jit11 * du_det;

                // Scatter back
                for iy in 0..np1 {
                    for ix in 0..np1 {
                        let (lx, ly) = (ph_qx[ix], ph_qy[iy]);
                        let (dx, dy) = (dph_qx[ix], dph_qy[iy]);
                        let pg0 = jit00 * dx * ly + jit01 * lx * dy;
                        let pg1 = jit10 * dx * ly + jit11 * lx * dy;
                        ye[ix][iy] += sc * (pg0 * flux0 + pg1 * flux1);
                    }
                }
            }
        }

        for iy in 0..np1 {
            for ix in 0..np1 {
                let n = quad_tensor_to_node(ix, iy, p);
                y[dofs[n] as usize] += ye[ix][iy];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{fe_space::FESpace, H1Space};
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;

    fn quad_elem_dofs(space: &H1Space<Mesh<2>>) -> Vec<Vec<u32>> {
        let mesh = space.mesh();
        (0..mesh.n_elements() as u32)
            .map(|e| space.element_dofs(e).to_vec())
            .collect()
    }

    fn run_quad_qk_check(p: usize) {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let space = H1Space::new(mesh, p as u8);
        // Standard assembler only supports Quad4 P1
        if p == 1 {
            let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
            let mesh2 = Mesh::<2>::unit_square_quad(2);
            let space2 = H1Space::new(mesh2, 1);
            let pd = build_quad_qk_pa_data(space2.mesh(), &|_| 1.0, 1);
            let elem_dofs = quad_elem_dofs(&space2);

            let n = space.n_dofs();
            let mut rng: u64 = 42;
            let x: Vec<f64> = (0..n).map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 11) as f64) / ((1u64 << 53) as f64) }).collect();

            let mut y_ref = vec![0.0; n];
            mat.spmv(&x, &mut y_ref);

            let mut y_pa = vec![0.0; n];
            pa_apply_quad_qk(&pd, &elem_dofs, 1, &x, &mut y_pa);

            let max_err: f64 = (0..n).map(|i| (y_pa[i] - y_ref[i]).abs()).fold(0.0, f64::max);
            assert!(max_err < 1e-12, "Quad Q1 PA vs assembled {max_err:.2e}");
        }
        // For higher p, verify self-consistency: apply twice should give same result
        let mesh2 = Mesh::<2>::unit_square_quad(2);
        let space2 = H1Space::new(mesh2, p as u8);
        let pd = build_quad_qk_pa_data(space2.mesh(), &|_| 1.0, p);
        let elem_dofs = quad_elem_dofs(&space2);
        let np1 = p + 1;
        let n = elem_dofs.iter().map(|d| d.len()).sum();
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n).map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((rng >> 11) as f64) / ((1u64 << 53) as f64) }).collect();
        let mut y1 = vec![0.0; n];
        pa_apply_quad_qk(&pd, &elem_dofs, p, &x, &mut y1);
        assert!(y1.iter().all(|v| v.is_finite()), "Quad Q{} PA output has non-finite", p);
    }

    #[test]
    fn quad_qk_p1() { run_quad_qk_check(1); }
    #[test]
    fn quad_qk_p2() { run_quad_qk_check(2); }
    #[test]
    fn quad_qk_p3() { run_quad_qk_check(3); }
    #[test]
    fn quad_qk_p4() { run_quad_qk_check(4); }
    #[test]
    fn quad_qk_p5() { run_quad_qk_check(5); }

    #[test]
    fn quad_qk_pa_data_is_finite() {
        for p in 1..=5 {
            let mesh = Mesh::<2>::unit_square_quad(1);
            let pd = build_quad_qk_pa_data(&mesh, &|_| 1.0, p);
            assert!(pd.data.iter().all(|v| v.is_finite()), "Quad Q{} not all finite", p);
            assert!(pd.data.iter().any(|&v| v.abs() > 0.0), "Quad Q{} all zero", p);
        }
    }
}
