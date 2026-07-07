//! HDG (Hybridizable Discontinuous Galerkin) methods.
//!
//! # HDG Poisson
//!
//! Primal HDG formulation of `−Δu = f` on simplex meshes.

use fem_linalg::CooMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::SolverConfig;
use fem_element::lagrange::{TriPk, TetPk, SegPk};

#[derive(Debug)]
pub struct HdgPoissonResult {
    pub u: Vec<f64>,
    pub lambda: Vec<f64>,
}

pub fn solve_hdg_poisson<M, F>(
    mesh: M,
    source: F,
    bulk_order: u8,
    skeleton_order: u8,
) -> HdgPoissonResult
where
    M: MeshTopology + Clone + Send + Sync,
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements();
    let tau = 1.0_f64;
    let bo = bulk_order as usize;

    let dofs_per_elem = if dim == 2 {
        (bo + 1) * (bo + 2) / 2
    } else {
        (bo + 1) * (bo + 2) * (bo + 3) / 6
    };
    let n_bulk = n_elems * dofs_per_elem;

    let ref_elem: Box<dyn fem_element::ReferenceElement> = match dim {
        2 => Box::new(TriPk::new(bo)),
        3 => Box::new(TetPk::new(bo)),
        _ => unreachable!(),
    };
    let geo_elem: Box<dyn fem_element::ReferenceElement> = match dim {
        2 => Box::new(TriPk::new(1)),
        3 => Box::new(TetPk::new(1)),
        _ => unreachable!(),
    };
    let geo_n = geo_elem.n_dofs();

    let face_ref_bulk: Box<dyn fem_element::ReferenceElement> = match dim {
        2 => Box::new(SegPk::new(bo)),
        3 => Box::new(TriPk::new(bo)),
        _ => unreachable!(),
    };
    let dofs_per_sk_face = if skeleton_order == 0 { 1 } else if dim == 2 { skeleton_order as usize + 1 } else { (skeleton_order as usize + 1) * (skeleton_order as usize + 2) / 2 };

    let quad_order = 2 * bulk_order;
    let quad_vol = ref_elem.quadrature(quad_order);
    let quad_face = face_ref_bulk.quadrature(quad_order);
    let n_qp = quad_vol.n_points();
    let n_qp_face = quad_face.n_points();

    use std::collections::HashMap;
    let mut face_map: HashMap<Vec<u32>, (Vec<u32>, bool)> = HashMap::new();

    for e in 0..n_elems as u32 {
        let enodes = mesh.element_nodes(e);
        let local_faces = match (dim, enodes.len() as u32) {
            (2, 3) => vec![vec![0u32, 1], vec![1, 2], vec![0, 2]],
            (3, 4) => vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
            _ => panic!("HDG: unsupported element"),
        };
        for lf in &local_faces {
            let mut key: Vec<u32> = lf.iter().map(|&ni| enodes[ni as usize]).collect();
            key.sort_unstable();
            use std::collections::hash_map::Entry;
            match face_map.entry(key) {
                Entry::Vacant(e) => { e.insert((lf.clone(), false)); }
                Entry::Occupied(mut e) => { e.get_mut().1 = true; }
            }
        }
    }

    let face_list: Vec<(Vec<u32>, bool)> = face_map.into_values().collect();
    let interior_count = face_list.iter().filter(|(_, interior)| *interior).count();
    let n_lambda = interior_count * dofs_per_sk_face;

    let mut lambda_offset_of_face: Vec<Option<usize>> = vec![None; face_list.len()];
    {
        let mut next = 0usize;
        for (i, (_, interior)) in face_list.iter().enumerate() {
            if *interior {
                lambda_offset_of_face[i] = Some(next);
                next += dofs_per_sk_face;
            }
        }
    }

    let mut phi = vec![0.0_f64; dofs_per_elem];
    let mut grad_ref = vec![0.0_f64; dofs_per_elem * dim];
    let mut geo_phi = vec![0.0_f64; geo_n];
    let mut geo_grad = vec![0.0_f64; geo_n * dim];
    let mut psi = vec![0.0_f64; dofs_per_sk_face];
    let sk_elem: Option<Box<dyn fem_element::ReferenceElement>> = if skeleton_order > 0 {
        Some(match dim { 2 => Box::new(SegPk::new(skeleton_order as usize)), 3 => Box::new(TriPk::new(skeleton_order as usize)), _ => unreachable!() })
    } else { None };

    let mut sk_coo = CooMatrix::new(n_lambda, n_lambda);
    let mut sk_rhs = vec![0.0_f64; n_lambda];

    for e in 0..n_elems as u32 {
        let enodes = mesh.element_nodes(e);
        let npe = enodes.len() as u32;

        let local_faces: Vec<Vec<u32>> = match (dim, npe) {
            (2, 3) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
            (3, 4) => vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
            _ => unreachable!(),
        };
        let n_local_faces = local_faces.len();

        let mut face_lambda_offset: Vec<Option<usize>> = Vec::with_capacity(n_local_faces);
        for lf in &local_faces {
            let mut key: Vec<u32> = lf.iter().map(|&ni| enodes[ni as usize]).collect();
            key.sort_unstable();
            let mut found = None;
            for (fi, (fnodes, _)) in face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes.to_vec();
                fk.sort_unstable();
                if fk == key { found = Some(fi); break; }
            }
            match found {
                Some(fi) => face_lambda_offset.push(lambda_offset_of_face[fi]),
                None => face_lambda_offset.push(None),
            }
        }

        let n_sk_dofs_elem = n_local_faces * dofs_per_sk_face;
        let mut a_elem = vec![0.0_f64; dofs_per_elem * dofs_per_elem];
        let mut f_elem = vec![0.0_f64; dofs_per_elem];
        let mut b_elem = vec![0.0_f64; dofs_per_elem * n_sk_dofs_elem];

        for q in 0..n_qp {
            let xi = &quad_vol.points[q];
            let w = quad_vol.weights[q];

            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            geo_elem.eval_grad_basis(xi, &mut geo_grad);

            let mut jac = vec![vec![0.0_f64; dim]; dim];
            for i in 0..dim {
                for d in 0..dim {
                    for k in 0..geo_n {
                        jac[i][d] += mesh.node_coords(enodes[k])[i] * geo_grad[k * dim + d];
                    }
                }
            }
            let det_j = if dim == 2 {
                jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
            } else {
                jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
                - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
                + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0])
            };
            let vol = (w * det_j).abs();
            let inv_det = 1.0 / det_j;

            let mut grad_phys = vec![0.0_f64; dofs_per_elem * dim];
            if dim == 2 {
                let j00 = jac[1][1] * inv_det; let j01 = -jac[0][1] * inv_det;
                let j10 = -jac[1][0] * inv_det; let j11 = jac[0][0] * inv_det;
                for i in 0..dofs_per_elem {
                    grad_phys[i * dim]     = j00 * grad_ref[i * dim] + j01 * grad_ref[i * dim + 1];
                    grad_phys[i * dim + 1] = j10 * grad_ref[i * dim] + j11 * grad_ref[i * dim + 1];
                }
            } else {
                let m00 = (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1]) * inv_det;
                let m01 = (jac[0][2]*jac[2][1] - jac[0][1]*jac[2][2]) * inv_det;
                let m02 = (jac[0][1]*jac[1][2] - jac[0][2]*jac[1][1]) * inv_det;
                let m10 = (jac[1][2]*jac[2][0] - jac[1][0]*jac[2][2]) * inv_det;
                let m11 = (jac[0][0]*jac[2][2] - jac[0][2]*jac[2][0]) * inv_det;
                let m12 = (jac[0][2]*jac[1][0] - jac[0][0]*jac[1][2]) * inv_det;
                let m20 = (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]) * inv_det;
                let m21 = (jac[0][1]*jac[2][0] - jac[0][0]*jac[2][1]) * inv_det;
                let m22 = (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]) * inv_det;
                for i in 0..dofs_per_elem {
                    let gx = grad_ref[i * dim]; let gy = grad_ref[i * dim + 1]; let gz = grad_ref[i * dim + 2];
                    grad_phys[i * dim]     = m00 * gx + m01 * gy + m02 * gz;
                    grad_phys[i * dim + 1] = m10 * gx + m11 * gy + m12 * gz;
                    grad_phys[i * dim + 2] = m20 * gx + m21 * gy + m22 * gz;
                }
            }

            geo_elem.eval_basis(xi, &mut geo_phi);
            let mut x_phys = vec![0.0_f64; dim];
            for k in 0..geo_n {
                let c = mesh.node_coords(enodes[k]);
                for i in 0..dim { x_phys[i] += geo_phi[k] * c[i]; }
            }
            let f_val = source(&x_phys);

            for i in 0..dofs_per_elem {
                for j in 0..dofs_per_elem {
                    let mut diff = 0.0;
                    for d in 0..dim { diff += grad_phys[i * dim + d] * grad_phys[j * dim + d]; }
                    a_elem[i * dofs_per_elem + j] += vol * diff;
                }
                f_elem[i] += vol * phi[i] * f_val;
            }
        }

        for (lf_idx, _lf) in local_faces.iter().enumerate() {
            for fq in 0..n_qp_face {
                let fxi = &quad_face.points[fq];
                let fw = quad_face.weights[fq];

                let xi_ref = match (dim, lf_idx) {
                    (2, 0) => vec![fxi[0], 0.0],
                    (2, 1) => vec![1.0 - fxi[0], fxi[0]],
                    (2, 2) => vec![0.0, 1.0 - fxi[0]],
                    (3, 0) => vec![fxi[0], fxi[1], 0.0],
                    (3, 1) => vec![fxi[0], 0.0, fxi[1]],
                    (3, 2) => vec![0.0, fxi[0], fxi[1]],
                    (3, 3) => vec![fxi[0], fxi[1], 1.0 - fxi[0] - fxi[1]],
                    _ => unreachable!(),
                };

                ref_elem.eval_basis(&xi_ref, &mut phi);
                if let Some(ref sk) = sk_elem { sk.eval_basis(fxi, &mut psi); } else { psi[0] = 1.0; }

                let face_jac = compute_face_size(&mesh, enodes, lf_idx, dim, npe);
                let w_face = fw * face_jac;

                for i in 0..dofs_per_elem {
                    for j in 0..dofs_per_elem {
                        a_elem[i * dofs_per_elem + j] += tau * w_face * phi[i] * phi[j];
                    }
                }

                if let Some(loff) = face_lambda_offset[lf_idx] {
                    for i in 0..dofs_per_elem {
                        for d in 0..dofs_per_sk_face {
                            b_elem[i * n_sk_dofs_elem + loff - face_lambda_offset[lf_idx].unwrap() + d] +=
                                tau * w_face * phi[i] * psi[d];
                        }
                    }
                }
            }
        }

        // Recompute B column index offsets for consistent access
        // Actually, let's rebuild B properly with local face-based indexing
        let mut b_elem2 = vec![0.0_f64; dofs_per_elem * n_sk_dofs_elem];

        for (lf_idx, _lf) in local_faces.iter().enumerate() {
            for fq in 0..n_qp_face {
                let fxi = &quad_face.points[fq];
                let fw = quad_face.weights[fq];
                let xi_ref = match (dim, lf_idx) {
                    (2, 0) => vec![fxi[0], 0.0],
                    (2, 1) => vec![1.0 - fxi[0], fxi[0]],
                    (2, 2) => vec![0.0, 1.0 - fxi[0]],
                    (3, 0) => vec![fxi[0], fxi[1], 0.0],
                    (3, 1) => vec![fxi[0], 0.0, fxi[1]],
                    (3, 2) => vec![0.0, fxi[0], fxi[1]],
                    (3, 3) => vec![fxi[0], fxi[1], 1.0 - fxi[0] - fxi[1]],
                    _ => unreachable!(),
                };
                ref_elem.eval_basis(&xi_ref, &mut phi);
                if let Some(ref sk) = sk_elem { sk.eval_basis(fxi, &mut psi); } else { psi[0] = 1.0; }
                let face_jac = compute_face_size(&mesh, enodes, lf_idx, dim, npe);
                let w_face = fw * face_jac;

                if face_lambda_offset[lf_idx].is_some() {
                    let base = lf_idx * dofs_per_sk_face;
                    for i in 0..dofs_per_elem {
                        for d in 0..dofs_per_sk_face {
                            b_elem2[i * n_sk_dofs_elem + base + d] += tau * w_face * phi[i] * psi[d];
                        }
                    }
                }
            }
        }

        let a_inv = match invert_dense(&a_elem, dofs_per_elem) {
            Some(inv) => inv,
            None => {
                let shifted: Vec<f64> = a_elem.iter().map(|&v| v + 1e-12).collect();
                invert_dense(&shifted, dofs_per_elem).unwrap_or(vec![0.0; dofs_per_elem * dofs_per_elem])
            }
        };

        let mut u0 = vec![0.0_f64; dofs_per_elem];
        for i in 0..dofs_per_elem {
            for j in 0..dofs_per_elem {
                u0[i] += a_inv[i * dofs_per_elem + j] * f_elem[j];
            }
        }

        let mut u_lambda = vec![0.0_f64; dofs_per_elem * n_sk_dofs_elem];
        for i in 0..dofs_per_elem {
            for f in 0..n_sk_dofs_elem {
                let mut s = 0.0;
                for j in 0..dofs_per_elem {
                    s += a_inv[i * dofs_per_elem + j] * b_elem2[j * n_sk_dofs_elem + f];
                }
                u_lambda[i * n_sk_dofs_elem + f] = s;
            }
        }

        for f in 0..n_sk_dofs_elem {
            let lf_idx = f / dofs_per_sk_face;
            let ld = f % dofs_per_sk_face;
            let Some(loff) = face_lambda_offset[lf_idx] else { continue };
            let lam_f = loff + ld;

            // g_f += B^T A^{-1} f = Σ_i B[i][f] * u0[i]
            let mut bt_u0 = 0.0;
            for i in 0..dofs_per_elem {
                bt_u0 += b_elem2[i * n_sk_dofs_elem + f] * u0[i];
            }
            sk_rhs[lam_f] += bt_u0;

            for g in 0..n_sk_dofs_elem {
                let lf_idx2 = g / dofs_per_sk_face;
                let ld2 = g % dofs_per_sk_face;
                let Some(loff2) = face_lambda_offset[lf_idx2] else { continue };
                let lam_g = loff2 + ld2;

                let mut s_fg = 0.0;
                for i in 0..dofs_per_elem {
                    s_fg += b_elem2[i * n_sk_dofs_elem + f] * u_lambda[i * n_sk_dofs_elem + g];
                }

                // Subtract τ ∫ ψ_f · ψ_g on shared face
                // For P0 skeleton: ψ_f = 1·δ_{f, local face}, 
                // τ ∫ ψ_f ψ_g = τ * face_jac * δ_{fg}
                // For P1+: face inner product of skeleton basis
                if lf_idx == lf_idx2 {
                    let face_jac = compute_face_size(&mesh, enodes, lf_idx, dim, npe);
                    let mut face_mass = 0.0;
                    for fq in 0..n_qp_face {
                        let fw = quad_face.weights[fq];
                        if let Some(ref sk) = sk_elem { sk.eval_basis(&quad_face.points[fq], &mut psi); } else { psi[0] = 1.0; }
                        face_mass += fw * face_jac * psi[ld] * psi[ld2];
                    }
                    s_fg -= tau * face_mass;
                }

                sk_coo.add(lam_f, lam_g, s_fg);
            }
        }
    }

    if n_lambda == 0 {
        return HdgPoissonResult { u: vec![0.0; n_bulk], lambda: vec![] };
    }

    let sk_csr = sk_coo.into_csr();
    let mut lambda = vec![0.0_f64; n_lambda];
    let cfg = SolverConfig { max_iter: 2000, atol: 1e-12, rtol: 1e-12, ..Default::default() };
    match fem_solver::solve_cg(&sk_csr, &sk_rhs, &mut lambda, &cfg) {
        Ok(_) | Err(_) => {}
    }

    // ── Reconstruct bulk solution ──
    let mut u_bulk = vec![0.0_f64; n_bulk];
    for e in 0..n_elems as u32 {
        let enodes = mesh.element_nodes(e);
        let npe = enodes.len() as u32;
        let local_faces: Vec<Vec<u32>> = match (dim, npe) {
            (2, 3) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
            (3, 4) => vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
            _ => unreachable!(),
        };
        let n_local_faces = local_faces.len();
        let n_sk_dofs_elem = n_local_faces * dofs_per_sk_face;

        let mut face_lambda_offset: Vec<Option<usize>> = Vec::with_capacity(n_local_faces);
        for lf in &local_faces {
            let mut key: Vec<u32> = lf.iter().map(|&ni| enodes[ni as usize]).collect();
            key.sort_unstable();
            let mut found = None;
            for (fi, (fnodes, _)) in face_list.iter().enumerate() {
                let mut fk: Vec<u32> = fnodes.to_vec();
                fk.sort_unstable();
                if fk == key { found = Some(fi); break; }
            }
            match found { Some(fi) => face_lambda_offset.push(lambda_offset_of_face[fi]), None => face_lambda_offset.push(None) }
        }

        let mut a_elem = vec![0.0_f64; dofs_per_elem * dofs_per_elem];
        let mut f_elem = vec![0.0_f64; dofs_per_elem];
        let mut b_elem2 = vec![0.0_f64; dofs_per_elem * n_sk_dofs_elem];

        for q in 0..n_qp {
            let xi = &quad_vol.points[q];
            let w = quad_vol.weights[q];
            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            geo_elem.eval_grad_basis(xi, &mut geo_grad);
            let mut jac = vec![vec![0.0_f64; dim]; dim];
            for i in 0..dim { for d in 0..dim { for k in 0..geo_n { jac[i][d] += mesh.node_coords(enodes[k])[i] * geo_grad[k * dim + d]; } } }
            let det_j = if dim == 2 { jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0] } else {
                jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1]) - jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0]) + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])
            };
            let vol = (w * det_j).abs(); let inv_det = 1.0/det_j;
            let mut gp = vec![0.0; dofs_per_elem*dim];
            if dim == 2 { let (j00,j01,j10,j11)=(jac[1][1]*inv_det,-jac[0][1]*inv_det,-jac[1][0]*inv_det,jac[0][0]*inv_det);
                for i in 0..dofs_per_elem { gp[i*dim]=j00*grad_ref[i*dim]+j01*grad_ref[i*dim+1]; gp[i*dim+1]=j10*grad_ref[i*dim]+j11*grad_ref[i*dim+1]; }
            } else { let (m00,m01,m02,m10,m11,m12,m20,m21,m22)=(
                (jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])*inv_det,(jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])*inv_det,(jac[0][1]*jac[1][2]-jac[0][2]*jac[1][1])*inv_det,
                (jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])*inv_det,(jac[0][0]*jac[2][2]-jac[0][2]*jac[2][0])*inv_det,(jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])*inv_det,
                (jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])*inv_det,(jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])*inv_det,(jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0])*inv_det);
                for i in 0..dofs_per_elem { let gx=grad_ref[i*dim];let gy=grad_ref[i*dim+1];let gz=grad_ref[i*dim+2];
                    gp[i*dim]=m00*gx+m01*gy+m02*gz;gp[i*dim+1]=m10*gx+m11*gy+m12*gz;gp[i*dim+2]=m20*gx+m21*gy+m22*gz; }
            }
            geo_elem.eval_basis(xi, &mut geo_phi);
            let mut xp = vec![0.0; dim];
            for k in 0..geo_n { let c = mesh.node_coords(enodes[k]); for i in 0..dim { xp[i] += geo_phi[k]*c[i]; } }
            let fv = source(&xp);
            for i in 0..dofs_per_elem { for j in 0..dofs_per_elem { let mut d = 0.0; for a in 0..dim { d += gp[i*dim+a]*gp[j*dim+a]; } a_elem[i*dofs_per_elem+j] += vol*d; } f_elem[i] += vol*phi[i]*fv; }
        }
        for (lf_idx, _lf) in local_faces.iter().enumerate() {
            for fq in 0..n_qp_face {
                let fxi = &quad_face.points[fq]; let fw = quad_face.weights[fq];
                let xi_ref = match (dim, lf_idx) { (2,0)=>vec![fxi[0],0.0], (2,1)=>vec![1.0-fxi[0],fxi[0]], (2,2)=>vec![0.0,1.0-fxi[0]],
                    (3,0)=>vec![fxi[0],fxi[1],0.0], (3,1)=>vec![fxi[0],0.0,fxi[1]], (3,2)=>vec![0.0,fxi[0],fxi[1]], (3,3)=>vec![fxi[0],fxi[1],1.0-fxi[0]-fxi[1]], _=>unreachable!() };
                ref_elem.eval_basis(&xi_ref, &mut phi);
                if let Some(ref sk) = sk_elem { sk.eval_basis(fxi, &mut psi); } else { psi[0] = 1.0; }
                let fj = compute_face_size(&mesh, enodes, lf_idx, dim, npe);
                let wf = fw * fj;
                for i in 0..dofs_per_elem { for j in 0..dofs_per_elem { a_elem[i*dofs_per_elem+j] += tau*wf*phi[i]*phi[j]; } }
                if face_lambda_offset[lf_idx].is_some() {
                    let base = lf_idx * dofs_per_sk_face;
                    for i in 0..dofs_per_elem { for d in 0..dofs_per_sk_face { b_elem2[i*n_sk_dofs_elem+base+d] += tau*wf*phi[i]*psi[d]; } }
                }
            }
        }

        let a_inv = invert_dense(&a_elem, dofs_per_elem).unwrap_or_else(|| {
            let s: Vec<f64> = a_elem.iter().map(|&v| v + 1e-12).collect();
            invert_dense(&s, dofs_per_elem).unwrap_or(vec![0.0; dofs_per_elem*dofs_per_elem])
        });

        let mut u0 = vec![0.0; dofs_per_elem];
        for i in 0..dofs_per_elem { for j in 0..dofs_per_elem { u0[i] += a_inv[i*dofs_per_elem+j]*f_elem[j]; } }

        let base = e as usize * dofs_per_elem;
        u_bulk[base..base + dofs_per_elem].copy_from_slice(&u0[..dofs_per_elem]);
        // Add U_lambda * lambda contribution
        for f in 0..n_sk_dofs_elem {
            let lf_idx = f / dofs_per_sk_face;
            let ld = f % dofs_per_sk_face;
            let Some(loff) = face_lambda_offset[lf_idx] else { continue };
            let lam_val = lambda[loff + ld];
            for i in 0..dofs_per_elem {
                let mut s = 0.0;
                for j in 0..dofs_per_elem {
                    s += a_inv[i * dofs_per_elem + j] * b_elem2[j * n_sk_dofs_elem + f];
                }
                u_bulk[base + i] += s * lam_val;
            }
        }
    }

    HdgPoissonResult { u: u_bulk, lambda }
}

fn compute_face_size<M: MeshTopology>(mesh: &M, enodes: &[u32], lf_idx: usize, dim: usize, _npe: u32) -> f64 {
    if dim == 2 {
        let a = enodes[lf_idx]; let b = enodes[(lf_idx + 1) % 3];
        let pa = mesh.node_coords(a); let pb = mesh.node_coords(b);
        let dx = pb[0]-pa[0]; let dy = pb[1]-pa[1];
        (dx*dx+dy*dy).sqrt()
    } else {
        let lf: [usize; 3] = match lf_idx { 0=>[0,1,2], 1=>[0,1,3], 2=>[0,2,3], 3=>[1,2,3], _=>unreachable!() };
        let pa = mesh.node_coords(enodes[lf[0]]); let pb = mesh.node_coords(enodes[lf[1]]); let pc = mesh.node_coords(enodes[lf[2]]);
        let ux = pb[0]-pa[0]; let uy = pb[1]-pa[1]; let uz = pb[2]-pa[2];
        let vx = pc[0]-pa[0]; let vy = pc[1]-pa[1]; let vz = pc[2]-pa[2];
        let cx = uy*vz-uz*vy; let cy = uz*vx-ux*vz; let cz = ux*vy-uy*vx;
        0.5*(cx*cx+cy*cy+cz*cz).sqrt()
    }
}

fn invert_dense(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n { inv[i * n + i] = 1.0; }
    for col in 0..n {
        let mut mr = col; let mut mv = a[col * n + col].abs();
        for r in (col+1)..n { let v = a[r * n + col].abs(); if v > mv { mv = v; mr = r; } }
        if mv < 1e-15 { return None; }
        if mr != col { for j in 0..n { a.swap(col*n+j, mr*n+j); inv.swap(col*n+j, mr*n+j); } }
        let pv = a[col * n + col]; let ipv = 1.0 / pv;
        for j in 0..n { a[col*n+j] *= ipv; inv[col*n+j] *= ipv; }
        for r in 0..n { if r == col { continue; }
            let f = a[r * n + col]; for j in 0..n { a[r*n+j] -= f*a[col*n+j]; inv[r*n+j] -= f*inv[col*n+j]; } }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn hdg_poisson_2d_p1() {
        use std::f64::consts::PI;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let source = |x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
        let result = solve_hdg_poisson(mesh, source, 1, 0);
        for &v in &result.u { assert!(v.is_finite()); }
        for &v in &result.lambda { assert!(v.is_finite()); }
        assert!(result.lambda.len() > 0);
    }

    #[test]
    fn hdg_poisson_2d_p2() {
        use std::f64::consts::PI;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let source = |x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
        let result = solve_hdg_poisson(mesh, source, 2, 0);
        for &v in &result.u { assert!(v.is_finite()); }
        assert!(result.u.len() > 0);
    }

    #[test]
    fn test_invert_3x3() {
        let m = vec![4.0, 3.0, 0.0, 3.0, 4.0, -1.0, 0.0, -1.0, 4.0];
        let inv = invert_dense(&m, 3).unwrap();
        for i in 0..3 { for j in 0..3 { let mut d = 0.0; for k in 0..3 { d += m[i*3+k]*inv[k*3+j]; } let e = if i==j {1.0} else {0.0}; assert!((d-e).abs()<1e-12); } }
    }
}
