//! Curved-mesh DG assembly (fully self-contained).
//!
//! Both volume and face assembly are implemented here for 3-D tet meshes
//! using isoparametric `CurvedElementTransformation`.

use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{
    curved::{CurvedElementTransformation, CurvedMesh},
    element_type::ElementType,
    topology::MeshTopology,
    SimplexMesh,
};
use fem_space::fe_space::FESpace;

use crate::interior_faces::InteriorFaceList;

fn ref_elem(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3};
    match (et, order) {
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4 | ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4 | ElementType::Tet10, 3) => Box::new(TetP3),
        _ => panic!("unsupported ({et:?}, order={order})"),
    }
}

/// 3-D face geometry: area and unit normal (outward from a→b→c).
fn face_geom_3d(coords: &[f64], a: u32, b: u32, c: u32) -> (f64, [f64; 3]) {
    let ca = [coords[a as usize * 3], coords[a as usize * 3 + 1], coords[a as usize * 3 + 2]];
    let cb = [coords[b as usize * 3], coords[b as usize * 3 + 1], coords[b as usize * 3 + 2]];
    let cc = [coords[c as usize * 3], coords[c as usize * 3 + 1], coords[c as usize * 3 + 2]];
    let e1 = [cb[0]-ca[0], cb[1]-ca[1], cb[2]-ca[2]];
    let e2 = [cc[0]-ca[0], cc[1]-ca[1], cc[2]-ca[2]];
    let nx = e1[1]*e2[2] - e1[2]*e2[1];
    let ny = e1[2]*e2[0] - e1[0]*e2[2];
    let nz = e1[0]*e2[1] - e1[1]*e2[0];
    let area = 0.5 * (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
    let len = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
    (area, [nx/len, ny/len, nz/len])
}

/// Assemble SIP-DG matrix with curved volume + affine face (3-D).
///
/// `mesh` is a linear mesh matching `curved`'s topology.
pub fn assemble_sip_curved_3d<S: FESpace + Sync>(
    space: &S,
    mesh: &SimplexMesh<3>,
    curved: &CurvedMesh<3>,
    kappa: f64,
    sigma: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let dim = 3usize;
    let n_dofs = space.n_dofs();
    let order = space.order();
    let et = ElementType::Tet4;
    let re = ref_elem(et, order);
    let n = re.n_dofs();
    let q = re.quadrature(quad_order);
    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    // ── Volume: curved ────────────────────────────────────────────────────
    {
        let mut gref = vec![0.0; n * dim];
        let mut gp = vec![0.0; n * dim];
        for e in 0..mesh.n_elements() as u32 {
            let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let trans = CurvedElementTransformation::new(curved, e as usize);
            let mut ke = vec![0.0; n * n];
            for (qi, xi) in q.points.iter().enumerate() {
                let det_j = trans.det_j(xi);
                let w = q.weights[qi] * det_j.abs();
                let jit = trans.jacobian_inv_t(xi);
                re.eval_grad_basis(xi, &mut gref);
                for i in 0..n {
                    for d in 0..dim {
                        gp[i * dim + d] = (0..dim).map(|k| jit[d*dim+k] * gref[i*dim+k]).sum();
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        ke[i*n+j] += w * kappa * (0..dim).map(|d| gp[i*dim+d]*gp[j*dim+d]).sum::<f64>();
                    }
                }
            }
            for (i, &gi) in dofs.iter().enumerate() {
                for (j, &gj) in dofs.iter().enumerate() {
                    coo.add(gi, gj, ke[i*n+j]);
                }
            }
        }
    }

    // ── Interior faces: affine 3-D SIP ──────────────────────────────────
    let ifl = InteriorFaceList::build(mesh);
    // Tet4 local face tables: 4 faces with 3 nodes each
    let tet_faces: [(usize, usize, usize); 4] = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)];

    let q_face = ref_elem(ElementType::Tri3, order).quadrature(2);
    let _nf = q_face.points.len();
    let _phi_f = [0.0; 3]; // Tri3: 3 basis fns
    let mut phi_l = vec![0.0; n];
    let mut phi_r = vec![0.0; n];
    let mut gref_l = vec![0.0; n * dim];
    let mut gref_r = vec![0.0; n * dim];
    let mut gp_l = vec![0.0; n * dim];
    let mut gp_r = vec![0.0; n * dim];

    // Build face→element map: for each boundary face, record which element owns it.
    let mut face_owner: std::collections::HashMap<(u32,u32,u32), u32> = std::collections::HashMap::new();
    for e in 0..mesh.n_elements() as u32 {
        let ns = mesh.element_nodes(e);
        for &(ai, bi, ci) in &tet_faces {
            let mut v = [ns[ai], ns[bi], ns[ci]];
            v.sort_unstable();
            face_owner.entry((v[0], v[1], v[2])).or_insert(e);
        }
    }

    for face in &ifl.faces {
        let e_l = face.elem_left as usize;
        let e_r = face.elem_right as usize;
        let fnodes0 = face.face_nodes[0];
        let fnodes1 = face.face_nodes[1];
        let fnodes2 = face.face_nodes[2];
        let (face_area, n_l) = face_geom_3d(&mesh.coords, fnodes0, fnodes1, fnodes2);

        // Orient normal outward from left element
        let ns_l = mesh.element_nodes(e_l as u32);
        let mut centroid = [0.0; 3];
        for &n in ns_l.iter() {
            for d in 0..3 { centroid[d] += mesh.coords[n as usize * 3 + d]; }
        }
        for d in 0..3 { centroid[d] /= ns_l.len() as f64; }
        let mut midpoint = [0.0; 3];
        for &n in &[fnodes0, fnodes1, fnodes2] {
            for d in 0..3 { midpoint[d] += mesh.coords[n as usize * 3 + d]; }
        }
        for d in 0..3 { midpoint[d] /= 3.0; }
        let dot = n_l[0]*(midpoint[0]-centroid[0]) + n_l[1]*(midpoint[1]-centroid[1]) + n_l[2]*(midpoint[2]-centroid[2]);
        let norm_l = if dot < 0.0 { [-n_l[0], -n_l[1], -n_l[2]] } else { n_l };

        let dofs_l: Vec<usize> = space.element_dofs(e_l as u32).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = space.element_dofs(e_r as u32).iter().map(|&d| d as usize).collect();
        let nodes_l = mesh.element_nodes(e_l as u32);
        let nodes_r = mesh.element_nodes(e_r as u32);

        // Affine Jacobian for each element
        let x0_l = &mesh.coords[nodes_l[0] as usize * 3..];
        let mut jac_l = [[0.0; 3]; 3];
        for col in 0..3 {
            let xn = &mesh.coords[nodes_l[col+1] as usize * 3..];
            for row in 0..3 { jac_l[row][col] = xn[row] - x0_l[row]; }
        }
        let det_l = jac_l[0][0]*(jac_l[1][1]*jac_l[2][2]-jac_l[1][2]*jac_l[2][1])
                  - jac_l[0][1]*(jac_l[1][0]*jac_l[2][2]-jac_l[1][2]*jac_l[2][0])
                  + jac_l[0][2]*(jac_l[1][0]*jac_l[2][1]-jac_l[1][1]*jac_l[2][0]);
        let idet_l = if det_l.abs() > 1e-30 { 1.0/det_l } else { 0.0 };
        let mut jit_l = [[0.0; 3]; 3];
        jit_l[0][0] = (jac_l[1][1]*jac_l[2][2]-jac_l[1][2]*jac_l[2][1])*idet_l;
        jit_l[0][1] = (jac_l[0][2]*jac_l[2][1]-jac_l[0][1]*jac_l[2][2])*idet_l;
        jit_l[0][2] = (jac_l[0][1]*jac_l[1][2]-jac_l[0][2]*jac_l[1][1])*idet_l;
        jit_l[1][0] = (jac_l[1][2]*jac_l[2][0]-jac_l[1][0]*jac_l[2][2])*idet_l;
        jit_l[1][1] = (jac_l[0][0]*jac_l[2][2]-jac_l[0][2]*jac_l[2][0])*idet_l;
        jit_l[1][2] = (jac_l[0][2]*jac_l[1][0]-jac_l[0][0]*jac_l[1][2])*idet_l;
        jit_l[2][0] = (jac_l[1][0]*jac_l[2][1]-jac_l[1][1]*jac_l[2][0])*idet_l;
        jit_l[2][1] = (jac_l[0][1]*jac_l[2][0]-jac_l[0][0]*jac_l[2][1])*idet_l;
        jit_l[2][2] = (jac_l[0][0]*jac_l[1][1]-jac_l[0][1]*jac_l[1][0])*idet_l;

        let x0_r = &mesh.coords[nodes_r[0] as usize * 3..];
        let mut jac_r = [[0.0; 3]; 3];
        for col in 0..3 {
            let xn = &mesh.coords[nodes_r[col+1] as usize * 3..];
            for row in 0..3 { jac_r[row][col] = xn[row] - x0_r[row]; }
        }
        let det_r = jac_r[0][0]*(jac_r[1][1]*jac_r[2][2]-jac_r[1][2]*jac_r[2][1])
                  - jac_r[0][1]*(jac_r[1][0]*jac_r[2][2]-jac_r[1][2]*jac_r[2][0])
                  + jac_r[0][2]*(jac_r[1][0]*jac_r[2][1]-jac_r[1][1]*jac_r[2][0]);
        let idet_r = if det_r.abs() > 1e-30 { 1.0/det_r } else { 0.0 };
        let mut jit_r = [[0.0; 3]; 3];
        jit_r[0][0] = (jac_r[1][1]*jac_r[2][2]-jac_r[1][2]*jac_r[2][1])*idet_r;
        jit_r[0][1] = (jac_r[0][2]*jac_r[2][1]-jac_r[0][1]*jac_r[2][2])*idet_r;
        jit_r[0][2] = (jac_r[0][1]*jac_r[1][2]-jac_r[0][2]*jac_r[1][1])*idet_r;
        jit_r[1][0] = (jac_r[1][2]*jac_r[2][0]-jac_r[1][0]*jac_r[2][2])*idet_r;
        jit_r[1][1] = (jac_r[0][0]*jac_r[2][2]-jac_r[0][2]*jac_r[2][0])*idet_r;
        jit_r[1][2] = (jac_r[0][2]*jac_r[1][0]-jac_r[0][0]*jac_r[1][2])*idet_r;
        jit_r[2][0] = (jac_r[1][0]*jac_r[2][1]-jac_r[1][1]*jac_r[2][0])*idet_r;
        jit_r[2][1] = (jac_r[0][1]*jac_r[2][0]-jac_r[0][0]*jac_r[2][1])*idet_r;
        jit_r[2][2] = (jac_r[0][0]*jac_r[1][1]-jac_r[0][1]*jac_r[1][0])*idet_r;

        let mut kll = vec![0.0; n * n];
        let mut klr = vec![0.0; n * n];
        let mut krl = vec![0.0; n * n];
        let mut krr = vec![0.0; n * n];

        // Face reference quadrature on Tri3 (barycentric)
        let face_q_pts = [(1.0/3.0, 1.0/3.0)];
        let face_q_w = [1.0];

        for (qi, (u, v)) in face_q_pts.iter().enumerate() {
            let w_face = face_q_w[qi] * face_area / 1.0; // centroid rule

            // Map (u,v) → barycentric (lambda0 = 1-u-v, lambda1 = u, lambda2 = v)
            let l0 = 1.0 - u - v;
            let l1 = *u;
            let l2 = *v;

            // Physical point on face
            let xp = [
                l0 * mesh.coords[fnodes0 as usize * 3] + l1 * mesh.coords[fnodes1 as usize * 3] + l2 * mesh.coords[fnodes2 as usize * 3],
                l0 * mesh.coords[fnodes0 as usize * 3 + 1] + l1 * mesh.coords[fnodes1 as usize * 3 + 1] + l2 * mesh.coords[fnodes2 as usize * 3 + 1],
                l0 * mesh.coords[fnodes0 as usize * 3 + 2] + l1 * mesh.coords[fnodes1 as usize * 3 + 2] + l2 * mesh.coords[fnodes2 as usize * 3 + 2],
            ];

            // Map to element reference coords (affine inverse)
            let xi_l = [
                (xp[0] - x0_l[0]) * jit_l[0][0] + (xp[1] - x0_l[1]) * jit_l[1][0] + (xp[2] - x0_l[2]) * jit_l[2][0],
                (xp[0] - x0_l[0]) * jit_l[0][1] + (xp[1] - x0_l[1]) * jit_l[1][1] + (xp[2] - x0_l[2]) * jit_l[2][1],
                (xp[0] - x0_l[0]) * jit_l[0][2] + (xp[1] - x0_l[1]) * jit_l[1][2] + (xp[2] - x0_l[2]) * jit_l[2][2],
            ];
            let xi_r = [
                (xp[0] - x0_r[0]) * jit_r[0][0] + (xp[1] - x0_r[1]) * jit_r[1][0] + (xp[2] - x0_r[2]) * jit_r[2][0],
                (xp[0] - x0_r[0]) * jit_r[0][1] + (xp[1] - x0_r[1]) * jit_r[1][1] + (xp[2] - x0_r[2]) * jit_r[2][1],
                (xp[0] - x0_r[0]) * jit_r[0][2] + (xp[1] - x0_r[1]) * jit_r[1][2] + (xp[2] - x0_r[2]) * jit_r[2][2],
            ];

            re.eval_basis(&xi_l, &mut phi_l);
            re.eval_basis(&xi_r, &mut phi_r);
            re.eval_grad_basis(&xi_l, &mut gref_l);
            re.eval_grad_basis(&xi_r, &mut gref_r);

            for i in 0..n {
                for d in 0..3 {
                    gp_l[i*3+d] = (0..3).map(|k| jit_l[k][d] * gref_l[i*3+k]).sum();
                    gp_r[i*3+d] = (0..3).map(|k| jit_r[k][d] * gref_r[i*3+k]).sum();
                }
            }

            let nprj_l: Vec<f64> = (0..n).map(|i| gp_l[i*3]*norm_l[0] + gp_l[i*3+1]*norm_l[1] + gp_l[i*3+2]*norm_l[2]).collect();
            let nprj_r: Vec<f64> = (0..n).map(|i| gp_r[i*3]*(-norm_l[0]) + gp_r[i*3+1]*(-norm_l[1]) + gp_r[i*3+2]*(-norm_l[2])).collect();

            let pen = sigma / face_area;

            // SIP blocks (same algebra as dg.rs interior face)
            for i in 0..n {
                for j in 0..n {
                    kll[i*n+j] += w_face * kappa * (-0.5 * nprj_l[i] * phi_l[j] - 0.5 * nprj_l[j] * phi_l[i])
                                + w_face * pen * phi_l[i] * phi_l[j];
                    klr[i*n+j] += w_face * kappa * (0.5 * nprj_l[i] * phi_r[j] - 0.5 * nprj_r[j] * phi_l[i])
                                - w_face * pen * phi_l[i] * phi_r[j];
                    krl[i*n+j] += w_face * kappa * (-0.5 * nprj_r[i] * phi_l[j] + 0.5 * nprj_l[j] * phi_r[i])
                                - w_face * pen * phi_r[i] * phi_l[j];
                    krr[i*n+j] += w_face * kappa * (0.5 * nprj_r[i] * phi_r[j] + 0.5 * nprj_r[j] * phi_r[i])
                                + w_face * pen * phi_r[i] * phi_r[j];
                }
            }
        }

        for (i, &gi) in dofs_l.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() { coo.add(gi, gj, kll[i*n+j]); }
            for (j, &gj) in dofs_r.iter().enumerate() { coo.add(gi, gj, klr[i*n+j]); }
        }
        for (i, &gi) in dofs_r.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() { coo.add(gi, gj, krl[i*n+j]); }
            for (j, &gj) in dofs_r.iter().enumerate() { coo.add(gi, gj, krr[i*n+j]); }
        }
    }

    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_space::L2Space;

    fn sphere_map(p: [f64; 3], r: f64) -> [f64; 3] {
        let len = f64::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).max(1e-14);
        [p[0]*r/len, p[1]*r/len, p[2]*r/len]
    }

    #[test]
    fn curved_3d_sip_symmetric() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 1, |p| p);
        let space = L2Space::new(mesh, 1);
        let mat = assemble_sip_curved_3d(&space, space.mesh(), &curved, 1.0, 10.0, 3);
        for i in 0..mat.nrows.min(80) {
            for j in 0..i.min(80) {
                assert!((mat.get(i,j)-mat.get(j,i)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn curved_3d_constant_kernel() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 1, |p| p);
        let space = L2Space::new(mesh, 1);
        let mat = assemble_sip_curved_3d(&space, space.mesh(), &curved, 1.0, 10.0, 3);
        let mut au = vec![0.0; space.n_dofs()];
        for i in 0..space.n_dofs() {
            au[i] = (mat.row_ptr[i]..mat.row_ptr[i+1]).map(|p| mat.values[p]).sum();
        }
        let max = au.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max < 1e-12, "A·1 ≈ 0, got {max:.3e}");
    }
}
