//! Weak Galerkin (WG) finite element method for Poisson.
//!
//! Reference: Wang & Ye (2013), "A weak Galerkin FEM for second-order elliptic problems".
//!
//! Bilinear form: a_h(u,v) = (kappa * grad_w u, grad_w v) + s(u,v)
//! where grad_w is the weak gradient and s is a face stabilizer.

use std::collections::HashMap;
use nalgebra::DMatrix;
use fem_core::types::DofId;
use fem_element::{
    ReferenceElement, lagrange::{TriPk, TetPk, SegPk},
};
use fem_element::quadrature::{tri_rule, tet_rule};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

// ─── Local helpers (mirrored from crate internals) ─────────────────────────

fn local_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut jac = DMatrix::zeros(dim, dim);
    for i in 0..dim { let xi = mesh.node_coords(nodes[1 + i]);
        for d in 0..dim { jac[(d, i)] = xi[d] - x0[d]; }
    }
    let det = jac.determinant();
    (jac, det)
}

fn local_phys_to_ref(jac: &DMatrix<f64>, x0: &[f64], xp: &[f64], dim: usize) -> Vec<f64> {
    let ji = jac.clone().try_inverse().unwrap_or(DMatrix::identity(dim, dim));
    let mut xi = vec![0.0; dim];
    for i in 0..dim { for j in 0..dim { xi[i] += ji[(i, j)] * (xp[j] - x0[j]); }}
    xi
}

fn face_geom_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> f64 {
    let c0 = mesh.node_coords(nodes[0]);
    let c1 = mesh.node_coords(nodes[1]);
    let dx = c1[0] - c0[0]; let dy = c1[1] - c0[1];
    (dx*dx + dy*dy).sqrt()
}

fn face_geom_3d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> f64 {
    let c0 = mesh.node_coords(nodes[0]);
    let c1 = mesh.node_coords(nodes[1]);
    let c2 = mesh.node_coords(nodes[2]);
    let u1 = [c1[0]-c0[0], c1[1]-c0[1], c1[2]-c0[2]];
    let u2 = [c2[0]-c0[0], c2[1]-c0[1], c2[2]-c0[2]];
    let nx = u1[1]*u2[2] - u1[2]*u2[1];
    let ny = u1[2]*u2[0] - u1[0]*u2[2];
    let nz = u1[0]*u2[1] - u1[1]*u2[0];
    (nx*nx + ny*ny + nz*nz).sqrt() / 2.0
}

fn build_face_elem_map<M: MeshTopology>(mesh: &M) -> HashMap<u32, u32> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    let n_bfaces = mesh.n_boundary_faces() as u32;
    for e in 0..mesh.n_elements() as u32 {
        let ns = mesh.element_nodes(e);
        let npe = ns.len();
        let dim = mesh.dim();
        // Build all faces of this element and check if they match a boundary face
        for fi in 0..npe {
            let (a, b, c) = match dim {
                2 => (ns[fi], ns[(fi+1)%npe], u32::MAX),
                3 => { if npe == 4 { (ns[TET_FACES[fi][0]], ns[TET_FACES[fi][1]], ns[TET_FACES[fi][2]]) }
                       else if npe == 8 { let f = HEX_FACES[fi]; (ns[f[0]], ns[f[1]], ns[f[2]]) }
                       else { continue; }}
                _ => continue,
            };
            for f in 0..n_bfaces {
                let bns = mesh.face_nodes(f);
                let matches = match dim {
                    2 => bns.len() == 2 && ((bns[0]==a&&bns[1]==b)||(bns[0]==b&&bns[1]==a)),
                    3 => bns.len() == 3 && bns.contains(&a) && bns.contains(&b) && bns.contains(&c),
                    _ => false,
                };
                if matches { map.insert(f, e); }
            }
        }
    }
    map
}

const TET_FACES: [[usize; 3]; 4] = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]];
const HEX_FACES: [[usize; 4]; 6] = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]];

// ─── Weak gradient matrix ──────────────────────────────────────────────────

fn weak_gradient_matrix<M: MeshTopology>(
    mesh: &M, e: u32, dim: usize, order: usize, quad_order: u8,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let ref_v: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(order)) }
                                           else { Box::new(TetPk::new(order)) };
    let n_v = ref_v.n_dofs();
    let os = if order > 0 { order - 1 } else { 0 };
    // For Sigma_h = [P_{k-1}]^d: P_0 has 1 DOF (constant), P_1+ uses TriPk/TetPk
    let (n_ss, use_const_basis) = if os == 0 { (1_usize, true) } else {
        let ref_s: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(os)) }
                                               else { Box::new(TetPk::new(os)) };
        (ref_s.n_dofs(), false)
    };
    let n_s = dim * n_ss;

    let qr = if dim == 2 { tri_rule(quad_order) } else { tet_rule(quad_order) };
    let nodes = mesh.element_nodes(e);
    let (jac, det_j) = local_jac(mesh, nodes, dim);
    if det_j.abs() < 1e-30 { return (DMatrix::zeros(n_v, n_s), DMatrix::zeros(n_s, n_s)); }
    let jit = jac.clone().try_inverse().unwrap().transpose();

    let mut G = DMatrix::zeros(n_v, n_s);
    let mut Ms = DMatrix::zeros(n_s, n_s);
    let mut pv = vec![0.0; n_v];
    let mut gv = vec![0.0; n_v * dim];
    let mut ps = vec![0.0; n_ss];
    let mut gsp = vec![0.0; n_ss * dim];

    for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
        let wq = w * det_j.abs();
        ref_v.eval_basis(pt, &mut pv);
        if use_const_basis {
            // P_0 on sigma: basis = 1, gradient = 0
            ps[0] = 1.0;
            for d in 0..dim { gsp[d] = 0.0; }
        } else {
            let ref_s: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(os)) }
                                                   else { Box::new(TetPk::new(os)) };
            let mut gs = vec![0.0; n_ss * dim];
            ref_s.eval_basis(pt, &mut ps);
            ref_s.eval_grad_basis(pt, &mut gs);
            for i in 0..n_ss {
                for d in 0..dim { gsp[i*dim+d] = (0..dim).map(|k| jit[(d,k)]*gs[i*dim+k]).sum(); }
            }
        }
        for i in 0..n_v { for j in 0..n_s {
            let sc = j / n_ss; let sd = j % n_ss;
            G[(i,j)] -= wq * pv[i] * gsp[sd*dim + sc];
        }}
        for p in 0..n_s { let pc = p/n_ss; let pd = p%n_ss;
            for q in 0..n_s { let qc = q/n_ss; let qd = q%n_ss;
                if pc == qc { Ms[(p,q)] += wq * ps[pd] * ps[qd]; }
            }
        }
    }
    (G, Ms)
}

// ─── WG Poisson assembly ──────────────────────────────────────────────────

pub fn assemble_wg_poisson<S: FESpace>(space: &S, quad_order: u8, penalty: f64) -> CsrMatrix<f64> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let n = space.n_dofs();
    let ne = mesh.n_elements();
    let order = space.order() as usize;
    use std::collections::HashMap;

    let mut coo = CooMatrix::new(n, n);

    for e in 0..ne as u32 {
        let (G, Ms) = weak_gradient_matrix(mesh, e, dim, order, quad_order);
        let nv = G.nrows(); let ns = G.ncols();
        let Gt = G.transpose();
        let X = Ms.clone().lu().solve(&Gt).unwrap_or(DMatrix::zeros(ns, nv));
        let Kl = &G * X;
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for i in 0..nv { for j in 0..nv {
            let v = Kl[(i,j)]; if v.abs() > 1e-30 { coo.add(dofs[i], dofs[j], v); }
        }}
    }

    // Face penalty (simplified: h^{-1} * penalty * ∫_F [u][v])
    let interior_faces = crate::InteriorFaceList::build(mesh);
    for f in &interior_faces.faces {
        let h = if dim == 2 { face_geom_2d(mesh, &f.face_nodes) }
                else { face_geom_3d(mesh, &f.face_nodes) };
        let alpha = penalty / h.max(1e-14);
        add_face_penalty(&mut coo, mesh, space, f.elem_left, f.elem_right, &f.face_nodes, alpha, quad_order);
    }
    let fe_map = build_face_elem_map(mesh);
    for bf in mesh.face_iter() {
        if let Some(&el) = fe_map.get(&bf) {
            let fnodes: Vec<u32> = mesh.face_nodes(bf).to_vec();
            let h = if dim == 2 { face_geom_2d(mesh, &fnodes) }
                    else { face_geom_3d(mesh, &fnodes) };
            let alpha = penalty / h.max(1e-14);
            add_face_penalty(&mut coo, mesh, space, el, el, &fnodes, alpha, quad_order);
        }
    }
    coo.into_csr()
}

fn add_face_penalty<M: MeshTopology, S: FESpace<Mesh=M>>(
    coo: &mut CooMatrix<f64>, mesh: &M, space: &S,
    el: u32, er: u32, fnodes: &[u32], alpha: f64, qo: u8,
) {
    let dim = mesh.dim() as usize;
    let order = space.order() as usize;
    let ref_e: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(order)) }
                                           else { Box::new(TetPk::new(order)) };
    let ne = ref_e.n_dofs();
    let qf = if dim == 2 { tri_rule(qo) } else { tri_rule(qo) };
    let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

    for (qi, xi) in qf.points.iter().enumerate() {
        let w = qf.weights[qi] * (if dim == 2 { face_geom_2d(mesh, fnodes) } else { face_geom_3d(mesh, fnodes) });
        let xp = if dim == 2 { let c = mesh.node_coords(fnodes[0]); let d = mesh.node_coords(fnodes[1]);
                [c[0]+xi[0]*(d[0]-c[0]), c[1]+xi[0]*(d[1]-c[1]), 0.0] }
               else { let c0=mesh.node_coords(fnodes[0]);let c1=mesh.node_coords(fnodes[1]);let c2=mesh.node_coords(fnodes[2]);
                [c0[0]+xi[0]*(c1[0]-c0[0])+xi[1]*(c2[0]-c0[0]),
                 c0[1]+xi[0]*(c1[1]-c0[1])+xi[1]*(c2[1]-c0[1]),
                 c0[2]+xi[0]*(c1[2]-c0[2])+xi[1]*(c2[2]-c0[2])] };
        let nl = mesh.element_nodes(el);
        let (jl, _) = local_jac(mesh, nl, dim);
        let xil = local_phys_to_ref(&jl, mesh.node_coords(nl[0]), &xp[..dim], dim);
        let mut pl = vec![0.0; ne]; ref_e.eval_basis(&xil, &mut pl);
        for i in 0..ne { for j in 0..ne {
            let v = alpha * w * pl[i] * pl[j];
            if v.abs() > 1e-30 { coo.add(dofs_l[i], dofs_l[j], v); }
        }}
        if el != er {
            let nr = mesh.element_nodes(er);
            let (jr, _) = local_jac(mesh, nr, dim);
            let xir = local_phys_to_ref(&jr, mesh.node_coords(nr[0]), &xp[..dim], dim);
            let mut pr = vec![0.0; ne]; ref_e.eval_basis(&xir, &mut pr);
            for i in 0..ne { for j in 0..ne {
                let v = alpha * w * pr[i] * pr[j];
                if v.abs() > 1e-30 { coo.add(dofs_r[i], dofs_r[j], v); }
            }}
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::L2Space;

    #[test] fn wg_poisson_spd() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let sp = L2Space::new(mesh, 1);
        let k = assemble_wg_poisson(&sp, 3, 10.0);
        let n = k.nrows;
        for i in 0..n { assert!(k.get(i,i) > 0.0, "diag[{i}]={}", k.get(i,i)); }
    }

    #[test] fn wg_poisson_cg_solves() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let sp = L2Space::new(mesh, 1);
        let k = assemble_wg_poisson(&sp, 3, 100.0);
        let n = k.nrows;
        let mut x = vec![0.0; n];
        let mut r = vec![1.0; n];
        let mut p = r.clone();
        let mut rr: f64 = r.iter().map(|v| v*v).sum();
        for _ in 0..300 {
            let mut ap = vec![0.0; n];
            for i in 0..n { for ptr in k.row_ptr[i]..k.row_ptr[i+1] { ap[i] += k.values[ptr] * p[k.col_idx[ptr] as usize]; }}
            let pap: f64 = p.iter().zip(ap.iter()).map(|(a,b)| a*b).sum();
            if pap.abs() < 1e-40 { break; }
            let al = rr / pap;
            for i in 0..n { x[i] += al * p[i]; r[i] -= al * ap[i]; }
            let rrn: f64 = r.iter().map(|v| v*v).sum();
            if rrn.sqrt() < 1e-8 { break; }
            let be = rrn / rr; rr = rrn;
            for i in 0..n { p[i] = r[i] + be * p[i]; }
        }
        let res: f64 = r.iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(res < 1e-6, "WG CG residual = {res:.3e}");
    }
}
