//! Weak Galerkin (WG) finite element method for Poisson.
//!
//! Reference: Wang & Ye (2013), "A weak Galerkin FEM for second-order elliptic problems".
//!
//! Bilinear form: a_h(u,v) = (kappa * grad_w u, grad_w v) + s(u,v)
//! where grad_w is the weak gradient and s is a face stabilizer.
//!
//! The face stabilizer's geometry (measure, physical point, element reference
//! point) comes from the family's shared isoparametric face path
//! (`super::wg_face_point` / `super::wg_face_measure`, i.e.
//! `dg_base::face_point_geom`); see [`super`] for what the pre-D810-1 chord
//! route got wrong.

use nalgebra::DMatrix;
use fem_element::{
    ReferenceElement, lagrange::{TriPk, TetPk},
};
use fem_element::quadrature::{tri_rule, tet_rule};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use super::{wg_boundary_face_map, wg_face_measure, wg_face_point, wg_face_rule};

// ─── Local helpers (mirrored from crate internals) ─────────────────────────

/// Element Jacobian from the element's **vertices** (the volume path's
/// geometry — the D808-4-class residual registered in [`super`]).
fn local_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut jac = DMatrix::zeros(dim, dim);
    for i in 0..dim { let xi = mesh.node_coords(nodes[1 + i]);
        for d in 0..dim { jac[(d, i)] = xi[d] - x0[d]; }
    }
    let det = jac.determinant();
    (jac, det)
}

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
    let _gv = vec![0.0; n_v * dim];
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
        let h = wg_face_measure(mesh, f.elem_left, &f.face_nodes, quad_order);
        let alpha = penalty / h.max(1e-14);
        add_face_penalty(&mut coo, mesh, space, f.elem_left, f.elem_right, &f.face_nodes, alpha, quad_order);
    }
    let fe_map = wg_boundary_face_map(mesh);
    for bf in mesh.face_iter() {
        if let Some(&el) = fe_map.get(&bf) {
            let fnodes: Vec<u32> = mesh.face_nodes(bf).to_vec();
            let h = wg_face_measure(mesh, el, &fnodes, quad_order);
            let alpha = penalty / h.max(1e-14);
            add_face_penalty(&mut coo, mesh, space, el, el, &fnodes, alpha, quad_order);
        }
    }
    coo.into_csr()
}

#[allow(clippy::too_many_arguments)]
fn add_face_penalty<M: MeshTopology, S: FESpace<Mesh=M>>(
    coo: &mut CooMatrix<f64>, mesh: &M, space: &S,
    el: u32, er: u32, fnodes: &[u32], alpha: f64, qo: u8,
) {
    let dim = mesh.dim() as usize;
    let order = space.order() as usize;
    let ref_e: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(order)) }
                                           else { Box::new(TetPk::new(order)) };
    let ne = ref_e.n_dofs();
    let qf = wg_face_rule(dim, qo);
    let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

    for (qi, xi) in qf.points.iter().enumerate() {
        // D810-1: the QP's measure is `ipw·|nor|` of the element's own
        // isoparametric face, and the basis is evaluated at the composed
        // reference point (`eip`) — no corner interpolation, no inversion.
        let g = wg_face_point(mesh, el, fnodes, xi);
        let w = qf.weights[qi] * g.nor_mag();
        let mut pl = vec![0.0; ne]; ref_e.eval_basis(&g.eip, &mut pl);
        for i in 0..ne { for j in 0..ne {
            let v = alpha * w * pl[i] * pl[j];
            if v.abs() > 1e-30 { coo.add(dofs_l[i], dofs_l[j], v); }
        }}
        if el != er {
            let gr = wg_face_point(mesh, er, fnodes, xi);
            let mut pr = vec![0.0; ne]; ref_e.eval_basis(&gr.eip, &mut pr);
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
    use fem_mesh::Mesh;
    use fem_space::L2Space;

    #[test] fn wg_poisson_spd() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let sp = L2Space::new(mesh, 1);
        let k = assemble_wg_poisson(&sp, 3, 10.0);
        let n = k.nrows;
        for i in 0..n { assert!(k.get(i,i) > 0.0, "diag[{i}]={}", k.get(i,i)); }
    }

    #[test] fn wg_poisson_cg_solves() {
        let mesh = Mesh::<2>::unit_square_tri(4);
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
