//! CR/P1 Stokes solver: Crouzeix–Raviart velocity + P1 pressure.
//!
//! Uses [`CRSpace`](fem_space::CRSpace) for edge-based DOF numbering and
//! [`ref_elem_cr`](crate::dg_advection::ref_elem_cr) for CR basis function
//! evaluation.  The element loop is inlined (not via `Assembler`) because
//! the generic assembler dispatches to Lagrange reference elements.
//!
//! ```text
//! −νΔu + ∇p = f,   ∇·u = 0
//! ```

use fem_core::NodeId;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_linalg::dense::{lu_factor, lu_solve};
use fem_mesh::topology::MeshTopology;
use fem_mesh::element_type::ElementType;
use fem_element::{ReferenceElement, lagrange::TriP1};
use fem_space::{FESpace, CRSpace};

use crate::dg_advection::{ref_elem_cr, simplex_jac, xform_grads};

/// Result of a CR/P1 Stokes solve.
pub struct CrStokesResult {
    pub u: Vec<f64>,
    pub p: Vec<f64>,
}

/// Solve the 2-D Stokes problem with CR/P1 elements via dense LU.
///
/// # Arguments
/// * `mesh` — Tri3 mesh.
/// * `f`    — body force per vertex `[f_x0, f_y0, f_x1, f_y1, …]`.  `&[]` for zero.
/// * `nu`   — kinematic viscosity.
/// * `bdry_dofs` — velocity DOFs to zero (no-slip).  Each entry is a *scalar*
///                 CR DOF index `edge_id * 2 + component` (0=x, 1=y).
pub fn solve_cr_stokes<M: MeshTopology + Clone + 'static>(
    mesh: &M,
    f: &[f64],
    nu: f64,
    bdry_dofs: &[usize],
) -> CrStokesResult {
    assert_eq!(mesh.dim(), 2, "CR/P1 Stokes is 2-D only");
    let n_elems = mesh.n_elements();
    let n_nodes = mesh.n_nodes();

    // ── 1. DOF numbering (CRSpace) ──────────────────────────────────
    // CRSpace provides edge-based DOFs; double for the two velocity components.
    let cr = CRSpace::new(mesh.clone(), 1);
    let n_edges = cr.n_dofs();
    let n_vel = n_edges * 2;
    let n_pres = n_nodes;
    let nsys = n_vel + n_pres;

    let ref_v = ref_elem_cr(ElementType::Tri3, 1);
    let ref_p = Box::new(TriP1);
    let n_ldofs_v = ref_v.n_dofs();  // 3
    let n_ldofs_p = ref_p.n_dofs();  // 3
    let dim = 2usize;

    // Pre-build element DOF tables for velocity (interleaved)
    let mut elem_vel_dofs = Vec::with_capacity(n_elems * n_ldofs_v * dim);
    for e in mesh.elem_iter() {
        let sd = cr.element_dofs(e);
        for k in 0..n_ldofs_v {
            for c in 0..dim {
                elem_vel_dofs.push(sd[k] as usize * 2 + c);
            }
        }
    }

    let mut coo_sys = CooMatrix::<f64>::new(nsys, nsys);
    let mut rhs_sys = vec![0.0_f64; nsys];
    let quad = ref_v.quadrature(3);

    for e in mesh.elem_iter() {
        let ns = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, ns, dim);
        let jit = jac.try_inverse().unwrap().transpose();

        let voff = e as usize * n_ldofs_v * dim;
        let vd = |li: usize, c: usize| -> usize { elem_vel_dofs[voff + li * dim + c] };
        let pd = |li: usize| -> usize { ns[li] as usize };

        let n_vec = n_ldofs_v * dim;
        let mut a_el = vec![0.0_f64; n_vec * n_vec];
        let mut b_el = vec![0.0_f64; n_ldofs_p * n_vec];
        let mut r_el = vec![0.0_f64; n_vec];
        let mut phi_v = vec![0.0_f64; n_ldofs_v];
        let mut phi_p = vec![0.0_f64; n_ldofs_p];
        let mut gref = vec![0.0_f64; n_ldofs_v * dim];
        let mut gphys = vec![0.0_f64; n_ldofs_v * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            ref_v.eval_basis(xi, &mut phi_v);
            ref_p.eval_basis(xi, &mut phi_p);
            ref_v.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs_v, dim);

            let fq = if f.is_empty() { [0.0;2] } else {
                let (mut fx, mut fy) = (0.0, 0.0);
                for k in 0..3 { fx += f[ns[k]as usize*2]*phi_v[k]; fy += f[ns[k]as usize*2+1]*phi_v[k]; }
                [fx, fy]
            };

            for i in 0..n_ldofs_v { for j in 0..n_ldofs_v {
                let s = nu*w*(gphys[i*dim]*gphys[j*dim] + gphys[i*dim+1]*gphys[j*dim+1]);
                for c in 0..dim { a_el[(i*dim+c)*n_vec + j*dim + c] += s; }
            }}
            for i in 0..n_ldofs_v { for c in 0..dim { r_el[i*dim+c] += w*fq[c]*phi_v[i]; }}
            for p in 0..n_ldofs_p { for v in 0..n_ldofs_v {
                b_el[p*n_vec + v*dim]   += -w*phi_p[p]*gphys[v*dim];
                b_el[p*n_vec + v*dim+1] += -w*phi_p[p]*gphys[v*dim+1];
            }}
        }

        for i in 0..n_vec { for j in 0..n_vec {
            coo_sys.add(vd(i/dim, i%dim), vd(j/dim, j%dim), a_el[i*n_vec + j]);
        }}
        for i in 0..n_vec { rhs_sys[vd(i/dim, i%dim)] += r_el[i]; }
        for p in 0..n_ldofs_p { for v in 0..n_vec {
            let val = b_el[p*n_vec + v];
            if val.abs() > 1e-30 {
                let r = n_vel + pd(p);
                let c = vd(v/dim, v%dim);
                coo_sys.add(r, c, val);
                coo_sys.add(c, r, val);
            }
        }}
    }

    // ── 2. Regularisation + BCs ──────────────────────────────────────
    let reg = 1e-12 * nu;
    for p in 0..n_pres { coo_sys.add(n_vel + p, n_vel + p, reg); }

    let mut sys_csr = coo_sys.into_csr();
    for &d in bdry_dofs { sys_csr.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs_sys); }

    // ── 3. Dense solve ───────────────────────────────────────────────
    let mut dense = sys_csr.to_dense();
    let mut piv = vec![0usize; nsys];
    lu_factor(&mut dense, nsys, &mut piv).expect("CR Stokes LU failed");
    lu_solve(&dense, nsys, &piv, &mut rhs_sys);

    CrStokesResult { u: rhs_sys[..n_vel].to_vec(), p: rhs_sys[n_vel..].to_vec() }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::FESpace;

    #[test]
    fn cr_stokes_fully_pinned_solves() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let cr = CRSpace::new(m.clone(), 1);
        use std::collections::{HashSet, HashMap};
        let mut key_eid: HashMap<(NodeId,NodeId), usize> = HashMap::new();
        for e in m.elem_iter() {
            let n = m.element_nodes(e);
            for li in 0..3 {
                let (a,b) = [(n[0],n[1]),(n[1],n[2]),(n[2],n[0])][li];
                let k = if a<b{(a,b)}else{(b,a)};
                key_eid.entry(k).or_insert(cr.element_dofs(e)[li] as usize);
            }
        }
        let mut bdry = Vec::new();
        for f in 0..m.n_faces() {
            let a=m.face_conn[2*f]; let b=m.face_conn[2*f+1];
            let k=if a<b{(a,b)}else{(b,a)};
            if let Some(&eid)=key_eid.get(&k) { bdry.push(eid*2); bdry.push(eid*2+1); }
        }
        let r = solve_cr_stokes(&m, &[], 1.0, &bdry);
        assert!(r.u.iter().all(|v|v.is_finite()));
        assert!(r.p.iter().all(|v|v.is_finite()));
    }

    #[test]
    fn cr_stokes_poiseuille() {
        let m = SimplexMesh::<2>::unit_square_tri(6);
        let nn = m.n_nodes();
        let nu = 1.0;
        let f: Vec<f64> = (0..nn).flat_map(|_| vec![8.0*nu, 0.0]).collect();
        let cr = CRSpace::new(m.clone(), 1);
        use std::collections::HashMap;
        let mut key_eid: HashMap<(NodeId,NodeId), usize> = HashMap::new();
        for e in m.elem_iter() {
            let n = m.element_nodes(e);
            for li in 0..3 {
                let (a,b)=[(n[0],n[1]),(n[1],n[2]),(n[2],n[0])][li];
                let k=if a<b{(a,b)}else{(b,a)};
                key_eid.entry(k).or_insert(cr.element_dofs(e)[li] as usize);
            }
        }
        let mut bdry = Vec::new();
        for f in 0..m.n_faces() {
            let a=m.face_conn[2*f]; let b=m.face_conn[2*f+1];
            let (ya,yb)=(m.coords_of(a)[1],m.coords_of(b)[1]);
            if (ya.abs()<1e-12&&yb.abs()<1e-12)||((ya-1.0).abs()<1e-12&&(yb-1.0).abs()<1e-12) {
                let k=if a<b{(a,b)}else{(b,a)};
                if let Some(&eid)=key_eid.get(&k){bdry.push(eid*2);bdry.push(eid*2+1);}
            }
        }
        let r=solve_cr_stokes(&m,&f,nu,&bdry);
        assert!(r.u.iter().all(|v|v.is_finite()));
        let max_vx=r.u.iter().step_by(2).fold(0.0_f64,|a,&b|a.max(b));
        assert!(max_vx>0.2,"max vx={max_vx}");
    }
}
