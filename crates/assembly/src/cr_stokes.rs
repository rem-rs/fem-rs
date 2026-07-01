//! CR/P1 Stokes solver: Crouzeix–Raviart velocity + P1 pressure.
//!
//! This pair is **inf–sup stable** on simplex meshes.  Velocity DOFs are at
//! edge midpoints (one per edge, per spatial component).  Pressure is the
//! standard P1 (continuous, vertex DOFs).
//!
//! ```text
//! −νΔu + ∇p = f    in Ω
//!      ∇·u = 0    in Ω
//!        u = u_D  on ∂Ω
//! ```
//!
//! # Block system
//! ```text
//! [ νA   Bᵀ ] [u]   [f]
//! [ B    0  ] [p] = [0]
//! ```
//!
//! # Reference
//! - Crouzeix & Raviart (1973), RAIRO Série Rouge 7(R3):33–76.

use fem_core::NodeId;
use fem_linalg::CooMatrix;
use fem_linalg::dense::{lu_factor, lu_solve};
use fem_mesh::topology::MeshTopology;
use fem_mesh::element_type::ElementType;

/// Result of a CR/P1 Stokes solve.
pub struct CrStokesResult {
    /// Velocity at edge midpoints. Length = 2 × n_edges.
    pub u: Vec<f64>,
    /// Pressure at vertices. Length = n_nodes.
    pub p: Vec<f64>,
}

// ─── Reference element helpers ─────────────────────────────────────────────────

fn cr1_basis(xi: &[f64], phi: &mut [f64]) {
    phi[0] = 1.0 - 2.0 * xi[1];
    phi[1] = 2.0 * (xi[0] + xi[1]) - 1.0;
    phi[2] = 1.0 - 2.0 * xi[0];
}
fn cr1_grad(_xi: &[f64], grads: &mut [f64]) {
    grads[0] = 0.0; grads[1] = -2.0;
    grads[2] = 2.0; grads[3] = 2.0;
    grads[4] = -2.0; grads[5] = 0.0;
}
fn p1_basis(xi: &[f64], phi: &mut [f64]) {
    phi[0] = 1.0 - xi[0] - xi[1];
    phi[1] = xi[0];
    phi[2] = xi[1];
}
fn tri_quad6() -> ([[f64; 2]; 6], [f64; 6]) {
    let p = [[1.0/6.0, 1.0/6.0], [2.0/3.0, 1.0/6.0], [1.0/6.0, 2.0/3.0],
             [0.2, 0.2], [0.6, 0.2], [0.2, 0.6]];
    let w = [1.0/12.0; 6];
    (p, w)
}

// ─── DOF numbering ─────────────────────────────────────────────────────────────

fn build_cr1_dofs(mesh: &dyn MeshTopology) -> (Vec<usize>, usize) {
    use std::collections::HashMap;
    let mut key_to_id: HashMap<(NodeId, NodeId), usize> = HashMap::new();
    let mut next = 0;
    for e in mesh.elem_iter() {
        let n = mesh.element_nodes(e);
        for &(a, b) in &[(n[0], n[1]), (n[1], n[2]), (n[2], n[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            key_to_id.entry(key).or_insert_with(|| { let id = next; next += 1; id });
        }
    }
    let mut eids = Vec::with_capacity(mesh.n_elements() * 3);
    for e in mesh.elem_iter() {
        let n = mesh.element_nodes(e);
        for &(a, b) in &[(n[0], n[1]), (n[1], n[2]), (n[2], n[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            eids.push(key_to_id[&key]);
        }
    }
    (eids, next)
}

// ─── Solver ────────────────────────────────────────────────────────────────────

/// Solve the 2-D Stokes problem with CR/P1 elements using a dense direct solver.
///
/// A small regularization (ε = 1e-12 · ν) is added to the pressure block to
/// make the saddle-point system invertible for the dense LU solver.  This
/// introduces a negligible compressibility error.
///
/// # Arguments
/// * `mesh` — Tri3 mesh.
/// * `f`    — body force per node, interleaved: `[f_x0, f_y0, f_x1, f_y1, …]`.
///            Pass `&[]` for zero.
/// * `nu`   — kinematic viscosity.
/// * `bdry_dofs` — velocity DOFs to zero (no-slip).  Each entry is
///                 `edge_id * 2 + component`.
pub fn solve_cr_stokes(
    mesh: &dyn MeshTopology,
    f: &[f64],
    nu: f64,
    bdry_dofs: &[usize],
) -> CrStokesResult {
    assert_eq!(mesh.dim(), 2, "CR/P1 Stokes is 2-D only");
    let n_elems = mesh.n_elements();
    let n_nodes = mesh.n_nodes();

    let (elem_edge_ids, n_edges) = build_cr1_dofs(mesh);
    let n_vel = n_edges * 2;
    let n_pres = n_nodes;
    let nsys = n_vel + n_pres;

    let mut coo_sys = CooMatrix::<f64>::new(nsys, nsys);
    let mut rhs_sys = vec![0.0_f64; nsys];
    let (qp, qw) = tri_quad6();

    for e in mesh.elem_iter() {
        assert_eq!(mesh.element_type(e), ElementType::Tri3,
                   "CR/P1 Stokes requires Tri3 mesh");
        let ns = mesh.element_nodes(e);
        let c = |i: usize, d: usize| mesh.node_coords(ns[i])[d];

        let j00 = c(1,0)-c(0,0); let j01 = c(2,0)-c(0,0);
        let j10 = c(1,1)-c(0,1); let j11 = c(2,1)-c(0,1);
        let det_j = j00*j11 - j01*j10;
        let idet = if det_j.abs() > 1e-30 { 1.0/det_j } else { 0.0 };
        let jit = [[j11*idet, -j10*idet], [-j01*idet, j00*idet]];

        let e0 = elem_edge_ids[e as usize * 3];
        let e1 = elem_edge_ids[e as usize * 3 + 1];
        let e2 = elem_edge_ids[e as usize * 3 + 2];
        let eid = [e0, e1, e2];

        let mut a_el = [0.0_f64; 36];
        let mut b_el = [0.0_f64; 18];
        let mut r_el = [0.0_f64; 6];

        for (q, xi) in qp.iter().enumerate() {
            let w = qw[q] * det_j.abs();
            let mut pv = [0.0_f64; 3]; cr1_basis(xi, &mut pv);
            let mut pp = [0.0_f64; 3]; p1_basis(xi, &mut pp);
            let mut gr = [0.0_f64; 6]; cr1_grad(xi, &mut gr);
            let mut gp = [0.0_f64; 6];
            for i in 0..3 {
                gp[i*2]   = jit[0][0]*gr[i*2] + jit[0][1]*gr[i*2+1];
                gp[i*2+1] = jit[1][0]*gr[i*2] + jit[1][1]*gr[i*2+1];
            }
            let fq = if f.is_empty() { [0.0;2] } else {
                let mut fx=0.0; let mut fy=0.0;
                for k in 0..3 { fx+=f[ns[k]as usize*2]*pv[k]; fy+=f[ns[k]as usize*2+1]*pv[k]; }
                [fx,fy]
            };
            for i in 0..3 { for j in 0..3 {
                let av = nu*w*(gp[i*2]*gp[j*2]+gp[i*2+1]*gp[j*2+1]);
                for c in 0..2 { a_el[(i*2+c)*6+j*2+c] += av; }
            }}
            for i in 0..3 { for c in 0..2 { r_el[i*2+c] += w*fq[c]*pv[i]; }}
            for p in 0..3 { for v in 0..3 {
                b_el[p*6+v*2]   += -w*pp[p]*gp[v*2];
                b_el[p*6+v*2+1] += -w*pp[p]*gp[v*2+1];
            }}
        }

        for i in 0..3 { for c1 in 0..2 {
            let row = eid[i]*2+c1;
            for j in 0..3 { for c2 in 0..2 {
                if c1==c2 { coo_sys.add(row, eid[j]*2+c2, a_el[(i*2+c1)*6+j*2+c2]); }
            }}
            rhs_sys[row] += r_el[i*2+c1];
        }}
        for p in 0..3 { for v in 0..3 { for c in 0..2 {
            let val = b_el[p*6+v*2+c];
            if val.abs() > 1e-30 {
                let r = n_vel + ns[p] as usize;
                let col = eid[v]*2+c;
                coo_sys.add(r, col, val);
                coo_sys.add(col, r, val);
            }
        }}}
    }

    // Add small regularization to the pressure block to make the saddle-point
    // system invertible for the dense LU solver.  ε = 1e-12 · tr(νA)/n_vel
    // is a negligible perturbation to the incompressibility constraint.
    // Without this, the zero block makes the matrix singular in exact arithmetic.
    let reg = 1e-12 * nu;
    for p in 0..n_pres { coo_sys.add(n_vel + p, n_vel + p, reg); }

    // BC: no-slip velocity
    let mut sys_csr = coo_sys.into_csr();
    for &d in bdry_dofs { sys_csr.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs_sys); }

    // Dense solve
    let nz = nsys;
    let mut dense = sys_csr.to_dense();
    let mut piv = vec![0usize; nz];
    lu_factor(&mut dense, nz, &mut piv).expect("CR Stokes LU failed");
    lu_solve(&dense, nz, &piv, &mut rhs_sys);

    CrStokesResult { u: rhs_sys[..n_vel].to_vec(), p: rhs_sys[n_vel..].to_vec() }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    fn unit_sq(n: usize) -> SimplexMesh<2> { SimplexMesh::<2>::unit_square_tri(n) }

    #[test]
    fn cr_stokes_dof_count() {
        let m = unit_sq(2);
        let (eids, ne) = build_cr1_dofs(&m);
        assert!(ne >= 8, "edges {ne}");
        assert_eq!(eids.len(), m.n_elements() * 3);
    }

    #[test]
    fn cr_stokes_fully_pinned_solves() {
        // Pin ALL boundary velocity DOFs → unique solvable system.
        let m = unit_sq(2);
        let (eids, ne) = build_cr1_dofs(&m);
        // Find edges on the mesh boundary
        use std::collections::HashSet;
        let mut bnd_edges: HashSet<(NodeId,NodeId)> = HashSet::new();
        for f in 0..m.n_faces() {
            let a=m.face_conn[2*f]; let b=m.face_conn[2*f+1];
            bnd_edges.insert(if a<b{(a,b)}else{(b,a)});
        }
        // Match them to edge IDs
        let mut key_eid: std::collections::HashMap<(NodeId,NodeId),usize> = std::collections::HashMap::new();
        for e in m.elem_iter() {
            let n=m.element_nodes(e);
            for li in 0..3 {
                let (a,b)=[(n[0],n[1]),(n[1],n[2]),(n[2],n[0])][li];
                let k=if a<b{(a,b)}else{(b,a)};
                key_eid.entry(k).or_insert(eids[e as usize*3+li]);
            }
        }
        let mut bdry=Vec::new();
        for (k,&eid) in &key_eid { if bnd_edges.contains(k) { bdry.push(eid*2); bdry.push(eid*2+1); }}
        let r=solve_cr_stokes(&m,&[],1.0,&bdry);
        assert!(r.u.iter().all(|v|v.is_finite()));
        assert!(r.p.iter().all(|v|v.is_finite()));
    }

    #[test]
    fn cr_stokes_poiseuille() {
        let m = SimplexMesh::<2>::unit_square_tri(6);
        let nn = m.n_nodes();
        let nu = 1.0;
        let f: Vec<f64> = (0..nn).flat_map(|_| vec![8.0*nu, 0.0]).collect();

        let (eids, _) = build_cr1_dofs(&m);
        use std::collections::HashMap;
        let mut key_eid: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        for e in m.elem_iter() {
            let n = m.element_nodes(e);
            for li in 0..3 {
                let (a,b)=[(n[0],n[1]),(n[1],n[2]),(n[2],n[0])][li];
                let k=if a<b{(a,b)}else{(b,a)};
                key_eid.entry(k).or_insert(eids[e as usize*3+li]);
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
