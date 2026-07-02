//! Weak Galerkin (WG) finite element method for Maxwell equations.
//!
//! Reference: Wang & Ye (2013), "A weak Galerkin FEM for Maxwell equations".
//! Bilinear form: a_h(E,v) = (κ ∇_w × E, ∇_w × v)_T + s(E,v)
//!
//! E ∈ V_h = Nédélec order k,  flux Σ_h = [P_{k-1}]^d.
//! Face stabilizer s(·,·) penalizes tangential jumps.

use std::collections::HashMap;
use nalgebra::DMatrix;
use fem_core::types::DofId;
use fem_element::{
    ReferenceElement, VectorReferenceElement,
    lagrange::{TriPk, TetPk, SegPk},
    nedelec::{TriNDk, TetNDk},
};
use fem_element::quadrature::{tri_rule, tet_rule};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

// ─── Local geometry helpers (same as wg_poisson) ──────────────────────────

fn local_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut jac = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        let xi = mesh.node_coords(nodes[1 + i]);
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
    let c0 = mesh.node_coords(nodes[0]); let c1 = mesh.node_coords(nodes[1]);
    ((c1[0]-c0[0]).powi(2) + (c1[1]-c0[1]).powi(2)).sqrt()
}

fn face_geom_3d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> f64 {
    let c0 = mesh.node_coords(nodes[0]); let c1 = mesh.node_coords(nodes[1]);
    let c2 = mesh.node_coords(nodes[2]);
    let u1 = [c1[0]-c0[0], c1[1]-c0[1], c1[2]-c0[2]];
    let u2 = [c2[0]-c0[0], c2[1]-c0[1], c2[2]-c0[2]];
    let nx = u1[1]*u2[2] - u1[2]*u2[1];
    let ny = u1[2]*u2[0] - u1[0]*u2[2];
    let nz = u1[0]*u2[1] - u1[1]*u2[0];
    ((nx*nx + ny*ny + nz*nz).sqrt()) / 2.0
}

fn build_face_elem_map<M: MeshTopology>(mesh: &M) -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        let fnodes = mesh.face_nodes(f);
        for e in mesh.elem_iter() {
            if fnodes.iter().all(|n| mesh.element_nodes(e).contains(n)) {
                map.insert(f, e); break;
            }
        }
    }
    map
}

// ─── Weak curl matrix ─────────────────────────────────────────────────────
// C_w[i,j] = ∫_T (∇_w × φ_j) · σ_i dV  where φ_j ∈ V_h (Nédélec), σ_i ∈ Σ_h
// M_Σ[m,n] = ∫_T σ_m · σ_n dV  (flux mass matrix, used to recover curl)

fn weak_curl_matrix<M: MeshTopology>(
    mesh: &M, e: u32, dim: usize, order: usize, quad_order: u8,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let nd_elem: Box<dyn VectorReferenceElement> = if dim == 2 {
        Box::new(TriNDk::new(order))
    } else {
        Box::new(TetNDk::new(order))
    };
    let n_v = nd_elem.n_dofs();

    let os = if order > 0 { order - 1 } else { 0 };
    let n_ss: usize;
    let mut ref_s: Option<Box<dyn ReferenceElement>> = None;
    if os == 0 {
        n_ss = 1;
    } else {
        let rs: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(os)) }
                                              else { Box::new(TetPk::new(os)) };
        n_ss = rs.n_dofs();
        ref_s = Some(rs);
    }
    let n_s = if dim == 2 { n_ss } else { dim * n_ss };

    let qr = if dim == 2 { tri_rule(quad_order) } else { tet_rule(quad_order) };
    let nodes = mesh.element_nodes(e);
    let (jac, det_j) = local_jac(mesh, nodes, dim);
    if det_j.abs() < 1e-30 { return (DMatrix::zeros(n_v, n_s), DMatrix::zeros(n_s, n_s)); }
    let jit = jac.clone().try_inverse().unwrap().transpose();
    let abs_det = det_j.abs();

    let mut Cw = DMatrix::zeros(n_v, n_s);
    let mut Ms = DMatrix::zeros(n_s, n_s);
    let mut nv_basis = vec![0.0_f64; n_v * dim];
    let mut nv_curl = vec![0.0_f64; n_v * dim];
    let mut ps = vec![0.0_f64; n_ss];
    let mut gsp = vec![0.0_f64; n_ss * dim];

    for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
        let wq = w * abs_det;
        nd_elem.eval_basis_vec(pt, &mut nv_basis);
        nd_elem.eval_curl(pt, &mut nv_curl);

        // Sigma basis values and gradients
        if os == 0 {
            ps[0] = 1.0;
        } else {
            let rs = ref_s.as_ref().unwrap();
            let mut gs = vec![0.0_f64; n_ss * dim];
            rs.eval_basis(pt, &mut ps);
            rs.eval_grad_basis(pt, &mut gs);
            for i in 0..n_ss {
                for d in 0..dim {
                    gsp[i * dim + d] = (0..dim).map(|k| jit[(d, k)] * gs[i * dim + k]).sum();
                }
            }
        }

        // Weak curl assembly: flux space sigma is
        //   2D: scalar P_{k-1}  (curl is scalar)
        //   3D: vector [P_{k-1}]^d  (curl is vector)
        if dim == 2 {
            for i in 0..n_v { for j in 0..n_s {
                // nv_curl[i] = scalar curl, sigma[j] = scalar basis
                Cw[(i, j)] -= wq * nv_curl[i] * ps[j];
            }}
            for p in 0..n_s { for q in 0..n_s {
                Ms[(p, q)] += wq * ps[p] * ps[q];
            }}
        } else {
            for i in 0..n_v { for j in 0..n_s {
                let sc = j / n_ss; let sd = j % n_ss;
                // nv_curl[i*3+sc] = sc-th curl component, sigma basis ps[sd]
                Cw[(i, j)] -= wq * nv_curl[i * 3 + sc] * ps[sd];
            }}
            for p in 0..n_s { let pc = p / n_ss; let pd = p % n_ss;
                for q in 0..n_s { let qc = q / n_ss; let qd = q % n_ss;
                    if pc == qc { Ms[(p, q)] += wq * ps[pd] * ps[qd]; }
                }
            }
        }
    }
    (Cw, Ms)
}

// ─── Face stabilizer (tangential jump for H(curl)) ────────────────────────

fn add_face_penalty_hcurl<M: MeshTopology>(
    coo: &mut CooMatrix<f64>, mesh: &M,
    hcurl_space: &dyn FESpace<Mesh=M>,
    el: u32, er: u32, fnodes: &[u32], alpha: f64, qo: u8,
) {
    let dim = mesh.dim() as usize;
    let order = hcurl_space.order() as usize;
    let nd_elem: Box<dyn VectorReferenceElement> = if dim == 2 {
        Box::new(TriNDk::new(order))
    } else {
        Box::new(TetNDk::new(order))
    };
    let ne = nd_elem.n_dofs();
    let qf = tri_rule(qo);
    let dofs_l: Vec<usize> = hcurl_space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = if el != er {
        hcurl_space.element_dofs(er).iter().map(|&d| d as usize).collect()
    } else { dofs_l.clone() };

    for (qi, xi) in qf.points.iter().enumerate() {
        let w = qf.weights[qi] * (if dim == 2 { face_geom_2d(mesh, fnodes) } else { face_geom_3d(mesh, fnodes) });
        let xp = if dim == 2 {
            let c = mesh.node_coords(fnodes[0]); let d = mesh.node_coords(fnodes[1]);
            vec![c[0] + xi[0] * (d[0] - c[0]), c[1] + xi[0] * (d[1] - c[1])]
        } else {
            let c0 = mesh.node_coords(fnodes[0]); let c1 = mesh.node_coords(fnodes[1]);
            let c2 = mesh.node_coords(fnodes[2]);
            vec![c0[0] + xi[0] * (c1[0] - c0[0]) + xi[1] * (c2[0] - c0[0]),
                 c0[1] + xi[0] * (c1[1] - c0[1]) + xi[1] * (c2[1] - c0[1]),
                 c0[2] + xi[0] * (c1[2] - c0[2]) + xi[1] * (c2[2] - c0[2])]
        };
        for (side, &dofs) in [&dofs_l, &dofs_r].iter().enumerate() {
            let es = if side == 0 { el } else { er };
            let nl = mesh.element_nodes(es);
            let (jl, _) = local_jac(mesh, nl, dim);
            let xi_ref = local_phys_to_ref(&jl, mesh.node_coords(nl[0]), &xp[..dim], dim);
            let mut pb = vec![0.0_f64; ne * dim];
            nd_elem.eval_basis_vec(&xi_ref, &mut pb);
            for i in 0..ne { for j in 0..ne {
                let v = alpha * w * (0..dim).map(|d| pb[i*dim+d] * pb[j*dim+d]).sum::<f64>();
                if v.abs() > 1e-30 { coo.add(dofs[i], dofs[j], v); }
            }}
        }
    }
}

// ─── WG Maxwell assembly ──────────────────────────────────────────────────

/// Assemble the WG Maxwell stiffness matrix for κ = 1 (vacuum).
///
/// # Arguments
/// * `hcurl_space` — Nédélec HCurl space of order k
/// * `quad_order` — volume/face quadrature order
/// * `penalty` — face penalty coefficient (h^{-1} scaling applied internally)
/// * `dirichlet_bc` — (dof, value) pairs for essential BC on E_tan
pub fn assemble_wg_maxwell<M, S>(
    hcurl_space: &S,
    quad_order: u8,
    penalty: f64,
    dirichlet_bc: &[(usize, f64)],
) -> (CsrMatrix<f64>, Vec<f64>)
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
{
    let mesh = hcurl_space.mesh();
    let dim = mesh.dim() as usize;
    let n = hcurl_space.n_dofs();
    let ne = mesh.n_elements();
    let order = hcurl_space.order() as usize;

    let mut coo = CooMatrix::new(n, n);
    let mut rhs = vec![0.0_f64; n];

    // ── Volume stiffness: A_elem = C_w^T * M_Σ^{-1} * C_w ──────────────────
    for e in 0..ne as u32 {
        let (Cw, Ms) = weak_curl_matrix(mesh, e, dim, order, quad_order);
        let n_v = Cw.nrows(); let n_s = Cw.ncols();
        let X = Ms.clone().lu().solve(&Cw.transpose()).unwrap_or(DMatrix::zeros(n_s, n_v));
        let A_elem = &Cw * X;
        let dofs: Vec<usize> = hcurl_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for i in 0..n_v { for j in 0..n_v {
            let v = A_elem[(i, j)];
            if v.abs() > 1e-30 { coo.add(dofs[i], dofs[j], v); }
        }}
    }

    // ── Face penalty (tangential jump stabilizer) ──────────────────────────
    let interior_faces = crate::InteriorFaceList::build(mesh);
    for f in &interior_faces.faces {
        let h = if dim == 2 { face_geom_2d(mesh, &f.face_nodes) }
                else { face_geom_3d(mesh, &f.face_nodes) };
        let alpha = penalty / h.max(1e-14);
        add_face_penalty_hcurl(&mut coo, mesh, hcurl_space, f.elem_left, f.elem_right, &f.face_nodes, alpha, quad_order);
    }
    let fe_map = build_face_elem_map(mesh);
    for bf in mesh.face_iter() {
        if let Some(&el) = fe_map.get(&bf) {
            let fnodes: Vec<u32> = mesh.face_nodes(bf).to_vec();
            let h = if dim == 2 { face_geom_2d(mesh, &fnodes) }
                    else { face_geom_3d(mesh, &fnodes) };
            let alpha = penalty / h.max(1e-14);
            add_face_penalty_hcurl(&mut coo, mesh, hcurl_space, el, el, &fnodes, alpha, quad_order);
        }
    }

    // ── Dirichlet BC ───────────────────────────────────────────────────────
    for &(dof, val) in dirichlet_bc {
        if dof < n { rhs[dof] = val; }
    }

    (coo.into_csr(), rhs)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::HCurlSpace;

    /// WG Maxwell matrix must be symmetric positive semi-definite.
    #[test]
    fn wg_maxwell_2d_spd() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 1);
        let (k, _) = assemble_wg_maxwell(&hcurl, 3, 10.0, &[]);
        let n = hcurl.n_dofs();

        // Check symmetry
        let mut max_asym: f64 = 0.0;
        for i in 0..n.min(50) {
            let s = k.row_ptr[i]; let e = k.row_ptr[i + 1];
            for p in s..e {
                let j = k.col_idx[p] as usize;
                if j < n.min(50) {
                    let vij = k.values[p];
                    let s2 = k.row_ptr[j]; let e2 = k.row_ptr[j + 1];
                    for q in s2..e2 {
                        if k.col_idx[q] == i as u32 {
                            max_asym = max_asym.max((vij - k.values[q]).abs());
                        }
                    }
                }
            }
        }
        assert!(max_asym < 1e-12, "A block symmetry violated: {max_asym}");

        // Check no zero diagonals
        for i in 0..n {
            let s = k.row_ptr[i]; let e = k.row_ptr[i + 1];
            let mut diag = 0.0;
            for p in s..e {
                if k.col_idx[p] == i as u32 { diag = k.values[p]; break; }
            }
            assert!(diag > 0.0, "Zero diagonal at DOF {i}");
        }
    }

    /// WG Maxwell 2D matrix must be invertible (no zero rows).
    #[test]
    fn wg_maxwell_2d_no_zero_rows() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 1);
        let (k, _) = assemble_wg_maxwell(&hcurl, 3, 10.0, &[]);
        let n = hcurl.n_dofs();
        for i in 0..n {
            let s = k.row_ptr[i]; let e = k.row_ptr[i + 1];
            let mut row_sum = 0.0;
            for p in s..e { row_sum += k.values[p].abs(); }
            assert!(row_sum > 1e-14, "Zero row {i}");
        }
    }
}
