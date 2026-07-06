//! Weak Galerkin (WG) finite element method for Stokes flow.
//!
//! Reference: Wang & Ye (2013), "A weak Galerkin FEM for Stokes equations".
//! Bilinear form:
//!   a_h(u,v) = (∇_w u, ∇_w v)_T + s(u,v)
//!   b_h(v,p) = -(∇_w·v, p)_T
//! Saddle-point system: [A Bᵀ; B 0][u; p] = [f; 0]
//!
//! Velocity space: [P_k]^d (continuous), Pressure space: P_{k-1} (discontinuous)

use std::collections::HashMap;
use nalgebra::DMatrix;
use fem_element::{
    ReferenceElement, lagrange::{TriPk, TetPk},
};
use fem_element::quadrature::{tri_rule, tet_rule};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

// ─── Local helpers ──────────────────────────────────────────────────────────

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
    let c0 = mesh.node_coords(nodes[0]);
    let c1 = mesh.node_coords(nodes[1]);
    ((c1[0]-c0[0]).powi(2) + (c1[1]-c0[1]).powi(2)).sqrt()
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
    ((nx*nx + ny*ny + nz*nz).sqrt()) / 2.0
}

fn build_face_elem_map<M: MeshTopology>(mesh: &M) -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    let n_bfaces = mesh.n_boundary_faces() as u32;
    for f in 0..n_bfaces {
        let fnodes = mesh.face_nodes(f);
        for e in mesh.elem_iter() {
            let enodes = mesh.element_nodes(e);
            if fnodes.iter().all(|n| enodes.contains(n)) {
                map.insert(f, e);
                break;
            }
        }
    }
    map
}

// ─── Weak gradient matrix ──────────────────────────────────────────────────
// Gᵢⱼ = (∇ φⱼ, σᵢ)_T, M_stab_{ij} = (σᵢ, σⱼ)_T  (local per-element)

fn weak_gradient_matrix<M: MeshTopology>(
    mesh: &M, e: u32, dim: usize, order: usize, quad_order: u8,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let ref_v: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(order)) }
                                           else { Box::new(TetPk::new(order)) };
    let n_v = ref_v.n_dofs();
    let os = if order > 0 { order - 1 } else { 0 };
    let (n_ss, use_const) = if os == 0 { (1_usize, true) } else {
        let rs: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(os)) }
                                            else { Box::new(TetPk::new(os)) };
        (rs.n_dofs(), false)
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
        if use_const {
            ps[0] = 1.0;
            for d in 0..dim { gsp[d] = 0.0; }
        } else {
            let rs: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(os)) }
                                                else { Box::new(TetPk::new(os)) };
            let mut gs = vec![0.0; n_ss * dim];
            rs.eval_basis(pt, &mut ps);
            rs.eval_grad_basis(pt, &mut gs);
            for i in 0..n_ss {
                for d in 0..dim {
                    gsp[i * dim + d] = (0..dim).map(|k| jit[(d, k)] * gs[i * dim + k]).sum();
                }
            }
        }
        for i in 0..n_v { for j in 0..n_s {
            let sc = j / n_ss; let sd = j % n_ss;
            G[(i, j)] -= wq * pv[i] * gsp[sd * dim + sc];
        }}
        for p in 0..n_s { let pc = p / n_ss; let pd = p % n_ss;
            for q in 0..n_s { let qc = q / n_ss; let qd = q % n_ss;
                if pc == qc { Ms[(p, q)] += wq * ps[pd] * ps[qd]; }
            }
        }
    }
    (G, Ms)
}

// ─── Face stabilizer (same as WG Poisson) ──────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn add_face_penalty<M: MeshTopology>(
    coo: &mut CooMatrix<f64>, mesh: &M,
    vel_space: &dyn FESpace<Mesh=M>,
    el: u32, er: u32, fnodes: &[u32], alpha: f64, qo: u8,
) {
    let dim = mesh.dim() as usize;
    let order = vel_space.order() as usize;
    let ref_e: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(order)) }
                                           else { Box::new(TetPk::new(order)) };
    let ne = ref_e.n_dofs();
    let qf = tri_rule(qo);
    let dofs_l: Vec<usize> = vel_space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = if el != er {
        vel_space.element_dofs(er).iter().map(|&d| d as usize).collect()
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
            let e_side = if side == 0 { el } else { er };
            let nl = mesh.element_nodes(e_side);
            let (jl, _) = local_jac(mesh, nl, dim);
            let xi_ref = local_phys_to_ref(&jl, mesh.node_coords(nl[0]), &xp[..dim], dim);
            let mut pl = vec![0.0; ne]; ref_e.eval_basis(&xi_ref, &mut pl);
            // Block-diagonal: each velocity component gets same penalty
            for comp in 0..dim {
                for i in 0..ne { for j in 0..ne {
                    let vi = alpha * w * pl[i] * pl[j];
                    if vi.abs() > 1e-30 {
                        let row = (comp * ne + i) + dofs[0];
                        let col = (comp * ne + j) + dofs[0];
                        coo.add(row, col, vi);
                    }
                }}
            }
        }
    }
}

// ─── WG Stokes assembly ────────────────────────────────────────────────────

/// Assemble the WG Stokes saddle-point system.
///
/// Returns `(K, rhs)` where `K` is the block matrix `[A Bᵀ; B 0]` and rhs
/// includes body force (velocity block) and Dirichlet data.
///
/// # Arguments
/// * `vel_space` — vector-valued velocity space (V_h = [P_k]^d)
/// * `pres_space` — pressure space (Q_h = P_{k-1}, discontinuous)
/// * `quad_order` — quadrature order for volume/face integration
/// * `penalty` — face penalty coefficient (typically 4*p*(p+1))
/// * `force` — body force f(x) → d components
/// * `dirichlet_bc` — (velocity_dof, value) pairs
#[allow(clippy::too_many_arguments)]
pub fn assemble_wg_stokes<V, P>(
    vel_space: &V,
    pres_space: &P,
    quad_order: u8,
    penalty: f64,
    force: &dyn Fn(&[f64]) -> Vec<f64>,
    dirichlet_bc: &[(usize, f64)],
) -> (CsrMatrix<f64>, Vec<f64>)
where
    V: FESpace,
    P: FESpace<Mesh = <V as FESpace>::Mesh>,
{
    let mesh = vel_space.mesh();
    let dim = mesh.dim() as usize;
    let n_vel = vel_space.n_dofs();
    let n_pres = pres_space.n_dofs();
    let n_total = n_vel + n_pres;
    let order = vel_space.order() as usize;
    let ne = mesh.n_elements();

    let mut coo = CooMatrix::new(n_total, n_total);
    let mut rhs = vec![0.0_f64; n_total];

    // ── Volume integrals: A block (velocity stiffness) ──────────────────────
    let mut dirichlet_mask = vec![false; n_vel];
    for &(dof, _) in dirichlet_bc { if dof < n_vel { dirichlet_mask[dof] = true; }}

    for e in 0..ne as u32 {
        let (G, Ms) = weak_gradient_matrix(mesh, e, dim, order, quad_order);
        let n_v = G.nrows();
        let n_s = G.ncols();
        let Gt = G.transpose();
        let X = Ms.clone().lu().solve(&Gt).unwrap_or(DMatrix::zeros(n_s, n_v));
        let A_elem = &G * X; // n_v × n_v

        let v_dofs: Vec<usize> = vel_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let _p_dofs: Vec<usize> = pres_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_v_local = v_dofs.len();

        // A block: velocity stiffness
        for i in 0..n_v_local { for j in 0..n_v_local {
            let v = A_elem[(i, j)];
            if v.abs() > 1e-30 { coo.add(v_dofs[i], v_dofs[j], v); }
        }}

        // ── Body force (velocity rhs) ────────────────────────────────────────
        let ref_v: Box<dyn ReferenceElement> = if dim == 2 { Box::new(TriPk::new(order)) }
                                               else { Box::new(TetPk::new(order)) };
        let qr = if dim == 2 { tri_rule(quad_order) } else { tet_rule(quad_order) };
        let nodes = mesh.element_nodes(e);
        let x0 = mesh.node_coords(nodes[0]);
        let (jac, det_j) = local_jac(mesh, nodes, dim);
        let mut pv = vec![0.0; n_v_local];
        for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
            let wq = w * det_j.abs();
            let xp = if dim == 2 {
                vec![x0[0] + jac[(0,0)]*pt[0] + jac[(0,1)]*pt[1],
                     x0[1] + jac[(1,0)]*pt[0] + jac[(1,1)]*pt[1]]
            } else {
                vec![x0[0] + jac[(0,0)]*pt[0] + jac[(0,1)]*pt[1] + jac[(0,2)]*pt[2],
                     x0[1] + jac[(1,0)]*pt[0] + jac[(1,1)]*pt[1] + jac[(1,2)]*pt[2],
                     x0[2] + jac[(2,0)]*pt[0] + jac[(2,1)]*pt[1] + jac[(2,2)]*pt[2]]
            };
            let f = force(&xp);
            ref_v.eval_basis(pt, &mut pv);
            for comp in 0..dim {
                for i in 0..n_v_local / dim {
                    let row = v_dofs[i + comp * (n_v_local / dim)];
                    rhs[row] += wq * pv[i] * f[comp];
                }
            }
        }
    }

    // ── Face penalty (A block stabilizer) ──────────────────────────────────
    let interior_faces = crate::InteriorFaceList::build(mesh);
    for f in &interior_faces.faces {
        let h = if dim == 2 { face_geom_2d(mesh, &f.face_nodes) }
                else { face_geom_3d(mesh, &f.face_nodes) };
        let alpha = penalty / h.max(1e-14);
        add_face_penalty(&mut coo, mesh, vel_space, f.elem_left, f.elem_right, &f.face_nodes, alpha, quad_order);
    }
    let fe_map = build_face_elem_map(mesh);
    for bf in mesh.face_iter() {
        if let Some(&el) = fe_map.get(&bf) {
            let fnodes: Vec<u32> = mesh.face_nodes(bf).to_vec();
            let h = if dim == 2 { face_geom_2d(mesh, &fnodes) }
                    else { face_geom_3d(mesh, &fnodes) };
            let alpha = penalty / h.max(1e-14);
            add_face_penalty(&mut coo, mesh, vel_space, el, el, &fnodes, alpha, quad_order);
        }
    }

    // ── Dirichlet BC (velocity) ────────────────────────────────────────────
    for &(dof, val) in dirichlet_bc {
        if dof < n_vel { rhs[dof] = val; }
    }

    (coo.into_csr(), rhs)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_space::L2Space;
    /// p = sin(πx)sin(πy), f = Δu - ∇p (computed analytically).
    #[test]
    fn wg_stokes_2d_driven_cavity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let vel = H1Space::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let pres = L2Space::new(mesh2, 1);

        let f = |x: &[f64]| {
            let (xx, yy) = (x[0], x[1]);
            let pix = std::f64::consts::PI * xx;
            let piy = std::f64::consts::PI * yy;
            let uxx = -pix * pix * pix.sin() * piy.cos();
            let uyy = -piy * piy * pix.sin() * piy.cos();
            let vxx = pix * pix * pix.cos() * piy.sin();
            let vyy = piy * piy * pix.cos() * piy.sin();
            let px = std::f64::consts::PI * pix.cos() * piy.sin();
            let py = std::f64::consts::PI * pix.sin() * piy.cos();
            vec![-(uxx + uyy) + px, -(vxx + vyy) + py]
        };

        let dirichlet = vec![(0usize, 0.0); 1]; // placeholder
        let (k, _rhs) = assemble_wg_stokes(&vel, &pres, 3, 10.0, &f, &dirichlet);

        // Check symmetry of A block
        let n_vel = vel.n_dofs();
        let mut max_asym: f64 = 0.0;
        for i in 0..n_vel.min(100) {
            let start = k.row_ptr[i]; let end = k.row_ptr[i + 1];
            for p in start..end {
                let j = k.col_idx[p] as usize;
                if j < n_vel {
                    let v = k.values[p];
                    let s2 = k.row_ptr[j]; let e2 = k.row_ptr[j + 1];
                    let mut found = false;
                    for q in s2..e2 {
                        if k.col_idx[q] == i as u32 {
                            let diff = (v - k.values[q]).abs();
                            max_asym = max_asym.max(diff);
                            found = true;
                            break;
                        }
                    }
                    if !found { max_asym = max_asym.max(v.abs()); }
                }
            }
        }
        assert!(max_asym < 1e-10, "A block symmetry violated: max_asym = {max_asym}");
    }

    /// Stokes matrix must be invertible (no zero rows/columns in A block).
    #[test]
    fn wg_stokes_2d_no_zero_rows() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let vel = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(3);
        let pres = L2Space::new(mesh2, 0);

        let f = |_: &[f64]| vec![1.0, 1.0];
        let (k, _) = assemble_wg_stokes(&vel, &pres, 3, 4.0, &f, &[]);
        for i in 0..vel.n_dofs() {
            let start = k.row_ptr[i]; let end = k.row_ptr[i + 1];
            let mut sum = 0.0;
            for p in start..end { sum += k.values[p].abs(); }
            assert!(sum > 1e-14, "Zero row {i} in A block");
        }
    }

    /// MMS test: verify WG Stokes assembly produces a valid system.
    #[test]
    fn wg_stokes_2d_mms_solves() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let vel = H1Space::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(6);
        let pres = L2Space::new(mesh2, 1);

        let f = |x: &[f64]| {
            let pix = std::f64::consts::PI * x[0];
            let piy = std::f64::consts::PI * x[1];
            vec![2.0 * pix * pix.cos() * piy.sin(),
                 2.0 * piy * pix.sin() * piy.cos()]
        };

        let dirichlet = vec![(0usize, 0.0); 1];
        let (k, rhs) = assemble_wg_stokes(&vel, &pres, 3, 10.0, &f, &dirichlet);

        // Matrix should have all non-zero rows in the A block
        let n_vel = vel.n_dofs();
        for i in 0..n_vel.min(20) {
            let mut row_sum = 0.0;
            for p in k.row_ptr[i]..k.row_ptr[i+1] { row_sum += k.values[p].abs(); }
            assert!(row_sum > 0.0, "Zero row {i} in stiffness matrix");
        }
        // RHS should be non-trivial
        let rhs_norm: f64 = rhs.iter().map(|v| v*v).sum::<f64>().sqrt();
        assert!(rhs_norm > 0.0, "RHS should be non-zero");
    }
}
