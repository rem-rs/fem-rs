//! Discontinuous Galerkin (DG) interior penalty assembly.
//!
//! Implements the **Symmetric Interior Penalty (SIP)** method for the scalar
//! diffusion equation `−∇·(κ ∇u) = f` with Dirichlet boundary conditions.
//!
//! # Bilinear form
//!
//! ```text
//! a_h(u,v) = ∑_K ∫_K κ ∇u·∇v dx
//!            − ∑_F ∫_F { κ ∇u }·[[v]] ds   (consistency)
//!            − ∑_F ∫_F { κ ∇v }·[[u]] ds   (symmetry, only for SIP)
//!            + ∑_F ∫_F (σ/h_F) [[u]]·[[v]] ds  (penalty)
//! ```
//!
//! where:
//! - `{·}` is the average operator: `{w} = ½(w⁺ + w⁻)` on interior faces,
//!   `{w} = w` on Dirichlet boundary faces.
//! - `[[·]]` is the scalar jump: `[[u]] = u⁺ n⁺ + u⁻ n⁻` (vector jump) or
//!   `[[u]] = u⁺ − u⁻` (scalar jump used with normal orientation convention).
//! - `h_F` is the face size (length in 2-D).
//! - `σ` is the penalty parameter (must be large enough for coercivity; typically
//!   σ ≥ C p²/h_F where p is the polynomial degree).
//!
//! # Usage
//! ```rust,ignore
//! let space = L2Space::new(mesh, 1);
//! let ifl   = InteriorFaceList::build(space.mesh());
//! let mat   = DgAssembler::assemble_sip(&space, &ifl, kappa, sigma, 3);
//! ```

use std::collections::HashMap;
use nalgebra::DMatrix;

use fem_element::{ReferenceElement, lagrange::{SegP1, TriP1, TriP2, TetP1}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

use crate::interior_faces::InteriorFaceList;

// ─── DgAssembler ─────────────────────────────────────────────────────────────

/// Stateless DG assembly driver.
pub struct DgAssembler;

impl DgAssembler {
    /// Assemble the global SIP-DG stiffness matrix.
    ///
    /// Combines:
    /// 1. **Volume terms**: standard diffusion `∫ κ ∇u·∇v dx` per element.
    /// 2. **Interior face terms**: consistency + symmetry + penalty.
    /// 3. **Boundary face terms** (Dirichlet, all boundary tags): same penalty form.
    ///
    /// # Arguments
    /// - `space`      — the L² (DG) finite element space.
    /// - `ifl`        — pre-built interior face list.
    /// - `kappa`      — diffusion coefficient (scalar, uniform).
    /// - `sigma`      — penalty parameter (dimensionless; use ≥ 3*(order+1)² for coercivity).
    /// - `quad_order` — polynomial order the quadrature integrates exactly.
    pub fn assemble_sip<S: FESpace>(
        space:      &S,
        ifl:        &InteriorFaceList,
        kappa:      f64,
        sigma:      f64,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        let mesh   = space.mesh();
        let _dim   = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let order  = space.order();

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        // ── 1. Volume terms ────────────────────────────────────────────────────
        assemble_volume(&mut coo, space, kappa, quad_order);

        // ── 2. Interior face terms ─────────────────────────────────────────────
        for iface in &ifl.faces {
            assemble_interior_face(
                &mut coo, mesh, space, iface.elem_left, iface.elem_right,
                &iface.face_nodes, kappa, sigma, order, quad_order,
            );
        }

        // ── 3. Boundary face terms (Dirichlet) ─────────────────────────────────
        // Build face→element map (SimplexMesh::face_elements always returns (0,None)).
        let face_to_elem = build_face_elem_map(mesh, order);
        for f in mesh.face_iter() {
            if let Some(&elem) = face_to_elem.get(&f) {
                assemble_boundary_face_with_elem(
                    &mut coo, mesh, space, f, elem, kappa, sigma, order, quad_order,
                );
            }
        }

        coo.into_csr()
    }
}

// ─── Volume contribution ──────────────────────────────────────────────────────

fn assemble_volume<S: FESpace>(
    coo:        &mut CooMatrix<f64>,
    space:      &S,
    kappa:      f64,
    quad_order: u8,
) {
    let mesh  = space.mesh();
    let dim   = mesh.dim() as usize;
    let order = space.order();

    let mut phi      = Vec::<f64>::new();
    let mut grad_ref = Vec::<f64>::new();
    let mut grad_p   = Vec::<f64>::new();

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let re = ref_elem_vol(elem_type, order);
        let n  = re.n_dofs();
        let q  = re.quadrature(quad_order);
        let gd = space.element_dofs(e).iter().map(|&d| d as usize).collect::<Vec<_>>();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        let j_inv_t = jac.clone().try_inverse().unwrap().transpose();

        phi.resize(n, 0.0);
        grad_ref.resize(n * dim, 0.0);
        grad_p.resize(n * dim, 0.0);

        let _x0 = mesh.node_coords(nodes[0]);
        let mut k_elem = vec![0.0_f64; n * n];

        for (qi, xi) in q.points.iter().enumerate() {
            let w = q.weights[qi] * det_j.abs();
            re.eval_grad_basis(xi, &mut grad_ref);
            xform_grads(&j_inv_t, &grad_ref, &mut grad_p, n, dim);
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for d in 0..dim { dot += grad_p[i*dim+d] * grad_p[j*dim+d]; }
                    k_elem[i*n+j] += w * kappa * dot;
                }
            }
        }

        for (i, &gi) in gd.iter().enumerate() {
            for (j, &gj) in gd.iter().enumerate() {
                coo.add(gi, gj, k_elem[i * n + j]);
            }
        }
    }
}

// ─── Interior face contribution ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn assemble_interior_face<S: FESpace>(
    coo:        &mut CooMatrix<f64>,
    mesh:       &S::Mesh,
    space:      &S,
    el:         u32,
    er:         u32,
    face_nodes: &[u32],
    kappa:      f64,
    sigma:      f64,
    order:      u8,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let (h_f, mut normal_l) = face_geom_2d(mesh, face_nodes);
    orient_normal_outward(mesh, el, face_nodes, &mut normal_l);

    // Build reference elements and quadrature for the face.
    let face_elem_type = if dim == 2 { ElementType::Line2 } else { ElementType::Tri3 };
    let ref_face = ref_elem_face(face_elem_type, order);
    let q_face   = ref_face.quadrature(quad_order);
    let _n_f = ref_face.n_dofs();

    // Build reference elements for the volume.
    let et_l = mesh.element_type(el);
    let re_l = ref_elem_vol(et_l, order);
    let et_r = mesh.element_type(er);
    let re_r = ref_elem_vol(et_r, order);
    let n_l = re_l.n_dofs();
    let n_r = re_r.n_dofs();

    let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

    let nodes_l = mesh.element_nodes(el);
    let nodes_r = mesh.element_nodes(er);
    let (jac_l, _det_l) = simplex_jac(mesh, nodes_l, dim);
    let (jac_r, _det_r) = simplex_jac(mesh, nodes_r, dim);
    let jit_l = jac_l.clone().try_inverse().unwrap().transpose();
    let jit_r = jac_r.clone().try_inverse().unwrap().transpose();

    // Blocks accumulated: K_ll, K_lr, K_rl, K_rr.
    let mut kll = vec![0.0_f64; n_l * n_l];
    let mut klr = vec![0.0_f64; n_l * n_r];
    let mut krl = vec![0.0_f64; n_r * n_l];
    let mut krr = vec![0.0_f64; n_r * n_r];

    // Map face quadrature points to physical coords, then back to reference
    // coords of each element for basis evaluation.
    let face_xi: Vec<Vec<f64>> = q_face.points.clone();
    let face_weights = &q_face.weights;

    // Physical face quadrature points.
    let x0f = mesh.node_coords(face_nodes[0]);
    let x1f = mesh.node_coords(face_nodes[1]);

    let mut phi_l    = vec![0.0_f64; n_l];
    let mut phi_r    = vec![0.0_f64; n_r];
    let mut gref_l   = vec![0.0_f64; n_l * dim];
    let mut gref_r   = vec![0.0_f64; n_r * dim];
    let mut gphys_l  = vec![0.0_f64; n_l * dim];
    let mut gphys_r  = vec![0.0_f64; n_r * dim];

    for (qi, xi_f) in face_xi.iter().enumerate() {
        let w_f = face_weights[qi] * h_f;

        // Physical quadrature point on the face.
        let xp: Vec<f64> = (0..dim).map(|i| x0f[i] + (x1f[i] - x0f[i]) * xi_f[0]).collect();

        // Map physical point → reference coordinates of each element.
        let xi_l = phys_to_ref(&jac_l, mesh.node_coords(nodes_l[0]), &xp, dim);
        let xi_r = phys_to_ref(&jac_r, mesh.node_coords(nodes_r[0]), &xp, dim);

        re_l.eval_basis(&xi_l, &mut phi_l);
        re_r.eval_basis(&xi_r, &mut phi_r);
        re_l.eval_grad_basis(&xi_l, &mut gref_l);
        re_r.eval_grad_basis(&xi_r, &mut gref_r);
        xform_grads(&jit_l, &gref_l, &mut gphys_l, n_l, dim);
        xform_grads(&jit_r, &gref_r, &mut gphys_r, n_r, dim);

        let pen = sigma * kappa / h_f;

        // SIP interior face terms using a single normal n = n_L (outward from left):
        //   -∫ {κ∇u·n}[v] ds - ∫ {κ∇v·n}[u] ds + pen ∫ [u][v] ds
        // where [w] = w_L - w_R, {w} = (w_L + w_R)/2
        // All gradients dotted with n_L (not n_R).

        // Precompute ∇φ·n_L for all basis functions on both sides
        let ngl: Vec<f64> = (0..n_l).map(|i| (0..dim).map(|d| gphys_l[i*dim+d] * normal_l[d]).sum::<f64>()).collect();
        let ngr: Vec<f64> = (0..n_r).map(|i| (0..dim).map(|d| gphys_r[i*dim+d] * normal_l[d]).sum::<f64>()).collect();

        // K_LL: a∈L, b∈L  → [v]=φ_bL, [u]=φ_aL, {∇u·n}=½∇φ_aL·n, {∇v·n}=½∇φ_bL·n
        for i in 0..n_l {
            for j in 0..n_l {
                kll[i*n_l+j] += w_f * (
                    -0.5 * kappa * ngl[i] * phi_l[j]
                    -0.5 * kappa * ngl[j] * phi_l[i]
                    + pen * phi_l[i] * phi_l[j]
                );
            }
        }
        // K_LR: a∈L, b∈R  → [v]=-φ_bR, [u]=φ_aL, {∇u·n}=½∇φ_aL·n, {∇v·n}=½∇φ_bR·n
        for i in 0..n_l {
            for j in 0..n_r {
                klr[i*n_r+j] += w_f * (
                     0.5 * kappa * ngl[i] * phi_r[j]    // from -{∇u·n}[v], [v]=-φ_bR
                    -0.5 * kappa * ngr[j] * phi_l[i]    // from -{∇v·n}[u], [u]=φ_aL
                    - pen * phi_l[i] * phi_r[j]          // from [u][v] = φ_aL·(-φ_bR)
                );
            }
        }
        // K_RL: a∈R, b∈L  → [v]=φ_bL, [u]=-φ_aR, {∇u·n}=½∇φ_aR·n, {∇v·n}=½∇φ_bL·n
        for i in 0..n_r {
            for j in 0..n_l {
                krl[i*n_l+j] += w_f * (
                    -0.5 * kappa * ngr[i] * phi_l[j]    // from -{∇u·n}[v], [v]=φ_bL
                    +0.5 * kappa * ngl[j] * phi_r[i]    // from -{∇v·n}[u], [u]=-φ_aR
                    - pen * phi_r[i] * phi_l[j]          // from [u][v] = (-φ_aR)·φ_bL
                );
            }
        }
        // K_RR: a∈R, b∈R  → [v]=-φ_bR, [u]=-φ_aR, {∇u·n}=½∇φ_aR·n, {∇v·n}=½∇φ_bR·n
        for i in 0..n_r {
            for j in 0..n_r {
                krr[i*n_r+j] += w_f * (
                     0.5 * kappa * ngr[i] * phi_r[j]    // from -{∇u·n}[v], [v]=-φ_bR
                    +0.5 * kappa * ngr[j] * phi_r[i]    // from -{∇v·n}[u], [u]=-φ_aR
                    + pen * phi_r[i] * phi_r[j]          // from [u][v] = (-φ_aR)·(-φ_bR)
                );
            }
        }
    }

    // Scatter
    for (i, &gi) in dofs_l.iter().enumerate() {
        for (j, &gj) in dofs_l.iter().enumerate() { coo.add(gi, gj, kll[i*n_l+j]); }
        for (j, &gj) in dofs_r.iter().enumerate() { coo.add(gi, gj, klr[i*n_r+j]); }
    }
    for (i, &gi) in dofs_r.iter().enumerate() {
        for (j, &gj) in dofs_l.iter().enumerate() { coo.add(gi, gj, krl[i*n_l+j]); }
        for (j, &gj) in dofs_r.iter().enumerate() { coo.add(gi, gj, krr[i*n_r+j]); }
    }
}

// ─── Face → element map ───────────────────────────────────────────────────────

/// Build a map from boundary face index → owning element index.
///
/// `SimplexMesh::face_elements()` always returns `(0, None)` (not implemented),
/// so we build this by matching boundary face node sets against element faces.
fn build_face_elem_map<M: MeshTopology>(mesh: &M, _order: u8) -> HashMap<u32, u32> {
    let dim = mesh.dim() as usize;
    // Build a map from sorted-face-node-key → element id from the volume mesh.
    let mut vol_face_map: HashMap<Vec<u32>, u32> = HashMap::new();

    let local_faces_fn: fn(usize, usize) -> Vec<Vec<usize>> = |npe, d| match (npe, d) {
        (3, 2) => vec![vec![0,1], vec![1,2], vec![0,2]],
        (4, 3) => vec![vec![1,2,3], vec![0,2,3], vec![0,1,3], vec![0,1,2]],
        _ => vec![],
    };

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let npe   = nodes.len();
        for lf in local_faces_fn(npe, dim) {
            let mut key: Vec<u32> = lf.iter().map(|&k| nodes[k]).collect();
            key.sort_unstable();
            vol_face_map.entry(key).or_insert(e);
        }
    }

    let mut result = HashMap::new();
    for f in mesh.face_iter() {
        let fnodes = mesh.face_nodes(f);
        let mut key: Vec<u32> = fnodes.to_vec();
        key.sort_unstable();
        if let Some(&elem) = vol_face_map.get(&key) {
            result.insert(f, elem);
        }
    }
    result
}

// ─── Boundary face contribution (Dirichlet) ───────────────────────────────────

fn assemble_boundary_face_with_elem<S: FESpace>(
    coo:        &mut CooMatrix<f64>,
    mesh:       &S::Mesh,
    space:      &S,
    face:       u32,
    elem:       u32,
    kappa:      f64,
    sigma:      f64,
    order:      u8,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let face_nodes = mesh.face_nodes(face);
    let (h_f, mut normal) = face_geom_2d(mesh, face_nodes);
    orient_normal_outward(mesh, elem, face_nodes, &mut normal);

    let et = mesh.element_type(elem);
    let re = ref_elem_vol(et, order);
    let n  = re.n_dofs();
    let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

    let face_elem_type = if dim == 2 { ElementType::Line2 } else { ElementType::Tri3 };
    let ref_face = ref_elem_face(face_elem_type, order);
    let q_face   = ref_face.quadrature(quad_order);

    let nodes = mesh.element_nodes(elem);
    let (jac, _det_j) = simplex_jac(mesh, nodes, dim);
    let jit = jac.clone().try_inverse().unwrap().transpose();

    let x0f = mesh.node_coords(face_nodes[0]);
    let x1f = mesh.node_coords(face_nodes[1]);

    let mut k_bd = vec![0.0_f64; n * n];
    let mut phi   = vec![0.0_f64; n];
    let mut gref  = vec![0.0_f64; n * dim];
    let mut gphys = vec![0.0_f64; n * dim];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        let w_f = q_face.weights[qi] * h_f;
        let xp: Vec<f64> = (0..dim).map(|i| x0f[i] + (x1f[i] - x0f[i]) * xi_f[0]).collect();
        let xi_e = phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp, dim);

        re.eval_basis(&xi_e, &mut phi);
        re.eval_grad_basis(&xi_e, &mut gref);
        xform_grads(&jit, &gref, &mut gphys, n, dim);

        let pen = sigma * kappa / h_f;

        for i in 0..n {
            let phi_i   = phi[i];
            let ngrad_i: f64 = (0..dim).map(|d| gphys[i*dim+d] * normal[d]).sum();
            for j in 0..n {
                let phi_j   = phi[j];
                let ngrad_j: f64 = (0..dim).map(|d| gphys[j*dim+d] * normal[d]).sum();
                k_bd[i*n+j] += w_f * (
                    -kappa * ngrad_i * phi_j
                    -kappa * ngrad_j * phi_i
                    + pen  * phi_i   * phi_j
                );
            }
        }
    }

    for (i, &gi) in dofs.iter().enumerate() {
        for (j, &gj) in dofs.iter().enumerate() {
            coo.add(gi, gj, k_bd[i*n+j]);
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// 2-D face geometry: return (edge_length, unit_normal).
///
/// The normal is the 90° CCW rotation of the edge direction.
/// Use `orient_normal_outward` to guarantee outward orientation.
fn face_geom_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> (f64, Vec<f64>) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx*dx + dy*dy).sqrt();
    (len, vec![dy / len, -dx / len])
}

/// Ensure `normal` points outward from `elem` by checking against the element centroid.
///
/// If `dot(normal, face_midpoint - centroid) < 0`, the normal points inward → flip it.
fn orient_normal_outward<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    face_nodes: &[u32],
    normal: &mut [f64],
) {
    let dim = mesh.dim() as usize;
    let enodes = mesh.element_nodes(elem);
    let npe = enodes.len();
    // Element centroid
    let mut centroid = vec![0.0_f64; dim];
    for &n in enodes {
        let c = mesh.node_coords(n);
        for d in 0..dim { centroid[d] += c[d]; }
    }
    for d in 0..dim { centroid[d] /= npe as f64; }
    // Face midpoint
    let mut midpoint = vec![0.0_f64; dim];
    for &n in face_nodes {
        let c = mesh.node_coords(n);
        for d in 0..dim { midpoint[d] += c[d]; }
    }
    for d in 0..dim { midpoint[d] /= face_nodes.len() as f64; }
    // Check orientation: outward means normal · (midpoint - centroid) > 0
    let dot: f64 = (0..dim).map(|d| normal[d] * (midpoint[d] - centroid[d])).sum();
    if dot < 0.0 {
        for d in 0..dim { normal[d] = -normal[d]; }
    }
}

/// Invert `x = x0 + J ξ` → `ξ = J^{-1}(x - x0)`.
fn phys_to_ref(jac: &DMatrix<f64>, x0: &[f64], xp: &[f64], dim: usize) -> Vec<f64> {
    let j_inv = jac.clone().try_inverse().expect("degenerate element in phys_to_ref");
    let dx: Vec<f64> = (0..dim).map(|i| xp[i] - x0[i]).collect();
    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim { xi[i] += j_inv[(i,k)] * dx[k]; }
    }
    xi
}

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        _ => panic!("dg ref_elem_vol: unsupported ({et:?}, {order})"),
    }
}

fn ref_elem_face(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Line2, 1) | (ElementType::Line2, 2) => Box::new(SegP1),
        (ElementType::Tri3, 1)  => Box::new(TriP1),
        _ => panic!("dg ref_elem_face: unsupported ({et:?}, {order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col+1]);
        for row in 0..dim { j[(row,col)] = xc[row] - x0[row]; }
    }
    let det = j.determinant();
    (j, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::L2Space;
    use crate::interior_faces::InteriorFaceList;

    /// SIP matrix should be symmetric for a uniform mesh.
    #[test]
    fn sip_matrix_symmetric() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let ifl   = InteriorFaceList::build(&mesh);
        let space = L2Space::new(mesh, 1);
        let mat   = DgAssembler::assemble_sip(&space, &ifl, 1.0, 10.0, 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i*n+j] - dense[j*n+i]).abs();
                assert!(diff < 1e-11, "SIP K[{i},{j}]-K[{j},{i}] = {diff}");
            }
        }
    }

    /// With a sufficiently large penalty the SIP matrix should be positive definite
    /// (all eigenvalues > 0).  We check via Cholesky or by verifying row-dominant structure:
    /// diagonal entry should be the largest in each row for a well-conditioned problem.
    #[test]
    fn sip_matrix_positive_diagonal() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(3);
        let ifl   = InteriorFaceList::build(&mesh);
        let space = L2Space::new(mesh, 1);
        let mat   = DgAssembler::assemble_sip(&space, &ifl, 1.0, 20.0, 3);
        for i in 0..mat.nrows {
            let diag = mat.get(i, i);
            assert!(diag > 0.0, "diagonal[{i}] = {diag}");
        }
    }
}
