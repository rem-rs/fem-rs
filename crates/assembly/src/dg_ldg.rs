//! LDG (Local Discontinuous Galerkin) method for diffusion.
//!
//! Bilinear form (interior penalty formulation):
//!
//! ```text
//! a_h(u,v) = Σ_K ∫_K κ ∇u·∇v dx
//!          - Σ_F ∫_F {κ ∇u·n} [v] ds          (consistency)
//!          - Σ_F ∫_F [u] {κ ∇v·n} ds          (symmetry)
//!          + Σ_F (η / h_F) ∫_F [u][v] ds      (stabilization)
//! ```
//!
//! This is structurally identical to SIP-DG but uses a smaller
//! stabilization parameter η ≈ O(1) (vs σ ≥ O(p²) for SIP).
//! The `β` switching parameter (upwind factor on the numerical flux)
//! is also supported: `β = 0` (central) or `β = ±½` (alternating).

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use crate::interior_faces::{InteriorFace, InteriorFaceList};

// ─── helpers (cloned from dg.rs to avoid coupling) ──────────────────────────

/// Compute face length (2D) or area (3D) and the unit normal pointing
/// outward from the reference side.
fn face_geom<M: MeshTopology>(
    mesh: &M,
    face_nodes: &[u32],
) -> (f64, Vec<f64>) {
    let d = mesh.dim() as usize;
    if d == 2 {
        let a = mesh.node_coords(face_nodes[0]);
        let b = mesh.node_coords(face_nodes[1]);
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let h = (dx * dx + dy * dy).sqrt().max(1e-30);
        let nx = -dy / h; // outward normal (rotated CCW)
        let ny = dx / h;
        (h, vec![nx, ny])
    } else {
        // 3-D face centroid (area-weighted)
        let a = mesh.node_coords(face_nodes[0]);
        let b = mesh.node_coords(face_nodes[1]);
        let c = mesh.node_coords(face_nodes[2]);
        let v1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let v2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let cr = [
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0],
        ];
        let area = 0.5 * (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt().max(1e-30);
        let nrm = (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt().max(1e-30);
        (area, vec![cr[0]/nrm, cr[1]/nrm, cr[2]/nrm])
    }
}

fn ref_shape_grads(dim: usize, npe: usize, xi: &[f64]) -> Vec<f64> {
    if dim == 2 && npe == 3 {
        // P1 triangle: ∇φ_ref
        vec![-1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
    } else if dim == 3 && npe == 4 {
        // P1 tet: ∇φ_ref
        vec![-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    } else {
        panic!("ref_shape_grads: unsupported (dim={dim}, npe={npe})");
    }
}

fn ref_shape_vals(dim: usize, npe: usize, xi: &[f64]) -> Vec<f64> {
    if dim == 2 && npe == 3 {
        let (x, y) = (xi[0], xi[1]);
        vec![1.0 - x - y, x, y]
    } else if dim == 3 && npe == 4 {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        vec![1.0 - x - y - z, x, y, z]
    } else {
        panic!("ref_shape_vals: unsupported (dim={dim}, npe={npe})");
    }
}

/// Physical gradient from reference gradient and Jacobian inverse.
fn phys_grad(j_inv: &[f64], grad_ref: &[f64], n_dofs: usize, dim: usize) -> Vec<f64> {
    let mut g = vec![0.0; n_dofs * dim];
    for i in 0..n_dofs {
        for d in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += j_inv[d * dim + k] * grad_ref[i * dim + k]; }
            g[i * dim + d] = s;
        }
    }
    g
}

/// Jacobian matrix (dim x dim) and its inverse for affine elements.
fn elem_jacobian<M: MeshTopology>(mesh: &M, elem: u32, dim: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let nodes = mesh.element_nodes(elem);
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = vec![0.0; dim * dim];
    for col in 0..dim {
        let xn = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim { j[row * dim + col] = xn[row] - x0[row]; }
    }
    let det = if dim == 2 {
        j[0] * j[3] - j[1] * j[2]
    } else {
        j[0]*(j[4]*j[8]-j[5]*j[7]) - j[1]*(j[3]*j[8]-j[5]*j[6]) + j[2]*(j[3]*j[7]-j[4]*j[6])
    };
    let inv_det = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
    let mut j_inv = vec![0.0; dim * dim];
    if dim == 2 {
        j_inv[0] = j[3] * inv_det; j_inv[1] = -j[1] * inv_det;
        j_inv[2] = -j[2] * inv_det; j_inv[3] = j[0] * inv_det;
    } else {
        let id = inv_det;
        j_inv[0] = (j[4]*j[8]-j[5]*j[7])*id; j_inv[1] = (j[2]*j[7]-j[1]*j[8])*id; j_inv[2] = (j[1]*j[5]-j[2]*j[4])*id;
        j_inv[3] = (j[5]*j[6]-j[3]*j[8])*id; j_inv[4] = (j[0]*j[8]-j[2]*j[6])*id; j_inv[5] = (j[2]*j[3]-j[0]*j[5])*id;
        j_inv[6] = (j[3]*j[7]-j[4]*j[6])*id; j_inv[7] = (j[1]*j[6]-j[0]*j[7])*id; j_inv[8] = (j[0]*j[4]-j[1]*j[3])*id;
    }
    (j, j_inv, det)
}

// ─── LDG Assembler ───────────────────────────────────────────────────────────

/// Assemble the LDG diffusion matrix.
///
/// # Arguments
/// * `space`    — FE space (must support `space_type() == L2` for DG)
/// * `ifl`      — precomputed `InteriorFaceList`
/// * `kappa`    — diffusion coefficient
/// * `eta`      — LDG stabilization parameter (O(1), e.g. 1.0–10.0)
/// * `beta`     — LDG upwind switching parameter (0.0 = central, ±0.5 = alternating)
/// * `quad_order` — quadrature order for volume term
pub fn assemble_ldg<S: FESpace + Sync>(
    space: &S,
    ifl: &InteriorFaceList,
    kappa: f64,
    eta: f64,
    beta: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mesh: &S::Mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let dim = mesh.dim() as usize;
    let ne = mesh.n_elements();
    let npe = if ne > 0 { mesh.element_nodes(0u32).len() } else { 0 };

    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    // ── 1. Volume term: ∫ κ ∇u·∇v dx ────────────────────────────────────
    // Reuse the standard DiffusionIntegrator via assemble_volume
    assemble_volume(space, mesh, kappa, quad_order, &mut coo);

    // ── 2. Interior face terms ───────────────────────────────────────────
    let q_rule = if dim == 2 {
        vec![1.0 / 3.0, 1.0 / 3.0] // single QP at centroid, weight = 1 (for face)
    } else {
        vec![0.25, 0.25, 0.25]
    };
    let ref_grad = ref_shape_grads(dim, npe, &q_rule);

    for face in &ifl.faces {
        let fnodes = &face.face_nodes;
        let e_l = face.elem_left;
        let e_r = face.elem_right;

        let (h_f, n_l) = face_geom(mesh, fnodes);
        let pen = eta / h_f;

        // Normal is already oriented outward from e_l's perspective
        let n_r: Vec<f64> = n_l.iter().map(|&x| -x).collect();

        // Jacobian for both elements
        let (_, j_inv_l, det_l) = elem_jacobian(mesh, e_l, dim);
        let (_, j_inv_r, det_r) = elem_jacobian(mesh, e_r, dim);

        let vol_l = if dim == 2 { 0.5 * det_l.abs() } else { det_l.abs() / 6.0 };
        let vol_r = if dim == 2 { 0.5 * det_r.abs() } else { det_r.abs() / 6.0 };

        // Physical gradients (constant for P1)
        let grad_l = phys_grad(&j_inv_l, &ref_grad, npe, dim);
        let grad_r = phys_grad(&j_inv_r, &ref_grad, npe, dim);

        // Normal-dotted gradients: ∇φ·n
        let ngl: Vec<f64> = (0..npe).map(|i|
            (0..dim).map(|k| grad_l[i * dim + k] * n_l[k]).sum()
        ).collect();
        let ngr: Vec<f64> = (0..npe).map(|i|
            (0..dim).map(|k| grad_r[i * dim + k] * n_r[k]).sum()
        ).collect();

        // Basis values at face QP (constant for P1)
        let phi = ref_shape_vals(dim, npe, &q_rule);

        let dofs_l = space.element_dofs(e_l);
        let dofs_r = space.element_dofs(e_r);

        // LDG numerical flux:
        //   û = {u} + β [u]      where {u} = (u⁺ + u⁻)/2, [u] = u⁺·n⁺ + u⁻·n⁻
        //   (∇u)^ = {∇u} - β [∇u] - η/h [u]
        //
        // For the standard LDG formulation the matrix blocks are:
        //   K_LL[i,j] = -0.5 * κ * ngl[i] * phi[j]  -0.5 * κ * ngl[j] * phi[i]
        //               + (η/h) * phi[i] * phi[j]
        //               + β * κ * (-ngl[i] * phi[j] + ngl[j] * phi[i])   [β-switch term]
        //   K_LR[i,j] = +0.5 * κ * ngl[i] * phi[j]  -0.5 * κ * ngr[j] * phi[i]
        //               - (η/h) * phi[i] * phi[j]
        //               + β * κ * (-ngl[i] * phi[j] - ngr[j] * phi[i])
        //   K_RL[i,j] = -0.5 * κ * ngr[i] * phi[j]  +0.5 * κ * ngl[j] * phi[i]
        //               - (η/h) * phi[i] * phi[j]
        //               + β * κ * (+ngr[i] * phi[j] + ngl[j] * phi[i])
        //   K_RR[i,j] = +0.5 * κ * ngr[i] * phi[j]  +0.5 * κ * ngr[j] * phi[i]
        //               + (η/h) * phi[i] * phi[j]
        //               + β * κ * (ngr[i] * phi[j] - ngr[j] * phi[i])

        for i in 0..npe {
            let di_l = dofs_l[i] as usize;
            let di_r = dofs_r[i] as usize;
            for j in 0..npe {
                let dj_l = dofs_l[j] as usize;
                let dj_r = dofs_r[j] as usize;

                // LDG stabilization with β-switching
                let k_ll = -0.5 * kappa * ngl[i] * phi[j]
                           - 0.5 * kappa * ngl[j] * phi[i]
                           + pen * phi[i] * phi[j]
                           + beta * kappa * (-ngl[i] * phi[j] + ngl[j] * phi[i]);

                let k_lr = 0.5 * kappa * ngl[i] * phi[j]
                           - 0.5 * kappa * ngr[j] * phi[i]
                           - pen * phi[i] * phi[j]
                           + beta * kappa * (-ngl[i] * phi[j] - ngr[j] * phi[i]);

                let k_rl = -0.5 * kappa * ngr[i] * phi[j]
                           + 0.5 * kappa * ngl[j] * phi[i]
                           - pen * phi[i] * phi[j]
                           + beta * kappa * (ngr[i] * phi[j] + ngl[j] * phi[i]);

                let k_rr = 0.5 * kappa * ngr[i] * phi[j]
                           + 0.5 * kappa * ngr[j] * phi[i]
                           + pen * phi[i] * phi[j]
                           + beta * kappa * (ngr[i] * phi[j] - ngr[j] * phi[i]);

                coo.add(di_l, dj_l, k_ll);
                coo.add(di_l, dj_r, k_lr);
                coo.add(di_r, dj_l, k_rl);
                coo.add(di_r, dj_r, k_rr);
            }
        }
    }

    coo.into_csr()
}

/// Volume assembly for LDG: ∫ κ ∇u·∇v dx using P1 single-point quadrature.
fn assemble_volume<S: FESpace>(
    space: &S,
    mesh: &S::Mesh,
    kappa: f64,
    _quad_order: u8,
    coo: &mut CooMatrix<f64>,
) where <S as FESpace>::Mesh: MeshTopology {
    let dim = mesh.dim() as usize;
    let ne = mesh.n_elements();
    let npe = if ne > 0 { mesh.element_nodes(0u32).len() } else { 0 };
    let q_rule = if dim == 2 { vec![1.0/3.0, 1.0/3.0] } else { vec![0.25, 0.25, 0.25] };
    let ref_grad = ref_shape_grads(dim, npe, &q_rule);

    for e in 0..ne as u32 {
        let dofs = space.element_dofs(e);
        let (_, j_inv, det) = elem_jacobian(mesh, e, dim);
        let vol = if dim == 2 { 0.5 * det.abs() } else { det.abs() / 6.0 };
        let grad = phys_grad(&j_inv, &ref_grad, npe, dim);

        for i in 0..npe {
            let di = dofs[i] as usize;
            for j in 0..npe {
                let mut val = 0.0;
                for k in 0..dim {
                    val += grad[i * dim + k] * grad[j * dim + k];
                }
                coo.add(di, dofs[j] as usize, kappa * val * vol);
            }
        }
    }
}

// ─── Convenience: assemble LDG system + RHS in one call ──────────────────────

/// Assemble the full LDG system matrix for Poisson `-κΔu = f`.
pub fn assemble_ldg_system<S: FESpace + Sync>(
    space: &S,
    ifl: &InteriorFaceList,
    kappa: f64,
    eta: f64,
    beta: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    assemble_ldg(space, ifl, kappa, eta, beta, quad_order)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::L2Space;
    use crate::interior_faces::InteriorFaceList;

    /// Verify LDG matrix symmetry (should be symmetric for β=0).
    #[test]
    fn ldg_matrix_is_symmetric_beta0() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let a = assemble_ldg(&space, &ifl, 1.0, 1.0, 0.0, 3);
        let dense = a.to_dense();
        let n = a.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                assert!(diff < 1e-12,
                    "LDG (β=0) should be symmetric: K[{i},{j}] - K[{j},{i}] = {diff}");
            }
        }
    }

    /// LDG with β=0.5 should be non-symmetric.
    #[test]
    fn ldg_matrix_is_non_symmetric_beta05() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let a = assemble_ldg(&space, &ifl, 1.0, 1.0, 0.5, 3);
        let dense = a.to_dense();
        let n = a.nrows;
        let mut asym = false;
        for i in 0..n {
            for j in 0..n {
                if (dense[i * n + j] - dense[j * n + i]).abs() > 1e-12 {
                    asym = true;
                }
            }
        }
        assert!(asym, "LDG with β=0.5 should be non-symmetric");
    }

    /// LDG should be positive-definite (positive diagonal for η=10).
    #[test]
    fn ldg_positive_diagonal() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let a = assemble_ldg(&space, &ifl, 1.0, 10.0, 0.0, 3);
        for i in 0..a.nrows {
            assert!(a.get(i, i) > 0.0,
                "LDG diagonal K[{i},{i}] = {} should be positive", a.get(i, i));
        }
    }
}
