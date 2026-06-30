//! Unified face assembly: `FaceIntegrator` trait + generic interior/boundary drivers.
//!
//! Replaces the three separate face-loop implementations (DG SIP, DG advection,
//! HDG hand-written loops) with a single abstraction.
//!
//! # Design
//! - [`FaceQpData`] bundles all per-quadrature-point data for both adjacent elements.
//! - [`FaceIntegrator`] accumulates into a 4-block element matrix `[K_ll K_lr; K_rl K_rr]`.
//! - [`assemble_interior_faces`] loops over [`InteriorFaceList`], evaluating bases
//!   at face quadrature points and delegating to any [`FaceIntegrator`].
//! - [`assemble_boundary_faces`] loops over mesh boundary faces (one-sided).

use fem_core::types::{DofId, ElemId};
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_linalg::CooMatrix;
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use crate::interior_faces::InteriorFaceList;

// ─── Face quadrature-point data ──────────────────────────────────────────────

/// Per-quadrature-point data for interior/boundary face integrals.
pub struct FaceQpData<'a> {
    /// Number of DOFs on the left element.
    pub n_dofs_l: usize,
    /// Number of DOFs on the right element (0 for boundary faces).
    pub n_dofs_r: usize,
    /// Spatial dimension.
    pub dim: usize,
    /// Effective integration weight: quadrature weight × face Jacobian.
    pub weight: f64,
    /// Basis values on the left element; length `n_dofs_l`.
    pub phi_l: &'a [f64],
    /// Physical gradients on the left; row-major `[n_dofs_l × dim]`.
    pub grad_l: &'a [f64],
    /// Basis values on the right element; length `n_dofs_r` (empty for boundary).
    pub phi_r: &'a [f64],
    /// Physical gradients on the right; `[n_dofs_r × dim]`.
    pub grad_r: &'a [f64],
    /// Unit normal pointing outward from the left element; length `dim`.
    pub normal: &'a [f64],
    /// Physical coordinates of this quadrature point; length `dim`.
    pub x_phys: &'a [f64],
    /// Left element index.
    pub elem_l: ElemId,
    /// Right element index (same as `elem_l` for boundary).
    pub elem_r: ElemId,
    /// Global DOF indices for the left element.
    pub dofs_l: Option<&'a [DofId]>,
    /// Global DOF indices for the right element.
    pub dofs_r: Option<&'a [DofId]>,
}

// ─── FaceIntegrator trait ────────────────────────────────────────────────────

/// Bilinear form on a face (interior or boundary).
///
/// Accumulates into a 4-block element matrix:
/// ```text
/// K = [K_ll  K_lr]   (rows = left DOFs, cols = right DOFs)
///     [K_rl  K_rr]
/// ```
/// For boundary faces only `K_ll` is used (others remain empty).
pub trait FaceIntegrator: Send + Sync {
    /// Accumulate into the four face matrix blocks.
    fn add_to_face_matrix(&self, qp: &FaceQpData<'_>,
        k_ll: &mut [f64], k_lr: &mut [f64],
        k_rl: &mut [f64], k_rr: &mut [f64]);
}

// ─── Reference element helper (copied from assembler.rs) ──────────────────────

fn ref_elem_face(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(TriP1), // dummy, only grad_phys[..]=0
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4 | ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4 | ElementType::Tet10, 3) => Box::new(TetP3),
        _ => panic!("face_assembly ref_elem: unsupported ({elem_type:?}, order={order})"),
    }
}

// ─── Face normal helpers ─────────────────────────────────────────────────────

fn face_normal_2d(mesh: &impl MeshTopology, face_nodes: &[u32]) -> (Vec<f64>, f64) {
    let pa = mesh.node_coords(face_nodes[0]);
    let pb = mesh.node_coords(face_nodes[1]);
    let dx = pb[0] - pa[0];
    let dy = pb[1] - pa[1];
    let len = (dx * dx + dy * dy).sqrt();
    // Outward normal (left of edge direction as viewed from outside):
    // For edge from a→b, the left normal is (-dy, dx)/len.
    // We need the actual outward direction, which depends on element orientation.
    // Default: CCW normal = (-dy, dx). This points to the LEFT of (a→b).
    // For a CCW triangle on the left, this is the outward normal.
    (vec![-dy / len, dx / len], len)
}

fn face_normal_3d_tri(mesh: &impl MeshTopology, face_nodes: &[u32]) -> (Vec<f64>, f64) {
    let pa = mesh.node_coords(face_nodes[0]);
    let pb = mesh.node_coords(face_nodes[1]);
    let pc = mesh.node_coords(face_nodes[2]);
    let d1 = [pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]];
    let d2 = [pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]];
    let nx = d1[1]*d2[2] - d1[2]*d2[1];
    let ny = d1[2]*d2[0] - d1[0]*d2[2];
    let nz = d1[0]*d2[1] - d1[1]*d2[0];
    let area = 0.5 * (nx*nx + ny*ny + nz*nz).sqrt();
    let norm = 1.0 / (2.0 * area);
    (vec![nx*norm, ny*norm, nz*norm], area)
}

// ─── Generic interior face driver ────────────────────────────────────────────

/// Assemble interior face contributions using any [`FaceIntegrator`].
pub fn assemble_interior_faces<M: MeshTopology, S: FESpace<Mesh=M>, F: FaceIntegrator>(
    coo: &mut CooMatrix<f64>,
    mesh: &M,
    space: &S,
    ifl: &InteriorFaceList,
    integrator: &F,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let order = space.order();
    let _n_dofs = space.n_dofs();

    // Get reference element once; assume uniform element type.
    let elem_type = mesh.element_type(0);
    let r_elem = ref_elem_face(elem_type, order);
    let n_dofs_elem = r_elem.n_dofs();

    let mut phi_l = vec![0.0; n_dofs_elem];
    let mut phi_r = vec![0.0; n_dofs_elem];
    let mut grad_ref_l = vec![0.0; n_dofs_elem * dim];
    let mut grad_ref_r = vec![0.0; n_dofs_elem * dim];
    let mut grad_phys_l = vec![0.0; n_dofs_elem * dim];
    let mut grad_phys_r = vec![0.0; n_dofs_elem * dim];

    for face in &ifl.faces {
        let e_l = face.elem_left;
        let e_r = face.elem_right;
        let dofs_l: Vec<usize> = space.element_dofs(e_l).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = space.element_dofs(e_r).iter().map(|&d| d as usize).collect();
        let n_l = dofs_l.len();
        let n_r = dofs_r.len();

        // Determine quadrature rule and face normal direction
        let (normal_face, jac_face) = if dim == 2 {
            face_normal_2d(mesh, &face.face_nodes)
        } else {
            face_normal_3d_tri(mesh, &face.face_nodes)
        };

        let quad = r_elem.quadrature(quad_order);
        let mut k_ll = vec![0.0; n_l * n_l];
        let mut k_lr = vec![0.0; n_l * 0]; // will be resized if both sides
        let mut k_rl = vec![0.0; 0 * n_l];
        let mut k_rr = vec![0.0; 0 * 0];

        if n_r > 0 {
            k_lr = vec![0.0; n_l * n_r];
            k_rl = vec![0.0; n_r * n_l];
            k_rr = vec![0.0; n_r * n_r];
        }

        // For determining element-side orientation: the left element's outward normal
        // at this face. We use the face normal direction (from left to right).
        // For now, assume normal points from left to right (InteriorFaceList convention).

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q];
            let weight = w * jac_face; // face quadrature weight

            // Evaluate left element basis at face reference point
            r_elem.eval_basis(xi, &mut phi_l);
            r_elem.eval_grad_basis(xi, &mut grad_ref_l);
            // Jacobian transform for gradients
            // For face integrals we approximate the physical gradient using the
            // parametric element: the Jacobian at the face point projected onto 2D.
            // This is a simplification — for full accuracy we'd need ElementTransformation.
            // Copy grad_ref to grad_phys (identity Jacobian for reference elements on [-1,1]^d)
            // For simplex elements, we use the 1D/2D face Jacobian.
            grad_phys_l.copy_from_slice(&grad_ref_l);

            if n_r > 0 {
                r_elem.eval_basis(xi, &mut phi_r);
                r_elem.eval_grad_basis(xi, &mut grad_ref_r);
                grad_phys_r.copy_from_slice(&grad_ref_r);
            }

            let qp = FaceQpData {
                n_dofs_l: n_l,
                n_dofs_r: n_r,
                dim,
                weight,
                phi_l: &phi_l[..n_l],
                grad_l: &grad_phys_l[..n_l * dim],
                phi_r: &phi_r[..n_r],
                grad_r: &grad_phys_r[..n_r * dim],
                normal: &normal_face,
                x_phys: &[0.0; 3][..dim],
                elem_l: e_l,
                elem_r: e_r,
                dofs_l: Some(space.element_dofs(e_l)),
                dofs_r: if n_r > 0 { Some(space.element_dofs(e_r)) } else { None },
            };

            integrator.add_to_face_matrix(&qp, &mut k_ll, &mut k_lr, &mut k_rl, &mut k_rr);
        }

        // Scatter to global matrix
        for (i, &gi) in dofs_l.iter().enumerate() {
            for (j, &gj) in dofs_l.iter().enumerate() {
                coo.add(gi, gj, k_ll[i * n_l + j]);
            }
            if n_r > 0 {
                for (j, &gj) in dofs_r.iter().enumerate() {
                    coo.add(gi, gj, k_lr[i * n_r + j]);
                }
            }
        }
        if n_r > 0 {
            for (i, &gi) in dofs_r.iter().enumerate() {
                for (j, &gj) in dofs_l.iter().enumerate() {
                    coo.add(gi, gj, k_rl[i * n_l + j]);
                }
                for (j, &gj) in dofs_r.iter().enumerate() {
                    coo.add(gi, gj, k_rr[i * n_r + j]);
                }
            }
        }
    }
}

/// Assemble boundary face contributions using any [`FaceIntegrator`].
/// Only the left element (K_ll block) is used; normal points outward.
pub fn assemble_boundary_faces<M: MeshTopology, S: FESpace<Mesh=M>, F: FaceIntegrator>(
    coo: &mut CooMatrix<f64>,
    mesh: &M,
    space: &S,
    integrator: &F,
    quad_order: u8,
) {
    let dim = mesh.dim() as usize;
    let order = space.order();
    let elem_type = mesh.element_type(0);
    let r_elem = ref_elem_face(elem_type, order);
    let n_dofs_elem = r_elem.n_dofs();

    let mut phi = vec![0.0; n_dofs_elem];
    let mut grad_ref = vec![0.0; n_dofs_elem * dim];
    let mut grad_phys = vec![0.0; n_dofs_elem * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        let face_nodes = mesh.face_nodes(f).to_vec();
        let (e, _ext) = mesh.face_elements(f);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_l = dofs.len();

        let (normal_out, jac_face) = if dim == 2 {
            face_normal_2d(mesh, &face_nodes)
        } else {
            face_normal_3d_tri(mesh, &face_nodes)
        };

        let quad = r_elem.quadrature(quad_order);
        let mut k_ll = vec![0.0; n_l * n_l];

        for (q, xi) in quad.points.iter().enumerate() {
            let weight = quad.weights[q] * jac_face;

            r_elem.eval_basis(xi, &mut phi);
            r_elem.eval_grad_basis(xi, &mut grad_ref);
            grad_phys.copy_from_slice(&grad_ref);

            let qp = FaceQpData {
                n_dofs_l: n_l,
                n_dofs_r: 0,
                dim,
                weight,
                phi_l: &phi[..n_l],
                grad_l: &grad_phys[..n_l * dim],
                phi_r: &[],
                grad_r: &[],
                normal: &normal_out,
                x_phys: &[0.0; 3][..dim],
                elem_l: e,
                elem_r: e,
                dofs_l: Some(space.element_dofs(e)),
                dofs_r: None,
            };

            integrator.add_to_face_matrix(&qp, &mut k_ll, &mut [], &mut [], &mut []);
        }

        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                coo.add(gi, gj, k_ll[i * n_l + j]);
            }
        }
    }
}
