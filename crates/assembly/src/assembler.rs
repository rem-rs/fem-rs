//! Global assembly loop.
//!
//! [`Assembler`] drives the element-by-element assembly of bilinear and linear
//! forms over the mesh.  It is stateless; all data comes from the [`FESpace`]
//! and integrators supplied at call time.

use nalgebra::DMatrix;

use fem_core::types::DofId;
use fem_element::{ReferenceElement, lagrange::{SegP1, SegP2, TetP1, TriP1, TriP2}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

use crate::integrator::{BdQpData, BoundaryBilinearIntegrator, BoundaryLinearIntegrator, BilinearIntegrator, LinearIntegrator, QpData};

// ─── Reference element factory ───────────────────────────────────────────────

/// Return the solution reference element matching `elem_type` and polynomial `order`.
fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tet4, 1)                           => Box::new(TetP1),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

/// Return the solution reference element for a boundary face.
fn ref_elem_face(face_elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (face_elem_type, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Tri3,  1) => Box::new(TriP1),
        _ => panic!("ref_elem_face: unsupported (element_type={face_elem_type:?}, order={order})"),
    }
}

// ─── Jacobian helpers ─────────────────────────────────────────────────────────

/// Build the Jacobian matrix `J[i, j] = x_{j+1}[i] − x_0[i]` for a simplex
/// with nodes listed in `geo_nodes` (first `dim+1` nodes are the vertices).
///
/// Returns `(J, det J)`.
fn simplex_jacobian<M: MeshTopology>(
    mesh: &M,
    geo_nodes: &[u32],
    dim: usize,
) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(geo_nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(geo_nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    let det = j.determinant();
    (j, det)
}

/// Compute physical coordinates: `x_phys = x0 + J * xi`.
fn phys_coords(x0: &[f64], j: &DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64> {
    let mut xp = x0.to_vec();
    for i in 0..dim {
        for k in 0..dim {
            xp[i] += j[(i, k)] * xi[k];
        }
    }
    xp
}

/// Transform reference gradients to physical gradients:
/// `grad_phys[i] = J^{−T} grad_ref[i]`.
fn transform_grads(
    j_inv_t: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n_ldofs: usize,
    dim: usize,
) {
    for i in 0..n_ldofs {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(j, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + j] = s;
        }
    }
}

// ─── Assembler ────────────────────────────────────────────────────────────────

/// Stateless assembly driver.
///
/// All methods are associated functions (no `self` needed) that take the
/// relevant space and integrators as arguments.
pub struct Assembler;

impl Assembler {
    // ── Volume bilinear form: K = Σ_e k_e ────────────────────────────────────

    /// Assemble the global stiffness matrix for a bilinear form.
    ///
    /// # Arguments
    /// * `space`       — finite element space (provides mesh + DOF map).
    /// * `integrators` — slice of bilinear-form contributions to accumulate.
    /// * `quad_order`  — polynomial order that the quadrature rule integrates exactly.
    ///
    /// # Returns
    /// Assembled `CsrMatrix<f64>` in CSR format.
    pub fn assemble_bilinear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],
        quad_order:  u8,
    ) -> CsrMatrix<f64> {
        let mesh   = space.mesh();
        let dim    = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let order  = space.order();

        // Accumulate in COO then convert to CSR.
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        let mut phi       = Vec::<f64>::new();
        let mut grad_ref  = Vec::<f64>::new();
        let mut grad_phys = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem  = ref_elem_vol(elem_type, order);
            let n_ldofs   = ref_elem.n_dofs();
            let quad      = ref_elem.quadrature(quad_order);

            let raw_dofs: Vec<DofId> =
                space.element_dofs(e).to_vec();
            let global_dofs: Vec<usize> =
                raw_dofs.iter().map(|&d| d as usize).collect();
            let n_elem_dofs = global_dofs.len(); // may be n_ldofs * dim for vector spaces
            let nodes = mesh.element_nodes(e);
            let elem_tag = mesh.element_tag(e);

            // Geometric Jacobian (linear/affine map from first dim+1 nodes).
            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.clone().try_inverse()
                .expect("degenerate element — zero-area/volume triangle")
                .transpose();

            let mut k_elem = vec![0.0_f64; n_elem_dofs * n_elem_dofs];
            phi.resize(n_ldofs, 0.0);
            grad_ref.resize(n_ldofs * dim, 0.0);
            grad_phys.resize(n_ldofs * dim, 0.0);

            let x0 = mesh.node_coords(nodes[0]);

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();

                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);
                let xp = phys_coords(x0, &jac, xi, dim);

                let qp = QpData {
                    n_dofs:    n_elem_dofs,
                    dim,
                    weight:    w,
                    phi:       &phi,
                    grad_phys: &grad_phys,
                    x_phys:    &xp,
                    elem_id:   e,
                    elem_tag,
                    elem_dofs: Some(&raw_dofs),
                };

                for integ in integrators {
                    integ.add_to_element_matrix(&qp, &mut k_elem);
                }
            }

            coo.add_element_matrix(&global_dofs, &k_elem);
        }

        coo.into_csr()
    }

    // ── Volume linear form: f = Σ_e f_e ──────────────────────────────────────

    /// Assemble the global load vector for a linear form.
    pub fn assemble_linear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn LinearIntegrator],
        quad_order:  u8,
    ) -> Vec<f64> {
        let mesh   = space.mesh();
        let dim    = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let order  = space.order();

        let mut rhs = vec![0.0_f64; n_dofs];

        let mut phi       = Vec::<f64>::new();
        let mut grad_ref  = Vec::<f64>::new();
        let mut grad_phys = Vec::<f64>::new();

        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem  = ref_elem_vol(elem_type, order);
            let n_ldofs   = ref_elem.n_dofs();
            let quad      = ref_elem.quadrature(quad_order);

            let raw_dofs: Vec<DofId> =
                space.element_dofs(e).to_vec();
            let global_dofs: Vec<usize> =
                raw_dofs.iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let elem_tag = mesh.element_tag(e);

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.clone().try_inverse().unwrap().transpose();

            let mut f_elem = vec![0.0_f64; n_ldofs];
            phi.resize(n_ldofs, 0.0);
            grad_ref.resize(n_ldofs * dim, 0.0);
            grad_phys.resize(n_ldofs * dim, 0.0);

            let x0 = mesh.node_coords(nodes[0]);

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();

                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);
                let xp = phys_coords(x0, &jac, xi, dim);

                let qp = QpData {
                    n_dofs:    n_ldofs,
                    dim,
                    weight:    w,
                    phi:       &phi,
                    grad_phys: &grad_phys,
                    x_phys:    &xp,
                    elem_id:   e,
                    elem_tag,
                    elem_dofs: Some(&raw_dofs),
                };

                for integ in integrators {
                    integ.add_to_element_vector(&qp, &mut f_elem);
                }
            }

            coo_add_element_vec(&global_dofs, &f_elem, &mut rhs);
        }

        rhs
    }

    // ── Boundary linear form ──────────────────────────────────────────────────

    /// Assemble boundary contributions (e.g. Neumann BCs) into a load vector.
    ///
    /// # Arguments
    /// * `n_dofs`      — total number of global DOFs.
    /// * `mesh`        — mesh topology.
    /// * `face_dofs`   — closure: `face_id → &[global_dof_id]` for each boundary face.
    /// * `integrators` — boundary linear integrators to accumulate.
    /// * `tags`        — only process boundary faces whose tag is in this list.
    /// * `quad_order`  — quadrature accuracy order.
    ///
    /// The closure `face_dofs` lets you pass either a P1 or P2 DOF list depending
    /// on your space (see [`face_dofs_p1`] and [`face_dofs_p2`] helpers).
    pub fn assemble_boundary_linear(
        n_dofs:      usize,
        mesh:        &dyn MeshTopology,
        face_dofs:   &dyn Fn(u32) -> Vec<DofId>,
        order:       u8,
        integrators: &[&dyn BoundaryLinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> Vec<f64> {
        let dim = mesh.dim() as usize;
        let mut rhs = vec![0.0_f64; n_dofs];

        for f in mesh.face_iter() {
            if !tags.contains(&mesh.face_tag(f)) { continue; }

            let fdofs: Vec<DofId> = face_dofs(f);
            let n_fdofs = fdofs.len();

            // Determine face element type from boundary face nodes count.
            let face_type = match mesh.face_nodes(f).len() {
                2 => ElementType::Line2,
                3 => ElementType::Tri3,
                _ => panic!("unsupported boundary face node count"),
            };
            let ref_elem = ref_elem_face(face_type, order);
            let quad = ref_elem.quadrature(quad_order);

            let face_nodes = mesh.face_nodes(f);

            // Face Jacobian and normal (2-D only for now).
            let (face_j_mag, normal) = face_jacobian_and_normal(mesh, face_nodes, dim);

            let mut phi    = vec![0.0_f64; n_fdofs];
            let mut f_face = vec![0.0_f64; n_fdofs];
            let x0 = mesh.node_coords(face_nodes[0]);

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * face_j_mag;

                ref_elem.eval_basis(xi, &mut phi);

                // Physical coordinate on face: x = x0 + (x1-x0)*t (for Line2, t = xi[0]).
                let xp: Vec<f64> = (0..dim).map(|i| {
                    let x1 = mesh.node_coords(face_nodes[1]);
                    x0[i] + (x1[i] - x0[i]) * xi[0]
                }).collect();

                let qp = BdQpData {
                    n_dofs:  n_fdofs,
                    dim,
                    weight:  w,
                    phi:     &phi,
                    x_phys:  &xp,
                    normal:  &normal,
                    elem_id: 0, // boundary faces: owner element not tracked yet
                    elem_tag: mesh.face_tag(f),
                };

                for integ in integrators {
                    integ.add_to_face_vector(&qp, &mut f_face);
                }
            }

            let global: Vec<usize> = fdofs.iter().map(|&d| d as usize).collect();
            coo_add_element_vec(&global, &f_face, &mut rhs);
        }

        rhs
    }

    // ── Boundary bilinear form ───────────────────────────────────────────────

    /// Assemble a boundary bilinear form (e.g. boundary mass ∫_Γ α u v ds).
    ///
    /// # Arguments
    /// * `n_dofs`      — total number of global DOFs.
    /// * `mesh`        — mesh topology.
    /// * `face_dofs`   — closure: `face_id → &[global_dof_id]` for each boundary face.
    /// * `order`       — polynomial order of the face reference element.
    /// * `integrators` — boundary bilinear integrators to accumulate.
    /// * `tags`        — only process boundary faces whose tag is in this list.
    /// * `quad_order`  — quadrature accuracy order.
    pub fn assemble_boundary_bilinear(
        n_dofs:      usize,
        mesh:        &dyn MeshTopology,
        face_dofs:   &dyn Fn(u32) -> Vec<DofId>,
        order:       u8,
        integrators: &[&dyn BoundaryBilinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> CsrMatrix<f64> {
        let dim = mesh.dim() as usize;
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        for f in mesh.face_iter() {
            if !tags.contains(&mesh.face_tag(f)) { continue; }

            let fdofs: Vec<DofId> = face_dofs(f);
            let n_fdofs = fdofs.len();

            let face_type = match mesh.face_nodes(f).len() {
                2 => ElementType::Line2,
                3 => ElementType::Tri3,
                _ => panic!("unsupported boundary face node count"),
            };
            let ref_elem = ref_elem_face(face_type, order);
            let quad = ref_elem.quadrature(quad_order);

            let face_nodes = mesh.face_nodes(f);
            let (face_j_mag, normal) = face_jacobian_and_normal(mesh, face_nodes, dim);

            let mut phi    = vec![0.0_f64; n_fdofs];
            let mut k_face = vec![0.0_f64; n_fdofs * n_fdofs];
            let x0 = mesh.node_coords(face_nodes[0]);

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * face_j_mag;

                ref_elem.eval_basis(xi, &mut phi);

                let xp: Vec<f64> = (0..dim).map(|i| {
                    let x1 = mesh.node_coords(face_nodes[1]);
                    x0[i] + (x1[i] - x0[i]) * xi[0]
                }).collect();

                let qp = BdQpData {
                    n_dofs:  n_fdofs,
                    dim,
                    weight:  w,
                    phi:     &phi,
                    x_phys:  &xp,
                    normal:  &normal,
                    elem_id: 0,
                    elem_tag: mesh.face_tag(f),
                };

                for integ in integrators {
                    integ.add_to_face_matrix(&qp, &mut k_face);
                }
            }

            let global: Vec<usize> = fdofs.iter().map(|&d| d as usize).collect();
            coo.add_element_matrix(&global, &k_face);
        }

        coo.into_csr()
    }
}

// ─── Face Jacobian and normal (2-D) ──────────────────────────────────────────

/// Compute the face Jacobian magnitude and outward unit normal for a 2-D boundary edge.
///
/// Returns `(|J_face|, n)` where `|J_face|` is the edge length and `n` is the
/// unit outward normal (rotated 90° from the edge tangent, pointing away from
/// the interior by convention `n = (dy, -dx) / |J_face|`).
fn face_jacobian_and_normal(
    mesh:       &dyn MeshTopology,
    face_nodes: &[u32],
    dim:        usize,
) -> (f64, Vec<f64>) {
    assert_eq!(dim, 2, "face_jacobian_and_normal currently only supports 2-D meshes");
    let x0 = mesh.node_coords(face_nodes[0]);
    let x1 = mesh.node_coords(face_nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();
    // Outward normal convention: rotate tangent (dx,dy) by -90° → (dy, -dx)
    let normal = vec![dy / len, -dx / len];
    (len, normal)
}

// ─── Scatter helper ───────────────────────────────────────────────────────────

/// Scatter `f_elem` into `rhs` at global DOF indices `dofs`.
#[inline]
fn coo_add_element_vec(dofs: &[usize], f_elem: &[f64], rhs: &mut [f64]) {
    for (&d, &v) in dofs.iter().zip(f_elem.iter()) {
        rhs[d] += v;
    }
}

// ─── Face DOF helpers ─────────────────────────────────────────────────────────

/// Build the face DOF list for a P1 space: face node indices only.
///
/// Use this as the `face_dofs` closure in [`Assembler::assemble_boundary_linear`]
/// for H1/P1 and L2/P0 or P1 spaces.
pub fn face_dofs_p1(mesh: &dyn MeshTopology) -> impl Fn(u32) -> Vec<DofId> + '_ {
    move |f| mesh.face_nodes(f).iter().map(|&n| n as DofId).collect()
}

/// Build the face DOF list for a P2 H1 space.
///
/// For each boundary face `f`, the face DOFs are the two vertex DOFs plus the
/// edge-midpoint DOF shared between them.  The edge-midpoint DOF is found by
/// looking at the element that owns the face and matching the edge in its DOF table.
///
/// # Panics
/// Panics if the face is not owned by any element or if the vertices cannot be
/// matched in the element's DOF table (programming error).
pub fn face_dofs_p2<S>(space: &S) -> impl Fn(u32) -> Vec<DofId> + '_
where
    S: FESpace,
    S::Mesh: MeshTopology,
{
    move |f| {
        let mesh = space.mesh();
        let fn_nodes = mesh.face_nodes(f);
        let (elem, _) = mesh.face_elements(f);
        let elem_nodes = mesh.element_nodes(elem);
        let elem_dofs  = space.element_dofs(elem);

        // Find local vertex positions of the two face nodes.
        let pos_a = elem_nodes.iter().position(|&n| n == fn_nodes[0])
            .expect("face node 0 not in element");
        let pos_b = elem_nodes.iter().position(|&n| n == fn_nodes[1])
            .expect("face node 1 not in element");

        let dof_a = elem_dofs[pos_a];
        let dof_b = elem_dofs[pos_b];

        // For TriP2 the edge DOF positions relative to vertex positions are:
        //   edge(v0→v1) = dofs[3],  edge(v1→v2) = dofs[4],  edge(v0→v2) = dofs[5]
        // Generalised: edge DOF for sorted (min_pos, max_pos) in {(0,1),(1,2),(0,2)}.
        let edge_dof = find_edge_dof(elem_nodes, elem_dofs, pos_a, pos_b);

        vec![dof_a, dof_b, edge_dof]
    }
}

/// Return the edge-midpoint DOF for the edge between local vertex positions `a` and `b`
/// in a TriP2 element (with 6 DOFs: 3 vertex + 3 edge).
fn find_edge_dof(elem_nodes: &[u32], elem_dofs: &[DofId], pos_a: usize, pos_b: usize) -> DofId {
    let (lo, hi) = if pos_a < pos_b { (pos_a, pos_b) } else { (pos_b, pos_a) };
    // TriP2 edge DOF mapping: (0,1)→3, (1,2)→4, (0,2)→5
    let _ = elem_nodes; // used via pos_a/pos_b
    let edge_local = match (lo, hi) {
        (0, 1) => 3,
        (1, 2) => 4,
        (0, 2) => 5,
        _ => panic!("find_edge_dof: unexpected vertex pair ({lo},{hi}) — only TriP2 supported"),
    };
    elem_dofs[edge_local]
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, fe_space::FESpace};

    #[test]
    fn assemble_bilinear_p1_returns_correct_size() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        // Diffusion integrator stub (adds nothing) — just test shape.
        struct Zero;
        impl BilinearIntegrator for Zero {
            fn add_to_element_matrix(&self, _: &QpData<'_>, _: &mut [f64]) {}
        }
        let mat = Assembler::assemble_bilinear(&space, &[&Zero], 2);
        assert_eq!(mat.nrows, n);
        assert_eq!(mat.ncols, n);
    }

    #[test]
    fn assemble_linear_p1_returns_correct_size() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        struct Zero;
        impl LinearIntegrator for Zero {
            fn add_to_element_vector(&self, _: &QpData<'_>, _: &mut [f64]) {}
        }
        let rhs = Assembler::assemble_linear(&space, &[&Zero], 2);
        assert_eq!(rhs.len(), n);
    }
}
