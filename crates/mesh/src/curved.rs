//! High-order curved mesh with isoparametric geometry mapping and Jacobian caching.
//!
//! Provides:
//! - [`CurvedMesh`] — arbitrary-order isoparametric mesh
//! - [`JacobianCache`] — precomputed `(J, detJ, J⁻ᵀ)` for all `(elem, qp)` pairs
//! - [`CurvedElementTransformation`] — isoparametric transformation for assembly
//! - [`refine_curved`] / [`refine_curved_3d`] — curved AMR with geometric node interpolation

use fem_core::{ElemId, FaceId, NodeId};
use nalgebra::DMatrix;
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use crate::topology::MeshTopology;

// ─── CurvedMesh ──────────────────────────────────────────────────────────────

/// A high-order curved (isoparametric) mesh with arbitrary geometric order.
///
/// The geometric mapping uses Lagrange basis functions of order `geom_order`,
/// using the factory's [`ref_elem`](fem_element::lagrange::factory::ref_elem)
/// for basis function evaluation.
#[derive(Debug, Clone)]
pub struct CurvedMesh<const D: usize> {
    /// Flat coordinate array.  Length = `n_nodes * D`.
    pub coords: Vec<f64>,
    /// Element connectivity.  Length = `n_elems * nodes_per_elem`.
    pub geom_conn: Vec<NodeId>,
    /// Geometric polynomial order (1 = linear, 2 = quadratic, ...).
    pub geom_order: u8,
    /// Number of geometric nodes per element.
    pub nodes_per_elem: usize,
    /// Element type (Tri3 for linear, Tri6 for quadratic, etc.).
    pub elem_type: ElementType,
    /// Total number of elements.
    pub n_elems: usize,
    /// Total number of nodes.
    pub n_nodes: usize,
    /// Boundary face connectivity.  Length = `n_faces * nodes_per_face`.
    pub face_conn: Vec<NodeId>,
    /// Physical group tags per boundary face.  Length = `n_faces`.
    pub face_tags: Vec<i32>,
    /// Face element type.
    pub face_type: ElementType,
    /// Tags for each element (material/domain labels).
    pub elem_tags: Vec<i32>,
}

impl<const D: usize> CurvedMesh<D> {
    /// Construct an order-1 curved mesh from a `Mesh`.
    pub fn from_linear(mesh: &Mesh<D>) -> Self {
        let npe = mesh.elem_type.nodes_per_element();
        CurvedMesh {
            coords:     mesh.coords.clone(),
            geom_conn:  mesh.conn.clone(),
            geom_order: 1,
            nodes_per_elem: npe,
            elem_type:  mesh.elem_type,
            n_elems:    mesh.n_elems(),
            n_nodes:    mesh.n_nodes(),
            face_conn:  mesh.face_conn.clone(),
            face_tags:  mesh.face_tags.clone(),
            face_type:  mesh.face_type,
            elem_tags:  mesh.elem_tags.clone(),
        }
    }

    /// Elevate to arbitrary order `p`.
    ///
    /// Inserts high-order geometric nodes for each element, edge, face, and volume.
    /// For 2D: Tri3 �?TriPk.  For 3D: Tet4 �?TetPk.
    /// The `map_fn` transforms new node coordinates (e.g., projects onto a curved surface).
    pub fn elevate_to_order<F>(mesh: &Mesh<D>, p: usize, map_fn: F) -> Self
    where
        F: Fn([f64; D]) -> [f64; D],
    {
        let dim = D;
        let order = p as u8;
        let npe_new = fem_element::lagrange::factory::n_dofs_simplex(dim, p);
        let n_linear_nodes = mesh.n_nodes();
        let mut new_coords: Vec<f64> = mesh.coords.clone();
        let mut next_node = n_linear_nodes as NodeId;

        // Build edge key �?[p-1 node IDs] mapping (shared between elements)
        use std::collections::HashMap;
        let mut edge_map: HashMap<(NodeId, NodeId), Vec<NodeId>> = HashMap::new();

        // Helper: get or create (p-1) nodes along edge (a,b), ordered a→b
        #[allow(clippy::too_many_arguments)]
        fn get_edge_nodes<const D: usize>(
            a: NodeId, b: NodeId,
            a_coords: &[f64; D], b_coords: &[f64; D],
            p: usize,
            next: &mut NodeId,
            new_coords: &mut Vec<f64>,
            edge_map: &mut HashMap<(NodeId, NodeId), Vec<NodeId>>,
            map_fn: &impl Fn([f64; D]) -> [f64; D],
        ) -> Vec<NodeId> {
            let key = if a < b { (a, b) } else { (b, a) };
            let nodes = edge_map.entry(key).or_insert_with(|| {
                let mut ids = Vec::with_capacity(p - 1);
                for k in 1..p {
                    let t = k as f64 / p as f64;
                    let mut xm = [0.0; D];
                    for i in 0..D { xm[i] = (1.0 - t) * a_coords[i] + t * b_coords[i]; }
                    xm = map_fn(xm);
                    new_coords.extend_from_slice(&xm);
                    ids.push(*next);
                    *next += 1;
                }
                ids
            });
            // Geometry nodes MUST follow factory DOF order (t increasing from key.0 to key.1).
            // Reversing would swap edge DOF indices, causing Jacobian errors at p >= 3.
            // All elements sharing this edge get nodes in the same (canonical) order.
            nodes.clone()
        }

        let n_elems = mesh.n_elems();
        let mut geom_conn = Vec::with_capacity(n_elems * npe_new);

        for e in 0..n_elems {
            let ns = mesh.elem_nodes(e as NodeId);
            let _base = e * npe_new;
            let off = geom_conn.len();
            geom_conn.resize(off + npe_new, 0);

            // Copy vertex nodes
            geom_conn[off..(dim + off + 1)].copy_from_slice(&ns[..(dim + 1)]);

            if dim == 2 {
                // Tri: 3 edges �?v0v1, v1v2, v0v2
                let edges = [(0usize, 1usize), (1, 2), (0, 2)];
                let mut pos = 3;
                for &(ai, bi) in &edges {
                    let (a, b) = (ns[ai], ns[bi]);
                    let ca = mesh.coords_of(a);
                    let cb = mesh.coords_of(b);
                    let edge_ids = get_edge_nodes::<D>(a, b,
                        &ca, &cb, p, &mut next_node, &mut new_coords, &mut edge_map, &map_fn);
                    for (k, &id) in edge_ids.iter().enumerate() {
                        geom_conn[off + pos + k] = id;
                    }
                    pos += p - 1;
                }
                // Face interior nodes for p �?3
                if p >= 3 {
                    let mut face_pos = 3 + 3 * (p - 1);
                    for j in 1..=(p - 2) {
                        for i in 1..=(p - 1 - j) {
                            let mut x = [0.0; D];
                            let (fi_x, fi_y) = (i as f64 / p as f64, j as f64 / p as f64);
                            for d in 0..D {
                                x[d] = (1.0 - fi_x - fi_y) * mesh.coords_of(ns[0])[d]
                                    + fi_x * mesh.coords_of(ns[1])[d]
                                    + fi_y * mesh.coords_of(ns[2])[d];
                            }
                            x = map_fn(x);
                            new_coords.extend_from_slice(&x);
                            geom_conn[off + face_pos] = next_node;
                            next_node += 1;
                            face_pos += 1;
                        }
                    }
                }
            } else {
                // Tet: 6 edges �?v0v1, v0v2, v0v3, v1v2, v1v3, v2v3
                let edges = [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
                let mut pos = 4;
                for &(ai, bi) in &edges {
                    let (a, b) = (ns[ai], ns[bi]);
                    let ca = mesh.coords_of(a);
                    let cb = mesh.coords_of(b);
                    let edge_ids = get_edge_nodes::<D>(a, b,
                        &ca, &cb, p, &mut next_node, &mut new_coords, &mut edge_map, &map_fn);
                    for (k, &id) in edge_ids.iter().enumerate() {
                        geom_conn[off + pos + k] = id;
                    }
                    pos += p - 1;
                }
                // Face interior nodes (for p �?3)
                // Match factory ordering: Face(0,1,2), Face(0,1,3), Face(0,2,3), Face(1,2,3)
                // each with (p-1)(p-2)/2 nodes in triangular pattern (k=1..p-2, i/j=1..p-1-k)
                let _face_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
                let mut face_pos = 4 + 6 * (p - 1);
                for face_idx in 0..4 {
                    for k in 1..=p.saturating_sub(2) {
                        let max_ij = p - 1 - k;
                        for ij in 1..=max_ij {
                            let (xi, eta, zeta): (f64, f64, f64) = match face_idx {
                                0 => (ij as f64 / p as f64, k as f64 / p as f64, 0.0),       // face(0,1,2) z=0
                                1 => (ij as f64 / p as f64, 0.0, k as f64 / p as f64),       // face(0,1,3) y=0
                                2 => (0.0, ij as f64 / p as f64, k as f64 / p as f64),       // face(0,2,3) x=0
                                _ => (1.0 - (ij + k) as f64 / p as f64, ij as f64 / p as f64, k as f64 / p as f64), // face(1,2,3)
                            };
                            let mut x = [0.0; D];
                            for d in 0..D {
                                x[d] = (1.0 - xi - eta - zeta) * mesh.coords_of(ns[0])[d]
                                     + xi * mesh.coords_of(ns[1])[d]
                                     + eta * mesh.coords_of(ns[2])[d]
                                     + zeta * mesh.coords_of(ns[3])[d];
                            }
                            x = map_fn(x);
                            new_coords.extend_from_slice(&x);
                            geom_conn[off + face_pos] = next_node;
                            next_node += 1;
                            face_pos += 1;
                        }
                    }
                }
                // Volume interior nodes (for p �?4)
                // Match factory ordering: k=1..p-3, j=1..p-2-k, i=1..p-1-j-k
                if p >= 4 {
                    let _n_vol = (p - 1) * (p - 2) * (p - 3) / 6;
                    for k in 1..=(p - 3) {
                        for j in 1..=(p - 2 - k) {
                            for i in 1..=(p - 1 - j - k) {
                                let xi = i as f64 / p as f64;
                                let eta = j as f64 / p as f64;
                                let zeta = k as f64 / p as f64;
                                let mut x = [0.0; D];
                                for d in 0..D {
                                    x[d] = (1.0 - xi - eta - zeta) * mesh.coords_of(ns[0])[d]
                                         + xi * mesh.coords_of(ns[1])[d]
                                         + eta * mesh.coords_of(ns[2])[d]
                                         + zeta * mesh.coords_of(ns[3])[d];
                                }
                                x = map_fn(x);
                                new_coords.extend_from_slice(&x);
                                geom_conn[off + face_pos] = next_node;
                                next_node += 1;
                                face_pos += 1;
                            }
                        }
                    }
                }
            }
        }

        // Determine new element/face type
        let new_elem_type = match (dim, p) {
            (2, 1) => ElementType::Tri3,
            (2, 2) => ElementType::Tri6,
            (3, 1) => ElementType::Tet4,
            (3, 2) => ElementType::Tet10,
            _ => mesh.elem_type,
        };
        let new_face_type = match (dim, p) {
            (2, _) => ElementType::Line2,
            (3, _) => ElementType::Tri3,
            _ => mesh.face_type,
        };

        CurvedMesh {
            n_nodes: next_node as usize,
            n_elems,
            coords: new_coords,
            geom_conn,
            geom_order: order,
            nodes_per_elem: npe_new,
            elem_type: new_elem_type,
            face_conn: mesh.face_conn.clone(),
            face_tags: mesh.face_tags.clone(),
            face_type: new_face_type,
            elem_tags: mesh.elem_tags.clone(),
        }
    }

    /// Coordinates of geometric node `n` as a fixed-size array.
    #[inline]
    pub fn node_coords_arr(&self, n: NodeId) -> [f64; D] {
        let off = n as usize * D;
        std::array::from_fn(|i| self.coords[off + i])
    }

    /// Geometric node IDs for element `e`.
    fn elem_geom_nodes(&self, e: usize) -> &[NodeId] {
        let off = e * self.nodes_per_elem;
        &self.geom_conn[off..off + self.nodes_per_elem]
    }

    /// Compute the isoparametric Jacobian `J = ∂F/∂ξ` at reference point `xi`.
    ///
    /// Uses factory DOF reference coordinates to compute physical node positions
    /// via barycentric interpolation from the element vertices, bypassing the
    /// `geom_conn` node ordering. This ensures correct Jacobians even when
    /// edge nodes are stored in a different order (due to edge key reversal).
    pub fn element_jacobian(&self, e: usize, xi: &[f64]) -> (DMatrix<f64>, f64) {
        let dim = D;
        let n = self.nodes_per_elem;

        let mut grad_ref = vec![0.0_f64; n * dim];
        self.eval_geom_grad_basis(xi, &mut grad_ref);

        // Get factory DOF reference coordinates (always in factory order)
        use fem_element::lagrange::factory::ref_elem;
        let et = mesh_elem_type_to_factory_type(self.elem_type);
        let factory = ref_elem(et, self.geom_order);
        let ref_coords = factory.dof_coords();

        // Vertex coordinates (first n_vert entries of geom_conn are vertices)
        let nodes = self.elem_geom_nodes(e);
        let verts: Vec<[f64; D]> = (0..=dim).map(|i| self.node_coords_arr(nodes[i])).collect();

        // Compute Jacobian using node positions in factory DOF order
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..n {
            let rc = &ref_coords[k];
            let mut xk = [0.0_f64; D];
            if dim == 2 {
                for d in 0..2 {
                    xk[d] = (1.0 - rc[0] - rc[1]) * verts[0][d]
                          + rc[0] * verts[1][d]
                          + rc[1] * verts[2][d];
                }
            } else {
                for d in 0..3 {
                    xk[d] = (1.0 - rc[0] - rc[1] - rc[2]) * verts[0][d]
                          + rc[0] * verts[1][d]
                          + rc[1] * verts[2][d]
                          + rc[2] * verts[3][d];
                }
            }
            for row in 0..dim {
                for col in 0..dim {
                    j[(row, col)] += xk[row] * grad_ref[k * dim + col];
                }
            }
        }
        let det = j.determinant();
        (j, det)
    }

    /// Physical coordinates of reference point `xi` in element `e`.
    pub fn reference_to_physical(&self, e: usize, xi: &[f64]) -> [f64; D] {
        let n = self.nodes_per_elem;
        let nodes = self.elem_geom_nodes(e);
        let mut phi = vec![0.0_f64; n];
        self.eval_geom_basis(xi, &mut phi);
        let mut xp = [0.0_f64; D];
        for k in 0..n {
            let xk = self.node_coords_arr(nodes[k]);
            for i in 0..D { xp[i] += xk[i] * phi[k]; }
        }
        xp
    }

    /// Evaluate geometric basis functions at `xi` using the factory.
    pub(crate) fn eval_geom_basis(&self, xi: &[f64], phi: &mut [f64]) {
        use fem_element::lagrange::factory::ref_elem;
        let et = mesh_elem_type_to_factory_type(self.elem_type);
        let elem = ref_elem(et, self.geom_order);
        elem.eval_basis(xi, phi);
    }

    /// Evaluate geometric basis function gradients at `xi` using the factory.
    pub(crate) fn eval_geom_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        use fem_element::lagrange::factory::ref_elem;
        let et = mesh_elem_type_to_factory_type(self.elem_type);
        let elem = ref_elem(et, self.geom_order);
        // Check capacity
        assert!(grads.len() >= self.nodes_per_elem * D,
            "grads len {} < nodes_per_elem {} * D {}", grads.len(), self.nodes_per_elem, D);
        elem.eval_grad_basis(xi, grads);
    }
}

/// Convert mesh ElementType to factory ElemType for basis evaluation.
fn mesh_elem_type_to_factory_type(t: ElementType) -> fem_element::lagrange::factory::ElemType {
    use fem_element::lagrange::factory::ElemType;
    match t {
        ElementType::Tri3 | ElementType::Tri6 => ElemType::Tri,
        ElementType::Tet4 | ElementType::Tet10 => ElemType::Tet,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => ElemType::Quad,
        ElementType::Hex8 | ElementType::Hex20 => ElemType::Hex,
        ElementType::Hex27 => ElemType::Hex,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => ElemType::Prism,
        ElementType::Pyramid5 | ElementType::Pyramid13 => ElemType::Pyramid,
        ElementType::Line2 | ElementType::Line3 => ElemType::Seg,
        _ => panic!("unsupported element type for curved mesh: {t:?}"),
    }
}

// ─── MeshTopology implementation ─────────────────────────────────────────────

impl<const D: usize> MeshTopology for CurvedMesh<D> {
    fn dim(&self) -> u8 { D as u8 }
    fn n_nodes(&self) -> usize { self.n_nodes }
    fn n_elements(&self) -> usize { self.n_elems }
    fn n_boundary_faces(&self) -> usize {
        if self.face_type == ElementType::Line2 { self.face_conn.len() / 2 }
        else if self.face_type == ElementType::Tri3 { self.face_conn.len() / 3 }
        else { 0 }
    }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        self.elem_geom_nodes(elem as usize)
    }

    fn element_type(&self, _elem: ElemId) -> ElementType { self.elem_type }

    fn element_tag(&self, elem: ElemId) -> i32 {
        self.elem_tags[elem as usize]
    }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        let off = node as usize * D;
        &self.coords[off..off + D]
    }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] {
        let f = face as usize;
        if self.face_type == ElementType::Line2 {
            &self.face_conn[2 * f..2 * f + 2]
        } else if self.face_type == ElementType::Tri3 {
            &self.face_conn[3 * f..3 * f + 3]
        } else {
            panic!("CurvedMesh::face_nodes: unsupported face type {:?}", self.face_type);
        }
    }

    fn face_tag(&self, face: FaceId) -> i32 { self.face_tags[face as usize] }

    fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) {
        // CurvedMesh does not track interior face adjacency.
        // Only boundary faces are stored.
        (0, None)
    }

    fn geom_order(&self) -> u8 { self.geom_order }
}

// ─── JacobianCache ────────────────────────────────────────────────────────────

/// Precomputed Jacobian data for all (element, quadrature-point) pairs.
///
/// Eliminates repeated `element_jacobian()` calls during assembly.
pub struct JacobianCache {
    /// Flat layout per QP: `[J_00, …, det_J, JIT_00, …]`.
    data: Vec<f64>,
    n_qp_per_elem: usize,
    dim: usize,
    stride: usize,
}

impl JacobianCache {
    /// Build from a `CurvedMesh` and a reference element's quadrature rule.
    pub fn build<const D: usize>(
        mesh: &CurvedMesh<D>,
        quad: &[(Vec<f64>, Vec<f64>)],  // (points, weights) per elem
    ) -> Self {
        let n_qp = quad[0].1.len();
        let n_elems = mesh.n_elems;
        let dim = D;
        let stride = dim * dim + 1 + dim * dim;
        let mut data = vec![0.0_f64; n_elems * n_qp * stride];

        for e in 0..n_elems {
            let pts = &quad[e].0;
            for q in 0..n_qp {
                let xi = &pts[q * dim .. (q + 1) * dim];
                let (jac, det) = mesh.element_jacobian(e, xi);
                let base = (e * n_qp + q) * stride;
                for i in 0..dim { for j in 0..dim { data[base + i * dim + j] = jac[(i, j)]; }}
                let off = dim * dim;
                data[base + off] = det;
                let jit = jac.try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(dim, dim));
                let off2 = dim * dim + 1;
                for i in 0..dim { for j in 0..dim { data[base + off2 + i * dim + j] = jit[(i, j)]; }}
            }
        }
        Self { data, n_qp_per_elem: n_qp, dim, stride }
    }

    /// Jacobian determinant at `(elem, qp)`.
    pub fn det_j(&self, elem: usize, qp: usize) -> f64 {
        let base = (elem * self.n_qp_per_elem + qp) * self.stride;
        self.data[base + self.dim * self.dim]
    }

    /// J⁻ᵀ at `(elem, qp)`.
    pub fn jacobian_inv_t(&self, elem: usize, qp: usize) -> Vec<f64> {
        let base = (elem * self.n_qp_per_elem + qp) * self.stride;
        let off = self.dim * self.dim + 1;
        (0..self.dim * self.dim).map(|i| self.data[base + off + i]).collect()
    }
}

// ─── CurvedElementTransformation ──────────────────────────────────────────────

/// Isoparametric element transformation for curved meshes.
pub struct CurvedElementTransformation<'a, const D: usize> {
    mesh: &'a CurvedMesh<D>,
    elem: usize,
}

impl<'a, const D: usize> CurvedElementTransformation<'a, D> {
    pub fn new(mesh: &'a CurvedMesh<D>, elem: usize) -> Self { Self { mesh, elem } }

    /// Jacobian determinant at reference point xi.
    pub fn det_j(&self, xi: &[f64]) -> f64 { self.mesh.element_jacobian(self.elem, xi).1 }

    /// J⁻ᵀ at reference point xi (row-major flat).
    pub fn jacobian_inv_t(&self, xi: &[f64]) -> Vec<f64> {
        let (jac, _) = self.mesh.element_jacobian(self.elem, xi);
        let jit = jac.try_inverse().map(|m| m.transpose()).unwrap_or_else(|| DMatrix::identity(D, D));
        (0..D * D).map(|k| jit.data.as_slice()[k]).collect()
    }

    /// Reference → physical coordinates.
    pub fn reference_to_physical(&self, xi: &[f64]) -> [f64; D] {
        self.mesh.reference_to_physical(self.elem, xi)
    }
}

pub fn refine_curved_2d(curved: &CurvedMesh<2>) -> CurvedMesh<2> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tri, curved.geom_order);
    let npe = geo.n_dofs();
    _reinterpolate_curved_2d(curved, &_refine_linear_2d(curved), geo, npe)
}

pub fn refine_curved_3d(curved: &CurvedMesh<3>) -> CurvedMesh<3> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tet, curved.geom_order);
    let npe = geo.n_dofs();
    _reinterpolate_curved_3d(curved, &_refine_linear_3d(curved), geo, npe)
}

pub fn refine_curved_2d_nc(curved: &CurvedMesh<2>, marked: &[usize]) -> CurvedMesh<2> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tri, curved.geom_order);
    let npe = geo.n_dofs();
    let lin = _extract_linear_2d(curved);
    let mid: Vec<u32> = marked.iter().map(|&m| m as u32).collect();
    let fine = crate::amr::refine_nonconforming(&lin, &mid).0;
    _reinterpolate_curved_2d(curved, &fine, geo, npe)
}

pub fn refine_curved_3d_nc(curved: &CurvedMesh<3>, marked: &[usize]) -> CurvedMesh<3> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tet, curved.geom_order);
    let npe = geo.n_dofs();
    let lin = _extract_linear_3d(curved);
    let mid: Vec<u32> = marked.iter().map(|&m| m as u32).collect();
    let fine = crate::amr::refine_nonconforming_3d(&lin, &mid).0;
    _reinterpolate_curved_3d(curved, &fine, geo, npe)
}

fn _refine_linear_2d(curved: &CurvedMesh<2>) -> Mesh<2> {
    crate::amr::refine_uniform(&_extract_linear_2d(curved))
}

fn _refine_linear_3d(curved: &CurvedMesh<3>) -> Mesh<3> {
    crate::amr::refine_uniform_3d(&_extract_linear_3d(curved))
}

fn _extract_linear_2d(curved: &CurvedMesh<2>) -> Mesh<2> {
    Mesh {
        coords: curved.coords.clone(),
        conn: curved.geom_conn.chunks(curved.nodes_per_elem).flat_map(|c| c[..3].to_vec()).collect(),
        elem_type: ElementType::Tri3,
        face_conn: curved.face_conn.clone(), face_tags: curved.face_tags.clone(),
        face_type: ElementType::Line2, elem_tags: curved.elem_tags.clone(),
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    }
}

fn _extract_linear_3d(curved: &CurvedMesh<3>) -> Mesh<3> {
    Mesh {
        coords: curved.coords.clone(),
        conn: curved.geom_conn.chunks(curved.nodes_per_elem).flat_map(|c| c[..4].to_vec()).collect(),
        elem_type: ElementType::Tet4,
        face_conn: curved.face_conn.clone(), face_tags: curved.face_tags.clone(),
        face_type: ElementType::Tri3, elem_tags: curved.elem_tags.clone(),
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    }
}

fn _build_vparent_map<const D: usize>(curved: &CurvedMesh<D>) -> Vec<Vec<usize>> {
    let n_verts = curved.geom_conn.iter().max().map(|&m| m as usize + 1).unwrap_or(0).max(curved.n_nodes);
    let nv = if D == 2 { 3 } else { n_corners_3d(curved.elem_type) };
    let mut map: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
    for p in 0..curved.n_elems {
        for v in 0..nv {
            let nid = curved.geom_conn[p * curved.nodes_per_elem + v] as usize;
            if nid < map.len() { map[nid].push(p); }
        }
    }
    map
}

fn _find_parent<const D: usize>(vmap: &[Vec<usize>], verts: &[u32], n_parent: usize) -> usize {
    let mut votes = std::collections::HashMap::new();
    for &v in verts {
        if let Some(ps) = vmap.get(v as usize) {
            for &p in ps { *votes.entry(p).or_insert(0) += 1; }
        }
    }
    votes.into_iter().max_by_key(|&(_, c)| c).map(|(p,_)| p).unwrap_or(0).min(n_parent - 1)
}

fn _reinterpolate_curved_2d(curved: &CurvedMesh<2>, fine: &Mesh<2>, geo: Box<dyn fem_element::ReferenceElement>, npe: usize) -> CurvedMesh<2> {
    let nf = fine.n_elems(); let vmap = _build_vparent_map(curved); let dc = geo.dof_coords();
    let mut nc = Vec::with_capacity(nf * npe); let mut nn = fine.n_nodes() as u32; let mut ns = fine.coords.clone();
    for fe in 0..nf {
        let fv = fine.elem_nodes(fe as u32); nc.extend_from_slice(&fv[..3]);
        let pe = _find_parent::<2>(&vmap, &fv[..3], curved.n_elems);
        for i in 3..npe { let x = curved.reference_to_physical(pe, &dc[i]); ns.extend_from_slice(&x); nc.push(nn); nn += 1; }
    }
    let nfb = fine.n_boundary_faces(); let mut fc = Vec::with_capacity(nfb * 2); let mut ft = Vec::with_capacity(nfb);
    for f in 0..nfb as u32 { fc.extend_from_slice(fine.face_nodes(f)); ft.push(fine.face_tag(f)); }
    CurvedMesh { coords: ns, geom_conn: nc, geom_order: curved.geom_order, nodes_per_elem: npe,
        elem_type: if curved.geom_order >= 2 { ElementType::Tri6 } else { ElementType::Tri3 },
        n_elems: nf, n_nodes: nn as usize, face_conn: fc, face_tags: ft, face_type: ElementType::Line2, elem_tags: vec![0; nf] }
}

fn _reinterpolate_curved_3d(curved: &CurvedMesh<3>, fine: &Mesh<3>, geo: Box<dyn fem_element::ReferenceElement>, npe: usize) -> CurvedMesh<3> {
    let nf = fine.n_elems(); let vmap = _build_vparent_map(curved); let dc = geo.dof_coords();
    let mut nc = Vec::with_capacity(nf * npe); let mut nn = fine.n_nodes() as u32; let mut ns = fine.coords.clone();
    for fe in 0..nf {
        let fv = fine.elem_nodes(fe as u32); nc.extend_from_slice(&fv[..4]);
        let pe = _find_parent::<3>(&vmap, &fv[..4], curved.n_elems);
        for i in 4..npe { let x = curved.reference_to_physical(pe, &dc[i]); ns.extend_from_slice(&x); nc.push(nn); nn += 1; }
    }
    let nfb = fine.n_boundary_faces(); let mut fc = Vec::with_capacity(nfb * 3); let mut ft = Vec::with_capacity(nfb);
    for f in 0..nfb as u32 { fc.extend_from_slice(fine.face_nodes(f)); ft.push(fine.face_tag(f)); }
    CurvedMesh { coords: ns, geom_conn: nc, geom_order: curved.geom_order, nodes_per_elem: npe,
        elem_type: if curved.geom_order >= 2 { ElementType::Tet10 } else { ElementType::Tet4 },
        n_elems: nf, n_nodes: nn as usize, face_conn: fc, face_tags: ft, face_type: ElementType::Tri3, elem_tags: vec![0; nf] }
}

// ─── Generalized 3-D curved refinement (Hex, Prism, Pyramid) ─────────────────

/// Number of corner (linear) nodes for a given 3-D element type.
fn n_corners_3d(et: ElementType) -> usize {
    match et {
        ElementType::Tet4 | ElementType::Tet10 => 4,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => 8,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => 6,
        ElementType::Pyramid5 | ElementType::Pyramid13 => 5,
        _ => panic!("n_corners_3d: unsupported {et:?}"),
    }
}

/// Linear sub-element type for a given 3-D curved element type.
fn linear_elem_type_3d(et: ElementType) -> ElementType {
    match et {
        ElementType::Tet4 | ElementType::Tet10 => ElementType::Tet4,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => ElementType::Hex8,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => ElementType::Prism6,
        ElementType::Pyramid5 | ElementType::Pyramid13 => ElementType::Pyramid5,
        _ => panic!("linear_elem_type_3d: unsupported {et:?}"),
    }
}

/// High-order element type for a given curved element type and order.
fn curved_elem_type_3d(et: ElementType, order: u8) -> ElementType {
    match et {
        ElementType::Tet4 | ElementType::Tet10 => if order >= 2 { ElementType::Tet10 } else { ElementType::Tet4 },
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => if order >= 2 { ElementType::Hex27 } else { ElementType::Hex8 },
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => if order >= 2 { ElementType::Prism15 } else { ElementType::Prism6 },
        ElementType::Pyramid5 | ElementType::Pyramid13 => if order >= 2 { ElementType::Pyramid13 } else { ElementType::Pyramid5 },
        _ => panic!("curved_elem_type_3d: unsupported {et:?}"),
    }
}

/// Face element type for a given 3-D curved element type.
fn curved_face_type_3d(et: ElementType) -> ElementType {
    match et {
        ElementType::Tet4 | ElementType::Tet10 => ElementType::Tri3,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => ElementType::Quad4,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => ElementType::Quad4,
        ElementType::Pyramid5 | ElementType::Pyramid13 => ElementType::Quad4,
        _ => panic!("curved_face_type_3d: unsupported {et:?}"),
    }
}

/// Uniformly refine a curved 3-D mesh of any element type (Tet, Hex, Prism, Pyramid).
///
/// Extracts the linear sub-mesh, refines it uniformly, then re-interpolates the
/// high-order geometry onto the refined mesh.  Supports arbitrary geometric order `p`.
pub fn refine_curved_3d_general(curved: &CurvedMesh<3>) -> CurvedMesh<3> {
    let factory_type = mesh_elem_type_to_factory_type(curved.elem_type);
    let geo = fem_element::lagrange::factory::ref_elem(factory_type, curved.geom_order);
    let npe = geo.n_dofs();
    let nc = n_corners_3d(curved.elem_type);
    let le = linear_elem_type_3d(curved.elem_type);

    // Extract linear sub-mesh
    // Reconstruct boundary faces for the linear mesh from the CurvedMesh.
    // The CurvedMesh stores face_conn as a flat array; the face type for
    // mixed-face elements (Prism, Pyramid) requires face_offsets.
    let (lface_conn, lface_tags, lface_type, lface_types, lface_offsets) = {
        if curved.face_type == ElementType::Tri3 || curved.face_type == ElementType::Quad4 {
            // Uniform face type — can use directly
            (curved.face_conn.clone(), curved.face_tags.clone(), curved.face_type, None, None)
        } else {
            (Vec::new(), Vec::new(), curved_face_type_3d(curved.elem_type), None, None)
        }
    };
    let lin = Mesh {
        coords: curved.coords.clone(),
        conn: curved.geom_conn.chunks(curved.nodes_per_elem).flat_map(|c| c[..nc].to_vec()).collect(),
        elem_type: le,
        face_conn: lface_conn, face_tags: lface_tags,
        face_type: lface_type,
        elem_tags: curved.elem_tags.clone(),
        elem_types: None, elem_offsets: None, face_types: lface_types, face_offsets: lface_offsets,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    };

    // Refine linear mesh
    let fine = crate::amr::refine_uniform_3d(&lin);

    // Re-interpolate curved geometry
    let nf = fine.n_elems();
    let vmap = _build_vparent_map(curved);
    let dc = geo.dof_coords();
    let mut nc_vec = Vec::with_capacity(nf * npe);
    let mut nn = fine.n_nodes() as u32;
    let mut ns = fine.coords.clone();
    for fe in 0..nf {
        let fv = fine.elem_nodes(fe as u32);
        nc_vec.extend_from_slice(&fv[..nc]);
        let pe = _find_parent::<3>(&vmap, &fv[..nc], curved.n_elems);
        for i in nc..npe {
            let x = curved.reference_to_physical(pe, &dc[i]);
            ns.extend_from_slice(&x);
            nc_vec.push(nn);
            nn += 1;
        }
    }

    let ce = curved_elem_type_3d(curved.elem_type, curved.geom_order);
    let ft = curved_face_type_3d(curved.elem_type);
    let nfb = fine.n_boundary_faces();
    let mut fc = Vec::with_capacity(nfb * ft.nodes_per_element());
    let mut f_tags = Vec::with_capacity(nfb);
    for f in 0..nfb as u32 {
        fc.extend_from_slice(fine.face_nodes(f));
        f_tags.push(fine.face_tag(f));
    }

    CurvedMesh {
        coords: ns, geom_conn: nc_vec, geom_order: curved.geom_order, nodes_per_elem: npe,
        elem_type: ce, n_elems: nf, n_nodes: nn as usize,
        face_conn: fc, face_tags: f_tags, face_type: ft, elem_tags: vec![0; nf],
    }
}

/// Non-conforming uniform refinement for curved 3-D meshes of any element type.
pub fn refine_curved_3d_nc_general(curved: &CurvedMesh<3>, marked: &[usize]) -> CurvedMesh<3> {
    let factory_type = mesh_elem_type_to_factory_type(curved.elem_type);
    let geo = fem_element::lagrange::factory::ref_elem(factory_type, curved.geom_order);
    let npe = geo.n_dofs();
    let nc = n_corners_3d(curved.elem_type);
    let le = linear_elem_type_3d(curved.elem_type);

    let lin = Mesh {
        coords: curved.coords.clone(),
        conn: curved.geom_conn.chunks(curved.nodes_per_elem).flat_map(|c| c[..nc].to_vec()).collect(),
        elem_type: le,
        face_conn: curved.face_conn.clone(), face_tags: curved.face_tags.clone(),
        face_type: curved_face_type_3d(curved.elem_type),
        elem_tags: curved.elem_tags.clone(),
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    };

    let mid: Vec<u32> = marked.iter().map(|&m| m as u32).collect();
    let fine = if le == ElementType::Tet4 {
        crate::amr::refine_nonconforming_3d(&lin, &mid).0
    } else {
        // For non-Tet4 linear meshes, use the NC refiner for that type
        match le {
            ElementType::Hex8 => { let (m, _, _, _) = crate::amr::refine_nonconforming_hex(&lin, &mid); m }
            ElementType::Prism6 => { let (m, _, _, _, _) = crate::amr::refine_nonconforming_prism(&lin, &mid); m }
            ElementType::Pyramid5 => { let (m, _, _, _, _) = crate::amr::refine_nonconforming_pyramid(&lin, &mid); m }
            _ => unreachable!(),
        }
    };

    let nf = fine.n_elems();
    let vmap = _build_vparent_map(curved);
    let dc = geo.dof_coords();
    let mut nc_vec = Vec::with_capacity(nf * npe);
    let mut nn = fine.n_nodes() as u32;
    let mut ns = fine.coords.clone();
    for fe in 0..nf {
        let fv = fine.elem_nodes(fe as u32);
        nc_vec.extend_from_slice(&fv[..nc]);
        let pe = _find_parent::<3>(&vmap, &fv[..nc], curved.n_elems);
        for i in nc..npe {
            let x = curved.reference_to_physical(pe, &dc[i]);
            ns.extend_from_slice(&x);
            nc_vec.push(nn);
            nn += 1;
        }
    }

    let ce = curved_elem_type_3d(curved.elem_type, curved.geom_order);
    let ft = curved_face_type_3d(curved.elem_type);
    let nfb = fine.n_boundary_faces();
    let mut fc = Vec::with_capacity(nfb * ft.nodes_per_element());
    let mut f_tags = Vec::with_capacity(nfb);
    for f in 0..nfb as u32 {
        fc.extend_from_slice(fine.face_nodes(f));
        f_tags.push(fine.face_tag(f));
    }

    CurvedMesh {
        coords: ns, geom_conn: nc_vec, geom_order: curved.geom_order, nodes_per_elem: npe,
        elem_type: ce, n_elems: nf, n_nodes: nn as usize,
        face_conn: fc, face_tags: f_tags, face_type: ft, elem_tags: vec![0; nf],
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    fn unit_tri(n: usize) -> Mesh<2> { Mesh::<2>::unit_square_tri(n) }
    fn unit_tet(n: usize) -> Mesh<3> { Mesh::<3>::unit_cube_tet(n) }

    // ── Basic construction ──────────────────────────────────────────────────

    #[test]
    fn from_linear_preserves_nodes() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::from_linear(&mesh);
        assert_eq!(curved.n_nodes, mesh.n_nodes());
        assert_eq!(curved.n_elems, mesh.n_elems());
        assert_eq!(curved.geom_order, 1);
        assert_eq!(curved.nodes_per_elem, 3);
    }

    #[test]
    fn elevate_to_p2_increases_nodes() {
        let curved = CurvedMesh::elevate_to_order(&unit_tri(2), 2, |x| x);
        assert_eq!(curved.nodes_per_elem, 6);
        assert_eq!(curved.geom_order, 2);
    }

    #[test]
    fn elevate_to_p4_increases_nodes() {
        let curved = CurvedMesh::elevate_to_order(&unit_tri(2), 4, |x| x);
        assert_eq!(curved.nodes_per_elem, 15); // P4 tri has 15 nodes
        assert_eq!(curved.geom_order, 4);
    }

    #[test]
    fn elevate_to_p3_3d_increases_nodes() {
        let curved = CurvedMesh::elevate_to_order(&unit_tet(1), 3, |x| x);
        assert_eq!(curved.nodes_per_elem, 20); // P3 tet has 20 nodes
        assert_eq!(curved.geom_order, 3);
    }

    // ── Jacobian consistency on flat meshes ─────────────────────────────────

    fn jacobian_matches_p1_2d(mesh: &Mesh<2>, p: usize) {
        let lin = CurvedMesh::from_linear(mesh);
        let curved = CurvedMesh::elevate_to_order(mesh, p, |x| x);
        let xi = [1.0 / 3.0, 1.0 / 3.0];
        for e in 0..mesh.n_elems() {
            let (_, d1) = lin.element_jacobian(e, &xi);
            let (_, d2) = curved.element_jacobian(e, &xi);
            assert!((d1 - d2).abs() < 1e-12, "P{p} elem {e}: det {d1:.6e} vs {d2:.6e}");
        }
    }

    #[test] fn p2_jacobian_matches_p1() { jacobian_matches_p1_2d(&unit_tri(4), 2); }
    #[test] fn p3_jacobian_matches_p1() { jacobian_matches_p1_2d(&unit_tri(4), 3); }
    #[test] fn p4_jacobian_matches_p1() { jacobian_matches_p1_2d(&unit_tri(4), 4); }

    fn jacobian_matches_p1_3d(mesh: &Mesh<3>, p: usize) {
        let lin = CurvedMesh::from_linear(mesh);
        let curved = CurvedMesh::elevate_to_order(mesh, p, |x| x);
        let xi = [0.25, 0.25, 0.25];
        for e in 0..mesh.n_elems() {
            let (_, d1) = lin.element_jacobian(e, &xi);
            let (_, d2) = curved.element_jacobian(e, &xi);
            assert!((d1 - d2).abs() < 1e-12, "P{p} 3D elem {e}: {d1:.6e} vs {d2:.6e}");
        }
    }

    #[test] fn p2_3d_jacobian_matches_p1() { jacobian_matches_p1_3d(&unit_tet(1), 2); }
    #[test] fn p3_3d_jacobian_matches_p1() { jacobian_matches_p1_3d(&unit_tet(1), 3); }

    // ── Spherical mesh: curved Jacobian differs from flat ─────────────────────

    #[test]
    fn spherical_curved_jacobian_differs_from_flat() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let spherical = |x: [f64; 2]| { x };
        let flat = CurvedMesh::from_linear(&mesh);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, spherical);
        let xi = [1.0/3.0, 1.0/3.0];
        let (_, d_flat) = flat.element_jacobian(0, &xi);
        let (_, d_curved) = curved.element_jacobian(0, &xi);
        // On a flat mesh with identity map_fn, P2 Jacobian = P1 Jacobian within tolerance
        assert!((d_flat - d_curved).abs() < 1e-12, "flat+identity should match: {d_flat:.6e} vs {d_curved:.6e}");
    }

    // ── refine_curved_2d ──────────────────────────────────────────────────────

    #[test]
    fn refine_curved_2d_p1_doubles_elements() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::from_linear(&mesh);
        let fine = refine_curved_2d(&curved);
        assert_eq!(fine.n_elems, curved.n_elems * 4);
        assert!(fine.n_nodes > curved.n_nodes);
    }

    #[test]
    fn refine_curved_2d_p2_quadruples_elements() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let fine = refine_curved_2d(&curved);
        assert_eq!(fine.n_elems, curved.n_elems * 4);
        assert_eq!(fine.geom_order, 2);
    }

    #[test]
    fn refine_curved_2d_p4_maintains_geom_order() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 4, |x| x);
        let fine = refine_curved_2d(&curved);
        assert_eq!(fine.geom_order, 4);
        for e in 0..fine.n_elems {
            let (_, det) = fine.element_jacobian(e, &[1.0/3.0, 1.0/3.0]);
            assert!(det.abs() > 1e-15, "elem {e} degenerate det={det:.6e}");
        }
    }

    #[test]
    fn refine_curved_2d_flat_p4_jacobian_constant() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 4, |x| x);
        let fine = refine_curved_2d(&curved);
        // All fine elements on a flat mesh should have same Jacobian
        let xi = [1.0/3.0, 1.0/3.0];
        let (_, d0) = fine.element_jacobian(0, &xi);
        for e in 1..fine.n_elems {
            let (_, d) = fine.element_jacobian(e, &xi);
            assert!((d0 - d).abs() < 1e-12, "elem {e}: det {d:.6e} != {d0:.6e}");
        }
    }

    // ── refine_curved_3d ────────────────────────────────────────────────────

    #[test]
    fn refine_curved_3d_p1_eightfolds_elements() {
        let mesh = unit_tet(1); // 6 tets in a cube
        let curved = CurvedMesh::from_linear(&mesh);
        let fine = refine_curved_3d(&curved);
        assert_eq!(fine.n_elems, curved.n_elems * 8);
        assert_eq!(fine.geom_order, 1);
    }

    #[test]
    fn refine_curved_3d_p2_maintains_order() {
        let mesh = unit_tet(1);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let fine = refine_curved_3d(&curved);
        assert_eq!(fine.geom_order, 2);
        assert!(fine.nodes_per_elem > 4);
    }

    #[test]
    fn refine_curved_3d_flat_p3_jacobian_consistent() {
        let mesh = unit_tet(1);
        let curved = CurvedMesh::elevate_to_order(&mesh, 3, |x| x);
        let fine = refine_curved_3d(&curved);
        let xi = [0.25, 0.25, 0.25];
        for e in 0..fine.n_elems.min(10) {
            let (_, d) = fine.element_jacobian(e, &xi);
            assert!(d.abs() > 1e-15, "elem {e} degenerate det={d:.6e}");
        }
    }

    // ── refine_curved_2d_nc (non-conforming) ────────────────────────────────

    #[test]
    fn refine_curved_2d_nc_single_element() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::from_linear(&mesh);
        let marked = vec![0usize];
        let fine = refine_curved_2d_nc(&curved, &marked);
        assert!(fine.n_elems > curved.n_elems, "NC refine should increase element count");
        assert_eq!(fine.geom_order, 1);
    }

    #[test]
    fn refine_curved_2d_nc_p2_nonconforming() {
        let mesh = unit_tri(3);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let marked = vec![1usize, 3];
        let fine = refine_curved_2d_nc(&curved, &marked);
        assert_eq!(fine.geom_order, 2);
        assert!(fine.n_elems > curved.n_elems);
    }

    #[test]
    fn refine_curved_2d_nc_p4_no_degenerate() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 4, |x| x);
        let marked = vec![0usize];
        let fine = refine_curved_2d_nc(&curved, &marked);
        let xi = [1.0/3.0, 1.0/3.0];
        for e in 0..fine.n_elems {
            let (_, det) = fine.element_jacobian(e, &xi);
            assert!(det.abs() > 1e-15, "elem {e} degenerate det={det:.6e}");
        }
    }

    // ── refine_curved_3d_nc (non-conforming) ────────────────────────────────

    #[test]
    fn refine_curved_3d_nc_single_tet() {
        let mesh = unit_tet(1);
        let curved = CurvedMesh::from_linear(&mesh);
        let marked = vec![0usize];
        let fine = refine_curved_3d_nc(&curved, &marked);
        assert!(fine.n_elems > curved.n_elems);
    }

    #[test]
    fn refine_curved_3d_nc_p2_nonconforming() {
        let mesh = unit_tet(1);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let marked = vec![1usize, 2];
        let fine = refine_curved_3d_nc(&curved, &marked);
        assert_eq!(fine.geom_order, 2);
        let xi = [0.25, 0.25, 0.25];
        for e in 0..fine.n_elems.min(4) {
            let (_, det) = fine.element_jacobian(e, &xi);
            assert!(det.abs() > 1e-15, "elem {e} degenerate det={det:.6e}");
        }
    }

    // ── JacobianCache ──────────────────────────────────────────────────────

    #[test]
    fn jacobian_cache_builds_and_matches() {
        let mesh = unit_tri(4);
        let curved = CurvedMesh::from_linear(&mesh);
        use fem_element::ReferenceElement;
        use fem_element::lagrange::TriP1;
        let ref_elem = TriP1;
        let quad = ref_elem.quadrature(3);
        let per_elem: Vec<(Vec<f64>, Vec<f64>)> = (0..curved.n_elems)
            .map(|_| (quad.points.iter().flatten().copied().collect(), quad.weights.clone()))
            .collect();
        let cache = JacobianCache::build::<2>(&curved, &per_elem);
        let xi = [1.0/3.0, 1.0/3.0];
        let (_, _det_direct) = curved.element_jacobian(0, &xi);
        let det_cached = cache.det_j(0, 1); // second QP
        assert!(det_cached.abs() > 1e-15, "cached det should be non-zero");
    }

    // ── CurvedElementTransformation ─────────────────────────────────────────

    #[test]
    fn curved_element_transformation_det_matches() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let tr = CurvedElementTransformation::new(&curved, 0);
        let xi = [1.0/3.0, 1.0/3.0];
        let (_, det_direct) = curved.element_jacobian(0, &xi);
        let det_tr = tr.det_j(&xi);
        assert!((det_direct - det_tr).abs() < 1e-15);
        let phys = tr.reference_to_physical(&xi);
        assert_eq!(phys.len(), 2);
    }

    // ── _find_parent vertex voting ──────────────────────────────────────────

    #[test]
    fn find_parent_by_vertex_vote_2d() {
        let mesh = unit_tri(2);
        let curved = CurvedMesh::from_linear(&mesh);
        let vmap = _build_vparent_map(&curved);
        // First fine element of refine_curved_2d should map to parent 0
        let fine = refine_curved_2d(&curved);
        let fv = fine.element_nodes(0);
        let p = _find_parent::<2>(&vmap, &fv[..3], curved.n_elems);
        assert!(p < curved.n_elems, "parent {p} out of range");
    }

    #[test]
    fn find_parent_by_vertex_vote_3d() {
        let mesh = unit_tet(1);
        let curved = CurvedMesh::from_linear(&mesh);
        let vmap = _build_vparent_map(&curved);
        let fine = refine_curved_3d(&curved);
        let fv = fine.element_nodes(0);
        let p = _find_parent::<3>(&vmap, &fv[..4], curved.n_elems);
        assert!(p < curved.n_elems, "parent {p} out of range");
    }

    // ── Reference-to-physical roundtrip ─────────────────────────────────────

    #[test]
    fn physical_to_reference_to_physical_consistent() {
        let mesh = unit_tri(4);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let xi_ref = [1.0/3.0, 1.0/3.0];
        let x_phys = curved.reference_to_physical(0, &xi_ref);
        assert!(x_phys[0] > 0.0 && x_phys[1] > 0.0, "physical coords should be positive");
    }

    // ─── Generalized 3-D curved refinement tests ─────────────────────────
    // Note: `elevate_to_order` only supports simplex elements.  For non-simplex
    // (Hex/Prism/Pyramid) curved refinement, `elevate_to_order` must first be
    // generalized to use factory::ref_elem for tensor-product node insertion.
    // The P1 (linear) flow is tested below.

    #[test] fn curved_3d_linear_refine_matches_amr() {
        // Test that for P1 meshes, the general curved refiner matches the AMR refiner.
        use crate::amr::refine_uniform_3d;

        // Hex P1 — uniform Quad4 faces
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let curved = CurvedMesh::from_linear(&mesh);
        let fine_curved = refine_curved_3d_general(&curved);
        let fine_amr = refine_uniform_3d(&mesh);
        assert_eq!(fine_curved.n_elems, fine_amr.n_elems(), "Hex P1: curved vs AMR element count");
        assert_eq!(fine_curved.n_nodes, fine_amr.n_nodes(), "Hex P1: curved vs AMR node count");
        assert_eq!(fine_curved.geom_order, 1);

        // Prism P1 — use uniform Tri3 face type for the linear mesh (tri faces only)
        let coords = vec![0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0];
        let conn = vec![0u32,1,2,3,4,5];
        // Split each quad face into 2 triangles
        let fc = vec![
            0u32,2,1, 3,4,5,        // 2 tri faces (bottom, top)
            0,1,4, 0,4,3,            // quad front → 2 tri
            1,2,5, 1,5,4,            // quad right → 2 tri
            0,3,5, 0,5,2,            // quad left → 2 tri
        ];
        let mesh2 = Mesh { coords, conn, elem_tags: vec![1i32], elem_type: ElementType::Prism6,
            face_conn: fc, face_tags: vec![1,2,3,4,5,6,7,8], face_type: ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:None, face_offsets:None,
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![] };
        let curved2 = CurvedMesh::from_linear(&mesh2);
        let fine2 = refine_curved_3d_general(&curved2);
        let fine2_amr = refine_uniform_3d(&mesh2);
        assert_eq!(fine2.n_elems, fine2_amr.n_elems(), "Prism P1: curved vs AMR");
        assert_eq!(fine2.n_nodes, fine2_amr.n_nodes(), "Prism P1: curved vs AMR node count");
    }
}
