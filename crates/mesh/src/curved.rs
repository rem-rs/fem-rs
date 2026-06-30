//! High-order curved mesh with isoparametric geometry mapping.
//!
//! A `CurvedMesh<D>` stores a **geometric order** �?1 and the corresponding
//! higher-order (quadratic, cubic, �? node coordinates.  The isoparametric
//! mapping `F_K: K̂ �?K` is defined by the same Lagrange shape functions used
//! for the FE solution, so the Jacobian `J = ∂F/∂ξ` is computed from the
//! geometric nodal coordinates.
//!
//! # Isoparametric Jacobian
//! For a 2-D triangle with geometric nodes `x_0, �? x_{n-1}`:
//! ```text
//! F(ξ) = Σ�?x�?φ�?ξ),    J = ∂F/∂�? (2×2 matrix)
//! ```

use fem_core::{ElemId, FaceId, NodeId};
use nalgebra::DMatrix;
use crate::element_type::ElementType;
use crate::simplex::SimplexMesh;
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
    /// Construct an order-1 curved mesh from a `SimplexMesh`.
    pub fn from_linear(mesh: &SimplexMesh<D>) -> Self {
        let npe = if D == 2 { 3 } else { 4 };
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
    pub fn elevate_to_order<F>(mesh: &SimplexMesh<D>, p: usize, map_fn: F) -> Self
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
            let _base = e as usize * npe_new;
            let off = geom_conn.len();
            geom_conn.resize(off + npe_new, 0);

            // Copy vertex nodes
            for i in 0..=dim { geom_conn[off + i] = ns[i]; }

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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::SimplexMesh;

    #[test]
    fn from_linear_preserves_geometry() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let curved = CurvedMesh::from_linear(&mesh);
        assert_eq!(curved.n_elems, mesh.n_elems());
        assert_eq!(curved.n_nodes, mesh.n_nodes());
        assert_eq!(curved.geom_order, 1);
    }

    #[test]
    fn elevate_to_p2_increases_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        assert!(curved.n_nodes > mesh.n_nodes());
        assert_eq!(curved.geom_order, 2);
        assert_eq!(curved.nodes_per_elem, 6);
    }

    #[test]
    fn elevate_to_p4_increases_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 4, |x| x);
        // P4 triangle: (4+1)(4+2)/2 = 15 nodes per element
        assert_eq!(curved.nodes_per_elem, 15);
        assert_eq!(curved.geom_order, 4);
    }

    #[test]
    fn jacobian_positive_linear() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let curved = CurvedMesh::from_linear(&mesh);
        let xi = vec![1.0 / 3.0, 1.0 / 3.0];
        for e in 0..curved.n_elems {
            let (_j, det) = curved.element_jacobian(e, &xi);
            assert!(det > 0.0, "Element {e}: det(J) = {det}");
        }
    }

    #[test]
    fn p2_jacobian_matches_p1_on_flat_mesh() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let lin = CurvedMesh::from_linear(&mesh);
        let p2 = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let xi = vec![1.0 / 3.0, 1.0 / 3.0];
        for e in 0..mesh.n_elems() {
            let (_, det_lin) = lin.element_jacobian(e, &xi);
            let (_, det_p2) = p2.element_jacobian(e, &xi);
            assert!((det_lin - det_p2).abs() < 1e-12,
                "elem {e}: P1={det_lin:.6e}, P2={det_p2:.6e}");
        }
    }

    #[test]
    fn tet4_linear_mesh() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let curved = CurvedMesh::from_linear(&mesh);
        let xi = vec![0.25, 0.25, 0.25];
        for e in 0..curved.n_elems.min(10) {
            let (_j, det) = curved.element_jacobian(e, &xi);
            assert!(det > 0.0, "Tet {e}: det(J) = {det}");
        }
    }

    #[test]
    fn elevate_tet_to_p2() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let curved = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        assert!(curved.n_nodes > mesh.n_nodes());
        assert_eq!(curved.geom_order, 2);
        // P4 tet: (2+1)(2+2)(2+3)/6 = 10 nodes per element
        assert_eq!(curved.nodes_per_elem, 10);
    }

    #[test]
    fn p2_tet_jacobian_matches_p1() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let lin = CurvedMesh::from_linear(&mesh);
        let p2 = CurvedMesh::elevate_to_order(&mesh, 2, |x| x);
        let xi = vec![0.25, 0.25, 0.25];
        for e in 0..mesh.n_elems().min(10) {
            let (_, det_lin) = lin.element_jacobian(e, &xi);
            let (_, det_p2) = p2.element_jacobian(e, &xi);
            assert!((det_lin - det_p2).abs() < 1e-12);
        }
    }

    #[test]
    fn p3_tet_jacobian_matches_p1() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let lin = CurvedMesh::from_linear(&mesh);
        let p3 = CurvedMesh::elevate_to_order(&mesh, 3, |x| x);
        let xi = vec![0.25, 0.25, 0.25];
        for e in 0..mesh.n_elems() {
            let (_, det_lin) = lin.element_jacobian(e, &xi);
            let (_, det_p3) = p3.element_jacobian(e, &xi);
            assert!((det_lin - det_p3).abs() < 1e-12,
                "elem {e}: P1={det_lin:.6e}, P3={det_p3:.6e}");
        }
    }



    #[test]
    fn p3_jacobian_matches_p1_on_flat_mesh() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let lin = CurvedMesh::from_linear(&mesh);
        let p3 = CurvedMesh::elevate_to_order(&mesh, 3, |x| x);
        let xi = vec![1.0 / 3.0, 1.0 / 3.0];
        for e in 0..mesh.n_elems() {
            let (_, det_lin) = lin.element_jacobian(e, &xi);
            let (_, det_p3) = p3.element_jacobian(e, &xi);
            assert!((det_lin - det_p3).abs() < 1e-12,
                "elem {e}: P1={det_lin:.6e}, P3={det_p3:.6e}");
        }
    }

    #[test]
    fn p4_jacobian_matches_p1_on_flat_mesh() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let lin = CurvedMesh::from_linear(&mesh);
        let p4 = CurvedMesh::elevate_to_order(&mesh, 4, |x| x);
        let xi = vec![1.0 / 3.0, 1.0 / 3.0];
        for e in 0..mesh.n_elems() {
            let (_, det_lin) = lin.element_jacobian(e, &xi);
            let (_, det_p4) = p4.element_jacobian(e, &xi);
            assert!((det_lin - det_p4).abs() < 1e-12,
                "elem {e}: P1={det_lin:.6e}, P4={det_p4:.6e}");
        }
    }
}
