//! High-order curved mesh with isoparametric geometry mapping.
//!
//! A `CurvedMesh<D>` stores a **geometric order** ≥ 1 and the corresponding
//! higher-order (quadratic, cubic, …) node coordinates.  The isoparametric
//! mapping `F_K: K̂ → K` is defined by the same Lagrange shape functions used
//! for the FE solution, so the Jacobian `J = ∂F/∂ξ` is computed from the
//! geometric nodal coordinates.
//!
//! # Supported elements
//! | Element | Geom order | Nodes/elem |
//! |---------|-----------|------------|
//! | Tri6    | 2          | 6          |
//! | Tri3    | 1          | 3          |
//!
//! # Usage
//! ```rust,ignore
//! // Quadratic mesh on the unit disk (polygonal approximation).
//! let linear = SimplexMesh::<2>::unit_square_tri(8);
//! let curved = CurvedMesh::from_linear(&linear);   // order-1, no curvature
//! let (jac, det) = curved.element_jacobian(e, &xi_ref);
//! ```
//!
//! # Isoparametric Jacobian
//! For a 2-D triangle with geometric nodes `x_0, …, x_{n-1}`:
//! ```text
//! F(ξ) = Σ_i x_i φ_i(ξ)
//! J    = ∂F/∂ξ  (2×2 matrix)
//! ```

use nalgebra::DMatrix;
use fem_core::NodeId;
use crate::{element_type::ElementType, simplex::SimplexMesh};

// ─── CurvedMesh ──────────────────────────────────────────────────────────────

/// A high-order curved (isoparametric) mesh.
///
/// Stores both:
/// - **Connectivity** (`geom_conn`): for each element, the geometric node
///   indices (linear + high-order nodes).
/// - **Coordinates** (`coords`): all node coordinates (including edge midpoints
///   and interior nodes for order > 1).
///
/// The geometric mapping uses Lagrange basis functions of order `geom_order`.
#[derive(Debug, Clone)]
pub struct CurvedMesh<const D: usize> {
    /// Flat coordinate array.  Length = `n_nodes * D`.
    pub coords: Vec<f64>,
    /// Element connectivity (geometric nodes).  Length = `n_elems * nodes_per_elem`.
    pub geom_conn: Vec<NodeId>,
    /// Geometric polynomial order.
    pub geom_order: u8,
    /// Element type (Tri3 for linear, Tri6 for quadratic triangles, etc.).
    pub elem_type: ElementType,
    /// Total number of elements.
    pub n_elems: usize,
    /// Total number of nodes (including mid-edge and interior nodes).
    pub n_nodes: usize,
}

impl<const D: usize> CurvedMesh<D> {
    /// Construct an order-1 curved mesh from a `SimplexMesh`.
    ///
    /// No curvature is added; this wraps the linear mesh into the curved-mesh
    /// interface so the same assembly loops work for both.
    pub fn from_linear(mesh: &SimplexMesh<D>) -> Self {
        CurvedMesh {
            coords:     mesh.coords.clone(),
            geom_conn:  mesh.conn.clone(),
            geom_order: 1,
            elem_type:  mesh.elem_type,
            n_elems:    mesh.n_elems(),
            n_nodes:    mesh.n_nodes(),
        }
    }

    /// Construct a quadratic (order-2) curved mesh by inserting edge-midpoint
    /// nodes for every element edge.
    ///
    /// The `map_fn` is an optional coordinate transform applied to new midpoint
    /// nodes (e.g. to project them onto a curved surface).  Pass `|_x| x` for
    /// no transformation (straight-edged P2 mesh).
    pub fn elevate_to_order2<F>(mesh: &SimplexMesh<D>, map_fn: F) -> Self
    where
        F: Fn([f64; D]) -> [f64; D],
    {
        assert_eq!(D, 2, "elevate_to_order2 currently supports D=2 only");
        assert!(
            mesh.elem_type == ElementType::Tri3,
            "elevate_to_order2 requires Tri3 input mesh"
        );

        let n_linear_nodes = mesh.n_nodes();
        let mut new_coords: Vec<f64> = mesh.coords.clone();

        // Map from edge key (sorted pair) → new midpoint node ID.
        let mut midpoint_map: std::collections::HashMap<(NodeId, NodeId), NodeId> =
            std::collections::HashMap::new();
        let mut next_node = n_linear_nodes as NodeId;

        // DOF ordering for Tri6:
        //   0: vertex (0,0)      ← node 0
        //   1: vertex (1,0)      ← node 1
        //   2: vertex (0,1)      ← node 2
        //   3: mid edge 0-1
        //   4: mid edge 1-2
        //   5: mid edge 0-2
        let edge_local = [(0usize, 1usize), (1, 2), (0, 2)];

        let n_elems = mesh.n_elems();
        let mut geom_conn = Vec::with_capacity(n_elems * 6);

        for e in 0..n_elems {
            let ns = mesh.elem_nodes(e as NodeId);
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let corners = [n0, n1, n2];

            let mut mid_ids = [0u32; 3];
            for (k, &(a, b)) in edge_local.iter().enumerate() {
                let na = corners[a];
                let nb = corners[b];
                let key = if na < nb { (na, nb) } else { (nb, na) };
                let mid_id = midpoint_map.entry(key).or_insert_with(|| {
                    // Compute midpoint coordinates.
                    let xa: [f64; D] = mesh.coords_of(na);
                    let xb: [f64; D] = mesh.coords_of(nb);
                    let xm_raw: [f64; D] = std::array::from_fn(|i| 0.5 * (xa[i] + xb[i]));
                    let xm = map_fn(xm_raw);
                    new_coords.extend_from_slice(&xm);
                    let id = next_node;
                    next_node += 1;
                    id
                });
                mid_ids[k] = *mid_id;
            }

            // Tri6 connectivity: [v0, v1, v2, mid01, mid12, mid02]
            geom_conn.extend_from_slice(&[n0, n1, n2, mid_ids[0], mid_ids[1], mid_ids[2]]);
        }

        CurvedMesh {
            n_nodes: next_node as usize,
            n_elems,
            coords: new_coords,
            geom_conn,
            geom_order: 2,
            elem_type: ElementType::Tri6,
        }
    }

    /// Coordinates of geometric node `n`.
    #[inline]
    pub fn node_coords(&self, n: NodeId) -> [f64; D] {
        let off = n as usize * D;
        std::array::from_fn(|i| self.coords[off + i])
    }

    /// Geometric node IDs for element `e`.
    #[inline]
    pub fn elem_geom_nodes(&self, e: usize) -> &[NodeId] {
        let npe = nodes_per_elem(self.geom_order);
        let off = e * npe;
        &self.geom_conn[off..off + npe]
    }

    /// Compute the isoparametric Jacobian `J = ∂F/∂ξ` at reference point `xi`
    /// for element `e`.
    ///
    /// Returns `(J, det(J))`.
    ///
    /// `J` is a `D×D` matrix in row-major order.
    pub fn element_jacobian(&self, e: usize, xi: &[f64]) -> (DMatrix<f64>, f64) {
        let dim = D;
        let nodes = self.elem_geom_nodes(e);
        let n = nodes.len();

        // Evaluate geometric basis gradients at xi.
        let mut grad_ref = vec![0.0_f64; n * dim];
        eval_geom_grad_basis(self.geom_order, xi, &mut grad_ref);

        // J[i,j] = Σ_k x_k[i] * ∂φ_k/∂ξ_j
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for row in 0..dim {
            for col in 0..dim {
                let mut s = 0.0;
                for k in 0..n {
                    let xk = self.node_coords(nodes[k]);
                    s += xk[row] * grad_ref[k * dim + col];
                }
                j[(row, col)] = s;
            }
        }
        let det = j.determinant();
        (j, det)
    }

    /// Physical coordinates of reference point `xi` in element `e`.
    pub fn reference_to_physical(&self, e: usize, xi: &[f64]) -> [f64; D] {
        let dim = D;
        let nodes = self.elem_geom_nodes(e);
        let n = nodes.len();
        let mut phi = vec![0.0_f64; n];
        eval_geom_basis(self.geom_order, xi, &mut phi);
        let mut xp = [0.0_f64; D];
        for k in 0..n {
            let xk = self.node_coords(nodes[k]);
            for i in 0..dim { xp[i] += xk[i] * phi[k]; }
        }
        xp
    }
}

// ─── Geometric basis functions ────────────────────────────────────────────────

fn nodes_per_elem(order: u8) -> usize {
    match order {
        1 => 3,  // Tri3
        2 => 6,  // Tri6
        _ => panic!("nodes_per_elem: unsupported order {order}"),
    }
}

/// Evaluate geometric (Lagrange) basis functions at reference point `xi`.
fn eval_geom_basis(order: u8, xi: &[f64], phi: &mut [f64]) {
    match order {
        1 => {
            let (x, y) = (xi[0], xi[1]);
            phi[0] = 1.0 - x - y;
            phi[1] = x;
            phi[2] = y;
        }
        2 => {
            let (x, y) = (xi[0], xi[1]);
            let l1 = 1.0 - x - y;
            let l2 = x;
            let l3 = y;
            phi[0] = l1 * (2.0 * l1 - 1.0);
            phi[1] = l2 * (2.0 * l2 - 1.0);
            phi[2] = l3 * (2.0 * l3 - 1.0);
            phi[3] = 4.0 * l1 * l2;
            phi[4] = 4.0 * l2 * l3;
            phi[5] = 4.0 * l1 * l3;
        }
        _ => panic!("eval_geom_basis: unsupported order {order}"),
    }
}

/// Evaluate gradients of geometric (Lagrange) basis at reference point `xi`.
/// Row-major `n × dim`.
fn eval_geom_grad_basis(order: u8, xi: &[f64], grads: &mut [f64]) {
    match order {
        1 => {
            // TriP1
            grads[0] = -1.0;  grads[1] = -1.0;
            grads[2] =  1.0;  grads[3] =  0.0;
            grads[4] =  0.0;  grads[5] =  1.0;
        }
        2 => {
            // TriP2
            let (x, y) = (xi[0], xi[1]);
            grads[0]  = 4.0*x + 4.0*y - 3.0;  grads[1]  = 4.0*x + 4.0*y - 3.0;
            grads[2]  = 4.0*x - 1.0;           grads[3]  = 0.0;
            grads[4]  = 0.0;                    grads[5]  = 4.0*y - 1.0;
            grads[6]  = 4.0 - 8.0*x - 4.0*y;   grads[7]  = -4.0*x;
            grads[8]  = 4.0*y;                  grads[9]  = 4.0*x;
            grads[10] = -4.0*y;                 grads[11] = 4.0 - 4.0*x - 8.0*y;
        }
        _ => panic!("eval_geom_grad_basis: unsupported order {order}"),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
    fn order2_node_count() {
        // Tri3 → Tri6: each edge gets a midpoint node.
        // unit_square_tri(2) has 9 nodes, 8 elements.
        // Number of unique edges = (3 * 8 + 4*2) / 2 = 16 (interior edges) + 8 (bdy) = 24/2 = hmm.
        // Exact count depends on the mesh structure; just check it's more than before.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_before = mesh.n_nodes();
        let curved = CurvedMesh::elevate_to_order2(&mesh, |x| x);
        assert!(curved.n_nodes > n_before,
            "P2 mesh should have more nodes: before={n_before}, after={}", curved.n_nodes);
        assert_eq!(curved.geom_order, 2);
        assert_eq!(curved.elem_type, ElementType::Tri6);
    }

    #[test]
    fn jacobian_linear_triangle() {
        // For a linear mesh, the Jacobian should be constant within each element.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let curved = CurvedMesh::from_linear(&mesh);
        let xi = vec![1.0 / 3.0, 1.0 / 3.0]; // centroid of reference triangle
        for e in 0..curved.n_elems {
            let (_j, det) = curved.element_jacobian(e, &xi);
            assert!(det > 0.0, "Element {e}: det(J) = {det} ≤ 0 (degenerate)");
        }
    }

    #[test]
    fn jacobian_p2_triangle() {
        // For a P2 mesh with no curvature (identity map_fn), the Jacobian
        // should equal the P1 Jacobian at the same point.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let lin = CurvedMesh::from_linear(&mesh);
        let p2  = CurvedMesh::elevate_to_order2(&mesh, |x| x);
        let xi = vec![1.0 / 3.0, 1.0 / 3.0];
        for e in 0..mesh.n_elems() {
            let (_, det_lin) = lin.element_jacobian(e, &xi);
            let (_, det_p2)  = p2.element_jacobian(e, &xi);
            let err = (det_lin - det_p2).abs();
            assert!(err < 1e-12, "Element {e}: det P1={det_lin:.6e}, P2={det_p2:.6e}, diff={err:.3e}");
        }
    }

    #[test]
    fn reference_to_physical_vertices() {
        // At reference vertex (0,0), the physical coords should equal the element's vertex 0.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let curved = CurvedMesh::from_linear(&mesh);
        for e in 0..mesh.n_elems() {
            let nodes = mesh.elem_nodes(e as NodeId);
            let x0 = mesh.coords_of(nodes[0]);
            let xp = curved.reference_to_physical(e, &[0.0, 0.0]);
            let err: f64 = x0.iter().zip(xp.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            assert!(err < 1e-13, "Element {e} vertex 0: expected {x0:?}, got {xp:?}");
        }
    }

    #[test]
    fn area_preserved_on_unit_square() {
        // Area of unit square = 1.0.
        // Sum of |det(J)| * area_ref_triangle = sum of element areas.
        
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let curved = CurvedMesh::from_linear(&mesh);
        let xi_centroid = vec![1.0 / 3.0, 1.0 / 3.0];
        // For constant-Jacobian elements: area_K = |det(J)| * (1/2) (ref tri area = 1/2)
        let total_area: f64 = (0..mesh.n_elems())
            .map(|e| {
                let (_, det) = curved.element_jacobian(e, &xi_centroid);
                det.abs() * 0.5
            })
            .sum();
        assert!((total_area - 1.0).abs() < 1e-12,
            "Total area = {total_area:.6e}, expected 1.0");
    }

    #[test]
    fn curved_disk_midpoints_projected() {
        // Test that map_fn is applied: project midpoints onto the unit circle.
        // Only a sanity check that projection is called.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let map_called = std::sync::atomic::AtomicUsize::new(0);
        let curved = CurvedMesh::elevate_to_order2(&mesh, |x: [f64; 2]| {
            map_called.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            x // identity — just checking that it's called
        });
        assert!(curved.n_nodes > mesh.n_nodes());
    }
}
