//! H¹ trace finite element space on mesh boundaries.
//!
//! `H1TraceSpace` represents the trace of the H¹ space on boundary faces.
//! Each boundary face has DOFs corresponding to its vertex (P1) or vertex+edge
//! (P2) nodes.  The global DOF numbering is consecutive over boundary nodes only.
//!
//! This space is useful for:
//! - Boundary element methods (BEM)
//! - Enforcing trace constraints in domain decomposition
//! - Robin / impedance boundary conditions

use std::collections::BTreeSet;

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::EdgeKey;
use crate::fe_space::{FESpace, SpaceType};

/// Trace of H¹ on the mesh boundary.
///
/// DOFs are placed at boundary nodes.  P1 trace has one DOF per boundary
/// vertex; P2 trace additionally has one DOF per boundary edge midpoint.
///
/// For P1, `n_dofs` = number of unique boundary vertices.
pub struct H1TraceSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    /// Total number of trace DOFs.
    n_dofs: usize,
    /// Map from mesh NodeId → trace DOF index (u32::MAX if not on boundary).
    node_to_dof: Vec<DofId>,
    /// Flat per-face DOF arrays.  `face_dofs[f * dpf .. (f+1) * dpf]`.
    face_dofs: Vec<DofId>,
    /// DOFs per boundary face.
    dofs_per_face: usize,
    /// Boundary node coordinates (flat, `n_dofs * dim`).
    dof_coords: Vec<f64>,
    /// Number of boundary faces.
    n_bfaces: usize,
}

impl<M: MeshTopology> H1TraceSpace<M> {
    /// Build an H¹ trace space of order 1, 2, or 3 on the boundary faces of `mesh`.
    ///
    /// - P1 trace (`order = 1`): one DOF per boundary vertex.
    /// - P2 trace (`order = 2`): one DOF per boundary vertex + one per boundary edge midpoint.
    /// - P3 trace (`order = 3`): one DOF per boundary vertex + two per boundary edge interior
    ///   (at ⅓ and ⅔) + one per boundary triangle interior (3D only).
    ///
    /// # Panics
    /// Panics if `order > 3`.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!(order <= 3, "H1TraceSpace: only orders 1–3 supported, got {order}");
        let dim = mesh.dim() as usize;
        let n_bfaces = mesh.n_boundary_faces();

        // Collect all boundary nodes
        let mut bdr_nodes = BTreeSet::new();
        for f in 0..n_bfaces as u32 {
            for &n in mesh.face_nodes(f) {
                bdr_nodes.insert(n);
            }
        }

        // Map mesh node → trace DOF (vertex DOFs 0..n_vertex_dofs)
        let n_nodes = mesh.n_nodes();
        let mut node_to_dof = vec![u32::MAX; n_nodes];
        let mut dof_coords = Vec::new();
        let mut next_dof = 0u32;
        for &n in &bdr_nodes {
            node_to_dof[n as usize] = next_dof;
            let c = mesh.node_coords(n);
            dof_coords.extend_from_slice(&c[..dim]);
            next_dof += 1;
        }

        // For P2: assign DOFs for boundary edge midpoints.
        let mut edge_to_dof: std::collections::HashMap<EdgeKey, DofId> = std::collections::HashMap::new();
        if order == 2 {
            for f in 0..n_bfaces as u32 {
                let nodes = mesh.face_nodes(f);
                let edges: Vec<(u32, u32)> = if nodes.len() == 2 {
                    vec![(nodes[0], nodes[1])]
                } else {
                    vec![(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])]
                };
                for (a, b) in edges {
                    let key = EdgeKey::new(a, b);
                    edge_to_dof.entry(key).or_insert_with(|| {
                        let d = next_dof;
                        next_dof += 1;
                        let ca = mesh.node_coords(key.0);
                        let cb = mesh.node_coords(key.1);
                        for di in 0..dim { dof_coords.push(0.5 * (ca[di] + cb[di])); }
                        d
                    });
                }
            }
        }

        // For P3: assign two DOFs per boundary edge (at 1/3 and 2/3),
        // and one DOF per boundary triangle face (face centroid, 3D only).
        let mut edge_to_dof2: std::collections::HashMap<EdgeKey, [DofId; 2]> =
            std::collections::HashMap::new();
        let mut face_bubble_dof: Vec<DofId> = Vec::new(); // indexed by boundary face index
        if order == 3 {
            // Edge DOFs (two per edge, at t=1/3 and t=2/3 from the smaller-index vertex)
            for f in 0..n_bfaces as u32 {
                let nodes = mesh.face_nodes(f);
                let edges: Vec<(u32, u32)> = if nodes.len() == 2 {
                    vec![(nodes[0], nodes[1])]
                } else {
                    vec![(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])]
                };
                for (a, b) in edges {
                    let key = EdgeKey::new(a, b);
                    edge_to_dof2.entry(key).or_insert_with(|| {
                        let d0 = next_dof; next_dof += 1;
                        let d1 = next_dof; next_dof += 1;
                        // At 1/3 and 2/3 from key.0 to key.1
                        let ca = mesh.node_coords(key.0);
                        let cb = mesh.node_coords(key.1);
                        for di in 0..dim { dof_coords.push(ca[di]/3.0*2.0 + cb[di]/3.0); }
                        for di in 0..dim { dof_coords.push(ca[di]/3.0 + cb[di]/3.0*2.0); }
                        [d0, d1]
                    });
                }
            }
            // Face bubble DOFs (3D only: one per boundary triangle face at centroid)
            for f in 0..n_bfaces as u32 {
                let nodes = mesh.face_nodes(f);
                if nodes.len() >= 3 {
                    // Face centroid
                    let d = next_dof; next_dof += 1;
                    face_bubble_dof.push(d);
                    for di in 0..dim {
                        let avg = nodes.iter().map(|&n| mesh.node_coords(n)[di]).sum::<f64>()
                            / nodes.len() as f64;
                        dof_coords.push(avg);
                    }
                } else {
                    face_bubble_dof.push(u32::MAX); // no bubble for edge faces (2D)
                }
            }
        }

        let n_dofs = next_dof as usize;

        // Build per-face DOF connectivity.
        // P1 in 2D: 2 DOFs/face; P1 in 3D: 3 DOFs/face.
        // P2 in 2D: 3 DOFs/face; P2 in 3D: 6 DOFs/face.
        // P3 in 2D: 4 DOFs/face (2 verts + 2 edge); P3 in 3D: 10 DOFs/face.
        let n_verts_per_face = if n_bfaces > 0 { mesh.face_nodes(0).len() } else { 2 };
        let dofs_per_face = match (order, n_verts_per_face) {
            (1, v) => v,
            (2, 2) => 3,   // 2 verts + 1 edge midpoint
            (2, _) => 6,   // 3 verts + 3 edge midpoints
            (3, 2) => 4,   // 2 verts + 2 edge DOFs
            (3, _) => 10,  // 3 verts + 3*2 edge DOFs + 1 face bubble
            _      => n_verts_per_face,
        };

        let mut face_dofs = Vec::with_capacity(n_bfaces * dofs_per_face);
        for f in 0..n_bfaces as u32 {
            let nodes = mesh.face_nodes(f);
            // Vertex DOFs first.
            for &n in nodes {
                face_dofs.push(node_to_dof[n as usize]);
            }
            // Edge midpoint DOFs (P2).
            if order == 2 {
                let edges: Vec<(u32, u32)> = if nodes.len() == 2 {
                    vec![(nodes[0], nodes[1])]
                } else {
                    vec![(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])]
                };
                for (a, b) in edges {
                    let key = EdgeKey::new(a, b);
                    face_dofs.push(*edge_to_dof.get(&key).expect("edge midpoint DOF not found"));
                }
            }
            // Edge interior DOFs (P3: 2 per edge).
            if order == 3 {
                let edges: Vec<(u32, u32)> = if nodes.len() == 2 {
                    vec![(nodes[0], nodes[1])]
                } else {
                    vec![(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])]
                };
                for (a, b) in edges {
                    let key = EdgeKey::new(a, b);
                    let [d0, d1] = *edge_to_dof2.get(&key).expect("P3 edge DOF not found");
                    // Order: d0 is closer to key.0, d1 to key.1.
                    // If edge is (a,b) and key = sorted(a,b), canonical order matches key.0→key.1.
                    // For the face, the edge is (a,b); key is sorted.
                    // DOFs near key.0 first, then key.1.
                    face_dofs.push(d0);
                    face_dofs.push(d1);
                }
                // Face bubble (3D only)
                if nodes.len() >= 3 && !face_bubble_dof.is_empty() {
                    let bd = face_bubble_dof[f as usize];
                    if bd != u32::MAX { face_dofs.push(bd); }
                }
            }
        }

        H1TraceSpace {
            mesh,
            order,
            n_dofs,
            node_to_dof,
            face_dofs,
            dofs_per_face,
            dof_coords,
            n_bfaces,
        }
    }

    /// Number of boundary faces.
    pub fn n_boundary_faces(&self) -> usize { self.n_bfaces }

    /// DOF indices for boundary face `f`.
    pub fn face_dofs(&self, f: u32) -> &[DofId] {
        let start = f as usize * self.dofs_per_face;
        &self.face_dofs[start..start + self.dofs_per_face]
    }

    /// Map from mesh node ID to trace DOF (returns `None` if not on boundary).
    pub fn node_to_trace_dof(&self, node: u32) -> Option<DofId> {
        let d = self.node_to_dof[node as usize];
        if d == u32::MAX { None } else { Some(d) }
    }
}

// H1TraceSpace implements FESpace with `element_dofs` returning face DOFs.
// This is a boundary space, so "elements" are the boundary faces.
impl<M: MeshTopology> FESpace for H1TraceSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    /// Returns DOFs for boundary face `elem` (here "element" means boundary face).
    fn element_dofs(&self, elem: u32) -> &[DofId] {
        self.face_dofs(elem)
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let dim = self.mesh.dim() as usize;
        let n = self.n_dofs;
        let mut v = Vector::zeros(n);
        for dof in 0..n {
            let base = dof * dim;
            let coords = &self.dof_coords[base..base + dim];
            v.as_slice_mut()[dof] = f(coords);
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::H1 }

    fn order(&self) -> u8 { self.order }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn h1_trace_unit_square_dof_count() {
        // Unit square 4×4 mesh: (5×5 = 25 nodes).
        // Boundary nodes: 4 * 4 = 16 (perimeter of 4×4 grid).
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1TraceSpace::new(mesh, 1);
        assert_eq!(space.n_dofs(), 16, "4×4 unit square has 16 boundary nodes");
    }

    #[test]
    fn h1_trace_face_dofs_valid() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let space = H1TraceSpace::new(mesh, 1);
        for f in 0..space.n_boundary_faces() as u32 {
            let dofs = space.face_dofs(f);
            assert_eq!(dofs.len(), 2, "2-D boundary face should have 2 DOFs (P1)");
            for &d in dofs {
                assert!((d as usize) < space.n_dofs(), "DOF index out of range");
            }
        }
    }

    #[test]
    fn h1_trace_interpolate_constant() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1TraceSpace::new(mesh, 1);
        let v = space.interpolate(&|_x| 5.0);
        assert_eq!(v.len(), space.n_dofs());
        for &c in v.as_slice() {
            assert!((c - 5.0).abs() < 1e-14);
        }
    }

    #[test]
    fn h1_trace_node_mapping() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let space = H1TraceSpace::new(mesh, 1);
        // Interior nodes should map to None
        // Node at (1/3, 1/3) should not be on boundary for a 3×3 mesh
        // All corner nodes should map to Some
        assert!(space.node_to_trace_dof(0).is_some(), "corner node 0 should be on boundary");
    }

    #[test]
    fn h1_trace_p2_2d_dof_count() {
        // n=4 mesh: boundary nodes = 16, boundary edges = 16 (perimeter edges of 4×4 grid)
        // P2 trace DOFs = 16 + 16 = 32
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1TraceSpace::new(mesh, 2);
        // 16 boundary vertices + 16 boundary edge midpoints = 32
        assert_eq!(space.n_dofs(), 32, "P2 trace n=4: expected 32 DOFs");
    }

    #[test]
    fn h1_trace_p2_2d_face_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let space = H1TraceSpace::new(mesh, 2);
        for f in 0..space.n_boundary_faces() as u32 {
            let dofs = space.face_dofs(f);
            assert_eq!(dofs.len(), 3, "2-D P2 face should have 3 DOFs");
            for &d in dofs {
                assert!((d as usize) < space.n_dofs(), "P2 DOF out of range");
            }
        }
    }

    #[test]
    fn h1_trace_3d() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let space = H1TraceSpace::new(mesh, 1);
        // All surface nodes of a 2×2×2 cube
        // Total nodes = 3×3×3 = 27, interior = 1×1×1 = 1
        assert_eq!(space.n_dofs(), 26, "3×3×3 cube has 26 boundary nodes");
        for f in 0..space.n_boundary_faces() as u32 {
            let dofs = space.face_dofs(f);
            assert_eq!(dofs.len(), 3, "3-D boundary face should have 3 DOFs (Tri3 P1)");
        }
    }

    #[test]
    fn h1_trace_p3_2d_dof_count() {
        // For n=4 unit square tri mesh:
        // P3 trace: 16 boundary vertices + 16 boundary edges × 2 = 48 total
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1TraceSpace::new(mesh, 3);
        assert_eq!(space.n_dofs(), 16 + 16 * 2, "P3 trace 2D: 16 verts + 32 edge DOFs");
    }

    #[test]
    fn h1_trace_p3_2d_face_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let space = H1TraceSpace::new(mesh, 3);
        for f in 0..space.n_boundary_faces() as u32 {
            let dofs = space.face_dofs(f);
            assert_eq!(dofs.len(), 4, "2-D P3 face should have 4 DOFs (2 verts + 2 edge)");
            for &d in dofs {
                assert!((d as usize) < space.n_dofs(), "P3 DOF out of range");
            }
        }
    }
}
