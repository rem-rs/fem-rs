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
    /// Build an H¹ trace space of order 1 on the boundary faces of `mesh`.
    ///
    /// # Panics
    /// Panics if `order != 1` (P2 trace not yet implemented).
    pub fn new(mesh: M, order: u8) -> Self {
        assert_eq!(order, 1, "H1TraceSpace: only order 1 supported for now");
        let dim = mesh.dim() as usize;
        let n_bfaces = mesh.n_boundary_faces();

        // Collect all boundary nodes
        let mut bdr_nodes = BTreeSet::new();
        for f in 0..n_bfaces as u32 {
            for &n in mesh.face_nodes(f) {
                bdr_nodes.insert(n);
            }
        }

        // Map mesh node → trace DOF
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
        let n_dofs = next_dof as usize;

        // Build per-face DOF connectivity
        let dofs_per_face = mesh.face_nodes(0).len(); // P1: nodes per face
        let mut face_dofs = Vec::with_capacity(n_bfaces * dofs_per_face);
        for f in 0..n_bfaces as u32 {
            let nodes = mesh.face_nodes(f);
            for &n in nodes {
                face_dofs.push(node_to_dof[n as usize]);
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
    use fem_mesh::SimplexMesh;

    #[test]
    fn h1_trace_unit_square_dof_count() {
        // Unit square 4×4 mesh: (5×5 = 25 nodes).
        // Boundary nodes: 4 * 4 = 16 (perimeter of 4×4 grid).
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1TraceSpace::new(mesh, 1);
        assert_eq!(space.n_dofs(), 16, "4×4 unit square has 16 boundary nodes");
    }

    #[test]
    fn h1_trace_face_dofs_valid() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1TraceSpace::new(mesh, 1);
        let v = space.interpolate(&|_x| 5.0);
        assert_eq!(v.len(), space.n_dofs());
        for &c in v.as_slice() {
            assert!((c - 5.0).abs() < 1e-14);
        }
    }

    #[test]
    fn h1_trace_node_mapping() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let space = H1TraceSpace::new(mesh, 1);
        // Interior nodes should map to None
        // Node at (1/3, 1/3) should not be on boundary for a 3×3 mesh
        // All corner nodes should map to Some
        assert!(space.node_to_trace_dof(0).is_some(), "corner node 0 should be on boundary");
    }

    #[test]
    fn h1_trace_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let space = H1TraceSpace::new(mesh, 1);
        // All surface nodes of a 2×2×2 cube
        // Total nodes = 3×3×3 = 27, interior = 1×1×1 = 1
        assert_eq!(space.n_dofs(), 26, "3×3×3 cube has 26 boundary nodes");
        for f in 0..space.n_boundary_faces() as u32 {
            let dofs = space.face_dofs(f);
            assert_eq!(dofs.len(), 3, "3-D boundary face should have 3 DOFs (Tri3 P1)");
        }
    }
}
