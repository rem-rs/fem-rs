use fem_core::{ElemId, FaceId, NodeId};
use crate::element_type::ElementType;

/// Minimal mesh interface required by fem-rs for assembly and DOF management.
///
/// Implementors provide topological connectivity (which nodes belong to which
/// element) and geometric data (node coordinates).  Higher-level operations
/// (DOF numbering, quadrature, etc.) are built on top of this trait.
pub trait MeshTopology: Send + Sync {
    /// Spatial dimension of the embedding space (2 or 3).
    fn dim(&self) -> u8;

    /// Total number of mesh nodes (vertices).
    fn n_nodes(&self) -> usize;

    /// Total number of interior (volume/surface) elements.
    fn n_elements(&self) -> usize;

    /// Total number of boundary faces (edges in 2-D, faces in 3-D).
    fn n_boundary_faces(&self) -> usize;

    /// Flat slice of node indices belonging to element `elem`.
    ///
    /// Length equals `ElementType::nodes_per_element` for the mesh's element type.
    fn element_nodes(&self, elem: ElemId) -> &[NodeId];

    /// Geometric type of element `elem`.
    fn element_type(&self, elem: ElemId) -> ElementType;

    /// Physical group tag of element `elem` (material / domain label).
    fn element_tag(&self, elem: ElemId) -> i32;

    /// Flat slice of node coordinates for node `node`.
    ///
    /// Length equals `self.dim()`.  Coordinates are in physical space.
    fn node_coords(&self, node: NodeId) -> &[f64];

    /// Flat slice of node indices on boundary face `face`.
    fn face_nodes(&self, face: FaceId) -> &[NodeId];

    /// Physical group tag of boundary face `face` (boundary condition label).
    fn face_tag(&self, face: FaceId) -> i32;

    /// Elements sharing boundary face `face`.
    ///
    /// Returns `(interior_elem, None)` for mesh boundary faces,
    /// or `(elem_a, Some(elem_b))` for interior faces (when tracked).
    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>);

    /// Iterator over all element indices.
    fn elem_iter(&self) -> std::ops::Range<u32> {
        0..self.n_elements() as u32
    }

    /// Iterator over all boundary face indices.
    fn face_iter(&self) -> std::ops::Range<u32> {
        0..self.n_boundary_faces() as u32
    }

    /// Geometric polynomial order used for the isoparametric mapping.
    ///
    /// Returns 1 for standard (affine) meshes.  Curved / isoparametric meshes
    /// return their geometric order (2 for P2, 3 for P3, …).
    ///
    /// The assembler uses this to decide whether to use the affine Jacobian
    /// fast path or the full isoparametric Jacobian.
    fn geom_order(&self) -> u8 {
        1
    }
}
