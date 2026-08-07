use fem_core::{EdgeId, ElemId, FaceId, NodeId};
use crate::element_type::ElementType;

/// Minimal mesh interface required by fem-rs for assembly and DOF management.
///
/// Implementors provide topological connectivity (which nodes belong to which
/// element) and geometric data (node coordinates).  Higher-level operations
/// (DOF numbering, quadrature, etc.) are built on top of this trait.
pub trait MeshTopology: Send + Sync {
    /// Spatial dimension of the embedding space (2 or 3).
    fn dim(&self) -> u8;

    /// Topological (intrinsic) dimension of the mesh elements.
    ///
    /// For volume meshes (2-D Tri3 in 2-D, 3-D Tet4 in 3-D) this returns the
    /// same value as [`dim()`](Self::dim).  For **surface meshes** (2-D Tri3
    /// embedded in 3-D), `dim()` returns the embedding dimension (3) while
    /// `topological_dim()` returns the element dimension (2).
    ///
    /// The default implementation returns `self.dim()`.  Override in types
    /// where the embedding dimension differs from the element dimension.
    fn topological_dim(&self) -> u8 {
        self.dim()
    }

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

    // ─── Edge-level queries ──────────────────────────────────────────────────

    /// Total number of unique edges in the mesh.
    ///
    /// Returns 0 by default (implementations must build the edge map).
    fn n_edges(&self) -> usize { 0 }

    /// Flat slice of node indices belonging to edge `eid` (length = 2).
    ///
    /// Default panics — implement when `n_edges() > 0`.
    fn edge_nodes(&self, _eid: EdgeId) -> &[NodeId] {
        panic!("edge_nodes() not implemented");
    }

    /// Element(s) sharing edge `eid`.
    ///
    /// Returns `(elem, None)` for boundary edges, `(elem_a, Some(elem_b))` for
    /// interior edges.  Default panics — implement when `n_edges() > 0`.
    fn edge_elements(&self, _eid: EdgeId) -> (ElemId, Option<ElemId>) {
        panic!("edge_elements() not implemented");
    }

    /// Iterator over all edge indices.
    fn edge_iter(&self) -> std::ops::Range<u32> {
        0..self.n_edges() as u32
    }

    // ─── Face orientation ───────────────────────────────────────────────────

    /// Orientation of a boundary face with respect to a canonical reference.
    ///
    /// Returns `0` by default (identity).  Specific element types may return
    /// a rotation index (0, 1, 2, …) or flip flag.
    fn face_orientation(&self, _face: FaceId) -> u8 { 0 }

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

    /// High-order geometry node indices for element `elem`.
    ///
    /// Returns the node indices used for the isoparametric geometry mapping.
    /// For standard affine meshes this is the same as [`element_nodes`].
    /// For curved/high-order meshes it returns the full set of geometry DOF
    /// node indices (e.g. 6 for a quadratic triangle).
    ///
    /// The length may differ from [`element_nodes`] when the mesh uses
    /// high-order geometry.
    fn geometry_nodes(&self, elem: ElemId) -> &[NodeId] {
        self.element_nodes(elem)
    }

    /// Coordinates of a high-order geometry node.
    ///
    /// For flat (affine) meshes this returns the same as [`node_coords`].
    /// For curved meshes (with [`GeometryData`](crate::GeometryData)), this
    /// returns the high-order node coordinate which may differ from the linear
    /// mesh vertex coordinate.
    fn geom_coords_of(&self, node: NodeId) -> &[f64] {
        self.node_coords(node)
    }

    /// Total number of high-order geometry nodes.
    ///
    /// For flat meshes this equals [`n_nodes`](Self::n_nodes).  For meshes
    /// with per-element geometry (curved / geometrically periodic), it is the
    /// size of the geometry node table that [`geometry_nodes`](Self::geometry_nodes)
    /// and [`geom_coords_of`](Self::geom_coords_of) index into.
    fn geom_n_nodes(&self) -> usize {
        self.n_nodes()
    }

    /// MFEM vertex-view order for NC meshes: `Some(view)` means vertex DOF `d`
    /// refers to physical node `view[d]` (MFEM `UpdateVertices` top-level then
    /// SFC ordering).  `None` (default) means vertex DOF `d` is node `d`.
    fn nc_vertex_view(&self) -> Option<&[NodeId]> {
        None
    }

    /// Clone the mesh into a boxed trait object.
    /// Required for dimension-agnostic code with dynamic dispatch.
    fn clone_mesh(&self) -> Box<dyn MeshTopology + Send + Sync + 'static> {
        panic!("clone_mesh not implemented for this MeshTopology type");
    }

    /// Downcast to `dyn Any` for dimension-specific dispatch.
    fn as_any(&self) -> &dyn std::any::Any {
        panic!("as_any not implemented for this MeshTopology type");
    }
}

impl MeshTopology for Box<dyn MeshTopology + Send + Sync + 'static> {
    fn dim(&self) -> u8 { (**self).dim() }
    fn topological_dim(&self) -> u8 { (**self).topological_dim() }
    fn n_nodes(&self) -> usize { (**self).n_nodes() }
    fn n_elements(&self) -> usize { (**self).n_elements() }
    fn n_boundary_faces(&self) -> usize { (**self).n_boundary_faces() }
    fn element_nodes(&self, elem: ElemId) -> &[NodeId] { (**self).element_nodes(elem) }
    fn element_type(&self, elem: ElemId) -> ElementType { (**self).element_type(elem) }
    fn element_tag(&self, elem: ElemId) -> i32 { (**self).element_tag(elem) }
    fn node_coords(&self, node: NodeId) -> &[f64] { (**self).node_coords(node) }
    fn face_nodes(&self, face: FaceId) -> &[NodeId] { (**self).face_nodes(face) }
    fn face_tag(&self, face: FaceId) -> i32 { (**self).face_tag(face) }
    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) { (**self).face_elements(face) }
    fn n_edges(&self) -> usize { (**self).n_edges() }
    fn edge_nodes(&self, eid: EdgeId) -> &[NodeId] { (**self).edge_nodes(eid) }
    fn edge_elements(&self, eid: EdgeId) -> (ElemId, Option<ElemId>) { (**self).edge_elements(eid) }
    fn face_orientation(&self, face: FaceId) -> u8 { (**self).face_orientation(face) }
    fn geom_order(&self) -> u8 { (**self).geom_order() }
    fn geometry_nodes(&self, elem: ElemId) -> &[NodeId] { (**self).geometry_nodes(elem) }
    fn geom_coords_of(&self, node: NodeId) -> &[f64] { (**self).geom_coords_of(node) }
    fn clone_mesh(&self) -> Box<dyn MeshTopology + Send + Sync + 'static> { (**self).clone_mesh() }
    fn as_any(&self) -> &dyn std::any::Any { (**self).as_any() }
}
