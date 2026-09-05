//! Face-to-edge mapping table for mixed bilinear forms.
//!
//! In 3-D, each face (triangular or quadrilateral) has a set of edges that
//! form its boundary.  This table provides that mapping, which is required
//! for assembling HCurl × HDiv mixed forms (RT × ND coupling) and for
//! parallel mixed assembly.
//!
//! - 2-D: each "face" is an edge, so the table is trivially the identity.
//! - 3-D: each face has 3 edges (triangle) or 4 edges (quadrilateral).
//!
//! This mirrors MFEM's `MixedBilinearForm::GetFaceEdgeTable` semantics.

use std::collections::HashMap;
use fem_mesh::topology::MeshTopology;
use fem_space::dof_manager::{EdgeKey, FaceKey, QuadFaceKey};

/// Face-to-edge mapping.
#[derive(Debug, Clone)]
pub enum FaceEdgeTable {
    /// 2-D: map from edge key to itself.
    Edges2D(HashMap<EdgeKey, EdgeKey>),
    /// 3-D: triangular faces → 3 edges.
    TriFaces3D(HashMap<FaceKey, [EdgeKey; 3]>),
    /// 3-D: quadrilateral faces → 4 edges.
    QuadFaces3D(HashMap<QuadFaceKey, [EdgeKey; 4]>),
}

impl FaceEdgeTable {
    /// Build the face-edge table for a mesh.
    pub fn build<M: MeshTopology>(mesh: &M) -> Self {
        let dim = mesh.dim();
        if dim == 2 {
            Self::build_2d(mesh)
        } else {
            Self::build_3d(mesh)
        }
    }

    fn build_2d<M: MeshTopology>(mesh: &M) -> Self {
        let mut map = HashMap::new();
        for fid in mesh.face_iter() {
            let nodes = mesh.face_nodes(fid);
            if nodes.len() >= 2 {
                let key = EdgeKey::new(nodes[0], nodes[1]);
                map.insert(key, key);
            }
        }
        FaceEdgeTable::Edges2D(map)
    }

    fn build_3d<M: MeshTopology>(mesh: &M) -> Self {
        let mut tri_map: HashMap<FaceKey, [EdgeKey; 3]> = HashMap::new();
        let mut quad_map: HashMap<QuadFaceKey, [EdgeKey; 4]> = HashMap::new();

        for fid in mesh.face_iter() {
            let nodes = mesh.face_nodes(fid);
            if nodes.len() == 3 {
                let e0 = EdgeKey::new(nodes[0], nodes[1]);
                let e1 = EdgeKey::new(nodes[1], nodes[2]);
                let e2 = EdgeKey::new(nodes[2], nodes[0]);
                let fkey = FaceKey::new(nodes[0], nodes[1], nodes[2]);
                tri_map.insert(fkey, [e0, e1, e2]);
            } else if nodes.len() == 4 {
                let e0 = EdgeKey::new(nodes[0], nodes[1]);
                let e1 = EdgeKey::new(nodes[1], nodes[2]);
                let e2 = EdgeKey::new(nodes[2], nodes[3]);
                let e3 = EdgeKey::new(nodes[3], nodes[0]);
                let fkey = QuadFaceKey::new(nodes[0], nodes[1], nodes[2], nodes[3]);
                quad_map.insert(fkey, [e0, e1, e2, e3]);
            }
        }

        if quad_map.is_empty() {
            FaceEdgeTable::TriFaces3D(tri_map)
        } else {
            FaceEdgeTable::QuadFaces3D(quad_map)
        }
    }

    pub fn tri_face_edges(&self, face: FaceKey) -> Option<&[EdgeKey; 3]> {
        match self {
            FaceEdgeTable::TriFaces3D(map) => map.get(&face),
            _ => None,
        }
    }

    pub fn quad_face_edges(&self, face: QuadFaceKey) -> Option<&[EdgeKey; 4]> {
        match self {
            FaceEdgeTable::QuadFaces3D(map) => map.get(&face),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            FaceEdgeTable::Edges2D(map) => map.len(),
            FaceEdgeTable::TriFaces3D(map) => map.len(),
            FaceEdgeTable::QuadFaces3D(map) => map.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn face_edge_table_2d_builds() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let table = FaceEdgeTable::build(&mesh);
        assert!(!table.is_empty());
        assert!(table.len() > 0);
    }

    #[test]
    fn face_edge_table_3d_tri_builds() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let table = FaceEdgeTable::build(&mesh);
        assert!(!table.is_empty());
        match &table {
            FaceEdgeTable::TriFaces3D(map) => {
                assert!(!map.is_empty());
                for (_, edges) in map.iter() {
                    assert_eq!(edges.len(), 3);
                }
            }
            _ => panic!("Expected TriFaces3D for tet mesh"),
        }
    }

    #[test]
    fn face_edge_table_3d_hex_builds() {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let table = FaceEdgeTable::build(&mesh);
        assert!(!table.is_empty());
        match &table {
            FaceEdgeTable::QuadFaces3D(map) => {
                assert!(!map.is_empty());
                for (_, edges) in map.iter() {
                    assert_eq!(edges.len(), 4);
                }
            }
            _ => panic!("Expected QuadFaces3D for hex mesh"),
        }
    }

    #[test]
    fn face_edge_table_edges_are_valid() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let table = FaceEdgeTable::build(&mesh);
        match &table {
            FaceEdgeTable::Edges2D(map) => {
                for (k, v) in map.iter() {
                    assert_eq!(k, v, "2-D table should be identity");
                }
            }
            _ => panic!("Expected 2D table"),
        }
    }

    #[test]
    fn face_edge_table_face_has_distinct_edges() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let table = FaceEdgeTable::build(&mesh);
        if let FaceEdgeTable::TriFaces3D(map) = &table {
            for (_, edges) in map.iter() {
                assert_ne!(edges[0], edges[1]);
                assert_ne!(edges[1], edges[2]);
                assert_ne!(edges[0], edges[2]);
            }
        }
    }
}
