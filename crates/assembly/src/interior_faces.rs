//! Interior face list: pairs of elements sharing an interior edge/face.
//!
//! The `Mesh` only stores boundary faces.  For DG methods, the
//! assembly loop also needs to iterate over interior faces with the two
//! adjacent elements and the shared face geometry.
//!
//! Call [`InteriorFaceList::build`] once after mesh construction; then
//! iterate over [`InteriorFaceList::faces`].

use std::collections::HashMap;
use fem_core::types::{ElemId, NodeId};
use fem_mesh::topology::MeshTopology;

/// One interior face with its two neighbouring elements and the local
/// vertex indices (for normal computation).
#[derive(Debug, Clone)]
pub struct InteriorFace {
    /// First element (the "left" element).
    pub elem_left:  ElemId,
    /// Second element (the "right" element).
    pub elem_right: ElemId,
    /// Node indices of the shared face (2 for 2-D edges, 3 for 3-D triangles).
    pub face_nodes: Vec<NodeId>,
}

/// Pre-computed list of all interior faces in a mesh.
///
/// Built from the element connectivity in O(n_elems × nodes_per_face) time
/// using an edge/face-key → element hash map.
#[derive(Debug, Clone)]
pub struct InteriorFaceList {
    pub faces: Vec<InteriorFace>,
}

impl InteriorFaceList {
    /// Build the interior face list from `mesh`.
    ///
    /// Works for 2-D meshes (triangles, quads) and 3-D meshes (tetrahedra).
    /// For periodic meshes, call `mesh.detect_periodic_boundary()` first to
    /// merge periodic node pairs, then `InteriorFaceList` will automatically
    /// detect all interior faces via node-key matching.
    pub fn build<M: MeshTopology>(mesh: &M) -> Self {
        let dim = mesh.dim() as usize;

        // Map from sorted face key → (elem_id, face_node_indices unsorted)
        let mut face_map: HashMap<Vec<NodeId>, (ElemId, Vec<NodeId>)> = HashMap::new();
        let mut interior = Vec::new();

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let npe = nodes.len();

            // Enumerate faces of this element.
            // For a triangle (3 nodes): faces are pairs (0,1),(1,2),(0,2).
            // For a quad (4 nodes in 2D): faces are pairs (0,1),(1,2),(2,3),(3,0).
            // For a tet (4 nodes in 3D): faces are triples (0,1,2),(0,1,3),(0,2,3),(1,2,3).
            let local_faces = local_faces(npe, dim);

            for lf in &local_faces {
                let mut key: Vec<NodeId> = lf.iter().map(|&k| nodes[k]).collect();
                key.sort_unstable();

                match face_map.remove(&key) {
                    None => {
                        // First time we see this face.
                        let face_nodes: Vec<NodeId> = lf.iter().map(|&k| nodes[k]).collect();
                        face_map.insert(key, (e, face_nodes));
                    }
                    Some((other_elem, face_nodes)) => {
                        // Second time → interior face.
                        interior.push(InteriorFace {
                            elem_left:  other_elem,
                            elem_right: e,
                            face_nodes,
                        });
                    }
                }
            }
        }
        // Remaining entries in face_map are boundary faces (one element only) — ignore.

        InteriorFaceList { faces: interior }
    }

    /// Number of interior faces.
    pub fn len(&self) -> usize { self.faces.len() }
    pub fn is_empty(&self) -> bool { self.faces.is_empty() }
}

/// Returns the local node index sets of the `faces_per_elem` faces of an element.
fn local_faces(npe: usize, dim: usize) -> Vec<Vec<usize>> {
    match (npe, dim) {
        (3, 2) => vec![vec![0,1], vec![1,2], vec![0,2]], // triangle edges
        (4, 2) => vec![vec![0,1], vec![1,2], vec![2,3], vec![3,0]], // quad edges
        (4, 3) => vec![vec![1,2,3], vec![0,2,3], vec![0,1,3], vec![0,1,2]], // tet faces
        _ => panic!("local_faces: unsupported (npe={npe}, dim={dim})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// A 1×1 unit square split into 2 triangles → exactly 1 interior edge.
    #[test]
    fn single_square_interior_faces() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let ifl = InteriorFaceList::build(&mesh);
        assert_eq!(ifl.len(), 1, "Expected 1 interior face, got {}", ifl.len());
        assert_eq!(ifl.faces[0].face_nodes.len(), 2);
    }

    /// An n×n unit-square mesh has 2n²−n interior edges for a structured mesh.
    #[test]
    fn unit_square_interior_face_count() {
        let n = 4usize;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let ifl = InteriorFaceList::build(&mesh);
        // Each of the 2n² triangles has 3 edges; total edge-slots = 6n².
        // Boundary edges = 4n, so interior face-slots = 6n² - 4n.
        // Each interior face is shared by 2 elements, so n_interior = (6n²-4n)/2 = 3n²-2n.
        let expected = 3 * n * n - 2 * n;
        assert_eq!(ifl.len(), expected, "n={n}: expected {expected}, got {}", ifl.len());
    }
}
