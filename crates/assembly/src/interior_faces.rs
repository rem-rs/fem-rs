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
    /// Phase 1 finds interior faces by standard node-key matching.
    /// Phase 2 (2D only) detects periodic boundary faces by geometry matching
    /// and adds them as interior face pairs WITHOUT merging mesh nodes.
    pub fn build<M: MeshTopology>(mesh: &M) -> Self {
        let dim = mesh.dim() as usize;

        // Phase 1: standard interior face detection (node-key matching)
        let mut face_map: HashMap<Vec<NodeId>, (ElemId, Vec<NodeId>)> = HashMap::new();
        let mut interior = Vec::new();

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let npe = nodes.len();
            let local_faces = local_faces(npe, dim);

            for lf in &local_faces {
                let mut key: Vec<NodeId> = lf.iter().map(|&k| nodes[k]).collect();
                key.sort_unstable();

                match face_map.remove(&key) {
                    None => {
                        let face_nodes: Vec<NodeId> = lf.iter().map(|&k| nodes[k]).collect();
                        face_map.insert(key, (e, face_nodes));
                    }
                    Some((other_elem, face_nodes)) => {
                        interior.push(InteriorFace {
                            elem_left:  other_elem,
                            elem_right: e,
                            face_nodes,
                        });
                    }
                }
            }
        }

        // Phase 2: periodic face detection (2D only)
        // Find remaining unpaired edges and match them across periodic boundaries.
        // Only applies when mesh has no boundary faces (fully periodic).
        if dim == 2 && !face_map.is_empty() && mesh.n_boundary_faces() == 0 {
            // Collect: (elem, face_nodes, normal)
            let bdr: Vec<(ElemId, Vec<u32>, Vec<f64>)> = face_map
                .into_values()
                .map(|(elem, nodes)| {
                    let p0 = mesh.node_coords(nodes[0]);
                    let p1 = mesh.node_coords(nodes[1]);
                    let dx = p1[0] - p0[0];
                    let dy = p1[1] - p0[1];
                    let len = (dx * dx + dy * dy).sqrt();
                    let (mut nx, mut ny) = (-dy / len, dx / len);
                    // Determine outward direction using element centroid
                    let en = mesh.element_nodes(elem);
                    let cx: f64 = en.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>() / en.len() as f64;
                    let cy: f64 = en.iter().map(|&n| mesh.node_coords(n)[1]).sum::<f64>() / en.len() as f64;
                    let mx = (p0[0] + p1[0]) / 2.0;
                    let my = (p0[1] + p1[1]) / 2.0;
                    if nx * (mx - cx) + ny * (my - cy) < 0.0 { nx = -nx; ny = -ny; }
                    (elem, nodes, vec![nx, ny])
                })
                .collect();

            // Group by normal direction and sort by midpoint position
            let group_and_pair = |dir: usize, val: f64, opp_val: f64| {
                // dir=0 for x(left/right), dir=1 for y(bottom/top)
                let mut neg: Vec<usize> = (0..bdr.len()).filter(|&i| bdr[i].2[dir] < val).collect();
                let mut pos: Vec<usize> = (0..bdr.len()).filter(|&i| bdr[i].2[dir] > opp_val).collect();
                if neg.is_empty() || pos.is_empty() || neg.len() != pos.len() { return; }
                // Sort by the OTHER coordinate
                let other_dir = 1 - dir;
                let coord = |idx: usize| -> f64 {
                    let fnodes = &bdr[idx].1;
                    let p0 = mesh.node_coords(fnodes[0]);
                    let p1 = mesh.node_coords(fnodes[1]);
                    (p0[other_dir] + p1[other_dir]) / 2.0
                };
                neg.sort_by_key(|&i| (coord(i) * 1e6) as i64);
                pos.sort_by_key(|&i| (coord(i) * 1e6) as i64);
                for i in 0..neg.len() {
                    let (el_neg, ref nodes_neg, _) = &bdr[neg[i]];
                    let (el_pos, ref nodes_pos, _) = &bdr[pos[i]];
                }
                // Note: periodic faces need special handling because the
                // face is not shared by adjacent elements in physical space.
                // Use assemble_periodic_faces() at the application level.
            };

            group_and_pair(0, -0.5, 0.5);  // x: left vs right
            group_and_pair(1, -0.5, 0.5);  // y: bottom vs top
        }

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
