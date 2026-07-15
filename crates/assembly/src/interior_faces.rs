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
    /// Supports periodic boundary detection: boundary faces at opposite sides
    /// of a periodic domain are paired and added as interior faces.
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

        // Phase 2: periodic face detection — pair un-matched boundary faces
        // by spatial position (for periodic meshes like periodic-square.mesh).
        if dim == 2 && !face_map.is_empty() {
            let mut boundary_edges: Vec<(ElemId, Vec<NodeId>)> = face_map.into_values().collect();
            let mut used = vec![false; boundary_edges.len()];

            for i in 0..boundary_edges.len() {
                if used[i] { continue; }
                for j in (i + 1)..boundary_edges.len() {
                    if used[j] { continue; }
                    if are_periodic_partners_2d(mesh, &boundary_edges[i], &boundary_edges[j]) {
                        // Found a periodic pair — add as interior face.
                        // Use the face_nodes from the entry with lower element index as canonical.
                        let (el, fn_i) = &boundary_edges[i];
                        let (er, _fn_j) = &boundary_edges[j];
                        interior.push(InteriorFace {
                            elem_left:  *el.min(er),
                            elem_right: *el.max(er),
                            face_nodes: fn_i.clone(),
                        });
                        used[i] = true;
                        used[j] = true;
                        break;
                    }
                }
            }
        }

        InteriorFaceList { faces: interior }
    }

    /// Number of interior faces.
    pub fn len(&self) -> usize { self.faces.len() }
    pub fn is_empty(&self) -> bool { self.faces.is_empty() }
}

/// Check if two boundary edges in 2D are periodic partners.
///
/// Two edges are periodic partners if they have approximately the same length,
/// their normals point in opposite directions, and they are at corresponding
/// positions on opposite sides of the domain (within tolerance).
fn are_periodic_partners_2d<M: MeshTopology>(
    mesh: &M,
    entry_a: &(ElemId, Vec<NodeId>),
    entry_b: &(ElemId, Vec<NodeId>),
) -> bool {
    let (elem_a, nodes_a) = entry_a;
    let (elem_b, nodes_b) = entry_b;

    // Edge vectors
    let p0a = mesh.node_coords(nodes_a[0]);
    let p1a = mesh.node_coords(nodes_a[1]);
    let p0b = mesh.node_coords(nodes_b[0]);
    let p1b = mesh.node_coords(nodes_b[1]);

    let dxa = p1a[0] - p0a[0]; let dya = p1a[1] - p0a[1];
    let dxb = p1b[0] - p0b[0]; let dyb = p1b[1] - p0b[1];

    let len_a = (dxa*dxa + dya*dya).sqrt();
    let len_b = (dxb*dxb + dyb*dyb).sqrt();

    // Edge lengths should match within 1%
    if (len_a - len_b).abs() > 0.01 * len_a.max(len_b) {
        return false;
    }

    // Compute outward normals (left-normal of edge direction from elem perspective)
    // For the left element, normal = (-dy, dx)/len
    let nx_a = -dya / len_a; let ny_a = dxa / len_a;
    let nx_b = -dyb / len_b; let ny_b = dxb / len_b;

    // Ensure normals point outward from elements
    let centroid_a = element_centroid(mesh, *elem_a);
    let centroid_b = element_centroid(mesh, *elem_b);
    let mid_a = [(p0a[0] + p1a[0]) / 2.0, (p0a[1] + p1a[1]) / 2.0];
    let mid_b = [(p0b[0] + p1b[0]) / 2.0, (p0b[1] + p1b[1]) / 2.0];

    // Normal from centroid to midpoint should point outward
    let dot_a = nx_a * (mid_a[0] - centroid_a[0]) + ny_a * (mid_a[1] - centroid_a[1]);
    let dot_b = nx_b * (mid_b[0] - centroid_b[0]) + ny_b * (mid_b[1] - centroid_b[1]);

    // Flip normals if they point inward
    let (nx_a, ny_a) = if dot_a < 0.0 { (-nx_a, -ny_a) } else { (nx_a, ny_a) };
    let (nx_b, ny_b) = if dot_b < 0.0 { (-nx_b, -ny_b) } else { (nx_b, ny_b) };

    // Periodic partners have opposite outward normals
    let n_dot = nx_a * nx_b + ny_a * ny_b;
    if n_dot > -0.5 {  // opposite means dot ≈ -1
        return false;
    }

    // For periodic pairing, the node positions of edge A should map to
    // node positions of edge B under a single translation vector.
    // Compute two possible translation vectors:
    //   shift1: node_a[0] → node_b[0], node_a[1] → node_b[1]
    //   shift2: node_a[0] → node_b[1], node_a[1] → node_b[0] (swapped orientation)
    let tol = 1e-8;

    let shift1_x = p0b[0] - p0a[0]; let shift1_y = p0b[1] - p0a[1];
    let match_shift1 = (p1b[0] - p1a[0] - shift1_x).abs() < tol
                    && (p1b[1] - p1a[1] - shift1_y).abs() < tol;

    let shift2_x = p1b[0] - p0a[0]; let shift2_y = p1b[1] - p0a[1];
    let match_shift2 = (p0b[0] - p1a[0] - shift2_x).abs() < tol
                    && (p0b[1] - p1a[1] - shift2_y).abs() < tol;

    match_shift1 || match_shift2
}

/// Compute the centroid of a 2D element.
fn element_centroid<M: MeshTopology>(mesh: &M, elem: ElemId) -> [f64; 2] {
    let nodes = mesh.element_nodes(elem);
    let mut cx = 0.0; let mut cy = 0.0;
    for &n in nodes {
        let c = mesh.node_coords(n);
        cx += c[0]; cy += c[1];
    }
    let n = nodes.len() as f64;
    [cx / n, cy / n]
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
