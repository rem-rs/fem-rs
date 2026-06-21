//! Pure-Rust mesh graph partitioning.
//!
//! Builds an element **dual graph** from a [`SimplexMesh`] and partitions it
//! using graph algorithms (BFS k-way).
//!
//! # Graph format
//! The dual graph is returned in standard CSR adjacency format (`xadj`, `adjncy`),
//! compatible with METIS C API conventions.
//!
//! # Example
//! ```
//! use fem_rmetis::{build_dual_graph, partition_bfs_kway};
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(8);
//! let (xadj, adjncy) = build_dual_graph(&mesh);
//! let parts = partition_bfs_kway(mesh.n_elems(), &xadj, &adjncy, 4);
//! assert_eq!(parts.len(), mesh.n_elems());
//! ```

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};
use fem_mesh::SimplexMesh;

// ─── Dual graph construction ─────────────────────────────────────────────────

/// Element facets (edges in 2D, faces in 3D) for building the dual graph.
///
/// Two elements are neighbours in the dual graph iff they share a full facet
/// with identical node set.
///
/// Supports Tri3, Quad4 (2D); Tet4, Pyramid5, Prism6, Hex8 (3D).
pub fn local_faces(nodes: &[NodeId], dim: usize) -> Vec<Vec<NodeId>> {
    match (nodes.len(), dim) {
        (3, 2) => vec![
            vec![nodes[0], nodes[1]],
            vec![nodes[1], nodes[2]],
            vec![nodes[0], nodes[2]],
        ],
        (4, 2) => vec![
            vec![nodes[0], nodes[1]],
            vec![nodes[1], nodes[2]],
            vec![nodes[2], nodes[3]],
            vec![nodes[3], nodes[0]],
        ],
        (4, 3) => vec![
            vec![nodes[1], nodes[2], nodes[3]],
            vec![nodes[0], nodes[2], nodes[3]],
            vec![nodes[0], nodes[1], nodes[3]],
            vec![nodes[0], nodes[1], nodes[2]],
        ],
        (5, 3) => vec![
            vec![nodes[0], nodes[1], nodes[2], nodes[3]],
            vec![nodes[0], nodes[1], nodes[4]],
            vec![nodes[1], nodes[2], nodes[4]],
            vec![nodes[2], nodes[3], nodes[4]],
            vec![nodes[3], nodes[0], nodes[4]],
        ],
        (6, 3) => vec![
            vec![nodes[0], nodes[1], nodes[2]],
            vec![nodes[3], nodes[4], nodes[5]],
            vec![nodes[0], nodes[1], nodes[4], nodes[3]],
            vec![nodes[1], nodes[2], nodes[5], nodes[4]],
            vec![nodes[2], nodes[0], nodes[3], nodes[5]],
        ],
        (8, 3) => vec![
            vec![nodes[0], nodes[1], nodes[2], nodes[3]],
            vec![nodes[4], nodes[5], nodes[6], nodes[7]],
            vec![nodes[0], nodes[1], nodes[5], nodes[4]],
            vec![nodes[2], nodes[3], nodes[7], nodes[6]],
            vec![nodes[0], nodes[3], nodes[7], nodes[4]],
            vec![nodes[1], nodes[2], nodes[6], nodes[5]],
        ],
        _ => vec![],
    }
}

/// Build the element dual graph of a simplex mesh.
///
/// Returns `(xadj, adjncy)` in CSR adjacency format, matching the METIS C API:
/// - `xadj[e]..xadj[e+1]` gives adjacency slice for element `e`.
/// - Edges are stored twice (once per direction).
pub fn build_dual_graph<const D: usize>(mesh: &SimplexMesh<D>) -> (Vec<i32>, Vec<i32>) {
    let n_elems = mesh.n_elems();
    let dim = D;

    let mut face_map: HashMap<Vec<NodeId>, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let nodes = mesh.elem_nodes(e);
        for lf in local_faces(nodes, dim) {
            let mut key = lf;
            key.sort_unstable();
            face_map.entry(key).or_default().push(e);
        }
    }

    let mut adj: Vec<Vec<ElemId>> = vec![Vec::new(); n_elems];
    for elems in face_map.values() {
        if elems.len() == 2 {
            adj[elems[0] as usize].push(elems[1]);
            adj[elems[1] as usize].push(elems[0]);
        }
    }

    let mut xadj = vec![0_i32; n_elems + 1];
    let mut adjncy = Vec::<i32>::new();
    for (e, nbrs) in adj.iter().enumerate() {
        xadj[e + 1] = xadj[e] + nbrs.len() as i32;
        adjncy.extend(nbrs.iter().map(|&n| n as i32));
    }
    (xadj, adjncy)
}

// ─── BFS k-way partition ─────────────────────────────────────────────────────

/// Partition the dual graph into `k` balanced parts using BFS flood-fill from
/// evenly spaced seeds.
///
/// This is a **greedy** heuristic (not multilevel).  It produces connected
/// partitions with reasonable balance for moderate mesh sizes.
pub fn partition_bfs_kway(n: usize, xadj: &[i32], adjncy: &[i32], k: usize) -> Vec<i32> {
    const UNSET: i32 = -1;
    let mut part = vec![UNSET; n];
    let mut queue = std::collections::VecDeque::<usize>::new();

    for p in 0..k {
        let seed = (p * n) / k;
        if part[seed] == UNSET {
            part[seed] = p as i32;
            queue.push_back(seed);
        }
    }

    while let Some(e) = queue.pop_front() {
        let owner = part[e];
        for j in xadj[e] as usize..xadj[e + 1] as usize {
            let nb = adjncy[j] as usize;
            if part[nb] == UNSET {
                part[nb] = owner;
                queue.push_back(nb);
            }
        }
    }

    for i in 0..n {
        if part[i] == UNSET {
            part[i] = (i % k) as i32;
        }
    }

    part
}

/// Convenience: partition a `SimplexMesh` into `nparts` balanced parts.
///
/// Builds the dual graph, runs BFS k-way, returns `partition[e]` for each element.
pub fn partition_mesh<const D: usize>(
    mesh: &SimplexMesh<D>,
    nparts: usize,
) -> Vec<i32> {
    if nparts <= 1 {
        return vec![0_i32; mesh.n_elems()];
    }
    let (xadj, adjncy) = build_dual_graph(mesh);
    partition_bfs_kway(mesh.n_elems(), &xadj, &adjncy, nparts)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{ElementType, SimplexMesh};

    fn two_prisms_sharing_triangle() -> SimplexMesh<3> {
        let coords = vec![0.0_f64; 9 * 3];
        let conn = vec![
            0u32, 1, 2, 3, 4, 5,
            3, 4, 5, 6, 7, 8,
        ];
        SimplexMesh::uniform(coords, conn, vec![1, 1], ElementType::Prism6, vec![], vec![], ElementType::Tri3)
    }

    #[test]
    fn dual_graph_prism_pair() {
        let mesh = two_prisms_sharing_triangle();
        let (_xadj, adjncy) = build_dual_graph(&mesh);
        assert_eq!(adjncy.len(), 2, "expected 1 undirected edge");
    }

    #[test]
    fn partition_all_elements_assigned() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let parts = partition_mesh(&mesh, 4);
        assert_eq!(parts.len(), mesh.n_elems());
        assert!(parts.iter().all(|&p| p >= 0 && p < 4));
    }

    #[test]
    fn partition_balanced() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n = mesh.n_elems();
        let parts = partition_mesh(&mesh, 4);
        let mut counts = vec![0usize; 4];
        for &p in &parts { counts[p as usize] += 1; }
        let ideal = n as f64 / 4.0;
        for &c in &counts {
            assert!((c as f64 - ideal).abs() / ideal < 0.6);
        }
    }

    #[test]
    fn partition_single_part_is_identity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let parts = partition_mesh(&mesh, 1);
        assert!(parts.iter().all(|&p| p == 0));
    }

    #[test]
    fn partition_tet_mesh() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let parts = partition_mesh(&mesh, 2);
        assert_eq!(parts.len(), mesh.n_elems());
        assert!(parts.iter().all(|&p| p == 0 || p == 1));
    }
}
