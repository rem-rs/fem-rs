//! METIS-based mesh partitioning.
//!
//! [`MetisPartitioner`] builds a **dual graph** from the element connectivity
//! (elements are vertices, shared faces/edges are graph edges), then partitions
//! the mesh.  Currently uses a greedy graph-coloring fallback (no external
//! METIS dependency); the API is compatible with a future rmetis backend.
//!
//! # Usage
//! ```rust,ignore
//! use fem_parallel::metis::{MetisPartitioner, MetisOptions};
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(16);
//! let opt  = MetisOptions::default();
//! let parts = MetisPartitioner::partition_mesh(&mesh, 4, &opt).unwrap();
//! // parts[e] = rank that owns element e (0..nparts)
//! ```
//!
//! The `partition_simplex_metis` convenience function wraps this into a
//! [`ParallelMesh`] for use in the parallel pipeline.

use std::collections::HashMap;

use fem_core::{ElemId, NodeId, Rank};
use fem_mesh::SimplexMesh;

use crate::{Comm, MeshPartition, par_mesh::ParallelMesh};
use crate::mesh_serde;
use crate::par_simplex::{extract_submesh_from_partition, STREAM_TAG_BASE};

// ─── Options ──────────────────────────────────────────────────────────────────

/// Options for the METIS partitioner.
#[derive(Debug, Clone, Default)]
pub struct MetisOptions {
    /// If true, print partition statistics to stdout.
    pub verbose: bool,
}

// ─── MetisPartitioner ─────────────────────────────────────────────────────────

/// Mesh partitioner using a greedy graph-bisection heuristic.
///
/// This provides balanced partitions without an external METIS dependency.
/// For production use with large meshes, link against METIS via a feature flag.
pub struct MetisPartitioner;

impl MetisPartitioner {
    /// Partition a simplex mesh into `nparts` balanced parts.
    ///
    /// Returns a vector of length `n_elems` where `partition[e]` is the rank
    /// (0..nparts) assigned to element `e`.
    pub fn partition_mesh<const D: usize>(
        mesh:   &SimplexMesh<D>,
        nparts: usize,
        opts:   &MetisOptions,
    ) -> Result<Vec<Rank>, String> {
        assert!(nparts >= 1, "nparts must be ≥ 1");
        let n_elems = mesh.n_elems();
        assert!(n_elems > 0, "mesh has no elements");

        if nparts == 1 {
            return Ok(vec![0; n_elems]);
        }

        // Build dual graph
        let (xadj, adjncy) = build_dual_graph(mesh);

        // Greedy BFS-based k-way partitioning
        let partition = bfs_kway_partition(n_elems, &xadj, &adjncy, nparts);

        if opts.verbose {
            let mut counts = vec![0usize; nparts];
            for &p in &partition { counts[p as usize] += 1; }
            println!("[MetisPartitioner] nparts={nparts}, counts={counts:?}");
        }

        Ok(partition)
    }
}

// ─── BFS k-way partitioner ────────────────────────────────────────────────────

/// Simple BFS-based k-way partitioner.
///
/// Grows k regions simultaneously from seed elements placed uniformly.
fn bfs_kway_partition(n: usize, xadj: &[i32], adjncy: &[i32], k: usize) -> Vec<Rank> {
    const UNSET: Rank = -1;
    let mut part = vec![UNSET; n];
    let mut queue: std::collections::VecDeque<usize> = Default::default();

    // Place k seeds spaced evenly
    for p in 0..k {
        let seed = (p * n) / k;
        if part[seed] == UNSET {
            part[seed] = p as Rank;
            queue.push_back(seed);
        }
    }

    // BFS flood-fill
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

    // Assign any remaining unvisited elements (disconnected components)
    for i in 0..n {
        if part[i] == UNSET {
            part[i] = (i % k) as Rank;
        }
    }

    part
}

// ─── Dual graph builder ───────────────────────────────────────────────────────

fn build_dual_graph<const D: usize>(mesh: &SimplexMesh<D>) -> (Vec<i32>, Vec<i32>) {
    let n_elems = mesh.n_elems();

    let mut face_map: HashMap<Vec<NodeId>, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let nodes = mesh.elem_nodes(e);
        for lf in local_faces_of_elem::<D>(nodes) {
            let mut key = lf;
            key.sort_unstable();
            face_map.entry(key).or_default().push(e);
        }
    }

    let mut adj: Vec<Vec<ElemId>> = vec![Vec::new(); n_elems];
    for (_key, elems) in &face_map {
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

fn local_faces_of_elem<const D: usize>(nodes: &[NodeId]) -> Vec<Vec<NodeId>> {
    match (nodes.len(), D) {
        (3, 2) => vec![
            vec![nodes[0], nodes[1]],
            vec![nodes[1], nodes[2]],
            vec![nodes[0], nodes[2]],
        ],
        (4, 3) => vec![
            vec![nodes[1], nodes[2], nodes[3]],
            vec![nodes[0], nodes[2], nodes[3]],
            vec![nodes[0], nodes[1], nodes[3]],
            vec![nodes[0], nodes[1], nodes[2]],
        ],
        _ => vec![],
    }
}

// ─── partition_simplex_metis ──────────────────────────────────────────────────

/// Distribute `mesh` across `comm.size()` ranks using k-way partitioning.
///
/// **Note**: every rank must hold the full serial mesh.  For a
/// memory-efficient alternative where only rank 0 holds the mesh, use
/// [`partition_simplex_metis_streaming`].
pub fn partition_simplex_metis<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &Comm,
    opts: &MetisOptions,
) -> ParallelMesh<SimplexMesh<D>> {
    let n_elems = mesh.n_elems();
    let n_nodes_total = mesh.n_nodes();
    assert!(n_elems > 0, "partition_simplex_metis: mesh has no elements");

    let size = comm.size();
    if size == 1 {
        let partition = MeshPartition::new_serial(n_nodes_total, n_elems);
        return ParallelMesh::new(mesh.clone(), comm.clone(), partition);
    }

    let elem_part = MetisPartitioner::partition_mesh(mesh, size, opts)
        .expect("partitioning failed");

    let (local_mesh, partition) = extract_submesh_from_partition(
        mesh, comm.rank(), &elem_part,
    );
    ParallelMesh::new(local_mesh, comm.clone(), partition)
}

// ─── partition_simplex_metis_streaming ────────────────────────────────────────

/// Streaming METIS partition: only rank 0 holds the full mesh.
///
/// Rank 0 runs the METIS graph partitioner, then sends each rank's sub-mesh
/// via point-to-point messages.  Other ranks receive their sub-mesh without
/// ever loading the full mesh.
///
/// # Arguments
/// * `mesh` — `Some(&full_mesh)` on rank 0, `None` on other ranks.
/// * `comm` — communicator spanning all ranks.
/// * `opts` — METIS options (verbose flag, etc.).
///
/// # Errors
/// Returns `Err` if partitioning or mesh decode fails.
pub fn partition_simplex_metis_streaming<const D: usize>(
    mesh: Option<&SimplexMesh<D>>,
    comm: &Comm,
    opts: &MetisOptions,
) -> Result<ParallelMesh<SimplexMesh<D>>, String> {
    let size = comm.size();

    if size == 1 {
        let m = mesh.ok_or("rank 0 must provide the mesh")?;
        let partition = MeshPartition::new_serial(m.n_nodes(), m.n_elems());
        return Ok(ParallelMesh::new(m.clone(), comm.clone(), partition));
    }

    if comm.is_root() {
        let m = mesh.ok_or("rank 0 must provide the mesh")?;

        let elem_part = MetisPartitioner::partition_mesh(m, size, opts)
            .map_err(|e| e.to_string())?;

        // Send sub-meshes to ranks 1..N-1.
        for target in 1..size as Rank {
            let (sub_mesh, sub_part) = extract_submesh_from_partition(
                m, target, &elem_part,
            );
            let encoded = mesh_serde::encode_submesh(&sub_mesh, &sub_part);
            comm.send_bytes(target, STREAM_TAG_BASE + target, &encoded);
        }

        // Extract rank 0's own sub-mesh.
        let (local_mesh, partition) = extract_submesh_from_partition(
            m, 0, &elem_part,
        );
        Ok(ParallelMesh::new(local_mesh, comm.clone(), partition))
    } else {
        let local_rank = comm.rank();
        let buf = comm.recv_bytes(0, STREAM_TAG_BASE + local_rank);
        let (local_mesh, partition) = mesh_serde::decode_submesh::<D>(&buf)?;
        Ok(ParallelMesh::new(local_mesh, comm.clone(), partition))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn partition_covers_all_elements() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_elems = mesh.n_elems();
        let nparts = 4;
        let parts = MetisPartitioner::partition_mesh(&mesh, nparts, &MetisOptions::default())
            .unwrap();
        assert_eq!(parts.len(), n_elems);
        for &p in &parts {
            assert!((p as usize) < nparts, "partition out of range: {p}");
        }
    }

    #[test]
    fn partition_balanced() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_elems = mesh.n_elems();
        let nparts = 4;
        let parts = MetisPartitioner::partition_mesh(&mesh, nparts, &MetisOptions::default())
            .unwrap();
        let mut counts = vec![0usize; nparts];
        for &p in &parts { counts[p as usize] += 1; }
        let ideal = n_elems as f64 / nparts as f64;
        for (i, &c) in counts.iter().enumerate() {
            let imbalance = (c as f64 - ideal).abs() / ideal;
            assert!(imbalance < 0.6, "part {i}: count={c}, ideal={ideal:.1}, imbalance={imbalance:.2}");
        }
    }

    #[test]
    fn partition_single_part_is_identity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n = mesh.n_elems();
        let parts = MetisPartitioner::partition_mesh(&mesh, 1, &MetisOptions::default()).unwrap();
        assert!(parts.iter().all(|&p| p == 0));
        assert_eq!(parts.len(), n);
    }

    #[test]
    fn partition_simplex_serial() {
        use crate::mpi_test_env::test_world_comm;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let comm = test_world_comm();
        let pmesh = partition_simplex_metis(&mesh, &comm, &MetisOptions::default());
        assert_eq!(pmesh.global_n_elems(), mesh.n_elems());
        assert_eq!(pmesh.global_n_nodes(), mesh.n_nodes());
        pmesh.local_mesh().check().expect("local mesh failed check");
    }
}
