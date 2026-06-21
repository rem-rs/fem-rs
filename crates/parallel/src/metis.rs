//! Mesh partitioning using [`fem_rmetis`].
//!
//! Wraps the dual-graph builder and partitioner from `fem-rmetis` into the
//! parallel mesh pipeline ([`ParallelMesh`] + [`extract_submesh_from_partition`]).
//!
//! # Usage
//! ```rust,ignore
//! use fem_parallel::metis::{MetisPartitioner, MetisOptions};
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(16);
//! let opt  = MetisOptions::default();
//! let parts = MetisPartitioner::partition_mesh(&mesh, 4, &opt).unwrap();
//! ```

use fem_core::Rank;
use fem_mesh::SimplexMesh;

use crate::{Comm, MeshPartition, par_mesh::ParallelMesh};
use crate::mesh_serde;
use crate::par_simplex::{extract_submesh_from_partition, STREAM_TAG_BASE};

// ─── Options ─────────────────────────────────────────────────────────────────

/// Options for the METIS partitioner.
#[derive(Debug, Clone, Default)]
pub struct MetisOptions {
    /// If true, print partition statistics to stdout.
    pub verbose: bool,
}

// ─── MetisPartitioner ─────────────────────────────────────────────────────────

/// Mesh partitioner backed by [`fem_rmetis`].
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
        assert!(mesh.n_elems() > 0, "mesh has no elements");

        let partition = fem_rmetis::partition_mesh(mesh, nparts);

        if opts.verbose {
            let mut counts = vec![0usize; nparts];
            for &p in &partition { counts[p as usize] += 1; }
            println!("[MetisPartitioner] nparts={nparts}, counts={counts:?}");
        }

        Ok(partition)
    }
}

// ─── partition_simplex_metis ──────────────────────────────────────────────────

/// Distribute `mesh` across `comm.size()` ranks using k-way partitioning.
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

// ─── partition_simplex_metis_streaming ─────────────────────────────────────────

/// Streaming partition: only rank 0 holds the full mesh.
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
        let elem_part = MetisPartitioner::partition_mesh(m, size, opts)?;

        for target in 1..size as Rank {
            let (sub_mesh, sub_part) = extract_submesh_from_partition(
                m, target, &elem_part,
            );
            let encoded = mesh_serde::encode_submesh(&sub_mesh, &sub_part);
            comm.send_bytes(target, STREAM_TAG_BASE + target, &encoded);
        }

        let (local_mesh, partition) = extract_submesh_from_partition(m, 0, &elem_part);
        Ok(ParallelMesh::new(local_mesh, comm.clone(), partition))
    } else {
        let local_rank = comm.rank();
        let buf = comm.recv_bytes(0, STREAM_TAG_BASE + local_rank);
        let (local_mesh, partition) = mesh_serde::decode_submesh::<D>(&buf)?;
        Ok(ParallelMesh::new(local_mesh, comm.clone(), partition))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{ElementType, SimplexMesh};

    #[test]
    fn partition_covers_all_elements() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_elems = mesh.n_elems();
        let parts = MetisPartitioner::partition_mesh(&mesh, 4, &MetisOptions::default()).unwrap();
        assert_eq!(parts.len(), n_elems);
        assert!(parts.iter().all(|&p| (p as usize) < 4));
    }

    #[test]
    fn partition_balanced() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_elems = mesh.n_elems();
        let nparts = 4;
        let parts = MetisPartitioner::partition_mesh(&mesh, nparts, &MetisOptions::default()).unwrap();
        let mut counts = vec![0usize; nparts];
        for &p in &parts { counts[p as usize] += 1; }
        let ideal = n_elems as f64 / nparts as f64;
        for &c in &counts {
            assert!((c as f64 - ideal).abs() / ideal < 0.6);
        }
    }

    #[test]
    fn partition_single_part_is_identity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let parts = MetisPartitioner::partition_mesh(&mesh, 1, &MetisOptions::default()).unwrap();
        assert!(parts.iter().all(|&p| p == 0));
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
