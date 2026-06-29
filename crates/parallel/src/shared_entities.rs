//! Shared topological entities (vertices, edges, faces) across MPI ranks.
//!
//! When a mesh is partitioned, each rank owns a subset of mesh entities.
//! Entities that lie on the partition boundary are **shared**: one rank
//! "owns" the entity (e.g., the rank with the lowest global ID) and other
//! ranks that have the entity as a ghost are "neighbors".  The
//! [`SharedEntities`] struct tracks this relationship for vertices, edges,
//! and faces, enabling consistent ghost exchange across all entity types.

use std::collections::HashMap;
use fem_core::Rank;
use fem_mesh::topology::MeshTopology;
use crate::partition::MeshPartition;

/// A vertex/edge/face shared with a neighbour rank.
#[derive(Debug, Clone)]
pub struct SharedEntity {
    /// Local ID of this entity on the current rank.
    pub local_id: u32,
    /// Neighbour rank that also has this entity as a ghost.
    pub neighbor_rank: Rank,
}

/// Shared entity groups indexed by neighbour rank.
///
/// Provides separate maps for vertices, edges, and faces so that
/// H¹ (vertex‑based), H(curl) (edge‑based) and H(div) (face‑based)
/// exchange logic can each use the appropriate topology level.
#[derive(Debug, Clone, Default)]
pub struct SharedEntities {
    pub vertices: HashMap<Rank, Vec<SharedEntity>>,
    /// Shared edges (only populated when `build_edges` / `build_faces` is called).
    pub edges: HashMap<Rank, Vec<SharedEntity>>,
    /// Shared faces (only populated when `build_faces` is called).
    pub faces: HashMap<Rank, Vec<SharedEntity>>,
}

impl SharedEntities {
    /// Build shared‑vertex list from a partition.
    ///
    /// A vertex is "shared" with rank R if this rank owns it and rank R has
    /// it as a ghost (or vice versa).  The current implementation records
    /// the ghost→owner direction, i.e. which neighbours hold copies of
    /// our owned vertices.
    pub fn from_partition(partition: &MeshPartition, comm_rank: Rank) -> Self {
        let mut vertices: HashMap<Rank, Vec<SharedEntity>> = HashMap::new();
        // Owned‑node side: for each owned node, find which ranks have it as ghost.
        // We derive this from the ghost‑node list: for each ghost node, the owner
        // rank R shares that node with us.
        for local_id in partition.n_owned_nodes..partition.n_owned_nodes + partition.n_ghost_nodes {
            let owner = partition.node_owner[local_id];
            if owner != comm_rank {
                vertices.entry(owner).or_default().push(SharedEntity {
                    local_id: local_id as u32,
                    neighbor_rank: owner,
                });
            }
        }
        SharedEntities { vertices, edges: HashMap::new(), faces: HashMap::new() }
    }

    /// Build shared‑edge and shared‑face groups from mesh topology + partition.
    ///
    /// An edge (or face) is "shared" with rank R if all its constituent nodes
    /// are shared with rank R *and* the entity lies on the partition boundary
    /// (i.e., not all nodes are owned by this rank).
    ///
    /// This is a topological derivation: it does not require additional
    /// all‑to‑all communication.  Call after [`from_partition`].
    pub fn build_edges_faces<M: MeshTopology>(&mut self, mesh: &M, partition: &MeshPartition, comm_rank: Rank) {
        let owned_end = partition.n_owned_nodes;
        let is_owned = |lid: usize| -> bool { lid < owned_end };

        let n_edges = 0;
        let _ = n_edges;
        // Edge sharing: check each element's edge definition.
        // For each unique sorted node pair (edge key), determine if that edge
        // is "shared" (at least one node is ghost).  If so, the neighbor rank
        // is the owner of the ghost node (or the first ghost node owner).
        //
        // Note: SimplexMesh does not currently expose a direct edge enumeration;
        // this requires an `edge_conn` / `edge_to_elem` table (P4-1 gap).
        // For now, we skip edge/face population — the infrastructure is
        // ready to be filled once the mesh topology provides those tables.
        //
        //   for e in mesh.elem_iter() {
        //       let nodes = mesh.element_nodes(e);
        //       for (i, j) in local_edge_pairs(mesh.element_type(e)) {
        //           let (ni, nj) = (nodes[i] as usize, nodes[j] as usize);
        //           let li = find_local(partition, ni);
        //           let lj = find_local(partition, nj);
        //           if !is_owned(li) || !is_owned(lj) { /* ghost edge */ }
        //       }
        //   }

        // Face sharing (3-D, triangular/quad faces): same structural pattern.
        //
        //   for e in mesh.elem_iter() {
        //       let nodes = mesh.element_nodes(e);
        //       for face_nodes in local_face_defs(mesh.element_type(e)) {
        //           // Determine if this face lies on partition boundary.
        //       }
        //   }
        //
        // Both blocks will be activated when `MeshTopology::n_edges()` and
        // edge/face enumeration are added (see improvement plan P4-1).
    }

    /// Convenience: build from partition + mesh topology in one call.
    pub fn from_partition_with_mesh<M: MeshTopology>(
        partition: &MeshPartition,
        comm_rank: Rank,
        mesh: &M,
    ) -> Self {
        let mut se = Self::from_partition(partition, comm_rank);
        se.build_edges_faces(mesh, partition, comm_rank);
        se
    }

    pub fn is_empty(&self) -> bool { self.vertices.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use crate::par_simplex::partition_simplex;
    use crate::mpi_test_env::test_world_comm;

    #[test]
    fn shared_vertices_on_serial_rank() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_total_nodes = mesh.n_nodes();
        let comm_rank: Rank = 0;
        let pmesh = partition_simplex(&mesh, &test_world_comm());
        let se = SharedEntities::from_partition(pmesh.partition(), comm_rank);
        let n_shared: usize = se.vertices.values().map(|v| v.len()).sum();
        eprintln!("Serial rank: {n_shared} shared vertices out of {n_total_nodes} total");
        assert_eq!(n_shared, 0, "serial partition has no shared vertices");
    }

    #[test]
    fn shared_edges_placeholder_does_not_panic() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let pmesh = partition_simplex(&mesh, &test_world_comm());
        let mut se = SharedEntities::from_partition(pmesh.partition(), 0);
        se.build_edges_faces(&mesh, pmesh.partition(), 0);
    }
}
