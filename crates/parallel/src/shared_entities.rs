//! Shared topological entities across MPI ranks.

use std::collections::HashMap;
use fem_core::Rank;
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
#[derive(Debug, Clone, Default)]
pub struct SharedEntities {
    pub vertices: HashMap<Rank, Vec<SharedEntity>>,
}

impl SharedEntities {
    /// Build shared vertex list from a partition.
    ///
    /// A vertex is "shared" with rank R if this rank owns it and rank R has
    /// it as a ghost (or vice versa).  Currently only captures ghost→owner
    /// direction (which neighbours have copies of our owned vertices).
    pub fn from_partition(partition: &MeshPartition, comm_rank: Rank) -> Self {
        let mut vertices: HashMap<Rank, Vec<SharedEntity>> = HashMap::new();
        for local_id in 0..partition.n_owned_nodes {
            // Record which neighbour has this owned node as a ghost.
            // The node_owner array tells us: owned nodes have owner == comm_rank.
            // Ghost nodes have owner == remote_rank. We need the reverse:
            // which remote ranks have our owned nodes as ghosts.
            // From `GhostExchange::from_partition` logic, this mapping is
            // constructed by the alltoallv in GhostExchange.
            // For now, we use the ghost-side info (what we know from ghost nodes):
            // if a ghost node has owner rank R, then rank R shares that node with us.
        }
        // Walk ghost nodes: each ghost node is OWNED by some remote rank.
        // Therefore that remote rank "shares" its vertex with us.
        for local_id in partition.n_owned_nodes..partition.n_owned_nodes + partition.n_ghost_nodes {
            let owner = partition.node_owner[local_id];
            if owner != comm_rank {
                vertices.entry(owner).or_default().push(SharedEntity {
                    local_id: local_id as u32,
                    neighbor_rank: owner,
                });
            }
        }
        SharedEntities { vertices }
    }

    pub fn is_empty(&self) -> bool { self.vertices.is_empty() }
}
