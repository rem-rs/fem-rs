//! DOF-level ownership for a parallel FE space.
//!
//! [`DofPartition`] extends mesh-level partitioning to DOF indices.  For
//! H1/P1, DOFs map 1:1 to nodes, so this mirrors [`MeshPartition`]'s node
//! layout.  Higher-order spaces (P2) will add edge/face DOFs with analogous
//! ownership rules.

use std::collections::HashMap;
use fem_core::Rank;
use crate::comm::Comm;
use crate::partition::MeshPartition;

/// DOF-level partition descriptor for one MPI rank.
///
/// Local DOF layout (contiguous in memory):
/// ```text
/// [ owned DOFs 0 .. n_owned )  [ ghost DOFs n_owned .. n_owned+n_ghost )
/// ```
#[derive(Debug, Clone)]
pub struct DofPartition {
    /// Number of locally owned DOFs.
    pub n_owned_dofs: usize,
    /// Number of ghost DOFs.
    pub n_ghost_dofs: usize,
    /// Global DOF IDs for all local DOFs, length `n_owned + n_ghost`.
    pub global_dof_ids: Vec<u32>,
    /// Owner rank for each local DOF.
    pub dof_owner: Vec<Rank>,
    /// Starting global DOF index for this rank's owned range.
    pub global_dof_offset: usize,
    /// Global -> local DOF mapping.
    dof_global_to_local: HashMap<u32, u32>,
}

impl DofPartition {
    /// Build a DOF partition for P1 (DOFs = nodes) from a mesh partition.
    ///
    /// Computes global DOF offsets via an exclusive prefix sum across ranks.
    pub fn from_mesh_partition(partition: &MeshPartition, comm: &Comm) -> Self {
        let n_owned = partition.n_owned_nodes;
        let n_ghost = partition.n_ghost_nodes;

        // Copy node ownership as DOF ownership.
        let global_dof_ids = partition.global_node_ids.clone();
        let dof_owner = partition.node_owner.clone();

        // Compute global DOF offset for this rank.
        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;

        // Build reverse lookup.
        let dof_global_to_local: HashMap<u32, u32> = global_dof_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();

        DofPartition {
            n_owned_dofs: n_owned,
            n_ghost_dofs: n_ghost,
            global_dof_ids,
            dof_owner,
            global_dof_offset,
            dof_global_to_local,
        }
    }

    /// Total local DOF count (owned + ghost).
    #[inline]
    pub fn n_total_dofs(&self) -> usize {
        self.n_owned_dofs + self.n_ghost_dofs
    }

    /// Global ID of a local DOF.
    #[inline]
    pub fn global_dof(&self, local_id: u32) -> u32 {
        self.global_dof_ids[local_id as usize]
    }

    /// Local ID of a global DOF, or `None` if not present on this rank.
    #[inline]
    pub fn local_dof(&self, global_id: u32) -> Option<u32> {
        self.dof_global_to_local.get(&global_id).copied()
    }

    /// `true` if `local_id` refers to an owned (non-ghost) DOF.
    #[inline]
    pub fn is_owned_dof(&self, local_id: u32) -> bool {
        (local_id as usize) < self.n_owned_dofs
    }

    /// Owner rank of local DOF `local_id`.
    #[inline]
    pub fn dof_owner(&self, local_id: u32) -> Rank {
        self.dof_owner[local_id as usize]
    }

    /// Iterate over ghost DOFs: yields `(local_id, owner_rank)`.
    pub fn ghost_dofs(&self) -> impl Iterator<Item = (u32, Rank)> + '_ {
        let start = self.n_owned_dofs;
        (start..self.n_total_dofs()).map(move |lid| {
            (lid as u32, self.dof_owner[lid])
        })
    }
}

/// Exclusive prefix sum across MPI ranks.
///
/// Each rank contributes `local_val`; the result on rank `r` is the sum of
/// `local_val` from ranks `0, 1, ..., r-1`.  Rank 0 always gets 0.
///
/// Uses a simple linear chain of send/recv for correctness with any backend.
fn exclusive_scan_i64(comm: &Comm, local_val: i64) -> i64 {
    let rank = comm.rank();
    let size = comm.size();

    if size <= 1 {
        return 0;
    }

    const TAG: i32 = 0x6000;

    if rank == 0 {
        // Send our partial sum (just our value) to rank 1.
        let my_sum = local_val;
        comm.send_bytes(1, TAG, &my_sum.to_le_bytes());
        0
    } else {
        // Receive the prefix sum from the previous rank.
        let prev_bytes = comm.recv_bytes(rank - 1, TAG);
        let prev_sum = i64::from_le_bytes(prev_bytes.try_into().unwrap());

        // Forward cumulative sum to next rank if we're not last.
        if (rank as usize) < size - 1 {
            let my_sum = prev_sum + local_val;
            comm.send_bytes(rank + 1, TAG, &my_sum.to_le_bytes());
        }

        prev_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_simplex::partition_simplex;
    use fem_mesh::SimplexMesh;
    use std::sync::{Arc, Mutex};

    #[test]
    fn dof_partition_p1_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_nodes = mesh.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let dof_part = DofPartition::from_mesh_partition(pmesh.partition(), &comm);

            assert_eq!(dof_part.n_owned_dofs, n_nodes);
            assert_eq!(dof_part.n_ghost_dofs, 0);
            assert_eq!(dof_part.global_dof_offset, 0);
            assert_eq!(dof_part.n_total_dofs(), n_nodes);
        });
    }

    #[test]
    fn dof_partition_p1_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let total_nodes = mesh.n_nodes();

        let results = Arc::new(Mutex::new(Vec::new()));

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        let results_clone = Arc::clone(&results);
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let dof_part = DofPartition::from_mesh_partition(pmesh.partition(), &comm);

            let rank = comm.rank();
            let n_owned = dof_part.n_owned_dofs;
            let offset = dof_part.global_dof_offset;

            results_clone.lock().unwrap().push((rank, n_owned, offset));

            // Verify global total of owned DOFs = total nodes.
            let global_owned = comm.allreduce_sum_i64(n_owned as i64) as usize;
            assert_eq!(global_owned, total_nodes,
                "sum of owned DOFs ({global_owned}) != total nodes ({total_nodes})");

            // Verify all owned DOFs are actually owned by this rank.
            for lid in 0..dof_part.n_owned_dofs as u32 {
                assert!(dof_part.is_owned_dof(lid));
                assert_eq!(dof_part.dof_owner(lid), rank);
            }

            // Verify ghost DOFs are owned by another rank.
            for (lid, owner) in dof_part.ghost_dofs() {
                assert!(!dof_part.is_owned_dof(lid));
                assert_ne!(owner, rank);
            }
        });

        // Verify offsets are consecutive.
        let mut res = results.lock().unwrap().clone();
        res.sort_by_key(|(r, _, _)| *r);
        assert_eq!(res[0].2, 0, "rank 0 offset must be 0");
        assert_eq!(res[1].2, res[0].1, "rank 1 offset must equal rank 0's n_owned");
    }

    #[test]
    fn exclusive_scan_four_ranks() {
        let results = Arc::new(Mutex::new(Vec::new()));

        let launcher = ThreadLauncher::new(WorkerConfig::new(4));
        let results_clone = Arc::clone(&results);
        launcher.launch(move |comm| {
            let rank = comm.rank();
            // Each rank contributes rank+1.
            let val = (rank + 1) as i64;
            let scan = exclusive_scan_i64(&comm, val);
            results_clone.lock().unwrap().push((rank, scan));
        });

        let mut res = results.lock().unwrap().clone();
        res.sort_by_key(|(r, _)| *r);
        // Expected: rank 0 -> 0, rank 1 -> 1, rank 2 -> 1+2=3, rank 3 -> 1+2+3=6
        assert_eq!(res[0].1, 0);
        assert_eq!(res[1].1, 1);
        assert_eq!(res[2].1, 3);
        assert_eq!(res[3].1, 6);
    }
}
