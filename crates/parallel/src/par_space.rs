//! Parallel finite element space.
//!
//! [`ParallelFESpace`] wraps a serial [`FESpace`] with DOF-level partitioning
//! and ghost exchange, enabling parallel assembly and solve.

use std::sync::Arc;

use fem_space::fe_space::FESpace;
use fem_mesh::topology::MeshTopology;

use crate::comm::Comm;
use crate::dof_partition::DofPartition;
use crate::ghost::GhostExchange;
use crate::par_mesh::ParallelMesh;

/// A parallel finite element space: wraps a serial FESpace with DOF-level
/// partitioning and ghost exchange.
///
/// For P1 spaces, DOFs correspond 1:1 with mesh nodes and the DOF ghost
/// exchange mirrors the node ghost exchange.
pub struct ParallelFESpace<S: FESpace> {
    local_space: S,
    dof_partition: DofPartition,
    dof_ghost_exchange: Arc<GhostExchange>,
    comm: Comm,
    n_global_dofs: usize,
}

impl<S: FESpace> ParallelFESpace<S>
where
    S::Mesh: MeshTopology,
{
    /// Build a parallel FE space from a local space and parallel mesh.
    ///
    /// The DOF partition is derived from the mesh partition (P1: DOFs = nodes).
    /// A DOF-level ghost exchange is built for communicating DOF values between
    /// ranks.
    pub fn new<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        comm: Comm,
    ) -> Self {
        let dof_partition = DofPartition::from_mesh_partition(par_mesh.partition(), &comm);

        // Build DOF-level ghost exchange using the same algorithm as the mesh
        // ghost exchange but based on DOF ownership.
        let dof_ghost_exchange = Arc::new(build_dof_ghost_exchange(&dof_partition, &comm));

        let n_global_dofs = comm.allreduce_sum_i64(dof_partition.n_owned_dofs as i64) as usize;

        ParallelFESpace {
            local_space,
            dof_partition,
            dof_ghost_exchange,
            comm,
            n_global_dofs,
        }
    }

    /// Reference to the local (serial) FE space.
    #[inline]
    pub fn local_space(&self) -> &S { &self.local_space }

    /// Reference to the DOF partition.
    #[inline]
    pub fn dof_partition(&self) -> &DofPartition { &self.dof_partition }

    /// Total number of DOFs across all ranks.
    #[inline]
    pub fn n_global_dofs(&self) -> usize { self.n_global_dofs }

    /// Number of local DOFs (owned + ghost).
    #[inline]
    pub fn n_local_dofs(&self) -> usize { self.dof_partition.n_total_dofs() }

    /// The MPI communicator.
    #[inline]
    pub fn comm(&self) -> &Comm { &self.comm }

    /// Arc-wrapped DOF ghost exchange (shared with ParVector/ParCsrMatrix).
    #[inline]
    pub fn dof_ghost_exchange_arc(&self) -> Arc<GhostExchange> {
        Arc::clone(&self.dof_ghost_exchange)
    }

    /// Forward exchange: propagate owned DOF values into ghost slots.
    pub fn forward_dof_exchange(&self, data: &mut [f64]) {
        self.dof_ghost_exchange.forward(&self.comm, data);
    }

    /// Reverse exchange: accumulate ghost DOF contributions back to owners.
    pub fn reverse_dof_exchange(&self, data: &mut [f64]) {
        self.dof_ghost_exchange.reverse(&self.comm, data);
    }
}

/// Build a `GhostExchange` from DOF ownership data.
///
/// Uses the same algorithm as `GhostExchange::from_partition` but with DOF
/// indices instead of node indices.
fn build_dof_ghost_exchange(dof_part: &DofPartition, comm: &Comm) -> GhostExchange {
    use crate::partition::MeshPartition;

    // Reuse the existing GhostExchange::from_partition by building a
    // temporary MeshPartition with DOF data.  This avoids duplicating the
    // exchange-pattern construction logic.
    let tmp_partition = MeshPartition::from_partitioner(
        &dof_part.global_dof_ids[..dof_part.n_owned_dofs],
        &dof_part.ghost_dofs().map(|(lid, owner)| {
            (dof_part.global_dof(lid), owner)
        }).collect::<Vec<_>>(),
        &[], // no elements needed
        comm.rank(),
    );

    GhostExchange::from_partition(&tmp_partition, comm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_simplex::partition_simplex;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    #[test]
    fn par_space_global_dofs_match_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let serial_n_dofs = mesh.n_nodes(); // P1

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            assert_eq!(par_space.n_global_dofs(), serial_n_dofs);
        });
    }

    #[test]
    fn par_space_ghost_exchange() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let n_local = par_space.n_local_dofs();
            let n_owned = par_space.dof_partition().n_owned_dofs;

            // Set owned DOFs to their global ID, ghost DOFs to -1.
            let mut data = vec![-1.0_f64; n_local];
            for lid in 0..n_owned {
                let gid = par_space.dof_partition().global_dof(lid as u32);
                data[lid] = gid as f64;
            }

            // Forward exchange should fill ghost slots with the correct global IDs.
            par_space.forward_dof_exchange(&mut data);

            for lid in n_owned..n_local {
                let expected = par_space.dof_partition().global_dof(lid as u32) as f64;
                assert!(
                    (data[lid] - expected).abs() < 1e-14,
                    "rank {}: ghost DOF local={lid} expected {expected}, got {}",
                    comm.rank(), data[lid]
                );
            }
        });
    }
}
