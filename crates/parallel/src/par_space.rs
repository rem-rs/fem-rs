//! Parallel finite element space.
//!
//! [`ParallelFESpace`] wraps a serial [`FESpace`] with DOF-level partitioning
//! and ghost exchange, enabling parallel assembly and solve.

use std::sync::Arc;

use fem_space::fe_space::FESpace;
use fem_space::dof_manager::DofManager;
use fem_mesh::topology::MeshTopology;

use crate::comm::Comm;
use crate::dof_partition::DofPartition;
use crate::ghost::GhostExchange;
use crate::par_mesh::ParallelMesh;

/// A parallel finite element space: wraps a serial FESpace with DOF-level
/// partitioning and ghost exchange.
///
/// For P1 spaces, DOFs correspond 1:1 with mesh nodes.  For P2, edge DOFs
/// are added with ownership based on the minimum-owner-rank rule.
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
    /// For P2+ spaces, use [`new_with_dof_manager`](Self::new_with_dof_manager).
    pub fn new<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        comm: Comm,
    ) -> Self {
        let dof_partition = DofPartition::from_mesh_partition(par_mesh.partition(), &comm);
        Self::finish(local_space, dof_partition, &comm)
    }

    /// Build a parallel FE space with an explicit `DofManager`.
    ///
    /// This constructor supports P2 (and future higher-order) spaces by using
    /// the edge-to-DOF mapping from the `DofManager` to determine edge DOF
    /// ownership across ranks.
    pub fn new_with_dof_manager<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        dof_manager: &DofManager,
        comm: Comm,
    ) -> Self {
        let dof_partition = DofPartition::from_dof_manager(
            dof_manager, par_mesh.partition(), &comm,
        );
        Self::finish(local_space, dof_partition, &comm)
    }

    /// Build a parallel FE space for edge-DOF-only spaces (H(curl), H(div) 2D).
    ///
    /// Uses edge-based DOF partitioning where `owner(edge) = min(owner(endpoints))`.
    pub fn new_for_edge_space<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        comm: Comm,
    ) -> Self {
        let dof_partition = DofPartition::from_edge_space(
            &local_space, par_mesh.partition(), &comm,
        );
        Self::finish(local_space, dof_partition, &comm)
    }

    /// Common construction: build ghost exchange and count global DOFs.
    fn finish(local_space: S, dof_partition: DofPartition, comm: &Comm) -> Self {
        let dof_ghost_exchange = Arc::new(build_dof_ghost_exchange(&dof_partition, comm));
        let n_global_dofs = comm.allreduce_sum_i64(dof_partition.n_owned_dofs as i64) as usize;

        ParallelFESpace {
            local_space,
            dof_partition,
            dof_ghost_exchange,
            comm: comm.clone(),
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
fn build_dof_ghost_exchange(dof_part: &DofPartition, comm: &Comm) -> GhostExchange {
    use crate::partition::MeshPartition;

    let tmp_partition = MeshPartition::from_partitioner(
        &dof_part.global_dof_ids[..dof_part.n_owned_dofs],
        &dof_part.ghost_dofs().map(|(lid, owner)| {
            (dof_part.global_dof(lid), owner)
        }).collect::<Vec<_>>(),
        &[],
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
    use fem_space::dof_manager::DofManager;

    #[test]
    fn par_space_global_dofs_match_serial_p1() {
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
    fn par_space_global_dofs_match_serial_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let serial_space = H1Space::new(mesh.clone(), 2);
        let serial_n_dofs = serial_space.n_dofs();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = DofManager::new(&local_mesh, 2);
            let local_space = H1Space::new(local_mesh, 2);
            let par_space = ParallelFESpace::new_with_dof_manager(
                local_space, &pmesh, &dm, comm.clone(),
            );

            assert_eq!(par_space.n_global_dofs(), serial_n_dofs);
        });
    }

    #[test]
    fn par_space_ghost_exchange_p1() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let n_local = par_space.n_local_dofs();
            let n_owned = par_space.dof_partition().n_owned_dofs;

            let mut data = vec![-1.0_f64; n_local];
            for lid in 0..n_owned {
                let gid = par_space.dof_partition().global_dof(lid as u32);
                data[lid] = gid as f64;
            }

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

    #[test]
    fn par_space_ghost_exchange_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = DofManager::new(&local_mesh, 2);
            let local_space = H1Space::new(local_mesh, 2);
            let par_space = ParallelFESpace::new_with_dof_manager(
                local_space, &pmesh, &dm, comm.clone(),
            );

            let n_local = par_space.n_local_dofs();
            let n_owned = par_space.dof_partition().n_owned_dofs;

            let mut data = vec![-1.0_f64; n_local];
            for lid in 0..n_owned {
                let gid = par_space.dof_partition().global_dof(lid as u32);
                data[lid] = gid as f64;
            }

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
