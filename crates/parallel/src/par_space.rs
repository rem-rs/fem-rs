//! Parallel finite element space.
//!
//! [`ParallelFESpace`] wraps a serial [`FESpace`] with DOF-level partitioning
//! and ghost exchange, enabling parallel assembly and solve.

use std::sync::Arc;

use fem_space::fe_space::{FESpace, SpaceType};
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
// MFEM: ParFiniteElementSpace
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
    /// H(curl) spaces always use the edge-based partition
    /// ([`DofPartition::from_edge_space`]); H(div) uses the edge partition in
    /// 2-D (RT dofs live on edges) and the face partition in 3-D
    /// ([`DofPartition::from_face_space`]).  For other P2+ spaces, use
    /// [`new_with_dof_manager`](Self::new_with_dof_manager).
    pub fn new<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        comm: Comm,
    ) -> Self {
        let dof_partition = match local_space.space_type() {
            SpaceType::HCurl => {
                DofPartition::from_edge_space(&local_space, par_mesh.partition(), &comm)
            }
            SpaceType::HDiv => {
                if local_space.mesh().topological_dim() == 3 {
                    DofPartition::from_face_space(&local_space, par_mesh.partition(), &comm)
                } else {
                    DofPartition::from_edge_space(&local_space, par_mesh.partition(), &comm)
                }
            }
            _ => DofPartition::from_mesh_partition(par_mesh.partition(), &comm),
        };
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

    /// Build a parallel FE space for a vector-valued space (vdim components)
    /// with byNODES block DOF layout (matches `VectorH1Space`).
    pub fn new_vector<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        vdim: usize,
        comm: Comm,
    ) -> Self {
        let dof_partition = DofPartition::from_vector_space(par_mesh.partition(), &comm, vdim);
        Self::finish(local_space, dof_partition, &comm)
    }

    /// Build a parallel FE space from an already-constructed `DofPartition`
    /// (e.g. [`DofPartition::from_l2_space`](crate::dof_partition::DofPartition::from_l2_space)).
    pub fn new_with_dof_partition(
        local_space: S,
        dof_partition: DofPartition,
        comm: Comm,
    ) -> Self {
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
        &[],  // owned elements (none — this is a DOF-based partition)
        &[],  // ghost elements (none)
        comm.rank(),
    );

    GhostExchange::from_partition(&tmp_partition, comm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_partition::partition_mesh;
    use crate::par_vector::ParVector;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use fem_space::dof_manager::DofManager;
    use fem_space::VectorH1Space;

    #[test]
    fn par_space_vector_h1_global_dofs_and_ghost_exchange() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let serial_n_dofs = 2 * mesh.n_nodes(); // vdim=2 × P1 nodes

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space =
                VectorH1Space::new(pmesh.local_mesh().clone(), 1, 2);
            let par_space =
                ParallelFESpace::new_vector(local_space, &pmesh, 2, comm.clone());

            assert_eq!(par_space.n_global_dofs(), serial_n_dofs);

            // Ghost exchange: fill owned with global ids, check ghosts receive
            // the matching global id from the owner rank.
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
                    comm.rank(),
                    data[lid]
                );
            }
        });
    }

    #[test]
    fn par_space_global_dofs_match_serial_p1() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let serial_n_dofs = mesh.n_nodes(); // P1

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            assert_eq!(par_space.n_global_dofs(), serial_n_dofs);
        });
    }

    #[test]
    fn par_space_global_dofs_match_serial_p2() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let serial_space = H1Space::new(mesh.clone(), 2);
        let serial_n_dofs = serial_space.n_dofs();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
    fn par_vector_h1_elasticity_matrix_consistent_across_partitions() {
        use fem_assembly::postproc::coefficient::PWConstCoeff;
        use fem_assembly::standard::ElasticityIntegrator;
        use std::sync::{Arc, Mutex};

        let mesh = Mesh::<2>::unit_square_tri(4);

        // Collect global (gid, y) after spmv with x = gid, from rank 0.
        fn run_partition<const N: usize>(
            mesh: Mesh<2>,
        ) -> Vec<(u32, f64)> {
            let out: Arc<Mutex<Option<Vec<(u32, f64)>>>> = Arc::new(Mutex::new(None));
            let out2 = Arc::clone(&out);
            let launcher = ThreadLauncher::new(WorkerConfig::new(N));
            launcher.launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let local = VectorH1Space::new(pmesh.local_mesh().clone(), 1, 2);
                let ps = ParallelFESpace::new_vector(local, &pmesh, 2, comm.clone());
                let lam = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
                let mu = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
                let el = ElasticityIntegrator::new(lam, mu);
                let a = ParAssembler::assemble_bilinear(&ps, &[&el], 3);
                let dp = ps.dof_partition();
                let mut x = ParVector::zeros(&ps);
                for pid in 0..dp.n_owned_dofs {
                    x.as_slice_mut()[pid] = dp.global_dof(pid as u32) as f64;
                }
                let mut y = ParVector::zeros(&ps);
                a.spmv(&mut x, &mut y);
                let owned: Vec<(u32, f64)> = (0..dp.n_owned_dofs)
                    .map(|pid| (dp.global_dof(pid as u32), y.as_slice()[pid]))
                    .collect();
                if comm.rank() == 0 {
                    let mut all = owned.clone();
                    for src in 1..comm.size() as i32 {
                        let gids: Vec<u32> = comm.recv(src, 101);
                        let vals: Vec<f64> = comm.recv(src, 102);
                        all.extend(gids.into_iter().zip(vals));
                    }
                    all.sort_unstable_by_key(|&(g, _)| g);
                    *out2.lock().unwrap() = Some(all);
                } else {
                    let gids: Vec<u32> = owned.iter().map(|&(g, _)| g).collect();
                    let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                    comm.send(0, 101, &gids);
                    comm.send(0, 102, &vals);
                }
            });
            let mut guard = out.lock().unwrap();
            guard.take().unwrap()
        }

        let np1 = run_partition::<1>(mesh.clone());
        let np2 = run_partition::<2>(mesh.clone());
        assert_eq!(np1.len(), np2.len(), "global dof count mismatch");
        let mut max_diff = 0.0_f64;
        let mut n_bad = 0;
        for ((g1, y1), (g2, y2)) in np1.iter().zip(np2.iter()) {
            assert_eq!(g1, g2, "gid order mismatch");
            let d = (y1 - y2).abs();
            if d > 1e-12 {
                n_bad += 1;
            }
            max_diff = max_diff.max(d);
        }
        assert!(
            max_diff < 1e-10,
            "matrix differs across partitions: max_diff={max_diff:e}, n_bad={n_bad}"
        );
    }

    #[test]
    fn par_space_ghost_exchange_p1() {
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
