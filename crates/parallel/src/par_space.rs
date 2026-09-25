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
    /// H1 order ≥ 2 uses the space's own [`DofManager`] so edge/face DOFs are
    /// partitioned too ([`DofPartition::from_dof_manager`]); the mesh-node
    /// partition would only cover the vertices (cylinder-hex Q2: 364 instead of
    /// 2443 DOFs).  H(curl) spaces always use the edge-based partition
    /// ([`DofPartition::from_edge_space`]); H(div) uses the edge partition in
    /// 2-D (RT dofs live on edges) and the face partition in 3-D
    /// ([`DofPartition::from_face_space`]).  For P2+ spaces with a `DofManager`
    /// in hand, [`new_with_dof_manager`](Self::new_with_dof_manager) is
    /// equivalent and explicit.
    pub fn new<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        comm: Comm,
    ) -> Self
    where
        S: 'static,
        M: 'static,
    {
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
            SpaceType::L2 => {
                // Discontinuous L2 DOFs are owned by their element (no sharing):
                // the generic `from_mesh_partition` fallback is wrong (it counts
                // P1-style node DOFs, e.g. 36 nodes vs 48 P0 elements).
                let l2 = (&local_space as &dyn std::any::Any)
                    .downcast_ref::<fem_space::L2Space<M>>()
                    .expect("SpaceType::L2 requires fem_space::L2Space");
                DofPartition::from_l2_space(l2, par_mesh.partition(), &comm)
            }
            // H¹ (and the other node-based spaces: H¹ orders, the
            // non-conforming `CrSpace` and the IGA spaces).  Order 1 has one
            // DOF per node, so the node partition is exact; from order 2 on,
            // edge/face/interior DOFs must come from the `DofManager` (only
            // `H1Space` exposes one — `new_with_dof_manager` is the entry point
            // for the other families).
            _ => match local_space.order() {
                0 | 1 => DofPartition::from_mesh_partition(par_mesh.partition(), &comm),
                _ => match (&local_space as &dyn std::any::Any)
                    .downcast_ref::<fem_space::H1Space<M>>()
                {
                    Some(h1) => DofPartition::from_dof_manager(
                        h1.dof_manager(),
                        par_mesh.partition(),
                        &comm,
                    ),
                    None => DofPartition::from_mesh_partition(par_mesh.partition(), &comm),
                },
            },
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
        Self::new_for_edge_space_ordered(local_space, par_mesh, comm, None)
    }

    /// [`Self::new_for_edge_space`] with the MFEM `UpdateVertices` node
    /// creation order (`gid → order`); see
    /// [`DofPartition::from_edge_space_ordered`] — needed after an AMR
    /// partition rebuild whose gids are not in MFEM creation order
    /// (pex6 np2 RT0 orientation).
    pub fn new_for_edge_space_ordered<M: MeshTopology>(
        local_space: S,
        par_mesh: &ParallelMesh<M>,
        comm: Comm,
        creation_order: Option<&std::collections::HashMap<u32, u32>>,
    ) -> Self {
        let dof_partition = DofPartition::from_edge_space_ordered(
            &local_space, par_mesh.partition(), &comm, creation_order,
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

    /// Distributed essential-boundary detection — the fem-rs counterpart of
    /// MFEM `ParFiniteElementSpace::GetEssentialTrueDofs`
    /// (fem/pfespace.cpp:1165); `GetBoundaryTrueDofs` (fem/fespace.hpp:1379)
    /// is the same entry with **all** boundary attributes marked.
    ///
    /// Returns the **rank-local full-dof ids** (the space's own dof numbering,
    /// the same id space the serial collectors
    /// `fem_space::constraints::boundary_dofs*` return) of the essential set,
    /// synchronized across the dof ghost halo so that every rank sees the
    /// complete, owner-consistent set:
    ///
    /// 1. rank-local detection on the local mesh (MFEM:
    ///    `FiniteElementSpace::GetEssentialVDofs`);
    /// 2. the 0/1 marker is OR-synchronized over the dof halo — reverse
    ///    (ghost → owner) then threshold, forward (owner → ghost) — MFEM:
    ///    `ParFiniteElementSpace::Synchronize`, "implement allreduce(|) as
    ///    reduce(|) + broadcast" (pfespace.cpp:1142);
    /// 3. the caller restricts to true dofs by mapping each id through
    ///    `dof_partition().permute_dof`: slots `< n_owned_dofs` are the true
    ///    dofs (global true-dof id = `global_dof(pid)`), the rest are ghosts
    ///    whose owner reports the same dof — the fem-rs equivalent of MFEM's
    ///    `GetRestrictionMatrix()->BooleanMult` (pfespace.cpp:1181).
    ///
    /// This closes the D124 defect: a rank that owns a shared-entity dof but
    /// holds none of its boundary faces (face ownership follows the minimum
    /// global node id in `partition_mesh::extract_local_faces`, dof ownership
    /// the per-family minimum-owner rules) nevertheless reports the dof,
    /// because the rank that holds the face pushes the marker through the
    /// halo — what MFEM's `GetEssentialTrueDofs` provides for free.
    ///
    /// At `comm.size() == 1` the halo is trivial and the result is the serial
    /// collector's output **bitwise**.
    ///
    /// Supported families: `H1Space` (any order, via `DofManager`),
    /// `HCurlSpace`, `HDivSpace` — the same dispatch as
    /// [`ParallelFESpace::new`].  (Vector spaces route through their scalar
    /// space / vdim expansion by the caller, as in `mfem_pex2`.)
    pub fn essential_true_dofs(
        &self,
        bdr_attr_is_ess: &[i32],
    ) -> Vec<fem_core::types::DofId>
    where
        S: 'static,
        S::Mesh: 'static,
    {
        use std::any::Any;

        use fem_space::constraints::{boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv};

        // 1. Rank-local detection (MFEM: FiniteElementSpace::GetEssentialVDofs
        //    on the local mesh).
        let mesh = self.local_space.mesh();
        let local: Vec<fem_core::types::DofId> = {
            let any = &self.local_space as &dyn Any;
            if let Some(h1) = any.downcast_ref::<fem_space::H1Space<S::Mesh>>() {
                boundary_dofs(mesh, h1.dof_manager(), bdr_attr_is_ess)
            } else if let Some(nd) = any.downcast_ref::<fem_space::HCurlSpace<S::Mesh>>() {
                boundary_dofs_hcurl(mesh, nd, bdr_attr_is_ess)
            } else if let Some(rt) = any.downcast_ref::<fem_space::HDivSpace<S::Mesh>>() {
                boundary_dofs_hdiv(mesh, rt, bdr_attr_is_ess)
            } else {
                panic!(
                    "ParallelFESpace::essential_true_dofs: no boundary collector for \
                     this space family (supported: H1Space, HCurlSpace, HDivSpace)"
                )
            }
        };

        let dp = &self.dof_partition;
        let n_owned = dp.n_owned_dofs;
        let n_total = dp.n_total_dofs();

        // 2. Synchronize the marker over the dof halo (MFEM:
        //    ParFiniteElementSpace::Synchronize — allreduce(BitOR) over each
        //    dof's group).  The 0/1 flags accumulate additively in `reverse`,
        //    so the owner-side test is `> 0.5`.
        let mut marker = vec![0.0_f64; n_total];
        for &d in &local {
            marker[dp.permute_dof(d) as usize] = 1.0;
        }
        self.reverse_dof_exchange(&mut marker);

        // 3. Restrict to the owned (true-dof) slots — MFEM:
        //    GetRestrictionMatrix()->BooleanMult (pfespace.cpp:1181) — and
        //    push the owner's marker back into the ghost slots (MFEM:
        //    Synchronize's Bcast leg) so ghosts agree with their owner.
        let mut true_marker = vec![0.0_f64; n_total];
        for (i, flag) in true_marker.iter_mut().enumerate().take(n_owned) {
            *flag = if marker[i] > 0.5 { 1.0 } else { 0.0 };
        }
        self.forward_dof_exchange(&mut true_marker);

        let mut out: Vec<fem_core::types::DofId> = (0..n_total)
            .filter(|&i| marker[i] > 0.5 || true_marker[i] > 0.5)
            .map(|i| dp.unpermute_dof(i as u32))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
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

    /// P2 (quadratic H¹) DOF partitioning must be partition-invariant: assemble
    /// the diffusion matrix on np1 and np2, apply `y = A x` with `x = gid`, and
    /// compare `y` for every owned DOF keyed by its **canonical mesh entity**
    /// (vertex global node / sorted global node pair for an edge).  P2 edge DOF
    /// gids are assigned per-rank in contiguous owner blocks, so they are NOT
    /// comparable across partitions — the entity key is.  This exercises the
    /// P2 vertex+edge ownership rule, the `dm_to_partition` permutation and the
    /// ghost edge global-id exchange end to end.
    ///
    /// D122-1 (round 73) moved the edge owner from the min *endpoint* owner to
    /// the min **element** owner (MFEM `GroupTopology` share-set minimum), which
    /// changes the np ≥ 2 id layout; `x` is therefore valued by entity here
    /// rather than by `global_dof` (see the note at the fill site) — the
    /// operator itself is unchanged.
    #[test]
    fn par_space_p2_matrix_consistent_across_partitions() {
        use fem_assembly::standard::DiffusionIntegrator;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        let mesh = Mesh::<2>::unit_square_tri(4);

        // (entity kind, a, b, y): kind 0 = vertex (a = global node), kind 1 =
        // edge (a, b = sorted global node pair).
        type Entity = (u8, u32, u32);

        fn run_partition<const N: usize>(mesh: Mesh<2>) -> Vec<(Entity, f64)> {
            let out: Arc<Mutex<Option<Vec<(Entity, f64)>>>> = Arc::new(Mutex::new(None));
            let out2 = Arc::clone(&out);
            let launcher = ThreadLauncher::new(WorkerConfig::new(N));
            launcher.launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let local_mesh = pmesh.local_mesh().clone();
                let dm = DofManager::new(&local_mesh, 2);
                let local = H1Space::new(local_mesh, 2);
                let ps = ParallelFESpace::new_with_dof_manager(local, &pmesh, &dm, comm.clone());

                // DofManager local DOF -> canonical entity, read from the same
                // maps `from_dof_manager` uses (vertex DOFs are node ids for the
                // Tri3 P2 layout; edge DOFs live in `edge_pk_map`).
                let part = pmesh.partition();
                let mut entity_of: HashMap<u32, Entity> = HashMap::new();
                for lid in 0..dm.n_vertex_dofs as u32 {
                    entity_of.insert(lid, (0, part.global_node(lid), 0));
                }
                for (&fem_space::dof_manager::EdgeKey(a, b), dofs) in &dm.edge_pk_map {
                    let (ga, gb) = (part.global_node(a), part.global_node(b));
                    for &d in dofs {
                        entity_of.insert(d, (1, ga.min(gb), ga.max(gb)));
                    }
                }

                let diff = DiffusionIntegrator { kappa: 1.0 };
                let a_mat = ParAssembler::assemble_bilinear(&ps, &[&diff], 4);
                let dp = ps.dof_partition();
                // D122-1 (round 73): `x` is keyed by the **entity**, not by the
                // global DOF id.  The ids (`global_dof_ids`) are assigned
                // owner-block by owner-block, so they differ across rank counts
                // for the same entity; the pre-D122-1 numbering only made
                // `A·gid` agree because rank 0's owned set was then a prefix of
                // the canonical id order (verified by replaying that numbering:
                // it reproduces the np = 1 result bit for bit).  The *operator*
                // by entity is the partition-independent object, and it is
                // pinned per entry in
                // `crates/parallel/tests/d122r73_q2_diag.rs`.
                let mut x = ParVector::zeros(&ps);
                for pid in 0..dp.n_owned_dofs {
                    let dm_id = dp.unpermute_dof(pid as u32);
                    let (k, a, b) = entity_of[&dm_id];
                    x.as_slice_mut()[pid] =
                        k as f64 + a as f64 / 3.0 + b as f64 / 9.0;
                }
                let mut y = ParVector::zeros(&ps);
                a_mat.spmv(&mut x, &mut y);

                let owned: Vec<(Entity, f64)> = (0..dp.n_owned_dofs)
                    .map(|pid| {
                        let dm_id = dp.unpermute_dof(pid as u32);
                        let e = *entity_of
                            .get(&dm_id)
                            .expect("P2 consistency test: owned DOF not classified");
                        (e, y.as_slice()[pid])
                    })
                    .collect();

                if comm.rank() == 0 {
                    let mut all = owned.clone();
                    for src in 1..comm.size() as i32 {
                        let flat: Vec<u32> = comm.recv(src, 121);
                        let vals: Vec<f64> = comm.recv(src, 122);
                        all.extend(
                            flat.chunks_exact(3)
                                .map(|c| (c[0] as u8, c[1], c[2]))
                                .zip(vals),
                        );
                    }
                    all.sort_unstable_by_key(|&(e, _)| e);
                    *out2.lock().unwrap() = Some(all);
                } else {
                    let flat: Vec<u32> = owned
                        .iter()
                        .flat_map(|&((k, a, b), _)| [k as u32, a, b])
                        .collect();
                    let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                    comm.send(0, 121, &flat);
                    comm.send(0, 122, &vals);
                }
            });
            let mut guard = out.lock().unwrap();
            guard.take().unwrap()
        }

        let np1 = run_partition::<1>(mesh.clone());
        let np2 = run_partition::<2>(mesh.clone());
        assert_eq!(np1.len(), np2.len(), "P2 global DOF count mismatch");
        let mut max_diff = 0.0_f64;
        for ((e1, y1), (e2, y2)) in np1.iter().zip(np2.iter()) {
            assert_eq!(e1, e2, "P2 entity key order mismatch");
            max_diff = max_diff.max((y1 - y2).abs());
        }
        assert!(
            max_diff < 1e-9,
            "P2 matrix differs across partitions: max_diff={max_diff:e}"
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
