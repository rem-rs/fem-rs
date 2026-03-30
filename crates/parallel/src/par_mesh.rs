//! Distributed mesh: [`ParallelMesh<M>`].
//!
//! A `ParallelMesh` wraps a *local* mesh `M` (the portion owned by this MPI
//! rank) together with the communication metadata needed to perform ghost-node
//! exchanges and global reductions.
//!
//! ## Index conventions
//!
//! | Term | Meaning |
//! |------|---------|
//! | *local node ID* | Index into this rank's node array `[0, n_total_nodes)` |
//! | *owned node* | `local_id < n_owned_nodes`; this rank holds the authoritative value |
//! | *ghost node* | `local_id >= n_owned_nodes`; read-only copy owned by a neighbor |
//! | *global node ID* | Mesh-wide unique index (same as serial mesh node index) |
//! | *local elem ID* | Index into this rank's element array `[0, n_local_elems)` |
//! | *global elem ID* | Mesh-wide unique element index |
//!
//! ## `MeshTopology` delegation
//!
//! `ParallelMesh<M>` implements `MeshTopology` by forwarding all queries to
//! the local mesh `M`.  The counts (`n_nodes`, `n_elements`, …) therefore
//! return **local** values; use `global_n_nodes()` / `global_n_elems()` for
//! mesh-wide totals.

use fem_core::{ElemId, FaceId, NodeId};
use fem_mesh::{ElementType, MeshTopology};

use crate::{Comm, GhostExchange, MeshPartition};

// ── ParallelMesh ─────────────────────────────────────────────────────────────

/// A distributed mesh: local sub-mesh + MPI partition metadata.
///
/// Construct via [`partition_simplex`](super::par_simplex::partition_simplex)
/// or any custom partitioner that builds a [`MeshPartition`] describing the
/// local node/element ownership.
pub struct ParallelMesh<M: MeshTopology> {
    /// Sub-mesh local to this rank (local node / element indices).
    local_mesh: M,
    /// MPI communicator (cloneable `Arc`-backed handle).
    comm: Comm,
    /// Node/element ownership description.
    partition: MeshPartition,
    /// Pre-computed ghost-exchange communication pattern.
    ghost_exchange: GhostExchange,
    /// Sum of owned nodes across all ranks (gathered once at construction).
    global_n_nodes: usize,
    /// Sum of local elements across all ranks (gathered once at construction).
    global_n_elems: usize,
}

impl<M: MeshTopology> ParallelMesh<M> {
    // ── constructor ──────────────────────────────────────────────────────────

    /// Build a `ParallelMesh` from a pre-partitioned local mesh.
    ///
    /// Performs a single `allreduce_sum` to compute global mesh statistics and
    /// builds the [`GhostExchange`] pattern from the partition.
    pub fn new(local_mesh: M, comm: Comm, partition: MeshPartition) -> Self {
        let global_n_nodes =
            comm.allreduce_sum_i64(partition.n_owned_nodes as i64) as usize;
        let global_n_elems =
            comm.allreduce_sum_i64(partition.n_local_elems as i64) as usize;
        let ghost_exchange = GhostExchange::from_partition(&partition, &comm);

        ParallelMesh {
            local_mesh,
            comm,
            partition,
            ghost_exchange,
            global_n_nodes,
            global_n_elems,
        }
    }

    // ── accessors ────────────────────────────────────────────────────────────

    /// Reference to the local sub-mesh (local indices throughout).
    #[inline]
    pub fn local_mesh(&self) -> &M { &self.local_mesh }

    /// MPI communicator.
    #[inline]
    pub fn comm(&self) -> &Comm { &self.comm }

    /// Node/element ownership descriptor.
    #[inline]
    pub fn partition(&self) -> &MeshPartition { &self.partition }

    /// Pre-built ghost-exchange communication pattern.
    #[inline]
    pub fn ghost_exchange(&self) -> &GhostExchange { &self.ghost_exchange }

    // ── global topology ──────────────────────────────────────────────────────

    /// Mesh-wide total number of nodes (sum of owned nodes across all ranks).
    #[inline]
    pub fn global_n_nodes(&self) -> usize { self.global_n_nodes }

    /// Mesh-wide total number of elements (sum across all ranks).
    #[inline]
    pub fn global_n_elems(&self) -> usize { self.global_n_elems }

    // ── local partition sizes ────────────────────────────────────────────────

    /// Number of nodes owned by this rank (authoritative copies).
    #[inline]
    pub fn n_owned_nodes(&self) -> usize { self.partition.n_owned_nodes }

    /// Number of ghost nodes on this rank (read-only mirror copies).
    #[inline]
    pub fn n_ghost_nodes(&self) -> usize { self.partition.n_ghost_nodes }

    /// Total local node count = owned + ghost.
    #[inline]
    pub fn n_total_nodes(&self) -> usize { self.partition.n_total_nodes() }

    // ── index translation ────────────────────────────────────────────────────

    /// Global node ID for a local node index.
    #[inline]
    pub fn global_node_id(&self, local: u32) -> NodeId {
        self.partition.global_node(local)
    }

    /// Global element ID for a local element index.
    #[inline]
    pub fn global_elem_id(&self, local: u32) -> NodeId {
        self.partition.global_elem(local)
    }

    /// Local node ID for a global node, or `None` if not on this rank.
    #[inline]
    pub fn local_node_id(&self, global: NodeId) -> Option<u32> {
        self.partition.local_node(global)
    }

    // ── ghost exchange helpers ───────────────────────────────────────────────

    /// Propagate owned-node `f64` values into ghost slots.
    ///
    /// `data` must have length `>= n_total_nodes()`.
    pub fn forward_exchange(&self, data: &mut [f64]) {
        self.ghost_exchange.forward(&self.comm, data);
    }

    /// Accumulate ghost `f64` contributions back to their owner slots.
    ///
    /// Ghost slots are zeroed after the accumulation.
    pub fn reverse_exchange(&self, data: &mut [f64]) {
        self.ghost_exchange.reverse(&self.comm, data);
    }

    // ── global reductions ────────────────────────────────────────────────────

    /// AllReduce sum of `local` across all ranks.
    #[inline]
    pub fn allreduce_sum(&self, local: f64) -> f64 {
        self.comm.allreduce_sum_f64(local)
    }

    /// Sum a slice over owned nodes across all ranks.
    ///
    /// Only owned-node values contribute (ghost slots are skipped) to avoid
    /// double-counting shared data.
    pub fn global_sum_owned(&self, data: &[f64]) -> f64 {
        let local: f64 = (0..self.partition.n_owned_nodes)
            .map(|lid| data[lid])
            .sum();
        self.comm.allreduce_sum_f64(local)
    }
}

// ── MeshTopology delegation ───────────────────────────────────────────────────

impl<M: MeshTopology> MeshTopology for ParallelMesh<M> {
    /// Spatial dimension (same as underlying mesh).
    fn dim(&self) -> u8 { self.local_mesh.dim() }

    /// Local node count (owned + ghost).
    fn n_nodes(&self) -> usize { self.local_mesh.n_nodes() }

    /// Local element count (owned elements only).
    fn n_elements(&self) -> usize { self.local_mesh.n_elements() }

    /// Local boundary face count.
    fn n_boundary_faces(&self) -> usize { self.local_mesh.n_boundary_faces() }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        self.local_mesh.element_nodes(elem)
    }

    fn element_type(&self, elem: ElemId) -> ElementType {
        self.local_mesh.element_type(elem)
    }

    fn element_tag(&self, elem: ElemId) -> i32 {
        self.local_mesh.element_tag(elem)
    }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        self.local_mesh.node_coords(node)
    }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] {
        self.local_mesh.face_nodes(face)
    }

    fn face_tag(&self, face: FaceId) -> i32 {
        self.local_mesh.face_tag(face)
    }

    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) {
        self.local_mesh.face_elements(face)
    }
}
