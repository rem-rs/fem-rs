//! Mesh partition descriptor.
//!
//! A [`MeshPartition`] describes how mesh entities (nodes and elements) are
//! distributed across MPI ranks.  It is the foundation on which
//! `ParallelMesh` will be built in Phase 10.
//!
//! ## Ownership model
//!
//! Each mesh node belongs to exactly one rank (its *owner*).  Nodes that are
//! geometrically shared across a partition boundary are kept as *ghost* copies
//! on the ranks that need them.  Ghost nodes are read-only; updates flow from
//! owner to ghosts via [`GhostExchange`](super::ghost::GhostExchange).
//!
//! Local node layout (contiguous in memory):
//! ```text
//! [ owned nodes 0 .. n_owned )  [ ghost nodes n_owned .. n_owned+n_ghost )
//! ```
//!
//! Elements are owned by the rank that holds all their nodes, or by explicit
//! assignment from the partitioner.  Ghost elements are not stored separately;
//! each element is owned by exactly one rank.

use std::collections::HashMap;
use fem_core::{ElemId, NodeId, Rank};

// ── MeshPartition ─────────────────────────────────────────────────────────────

/// Describes the local share of a distributed mesh on one MPI rank.
///
/// Index convention
/// ----------------
/// * *local node ID* — index into the local node array `[0, n_owned + n_ghost)`.
/// * *local element ID* — index into the local element array `[0, n_local_elems)`.
/// * *global node/element ID* — mesh-wide unique index assigned by the
///   partitioner (same numbering as the serial mesh).
#[derive(Debug, Clone)]
pub struct MeshPartition {
    // ── node ownership ──────────────────────────────────────────────────────

    /// Number of locally *owned* nodes (rank is authoritative for these).
    pub n_owned_nodes: usize,

    /// Number of *ghost* nodes (owned by a neighboring rank, kept here for
    /// stencil completeness).
    pub n_ghost_nodes: usize,

    /// Global node IDs for every local node, length `n_owned_nodes + n_ghost_nodes`.
    ///
    /// `global_node_ids[local_id] = global_id`
    pub global_node_ids: Vec<NodeId>,

    /// MPI rank that owns each local+ghost node, length `n_owned_nodes + n_ghost_nodes`.
    ///
    /// For owned nodes this equals the local rank; for ghost nodes it is the
    /// remote rank that holds the authoritative copy.
    pub node_owner: Vec<Rank>,

    // ── element ownership ───────────────────────────────────────────────────

    /// Number of locally owned elements.
    pub n_local_elems: usize,

    /// Global element IDs for every local element, length `n_local_elems`.
    pub global_elem_ids: Vec<ElemId>,

    // ── reverse lookup ──────────────────────────────────────────────────────

    /// Global → local node ID mapping (covers both owned and ghost nodes).
    ///
    /// Built lazily; call [`MeshPartition::build_lookup`] after construction.
    node_global_to_local: HashMap<NodeId, u32>,

    /// Global → local element ID mapping.
    elem_global_to_local: HashMap<ElemId, u32>,
}

impl MeshPartition {
    // ── constructors ─────────────────────────────────────────────────────────

    /// Create a trivial single-rank partition for a serial mesh.
    ///
    /// All `n_nodes` nodes and `n_elems` elements are owned by rank 0.
    /// Useful for testing and for the MPI-disabled build path.
    pub fn new_serial(n_nodes: usize, n_elems: usize) -> Self {
        let global_node_ids: Vec<NodeId> = (0..n_nodes as u32).collect();
        let node_owner: Vec<Rank> = vec![0; n_nodes];
        let global_elem_ids: Vec<ElemId> = (0..n_elems as u32).collect();

        let node_global_to_local: HashMap<_, _> = global_node_ids
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local as u32))
            .collect();
        let elem_global_to_local: HashMap<_, _> = global_elem_ids
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local as u32))
            .collect();

        MeshPartition {
            n_owned_nodes: n_nodes,
            n_ghost_nodes: 0,
            global_node_ids,
            node_owner,
            n_local_elems: n_elems,
            global_elem_ids,
            node_global_to_local,
            elem_global_to_local,
        }
    }

    /// Construct from raw arrays (used by the partitioner in Phase 10).
    ///
    /// # Parameters
    /// * `owned_global_nodes` — global IDs of nodes owned by this rank,
    ///   in any order.
    /// * `ghost_global_nodes` — global IDs of ghost nodes together with
    ///   their owner ranks.
    /// * `local_global_elems` — global IDs of elements assigned to this rank.
    pub fn from_partitioner(
        owned_global_nodes: &[NodeId],
        ghost_global_nodes: &[(NodeId, Rank)],
        local_global_elems: &[ElemId],
        local_rank: Rank,
    ) -> Self {
        let n_owned = owned_global_nodes.len();
        let n_ghost = ghost_global_nodes.len();
        let total_nodes = n_owned + n_ghost;

        let mut global_node_ids = Vec::with_capacity(total_nodes);
        let mut node_owner = Vec::with_capacity(total_nodes);

        for &gid in owned_global_nodes {
            global_node_ids.push(gid);
            node_owner.push(local_rank);
        }
        for &(gid, owner) in ghost_global_nodes {
            global_node_ids.push(gid);
            node_owner.push(owner);
        }

        let global_elem_ids = local_global_elems.to_vec();

        let node_global_to_local = global_node_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();
        let elem_global_to_local = global_elem_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();

        MeshPartition {
            n_owned_nodes: n_owned,
            n_ghost_nodes: n_ghost,
            global_node_ids,
            node_owner,
            n_local_elems: local_global_elems.len(),
            global_elem_ids,
            node_global_to_local,
            elem_global_to_local,
        }
    }

    // ── queries ──────────────────────────────────────────────────────────────

    /// Total local node count (owned + ghost).
    #[inline]
    pub fn n_total_nodes(&self) -> usize {
        self.n_owned_nodes + self.n_ghost_nodes
    }

    /// `true` if `local_id` refers to an owned (non-ghost) node.
    #[inline]
    pub fn is_owned_node(&self, local_id: u32) -> bool {
        (local_id as usize) < self.n_owned_nodes
    }

    /// Global ID of a local node.
    ///
    /// # Panics
    /// Panics if `local_id >= n_total_nodes()`.
    #[inline]
    pub fn global_node(&self, local_id: u32) -> NodeId {
        self.global_node_ids[local_id as usize]
    }

    /// Local ID of a global node, or `None` if not present on this rank.
    #[inline]
    pub fn local_node(&self, global_id: NodeId) -> Option<u32> {
        self.node_global_to_local.get(&global_id).copied()
    }

    /// Global ID of a local element.
    #[inline]
    pub fn global_elem(&self, local_id: u32) -> ElemId {
        self.global_elem_ids[local_id as usize]
    }

    /// Local ID of a global element, or `None` if not present on this rank.
    #[inline]
    pub fn local_elem(&self, global_id: ElemId) -> Option<u32> {
        self.elem_global_to_local.get(&global_id).copied()
    }

    /// Owner rank of local node `local_id`.
    #[inline]
    pub fn node_owner(&self, local_id: u32) -> Rank {
        self.node_owner[local_id as usize]
    }

    /// Construct from raw flat arrays (used by streaming mesh deserialisation).
    ///
    /// Builds the internal global→local lookup tables automatically.
    ///
    /// # Panics
    /// Panics if `global_node_ids.len() != n_owned_nodes + n_ghost_nodes` or
    /// `node_owner.len() != n_owned_nodes + n_ghost_nodes`.
    pub fn from_raw(
        n_owned_nodes: usize,
        n_ghost_nodes: usize,
        global_node_ids: Vec<NodeId>,
        node_owner: Vec<Rank>,
        global_elem_ids: Vec<ElemId>,
    ) -> Self {
        let total = n_owned_nodes + n_ghost_nodes;
        assert_eq!(global_node_ids.len(), total,
            "global_node_ids.len()={} != n_owned+n_ghost={}",
            global_node_ids.len(), total);
        assert_eq!(node_owner.len(), total,
            "node_owner.len()={} != n_owned+n_ghost={}",
            node_owner.len(), total);

        let n_local_elems = global_elem_ids.len();
        let node_global_to_local = global_node_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();
        let elem_global_to_local = global_elem_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();

        MeshPartition {
            n_owned_nodes,
            n_ghost_nodes,
            global_node_ids,
            node_owner,
            n_local_elems,
            global_elem_ids,
            node_global_to_local,
            elem_global_to_local,
        }
    }

    /// Rebuild the global→local lookup tables.
    ///
    /// Call this if `global_node_ids` or `global_elem_ids` were mutated after
    /// construction.
    pub fn build_lookup(&mut self) {
        self.node_global_to_local = self
            .global_node_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();
        self.elem_global_to_local = self
            .global_elem_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();
    }

    /// Iterate over local IDs of ghost nodes together with their owner ranks.
    ///
    /// Yields `(local_id, owner_rank)` for every ghost node.
    pub fn ghost_nodes(&self) -> impl Iterator<Item = (u32, Rank)> + '_ {
        let start = self.n_owned_nodes;
        (start..self.n_total_nodes()).map(move |lid| {
            (lid as u32, self.node_owner[lid])
        })
    }
}
