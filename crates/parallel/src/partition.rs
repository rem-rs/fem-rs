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
use fem_mesh::ElementType;

// ── EntityOwnership (D807-1) ──────────────────────────────────────────────────

/// Traversal-independent entity ownership, published by the mesh extraction.
///
/// [`crate::par_partition::partition_mesh`] holds the **full** serial mesh and
/// the full element-partition vector, so it can answer, for every node / edge /
/// facet of the local sub-mesh, the two questions `DofPartition` used to answer
/// from its *local element traversal*:
///
/// 1. **who owns the entity** — the MFEM `GroupTopology` rule: the minimum rank
///    over *all* elements (mesh-wide) that hold it; and
/// 2. **which element anchors its canonical face basis** — the minimum global
///    element id among those holders (D412 / D122-3), together with that
///    element's **own vertex list** (D813-1).
///
/// Both are pure functions of the full mesh + partition vector, so they do not
/// depend on which elements this rank happens to carry.  Publishing them lets
/// the ghost layer be cut to the smallest set the rules actually need — see the
/// layer discussion in [`crate::par_partition`].
///
/// D813-1: the anchor element's `(ElementType, global vertex list)` travels
/// with the ownership tables so the DOF partition can rebuild the facet's
/// **canonical face-DOF frame** on a rank that does not carry the anchor element
/// (`crates/space`: `HCurlSpace::facet_slots_against_published_anchor`).  Only
/// the facet's own vertices are ever read from the frame — every family's frame
/// is face-local — so the anchor element itself need not be in the ghost layer.
///
/// `None` on partitions that were not produced by the extraction (the serial
/// wrapper, AMR/repartition rebuilds); `DofPartition` then falls back to the
/// traversal-derived rules, which is byte-identical to the pre-D807 behaviour.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EntityOwnership {
    /// `(global node id, owning rank)`, sorted by the global id.
    node_owner: Vec<(NodeId, Rank)>,
    /// Edge (sorted global node pair) → minimum rank over its holders.
    edge_owner: HashMap<(NodeId, NodeId), Rank>,
    /// Facet (sorted global vertex list) → `(minimum rank over its holders,
    /// minimum global element id among them)`.
    facet: HashMap<Vec<NodeId>, (Rank, ElemId)>,
    /// Anchor global element id → `(element type, global vertex list in the
    /// element's own slot order)` of that anchor (D813-1).  One entry per
    /// element that is the canonical anchor of at least one published facet.
    facet_anchor_elem: HashMap<ElemId, (ElementType, Vec<NodeId>)>,
}

impl EntityOwnership {
    /// Build from the four tables (the extraction is the only producer).
    pub(crate) fn new(
        mut node_owner: Vec<(NodeId, Rank)>,
        edge_owner: HashMap<(NodeId, NodeId), Rank>,
        facet: HashMap<Vec<NodeId>, (Rank, ElemId)>,
        facet_anchor_elem: HashMap<ElemId, (ElementType, Vec<NodeId>)>,
    ) -> Self {
        node_owner.sort_unstable();
        node_owner.dedup_by_key(|e| e.0);
        EntityOwnership { node_owner, edge_owner, facet, facet_anchor_elem }
    }

    /// Owner (minimum rank over the holders) of global node `gid`.
    pub fn node_owner(&self, gid: NodeId) -> Option<Rank> {
        self.node_owner
            .binary_search_by_key(&gid, |&(g, _)| g)
            .ok()
            .map(|i| self.node_owner[i].1)
    }

    /// Owner of the edge `(a, b)`, keyed by the sorted global node pair.
    pub fn edge_owner(&self, a: NodeId, b: NodeId) -> Option<Rank> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.edge_owner.get(&key).copied()
    }

    /// Owner of the facet whose **sorted** global vertex list is `verts`.
    pub fn facet_owner(&self, verts: &[NodeId]) -> Option<Rank> {
        self.facet.get(verts).map(|&(owner, _)| owner)
    }

    /// Canonical anchor element (minimum global element id among the facet's
    /// holders) of the facet whose sorted global vertex list is `verts`.
    pub fn facet_anchor(&self, verts: &[NodeId]) -> Option<ElemId> {
        self.facet.get(verts).map(|&(_, anchor)| anchor)
    }

    /// `(owner, canonical anchor element)` of the facet whose **sorted** global
    /// vertex list is `verts`.
    pub fn facet(&self, verts: &[NodeId]) -> Option<(Rank, ElemId)> {
        self.facet.get(verts).copied()
    }

    /// Number of published node owners.
    pub fn n_node_owners(&self) -> usize { self.node_owner.len() }
    /// Number of published edge owners.
    pub fn n_edge_owners(&self) -> usize { self.edge_owner.len() }
    /// Number of published facets.
    pub fn n_facets(&self) -> usize { self.facet.len() }
    /// Number of published anchor elements.
    pub fn n_facet_anchor_elems(&self) -> usize { self.facet_anchor_elem.len() }

    /// `(ElementType, global vertex list)` of the canonical anchor element
    /// `gid`, in the element's own vertex slot order — the data
    /// `HCurlSpace::facet_slots_against_published_anchor` needs to rebuild a
    /// facet's canonical frame without the element itself (D813-1).
    pub fn facet_anchor_element(&self, gid: ElemId) -> Option<(ElementType, &[NodeId])> {
        self.facet_anchor_elem.get(&gid).map(|(et, v)| (*et, v.as_slice()))
    }

    /// Anchor-element table: global element id → `(type, vertex list)`.
    pub fn facet_anchor_elem_table(&self) -> &HashMap<ElemId, (ElementType, Vec<NodeId>)> {
        &self.facet_anchor_elem
    }

    /// Node-owner table: `(global node id, owner)`, sorted by the global id.
    pub fn node_owner_table(&self) -> &[(NodeId, Rank)] { &self.node_owner }
    /// Edge-owner table: sorted global node pair → owner.
    pub fn edge_owner_table(&self) -> &HashMap<(NodeId, NodeId), Rank> {
        &self.edge_owner
    }
    /// Facet table: sorted global vertex list → `(owner, anchor element)`.
    pub fn facet_table(&self) -> &HashMap<Vec<NodeId>, (Rank, ElemId)> {
        &self.facet
    }
}

// ── MeshPartition ─────────────────────────────────────────────────────────────

/// Describes the local share of a distributed mesh on one MPI rank.
///
/// Index convention
/// ----------------
/// * *local node ID* — index into the local node array `[0, n_owned + n_ghost)`.
/// * *local element ID* — index into the local element array `[0, n_owned_elems + n_ghost_elems)`.
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

    /// `true` when local node ids ARE the global ids (MFEM ParMesh
    /// semantics, opt-in via [`crate::par_partition::partition_mesh_identity`]).
    /// In this mode `global_node(local_id)` returns `local_id` itself and
    /// `node_owner`/`is_owned_node` resolve through the global→compact map.
    /// Needed so FE-space construction (HDiv/HCurl edge orientation, DOF
    /// order) is identical across ranks for RTk/NDk spaces.
    pub node_id_identity: bool,

    // ── element ownership ───────────────────────────────────────────────────

    /// Number of locally owned elements.
    pub n_owned_elems: usize,

    /// Number of ghost elements (owned by neighbor ranks, kept for stencil).
    pub n_ghost_elems: usize,

    /// Global element IDs for every local element (owned + ghost), length `n_owned_elems + n_ghost_elems`.
    pub global_elem_ids: Vec<ElemId>,

    /// MPI rank that owns each local element, length `n_owned_elems + n_ghost_elems`.
    pub elem_owner: Vec<Rank>,

    /// Traversal-independent entity ownership + canonical facet anchors,
    /// published by the extraction (D807-1).  `None` on partitions built
    /// without the full mesh (serial wrapper, AMR/refine rebuilds, legacy wire
    /// format); `DofPartition` then keeps its traversal-derived rules.
    pub entities: Option<EntityOwnership>,

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
            n_owned_elems: n_elems,
            n_ghost_elems: 0,
            global_elem_ids,
            elem_owner: vec![0; n_elems],
            node_global_to_local,
            elem_global_to_local,
            node_id_identity: false,
            entities: None,
        }
    }

    /// Construct from raw arrays (used by the partitioner in Phase 10).
    ///
    /// # Parameters
    /// * `owned_global_nodes` — global IDs of nodes owned by this rank,
    ///   in any order.
    /// * `ghost_global_nodes` — global IDs of ghost nodes together with
    ///   their owner ranks.
    /// * `owned_global_elems` — global IDs of elements owned by this rank.
    /// * `ghost_global_elems` — global IDs of ghost elements with their owners.
    pub fn from_partitioner(
        owned_global_nodes: &[NodeId],
        ghost_global_nodes: &[(NodeId, Rank)],
        owned_global_elems: &[ElemId],
        ghost_global_elems: &[(ElemId, Rank)],
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

        let n_owned_elems = owned_global_elems.len();
        let n_ghost_elems = ghost_global_elems.len();

        let mut global_elem_ids = Vec::with_capacity(n_owned_elems + n_ghost_elems);
        let mut elem_owner = Vec::with_capacity(n_owned_elems + n_ghost_elems);
        for &gid in owned_global_elems {
            global_elem_ids.push(gid);
            elem_owner.push(local_rank);
        }
        for &(gid, owner) in ghost_global_elems {
            global_elem_ids.push(gid);
            elem_owner.push(owner);
        }

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
            n_owned_elems,
            n_ghost_elems,
            global_elem_ids,
            elem_owner,
            node_global_to_local,
            elem_global_to_local,
            node_id_identity: false,
            entities: None,
        }
    }

    /// Total local node count (owned + ghost).
    #[inline]
    pub fn n_total_nodes(&self) -> usize {
        self.n_owned_nodes + self.n_ghost_nodes
    }

    /// `true` if `local_id` refers to an owned (non-ghost) node.
    #[inline]
    pub fn is_owned_node(&self, local_id: u32) -> bool {
        if self.node_id_identity {
            self.node_global_to_local
                .get(&local_id)
                .map(|&c| (c as usize) < self.n_owned_nodes)
                .unwrap_or(false)
        } else {
            (local_id as usize) < self.n_owned_nodes
        }
    }

    /// Global ID of a local node.
    ///
    /// In identity mode (`node_id_identity`) the local id IS the global id.
    ///
    /// # Panics
    /// Panics if `local_id >= n_total_nodes()` (non-identity mode).
    #[inline]
    pub fn global_node(&self, local_id: u32) -> NodeId {
        if self.node_id_identity {
            local_id
        } else {
            if (local_id as usize) >= self.global_node_ids.len() {
                if std::env::var("PEX15_DBG").is_ok() {
                    eprintln!(
                        "[dbg-gn] global_node({local_id}) OOB len={} n_owned={} n_total={}",
                        self.global_node_ids.len(),
                        self.n_owned_nodes,
                        self.n_owned_nodes + self.n_ghost_nodes
                    );
                }
            }
            self.global_node_ids[local_id as usize]
        }
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
        if self.node_id_identity {
            self.node_global_to_local
                .get(&local_id)
                .map(|&c| self.node_owner[c as usize])
                .unwrap_or(0)
        } else {
            self.node_owner[local_id as usize]
        }
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
        n_owned_elems: usize,
        n_ghost_elems: usize,
        global_node_ids: Vec<NodeId>,
        node_owner: Vec<Rank>,
        global_elem_ids: Vec<ElemId>,
        elem_owner: Vec<Rank>,
    ) -> Self {
        let total = n_owned_nodes + n_ghost_nodes;
        assert_eq!(global_node_ids.len(), total,
            "global_node_ids.len()={} != n_owned+n_ghost={}",
            global_node_ids.len(), total);
        assert_eq!(node_owner.len(), total,
            "node_owner.len()={} != n_owned+n_ghost={}",
            node_owner.len(), total);
        assert_eq!(global_elem_ids.len(), n_owned_elems + n_ghost_elems,
            "global_elem_ids.len()={} != n_owned_elems + n_ghost_elems = {}",
            global_elem_ids.len(), n_owned_elems + n_ghost_elems);
        assert_eq!(elem_owner.len(), global_elem_ids.len(),
            "elem_owner.len()={} != global_elem_ids.len()={}",
            elem_owner.len(), global_elem_ids.len());

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
            n_owned_elems,
            n_ghost_elems,
            global_elem_ids,
            elem_owner,
            node_global_to_local,
            elem_global_to_local,
            node_id_identity: false,
            entities: None,
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
