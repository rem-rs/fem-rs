//! DOF-level ownership for a parallel FE space.
//!
//! [`DofPartition`] extends mesh-level partitioning to DOF indices.  For
//! H1/P1, DOFs map 1:1 to nodes.  For P2, edge DOFs are added with
//! ownership: `owner(edge(a,b)) = min(owner(a), owner(b))`.

use std::collections::HashMap;
use fem_core::Rank;
use fem_mesh::topology::MeshTopology;
use fem_space::dof_manager::{DofManager, EdgeKey};
use fem_space::fe_space::FESpace;
use crate::comm::Comm;
use crate::partition::MeshPartition;

// ── EdgeInfo (internal) ─────────────────────────────────────────────────────

/// Metadata for one edge DOF used during P2 partitioning.
struct EdgeDofInfo {
    local_dof_id: u32,
    global_node_a: u32,   // min of global endpoints
    global_node_b: u32,   // max of global endpoints
    owner: Rank,
    /// Cross-rank-consistent DOF identifier used as part of the ghost
    /// exchange key.  For FE spaces whose edge DOFs carry global numbering
    /// (RTk/NDk: `local_dof_id` is a global DOF id) this is `local_dof_id`;
    /// for DofManager P2 (1 DOF per edge, local numbering differs per rank)
    /// this is `0` (the node pair alone identifies the edge uniquely).
    dof_key: u32,
}

/// Metadata for one face DOF of a 3-D H(div) space (RT0/RT1).
struct FaceDofInfo {
    local_dof_id: u32,
    /// Face key: the 3 smallest global vertex ids of the face.
    face_key: (u32, u32, u32),
    /// Position of the DOF within its face (0 for RT0, 0..dofs_per_face for
    /// RTk): cross-rank consistent because it is derived from the DOF's
    /// position inside the element's face block, which all ranks compute the
    /// same way.
    pos: u32,
    owner: Rank,
}

// ── DofPartition ────────────────────────────────────────────────────────────

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
    /// Permutation from DofManager's local DOF numbering to partition's
    /// [owned|ghost] layout.  `perm[dm_local_id] = partition_local_id`.
    /// Empty for P1 (identity permutation).
    pub(crate) dm_to_partition: Vec<u32>,
    /// Inverse permutation: `partition_to_dm[partition_local_id] = dm_local_id`.
    /// Empty for P1 (identity permutation).
    pub(crate) partition_to_dm: Vec<u32>,
    /// Per-DOF sign correction (±1.0) for H(curl)/H(div) edge spaces.
    ///
    /// For vector FE spaces the basis function sign depends on the local
    /// vertex ordering which may disagree with the canonical global ordering
    /// after mesh partitioning.  `sign_corrections[dm_local_id]` is `+1.0`
    /// when the local sign agrees with the global convention and `−1.0`
    /// otherwise.  Empty when no correction is needed (P1, P2, serial).
    ///
    /// Callers that permute matrix/vector data must apply
    /// `val *= sign_correction(row) * sign_correction(col)` (matrix) or
    /// `val *= sign_correction(i)` (vector) during permutation.
    pub(crate) sign_corrections: Vec<f64>,
}

impl DofPartition {
    /// Build a DOF partition for P1 (DOFs = nodes) from a mesh partition.
    pub fn from_mesh_partition(partition: &MeshPartition, comm: &Comm) -> Self {
        let n_owned = partition.n_owned_nodes;
        let n_ghost = partition.n_ghost_nodes;

        let global_dof_ids = partition.global_node_ids.clone();
        let dof_owner = partition.node_owner.clone();
        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;

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
            dm_to_partition: Vec::new(), // identity for P1
            partition_to_dm: Vec::new(),
            sign_corrections: Vec::new(),
        }
    }

    /// Build a DOF partition for a discontinuous L2 space: every element owns
    /// its `dofs_per_elem` DOFs.  Global DOF IDs are
    /// `global_elem_id * dofs_per_elem + j`, which is identical on every rank
    /// holding that element (no exchange needed).  The local DOF layout
    /// (element-traversal order) already matches the [owned | ghost] element
    /// ordering, so the permutation is the identity.
    pub fn from_l2_space<M: MeshTopology>(
        space: &fem_space::L2Space<M>,
        partition: &MeshPartition,
        comm: &Comm,
    ) -> Self {
        let _local_rank = comm.rank();
        let dofs_per_elem = space.element_dofs(0).len();
        let n_local_elems = partition.n_owned_elems + partition.n_ghost_elems;
        let n_owned = partition.n_owned_elems * dofs_per_elem;
        let n_ghost = partition.n_ghost_elems * dofs_per_elem;

        let mut global_dof_ids = Vec::with_capacity(n_owned + n_ghost);
        let mut dof_owner = Vec::with_capacity(n_owned + n_ghost);
        for e in 0..n_local_elems {
            let ge = partition.global_elem(e as u32);
            let owner = partition.elem_owner[e];
            for j in 0..dofs_per_elem {
                global_dof_ids.push(ge * dofs_per_elem as u32 + j as u32);
                dof_owner.push(owner);
            }
        }

        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;
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
            dm_to_partition: Vec::new(), // identity: element order == partition order
            partition_to_dm: Vec::new(),
            sign_corrections: Vec::new(),
        }
    }

    /// Build a DOF partition for a vector FE space (vdim components) using the
    /// byNODES block layout: `dof(vd, node) = node + vd * n_global_nodes`,
    /// which matches `VectorH1Space` (see crates/space/src/vector_h1.rs).
    ///
    /// Local partition layout follows the `[owned | ghost]` convention used by
    /// `ParVector` / `ParCsrMatrix`:
    /// - owned segment:  `vd * n_owned_nodes + node` (node in owned range)
    /// - ghost segment:  `n_owned + vd * n_ghost_nodes + i` (ghost index i)
    ///
    /// The `DofManager` (space) ordering is `vd * n_local_nodes + node`
    /// (component blocks of all local nodes), so a permutation
    /// `dm_to_partition` / `partition_to_dm` is installed.
    pub fn from_vector_space(partition: &MeshPartition, comm: &Comm, vdim: usize) -> Self {
        let n_global_nodes = comm.allreduce_sum_i64(partition.n_owned_nodes as i64) as usize;
        let n_owned_nodes = partition.n_owned_nodes;
        let n_ghost_nodes = partition.n_ghost_nodes;
        let n_local_nodes = partition.global_node_ids.len();
        let n_owned = n_owned_nodes * vdim;
        let n_ghost = n_ghost_nodes * vdim;

        let mut global_dof_ids = Vec::with_capacity(n_owned + n_ghost);
        let mut dof_owner = Vec::with_capacity(n_owned + n_ghost);
        for vd in 0..vdim {
            let block_base = (vd * n_global_nodes) as u32;
            for lid in 0..n_owned_nodes as u32 {
                global_dof_ids.push(partition.global_node(lid) + block_base);
                dof_owner.push(comm.rank());
            }
        }
        for vd in 0..vdim {
            let block_base = (vd * n_global_nodes) as u32;
            for (i, (lid, owner)) in partition.ghost_nodes().enumerate() {
                let _ = i;
                let gid = partition.global_node(lid);
                global_dof_ids.push(gid + block_base);
                dof_owner.push(owner);
            }
        }
        // NOTE: ghost segment layout above MUST match the permutation in
        // dm_to_partition below: both are vd-outer, ghost-node-inner
        // (gid = n_owned + vd * n_ghost_nodes + i).

        // Permutation: DofManager (space) ordering → partition [owned|ghost] ordering.
        let mut dm_to_partition = vec![0u32; n_local_nodes * vdim];
        for vd in 0..vdim {
            for node in 0..n_local_nodes {
                let dm_id = vd * n_local_nodes + node;
                let p_id = if node < n_owned_nodes {
                    vd * n_owned_nodes + node
                } else {
                    n_owned + vd * n_ghost_nodes + (node - n_owned_nodes)
                };
                dm_to_partition[dm_id] = p_id as u32;
            }
        }
        let mut partition_to_dm = vec![0u32; n_owned + n_ghost];
        for (dm_id, &p_id) in dm_to_partition.iter().enumerate() {
            partition_to_dm[p_id as usize] = dm_id as u32;
        }

        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;
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
            dm_to_partition,
            partition_to_dm,
            sign_corrections: Vec::new(),
        }
    }

    /// Build a DOF partition from a `DofManager` and `MeshPartition`.
    ///
    /// For P1, delegates to `from_mesh_partition`.  For P2, adds edge DOFs
    /// with ownership rule: `owner(edge) = min(owner(endpoint_a), owner(endpoint_b))`.
    ///
    /// Global DOF IDs:
    /// - Vertex DOFs keep their global node IDs `[0, total_global_vertices)`.
    /// - Edge DOFs get IDs `[total_global_vertices, total_global_vertices + total_global_edges)`,
    ///   assigned via a prefix scan on owned-edge counts.
    pub fn from_dof_manager(
        dof_manager: &DofManager,
        partition: &MeshPartition,
        comm: &Comm,
    ) -> Self {
        if dof_manager.order == 1 {
            return Self::from_mesh_partition(partition, comm);
        }
        // P2 (quads: 1 edge DOF + 1 center DOF per element) and P3 (quads:
        // p-1 edge DOFs + (p-1)² bubble DOFs per element) share the same
        // partition machinery: vertices, then edges (sorted by global node
        // pair), then element-interior DOFs (element order).
        assert!(
            dof_manager.order >= 2 && dof_manager.order <= 4,
            "DofPartition: only P2..=P4 (quad) supported"
        );
        let order = dof_manager.order as usize;
        let _interior_dofs_per = if order == 2 { 1 } else { (order - 1) * (order - 1) };

        let local_rank = comm.rank();
        let n_owned_vertices = partition.n_owned_nodes;
        let n_ghost_vertices = partition.n_ghost_nodes;

        // ── Classify edge DOFs as owned or ghost ────────────────────────────
        let mut owned_edges: Vec<EdgeDofInfo> = Vec::new();
        let mut ghost_edges: Vec<EdgeDofInfo> = Vec::new();

        // Each canonical edge -> its DOFs (P2: 1, P3: p-1), ordered from the
        // near-first-vertex side (DofManager convention).
        let edge_dofs_of: Vec<(EdgeKey, Vec<u32>)> = if order == 2 {
            dof_manager
                .edge_dof_map
                .iter()
                .map(|(&k, &d)| (k, vec![d]))
                .collect()
        } else {
            dof_manager
                .edge_pk_map
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect()
        };
        for (EdgeKey(local_a, local_b), dofs) in edge_dofs_of {
            let ga = partition.global_node(local_a);
            let gb = partition.global_node(local_b);
            let owner_a = partition.node_owner(local_a);
            let owner_b = partition.node_owner(local_b);
            let edge_owner = owner_a.min(owner_b);
            let (gna, gnb) = (ga.min(gb), ga.max(gb));

            // The DofManager's edge-DOF order is "near the first *local*
            // endpoint" — but local node ids are renumbered per-rank after
            // partitioning, so which of the DOFs is "first" is not
            // cross-rank consistent (it can flip for the same global edge).
            // dof_key must identify the DOF's position along the edge counted
            // from the GLOBAL min endpoint (0..per_edge-1).  For a single DOF
            // per edge (P2) dof_key is trivially 0 and the DofManager stores
            // no edge coordinates.  For P3+ (multiple DOFs per edge) the key
            // is the rank-consistent order of the DOFs' parameter along the
            // edge — a plain near-endpoint test collides for mid-edge DOFs
            // (e.g. P4's center GLL node satisfies da == db and would share
            // the near-min key with the quarter-point DOF, giving two ghost
            // slots the same global id → duplicated columns in spmv).
            let per_edge = dofs.len();
            let (ca, cb) = (dof_manager.dof_coord(local_a), dof_manager.dof_coord(local_b));
            if per_edge == 1 {
                let info = EdgeDofInfo {
                    local_dof_id: dofs[0],
                    global_node_a: gna,
                    global_node_b: gnb,
                    owner: edge_owner,
                    dof_key: 0,
                };
                if edge_owner == local_rank {
                    owned_edges.push(info);
                } else {
                    ghost_edges.push(info);
                }
            } else {
                // Parameter t ∈ [0,1] of each DOF along (ca → cb), then
                // re-based to the global-min endpoint; the sorted order is
                // the rank-independent dof_key.
                let mut t_sorted: Vec<(f64, u32)> = dofs.iter().map(|&local_dof_id| {
                    let c = dof_manager.dof_coord(local_dof_id);
                    let mut t = 0.0;
                    let mut len2 = 0.0;
                    for d in 0..c.len() {
                        let ab = cb[d] - ca[d];
                        t += (c[d] - ca[d]) * ab;
                        len2 += ab * ab;
                    }
                    let t_ab = if len2 > 0.0 { t / len2 } else { 0.0 };
                    let t_global = if ga < gb { t_ab } else { 1.0 - t_ab };
                    (t_global, local_dof_id)
                }).collect();
                t_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let key_of: std::collections::HashMap<u32, u32> = t_sorted.iter().enumerate()
                    .map(|(k, &(_, did))| (did, k as u32)).collect();
                for &local_dof_id in dofs.iter() {
                    let info = EdgeDofInfo {
                        local_dof_id,
                        global_node_a: gna,
                        global_node_b: gnb,
                        owner: edge_owner,
                        dof_key: key_of[&local_dof_id],
                    };
                    if edge_owner == local_rank {
                        owned_edges.push(info);
                    } else {
                        ghost_edges.push(info);
                    }
                }
            }
        }

        // Deterministic ordering by sorted global node pair, then within-edge index.
        owned_edges.sort_by_key(|e| (e.global_node_a, e.global_node_b, e.dof_key));
        ghost_edges.sort_by_key(|e| (e.global_node_a, e.global_node_b, e.dof_key));

        let n_owned_edges = owned_edges.len();
        let n_ghost_edges = ghost_edges.len();

        // ── Element-interior DOFs (P2 quad: 1 center; P3 quad: (p-1)²
        //    bubble DOFs) ───────────────────────────────────────────────────
        // Interior DOFs are those in `element_dofs` that are neither vertices
        // (id < n_vertex_dofs) nor edge DOFs.  P2 triangles have none.
        // Ownership = the element's owner; the global id =
        // n_global_vertices + n_global_edges + global_elem_id ·
        // interior_dofs_per + k (unique on every rank holding the element,
        // no exchange needed).
        let edge_dof_set: std::collections::HashSet<u32> = if order == 2 {
            dof_manager.edge_dof_map.values().copied().collect()
        } else {
            dof_manager
                .edge_pk_map
                .values()
                .flatten()
                .copied()
                .collect()
        };
        let n_total_vertices_loc = dof_manager.n_vertex_dofs as u32;
        let mut owned_interior: Vec<(Vec<u32>, u32)> = Vec::new(); // (dm dofs, local elem)
        let mut ghost_interior: Vec<(Vec<u32>, u32, Rank)> = Vec::new();
        let n_local_elems = partition.n_owned_elems + partition.n_ghost_elems;
        for e in 0..n_local_elems {
            let owner = partition.elem_owner[e];
            let interior: Vec<u32> = dof_manager
                .element_dofs(e as u32)
                .iter()
                .filter(|&&d| d >= n_total_vertices_loc && !edge_dof_set.contains(&d))
                .copied()
                .collect();
            if interior.is_empty() {
                continue;
            }
            if owner == local_rank {
                owned_interior.push((interior, e as u32));
            } else {
                ghost_interior.push((interior, e as u32, owner));
            }
        }
        let interior_dofs_per = if order == 2 { 1 } else { (order - 1) * (order - 1) };
        let n_owned_interior = owned_interior.len() * interior_dofs_per;
        let n_ghost_interior = ghost_interior.len() * interior_dofs_per;

        let n_owned = n_owned_vertices + n_owned_edges + n_owned_interior;
        let n_ghost = n_ghost_vertices + n_ghost_edges + n_ghost_interior;

        // ── Compute global offsets ──────────────────────────────────────────
        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;
        let total_global_vertices = comm.allreduce_sum_i64(n_owned_vertices as i64) as u32;
        let edge_offset = exclusive_scan_i64(comm, n_owned_edges as i64) as u32;
        let n_global_edges = comm.allreduce_sum_i64(n_owned_edges as i64) as u32;

        // ── Build owned DOF arrays ──────────────────────────────────────────
        let total = n_owned + n_ghost;
        let mut global_dof_ids = Vec::with_capacity(total);
        let mut dof_owner_vec = Vec::with_capacity(total);

        // Owned vertices: global ID = global node ID.
        for lid in 0..n_owned_vertices as u32 {
            // Read the stored global id directly (global_node() is identity
            // in identity mode and would return the compact slot itself).
            global_dof_ids.push(partition.global_node_ids[lid as usize]);
            dof_owner_vec.push(local_rank);
        }

        // Owned edges: global ID = total_global_vertices + edge_offset + i.
        let mut owned_edge_global_map: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for (i, edge) in owned_edges.iter().enumerate() {
            let gid = total_global_vertices + edge_offset + i as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_edge_global_map.insert(
                (edge.global_node_a, edge.global_node_b, edge.dof_key),
                gid,
            );
        }

        // Owned element-interior DOFs: global ID = n_global_vertices +
        // n_global_edges + global_elem_id · interior_dofs_per + k (element
        // order is the canonical interior numbering).
        for (bubble, le) in &owned_interior {
            let ge = partition.global_elem(*le);
            for (k, &d) in bubble.iter().enumerate() {
                let gid = total_global_vertices
                    + n_global_edges
                    + ge * interior_dofs_per as u32
                    + k as u32;
                global_dof_ids.push(gid);
                dof_owner_vec.push(local_rank);
                let _ = d;
            }
        }

        // ── Build ghost DOF arrays ──────────────────────────────────────────

        // Ghost vertices.
        for lid in n_owned_vertices..(n_owned_vertices + n_ghost_vertices) {
            // `lid` is a compact slot: read the stored global id directly
            // (global_node() is identity in identity mode and would return
            // the compact slot itself).  node_owner() needs the compact slot
            // in compact mode and the global id in identity mode.
            let gid = partition.global_node_ids[lid as usize];
            global_dof_ids.push(gid);
            let owner = if partition.node_id_identity {
                partition.node_owner(gid)
            } else {
                partition.node_owner(lid as u32)
            };
            dof_owner_vec.push(owner);
        }

        let ghost_edge_gids = exchange_ghost_edge_ids(
            &ghost_edges, &owned_edge_global_map, comm,
        );
        for (i, edge) in ghost_edges.iter().enumerate() {
            global_dof_ids.push(ghost_edge_gids[i]);
            dof_owner_vec.push(edge.owner);
        }

        // Ghost element-interior DOFs (same gid formula as owned — element
        // global ids are shared, so no exchange needed).
        for (bubble, le, owner) in &ghost_interior {
            let ge = partition.global_elem(*le);
            for (k, &d) in bubble.iter().enumerate() {
                let gid = total_global_vertices
                    + n_global_edges
                    + ge * interior_dofs_per as u32
                    + k as u32;
                global_dof_ids.push(gid);
                dof_owner_vec.push(*owner);
                let _ = d;
            }
        }
        debug_assert_eq!(global_dof_ids.len(), total);

        // ── Build dm_to_partition permutation ───────────────────────────────
        // Maps DofManager's local DOF ID → partition's local DOF ID.
        // Partition layout:
        //   [owned_vertices | owned_edges | owned_interior |
        //    ghost_vertices | ghost_edges | ghost_interior]
        // DofManager layout:
        //   [all_local_vertices | all_edges_in_enum_order | interior_in_elem_order]
        let n_dm_dofs = dof_manager.n_dofs;
        let mut dm_to_partition = vec![0u32; n_dm_dofs];

        // Vertices: DM IDs 0..n_owned_vertices → partition 0..n_owned_vertices (unchanged)
        for i in 0..n_owned_vertices {
            dm_to_partition[i] = i as u32;
        }
        // Ghost vertices: DM IDs n_owned_vertices..n_total_vertices → partition n_owned..n_owned+n_ghost_vertices
        let n_total_vertices = n_owned_vertices + n_ghost_vertices;
        for i in n_owned_vertices..n_total_vertices {
            dm_to_partition[i] = (n_owned + (i - n_owned_vertices)) as u32;
        }
        // Owned edges
        for (i, edge) in owned_edges.iter().enumerate() {
            dm_to_partition[edge.local_dof_id as usize] = (n_owned_vertices + i) as u32;
        }
        // Ghost edges
        for (i, edge) in ghost_edges.iter().enumerate() {
            dm_to_partition[edge.local_dof_id as usize] =
                (n_owned + n_ghost_vertices + i) as u32;
        }
        // Element-interior DOFs: partition owned segment after edges, ghost
        // segment after ghost edges.
        for (i, (bubble, _)) in owned_interior.iter().enumerate() {
            for (k, &d) in bubble.iter().enumerate() {
                dm_to_partition[d as usize] =
                    (n_owned_vertices + n_owned_edges + i * interior_dofs_per + k) as u32;
            }
        }
        for (i, (bubble, _, _)) in ghost_interior.iter().enumerate() {
            for (k, &d) in bubble.iter().enumerate() {
                dm_to_partition[d as usize] = (n_owned
                    + n_ghost_vertices
                    + n_ghost_edges
                    + i * interior_dofs_per
                    + k) as u32;
            }
        }

        // Build inverse permutation.
        let mut partition_to_dm = vec![0u32; n_dm_dofs];
        for (dm_id, &part_id) in dm_to_partition.iter().enumerate() {
            partition_to_dm[part_id as usize] = dm_id as u32;
        }

        // ── Reverse lookup ──────────────────────────────────────────────────
        let dof_global_to_local: HashMap<u32, u32> = global_dof_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();

        DofPartition {
            n_owned_dofs: n_owned,
            n_ghost_dofs: n_ghost,
            global_dof_ids,
            dof_owner: dof_owner_vec,
            global_dof_offset,
            dof_global_to_local,
            dm_to_partition,
            partition_to_dm,
            sign_corrections: Vec::new(), // P2 H1 DOFs are sign-invariant
        }
    }

    /// Build a DOF partition for an edge-DOF-only space (H(curl) or H(div) 2D).
    ///
    /// For H(curl) ND1 and H(div) RT0 on triangles, DOFs are edges — no vertex DOFs.
    /// Edge ownership: `owner(edge(a,b)) = min(owner(a), owner(b))`.
    ///
    /// The permutation maps from the serial space's DOF ordering (edge enum order)
    /// to the partition layout: `[owned_edges | ghost_edges]`.
    ///
    /// **Sign corrections** — H(curl) / H(div) basis functions carry a sign that
    /// depends on the local vertex ordering (`nodes[li] < nodes[lj]`).  After
    /// mesh partitioning, local vertex IDs can disagree with the canonical
    /// global ordering, flipping the sign on some edges.  This method computes
    /// a per-DOF correction `d_i = global_sign / local_sign ∈ {-1, +1}` and
    /// stores it in [`DofPartition::sign_corrections`] so that callers can
    /// transform to the globally consistent basis during permutation.
    pub fn from_edge_space<S: FESpace>(
        space: &S,
        partition: &MeshPartition,
        comm: &Comm,
    ) -> Self
    where
        S::Mesh: MeshTopology,
    {
        let local_rank = comm.rank();
        let mesh = space.mesh();
        let n_space_dofs = space.n_dofs();

        let dim = mesh.dim() as usize;

        let space_type = space.space_type();

        // Per-element-type local edge tables, matching the serial
        // `HCurlSpace` / `HDivSpace` element ordering (fem_space::hcurl /
        // fem_space::hdiv).  Mixed 3-D meshes (tet + hex + prism, e.g.
        // fichera-mixed.mesh used by pex34) require the table of *each*
        // element, not just element 0's.
        fn edges_for_elem(space_type: fem_space::fe_space::SpaceType, dim: u8, et: fem_mesh::ElementType) -> &'static [(usize, usize)] {
            use fem_mesh::ElementType;
            match (space_type, dim, et) {
                (fem_space::fe_space::SpaceType::HCurl, 2, ElementType::Quad4 | ElementType::Quad8) => {
                    &[(0, 1), (1, 2), (2, 3), (3, 0)]
                }
                (fem_space::fe_space::SpaceType::HCurl, 2, _) => &[(0, 1), (1, 2), (0, 2)],
                (fem_space::fe_space::SpaceType::HDiv, 2, ElementType::Quad4 | ElementType::Quad8) => {
                    &[(0, 1), (1, 2), (2, 3), (3, 0)]
                }
                (fem_space::fe_space::SpaceType::HDiv, 2, _) => &[(1, 2), (0, 2), (0, 1)],
                // 3-D HCurl (NDk): tet (6), hex (12, MFEM CUBE Edges), prism
                // (9) and pyramid (8) edge tables from fem_space::hcurl.
                (fem_space::fe_space::SpaceType::HCurl, 3, ElementType::Tet4 | ElementType::Tet10) => {
                    &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                }
                (fem_space::fe_space::SpaceType::HCurl, 3, ElementType::Hex8 | ElementType::Hex20) => {
                    &[
                        (0, 1), (1, 2), (3, 2), (0, 3),
                        (4, 5), (5, 6), (7, 6), (4, 7),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ]
                }
                (fem_space::fe_space::SpaceType::HCurl, 3, ElementType::Prism6) => {
                    &[
                        (0, 1), (1, 2), (0, 2), // bottom tri
                        (3, 4), (4, 5), (3, 5), // top tri
                        (0, 3), (1, 4), (2, 5), // verticals
                    ]
                }
                (fem_space::fe_space::SpaceType::HCurl, 3, ElementType::Pyramid5) => {
                    &[
                        (0, 1), (1, 2), (2, 3), (3, 0), // base
                        (0, 4), (1, 4), (2, 4), (3, 4), // apex
                    ]
                }
                _ => &[(0, 1), (1, 2), (0, 2)],
            }
        }

        // DOFs per edge for the supported spaces: NDk has k per edge, RTk has
        // k+1 per edge (RT0/ND1 → 1).  Higher-order 3-D mixed meshes are not
        // supported by this path.
        let order = space.order() as usize;
        let dofs_per_edge = match space_type {
            fem_space::fe_space::SpaceType::HCurl => order.max(1),
            fem_space::fe_space::SpaceType::HDiv => order + 1,
            _ => 1,
        };

        // ── Step 1: Map each DOF to its canonical edge (or interior) ──────────
        //
        // For higher-order spaces, DOFs are grouped:
        //   [edge0 × dofs_per_edge, edge1 × dofs_per_edge, ..., interior...]
        let mut dof_to_edge: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut dof_to_edge_pos: HashMap<u32, u32> = HashMap::new(); // dof_id -> position within its edge
        let mut interior_dofs: Vec<(u32, u32, u32)> = Vec::new(); // (dof_id, local_elem_id, dof_idx_in_elem)
        let mut sign_corr: Vec<f64> = vec![1.0; n_space_dofs];

        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let local_edges = edges_for_elem(space_type, dim as u8, et);
            let edge_dofs_total = local_edges.len() * dofs_per_edge;
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            for (i, &dof_id) in dofs.iter().enumerate() {
                if dof_to_edge.contains_key(&dof_id) || interior_dofs.iter().any(|&(d, _, _)| d == dof_id) {
                    continue;
                }

                if i < edge_dofs_total {
                    // Edge DOF: map to the corresponding edge of this element.
                    let edge_idx = i / dofs_per_edge;
                    let (a, b) = local_edges[edge_idx];
                    let local_a = nodes[a];
                    let local_b = nodes[b];
                    let ga = partition.global_node(local_a);
                    let gb = partition.global_node(local_b);

                    dof_to_edge.insert(dof_id, (ga.min(gb), ga.max(gb)));
                    // Physical per-edge position (mom0/mom1): the edge's DOFs
                    // are [first, first+dofs_per_edge) in the space's global
                    // numbering, so `dof_id - first` identifies the moment
                    // regardless of the element's edge orientation (rev).
                    // Using `i % dofs_per_edge` instead is WRONG: it depends
                    // on which element first "sees" the edge, and that
                    // traversal order differs between ranks.
                    let edge_start = edge_idx * dofs_per_edge;
                    let edge_first = dofs[edge_start..edge_start + dofs_per_edge]
                        .iter()
                        .min()
                        .copied()
                        .unwrap_or(dof_id);
                    dof_to_edge_pos.insert(dof_id, dof_id - edge_first);

                    // Sign correction: local_sign * d = global_sign, so d = global_sign / local_sign.
                    let local_sign: f64 = if local_a < local_b { 1.0 } else { -1.0 };
                    let global_sign: f64 = if ga < gb { 1.0 } else { -1.0 };
                    sign_corr[dof_id as usize] = global_sign / local_sign;
                } else {
                    // Interior DOF: record with element info for ownership later.
                    interior_dofs.push((dof_id, e, i as u32));
                }
            }
        }

        // ── Step 2: Classify DOFs as owned or ghost ────────────────────────────
        let mut owned_edges: Vec<EdgeDofInfo> = Vec::new();
        let mut ghost_edges: Vec<EdgeDofInfo> = Vec::new();

        // Build global-to-local node map for ownership lookup.
        // In identity mode local node ids ARE global ids, so iterate the
        // local mesh's full node range (owned + ghost + unused holes);
        // in compact mode the range equals owned+ghost count.
        let n_total_nodes = mesh.n_nodes() as u32;
        let mut global_to_local_node: HashMap<u32, u32> = HashMap::new();
        for lid in 0..n_total_nodes {
            global_to_local_node.insert(partition.global_node(lid), lid);
        }

        for (&dof_id, &(ga, gb)) in &dof_to_edge {
            let owner_a = if partition.node_id_identity {
                // Identity mode: ga is already a local-mesh node id.
                partition.node_owner(ga)
            } else {
                global_to_local_node.get(&ga)
                    .map(|&lid| partition.node_owner(lid))
                    .unwrap_or(Rank::MAX)
            };
            let owner_b = if partition.node_id_identity {
                partition.node_owner(gb)
            } else {
                global_to_local_node.get(&gb)
                    .map(|&lid| partition.node_owner(lid))
                    .unwrap_or(Rank::MAX)
            };
            let edge_owner = owner_a.min(owner_b);

            let info = EdgeDofInfo {
                local_dof_id: dof_id,
                global_node_a: ga,
                global_node_b: gb,
                owner: edge_owner,
                // In-edge position (0..dofs_per_edge): cross-rank consistent
                // (each edge's DOFs appear in global DOF order inside the
                // element DOF list; i%2 == dof_id - first regardless of the
                // local edge enumeration).  The raw DOF id is NOT usable:
                // HDiv/HCurl spaces number DOFs by local element traversal,
                // which differs between ranks.
                dof_key: *dof_to_edge_pos.get(&dof_id).unwrap_or(&0),
            };

            if edge_owner == local_rank {
                owned_edges.push(info);
            } else {
                ghost_edges.push(info);
            }
        }

        // Deterministic ordering by sorted global node pair.
        owned_edges.sort_by_key(|e| (e.global_node_a, e.global_node_b));
        ghost_edges.sort_by_key(|e| (e.global_node_a, e.global_node_b));

        // ── Step 2b: Process interior DOFs ───────────────────────────────────
        // Interior DOFs are classified by their element ownership.
        // We use the partition's elem_owner array (owned elements first, then ghost).
        let mut owned_interior: Vec<(u32, u32, u32)> = Vec::new(); // (dof_id, elem_gid, dof_idx)
        let mut ghost_interior: Vec<(u32, u32, u32, Rank)> = Vec::new();

        for &(dof_id, le, dof_idx) in &interior_dofs {
            let elem_gid = partition.global_elem(le);
            let owner = if (le as usize) < partition.n_owned_elems {
                local_rank
            } else {
                partition.elem_owner[le as usize]
            };
            if owner == local_rank {
                owned_interior.push((dof_id, elem_gid, dof_idx));
            } else {
                ghost_interior.push((dof_id, elem_gid, dof_idx, owner));
            }
        }
        owned_interior.sort();
        ghost_interior.sort_by_key(|&(dof_id, _, _, _)| dof_id);

        let n_owned_edge = owned_edges.len();
        let n_ghost_edge = ghost_edges.len();
        let n_owned_interior = owned_interior.len();
        let n_ghost_interior = ghost_interior.len();
        let n_owned = n_owned_edge + n_owned_interior;
        let n_ghost = n_ghost_edge + n_ghost_interior;
        let total = n_owned + n_ghost;

        // Note: interior DOFs may not cover all remaining DOFs if the space
        // has interior DOFs that weren't found by element iteration. The
        // assertion below only applies when we've classified every DOF.
        if total != n_space_dofs {
            // This can happen for unsupported element types — log a warning
            // but don't crash. The DOFs not found will be unclassified and
            // may cause issues downstream.
            eprintln!("  Warning: from_edge_space classified {total}/{n_space_dofs} DOFs \
                (edge={}, interior={}). Missing DOFs may cause errors.",
                n_owned_edge + n_ghost_edge, n_owned_interior + n_ghost_interior);
        }

        // ── Step 3: Compute global offsets ─────────────────────────────────────
        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;
        let edge_offset = global_dof_offset as u32;

        // ── Step 4: Build global DOF IDs ───────────────────────────────────────
        let mut global_dof_ids = Vec::with_capacity(total);
        let mut dof_owner_vec = Vec::with_capacity(total);

        // 4a. Owned DOFs: edges first, then interior DOFs.
        //
        // Partition layout is [owned | ghost]; n_owned_dofs = n_owned_edge +
        // n_owned_interior, so ALL owned DOFs must precede the ghost segment.
        // (Previously ghost edges were pushed right after owned edges, which
        // mixed ghost DOFs into the owned segment once interior DOFs exist —
        // RT0 escaped because it has no interior DOFs; RT1/RT2 break.)
        //
        // NOTE: RTk spaces have `dofs_per_edge` DOFs per edge (e.g. RT1: 2).
        // The map key must include the DOF id — using only the node pair
        // collapses the edge's DOFs to a single gid and the ghost exchange
        // returns the same gid for all of them (duplicate global DOFs).
        let mut owned_edge_global_map: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for (i, edge) in owned_edges.iter().enumerate() {
            let gid = edge_offset + i as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_edge_global_map.insert(
                (edge.global_node_a, edge.global_node_b, edge.dof_key),
                gid,
            );
        }

        // Owned interior DOFs.
        let owned_interior_offset = edge_offset + owned_edges.len() as u32;
        let mut owned_interior_map: HashMap<(u32, u32), u32> = HashMap::new();
        for (j, &(_dof_id, elem_gid, dof_idx)) in owned_interior.iter().enumerate() {
            let gid = owned_interior_offset + j as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_interior_map.insert((elem_gid, dof_idx), gid);
        }

        // 4b. Ghost DOFs: edges then interior.
        let ghost_edge_gids = exchange_ghost_edge_ids(
            &ghost_edges, &owned_edge_global_map, comm,
        );
        for (i, edge) in ghost_edges.iter().enumerate() {
            global_dof_ids.push(ghost_edge_gids[i]);
            dof_owner_vec.push(edge.owner);
        }

        let ghost_interior_gids = exchange_ghost_interior_ids(
            &ghost_interior, &owned_interior_map, comm,
        );
        for &gid in &ghost_interior_gids {
            global_dof_ids.push(gid);
        }
        for &(_, _, _, owner) in &ghost_interior {
            // Global ID already set by exchange.
            dof_owner_vec.push(owner);
        }

        // ── Step 5: Build permutation ──────────────────────────────────────────
        let mut dm_to_partition = vec![0u32; n_space_dofs];
        let mut partition_to_dm = vec![0u32; n_space_dofs];

        // Edge DOFs
        for (i, edge) in owned_edges.iter().enumerate() {
            dm_to_partition[edge.local_dof_id as usize] = i as u32;
        }
        // Owned interior DOFs follow owned edges.
        for (j, &(dof_id, _, _)) in owned_interior.iter().enumerate() {
            dm_to_partition[dof_id as usize] = (n_owned_edge + j) as u32;
        }
        // Ghost edges/interior start after the whole owned segment.
        for (i, edge) in ghost_edges.iter().enumerate() {
            dm_to_partition[edge.local_dof_id as usize] = (n_owned + i) as u32;
        }
        for (j, &(dof_id, _, _, _)) in ghost_interior.iter().enumerate() {
            dm_to_partition[dof_id as usize] = (n_owned + n_ghost_edge + j) as u32;
        }
        // Build reverse permutation
        for (dm_id, &part_id) in dm_to_partition.iter().enumerate() {
            partition_to_dm[part_id as usize] = dm_id as u32;
        }

        let dof_global_to_local: HashMap<u32, u32> = global_dof_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();

        DofPartition {
            n_owned_dofs: n_owned,
            n_ghost_dofs: n_ghost,
            global_dof_ids,
            dof_owner: dof_owner_vec,
            global_dof_offset,
            dof_global_to_local,
            dm_to_partition,
            partition_to_dm,
            sign_corrections: sign_corr,
        }
    }

    /// Build a DOF partition for 3-D H(div) (Raviart-Thomas) face-based spaces.
    ///
    /// RT0 (hex: 6 quad-face DOFs, tet: 4 tri-face DOFs; RT1 hex: 4 DOFs per
    /// quad face + 12 interior DOFs).  Face DOF ownership:
    /// `owner(face) = min(owner of the face's vertices)`.
    ///
    /// The canonical face orientation follows MFEM's Elem1 rule (the element
    /// with the **minimum global element id** among the elements sharing the
    /// face), so the partition sign corrections are cross-rank consistent even
    /// though the space's own first-seen (local traversal) canonical face may
    /// differ between ranks.
    pub fn from_face_space<S: FESpace>(
        space: &S,
        partition: &MeshPartition,
        comm: &Comm,
    ) -> Self
    where
        S::Mesh: MeshTopology,
    {
        let local_rank = comm.rank();
        let mesh = space.mesh();
        let n_space_dofs = space.n_dofs();
        let dim = mesh.dim() as usize;
        assert_eq!(dim, 3, "from_face_space: requires a 3-D mesh");
        assert!(
            mesh.n_elements() > 0,
            "from_face_space: empty local mesh"
        );

        let elem0 = mesh.element_type(0);
        let _ = elem0;
        // Per-element-type local face tables in the **canonical (MFEM
        // FaceVert) vertex order** — the order the serial `HDivSpace` uses for
        // its element face signs.  `from_face_space` must record first_seen /
        // min-gid faces in this same order, otherwise the partition sign
        // corrections do not line up with the space's element signs on
        // tet/prism faces (whose plain face lists differ from the canonical
        // order; hex faces already equal their canonical order).
        // Each entry: (canonical vertex order, is_tri).
        fn faces_for_elem(
            et: fem_mesh::ElementType,
            order: usize,
        ) -> (Vec<(Vec<usize>, bool)>, usize) {
            use fem_mesh::ElementType;
            match et {
                ElementType::Tet4 | ElementType::Tet10 => (
                    vec![
                        (vec![1, 2, 3], true),
                        (vec![0, 3, 2], true),
                        (vec![0, 1, 3], true),
                        (vec![0, 2, 1], true),
                    ],
                    0,
                ),
                ElementType::Hex8 => (
                    vec![
                        (vec![3, 2, 1, 0], false), // z=-1 (bottom)
                        (vec![0, 1, 5, 4], false), // y=-1 (front)
                        (vec![1, 2, 6, 5], false), // x=+1 (right)
                        (vec![2, 3, 7, 6], false), // y=+1 (back)
                        (vec![3, 0, 4, 7], false), // x=-1 (left)
                        (vec![4, 5, 6, 7], false), // z=+1 (top)
                    ],
                    if order == 0 { 0 } else { 12 },
                ),
                ElementType::Prism6 => (
                    vec![
                        (vec![0, 2, 1], true),  // bottom (tri)
                        (vec![3, 4, 5], true),  // top (tri)
                        (vec![0, 1, 4, 3], false), // quad (front)
                        (vec![1, 2, 5, 4], false), // quad (right)
                        (vec![2, 0, 3, 5], false), // quad (left)
                    ],
                    0,
                ),
                _ => panic!("from_face_space: unsupported element type {et:?}"),
            }
        }
        let dofs_per_face = space.order() as usize + 1;

        // Per-face metadata: the space-canonical (first-seen) vertex order,
        // the min-global-element-id face order (global canonical), and the
        // face's owner rank (min over the *elements* sharing the face — the
        // owner must actually hold the face locally, which the vertex-min
        // rule does not guarantee).
        struct FaceInfo {
            first_order: Vec<u32>,
            min_gid_elem: u32,
            min_gid_order: Vec<u32>,
            min_elem_owner: Rank,
        }

        let mut face_info: HashMap<(u32, u32, u32), FaceInfo> = HashMap::new();
        let mut dof_to_face: HashMap<u32, (u32, u32, u32)> = HashMap::new();
        let mut dof_to_pos: HashMap<u32, u32> = HashMap::new();
        let mut interior_dofs: Vec<(u32, u32, u32)> = Vec::new(); // (dof_id, local_elem, dof_idx)
        let mut sign_corr: Vec<f64> = vec![1.0; n_space_dofs];

        for e in mesh.elem_iter() {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let elem_gid = partition.global_elem(e);
            let (faces, _n_interior) = faces_for_elem(mesh.element_type(e), space.order() as usize);
            let n_face_dofs_total = faces.len() * dofs_per_face;

            for (i, &dof_id) in dofs.iter().enumerate() {
                if i >= n_face_dofs_total {
                    interior_dofs.push((dof_id, e, i as u32));
                    continue;
                }
                let face_idx = i / dofs_per_face;
                let pos = i % dofs_per_face;
                let (fv, _is_tri) = &faces[face_idx];
                let verts_global: Vec<u32> =
                    fv.iter().map(|&li| partition.global_node(nodes[li])).collect();
                // Face key: the 3 smallest global vertex ids of the *whole*
                // face (HDivSpace FaceKey convention) — the local face vertex
                // order differs between the two elements sharing a face, so
                // taking the first 3 of the local ordering would produce
                // different keys on different ranks.
                let mut v4 = verts_global.clone();
                v4.sort_unstable();
                let key = (v4[0], v4[1], v4[2]);

                let entry = face_info.entry(key).or_insert_with(|| {
                    let owner = if (e as usize) < partition.n_owned_elems {
                        local_rank
                    } else {
                        partition.elem_owner[e as usize]
                    };
                    FaceInfo {
                        first_order: verts_global.clone(),
                        min_gid_elem: elem_gid,
                        min_gid_order: verts_global.clone(),
                        min_elem_owner: owner,
                    }
                });
                if !dof_to_face.contains_key(&dof_id) {
                    // First element in traversal maps the DOF: this is the
                    // space's canonical face orientation (hdiv.rs face_canon
                    // first-seen rule), in the traversal's vertex order.
                    entry.first_order = verts_global.clone();
                    dof_to_face.insert(dof_id, key);
                    dof_to_pos.insert(dof_id, pos as u32);
                }
                if elem_gid < entry.min_gid_elem {
                    entry.min_gid_elem = elem_gid;
                    entry.min_gid_order = verts_global.clone();
                }
                let owner = if (e as usize) < partition.n_owned_elems {
                    local_rank
                } else {
                    partition.elem_owner[e as usize]
                };
                if owner < entry.min_elem_owner {
                    entry.min_elem_owner = owner;
                }
            }
        }

        // ── Step 2: classify owned / ghost faces and compute sign corrections
        let mut owned_faces: Vec<FaceDofInfo> = Vec::new();
        let mut ghost_faces: Vec<FaceDofInfo> = Vec::new();

        for (&dof_id, &key) in &dof_to_face {
            let info = &face_info[&key];
            let owner = info.min_elem_owner;

            // Sign correction: parity(space canonical first-seen order, global
            // canonical min-gid-element order).  Multiplying the space's
            // element signs by this yields each element's sign relative to the
            // global canonical face orientation, which is cross-rank
            // consistent.
            let sign = if info.first_order.len() == 4 {
                fem_space::hdiv::rt_face_sign(fem_space::hdiv::quad_orientation(
                    [
                        info.first_order[0],
                        info.first_order[1],
                        info.first_order[2],
                        info.first_order[3],
                    ],
                    [
                        info.min_gid_order[0],
                        info.min_gid_order[1],
                        info.min_gid_order[2],
                        info.min_gid_order[3],
                    ],
                ))
            } else {
                fem_space::hdiv::rt_face_sign(fem_space::hdiv::tri_orientation(
                    [info.first_order[0], info.first_order[1], info.first_order[2]],
                    [
                        info.min_gid_order[0],
                        info.min_gid_order[1],
                        info.min_gid_order[2],
                    ],
                ))
            };
            sign_corr[dof_id as usize] = sign;

            let f = FaceDofInfo {
                local_dof_id: dof_id,
                face_key: key,
                pos: dof_to_pos[&dof_id],
                owner,
            };
            if owner == local_rank {
                owned_faces.push(f);
            } else {
                ghost_faces.push(f);
            }
        }

        // Deterministic ordering by (face key, position within face).
        owned_faces.sort_by_key(|f| (f.face_key.0, f.face_key.1, f.face_key.2, f.pos));
        ghost_faces.sort_by_key(|f| (f.face_key.0, f.face_key.1, f.face_key.2, f.pos));

        // ── Step 2b: interior DOFs (element-owned) ───────────────────────────
        let mut owned_interior: Vec<(u32, u32, u32)> = Vec::new();
        let mut ghost_interior: Vec<(u32, u32, u32, Rank)> = Vec::new();
        for &(dof_id, le, dof_idx) in &interior_dofs {
            let elem_gid = partition.global_elem(le);
            let owner = if (le as usize) < partition.n_owned_elems {
                local_rank
            } else {
                partition.elem_owner[le as usize]
            };
            if owner == local_rank {
                owned_interior.push((dof_id, elem_gid, dof_idx));
            } else {
                ghost_interior.push((dof_id, elem_gid, dof_idx, owner));
            }
        }
        owned_interior.sort();
        ghost_interior.sort_by_key(|&(dof_id, _, _, _)| dof_id);

        let n_owned_face = owned_faces.len();
        let n_ghost_face = ghost_faces.len();
        let n_owned_interior = owned_interior.len();
        let n_ghost_interior = ghost_interior.len();
        let n_owned = n_owned_face + n_owned_interior;
        let n_ghost = n_ghost_face + n_ghost_interior;
        let total = n_owned + n_ghost;

        if total != n_space_dofs {
            eprintln!(
                "  Warning: from_face_space classified {total}/{n_space_dofs} DOFs \
                 (face={}, interior={}). Missing DOFs may cause errors.",
                n_owned_face + n_ghost_face,
                n_owned_interior + n_ghost_interior
            );
        }

        // ── Step 3: global offsets ─────────────────────────────────────────────
        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;

        // ── Step 4: build global DOF IDs ───────────────────────────────────────
        let mut global_dof_ids = Vec::with_capacity(total);
        let mut dof_owner_vec = Vec::with_capacity(total);

        let mut owned_face_global_map: HashMap<(u32, u32, u32, u32), u32> = HashMap::new();
        for (i, f) in owned_faces.iter().enumerate() {
            let gid = global_dof_offset as u32 + i as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_face_global_map.insert((f.face_key.0, f.face_key.1, f.face_key.2, f.pos), gid);
        }

        let owned_interior_offset = global_dof_offset + owned_faces.len();
        let mut owned_interior_map: HashMap<(u32, u32), u32> = HashMap::new();
        for (j, &(_dof_id, elem_gid, dof_idx)) in owned_interior.iter().enumerate() {
            let gid = owned_interior_offset as u32 + j as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_interior_map.insert((elem_gid, dof_idx), gid);
        }

        let ghost_face_gids = exchange_ghost_face_ids(&ghost_faces, &owned_face_global_map, comm);
        for (i, f) in ghost_faces.iter().enumerate() {
            global_dof_ids.push(ghost_face_gids[i]);
            dof_owner_vec.push(f.owner);
        }

        let ghost_interior_gids =
            exchange_ghost_interior_ids(&ghost_interior, &owned_interior_map, comm);
        for &gid in &ghost_interior_gids {
            global_dof_ids.push(gid);
        }
        for &(_, _, _, owner) in &ghost_interior {
            dof_owner_vec.push(owner);
        }

        // ── Step 5: permutation ────────────────────────────────────────────────
        let mut dm_to_partition = vec![0u32; n_space_dofs];
        let mut partition_to_dm = vec![0u32; n_space_dofs];

        for (i, f) in owned_faces.iter().enumerate() {
            dm_to_partition[f.local_dof_id as usize] = i as u32;
        }
        for (j, &(dof_id, _, _)) in owned_interior.iter().enumerate() {
            dm_to_partition[dof_id as usize] = (n_owned_face + j) as u32;
        }
        for (i, f) in ghost_faces.iter().enumerate() {
            dm_to_partition[f.local_dof_id as usize] = (n_owned + i) as u32;
        }
        for (j, &(dof_id, _, _, _)) in ghost_interior.iter().enumerate() {
            dm_to_partition[dof_id as usize] = (n_owned + n_ghost_face + j) as u32;
        }
        for (dm_id, &part_id) in dm_to_partition.iter().enumerate() {
            partition_to_dm[part_id as usize] = dm_id as u32;
        }

        let dof_global_to_local: HashMap<u32, u32> = global_dof_ids
            .iter()
            .enumerate()
            .map(|(lid, &gid)| (gid, lid as u32))
            .collect();

        DofPartition {
            n_owned_dofs: n_owned,
            n_ghost_dofs: n_ghost,
            global_dof_ids,
            dof_owner: dof_owner_vec,
            global_dof_offset,
            dof_global_to_local,
            dm_to_partition,
            partition_to_dm,
            sign_corrections: sign_corr,
        }
    }

    /// Total local DOF count (owned + ghost).
    #[inline]
    pub fn n_total_dofs(&self) -> usize { self.n_owned_dofs + self.n_ghost_dofs }

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

    /// `true` if DOF reordering is needed (P2+).
    #[inline]
    pub fn needs_permutation(&self) -> bool {
        !self.dm_to_partition.is_empty()
    }

    /// `true` if sign corrections must be applied during permutation.
    ///
    /// This is the case for H(curl)/H(div) spaces where the local mesh's
    /// vertex ordering may disagree with the canonical global ordering,
    /// causing edge basis function signs to flip.
    #[inline]
    pub fn needs_sign_correction(&self) -> bool {
        !self.sign_corrections.is_empty()
    }

    /// Sign correction factor (±1.0) for a DofManager-local DOF.
    ///
    /// Returns `+1.0` if no correction is needed (P1, P2, or matching signs).
    #[inline]
    pub fn sign_correction(&self, dm_local_id: u32) -> f64 {
        if self.sign_corrections.is_empty() {
            1.0
        } else {
            self.sign_corrections[dm_local_id as usize]
        }
    }

    /// Map a DofManager local DOF ID to the partition's local DOF ID.
    /// Returns the input unchanged for P1 (identity).
    #[inline]
    pub fn permute_dof(&self, dm_local_id: u32) -> u32 {
        if self.dm_to_partition.is_empty() {
            dm_local_id
        } else {
            self.dm_to_partition[dm_local_id as usize]
        }
    }

    /// Map a partition local DOF ID back to DofManager's local DOF ID.
    /// Returns the input unchanged for P1 (identity).
    #[inline]
    pub fn unpermute_dof(&self, partition_local_id: u32) -> u32 {
        if self.dm_to_partition.is_empty() {
            partition_local_id
        } else {
            self.partition_to_dm[partition_local_id as usize]
        }
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

// ── Ghost edge ID exchange ──────────────────────────────────────────────────

/// Exchange global DOF IDs for ghost edge DOFs via alltoallv.
///
/// Each rank sends its ghost edges (identified by sorted global node pairs) to
/// the owner rank.  The owner looks up the global DOF ID and sends it back.
fn exchange_ghost_edge_ids(
    ghost_edges: &[EdgeDofInfo],
    owned_edge_global_map: &HashMap<(u32, u32, u32), u32>,
    comm: &Comm,
) -> Vec<u32> {
    if comm.size() <= 1 || ghost_edges.is_empty() {
        return Vec::new();
    }

    // Group ghost edges by owner rank. Each request carries the local DOF id
    // so edges with multiple DOFs (RTk: dofs_per_edge > 1) resolve to
    // distinct global DOF ids on the owner.
    let mut requests_by_owner: HashMap<Rank, Vec<(usize, u32, u32, u32)>> = HashMap::new();
    for (i, edge) in ghost_edges.iter().enumerate() {
        requests_by_owner.entry(edge.owner).or_default()
            .push((i, edge.global_node_a, edge.global_node_b, edge.dof_key));
    }

    // Phase 1: send edge requests (node pair + dof id) to owners.
    let sends: Vec<(Rank, Vec<u8>)> = requests_by_owner
        .iter()
        .map(|(&owner, edges)| {
            let bytes: Vec<u8> = edges.iter()
                .flat_map(|&(_, a, b, d)| {
                    let mut buf = [0u8; 12];
                    buf[..4].copy_from_slice(&a.to_le_bytes());
                    buf[4..8].copy_from_slice(&b.to_le_bytes());
                    buf[8..].copy_from_slice(&d.to_le_bytes());
                    buf
                })
                .collect();
            (owner, bytes)
        })
        .collect();

    let received = comm.alltoallv_bytes(&sends);

    // Phase 2: owners look up global DOF IDs and reply.
    let replies: Vec<(Rank, Vec<u8>)> = received.iter()
        .map(|(requester, bytes)| {
            debug_assert_eq!(bytes.len() % 12, 0);
            let reply_bytes: Vec<u8> = bytes.chunks_exact(12)
                .flat_map(|chunk| {
                    let a = u32::from_le_bytes(chunk[..4].try_into().unwrap());
                    let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    let d = u32::from_le_bytes(chunk[8..].try_into().unwrap());
                    let gid = owned_edge_global_map.get(&(a, b, d))
                        .unwrap_or_else(|| panic!(
                            "exchange_ghost_edge_ids: rank {} requested edge ({a},{b}) dof {d} \
                             but this rank does not own it", requester
                        ));
                    gid.to_le_bytes()
                })
                .collect();
            (*requester, reply_bytes)
        })
        .collect();

    let reply_received = comm.alltoallv_bytes(&replies);

    // Phase 3: decode replies into the original ghost-edge order.
    let mut result = vec![0u32; ghost_edges.len()];
    for (responder, bytes) in &reply_received {
        let gids: Vec<u32> = bytes.chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let request_indices = &requests_by_owner[responder];
        assert_eq!(gids.len(), request_indices.len());
        for (j, &(orig_idx, _, _, _)) in request_indices.iter().enumerate() {
            result[orig_idx] = gids[j];
        }
    }

    result
}

/// Exchange global DOF IDs for ghost face DOFs via alltoallv.
///
/// Each rank sends its ghost faces (identified by the 3-vertex face key and
/// the position within the face) to the owner rank, which looks up the global
/// DOF ID and replies.
fn exchange_ghost_face_ids(
    ghost_faces: &[FaceDofInfo],
    owned_face_global_map: &HashMap<(u32, u32, u32, u32), u32>,
    comm: &Comm,
) -> Vec<u32> {
    if comm.size() <= 1 || ghost_faces.is_empty() {
        return Vec::new();
    }

    let mut requests_by_owner: HashMap<Rank, Vec<(usize, u32, u32, u32, u32)>> = HashMap::new();
    for (i, f) in ghost_faces.iter().enumerate() {
        requests_by_owner.entry(f.owner).or_default().push((
            i,
            f.face_key.0,
            f.face_key.1,
            f.face_key.2,
            f.pos,
        ));
    }

    // Phase 1: send face requests (3-vertex key + pos) to owners.
    let sends: Vec<(Rank, Vec<u8>)> = requests_by_owner
        .iter()
        .map(|(&owner, entries)| {
            let bytes: Vec<u8> = entries
                .iter()
                .flat_map(|&(_, a, b, c, p)| {
                    let mut buf = [0u8; 16];
                    buf[..4].copy_from_slice(&a.to_le_bytes());
                    buf[4..8].copy_from_slice(&b.to_le_bytes());
                    buf[8..12].copy_from_slice(&c.to_le_bytes());
                    buf[12..].copy_from_slice(&p.to_le_bytes());
                    buf
                })
                .collect();
            (owner, bytes)
        })
        .collect();

    let received = comm.alltoallv_bytes(&sends);

    // Phase 2: owners look up global DOF IDs and reply.
    let replies: Vec<(Rank, Vec<u8>)> = received
        .iter()
        .map(|(requester, bytes)| {
            debug_assert_eq!(bytes.len() % 16, 0);
            let reply_bytes: Vec<u8> = bytes
                .chunks_exact(16)
                .flat_map(|chunk| {
                    let a = u32::from_le_bytes(chunk[..4].try_into().unwrap());
                    let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    let c = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                    let p = u32::from_le_bytes(chunk[12..].try_into().unwrap());
                    let gid = owned_face_global_map
                        .get(&(a, b, c, p))
                        .unwrap_or_else(|| {
                            panic!(
                                "exchange_ghost_face_ids: rank {} requested face ({a},{b},{c}) \
                                 pos {p} but this rank does not own it",
                                requester
                            )
                        });
                    gid.to_le_bytes()
                })
                .collect();
            (*requester, reply_bytes)
        })
        .collect();

    let reply_received = comm.alltoallv_bytes(&replies);

    // Phase 3: decode replies into the original ghost-face order.
    let mut result = vec![0u32; ghost_faces.len()];
    for (responder, bytes) in &reply_received {
        let gids: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let request_indices = &requests_by_owner[responder];
        assert_eq!(gids.len(), request_indices.len());
        for (j, &(orig_idx, _, _, _, _)) in request_indices.iter().enumerate() {
            result[orig_idx] = gids[j];
        }
    }

    result
}

/// Exchange global IDs for ghost interior DOFs.
///
/// Interior ghost DOFs are identified by (elem_gid, dof_idx) pairs. Each ghost
/// rank sends these pairs to the element's owning rank, which replies with the
/// corresponding global DOF ID.
fn exchange_ghost_interior_ids(
    ghost_interior: &[(u32, u32, u32, Rank)], // (dof_id, elem_gid, dof_idx, owner)
    owned_interior_map: &HashMap<(u32, u32), u32>,
    comm: &Comm,
) -> Vec<u32> {
    if comm.size() <= 1 || ghost_interior.is_empty() {
        return Vec::new();
    }

    // Group ghost interior DOFs by owner rank.
    let mut requests_by_owner: HashMap<Rank, Vec<(usize, u32, u32)>> = HashMap::new();
    for (i, &(_dof_id, elem_gid, dof_idx, owner)) in ghost_interior.iter().enumerate() {
        requests_by_owner.entry(owner).or_default()
            .push((i, elem_gid, dof_idx));
    }

    // Phase 1: send (elem_gid, dof_idx) pairs to owners.
    let sends: Vec<(Rank, Vec<u8>)> = requests_by_owner
        .iter()
        .map(|(&owner, entries)| {
            let bytes: Vec<u8> = entries.iter()
                .flat_map(|&(_, elem_gid, dof_idx)| {
                    let mut buf = [0u8; 8];
                    buf[..4].copy_from_slice(&elem_gid.to_le_bytes());
                    buf[4..].copy_from_slice(&dof_idx.to_le_bytes());
                    buf
                })
                .collect();
            (owner, bytes)
        })
        .collect();
    let received = comm.alltoallv_bytes(&sends);

    // Phase 2: owners look up global DOF IDs and reply.
    let replies: Vec<(Rank, Vec<u8>)> = received.iter()
        .map(|(requester, bytes)| {
            debug_assert_eq!(bytes.len() % 8, 0);
            let reply_bytes: Vec<u8> = bytes.chunks_exact(8)
                .flat_map(|chunk| {
                    let elem_gid = u32::from_le_bytes(chunk[..4].try_into().unwrap());
                    let dof_idx = u32::from_le_bytes(chunk[4..].try_into().unwrap());
                    let gid = owned_interior_map.get(&(elem_gid, dof_idx))
                        .copied()
                        .unwrap_or_else(|| {
                            eprintln!("  Warning: exchange_ghost_interior_ids: rank {requester} requested \
                                interior DOF (elem={elem_gid}, idx={dof_idx}) not found, \
                                using sentinel GID");
                            u32::MAX
                        });
                    gid.to_le_bytes()
                })
                .collect();
            (*requester, reply_bytes)
        })
        .collect();
    let reply_received = comm.alltoallv_bytes(&replies);

    // Phase 3: decode replies into the original order.
    let mut result = vec![0u32; ghost_interior.len()];
    for (responder, bytes) in &reply_received {
        let gids: Vec<u32> = bytes.chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let request_indices = &requests_by_owner[responder];
        assert_eq!(gids.len(), request_indices.len());
        for (j, &(orig_idx, _, _)) in request_indices.iter().enumerate() {
            result[orig_idx] = gids[j];
        }
    }

    result
}

// ── Prefix scan ─────────────────────────────────────────────────────────────

/// Exclusive prefix sum across MPI ranks.
///
/// Each rank contributes `local_val`; the result on rank `r` is the sum of
/// `local_val` from ranks `0, 1, ..., r-1`.  Rank 0 always gets 0.
fn exclusive_scan_i64(comm: &Comm, local_val: i64) -> i64 {
    let rank = comm.rank();
    let size = comm.size();

    if size <= 1 {
        return 0;
    }

    const TAG: i32 = 0x6000;

    if rank == 0 {
        let my_sum = local_val;
        comm.send_bytes(1, TAG, &my_sum.to_le_bytes());
        0
    } else {
        let prev_bytes = comm.recv_bytes(rank - 1, TAG);
        let prev_sum = i64::from_le_bytes(prev_bytes.try_into().unwrap());

        if (rank as usize) < size - 1 {
            let my_sum = prev_sum + local_val;
            comm.send_bytes(rank + 1, TAG, &my_sum.to_le_bytes());
        }

        prev_sum
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_partition::partition_mesh;
    use fem_mesh::Mesh;
    use fem_space::dof_manager::DofManager;
    use std::sync::{Arc, Mutex};

    #[test]
    fn dof_partition_p1_serial() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n_nodes = mesh.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let dof_part = DofPartition::from_mesh_partition(pmesh.partition(), &comm);

            assert_eq!(dof_part.n_owned_dofs, n_nodes);
            assert_eq!(dof_part.n_ghost_dofs, 0);
            assert_eq!(dof_part.global_dof_offset, 0);
            assert_eq!(dof_part.n_total_dofs(), n_nodes);
        });
    }

    #[test]
    fn dof_partition_p1_two_ranks() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let total_nodes = mesh.n_nodes();

        let results = Arc::new(Mutex::new(Vec::new()));

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        let results_clone = Arc::clone(&results);
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let dof_part = DofPartition::from_mesh_partition(pmesh.partition(), &comm);

            let rank = comm.rank();
            let n_owned = dof_part.n_owned_dofs;
            let offset = dof_part.global_dof_offset;

            results_clone.lock().unwrap().push((rank, n_owned, offset));

            let global_owned = comm.allreduce_sum_i64(n_owned as i64) as usize;
            assert_eq!(global_owned, total_nodes,
                "sum of owned DOFs ({global_owned}) != total nodes ({total_nodes})");

            for lid in 0..dof_part.n_owned_dofs as u32 {
                assert!(dof_part.is_owned_dof(lid));
                assert_eq!(dof_part.dof_owner(lid), rank);
            }

            for (lid, owner) in dof_part.ghost_dofs() {
                assert!(!dof_part.is_owned_dof(lid));
                assert_ne!(owner, rank);
            }
        });

        let mut res = results.lock().unwrap().clone();
        res.sort_by_key(|(r, _, _)| *r);
        assert_eq!(res[0].2, 0, "rank 0 offset must be 0");
        assert_eq!(res[1].2, res[0].1, "rank 1 offset must equal rank 0's n_owned");
    }

    #[test]
    fn dof_partition_p2_serial() {
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let dm = DofManager::new(pmesh.local_mesh(), 2);
            let dof_part = DofPartition::from_dof_manager(&dm, pmesh.partition(), &comm);

            assert_eq!(dof_part.n_owned_dofs, dm.n_dofs);
            assert_eq!(dof_part.n_ghost_dofs, 0);
            assert_eq!(dof_part.global_dof_offset, 0);
            assert_eq!(dof_part.n_total_dofs(), dm.n_dofs);
        });
    }

    #[test]
    fn dof_partition_p2_two_ranks() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let serial_dm = DofManager::new(&mesh, 2);
        let serial_n_dofs = serial_dm.n_dofs;

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let dm = DofManager::new(pmesh.local_mesh(), 2);
            let dof_part = DofPartition::from_dof_manager(&dm, pmesh.partition(), &comm);

            let rank = comm.rank();

            let global_owned = comm.allreduce_sum_i64(dof_part.n_owned_dofs as i64) as usize;
            assert_eq!(global_owned, serial_n_dofs,
                "rank {rank}: sum of owned P2 DOFs ({global_owned}) != serial ({serial_n_dofs})");

            for lid in 0..dof_part.n_owned_dofs as u32 {
                assert!(dof_part.is_owned_dof(lid));
                assert_eq!(dof_part.dof_owner(lid), rank);
            }

            for (lid, owner) in dof_part.ghost_dofs() {
                assert!(!dof_part.is_owned_dof(lid));
                assert_ne!(owner, rank);
            }

            // Verify all global DOF IDs are unique within this rank.
            let mut seen = std::collections::HashSet::new();
            for lid in 0..dof_part.n_total_dofs() as u32 {
                let gid = dof_part.global_dof(lid);
                assert!(seen.insert(gid),
                    "rank {rank}: duplicate global DOF ID {gid} at local {lid}");
            }
        });
    }

    #[test]
    fn exclusive_scan_four_ranks() {
        let results = Arc::new(Mutex::new(Vec::new()));

        let launcher = ThreadLauncher::new(WorkerConfig::new(4));
        let results_clone = Arc::clone(&results);
        launcher.launch(move |comm| {
            let rank = comm.rank();
            let val = (rank + 1) as i64;
            let scan = exclusive_scan_i64(&comm, val);
            results_clone.lock().unwrap().push((rank, scan));
        });

        let mut res = results.lock().unwrap().clone();
        res.sort_by_key(|(r, _)| *r);
        assert_eq!(res[0].1, 0);
        assert_eq!(res[1].1, 1);
        assert_eq!(res[2].1, 3);
        assert_eq!(res[3].1, 6);
    }

    /// Assemble the H(div) or H(curl) mass matrix on a partitioned 3-D hex
    /// mesh, apply `y = M x` with `x = gid` and return `(gid, y)` for all
    /// owned DOFs.  The returned map must be identical for every partition
    /// (np1 / np2 / np4) — this validates edge/face DOF ownership, the
    /// permutation and the sign corrections.
    #[test]
    fn hex_nd1_mass_consistent_across_partitions() {
        let mut mesh = Mesh::<3>::unit_cube_hex(2);
        for _ in 0..2 {
            mesh = fem_mesh::refine_uniform_3d(&mesh);
        }
        // Hex local edges (matching HCurlSpace HEX_EDGES).
        const HEX_EDGES: [(usize, usize); 12] = [
            (0, 1), (1, 2), (3, 2), (0, 3),
            (4, 5), (5, 6), (7, 6), (4, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];

        // Returns (edge-node-pair, y) for every owned dof after `y = M x`
        // with `x = 1` (constant field).  The edge pair uses global node ids,
        // which are partition-invariant (unlike the edge DOF gids).
        let run = |np: usize, mesh: Mesh<3>| -> Vec<((u32, u32), f64)> {
            let out: Arc<Mutex<Option<Vec<((u32, u32), f64)>>>> = Arc::new(Mutex::new(None));
            let out2 = Arc::clone(&out);
            let launcher = ThreadLauncher::new(WorkerConfig::new(np));
            launcher.launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let lm = pmesh.local_mesh().clone();
                let local_space = fem_space::HCurlSpace::new(lm, 1);
                let dp =
                    DofPartition::from_edge_space(&local_space, pmesh.partition(), &comm);
                let ps = crate::par_space::ParallelFESpace::new_with_dof_partition(
                    local_space,
                    dp,
                    comm.clone(),
                );

                use crate::par_vector::ParVector;
                use crate::par_vector_assembler::ParVectorAssembler;
                use fem_assembly::standard::VectorMassIntegrator;
                let mass = ParVectorAssembler::assemble_bilinear(
                    &ps,
                    &[&VectorMassIntegrator { alpha: 1.0 }],
                    3,
                );
                let mut x = ParVector::zeros(&ps);
                for pid in 0..ps.dof_partition().n_owned_dofs {
                    x.as_slice_mut()[pid] = 1.0;
                }
                x.update_ghosts();
                let mut y = ParVector::zeros(&ps);
                mass.spmv(&mut x, &mut y);

                // dof (dm id) -> canonical edge node pair (global ids).
                let mut dof_edge: std::collections::HashMap<u32, (u32, u32)> =
                    std::collections::HashMap::new();
                {
                    let sp = ps.local_space();
                    let mesh3 = sp.mesh();
                    for e in mesh3.elem_iter() {
                        let dofs = sp.element_dofs(e);
                        let nodes = mesh3.element_nodes(e);
                        for (i, &d) in dofs.iter().enumerate() {
                            let (li, lj) = HEX_EDGES[i];
                            let (a, b) = (
                                pmesh.partition().global_node(nodes[li]),
                                pmesh.partition().global_node(nodes[lj]),
                            );
                            dof_edge.entry(d as u32).or_insert((a.min(b), a.max(b)));
                        }
                    }
                }
                let dp = ps.dof_partition();
                let owned: Vec<((u32, u32), f64)> = (0..dp.n_owned_dofs)
                    .map(|pid| {
                        let dm = dp.unpermute_dof(pid as u32);
                        let key = *dof_edge
                            .get(&dm)
                            .expect("nd1 test: dof not in edge map");
                        (key, y.as_slice()[pid])
                    })
                    .collect();
                if comm.rank() == 0 {
                    let mut all = owned.clone();
                    for src in 1..comm.size() as i32 {
                        let flat: Vec<u32> = comm.recv(src, 201);
                        let keys: Vec<(u32, u32)> = flat
                            .chunks_exact(2)
                            .map(|c| (c[0], c[1]))
                            .collect();
                        let vals: Vec<f64> = comm.recv(src, 202);
                        all.extend(keys.into_iter().zip(vals));
                    }
                    all.sort_unstable_by_key(|&(k, _)| k);
                    *out2.lock().unwrap() = Some(all);
                } else {
                    let keys: Vec<(u32, u32)> = owned.iter().map(|&(k, _)| k).collect();
                    let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                    // Tuples are not Pod: pack into flat u32 pairs.
                    let flat: Vec<u32> = keys.iter().flat_map(|&(a, b)| [a, b]).collect();
                    comm.send(0, 201, &flat);
                    comm.send(0, 202, &vals);
                }
            });
            let mut guard = out.lock().unwrap();
            guard.take().unwrap()
        };

        let np1 = run(1, mesh.clone());
        let np2 = run(2, mesh.clone());
        let np4 = run(4, mesh.clone());

        assert_eq!(np1.len(), np2.len(), "nd1: np1/np2 global dof count mismatch");
        assert_eq!(np1.len(), np4.len(), "nd1: np1/np4 global dof count mismatch");
        let mut max_diff = 0.0_f64;
        for ((k1, y1), (k2, y2)) in np1.iter().zip(np2.iter()) {
            assert_eq!(k1, k2, "nd1: edge-key order mismatch np1 vs np2");
            max_diff = max_diff.max((y1 - y2).abs());
        }
        assert!(
            max_diff < 1e-10,
            "hex ND1 mass differs across partitions (np1 vs np2): max_diff={max_diff:e}"
        );
        let mut max_diff4 = 0.0_f64;
        for ((k1, y1), (k4, y4)) in np1.iter().zip(np4.iter()) {
            assert_eq!(k1, k4, "nd1: edge-key order mismatch np1 vs np4");
            max_diff4 = max_diff4.max((y1 - y4).abs());
        }
        assert!(
            max_diff4 < 1e-10,
            "hex ND1 mass differs across partitions (np1 vs np4): max_diff={max_diff4:e}"
        );
    }

    #[test]
    fn hex_rt0_mass_consistent_across_partitions() {
        let mut mesh = Mesh::<3>::unit_cube_hex(2);
        for _ in 0..2 {
            mesh = fem_mesh::refine_uniform_3d(&mesh);
        }

        // Returns (face-3-vertex-key, y) for every owned dof after `y = M x`
        // with `x = 1`.  The face key uses global vertex ids (partition
        // invariant), unlike the face DOF gids.
        let run = |np: usize, mesh: Mesh<3>| -> Vec<((u32, u32, u32), f64)> {
            let out: Arc<Mutex<Option<Vec<((u32, u32, u32), f64)>>>> = Arc::new(Mutex::new(None));
            let out2 = Arc::clone(&out);
            let launcher = ThreadLauncher::new(WorkerConfig::new(np));
            launcher.launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let lm = pmesh.local_mesh().clone();
                let local_space = fem_space::HDivSpace::new(lm, 0);
                let dp =
                    DofPartition::from_face_space(&local_space, pmesh.partition(), &comm);
                let ps = crate::par_space::ParallelFESpace::new_with_dof_partition(
                    local_space,
                    dp,
                    comm.clone(),
                );

                use crate::par_vector::ParVector;
                use crate::par_vector_assembler::ParVectorAssembler;
                use fem_assembly::standard::VectorMassIntegrator;
                let mass = ParVectorAssembler::assemble_bilinear(
                    &ps,
                    &[&VectorMassIntegrator { alpha: 1.0 }],
                    3,
                );
                let mut x = ParVector::zeros(&ps);
                for pid in 0..ps.dof_partition().n_owned_dofs {
                    x.as_slice_mut()[pid] = 1.0;
                }
                x.update_ghosts();
                let mut y = ParVector::zeros(&ps);
                mass.spmv(&mut x, &mut y);

                // dof (dm id) -> canonical face key (3 smallest global ids).
                // Hex local faces (matching HDivSpace HEX_FACES).
                const HEX_FACES: [[usize; 4]; 6] = [
                    [3, 2, 1, 0],
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7],
                    [4, 5, 6, 7],
                ];
                let mut dof_face: std::collections::HashMap<u32, (u32, u32, u32)> =
                    std::collections::HashMap::new();
                {
                    let sp = ps.local_space();
                    let mesh3 = sp.mesh();
                    for e in mesh3.elem_iter() {
                        let dofs = sp.element_dofs(e);
                        let nodes = mesh3.element_nodes(e);
                        for (i, &d) in dofs.iter().enumerate() {
                            if i >= HEX_FACES.len() {
                                break;
                            }
                            let fv = HEX_FACES[i];
                            let mut v4: Vec<u32> = fv
                                .iter()
                                .map(|&li| pmesh.partition().global_node(nodes[li]))
                                .collect();
                            v4.sort_unstable();
                            dof_face
                                .entry(d as u32)
                                .or_insert((v4[0], v4[1], v4[2]));
                        }
                    }
                }
                let dp = ps.dof_partition();
                let owned: Vec<((u32, u32, u32), f64)> = (0..dp.n_owned_dofs)
                    .map(|pid| {
                        let dm = dp.unpermute_dof(pid as u32);
                        let key = *dof_face
                            .get(&dm)
                            .expect("rt0 test: dof not in face map");
                        (key, y.as_slice()[pid])
                    })
                    .collect();
                if comm.rank() == 0 {
                    let mut all = owned.clone();
                    for src in 1..comm.size() as i32 {
                        let flat: Vec<u32> = comm.recv(src, 201);
                        let keys: Vec<(u32, u32, u32)> = flat
                            .chunks_exact(3)
                            .map(|c| (c[0], c[1], c[2]))
                            .collect();
                        let vals: Vec<f64> = comm.recv(src, 202);
                        all.extend(keys.into_iter().zip(vals));
                    }
                    all.sort_unstable_by_key(|&(k, _)| k);
                    *out2.lock().unwrap() = Some(all);
                } else {
                    let keys: Vec<(u32, u32, u32)> = owned.iter().map(|&(k, _)| k).collect();
                    let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                    let flat: Vec<u32> = keys.iter().flat_map(|&(a, b, c)| [a, b, c]).collect();
                    comm.send(0, 201, &flat);
                    comm.send(0, 202, &vals);
                }
            });
            let mut guard = out.lock().unwrap();
            guard.take().unwrap()
        };

        let np1 = run(1, mesh.clone());
        let np2 = run(2, mesh.clone());
        let np4 = run(4, mesh.clone());

        assert_eq!(np1.len(), np2.len(), "rt0: np1/np2 global dof count mismatch");
        assert_eq!(np1.len(), np4.len(), "rt0: np1/np4 global dof count mismatch");
        let mut max_diff = 0.0_f64;
        for ((k1, y1), (k2, y2)) in np1.iter().zip(np2.iter()) {
            assert_eq!(k1, k2, "rt0: face-key order mismatch np1 vs np2");
            max_diff = max_diff.max((y1 - y2).abs());
        }
        assert!(
            max_diff < 1e-10,
            "hex RT0 mass differs across partitions (np1 vs np2): max_diff={max_diff:e}"
        );
        let mut max_diff4 = 0.0_f64;
        for ((k1, y1), (k4, y4)) in np1.iter().zip(np4.iter()) {
            assert_eq!(k1, k4, "rt0: face-key order mismatch np1 vs np4");
            max_diff4 = max_diff4.max((y1 - y4).abs());
        }
        assert!(
            max_diff4 < 1e-10,
            "hex RT0 mass differs across partitions (np1 vs np4): max_diff={max_diff4:e}"
        );
    }
}
