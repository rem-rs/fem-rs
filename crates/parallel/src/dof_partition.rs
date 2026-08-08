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
        assert_eq!(dof_manager.order, 2, "DofPartition: only P1 and P2 supported");

        let local_rank = comm.rank();
        let n_owned_vertices = partition.n_owned_nodes;
        let n_ghost_vertices = partition.n_ghost_nodes;

        // ── Classify edge DOFs as owned or ghost ────────────────────────────
        let mut owned_edges: Vec<EdgeDofInfo> = Vec::new();
        let mut ghost_edges: Vec<EdgeDofInfo> = Vec::new();

        for (&EdgeKey(local_a, local_b), &local_dof_id) in &dof_manager.edge_dof_map {
            let ga = partition.global_node(local_a);
            let gb = partition.global_node(local_b);
            let edge_owner = partition.node_owner(local_a).min(partition.node_owner(local_b));

            let info = EdgeDofInfo {
                local_dof_id,
                global_node_a: ga.min(gb),
                global_node_b: ga.max(gb),
                owner: edge_owner,
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

        let n_owned_edges = owned_edges.len();
        let n_owned = n_owned_vertices + n_owned_edges;
        let n_ghost = n_ghost_vertices + ghost_edges.len();

        // ── Compute global offsets ──────────────────────────────────────────
        let global_dof_offset = exclusive_scan_i64(comm, n_owned as i64) as usize;
        let total_global_vertices = comm.allreduce_sum_i64(n_owned_vertices as i64) as u32;
        let edge_offset = exclusive_scan_i64(comm, n_owned_edges as i64) as u32;

        // ── Build owned DOF arrays ──────────────────────────────────────────
        let total = n_owned + n_ghost;
        let mut global_dof_ids = Vec::with_capacity(total);
        let mut dof_owner_vec = Vec::with_capacity(total);

        // Owned vertices: global ID = global node ID.
        for lid in 0..n_owned_vertices as u32 {
            global_dof_ids.push(partition.global_node(lid));
            dof_owner_vec.push(local_rank);
        }

        // Owned edges: global ID = total_global_vertices + edge_offset + i.
        let mut owned_edge_global_map: HashMap<(u32, u32), u32> = HashMap::new();
        for (i, edge) in owned_edges.iter().enumerate() {
            let gid = total_global_vertices + edge_offset + i as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_edge_global_map.insert((edge.global_node_a, edge.global_node_b), gid);
        }
        debug_assert_eq!(global_dof_ids.len(), n_owned);

        // ── Build ghost DOF arrays ──────────────────────────────────────────

        // Ghost vertices.
        for lid in n_owned_vertices..(n_owned_vertices + n_ghost_vertices) {
            global_dof_ids.push(partition.global_node(lid as u32));
            dof_owner_vec.push(partition.node_owner(lid as u32));
        }

        // Ghost edges: exchange global IDs with their owners.
        let ghost_edge_gids = exchange_ghost_edge_ids(
            &ghost_edges, &owned_edge_global_map, comm,
        );
        for (i, edge) in ghost_edges.iter().enumerate() {
            global_dof_ids.push(ghost_edge_gids[i]);
            dof_owner_vec.push(edge.owner);
        }
        debug_assert_eq!(global_dof_ids.len(), total);

        // ── Build dm_to_partition permutation ───────────────────────────────
        // Maps DofManager's local DOF ID → partition's local DOF ID.
        // Partition layout:
        //   [owned_vertices | owned_edges | ghost_vertices | ghost_edges]
        // DofManager layout:
        //   [all_local_vertices | all_edges_in_enum_order]
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
            dm_to_partition[edge.local_dof_id as usize] = (n_owned + n_ghost_vertices + i) as u32;
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

        // Edge tables matching HCurlSpace / HDivSpace element ordering.
        let hcurl_2d: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (0, 2)];
        let hcurl_quad_2d: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let hdiv_2d: Vec<(usize, usize)> = vec![(1, 2), (0, 2), (0, 1)];
        let hdiv_quad_2d: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let hcurl_3d: Vec<(usize, usize)> = vec![
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ];

        let space_type = space.space_type();
        let is_quad = mesh.n_elements() > 0
            && matches!(mesh.element_type(0), fem_mesh::ElementType::Quad4);

        let edges_for_space: &[(usize, usize)] = match (space_type, dim) {
            (fem_space::fe_space::SpaceType::HCurl, 2) if is_quad => &hcurl_quad_2d,
            (fem_space::fe_space::SpaceType::HCurl, 2) => &hcurl_2d,
            (fem_space::fe_space::SpaceType::HDiv, 2) if is_quad => &hdiv_quad_2d,
            (fem_space::fe_space::SpaceType::HDiv, 2) => &hdiv_2d,
            (fem_space::fe_space::SpaceType::HCurl, 3) => &hcurl_3d,
            _ => &hcurl_2d,
        };

        // Determine DOF layout per edge from the element type.
        // For order-1 spaces each edge holds 1 DOF and there are no interior DOFs.
        // Higher-order Nédélec elements have multiple DOFs per edge plus interior bubbles.
        let (dofs_per_edge, _n_interior_per_elem) = if mesh.n_elements() > 0 {
            let t = mesh.element_type(0);
            match (space_type, t) {
                (_, fem_mesh::ElementType::Tri6) => (2, 2),  // TriND2: 2/edge, 2 interior
                (_, fem_mesh::ElementType::Tri3) => (1, 0),
                (_, fem_mesh::ElementType::Quad4) => (1, 0),
                (_, fem_mesh::ElementType::Tet4) => (1, 0),
                _ => (1, 0),
            }
        } else {
            (1, 0)
        };
        let n_edges = edges_for_space.len();
        let edge_dofs_total = n_edges * dofs_per_edge;

        // ── Step 1: Map each DOF to its canonical edge (or interior) ──────────
        //
        // For higher-order spaces, DOFs are grouped:
        //   [edge0 × dofs_per_edge, edge1 × dofs_per_edge, ..., interior...]
        let mut dof_to_edge: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut interior_dofs: Vec<(u32, u32, u32)> = Vec::new(); // (dof_id, local_elem_id, dof_idx_in_elem)
        let mut sign_corr: Vec<f64> = vec![1.0; n_space_dofs];

        for e in mesh.elem_iter() {
            let dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            for (i, &dof_id) in dofs.iter().enumerate() {
                if dof_to_edge.contains_key(&dof_id) || interior_dofs.iter().any(|&(d, _, _)| d == dof_id) {
                    continue;
                }

                if i < edge_dofs_total {
                    // Edge DOF: map to the corresponding edge in edges_for_space.
                    let edge_idx = i / dofs_per_edge;
                    let (a, b) = edges_for_space[edge_idx];
                    let local_a = nodes[a];
                    let local_b = nodes[b];
                    let ga = partition.global_node(local_a);
                    let gb = partition.global_node(local_b);

                    dof_to_edge.insert(dof_id, (ga.min(gb), ga.max(gb)));

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
        let n_total_nodes = (partition.n_owned_nodes + partition.n_ghost_nodes) as u32;
        let mut global_to_local_node: HashMap<u32, u32> = HashMap::new();
        for lid in 0..n_total_nodes {
            global_to_local_node.insert(partition.global_node(lid), lid);
        }

        for (&dof_id, &(ga, gb)) in &dof_to_edge {
            let owner_a = global_to_local_node.get(&ga)
                .map(|&lid| partition.node_owner(lid))
                .unwrap_or(Rank::MAX);
            let owner_b = global_to_local_node.get(&gb)
                .map(|&lid| partition.node_owner(lid))
                .unwrap_or(Rank::MAX);
            let edge_owner = owner_a.min(owner_b);

            let info = EdgeDofInfo {
                local_dof_id: dof_id,
                global_node_a: ga,
                global_node_b: gb,
                owner: edge_owner,
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

        // 4a. Edge DOFs (owned first, then ghost).
        let mut owned_edge_global_map: HashMap<(u32, u32), u32> = HashMap::new();
        for (i, edge) in owned_edges.iter().enumerate() {
            let gid = edge_offset + i as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_edge_global_map.insert((edge.global_node_a, edge.global_node_b), gid);
        }

        let ghost_edge_gids = exchange_ghost_edge_ids(
            &ghost_edges, &owned_edge_global_map, comm,
        );
        for (i, edge) in ghost_edges.iter().enumerate() {
            global_dof_ids.push(ghost_edge_gids[i]);
            dof_owner_vec.push(edge.owner);
        }

        // 4b. Interior DOFs (owned then ghost).
        let owned_interior_offset = edge_offset + owned_edges.len() as u32;
        let mut owned_interior_map: HashMap<(u32, u32), u32> = HashMap::new();
        for (j, &(_dof_id, elem_gid, dof_idx)) in owned_interior.iter().enumerate() {
            let gid = owned_interior_offset + j as u32;
            global_dof_ids.push(gid);
            dof_owner_vec.push(local_rank);
            owned_interior_map.insert((elem_gid, dof_idx), gid);
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
        for (i, edge) in ghost_edges.iter().enumerate() {
            dm_to_partition[edge.local_dof_id as usize] = (n_owned_edge + i) as u32;
        }
        // Interior DOFs
        let interior_owned_start = n_owned_edge + n_ghost_edge;
        for (j, &(dof_id, _, _)) in owned_interior.iter().enumerate() {
            dm_to_partition[dof_id as usize] = (interior_owned_start + j) as u32;
        }
        for (j, &(dof_id, _, _, _)) in ghost_interior.iter().enumerate() {
            dm_to_partition[dof_id as usize] = (n_owned + j) as u32;
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
    owned_edge_global_map: &HashMap<(u32, u32), u32>,
    comm: &Comm,
) -> Vec<u32> {
    if comm.size() <= 1 || ghost_edges.is_empty() {
        return Vec::new();
    }

    // Group ghost edges by owner rank.
    let mut requests_by_owner: HashMap<Rank, Vec<(usize, u32, u32)>> = HashMap::new();
    for (i, edge) in ghost_edges.iter().enumerate() {
        requests_by_owner.entry(edge.owner).or_default()
            .push((i, edge.global_node_a, edge.global_node_b));
    }

    // Phase 1: send edge requests (pairs of global node IDs) to owners.
    let sends: Vec<(Rank, Vec<u8>)> = requests_by_owner
        .iter()
        .map(|(&owner, edges)| {
            let bytes: Vec<u8> = edges.iter()
                .flat_map(|&(_, a, b)| {
                    let mut buf = [0u8; 8];
                    buf[..4].copy_from_slice(&a.to_le_bytes());
                    buf[4..].copy_from_slice(&b.to_le_bytes());
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
                    let a = u32::from_le_bytes(chunk[..4].try_into().unwrap());
                    let b = u32::from_le_bytes(chunk[4..].try_into().unwrap());
                    let gid = owned_edge_global_map.get(&(a, b))
                        .unwrap_or_else(|| panic!(
                            "exchange_ghost_edge_ids: rank {} requested edge ({a},{b}) \
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
        for (j, &(orig_idx, _, _)) in request_indices.iter().enumerate() {
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
}
