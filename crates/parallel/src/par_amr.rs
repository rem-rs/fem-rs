//! Parallel adaptive mesh refinement.
//!
//! Provides [`par_refine_marked`] for distributed non-conforming refinement and
//! [`par_repartition`] for load-rebalancing after refinement.

use std::collections::{HashMap, BTreeMap, BTreeSet, HashSet};

use fem_mesh::{Mesh, amr::NCState, amr::DerefineTree, amr::DerefineRecord, amr::derefine_marked, topology::MeshTopology, boundary::BoundaryTag};
use fem_core::types::{ElemId, NodeId, Rank};

use crate::{
    par_mesh::ParallelMesh,
    partition::MeshPartition,
    ghost::GhostExchange,
    mesh_serde::{encode_submesh, decode_submesh},
    par_partition::{partition_mesh_streaming, STREAM_TAG_BASE},
    Comm,
};

const REPART_TAG_BASE: i32 = 0x3800;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors from parallel AMR operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ParAmrError {
    RefinementError(String),
    RepartitionError(String),
    SerializationError(String),
}

impl std::fmt::Display for ParAmrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RefinementError(s) => write!(f, "Refinement error: {s}"),
            Self::RepartitionError(s) => write!(f, "Repartition error: {s}"),
            Self::SerializationError(s) => write!(f, "Serialization error: {s}"),
        }
    }
}

// ─── par_refine_marked ────────────────────────────────────────────────────────

/// Result of a single parallel AMR cycle.
pub struct ParRefinedMesh {
    pub par_mesh:    ParallelMesh<Mesh<2>>,
    pub nc_state:    NCState,
    pub solution:    Vec<f64>,
    pub n_new_elems: usize,
}

/// Perform one cycle of parallel non-conforming AMR.
///
/// `marked` are **local** element ids (owned + ghost) to refine; callers must
/// ensure the marks are globally consistent (every rank marks the same set of
/// global elements — see [`gather_marked_set`]).  The refined mesh's parallel
/// partition is rebuilt so that global element/node ids match the *serial*
/// NC-refinement sequence of the full mesh, children inherit the parent's
/// owner, and the ghost layer covers all cross-rank neighbours.
///
/// # Note on `nc_state`
/// The argument is accepted for signature compatibility but **ignored**: the
/// partition rebuild renumbers local node ids, which invalidates any
/// carried-over `NCState` (`active_midpoints`/`edge_level` keyed by local
/// ids).  Each call starts from a fresh state; midpoint reuse across rounds
/// is handled by the coordinate-midpoint fallback in
/// [`NCState::refine`](fem_mesh::amr::NCState::refine).  For round-trip
/// derefinement use [`par_refine_marked_with_tree`].
///
/// To later derefine, use [`par_refine_marked_with_tree`] which returns a
/// [`DerefineTree`] suitable for [`par_derefine_marked`].
pub fn par_refine_marked(
    par_mesh: &ParallelMesh<Mesh<2>>,
    _nc_state: NCState,
    marked:    &[ElemId],
    solution:  Option<&[f64]>,
) -> Result<ParRefinedMesh, ParAmrError> {
    let coarse_mesh = par_mesh.local_mesh().clone();
    let comm       = par_mesh.comm().clone();
    let rank = comm.rank();
    // Round-boundary synchronization: keeps every rank in the same collective
    // phase across AMR rounds (the channel backend's alltoallv/allreduce are
    // separate rendezvous sets, so a rank that runs ahead of the others could
    // otherwise wait on a different collective than the one being called).
    comm.barrier();

    // Fresh state per round: see the note above.
    let mut nc_state = NCState::new();
    let (refined_mesh, _constraints, midpoint_map) = nc_state.refine(&coarse_mesh, marked, 0);
    let n_new_elems = refined_mesh.n_elements();

    let prolongated = if let Some(sol) = solution {
        prolongate_p1(&coarse_mesh, &refined_mesh, sol)
    } else {
        vec![]
    };

    let (refined_mesh, new_partition, remap) = rebuild_partition_nc(
        &refined_mesh, par_mesh, marked, &midpoint_map, &comm,
    );
    // Reorder the prolongated solution to match the reordered node ids.
    let prolongated = reorder_solution(&prolongated, &remap);
    let new_par_mesh = ParallelMesh::new(refined_mesh, comm, new_partition);

    Ok(ParRefinedMesh {
        par_mesh: new_par_mesh,
        nc_state,
        solution: prolongated,
        n_new_elems,
    })
}

/// Parallel non-conforming AMR that also returns a [`DerefineTree`] for later
/// coarsening via [`par_derefine_marked`].
///
/// This is the counterpart of [`par_refine_marked`] that preserves the
/// parent→children provenance required for parallel derefinement.
pub fn par_refine_marked_with_tree(
    par_mesh: &ParallelMesh<Mesh<2>>,
    mut nc_state: NCState,
    marked:    &[ElemId],
    solution:  Option<&[f64]>,
) -> Result<(ParRefinedMesh, DerefineTree), ParAmrError> {
    let coarse_mesh = par_mesh.local_mesh().clone();
    let comm       = par_mesh.comm().clone();

    let (refined_mesh, _constraints, midpoint_map) = nc_state.refine(&coarse_mesh, marked, 0);
    let n_new_elems = refined_mesh.n_elements();

    let prolongated = if let Some(sol) = solution {
        prolongate_p1(&coarse_mesh, &refined_mesh, sol)
    } else {
        vec![]
    };

    let (refined_mesh, new_partition, remap) = rebuild_partition_nc(
        &refined_mesh, par_mesh, marked, &midpoint_map, &comm,
    );
    // Reorder the prolongated solution to match the reordered node ids.
    let prolongated = reorder_solution(&prolongated, &remap);
    let new_par_mesh = ParallelMesh::new(refined_mesh, comm, new_partition);

    let tree = build_derefine_tree_from_refine(&coarse_mesh, marked, &midpoint_map);

    Ok((ParRefinedMesh {
        par_mesh: new_par_mesh,
        nc_state,
        solution: prolongated,
        n_new_elems,
    }, tree))
}

// ─── cross-rank partition rebuild after NC refinement ───────────────────────

/// Tag base for the marked-set broadcast (0x39xx range is free).
const NC_AMR_MARK_TAG: i32 = 0x3910;

/// Allgather the globally-marked element set: each rank contributes the
/// global ids of the owned elements it marks; the merged set is replicated.
///
/// Uses `alltoallv` (a collective with a global rendezvous) so that every
/// rank crosses this communication point together — point-to-point root
/// collection would race across AMG-refinement rounds when thread scheduling
/// lets one rank run ahead of another.
fn gather_marked_set(comm: &Comm, owned_marked: &[u32]) -> BTreeSet<u32> {
    let mut set: BTreeSet<u32> = owned_marked.iter().copied().collect();
    if comm.size() > 1 {
        let mut payload = Vec::with_capacity(owned_marked.len() * 4);
        for g in owned_marked {
            payload.extend_from_slice(&g.to_le_bytes());
        }
        let sends: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
            .map(|r| (r, payload.clone()))
            .collect();
        for (_src, bytes) in comm.alltoallv_bytes(&sends) {
            for chunk in bytes.chunks_exact(4) {
                set.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
    }
    set
}

/// Gather one `i64` per rank into a vector replicated on every rank.
/// Collective (alltoallv) — see [`gather_marked_set`] for why.
fn gather_counts(comm: &Comm, my_count: i64) -> Vec<i64> {
    let n = comm.size() as i32;
    if n <= 1 {
        return vec![my_count];
    }
    let payload = my_count.to_le_bytes().to_vec();
    let sends: Vec<(Rank, Vec<u8>)> = (0..n).map(|r| (r, payload.clone())).collect();
    let recv = comm.alltoallv_bytes(&sends);
    let mut all = vec![0i64; n as usize];
    for (src, bytes) in recv {
        all[src as usize] = i64::from_le_bytes(bytes[..8].try_into().unwrap());
    }
    all
}


/// Rebuild the cross-rank [`MeshPartition`] after a local NC refinement.
///
/// Returns `(reordered_mesh, new_partition, remap)`: the reordered mesh keeps
/// local node ids in the `[owned | ghost]` segment order the partition
/// expects; `remap[new_id] = old_id` maps reordered ids back to the input
/// (NCState output) node numbering.
///
/// `refined` is the locally refined mesh produced by `NCState::refine`
/// (element order = coarse element order expanded 1 → 4 for marked elements,
/// owned-first).  `marked_local` are the local element ids that were refined
/// (owned + ghost; globally consistent).  `midpoint_map` maps local parent
/// edge `(a, b)` → local midpoint id.
///
/// Global ids match the *serial* NC refinement of the full mesh:
/// * element `g` (global) → `prefix[g] + k` (`k` in 0..4 if refined);
/// * edge midpoints are assigned in global-marked-element order with edge
///   order `(0,1), (1,2), (2,0)` and first-touch wins — identical on every
///   rank; the coordinator (owner of the smallest marked element referencing
///   the edge) answers cross-rank requests.
///
/// Children inherit the parent's owner; a midpoint is owned by the owner of
/// its smallest referencing marked element.
fn rebuild_partition_nc(
    refined: &Mesh<2>,
    par_mesh: &ParallelMesh<Mesh<2>>,
    marked_local: &[ElemId],
    midpoint_map: &HashMap<(NodeId, NodeId), NodeId>,
    comm: &Comm,
) -> (Mesh<2>, MeshPartition, Vec<u32>) {
    let rank = comm.rank();
    let partition = par_mesh.partition();
    let local_mesh = par_mesh.local_mesh();
    let rank = comm.rank();
    let n_owned_elems = partition.n_owned_elems;
    let n_ghost_elems = partition.n_ghost_elems;
    let n_local_elems = n_owned_elems + n_ghost_elems;
    let n_orig = local_mesh.n_nodes();
    let n_global_elems = par_mesh.global_n_elems();
    let n_global_old_nodes = par_mesh.global_n_nodes();
    let gid_of = |local: u32| partition.global_node(local);

    // 1. Global marked set (every rank contributes its owned marks).
    let mut owned_marked: Vec<u32> = marked_local
        .iter()
        .filter(|&&e| partition.elem_owner[e as usize] == rank)
        .map(|&e| partition.global_elem(e))
        .collect();
    owned_marked.sort_unstable();
    let global_marked = gather_marked_set(comm, &owned_marked);

    // 2. Serial NC-refinement sequence positions.
    let mut prefix = vec![0usize; n_global_elems + 1];
    for g in 0..n_global_elems {
        prefix[g + 1] =
            prefix[g] + if global_marked.contains(&(g as u32)) { 4 } else { 1 };
    }

    // 3. New element gids (local refined order = coarse order expanded).
    let n_refined = refined.n_elems();
    let mut new_elem_gid = vec![0u32; n_refined];
    let mut new_elem_owner = vec![0i32; n_refined];
    let mut expand = 0usize;
    for e in 0..n_local_elems {
        let g = partition.global_elem(e as u32) as usize;
        let cnt = if global_marked.contains(&(g as u32)) { 4 } else { 1 };
        for k in 0..cnt {
            new_elem_gid[expand + k] = (prefix[g] + k) as u32;
            new_elem_owner[expand + k] = partition.elem_owner[e];
        }
        expand += cnt;
    }
    debug_assert_eq!(expand, n_refined, "element count drift in NC refine");

    // 4. Edge midpoint gids.  Only *new* midpoints (mid >= n_orig) get fresh
    //    global ids; edges whose midpoint already exists from a previous
    //    round keep the id the old-node segment inherited from `partition`.
    //
    //    Each edge's global id is assigned by the owner of its smaller
    //    endpoint node — a globally unique, locally computable rank (every
    //    rank touching the edge sees both endpoints locally).  Ranks that
    //    see an edge request its id from the assigner; the assigner dedups,
    //    numbers its edges in sorted order within a prefix-summed range, and
    //    replies.  (A min-marked-element-based assigner would be unusable:
    //    the owner of the smallest marked element referencing an edge may
    //    not see the edge locally at all.)
    let new_mid_edges: HashSet<(NodeId, NodeId)> = midpoint_map
        .iter()
        .filter(|(_, &mid)| (mid as usize) >= n_orig)
        .map(|(&(a, b), _)| edge_key(a, b))
        .collect();
    let mut local_new_edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    // Local (edge → smallest marked element referencing it + its owner).
    // The owner is used for the midpoint *ownership* below; the assigner of
    // the edge id is the owner of the smaller endpoint (computed separately).
    let mut local_edge_ref: BTreeMap<(u32, u32), (ElemId, Rank)> = BTreeMap::new();
    for &e in marked_local {
        let ns = local_mesh.elem_nodes(e);
        let ge = partition.global_elem(e);
        let owner = partition.elem_owner[e as usize];
        for &(a, b) in &[(0usize, 1usize), (1usize, 2usize), (2usize, 0usize)] {
            if !new_mid_edges.contains(&edge_key(ns[a], ns[b])) {
                continue; // old midpoint (created in a previous round)
            }
            let ek = edge_key(gid_of(ns[a]), gid_of(ns[b]));
            local_new_edges.insert(ek);
            local_edge_ref
                .entry(ek)
                .and_modify(|(m, o)| {
                    if ge < *m {
                        *m = ge;
                        *o = owner;
                    }
                })
                .or_insert((ge, owner));
        }
    }

    // Global min-ref per edge (with owner), reduced via alltoallv.  A rank
    // may not see every element referencing an edge it touches (the ghost
    // layer only covers cross-rank *owned* neighbours), so the minimum must
    // be reduced across ranks.
    let mut payload = Vec::with_capacity(local_edge_ref.len() * 16);
    for (&(a, b), &(m, o)) in &local_edge_ref {
        payload.extend_from_slice(&a.to_le_bytes());
        payload.extend_from_slice(&b.to_le_bytes());
        payload.extend_from_slice(&m.to_le_bytes());
        payload.extend_from_slice(&(o as u32).to_le_bytes());
    }
    let sends_all: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
        .map(|r| (r, payload.clone()))
        .collect();
    let mut edge_ref_global: BTreeMap<(u32, u32), (ElemId, Rank)> = local_edge_ref.clone();
    if comm.size() > 1 {
        for (_src, bytes) in comm.alltoallv_bytes(&sends_all) {
            for chunk in bytes.chunks_exact(16) {
                let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let m = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                let o = i32::from_le_bytes(chunk[12..16].try_into().unwrap());
                let entry = edge_ref_global.entry((a, b)).or_insert((m, o));
                if m < entry.0 {
                    *entry = (m, o);
                }
            }
        }
    }

    // Group local edges by assigner rank (owner of the smaller endpoint).
    let mut req_map: BTreeMap<Rank, Vec<(u32, u32)>> = BTreeMap::new();
    for &ek in &local_new_edges {
        let min_n = ek.0.min(ek.1);
        let min_local = partition
            .local_node(min_n)
            .expect("rebuild_partition_nc: edge endpoint not present locally");
        let owner = partition.node_owner[min_local as usize];
        req_map.entry(owner).or_default().push(ek);
    }
    for v in req_map.values_mut() {
        v.sort_unstable();
    }

    // Send requests to assigners (including ourselves).  The single-rank
    // backend's alltoallv returns nothing, so bypass it there.
    let mut edge_gid: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    if comm.size() == 1 {
        for (idx, &ek) in local_new_edges.iter().enumerate() {
            edge_gid.insert(ek, (n_global_old_nodes + idx) as u32);
        }
    } else {
    let sends: Vec<(Rank, Vec<u8>)> = req_map
        .iter()
        .map(|(&dest, keys)| {
            let mut b = Vec::with_capacity(keys.len() * 8);
            for &(a, bb) in keys {
                b.extend_from_slice(&a.to_le_bytes());
                b.extend_from_slice(&bb.to_le_bytes());
            }
            (dest, b)
        })
        .collect();
    let incoming = comm.alltoallv_bytes(&sends);

    // Edges we assign (dedup, sorted) + per-requester edge order for replies.
    let mut my_edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut req_by_requester: BTreeMap<Rank, Vec<(u32, u32)>> = BTreeMap::new();
    for (requester, bytes) in &incoming {
        let mut keys = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
            keys.push((a, b));
        }
        req_by_requester.insert(*requester, keys.clone());
        my_edges.extend(keys);
    }

    // Prefix-summed id range for our edges.
    let all_counts = gather_counts(comm, my_edges.len() as i64);
    let base: usize = all_counts
        .iter()
        .take(rank as usize)
        .map(|&c| c as usize)
        .sum();

    // Assign ids in sorted edge order.
    for (idx, &ek) in my_edges.iter().enumerate() {
        edge_gid.insert(ek, (n_global_old_nodes + base + idx) as u32);
    }

    // Reply to every requester (same order as the request).
    let reply_payloads: Vec<(Rank, Vec<u8>)> = req_by_requester
        .iter()
        .map(|(requester, keys)| {
            let mut b = Vec::with_capacity(keys.len() * 4);
            for &ek in keys {
                b.extend_from_slice(&edge_gid[&ek].to_le_bytes());
            }
            (*requester, b)
        })
        .collect();
    let responses = comm.alltoallv_bytes(&reply_payloads);
    for (coord, bytes) in responses {
        let gids = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (&k, g) in req_map[&coord].iter().zip(gids) {
            edge_gid.insert(k, g);
        }
    }
    }
    debug_assert_eq!(
        edge_gid.len(),
        local_new_edges.len(),
        "missing midpoint ids in NC refine"
    );

    // 5. Cross-rank midpoints the local mesh references but did not create.
    //    A coarse edge (u,v) whose midpoint was created on another rank in an
    //    earlier round is invisible to us unless the partition rebuild adds
    //    it as a ghost node; without it a later round would create a
    //    duplicate midpoint with a different global id.  Exchange the global
    //    (edge → (gid, owner)) table and append such midpoints as ghost nodes.
    let mut global_mid: BTreeMap<(u32, u32), (u32, i32)> = BTreeMap::new();
    for (&ek, &g) in &edge_gid {
        let owner = edge_ref_global.get(&ek).map(|&(_, o)| o).unwrap_or(rank);
        global_mid.insert(ek, (g, owner));
    }
    if comm.size() > 1 {
        let mut mid_payload = Vec::with_capacity(edge_gid.len() * 16);
        for (&(a, b), &(g, o)) in &global_mid {
            mid_payload.extend_from_slice(&a.to_le_bytes());
            mid_payload.extend_from_slice(&b.to_le_bytes());
            mid_payload.extend_from_slice(&g.to_le_bytes());
            mid_payload.extend_from_slice(&(o as u32).to_le_bytes());
        }
        let mid_sends: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
            .map(|r| (r, mid_payload.clone()))
            .collect();
        for (_src, bytes) in comm.alltoallv_bytes(&mid_sends) {
            for chunk in bytes.chunks_exact(16) {
                let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let g = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                let o = i32::from_le_bytes(chunk[12..16].try_into().unwrap());
                global_mid.entry((a, b)).or_insert((g, o));
            }
        }
    }
    // Edges of the refined local mesh whose midpoint lives elsewhere.
    let mut extra_ghost: Vec<(u32, i32, f64, f64)> = Vec::new(); // (gid, owner, mx, my)
    {
        let mut seen: BTreeSet<(u32, u32)> = BTreeSet::new();
        for e in 0..refined.n_elems() as ElemId {
            let ns = refined.elem_nodes(e);
            for &(a, b) in &[(0usize, 1usize), (1usize, 2usize), (2usize, 0usize)] {
                let lk = edge_key(ns[a], ns[b]);
                if !seen.insert(lk) {
                    continue;
                }
                // Edges touching a midpoint created this round are not coarse
                // edges; their midpoints are already handled locally.
                if lk.0 >= n_orig as u32 || lk.1 >= n_orig as u32 {
                    continue;
                }
                let gk = edge_key(gid_of(lk.0), gid_of(lk.1));
                if !global_mid.contains_key(&gk) {
                    continue;
                }
                if edge_gid.contains_key(&gk) {
                    continue; // we created this midpoint
                }
                let (g, owner) = global_mid[&gk];
                if owner == rank {
                    continue;
                }
                let xa = refined.coords_of(ns[a]);
                let xb = refined.coords_of(ns[b]);
                extra_ghost.push((g, owner, 0.5 * (xa[0] + xb[0]), 0.5 * (xa[1] + xb[1])));
            }
        }
    }
    extra_ghost.sort_unstable_by_key(|&(g, o, mx, my)| (g, o, mx.to_bits(), my.to_bits()));
    extra_ghost.dedup();

    // 6. Global ids + owners of every new local node (plus extra ghosts).
    let n_total_new = refined.n_nodes() + extra_ghost.len();
    let mut new_gid = vec![0u32; n_total_new];
    let mut new_owner = vec![0i32; n_total_new];
    for i in 0..n_orig {
        new_gid[i] = partition.global_node_ids[i];
        new_owner[i] = partition.node_owner[i];
    }
    for (&(a, b), &mid) in midpoint_map {
        if (mid as usize) < n_orig {
            continue; // old midpoint from a previous round (already in old segment)
        }
        let gkey = edge_key(gid_of(a), gid_of(b));
        let g = edge_gid[&gkey];
        // Midpoint ownership: the owner of the smallest marked element
        // referencing the edge — that rank refines the edge and creates the
        // midpoint, so it holds the authoritative copy.
        let (_mr, owner) = edge_ref_global[&gkey];
        new_gid[mid as usize] = g;
        new_owner[mid as usize] = owner;
    }
    for (i, &(g, owner, _mx, _my)) in extra_ghost.iter().enumerate() {
        let id = refined.n_nodes() + i;
        new_gid[id] = g;
        new_owner[id] = owner;
    }

    // 7. Reorder local nodes into [owned | ghost] segments.
    let mut order: Vec<usize> = (0..n_total_new).collect();
    order.sort_by_key(|&i| (new_owner[i] != rank, i));
    let n_owned_new = order.iter().filter(|&&i| new_owner[i] == rank).count();
    let mut remap = vec![0u32; n_total_new];
    let mut new_coords = Vec::with_capacity(n_total_new * 2);
    for (new_id, &old_id) in order.iter().enumerate() {
        remap[old_id] = new_id as u32;
        let c = if old_id < refined.n_nodes() {
            refined.coords_of(old_id as u32)
        } else {
            let (_, _, mx, my) = extra_ghost[old_id - refined.n_nodes()];
            [mx, my]
        };
        new_coords.push(c[0]);
        new_coords.push(c[1]);
    }
    let new_conn: Vec<u32> = refined.conn.iter().map(|&n| remap[n as usize]).collect();
    let new_face_conn: Vec<u32> =
        refined.face_conn.iter().map(|&n| remap[n as usize]).collect();

    let refined_mesh = Mesh::uniform(
        new_coords,
        new_conn,
        refined.elem_tags.clone(),
        refined.elem_type,
        new_face_conn,
        refined.face_tags.clone(),
        refined.face_type,
    );

    // 7. New partition: owned elements first (their children), then ghosts.
    let mut owned_global_nodes: Vec<u32> = Vec::with_capacity(n_owned_new);
    let mut ghost_global_nodes: Vec<(u32, i32)> = Vec::with_capacity(n_total_new - n_owned_new);
    for &old_id in order.iter() {
        if new_owner[old_id] == rank {
            owned_global_nodes.push(new_gid[old_id]);
        } else {
            ghost_global_nodes.push((new_gid[old_id], new_owner[old_id]));
        }
    }
    let n_owned_refined = (0..n_owned_elems)
        .map(|e| {
            let g = partition.global_elem(e as u32) as usize;
            if global_marked.contains(&(g as u32)) { 4 } else { 1 }
        })
        .sum::<usize>();
    let mut owned_global_elems: Vec<u32> = Vec::with_capacity(n_owned_refined);
    let mut ghost_global_elems: Vec<(u32, i32)> = Vec::with_capacity(n_refined - n_owned_refined);
    for e in 0..n_refined {
        if e < n_owned_refined {
            owned_global_elems.push(new_elem_gid[e]);
        } else {
            ghost_global_elems.push((new_elem_gid[e], new_elem_owner[e]));
        }
    }

    let new_partition = MeshPartition::from_partitioner(
        &owned_global_nodes,
        &ghost_global_nodes,
        &owned_global_elems,
        &ghost_global_elems,
        rank,
    );
    // The reordered `refined_mesh` keeps local node ids = partition segment
    // order (owned first, then ghost), matching from_partitioner.
    (refined_mesh, new_partition, remap)
}

// ─── DerefineTree building ──────────────────────────────────────────────────

/// Canonical edge key (sorted node pair) — mirrors `fem_mesh::amr::bisect::edge_key`.
#[allow(dead_code)]
fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a < b { (a, b) } else { (b, a) }
}

/// Build a [`DerefineTree`] from the inputs and outputs of
/// [`NCState::refine`](fem_mesh::amr::NCState::refine).
///
/// `NCState::refine` performs red refinement (1 Tri3 → 4 children).  The
/// children occupy consecutive positions in the refined element array,
/// determined by how many elements before `e` were refined.  This function
/// reconstructs the `DerefineTree` so that [`derefine_marked`] can roll back
/// those refinements.
pub fn build_derefine_tree_from_refine(
    coarse_mesh: &Mesh<2>,
    marked: &[ElemId],
    midpoint_map: &HashMap<(NodeId, NodeId), NodeId>,
) -> DerefineTree {
    let marked_set: HashSet<ElemId> = marked.iter().copied().collect();
    let mut records = HashMap::new();
    let n_coarse = coarse_mesh.n_elems();
    let mut n_refined_before = 0usize;

    for e in 0..n_coarse as ElemId {
        if marked_set.contains(&e) {
            let ns = coarse_mesh.element_nodes(e);
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];

            // Children start at position e + 3 × (#refined before e) in the
            // refined element array (each refined element → 4 children,
            // each unrefined → 1 element).
            let start = e + 3 * n_refined_before as ElemId;

            records.insert(e, DerefineRecord {
                parent_nodes: [n0, n1, n2],
                parent_tag: coarse_mesh.elem_tags[e as usize],
                children: [start, start + 1, start + 2, start + 3],
            });
            n_refined_before += 1;
        }
    }

    DerefineTree { records, midpoint_map: midpoint_map.clone() }
}

// ─── Parallel derefinement ──────────────────────────────────────────────────

/// Result of a single parallel *de*refinement (coarsening) cycle.
///
/// Structurally identical to [`ParRefinedMesh`]; semantic difference is that
/// `n_new_elems` is the coarsened element count (smaller than before).
pub struct ParDerefinedMesh {
    pub par_mesh:    ParallelMesh<Mesh<2>>,
    pub nc_state:    NCState,
    pub solution:    Vec<f64>,
    pub n_new_elems: usize,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Return an unchanged [`ParDerefinedMesh`] — used when no derefinement occurs.
fn make_unchanged_derefined_mesh(
    local_mesh: &Mesh<2>,
    comm: crate::Comm,
    partition: &MeshPartition,
    nc_state: &NCState,
    solution: Option<&[f64]>,
) -> ParDerefinedMesh {
    ParDerefinedMesh {
        par_mesh: ParallelMesh::new(local_mesh.clone(), comm, partition.clone()),
        nc_state: nc_state.clone(),
        solution: solution.map(|s| s.to_vec()).unwrap_or_default(),
        n_new_elems: local_mesh.n_elems(),
    }
}

/// Compact a coarsened mesh by removing orphaned (unused) nodes and build a
/// partition that preserves the old partition's global-ID numbering.
///
/// After serial [`derefine_marked`], the mesh keeps *all* nodes from the fine
/// mesh, but only a subset (coarse-parent nodes + any non-derefined children)
/// are referenced by elements.  This function:
///
/// 1. Scans element and face connectivity for active node IDs.
/// 2. Compacts the node array (removes unused entries).
/// 3. Remaps connectivity to the compacted indices.
/// 4. Builds a [`MeshPartition`] that preserves the old partition's global
///    node/element IDs.
/// 5. Restricts the solution vector to only the active nodes.
fn compact_derefined_mesh(
    mesh: &Mesh<2>,
    old_partition: &MeshPartition,
    dropped_children: &HashSet<ElemId>,
    restored_parents: &[ElemId],
    solution: Option<&[f64]>,
    local_rank: Rank,
) -> (Mesh<2>, MeshPartition, Vec<f64>) {
    // ── 1. Collect active node IDs ──────────────────────────────────────────
    let mut active: HashSet<NodeId> = HashSet::new();
    for e in 0..mesh.n_elems() as ElemId {
        for &n in mesh.element_nodes(e) {
            active.insert(n);
        }
    }
    for &n in &mesh.face_conn {
        active.insert(n);
    }

    let mut sorted: Vec<NodeId> = active.into_iter().collect();
    sorted.sort_unstable();
    let n_active = sorted.len();

    // ── 2. Build old->new node mapping and new coordinate array ─────────────
    let mut old_to_new: HashMap<NodeId, NodeId> = HashMap::with_capacity(n_active);
    let mut new_coords = Vec::with_capacity(n_active * 2);
    let mut owned_global_nodes = Vec::with_capacity(n_active);

    for (new_id, &old_id) in sorted.iter().enumerate() {
        old_to_new.insert(old_id, new_id as NodeId);
        owned_global_nodes.push(old_partition.global_node_ids[old_id as usize]);
        let c = mesh.node_coords(old_id);
        new_coords.extend_from_slice(&c[..2]);
    }

    // ── 3. Remap connectivity ───────────────────────────────────────────────
    let new_conn: Vec<NodeId> = mesh.conn.iter()
        .map(|&n| old_to_new[&n])
        .collect();
    let new_face_conn: Vec<NodeId> = mesh.face_conn.iter()
        .map(|&n| old_to_new[&n])
        .collect();

    let new_mesh = Mesh::uniform(
        new_coords,
        new_conn,
        mesh.elem_tags.clone(),
        mesh.elem_type,
        new_face_conn,
        mesh.face_tags.clone(),
        mesh.face_type,
    );

    // ── 4. Build element partition ──────────────────────────────────────────
    let old_geids = &old_partition.global_elem_ids;
    let old_n_elems = old_geids.len();
    let new_n_elems = new_mesh.n_elems();

    let mut owned_global_elems = Vec::with_capacity(new_n_elems);
    for le in 0..old_n_elems as ElemId {
        if !dropped_children.contains(&le) {
            owned_global_elems.push(old_geids[le as usize]);
        }
    }
    owned_global_elems.extend_from_slice(restored_parents);

    debug_assert_eq!(
        owned_global_elems.len(), new_n_elems,
        "element count mismatch in compact_derefined_mesh"
    );

    let new_partition = MeshPartition::from_partitioner(
        &owned_global_nodes,
        &[], // no ghost nodes (serial partition after derefinement)
        &owned_global_elems,
        &[], // no ghost elems
        local_rank,
    );

    // ── 5. Restrict solution ────────────────────────────────────────────────
    let restricted = if let Some(sol) = solution {
        sorted.iter().map(|&old_id| {
            if (old_id as usize) < sol.len() { sol[old_id as usize] } else { 0.0 }
        }).collect()
    } else {
        vec![]
    };

    (new_mesh, new_partition, restricted)
}

/// Coarsen previously refined elements in a distributed mesh.
///
/// # Parallel coordination protocol
///
/// 1. Each rank filters `marked_parents` to only those parents where **all 4
///    children** reside on that rank (checked via the partition's
///    `global_elem_ids` / `local_elem` lookup).
/// 2. Per-parent `i64` votes (`1` = this rank owns all children, `0` = not)
///    exchanged via `allreduce_sum_i64`.  Only parents with vote-sum `== 1`
///    (children on exactly one rank) are coarsened — parents that straddle rank
///    boundaries are **skipped** (future work: migrate children first).
/// 3. Each rank builds a **local** [`DerefineTree`] whose children use local
///    element indices, then calls the serial [`derefine_marked`].
/// 4. The [`NCState`] is rolled back one level via `derefine_last()`.
/// 5. The mesh is compacted (orphaned midpoint nodes are removed) and the
///    solution is restricted to the active (coarse) nodes.
///
/// # MPI Safety
///
/// `marked_parents` **must be identical on all ranks**.  If different ranks
/// pass different parent lists, the per-parent `allreduce_sum_i64` will
/// deadlock or produce inconsistent results.  Collectives require identical
/// participation across ranks.
///
/// # Arguments
/// * `par_mesh` — partitioned refined mesh.
/// * `nc_state` — NCState from the refinement step (will be rolled back).
/// * `derefine_tree` — parent→children tree from [`build_derefine_tree_from_refine`].
/// * `marked_parents` — **global** ElemIds of parent elements to coarsen.
/// * `solution` — optional solution vector on the refined mesh.
pub fn par_derefine_marked(
    par_mesh: &ParallelMesh<Mesh<2>>,
    nc_state: &mut NCState,
    derefine_tree: &DerefineTree,
    marked_parents: &[ElemId],
    solution: Option<&[f64]>,
) -> Result<ParDerefinedMesh, ParAmrError> {
    let local_mesh = par_mesh.local_mesh();
    let partition = par_mesh.partition();
    let comm = par_mesh.comm().clone();

    if marked_parents.is_empty() {
        return Ok(make_unchanged_derefined_mesh(
            local_mesh, comm, partition, nc_state, solution,
        ));
    }

    // ── Step 1: filter to parents whose children are all on this rank ─────
    // Build a local DerefineTree with local element indices.
    let mut local_records: HashMap<ElemId, DerefineRecord> = HashMap::new();
    // Track per-parent "vote" for MPI coordination.
    let mut votes: Vec<i64> = Vec::with_capacity(marked_parents.len());

    for &parent_global in marked_parents {
        let rec = match derefine_tree.records.get(&parent_global) {
            Some(r) => r,
            None => { votes.push(0); continue; }
        };

        // Check all 4 children are present on this rank.
        let mut all_local = true;
        let mut local_children = [0u32; 4];
        for (j, &child_global) in rec.children.iter().enumerate() {
            match partition.local_elem(child_global) {
                Some(local_id) => local_children[j] = local_id,
                None => { all_local = false; break; }
            }
        }

        if all_local {
            local_records.insert(parent_global, DerefineRecord {
                parent_nodes: rec.parent_nodes,
                parent_tag: rec.parent_tag,
                children: local_children,
            });
            votes.push(1);
        } else {
            votes.push(0);
        }
    }

    // ── Step 2: MPI coordination via per-parent Allreduce ─────────────────
    // Sum votes across ranks.  A parent is derefinable iff sum == 1
    // (exactly one rank owns all its children).
    let size = comm.size();
    let global_votes: Vec<i64> = if size > 1 {
        let mut gv = votes.clone();
        // allreduce_sum_i64 operates on a single value; we loop.
        // For large numbers of parents this should be replaced with a
        // single alltoallv, but for typical AMR it's fine.
        for v in &mut gv {
            *v = comm.allreduce_sum_i64(*v);
        }
        gv
    } else {
        votes // no-op for single-rank
    };

    // Build final local tree from parents with sum == 1.
    let mut final_records: HashMap<ElemId, DerefineRecord> = HashMap::new();
    for (i, &parent_global) in marked_parents.iter().enumerate() {
        if global_votes[i] == 1 {
            if let Some(rec) = local_records.get(&parent_global) {
                final_records.insert(parent_global, rec.clone());
            }
        }
    }

    // ── Step 3: serial derefinement on the local mesh ─────────────────────
    let derefinable_parents: Vec<ElemId> = final_records.keys().copied().collect();
    if derefinable_parents.is_empty() {
        return Ok(make_unchanged_derefined_mesh(
            local_mesh, comm, partition, nc_state, solution,
        ));
    }

    // Compute dropped children BEFORE consuming final_records.
    let dropped_children: HashSet<ElemId> = final_records
        .values()
        .flat_map(|r| r.children.iter().copied())
        .collect();

    let local_tree = DerefineTree {
        records: final_records,
        midpoint_map: derefine_tree.midpoint_map.clone(),
    };

    let coarsened_mesh = derefine_marked(local_mesh, &local_tree, &derefinable_parents);

    // ── Step 4: roll back NCState ─────────────────────────────────────────
    nc_state.derefine_last();

    // ── Step 5: compact mesh, restrict solution, build partition ─────────
    let (compacted_mesh, new_partition, restricted) = compact_derefined_mesh(
        &coarsened_mesh,
        partition,
        &dropped_children,
        &derefinable_parents,
        solution,
        comm.rank() as Rank,
    );
    let new_n_elems = compacted_mesh.n_elems();
    let new_par_mesh = ParallelMesh::new(compacted_mesh, comm, new_partition);

    Ok(ParDerefinedMesh {
        par_mesh: new_par_mesh,
        nc_state: nc_state.clone(),
        solution: restricted,
        n_new_elems: new_n_elems,
    })
}

// ─── par_repartition ──────────────────────────────────────────────────────────

fn merge_submeshes(
    meshes: &[Mesh<2>],
    partitions: &[MeshPartition],
) -> Result<Mesh<2>, ParAmrError> {
    // Collect unique global nodes: global_id → coords
    let mut global_nodes: BTreeMap<NodeId, [f64; 2]> = BTreeMap::new();
    // Collect all elements keyed by global element ID
    let mut global_elems: BTreeMap<ElemId, (Vec<NodeId>, i32)> = BTreeMap::new();
    // Collect all boundary faces (deduplicated by node set)
    let mut global_faces: Vec<(Vec<NodeId>, BoundaryTag)> = Vec::new();

    for (mesh, part) in meshes.iter().zip(partitions.iter()) {
        let gn = &part.global_node_ids;
        // Nodes
        for local_id in 0..mesh.n_nodes() {
            let gid = gn[local_id];
            let cx = mesh.node_coords(local_id as u32);
            global_nodes.entry(gid).or_insert_with(|| [cx[0], cx[1]]);
        }
        // Elements
        for le in 0..mesh.n_elems() {
            let ge = part.global_elem_ids[le];
            let local_conn = mesh.element_nodes(le as u32);
            let global_conn: Vec<NodeId> = local_conn.iter().map(|&n| gn[n as usize]).collect();
            global_elems.entry(ge).or_insert((global_conn, mesh.elem_tags[le]));
        }
        // Boundary faces
        let n_faces = mesh.face_conn.len() / 3; // Tri face = 2 nodes, but in 2D boundary faces are edges
        // Actually for Mesh<2>, face_conn stores edge vertex pairs
        // face_conn is flat: each face has face_type.nodes_per_element() entries
        let f_npe = mesh.face_type.nodes_per_element();
        if f_npe > 0 && n_faces > 0 {
            let nf = mesh.face_conn.len() / f_npe;
            for fi in 0..nf {
                let global_fv: Vec<NodeId> = (0..f_npe)
                    .map(|k| gn[mesh.face_conn[fi * f_npe + k] as usize])
                    .collect();
                let tag = mesh.face_tags.get(fi).copied().unwrap_or(0);
                global_faces.push((global_fv, tag));
            }
        }
    }

    if global_nodes.is_empty() || global_elems.is_empty() {
        return Err(ParAmrError::RepartitionError(
            "merge_submeshes: no nodes or elements".into(),
        ));
    }

    // Build new local index mapping: sorted global ID → 0..n_global_nodes
    let new_id: HashMap<NodeId, NodeId> = global_nodes
        .keys()
        .enumerate()
        .map(|(i, &gid)| (gid, i as NodeId))
        .collect();

    let n_glob_nodes = global_nodes.len();
    let mut coords = Vec::with_capacity(n_glob_nodes * 2);
    for xy in global_nodes.values() {
        coords.push(xy[0]);
        coords.push(xy[1]);
    }

    let n_glob_elems = global_elems.len();
    let (elem_type, npe) = if n_glob_elems > 0 {
        let first_conn = &global_elems.values().next().unwrap().0;
        let npe = first_conn.len();
        (if npe == 3 { fem_mesh::ElementType::Tri3 } else { fem_mesh::ElementType::Tri6 }, npe)
    } else {
        return Err(ParAmrError::RepartitionError("no elements to merge".into()));
    };

    let mut conn = Vec::with_capacity(n_glob_elems * npe);
    let mut elem_tags = Vec::with_capacity(n_glob_elems);
    for (global_conn, tag) in global_elems.values() {
        for &gn_id in global_conn {
            conn.push(new_id[&gn_id]);
        }
        elem_tags.push(*tag);
    }

    // Boundary faces
    let face_type = fem_mesh::ElementType::Line2;
    let f_npe = 2usize;
    let mut face_conn = Vec::with_capacity(global_faces.len() * f_npe);
    let mut face_tags = Vec::with_capacity(global_faces.len());
    for (fv, tag) in &global_faces {
        for &gn_id in fv {
            face_conn.push(new_id[&gn_id]);
        }
        face_tags.push(*tag);
    }

    Ok(Mesh {
        coords,
        conn,
        elem_tags,
        elem_type,
        face_conn,
        face_tags,
        face_type,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    })
}

/// Re-distribute elements across MPI ranks after refinement.
///
/// Gathers all sub-meshes to rank 0, merges them into a single global mesh,
/// and redistributes via [`partition_mesh_streaming`].
pub fn par_repartition(
    par_mesh: ParallelMesh<Mesh<2>>,
) -> Result<ParallelMesh<Mesh<2>>, ParAmrError> {
    let comm = par_mesh.comm().clone();
    let size = comm.size();
    let rank = comm.rank();

    if size == 1 {
        return Ok(par_mesh);
    }

    let local_mesh = par_mesh.local_mesh().clone();
    let partition = par_mesh.partition().clone();

    if rank == 0 {
        // Collect all sub-meshes
        let mut meshes = vec![local_mesh];
        let mut parts = vec![partition];

        for src in 1..size as i32 {
            let buf = comm.recv_bytes(src, REPART_TAG_BASE + src);
            let (sub_mesh, sub_part) = decode_submesh::<2>(&buf)
                .map_err(ParAmrError::SerializationError)?;
            meshes.push(sub_mesh);
            parts.push(sub_part);
        }

        let global_mesh = merge_submeshes(&meshes, &parts)?;

        // Redistribute using the streaming partitioner
        partition_mesh_streaming(Some(&global_mesh), &comm)
            .map_err(ParAmrError::RepartitionError)
    } else {
        // Send our mesh to rank 0
        let encoded = encode_submesh(&local_mesh, &partition);
        comm.send_bytes(0, REPART_TAG_BASE + rank, &encoded);

        // Receive new partition from rank 0
        let buf = comm.recv_bytes(0, STREAM_TAG_BASE + rank);
        let (new_mesh, new_part) = decode_submesh::<2>(&buf)
            .map_err(ParAmrError::SerializationError)?;
        Ok(ParallelMesh::new(new_mesh, comm.clone(), new_part))
    }
}

/// Rebalance elements via SFC ordering + MPI ring exchange.
///
/// Unlike [`par_repartition`] (gather to rank 0), this does a **neighbour
/// exchange** on a ring topology: each rank keeps its SFC-lowest elements
/// up to the target count and sends the excess to the next rank.
///
/// Target per rank is `n_global / size`.  When a rank's local count exceeds
/// the target, the excess is sent clockwise.  When it is below target, it
/// receives from the anti-clockwise neighbour.
///
/// This is a **single-pass diffusive** scheme.  Full balance may require
/// multiple passes (call in a loop until imbalance < threshold).
pub fn sfc_rebalance_ring<const D: usize>(
    par_mesh: ParallelMesh<Mesh<D>>,
) -> Result<ParallelMesh<Mesh<D>>, ParAmrError> {
    let comm = par_mesh.comm().clone();
    let size = comm.size();
    let rank = comm.rank();

    if size <= 1 {
        return Ok(par_mesh);
    }

    let local_mesh = par_mesh.local_mesh().clone();
    let n_local = local_mesh.n_elems();
    let _partition = par_mesh.partition().clone();

    // 1. Compute global element count via broadcast from rank 0
    let mut n_global = n_local;
    if rank == 0 {
        for src in 1..size as i32 {
            let buf = comm.recv_bytes(src, REPART_TAG_BASE + 1000 + src);
            let count_bytes: [u8; 8] = buf[..8].try_into().unwrap();
            n_global += usize::from_le_bytes(count_bytes);
        }
        for dst in 1..size as i32 {
            comm.send_bytes(dst, REPART_TAG_BASE + 2000 + dst, &n_global.to_le_bytes());
        }
    } else {
        comm.send_bytes(0, REPART_TAG_BASE + 1000 + rank, &n_local.to_le_bytes());
        let buf = comm.recv_bytes(0, REPART_TAG_BASE + 2000 + rank);
        let count_bytes: [u8; 8] = buf[..8].try_into().unwrap();
        n_global = usize::from_le_bytes(count_bytes);
    }

    let target = n_global / size;
    let _excess = n_global % size;
    let size_i32 = size as i32;

    if n_global == 0 {
        return Ok(par_mesh);
    }

    // 2. SFC plan: keep target elements with lowest Morton keys
    let (keep_idx, send_idx) = sfc_rebalance_plan(&local_mesh, target.min(n_local));

    // 3. Build send submesh from excess elements
    let n_send = send_idx.len();
    let n_keep = keep_idx.len();

    if n_send == 0 && n_keep == n_local {
        return Ok(par_mesh);
    }

    let keep_mesh = extract_submesh_elements(&local_mesh, &keep_idx);
    let send_mesh = extract_submesh_elements(&local_mesh, &send_idx);

    // 4. Ring exchange: send to next rank, receive from previous
    let next_rank = (rank + 1) % size_i32;
    let prev_rank = if rank == 0 { size_i32 - 1 } else { rank - 1 };

    // Encode send submesh (with minimal partition info)
    let send_part = crate::partition::MeshPartition::new_serial(
        send_mesh.n_nodes(), send_mesh.n_elems());
    let encoded_send = crate::mesh_serde::encode_submesh::<D>(&send_mesh, &send_part);

    // Buffered send/recv to avoid deadlock on ring
    // Even ranks send first, odd ranks recv first (standard MPI pattern)
    let recv_buf = if (rank % 2) == 0 {
        comm.send_bytes(next_rank, REPART_TAG_BASE + 3000 + rank, &encoded_send);
        if n_send < n_local || size > 2 {
            // Expect data from prev rank if we're not the only sender
            Some(comm.recv_bytes(prev_rank, REPART_TAG_BASE + 3000 + prev_rank))
        } else {
            None
        }
    } else {
        let buf = if n_send < n_local || size > 2 {
            Some(comm.recv_bytes(prev_rank, REPART_TAG_BASE + 3000 + prev_rank))
        } else {
            None
        };
        comm.send_bytes(next_rank, REPART_TAG_BASE + 3000 + rank, &encoded_send);
        buf
    };

    // 5. Merge received elements into local mesh
    let final_mesh = if let Some(buf) = recv_buf {
        let (recv_mesh, _recv_part) = crate::mesh_serde::decode_submesh::<D>(&buf)
            .map_err(ParAmrError::SerializationError)?;
        merge_two_meshes(&keep_mesh, &recv_mesh)
    } else {
        keep_mesh
    };

    let new_part = crate::partition::MeshPartition::new_serial(
        final_mesh.n_nodes(), final_mesh.n_elems());
    Ok(ParallelMesh::new(final_mesh, comm, new_part))
}

// ─── SFC rebalancing ──────────────────────────────────────────────────────────

/// Compute a diffusive load-balancing plan based on SFC ordering.
///
/// Returns a list of elements to send to each neighbour rank.  The caller
/// should then use the existing [`par_repartition`] machinery to perform
/// the actual element exchange.
///
/// Step 1: sort local elements by Morton (Z-order) curve key of their centroid.
/// Step 2: keep the first `target` elements and mark the rest as send candidates.
/// Step 3: send candidates go to the next rank in the ring.
///
/// This is a **one-step diffusive** scheme: each rank talks only to its
/// clockwise neighbour.  Full balance is reached after O(P) ring passes.
pub fn sfc_rebalance_plan<const D: usize>(
    mesh: &Mesh<D>,
    target: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n_local = mesh.n_elems();
    if n_local <= target {
        return ((0..n_local).collect(), Vec::new());
    }

    // Compute SFC keys
    let centroids = compute_centroids_simple(mesh);
    let opts = crate::sfc::SfcOptions::default();
    let keys: Vec<u64> = centroids.iter()
        .map(|c| crate::sfc::morton_code::<D>(c, opts.bits_per_coord))
        .collect();

    // Sort by SFC key
    let mut indices: Vec<usize> = (0..n_local).collect();
    indices.sort_by_key(|&i| keys[i]);

    let keep: Vec<usize> = indices.iter().take(target).copied().collect();
    let send: Vec<usize> = indices.iter().skip(target).copied().collect();
    (keep, send)
}

/// Extract a subset of elements from a mesh into a new mesh.
fn extract_submesh_elements<const D: usize>(
    mesh: &Mesh<D>,
    elem_indices: &[usize],
) -> Mesh<D> {
    use std::collections::HashMap;
    let n = elem_indices.len();
    let mut new_conn = Vec::new();
    let mut new_tags = Vec::with_capacity(n);
    let mut node_map: HashMap<u32, u32> = HashMap::new();
    let mut new_coords = Vec::new();

    let mut next_node = 0u32;
    for &ei in elem_indices {
        let e = ei as u32;
        let nodes = mesh.element_nodes(e);
        let mut local_nodes = Vec::with_capacity(nodes.len());
        for &n in nodes {
            let new_n = *node_map.entry(n).or_insert_with(|| {
                let idx = next_node;
                next_node += 1;
                let c = mesh.node_coords(n);
                new_coords.extend_from_slice(&c[..D]);
                idx
            });
            local_nodes.push(new_n);
        }
        new_conn.extend_from_slice(&local_nodes);
        new_tags.push(mesh.elem_tags[e as usize]);
    }

    // Create a minimal mesh with the subset
    // (face data is not transferred; caller regenerates if needed)
    Mesh {
        coords: new_coords,
        conn: new_conn,
        elem_tags: new_tags,
        elem_type: mesh.elem_type,
        face_conn: Vec::new(),
        face_tags: Vec::new(),
        face_type: mesh.face_type,
        elem_types: mesh.elem_types.clone(),
        elem_offsets: mesh.elem_offsets.clone(),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(),
        edge_to_elem: Vec::new(),
        geometry: None, nc_vertex_view: None,
    }
}

/// Merge two meshes (concatenate nodes and elements).
fn merge_two_meshes<const D: usize>(
    mesh_a: &Mesh<D>,
    mesh_b: &Mesh<D>,
) -> Mesh<D> {
    let n_a_nodes = mesh_a.n_nodes();
    let mut coords = mesh_a.coords.clone();
    coords.extend_from_slice(&mesh_b.coords);

    let mut conn = mesh_a.conn.clone();
    let offset = n_a_nodes as u32;
    for &n in &mesh_b.conn {
        conn.push(n + offset);
    }

    let mut elem_tags = mesh_a.elem_tags.clone();
    elem_tags.extend_from_slice(&mesh_b.elem_tags);

    Mesh {
        coords,
        conn,
        elem_tags,
        elem_type: mesh_a.elem_type,
        face_conn: mesh_a.face_conn.clone(),
        face_tags: mesh_a.face_tags.clone(),
        face_type: mesh_a.face_type,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(),
        edge_to_elem: Vec::new(),
        geometry: None, nc_vertex_view: None,
    }
}

/// Compute element centroids (simplified, for D=2 and D=3).
fn compute_centroids_simple<const D: usize>(mesh: &Mesh<D>) -> Vec<[f64; D]> {
    let n_elems = mesh.n_elems();
    let mut centroids = Vec::with_capacity(n_elems);
    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let npe = nodes.len();
        let mut c = [0.0_f64; D];
        for &n in nodes {
            let nc = mesh.node_coords(n);
            for d in 0..D { c[d] += nc[d]; }
        }
        let inv = 1.0 / npe as f64;
        for d in 0..D { c[d] *= inv; }
        centroids.push(c);
    }
    centroids
}

// ─── Solution prolongation ────────────────────────────────────────────────────

/// Prolongate a P1 solution from coarse mesh to refined mesh.
///
/// Coarse-node values are copied directly. New midpoint nodes are
/// interpolated from the two nearest coarse nodes — exact for P1.
pub fn prolongate_p1(
    coarse:     &Mesh<2>,
    refined:    &Mesh<2>,
    sol_coarse: &[f64],
) -> Vec<f64> {
    let n_fine   = refined.n_nodes();
    let n_coarse = coarse.n_nodes();
    let mut sol_fine = vec![0.0_f64; n_fine];

    let n_copy = n_coarse.min(n_fine).min(sol_coarse.len());
    sol_fine[..n_copy].copy_from_slice(&sol_coarse[..n_copy]);

    for new_node in n_coarse..n_fine {
        let xp = refined.node_coords(new_node as u32);
        let mut best1 = (f64::MAX, 0.0_f64);
        let mut best2 = (f64::MAX, 0.0_f64);
        for cn in 0..n_coarse as u32 {
            let xc = coarse.node_coords(cn);
            let d2: f64 = xp.iter().zip(xc.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            if d2 < best1.0 {
                best2 = best1;
                best1 = (d2, sol_coarse[cn as usize]);
            } else if d2 < best2.0 {
                best2 = (d2, sol_coarse[cn as usize]);
            }
        }
        if best2.0 < f64::MAX
            && (best1.0 - best2.0).abs() < 1e-10 * (best1.0 + best2.0 + 1e-14)
        {
            sol_fine[new_node] = 0.5 * (best1.1 + best2.1);
        } else {
            sol_fine[new_node] = best1.1;
        }
    }
    sol_fine
}

/// Reorder a per-node vector from the NCState output numbering to the
/// reordered (`[owned | ghost]` segment) numbering produced by
/// [`rebuild_partition_nc`].  No-op for empty vectors.
fn reorder_solution(sol: &[f64], remap: &[u32]) -> Vec<f64> {
    if sol.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; remap.len()];
    for (new_id, &old_id) in remap.iter().enumerate() {
        // Extra ghost midpoints appended by the partition rebuild have no
        // solution value (they are not referenced by any local element).
        if (old_id as usize) < sol.len() {
            out[new_id] = sol[old_id as usize];
        }
    }
    out
}

/// Compute the global element count via allreduce.
///
/// Returns `(n_global, max_local)`.
fn compute_global_stats(n_local: usize, comm: &crate::Comm) -> (usize, usize) {
    let n_global = comm.allreduce_sum_i64(n_local as i64) as usize;
    // Approximate max with a simple ring to avoid needing allreduce_max.
    // Each rank sends local count to next rank; after size steps each has seen all.
    let size = comm.size();
    let rank = comm.rank();
    let mut max_seen = n_local;
    if size > 1 {
        let next = (rank + 1) % size as i32;
        let prev = if rank == 0 { size as i32 - 1 } else { rank - 1 };
        let mut buf = n_local.to_le_bytes().to_vec();
        for _step in 0..size - 1 {
            comm.send_bytes(next, REPART_TAG_BASE + 5000 + rank, &buf);
            let recv = comm.recv_bytes(prev, REPART_TAG_BASE + 5000 + prev);
            let incoming = usize::from_le_bytes(recv[..8].try_into().unwrap());
            max_seen = max_seen.max(incoming);
            buf = recv;
        }
    }
    (n_global, max_seen)
}

/// Compute the global load imbalance factor across all ranks.
///
/// Returns `max_local / ideal`.  Value > 1.0 means overloaded ranks exist.
pub fn compute_global_imbalance(n_local: usize, comm: &crate::Comm) -> f64 {
    let size = comm.size() as f64;
    if size <= 0.0 { return 0.0; }
    let n_global: usize = comm.allreduce_sum_i64(n_local as i64) as usize;
    if n_global == 0 { return 0.0; }
    let ideal = n_global as f64 / size;
    if ideal <= 0.0 { return 0.0; }
    let (_total, max_local) = compute_global_stats(n_local, comm);
    max_local as f64 / ideal
}

/// Multi-iteration diffusive load-balancing.
///
/// Unlike [`sfc_rebalance_ring`] (single-pass ring exchange), this function
/// performs **iterative nearest-neighbour diffusion**: each step exchanges
/// excess elements with the neighbour that has the most complementary load.
///
/// After each iteration the global imbalance is recomputed.  The process
/// stops when `imbalance < 1.0 + threshold` or `max_iters` is reached.
pub fn par_diffusive_rebalance<const D: usize>(
    par_mesh: ParallelMesh<Mesh<D>>,
    threshold: f64,
    max_iters: usize,
    n_diffuse: usize,
) -> Result<ParallelMesh<Mesh<D>>, ParAmrError> {
    let comm = par_mesh.comm().clone();
    let size = comm.size();
    if size <= 1 { return Ok(par_mesh); }

    let mut mesh = par_mesh;
    let rank = comm.rank();
    let size_i32 = size as i32;

    for _iter in 0..max_iters {
        let n_local = mesh.local_mesh().n_elems();
        let imb = compute_global_imbalance(n_local, &comm);
        if imb < 1.0 + threshold { break; }

        let n_global: usize = comm.allreduce_sum_i64(n_local as i64) as usize;
        let target = n_global / size;
        let n_excess = n_local.saturating_sub(target);
        let n_to_send = n_excess.min(n_diffuse);

        if n_to_send == 0 { continue; }

        // SFC plan: keep lowest keys, send highest keys
        let n_keep = n_local.saturating_sub(n_to_send);
        let (keep_idx, send_idx) = sfc_rebalance_plan(mesh.local_mesh(), n_keep);
        if send_idx.is_empty() { continue; }

        let keep_mesh = extract_submesh_elements(mesh.local_mesh(), &keep_idx);
        let send_mesh = extract_submesh_elements(mesh.local_mesh(), &send_idx);

        // Send to neighbour one step clockwise (ring diffusion)
        let next_rank = (rank + 1) % size_i32;
        let prev_rank = if rank == 0 { size_i32 - 1 } else { rank - 1 };

        let send_part = MeshPartition::new_serial(send_mesh.n_nodes(), send_mesh.n_elems());
        let encoded_send = crate::mesh_serde::encode_submesh::<D>(&send_mesh, &send_part);
        let tag = REPART_TAG_BASE + 4000 + (_iter as i32) * 100;

        let recv_buf = if (rank % 2) == 0 {
            comm.send_bytes(next_rank, tag + rank, &encoded_send);
            if n_to_send > 0 {
                Some(comm.recv_bytes(prev_rank, tag + prev_rank))
            } else {
                None
            }
        } else {
            let buf = if n_to_send > 0 {
                Some(comm.recv_bytes(prev_rank, tag + prev_rank))
            } else {
                None
            };
            comm.send_bytes(next_rank, tag + rank, &encoded_send);
            buf
        };

        let final_mesh = if let Some(buf) = recv_buf {
            let (recv_mesh, _recv_part) = crate::mesh_serde::decode_submesh::<D>(&buf)
                .map_err(ParAmrError::SerializationError)?;
            merge_two_meshes(&keep_mesh, &recv_mesh)
        } else {
            keep_mesh
        };

        let new_part = MeshPartition::new_serial(final_mesh.n_nodes(), final_mesh.n_elems());
        mesh = ParallelMesh::new(final_mesh, comm.clone(), new_part);
    }

    Ok(mesh)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{Mesh, amr::NCState};
    use crate::{
        par_mesh::ParallelMesh, partition::MeshPartition,
        backend::native::SerialBackend, comm::Comm,
        launcher::native::ThreadLauncher, par_partition::partition_mesh,
        WorkerConfig,
    };

    fn make_serial_par_mesh(n: usize) -> (ParallelMesh<Mesh<2>>, NCState) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elements());
        let comm = Comm::from_backend(Box::new(SerialBackend));
        let par_mesh = ParallelMesh::new(mesh, comm, partition);
        (par_mesh, NCState::new())
    }

    #[test]
    fn par_refine_increases_elements() {
        let (par_mesh, nc) = make_serial_par_mesh(2);
        let n_before = par_mesh.local_mesh().n_elements();
        let marked: Vec<ElemId> = vec![0, 1];
        let result = par_refine_marked(&par_mesh, nc, &marked, None).unwrap();
        assert!(result.par_mesh.local_mesh().n_elements() > n_before);
    }

    #[test]
    fn par_refine_no_marked_is_identity() {
        let (par_mesh, nc) = make_serial_par_mesh(3);
        let n_before = par_mesh.local_mesh().n_elements();
        let result = par_refine_marked(&par_mesh, nc, &[], None).unwrap();
        assert_eq!(result.par_mesh.local_mesh().n_elements(), n_before);
    }

    #[test]
    fn prolongate_constant_function() {
        let coarse = Mesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();
        let marked: Vec<ElemId> = (0..coarse.n_elements() as ElemId).collect();
        let (refined, _, _) = nc.refine(&coarse, &marked, 0);
        let sol_coarse = vec![3.14_f64; coarse.n_nodes()];
        let sol_fine = prolongate_p1(&coarse, &refined, &sol_coarse);
        for (i, &v) in sol_fine.iter().enumerate() {
            assert!((v - 3.14).abs() < 1e-12, "node {i}: got {v}");
        }
    }

    #[test]
    fn prolongate_linear_function() {
        let coarse = Mesh::<2>::unit_square_tri(4);
        let mut nc = NCState::new();
        let marked: Vec<ElemId> = (0..coarse.n_elements() as ElemId).collect();
        let (refined, _, _) = nc.refine(&coarse, &marked, 0);
        let sol_coarse: Vec<f64> = (0..coarse.n_nodes())
            .map(|i| { let c = coarse.node_coords(i as u32); c[0] + c[1] })
            .collect();
        let sol_fine = prolongate_p1(&coarse, &refined, &sol_coarse);
        let n_coarse = coarse.n_nodes();
        let max_err = (n_coarse..refined.n_nodes())
            .map(|i| {
                let c = refined.node_coords(i as u32);
                (sol_fine[i] - (c[0] + c[1])).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.15, "P1 prolongation error: {max_err}");
    }

    #[test]
    fn par_repartition_preserves_elements_single_rank() {
        let (par_mesh, _) = make_serial_par_mesh(4);
        let n = par_mesh.local_mesh().n_elements();
        let result = par_repartition(par_mesh).unwrap();
        assert_eq!(result.local_mesh().n_elements(), n);
    }

    #[test]
    fn merge_submeshes_round_trip() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let part = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elems());
        let merged = merge_submeshes(&[mesh.clone()], &[part]).unwrap();
        assert_eq!(merged.n_elems(), mesh.n_elems());
        assert_eq!(merged.n_nodes(), mesh.n_nodes());
        for le in 0..mesh.n_elems() as u32 {
            let orig: Vec<_> = mesh.element_nodes(le).iter().copied().collect();
            let merged_conn: Vec<_> = merged.element_nodes(le).iter().copied().collect();
            assert_eq!(orig, merged_conn, "elem {le} connectivity mismatch");
        }
    }

    // ─── SFC rebalancing tests ───────────────────────────────────────────────

    #[test]
    fn sfc_rebalance_plan_under_target_keeps_all() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements
        let n = mesh.n_elems();
        let (keep, send) = sfc_rebalance_plan(&mesh, n + 10);
        assert_eq!(keep.len(), n, "should keep all when target exceeds local count");
        assert!(send.is_empty(), "should send nothing when under target");
    }

    #[test]
    fn sfc_rebalance_plan_over_target_sends_excess() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements
        let n = mesh.n_elems();
        let target = n / 2;
        let (keep, send) = sfc_rebalance_plan(&mesh, target);
        assert_eq!(keep.len(), target, "should keep exactly target elements");
        assert_eq!(send.len(), n - target, "should send remaining elements");
    }

    #[test]
    fn sfc_rebalance_plan_elements_are_disjoint() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n = mesh.n_elems();
        let target = n / 2;
        let (keep, send) = sfc_rebalance_plan(&mesh, target);
        let mut all: Vec<usize> = keep.iter().chain(send.iter()).copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n, "keep + send should cover all elements without overlap");
        assert!(all.iter().all(|&i| i < n), "all indices should be in range");
    }

    #[test]
    fn extract_submesh_elements_preserves_connectivity() {
        let mesh = Mesh::<2>::unit_square_tri(3); // 18 elements
        let indices: Vec<usize> = (0..3).collect(); // first 3 elements
        let sub = extract_submesh_elements(&mesh, &indices);
        assert_eq!(sub.n_elems(), 3, "should have 3 elements");
        for (si, &mi) in indices.iter().enumerate() {
            let orig_nodes: Vec<u32> = mesh.element_nodes(mi as u32).iter().copied().collect();
            let sub_nodes: Vec<u32> = sub.element_nodes(si as u32).iter().copied().collect();
            assert_eq!(sub_nodes.len(), orig_nodes.len(), "elem {si} should have same npe");
        }
    }

    #[test]
    fn merge_two_meshes_concatenates() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let n = mesh.n_elems();
        let mid = n / 2;
        let left: Vec<usize> = (0..mid).collect();
        let right: Vec<usize> = (mid..n).collect();
        let mesh_a = extract_submesh_elements(&mesh, &left);
        let mesh_b = extract_submesh_elements(&mesh, &right);
        let merged = merge_two_meshes(&mesh_a, &mesh_b);
        assert_eq!(merged.n_elems(), n, "merged should have same total elements");
    }

    #[test]
    fn compute_centroids_simple_2d_triangle() {
        // A known triangle: (0,0), (1,0), (0,1) → centroid at (1/3, 1/3)
        let mesh = Mesh::<2> {
            coords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            conn: vec![0, 1, 2],
            elem_tags: vec![0],
            elem_type: fem_mesh::ElementType::Tri3,
            face_conn: Vec::new(),
            face_tags: Vec::new(),
            face_type: fem_mesh::ElementType::Line2,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: Vec::new(),
            edge_to_elem: Vec::new(),
            geometry: None, nc_vertex_view: None,
        };
        let centroids = compute_centroids_simple(&mesh);
        assert_eq!(centroids.len(), 1);
        assert!((centroids[0][0] - 1.0 / 3.0).abs() < 1e-14, "centroid x={}", centroids[0][0]);
        assert!((centroids[0][1] - 1.0 / 3.0).abs() < 1e-14, "centroid y={}", centroids[0][1]);
    }

    // ─── DerefineTree building tests ──────────────────────────────────────────

    #[test]
    fn build_derefine_tree_all_refined() {
        // unit_square_tri(1) = 2 triangles
        let mesh = Mesh::<2>::unit_square_tri(1);
        let n_elems = mesh.n_elems();
        assert_eq!(n_elems, 2, "unit_square_tri(1) should have 2 triangles");
        let marked: Vec<ElemId> = (0..n_elems as ElemId).collect();
        let midpoint_map = HashMap::new(); // positions are computed algorithmically
        let tree = build_derefine_tree_from_refine(&mesh, &marked, &midpoint_map);
        assert_eq!(tree.records.len(), 2, "2 parents");
        // Parent 0: first refined element → children at 0, 1, 2, 3
        let r0 = &tree.records[&0u32];
        assert_eq!(r0.children, [0, 1, 2, 3], "parent 0 children");
        // Parent 1: second refined element, 1 refined before → children at 4, 5, 6, 7
        let r1 = &tree.records[&1u32];
        assert_eq!(r1.children, [4, 5, 6, 7], "parent 1 children");
    }

    #[test]
    fn build_derefine_tree_partial_refine() {
        let mesh = Mesh::<2>::unit_square_tri(3); // 3 elements
        let marked: Vec<ElemId> = vec![0, 2]; // refine elements 0 and 2 only
        let midpoint_map = HashMap::new(); // content not important for position test
        let tree = build_derefine_tree_from_refine(&mesh, &marked, &midpoint_map);
        assert_eq!(tree.records.len(), 2, "2 parents");
        // Parent 0: first element, 0 refined before → children at 0, 1, 2, 3
        assert_eq!(tree.records[&0u32].children, [0, 1, 2, 3], "parent 0 children");
        // Parent 2: third element, 1 refined before (elem 0) → 2 + 3*1 = 5
        assert_eq!(tree.records[&2u32].children, [5, 6, 7, 8], "parent 2 children");
        // Element 1 (unrefined) has no record
        assert!(!tree.records.contains_key(&1u32), "elem 1 should not be in tree");
    }

    // ─── Derefine cycle tests (refine → derefine → original) ─────────────────

    #[test]
    fn refine_then_derefine_restores_element_count() {
        let (par_mesh, nc) = make_serial_par_mesh(2);
        let n_before = par_mesh.local_mesh().n_elems();
        let all_marked: Vec<ElemId> = (0..n_before as ElemId).collect();

        // Refine with tree
        let (refined, tree) =
            par_refine_marked_with_tree(&par_mesh, nc, &all_marked, None).unwrap();
        let n_refined = refined.par_mesh.local_mesh().n_elems();
        assert!(n_refined > n_before, "refined count {} > {}", n_refined, n_before);

        // Derefine all parents
        let parents: Vec<ElemId> = tree.parents();
        assert!(!parents.is_empty(), "should have parents to derefine");

        let mut nc2 = refined.nc_state;
        let result = par_derefine_marked(
            &refined.par_mesh,
            &mut nc2,
            &tree,
            &parents,
            None,
        ).unwrap();

        assert_eq!(
            result.par_mesh.local_mesh().n_elems(),
            n_before,
            "derefine should restore original element count"
        );
    }

    #[test]
    fn refine_then_derefine_partial() {
        // unit_square_tri(3) = 2*3*3 = 18 triangles
        let (par_mesh, nc) = make_serial_par_mesh(3);
        let n_before = par_mesh.local_mesh().n_elems();
        assert_eq!(n_before, 18, "unit_square_tri(3)");

        // Refine elements 0 and 2 only
        let marked: Vec<ElemId> = vec![0u32, 2u32];
        let (refined, tree) =
            par_refine_marked_with_tree(&par_mesh, nc, &marked, None).unwrap();
        let n_refined = refined.par_mesh.local_mesh().n_elems();

        // 2 refined (each →4) + 16 unrefined = 8 + 16 = 24
        assert_eq!(n_refined, 24, "refined: 2*4 + 16 unrefined");

        // Derefine both parents
        let mut nc2 = refined.nc_state;
        let result = par_derefine_marked(
            &refined.par_mesh,
            &mut nc2,
            &tree,
            &[0u32, 2u32],
            None,
        ).unwrap();

        assert_eq!(
            result.par_mesh.local_mesh().n_elems(),
            n_before,
            "should restore original count after partial derefine"
        );
    }

    #[test]
    fn derefine_preserves_solution() {
        let (par_mesh, nc) = make_serial_par_mesh(2);
        let solution: Vec<f64> = (0..par_mesh.local_mesh().n_nodes())
            .map(|i| i as f64 * 0.5)
            .collect();

        // Refine
        let all: Vec<ElemId> = (0..par_mesh.local_mesh().n_elems() as ElemId).collect();
        let (refined, tree) =
            par_refine_marked_with_tree(&par_mesh, nc, &all, Some(&solution)).unwrap();

        // Solution should have been prolongated (longer vector)
        assert!(refined.solution.len() >= solution.len());

        // Derefine
        let parents: Vec<ElemId> = tree.parents();
        let mut nc2 = refined.nc_state;
        let result = par_derefine_marked(
            &refined.par_mesh,
            &mut nc2,
            &tree,
            &parents,
            Some(&refined.solution),
        ).unwrap();

        // After derefine with compaction, solution length equals number of
        // active nodes (original coarse nodes, since all were derefined).
        assert_eq!(result.solution.len(), solution.len());

        // Corner node values should be preserved
        let n_original = solution.len();
        for i in 0..n_original {
            let diff = (result.solution[i] - solution[i]).abs();
            assert!(diff < 1e-14, "node {i}: expected {}, got {}", solution[i], result.solution[i]);
        }
    }

    #[test]
    fn par_refine_marked_with_tree_round_trip() {
        // Verify that par_refine_marked_with_tree returns a consistent tree.
        // unit_square_tri(2) = 8 elements
        let (par_mesh, nc) = make_serial_par_mesh(2);
        let n_elems = par_mesh.local_mesh().n_elems();
        assert_eq!(n_elems, 8, "unit_square_tri(2)");
        let all: Vec<ElemId> = (0..n_elems as ElemId).collect();
        let (_refined, tree) =
            par_refine_marked_with_tree(&par_mesh, nc, &all, None).unwrap();

        let parents = tree.parents();
        assert_eq!(parents.len(), n_elems, "all {} elements should have parent records", n_elems);
        for &p in &parents {
            let rec = tree.records.get(&p).unwrap();
            assert_eq!(rec.children.len(), 4, "each parent should have 4 children");
        }
    }

    // ─── cross-rank NC refine (rebuild_partition_nc) ────────────────────────

    /// Collect every rank's owned global elem ids on rank 0, sorted.
    fn gather_owned_elems(comm: &Comm, owned: &[u32]) -> Vec<u32> {
        if comm.is_root() {
            let mut all: Vec<u32> = owned.to_vec();
            for r in 1..comm.size() as i32 {
                all.extend(comm.recv::<u32>(r, NC_AMR_MARK_TAG + 100));
            }
            all.sort_unstable();
            all
        } else {
            comm.send(0, NC_AMR_MARK_TAG + 100, owned);
            Vec::new()
        }
    }

    /// Two-rank NC refine must produce the same global mesh as serial NC refine.
    #[test]
    fn par_refine_marked_two_ranks_matches_serial() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements, 25 nodes
        let marked_global: Vec<ElemId> = vec![0, 1, 2, 3, 16, 17, 18, 19];

        // Serial reference.
        let mut ser_nc = NCState::new();
        let (ser_refined, _, _) = ser_nc.refine(&mesh, &marked_global, 0);
        let n_ser_elems = ser_refined.n_elems();
        let n_ser_nodes = ser_refined.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let part = pmesh.partition();
            let n_local = part.n_owned_elems + part.n_ghost_elems;
            // Locally visible marked elements (owned + ghost, global consistency).
            let marked_local: Vec<ElemId> = (0..n_local)
                .filter(|&e| marked_global.contains(&part.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let nc = NCState::new();
            let res = par_refine_marked(&pmesh, nc, &marked_local, None).unwrap();
            let np = res.par_mesh.partition();

            // Global element/node counts match serial.
            let n_elems_g: usize =
                comm.allreduce_sum_i64(np.n_owned_elems as i64) as usize;
            let n_nodes_g: usize =
                comm.allreduce_sum_i64(np.n_owned_nodes as i64) as usize;
            assert_eq!(n_elems_g, n_ser_elems, "global elem count mismatch");
            assert_eq!(n_nodes_g, n_ser_nodes, "global node count mismatch");

            // Owned global elem ids partition [0, n_ser_elems) exactly once.
            let owned_gids: Vec<u32> = np.global_elem_ids[..np.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all.len(), n_ser_elems, "owned gid count");
                assert_eq!(all, (0..n_ser_elems as u32).collect::<Vec<_>>(),
                    "owned element gids must be a partition of the serial sequence");
            }

            // Same for nodes.
            let owned_nodes: Vec<u32> = np.global_node_ids[..np.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n.len(), n_ser_nodes, "owned node gid count");
                assert_eq!(all_n, (0..n_ser_nodes as u32).collect::<Vec<_>>(),
                    "owned node gids must be a partition of the serial sequence");
            }
        });
    }

    /// Two-rank refine followed by a second refine (marked children) keeps the
    /// same global mesh as two serial NC refine passes.
    #[test]
    fn par_refine_marked_two_rounds_matches_serial() {
        let mesh = Mesh::<2>::unit_square_tri(3); // 18 elements
        let marked_round1: Vec<ElemId> = vec![0, 1, 8, 9];

        // Serial reference: two consecutive NC refinements.
        let mut ser_nc = NCState::new();
        let (ser1, _, _) = ser_nc.refine(&mesh, &marked_round1, 0);
        // Mark a subset of round-1 children: the first child (id 0 → 0..4)
        // plus children of element 1 (ids 4..8).
        let marked_round2: Vec<ElemId> = vec![0, 1, 4, 5];
        let (ser2, _, _) = ser_nc.refine(&ser1, &marked_round2, 0);
        let n_ser = ser2.n_elems();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            // Round 1.
            let pmesh0 = partition_mesh(&mesh, &comm);
            let part0 = pmesh0.partition();
            let n0 = part0.n_owned_elems + part0.n_ghost_elems;
            let marked1: Vec<ElemId> = (0..n0)
                .filter(|&e| marked_round1.contains(&part0.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let nc1 = NCState::new();
            let r1 = par_refine_marked(&pmesh0, nc1, &marked1, None).unwrap();

            // Round 2 on the refined mesh.
            let part1 = r1.par_mesh.partition();
            let n1 = part1.n_owned_elems + part1.n_ghost_elems;
            let marked2: Vec<ElemId> = (0..n1)
                .filter(|&e| marked_round2.contains(&part1.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let r2 = par_refine_marked(&r1.par_mesh, r1.nc_state, &marked2, None).unwrap();
            let np2 = r2.par_mesh.partition();

            let n_elems_g: usize =
                comm.allreduce_sum_i64(np2.n_owned_elems as i64) as usize;
            assert_eq!(n_elems_g, n_ser, "two-round global elem count mismatch");
            let owned_gids: Vec<u32> = np2.global_elem_ids[..np2.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all.len(), n_ser);
                assert_eq!(all, (0..n_ser as u32).collect::<Vec<_>>());
            }
            // Nodes must also partition the serial node-id range.
            let n_ser_nodes = ser2.n_nodes();
            let owned_nodes: Vec<u32> = np2.global_node_ids[..np2.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n.len(), n_ser_nodes, "two-round owned node gid count");
                assert_eq!(all_n, (0..n_ser_nodes as u32).collect::<Vec<_>>(),
                    "two-round owned node gids must partition the serial sequence");
            }
        });
    }

    /// Regression: single round with a large mark set (pex6-like, 75% marked)
    /// must rebuild a consistent ghost layer.
    #[test]
    fn par_refine_marked_single_round_large_marks() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements

        let mut ser_nc = NCState::new();
        let round1_g: Vec<ElemId> = (0..24).collect(); // 75%
        let (ser1, _, _) = ser_nc.refine(&mesh, &round1_g, 0);
        let n_ser = ser1.n_elems();
        let n_ser_nodes = ser1.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh0 = partition_mesh(&mesh, &comm);
            let part0 = pmesh0.partition();
            let n0 = part0.n_owned_elems + part0.n_ghost_elems;
            let m1: Vec<ElemId> = (0..n0)
                .filter(|&e| round1_g.contains(&part0.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let r1 = par_refine_marked(&pmesh0, NCState::new(), &m1, None).unwrap();
            let np1 = r1.par_mesh.partition();

            let n_elems_g: usize =
                comm.allreduce_sum_i64(np1.n_owned_elems as i64) as usize;
            assert_eq!(n_elems_g, n_ser, "large-mark elem count mismatch");
            let owned_nodes: Vec<u32> = np1.global_node_ids[..np1.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n.len(), n_ser_nodes, "large-mark owned node gid count");
                assert_eq!(all_n, (0..n_ser_nodes as u32).collect::<Vec<_>>(),
                    "large-mark owned node gids must partition the serial sequence");
            }
        });
    }

    /// Regression: two rounds with large mark sets (pex6-like) must also
    /// rebuild a consistent ghost layer on the second round.
    #[test]
    fn par_refine_marked_two_rounds_large_marks() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements

        // Serial reference with the same mark fractions.
        let mut ser_nc = NCState::new();
        let round1_g: Vec<ElemId> = (0..24).collect(); // 75%
        let (ser1, _, _) = ser_nc.refine(&mesh, &round1_g, 0);
        let round2_g: Vec<ElemId> = (0..(ser1.n_elems() * 75 / 100) as ElemId).collect();
        let (ser2, _, _) = ser_nc.refine(&ser1, &round2_g, 0);
        let n_ser = ser2.n_elems();
        let n_ser_nodes = ser2.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh0 = partition_mesh(&mesh, &comm);
            let part0 = pmesh0.partition();
            let n0 = part0.n_owned_elems + part0.n_ghost_elems;
            let m1: Vec<ElemId> = (0..n0)
                .filter(|&e| round1_g.contains(&part0.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let r1 = par_refine_marked(&pmesh0, NCState::new(), &m1, None).unwrap();

            let part1 = r1.par_mesh.partition();
            let n1 = part1.n_owned_elems + part1.n_ghost_elems;
            let m2: Vec<ElemId> = (0..n1)
                .filter(|&e| round2_g.contains(&part1.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let r2 = par_refine_marked(&r1.par_mesh, r1.nc_state, &m2, None).unwrap();
            let np2 = r2.par_mesh.partition();

            let n_elems_g: usize =
                comm.allreduce_sum_i64(np2.n_owned_elems as i64) as usize;
            assert_eq!(n_elems_g, n_ser, "large-mark global elem count mismatch");
            let owned_gids: Vec<u32> = np2.global_elem_ids[..np2.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all, (0..n_ser as u32).collect::<Vec<_>>());
            }
            let owned_nodes: Vec<u32> = np2.global_node_ids[..np2.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n.len(), n_ser_nodes, "large-mark owned node gid count");
                assert_eq!(all_n, (0..n_ser_nodes as u32).collect::<Vec<_>>(),
                    "large-mark owned node gids must partition the serial sequence");
            }
        });
    }

    /// Regression: four rounds of large marks must keep running (pex6-like
    /// workload with cross-rank coarse-edge midpoints).
    #[test]
    fn par_refine_marked_four_rounds_large_marks() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let mut pmesh = partition_mesh(&mesh, &comm);
            let mut nc = NCState::new();
            let mut n_elems_g: usize = 0;
            for _round in 0..4 {
                let part = pmesh.partition();
                let n_local = part.n_owned_elems + part.n_ghost_elems;
                let n_global = pmesh.global_n_elems();
                // Mark 75% of the global element range (locally visible).
                let mark_upto = n_global * 3 / 4;
                let m: Vec<ElemId> = (0..n_local)
                    .filter(|&e| (part.global_elem(e as u32) as usize) < mark_upto)
                    .map(|e| e as ElemId)
                    .collect();
                let r = par_refine_marked(&pmesh, nc, &m, None).unwrap();
                n_elems_g = comm.allreduce_sum_i64(r.par_mesh.partition().n_owned_elems as i64) as usize;
                pmesh = r.par_mesh;
                nc = r.nc_state;
            }
            // Four rounds must complete and produce a strictly growing mesh.
            assert!(n_elems_g > 32 * 4, "expected strong growth, got {n_elems_g}");
        });
    }
}
