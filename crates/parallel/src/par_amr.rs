//! Parallel adaptive mesh refinement.
//!
//! Provides [`par_refine_marked`] for distributed non-conforming refinement and
//! [`par_repartition`] for load-rebalancing after refinement.

use std::collections::{HashMap, BTreeMap, BTreeSet, HashSet};

use fem_mesh::{Mesh, ElementType, amr::NCState, amr::DerefineTree, amr::DerefineRecord, amr::derefine_marked, topology::MeshTopology};
use fem_core::types::{ElemId, NodeId, Rank};

use crate::{par_mesh::ParallelMesh, partition::MeshPartition, Comm};

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
    // 5b. Cross-rank coarse-edge midpoints that this rank *reused* (rather
    // than created) must be reconciled with the global midpoint table.
    //
    // After a rebalance the partition changes: `NCState::refine`'s
    // `find_midpoint_node` coordinate fallback may match an *old* node M on
    // one rank (same midpoint coordinates, different gid — e.g. M is the
    // midpoint of another coarse edge) while the other rank creates a fresh
    // midpoint node.  `global_mid` (gathered from every rank) is
    // authoritative: every coarse edge's midpoint must be `global_mid[gk].0`
    // everywhere.  Remap the connectivity from the reused node to the
    // authoritative gid (appending it as an extra ghost when not present).
    let mut mid_remap: HashMap<u32, u32> = HashMap::new(); // local mid id → authoritative gid
    {
        let created_edges: HashSet<(u32, u32)> = midpoint_map
            .iter()
            .filter(|(_, &mid)| (mid as usize) >= n_orig)
            .map(|(&(a, b), _)| edge_key(gid_of(a), gid_of(b)))
            .collect();
        for (&(a, b), &mid) in midpoint_map {
            if (mid as usize) >= n_orig {
                continue; // we created it; its gid already agrees with global_mid
            }
            let gk = edge_key(gid_of(a), gid_of(b));
            if created_edges.contains(&gk) {
                continue;
            }
            let Some(&(target_gid, target_owner)) = global_mid.get(&gk) else {
                continue;
            };
            if gid_of(mid) == target_gid {
                continue; // already the authoritative midpoint
            }
            // The refined connectivity uses `mid`, but the global mesh uses
            // `target_gid`.  Make sure the target node is local (append as an
            // extra ghost if not) and remap the connectivity below.
            if partition.local_node(target_gid).is_none()
                && !extra_ghost.iter().any(|&(g, _, _, _)| g == target_gid)
            {
                let c = refined.coords_of(mid);
                extra_ghost.push((target_gid, target_owner, c[0], c[1]));
            }
            mid_remap.insert(mid, target_gid);
        }
    }
    extra_ghost.sort_unstable_by_key(|&(g, o, mx, my)| (g, o, mx.to_bits(), my.to_bits()));
    extra_ghost.dedup();
    if std::env::var("PEX6_TRACE").is_ok() && !mid_remap.is_empty() {
        eprintln!("[r{rank}] rebuild: mid_remap={mid_remap:?} extra_ghost_len={}",
            extra_ghost.len());
    }

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
    // Reordered-node lookup by global id (for mid_remap targets: a remapped
    // midpoint may live in the owned, ghost or extra-ghost segment).
    let mut gid_to_new: HashMap<u32, u32> = HashMap::with_capacity(n_total_new);
    for (new_id, &old_id) in order.iter().enumerate() {
        gid_to_new.insert(new_gid[old_id], new_id as u32);
    }
    let map_node = |n: u32| -> u32 {
        if let Some(&target_gid) = mid_remap.get(&n) {
            gid_to_new[&target_gid]
        } else {
            remap[n as usize]
        }
    };
    let new_conn: Vec<u32> = refined.conn.iter().map(|&n| map_node(n)).collect();
    let new_face_conn: Vec<u32> = refined.face_conn.iter().map(|&n| map_node(n)).collect();

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

/// Re-distribute elements across ranks for load rebalancing after AMR.
///
/// # Communication discipline
///
/// Every cross-rank exchange is a single collective `alltoallv` — each rank
/// broadcasts its local payload and all ranks call the same collective, so
/// there are **no** point-to-point root-gather/broadcast steps and ranks
/// cannot deadlock when they run at different speeds (the historical
/// root-collect version of this function was a pex6 deadlock source).
///
/// # Algorithm
///
/// 1. Each rank packs its *owned* elements `(gid, SFC key, tag, global conn)`,
///    its locally held boundary faces `(endpoint gids, tag)` and its local
///    node coordinates, and `alltoallv`-broadcasts them → every rank holds
///    the full element / face / node tables.
/// 2. The global element table is sorted by `(SFC key, gid)` — identical
///    input on every rank, so the order (and each element's global position)
///    is identical; the new owner is the evenly-divided interval of that
///    order (`±1` element per rank).
/// 3. Each rank keeps the elements whose new owner is itself as *owned* and
///    selects the *ghost* layer from the global table with the same rule as
///    [`crate::par_partition::partition_mesh`]: node-neighbours (share ≥ 1
///    node) plus one face-closure round (share ≥ 2 nodes), so every local
///    face sees both of its adjacent elements.
/// 4. Node ownership is recomputed as the owner of the smallest-gid element
///    referencing the node (a global invariant, identical on every rank from
///    the shared element table); boundary faces are assigned to the owner of
///    their adjacent element (pex9 semantics).
/// 5. The local mesh and [`MeshPartition`] are rebuilt with global element /
///    node ids **unchanged** (they match the serial mesh numbering), so a
///    subsequent [`par_refine_marked`] keeps working — rebalancing only moves
///    elements between ranks, it never renumbers the mesh.
///
/// # Supported meshes
///
/// 2-D `Tri3` / `Quad4` (the element types produced by pex6's NC refine);
/// mixed-type and 3-D meshes are rejected with
/// [`ParAmrError::RepartitionError`].
///
/// # Communication pattern (targeted, unlike the old full broadcast)
///
/// Only the small SFC keys (12 B/element) are exchanged full-broadcast; all
/// element/face/node payloads travel **targeted** to their new owner:
/// 1. **keys** — full alltoallv, every rank computes the global SFC order and
///    the new owner per element (identical on every rank);
/// 2. **elements** — sent only to their new owner (`gid | tag | conn`);
/// 3. **node refs** — each rank announces the nodes its owned elements
///    reference (with the referencing element) plus the local node coords to
///    the per-node *coordinator* `gid % size`;
/// 4. **reply** — coordinators send the full referencing set `E(n)`, the node
///    coords and the node owner (owner of the smallest-gid referencing
///    element) back to every referencing rank → node-neighbour ghosts;
/// 5. **requests** (nodes of ghost elements not yet covered) + boundary
///    **faces** (sent to the new owner of their adjacent element);
/// 6. **reply 2** → face-closure ghosts (elements sharing an edge with a
///    node-neighbour ghost);
/// 7./8. **request/reply 3** for nodes referenced only by face-closure ghosts.
///
/// Every step is a collective `alltoallv` (all ranks call it), so there is no
/// point-to-point root collection and no deadlock.  Element gids, the node
/// ownership rule (owner of the smallest-gid referencing element) and the
/// ghost layer are identical to the previous implementation.
pub fn par_repartition(
    par_mesh: ParallelMesh<Mesh<2>>,
) -> Result<ParallelMesh<Mesh<2>>, ParAmrError> {
    let comm = par_mesh.comm().clone();
    let size = comm.size();
    let rank = comm.rank();
    if size <= 1 {
        return Ok(par_mesh);
    }

    let local_mesh = par_mesh.local_mesh().clone();
    let partition = par_mesh.partition().clone();
    let elem_type = local_mesh.elem_type;
    let npe = elem_type.nodes_per_element();
    if !matches!(elem_type, ElementType::Tri3 | ElementType::Quad4) {
        return Err(ParAmrError::RepartitionError(format!(
            "par_repartition: unsupported element type {elem_type:?} (Tri3/Quad4 only)"
        )));
    }
    // Local element → its edges (canonical endpoint pairs, global ids).
    let edges_of = |conn: &[u32]| -> Vec<(u32, u32)> {
        match npe {
            3 => vec![
                edge_key(conn[0], conn[1]),
                edge_key(conn[1], conn[2]),
                edge_key(conn[2], conn[0]),
            ],
            _ => vec![
                edge_key(conn[0], conn[1]),
                edge_key(conn[1], conn[2]),
                edge_key(conn[2], conn[3]),
                edge_key(conn[3], conn[0]),
            ],
        }
    };
    let trace = std::env::var("PEX6_TRACE").is_ok();
    // Coordinator of a global node: every rank maps the same gid to the same
    // rank, so node-reference reductions are globally consistent.
    let coord_of = |node_gid: u32| -> Rank { (node_gid as usize % size) as Rank };

    // Small byte readers for the phase payloads (no captures).
    fn rd_u32(off: &mut usize, b: &[u8]) -> u32 {
        let s = &b[*off..*off + 4];
        *off += 4;
        u32::from_le_bytes(s.try_into().unwrap())
    }
    fn rd_u64(off: &mut usize, b: &[u8]) -> u64 {
        let s = &b[*off..*off + 8];
        *off += 8;
        u64::from_le_bytes(s.try_into().unwrap())
    }
    fn rd_f64(off: &mut usize, b: &[u8]) -> f64 {
        let s = &b[*off..*off + 8];
        *off += 8;
        f64::from_le_bytes(s.try_into().unwrap())
    }
    fn rd_vec(off: &mut usize, b: &[u8]) -> Vec<u32> {
        let n = rd_u32(off, b) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(rd_u32(off, b));
        }
        v
    }

    // Node info a rank needs about a node it references: coordinates, the
    // global owner (owner of the smallest-gid referencing element) and the
    // full referencing-element set E(n) with owner/tag/connectivity.
    struct NodeInfo {
        coords: [f64; 2],
        owner: i32,
        refs: Vec<(u32, i32, i32, Vec<u32>)>, // (elem gid, owner, tag, conn)
    }
    // Reply record sent by a coordinator for one node.
    struct NodeRec {
        node: u32,
        coords: [f64; 2],
        node_owner: i32,
        refs: Vec<(u32, i32, i32, Vec<u32>)>,
    }
    fn encode_node_records(reply_map: &HashMap<Rank, Vec<NodeRec>>) -> Vec<(Rank, Vec<u8>)> {
        reply_map
            .iter()
            .map(|(&d, list)| {
                let mut buf = Vec::new();
                buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for rec in list {
                    buf.extend_from_slice(&rec.node.to_le_bytes());
                    buf.extend_from_slice(&rec.coords[0].to_le_bytes());
                    buf.extend_from_slice(&rec.coords[1].to_le_bytes());
                    buf.extend_from_slice(&(rec.node_owner as u32).to_le_bytes());
                    buf.extend_from_slice(&(rec.refs.len() as u32).to_le_bytes());
                    for (eg, o, t, conn) in &rec.refs {
                        buf.extend_from_slice(&eg.to_le_bytes());
                        buf.extend_from_slice(&(*o as u32).to_le_bytes());
                        buf.extend_from_slice(&(*t as u32).to_le_bytes());
                        buf.extend_from_slice(&(conn.len() as u32).to_le_bytes());
                        for &gn in conn {
                            buf.extend_from_slice(&gn.to_le_bytes());
                        }
                    }
                }
                (d, buf)
            })
            .collect()
    }
    fn decode_node_replies(incoming: &[(Rank, Vec<u8>)], node_info: &mut HashMap<u32, NodeInfo>) {
        for (_src, bytes) in incoming {
            let mut off = 0usize;
            let n_records = rd_u32(&mut off, bytes) as usize;
            for _ in 0..n_records {
                let node = rd_u32(&mut off, bytes);
                let x = rd_f64(&mut off, bytes);
                let y = rd_f64(&mut off, bytes);
                let node_owner = rd_u32(&mut off, bytes) as i32;
                let n_refs = rd_u32(&mut off, bytes) as usize;
                let mut refs = Vec::with_capacity(n_refs);
                for _ in 0..n_refs {
                    let eg = rd_u32(&mut off, bytes);
                    let o = rd_u32(&mut off, bytes) as i32;
                    let t = rd_u32(&mut off, bytes) as i32;
                    let conn = rd_vec(&mut off, bytes);
                    refs.push((eg, o, t, conn));
                }
                node_info
                    .entry(node)
                    .or_insert(NodeInfo { coords: [x, y], owner: node_owner, refs });
            }
            debug_assert_eq!(off, bytes.len(), "par_repartition node reply overrun");
        }
    }

    // ── Phase 1: SFC keys (alltoallv, full broadcast — 12 B/elem) ──
    //  Each rank broadcasts its owned elements' (gid, key); every rank sorts
    //  the full set and computes the new owner per element (the same even
    //  split ±1 per rank as the previous implementation).  Only the small
    //  keys are exchanged — element connectivity/tags travel targeted below.
    let sfc_opts = crate::sfc::SfcOptions::default();
    let mut key_chunks: Vec<(u32, u64)> = Vec::with_capacity(partition.n_owned_elems);
    for e in 0..partition.n_owned_elems as ElemId {
        let gid = partition.global_elem(e);
        let ns = local_mesh.element_nodes(e);
        let mut c = [0.0f64; 2];
        for &n in ns {
            let nc = local_mesh.node_coords(n);
            c[0] += nc[0];
            c[1] += nc[1];
        }
        c[0] /= npe as f64;
        c[1] /= npe as f64;
        let key = crate::sfc::morton_code::<2>(&c, sfc_opts.bits_per_coord);
        key_chunks.push((gid, key));
    }
    let mut key_payload = Vec::with_capacity(key_chunks.len() * 12 + 4);
    key_payload.extend_from_slice(&(key_chunks.len() as u32).to_le_bytes());
    for (gid, key) in &key_chunks {
        key_payload.extend_from_slice(&gid.to_le_bytes());
        key_payload.extend_from_slice(&key.to_le_bytes());
    }
    let key_sends: Vec<(Rank, Vec<u8>)> =
        (0..size as i32).map(|r| (r, key_payload.clone())).collect();
    if trace {
        eprintln!("[r{rank}] par_repartition: phase1 keys ({} owned)", key_chunks.len());
    }
    let incoming = comm.alltoallv_bytes(&key_sends);
    let mut keys: Vec<(u32, u64)> = Vec::with_capacity(key_chunks.len() * size);
    for (_src, bytes) in &incoming {
        let mut off = 0usize;
        let n_elems = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_elems {
            let gid = rd_u32(&mut off, bytes);
            let key = rd_u64(&mut off, bytes);
            keys.push((gid, key));
        }
        debug_assert_eq!(off, bytes.len(), "par_repartition phase1 key overrun");
    }
    if keys.is_empty() {
        return Err(ParAmrError::RepartitionError(
            "par_repartition: empty global element set".into(),
        ));
    }
    let n_global = keys.len();
    let chunk = n_global.div_ceil(size);
    let n_first = n_global % size;
    let mut order: Vec<usize> = (0..n_global).collect();
    order.sort_by(|&i, &j| keys[i].1.cmp(&keys[j].1).then(keys[i].0.cmp(&keys[j].0)));
    let mut owner_of: HashMap<u32, i32> = HashMap::with_capacity(n_global);
    for (pos, &i) in order.iter().enumerate() {
        let o = if chunk == 1 {
            pos as i32
        } else if n_first == 0 || pos < n_first * chunk {
            (pos / chunk) as i32
        } else {
            (n_first + (pos - n_first * chunk) / (chunk - 1)) as i32
        };
        owner_of.insert(keys[i].0, o);
    }
    if trace {
        let mine = owner_of.values().filter(|&&o| o == rank).count();
        eprintln!("[r{rank}] par_repartition: phase1 done ({n_global} global, {mine} mine)");
    }

    // ── Phase 2: element migration (alltoallv, targeted to new owners) ──
    //  Local old owned elements are sent only to their new owner:
    //  `gid u32 | tag u32 | conn u32×npe`.  Each rank receives exactly the
    //  elements it will own.
    let mut elem_sends: HashMap<Rank, Vec<(u32, i32, Vec<u32>)>> = HashMap::new();
    for e in 0..partition.n_owned_elems as ElemId {
        let gid = partition.global_elem(e);
        let dest = owner_of[&gid];
        let ns = local_mesh.element_nodes(e);
        let gconn: Vec<u32> = ns.iter().map(|&n| partition.global_node(n)).collect();
        elem_sends
            .entry(dest)
            .or_default()
            .push((gid, local_mesh.elem_tags[e as usize], gconn));
    }
    let sends: Vec<(Rank, Vec<u8>)> = elem_sends
        .iter()
        .map(|(&d, list)| {
            let mut buf = Vec::with_capacity(list.len() * (8 + 4 * npe) + 4);
            buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for (gid, tag, conn) in list {
                buf.extend_from_slice(&gid.to_le_bytes());
                buf.extend_from_slice(&(*tag as u32).to_le_bytes());
                for &gn in conn {
                    buf.extend_from_slice(&gn.to_le_bytes());
                }
            }
            (d, buf)
        })
        .collect();
    if trace {
        eprintln!("[r{rank}] par_repartition: phase2 elems ({} owned → {} dests)",
            partition.n_owned_elems, elem_sends.len());
    }
    let incoming = comm.alltoallv_bytes(&sends);
    let mut owned_elems: Vec<(u32, i32, Vec<u32>)> = Vec::new();
    for (_src, bytes) in &incoming {
        let mut off = 0usize;
        let n_elems = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_elems {
            let gid = rd_u32(&mut off, bytes);
            let tag = rd_u32(&mut off, bytes) as i32;
            let mut conn = Vec::with_capacity(npe);
            for _ in 0..npe {
                conn.push(rd_u32(&mut off, bytes));
            }
            owned_elems.push((gid, tag, conn));
        }
        debug_assert_eq!(off, bytes.len(), "par_repartition phase2 element overrun");
    }
    owned_elems.sort_by_key(|e| e.0);
    if trace {
        eprintln!("[r{rank}] par_repartition: phase2 done ({} owned elems)", owned_elems.len());
    }

    // ── Phase 3: node-reference announcement + local node coords ──
    //  (alltoallv, targeted to the per-node coordinator `coord_of(gid)`).
    //  Announcement payload per destination:
    //    n_coords u32 | (gid u32 | x f64 | y f64)×n_coords
    //    n_refs   u32 | (node u32 | elem u32 | owner i32 | tag i32 |
    //                    nconn u32 | conn u32×nconn)×n_refs
    let mut ann: HashMap<Rank, (Vec<(u32, [f64; 2])>, Vec<(u32, u32, i32, i32, Vec<u32>)>)> =
        HashMap::new();
    for lid in 0..partition.n_total_nodes() as u32 {
        let gid = partition.global_node(lid);
        let c = local_mesh.node_coords(lid);
        ann.entry(coord_of(gid)).or_default().0.push((gid, [c[0], c[1]]));
    }
    for (gid, tag, conn) in &owned_elems {
        for &n in conn {
            ann.entry(coord_of(n))
                .or_default()
                .1
                .push((n, *gid, rank, *tag, conn.clone()));
        }
    }
    let sends: Vec<(Rank, Vec<u8>)> = ann
        .iter()
        .map(|(&d, (coords, refs))| {
            let mut buf = Vec::new();
            buf.extend_from_slice(&(coords.len() as u32).to_le_bytes());
            for (g, xy) in coords {
                buf.extend_from_slice(&g.to_le_bytes());
                buf.extend_from_slice(&xy[0].to_le_bytes());
                buf.extend_from_slice(&xy[1].to_le_bytes());
            }
            buf.extend_from_slice(&(refs.len() as u32).to_le_bytes());
            for (n, eg, o, t, conn) in refs {
                buf.extend_from_slice(&n.to_le_bytes());
                buf.extend_from_slice(&eg.to_le_bytes());
                buf.extend_from_slice(&(*o as u32).to_le_bytes());
                buf.extend_from_slice(&(*t as u32).to_le_bytes());
                buf.extend_from_slice(&(conn.len() as u32).to_le_bytes());
                for &gn in conn {
                    buf.extend_from_slice(&gn.to_le_bytes());
                }
            }
            (d, buf)
        })
        .collect();
    if trace {
        let n_refs: usize = ann.values().map(|(_, r)| r.len()).sum();
        eprintln!("[r{rank}] par_repartition: phase3 announce ({} coords, {n_refs} node-refs)",
            partition.n_total_nodes());
    }
    let incoming = comm.alltoallv_bytes(&sends);

    // Coordinator side: reduce the announcements into per-node tables.
    let mut coord_refs: HashMap<u32, Vec<(u32, i32, i32, Vec<u32>)>> = HashMap::new();
    let mut coord_coords: HashMap<u32, [f64; 2]> = HashMap::new();
    for (_src, bytes) in &incoming {
        let mut off = 0usize;
        let n_coords = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_coords {
            let gid = rd_u32(&mut off, bytes);
            let x = rd_f64(&mut off, bytes);
            let y = rd_f64(&mut off, bytes);
            coord_coords.entry(gid).or_insert([x, y]);
        }
        let n_refs = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_refs {
            let n = rd_u32(&mut off, bytes);
            let eg = rd_u32(&mut off, bytes);
            let o = rd_u32(&mut off, bytes) as i32;
            let t = rd_u32(&mut off, bytes) as i32;
            let conn = rd_vec(&mut off, bytes);
            coord_refs.entry(n).or_default().push((eg, o, t, conn));
        }
        debug_assert_eq!(off, bytes.len(), "par_repartition phase3 announce overrun");
    }

    // ── Phase 4: reply — coordinators send E(n) + coords + owner back to
    //  every referencing rank (alltoallv, targeted). ──
    let mut reply_map: HashMap<Rank, Vec<NodeRec>> = HashMap::new();
    for (n, refs) in &coord_refs {
        let coords = coord_coords.get(n).copied().unwrap_or([f64::NAN; 2]);
        let node_owner = refs
            .iter()
            .min_by_key(|(eg, _, _, _)| *eg)
            .map(|(_, o, _, _)| *o)
            .unwrap_or(rank);
        // Every referencing rank receives the *complete* referencing set
        // E(n) (not only its own entries) — receivers need the full set to
        // detect cross-rank neighbours.
        let mut by_owner: HashMap<i32, usize> = HashMap::new();
        for r in refs {
            *by_owner.entry(r.1).or_insert(0) += 1;
        }
        for o in by_owner.keys() {
            reply_map
                .entry(*o)
                .or_default()
                .push(NodeRec {
                    node: *n,
                    coords,
                    node_owner,
                    refs: refs.clone(),
                });
        }
    }
    let sends = encode_node_records(&reply_map);
    if trace {
        eprintln!("[r{rank}] par_repartition: phase4 reply ({} node records → {} dests)",
            reply_map.values().map(|l| l.len()).sum::<usize>(), reply_map.len());
    }
    let incoming = comm.alltoallv_bytes(&sends);
    let mut node_info: HashMap<u32, NodeInfo> = HashMap::new();
    decode_node_replies(&incoming, &mut node_info);
    if trace {
        eprintln!("[r{rank}] par_repartition: phase4 done ({} node infos)", node_info.len());
    }

    // ── Local ghost detection, pass 1: node-neighbours ──
    //  Every element in E(n) of a node referenced by our owned elements is a
    //  ghost (owner != rank).  Also build `elem_data` (gid → (tag, conn)) and
    //  `elem_owner_map` from the reference sets for later assembly.
    let mut elem_data: HashMap<u32, (i32, Vec<u32>)> = HashMap::new();
    let mut elem_owner_map: HashMap<u32, i32> = HashMap::new();
    for (gid, tag, conn) in &owned_elems {
        elem_data.insert(*gid, (*tag, conn.clone()));
        elem_owner_map.insert(*gid, rank);
    }
    for info in node_info.values() {
        for (eg, o, t, conn) in &info.refs {
            elem_data.entry(*eg).or_insert_with(|| (*t, conn.clone()));
            elem_owner_map.entry(*eg).or_insert(*o);
        }
    }
    let owned_set: HashSet<u32> = owned_elems.iter().map(|e| e.0).collect();
    let mut ghost_nn: HashSet<u32> = HashSet::new();
    for (gid, _tag, conn) in &owned_elems {
        for &n in conn {
            let info = node_info.get(&n).unwrap_or_else(|| {
                panic!("par_repartition: missing node info for owned-elem node {n} (elem {gid})")
            });
            for (eg, o, _t, _c) in &info.refs {
                if *o != rank {
                    ghost_nn.insert(*eg);
                }
            }
        }
    }

    // ── Phase 5: request node info for ghost-element nodes not yet covered,
    //  plus migrate boundary faces to the new owner of their adjacent
    //  element (alltoallv, targeted; one message per destination).
    //  Payload per destination:
    //    n_req u32 | (node u32)×n_req | n_faces u32 | (a u32 | b u32 | tag i32)×n_faces
    let mut req2: BTreeSet<u32> = BTreeSet::new();
    for &g in &ghost_nn {
        if let Some((_t, conn)) = elem_data.get(&g) {
            for &n in conn {
                if !node_info.contains_key(&n) {
                    req2.insert(n);
                }
            }
        }
    }
    let f_npe = local_mesh.face_type.nodes_per_element();
    let n_faces = if f_npe > 0 {
        local_mesh.face_conn.len() / f_npe
    } else {
        0
    };
    // Local old owned elements indexed by their edges (global gids); every
    // local boundary face is adjacent to exactly one of them.
    let mut old_edge_elem: HashMap<(u32, u32), u32> = HashMap::new();
    for e in 0..partition.n_owned_elems as ElemId {
        let gid = partition.global_elem(e);
        let ns = local_mesh.element_nodes(e);
        let gconn: Vec<u32> = ns.iter().map(|&n| partition.global_node(n)).collect();
        for ek in edges_of(&gconn) {
            old_edge_elem.entry(ek).or_insert(gid);
        }
    }
    let mut face_sends: HashMap<Rank, Vec<(u32, u32, i32)>> = HashMap::new();
    for fi in 0..n_faces {
        let a = partition.global_node(local_mesh.face_conn[fi * f_npe]);
        let b = partition.global_node(local_mesh.face_conn[fi * f_npe + 1]);
        let tag = local_mesh.face_tags.get(fi).copied().unwrap_or(0);
        let ek = (a.min(b), a.max(b));
        if let Some(&old_elem) = old_edge_elem.get(&ek) {
            let dest = owner_of[&old_elem];
            face_sends.entry(dest).or_default().push((a, b, tag));
        }
        // A boundary face without a local adjacent element cannot happen
        // (faces follow their adjacent element's owner); skip defensively.
    }
    let mut merged: HashMap<Rank, (Vec<u32>, Vec<(u32, u32, i32)>)> = HashMap::new();
    for &n in &req2 {
        merged.entry(coord_of(n)).or_default().0.push(n);
    }
    for (d, faces) in face_sends {
        merged.entry(d).or_default().1.extend(faces);
    }
    let sends: Vec<(Rank, Vec<u8>)> = merged
        .iter()
        .map(|(&d, (reqs, faces))| {
            let mut buf = Vec::new();
            buf.extend_from_slice(&(reqs.len() as u32).to_le_bytes());
            for &n in reqs {
                buf.extend_from_slice(&n.to_le_bytes());
            }
            buf.extend_from_slice(&(faces.len() as u32).to_le_bytes());
            for &(a, b, t) in faces {
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
                buf.extend_from_slice(&(t as u32).to_le_bytes());
            }
            (d, buf)
        })
        .collect();
    if trace {
        eprintln!("[r{rank}] par_repartition: phase5 requests ({}) + faces ({n_faces})",
            req2.len());
    }
    let incoming = comm.alltoallv_bytes(&sends);

    // ── Phase 6: reply to the phase-5 requests (alltoallv, targeted). ──
    let mut recv_faces: Vec<(u32, u32, i32)> = Vec::new();
    let mut req_replies: HashMap<Rank, Vec<NodeRec>> = HashMap::new();
    for (src, bytes) in &incoming {
        let mut off = 0usize;
        let n_req = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_req {
            let n = rd_u32(&mut off, bytes);
            if let Some(refs) = coord_refs.get(&n) {
                let coords = coord_coords.get(&n).copied().unwrap_or([f64::NAN; 2]);
                let node_owner = refs
                    .iter()
                    .min_by_key(|(eg, _, _, _)| *eg)
                    .map(|(_, o, _, _)| *o)
                    .unwrap_or(rank);
                req_replies
                    .entry(*src)
                    .or_default()
                    .push(NodeRec { node: n, coords, node_owner, refs: refs.clone() });
            }
            // Unknown requested node (no rank announced it) cannot happen for
            // nodes referenced by real elements; skip defensively.
        }
        let n_faces = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_faces {
            let a = rd_u32(&mut off, bytes);
            let b = rd_u32(&mut off, bytes);
            let t = rd_u32(&mut off, bytes) as i32;
            recv_faces.push((a, b, t));
        }
        debug_assert_eq!(off, bytes.len(), "par_repartition phase5/6 overrun");
    }
    let sends = encode_node_records(&req_replies);
    let incoming = comm.alltoallv_bytes(&sends);
    decode_node_replies(&incoming, &mut node_info);
    // Refresh elem_data with the newly arrived reference sets.
    for info in node_info.values() {
        for (eg, o, t, conn) in &info.refs {
            elem_data.entry(*eg).or_insert_with(|| (*t, conn.clone()));
            elem_owner_map.entry(*eg).or_insert(*o);
        }
    }
    if trace {
        eprintln!("[r{rank}] par_repartition: phase6 done ({} node infos, {} faces recv)",
            node_info.len(), recv_faces.len());
    }

    // ── Local ghost detection, pass 2: face closure ──
    //  Elements sharing an edge (≥ face_dim common nodes) with a
    //  node-neighbour ghost are ghosts too — the same single extra layer as
    //  the previous implementation (no recursion).  Elements sharing an edge
    //  with an *owned* element are already node-neighbours, so only the
    //  ghost elements' edges need checking.
    let mut ghost_fc: HashSet<u32> = HashSet::new();
    for &g in &ghost_nn {
        if let Some((_t, conn)) = elem_data.get(&g) {
            for (a, b) in edges_of(conn) {
                let (Some(ia), Some(ib)) = (node_info.get(&a), node_info.get(&b)) else {
                    continue; // both edges are covered by req2/reply above
                };
                let ea: HashSet<u32> = ia.refs.iter().map(|r| r.0).collect();
                for (eg, o, _t, _c) in &ib.refs {
                    if *o != rank
                        && ea.contains(eg)
                        && !owned_set.contains(eg)
                        && !ghost_nn.contains(eg)
                    {
                        ghost_fc.insert(*eg);
                    }
                }
            }
        }
    }

    // ── Phases 7/8: request + reply for nodes referenced only by
    //  face-closure ghosts (alltoallv, targeted). ──
    let mut req3: BTreeSet<u32> = BTreeSet::new();
    for &g in &ghost_fc {
        if let Some((_t, conn)) = elem_data.get(&g) {
            for &n in conn {
                if !node_info.contains_key(&n) {
                    req3.insert(n);
                }
            }
        }
    }
    let sends: Vec<(Rank, Vec<u8>)> = {
        let mut m: HashMap<Rank, Vec<u32>> = HashMap::new();
        for &n in &req3 {
            m.entry(coord_of(n)).or_default().push(n);
        }
        m.into_iter()
            .map(|(d, reqs)| {
                let mut buf = Vec::new();
                buf.extend_from_slice(&(reqs.len() as u32).to_le_bytes());
                for &n in &reqs {
                    buf.extend_from_slice(&n.to_le_bytes());
                }
                buf.extend_from_slice(&0u32.to_le_bytes()); // no faces
                (d, buf)
            })
            .collect()
    };
    let incoming = comm.alltoallv_bytes(&sends);
    let mut req_replies: HashMap<Rank, Vec<NodeRec>> = HashMap::new();
    for (src, bytes) in &incoming {
        let mut off = 0usize;
        let n_req = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_req {
            let n = rd_u32(&mut off, bytes);
            if let Some(refs) = coord_refs.get(&n) {
                let coords = coord_coords.get(&n).copied().unwrap_or([f64::NAN; 2]);
                let node_owner = refs
                    .iter()
                    .min_by_key(|(eg, _, _, _)| *eg)
                    .map(|(_, o, _, _)| *o)
                    .unwrap_or(rank);
                req_replies
                    .entry(*src)
                    .or_default()
                    .push(NodeRec { node: n, coords, node_owner, refs: refs.clone() });
            }
        }
        // No faces travel in the phase-7 request round; consume the (empty)
        // trailing face segment to keep the payload format uniform.
        let n_faces = rd_u32(&mut off, bytes) as usize;
        for _ in 0..n_faces {
            let _a = rd_u32(&mut off, bytes);
            let _b = rd_u32(&mut off, bytes);
            let _t = rd_u32(&mut off, bytes);
        }
        debug_assert_eq!(off, bytes.len(), "par_repartition phase7 overrun");
    }
    let sends = encode_node_records(&req_replies);
    let incoming = comm.alltoallv_bytes(&sends);
    decode_node_replies(&incoming, &mut node_info);
    for info in node_info.values() {
        for (eg, o, t, conn) in &info.refs {
            elem_data.entry(*eg).or_insert_with(|| (*t, conn.clone()));
            elem_owner_map.entry(*eg).or_insert(*o);
        }
    }
    if trace {
        eprintln!("[r{rank}] par_repartition: phases 7/8 done ({} node infos total)",
            node_info.len());
    }

    // ── Final assembly: local mesh + partition (ids unchanged) ──
    //  All local elements: owned + node-neighbour ghosts + face-closure
    //  ghosts, ordered by gid (owned first, matching the old layout).
    let mut all_elems: BTreeMap<u32, (i32, Vec<u32>)> = BTreeMap::new();
    for (gid, tag, conn) in &owned_elems {
        all_elems.insert(*gid, (*tag, conn.clone()));
    }
    for &g in ghost_nn.iter().chain(ghost_fc.iter()) {
        if let Some((t, conn)) = elem_data.get(&g) {
            all_elems.entry(g).or_insert_with(|| (*t, conn.clone()));
        } else {
            panic!("par_repartition: ghost elem {g} has no connectivity");
        }
    }
    let owned_gids: Vec<u32> = owned_elems.iter().map(|e| e.0).collect();
    let ghost_gids: Vec<u32> = {
        let mut v: Vec<u32> = all_elems
            .keys()
            .filter(|g| !owned_set.contains(g))
            .copied()
            .collect();
        v.sort_unstable();
        v
    };
    let ghost_owners: Vec<(u32, i32)> = ghost_gids
        .iter()
        .map(|&g| (g, elem_owner_map[&g]))
        .collect();

    // Node set of all local elements; every referenced node must have been
    // covered by the replies above.
    let mut all_node_set: BTreeSet<u32> = BTreeSet::new();
    for (_gid, (_t, conn)) in &all_elems {
        for &n in conn {
            all_node_set.insert(n);
        }
    }
    let owned_nodes: Vec<u32> = all_node_set
        .iter()
        .filter(|&&g| node_info[&g].owner == rank)
        .copied()
        .collect();
    let ghost_nodes: Vec<(u32, i32)> = all_node_set
        .iter()
        .filter(|&&g| node_info[&g].owner != rank)
        .map(|&g| (g, node_info[&g].owner))
        .collect();
    let mut g2l: HashMap<u32, u32> =
        HashMap::with_capacity(owned_nodes.len() + ghost_nodes.len());
    for (lid, &g) in owned_nodes.iter().enumerate() {
        g2l.insert(g, lid as u32);
    }
    let ghost_base = owned_nodes.len();
    for (idx, &(g, _)) in ghost_nodes.iter().enumerate() {
        g2l.insert(g, (ghost_base + idx) as u32);
    }
    let mut coords = Vec::with_capacity((owned_nodes.len() + ghost_nodes.len()) * 2);
    for &g in owned_nodes.iter().chain(ghost_nodes.iter().map(|(g, _)| g)) {
        let xy = node_info[&g].coords;
        coords.push(xy[0]);
        coords.push(xy[1]);
    }
    let n_local_elems = all_elems.len();
    let mut conn = Vec::with_capacity(n_local_elems * npe);
    let mut elem_tags = Vec::with_capacity(n_local_elems);
    // Element order must match the partition's `[owned | ghost]` segments
    // (each gid-ascending) — not the gid-interleaved BTreeMap iteration.
    for gid in owned_gids.iter().chain(ghost_gids.iter()) {
        let (tag, econn) = &all_elems[gid];
        for &n in econn {
            conn.push(g2l[&n]);
        }
        elem_tags.push(*tag);
    }
    // Boundary faces received in phase 5 (global gids → local ids).
    let mut face_conn = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    for (a, b, tag) in &recv_faces {
        face_conn.push(g2l[a]);
        face_conn.push(g2l[b]);
        face_tags.push(*tag);
    }

    // ── Partition rebuild (global ids unchanged) ──
    let mesh = Mesh::uniform(
        coords, conn, elem_tags, elem_type,
        face_conn, face_tags, local_mesh.face_type,
    );
    let new_partition = MeshPartition::from_partitioner(
        &owned_nodes,
        &ghost_nodes,
        &owned_gids,
        &ghost_owners,
        rank,
    );
    if trace {
        eprintln!("[r{rank}] par_repartition: owned_nodes={} ghost_nodes={} owned_elems={} ghost_elems={}",
            owned_nodes.len(), ghost_nodes.len(), owned_gids.len(), ghost_gids.len());
    }
    Ok(ParallelMesh::new(mesh, comm, new_partition))
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

    // ─── par_repartition (SFC alltoallv rebalancing) ──────────────────────────

    /// Two-rank rebalancing keeps the global mesh (element/node ids +
    /// element connectivity) identical to the serial input mesh, and balances
    /// the owned-element counts.
    #[test]
    fn par_repartition_two_ranks_preserves_global_ids() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements, 25 nodes
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let rb = par_repartition(pmesh).unwrap();
            let np = rb.partition();
            let lm = rb.local_mesh();

            // Global counts unchanged.
            let n_elems_g: usize =
                comm.allreduce_sum_i64(np.n_owned_elems as i64) as usize;
            let n_nodes_g: usize =
                comm.allreduce_sum_i64(np.n_owned_nodes as i64) as usize;
            assert_eq!(n_elems_g, 32, "global elem count: got {n_elems_g}");
            assert_eq!(n_nodes_g, 25, "global node count: got {n_nodes_g}");

            // Owned element gids partition [0, 32).
            let owned_gids: Vec<u32> = np.global_elem_ids[..np.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all, (0..32).collect::<Vec<_>>(),
                    "owned element gids must be a partition of the serial sequence");
            }
            // Owned node gids partition [0, 25).
            let owned_nodes: Vec<u32> = np.global_node_ids[..np.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n, (0..25).collect::<Vec<_>>(),
                    "owned node gids must be a partition of the serial sequence");
            }

            // Balance: with 32 elements and 2 ranks the SFC split is exactly
            // 16/16, so every rank must hold at least 16.
            let both_full = comm.allreduce_sum_i64((np.n_owned_elems >= 16) as i64);
            assert_eq!(both_full, 2, "both ranks should hold >= 16 elements");

            // Element connectivity (in global ids) must equal the serial mesh.
            let mut payload: Vec<u32> = Vec::new();
            for e in 0..np.n_owned_elems as ElemId {
                let g = np.global_elem(e);
                let ns = lm.element_nodes(e);
                let mut gc: Vec<u32> = ns.iter().map(|&n| np.global_node(n)).collect();
                gc.sort_unstable();
                payload.push(g);
                payload.extend_from_slice(&gc);
            }
            let mut flat: Vec<u32> = Vec::new();
            if comm.is_root() {
                flat = payload.clone();
                for r in 1..comm.size() as i32 {
                    flat.extend(comm.recv::<u32>(r, NC_AMR_MARK_TAG + 200));
                }
            } else {
                comm.send(0, NC_AMR_MARK_TAG + 200, &payload);
            }
            if comm.is_root() {
                assert_eq!(flat.len(), 32 * 4, "payload length");
                let mut serial: Vec<u32> = Vec::new();
                for g in 0..32u32 {
                    let mut gc: Vec<u32> = mesh.element_nodes(g).iter().copied().collect();
                    gc.sort_unstable();
                    serial.push(g);
                    serial.extend_from_slice(&gc);
                }
                assert_eq!(flat, serial, "element connectivity must match the serial mesh");
            }
        });
    }

    /// Two-rank rebalancing of a Quad4 mesh keeps the global element/node ids
    /// a partition of the serial range.
    #[test]
    fn par_repartition_two_ranks_quad4() {
        let mesh = Mesh::<2>::unit_square_quad(2); // 4 quads, 9 nodes
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let rb = par_repartition(pmesh).unwrap();
            let np = rb.partition();

            let n_elems_g: usize =
                comm.allreduce_sum_i64(np.n_owned_elems as i64) as usize;
            let n_nodes_g: usize =
                comm.allreduce_sum_i64(np.n_owned_nodes as i64) as usize;
            assert_eq!(n_elems_g, 4, "quad global elem count");
            assert_eq!(n_nodes_g, 9, "quad global node count");

            let owned_gids: Vec<u32> = np.global_elem_ids[..np.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all, (0..4).collect::<Vec<_>>(),
                    "quad owned element gids must partition the serial range");
            }
            let owned_nodes: Vec<u32> = np.global_node_ids[..np.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n, (0..9).collect::<Vec<_>>(),
                    "quad owned node gids must partition the serial range");
            }
        });
    }

    /// Rebalancing after NC refinement must preserve the refined global mesh
    /// (element/node gid coverage matches the serial NC-refine sequence).
    #[test]
    fn par_repartition_after_refine_two_ranks() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements
        // Serial reference: refine the first 75% (elements 0..24).
        let mut ser_nc = NCState::new();
        let marked_g: Vec<ElemId> = (0..24).collect();
        let (ser_refined, _, _) = ser_nc.refine(&mesh, &marked_g, 0);
        let n_ser = ser_refined.n_elems();
        let n_ser_nodes = ser_refined.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh0 = partition_mesh(&mesh, &comm);
            let part0 = pmesh0.partition();
            let n0 = part0.n_owned_elems + part0.n_ghost_elems;
            let m1: Vec<ElemId> = (0..n0)
                .filter(|&e| marked_g.contains(&part0.global_elem(e as u32)))
                .map(|e| e as ElemId)
                .collect();
            let r1 = par_refine_marked(&pmesh0, NCState::new(), &m1, None).unwrap();

            // Rebalance the refined mesh and re-check the global invariants.
            let rb = par_repartition(r1.par_mesh).unwrap();
            let np = rb.partition();

            let n_elems_g: usize =
                comm.allreduce_sum_i64(np.n_owned_elems as i64) as usize;
            let n_nodes_g: usize =
                comm.allreduce_sum_i64(np.n_owned_nodes as i64) as usize;
            assert_eq!(n_elems_g, n_ser, "rebalanced global elem count");
            assert_eq!(n_nodes_g, n_ser_nodes, "rebalanced global node count");

            let owned_gids: Vec<u32> = np.global_elem_ids[..np.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all, (0..n_ser as u32).collect::<Vec<_>>(),
                    "rebalanced owned element gids must partition the serial sequence");
            }
            let owned_nodes: Vec<u32> = np.global_node_ids[..np.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n, (0..n_ser_nodes as u32).collect::<Vec<_>>(),
                    "rebalanced owned node gids must partition the serial sequence");
            }
        });
    }

    /// A rebalanced mesh must remain usable by `par_refine_marked` — refine
    /// a rebalanced partition and verify global gid coverage still holds.
    #[test]
    fn par_refine_after_rebalance_two_ranks() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh0 = partition_mesh(&mesh, &comm);
            // Rebalance once (elements move between ranks).
            let rb = par_repartition(pmesh0).unwrap();

            // Refine the first 75% of the global range on the rebalanced mesh.
            let part = rb.partition();
            let n_local = part.n_owned_elems + part.n_ghost_elems;
            let n_global = rb.global_n_elems();
            let mark_upto = n_global * 3 / 4;
            let m: Vec<ElemId> = (0..n_local)
                .filter(|&e| (part.global_elem(e as u32) as usize) < mark_upto)
                .map(|e| e as ElemId)
                .collect();
            let r = par_refine_marked(&rb, NCState::new(), &m, None).unwrap();
            let np = r.par_mesh.partition();
            let n_elems_g: usize =
                comm.allreduce_sum_i64(np.n_owned_elems as i64) as usize;
            // 24 refined (×4) + 8 unrefined = 104.
            assert_eq!(n_elems_g, 24 * 4 + 8, "refine after rebalance count");
            let owned_gids: Vec<u32> = np.global_elem_ids[..np.n_owned_elems].to_vec();
            let all = gather_owned_elems(&comm, &owned_gids);
            if comm.is_root() {
                assert_eq!(all, (0..(24 * 4 + 8) as u32).collect::<Vec<_>>(),
                    "refine-after-rebalance gid coverage");
            }
        });
    }

    /// pex6-style workload: two rounds of refine(75%) + rebalance must keep
    /// node-owner consistency (every rank's ghost requests are satisfiable by
    /// the owning rank) — regression for the GhostExchange panic seen when
    /// rebalancing a mesh with cross-rank NC midpoints.
    #[test]
    fn pex6_style_refine_rebalance_refine_rebalance_two_ranks() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let mut pmesh = partition_mesh(&mesh, &comm);
            let mut nc = NCState::new();
            for _round in 0..2 {
                let part = pmesh.partition();
                let n_local = part.n_owned_elems + part.n_ghost_elems;
                let n_global = pmesh.global_n_elems();
                let mark_upto = n_global * 3 / 4;
                let m: Vec<ElemId> = (0..n_local)
                    .filter(|&e| (part.global_elem(e as u32) as usize) < mark_upto)
                    .map(|e| e as ElemId)
                    .collect();
                let r = par_refine_marked(&pmesh, nc, &m, None).unwrap();
                nc = r.nc_state;
                let rb = par_repartition(r.par_mesh).unwrap();
                pmesh = rb;
            }
            // Final consistency: owned node gids partition the global range.
            let np = pmesh.partition();
            let n_nodes_g: usize =
                comm.allreduce_sum_i64(np.n_owned_nodes as i64) as usize;
            let owned_nodes: Vec<u32> = np.global_node_ids[..np.n_owned_nodes].to_vec();
            let all_n = gather_owned_elems(&comm, &owned_nodes);
            if comm.is_root() {
                assert_eq!(all_n.len(), n_nodes_g, "owned node gid count");
                assert_eq!(all_n, (0..n_nodes_g as u32).collect::<Vec<_>>(),
                    "owned node gids must partition the serial sequence");
            }
        });
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
