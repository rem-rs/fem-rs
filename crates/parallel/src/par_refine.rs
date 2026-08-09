//! Distributed uniform refinement for 2-D parallel meshes.
//!
//! [`par_uniform_refine`] refines **every** element of a
//! [`ParallelMesh<Mesh<2>>`](crate::par_mesh::ParallelMesh) across ranks — the
//! distributed counterpart of [`fem_mesh::refine_uniform`].  Each element is
//! split 1 → 4 and the children inherit the parent's partition ownership, so
//! the partition topology (which rank owns which region) is preserved.
//!
//! # Global consistency
//!
//! * **Element ids**: child `k` of parent `g` gets `4·g + k` — identical on
//!   every rank that sees the parent (owned or ghost).
//! * **Edge-midpoint nodes**: the id is assigned by the *first* element that
//!   references the edge (smallest global element id), matching the traversal
//!   order of the serial [`fem_mesh::refine_uniform`].  The *owner* of that
//!   first element assigns the id from a prefix-summed range starting at
//!   `n_global_old_nodes`; other ranks that see the edge (through ghost
//!   elements) learn the id via an `alltoallv` request/reply.  Hence a
//!   cross-rank edge gets a single, identical global id on both sides.
//! * **Quad4 element-center nodes**: one per parent element; its id is
//!   `n_global_old_nodes + n_global_edges + parent_gid` — a pure function of
//!   the parent's global id, so no communication is needed.
//!
//! The result is a refined global mesh identical to serially refining the full
//! mesh with [`fem_mesh::refine_uniform`] and then partitioning — including
//! the global node numbering (single rank: bit-for-bit identical to serial).
//!
//! # Supported input
//!
//! * 2-D meshes with a **single** element type: `Tri3` (red refinement,
//!   1 → 4) or `Quad4` (conforming split).  Mixed-type meshes are not yet
//!   supported.
//! * **Compact** node partitions ([`partition_mesh`]); identity-node
//!   partitions ([`partition_mesh_identity`]) are not yet supported here.
//!
//! [`partition_mesh`]: crate::par_partition::partition_mesh
//! [`partition_mesh_identity`]: crate::par_partition::partition_mesh_identity

use std::collections::{BTreeMap, BTreeSet, HashMap};

use fem_core::{ElemId, NodeId, Rank};
use fem_mesh::{BoundaryTag, ElementType, Mesh};

use crate::comm::Comm;
use crate::par_mesh::ParallelMesh;
use crate::partition::MeshPartition;

/// Tag base for the midpoint-id request/reply exchanges.
const REFINE_GID_TAG: i32 = 0x3A00;

// ─── local refinement ────────────────────────────────────────────────────────

/// Result of refining the local sub-mesh.
struct LocalRefine {
    /// Refined mesh: nodes `[0..n_orig)` are the old nodes (same order),
    /// nodes `[n_orig..)` are new (edge midpoints sorted by key, then quad
    /// centers in parent-element order).  Elements are `4` consecutive
    /// children per parent, parents in local element order (owned first,
    /// then ghost).
    mesh: Mesh<2>,
    /// Sorted edge key `(min, max)` of *local* node ids → local id of the
    /// new midpoint node.
    edge_mid: HashMap<(u32, u32), u32>,
    /// Number of old nodes in the input mesh.
    n_orig: usize,
    /// Number of unique edges (== number of edge-midpoint nodes).
    n_edges: usize,
}

#[inline]
fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Refine a pure-Tri3 or pure-Quad4 2-D mesh (all elements 1 → 4).
///
/// The child layout matches the serial [`fem_mesh::refine_uniform`] exactly:
/// * Tri3  (red refinement): `[n0,m01,m02]`, `[m01,n1,m12]`, `[m02,m12,n2]`,
///   `[m01,m12,m02]`.
/// * Quad4: `[v0,e0,c,e3]`, `[e0,v1,e1,c]`, `[c,e1,v2,e2]`, `[e3,c,e2,v3]`.
fn refine_local(mesh: &Mesh<2>) -> LocalRefine {
    match mesh.elem_type {
        ElementType::Tri3 => refine_local_tri3(mesh),
        ElementType::Quad4 => refine_local_quad4(mesh),
        other => panic!(
            "par_uniform_refine: only Tri3/Quad4 supported, got {other:?}"
        ),
    }
}

fn refine_local_tri3(mesh: &Mesh<2>) -> LocalRefine {
    let n_orig = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // Unique edges of the whole local mesh (owned + ghost elements).
    let mut edge_set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        edge_set.insert(key(ns[0], ns[1]));
        edge_set.insert(key(ns[1], ns[2]));
        edge_set.insert(key(ns[2], ns[0]));
    }
    let n_edges = edge_set.len();

    // Edge midpoints: new node ids `n_orig + idx` in sorted-key order.
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::with_capacity(n_edges);
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    for (idx, &(a, b)) in edge_set.iter().enumerate() {
        let mid = (n_orig + idx) as u32;
        edge_mid.insert((a, b), mid);
        let ca = mesh.coords_of(a);
        let cb = mesh.coords_of(b);
        new_coords.push((ca[0] + cb[0]) * 0.5);
        new_coords.push((ca[1] + cb[1]) * 0.5);
    }

    // Children (red refinement, MFEM Tri3 pattern).
    let mut new_conn = Vec::with_capacity(n_elems * 4 * 3);
    let mut new_tags = Vec::with_capacity(n_elems * 4);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let m01 = edge_mid[&key(ns[0], ns[1])];
        let m12 = edge_mid[&key(ns[1], ns[2])];
        let m02 = edge_mid[&key(ns[2], ns[0])];
        let tag = mesh.elem_tags[e as usize];
        new_conn.extend_from_slice(&[ns[0], m01, m02]);
        new_conn.extend_from_slice(&[m01, ns[1], m12]);
        new_conn.extend_from_slice(&[m02, m12, ns[2]]);
        new_conn.extend_from_slice(&[m01, m12, m02]);
        new_tags.extend_from_slice(&[tag, tag, tag, tag]);
    }

    // Boundary faces: split each old face at its midpoint.
    let mut new_face_conn = Vec::new();
    let mut new_face_tags = Vec::new();
    for f in 0..mesh.n_faces() {
        let a = mesh.face_conn[f * 2];
        let b = mesh.face_conn[f * 2 + 1];
        let tag = mesh.face_tags[f];
        if let Some(&mid) = edge_mid.get(&key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]);
            new_face_conn.extend_from_slice(&[mid, b]);
            new_face_tags.extend_from_slice(&[tag, tag]);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let refined = Mesh::uniform(
        new_coords,
        new_conn,
        new_tags,
        ElementType::Tri3,
        new_face_conn,
        new_face_tags,
        ElementType::Line2,
    );
    LocalRefine { mesh: refined, edge_mid, n_orig, n_edges }
}

fn refine_local_quad4(mesh: &Mesh<2>) -> LocalRefine {
    let n_orig = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

    // Unique edges.
    let mut edge_set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(li, lj) in &QUAD_EDGES {
            edge_set.insert(key(ns[li], ns[lj]));
        }
    }
    let n_edges = edge_set.len();

    // New node layout: [old | edge midpoints | element centers].
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::with_capacity(n_edges);
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    for (idx, &(a, b)) in edge_set.iter().enumerate() {
        let mid = (n_orig + idx) as u32;
        edge_mid.insert((a, b), mid);
        let ca = mesh.coords_of(a);
        let cb = mesh.coords_of(b);
        new_coords.push((ca[0] + cb[0]) * 0.5);
        new_coords.push((ca[1] + cb[1]) * 0.5);
    }
    let mut center_local: Vec<u32> = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let cid = (n_orig + n_edges + e) as u32;
        center_local.push(cid);
        let mut sx = 0.0;
        let mut sy = 0.0;
        for &n in ns {
            let c = mesh.coords_of(n);
            sx += c[0];
            sy += c[1];
        }
        new_coords.push(sx / 4.0);
        new_coords.push(sy / 4.0);
    }

    // Children (MFEM Quad4 pattern, shared parent center).
    let mut new_conn = Vec::with_capacity(n_elems * 4 * 4);
    let mut new_tags = Vec::with_capacity(n_elems * 4);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let v = [ns[0], ns[1], ns[2], ns[3]];
        let c = center_local[e as usize];
        let e_mid: [u32; 4] = core::array::from_fn(|li| {
            let (a, b) = QUAD_EDGES[li];
            edge_mid[&key(v[a], v[b])]
        });
        let tag = mesh.elem_tags[e as usize];
        new_conn.extend_from_slice(&[v[0], e_mid[0], c, e_mid[3]]);
        new_conn.extend_from_slice(&[e_mid[0], v[1], e_mid[1], c]);
        new_conn.extend_from_slice(&[c, e_mid[1], v[2], e_mid[2]]);
        new_conn.extend_from_slice(&[e_mid[3], c, e_mid[2], v[3]]);
        new_tags.extend_from_slice(&[tag, tag, tag, tag]);
    }

    // Boundary faces.
    let mut new_face_conn = Vec::new();
    let mut new_face_tags = Vec::new();
    for f in 0..mesh.n_faces() {
        let a = mesh.face_conn[f * 2];
        let b = mesh.face_conn[f * 2 + 1];
        let tag = mesh.face_tags[f];
        if let Some(&mid) = edge_mid.get(&key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]);
            new_face_conn.extend_from_slice(&[mid, b]);
            new_face_tags.extend_from_slice(&[tag, tag]);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let refined = Mesh::uniform(
        new_coords,
        new_conn,
        new_tags,
        ElementType::Quad4,
        new_face_conn,
        new_face_tags,
        ElementType::Line2,
    );
    LocalRefine { mesh: refined, edge_mid, n_orig, n_edges }
}

// ─── public entry point ──────────────────────────────────────────────────────

/// Uniformly refine a partitioned 2-D mesh in parallel.
///
/// Every element (owned and ghost) is split 1 → 4; children keep the parent's
/// owner.  New edge-midpoint nodes get globally consistent ids via the
/// coordinator scheme described in the module docs, so the refined *global*
/// mesh matches a serial [`fem_mesh::refine_uniform`] of the full mesh.
pub fn par_uniform_refine(par_mesh: &ParallelMesh<Mesh<2>>) -> ParallelMesh<Mesh<2>> {
    let partition = par_mesh.partition();
    let local_mesh = par_mesh.local_mesh();
    let comm = par_mesh.comm().clone();

    assert!(
        !partition.node_id_identity,
        "par_uniform_refine: identity-node partitions are not supported yet \
         (use the default compact partition_mesh)"
    );

    // 1. Refine the local mesh (owned + ghost elements).
    let lr = refine_local(local_mesh);
    let refined = lr.mesh;
    let n_orig = lr.n_orig;
    let n_owned_elems = partition.n_owned_elems;
    let n_ghost_elems = partition.n_ghost_elems;
    let n_local_elems = n_owned_elems + n_ghost_elems;
    let n_orig_global = par_mesh.global_n_nodes();
    let my_rank = comm.rank();

    let gid_of = |local: u32| partition.global_node(local);

    // 2. Old edges (owned + ghost elements) keyed by global node ids.
    let mut old_edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for e in 0..n_local_elems as ElemId {
        let ns = local_mesh.elem_nodes(e);
        match local_mesh.elem_type {
            ElementType::Tri3 => {
                old_edges.insert(key(gid_of(ns[0]), gid_of(ns[1])));
                old_edges.insert(key(gid_of(ns[1]), gid_of(ns[2])));
                old_edges.insert(key(gid_of(ns[2]), gid_of(ns[0])));
            }
            ElementType::Quad4 => {
                for &(li, lj) in &[(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
                    old_edges.insert(key(gid_of(ns[li]), gid_of(ns[lj])));
                }
            }
            other => panic!("par_uniform_refine: unsupported element type {other:?}"),
        }
    }

    // 3. Global serial insertion order: the id of an edge-midpoint node is
    //    assigned by the *first* element (smallest global id) that references
    //    the edge, matching `refine_uniform`'s traversal order.  Every element
    //    referencing a locally-visible edge is itself visible locally (via the
    //    ghost layer), so each rank can compute the global minimum referencing
    //    element for every local edge without extra communication.
    let elem_edges_gid: Vec<Vec<(u32, u32)>> = (0..n_local_elems)
        .map(|e| {
            let ns = local_mesh.elem_nodes(e as ElemId);
            match local_mesh.elem_type {
                ElementType::Tri3 => vec![
                    key(gid_of(ns[0]), gid_of(ns[1])),
                    key(gid_of(ns[1]), gid_of(ns[2])),
                    key(gid_of(ns[2]), gid_of(ns[0])),
                ],
                ElementType::Quad4 => (0..4)
                    .map(|li| key(gid_of(ns[li]), gid_of(ns[(li + 1) % 4])))
                    .collect(),
                other => panic!("par_uniform_refine: unsupported element type {other:?}"),
            }
        })
        .collect();

    let mut edge_min_ref: BTreeMap<(u32, u32), ElemId> = BTreeMap::new();
    for e in 0..n_local_elems {
        let ge = partition.global_elem(e as u32);
        for &ek in &elem_edges_gid[e] {
            edge_min_ref
                .entry(ek)
                .and_modify(|m| *m = (*m).min(ge))
                .or_insert(ge);
        }
    }

    // Per owned element (global-id order): the new edges it assigns.
    // off[e] = number of new edges assigned by owned elements before e.
    let mut off = vec![0usize; n_owned_elems + 1];
    for e in 0..n_owned_elems {
        let ge = partition.global_elem(e as u32);
        let n_new = elem_edges_gid[e]
            .iter()
            .filter(|&&ek| edge_min_ref[&ek] == ge)
            .count();
        off[e + 1] = off[e] + n_new;
    }
    let total_rank = off[n_owned_elems];
    let all_counts = gather_counts(&comm, total_rank as i64);
    let base: usize = all_counts
        .iter()
        .take(comm.rank() as usize)
        .map(|&c| c as usize)
        .sum();
    let n_global_edges: usize = all_counts.iter().map(|&c| c as usize).sum();

    // 4. Assign midpoint ids in serial insertion order (element-major, then
    //    local edge order).
    let mut edge_gid: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for e in 0..n_owned_elems {
        let ge = partition.global_elem(e as u32);
        let mut next = base + off[e];
        for &ek in &elem_edges_gid[e] {
            if edge_min_ref[&ek] == ge {
                edge_gid.insert(ek, (n_orig_global + next) as u32);
                next += 1;
            }
        }
        debug_assert_eq!(next, base + off[e + 1], "edge count drift");
    }

    // 5. Request ids for edges whose min-ref element lives on another rank;
    //    answer requests for edges we assigned.
    let mut need_edges: BTreeMap<Rank, Vec<(u32, u32)>> = BTreeMap::new();
    for &ek in &old_edges {
        if edge_gid.contains_key(&ek) {
            continue;
        }
        let mr = edge_min_ref[&ek];
        let local_mr = partition
            .local_elem(mr)
            .expect("par_uniform_refine: min-ref element not present locally");
        let owner = partition.elem_owner[local_mr as usize];
        need_edges.entry(owner).or_default().push(ek);
    }
    for keys in need_edges.values_mut() {
        keys.sort_unstable();
    }
    let replies = answer_edge_requests(&comm, &edge_gid, &need_edges);
    for (coord, keys) in &need_edges {
        let gids = &replies[coord];
        for (k, &g) in keys.iter().zip(gids) {
            edge_gid.insert(*k, g);
        }
    }
    debug_assert_eq!(
        edge_gid.len(),
        old_edges.len(),
        "par_uniform_refine: missing midpoint ids"
    );

    // 6. Global ids + owners of every new local node.
    let n_total_new = refined.n_nodes();
    let mut new_gid = vec![0u32; n_total_new];
    let mut new_owner = vec![0i32; n_total_new];
    for i in 0..n_orig {
        new_gid[i] = partition.global_node_ids[i];
        new_owner[i] = partition.node_owner[i];
    }
    for (&(a, b), &mid) in &lr.edge_mid {
        let (ga, gb) = (gid_of(a), gid_of(b));
        let gkey = key(ga, gb);
        let g = edge_gid[&gkey];
        let mr = edge_min_ref[&gkey];
        let local_mr = partition
            .local_elem(mr)
            .expect("par_uniform_refine: min-ref element not present locally");
        new_gid[mid as usize] = g;
        new_owner[mid as usize] = partition.elem_owner[local_mr as usize];
    }
    if local_mesh.elem_type == ElementType::Quad4 {
        // Centers: one per parent element, id = f(parent gid).
        for e in 0..n_local_elems {
            let cid = n_orig + lr.n_edges + e;
            let pg = partition.global_elem(e as u32);
            new_gid[cid] = (n_orig_global + n_global_edges + pg as usize) as u32;
            new_owner[cid] = partition.elem_owner[e];
        }
    }

    // 7. Reorder local nodes into [owned | ghost] segments.
    let mut order: Vec<usize> = (0..n_total_new).collect();
    order.sort_by_key(|&i| (new_owner[i] != my_rank, i));
    let n_owned_new = order.iter().filter(|&&i| new_owner[i] == my_rank).count();
    let mut remap = vec![0u32; n_total_new];
    let mut new_coords = Vec::with_capacity(n_total_new * 2);
    for (new_id, &old_id) in order.iter().enumerate() {
        remap[old_id] = new_id as u32;
        let c = refined.coords_of(old_id as u32);
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

    // 8. New partition: element ids are 4·parent + k (owned first, then ghost);
    //    node ids from `new_gid`/`new_owner`.
    let mut owned_global_nodes: Vec<u32> = Vec::with_capacity(n_owned_new);
    let mut ghost_global_nodes: Vec<(u32, i32)> = Vec::with_capacity(n_total_new - n_owned_new);
    for &old_id in order.iter() {
        if new_owner[old_id] == my_rank {
            owned_global_nodes.push(new_gid[old_id]);
        } else {
            ghost_global_nodes.push((new_gid[old_id], new_owner[old_id]));
        }
    }
    let mut owned_global_elems: Vec<u32> = Vec::with_capacity(n_owned_elems * 4);
    let mut ghost_global_elems: Vec<(u32, i32)> = Vec::with_capacity(n_ghost_elems * 4);
    for e in 0..n_owned_elems {
        let pg = partition.global_elem(e as u32);
        for k in 0..4u32 {
            owned_global_elems.push(4 * pg + k);
        }
    }
    for e in n_owned_elems..n_local_elems {
        let pg = partition.global_elem(e as u32);
        let owner = partition.elem_owner[e];
        for k in 0..4u32 {
            ghost_global_elems.push((4 * pg + k, owner));
        }
    }

    let new_partition = MeshPartition::from_partitioner(
        &owned_global_nodes,
        &ghost_global_nodes,
        &owned_global_elems,
        &ghost_global_elems,
        my_rank,
    );

    ParallelMesh::new(refined_mesh, comm, new_partition)
}

// ─── communication helpers ───────────────────────────────────────────────────

/// Gather one `i64` per rank into a vector replicated on every rank.
fn gather_counts(comm: &Comm, my_count: i64) -> Vec<i64> {
    let n = comm.size() as i32;
    let mut all = vec![0i64; n as usize];
    if comm.is_root() {
        all[0] = my_count;
        for r in 1..n {
            let v = comm.recv::<i64>(r, REFINE_GID_TAG);
            all[r as usize] = v[0];
        }
        let mut buf = Vec::new();
        for c in &all {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        comm.broadcast_bytes(0, &mut buf);
    } else {
        comm.send(0, REFINE_GID_TAG, &[my_count]);
        let mut buf = Vec::new();
        comm.broadcast_bytes(0, &mut buf);
        all = buf
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
    }
    all
}

/// Answer midpoint-id requests from other ranks.
///
/// Requests arrive via `alltoallv` as sorted `(u32, u32)` key lists; replies
/// are the corresponding ids in the same order.  Returns a map
/// `coordinator → Vec<gid>` (only for coordinators we asked).
fn answer_edge_requests(
    comm: &Comm,
    edge_gid: &BTreeMap<(u32, u32), u32>,
    need_edges: &BTreeMap<Rank, Vec<(u32, u32)>>,
) -> BTreeMap<Rank, Vec<u32>> {
    // Encode our requests.
    let sends: Vec<(Rank, Vec<u8>)> = need_edges
        .iter()
        .map(|(&coord, keys)| {
            let mut bytes = Vec::with_capacity(keys.len() * 8);
            for &(a, b) in keys {
                bytes.extend_from_slice(&a.to_le_bytes());
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            (coord, bytes)
        })
        .collect();

    // Incoming requests (this rank is coordinator for some edges).
    let incoming = comm.alltoallv_bytes(&sends);
    let mut reply_payloads: Vec<(Rank, Vec<u8>)> = Vec::new();
    for (requester, bytes) in incoming {
        let mut gids = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
            gids.push(
                *edge_gid
                    .get(&key(a, b))
                    .expect("par_uniform_refine: request for an edge I don't coordinate"),
            );
        }
        let mut out = Vec::with_capacity(gids.len() * 4);
        for g in gids {
            out.extend_from_slice(&g.to_le_bytes());
        }
        reply_payloads.push((requester, out));
    }

    // Deliver replies.
    let responses = comm.alltoallv_bytes(&reply_payloads);
    let mut replies: BTreeMap<Rank, Vec<u32>> = BTreeMap::new();
    for (coord, bytes) in responses {
        let gids = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        replies.insert(coord, gids);
    }
    replies
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::native::SerialBackend;
    use crate::comm::Comm;
    use crate::launcher::native::ThreadLauncher;
    use crate::par_partition::partition_mesh;
    use crate::WorkerConfig;
    use fem_mesh::refine_uniform;

    #[test]
    fn single_rank_tri3_matches_serial() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let serial = refine_uniform(&mesh);
        let comm = Comm::from_backend(Box::new(SerialBackend));
        let pm = partition_mesh(&mesh, &comm);
        let refined = par_uniform_refine(&pm);
        assert_eq!(refined.global_n_elems(), serial.n_elems());
        assert_eq!(refined.global_n_nodes(), serial.n_nodes());
        // Node coordinates must be identical (gid order == serial order).
        let local = refined.local_mesh();
        let part = refined.partition();
        for i in 0..part.n_owned_nodes {
            let c = local.coords_of(i as u32);
            let gid = part.global_node_ids[i] as usize;
            let sc = serial.coords_of(gid as u32);
            assert!((c[0] - sc[0]).abs() < 1e-12 && (c[1] - sc[1]).abs() < 1e-12,
                "node {i} gid {gid}: ({},{}) vs ({},{})",
                c[0], c[1], sc[0], sc[1]);
        }
    }

    #[test]
    fn single_rank_quad4_matches_serial() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let serial = refine_uniform(&mesh);
        let comm = Comm::from_backend(Box::new(SerialBackend));
        let pm = partition_mesh(&mesh, &comm);
        let refined = par_uniform_refine(&pm);
        assert_eq!(refined.global_n_elems(), serial.n_elems());
        assert_eq!(refined.global_n_nodes(), serial.n_nodes());
        let local = refined.local_mesh();
        let part = refined.partition();
        for i in 0..part.n_owned_nodes {
            let c = local.coords_of(i as u32);
            let gid = part.global_node_ids[i] as usize;
            let sc = serial.coords_of(gid as u32);
            assert!((c[0] - sc[0]).abs() < 1e-12 && (c[1] - sc[1]).abs() < 1e-12,
                "node {i} gid {gid}: ({},{}) vs ({},{})",
                c[0], c[1], sc[0], sc[1]);
        }
    }

    #[test]
    fn two_ranks_matches_serial_global_mesh() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let serial = refine_uniform(&mesh);
        let n_nodes = serial.n_nodes();
        let n_elems = serial.n_elems();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pm = partition_mesh(&mesh, &comm);
            let refined = par_uniform_refine(&pm);
            let _n = comm.size();

            // Global counts must match serial.
            assert_eq!(refined.global_n_elems() as usize, n_elems);
            assert_eq!(refined.global_n_nodes() as usize, n_nodes);

            // Collect all owned (gid, x, y) from every rank and verify:
            //  - gids are unique
            //  - coordinates match the serial mesh at that gid
            let owned: Vec<(u32, f64, f64)> = (0..refined.n_owned_nodes())
                .map(|i| {
                    let c = refined.local_mesh().coords_of(i as u32);
                    (refined.partition().global_node_ids[i], c[0], c[1])
                })
                .collect();
            let flat: Vec<f64> = owned.iter().flat_map(|&(g, x, y)| [g as f64, x, y]).collect();
            if comm.rank() == 0 {
                let mut all = flat.clone();
                for r in 1..comm.size() as i32 {
                    let recv = comm.recv::<f64>(r, 0x3A21);
                    all.extend_from_slice(&recv);
                }
                let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
                for chunk in all.chunks_exact(3) {
                    let g = chunk[0] as u32;
                    let x = chunk[1];
                    let y = chunk[2];
                    assert!(seen.insert(g), "duplicate global node id {g}");
                    let sc = serial.coords_of(g as u32);
                    assert!((x - sc[0]).abs() < 1e-12 && (y - sc[1]).abs() < 1e-12,
                        "gid {g}: ({x},{y}) vs serial ({},{})", sc[0], sc[1]);
                }
                assert_eq!(seen.len(), n_nodes, "node count mismatch");
            } else {
                comm.send(0, 0x3A21, &flat);
            }
        });
    }

    #[test]
    fn two_ranks_quad4_matches_serial_global_mesh() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        let serial = refine_uniform(&mesh);
        let n_nodes = serial.n_nodes();
        let n_elems = serial.n_elems();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pm = partition_mesh(&mesh, &comm);
            let refined = par_uniform_refine(&pm);
            assert_eq!(refined.global_n_elems() as usize, n_elems);
            assert_eq!(refined.global_n_nodes() as usize, n_nodes);

            let owned: Vec<(u32, f64, f64)> = (0..refined.n_owned_nodes())
                .map(|i| {
                    let c = refined.local_mesh().coords_of(i as u32);
                    (refined.partition().global_node_ids[i], c[0], c[1])
                })
                .collect();
            let flat: Vec<f64> = owned.iter().flat_map(|&(g, x, y)| [g as f64, x, y]).collect();
            if comm.rank() == 0 {
                let mut all = flat.clone();
                for r in 1..comm.size() as i32 {
                    let recv = comm.recv::<f64>(r, 0x3A21);
                    all.extend_from_slice(&recv);
                }
                let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
                for chunk in all.chunks_exact(3) {
                    let g = chunk[0] as u32;
                    let x = chunk[1];
                    let y = chunk[2];
                    assert!(seen.insert(g), "duplicate global node id {g}");
                    let sc = serial.coords_of(g as u32);
                    assert!((x - sc[0]).abs() < 1e-12 && (y - sc[1]).abs() < 1e-12,
                        "gid {g}: ({x},{y}) vs serial ({},{})", sc[0], sc[1]);
                }
                assert_eq!(seen.len(), n_nodes, "node count mismatch");
            } else {
                comm.send(0, 0x3A21, &flat);
            }
        });
    }
}
