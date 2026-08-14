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
//! * Compact node partitions ([`partition_mesh`]) **and** identity-node
//!   partitions ([`partition_mesh_identity`]).  Identity inputs refine to a
//!   compact partition (holes are dropped; the refined global mesh matches
//!   the serial refinement).
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
    /// centers in quad-parent order).  Elements are `4` consecutive
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
    /// Local parent element id → local id of the new element-center node
    /// (Quad4 parents only; empty for pure-Tri3 meshes).
    center_of: HashMap<u32, u32>,
}

#[inline]
fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Refine a pure-Tri3, pure-Quad4, or mixed Tri3+Quad4 2-D mesh (all
/// elements 1 → 4).
///
/// The child layout matches the serial [`fem_mesh::refine_uniform`] exactly:
/// * Tri3  (red refinement): `[n0,m01,m02]`, `[m01,n1,m12]`, `[m02,m12,n2]`,
///   `[m01,m12,m02]`.
/// * Quad4: `[v0,e0,c,e3]`, `[e0,v1,e1,c]`, `[c,e1,v2,e2]`, `[e3,c,e2,v3]`.
///
/// Mixed meshes (Tri3 + Quad4, `elem_types` present) refine each element by
/// its own type with a shared edge-midpoint map and one center node per
/// Quad4 parent (MFEM `UniformRefinement2D_base` semantics).
fn refine_local(mesh: &Mesh<2>) -> LocalRefine {
    if mesh.elem_types.is_some() {
        return refine_local_mixed(mesh);
    }
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

    // Edge midpoints: new node ids `n_orig + idx` in **element × local-edge
    // order** (first occurrence), matching the serial `refine_uniform_2d`
    // edge map (MFEM el_to_edge).  A BTreeSet (sorted-key) enumeration would
    // renumber the midpoints differently than the serial path, breaking the
    // P2/P3 DofManager edge numbering on partitioned meshes.
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_mid = n_orig as u32;
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(li, lj) in &[(0usize, 1usize), (1, 2), (2, 0)] {
            let k = key(ns[li], ns[lj]);
            if !edge_mid.contains_key(&k) {
                let mid = next_mid;
                next_mid += 1;
                edge_mid.insert(k, mid);
                let ca = mesh.coords_of(ns[li]);
                let cb = mesh.coords_of(ns[lj]);
                new_coords.push((ca[0] + cb[0]) * 0.5);
                new_coords.push((ca[1] + cb[1]) * 0.5);
            }
        }
    }
    let n_edges = edge_mid.len();

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
    LocalRefine {
        mesh: refined,
        edge_mid,
        n_orig,
        n_edges,
        center_of: HashMap::new(),
    }
}

fn refine_local_quad4(mesh: &Mesh<2>) -> LocalRefine {
    let n_orig = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

    // New node layout: [old | edge midpoints | element centers].
    // Edge midpoints in element × local-edge order (serial-matching), not
    // sorted-key order — see refine_local_tri3.
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_mid = n_orig as u32;
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(li, lj) in &QUAD_EDGES {
            let k = key(ns[li], ns[lj]);
            if !edge_mid.contains_key(&k) {
                let mid = next_mid;
                next_mid += 1;
                edge_mid.insert(k, mid);
                let ca = mesh.coords_of(ns[li]);
                let cb = mesh.coords_of(ns[lj]);
                new_coords.push((ca[0] + cb[0]) * 0.5);
                new_coords.push((ca[1] + cb[1]) * 0.5);
            }
        }
    }
    let n_edges = edge_mid.len();
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
    let center_of: HashMap<u32, u32> = (0..n_elems as u32)
        .map(|e| (e, (n_orig + n_edges + e as usize) as u32))
        .collect();
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
    LocalRefine {
        mesh: refined,
        edge_mid,
        n_orig,
        n_edges,
        center_of,
    }
}

/// Uniform refinement of a **mixed** Tri3 + Quad4 2-D mesh (parallel
/// counterpart of the serial [`fem_mesh::refine_uniform`] mixed path /
/// MFEM `UniformRefinement2D_base`).
///
/// One shared edge-midpoint map covers both element types; each element
/// splits 1 → 4 by its own type (Tri3 red refinement / Quad4 four-way with a
/// shared parent center).  New node ids: `[0..n_orig)` old nodes, then edge
/// midpoints in sorted-key order, then quad centers in quad-parent order.
fn refine_local_mixed(mesh: &Mesh<2>) -> LocalRefine {
    const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

    let n_orig = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // Edge midpoints: new node ids `n_orig + idx` in element × local-edge
    // order (serial-matching; see refine_local_tri3).
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_mid = n_orig as u32;
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let edges = match mesh.element_type_at(e) {
            ElementType::Tri3 => &TRI_EDGES[..],
            ElementType::Quad4 => &QUAD_EDGES[..],
            other => panic!("par_uniform_refine: unsupported element type {other:?}"),
        };
        for &(li, lj) in edges {
            let k = key(ns[li], ns[lj]);
            if !edge_mid.contains_key(&k) {
                let mid = next_mid;
                next_mid += 1;
                edge_mid.insert(k, mid);
                let ca = mesh.coords_of(ns[li]);
                let cb = mesh.coords_of(ns[lj]);
                new_coords.push((ca[0] + cb[0]) * 0.5);
                new_coords.push((ca[1] + cb[1]) * 0.5);
            }
        }
    }
    let n_edges = edge_mid.len();

    // Quad centers: one per Quad4 parent, id = n_orig + n_edges + quad_idx.
    let mut center_of: HashMap<u32, u32> = HashMap::new();
    let mut quad_counter = 0usize;
    for e in 0..n_elems as ElemId {
        if mesh.element_type_at(e) != ElementType::Quad4 {
            continue;
        }
        let ns = mesh.elem_nodes(e);
        let cid = (n_orig + n_edges + quad_counter) as u32;
        quad_counter += 1;
        center_of.insert(e, cid);
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

    // Children: per-element type, 4 per parent (red refinement for Tri3,
    // four-way with shared center for Quad4).
    let mut new_conn = Vec::with_capacity(n_elems * 4 * 4);
    let mut new_tags = Vec::with_capacity(n_elems * 4);
    let mut new_types = Vec::with_capacity(n_elems * 4);
    let mut new_offsets = Vec::with_capacity(n_elems * 4 + 1);
    new_offsets.push(0usize);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];
        match mesh.element_type_at(e) {
            ElementType::Tri3 => {
                let m01 = edge_mid[&key(ns[0], ns[1])];
                let m12 = edge_mid[&key(ns[1], ns[2])];
                let m02 = edge_mid[&key(ns[2], ns[0])];
                for c in [
                    [ns[0], m01, m02],
                    [m01, ns[1], m12],
                    [m02, m12, ns[2]],
                    [m01, m12, m02],
                ] {
                    new_conn.extend_from_slice(&c);
                    new_types.push(ElementType::Tri3);
                    new_offsets.push(new_conn.len());
                    new_tags.push(tag);
                }
            }
            ElementType::Quad4 => {
                let v = [ns[0], ns[1], ns[2], ns[3]];
                let center = center_of[&e];
                let e_mid: [u32; 4] = core::array::from_fn(|li| {
                    let (a, b) = QUAD_EDGES[li];
                    edge_mid[&key(v[a], v[b])]
                });
                for c in [
                    [v[0], e_mid[0], center, e_mid[3]],
                    [e_mid[0], v[1], e_mid[1], center],
                    [center, e_mid[1], v[2], e_mid[2]],
                    [e_mid[3], center, e_mid[2], v[3]],
                ] {
                    new_conn.extend_from_slice(&c);
                    new_types.push(ElementType::Quad4);
                    new_offsets.push(new_conn.len());
                    new_tags.push(tag);
                }
            }
            other => panic!("par_uniform_refine: unsupported element type {other:?}"),
        }
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

    let refined = Mesh {
        coords: new_coords,
        conn: new_conn,
        elem_tags: new_tags,
        elem_type: mesh.elem_type,
        face_conn: new_face_conn,
        face_tags: new_face_tags,
        face_type: ElementType::Line2,
        elem_types: Some(new_types),
        elem_offsets: Some(new_offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
    };
    LocalRefine {
        mesh: refined,
        edge_mid,
        n_orig,
        n_edges,
        center_of,
    }
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

    // 1. Refine the local mesh (owned + ghost elements).
    let lr = refine_local(local_mesh);
    let refined = lr.mesh;
    let n_orig = lr.n_orig;
    let n_owned_elems = partition.n_owned_elems;
    let n_ghost_elems = partition.n_ghost_elems;
    let n_local_elems = n_owned_elems + n_ghost_elems;
    // Start of the new-node id range.  `global_n_nodes()` is the number of
    // *referenced* nodes, but the input partition may contain unreferenced
    // "hole" gids (e.g. isolated vertices from the original mesh that every
    // refinement keeps but the referenced-only partition drops): old gids
    // then extend past `global_n_nodes`, and new edge/center ids starting at
    // `global_n_nodes` collide with them (pex39 np1 second refine produced
    // 1024 duplicate gids == the quad-center ids).  Use max(old gid) + 1,
    // globally consistent.
    let local_max_gid = par_mesh
        .partition()
        .global_node_ids
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as i64;
    let all_maxes = gather_counts(&comm, local_max_gid);
    let n_orig_global = (*all_maxes.iter().max().unwrap()) as u32 + 1;
    let my_rank = comm.rank();

    let gid_of = |local: u32| partition.global_node(local);

    // 2. Old edges (owned + ghost elements) keyed by global node ids.
    let mut old_edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for e in 0..n_local_elems as ElemId {
        let ns = local_mesh.elem_nodes(e);
        match local_mesh.element_type_at(e) {
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
            match local_mesh.element_type_at(e as ElemId) {
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

    let mut edge_min_ref: BTreeMap<(u32, u32), (ElemId, Rank)> = BTreeMap::new();
    for e in 0..n_local_elems {
        let ge = partition.global_elem(e as u32);
        let ow = partition.elem_owner[e];
        for &ek in &elem_edges_gid[e] {
            edge_min_ref
                .entry(ek)
                .and_modify(|m| {
                    if ge < m.0 {
                        *m = (ge, ow);
                    }
                })
                .or_insert((ge, ow));
        }
    }

    // 3b. Global min-ref agreement.  With an asymmetric ghost layer (np >= 4
    //     the two elements sharing an edge are not always mutually visible),
    //     ranks can disagree on the minimum referencing element of a
    //     cross-rank edge; the edge-midpoint request/answer routing in steps
    //     4-5 then deadlocks/panics (pex39 np4 hit "request for an edge I
    //     don't coordinate").  Exchange (edge, min_elem, owner) triples and
    //     merge to the global minimum so every rank routes to the same
    //     coordinator.
    let payload: Vec<(u32, u32, u32, i32)> = edge_min_ref
        .iter()
        .map(|(&(a, b), &(mr, ow))| (a, b, mr, ow))
        .collect();
    let mut sends: Vec<(Rank, Vec<u8>)> = Vec::new();
    for r in 0..comm.size() as i32 {
        if r == my_rank {
            continue;
        }
        let mut bytes = Vec::with_capacity(payload.len() * 16);
        for &(a, b, mr, ow) in &payload {
            bytes.extend_from_slice(&a.to_le_bytes());
            bytes.extend_from_slice(&b.to_le_bytes());
            bytes.extend_from_slice(&mr.to_le_bytes());
            bytes.extend_from_slice(&ow.to_le_bytes());
        }
        sends.push((r as Rank, bytes));
    }
    let incoming = comm.alltoallv_bytes(&sends);
    for (_, bytes) in incoming {
        for chunk in bytes.chunks_exact(16) {
            let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let mr = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let ow = i32::from_le_bytes(chunk[12..16].try_into().unwrap());
            let ek = key(a, b);
            edge_min_ref
                .entry(ek)
                .and_modify(|cur| {
                    if mr < cur.0 {
                        *cur = (mr, ow);
                    }
                })
                .or_insert((mr, ow));
        }
    }

    // Per owned element (global-id order): the new edges it assigns.
    // off[e] = number of new edges assigned by owned elements before e.
    let mut off = vec![0usize; n_owned_elems + 1];
    for e in 0..n_owned_elems {
        let ge = partition.global_elem(e as u32);
        let n_new = elem_edges_gid[e]
            .iter()
            .filter(|&&ek| edge_min_ref[&ek].0 == ge)
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
            if edge_min_ref[&ek].0 == ge {
                edge_gid.insert(ek, (n_orig_global as usize + next) as u32);
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
        let owner = edge_min_ref[&ek].1;
        need_edges.entry(owner).or_default().push(ek);
    }
    for keys in need_edges.values_mut() {
        keys.sort_unstable();
    }
    let replies = answer_edge_requests(&comm, &edge_gid, &need_edges);    for (coord, keys) in &need_edges {
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
    //    `global_node`/`node_owner` are identity-aware (in identity mode the
    //    local id IS the global id, so the compact `global_node_ids` array
    //    cannot be indexed directly).
    let n_total_new = refined.n_nodes();
    let mut new_gid = vec![0u32; n_total_new];
    let mut new_owner = vec![0i32; n_total_new];
    for i in 0..n_orig {
        new_gid[i] = partition.global_node(i as u32);        new_owner[i] = partition.node_owner(i as u32);
    }
    for (&(a, b), &mid) in &lr.edge_mid {
        let (ga, gb) = (gid_of(a), gid_of(b));
        let gkey = key(ga, gb);
        let g = edge_gid[&gkey];
        let (_, owner) = edge_min_ref[&gkey];
        new_gid[mid as usize] = g;
        new_owner[mid as usize] = owner;
    }
    if !lr.center_of.is_empty() {
        // Centers: one per Quad4 parent, id = f(parent gid).
        for e in 0..n_local_elems {
            if let Some(&cid) = lr.center_of.get(&(e as u32)) {
                let pg = partition.global_elem(e as u32);
                new_gid[cid as usize] = (n_orig_global as usize + n_global_edges + pg as usize) as u32;
                new_owner[cid as usize] = partition.elem_owner[e];
            }
        }
    }

    // 7. Reorder local nodes into [owned | ghost] segments.  In identity mode
    //    the local node id space has holes (ids ARE global ids); only nodes
    //    referenced by the refined mesh participate.
    let mut referenced = vec![false; n_total_new];
    for &n in &refined.conn {
        referenced[n as usize] = true;
    }
    let mut order: Vec<usize> = (0..n_total_new).filter(|&i| referenced[i]).collect();
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

    let refined_mesh = if refined.elem_types.is_some() {
        // Mixed Tri3+Quad4: preserve per-element types/offsets.
        Mesh {
            coords: new_coords,
            conn: new_conn,
            elem_tags: refined.elem_tags.clone(),
            elem_type: refined.elem_type,
            face_conn: new_face_conn,
            face_tags: refined.face_tags.clone(),
            face_type: refined.face_type,
            elem_types: refined.elem_types.clone(),
            elem_offsets: refined.elem_offsets.clone(),
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
        }
    } else {
        Mesh::uniform(
            new_coords,
            new_conn,
            refined.elem_tags.clone(),
            refined.elem_type,
            new_face_conn,
            refined.face_tags.clone(),
            refined.face_type,
        )
    };

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

    /// Mixed Tri3+Quad4 meshes must refine per-element type with a shared
    /// edge map, matching the serial mixed path (MFEM
    /// `UniformRefinement2D_base`).
    fn build_mixed_mesh() -> Mesh<2> {
        // 6 nodes: unit square with a vertical mid-edge.
        #[rustfmt::skip]
        let coords = vec![
            0.0, 0.0, // 0
            0.5, 0.0, // 1
            1.0, 0.0, // 2
            0.0, 1.0, // 3
            0.5, 1.0, // 4
            1.0, 1.0, // 5
        ];
        // Quad4 {0,1,4,3} left half + Tri3 {1,2,5} + Tri3 {1,5,4} right half.
        let conn: Vec<u32> = vec![0, 1, 4, 3, 1, 2, 5, 1, 5, 4];
        let elem_offsets = vec![0usize, 4, 7, 10];
        let elem_types = vec![
            ElementType::Quad4,
            ElementType::Tri3,
            ElementType::Tri3,
        ];
        let elem_tags = vec![0i32; 3];
        let face_conn: Vec<u32> = vec![0, 1, 1, 2, 2, 5, 5, 4, 4, 3, 3, 0];
        let face_tags: Vec<BoundaryTag> = vec![1, 1, 2, 3, 3, 4];
        Mesh::<2> {
            coords,
            conn,
            elem_tags,
            elem_type: ElementType::Tri3,
            face_conn,
            face_tags,
            face_type: ElementType::Line2,
            elem_types: Some(elem_types),
            elem_offsets: Some(elem_offsets),
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
        }
    }

    #[test]
    fn single_rank_mixed_matches_serial() {
        let mesh = build_mixed_mesh();
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
            assert!(
                (c[0] - sc[0]).abs() < 1e-12 && (c[1] - sc[1]).abs() < 1e-12,
                "node {i} gid {gid}: ({},{}) vs ({},{})",
                c[0],
                c[1],
                sc[0],
                sc[1]
            );
        }
    }

    #[test]
    fn two_ranks_mixed_matches_serial_global_mesh() {
        let mesh = build_mixed_mesh();
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
            let flat: Vec<f64> =
                owned.iter().flat_map(|&(g, x, y)| [g as f64, x, y]).collect();
            if comm.rank() == 0 {
                let mut all = flat.clone();
                for r in 1..comm.size() as i32 {
                    let recv = comm.recv::<f64>(r, 0x3A21);
                    all.extend_from_slice(&recv);
                }
                let mut seen: std::collections::HashSet<u32> =
                    std::collections::HashSet::new();
                for chunk in all.chunks_exact(3) {
                    let g = chunk[0] as u32;
                    let x = chunk[1];
                    let y = chunk[2];
                    assert!(seen.insert(g), "duplicate global node id {g}");
                    let sc = serial.coords_of(g as u32);
                    assert!(
                        (x - sc[0]).abs() < 1e-12 && (y - sc[1]).abs() < 1e-12,
                        "gid {g}: ({x},{y}) vs serial ({},{})",
                        sc[0],
                        sc[1]
                    );
                }
                assert_eq!(seen.len(), n_nodes, "node count mismatch");
            } else {
                comm.send(0, 0x3A21, &flat);
            }
        });
    }

    /// Identity-node partitions must also refine: the local node ids ARE the
    /// global ids (with holes), and the refined partition must be rebuilt
    /// correctly (nodes referenced by the refined mesh only, holes dropped).
    #[test]
    fn identity_partition_refines_consistently() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 elements, 25 nodes
        let serial = refine_uniform(&mesh);
        let n_elems = serial.n_elems();
        let n_nodes = serial.n_nodes();

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = crate::par_partition::partition_mesh_identity(&mesh, &comm);
            let refined = par_uniform_refine(&pmesh);
            let part = refined.partition();

            let n_elems_g: usize =
                comm.allreduce_sum_i64(part.n_owned_elems as i64) as usize;
            let n_nodes_g: usize =
                comm.allreduce_sum_i64(part.n_owned_nodes as i64) as usize;
            assert_eq!(n_elems_g, n_elems, "identity refine elem count");
            assert_eq!(n_nodes_g, n_nodes, "identity refine node count");

            // Owned node gids partition [0, n_nodes) exactly once.
            let owned: Vec<u32> = part.global_node_ids[..part.n_owned_nodes].to_vec();
            let mut flat = owned.clone();
            if comm.rank() == 0 {
                for r in 1..comm.size() as i32 {
                    let recv = comm.recv::<u32>(r, 0x3A22);
                    flat.extend_from_slice(&recv);
                }
                flat.sort_unstable();
                assert_eq!(flat, (0..n_nodes as u32).collect::<Vec<_>>(),
                    "identity refine owned gids must cover the serial range");
            } else {
                comm.send(0, 0x3A22, &flat);
            }
        });
    }
}
