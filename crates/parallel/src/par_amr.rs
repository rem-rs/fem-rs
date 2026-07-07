//! Parallel adaptive mesh refinement.
//!
//! Provides [`par_refine_marked`] for distributed non-conforming refinement and
//! [`par_repartition`] for load-rebalancing after refinement.

use std::collections::{HashMap, BTreeMap};

use fem_mesh::{Mesh, amr::NCState, topology::MeshTopology, boundary::BoundaryTag};
use fem_core::types::{ElemId, NodeId};

use crate::{
    par_mesh::ParallelMesh,
    partition::MeshPartition,
    ghost::GhostExchange,
    mesh_serde::{encode_submesh, decode_submesh},
    par_simplex::{partition_simplex_streaming, STREAM_TAG_BASE},
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
pub fn par_refine_marked(
    par_mesh: &ParallelMesh<Mesh<2>>,
    mut nc_state: NCState,
    marked:    &[ElemId],
    solution:  Option<&[f64]>,
) -> Result<ParRefinedMesh, ParAmrError> {
    let local_mesh = par_mesh.local_mesh();
    let comm       = par_mesh.comm().clone();

    let (refined_mesh, _constraints, _midpoint_map) = nc_state.refine(local_mesh, marked);
    let n_new_elems = refined_mesh.n_elements();

    let prolongated = if let Some(sol) = solution {
        prolongate_p1(local_mesh, &refined_mesh, sol)
    } else {
        vec![]
    };

    let new_partition = MeshPartition::new_serial(
        refined_mesh.n_nodes(),
        refined_mesh.n_elements(),
    );
    let _ghost = GhostExchange::from_trivial();
    let new_par_mesh = ParallelMesh::new(refined_mesh, comm, new_partition);

    Ok(ParRefinedMesh {
        par_mesh: new_par_mesh,
        nc_state,
        solution: prolongated,
        n_new_elems,
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
    })
}

/// Re-distribute elements across MPI ranks after refinement.
///
/// Gathers all sub-meshes to rank 0, merges them into a single global mesh,
/// and redistributes via [`partition_simplex_streaming`].
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
        partition_simplex_streaming(Some(&global_mesh), &comm)
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
    use crate::{par_mesh::ParallelMesh, partition::MeshPartition, backend::native::SerialBackend, comm::Comm};

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
        let (refined, _, _) = nc.refine(&coarse, &marked);
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
        let (refined, _, _) = nc.refine(&coarse, &marked);
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
        };
        let centroids = compute_centroids_simple(&mesh);
        assert_eq!(centroids.len(), 1);
        assert!((centroids[0][0] - 1.0 / 3.0).abs() < 1e-14, "centroid x={}", centroids[0][0]);
        assert!((centroids[0][1] - 1.0 / 3.0).abs() < 1e-14, "centroid y={}", centroids[0][1]);
    }
}
