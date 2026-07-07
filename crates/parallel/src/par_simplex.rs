//! Partitioner for [`Mesh<D>`].
//!
//! [`partition_simplex`] distributes a serial `Mesh<D>` across the
//! ranks of a [`Comm`] using a **contiguous element partition**:
//!
//! * Rank `r` owns elements `[r·chunk, (r+1)·chunk)`.
//! * Node ownership: a node is owned by the lowest-rank process whose element
//!   chunk contains it.  Iteration order ensures this is simply the first rank
//!   to "see" the node when sweeping elements 0 → n_elems.
//! * Boundary faces: assigned to the rank that owns the minimum-index node of
//!   the face (unique and consistent across ranks).
//!
//! Mixed **volume** connectivity (`elem_offsets` / `elem_types`) and mixed **boundary**
//! faces (`face_offsets` / `face_types`) from the global mesh are preserved on each
//! rank’s local sub-mesh (needed for prism/pyramid-dominated 3D meshes).
//!
//! ## Single-rank fast path
//!
//! When `comm.size() == 1` the full mesh is wrapped as-is with
//! [`MeshPartition::new_serial`]; no copying or remapping is done.
//!
//! ## Multi-rank behaviour
//!
//! Every rank receives the **full** serial mesh and extracts its local portion.
//! This "replicated-then-extract" strategy is memory-inefficient for very large
//! meshes but is correct and straightforward for the initial implementation.
//! A streaming partitioner (where only rank 0 reads the mesh and distributes
//! via MPI) can replace this later.

use std::collections::{BTreeSet, HashMap};

use fem_core::{FaceId, NodeId, Rank};
use fem_mesh::{ElementType, Mesh};

use crate::{Comm, MeshPartition, par_mesh::ParallelMesh};
use crate::mesh_serde;

// ── public entry point ────────────────────────────────────────────────────────

/// Distribute `mesh` across all ranks of `comm`.
///
/// Returns a [`ParallelMesh`] whose local sub-mesh contains only the elements
/// and nodes (owned + ghost) assigned to the calling rank.
///
/// ## Multi-rank behaviour
///
/// Rank 0 partitions the mesh and sends each rank's sub-mesh via point-to-point
/// messages.  Other ranks receive their sub-mesh without ever loading the full
/// mesh (streaming approach).  This is memory-efficient: rank 0 holds the full
/// mesh; other ranks hold only their local portion.
///
/// For the **replicate-then-extract** fallback (all ranks hold the full mesh),
/// use [`partition_simplex_replicated`].
///
/// # Panics
/// Panics if the mesh has zero elements.
pub fn partition_simplex<const D: usize>(
    mesh: &Mesh<D>,
    comm: &Comm,
) -> ParallelMesh<Mesh<D>> {
    if comm.size() == 1 {
        let n = mesh.n_elems();
        assert!(n > 0, "partition_simplex: mesh has no elements");
        let partition = MeshPartition::new_serial(mesh.n_nodes(), n);
        return ParallelMesh::new(mesh.clone(), comm.clone(), partition);
    }
    // Multi-rank: use streaming path (rank 0 partitions, sends sub-meshes).
    // Non-root ranks receive their portion without loading the full mesh.
    partition_simplex_streaming(Some(mesh), comm)
        .expect("partition_simplex streaming failed")
}

/// Replicated-then-extract partitioner (all ranks hold the full mesh).
///
/// This is the **fallback** for environments where point-to-point messaging
/// is unavailable (e.g., WASM Workers with limited channels).  On native
/// platforms, [`partition_simplex`] uses streaming by default.
pub fn partition_simplex_replicated<const D: usize>(
    mesh: &Mesh<D>,
    comm: &Comm,
) -> ParallelMesh<Mesh<D>> {
    let n_elems = mesh.n_elems();
    let n_nodes_total = mesh.n_nodes();
    assert!(n_elems > 0, "partition_simplex: mesh has no elements");

    // ── single-rank fast path ────────────────────────────────────────────────
    if comm.size() == 1 {
        let partition = MeshPartition::new_serial(n_nodes_total, n_elems);
        return ParallelMesh::new(mesh.clone(), comm.clone(), partition);
    }

    // ── multi-rank partitioning ──────────────────────────────────────────────
    let (local_mesh, partition) = extract_submesh_for_rank(
        mesh, comm.rank(), comm.size(),
    );
    ParallelMesh::new(local_mesh, comm.clone(), partition)
}

// ── streaming partition ──────────────────────────────────────────────────────

/// Streaming mesh partition tag base (avoids ghost `0x1000`/`0x2000` and
/// alltoallv `0x4000`/`0x5000`).
pub(crate) const STREAM_TAG_BASE: i32 = 0x3700;

/// Distribute a mesh using streaming: only rank 0 holds the full mesh.
///
/// Rank 0 partitions the mesh and sends each rank's sub-mesh via point-to-point
/// messages.  Other ranks receive their sub-mesh without ever loading the full
/// mesh — saving memory on WASM workers.
///
/// # Arguments
/// * `mesh` — `Some(&full_mesh)` on rank 0, `None` on other ranks.
/// * `comm` — communicator spanning all ranks.
///
/// # Errors
/// Returns `Err` if the binary mesh decode fails on a receiving rank.
pub fn partition_simplex_streaming<const D: usize>(
    mesh: Option<&Mesh<D>>,
    comm: &Comm,
) -> Result<ParallelMesh<Mesh<D>>, String> {
    let size = comm.size();

    // ── single-rank fast path ────────────────────────────────────────────────
    if size == 1 {
        let m = mesh.ok_or("rank 0 must provide the mesh")?;
        let partition = MeshPartition::new_serial(m.n_nodes(), m.n_elems());
        return Ok(ParallelMesh::new(m.clone(), comm.clone(), partition));
    }

    if comm.is_root() {
        // ── root: partition and distribute ────────────────────────────────────
        let m = mesh.ok_or("rank 0 must provide the mesh")?;

        // Send sub-meshes to ranks 1..N-1.
        for target in 1..size as Rank {
            let (sub_mesh, sub_part) = extract_submesh_for_rank(m, target, size);
            let encoded = mesh_serde::encode_submesh(&sub_mesh, &sub_part);
            comm.send_bytes(target, STREAM_TAG_BASE + target, &encoded);
        }

        // Extract rank 0's own sub-mesh.
        let (local_mesh, partition) = extract_submesh_for_rank(m, 0, size);
        Ok(ParallelMesh::new(local_mesh, comm.clone(), partition))
    } else {
        // ── non-root: receive sub-mesh ───────────────────────────────────────
        let local_rank = comm.rank();
        let buf = comm.recv_bytes(0, STREAM_TAG_BASE + local_rank);
        let (local_mesh, partition) = mesh_serde::decode_submesh::<D>(&buf)?;
        Ok(ParallelMesh::new(local_mesh, comm.clone(), partition))
    }
}

// ── extract_submesh_for_rank ─────────────────────────────────────────────────

/// Extract the sub-mesh for a given rank using contiguous element blocks.
///
/// This is a convenience wrapper around [`extract_submesh_from_partition`] that
/// builds a contiguous-block partition vector internally.
fn extract_submesh_for_rank<const D: usize>(
    mesh: &Mesh<D>,
    target_rank: Rank,
    n_ranks: usize,
) -> (Mesh<D>, MeshPartition) {
    let n_elems = mesh.n_elems();
    let chunk = n_elems.div_ceil(n_ranks);
    let elem_part: Vec<Rank> = (0..n_elems)
        .map(|e| (e / chunk) as Rank)
        .collect();
    extract_submesh_from_partition(mesh, target_rank, &elem_part)
}

/// Extract the sub-mesh and partition descriptor for a given rank from an
/// arbitrary element partition vector.
///
/// This is the shared core used by both the contiguous-block partitioner
/// (`partition_simplex`) and the METIS graph partitioner
/// (`partition_simplex_metis`).
///
/// # Arguments
/// * `mesh` — the full serial mesh.
/// * `target_rank` — the rank whose sub-mesh to extract.
/// * `elem_part` — `elem_part[e]` is the rank that owns element `e`.
pub(crate) fn extract_submesh_from_partition<const D: usize>(
    mesh: &Mesh<D>,
    target_rank: Rank,
    elem_part: &[Rank],
) -> (Mesh<D>, MeshPartition) {
    let n_elems = mesh.n_elems();

    // 1. Collect elements owned by target_rank.
    let local_elem_gids: Vec<u32> = (0..n_elems as u32)
        .filter(|&e| elem_part[e as usize] == target_rank)
        .collect();

    // 2. Node ownership: owner = rank of first element containing the node.
    let node_owners = compute_node_owners_from_partition(mesh, elem_part);

    // 3. Collect all nodes touched by local (owned) elements.
    let mut node_set: BTreeSet<NodeId> = BTreeSet::new();
    for &ge in &local_elem_gids {
        for &n in mesh.elem_nodes(ge) {
            node_set.insert(n);
        }
    }

    // 3b. Find ghost elements: elements NOT owned by this rank that share at
    // least one node with our owned elements.
    let mut ghost_elem_gids: Vec<u32> = Vec::new();
    for e in 0..n_elems as u32 {
        if elem_part[e as usize] == target_rank { continue; }
        let shares_node = mesh.elem_nodes(e).iter().any(|n| node_set.contains(n));
        if shares_node {
            ghost_elem_gids.push(e);
        }
    }

    // 3c. Add nodes from ghost elements to the node set.
    for &ge in &ghost_elem_gids {
        for &n in mesh.elem_nodes(ge) {
            node_set.insert(n);
        }
    }

    // 4. Classify nodes as owned vs ghost.
    let mut owned_global: Vec<NodeId> = Vec::new();
    let mut ghost_global: Vec<(NodeId, Rank)> = Vec::new();
    for gn in &node_set {
        let owner = node_owners[*gn as usize];
        if owner == target_rank {
            owned_global.push(*gn);
        } else {
            ghost_global.push((*gn, owner));
        }
    }

    // 4b. Build global → local node mapping (owned first, then ghost).
    let ghost_base = owned_global.len();
    let mut g2l: HashMap<NodeId, u32> =
        HashMap::with_capacity(owned_global.len() + ghost_global.len());
    for (lid, &gn) in owned_global.iter().enumerate() {
        g2l.insert(gn, lid as u32);
    }
    for (idx, &(gn, _)) in ghost_global.iter().enumerate() {
        g2l.insert(gn, (ghost_base + idx) as u32);
    }

    // 5. Build local coordinate array (owned first, then ghost).
    let total_local_nodes = g2l.len();
    let mut local_coords = Vec::with_capacity(total_local_nodes * D);
    for &gn in owned_global.iter()
        .chain(ghost_global.iter().map(|(gn, _)| gn))
    {
        local_coords.extend_from_slice(&mesh.coords_of(gn));
    }

    // 6. Build local connectivity with remapped node IDs (owned + ghost elements).
    let all_local_elems = local_elem_gids.len() + ghost_elem_gids.len();
    let mixed_vol = mesh.elem_offsets.is_some();
    let npe_hint = if mixed_vol { 8 } else { mesh.elem_type.nodes_per_element() };
    let mut local_conn = Vec::with_capacity(all_local_elems.saturating_mul(npe_hint));
    let mut local_elem_tags = Vec::with_capacity(all_local_elems);
    let mut local_elem_types = if mixed_vol { Some(Vec::new()) } else { None };
    let mut local_elem_offsets = if mixed_vol { Some(vec![0usize]) } else { None };
    for &ge in local_elem_gids.iter().chain(ghost_elem_gids.iter()) {
        for &gn in mesh.elem_nodes(ge) {
            local_conn.push(g2l[&gn]);
        }
        local_elem_tags.push(mesh.elem_tags[ge as usize]);
        if mixed_vol {
            local_elem_types.as_mut().unwrap().push(mesh.element_type_at(ge));
            local_elem_offsets.as_mut().unwrap().push(local_conn.len());
        }
    }

    // 7. Assign boundary faces to this rank.
    let (local_face_conn, local_face_tags, local_face_types, local_face_offsets) =
        extract_local_faces(mesh, &g2l, &node_owners, target_rank);

    // 8. Assemble the local sub-mesh.
    let mut local_mesh = Mesh::uniform(
        local_coords, local_conn, local_elem_tags, mesh.elem_type,
        local_face_conn, local_face_tags, mesh.face_type,
    );
    local_mesh.elem_types = local_elem_types;
    local_mesh.elem_offsets = local_elem_offsets;
    local_mesh.face_types = local_face_types;
    local_mesh.face_offsets = local_face_offsets;

    let partition = MeshPartition::from_partitioner(
        &owned_global,
        &ghost_global,
        &local_elem_gids,
        target_rank,
    );

    (local_mesh, partition)
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// For each global node, compute which rank owns it given an arbitrary
/// element partition vector.
///
/// A node is owned by the rank that owns the lowest-indexed element containing
/// it.  Sweeping elements 0 → n_elems in order, the first rank to "see" a node
/// becomes its owner.
pub(crate) fn compute_node_owners_from_partition<const D: usize>(
    mesh: &Mesh<D>,
    elem_part: &[Rank],
) -> Vec<Rank> {
    let n_nodes = mesh.n_nodes();
    let mut owners = vec![-1_i32; n_nodes];
    for (e, &rank) in elem_part.iter().enumerate() {
        for &n in mesh.elem_nodes(e as u32) {
            if owners[n as usize] < 0 {
                owners[n as usize] = rank;
            }
        }
    }
    for o in &mut owners {
        if *o < 0 { *o = 0; }
    }
    owners
}

/// Extract boundary faces that belong to this rank.
///
/// Assignment rule: a boundary face belongs to rank `r` iff the minimum
/// global node ID among its nodes is owned by `r`.  This is uniquely defined
/// and consistent across ranks.
#[allow(clippy::type_complexity)]
fn extract_local_faces<const D: usize>(
    mesh: &Mesh<D>,
    g2l: &HashMap<NodeId, u32>,
    node_owners: &[Rank],
    local_rank: Rank,
) -> (
    Vec<NodeId>,
    Vec<i32>,
    Option<Vec<ElementType>>,
    Option<Vec<usize>>,
) {
    let n_bfaces = mesh.n_faces();
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mixed_bnd = mesh.face_offsets.is_some();
    let mut face_types_loc = if mixed_bnd { Some(Vec::new()) } else { None };
    let mut face_offsets_loc = if mixed_bnd { Some(vec![0usize]) } else { None };

    for f in 0..n_bfaces as u32 {
        let bnodes = mesh.bface_nodes(f as FaceId);

        // All face nodes must be present in the local node set.
        if bnodes.iter().any(|gn| !g2l.contains_key(gn)) {
            continue;
        }

        // Assign to the rank owning the minimum-index face node.
        let min_gn = *bnodes.iter().min().expect("face has no nodes");
        if node_owners[min_gn as usize] != local_rank {
            continue;
        }

        if mixed_bnd {
            face_types_loc
                .as_mut()
                .expect("face_types")
                .push(mesh.face_type_at(f as FaceId));
        }
        for &gn in bnodes {
            face_conn.push(g2l[&gn]);
        }
        face_tags.push(mesh.face_tags[f as usize]);
        if let Some(ref mut off) = face_offsets_loc {
            off.push(face_conn.len());
        }
    }

    (face_conn, face_tags, face_types_loc, face_offsets_loc)
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::MeshTopology;
    use crate::mpi_test_env::test_world_comm;

    fn serial_comm() -> Comm {
        test_world_comm()
    }

    #[test]
    fn serial_partition_counts() {
        let n = 4usize;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let comm = serial_comm();

        let pmesh = partition_simplex(&mesh, &comm);

        // Global stats must match the original serial mesh.
        assert_eq!(pmesh.global_n_nodes(), mesh.n_nodes(),
            "global node count mismatch");
        assert_eq!(pmesh.global_n_elems(), mesh.n_elems(),
            "global element count mismatch");

        // Single-rank: all nodes are owned, none are ghost.
        assert_eq!(pmesh.n_owned_nodes(), mesh.n_nodes());
        assert_eq!(pmesh.n_ghost_nodes(), 0);

        // Local mesh counts equal global (single rank).
        assert_eq!(pmesh.local_mesh().n_nodes(), mesh.n_nodes());
        assert_eq!(pmesh.local_mesh().n_elements(), mesh.n_elems());
        assert_eq!(pmesh.local_mesh().n_boundary_faces(), mesh.n_faces());
    }

    #[test]
    fn serial_partition_coords_preserved() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);

        // All node coordinates must be preserved.
        for n in 0..mesh.n_nodes() as u32 {
            let orig = mesh.node_coords(n);
            let local = pmesh.node_coords(n);
            assert_eq!(orig, local,
                "coords mismatch for node {n}");
        }
    }

    #[test]
    fn serial_partition_connectivity_preserved() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);

        // Single-rank: element connectivity is identical to serial.
        for e in 0..mesh.n_elems() as u32 {
            assert_eq!(
                mesh.element_nodes(e),
                pmesh.element_nodes(e),
                "connectivity mismatch for element {e}"
            );
        }
    }

    #[test]
    fn serial_partition_global_id_mapping() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);

        // Single-rank: global IDs == local IDs.
        for lid in 0..mesh.n_nodes() as u32 {
            assert_eq!(pmesh.global_node_id(lid), lid,
                "global_node_id mismatch at lid={lid}");
        }
        for lid in 0..mesh.n_elems() as u32 {
            assert_eq!(pmesh.global_elem_id(lid), lid,
                "global_elem_id mismatch at lid={lid}");
        }
    }

    #[test]
    fn serial_partition_global_sum() {
        let n = 4usize;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);

        // Sum x-coordinates of all owned nodes via allreduce.
        // unit_square_tri(4): nodes at (i/4, j/4) for i,j in 0..=4.
        // Sum of x = (n+1)² / 2 = 5² / 2 = 12.5
        let xs: Vec<f64> = (0..pmesh.n_total_nodes())
            .map(|lid| pmesh.node_coords(lid as u32)[0])
            .collect();
        let global_sum_x = pmesh.global_sum_owned(&xs);
        let expected = (n + 1) as f64 * (n + 1) as f64 / 2.0;
        assert!(
            (global_sum_x - expected).abs() < 1e-12,
            "global sum of x coords = {global_sum_x}, expected {expected}"
        );
    }

    #[test]
    fn serial_partition_ghost_exchange_trivial() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);

        // Single rank: ghost exchange is a no-op.
        assert!(pmesh.ghost_exchange().is_trivial());
        assert_eq!(pmesh.ghost_exchange().n_neighbours(), 0);

        let mut data: Vec<f64> = (0..pmesh.n_total_nodes()).map(|i| i as f64).collect();
        let original = data.clone();
        pmesh.forward_exchange(&mut data);
        assert_eq!(data, original, "forward exchange mutated data (should be no-op)");
    }

    #[test]
    fn local_mesh_passes_check() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);
        pmesh.local_mesh().check().expect("local mesh check failed");
    }

    /// `Prism6` with Tri3 caps + Quad4 sides — must keep `face_offsets` through extraction.
    fn unit_prism_mixed_boundary() -> Mesh<3> {
        let coords: Vec<f64> = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
        ];
        let conn = vec![0u32, 1, 2, 3, 4, 5];
        let elem_tags = vec![1i32];
        let face_conn: Vec<u32> = vec![
            0, 1, 2,
            3, 4, 5,
            0, 1, 4, 3,
            1, 2, 5, 4,
            2, 0, 3, 5,
        ];
        let face_tags = vec![1i32, 2, 3, 3, 3];
        let face_types = vec![
            ElementType::Tri3,
            ElementType::Tri3,
            ElementType::Quad4,
            ElementType::Quad4,
            ElementType::Quad4,
        ];
        let face_offsets = vec![0usize, 3, 6, 10, 14, 18];
        let mut m = Mesh::uniform(
            coords,
            conn,
            elem_tags,
            ElementType::Prism6,
            face_conn,
            face_tags,
            ElementType::Tri3,
        );
        m.face_types = Some(face_types);
        m.face_offsets = Some(face_offsets);
        m
    }

    #[test]
    fn extract_submesh_preserves_mixed_prism_boundary() {
        let mesh = unit_prism_mixed_boundary();
        mesh.check().expect("fixture");
        let elem_part = vec![0 as Rank; mesh.n_elems()];
        let (local, _) = extract_submesh_from_partition(&mesh, 0, &elem_part);
        assert_eq!(local.n_faces(), 5);
        assert!(local.face_offsets.is_some());
        assert_eq!(local.face_offsets.as_ref().unwrap().len(), 6);
        local.check().expect("local mesh");
    }
}
