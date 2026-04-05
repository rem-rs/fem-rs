//! Partitioner for [`SimplexMesh<D>`].
//!
//! [`partition_simplex`] distributes a serial `SimplexMesh<D>` across the
//! ranks of a [`Comm`] using a **contiguous element partition**:
//!
//! * Rank `r` owns elements `[r·chunk, (r+1)·chunk)`.
//! * Node ownership: a node is owned by the lowest-rank process whose element
//!   chunk contains it.  Iteration order ensures this is simply the first rank
//!   to "see" the node when sweeping elements 0 → n_elems.
//! * Boundary faces: assigned to the rank that owns the minimum-index node of
//!   the face (unique and consistent across ranks).
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
use fem_mesh::SimplexMesh;

use crate::{Comm, MeshPartition, par_mesh::ParallelMesh};

// ── public entry point ────────────────────────────────────────────────────────

/// Distribute `mesh` across all ranks of `comm`.
///
/// Returns a [`ParallelMesh`] whose local sub-mesh contains only the elements
/// and nodes (owned + ghost) assigned to the calling rank.
///
/// # Panics
/// Panics if the mesh has zero elements.
pub fn partition_simplex<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &Comm,
) -> ParallelMesh<SimplexMesh<D>> {
    let n_elems = mesh.n_elems();
    let n_nodes_total = mesh.n_nodes();
    assert!(n_elems > 0, "partition_simplex: mesh has no elements");

    // ── single-rank fast path ────────────────────────────────────────────────
    if comm.size() == 1 {
        let partition = MeshPartition::new_serial(n_nodes_total, n_elems);
        return ParallelMesh::new(mesh.clone(), comm.clone(), partition);
    }

    // ── multi-rank partitioning ──────────────────────────────────────────────
    let size = comm.size();
    let local_rank = comm.rank();

    // 1. Assign elements to ranks (contiguous blocks).
    let chunk = n_elems.div_ceil(size);
    let e_start = (local_rank as usize * chunk).min(n_elems);
    let e_end = (e_start + chunk).min(n_elems);
    let local_elem_gids: Vec<u32> = (e_start..e_end).map(|e| e as u32).collect();

    // 2. Node ownership: owner = rank of first element containing the node.
    let node_owners = compute_node_owners(mesh, n_nodes_total, n_elems, size);

    // 3. Collect all nodes touched by local (owned) elements.
    let mut node_set: BTreeSet<NodeId> = BTreeSet::new();
    for &ge in &local_elem_gids {
        for &n in mesh.elem_nodes(ge) {
            node_set.insert(n);
        }
    }

    // 3b. Find ghost elements: elements NOT owned by this rank that share at
    // least one node with our owned elements.  This ensures that all
    // contributions to owned-row DOFs are captured during local assembly.
    let mut ghost_elem_gids: Vec<u32> = Vec::new();
    for e in 0..n_elems as u32 {
        let owner_rank = (e as usize / chunk) as Rank;
        if owner_rank == local_rank { continue; }
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
        if owner == local_rank {
            owned_global.push(*gn);
        } else {
            ghost_global.push((*gn, owner));
        }
    }

    // 4. Build global → local node mapping (owned first, then ghost).
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
    let npe = mesh.elem_type.nodes_per_element();
    let all_local_elems = local_elem_gids.len() + ghost_elem_gids.len();
    let mut local_conn = Vec::with_capacity(all_local_elems * npe);
    let mut local_elem_tags = Vec::with_capacity(all_local_elems);
    for &ge in local_elem_gids.iter().chain(ghost_elem_gids.iter()) {
        for &gn in mesh.elem_nodes(ge) {
            local_conn.push(g2l[&gn]);
        }
        local_elem_tags.push(mesh.elem_tags[ge as usize]);
    }

    // 7. Assign boundary faces to this rank.
    let (local_face_conn, local_face_tags) =
        extract_local_faces(mesh, &g2l, &node_owners, local_rank);

    // 8. Assemble the local sub-mesh.
    let local_mesh = SimplexMesh::<D> {
        coords:     local_coords,
        conn:       local_conn,
        elem_tags:  local_elem_tags,
        elem_type:  mesh.elem_type,
        face_conn:  local_face_conn,
        face_tags:  local_face_tags,
        face_type:  mesh.face_type,
    };

    let partition = MeshPartition::from_partitioner(
        &owned_global,
        &ghost_global,
        &local_elem_gids,  // only owned elements in the partition
        local_rank,
    );

    ParallelMesh::new(local_mesh, comm.clone(), partition)
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// For each global node, compute which rank owns it.
///
/// A node is owned by the rank that owns the lowest-indexed element containing
/// it.  Sweeping elements 0 → n_elems in order, the first rank to "see" a node
/// becomes its owner.
fn compute_node_owners<const D: usize>(
    mesh: &SimplexMesh<D>,
    n_nodes: usize,
    n_elems: usize,
    n_ranks: usize,
) -> Vec<Rank> {
    let chunk = n_elems.div_ceil(n_ranks);
    let mut owners = vec![-1_i32; n_nodes];
    for e in 0..n_elems {
        let rank = (e / chunk) as Rank;
        for &n in mesh.elem_nodes(e as u32) {
            if owners[n as usize] < 0 {
                owners[n as usize] = rank;
            }
        }
    }
    // Isolated nodes (not in any element) go to rank 0.
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
fn extract_local_faces<const D: usize>(
    mesh: &SimplexMesh<D>,
    g2l: &HashMap<NodeId, u32>,
    node_owners: &[Rank],
    local_rank: Rank,
) -> (Vec<NodeId>, Vec<i32>) {
    let n_bfaces = mesh.n_faces();
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();

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

        for &gn in bnodes {
            face_conn.push(g2l[&gn]);
        }
        face_tags.push(mesh.face_tags[f as usize]);
    }

    (face_conn, face_tags)
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::MeshTopology;
    use crate::launcher::{Launcher, native::MpiLauncher};

    fn serial_comm() -> Comm {
        // Use MpiLauncher which falls back to SerialBackend when `mpi` feature
        // is not enabled.
        MpiLauncher::init().expect("MPI already initialised").world_comm()
    }

    #[test]
    fn serial_partition_counts() {
        let n = 4usize;
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let comm = serial_comm();
        let pmesh = partition_simplex(&mesh, &comm);
        pmesh.local_mesh().check().expect("local mesh check failed");
    }
}
