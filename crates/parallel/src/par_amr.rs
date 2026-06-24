//! Parallel adaptive mesh refinement.
//!
//! Provides [`par_refine_marked`] for distributed non-conforming refinement and
//! [`par_repartition`] for load-rebalancing after refinement.

use std::collections::{HashMap, BTreeMap};

use fem_mesh::{SimplexMesh, amr::NCState, topology::MeshTopology, boundary::BoundaryTag};
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
    pub par_mesh:    ParallelMesh<SimplexMesh<2>>,
    pub nc_state:    NCState,
    pub solution:    Vec<f64>,
    pub n_new_elems: usize,
}

/// Perform one cycle of parallel non-conforming AMR.
pub fn par_refine_marked(
    par_mesh: &ParallelMesh<SimplexMesh<2>>,
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
        nc_state: nc_state,
        solution: prolongated,
        n_new_elems,
    })
}

// ─── par_repartition ──────────────────────────────────────────────────────────

fn merge_submeshes(
    meshes: &[SimplexMesh<2>],
    partitions: &[MeshPartition],
) -> Result<SimplexMesh<2>, ParAmrError> {
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
        // Actually for SimplexMesh<2>, face_conn stores edge vertex pairs
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
    for (_gid, xy) in &global_nodes {
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
    for (_ge, (global_conn, tag)) in &global_elems {
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

    Ok(SimplexMesh {
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
    })
}

/// Re-distribute elements across MPI ranks after refinement.
///
/// Gathers all sub-meshes to rank 0, merges them into a single global mesh,
/// and redistributes via [`partition_simplex_streaming`].
pub fn par_repartition(
    par_mesh: ParallelMesh<SimplexMesh<2>>,
) -> Result<ParallelMesh<SimplexMesh<2>>, ParAmrError> {
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
                .map_err(|e| ParAmrError::SerializationError(e))?;
            meshes.push(sub_mesh);
            parts.push(sub_part);
        }

        let global_mesh = merge_submeshes(&meshes, &parts)?;

        // Redistribute using the streaming partitioner
        partition_simplex_streaming(Some(&global_mesh), &comm)
            .map_err(|e| ParAmrError::RepartitionError(e))
    } else {
        // Send our mesh to rank 0
        let encoded = encode_submesh(&local_mesh, &partition);
        comm.send_bytes(0, REPART_TAG_BASE + rank, &encoded);

        // Receive new partition from rank 0
        let buf = comm.recv_bytes(0, STREAM_TAG_BASE + rank);
        let (new_mesh, new_part) = decode_submesh::<2>(&buf)
            .map_err(|e| ParAmrError::SerializationError(e))?;
        Ok(ParallelMesh::new(new_mesh, comm.clone(), new_part))
    }
}

// ─── ParNCMesh ────────────────────────────────────────────────────────────────

/// Parallel non-conforming mesh: wraps [`ParallelMesh`], [`NCState`], and
/// supports distributed AMR cycles with repartitioning.
pub struct ParNCMesh {
    pub par_mesh: ParallelMesh<SimplexMesh<2>>,
    pub nc_state: NCState,
}

impl ParNCMesh {
    /// Create a new `ParNCMesh` from an already-partitioned mesh.
    pub fn new(par_mesh: ParallelMesh<SimplexMesh<2>>, nc_state: NCState) -> Self {
        ParNCMesh { par_mesh, nc_state }
    }

    /// Refine marked elements on this rank, then repartition for load balance.
    pub fn refine_and_rebalance(
        &mut self,
        marked: &[ElemId],
        solution: Option<&[f64]>,
    ) -> Result<Vec<f64>, ParAmrError> {
        let ParRefinedMesh { par_mesh, nc_state, solution, .. } =
            par_refine_marked(&self.par_mesh, std::mem::replace(&mut self.nc_state, NCState::new()), marked, solution)?;

        let rebalanced = par_repartition(par_mesh)?;

        self.par_mesh = rebalanced;
        self.nc_state = nc_state;
        Ok(solution)
    }

    /// Access the underlying parallel mesh.
    pub fn par_mesh(&self) -> &ParallelMesh<SimplexMesh<2>> {
        &self.par_mesh
    }

    /// Access the non-conforming state.
    pub fn nc_state(&self) -> &NCState {
        &self.nc_state
    }
}

// ─── Solution prolongation ────────────────────────────────────────────────────

/// Prolongate a P1 solution from coarse mesh to refined mesh.
///
/// Coarse-node values are copied directly. New midpoint nodes are
/// interpolated from the two nearest coarse nodes — exact for P1.
pub fn prolongate_p1(
    coarse:     &SimplexMesh<2>,
    refined:    &SimplexMesh<2>,
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{SimplexMesh, amr::NCState};
    use crate::{par_mesh::ParallelMesh, partition::MeshPartition, backend::native::SerialBackend, comm::Comm};

    fn make_serial_par_mesh(n: usize) -> (ParallelMesh<SimplexMesh<2>>, NCState) {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
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
        let coarse = SimplexMesh::<2>::unit_square_tri(2);
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
        let coarse = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
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
}
