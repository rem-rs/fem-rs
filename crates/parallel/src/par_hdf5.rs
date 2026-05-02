//! # Parallel HDF5 I/O coordination
//!
//! Provides coordinated write operations for distributed mesh + field data.
//!
//! ## Modes
//!
//! * **Per-rank mode** (default): Each MPI rank writes its own HDF5 file
//!   (`{base_path}_rank_{rank}.h5`) containing its local mesh and field
//!   data.  Rank 0 then writes an XDMF metadata file (`{base_path}.xmf`)
//!   referencing all rank files.
//!
//! * **Gather mode**: All ranks send their local data to rank 0, which
//!   assembles the full mesh and fields and writes a single HDF5 file
//!   (`{base_path}.h5`) plus an XDMF file.
//!
//! This module is feature-gated behind `cfg(feature = "hdf5")`.


use fem_core::FemResult;
use fem_io::hdf5::{write_mesh_and_fields, Hdf5WriteOptions};
use fem_io::xdmf::{write_xdmf, XdmfCenter, XdmfField};
use fem_mesh::{ElementType, SimplexMesh};

use crate::comm::Comm;
use crate::par_vector::ParVector;

// ── Write mode ───────────────────────────────────────────────────────────────

/// Whether to write per-rank files or a single gathered file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelWriteMode {
    /// Each rank writes its own file; rank 0 writes XDMF metadata.
    PerRank,
    /// All data is gathered to rank 0, which writes one file + XDMF.
    Gather,
}

impl Default for ParallelWriteMode {
    fn default() -> Self {
        ParallelWriteMode::PerRank
    }
}

// ── Extended write options ───────────────────────────────────────────────────

/// Options for parallel HDF5 writes.
#[derive(Debug, Clone)]
pub struct ParHdf5Options {
    /// Underlying HDF5 write options (compression, overwrite).
    pub hdf5_options: Hdf5WriteOptions,
    /// Parallel write mode.
    pub mode: ParallelWriteMode,
}

impl Default for ParHdf5Options {
    fn default() -> Self {
        Self {
            hdf5_options: Hdf5WriteOptions::default(),
            mode: ParallelWriteMode::PerRank,
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Write a distributed mesh + fields to HDF5 files + XDMF metadata.
///
/// # Arguments
///
/// * `mesh` — Local sub-mesh for this rank (see `ParallelMesh::local_mesh()`).
/// * `fields` — Named field vectors with their space type (e.g. `"H1"`).
/// * `base_path` — Base path for output files (without extension).
/// * `comm` — MPI communicator.
/// * `options` — Write options including mode selection.
pub fn par_write_mesh_and_fields<const D: usize>(
    mesh: &SimplexMesh<D>,
    fields: &[(&str, &ParVector, &str)],
    base_path: &str,
    comm: &Comm,
    options: &ParHdf5Options,
) -> FemResult<()> {
    match options.mode {
        ParallelWriteMode::PerRank => {
            par_write_per_rank(mesh, fields, base_path, comm, &options.hdf5_options)
        }
        ParallelWriteMode::Gather => {
            par_write_gather(mesh, fields, base_path, comm, &options.hdf5_options)
        }
    }
}

// ── Per-rank mode ────────────────────────────────────────────────────────────

/// Per-rank write: each rank writes its own HDF5, rank 0 writes XDMF.
fn par_write_per_rank<const D: usize>(
    mesh: &SimplexMesh<D>,
    fields: &[(&str, &ParVector, &str)],
    base_path: &str,
    comm: &Comm,
    h5_opts: &Hdf5WriteOptions,
) -> FemResult<()> {
    let _rank = comm.rank();
    let n_ranks = comm.size();

    // ── Each rank writes its own HDF5 ────────────────────────────────────────
    let rank_path = format!("{base_path}_rank_{_rank}.h5");

    let fields_prepared: Vec<(&str, &[f64], &str)> = fields
        .iter()
        .map(|(name, vec, space)| (*name, vec.owned_slice(), *space))
        .collect();

    write_mesh_and_fields(&rank_path, mesh, &fields_prepared, h5_opts).map_err(|e| {
        fem_core::FemError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("HDF5 write error on rank {_rank}: {e}"),
        ))
    })?;

    // ── Gather metadata on rank 0 ────────────────────────────────────────────
    let _n_nodes_local = mesh.n_nodes();
    let _n_elems_local = mesh.n_elems();

    let all_n_nodes = gather_usizes(_n_nodes_local, comm);
    let all_n_elems = gather_usizes(_n_elems_local, comm);

    if comm.is_root() {
        let xdmf_fields: Vec<XdmfField> = fields
            .iter()
            .map(|(name, _vec, _space)| XdmfField {
                name: name.to_string(),
                hdf5_path: format!("{base_path}_rank_{{}}.h5"),
                dataset_path: format!("/fields/{name}/values"),
                center: XdmfCenter::Node,
            })
            .collect();

        let xdmf_path = format!("{base_path}.xmf");

        write_xdmf(
            &xdmf_path,
            n_ranks,
            mesh.elem_type,
            D,
            &all_n_nodes,
            &all_n_elems,
            &format!("{base_path}_rank_{{}}.h5"),
            &xdmf_fields,
        )?;
    }

    comm.barrier();
    Ok(())
}

// ── Gather mode ──────────────────────────────────────────────────────────────

/// Gather mode: rank 0 collects all local sub-meshes + fields and writes one file.
fn par_write_gather<const D: usize>(
    mesh: &SimplexMesh<D>,
    fields: &[(&str, &ParVector, &str)],
    base_path: &str,
    comm: &Comm,
    h5_opts: &Hdf5WriteOptions,
) -> FemResult<()> {
    let _rank = comm.rank();
    let n_ranks = comm.size();

    let _n_nodes_local = mesh.n_nodes();
    let _n_elems_local = mesh.n_elems();

    let all_n_nodes = gather_usizes(_n_nodes_local, comm);
    let all_n_elems = gather_usizes(_n_elems_local, comm);

    // Gather mesh data to rank 0
    let gathered_mesh = gather_mesh(mesh, n_ranks, &all_n_nodes, &all_n_elems, comm)?;

    // Gather field data to rank 0
    let gathered_fields: Vec<Vec<Vec<f64>>> = fields
        .iter()
        .map(|(_, vec, _)| {
            let local_data = vec.owned_slice().to_vec();
            gather_f64s(&local_data, n_ranks, comm)
        })
        .collect();

    if comm.is_root() {
        // Combine gathered data
        let all_coords = gathered_mesh.coords;
        let all_conn = gathered_mesh.conn;
        let all_elem_tags = gathered_mesh.elem_tags;

        // Create unified mesh
        let unified_elem_type = gathered_mesh.elem_type;
        let n_total_nodes = all_coords.len() / D;
        let n_total_elems = all_conn.len() / unified_elem_type.nodes_per_element();

        let unified_mesh: SimplexMesh<D> = SimplexMesh {
            coords: all_coords,
            conn: all_conn,
            elem_tags: all_elem_tags,
            elem_type: unified_elem_type,
            face_conn: vec![],
            face_tags: vec![],
            face_type: mesh.face_type,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
        };

        // Build gathered field slices
        let field_slices: Vec<(&str, Vec<f64>, &str)> = fields
            .iter()
            .zip(gathered_fields.iter())
            .enumerate()
            .map(|(_i, ((name, _, space), gathered))| {
                let flat: Vec<f64> = gathered.iter().flat_map(|v| v.iter()).copied().collect();
                (*name, flat, *space)
            })
            .collect();

        let refs: Vec<(&str, &[f64], &str)> = field_slices
            .iter()
            .map(|(n, v, s)| (*n, v.as_slice(), *s))
            .collect();

        let h5_path = format!("{base_path}.h5");
        write_mesh_and_fields(&h5_path, &unified_mesh, &refs, h5_opts).map_err(|e| {
            fem_core::FemError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("HDF5 gather write error: {e}"),
            ))
        })?;

        // Write XDMF
        let xdmf_fields: Vec<XdmfField> = fields
            .iter()
            .map(|(name, _vec, _space)| XdmfField {
                name: name.to_string(),
                hdf5_path: format!("{base_path}.h5"),
                dataset_path: format!("/fields/{name}/values"),
                center: XdmfCenter::Node,
            })
            .collect();

        write_xdmf(
            format!("{base_path}.xmf"),
            1, // serial XDMF (single HDF5 file)
            mesh.elem_type,
            D,
            &[n_total_nodes],
            &[n_total_elems],
            &format!("{base_path}.h5"),
            &xdmf_fields,
        )?;
    }

    comm.barrier();
    Ok(())
}

// ── Communication helpers ────────────────────────────────────────────────────

/// Gather a `usize` from every rank to a `Vec<usize>` on rank 0.
fn gather_usizes(local: usize, comm: &Comm) -> Vec<usize> {
    let n_ranks = comm.size();
    let mut result = vec![0usize; n_ranks];

    if n_ranks == 1 {
        result[0] = local;
        return result;
    }

    // Convert to u64 for network transport
    let local_u64 = local as u64;
    let bytes = bytemuck::bytes_of(&local_u64);

    if comm.is_root() {
        result[0] = local;
        for src in 1..n_ranks {
            let recv_bytes = comm.recv_bytes(src as i32, 1001);
            let val =
                u64::from_le_bytes(recv_bytes.try_into().expect("gather_usizes: bad recv size"));
            result[src] = val as usize;
        }
    } else {
        comm.send_bytes(0, 1001, bytes);
    }

    result
}

/// Gather `&[f64]` from every rank to a `Vec<Vec<f64>>` on rank 0.
fn gather_f64s(local: &[f64], n_ranks: usize, comm: &Comm) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(n_ranks);

    if n_ranks == 1 {
        result.push(local.to_vec());
        return result;
    }

    if comm.is_root() {
        result.push(local.to_vec());
        for src in 1..n_ranks {
            let recv = comm.recv::<f64>(src as i32, 1002);
            result.push(recv);
        }
    } else {
        comm.send(0, 1002, local);
    }

    result
}

/// Gather the local mesh to rank 0 and reassemble a unified `SimplexMesh`.
struct GatheredMeshData {
    coords: Vec<f64>,
    conn: Vec<u32>, // global node IDs
    elem_tags: Vec<i32>,
    elem_type: ElementType,
    face_type: ElementType,
}

fn gather_mesh<const D: usize>(
    mesh: &SimplexMesh<D>,
    n_ranks: usize,
    all_n_nodes: &[usize],
    all_n_elems: &[usize],
    comm: &Comm,
) -> FemResult<GatheredMeshData> {
    if comm.is_root() {
        // Compute cumulative node/elem offsets
        let mut node_offset = 0usize;
        let mut _elem_offset = 0usize;

        let mut all_coords = mesh.coords.clone();
        let mut all_conn: Vec<u32> = mesh.conn.iter().map(|&n| n + node_offset as u32).collect();
        let mut all_elem_tags = mesh.elem_tags.clone();

        // The rank 0 local mesh already uses local node IDs 0..n_owned.
        // We need to remap them to global: global = local + cumulative_node_offset.
        // Actually, the local mesh uses node IDs that are local to this rank.
        // For a proper gather, we'd need node ownership info from the partitioner.
        // For simplicity, we treat each rank's local nodes as consecutive blocks.
        //
        // NOTE: This simplified gather assumes contiguous per-rank node ranges,
        // which matches the contiguous partition strategy used by partition_simplex.
        node_offset += all_n_nodes[0];
        _elem_offset += all_n_elems[0];

        for src in 1..n_ranks {
            // Receive coords
            let n_src_nodes = all_n_nodes[src];
            let n_src_elems = all_n_elems[src];
            let _npe = mesh.elem_type.nodes_per_element();

            let recv_coords: Vec<f64> = comm.recv(src as i32, 1010);
            let recv_conn: Vec<u32> = comm.recv(src as i32, 1011);
            let recv_tags: Vec<i32> = comm.recv(src as i32, 1012);

            // Remap connectivity to global node IDs
            let remapped_conn: Vec<u32> =
                recv_conn.iter().map(|&n| n + node_offset as u32).collect();

            all_coords.extend_from_slice(&recv_coords);
            all_conn.extend_from_slice(&remapped_conn);
            all_elem_tags.extend_from_slice(&recv_tags);

            node_offset += n_src_nodes;
            _elem_offset += n_src_elems;
        }

        Ok(GatheredMeshData {
            coords: all_coords,
            conn: all_conn,
            elem_tags: all_elem_tags,
            elem_type: mesh.elem_type,
            face_type: mesh.face_type,
        })
    } else {
        // Non-root ranks send their local mesh data
        let _n_nodes_local = mesh.n_nodes();
        let _n_elems_local = mesh.n_elems();

        // Send coords
        comm.send::<f64>(0, 1010, &mesh.coords);
        // Send connectivity as u32
        let conn_u32: Vec<u32> = mesh.conn.clone();
        comm.send::<u32>(0, 1011, &conn_u32);
        // Send elem tags
        comm.send::<i32>(0, 1012, &mesh.elem_tags);

        // Return dummy on non-root
        Ok(GatheredMeshData {
            coords: vec![],
            conn: vec![],
            elem_tags: vec![],
            elem_type: mesh.elem_type,
            face_type: mesh.face_type,
        })
    }
}
