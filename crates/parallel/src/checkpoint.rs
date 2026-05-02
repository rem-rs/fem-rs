//! # Checkpoint / restart for parallel FEM simulations
//!
//! Provides save/restore of distributed solver state (vectors, matrices,
//! time, step) to/from HDF5 files.
//!
//! ## File layout
//!
//! Each rank writes its local portion to `{path}_rank_{rank}.chk`.  Rank 0
//! additionally writes a small metadata file `{path}.meta` with scalar values
//! and field names that are identical across all ranks.
//!
//! Per-rank checkpoint file:
//! ```text
//! /time             — f64 (wall-clock time)
//! /step             — u64 (timestep counter)
//! /fields/<name>    — [n_owned] f64 (local owned DOF values)
//! /matrix/diag/
//!   row_ptr         — [n_owned+1] i64  (CSR row pointers)
//!   col_idx         — [nnz] i64         (CSR column indices)
//!   values          — [nnz] f64         (CSR values)
//! /matrix/offd/
//!   row_ptr         — [n_owned+1] i64
//!   col_idx         — [nnz] i64
//!   values          — [nnz] f64
//! ```
//!
//! Metadata file (rank 0 only):
//! ```text
//! /time             — f64
//! /step             — u64
//! /n_fields         — u64
//! /field_names      — [n_fields] string (variable-length)
//! ```

#![cfg(feature = "hdf5")]

use std::sync::Arc;
use std::str::FromStr;

use fem_core::{FemError, FemResult};
use fem_linalg::CsrMatrix;
use fem_mesh::SimplexMesh;
use fem_space::fe_space::FESpace;

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_csr::ParCsrMatrix;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;

// ── CheckpointData ───────────────────────────────────────────────────────────

/// Distributed solver state to checkpoint.
pub struct CheckpointData {
    /// Simulation time.
    pub time: f64,
    /// Timestep count.
    pub step: u64,
    /// Named field vectors (each rank holds its local portion).
    pub fields: Vec<(String, ParVector)>,
    /// Optional distributed matrix.
    pub matrix: Option<ParCsrMatrix>,
}

// ── Writer ───────────────────────────────────────────────────────────────────

/// Write a distributed checkpoint.
///
/// Each rank writes its local data to `{path}_rank_{rank}.chk`.
/// Rank 0 additionally writes `{path}.meta` with global metadata.
pub fn write_checkpoint(path: &str, data: &CheckpointData, comm: &Comm) -> FemResult<()> {
    let rank = comm.rank();
    let rank_path = format!("{path}_rank_{rank}.chk");

    let file = hdf5::File::create(&rank_path).map_err(|e| hdf5_error(e))?;

    // Write time and step
    write_scalar(&file, "time", &data.time)?;
    write_scalar(&file, "step", &data.step)?;

    // Write fields
    if !data.fields.is_empty() {
        let fg = file.create_group("fields").map_err(|e| hdf5_error(e))?;
        for (name, vec) in &data.fields {
            let grp = fg.create_group(name.as_str()).map_err(|e| hdf5_error(e))?;
            let owned = vec.owned_slice();
            let dset = grp
                .new_dataset::<f64>()
                .shape((owned.len(),))
                .create("values")
                .map_err(|e| hdf5_error(e))?;
            dset.write(owned).map_err(|e| hdf5_error(e))?;
        }
    }

    // Write matrix
    if let Some(ref mat) = data.matrix {
        let mg = file.create_group("matrix").map_err(|e| hdf5_error(e))?;

        write_csr_block(&mg, "diag", &mat.diag)?;
        write_csr_block(&mg, "offd", &mat.offd)?;
    }

    // ── Rank 0 writes metadata ──────────────────────────────────────────────
    if comm.is_root() {
        let meta_path = format!("{path}.meta");
        let meta_file = hdf5::File::create(&meta_path).map_err(|e| hdf5_error(e))?;

        write_scalar(&meta_file, "time", &data.time)?;
        write_scalar(&meta_file, "step", &data.step)?;

        let n_fields = data.fields.len() as u64;
        write_scalar(&meta_file, "n_fields", &n_fields)?;

        if !data.fields.is_empty() {
            use hdf5::types::VarLenUnicode;
            let fg = meta_file
                .create_group("fields")
                .map_err(|e| hdf5_error(e))?;
            // Write field names as a 1D dataset of VarLenUnicode strings
            let names: Vec<VarLenUnicode> = data
                .fields
                .iter()
                .map(|(n, _)| VarLenUnicode::from_str(n).unwrap())
                .collect();
            let dset = fg
                .new_dataset::<hdf5::types::VarLenUnicode>()
                .shape((names.len(),))
                .create("names")
                .map_err(|e| hdf5_error(e))?;
            dset.write(&names).map_err(|e| hdf5_error(e))?;
        }
    }

    comm.barrier();
    Ok(())
}

// ── Reader ───────────────────────────────────────────────────────────────────

/// Read a distributed checkpoint, restoring per-rank state.
///
/// The `dof_ghost_exchange`, `space`, and `local_mesh` arguments are used to
/// reconstruct `ParVector` and `ParCsrMatrix` instances with the correct
/// local layouts.  Only owned DOF values are restored; ghost slots are
/// zero-initialised (the caller should call `update_ghosts()` if needed).
pub fn read_checkpoint<const D: usize>(
    path: &str,
    comm: &Comm,
    dof_ghost_exchange: Arc<GhostExchange>,
    space: &ParallelFESpace<impl FESpace>,
    _local_mesh: &SimplexMesh<D>,
) -> FemResult<CheckpointData> {
    let rank = comm.rank();
    let rank_path = format!("{path}_rank_{rank}.chk");

    let file = hdf5::File::open(&rank_path).map_err(|e| hdf5_error(e))?;

    // Read time and step
    let time: f64 = read_scalar(&file, "time")?;
    let step: u64 = read_scalar(&file, "step")?;

    // Read metadata (available on all ranks from their own file)
    let meta_fields: Vec<(String, usize)> = if let Ok(fg) = file.group("fields") {
        let mut names = Vec::new();
        for member in fg.member_names().map_err(|e| hdf5_error(e))? {
            let grp = fg.group(&member).map_err(|e| hdf5_error(e))?;
            let dset = grp.dataset("values").map_err(|e| hdf5_error(e))?;
            let shape = dset.shape();
            let n = shape.first().copied().unwrap_or(0);
            names.push((member, n));
        }
        names
    } else {
        Vec::new()
    };

    // Read field data and reconstruct ParVectors
    let n_owned = space.dof_partition().n_owned_dofs;
    let n_ghost = space.dof_partition().n_ghost_dofs;
    let ge = dof_ghost_exchange;
    let comm_clone = comm.clone();

    let mut fields = Vec::with_capacity(meta_fields.len());
    for (name, _n) in &meta_fields {
        let mut vec = ParVector::zeros_raw(n_owned, n_ghost, Arc::clone(&ge), comm_clone.clone());

        let owned_slice = vec.owned_slice_mut();
        if let Ok(fg) = file.group("fields") {
            if let Ok(grp) = fg.group(name) {
                if let Ok(dset) = grp.dataset("values") {
                    let data: Vec<f64> = dset.read_1d().map_err(|e| hdf5_error(e))?.to_vec();
                    let len = data.len().min(owned_slice.len());
                    owned_slice[..len].copy_from_slice(&data[..len]);
                }
            }
        }
        fields.push((name.clone(), vec));
    }

    // Read matrix
    let matrix = if let Ok(mg) = file.group("matrix") {
        let diag = read_csr_block(&mg, "diag")?;
        let offd = read_csr_block(&mg, "offd")?;
        let n_ghost_mat = offd.ncols;
        Some(ParCsrMatrix::from_blocks(
            diag,
            offd,
            n_owned,
            n_ghost_mat,
            ge,
            comm_clone,
        ))
    } else {
        None
    };

    Ok(CheckpointData {
        time,
        step,
        fields,
        matrix,
    })
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn hdf5_error(e: hdf5::Error) -> FemError {
    FemError::Io(std::io::Error::new(
        std::io::ErrorKind::Other,
        e.to_string(),
    ))
}

fn write_scalar<T: hdf5::H5Type>(file: &hdf5::File, name: &str, val: &T) -> FemResult<()> {
    let dset = file
        .new_dataset::<T>()
        .shape((1,))
        .create(name)
        .map_err(|e| hdf5_error(e))?;
    dset.write(std::slice::from_ref(val)).map_err(|e| hdf5_error(e))?;
    Ok(())
}

fn read_scalar<T: hdf5::H5Type + Copy>(file: &hdf5::File, name: &str) -> FemResult<T> {
    let dset = file.dataset(name).map_err(|e| hdf5_error(e))?;
    let data: Vec<T> = dset.read_1d().map_err(|e| hdf5_error(e))?.to_vec();
    if data.is_empty() {
        return Err(FemError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("checkpoint: dataset '{name}' is empty"),
        )));
    }
    Ok(data[0])
}

fn write_csr_block(parent: &hdf5::Group, name: &str, csr: &CsrMatrix<f64>) -> FemResult<()> {
    let grp = parent.create_group(name).map_err(|e| hdf5_error(e))?;

    // row_ptr: [nrows+1] i64
    let rp_i64: Vec<i64> = csr.row_ptr.iter().map(|&v| v as i64).collect();
    let dset = grp
        .new_dataset::<i64>()
        .shape((rp_i64.len(),))
        .create("row_ptr")
        .map_err(|e| hdf5_error(e))?;
    dset.write(rp_i64.as_slice()).map_err(|e| hdf5_error(e))?;

    // col_idx: [nnz] i64 (CsrMatrix uses u32 internally)
    let ci_i64: Vec<i64> = csr.col_idx.iter().map(|&v| v as i64).collect();
    let dset = grp
        .new_dataset::<i64>()
        .shape((ci_i64.len(),))
        .create("col_idx")
        .map_err(|e| hdf5_error(e))?;
    dset.write(ci_i64.as_slice()).map_err(|e| hdf5_error(e))?;

    // values: [nnz] f64
    let dset = grp
        .new_dataset::<f64>()
        .shape((csr.values.len(),))
        .create("values")
        .map_err(|e| hdf5_error(e))?;
    dset.write(csr.values.as_slice())
        .map_err(|e| hdf5_error(e))?;

    Ok(())
}

fn read_csr_block(parent: &hdf5::Group, name: &str) -> FemResult<CsrMatrix<f64>> {
    let grp = parent.group(name).map_err(|e| hdf5_error(e))?;

    // row_ptr
    let rp_raw: Vec<i64> = grp
        .dataset("row_ptr")
        .map_err(|e| hdf5_error(e))?
        .read_1d()
        .map_err(|e| hdf5_error(e))?
        .to_vec();
    let row_ptr: Vec<usize> = rp_raw.iter().map(|&v| v as usize).collect();

    let nrows = if row_ptr.is_empty() {
        0
    } else {
        row_ptr.len() - 1
    };

    // col_idx
    let ci_raw: Vec<i64> = grp
        .dataset("col_idx")
        .map_err(|e| hdf5_error(e))?
        .read_1d()
        .map_err(|e| hdf5_error(e))?
        .to_vec();
    let col_idx: Vec<u32> = ci_raw.iter().map(|&v| v as u32).collect();

    // values
    let values: Vec<f64> = grp
        .dataset("values")
        .map_err(|e| hdf5_error(e))?
        .read_1d()
        .map_err(|e| hdf5_error(e))?
        .to_vec();

    let ncols = col_idx
        .iter()
        .max()
        .copied()
        .map(|c| c as usize + 1)
        .unwrap_or(0);

    Ok(CsrMatrix {
        nrows,
        ncols,
        row_ptr,
        col_idx,
        values,
    })
}
