# fem-io-hdf5-parallel

Parallel HDF5 I/O helper crate for fem-rs.

## Scope

This crate provides:

- Rank-partitioned HDF5 read/write (`/steps/.../partitions/rank_XXXXXX/...`)
- Checkpoint metadata and schema versioning
- Bundle writer with optional mesh metadata (`/steps/.../mesh_meta/*`)
- Rank-local restart reads by step or latest step
- Layout validator for checkpoint consistency checks
- Global field materialization for visualization
- Global field read helpers (`read_global_field_f64`, `read_global_field_slice_f64`)
- Minimal XDMF sidecar writers (single-step and temporal collection)

## Features

- `default` (pure-Rust backend): API is fully usable without native HDF5 installation.
- `hdf5`: Enables native HDF5 backend via crate `hdf5` (optional).
- `hdf5-mpi`: Enables MPI-coordinated native backend (rank writes + direct hyperslab global writes, with root materialization compatibility fallback).

## Typical flow

1. Each rank writes local chunks with `write_checkpoint_step_f64`.
2. Build a contiguous field for visualization with `materialize_global_field_f64`.
3. Emit XDMF sidecar with `write_xdmf_polyvertex_scalar_timeseries_sidecar`.
4. Validate checkpoint with `validate_checkpoint_layout`.
5. Restart with `read_checkpoint_field_f64_latest` (or `_at_step`).

Backend selection API:

- `IoBackend::Partitioned` (stable baseline)
- `IoBackend::MpiCollective` (MPI-coordinated checkpoint path with direct HDF5 hyperslab global write path)

## Build notes for optional `hdf5`

Pure-Rust default builds do not require native HDF5. Native `hdf5` builds are optional.

`hdf5-sys` requires HDF5 headers/libs. On systems without HDF5 installed,
`cargo check --features hdf5` may fail with a missing HDF5 root error.

Set one of these before building:

- `HDF5_DIR` to your HDF5 installation root
- or `HDF5_INCLUDE_DIR` and `HDF5_LIB_DIR`

Example (PowerShell):

```powershell
$env:HDF5_DIR = "C:\hdf5"
cargo check -p fem-io-hdf5-parallel --features hdf5
```
