# fem-io-hdf5-parallel

Parallel HDF5 I/O helper crate for fem-rs.

## Scope

This crate provides:

- Rank-partitioned checkpoint read/write (`/steps/.../partitions/rank_XXXXXX/...`)
- Checkpoint metadata and schema versioning
- Bundle writer with optional mesh metadata (`/steps/.../mesh_meta/*`)
- Rank-local restart reads by step or latest step
- Layout validator for checkpoint consistency checks
- Global field materialization for visualization
- Global field read helpers (`read_global_field_f64`, `read_global_field_slice_f64`)
- Minimal XDMF sidecar writers (single-step and temporal collection)

## Features

- `default` (pure-Rust backend): MessagePack “portable” storage — no HDF5 dependency.
- `hdf5`: Enables the **pure Rust** [`rust-hdf5`](https://crates.io/crates/rust-hdf5) crate (optional). **No** system `libhdf5` / `hdf5-sys` / legacy `hdf5` crate.
- `hdf5-mpi`: MPI-coordinated checkpoint staging (`mpi` + `hdf5` feature). Rank-partitioned writes plus optional root materialization of global fields. Multi-rank **hyperslab** staging of shared globals is not supported with `rust-hdf5` (see crate `SUPPORTS_HYPERSLAB`).
- `phdf5`: Alias for `hdf5-mpi` (same dependency set).

### Native checkpoint schema (`hdf5` / `rust-hdf5`)

Files use **schema version 2** (`/meta/schema_version` as a length-1 `u64` dataset). Step time and step index are stored as scalar datasets under each `/steps/step_XXXXXXXX/` group. This is **not** byte-compatible with older libhdf5-based checkpoints (schema v1 with group attributes).

## Typical flow

1. Each rank writes local chunks with `write_checkpoint_step_f64`.
2. Build a contiguous field for visualization with `materialize_global_field_f64`.
3. Emit XDMF sidecar with `write_xdmf_polyvertex_scalar_timeseries_sidecar`.
4. Validate checkpoint with `validate_checkpoint_layout`.
5. Restart with `read_checkpoint_field_f64_latest` (or `_at_step`).

Backend selection API:

- `IoBackend::Partitioned` (stable baseline)
- `IoBackend::MpiCollective` (MPI-coordinated path; hyperslab staging is skipped when unsupported)

## Build

Pure-Rust default builds do not pull in `rust-hdf5`. With `--features hdf5`, only the `rust-hdf5` crate is built — **no** `HDF5_DIR` or C toolchain for libhdf5.

Example:

```powershell
cargo check -p fem-io-hdf5-parallel --features hdf5
```
