//! Parallel HDF5 I/O helpers.
//!
//! This crate isolates HDF5-related parallel read/write concerns behind a small,
//! feature-gated API.
//!
//! Design goals:
//! - Keep the workspace buildable without HDF5 installed.
//! - Offer a rank-partitioned file layout that is deterministic and restart-friendly.
//! - Optional **native checkpoint** (`hdf5` feature): pure-Rust [`rust_hdf5`] backend (schema v2),
//!   no system libhdf5.
//!
//! [`rust_hdf5`]: https://crates.io/crates/rust-hdf5

use serde::{Deserialize, Serialize};

#[cfg(feature = "hdf5")]
mod hdf5_rust_impl;

use thiserror::Error;

/// Schema version for on-disk checkpoint layout.
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// I/O backend selection for checkpoint operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoBackend {
    /// Rank-partitioned independent I/O (current stable baseline).
    Partitioned,
    /// MPI-enabled backend (currently compatibility mode over partitioned I/O).
    MpiCollective,
}

/// Runtime configuration for partitioned parallel I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelIoConfig {
    /// Number of ranks participating in the write/read.
    pub world_size: usize,
    /// Current rank id in [0, world_size).
    pub rank: usize,
}

/// Per-rank field payload for checkpoint I/O.
#[derive(Debug, Clone)]
pub struct RankFieldF64 {
    /// Field name (e.g. "u", "pressure", "temperature").
    pub name: String,
    /// Global starting offset for this rank's chunk.
    pub global_offset: u64,
    /// Global length of the full field vector.
    pub global_len: u64,
    /// Local values owned by this rank.
    pub values: Vec<f64>,
}

/// Metadata returned when reading a rank-local field from checkpoint.
#[derive(Debug, Clone)]
pub struct RankFieldReadF64 {
    pub step: u64,
    pub time: f64,
    pub global_offset: u64,
    pub global_len: u64,
    pub values: Vec<f64>,
}

/// Optional mesh-level metadata stored alongside a checkpoint step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMeshMeta {
    pub dim: u8,
    pub n_vertices: u64,
    pub n_elements: u64,
}

/// Bundle payload for one checkpoint step.
#[derive(Debug, Clone)]
pub struct CheckpointBundleF64 {
    pub mesh_meta: Option<CheckpointMeshMeta>,
    pub fields: Vec<RankFieldF64>,
}

/// Full read result for a checkpoint bundle step.
#[derive(Debug, Clone)]
pub struct CheckpointBundleReadF64 {
    pub step: u64,
    pub time: f64,
    pub mesh_meta: Option<CheckpointMeshMeta>,
    pub fields: Vec<(String, RankFieldReadF64)>,
}

/// Per-step checkpoint validation summary.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointStepInfo {
    pub step: u64,
    pub time: f64,
    pub partition_count: usize,
}

/// High-level checkpoint validation output.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointValidationReport {
    pub schema_version: u32,
    pub world_size: usize,
    pub steps: Vec<CheckpointStepInfo>,
    pub warnings: Vec<String>,
}

impl ParallelIoConfig {
    pub fn validate(&self) -> Result<(), Hdf5ParallelError> {
        if self.world_size == 0 {
            return Err(Hdf5ParallelError::InvalidConfig("world_size must be > 0"));
        }
        if self.rank >= self.world_size {
            return Err(Hdf5ParallelError::InvalidConfig("rank must be < world_size"));
        }
        Ok(())
    }

    pub fn rank_group_name(&self) -> String {
        format!("rank_{:06}", self.rank)
    }
}

#[derive(Debug, Error)]
pub enum Hdf5ParallelError {
    #[error("invalid parallel I/O config: {0}")]
    InvalidConfig(&'static str),

    #[error("hdf5 feature is disabled; enable crate feature `hdf5`")]
    Hdf5FeatureDisabled,

    #[error("hdf5-mpi feature is disabled; enable crate feature `hdf5-mpi`")]
    Hdf5MpiFeatureDisabled,

    #[error("HDF5 backend error: {0}")]
    Backend(String),

    #[error("checkpoint metadata is missing or invalid: {0}")]
    InvalidCheckpoint(String),
}

#[cfg(not(feature = "hdf5"))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PortableCheckpointDb {
    schema_version: u32,
    world_size: usize,
    rank_partitions: std::collections::BTreeMap<usize, std::collections::BTreeMap<String, Vec<f64>>>,
    steps: std::collections::BTreeMap<u64, PortableStep>,
}

#[cfg(not(feature = "hdf5"))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PortableStep {
    time: f64,
    mesh_meta: Option<CheckpointMeshMeta>,
    partitions: std::collections::BTreeMap<usize, std::collections::BTreeMap<String, PortableField>>, 
    global_fields: std::collections::BTreeMap<String, Vec<f64>>,
}

#[cfg(not(feature = "hdf5"))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PortableField {
    global_offset: u64,
    global_len: u64,
    values: Vec<f64>,
}

#[cfg(not(feature = "hdf5"))]
fn portable_load_db(file_path: &str) -> Result<PortableCheckpointDb, Hdf5ParallelError> {
    use std::path::Path;

    if !Path::new(file_path).exists() {
        return Ok(PortableCheckpointDb {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            ..PortableCheckpointDb::default()
        });
    }

    let bytes = std::fs::read(file_path)
        .map_err(|e| Hdf5ParallelError::Backend(format!("portable read failed: {e}")))?;
    let mut db: PortableCheckpointDb = rmp_serde::from_slice(&bytes)
        .map_err(|e| Hdf5ParallelError::Backend(format!("portable decode failed: {e}")))?;
    if db.schema_version == 0 {
        db.schema_version = CHECKPOINT_SCHEMA_VERSION;
    }
    Ok(db)
}

#[cfg(not(feature = "hdf5"))]
fn portable_save_db(file_path: &str, db: &PortableCheckpointDb) -> Result<(), Hdf5ParallelError> {
    if let Some(parent) = std::path::Path::new(file_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Hdf5ParallelError::Backend(format!("portable mkdir failed: {e}")))?;
        }
    }

    let bytes = rmp_serde::to_vec_named(db)
        .map_err(|e| Hdf5ParallelError::Backend(format!("portable encode failed: {e}")))?;
    std::fs::write(file_path, bytes)
        .map_err(|e| Hdf5ParallelError::Backend(format!("portable write failed: {e}")))
}

#[cfg(not(feature = "hdf5"))]
fn portable_ensure_world_size(
    db: &mut PortableCheckpointDb,
    world_size: usize,
) -> Result<(), Hdf5ParallelError> {
    if db.world_size == 0 {
        db.world_size = world_size;
        return Ok(());
    }
    if db.world_size != world_size {
        return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
            "world_size mismatch: file={}, requested={}",
            db.world_size, world_size
        )));
    }
    Ok(())
}

/// Backend-dispatching checkpoint writer.
///
/// This API lets callers switch between stable partitioned I/O and a future
/// MPI-collective path without changing call sites.
pub fn write_checkpoint_step_f64_with_backend(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    time: f64,
    fields: &[RankFieldF64],
    backend: IoBackend,
) -> Result<(), Hdf5ParallelError> {
    match backend {
        IoBackend::Partitioned => write_checkpoint_step_f64(file_path, cfg, step, time, fields),
        IoBackend::MpiCollective => write_checkpoint_step_f64_mpi_collective(
            file_path, cfg, step, time, fields,
        ),
    }
}

/// MPI-enabled checkpoint writer.
///
/// In `hdf5-mpi` builds this performs MPI-coordinated checkpoint staging:
/// each rank writes its partition, then rank 0 materializes global fields for
/// the step. If the live MPI communicator does not match `cfg`, the function
/// safely falls back to partitioned mode.
pub fn write_checkpoint_step_f64_mpi_collective(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    time: f64,
    fields: &[RankFieldF64],
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5-mpi")]
    {
        use mpi::topology::SimpleCommunicator;
        use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Source};

        let world = SimpleCommunicator::world();
        let comm_rank = world.rank() as usize;
        let comm_size = world.size() as usize;

        // If caller config does not match active communicator, degrade to
        // deterministic partitioned write rather than failing unexpectedly.
        if comm_size != cfg.world_size || comm_rank != cfg.rank {
            return write_checkpoint_step_f64(file_path, cfg, step, time, fields);
        }

        write_checkpoint_step_f64(file_path, cfg, step, time, fields)?;

        #[cfg(feature = "hdf5")]
        if hdf5_rust_impl::SUPPORTS_HYPERSLAB {
            let _ = write_checkpoint_step_f64_hyperslab(file_path, cfg, step, time, fields);
        }

        world.barrier();

        const TAG_FIELD_NAMES: i32 = 0x4A10;

        if comm_rank == 0 {
            let mut names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();

            for src in 1..comm_size as i32 {
                let (msg, _status) = world
                    .process_at_rank(src)
                    .receive_vec_with_tag::<u8>(TAG_FIELD_NAMES);
                if let Ok(txt) = String::from_utf8(msg) {
                    for n in txt.split('\n') {
                        if !n.is_empty() {
                            names.push(n.to_string());
                        }
                    }
                }
            }

            names.sort_unstable();
            names.dedup();

            for name in names {
                // Materialize if all partitions for this field exist.
                // Missing partitions are treated as non-fatal and skipped.
                if materialize_global_field_f64(file_path, cfg.world_size, step, &name).is_err() {
                    continue;
                }
            }
        } else {
            let payload = fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes();
            world
                .process_at_rank(0)
                .send_with_tag(payload.as_slice(), TAG_FIELD_NAMES);
        }

        world.barrier();
        Ok(())
    }

    #[cfg(not(feature = "hdf5-mpi"))]
    {
        write_checkpoint_step_f64(file_path, cfg, step, time, fields)
    }
}

/// Write rank-local chunks directly into global field datasets via HDF5 slices.
///
/// Output dataset path:
/// `/global_fields/step_XXXXXXXX/<field_name>`
///
/// This API is intended as the direct hyperslab write path for MPI-coordinated
/// checkpoints. It is also safe for single-process staged writes.
pub fn write_checkpoint_step_f64_hyperslab(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    time: f64,
    fields: &[RankFieldF64],
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::write_checkpoint_step_f64_hyperslab(
            file_path, cfg, step, time, fields,
        );
    }

    #[cfg(not(feature = "hdf5"))]
    {
        write_checkpoint_step_f64(file_path, cfg, step, time, fields)
    }
}

/// Write one checkpoint step using a bundle schema.
///
/// This extends the rank-partitioned field layout with optional mesh metadata
/// under `/steps/step_XXXXXXXX/mesh_meta/*`.
pub fn write_checkpoint_step_bundle_f64(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    time: f64,
    bundle: &CheckpointBundleF64,
    backend: IoBackend,
) -> Result<(), Hdf5ParallelError> {
    write_checkpoint_step_f64_with_backend(file_path, cfg, step, time, &bundle.fields, backend)?;

    #[cfg(feature = "hdf5")]
    {
        if let Some(meta) = bundle.mesh_meta {
            hdf5_rust_impl::write_checkpoint_step_bundle_f64_mesh_meta(
                file_path, cfg, step, time, meta,
            )?;
        }
    }

    #[cfg(not(feature = "hdf5"))]
    {
        if let Some(meta) = bundle.mesh_meta {
            let mut db = portable_load_db(file_path)?;
            portable_ensure_world_size(&mut db, cfg.world_size)?;
            let step_entry = db.steps.entry(step).or_insert_with(PortableStep::default);
            step_entry.time = time;
            step_entry.mesh_meta = Some(meta);
            portable_save_db(file_path, &db)?;
        }
    }

    Ok(())
}

/// Write a local rank partition as a per-rank dataset.
///
/// File layout:
/// - `/meta/world_size` (attribute)
/// - `/partitions/rank_XXXXXX/<dataset>`
pub fn write_rank_partition_f64(
    file_path: &str,
    dataset: &str,
    local_values: &[f64],
    cfg: ParallelIoConfig,
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::write_rank_partition_f64(file_path, dataset, local_values, cfg);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let mut db = portable_load_db(file_path)?;
        portable_ensure_world_size(&mut db, cfg.world_size)?;
        let rank_entry = db
            .rank_partitions
            .entry(cfg.rank)
            .or_insert_with(std::collections::BTreeMap::new);
        rank_entry.insert(dataset.to_string(), local_values.to_vec());
        portable_save_db(file_path, &db)
    }
}

/// Read a local rank partition from a per-rank dataset.
pub fn read_rank_partition_f64(
    file_path: &str,
    dataset: &str,
    cfg: ParallelIoConfig,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::read_rank_partition_f64(file_path, dataset, cfg);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        let vals = db
            .rank_partitions
            .get(&cfg.rank)
            .and_then(|m| m.get(dataset))
            .cloned()
            .ok_or_else(|| {
                Hdf5ParallelError::InvalidCheckpoint(format!(
                    "missing portable partition rank={} dataset={}",
                    cfg.rank, dataset
                ))
            })?;
        Ok(vals)
    }
}

/// Write one checkpoint step for all local rank fields.
///
/// Layout:
/// - `/meta/*` (schema/world size)
/// - `/steps/step_XXXXXXXX/` (step-specific metadata)
/// - `/steps/step_XXXXXXXX/partitions/rank_XXXXXX/<field_name>` datasets
pub fn write_checkpoint_step_f64(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    time: f64,
    fields: &[RankFieldF64],
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::write_checkpoint_step_f64(file_path, cfg, step, time, fields);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let mut db = portable_load_db(file_path)?;
        portable_ensure_world_size(&mut db, cfg.world_size)?;

        let step_entry = db.steps.entry(step).or_insert_with(PortableStep::default);
        step_entry.time = time;
        let rank_fields = step_entry
            .partitions
            .entry(cfg.rank)
            .or_insert_with(std::collections::BTreeMap::new);

        for f in fields {
            rank_fields.insert(
                f.name.clone(),
                PortableField {
                    global_offset: f.global_offset,
                    global_len: f.global_len,
                    values: f.values.clone(),
                },
            );
        }

        portable_save_db(file_path, &db)
    }
}

/// Read one rank-local field from a specified checkpoint step.
pub fn read_checkpoint_field_f64_at_step(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    field_name: &str,
) -> Result<RankFieldReadF64, Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::read_checkpoint_field_f64_at_step(file_path, cfg, step, field_name);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        let step_entry = db
            .steps
            .get(&step)
            .ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint(format!("missing step {step}")))?;
        let rank_fields = step_entry.partitions.get(&cfg.rank).ok_or_else(|| {
            Hdf5ParallelError::InvalidCheckpoint(format!(
                "missing partition rank={} at step {}",
                cfg.rank, step
            ))
        })?;
        let field = rank_fields.get(field_name).ok_or_else(|| {
            Hdf5ParallelError::InvalidCheckpoint(format!(
                "missing field '{}' at step {} rank {}",
                field_name, step, cfg.rank
            ))
        })?;

        Ok(RankFieldReadF64 {
            step,
            time: step_entry.time,
            global_offset: field.global_offset,
            global_len: field.global_len,
            values: field.values.clone(),
        })
    }
}

/// Read one rank-local field from the latest checkpoint step.
pub fn read_checkpoint_field_f64_latest(
    file_path: &str,
    cfg: ParallelIoConfig,
    field_name: &str,
) -> Result<RankFieldReadF64, Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::read_checkpoint_field_f64_latest(file_path, cfg, field_name);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        let step = db
            .steps
            .keys()
            .copied()
            .max()
            .ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint("no step entries found".into()))?;
        read_checkpoint_field_f64_at_step(file_path, cfg, step, field_name)
    }
}

/// Read multiple rank-local fields from the latest checkpoint step.
pub fn read_checkpoint_fields_f64_latest(
    file_path: &str,
    cfg: ParallelIoConfig,
    field_names: &[&str],
) -> Result<Vec<(String, RankFieldReadF64)>, Hdf5ParallelError> {
    let mut out = Vec::with_capacity(field_names.len());
    for &name in field_names {
        let r = read_checkpoint_field_f64_latest(file_path, cfg, name)?;
        out.push((name.to_string(), r));
    }
    Ok(out)
}

/// Validate checkpoint structure and rank field consistency.
///
/// Checks schema metadata, step layout, and that each field chunk is bounded
/// by its reported `global_len` with non-overlapping rank ranges.
pub fn validate_checkpoint_layout(
    file_path: &str,
    expected_world_size: Option<usize>,
) -> Result<CheckpointValidationReport, Hdf5ParallelError> {
    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::validate_checkpoint_layout(file_path, expected_world_size);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        if db.world_size == 0 {
            return Err(Hdf5ParallelError::InvalidCheckpoint(
                "portable checkpoint world_size is zero".to_string(),
            ));
        }
        if let Some(exp) = expected_world_size {
            if exp != db.world_size {
                return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                    "world_size mismatch: file={}, expected={}",
                    db.world_size, exp
                )));
            }
        }

        let mut steps: Vec<u64> = db.steps.keys().copied().collect();
        steps.sort_unstable();

        let mut step_infos = Vec::with_capacity(steps.len());
        let mut warnings = Vec::new();

        for step in steps {
            let step_entry = db.steps.get(&step).expect("step key exists");
            let mut field_ranges: std::collections::HashMap<String, (u64, Vec<(u64, u64)>)> =
                std::collections::HashMap::new();

            for rank in 0..db.world_size {
                let rank_fields = step_entry.partitions.get(&rank).ok_or_else(|| {
                    Hdf5ParallelError::InvalidCheckpoint(format!(
                        "missing partition rank={} at step {}",
                        rank, step
                    ))
                })?;

                for (fname, f) in rank_fields {
                    let end = f.global_offset.saturating_add(f.values.len() as u64);
                    if end > f.global_len {
                        return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                            "chunk out of bounds in step {} rank {} field {}: [{}, {}) > {}",
                            step, rank, fname, f.global_offset, end, f.global_len
                        )));
                    }
                    let entry = field_ranges
                        .entry(fname.clone())
                        .or_insert_with(|| (f.global_len, Vec::new()));
                    if entry.0 != f.global_len {
                        return Err(Hdf5ParallelError::InvalidCheckpoint(
                            "global_len mismatch across ranks".to_string(),
                        ));
                    }
                    entry.1.push((f.global_offset, end));
                }
            }

            for (fname, (gl, mut ranges)) in field_ranges {
                ranges.sort_unstable_by_key(|r| r.0);
                let mut covered = 0u64;
                let mut prev_end = 0u64;
                for (off, end) in ranges {
                    if off < prev_end {
                        return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                            "overlapping chunks in step {} field {}",
                            step, fname
                        )));
                    }
                    covered = covered.saturating_add(end.saturating_sub(off));
                    prev_end = end;
                }
                if covered != gl {
                    warnings.push(format!(
                        "incomplete coverage in step {} field {}: covered={}, global_len={}",
                        step, fname, covered, gl
                    ));
                }
            }

            step_infos.push(CheckpointStepInfo {
                step,
                time: step_entry.time,
                partition_count: step_entry.partitions.len(),
            });
        }

        Ok(CheckpointValidationReport {
            schema_version: db.schema_version,
            world_size: db.world_size,
            steps: step_infos,
            warnings,
        })
    }
}

/// Materialize a full global field dataset from per-rank partitions.
///
/// Output dataset path:
/// `/global_fields/step_XXXXXXXX/<field_name>`
///
/// Returns the global field length.
pub fn materialize_global_field_f64(
    file_path: &str,
    world_size: usize,
    step: u64,
    field_name: &str,
) -> Result<u64, Hdf5ParallelError> {
    if world_size == 0 {
        return Err(Hdf5ParallelError::InvalidConfig("world_size must be > 0"));
    }

    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::materialize_global_field_f64(file_path, world_size, step, field_name);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let mut db = portable_load_db(file_path)?;
        portable_ensure_world_size(&mut db, world_size)?;
        let step_entry = db
            .steps
            .get_mut(&step)
            .ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint(format!("missing step {step}")))?;

        let mut global_len: Option<u64> = None;
        let mut chunks: Vec<(u64, Vec<f64>)> = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let rank_fields = step_entry.partitions.get(&rank).ok_or_else(|| {
                Hdf5ParallelError::InvalidCheckpoint(format!(
                    "missing partition rank={} at step {}",
                    rank, step
                ))
            })?;
            let f = rank_fields.get(field_name).ok_or_else(|| {
                Hdf5ParallelError::InvalidCheckpoint(format!(
                    "missing field '{}' at step {} rank {}",
                    field_name, step, rank
                ))
            })?;
            if let Some(prev) = global_len {
                if prev != f.global_len {
                    return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                        "global_len mismatch across ranks: {} vs {}",
                        prev, f.global_len
                    )));
                }
            } else {
                global_len = Some(f.global_len);
            }
            chunks.push((f.global_offset, f.values.clone()));
        }

        let gl = global_len.ok_or_else(|| {
            Hdf5ParallelError::InvalidCheckpoint("no rank chunks found for global materialization".into())
        })?;
        let mut global = vec![0.0f64; gl as usize];
        for (off, vals) in chunks {
            let start = off as usize;
            let end = start + vals.len();
            if end > global.len() {
                return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                    "partition out of bounds: [{}, {}) vs global {}",
                    start,
                    end,
                    global.len()
                )));
            }
            global[start..end].copy_from_slice(&vals);
        }

        step_entry
            .global_fields
            .insert(field_name.to_string(), global);
        portable_save_db(file_path, &db)?;
        Ok(gl)
    }
}

/// Read a full global field dataset produced under `/global_fields/step_XXXXXXXX/<field_name>`.
pub fn read_global_field_f64(
    file_path: &str,
    step: u64,
    field_name: &str,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::read_global_field_f64(file_path, step, field_name);
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        let step_entry = db
            .steps
            .get(&step)
            .ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint(format!("missing step {step}")))?;
        step_entry
            .global_fields
            .get(field_name)
            .cloned()
            .ok_or_else(|| {
                Hdf5ParallelError::InvalidCheckpoint(format!(
                    "missing global field '{}' at step {}",
                    field_name, step
                ))
            })
    }
}

/// Read a slice from a global field dataset produced under `/global_fields/step_XXXXXXXX/<field_name>`.
pub fn read_global_field_slice_f64(
    file_path: &str,
    step: u64,
    field_name: &str,
    global_offset: u64,
    local_len: usize,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    #[cfg(feature = "hdf5")]
    {
        return hdf5_rust_impl::read_global_field_slice_f64(
            file_path,
            step,
            field_name,
            global_offset,
            local_len,
        );
    }

    #[cfg(not(feature = "hdf5"))]
    {
        let values = read_global_field_f64(file_path, step, field_name)?;
        let start = global_offset as usize;
        let end = start.saturating_add(local_len);
        if end > values.len() {
            return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                "requested global slice [{}, {}) exceeds dataset length {}",
                start,
                end,
                values.len()
            )));
        }
        Ok(values[start..end].to_vec())
    }
}

/// Read each named field for the given rank using hyperslab (slice) access.
///
/// This is the read-side counterpart to [`write_checkpoint_step_f64_hyperslab`].
/// Each rank specifies its `global_offset` and `local_len`; the function reads
/// only that rank's contiguous slice from the global field dataset stored at
///
/// `/global_fields/step_XXXXXXXX/<field_name>`
///
/// When the `hdf5` feature is enabled, this uses native HDF5 hyperslab I/O.
/// Otherwise it falls back to reading the full global field and slicing in memory
/// (functionally identical, slightly higher I/O overhead).
///
/// # Arguments
/// * `file_path`  — Path to the checkpoint file.
/// * `cfg`        — Parallel configuration (rank, world_size).
/// * `step`       — Checkpoint step index.
/// * `requests`   — Slice of `(field_name, global_offset, local_len)` tuples.
///
/// # Returns
/// A `Vec<(String, Vec<f64>)>` with the local data for each requested field,
/// in the same order as `requests`.
pub fn read_checkpoint_step_f64_hyperslab(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    requests: &[(&str, u64, usize)],
) -> Result<Vec<(String, Vec<f64>)>, Hdf5ParallelError> {
    cfg.validate()?;

    let mut out = Vec::with_capacity(requests.len());
    for &(field_name, global_offset, local_len) in requests {
        let values = read_global_field_slice_f64(file_path, step, field_name, global_offset, local_len)?;
        out.push((field_name.to_string(), values));
    }
    Ok(out)
}

/// Read a full checkpoint bundle (all named fields + optional mesh meta) at a given step.
///
/// This is the read-side counterpart to [`write_checkpoint_step_bundle_f64`].
/// Each element of `field_names` is read for this rank and returned as a
/// `(name, RankFieldReadF64)` pair.
pub fn read_checkpoint_bundle_f64(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    field_names: &[&str],
) -> Result<CheckpointBundleReadF64, Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        let step_entry = db
            .steps
            .get(&step)
            .ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint(format!("missing step {step}")))?;
        let time = step_entry.time;
        let mesh_meta = step_entry.mesh_meta;
        let rank_fields = step_entry.partitions.get(&cfg.rank).ok_or_else(|| {
            Hdf5ParallelError::InvalidCheckpoint(format!(
                "missing partition rank={} at step {}",
                cfg.rank, step
            ))
        })?;
        let mut fields = Vec::with_capacity(field_names.len());
        for &name in field_names {
            let f = rank_fields.get(name).ok_or_else(|| {
                Hdf5ParallelError::InvalidCheckpoint(format!(
                    "missing field '{name}' at step {step} rank {}",
                    cfg.rank
                ))
            })?;
            fields.push((
                name.to_string(),
                RankFieldReadF64 {
                    step,
                    time,
                    global_offset: f.global_offset,
                    global_len: f.global_len,
                    values: f.values.clone(),
                },
            ));
        }
        return Ok(CheckpointBundleReadF64 { step, time, mesh_meta, fields });
    }

    #[cfg(feature = "hdf5")]
    {
        // Under the HDF5 feature path, delegate to the existing per-field reader.
        let mut fields = Vec::with_capacity(field_names.len());
        let mut time = 0.0;
        let mut mesh_meta = None;
        for &name in field_names {
            let r = hdf5_rust_impl::read_checkpoint_field_f64_at_step(file_path, cfg, step, name)?;
            time = r.time;
            fields.push((name.to_string(), r));
        }
        // Mesh meta requires a dedicated read; attempt it if possible.
        // (best-effort; errors are silently swallowed.)
        if let Ok(db) = portable_load_db(file_path) {
            if let Some(se) = db.steps.get(&step) {
                mesh_meta = se.mesh_meta;
                time = se.time;
            }
        }
        return Ok(CheckpointBundleReadF64 { step, time, mesh_meta, fields });
    }
}

/// Read the latest checkpoint bundle for the specified field names.
pub fn read_checkpoint_bundle_f64_latest(
    file_path: &str,
    cfg: ParallelIoConfig,
    field_names: &[&str],
) -> Result<CheckpointBundleReadF64, Hdf5ParallelError> {
    cfg.validate()?;

    #[cfg(not(feature = "hdf5"))]
    {
        let db = portable_load_db(file_path)?;
        let step = db
            .steps
            .keys()
            .copied()
            .max()
            .ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint("checkpoint has no steps".into()))?;
        return read_checkpoint_bundle_f64(file_path, cfg, step, field_names);
    }

    #[cfg(feature = "hdf5")]
    {
        let mut fields = Vec::with_capacity(field_names.len());
        let mut time = 0.0;
        let mut step = 0u64;
        for &name in field_names {
            let r = hdf5_rust_impl::read_checkpoint_field_f64_latest(file_path, cfg, name)?;
            step = r.step;
            time = r.time;
            fields.push((name.to_string(), r));
        }
        return Ok(CheckpointBundleReadF64 { step, time, mesh_meta: None, fields });
    }
}

/// Write a minimal XDMF sidecar describing one scalar field at one time step.
///
/// This helper intentionally emits a compact Polyvertex grid to make the
/// checkpoint field quickly inspectable in XDMF-capable tools.
/// Call [`materialize_global_field_f64`] first so the referenced dataset exists.
pub fn write_xdmf_polyvertex_scalar_sidecar(
    xdmf_path: &str,
    hdf5_path: &str,
    field_name: &str,
    global_len: u64,
    step: u64,
    time: f64,
) -> Result<(), Hdf5ParallelError> {
    use std::fmt::Write as _;
    let mut xml = String::new();
    let _ = writeln!(&mut xml, "<?xml version=\"1.0\" ?>");
    let _ = writeln!(&mut xml, "<Xdmf Version=\"3.0\">");
    let _ = writeln!(&mut xml, "  <Domain>");
    let _ = writeln!(&mut xml, "    <Grid Name=\"checkpoint_step_{:08}\" GridType=\"Uniform\">", step);
    let _ = writeln!(&mut xml, "      <Time Value=\"{}\" />", time);
    let _ = writeln!(&mut xml, "      <Topology TopologyType=\"Polyvertex\" NumberOfElements=\"{}\"/>", global_len);
    let _ = writeln!(&mut xml, "      <Geometry GeometryType=\"XYZ\">");
    let _ = writeln!(&mut xml, "        <DataItem Dimensions=\"{} 3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">", global_len);
    let _ = writeln!(&mut xml, "          0 0 0");
    let _ = writeln!(&mut xml, "        </DataItem>");
    let _ = writeln!(&mut xml, "      </Geometry>");
    let _ = writeln!(&mut xml, "      <Attribute Name=\"{}\" AttributeType=\"Scalar\" Center=\"Node\">", field_name);
    let _ = writeln!(&mut xml, "        <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">", global_len);
    let _ = writeln!(&mut xml, "          {}:/global_fields/step_{:08}/{}", hdf5_path.replace('\\', "/"), step, field_name);
    let _ = writeln!(&mut xml, "        </DataItem>");
    let _ = writeln!(&mut xml, "      </Attribute>");
    let _ = writeln!(&mut xml, "    </Grid>");
    let _ = writeln!(&mut xml, "  </Domain>");
    let _ = writeln!(&mut xml, "</Xdmf>");

    std::fs::write(xdmf_path, xml).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))
}

/// Write flat mesh node coordinates into the checkpoint as a global field.
///
/// The coordinates are stored under the special dataset key `__mesh_coords__`
/// so they survive alongside field data and can be retrieved without any
/// per-rank information at restart.
///
/// `dim` is the spatial dimension (typically 3); `coords` is the flat
/// row-major array of length `n_nodes * dim`.
pub fn write_checkpoint_mesh_coords(
    file_path: &str,
    step: u64,
    time: f64,
    _dim: usize,
    coords: &[f64],
) -> Result<(), Hdf5ParallelError> {
    if coords.is_empty() {
        return Ok(());
    }
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let n = coords.len() as u64;
    write_checkpoint_step_f64(
        file_path,
        cfg,
        step,
        time,
        &[RankFieldF64 {
            name: "__mesh_coords__".into(),
            global_offset: 0,
            global_len: n,
            values: coords.to_vec(),
        }],
    )?;
    materialize_global_field_f64(file_path, 1, step, "__mesh_coords__")?;
    Ok(())
}

/// Read mesh node coordinates previously saved with [`write_checkpoint_mesh_coords`].
///
/// Returns the flat row-major coordinate array `[x0,y0,z0, x1,y1,z1, …]`.
pub fn read_checkpoint_mesh_coords(
    file_path: &str,
    step: u64,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    read_global_field_f64(file_path, step, "__mesh_coords__")
}

/// Read mesh node coordinates from the latest checkpoint step.
///
/// Returns `(step, flat_coords)`.
pub fn read_checkpoint_mesh_coords_latest(
    file_path: &str,
) -> Result<(u64, Vec<f64>), Hdf5ParallelError> {
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let r = read_checkpoint_field_f64_latest(file_path, cfg, "__mesh_coords__")?;
    Ok((r.step, r.values))
}

/// Write a temporal XDMF sidecar for one scalar field across multiple steps.
///
/// Call [`materialize_global_field_f64`] for each listed step before writing.
pub fn write_xdmf_polyvertex_scalar_timeseries_sidecar(
    xdmf_path: &str,
    hdf5_path: &str,
    field_name: &str,
    global_len: u64,
    steps: &[(u64, f64)],
) -> Result<(), Hdf5ParallelError> {
    use std::fmt::Write as _;
    let mut xml = String::new();
    let _ = writeln!(&mut xml, "<?xml version=\"1.0\" ?>");
    let _ = writeln!(&mut xml, "<Xdmf Version=\"3.0\">");
    let _ = writeln!(&mut xml, "  <Domain>");
    let _ = writeln!(
        &mut xml,
        "    <Grid Name=\"checkpoint_series\" GridType=\"Collection\" CollectionType=\"Temporal\">"
    );

    for (step, time) in steps {
        let _ = writeln!(
            &mut xml,
            "      <Grid Name=\"checkpoint_step_{:08}\" GridType=\"Uniform\">",
            step
        );
        let _ = writeln!(&mut xml, "        <Time Value=\"{}\" />", time);
        let _ = writeln!(
            &mut xml,
            "        <Topology TopologyType=\"Polyvertex\" NumberOfElements=\"{}\"/>",
            global_len
        );
        let _ = writeln!(&mut xml, "        <Geometry GeometryType=\"XYZ\">");
        let _ = writeln!(
            &mut xml,
            "          <DataItem Dimensions=\"{} 3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">",
            global_len
        );
        let _ = writeln!(&mut xml, "            0 0 0");
        let _ = writeln!(&mut xml, "          </DataItem>");
        let _ = writeln!(&mut xml, "        </Geometry>");
        let _ = writeln!(
            &mut xml,
            "        <Attribute Name=\"{}\" AttributeType=\"Scalar\" Center=\"Node\">",
            field_name
        );
        let _ = writeln!(
            &mut xml,
            "          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">",
            global_len
        );
        let _ = writeln!(
            &mut xml,
            "            {}:/global_fields/step_{:08}/{}",
            hdf5_path.replace('\\', "/"),
            step,
            field_name
        );
        let _ = writeln!(&mut xml, "          </DataItem>");
        let _ = writeln!(&mut xml, "        </Attribute>");
        let _ = writeln!(&mut xml, "      </Grid>");
    }

    let _ = writeln!(&mut xml, "    </Grid>");
    let _ = writeln!(&mut xml, "  </Domain>");
    let _ = writeln!(&mut xml, "</Xdmf>");

    std::fs::write(xdmf_path, xml).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))
}

#[cfg(all(test, feature = "hdf5"))]
mod restart_tests {
    use super::*;

    fn make_rank_field_u(global: &[f64], rank: usize, world_size: usize) -> RankFieldF64 {
        let local_len = global.len() / world_size;
        let start = rank * local_len;
        let end = start + local_len;
        RankFieldF64 {
            name: "u".to_string(),
            global_offset: start as u64,
            global_len: global.len() as u64,
            values: global[start..end].to_vec(),
        }
    }

    fn advance_state(global: &mut [f64], step: u64) {
        for (i, v) in global.iter_mut().enumerate() {
            *v += 0.5 * (step as f64 + 1.0) + (i as f64) * 0.01;
        }
    }

    #[test]
    fn checkpoint_restart_continuation_matches_uninterrupted_baseline() {
        let world_size = 2usize;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "fem_io_hdf5_parallel_restart_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        ));
        let file_path = path.to_string_lossy().to_string();

        let mut interrupted = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        for step in 0..=2u64 {
            advance_state(&mut interrupted, step);
            for rank in 0..world_size {
                let cfg = ParallelIoConfig { world_size, rank };
                let field = make_rank_field_u(&interrupted, rank, world_size);
                write_checkpoint_step_f64(&file_path, cfg, step, step as f64 * 0.1, &[field])
                    .expect("write initial checkpoint step");
            }
        }

        let mut resumed = Vec::new();
        for rank in 0..world_size {
            let cfg = ParallelIoConfig { world_size, rank };
            let read = read_checkpoint_field_f64_latest(&file_path, cfg, "u")
                .expect("read latest restart field");
            resumed.extend_from_slice(&read.values);
            assert_eq!(read.step, 2);
        }

        for step in 3..=4u64 {
            advance_state(&mut resumed, step);
            for rank in 0..world_size {
                let cfg = ParallelIoConfig { world_size, rank };
                let field = make_rank_field_u(&resumed, rank, world_size);
                write_checkpoint_step_f64(&file_path, cfg, step, step as f64 * 0.1, &[field])
                    .expect("write resumed checkpoint step");
            }
        }

        let mut baseline = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for step in 0..=4u64 {
            advance_state(&mut baseline, step);
        }

        let mut latest = Vec::new();
        for rank in 0..world_size {
            let cfg = ParallelIoConfig { world_size, rank };
            let read = read_checkpoint_field_f64_latest(&file_path, cfg, "u")
                .expect("read latest final field");
            latest.extend_from_slice(&read.values);
            assert_eq!(read.step, 4);
            assert!((read.time - 0.4).abs() < 1.0e-12);
        }

        assert_eq!(latest.len(), baseline.len());
        for (a, b) in latest.iter().zip(baseline.iter()) {
            assert!((a - b).abs() < 1.0e-12, "latest={a}, baseline={b}");
        }

        let report = validate_checkpoint_layout(&file_path, Some(world_size))
            .expect("validate checkpoint layout");
        assert_eq!(report.schema_version, hdf5_rust_impl::SCHEMA_VERSION_RUST);
        assert_eq!(report.steps.len(), 5);

        let _ = std::fs::remove_file(&file_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_cfg_validation() {
        assert!(ParallelIoConfig { world_size: 4, rank: 2 }.validate().is_ok());
        assert!(ParallelIoConfig { world_size: 0, rank: 0 }.validate().is_err());
        assert!(ParallelIoConfig { world_size: 2, rank: 2 }.validate().is_err());
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn parse_step_name_ok() {
        assert_eq!(super::parse_step_name("step_00000042"), Some(42));
        assert_eq!(super::parse_step_name("other"), None);
    }

    #[test]
    fn portable_backend_roundtrip_without_hdf5_feature() {
        let world_size = 2usize;
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_portable_checkpoint_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        ));
        let file_path = p.to_string_lossy().to_string();

        for rank in 0..world_size {
            write_checkpoint_step_f64(
                &file_path,
                ParallelIoConfig { world_size, rank },
                0,
                0.0,
                &[RankFieldF64 {
                    name: "u".into(),
                    global_offset: (rank * 2) as u64,
                    global_len: 4,
                    values: vec![rank as f64 + 1.0, rank as f64 + 2.0],
                }],
            )
            .expect("portable step write should succeed");
        }

        let gl = materialize_global_field_f64(&file_path, world_size, 0, "u")
            .expect("portable global materialization should succeed");
        assert_eq!(gl, 4);

        let full = read_global_field_f64(&file_path, 0, "u")
            .expect("portable global read should succeed");
        assert_eq!(full, vec![1.0, 2.0, 2.0, 3.0]);

        let s = read_global_field_slice_f64(&file_path, 0, "u", 1, 2)
            .expect("portable global slice should succeed");
        assert_eq!(s, vec![2.0, 2.0]);

        let r = read_checkpoint_field_f64_latest(
            &file_path,
            ParallelIoConfig { world_size, rank: 1 },
            "u",
        )
        .expect("portable latest rank read should succeed");
        assert_eq!(r.step, 0);
        assert_eq!(r.global_offset, 2);
        assert_eq!(r.values, vec![2.0, 3.0]);

        let report = validate_checkpoint_layout(&file_path, Some(world_size))
            .expect("portable layout validation should succeed");
        assert_eq!(report.steps.len(), 1);
        assert_eq!(report.world_size, world_size);

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn mpi_backend_falls_back_to_portable_when_mpi_feature_is_absent() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_portable_checkpoint_mpi_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        ));
        let file_path = p.to_string_lossy().to_string();

        write_checkpoint_step_f64_with_backend(
            &file_path,
            ParallelIoConfig { world_size: 1, rank: 0 },
            0,
            0.0,
            &[RankFieldF64 {
                name: "u".into(),
                global_offset: 0,
                global_len: 3,
                values: vec![1.0, 2.0, 3.0],
            }],
            IoBackend::MpiCollective,
        )
        .expect("mpi backend should fall back to portable partitioned mode");

        let _ = materialize_global_field_f64(&file_path, 1, 0, "u")
            .expect("portable materialization should succeed");
        let vals = read_global_field_f64(&file_path, 0, "u")
            .expect("portable global read should succeed");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn xdmf_sidecar_writes_file() {
        let mut p = std::env::temp_dir();
        p.push(format!("fem_hdf5_xdmf_test_{}.xdmf", std::process::id()));
        let path = p.to_string_lossy().to_string();
        write_xdmf_polyvertex_scalar_sidecar(&path, "checkpoint.h5", "u", 8, 3, 0.125)
            .expect("xdmf write failed");
        let txt = std::fs::read_to_string(&path).expect("xdmf read failed");
        assert!(txt.contains("checkpoint.h5:/global_fields/step_00000003/u"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn xdmf_timeseries_sidecar_writes_file() {
        let mut p = std::env::temp_dir();
        p.push(format!("fem_hdf5_xdmf_ts_test_{}.xdmf", std::process::id()));
        let path = p.to_string_lossy().to_string();
        write_xdmf_polyvertex_scalar_timeseries_sidecar(
            &path,
            "checkpoint.h5",
            "u",
            8,
            &[(0, 0.0), (1, 0.1), (2, 0.2)],
        )
        .expect("xdmf time-series write failed");
        let txt = std::fs::read_to_string(&path).expect("xdmf time-series read failed");
        assert!(txt.contains("CollectionType=\"Temporal\""));
        assert!(txt.contains("checkpoint.h5:/global_fields/step_00000002/u"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn portable_hyperslab_aliases_partitioned_writer() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_portable_checkpoint_hyperslab_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        ));
        let file_path = p.to_string_lossy().to_string();

        write_checkpoint_step_f64_hyperslab(
            &file_path,
            ParallelIoConfig { world_size: 1, rank: 0 },
            0,
            0.0,
            &[RankFieldF64 {
                name: "u".into(),
                global_offset: 0,
                global_len: 3,
                values: vec![10.0, 20.0, 30.0],
            }],
        )
        .expect("portable hyperslab writer should fall back to step writer");

        let _ = materialize_global_field_f64(&file_path, 1, 0, "u")
            .expect("portable materialization should succeed");
        let vals = read_global_field_f64(&file_path, 0, "u")
            .expect("portable global read should succeed");
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);

        let _ = std::fs::remove_file(file_path);
    }

    // ─── read_checkpoint_step_f64_hyperslab tests ──────────────────────────

    #[test]
    fn hyperslab_read_single_rank_full_field() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_hsr_full_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = p.to_string_lossy().to_string();
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        write_checkpoint_step_f64(&path, cfg, 0, 0.0, &[RankFieldF64 {
            name: "pressure".into(),
            global_offset: 0,
            global_len: 4,
            values: vec![1.0, 2.0, 3.0, 4.0],
        }]).unwrap();
        let _ = materialize_global_field_f64(&path, 1, 0, "pressure").unwrap();

        let result = read_checkpoint_step_f64_hyperslab(
            &path, cfg, 0, &[("pressure", 0, 4)],
        ).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "pressure");
        assert_eq!(result[0].1, vec![1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hyperslab_read_rank_local_slice() {
        // Simulate 2-rank setup: rank 0 owns [0..3], rank 1 owns [3..5].
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_hsr_slice_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = p.to_string_lossy().to_string();

        let cfg0 = ParallelIoConfig { world_size: 2, rank: 0 };
        let cfg1 = ParallelIoConfig { world_size: 2, rank: 1 };

        write_checkpoint_step_f64(&path, cfg0, 1, 0.5, &[RankFieldF64 {
            name: "T".into(), global_offset: 0, global_len: 5, values: vec![10.0, 11.0, 12.0],
        }]).unwrap();
        write_checkpoint_step_f64(&path, cfg1, 1, 0.5, &[RankFieldF64 {
            name: "T".into(), global_offset: 3, global_len: 5, values: vec![13.0, 14.0],
        }]).unwrap();
        let _ = materialize_global_field_f64(&path, 2, 1, "T").unwrap();

        // Rank 0 reads its slice.
        let r0 = read_checkpoint_step_f64_hyperslab(&path, cfg0, 1, &[("T", 0, 3)]).unwrap();
        assert_eq!(r0[0].1, vec![10.0, 11.0, 12.0]);

        // Rank 1 reads its slice.
        let r1 = read_checkpoint_step_f64_hyperslab(&path, cfg1, 1, &[("T", 3, 2)]).unwrap();
        assert_eq!(r1[0].1, vec![13.0, 14.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hyperslab_read_multi_field() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_hsr_multi_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = p.to_string_lossy().to_string();
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        write_checkpoint_step_f64(&path, cfg, 2, 1.0, &[
            RankFieldF64 { name: "u".into(), global_offset: 0, global_len: 3, values: vec![1.0, 2.0, 3.0] },
            RankFieldF64 { name: "v".into(), global_offset: 0, global_len: 2, values: vec![5.0, 6.0] },
        ]).unwrap();
        let _ = materialize_global_field_f64(&path, 1, 2, "u").unwrap();
        let _ = materialize_global_field_f64(&path, 1, 2, "v").unwrap();

        let result = read_checkpoint_step_f64_hyperslab(
            &path, cfg, 2, &[("u", 0, 3), ("v", 0, 2)],
        ).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "u"); assert_eq!(result[0].1, vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1].0, "v"); assert_eq!(result[1].1, vec![5.0, 6.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hyperslab_read_out_of_bounds_returns_error() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_hsr_oob_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = p.to_string_lossy().to_string();
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        write_checkpoint_step_f64(&path, cfg, 0, 0.0, &[RankFieldF64 {
            name: "x".into(), global_offset: 0, global_len: 3, values: vec![1.0, 2.0, 3.0],
        }]).unwrap();
        let _ = materialize_global_field_f64(&path, 1, 0, "x").unwrap();

        // Request slice beyond dataset length.
        let res = read_checkpoint_step_f64_hyperslab(&path, cfg, 0, &[("x", 2, 5)]);
        assert!(res.is_err(), "out-of-bounds slice should return an error");

        let _ = std::fs::remove_file(&path);
    }

    // ─── read_checkpoint_bundle_f64 tests ────────────────────────────────────

    fn make_bundle_path(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_bundle_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn bundle_write_then_read_single_field() {
        let path = make_bundle_path("single");
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        let bundle = CheckpointBundleF64 {
            mesh_meta: Some(CheckpointMeshMeta { dim: 3, n_vertices: 8, n_elements: 2 }),
            fields: vec![RankFieldF64 {
                name: "pressure".into(),
                global_offset: 0,
                global_len: 4,
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
        };
        write_checkpoint_step_bundle_f64(&path, cfg, 5, 0.5, &bundle, IoBackend::Partitioned)
            .unwrap();

        let read = read_checkpoint_bundle_f64(&path, cfg, 5, &["pressure"]).unwrap();
        assert_eq!(read.step, 5);
        assert!((read.time - 0.5).abs() < 1e-12);
        let (name, field) = &read.fields[0];
        assert_eq!(name, "pressure");
        assert_eq!(field.values, vec![1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bundle_read_multi_field() {
        let path = make_bundle_path("multi");
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        let bundle = CheckpointBundleF64 {
            mesh_meta: None,
            fields: vec![
                RankFieldF64 { name: "u".into(), global_offset: 0, global_len: 3, values: vec![1.0, 2.0, 3.0] },
                RankFieldF64 { name: "v".into(), global_offset: 0, global_len: 3, values: vec![4.0, 5.0, 6.0] },
            ],
        };
        write_checkpoint_step_bundle_f64(&path, cfg, 0, 1.0, &bundle, IoBackend::Partitioned)
            .unwrap();

        let read = read_checkpoint_bundle_f64(&path, cfg, 0, &["u", "v"]).unwrap();
        assert_eq!(read.fields.len(), 2);
        let u_vals = &read.fields[0].1.values;
        let v_vals = &read.fields[1].1.values;
        assert_eq!(u_vals, &vec![1.0, 2.0, 3.0]);
        assert_eq!(v_vals, &vec![4.0, 5.0, 6.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bundle_read_latest_step() {
        let path = make_bundle_path("latest");
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        let field = |v: f64| CheckpointBundleF64 {
            mesh_meta: None,
            fields: vec![RankFieldF64 { name: "q".into(), global_offset: 0, global_len: 1, values: vec![v] }],
        };
        write_checkpoint_step_bundle_f64(&path, cfg, 1, 0.1, &field(10.0), IoBackend::Partitioned).unwrap();
        write_checkpoint_step_bundle_f64(&path, cfg, 3, 0.3, &field(30.0), IoBackend::Partitioned).unwrap();

        let read = read_checkpoint_bundle_f64_latest(&path, cfg, &["q"]).unwrap();
        assert_eq!(read.step, 3);
        assert_eq!(read.fields[0].1.values, vec![30.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bundle_read_missing_field_returns_error() {
        let path = make_bundle_path("err");
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        let bundle = CheckpointBundleF64 {
            mesh_meta: None,
            fields: vec![RankFieldF64 { name: "u".into(), global_offset: 0, global_len: 2, values: vec![1.0, 2.0] }],
        };
        write_checkpoint_step_bundle_f64(&path, cfg, 0, 0.0, &bundle, IoBackend::Partitioned).unwrap();

        let res = read_checkpoint_bundle_f64(&path, cfg, 0, &["nonexistent"]);
        assert!(res.is_err(), "reading a missing field should return an error");

        let _ = std::fs::remove_file(&path);
    }

    // ─── mesh coords checkpoint tests ────────────────────────────────────────

    fn make_coords_path(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_mesh_coords_{}_{}_{}.chk",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn write_then_read_mesh_coords_single_step() {
        let path = make_coords_path("single");
        let coords = vec![0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0];
        write_checkpoint_mesh_coords(&path, 0, 0.0, 3, &coords)
            .expect("write mesh coords should succeed");
        let out = read_checkpoint_mesh_coords(&path, 0)
            .expect("read mesh coords should succeed");
        assert_eq!(out.len(), 9, "9 floats for 3 nodes × 3D");
        for (a, b) in coords.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mesh_coords_latest_returns_last_step() {
        let path = make_coords_path("latest");
        let coords0 = vec![0.0, 0.0, 0.0,  1.0, 0.0, 0.0];
        let coords1 = vec![0.0, 0.0, 0.5,  1.0, 0.0, 0.5];
        write_checkpoint_mesh_coords(&path, 1, 0.1, 3, &coords0).unwrap();
        write_checkpoint_mesh_coords(&path, 3, 0.3, 3, &coords1).unwrap();
        let (step, out) = read_checkpoint_mesh_coords_latest(&path)
            .expect("read latest mesh coords should succeed");
        assert_eq!(step, 3);
        for (a, b) in coords1.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mesh_coords_coexist_with_field_data() {
        let path = make_coords_path("coexist");
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
        // Write field data and mesh coords at the same step.
        write_checkpoint_step_f64(&path, cfg, 2, 0.2, &[RankFieldF64 {
            name: "u".into(),
            global_offset: 0,
            global_len: 3,
            values: vec![1.0, 2.0, 3.0],
        }]).unwrap();
        let coords = vec![0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.5, 1.0, 0.0];
        write_checkpoint_mesh_coords(&path, 2, 0.2, 3, &coords).unwrap();

        // Both should be readable.
        let field = read_checkpoint_field_f64_at_step(&path, cfg, 2, "u").unwrap();
        assert_eq!(field.values, vec![1.0, 2.0, 3.0]);

        let out = read_checkpoint_mesh_coords(&path, 2).unwrap();
        assert_eq!(out.len(), 9);
        for (a, b) in coords.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mesh_coords_missing_step_returns_error() {
        let path = make_coords_path("missing");
        let coords = vec![0.0, 0.0, 0.0];
        write_checkpoint_mesh_coords(&path, 0, 0.0, 3, &coords).unwrap();
        let res = read_checkpoint_mesh_coords(&path, 99);
        assert!(res.is_err(), "reading coords at non-existent step should fail");
        let _ = std::fs::remove_file(&path);
    }

    // ─── multi-step hyperslab restart integration tests ─────────────────────

    fn make_restart_path(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fem_restart_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        p.to_string_lossy().to_string()
    }

    /// Full checkpoint-restart cycle: write N steps across 2 simulated ranks,
    /// then read back each step via hyperslab and verify data integrity.
    #[test]
    fn hyperslab_multi_step_restart_roundtrip() {
        let path = make_restart_path("multistep");

        // Simulated 2-rank world; rank 0 owns nodes [0..4], rank 1 owns [4..8].
        let cfg0 = ParallelIoConfig { world_size: 2, rank: 0 };
        let cfg1 = ParallelIoConfig { world_size: 2, rank: 1 };

        let steps: u64 = 3;
        for step in 0..steps {
            let t = step as f64 * 0.1;
            // Temperature field: rank 0 values = step*10 + [1..4], rank 1 = step*10 + [5..8]
            let r0_vals: Vec<f64> = (1u64..=4).map(|i| (step * 10 + i) as f64).collect();
            let r1_vals: Vec<f64> = (5u64..=8).map(|i| (step * 10 + i) as f64).collect();

            write_checkpoint_step_f64(&path, cfg0, step, t, &[RankFieldF64 {
                name: "T".into(), global_offset: 0, global_len: 8, values: r0_vals.clone(),
            }]).unwrap();
            write_checkpoint_step_f64(&path, cfg1, step, t, &[RankFieldF64 {
                name: "T".into(), global_offset: 4, global_len: 8, values: r1_vals.clone(),
            }]).unwrap();
            materialize_global_field_f64(&path, 2, step, "T").unwrap();
        }

        // Restart from step 1: verify both ranks can re-read their local data.
        let restart_step = 1u64;
        let r0_back = read_checkpoint_step_f64_hyperslab(
            &path, cfg0, restart_step, &[("T", 0, 4)],
        ).unwrap();
        let r1_back = read_checkpoint_step_f64_hyperslab(
            &path, cfg1, restart_step, &[("T", 4, 4)],
        ).unwrap();

        // Step 1: expected values = 1*10 + [1..4] for rank 0, 1*10 + [5..8] for rank 1.
        let expected_r0: Vec<f64> = (1u64..=4).map(|i| (10 + i) as f64).collect();
        let expected_r1: Vec<f64> = (5u64..=8).map(|i| (10 + i) as f64).collect();
        assert_eq!(r0_back[0].1, expected_r0, "rank 0 restart data mismatch");
        assert_eq!(r1_back[0].1, expected_r1, "rank 1 restart data mismatch");

        let _ = std::fs::remove_file(&path);
    }

    /// Verify that mesh coords checkpoint is preserved across restart steps.
    #[test]
    fn hyperslab_restart_with_mesh_coords_consistent() {
        let path = make_restart_path("meshcoords");
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };

        // Write 3 steps with evolving mesh coords and field data.
        for step in 0u64..3 {
            let t = step as f64 * 0.25;
            let coords: Vec<f64> = (0..6).map(|i| (step as f64 * 0.1) + i as f64).collect();
            write_checkpoint_mesh_coords(&path, step, t, 3, &coords).unwrap();
            write_checkpoint_step_f64(&path, cfg, step, t, &[RankFieldF64 {
                name: "phi".into(), global_offset: 0, global_len: 2,
                values: vec![step as f64, step as f64 * 2.0],
            }]).unwrap();
            materialize_global_field_f64(&path, 1, step, "phi").unwrap();
        }

        // Restart from step 2: verify coords and field both consistent.
        let (latest_step, latest_coords) = read_checkpoint_mesh_coords_latest(&path).unwrap();
        assert_eq!(latest_step, 2, "latest coord step should be 2");
        let expected_coords: Vec<f64> = (0..6).map(|i| 0.2 + i as f64).collect();
        for (a, b) in latest_coords.iter().zip(expected_coords.iter()) {
            assert!((a - b).abs() < 1e-12, "coord mismatch: {a} != {b}");
        }

        let phi_back = read_checkpoint_step_f64_hyperslab(
            &path, cfg, 2, &[("phi", 0, 2)],
        ).unwrap();
        assert_eq!(phi_back[0].1, vec![2.0, 4.0], "field data at restart step 2 mismatch");

        let _ = std::fs::remove_file(&path);
    }

    /// Validate that checkpoint layout passes validation after multi-step write.
    #[test]
    fn hyperslab_multi_step_layout_validates() {
        let path = make_restart_path("validate");
        let cfg0 = ParallelIoConfig { world_size: 2, rank: 0 };
        let cfg1 = ParallelIoConfig { world_size: 2, rank: 1 };

        for step in 0u64..4 {
            let t = step as f64 * 0.05;
            write_checkpoint_step_f64(&path, cfg0, step, t, &[RankFieldF64 {
                name: "u".into(), global_offset: 0, global_len: 6, values: vec![1.0, 2.0, 3.0],
            }]).unwrap();
            write_checkpoint_step_f64(&path, cfg1, step, t, &[RankFieldF64 {
                name: "u".into(), global_offset: 3, global_len: 6, values: vec![4.0, 5.0, 6.0],
            }]).unwrap();
            materialize_global_field_f64(&path, 2, step, "u").unwrap();
        }

        let report = validate_checkpoint_layout(&path, Some(2)).unwrap();
        assert!(report.warnings.is_empty(), "unexpected warnings: {:?}", report.warnings);
        assert_eq!(report.steps.len(), 4);
        assert_eq!(report.world_size, 2);

        let _ = std::fs::remove_file(&path);
    }
}
