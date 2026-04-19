//! Parallel HDF5 I/O helpers.
//!
//! This crate isolates HDF5-related parallel read/write concerns behind a small,
//! feature-gated API.
//!
//! Design goals:
//! - Keep the workspace buildable without HDF5 installed.
//! - Offer a rank-partitioned file layout that is deterministic and restart-friendly.
//! - Provide a thin API that can later be upgraded to MPI-collective I/O.

use serde::{Deserialize, Serialize};
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

    #[cfg(feature = "hdf5")]
    fn rank_group_name(&self) -> String {
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
        use mpi::topology::SystemCommunicator;
        use mpi::traits::{Communicator, CommunicatorCollectives};

        let world = SystemCommunicator::world();
        let comm_rank = world.rank() as usize;
        let comm_size = world.size() as usize;

        // If caller config does not match active communicator, degrade to
        // deterministic partitioned write rather than failing unexpectedly.
        if comm_size != cfg.world_size || comm_rank != cfg.rank {
            return write_checkpoint_step_f64(file_path, cfg, step, time, fields);
        }

        write_checkpoint_step_f64(file_path, cfg, step, time, fields)?;

        // Try direct global-dataset hyperslab writes first.
        // Keep the root materialization path below as compatibility fallback.
        let _ = write_checkpoint_step_f64_hyperslab(file_path, cfg, step, time, fields);
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
        use hdf5::File;

        let file = if std::path::Path::new(file_path).exists() {
            File::open_rw(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        } else {
            File::create(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        };

        let meta = match file.group("meta") {
            Ok(g) => g,
            Err(_) => file.create_group("meta").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        upsert_attr_u64(&meta, "world_size", cfg.world_size as u64)?;
        upsert_attr_u64(&meta, "schema_version", CHECKPOINT_SCHEMA_VERSION as u64)?;

        let steps = match file.group("steps") {
            Ok(g) => g,
            Err(_) => file.create_group("steps").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        let step_name = format!("step_{:08}", step);
        let step_group = match steps.group(&step_name) {
            Ok(g) => g,
            Err(_) => steps.create_group(&step_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        upsert_attr_u64(&step_group, "step", step)?;
        upsert_attr_f64(&step_group, "time", time)?;

        let globals = match file.group("global_fields") {
            Ok(g) => g,
            Err(_) => file.create_group("global_fields").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        let step_global = match globals.group(&step_name) {
            Ok(g) => g,
            Err(_) => globals.create_group(&step_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };

        for f in fields {
            let gl = f.global_len as usize;
            let start = f.global_offset as usize;
            let end = start.saturating_add(f.values.len());
            if end > gl {
                return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                    "hyperslab out of bounds for {}: [{start},{end}) > global_len={gl}",
                    f.name
                )));
            }

            let ds = if step_global.link_exists(&f.name) {
                let ds = step_global
                    .dataset(&f.name)
                    .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
                if ds.size() != gl {
                    return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                        "dataset size mismatch for {}: existing={}, expected={}",
                        f.name,
                        ds.size(),
                        gl
                    )));
                }
                ds
            } else {
                step_global
                    .new_dataset::<f64>()
                    .shape([gl])
                    .create(f.name.as_str())
                    .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
            };

            ds.write_slice(&f.values, start..end)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            upsert_attr_u64(&ds, "global_len", f.global_len)?;
        }

        return Ok(());
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
            use hdf5::File;

            let file = hdf5::File::open_rw(file_path)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let steps = file
                .group("steps")
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let step_name = format!("step_{:08}", step);
            let step_group = steps
                .group(&step_name)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let mesh_group = match step_group.group("mesh_meta") {
                Ok(g) => g,
                Err(_) => step_group
                    .create_group("mesh_meta")
                    .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
            };

            upsert_attr_u64(&mesh_group, "dim", meta.dim as u64)?;
            upsert_attr_u64(&mesh_group, "n_vertices", meta.n_vertices)?;
            upsert_attr_u64(&mesh_group, "n_elements", meta.n_elements)?;
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
        use hdf5::File;

        let file = if std::path::Path::new(file_path).exists() {
            File::open_rw(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        } else {
            File::create(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        };

        let meta = match file.group("meta") {
            Ok(g) => g,
            Err(_) => file.create_group("meta").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        if let Ok(attr) = meta.attr("world_size") {
            attr.write_scalar(&(cfg.world_size as u64))
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        } else {
            meta.new_attr::<u64>()
                .create("world_size")
                .and_then(|a| a.write_scalar(&(cfg.world_size as u64)))
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        }

        let parts = match file.group("partitions") {
            Ok(g) => g,
            Err(_) => file.create_group("partitions").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        let rank_group_name = cfg.rank_group_name();
        let rank_group = match parts.group(&rank_group_name) {
            Ok(g) => g,
            Err(_) => parts.create_group(&rank_group_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };

        if rank_group.link_exists(dataset) {
            rank_group
                .unlink(dataset)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        }

        rank_group
            .new_dataset_builder()
            .with_data(local_values)
            .create(dataset)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

        return Ok(());
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
        use hdf5::File;

        let file = File::open(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let parts = file.group("partitions").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let rank_group = parts
            .group(&cfg.rank_group_name())
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let ds = rank_group.dataset(dataset).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let data = ds.read_raw::<f64>().map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        return Ok(data);
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
        use hdf5::File;

        let file = if std::path::Path::new(file_path).exists() {
            File::open_rw(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        } else {
            File::create(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        };

        let meta = match file.group("meta") {
            Ok(g) => g,
            Err(_) => file.create_group("meta").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        upsert_attr_u64(&meta, "world_size", cfg.world_size as u64)?;
        upsert_attr_u64(&meta, "schema_version", CHECKPOINT_SCHEMA_VERSION as u64)?;

        let steps = match file.group("steps") {
            Ok(g) => g,
            Err(_) => file.create_group("steps").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        let step_name = format!("step_{:08}", step);
        let step_group = match steps.group(&step_name) {
            Ok(g) => g,
            Err(_) => steps.create_group(&step_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        upsert_attr_u64(&step_group, "step", step)?;
        upsert_attr_f64(&step_group, "time", time)?;

        let parts = match step_group.group("partitions") {
            Ok(g) => g,
            Err(_) => step_group.create_group("partitions").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        let rank_group_name = cfg.rank_group_name();
        let rank_group = match parts.group(&rank_group_name) {
            Ok(g) => g,
            Err(_) => parts.create_group(&rank_group_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };

        for f in fields {
            if rank_group.link_exists(&f.name) {
                rank_group
                    .unlink(&f.name)
                    .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            }
            rank_group
                .new_dataset_builder()
                .with_data(&f.values)
                .create(f.name.as_str())
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

            let ds = rank_group
                .dataset(&f.name)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            upsert_attr_u64(&ds, "global_offset", f.global_offset)?;
            upsert_attr_u64(&ds, "global_len", f.global_len)?;
        }

        return Ok(());
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
        use hdf5::File;

        let file = File::open(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let steps = file.group("steps").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let step_name = format!("step_{:08}", step);
        let step_group = steps.group(&step_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

        let time = step_group
            .attr("time")
            .and_then(|a| a.read_scalar::<f64>())
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing time attr: {e}")))?;

        let parts = step_group.group("partitions").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let rank_group = parts
            .group(&cfg.rank_group_name())
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let ds = rank_group
            .dataset(field_name)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

        let values = ds.read_raw::<f64>().map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let global_offset = ds
            .attr("global_offset")
            .and_then(|a| a.read_scalar::<u64>())
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing global_offset attr: {e}")))?;
        let global_len = ds
            .attr("global_len")
            .and_then(|a| a.read_scalar::<u64>())
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing global_len attr: {e}")))?;

        return Ok(RankFieldReadF64 {
            step,
            time,
            global_offset,
            global_len,
            values,
        });
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
        use hdf5::File;

        let file = File::open(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let steps = file.group("steps").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let mut max_step: Option<u64> = None;

        for name in steps.member_names().map_err(|e| Hdf5ParallelError::Backend(e.to_string()))? {
            if let Some(s) = parse_step_name(&name) {
                max_step = Some(max_step.map_or(s, |m| m.max(s)));
            }
        }

        let step = max_step.ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint("no step_* groups found".into()))?;
        return read_checkpoint_field_f64_at_step(file_path, cfg, step, field_name);
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
        use hdf5::File;

        let file = File::open(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let meta = file
            .group("meta")
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing /meta: {e}")))?;

        let world_size = meta
            .attr("world_size")
            .and_then(|a| a.read_scalar::<u64>())
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing world_size attr: {e}")))?
            as usize;
        let schema_version = meta
            .attr("schema_version")
            .and_then(|a| a.read_scalar::<u64>())
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing schema_version attr: {e}")))?
            as u32;

        if let Some(exp) = expected_world_size {
            if exp != world_size {
                return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                    "world_size mismatch: file={world_size}, expected={exp}"
                )));
            }
        }

        let steps_group = file
            .group("steps")
            .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing /steps: {e}")))?;
        let mut step_infos = Vec::new();
        let mut warnings = Vec::new();

        let mut step_ids = Vec::new();
        for name in steps_group
            .member_names()
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
        {
            if let Some(step) = parse_step_name(&name) {
                step_ids.push(step);
            } else {
                warnings.push(format!("ignored non-step group: {name}"));
            }
        }
        step_ids.sort_unstable();

        for step in step_ids {
            let step_name = format!("step_{:08}", step);
            let step_group = steps_group
                .group(&step_name)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let time = step_group
                .attr("time")
                .and_then(|a| a.read_scalar::<f64>())
                .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing time attr in {step_name}: {e}")))?;
            let parts = step_group
                .group("partitions")
                .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing partitions in {step_name}: {e}")))?;

            let mut field_ranges: std::collections::HashMap<String, (u64, Vec<(u64, u64)>)> =
                std::collections::HashMap::new();
            let mut present_ranks = 0usize;
            for rank in 0..world_size {
                let rank_name = format!("rank_{:06}", rank);
                let rank_group = parts.group(&rank_name).map_err(|e| {
                    Hdf5ParallelError::InvalidCheckpoint(format!(
                        "missing rank group {rank_name} in {step_name}: {e}"
                    ))
                })?;
                present_ranks += 1;

                for fname in rank_group
                    .member_names()
                    .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
                {
                    let ds = rank_group
                        .dataset(&fname)
                        .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
                    let off = ds
                        .attr("global_offset")
                        .and_then(|a| a.read_scalar::<u64>())
                        .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!(
                            "missing global_offset attr in {step_name}/{rank_name}/{fname}: {e}"
                        )))?;
                    let gl = ds
                        .attr("global_len")
                        .and_then(|a| a.read_scalar::<u64>())
                        .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!(
                            "missing global_len attr in {step_name}/{rank_name}/{fname}: {e}"
                        )))?;
                    let nloc = ds.size() as u64;
                    let end = off.saturating_add(nloc);
                    if end > gl {
                        return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                            "chunk out of bounds in {step_name}/{rank_name}/{fname}: [{off},{end}) > global_len={gl}"
                        )));
                    }

                    let entry = field_ranges
                        .entry(fname)
                        .or_insert_with(|| (gl, Vec::new()));
                    if entry.0 != gl {
                        return Err(Hdf5ParallelError::InvalidCheckpoint(
                            "global_len mismatch across ranks".to_string(),
                        ));
                    }
                    entry.1.push((off, end));
                }
            }

            for (fname, (gl, mut ranges)) in field_ranges {
                ranges.sort_unstable_by_key(|r| r.0);
                let mut covered = 0u64;
                let mut prev_end = 0u64;
                for (off, end) in ranges {
                    if off < prev_end {
                        return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                            "overlapping chunks in {step_name}/{fname}"
                        )));
                    }
                    covered = covered.saturating_add(end.saturating_sub(off));
                    prev_end = end;
                }
                if covered != gl {
                    warnings.push(format!(
                        "incomplete coverage in {step_name}/{fname}: covered={covered}, global_len={gl}"
                    ));
                }
            }

            step_infos.push(CheckpointStepInfo {
                step,
                time,
                partition_count: present_ranks,
            });
        }

        return Ok(CheckpointValidationReport {
            schema_version,
            world_size,
            steps: step_infos,
            warnings,
        });
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
        use hdf5::File;

        let file = File::open_rw(file_path).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let step_name = format!("step_{:08}", step);
        let step_group = file
            .group("steps")
            .and_then(|g| g.group(&step_name))
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let parts = step_group
            .group("partitions")
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

        let mut global_len: Option<u64> = None;
        let mut chunks: Vec<(u64, Vec<f64>)> = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let rank_name = format!("rank_{:06}", rank);
            let rank_group = parts
                .group(&rank_name)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let ds = rank_group
                .dataset(field_name)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let vals = ds.read_raw::<f64>().map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
            let off = ds
                .attr("global_offset")
                .and_then(|a| a.read_scalar::<u64>())
                .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing global_offset attr: {e}")))?;
            let gl = ds
                .attr("global_len")
                .and_then(|a| a.read_scalar::<u64>())
                .map_err(|e| Hdf5ParallelError::InvalidCheckpoint(format!("missing global_len attr: {e}")))?;

            if let Some(prev) = global_len {
                if prev != gl {
                    return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                        "global_len mismatch across ranks: {prev} vs {gl}"
                    )));
                }
            } else {
                global_len = Some(gl);
            }

            chunks.push((off, vals));
        }

        let gl = global_len.ok_or_else(|| Hdf5ParallelError::InvalidCheckpoint("no rank chunks found".into()))?;
        let mut global = vec![0.0f64; gl as usize];

        for (off, vals) in chunks {
            let start = off as usize;
            let end = start + vals.len();
            if end > global.len() {
                return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                    "partition out of bounds: [{start},{end}) vs global {}",
                    global.len()
                )));
            }
            global[start..end].copy_from_slice(&vals);
        }

        let globals = match file.group("global_fields") {
            Ok(g) => g,
            Err(_) => file.create_group("global_fields").map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };
        let step_global = match globals.group(&step_name) {
            Ok(g) => g,
            Err(_) => globals.create_group(&step_name).map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?,
        };

        if step_global.link_exists(field_name) {
            step_global
                .unlink(field_name)
                .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        }
        step_global
            .new_dataset_builder()
            .with_data(&global)
            .create(field_name)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

        return Ok(gl);
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
        let file = hdf5::File::open(file_path)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let step_name = format!("step_{:08}", step);
        let ds = file
            .group("global_fields")
            .and_then(|g| g.group(&step_name))
            .and_then(|g| g.dataset(field_name))
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let vals = ds
            .read_raw::<f64>()
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        return Ok(vals);
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
        let file = hdf5::File::open(file_path)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;
        let step_name = format!("step_{:08}", step);
        let ds = file
            .group("global_fields")
            .and_then(|g| g.group(&step_name))
            .and_then(|g| g.dataset(field_name))
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?;

        let start = global_offset as usize;
        let end = start.saturating_add(local_len);
        if end > ds.size() {
            return Err(Hdf5ParallelError::InvalidCheckpoint(format!(
                "requested global slice [{start},{end}) exceeds dataset length {}",
                ds.size()
            )));
        }

        let vals = ds
            .read_slice_1d::<f64, _>(start..end)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))?
            .to_vec();
        return Ok(vals);
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

#[cfg(feature = "hdf5")]
fn upsert_attr_u64<T>(obj: &T, name: &str, value: u64) -> Result<(), Hdf5ParallelError>
where
    T: std::ops::Deref<Target = hdf5::Location>,
{
    if let Ok(attr) = obj.attr(name) {
        attr.write_scalar(&value)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))
    } else {
        obj.new_attr::<u64>()
            .create(name)
            .and_then(|a| a.write_scalar(&value))
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))
    }
}

#[cfg(feature = "hdf5")]
fn upsert_attr_f64<T>(obj: &T, name: &str, value: f64) -> Result<(), Hdf5ParallelError>
where
    T: std::ops::Deref<Target = hdf5::Location>,
{
    if let Ok(attr) = obj.attr(name) {
        attr.write_scalar(&value)
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))
    } else {
        obj.new_attr::<f64>()
            .create(name)
            .and_then(|a| a.write_scalar(&value))
            .map_err(|e| Hdf5ParallelError::Backend(e.to_string()))
    }
}

#[cfg(feature = "hdf5")]
fn parse_step_name(name: &str) -> Option<u64> {
    let pfx = "step_";
    if !name.starts_with(pfx) {
        return None;
    }
    name[pfx.len()..].parse::<u64>().ok()
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
        assert_eq!(report.schema_version, CHECKPOINT_SCHEMA_VERSION);
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
}

#[cfg(all(test, feature = "hdf5"))]
mod hyperslab_tests {
    use super::*;

    #[test]
    fn hyperslab_staged_two_rank_write_builds_global_field() {
        let world_size = 2usize;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "fem_io_hdf5_hyperslab_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        ));
        let file_path = path.to_string_lossy().to_string();

        let step = 7u64;
        let global = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for rank in 0..world_size {
            let start = rank * 3;
            let field = RankFieldF64 {
                name: "u".to_string(),
                global_offset: start as u64,
                global_len: global.len() as u64,
                values: global[start..start + 3].to_vec(),
            };
            write_checkpoint_step_f64_hyperslab(
                &file_path,
                ParallelIoConfig { world_size, rank },
                step,
                0.7,
                &[field],
            )
            .expect("hyperslab staged write");
        }

        #[cfg(feature = "hdf5")]
        {
            let f = hdf5::File::open(&file_path).expect("open hdf5");
            let ds = f
                .group("global_fields")
                .and_then(|g| g.group("step_00000007"))
                .and_then(|g| g.dataset("u"))
                .expect("read global hyperslab field");
            let vals = ds.read_raw::<f64>().expect("read raw");
            assert_eq!(vals, global);
        }

        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn hyperslab_global_read_and_slice_work() {
        let world_size = 2usize;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "fem_io_hdf5_hyperslab_read_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        ));
        let file_path = path.to_string_lossy().to_string();

        let step = 3u64;
        let global = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
        for rank in 0..world_size {
            let start = rank * 3;
            let field = RankFieldF64 {
                name: "u".to_string(),
                global_offset: start as u64,
                global_len: global.len() as u64,
                values: global[start..start + 3].to_vec(),
            };
            write_checkpoint_step_f64_hyperslab(
                &file_path,
                ParallelIoConfig { world_size, rank },
                step,
                0.3,
                &[field],
            )
            .expect("hyperslab staged write");
        }

        let all = read_global_field_f64(&file_path, step, "u").expect("read global field");
        assert_eq!(all, global);

        let slice0 = read_global_field_slice_f64(&file_path, step, "u", 0, 3)
            .expect("read first slice");
        assert_eq!(slice0, vec![10.0, 11.0, 12.0]);
        let slice1 = read_global_field_slice_f64(&file_path, step, "u", 3, 3)
            .expect("read second slice");
        assert_eq!(slice1, vec![20.0, 21.0, 22.0]);

        let _ = std::fs::remove_file(&file_path);
    }
}
