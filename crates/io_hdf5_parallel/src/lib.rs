//! Parallel HDF5 I/O helpers.
//!
//! This crate isolates HDF5-related parallel read/write concerns behind a small,
//! feature-gated API.
//!
//! Design goals:
//! - Keep the workspace buildable without HDF5 installed.
//! - Offer a rank-partitioned file layout that is deterministic and restart-friendly.
//! - Provide a thin API that can later be upgraded to MPI-collective I/O.

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
#[derive(Debug, Clone, Copy, PartialEq)]
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
        let _ = (file_path, step, time, fields);
        Err(Hdf5ParallelError::Hdf5MpiFeatureDisabled)
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
        let _ = bundle;
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
        let _ = (file_path, dataset, local_values);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
        let _ = (file_path, dataset);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
        let _ = (file_path, step, time, fields);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
        let _ = (file_path, step, field_name);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
        let _ = (file_path, field_name);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
        let _ = (file_path, expected_world_size);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
        let _ = (file_path, step, field_name);
        Err(Hdf5ParallelError::Hdf5FeatureDisabled)
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
    fn no_hdf5_feature_returns_expected_error() {
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
        let err = write_checkpoint_step_f64(
            "dummy.h5",
            cfg,
            0,
            0.0,
            &[RankFieldF64 {
                name: "u".into(),
                global_offset: 0,
                global_len: 3,
                values: vec![1.0, 2.0, 3.0],
            }],
        )
        .expect_err("expected feature disabled in default build");

        match err {
            Hdf5ParallelError::Hdf5FeatureDisabled => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mpi_backend_without_feature_reports_expected_error() {
        let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
        let err = write_checkpoint_step_f64_with_backend(
            "dummy.h5",
            cfg,
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
        .expect_err("expected hdf5-mpi feature disabled in default build");

        match err {
            Hdf5ParallelError::Hdf5MpiFeatureDisabled => {}
            other => panic!("unexpected error: {other}"),
        }
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
    fn materialize_global_without_hdf5_feature_errors() {
        let err = materialize_global_field_f64("dummy.h5", 2, 0, "u")
            .expect_err("expected feature disabled in default build");
        match err {
            Hdf5ParallelError::Hdf5FeatureDisabled => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn validate_layout_without_hdf5_feature_errors() {
        let err = validate_checkpoint_layout("dummy.h5", Some(2))
            .expect_err("expected feature disabled in default build");
        match err {
            Hdf5ParallelError::Hdf5FeatureDisabled => {}
            other => panic!("unexpected error: {other}"),
        }
    }
}
