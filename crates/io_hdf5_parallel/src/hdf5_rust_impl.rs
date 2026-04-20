//! Checkpoint I/O backed by the pure-Rust [`rust_hdf5`](https://crates.io/crates/rust-hdf5) crate.
//!
//! On-disk layout uses **schema version 2** (scalar datasets for `/meta` and step metadata).
//! Legacy libhdf5 / `hdf5` 0.8 checkpoints (schema v1, group attributes) are not read here.
//!
//! `rust_hdf5` writers cannot reopen existing datasets by name, so multi-rank incremental
//! hyperslab writes to a shared global dataset are **not supported**; see [`SUPPORTS_HYPERSLAB`].

use std::path::Path;

use rust_hdf5::{H5File, H5Group};

use crate::{
    CheckpointMeshMeta, CheckpointStepInfo, CheckpointValidationReport, Hdf5ParallelError,
    ParallelIoConfig, RankFieldF64, RankFieldReadF64,
};

/// Schema version stored at `/meta/schema_version` for this backend.
pub const SCHEMA_VERSION_RUST: u32 = 2;

/// Whether [`crate::write_checkpoint_step_f64_hyperslab`] can stage multi-rank global hyperslabs.
pub const SUPPORTS_HYPERSLAB: bool = false;

fn be(s: impl ToString) -> Hdf5ParallelError {
    Hdf5ParallelError::Backend(s.to_string())
}

fn ic(s: impl ToString) -> Hdf5ParallelError {
    Hdf5ParallelError::InvalidCheckpoint(s.to_string())
}

fn parse_step_name(name: &str) -> Option<u64> {
    let pfx = "step_";
    if !name.starts_with(pfx) {
        return None;
    }
    name[pfx.len()..].parse::<u64>().ok()
}

fn open_read(path: &str) -> Result<H5File, Hdf5ParallelError> {
    H5File::open(path).map_err(be)
}

fn open_rw(path: &str) -> Result<H5File, Hdf5ParallelError> {
    H5File::open_rw(path).map_err(be)
}

fn create_new(path: &str) -> Result<H5File, Hdf5ParallelError> {
    H5File::create(path).map_err(be)
}

fn root_group(file: &H5File) -> H5Group {
    file.root_group()
}

fn read_scalar_u64_dataset(file: &H5File, path: &str) -> Result<u64, Hdf5ParallelError> {
    let ds = file.dataset(path).map_err(be)?;
    let v = ds.read_raw::<u64>().map_err(be)?;
    v.first().copied().ok_or_else(|| ic(format!("empty dataset {path}")))
}

fn read_scalar_f64_dataset(file: &H5File, path: &str) -> Result<f64, Hdf5ParallelError> {
    let ds = file.dataset(path).map_err(be)?;
    let v = ds.read_raw::<f64>().map_err(be)?;
    v.first().copied().ok_or_else(|| ic(format!("empty dataset {path}")))
}

fn write_scalar_u64_ds(g: &H5Group, name: &str, value: u64) -> Result<(), Hdf5ParallelError> {
    let ds = g
        .new_dataset::<u64>()
        .shape(&[1])
        .create(name)
        .map_err(be)?;
    ds.write_raw(&[value]).map_err(be)?;
    Ok(())
}

fn write_scalar_f64_ds(g: &H5Group, name: &str, value: f64) -> Result<(), Hdf5ParallelError> {
    let ds = g
        .new_dataset::<f64>()
        .shape(&[1])
        .create(name)
        .map_err(be)?;
    ds.write_raw(&[value]).map_err(be)?;
    Ok(())
}

fn ensure_child_group(parent: &H5Group, name: &str) -> Result<H5Group, Hdf5ParallelError> {
    let groups = parent.group_names().map_err(be)?;
    if groups.iter().any(|g| g == name) {
        return parent.group(name).map_err(be);
    }
    parent.create_group(name).map_err(be)
}

/// Verify schema v2 and `world_size` before appending with a writer handle.
fn verify_reader_meta(file: &H5File, cfg: ParallelIoConfig) -> Result<(), Hdf5ParallelError> {
    let ws = read_scalar_u64_dataset(file, "meta/world_size")?;
    if ws != cfg.world_size as u64 {
        return Err(ic(format!(
            "world_size mismatch: file={ws}, expected={}",
            cfg.world_size
        )));
    }
    let sv = read_scalar_u64_dataset(file, "meta/schema_version")?;
    if sv != SCHEMA_VERSION_RUST {
        return Err(ic(format!(
            "unsupported schema_version in rust-hdf5 backend: file={sv}, expected {SCHEMA_VERSION_RUST}"
        )));
    }
    Ok(())
}

fn init_new_checkpoint(file: &H5File, cfg: ParallelIoConfig) -> Result<(), Hdf5ParallelError> {
    let root = root_group(file);
    let meta = ensure_child_group(&root, "meta")?;
    write_scalar_u64_ds(&meta, "world_size", cfg.world_size as u64)?;
    write_scalar_u64_ds(&meta, "schema_version", SCHEMA_VERSION_RUST)?;
    let _ = ensure_child_group(&root, "steps")?;
    Ok(())
}

pub fn write_checkpoint_step_f64(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    time: f64,
    fields: &[RankFieldF64],
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;

    let file = if Path::new(file_path).exists() {
        {
            let r = open_read(file_path)?;
            verify_reader_meta(&r, cfg)?;
        }
        open_rw(file_path)?
    } else {
        let f = create_new(file_path)?;
        init_new_checkpoint(&f, cfg)?;
        f
    };

    let root = root_group(&file);
    let steps = ensure_child_group(&root, "steps")?;
    let step_name = format!("step_{:08}", step);
    let step_g = ensure_child_group(&steps, &step_name)?;
    let step_children = step_g.dataset_names().map_err(be)?;
    if !step_children.iter().any(|n| n == "time") {
        write_scalar_u64_ds(&step_g, "step", step)?;
        write_scalar_f64_ds(&step_g, "time", time)?;
    }

    let parts = ensure_child_group(&step_g, "partitions")?;
    let rank_g = ensure_child_group(&parts, &cfg.rank_group_name())?;

    for f in fields {
        let ds = rank_g
            .new_dataset::<f64>()
            .shape(&[f.values.len()])
            .create(f.name.as_str())
            .map_err(be)?;
        ds.write_raw(&f.values).map_err(be)?;
        let a1 = ds
            .new_attr::<u64>()
            .shape(())
            .create("global_offset")
            .map_err(be)?;
        a1.write_numeric(&f.global_offset).map_err(be)?;
        let a2 = ds
            .new_attr::<u64>()
            .shape(())
            .create("global_len")
            .map_err(be)?;
        a2.write_numeric(&f.global_len).map_err(be)?;
    }

    Ok(())
}

pub fn write_checkpoint_step_f64_hyperslab(
    _file_path: &str,
    _cfg: ParallelIoConfig,
    _step: u64,
    _time: f64,
    _fields: &[RankFieldF64],
) -> Result<(), Hdf5ParallelError> {
    Err(be(
        "rust_hdf5 backend: multi-rank hyperslab staging to shared global datasets is not supported",
    ))
}

pub fn write_checkpoint_step_bundle_f64_mesh_meta(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    _time: f64,
    meta: CheckpointMeshMeta,
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;
    let file = open_rw(file_path)?;
    verify_reader_meta(&file, cfg)?;
    let root = root_group(&file);
    let steps = ensure_child_group(&root, "steps")?;
    let step_name = format!("step_{:08}", step);
    let step_g = ensure_child_group(&steps, &step_name)?;
    let mesh = ensure_child_group(&step_g, "mesh_meta")?;

    if mesh.dataset_names().map_err(be)?.is_empty() {
        write_scalar_u64_ds(&mesh, "dim", meta.dim as u64)?;
        write_scalar_u64_ds(&mesh, "n_vertices", meta.n_vertices)?;
        write_scalar_u64_ds(&mesh, "n_elements", meta.n_elements)?;
    }
    Ok(())
}

pub fn write_rank_partition_f64(
    file_path: &str,
    dataset: &str,
    local_values: &[f64],
    cfg: ParallelIoConfig,
) -> Result<(), Hdf5ParallelError> {
    cfg.validate()?;

    let file = if Path::new(file_path).exists() {
        {
            let r = open_read(file_path)?;
            verify_reader_meta(&r, cfg)?;
        }
        open_rw(file_path)?
    } else {
        let f = create_new(file_path)?;
        init_new_checkpoint(&f, cfg)?;
        f
    };

    let root = root_group(&file);
    let parts = ensure_child_group(&root, "partitions")?;
    let rank_g = ensure_child_group(&parts, &cfg.rank_group_name())?;

    let ds = rank_g
        .new_dataset::<f64>()
        .shape(&[local_values.len()])
        .create(dataset)
        .map_err(be)?;
    ds.write_raw(local_values).map_err(be)?;
    Ok(())
}

pub fn read_rank_partition_f64(
    file_path: &str,
    dataset: &str,
    cfg: ParallelIoConfig,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    cfg.validate()?;
    let file = open_read(file_path)?;
    let path = format!(
        "partitions/{}/{}",
        cfg.rank_group_name(),
        dataset.trim_start_matches('/')
    );
    let ds = file.dataset(&path).map_err(be)?;
    ds.read_raw::<f64>().map_err(be)
}

pub fn read_checkpoint_field_f64_at_step(
    file_path: &str,
    cfg: ParallelIoConfig,
    step: u64,
    field_name: &str,
) -> Result<RankFieldReadF64, Hdf5ParallelError> {
    cfg.validate()?;
    let file = open_read(file_path)?;
    let step_name = format!("step_{:08}", step);
    let time = read_scalar_f64_dataset(&file, &format!("steps/{step_name}/time"))?;

    let path = format!(
        "steps/{}/partitions/{}/{}",
        step_name,
        cfg.rank_group_name(),
        field_name
    );
    let ds = file.dataset(&path).map_err(be)?;
    let values = ds.read_raw::<f64>().map_err(be)?;
    let global_offset = ds
        .attr("global_offset")
        .map_err(|e| ic(format!("missing global_offset: {e}")))?
        .read_numeric::<u64>()
        .map_err(be)?;
    let global_len = ds
        .attr("global_len")
        .map_err(|e| ic(format!("missing global_len: {e}")))?
        .read_numeric::<u64>()
        .map_err(be)?;

    Ok(RankFieldReadF64 {
        step,
        time,
        global_offset,
        global_len,
        values,
    })
}

pub fn read_checkpoint_field_f64_latest(
    file_path: &str,
    cfg: ParallelIoConfig,
    field_name: &str,
) -> Result<RankFieldReadF64, Hdf5ParallelError> {
    cfg.validate()?;
    let file = open_read(file_path)?;
    let steps = root_group(&file)
        .group("steps")
        .map_err(|e| ic(format!("missing /steps: {e}")))?;
    let mut max_step: Option<u64> = None;
    for name in steps.group_names().map_err(be)? {
        if let Some(s) = parse_step_name(&name) {
            max_step = Some(max_step.map_or(s, |m| m.max(s)));
        }
    }
    let step = max_step.ok_or_else(|| ic("no step_* groups found"))?;
    read_checkpoint_field_f64_at_step(file_path, cfg, step, field_name)
}

pub fn validate_checkpoint_layout(
    file_path: &str,
    expected_world_size: Option<usize>,
) -> Result<CheckpointValidationReport, Hdf5ParallelError> {
    let file = open_read(file_path)?;
    let world_size = read_scalar_u64_dataset(&file, "meta/world_size")? as usize;
    let schema_version = read_scalar_u64_dataset(&file, "meta/schema_version")? as u32;

    if schema_version != SCHEMA_VERSION_RUST {
        return Err(ic(format!(
            "unsupported schema_version for rust-hdf5 validator: {schema_version}"
        )));
    }

    if let Some(exp) = expected_world_size {
        if exp != world_size {
            return Err(ic(format!(
                "world_size mismatch: file={world_size}, expected={exp}"
            )));
        }
    }

    let steps_g = root_group(&file)
        .group("steps")
        .map_err(|e| ic(format!("missing /steps: {e}")))?;
    let mut step_ids = Vec::new();
    for name in steps_g.group_names().map_err(be)? {
        if let Some(s) = parse_step_name(&name) {
            step_ids.push(s);
        }
    }
    step_ids.sort_unstable();

    let mut step_infos = Vec::new();
    let mut warnings = Vec::new();

    for step in step_ids {
        let step_name = format!("step_{:08}", step);
        let time = read_scalar_f64_dataset(&file, &format!("steps/{step_name}/time"))?;
        let parts = steps_g
            .group(&step_name)
            .map_err(|e| ic(format!("missing step group {step_name}: {e}")))?
            .group("partitions")
            .map_err(|e| ic(format!("missing partitions in {step_name}: {e}")))?;

        let mut field_ranges: std::collections::HashMap<String, (u64, Vec<(u64, u64)>)> =
            std::collections::HashMap::new();
        let mut present_ranks = 0usize;

        for rank in 0..world_size {
            let rank_name = format!("rank_{:06}", rank);
            let rank_group = parts.group(&rank_name).map_err(|e| {
                ic(format!("missing rank group {rank_name} in {step_name}: {e}"))
            })?;
            present_ranks += 1;

            let mut names = rank_group.dataset_names().map_err(be)?;
            names.sort_unstable();
            for fname in names {
                let ds_path =
                    format!("steps/{step_name}/partitions/{rank_name}/{fname}");
                let ds = file.dataset(&ds_path).map_err(be)?;
                let off = ds
                    .attr("global_offset")
                    .map_err(|e| {
                        ic(format!(
                            "missing global_offset in {step_name}/{rank_name}/{fname}: {e}"
                        ))
                    })?
                    .read_numeric::<u64>()
                    .map_err(be)?;
                let gl = ds
                    .attr("global_len")
                    .map_err(|e| {
                        ic(format!("missing global_len in {step_name}/{rank_name}/{fname}: {e}"))
                    })?
                    .read_numeric::<u64>()
                    .map_err(be)?;
                let nloc = ds.total_elements() as u64;
                let end = off.saturating_add(nloc);
                if end > gl {
                    return Err(ic(format!(
                        "chunk out of bounds in {step_name}/{rank_name}/{fname}: [{off},{end}) > global_len={gl}"
                    )));
                }

                let entry = field_ranges
                    .entry(fname)
                    .or_insert_with(|| (gl, Vec::new()));
                if entry.0 != gl {
                    return Err(ic("global_len mismatch across ranks".to_string()));
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
                    return Err(ic(format!(
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

    Ok(CheckpointValidationReport {
        schema_version,
        world_size,
        steps: step_infos,
        warnings,
    })
}

pub fn materialize_global_field_f64(
    file_path: &str,
    world_size: usize,
    step: u64,
    field_name: &str,
) -> Result<u64, Hdf5ParallelError> {
    if world_size == 0 {
        return Err(Hdf5ParallelError::InvalidConfig("world_size must be > 0"));
    }

    let cfg0 = ParallelIoConfig {
        world_size,
        rank: 0,
    };

    let file = open_read(file_path)?;
    verify_reader_meta(&file, cfg0)?;
    let step_name = format!("step_{:08}", step);

    let mut global_len: Option<u64> = None;
    let mut chunks: Vec<(u64, Vec<f64>)> = Vec::with_capacity(world_size);

    for rank in 0..world_size {
        let rank_name = format!("rank_{:06}", rank);
        let ds_path = format!("steps/{step_name}/partitions/{rank_name}/{field_name}");
        let ds = file.dataset(&ds_path).map_err(be)?;
        let vals = ds.read_raw::<f64>().map_err(be)?;
        let off = ds
            .attr("global_offset")
            .map_err(|e| ic(format!("missing global_offset attr: {e}")))?
            .read_numeric::<u64>()
            .map_err(be)?;
        let gl = ds
            .attr("global_len")
            .map_err(|e| ic(format!("missing global_len attr: {e}")))?
            .read_numeric::<u64>()
            .map_err(be)?;

        if let Some(prev) = global_len {
            if prev != gl {
                return Err(ic(format!(
                    "global_len mismatch across ranks: {prev} vs {gl}"
                )));
            }
        } else {
            global_len = Some(gl);
        }
        chunks.push((off, vals));
    }

    let gl = global_len.ok_or_else(|| ic("no rank chunks found"))?;
    let mut global = vec![0.0f64; gl as usize];
    for (off, vals) in chunks {
        let start = off as usize;
        let end = start + vals.len();
        if end > global.len() {
            return Err(ic(format!(
                "partition out of bounds: [{start},{end}) vs global {}",
                global.len()
            )));
        }
        global[start..end].copy_from_slice(&vals);
    }

    drop(file);

    {
        let r = open_read(file_path)?;
        verify_reader_meta(
            &r,
            ParallelIoConfig {
                world_size,
                rank: 0,
            },
        )?;
    }

    let file = open_rw(file_path)?;
    let root = root_group(&file);
    let globals = ensure_child_group(&root, "global_fields")?;
    let step_global = ensure_child_group(&globals, &step_name)?;

    let ds = step_global
        .new_dataset::<f64>()
        .shape(&[global.len()])
        .create(field_name)
        .map_err(be)?;
    ds.write_raw(&global).map_err(be)?;

    Ok(gl)
}

pub fn read_global_field_f64(
    file_path: &str,
    step: u64,
    field_name: &str,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    let file = open_read(file_path)?;
    let step_name = format!("step_{:08}", step);
    let path = format!("global_fields/{step_name}/{field_name}");
    let ds = file.dataset(&path).map_err(be)?;
    ds.read_raw::<f64>().map_err(be)
}

pub fn read_global_field_slice_f64(
    file_path: &str,
    step: u64,
    field_name: &str,
    global_offset: u64,
    local_len: usize,
) -> Result<Vec<f64>, Hdf5ParallelError> {
    let file = open_read(file_path)?;
    let step_name = format!("step_{:08}", step);
    let path = format!("global_fields/{step_name}/{field_name}");
    let ds = file.dataset(&path).map_err(be)?;
    let n = ds.total_elements();
    let start = global_offset as usize;
    let end = start.saturating_add(local_len);
    if end > n {
        return Err(ic(format!(
            "requested global slice [{start},{end}) exceeds dataset length {n}"
        )));
    }
    ds.read_slice::<f64>(&[start], &[local_len]).map_err(be)
}
