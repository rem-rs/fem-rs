//! mfem_ex43_hdf5_checkpoint - parallel checkpoint/restart baseline demo.
//!
//! This example demonstrates rank-partitioned checkpoint writing, latest-step
//! restart read, and XDMF sidecar generation using fem-io-hdf5-parallel.

use fem_io_hdf5_parallel::{
    CheckpointBundleF64, CheckpointMeshMeta, Hdf5ParallelError, IoBackend, ParallelIoConfig,
    RankFieldF64, materialize_global_field_f64, read_checkpoint_field_f64_at_step,
    read_checkpoint_field_f64_latest, read_global_field_f64, validate_checkpoint_layout, write_checkpoint_step_bundle_f64,
    write_xdmf_polyvertex_scalar_timeseries_sidecar,
};

fn main() {
    let args = parse_args();

    match run_checkpoint_demo(&args) {
        DemoOutcome::Completed(result) => {
            println!("=== mfem_ex43_hdf5_checkpoint (baseline) ===");
            println!("  out_h5={}, out_xdmf={}", args.out_h5, args.out_xdmf);
            println!("  backend={}", match args.backend { IoBackend::Partitioned => "partitioned", IoBackend::MpiCollective => "mpi" });
            println!("  restart_step={}", args.restart_step.map_or("latest".into(), |s| s.to_string()));
            println!(
                "  validation: schema={}, steps={}, warnings={}",
                result.schema_version,
                result.layout_steps,
                result.layout_warnings
            );
            println!(
                "  restart: step={}, time={:.3}, offset={}, len={}, local={:?}",
                result.restart_step,
                result.restart_time,
                result.restart_global_offset,
                result.restart_global_len,
                result.restart_values
            );
            println!("  PASS");
        }
        DemoOutcome::Hdf5Disabled => {
            println!("=== mfem_ex43_hdf5_checkpoint (baseline) ===");
            println!("  out_h5={}, out_xdmf={}", args.out_h5, args.out_xdmf);
            println!("  backend=partitioned");
            println!("  restart_step={}", args.restart_step.map_or("latest".into(), |s| s.to_string()));
            println!("  HDF5 backend disabled (build without feature `hdf5`)");
            println!("  To enable real checkpoint I/O: cargo run --example mfem_ex43_hdf5_checkpoint --features fem-io-hdf5-parallel/hdf5");
            println!("  PASS (API fallback verified)");
        }
        DemoOutcome::Hdf5MpiDisabled => {
            println!("=== mfem_ex43_hdf5_checkpoint (baseline) ===");
            println!("  out_h5={}, out_xdmf={}", args.out_h5, args.out_xdmf);
            println!("  backend=mpi");
            println!("  restart_step={}", args.restart_step.map_or("latest".into(), |s| s.to_string()));
            println!("  MPI HDF5 backend disabled (build without feature `hdf5-mpi`)");
            println!("  Use partitioned mode or enable: cargo run --example mfem_ex43_hdf5_checkpoint --features fem-io-hdf5-parallel/hdf5-mpi -- --backend mpi");
            println!("  PASS (MPI backend fallback verified)");
        }
    }
}

#[derive(Debug)]
enum DemoOutcome {
    Completed(DemoResult),
    Hdf5Disabled,
    Hdf5MpiDisabled,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct DemoResult {
    schema_version: u32,
    layout_steps: usize,
    layout_warnings: usize,
    restart_step: u64,
    restart_time: f64,
    restart_global_offset: u64,
    restart_global_len: u64,
    restart_values: Vec<f64>,
    latest_rank0_values: Vec<f64>,
    latest_rank1_values: Vec<f64>,
    latest_global_values: Vec<f64>,
    xdmf_text: String,
}

fn expected_rank_values(step: u64, rank: usize) -> Vec<f64> {
    let t = 0.1 * step as f64;
    let start = if rank == 0 { 0 } else { 4 };
    let end = start + 4;
    (start..end).map(|i| i as f64 + t).collect()
}

fn expected_global_values(step: u64) -> Vec<f64> {
    (0..8).map(|i| i as f64 + 0.1 * step as f64).collect()
}

fn run_checkpoint_demo(args: &Args) -> DemoOutcome {
    let world_size = 2usize;
    let global_len = 8u64;
    let requested_restart_step = args.restart_step.unwrap_or(2);

    if let Some(parent) = std::path::Path::new(&args.out_h5).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    if let Some(parent) = std::path::Path::new(&args.out_xdmf).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let _ = std::fs::remove_file(&args.out_h5);
    let _ = std::fs::remove_file(&args.out_xdmf);

    // Synthetic transient field on 2 rank partitions.
    // rank 0 owns [0..4), rank 1 owns [4..8).
    let mesh_meta = CheckpointMeshMeta {
        dim: 1,
        n_vertices: global_len,
        n_elements: global_len.saturating_sub(1),
    };

    for step in 0..=2u64 {
        let t = 0.1 * step as f64;

        let rank0 = make_rank_field(0, global_len, t);
        let rank1 = make_rank_field(1, global_len, t);

        for (rank, field) in [(0usize, rank0), (1usize, rank1)] {
            let cfg = ParallelIoConfig { world_size, rank };
            let bundle = CheckpointBundleF64 {
                mesh_meta: Some(mesh_meta),
                fields: vec![field],
            };
            match write_checkpoint_step_bundle_f64(&args.out_h5, cfg, step, t, &bundle, args.backend) {
                Ok(()) => {}
                Err(Hdf5ParallelError::Hdf5FeatureDisabled) => return DemoOutcome::Hdf5Disabled,
                Err(Hdf5ParallelError::Hdf5MpiFeatureDisabled) => return DemoOutcome::Hdf5MpiDisabled,
                Err(e) => panic!("checkpoint write failed: {e}"),
            }
        }
    }

    // Build global datasets for visualization and emit a temporal XDMF sidecar.
    let mut time_steps = Vec::new();
    let mut gl = 0u64;
    for step in 0..=2u64 {
        gl = materialize_global_field_f64(&args.out_h5, world_size, step, "u")
            .expect("materialize global field failed");
        time_steps.push((step, 0.1 * step as f64));
    }
    write_xdmf_polyvertex_scalar_timeseries_sidecar(&args.out_xdmf, &args.out_h5, "u", gl, &time_steps)
        .expect("xdmf write failed");

    let report = validate_checkpoint_layout(&args.out_h5, Some(world_size)).expect("layout validation failed");

    // Restart read for rank 1 from selected step (or latest).
    let restart = if let Some(step) = args.restart_step {
        read_checkpoint_field_f64_at_step(
            &args.out_h5,
            ParallelIoConfig { world_size, rank: 1 },
            step,
            "u",
        )
    } else {
        read_checkpoint_field_f64_latest(
            &args.out_h5,
            ParallelIoConfig { world_size, rank: 1 },
            "u",
        )
    }
    .expect("restart read failed");

    let latest_rank0 = read_checkpoint_field_f64_latest(
        &args.out_h5,
        ParallelIoConfig { world_size, rank: 0 },
        "u",
    )
    .expect("latest rank-0 read failed");
    let latest_rank1 = read_checkpoint_field_f64_latest(
        &args.out_h5,
        ParallelIoConfig { world_size, rank: 1 },
        "u",
    )
    .expect("latest rank-1 read failed");
    let latest_global = read_global_field_f64(&args.out_h5, 2, "u")
        .expect("read global field failed");
    let xdmf_text = std::fs::read_to_string(&args.out_xdmf)
        .expect("xdmf read failed");

    assert_eq!(restart.step, requested_restart_step);
    assert!((restart.time - 0.1 * requested_restart_step as f64).abs() < 1.0e-12);
    assert_eq!(restart.global_offset, 4);
    assert_eq!(restart.global_len, 8);
    assert_eq!(restart.values.len(), 4);
    assert_eq!(restart.values, expected_rank_values(requested_restart_step, 1));
    assert_eq!(latest_rank0.values, expected_rank_values(2, 0));
    assert_eq!(latest_rank1.values, expected_rank_values(2, 1));
    assert_eq!(latest_global, expected_global_values(2));

    DemoOutcome::Completed(DemoResult {
        schema_version: report.schema_version,
        layout_steps: report.steps.len(),
        layout_warnings: report.warnings.len(),
        restart_step: restart.step,
        restart_time: restart.time,
        restart_global_offset: restart.global_offset,
        restart_global_len: restart.global_len,
        restart_values: restart.values,
        latest_rank0_values: latest_rank0.values,
        latest_rank1_values: latest_rank1.values,
        latest_global_values: latest_global,
        xdmf_text,
    })
}

fn make_rank_field(rank: usize, global_len: u64, t: f64) -> RankFieldF64 {
    let start = if rank == 0 { 0 } else { 4 };
    let end = start + 4;
    RankFieldF64 {
        name: "u".into(),
        global_offset: start as u64,
        global_len,
        values: (start..end).map(|i| i as f64 + t).collect(),
    }
}

struct Args {
    out_h5: String,
    out_xdmf: String,
    backend: IoBackend,
    restart_step: Option<u64>,
}

fn parse_args() -> Args {
    let mut args = Args {
        out_h5: "output/checkpoint_demo.h5".into(),
        out_xdmf: "output/checkpoint_demo.xdmf".into(),
        backend: IoBackend::Partitioned,
        restart_step: None,
    };

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out-h5" => args.out_h5 = it.next().unwrap_or_else(|| args.out_h5.clone()),
            "--out-xdmf" => args.out_xdmf = it.next().unwrap_or_else(|| args.out_xdmf.clone()),
            "--backend" => {
                let b = it.next().unwrap_or_else(|| "partitioned".into()).to_lowercase();
                args.backend = if b == "mpi" {
                    IoBackend::MpiCollective
                } else {
                    IoBackend::Partitioned
                };
            }
            "--restart-step" => {
                let s = it.next().unwrap_or_else(|| "latest".into()).to_lowercase();
                args.restart_step = if s == "latest" { None } else { s.parse::<u64>().ok() };
            }
            _ => {}
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths(tag: &str) -> (String, String) {
        let mut base = std::env::temp_dir();
        let unique = format!(
            "fem_ex43_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        );
        base.push(unique);
        let h5 = base.with_extension("h5").to_string_lossy().to_string();
        let xdmf = base.with_extension("xdmf").to_string_lossy().to_string();
        (h5, xdmf)
    }

    #[test]
    fn ex43_checkpoint_latest_restart_matches_expected_partition() {
        let (out_h5, out_xdmf) = temp_paths("latest");
        let args = Args {
            out_h5: out_h5.clone(),
            out_xdmf: out_xdmf.clone(),
            backend: IoBackend::Partitioned,
            restart_step: None,
        };

        match run_checkpoint_demo(&args) {
            DemoOutcome::Completed(result) => {
                assert_eq!(result.layout_steps, 3);
                assert_eq!(result.layout_warnings, 0);
                assert_eq!(result.restart_step, 2);
                assert!((result.restart_time - 0.2).abs() < 1.0e-12);
                assert_eq!(result.restart_global_offset, 4);
                assert_eq!(result.restart_global_len, 8);
                assert_eq!(result.restart_values, expected_rank_values(2, 1));
                assert_eq!(result.latest_rank0_values, expected_rank_values(2, 0));
                assert_eq!(result.latest_rank1_values, expected_rank_values(2, 1));
                assert_eq!(result.latest_global_values, expected_global_values(2));
                let stitched: Vec<f64> = result
                    .latest_rank0_values
                    .iter()
                    .chain(result.latest_rank1_values.iter())
                    .copied()
                    .collect();
                assert_eq!(stitched, result.latest_global_values);
                assert!(result.xdmf_text.contains("CollectionType=\"Temporal\""));
                assert!(result.xdmf_text.contains("Value=\"0\""));
                assert!(result.xdmf_text.contains("Value=\"0.1\""));
                assert!(result.xdmf_text.contains("Value=\"0.2\""));
                assert!(result.xdmf_text.contains("/global_fields/step_00000000/u"));
                assert!(result.xdmf_text.contains("/global_fields/step_00000001/u"));
                assert!(result.xdmf_text.contains("/global_fields/step_00000002/u"));
            }
            DemoOutcome::Hdf5Disabled => {}
            DemoOutcome::Hdf5MpiDisabled => panic!("partitioned backend should not report hdf5-mpi feature disabled"),
        }

        let _ = std::fs::remove_file(out_h5);
        let _ = std::fs::remove_file(out_xdmf);
    }

    #[test]
    fn ex43_checkpoint_can_restart_from_requested_step() {
        let (out_h5, out_xdmf) = temp_paths("step1");
        let args = Args {
            out_h5: out_h5.clone(),
            out_xdmf: out_xdmf.clone(),
            backend: IoBackend::Partitioned,
            restart_step: Some(1),
        };

        match run_checkpoint_demo(&args) {
            DemoOutcome::Completed(result) => {
                assert_eq!(result.restart_step, 1);
                assert!((result.restart_time - 0.1).abs() < 1.0e-12);
                assert_eq!(result.restart_global_offset, 4);
                assert_eq!(result.restart_global_len, 8);
                assert_eq!(result.restart_values, expected_rank_values(1, 1));
                assert_eq!(result.latest_rank0_values, expected_rank_values(2, 0));
                assert_eq!(result.latest_rank1_values, expected_rank_values(2, 1));
                assert_eq!(result.latest_global_values, expected_global_values(2));
            }
            DemoOutcome::Hdf5Disabled => {}
            DemoOutcome::Hdf5MpiDisabled => panic!("partitioned backend should not report hdf5-mpi feature disabled"),
        }

        let _ = std::fs::remove_file(out_h5);
        let _ = std::fs::remove_file(out_xdmf);
    }

    #[test]
    fn ex43_mpi_backend_reports_feature_gate_or_succeeds() {
        let (out_h5, out_xdmf) = temp_paths("mpi");
        let args = Args {
            out_h5: out_h5.clone(),
            out_xdmf: out_xdmf.clone(),
            backend: IoBackend::MpiCollective,
            restart_step: None,
        };

        match run_checkpoint_demo(&args) {
            DemoOutcome::Completed(result) => {
                assert_eq!(result.layout_steps, 3);
                assert_eq!(result.layout_warnings, 0);
                assert_eq!(result.restart_values.len(), 4);
                assert_eq!(result.latest_global_values, expected_global_values(2));
            }
            DemoOutcome::Hdf5Disabled | DemoOutcome::Hdf5MpiDisabled => {}
        }

        let _ = std::fs::remove_file(out_h5);
        let _ = std::fs::remove_file(out_xdmf);
    }

    #[test]
    fn ex43_rank_field_generation_matches_partition_contract() {
        let rank0 = make_rank_field(0, 8, 0.2);
        let rank1 = make_rank_field(1, 8, 0.2);

        assert_eq!(rank0.global_offset, 0);
        assert_eq!(rank1.global_offset, 4);
        assert_eq!(rank0.global_len, 8);
        assert_eq!(rank1.global_len, 8);
        assert_eq!(rank0.values, expected_rank_values(2, 0));
        assert_eq!(rank1.values, expected_rank_values(2, 1));
        assert_eq!(
            rank0.values
                .iter()
                .chain(rank1.values.iter())
                .copied()
                .collect::<Vec<_>>(),
            expected_global_values(2)
        );
    }
}

