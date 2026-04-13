//! ex43_hdf5_checkpoint - parallel checkpoint/restart baseline demo.
//!
//! This example demonstrates rank-partitioned checkpoint writing, latest-step
//! restart read, and XDMF sidecar generation using fem-io-hdf5-parallel.

use fem_io_hdf5_parallel::{
    CheckpointBundleF64, CheckpointMeshMeta, Hdf5ParallelError, IoBackend, ParallelIoConfig,
    RankFieldF64, materialize_global_field_f64, read_checkpoint_field_f64_at_step,
    read_checkpoint_field_f64_latest, validate_checkpoint_layout, write_checkpoint_step_bundle_f64,
    write_xdmf_polyvertex_scalar_timeseries_sidecar,
};

fn main() {
    let args = parse_args();

    println!("=== ex43_hdf5_checkpoint (baseline) ===");
    println!("  out_h5={}, out_xdmf={}", args.out_h5, args.out_xdmf);
    println!("  backend={}", match args.backend { IoBackend::Partitioned => "partitioned", IoBackend::MpiCollective => "mpi" });
    println!("  restart_step={}", args.restart_step.map_or("latest".into(), |s| s.to_string()));

    let world_size = 2usize;
    let global_len = 8u64;

    // Synthetic transient field on 2 rank partitions.
    // rank 0 owns [0..4), rank 1 owns [4..8).
    let mesh_meta = CheckpointMeshMeta {
        dim: 1,
        n_vertices: global_len,
        n_elements: global_len.saturating_sub(1),
    };

    for step in 0..=2u64 {
        let t = 0.1 * step as f64;

        let rank0 = RankFieldF64 {
            name: "u".into(),
            global_offset: 0,
            global_len,
            values: (0..4).map(|i| (i as f64) + t).collect(),
        };
        let rank1 = RankFieldF64 {
            name: "u".into(),
            global_offset: 4,
            global_len,
            values: (4..8).map(|i| (i as f64) + t).collect(),
        };

        for (rank, field) in [(0usize, rank0), (1usize, rank1)] {
            let cfg = ParallelIoConfig { world_size, rank };
            let bundle = CheckpointBundleF64 {
                mesh_meta: Some(mesh_meta),
                fields: vec![field],
            };
            match write_checkpoint_step_bundle_f64(&args.out_h5, cfg, step, t, &bundle, args.backend) {
                Ok(()) => {}
                Err(Hdf5ParallelError::Hdf5FeatureDisabled) => {
                    println!("  HDF5 backend disabled (build without feature `hdf5`)");
                    println!("  To enable real checkpoint I/O: cargo run --example ex43_hdf5_checkpoint --features fem-io-hdf5-parallel/hdf5");
                    println!("  PASS (API fallback verified)");
                    return;
                }
                Err(Hdf5ParallelError::Hdf5MpiFeatureDisabled) => {
                    println!("  MPI HDF5 backend disabled (build without feature `hdf5-mpi`)");
                    println!("  Use partitioned mode or enable: cargo run --example ex43_hdf5_checkpoint --features fem-io-hdf5-parallel/hdf5-mpi -- --backend mpi");
                    println!("  PASS (MPI backend fallback verified)");
                    return;
                }
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
    println!(
        "  validation: schema={}, steps={}, warnings={}",
        report.schema_version,
        report.steps.len(),
        report.warnings.len()
    );

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

    println!(
        "  restart: step={}, time={:.3}, offset={}, len={}, local={:?}",
        restart.step, restart.time, restart.global_offset, restart.global_len, restart.values
    );

    assert_eq!(restart.step, 2);
    assert_eq!(restart.global_offset, 4);
    assert_eq!(restart.global_len, 8);
    assert_eq!(restart.values.len(), 4);

    println!("  PASS");
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
