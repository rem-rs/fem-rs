//! Driver-style checkpoint: one OS process calls `IoBackend::MpiCollective` once per
//! logical rank while the MPI world size is **1**, so the crate falls back to
//! partitioned writes (no ordered MPI section). Ensures the MPI feature does not
//! require `mpiexec -n N` matching `world_size` for this common test / demo pattern.
//!
//! If the test binary is launched with `mpiexec -n 2+`, each process would run the
//! full loop and corrupt the file — skip when MPI world size is not 1.

use fem_io_hdf5_parallel::{
    IoBackend, ParallelIoConfig, RankFieldF64, materialize_global_field_f64, read_global_field_f64,
    write_checkpoint_step_f64_with_backend,
};

#[test]
fn mpi_collective_serial_driver_two_logical_ranks_one_mpi_process() {
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::Communicator;
    if SimpleCommunicator::world().size() != 1 {
        return;
    }

    let world_size = 2usize;
    let mut path = std::env::temp_dir();
    path.push(format!(
        "fem_io_mpi_fallback_{}_{}.h5",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let file_path = path.to_string_lossy().to_string();
    let _ = std::fs::remove_file(&file_path);

    for rank in 0..world_size {
        let cfg = ParallelIoConfig { world_size, rank };
        let field = RankFieldF64 {
            name: "u".into(),
            global_offset: (rank * 3) as u64,
            global_len: 6,
            values: vec![rank as f64 + 10.0, rank as f64 + 11.0, rank as f64 + 12.0],
        };
        write_checkpoint_step_f64_with_backend(
            &file_path,
            cfg,
            0,
            0.0,
            &[field],
            IoBackend::MpiCollective,
        )
        .expect("checkpoint write with MPI backend + mismatched world should succeed");
    }

    materialize_global_field_f64(&file_path, world_size, 0, "u").expect("materialize");
    let g = read_global_field_f64(&file_path, 0, "u").expect("read global");
    assert_eq!(g, vec![10.0, 11.0, 12.0, 11.0, 12.0, 13.0]);

    let _ = std::fs::remove_file(&file_path);
}
