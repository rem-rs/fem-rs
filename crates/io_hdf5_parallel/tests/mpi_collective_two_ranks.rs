//! Two **MPI processes** (`mpiexec -n 2`), each with matching `ParallelIoConfig`, write one
//! step via `IoBackend::MpiCollective` (ordered `rust-hdf5` access + rank-0 materialization).
//!
//! Under plain `cargo test` (world size 1) this test returns immediately. To exercise it:
//! `mpiexec -n 2 cargo test -p fem-io-hdf5-parallel --features hdf5-mpi --test mpi_collective_two_ranks`

use fem_io_hdf5_parallel::{
    IoBackend, ParallelIoConfig, RankFieldF64, read_checkpoint_field_f64_latest, read_global_field_f64,
    write_checkpoint_step_f64_with_backend,
};
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};

#[test]
fn two_mpi_ranks_mpi_collective_single_step_global_matches() {
    let world = SimpleCommunicator::world();
    if world.size() != 2 {
        return;
    }

    let rank = world.rank() as usize;
    let world_size = 2usize;

    let local_tag = std::process::id() as i64 + 1_000_000_000 * (rank as i64);
    let mut file_tag = 0_i64;
    world.all_reduce_into(&local_tag, &mut file_tag, &SystemOperation::sum());

    let mut path = std::env::temp_dir();
    path.push(format!("fem_io_mpi_collective_2r_{file_tag}.h5"));
    let file_path = path.to_string_lossy().to_string();

    if rank == 0 {
        let _ = std::fs::remove_file(&file_path);
    }
    world.barrier();

    let field = RankFieldF64 {
        name: "u".into(),
        global_offset: if rank == 0 { 0 } else { 4 },
        global_len: 8,
        values: if rank == 0 {
            vec![10.0, 11.0, 12.0, 13.0]
        } else {
            vec![14.0, 15.0, 16.0, 17.0]
        },
    };

    let cfg = ParallelIoConfig { world_size, rank };
    write_checkpoint_step_f64_with_backend(
        &file_path,
        cfg,
        0,
        0.0,
        &[field],
        IoBackend::MpiCollective,
    )
    .expect("MPI collective checkpoint write");

    world.barrier();

    if rank == 0 {
        let g = read_global_field_f64(&file_path, 0, "u").expect("read global u after materialize");
        assert_eq!(
            g,
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "rank 0 global field"
        );
    }

    let read = read_checkpoint_field_f64_latest(
        &file_path,
        ParallelIoConfig { world_size, rank },
        "u",
    )
    .expect("rank-local latest read");
    assert_eq!(read.global_len, 8);
    if rank == 0 {
        assert_eq!(read.values, vec![10.0, 11.0, 12.0, 13.0]);
    } else {
        assert_eq!(read.values, vec![14.0, 15.0, 16.0, 17.0]);
    }

    world.barrier();
    if rank == 0 {
        let _ = std::fs::remove_file(&file_path);
    }
}
