//! Integration tests for the in-process [`ThreadLauncher`] and channel backend.
//!
//! These tests exercise the full parallel stack — barrier, allreduce,
//! point-to-point, alltoallv, ghost exchange — using OS threads without
//! requiring an MPI installation.

use std::sync::{Arc, Mutex};

use fem_parallel::{
    launcher::native::ThreadLauncher,
    par_simplex::partition_simplex,
    GhostExchange,
    WorkerConfig,
};
use fem_mesh::SimplexMesh;

// ── helpers ───────────────────────────────────────────────────────────────────

fn launcher(n: usize) -> ThreadLauncher {
    ThreadLauncher::new(WorkerConfig::new(n))
}

// ── basic topology ────────────────────────────────────────────────────────────

#[test]
fn thread_single_worker_rank() {
    launcher(1).launch(|comm| {
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.size(), 1);
    });
}

#[test]
fn thread_multi_worker_ranks() {
    let ranks_seen = Arc::new(Mutex::new(Vec::new()));
    let ranks_seen2 = Arc::clone(&ranks_seen);
    launcher(4).launch(move |comm| {
        let mut guard = ranks_seen2.lock().unwrap();
        guard.push((comm.rank(), comm.size()));
    });
    let mut seen = ranks_seen.lock().unwrap();
    seen.sort();
    assert_eq!(seen.len(), 4);
    for (i, &(rank, size)) in seen.iter().enumerate() {
        assert_eq!(rank as usize, i);
        assert_eq!(size, 4);
    }
}

// ── barrier ───────────────────────────────────────────────────────────────────

#[test]
fn thread_barrier_does_not_deadlock() {
    launcher(4).launch(|comm| {
        comm.barrier();
        comm.barrier(); // second call to verify generation counting
    });
}

// ── allreduce ─────────────────────────────────────────────────────────────────

#[test]
fn thread_allreduce_sum_f64() {
    // 4 ranks each contribute (rank + 1.0): sum = 1+2+3+4 = 10
    launcher(4).launch(|comm| {
        let local  = comm.rank() as f64 + 1.0;
        let global = comm.allreduce_sum_f64(local);
        assert!((global - 10.0).abs() < 1e-12, "expected sum=10, got {global}");
    });
}

#[test]
fn thread_allreduce_sum_i64() {
    // 8 ranks: sum of 0..7 = 28
    launcher(8).launch(|comm| {
        let local  = comm.rank() as i64;
        let global = comm.allreduce_sum_i64(local);
        assert_eq!(global, 28);
    });
}

#[test]
fn thread_allreduce_repeated() {
    // Verify generation tracking allows back-to-back allreduce calls.
    launcher(4).launch(|comm| {
        for round in 0..5_i64 {
            let local  = comm.rank() as i64;
            let global = comm.allreduce_sum_i64(local);
            // 0+1+2+3 = 6 every round
            assert_eq!(global, 6, "round {round}: expected 6, got {global}");
        }
    });
}

// ── broadcast ────────────────────────────────────────────────────────────────

#[test]
fn thread_broadcast_bytes() {
    launcher(4).launch(|comm| {
        let mut buf = if comm.rank() == 0 {
            b"hello_fem".to_vec()
        } else {
            vec![]
        };
        comm.broadcast_bytes(0, &mut buf);
        assert_eq!(buf, b"hello_fem".to_vec(),
            "rank {} got wrong broadcast", comm.rank());
    });
}

// ── point-to-point ────────────────────────────────────────────────────────────

#[test]
fn thread_send_recv_ring() {
    // Each rank sends its rank to the next (mod n).
    launcher(4).launch(|comm| {
        let n    = comm.size() as i32;
        let rank = comm.rank();
        let next = ((rank + 1) % n) as fem_core::Rank;
        let prev = ((rank + n - 1) % n) as fem_core::Rank;

        // Post send first (async in channel backend).
        comm.send_bytes(next, 42, &(rank as u32).to_le_bytes());
        let recv = comm.recv_bytes(prev, 42);
        let val  = u32::from_le_bytes(recv.try_into().unwrap());
        assert_eq!(val as i32, (rank + n - 1) % n,
            "rank {rank}: expected {}, got {val}", (rank + n - 1) % n);
    });
}

// ── alltoallv ─────────────────────────────────────────────────────────────────

#[test]
fn thread_alltoallv_full() {
    // Each rank sends (rank, b"from_{rank}") to every other rank.
    launcher(4).launch(|comm| {
        use fem_core::Rank;
        let n    = comm.size();
        let rank = comm.rank();

        let sends: Vec<(Rank, Vec<u8>)> = (0..n)
            .filter(|&d| d as i32 != rank)
            .map(|d| (d as Rank, format!("from_{rank}").into_bytes()))
            .collect();

        let recv = comm.alltoallv_bytes(&sends);
        assert_eq!(recv.len(), n - 1, "rank {rank}: expected {} msgs", n - 1);

        for (src, data) in &recv {
            let expected = format!("from_{src}");
            assert_eq!(
                data, expected.as_bytes(),
                "rank {rank}: bad payload from {src}",
            );
        }
    });
}

#[test]
fn thread_alltoallv_sparse() {
    // Only rank 0 sends to rank 2; all others send nothing.
    launcher(4).launch(|comm| {
        use fem_core::Rank;
        let rank = comm.rank();

        let sends: Vec<(Rank, Vec<u8>)> = if rank == 0 {
            vec![(2, b"ping".to_vec())]
        } else {
            vec![]
        };

        let recv = comm.alltoallv_bytes(&sends);

        if rank == 2 {
            assert_eq!(recv.len(), 1);
            assert_eq!(recv[0], (0, b"ping".to_vec()));
        } else {
            assert!(recv.is_empty(), "rank {rank}: expected no messages");
        }
    });
}

// ── ghost exchange ────────────────────────────────────────────────────────────

/// Partition a 2×2 unit-square mesh (32 triangles, 4 ranks) and verify that
/// ghost exchange correctly propagates owned-node values to ghost copies.
#[test]
fn ghost_exchange_forward_2d() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4); // 4×4 grid → 32 triangles
    let n_total_nodes = mesh.n_nodes();

    // Shared storage so every thread can write its owned values and the test
    // thread can verify them after join.
    let results = Arc::new(Mutex::new(Vec::<(i32 /*rank*/, bool /*ok*/)>::new()));
    let results2 = Arc::clone(&results);
    let mesh_arc = Arc::new(mesh);

    launcher(4).launch(move |comm| {
        let par_mesh = partition_simplex(&mesh_arc, &comm);
        let partition = par_mesh.partition();
        let exchange  = GhostExchange::from_partition(partition, &comm);

        // Initialise data: owned nodes get value = global_node_id, ghosts = -1.
        let mut data = vec![-1.0_f64; partition.n_total_nodes()];
        for lid in 0..partition.n_owned_nodes as u32 {
            data[lid as usize] = partition.global_node(lid) as f64;
        }

        // Forward: propagate owned values to ghosts on neighbours.
        exchange.forward(&comm, &mut data);

        // After forward, every ghost should equal the global node ID.
        let mut ok = true;
        for (lid, _owner) in partition.ghost_nodes() {
            let gid      = partition.global_node(lid);
            let expected = gid as f64;
            if (data[lid as usize] - expected).abs() > 1e-12 {
                ok = false;
            }
        }

        // Sanity: no value should still be -1 (including ghosts).
        for v in &data {
            if *v < -0.5 { ok = false; }
        }

        let _ = n_total_nodes; // suppress unused warning
        results2.lock().unwrap().push((comm.rank(), ok));
    });

    let res = results.lock().unwrap();
    assert_eq!(res.len(), 4, "expected 4 thread results");
    for &(rank, ok) in res.iter() {
        assert!(ok, "rank {rank}: ghost forward exchange failed");
    }
}

/// Verify that reverse exchange accumulates ghost contributions to owned slots.
#[test]
fn ghost_exchange_reverse_2d() {
    let mesh    = SimplexMesh::<2>::unit_square_tri(4);
    let results = Arc::new(Mutex::new(Vec::<(i32, bool)>::new()));
    let results2 = Arc::clone(&results);
    let mesh_arc = Arc::new(mesh);

    launcher(4).launch(move |comm| {
        let par_mesh  = partition_simplex(&mesh_arc, &comm);
        let partition = par_mesh.partition();
        let exchange  = GhostExchange::from_partition(partition, &comm);

        // Every node (owned + ghost) gets value 1.0 — simulates an assembly
        // where ghost contributions have been accumulated locally.
        let mut data = vec![1.0_f64; partition.n_total_nodes()];

        // Reverse: sum ghost contributions into owned nodes.
        exchange.reverse(&comm, &mut data);

        // After reverse:
        // - Ghost slots must be 0 (zeroed by reverse).
        // - Owned slots must be ≥ 1 (accumulated contributions).
        let mut ok = true;
        for (lid, _owner) in partition.ghost_nodes() {
            if data[lid as usize].abs() > 1e-12 { ok = false; }
        }
        for lid in 0..partition.n_owned_nodes as u32 {
            if data[lid as usize] < 1.0 - 1e-12 { ok = false; }
        }

        results2.lock().unwrap().push((comm.rank(), ok));
    });

    let res = results.lock().unwrap();
    assert_eq!(res.len(), 4);
    for &(rank, ok) in res.iter() {
        assert!(ok, "rank {rank}: ghost reverse exchange failed");
    }
}
