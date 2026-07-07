//! Integration tests for the in-process [`ThreadLauncher`] and channel backend.
//!
//! These tests exercise the full parallel stack — barrier, allreduce,
//! point-to-point, alltoallv, ghost exchange — using OS threads without
//! requiring an MPI installation.

use std::sync::{Arc, Mutex};

use fem_parallel::{
    launcher::native::ThreadLauncher,
    metis::{MetisOptions, partition_simplex_metis, partition_simplex_metis_streaming},
    par_simplex::{partition_simplex, partition_simplex_streaming},
    GhostExchange,
    WorkerConfig,
};
use fem_mesh::{ElementType, Mesh};

// ── helpers ───────────────────────────────────────────────────────────────────

/// One `Prism6` with Tri3 caps + Quad4 sides (mixed `face_offsets`).
fn unit_prism_mixed_boundary() -> Mesh<3> {
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
    ];
    let conn = vec![0u32, 1, 2, 3, 4, 5];
    let elem_tags = vec![1i32];
    let face_conn: Vec<u32> = vec![
        0, 1, 2,
        3, 4, 5,
        0, 1, 4, 3,
        1, 2, 5, 4,
        2, 0, 3, 5,
    ];
    let face_tags = vec![1i32, 2, 3, 3, 3];
    let face_types = vec![
        ElementType::Tri3,
        ElementType::Tri3,
        ElementType::Quad4,
        ElementType::Quad4,
        ElementType::Quad4,
    ];
    let face_offsets = vec![0usize, 3, 6, 10, 14, 18];
    let mut m = Mesh::uniform(
        coords,
        conn,
        elem_tags,
        ElementType::Prism6,
        face_conn,
        face_tags,
        ElementType::Tri3,
    );
    m.face_types = Some(face_types);
    m.face_offsets = Some(face_offsets);
    m
}

/// Two disjoint prisms (12 nodes, 2 elems, 10 boundary faces, mixed offsets).
fn two_disjoint_prisms_mixed_boundary() -> Mesh<3> {
    let o = 6u32;
    let coords: Vec<f64> = vec![
        // prism 0
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        // prism 1 (translated in +x)
        3.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
        3.0, 1.0, 0.0,
        3.0, 0.0, 1.0,
        4.0, 0.0, 1.0,
        3.0, 1.0, 1.0,
    ];
    let conn: Vec<u32> = vec![0, 1, 2, 3, 4, 5, o, o + 1, o + 2, o + 3, o + 4, o + 5];
    let elem_tags = vec![1i32, 1];
    let mut face_conn: Vec<u32> = vec![
        0, 1, 2,
        3, 4, 5,
        0, 1, 4, 3,
        1, 2, 5, 4,
        2, 0, 3, 5,
    ];
    // second prism faces (node ids +6)
    face_conn.extend_from_slice(&[
        o, o + 1, o + 2,
        o + 3, o + 4, o + 5,
        o, o + 1, o + 4, o + 3,
        o + 1, o + 2, o + 5, o + 4,
        o + 2, o, o + 3, o + 5,
    ]);
    let face_types: Vec<ElementType> = vec![
        ElementType::Tri3,
        ElementType::Tri3,
        ElementType::Quad4,
        ElementType::Quad4,
        ElementType::Quad4,
        ElementType::Tri3,
        ElementType::Tri3,
        ElementType::Quad4,
        ElementType::Quad4,
        ElementType::Quad4,
    ];
    let face_tags = vec![1, 2, 3, 3, 3, 1, 2, 3, 3, 3];
    let face_offsets: Vec<usize> = vec![
        0, 3, 6, 10, 14, 18, 21, 24, 28, 32, 36,
    ];
    let mut m = Mesh::uniform(
        coords,
        conn,
        elem_tags,
        ElementType::Prism6,
        face_conn,
        face_tags,
        ElementType::Tri3,
    );
    m.face_types = Some(face_types);
    m.face_offsets = Some(face_offsets);
    m
}

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
    let mesh = Mesh::<2>::unit_square_tri(4); // 4×4 grid → 32 triangles
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
    let mesh    = Mesh::<2>::unit_square_tri(4);
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

// ── Comm::split ─────────────────────────────────────────────────────────────

#[test]
fn comm_split_even_odd() {
    // Split 4 ranks into even (0,2) and odd (1,3) sub-communicators.
    let results = Arc::new(Mutex::new(Vec::new()));

    let l = launcher(4);
    let results_clone = Arc::clone(&results);
    l.launch(move |comm| {
        let rank = comm.rank();
        let color = rank % 2; // 0=even, 1=odd
        let key = rank;       // preserve ordering

        let sub_comm = comm.split(color, key);

        let sub_rank = sub_comm.rank();
        let sub_size = sub_comm.size();

        results_clone.lock().unwrap().push((rank, color, sub_rank, sub_size));

        // Test allreduce within sub-communicator.
        let sum = sub_comm.allreduce_sum_i64(1);
        assert_eq!(sum, 2, "rank {rank}: sub-allreduce should sum to 2 (2 ranks per group)");

        // Test that sub-communicator ranks are correct.
        assert_eq!(sub_size, 2, "rank {rank}: sub-comm should have 2 ranks");
    });

    let mut res = results.lock().unwrap().clone();
    res.sort_by_key(|(r, _, _, _)| *r);

    // rank 0: color=0, sub_rank=0, sub_size=2
    assert_eq!(res[0], (0, 0, 0, 2));
    // rank 1: color=1, sub_rank=0, sub_size=2
    assert_eq!(res[1], (1, 1, 0, 2));
    // rank 2: color=0, sub_rank=1, sub_size=2
    assert_eq!(res[2], (2, 0, 1, 2));
    // rank 3: color=1, sub_rank=1, sub_size=2
    assert_eq!(res[3], (3, 1, 1, 2));
}

#[test]
fn comm_split_single_group() {
    // All ranks same color → sub-communicator = original.
    let l = launcher(3);
    l.launch(move |comm| {
        let sub_comm = comm.split(0, comm.rank());
        assert_eq!(sub_comm.size(), 3);
        assert_eq!(sub_comm.rank(), comm.rank());

        let sum = sub_comm.allreduce_sum_f64(1.0);
        assert!((sum - 3.0).abs() < 1e-14);
    });
}

// ── streaming partition ─────────────────────────────────────────────────────

#[test]
fn streaming_single_rank() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    launcher(1).launch(move |comm| {
        let pmesh = partition_simplex_streaming(Some(&mesh), &comm)
            .expect("streaming partition failed");
        assert_eq!(pmesh.global_n_nodes(), mesh.n_nodes());
        assert_eq!(pmesh.global_n_elems(), mesh.n_elems());
        assert_eq!(pmesh.n_ghost_nodes(), 0);
    });
}

#[test]
fn streaming_matches_replicated_2_ranks() {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
    let mesh2 = Arc::clone(&mesh);

    // Replicated partition for reference.
    let replicated_nodes = Arc::new(Mutex::new(Vec::new()));
    let replicated_elems = Arc::new(Mutex::new(Vec::new()));
    let rn = Arc::clone(&replicated_nodes);
    let re = Arc::clone(&replicated_elems);
    let mesh_ref = Arc::clone(&mesh);
    launcher(2).launch(move |comm| {
        let pmesh = partition_simplex(&mesh_ref, &comm);
        rn.lock().unwrap().push((comm.rank(), pmesh.n_owned_nodes(), pmesh.n_ghost_nodes()));
        re.lock().unwrap().push((comm.rank(), pmesh.global_n_nodes(), pmesh.global_n_elems()));
    });

    // Streaming partition.
    let streaming_nodes = Arc::new(Mutex::new(Vec::new()));
    let streaming_elems = Arc::new(Mutex::new(Vec::new()));
    let sn = Arc::clone(&streaming_nodes);
    let se = Arc::clone(&streaming_elems);
    launcher(2).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh2) } else { None };
        let pmesh = partition_simplex_streaming(mesh_opt, &comm)
            .expect("streaming partition failed");
        sn.lock().unwrap().push((comm.rank(), pmesh.n_owned_nodes(), pmesh.n_ghost_nodes()));
        se.lock().unwrap().push((comm.rank(), pmesh.global_n_nodes(), pmesh.global_n_elems()));
    });

    let mut rn = replicated_nodes.lock().unwrap().clone();
    let mut sn = streaming_nodes.lock().unwrap().clone();
    rn.sort_by_key(|t| t.0);
    sn.sort_by_key(|t| t.0);
    assert_eq!(rn, sn, "owned/ghost node counts must match between replicated and streaming");

    let mut re = replicated_elems.lock().unwrap().clone();
    let mut se = streaming_elems.lock().unwrap().clone();
    re.sort_by_key(|t| t.0);
    se.sort_by_key(|t| t.0);
    assert_eq!(re, se, "global node/elem counts must match");
}

#[test]
fn streaming_ghost_exchange_after_partition() {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(4));
    launcher(2).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
        let pmesh = partition_simplex_streaming(mesh_opt, &comm)
            .expect("streaming partition failed");

        // Set owned nodes to their global ID, ghosts to -1.
        let n_total = pmesh.n_total_nodes();
        let mut data = vec![-1.0_f64; n_total];
        for lid in 0..pmesh.n_owned_nodes() {
            data[lid] = pmesh.global_node_id(lid as u32) as f64;
        }

        // Forward exchange should fill ghost slots.
        pmesh.forward_exchange(&mut data);

        for lid in 0..n_total {
            let expected = pmesh.global_node_id(lid as u32) as f64;
            assert!(
                (data[lid] - expected).abs() < 1e-12,
                "rank {}: data[{lid}] = {}, expected {} (global={})",
                comm.rank(), data[lid], expected, pmesh.global_node_id(lid as u32),
            );
        }
    });
}

#[test]
fn streaming_4_ranks() {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
    let total_elems = mesh.n_elems();
    let total_nodes = mesh.n_nodes();

    let results = Arc::new(Mutex::new(Vec::new()));
    let res = Arc::clone(&results);
    launcher(4).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
        let pmesh = partition_simplex_streaming(mesh_opt, &comm)
            .expect("streaming partition failed");

        assert_eq!(pmesh.global_n_elems(), total_elems);
        assert_eq!(pmesh.global_n_nodes(), total_nodes);

        pmesh.local_mesh().check().expect("local mesh check failed");

        res.lock().unwrap().push(comm.rank());
    });

    let mut r = results.lock().unwrap().clone();
    r.sort();
    assert_eq!(r, vec![0, 1, 2, 3]);
}

#[test]
fn streaming_prism_mixed_boundary_single_rank() {
    let mesh = Arc::new(unit_prism_mixed_boundary());
    mesh.check().expect("fixture prism");
    launcher(1).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
        let pmesh = partition_simplex_streaming(mesh_opt, &comm)
            .expect("streaming partition failed");
        assert_eq!(pmesh.global_n_nodes(), mesh.n_nodes());
        assert_eq!(pmesh.global_n_elems(), mesh.n_elems());
        let local = pmesh.local_mesh();
        local.check().expect("local mesh");
        assert_eq!(local.n_faces(), 5, "prism has 5 boundary faces");
        assert!(local.face_offsets.is_some(), "mixed boundary must round-trip");
    });
}

#[test]
fn streaming_two_prisms_matches_replicated_2_ranks() {
    let mesh = Arc::new(two_disjoint_prisms_mixed_boundary());
    mesh.check().expect("fixture two prisms");
    let mesh2 = Arc::clone(&mesh);

    let replicated = Arc::new(Mutex::new(Vec::new()));
    let rep = Arc::clone(&replicated);
    let mesh_ref = Arc::clone(&mesh);
    launcher(2).launch(move |comm| {
        let pmesh = partition_simplex(&mesh_ref, &comm);
        rep.lock().unwrap().push((
            comm.rank(),
            pmesh.n_owned_nodes(),
            pmesh.n_ghost_nodes(),
            pmesh.local_mesh().n_faces(),
        ));
    });

    let streaming = Arc::new(Mutex::new(Vec::new()));
    let stm = Arc::clone(&streaming);
    launcher(2).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh2) } else { None };
        let pmesh = partition_simplex_streaming(mesh_opt, &comm)
            .expect("streaming partition failed");
        stm.lock().unwrap().push((
            comm.rank(),
            pmesh.n_owned_nodes(),
            pmesh.n_ghost_nodes(),
            pmesh.local_mesh().n_faces(),
        ));
    });

    let mut a = replicated.lock().unwrap().clone();
    let mut b = streaming.lock().unwrap().clone();
    a.sort_by_key(|t| t.0);
    b.sort_by_key(|t| t.0);
    assert_eq!(a, b, "replicated vs streaming: per-rank nodes/ghosts/local faces");
    let sum_faces: usize = a.iter().map(|t| t.3).sum();
    assert_eq!(sum_faces, 10, "each prism contributes 5 boundary faces");
}

// ── METIS streaming partition ───────────────────────────────────────────────

#[test]
fn metis_streaming_2_ranks() {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
    let total_elems = mesh.n_elems();
    let total_nodes = mesh.n_nodes();
    let opts = MetisOptions::default();

    launcher(2).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
        let pmesh = partition_simplex_metis_streaming(mesh_opt, &comm, &opts)
            .expect("METIS streaming partition failed");

        assert_eq!(pmesh.global_n_elems(), total_elems);
        assert_eq!(pmesh.global_n_nodes(), total_nodes);
        pmesh.local_mesh().check().expect("local mesh check failed");
    });
}

#[test]
fn metis_streaming_4_ranks() {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
    let total_elems = mesh.n_elems();
    let total_nodes = mesh.n_nodes();
    let opts = MetisOptions::default();

    let results = Arc::new(Mutex::new(Vec::new()));
    let res = Arc::clone(&results);
    launcher(4).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
        let pmesh = partition_simplex_metis_streaming(mesh_opt, &comm, &opts)
            .expect("METIS streaming partition failed");

        assert_eq!(pmesh.global_n_elems(), total_elems);
        assert_eq!(pmesh.global_n_nodes(), total_nodes);
        pmesh.local_mesh().check().expect("local mesh check failed");

        res.lock().unwrap().push(comm.rank());
    });

    let mut r = results.lock().unwrap().clone();
    r.sort();
    assert_eq!(r, vec![0, 1, 2, 3]);
}

#[test]
fn metis_streaming_ghost_exchange() {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
    let opts = MetisOptions::default();

    launcher(2).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh) } else { None };
        let pmesh = partition_simplex_metis_streaming(mesh_opt, &comm, &opts)
            .expect("METIS streaming partition failed");

        let n_total = pmesh.n_total_nodes();
        let mut data = vec![-1.0_f64; n_total];
        for lid in 0..pmesh.n_owned_nodes() {
            data[lid] = pmesh.global_node_id(lid as u32) as f64;
        }

        pmesh.forward_exchange(&mut data);

        for lid in 0..n_total {
            let expected = pmesh.global_node_id(lid as u32) as f64;
            assert!(
                (data[lid] - expected).abs() < 1e-12,
                "rank {}: data[{lid}] = {}, expected {}",
                comm.rank(), data[lid], expected,
            );
        }
    });
}

#[test]
fn metis_streaming_two_disjoint_prisms_matches_replicated_2_ranks() {
    let mesh = Arc::new(two_disjoint_prisms_mixed_boundary());
    mesh.check().expect("fixture two prisms");
    let mesh2 = Arc::clone(&mesh);
    let opts_rep = MetisOptions::default();
    let opts_stm = MetisOptions::default();

    let replicated = Arc::new(Mutex::new(Vec::new()));
    let rep = Arc::clone(&replicated);
    let mesh_ref = Arc::clone(&mesh);
    launcher(2).launch(move |comm| {
        let pmesh = partition_simplex_metis(&mesh_ref, &comm, &opts_rep);
        rep.lock().unwrap().push((
            comm.rank(),
            pmesh.n_owned_nodes(),
            pmesh.n_ghost_nodes(),
            pmesh.local_mesh().n_faces(),
        ));
    });

    let streaming = Arc::new(Mutex::new(Vec::new()));
    let stm = Arc::clone(&streaming);
    launcher(2).launch(move |comm| {
        let mesh_opt = if comm.is_root() { Some(&*mesh2) } else { None };
        let pmesh = partition_simplex_metis_streaming(mesh_opt, &comm, &opts_stm)
            .expect("METIS streaming partition failed");
        stm.lock().unwrap().push((
            comm.rank(),
            pmesh.n_owned_nodes(),
            pmesh.n_ghost_nodes(),
            pmesh.local_mesh().n_faces(),
        ));
    });

    let mut a = replicated.lock().unwrap().clone();
    let mut b = streaming.lock().unwrap().clone();
    a.sort_by_key(|t| t.0);
    b.sort_by_key(|t| t.0);
    assert_eq!(
        a, b,
        "METIS replicated vs streaming: per-rank nodes/ghosts/local faces"
    );
    let sum_faces: usize = a.iter().map(|t| t.3).sum();
    assert_eq!(sum_faces, 10, "each prism contributes 5 boundary faces");
}
