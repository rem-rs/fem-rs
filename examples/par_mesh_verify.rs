//! `par_mesh_verify` — correctness verification for `ParallelMesh`.
//!
//! Exercises the full pipeline from serial mesh generation through distributed
//! partitioning to global reductions, checking every invariant that must hold
//! regardless of the number of MPI ranks.
//!
//! ## Running
//!
//! ```sh
//! # Serial (no MPI install needed):
//! cargo run --example par_mesh_verify
//!
//! # With MPI (requires `mpi` feature and an MPI installation):
//! cargo run --example par_mesh_verify --features fem-parallel/mpi
//! mpirun -n 4 target/debug/examples/par_mesh_verify
//! ```
//!
//! ## Verification checklist
//!
//! 1. **Global count invariant** — sum of local element counts == serial total
//! 2. **Node ownership invariant** — sum of owned node counts == serial total
//! 3. **Coordinate preservation** — each owned node's coordinates match serial
//! 4. **Connectivity preservation** — element connectivity consistent with serial
//! 5. **Boundary face completeness** — all 4 sides of the unit square present
//! 6. **Global reduction** — allreduce sum of x-coordinates == analytic value
//! 7. **Ghost exchange no-op** — trivial exchange leaves data unchanged (serial)
//! 8. **Local mesh validity** — `SimplexMesh::check()` passes on local mesh

use fem_mesh::{MeshTopology, SimplexMesh};
use fem_parallel::{
    launcher::{Launcher, native::MpiLauncher},
    partition_simplex, ParallelMesh,
};

fn main() {
    env_logger::init();

    // ── initialise ────────────────────────────────────────────────────────────
    let launcher = MpiLauncher::init().expect("MPI already initialised");
    let comm = launcher.world_comm();
    let rank = comm.rank();
    let size = comm.size();

    if rank == 0 {
        println!("par_mesh_verify: rank {} / {} processes", rank, size);
        println!("──────────────────────────────────────────────────────────────");
    }
    comm.barrier();

    // ── build serial mesh ─────────────────────────────────────────────────────
    let n = 8usize;   // n × n cell grid → 2n² triangles, (n+1)² nodes
    let serial_mesh = SimplexMesh::<2>::unit_square_tri(n);

    let expected_nodes = (n + 1) * (n + 1); // 81
    let expected_elems = 2 * n * n;          // 128
    let expected_faces = 4 * n;              // 32

    assert_eq!(serial_mesh.n_nodes(),    expected_nodes);
    assert_eq!(serial_mesh.n_elems(),    expected_elems);
    assert_eq!(serial_mesh.n_boundary_faces(), expected_faces);

    // ── partition ─────────────────────────────────────────────────────────────
    let pmesh: ParallelMesh<SimplexMesh<2>> = partition_simplex(&serial_mesh, &comm);

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 1: global element count
    // ─────────────────────────────────────────────────────────────────────────
    assert_eq!(
        pmesh.global_n_elems(), expected_elems,
        "[rank {rank}] global_n_elems mismatch"
    );

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 2: global node count (owned nodes sum to serial total)
    // ─────────────────────────────────────────────────────────────────────────
    assert_eq!(
        pmesh.global_n_nodes(), expected_nodes,
        "[rank {rank}] global_n_nodes mismatch"
    );

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 3: every local element count sums to the global total
    // ─────────────────────────────────────────────────────────────────────────
    let local_elems_i64 = pmesh.local_mesh().n_elements() as i64;
    let sum_elems = comm.allreduce_sum_i64(local_elems_i64) as usize;
    assert_eq!(
        sum_elems, expected_elems,
        "[rank {rank}] allreduce sum of local element counts != expected"
    );

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 4: owned node count sums to global total
    // ─────────────────────────────────────────────────────────────────────────
    let owned_i64 = pmesh.n_owned_nodes() as i64;
    let sum_owned = comm.allreduce_sum_i64(owned_i64) as usize;
    assert_eq!(
        sum_owned, expected_nodes,
        "[rank {rank}] sum of n_owned_nodes != expected_nodes"
    );

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 5: owned-node coordinates match serial mesh
    // ─────────────────────────────────────────────────────────────────────────
    for lid in 0..pmesh.n_owned_nodes() as u32 {
        let gid = pmesh.global_node_id(lid);
        let local_coords  = pmesh.node_coords(lid);
        let serial_coords = serial_mesh.node_coords(gid);
        for (d, (&lc, &sc)) in local_coords.iter().zip(serial_coords.iter()).enumerate() {
            assert!(
                (lc - sc).abs() < 1e-14,
                "[rank {rank}] coord dim {d} mismatch for local node {lid} \
                 (global {gid}): got {lc}, expected {sc}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 6: element connectivity consistent with serial (single-rank only)
    // ─────────────────────────────────────────────────────────────────────────
    if size == 1 {
        for le in 0..pmesh.local_mesh().n_elements() as u32 {
            let ge = pmesh.global_elem_id(le);
            let local_nodes  = pmesh.element_nodes(le);
            let serial_nodes = serial_mesh.element_nodes(ge);
            assert_eq!(
                local_nodes, serial_nodes,
                "[rank 0] connectivity mismatch for local elem {le} (global {ge})"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 7: boundary faces — all 4 sides of [0,1]² must appear (single-rank)
    // ─────────────────────────────────────────────────────────────────────────
    if size == 1 {
        let tags: std::collections::HashSet<i32> = (0..pmesh.local_mesh().n_boundary_faces() as u32)
            .map(|f| pmesh.face_tag(f))
            .collect();
        for expected_tag in [1, 2, 3, 4] {
            assert!(
                tags.contains(&expected_tag),
                "[rank 0] boundary tag {expected_tag} missing from local mesh"
            );
        }
        assert_eq!(
            pmesh.local_mesh().n_boundary_faces(), expected_faces,
            "[rank 0] boundary face count mismatch"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 8: global sum of x-coordinates via allreduce
    //
    //   unit_square_tri(n): nodes at (i/n, j/n), i,j ∈ [0,n].
    //   Sum of x over all nodes
    //     = Σ_{j=0}^{n} Σ_{i=0}^{n} i/n
    //     = (n+1) × Σ_{i=0}^{n} i/n
    //     = (n+1) × (1/n) × n(n+1)/2
    //     = (n+1)² / 2.
    //   For n = 8: 9² / 2 = 40.5.
    // ─────────────────────────────────────────────────────────────────────────
    let xs: Vec<f64> = (0..pmesh.n_total_nodes())
        .map(|lid| pmesh.node_coords(lid as u32)[0])
        .collect();
    let global_sum_x = pmesh.global_sum_owned(&xs);
    let expected_sum_x = (n + 1) as f64 * (n + 1) as f64 / 2.0;
    assert!(
        (global_sum_x - expected_sum_x).abs() < 1e-10,
        "[rank {rank}] global sum of x coords = {global_sum_x:.6}, \
         expected {expected_sum_x:.6}"
    );

    // Repeat for y-coordinates (same value by symmetry).
    let ys: Vec<f64> = (0..pmesh.n_total_nodes())
        .map(|lid| pmesh.node_coords(lid as u32)[1])
        .collect();
    let global_sum_y = pmesh.global_sum_owned(&ys);
    assert!(
        (global_sum_y - expected_sum_x).abs() < 1e-10,
        "[rank {rank}] global sum of y coords = {global_sum_y:.6}, \
         expected {expected_sum_x:.6}"
    );

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 9: ghost exchange is trivial on a single rank
    // ─────────────────────────────────────────────────────────────────────────
    if size == 1 {
        assert!(pmesh.ghost_exchange().is_trivial(),
            "[rank 0] ghost exchange should be trivial for single rank");

        let mut data: Vec<f64> = (0..pmesh.n_total_nodes()).map(|i| i as f64).collect();
        let before = data.clone();
        pmesh.forward_exchange(&mut data);
        assert_eq!(data, before,
            "[rank 0] forward_exchange mutated data (should be no-op on single rank)");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CHECK 10: local mesh passes internal consistency check
    // ─────────────────────────────────────────────────────────────────────────
    pmesh.local_mesh().check().unwrap_or_else(|e| {
        panic!("[rank {rank}] local mesh check() failed: {e}");
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────────────
    comm.barrier();
    if rank == 0 {
        println!();
        println!("Mesh: {}×{} triangular grid on [0,1]²", n, n);
        println!("  Serial: {} nodes, {} elements, {} boundary edges",
                 expected_nodes, expected_elems, expected_faces);
        println!("  Processes: {}", size);
    }

    // Per-rank report (in-order via barrier — approximate for demo).
    for r in 0..size as i32 {
        if rank == r {
            println!(
                "  Rank {:2}: {:4} owned nodes, {:2} ghost nodes, {:4} local elems, \
                 {:3} local bfaces",
                rank,
                pmesh.n_owned_nodes(),
                pmesh.n_ghost_nodes(),
                pmesh.local_mesh().n_elements(),
                pmesh.local_mesh().n_boundary_faces(),
            );
        }
        comm.barrier();
    }

    if rank == 0 {
        println!();
        println!("Global sums (via allreduce):");
        println!("  Σ x-coords = {:.6}  (expected {:.6})", global_sum_x, expected_sum_x);
        println!("  Σ y-coords = {:.6}  (expected {:.6})", global_sum_y, expected_sum_x);
        println!();
        println!("✓ all checks passed");
    }
}
