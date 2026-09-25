//! D803/D786 (round 74): empty-rank robustness — np > n_elements.
//!
//! When the mesh has fewer elements than ranks, `partition_mesh` hands the
//! trailing ranks an **empty** local mesh.  Those ranks used to panic in the
//! space constructors (`HCurlSpace::new`: "mesh must contain at least one
//! element") and deadlock the launcher rendezvous.  The spaces must accept a
//! 0-element mesh (empty dof tables, no owned dofs) so an over-partitioned
//! run constructs, exchanges, and solves.

use fem_mesh::topology::MeshTopology as _;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::launcher::WorkerConfig;
use fem_parallel::par_partition::partition_mesh;
use fem_space::{HCurlSpace, H1Space};
use std::sync::{Arc, Mutex};

/// A 2-element hexahedron slab: fewer elements than the 4 ranks below.
fn two_hex_mesh() -> fem_mesh::Mesh<3> {
    let c: Vec<f64> = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0],
    ]
    .iter()
    .flat_map(|p| p.iter().copied())
    .collect();
    let hexes: [[u32; 8]; 2] = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 2, 5, 10, 11, 6],
    ];
    let e: Vec<u32> = hexes.iter().flat_map(|h| h.iter().copied()).collect();
    fem_mesh::Mesh::<3>::uniform(c, e, vec![1; 2], fem_mesh::ElementType::Hex8, vec![], vec![], fem_mesh::ElementType::Quad4)
}

#[test]
fn d803_spaces_construct_on_empty_ranks() {
    let mesh = two_hex_mesh();
    assert_eq!(mesh.n_elems(), 2, "fixture: 2 elements");

    let reports: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let reports_rank = Arc::clone(&reports);
    ThreadLauncher::new(WorkerConfig::new(4)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let report = match (
            lm.n_elems(),
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _h1 = H1Space::new(lm.clone(), 1);
                let _nd = HCurlSpace::new(lm.clone(), 1);
            })),
        ) {
            (ne, Ok(())) => format!("rank {}: ne={ne} spaces ok", comm.rank()),
            (ne, Err(_)) => format!("rank {}: ne={ne} SPACE CONSTRUCTION PANICKED", comm.rank()),
        };
        reports_rank.lock().unwrap().push(report);
        // The parallel wrappers too (partition plumbing + ghost tables on an
        // empty rank — D504 territory).
        let wrapped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let h1_par = fem_parallel::par_space::ParallelFESpace::new(
                H1Space::new(lm.clone(), 1), &pmesh, comm.clone());
            let nd_par = fem_parallel::par_space::ParallelFESpace::new(
                HCurlSpace::new(lm.clone(), 1), &pmesh, comm.clone());
            (h1_par.n_global_dofs(), nd_par.n_global_dofs())
        }));
        match wrapped {
            Ok((h1g, ndg)) => println!("rank {}: parallel wrappers ok (global {h1g}/{ndg})", comm.rank()),
            Err(_) => println!("rank {}: PARALLEL WRAPPER PANICKED", comm.rank()),
        }
    });
    let mut got = reports.lock().unwrap().clone();
    got.sort();
    for l in &got {
        println!("{l}");
    }
    assert_eq!(got.len(), 4, "all four ranks must report");
    let empty_ranks: Vec<&String> = got.iter().filter(|l| l.contains("ne=0")).collect();
    assert_eq!(empty_ranks.len(), 2, "two ranks get an empty partition");
    assert!(
        got.iter().all(|l| !l.contains("PANICKED")),
        "space construction must not panic on an empty local mesh"
    );
}
