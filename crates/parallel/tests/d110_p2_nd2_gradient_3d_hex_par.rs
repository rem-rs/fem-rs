//! D110 (parallel): `ParDiscreteLinearOperator::gradient` for `H¹(P2) →
//! H(curl) ND2` on 3-D hexahedra.
//!
//! The serial entry point gained its 3-D hex path with D110; this pins the
//! parallel wrapper's contract — owned H(curl) rows × all local (owned +
//! ghost) H¹ columns, with the H(curl) partition's by-DOF sign corrections —
//! against the serial gradient assembled on the same local mesh, permuted
//! with both partitions and truncated to the owned rows.
//!
//! The single-rank case is the runnable one.  The multi-rank case is
//! `#[ignore]`d because it hits a **pre-existing** defect that is independent
//! of the gradient: at ≥ 2 ranks the parallel Nédélec `ND2` (and `RT1`)
//! DOF partition asks the neighbour for the *face/interior* DOF ids of ghost
//! elements, and the ghost-interior exchange cannot resolve them, so it
//! emits
//!
//! ```text
//! Warning: exchange_ghost_interior_ids: rank 1 requested interior DOF
//!          (elem=16, idx=24) not found, using sentinel GID
//! ```
//!
//! and then aborts in
//! `GhostExchange::from_partition` ("requested global node 4294967295 but
//! this rank does not own it", `crates/parallel/src/ghost.rs:178`) for
//! smaller meshes.  The same warnings already appear when the *joule*
//! miniapp builds its four spaces at `--ranks 2/3/4` (no gradient involved),
//! so the fix belongs in the DOF-partition / ghost-exchange layer, not here.

use fem_assembly::DiscreteLinearOperator;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::launcher::WorkerConfig;
use fem_parallel::par_discrete_operator::ParDiscreteLinearOperator;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_space::fe_space::FESpace as _;
use fem_space::{H1Space, HCurlSpace};
use std::sync::{Arc, Mutex};

/// One rank's observations: `(rank, owned H(curl) rows checked, max deviation)`.
type RankReport = (i32, usize, f64);

fn run_at_ranks(mesh: Mesh<3>, n_ranks: usize) -> Vec<RankReport> {
    let reports: Arc<Mutex<Vec<RankReport>>> = Arc::new(Mutex::new(Vec::new()));
    let reports_rank = Arc::clone(&reports);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        // Same constructors as `miniapps/electromagnetics/joule.rs`.
        let h1_par = ParallelFESpace::new(H1Space::new(lm.clone(), 2), &pmesh, comm.clone());
        let nd_par = ParallelFESpace::new(HCurlSpace::new(lm.clone(), 2), &pmesh, comm.clone());

        let g = ParDiscreteLinearOperator::gradient(&h1_par, &nd_par);
        let h1p = h1_par.dof_partition();
        let ndp = nd_par.dof_partition();

        let shape_ok = g.nrows == ndp.n_owned_dofs && g.ncols == h1p.n_total_dofs();

        let loc = DiscreteLinearOperator::gradient(h1_par.local_space(), nd_par.local_space())
            .expect("serial P2→ND2 gradient on the local mesh");
        let n_nd_local = nd_par.local_space().n_dofs();
        let n_h1_local = h1_par.local_space().n_dofs();

        let mut max_dev = 0.0_f64;
        let mut checked = 0usize;
        if shape_ok && loc.nrows == n_nd_local && loc.ncols == n_h1_local {
            for d in 0..n_nd_local as u32 {
                let p = ndp.permute_dof(d) as usize;
                if p >= ndp.n_owned_dofs {
                    continue; // ghost H(curl) row — deliberately dropped
                }
                checked += 1;
                let sr = ndp.sign_correction(d);
                for c in 0..n_h1_local as u32 {
                    let q = h1p.permute_dof(c) as usize;
                    let want = loc.get(d as usize, c as usize) * sr;
                    max_dev = max_dev.max((want - g.get(p, q)).abs());
                }
            }
        } else {
            checked = usize::MAX; // poison the outside assertion
        }
        reports_rank.lock().unwrap().push((comm.rank(), checked, max_dev));
    });
    Arc::try_unwrap(reports)
        .expect("single owner after launch")
        .into_inner()
        .unwrap()
}

fn check_reports(rep: &[RankReport], n_ranks: usize) {
    assert_eq!(rep.len(), n_ranks, "{n_ranks}-rank launch must visit every rank");
    for &(rank, checked, dev) in rep {
        assert_ne!(
            checked,
            usize::MAX,
            "{n_ranks} ranks, rank {rank}: parallel gradient shape mismatch \
             (expected owned H(curl) rows × local H¹ columns)"
        );
        assert!(checked > 0, "{n_ranks} ranks, rank {rank}: no owned rows checked");
        assert!(
            dev < 1e-12,
            "{n_ranks} ranks, rank {rank}: owned-H(curl)-row parallel gradient vs \
             permuted local gradient max dev {dev:.3e}"
        );
    }
}

#[test]
fn par_p2_nd2_gradient_hex3d_matches_the_serial_gradient_at_one_rank() {
    let rep = run_at_ranks(Mesh::<3>::unit_cube_hex(2), 1);
    check_reports(&rep, 1);
}

/// Multi-rank: see the module docs — blocked by the pre-existing ND2/RT1
/// parallel DOF-partition defect in `crates/parallel` (sentinel ghost GIDs for
/// face/interior DOFs of ghost elements, then a `GhostExchange` panic).
#[test]
#[ignore = "pre-existing ND2/RT1 parallel DOF-partition defect at >=2 ranks \
            (exchange_ghost_interior_ids sentinel GIDs -> GhostExchange panic)"]
fn par_p2_nd2_gradient_hex3d_matches_the_serial_gradient_multi_rank() {
    for n_ranks in [2usize, 4] {
        let rep = run_at_ranks(Mesh::<3>::unit_cube_hex(4), n_ranks);
        check_reports(&rep, n_ranks);
    }
}
