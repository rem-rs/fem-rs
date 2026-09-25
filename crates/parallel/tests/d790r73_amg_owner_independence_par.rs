//! D790-1 (round 73): the AMG preconditioner's quality must not depend on the
//! **owner split** of the distributed operator.
//!
//! D122-1 replaced every rank's owned set with exactly the DOFs it holds (it
//! used to carry a ring of the neighbour's interior DOFs).  The AMG's
//! coarsening was *block-local* in those days: aggregates were built on owned
//! points only and the coarse levels were block-diagonal, so the interface
//! error component had no coarse representation at all — the ring had been
//! coupling the coarse blocks by accident.  The pex3 configuration measured
//! here went from **102 converged iterations to a 10000-iteration stall
//! (8.7e-8)** at np = 2 (`tmp/d122r73/final_runs.txt`, pre-fix binary of
//! 19:21).  The fix makes the coarsening ghost-aware in every mode (full-row
//! strength + cross-rank aggregate merging + coupled coarse levels), which is
//! bit-identical on one rank, so the np = 1 red lines (`-o 1 --ranks 1` =
//! 2.70053050699196e-2 / 87 it, `-o 2 --ranks 1` = 3.05558571614408e-4) did not
//! move.
//!
//! Fixture: `data/star.mesh` + 4 uniform refinements, ND1 (H(curl) order 1),
//! `curl curl + 1` with the manufactured pex3 source and non-homogeneous PEC
//! elimination — the pex3 default configuration, reproduced in-process so the
//! pin needs no example binary.  Measured with the fix: np = 1 → 87 it,
//! np = 2 → 95 it (`tmp/d122r73/amg_fix_pex3.txt`).

use std::sync::{Arc, Mutex};

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
use fem_mesh::amr::refine_uniform;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector::ParVector;
use fem_parallel::{
    par_solve_pcg_amg, ParAmgConfig, ParVectorAssembler, SmootherType, WorkerConfig,
};
use fem_space::HCurlSpace;
use fem_solver::SolverConfig;

const KAPPA: f64 = std::f64::consts::PI;

/// pex3's manufactured field `(sin κy, sin κx)`.
fn exact_e(x: &[f64]) -> [f64; 2] {
    [(KAPPA * x[1]).sin(), (KAPPA * x[0]).sin()]
}

/// pex3's source term: `(curl curl + 1)E = (1 + κ²)E` with `E` divergence-free.
struct Src;
impl VectorLinearIntegrator for Src {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
        let c = 1.0 + KAPPA * KAPPA;
        let fx = c * (KAPPA * qp.x_phys[1]).sin();
        let fy = c * (KAPPA * qp.x_phys[0]).sin();
        for i in 0..qp.n_dofs {
            fe[i] += qp.weight * (qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy);
        }
    }
}

/// `data/star.mesh` refined 4 times (pex3's default: `ref_levels = 2` serial +
/// 2 parallel levels folded into the serial pass).
fn pex3_mesh() -> Mesh<2> {
    let path = format!("{}/../../data/star.mesh", env!("CARGO_MANIFEST_DIR"));
    let base = fem_io::mfem::read_mfem_file(&path)
        .expect("data/star.mesh")
        .mesh2d
        .expect("star.mesh is 2-D");
    let mut m = base;
    for _ in 0..4 {
        m = refine_uniform(&m);
    }
    m
}

/// pex3's order<2 AMG configuration (round-67 baseline shape).
fn pex3_amg_cfg() -> ParAmgConfig {
    ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        n_pre_smooth: 2,
        n_post_smooth: 2,
        smoothed_prolongation: true,
        block_size: 1,
        use_global_aggregation: false,
        ..ParAmgConfig::default()
    }
}

/// One rank count: `(iterations, converged, ‖E_h − E‖_{L²})`.
fn solve_pex3(n_ranks: usize) -> (usize, bool, f64) {
    let mesh = pex3_mesh();
    let out: Arc<Mutex<(usize, bool, f64)>> = Arc::new(Mutex::new((0, false, 0.0)));
    let out_rank = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let ps = ParallelFESpace::new_for_edge_space(HCurlSpace::new(lm, 1), &pm, comm.clone());
        let dp = ps.dof_partition();

        let mut stiff = ParVectorAssembler::assemble_bilinear(
            &ps,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
            4,
        );
        let mut rhs = ParVectorAssembler::assemble_linear(&ps, &[&Src], 4);

        // MFEM `FormLinearSystem`: projected exact values on the essential
        // (PEC) true DOFs, DIAG_KEEP elimination.
        let x0 = ps.local_space().interpolate_vector(&|p| exact_e(p).to_vec());
        let x0_perm = {
            let n = dp.n_total_dofs();
            let mut out = vec![0.0; n];
            for (d, &v) in x0.as_slice().iter().enumerate() {
                out[dp.permute_dof(d as u32) as usize] = v * dp.sign_correction(d as u32);
            }
            out
        };
        let u0 = ParVector::from_local_raw(
            x0_perm,
            dp.n_owned_dofs,
            ps.dof_ghost_exchange_arc(),
            comm.clone(),
        );
        let mut owned_ess: Vec<(usize, f64)> = Vec::new();
        let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
        for &d in ps.essential_true_dofs(&[1]).iter() {
            let pid = dp.permute_dof(d) as usize;
            let v = u0.as_slice()[pid];
            if pid < dp.n_owned_dofs {
                owned_ess.push((pid, v));
            } else {
                ghost_ess.push((pid - dp.n_owned_dofs, v));
            }
        }
        for &(pid, v) in &owned_ess {
            stiff.apply_dirichlet_par_keep_diag(pid, v, &mut rhs);
        }
        stiff.apply_ghost_ess_columns(&ghost_ess, &mut rhs);

        let mut u = ParVector::from_local_raw(
            vec![0.0; dp.n_total_dofs()],
            dp.n_owned_dofs,
            ps.dof_ghost_exchange_arc(),
            comm.clone(),
        );
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 10000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = par_solve_pcg_amg(&stiff, &rhs, &mut u, &pex3_amg_cfg(), &cfg)
            .expect("PCG+AMG solve failed");

        // A rank-count-independent functional of the solution: the global L² of
        // the coefficient vector (`Σ` over every global DOF exactly once — the
        // same physical field at any rank count, unlike any per-rank slice).
        let norm = u.global_norm();
        if comm.is_root() {
            *out_rank.lock().unwrap() = (res.iterations, res.converged, norm);
        }
    });
    let r = *out.lock().unwrap();
    r
}

/// The ownership-independence pin: at np = 1 and np = 2 the pex3 system must
/// converge, the 2-rank iteration count must stay in the same regime as the
/// 1-rank one, and the solved field must not depend on the split.  With the
/// block-local coarsening the 2-rank run stalled at `max_iter` (10000) — red;
/// green it is 95 iterations against np = 1's 87 (`amg_fix_pex3.txt`).
#[test]
fn d790r73_amg_quality_is_owner_split_independent() {
    let (it1, conv1, n1) = solve_pex3(1);
    let (it2, conv2, n2) = solve_pex3(2);
    println!(
        "D790R73AMG np1: {it1} it (converged={conv1}) ‖u‖={n1:.17e}\n\
         D790R73AMG np2: {it2} it (converged={conv2}) ‖u‖={n2:.17e}"
    );
    assert!(conv1, "np=1: PCG+AMG did not converge ({it1} it)");
    assert!(
        conv2,
        "np=2: PCG+AMG did not converge ({it2} it) — the coarsening is \
         block-diagonal again (D790-1)"
    );
    assert!(
        it2 <= 4 * it1 + 20,
        "np=2 needed {it2} iterations against np=1's {it1}: the AMG quality \
         depends on the owner split (D790-1)"
    );
    assert!(
        (n1 - n2).abs() <= 1e-6 * n1.abs(),
        "the solved field depends on the rank count: ‖u‖ = {n1:.17e} (np=1) vs \
         {n2:.17e} (np=2)"
    );
}
