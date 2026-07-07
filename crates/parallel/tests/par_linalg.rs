//! Parallel linear algebra integration tests.
//!
//! Exercises ParVector global ops, ParCsrMatrix SpMV, AMG PCG, and RAS
//! preconditioner convergence using ThreadLauncher (no MPI required).
//!
//! All tests pass both individually and together (108 tests in fem-parallel).

use std::sync::{Arc, Mutex};

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::{
    ParAssembler, ParVector, ParallelFESpace, WorkerConfig,
    par_partition::partition_mesh,
    par_solve_pcg_amg, ParAmgConfig,
    par_solve_pcg_ras, RasConfig, RasLocalSolverKind,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn launcher(n: usize) -> ThreadLauncher {
    ThreadLauncher::new(WorkerConfig::new(n))
}

/// Build a parallel Poisson system on unit square for testing.
fn build_poisson(
    comm: &fem_parallel::comm::Comm,
    mesh_n: usize,
    order: u8,
) -> (fem_parallel::ParCsrMatrix, ParVector, ParallelFESpace<H1Space<Mesh<2>>>) {
    let mesh = Arc::new(Mesh::<2>::unit_square_tri(mesh_n));
    let par_mesh = partition_mesh(&mesh, comm);
    let local_mesh = par_mesh.local_mesh().clone();
    let local_space = H1Space::new(local_mesh, order);
    let par_space = ParallelFESpace::new(local_space, &par_mesh, comm.clone());

    use std::f64::consts::PI;
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let mut a = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

    let dp = par_space.dof_partition();
    let bc_dofs = boundary_dofs(par_space.local_space().mesh(),
        par_space.local_space().dof_manager(), &[1, 2, 3, 4]);
    for &d in &bc_dofs {
        let pid = dp.permute_dof(d) as usize;
        if pid < dp.n_owned_dofs {
            a.apply_dirichlet_par(pid, 0.0, &mut rhs);
        }
    }

    (a, rhs, par_space)
}

// ── 1. ParCsrMatrix / ParVector basic ops ───────────────────────────────────

#[test]
fn par_vector_global_ops_consistent() {
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = Arc::clone(&results);
    launcher(2).launch(move |comm| {
        let (_a, _rhs, par_space) = build_poisson(&comm, 6, 1);
        let dp = par_space.dof_partition();
        let n_owned = dp.n_owned_dofs;

        let mut v = ParVector::zeros_from_space(&par_space);
        for pid in 0..n_owned { v.as_slice_mut()[pid] = 1.0; }
        v.update_ghosts();

        let dot = v.global_dot(&v);
        let n_global = par_space.n_global_dofs() as f64;
        assert!((dot - n_global).abs() < 1e-12);

        let norm = v.global_norm();
        assert!((norm - n_global.sqrt()).abs() < 1e-12);

        let mut y = ParVector::zeros_like(&v);
        y.axpy(3.0, &v);
        let y_dot = y.global_dot(&y);
        let expected = (3.0 * n_global.sqrt()).powi(2);
        assert!((y_dot - expected).abs() < 1e-10);

        r.lock().unwrap().push(comm.rank());
    });
    assert_eq!(results.lock().unwrap().len(), 2);
}

#[test]
fn par_cg_solves_poisson_on_2_ranks() {
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = Arc::clone(&results);
    launcher(2).launch(move |comm| {
        let (a, rhs, ps) = build_poisson(&comm, 8, 1);
        let mut u = ParVector::zeros_from_space(&ps);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let result = fem_parallel::par_solve_cg(&a, &rhs, &mut u, &cfg);
        assert!(result.is_ok());
        let r2 = result.unwrap();
        assert!(r2.converged, "par_solve_cg not converged (iters={})", r2.iterations);
        r.lock().unwrap().push((comm.rank(), r2.converged, r2.iterations));
    });
    let all = results.lock().unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|(_, c, _)| *c), "All ranks must converge");
}

// ── 2. GhostExchange is already tested in thread_launcher.rs ──────────────────

// ── 3. Parallel AMG PCG ───────────────────────────────────────────────────────

#[test]
fn par_amg_pcg_converges_on_2_ranks() {
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = Arc::clone(&results);
    launcher(2).launch(move |comm| {
        let (a, rhs, ps) = build_poisson(&comm, 12, 1);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 100, verbose: false, ..SolverConfig::default() };
        let amg_cfg = ParAmgConfig { max_levels: 10, coarse_size: 10, ..ParAmgConfig::default() };
        let mut u = ParVector::zeros_from_space(&ps);
        let result = par_solve_pcg_amg(&a, &rhs, &mut u, &amg_cfg, &cfg);
        assert!(result.is_ok(), "AMG PCG failed: {result:?}");
        let r2 = result.unwrap();
        assert!(r2.converged, "AMG PCG not converged on rank {} (iters={}, residual={:.3e})",
            comm.rank(), r2.iterations, r2.final_residual);
        r.lock().unwrap().push((comm.rank(), r2.converged, r2.final_residual));
    });
    let all = results.lock().unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|(_, c, _)| *c), "All ranks must converge");
}

// ── 4. RAS preconditioner convergence ─────────────────────────────────────────

#[test]
fn ras_pcg_converges_on_2_ranks() {
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = Arc::clone(&results);
    launcher(2).launch(move |comm| {
        let (a, rhs, ps) = build_poisson(&comm, 12, 1);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, verbose: false, ..SolverConfig::default() };
        let ras_cfg = RasConfig { overlap: 1, local_solver: RasLocalSolverKind::DiagJacobi, ..RasConfig::default() };
        let mut u = ParVector::zeros_from_space(&ps);
        let result = par_solve_pcg_ras(&a, &rhs, &mut u, &ras_cfg, &cfg);
        assert!(result.is_ok(), "RAS PCG failed: {result:?}");
        let r2 = result.unwrap();
        assert!(r2.converged, "RAS PCG not converged on rank {} (iters={}, residual={:.3e})",
            comm.rank(), r2.iterations, r2.final_residual);
        r.lock().unwrap().push((comm.rank(), r2.converged, r2.final_residual));
    });
    let all = results.lock().unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|(_, c, _)| *c), "All ranks must converge");
}

// ── 5. Par MMS ──────────────────────────────────────────────────────────────

#[test]
fn par_mms_p1_two_ranks() {
    use std::f64::consts::PI;
    let src_fn = |x: &[f64]| 2.0*PI*PI*(PI*x[0]).sin()*(PI*x[1]).sin();
    launcher(2).launch(move |comm| {
        let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
        let pm = partition_mesh(&mesh, &comm);
        let ls = H1Space::new(pm.local_mesh().clone(), 1);
        let ps = ParallelFESpace::new(ls, &pm, comm.clone());
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&ps, &[&diff], 3);
        let src = DomainSourceIntegrator::new(&src_fn);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&src], 3);
        let dp = ps.dof_partition();
        for &d in &boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[1,2,3,4]) {
            let pid = dp.permute_dof(d) as usize;
            if pid < dp.n_owned_dofs { a.apply_dirichlet_par(pid, 0.0, &mut rhs); }
        }
        let mut u = ParVector::zeros_from_space(&ps);
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let res = fem_parallel::par_solve_cg(&a, &rhs, &mut u, &cfg).unwrap();
        assert!(res.converged, "P1 2-rank MMS CG not converged");
        if comm.is_root() { eprintln!("P1 2-rank MMS OK: {} iters", res.iterations); }
    });
}

#[test]
fn par_mms_p2_two_ranks() {
    use std::f64::consts::PI;
    let src_fn = |x: &[f64]| 2.0*PI*PI*(PI*x[0]).sin()*(PI*x[1]).sin();
    launcher(2).launch(move |comm| {
        let mesh = Arc::new(Mesh::<2>::unit_square_tri(8));
        let pm = partition_mesh(&mesh, &comm);
        let lm = pm.local_mesh().clone();
        let ls = H1Space::new(lm.clone(), 2);
        let dm = fem_space::dof_manager::DofManager::new(&lm, 2);
        let ps = ParallelFESpace::new_with_dof_manager(ls, &pm, &dm, comm.clone());
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&ps, &[&diff], 4);
        let src = DomainSourceIntegrator::new(&src_fn);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&src], 5);
        let dp = ps.dof_partition();
        for &d in &boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[1,2,3,4]) {
            let pid = dp.permute_dof(d) as usize;
            if pid < dp.n_owned_dofs { a.apply_dirichlet_par(pid, 0.0, &mut rhs); }
        }
        let mut u = ParVector::zeros_from_space(&ps);
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let res = fem_parallel::par_solve_cg(&a, &rhs, &mut u, &cfg).unwrap();
        assert!(res.converged, "P2 2-rank MMS CG not converged");
        if comm.is_root() { eprintln!("P2 2-rank MMS OK: {} iters", res.iterations); }
    });
}

#[test]
fn par_mms_p1_four_ranks_matches_two_ranks() {
    use std::f64::consts::PI;
    let src_fn = |x: &[f64]| 2.0*PI*PI*(PI*x[0]).sin()*(PI*x[1]).sin();
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = Arc::clone(&results);
    launcher(4).launch(move |comm| {
        let mesh = Arc::new(Mesh::<2>::unit_square_tri(12));
        let pm = partition_mesh(&mesh, &comm);
        let ls = H1Space::new(pm.local_mesh().clone(), 1);
        let ps = ParallelFESpace::new(ls, &pm, comm.clone());
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&ps, &[&diff], 3);
        let src = DomainSourceIntegrator::new(&src_fn);
        let mut rhs = ParAssembler::assemble_linear(&ps, &[&src], 3);
        let dp = ps.dof_partition();
        for &d in &boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[1,2,3,4]) {
            let pid = dp.permute_dof(d) as usize;
            if pid < dp.n_owned_dofs { a.apply_dirichlet_par(pid, 0.0, &mut rhs); }
        }
        let mut u = ParVector::zeros_from_space(&ps);
        let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let res = fem_parallel::par_solve_cg(&a, &rhs, &mut u, &cfg).unwrap();
        assert!(res.converged, "P1 4-rank MMS CG not converged");
        r.lock().unwrap().push(res.converged);
    });
    let all = results.lock().unwrap();
    assert_eq!(all.len(), 4, "all 4 ranks must report");
    assert!(all.iter().all(|&c| c), "all ranks must converge");
}
