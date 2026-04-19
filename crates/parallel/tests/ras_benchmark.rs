use std::time::Instant;
use std::path::Path;
use std::{env, fs};
use std::sync::{Arc, Mutex};

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    par_solve_gmres_ras, par_solve_pcg_ras, partition_simplex, summarize_ras_hpc,
    ParAssembler, ParallelFESpace, RasConfig, RasLocalSolverKind,
};
use fem_parallel::launcher::{native::ThreadLauncher, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

#[derive(Debug, Clone)]
struct ScalingRow {
    mode: &'static str,
    ranks: usize,
    mesh_n: usize,
    dofs: usize,
    iterations: usize,
    final_residual: f64,
    time_ms: f64,
    owned: usize,
    ghost: usize,
    nnz_diag: usize,
    nnz_offd: usize,
    owned_cv: f64,
    ghost_cv: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaturityScore {
    Pass,
    Warn,
    Fail,
}

impl MaturityScore {
    fn as_str(self) -> &'static str {
        match self {
            MaturityScore::Pass => "pass",
            MaturityScore::Warn => "warn",
            MaturityScore::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalingGates {
    strong_eff_warn: f64,
    strong_eff_fail: f64,
    weak_growth_warn: f64,
    weak_growth_fail: f64,
    owned_cv_warn: f64,
    owned_cv_fail: f64,
    ghost_cv_warn: f64,
    ghost_cv_fail: f64,
}

impl Default for ScalingGates {
    fn default() -> Self {
        Self {
            strong_eff_warn: 0.50,
            strong_eff_fail: 0.30,
            weak_growth_warn: 3.00,
            weak_growth_fail: 6.00,
            owned_cv_warn: 0.20,
            owned_cv_fail: 0.35,
            ghost_cv_warn: 0.35,
            ghost_cv_fail: 0.55,
        }
    }
}

fn read_gate(name: &str, default: f64) -> f64 {
    match env::var(name) {
        Ok(v) => v.parse::<f64>().ok().filter(|x| x.is_finite()).unwrap_or(default),
        Err(_) => default,
    }
}

fn scaling_gates_from_env() -> ScalingGates {
    let d = ScalingGates::default();
    ScalingGates {
        strong_eff_warn: read_gate("RAS_STRONG_EFF_WARN", d.strong_eff_warn),
        strong_eff_fail: read_gate("RAS_STRONG_EFF_FAIL", d.strong_eff_fail),
        weak_growth_warn: read_gate("RAS_WEAK_GROWTH_WARN", d.weak_growth_warn),
        weak_growth_fail: read_gate("RAS_WEAK_GROWTH_FAIL", d.weak_growth_fail),
        owned_cv_warn: read_gate("RAS_OWNED_CV_WARN", d.owned_cv_warn),
        owned_cv_fail: read_gate("RAS_OWNED_CV_FAIL", d.owned_cv_fail),
        ghost_cv_warn: read_gate("RAS_GHOST_CV_WARN", d.ghost_cv_warn),
        ghost_cv_fail: read_gate("RAS_GHOST_CV_FAIL", d.ghost_cv_fail),
    }
}

fn score_mode_metric(value: f64, warn: f64, fail: f64, higher_is_better: bool) -> MaturityScore {
    if higher_is_better {
        if value < fail {
            MaturityScore::Fail
        } else if value < warn {
            MaturityScore::Warn
        } else {
            MaturityScore::Pass
        }
    } else if value > fail {
        MaturityScore::Fail
    } else if value > warn {
        MaturityScore::Warn
    } else {
        MaturityScore::Pass
    }
}

fn combine_scores(scores: &[MaturityScore]) -> MaturityScore {
    if scores.iter().any(|s| *s == MaturityScore::Fail) {
        MaturityScore::Fail
    } else if scores.iter().any(|s| *s == MaturityScore::Warn) {
        MaturityScore::Warn
    } else {
        MaturityScore::Pass
    }
}

fn run_scaling_point(ranks: usize, mode: &'static str, mesh_n: usize) -> ScalingRow {
    let mesh = SimplexMesh::<2>::unit_square_tri(mesh_n);
    let launcher = ThreadLauncher::new(WorkerConfig::new(ranks));
    let out: Arc<Mutex<Option<ScalingRow>>> = Arc::new(Mutex::new(None));
    let out_shared = Arc::clone(&out);

    launcher.launch(move |comm| {
        let pmesh = partition_simplex(&mesh, &comm);
        let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
        let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
        let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

        let dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
        for &d in &bc_dofs {
            let lid = d as usize;
            if lid < par_space.dof_partition().n_owned_dofs {
                a_mat.apply_dirichlet_row(lid, 0.0, rhs.as_slice_mut());
            }
        }

        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 3000,
            ..SolverConfig::default()
        };
        let ras_cfg = RasConfig {
            overlap: 1,
            local_solver: RasLocalSolverKind::Ilu0,
            ..RasConfig::default()
        };

        let mut u = fem_parallel::ParVector::zeros(&par_space);
        let t0 = Instant::now();
        let res = par_solve_pcg_ras(&a_mat, &rhs, &mut u, &ras_cfg, &cfg)
            .unwrap_or_else(|e| panic!("pcg_ras_ilu0_ov1 failed: {}", e));
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        let hpc = summarize_ras_hpc(&a_mat);

        assert!(
            res.final_residual <= 1e-6,
            "scaling run residual too high on rank {}: {:.3e}",
            comm.rank(),
            res.final_residual
        );

        if comm.is_root() {
            let row = ScalingRow {
                mode,
                ranks,
                mesh_n,
                dofs: hpc.global_owned_dofs,
                iterations: res.iterations,
                final_residual: res.final_residual,
                time_ms: dt,
                owned: hpc.global_owned_dofs,
                ghost: hpc.global_ghost_dofs,
                nnz_diag: hpc.global_diag_nnz,
                nnz_offd: hpc.global_offd_nnz,
                owned_cv: hpc.owned_dofs_cv,
                ghost_cv: hpc.ghost_dofs_cv,
            };
            *out_shared.lock().expect("scaling output mutex poisoned") = Some(row);
        }
    });

    let row = out
        .lock()
        .expect("scaling output mutex poisoned")
        .take()
        .expect("root scaling row was not produced");
    row
}

#[test]
#[ignore = "benchmark-style timing test; run explicitly"]
fn ras_benchmark_report_two_ranks() {
    let mesh = SimplexMesh::<2>::unit_square_tri(24);
    let launcher = ThreadLauncher::new(WorkerConfig::new(2));

    launcher.launch(move |comm| {
        let pmesh = partition_simplex(&mesh, &comm);
        let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
        let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
        let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

        let dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
        for &d in &bc_dofs {
            let lid = d as usize;
            if lid < par_space.dof_partition().n_owned_dofs {
                a_mat.apply_dirichlet_row(lid, 0.0, rhs.as_slice_mut());
            }
        }

        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 2000,
            ..SolverConfig::default()
        };

        let hpc = summarize_ras_hpc(&a_mat);

        let mut rows: Vec<(String, usize, f64, f64)> = Vec::new();

        {
            let mut run_pcg = |name: &str, ras_cfg: RasConfig| {
                let mut u = fem_parallel::ParVector::zeros(&par_space);
                let t0 = Instant::now();
                let res = par_solve_pcg_ras(&a_mat, &rhs, &mut u, &ras_cfg, &cfg)
                    .unwrap_or_else(|e| panic!("{} failed: {}", name, e));
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                rows.push((name.to_string(), res.iterations, res.final_residual, dt));
            };

            run_pcg(
                "pcg_ras_diag_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_pcg(
                "pcg_ras_diag_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_pcg(
                "pcg_ras_ilu0_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
            run_pcg(
                "pcg_ras_ilu0_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
        }

        {
            let mut run_gmres = |name: &str, ras_cfg: RasConfig| {
                let mut u = fem_parallel::ParVector::zeros(&par_space);
                let t0 = Instant::now();
                let res = par_solve_gmres_ras(&a_mat, &rhs, &mut u, &ras_cfg, 30, &cfg)
                    .unwrap_or_else(|e| panic!("{} failed: {}", name, e));
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                rows.push((name.to_string(), res.iterations, res.final_residual, dt));
            };

            run_gmres(
                "gmres_ras_diag_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_gmres(
                "gmres_ras_diag_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_gmres(
                "gmres_ras_ilu0_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
            run_gmres(
                "gmres_ras_ilu0_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
        }

        if comm.is_root() {
            println!("\n=== RAS Benchmark (2 ranks, mesh=unit_square_tri(24)) ===");
            println!(
                "hpc_summary,ranks={},owned={},ghost={},nnz_diag={},nnz_offd={},owned_cv={:.4},ghost_cv={:.4}",
                hpc.n_ranks,
                hpc.global_owned_dofs,
                hpc.global_ghost_dofs,
                hpc.global_diag_nnz,
                hpc.global_offd_nnz,
                hpc.owned_dofs_cv,
                hpc.ghost_dofs_cv,
            );
            println!(
                "case,iterations,final_residual,time_ms,ranks,owned,ghost,nnz_diag,nnz_offd,owned_cv,ghost_cv"
            );
            for (name, it, rr, ms) in &rows {
                println!(
                    "{},{},{:.3e},{:.3},{},{},{},{},{},{:.4},{:.4}",
                    name,
                    it,
                    rr,
                    ms,
                    hpc.n_ranks,
                    hpc.global_owned_dofs,
                    hpc.global_ghost_dofs,
                    hpc.global_diag_nnz,
                    hpc.global_offd_nnz,
                    hpc.owned_dofs_cv,
                    hpc.ghost_dofs_cv,
                );
            }

            if let Ok(path) = env::var("RAS_BENCH_CSV") {
                let mut csv = String::from(
                    "case,iterations,final_residual,time_ms,ranks,owned,ghost,nnz_diag,nnz_offd,owned_cv,ghost_cv\n",
                );
                for (name, it, rr, ms) in &rows {
                    csv.push_str(&format!(
                        "{},{},{:.6e},{:.6},{},{},{},{},{},{:.6},{:.6}\n",
                        name,
                        it,
                        rr,
                        ms,
                        hpc.n_ranks,
                        hpc.global_owned_dofs,
                        hpc.global_ghost_dofs,
                        hpc.global_diag_nnz,
                        hpc.global_offd_nnz,
                        hpc.owned_dofs_cv,
                        hpc.ghost_dofs_cv,
                    ));
                }
                if let Some(parent) = Path::new(&path).parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent).unwrap_or_else(|e| {
                            panic!(
                                "failed to create parent directory for RAS_BENCH_CSV {}: {}",
                                path, e
                            )
                        });
                    }
                }
                fs::write(&path, csv)
                    .unwrap_or_else(|e| panic!("failed to write RAS_BENCH_CSV to {}: {}", path, e));
                println!("ras benchmark csv written to {}", path);
            }
        }

        for (_name, _it, rr, _ms) in rows {
            assert!(rr <= 1e-6, "benchmark run residual too high: {:.3e}", rr);
        }
    });
}

#[test]
#[ignore = "benchmark-style scaling test; run explicitly"]
fn ras_scaling_report_pcg_ilu0_overlap1() {
    let ranks_list = [1usize, 2, 4, 8];
    let strong_mesh_n = 32usize;
    let weak_base_mesh_n = 16usize;

    let mut rows: Vec<ScalingRow> = Vec::new();

    for &ranks in &ranks_list {
        rows.push(run_scaling_point(ranks, "strong", strong_mesh_n));
    }
    for &ranks in &ranks_list {
        let mesh_n = ((weak_base_mesh_n as f64) * (ranks as f64).sqrt()).round() as usize;
        rows.push(run_scaling_point(ranks, "weak", mesh_n.max(weak_base_mesh_n)));
    }

    rows.sort_by_key(|r| (r.mode, r.ranks));

    let strong_t1 = rows
        .iter()
        .find(|r| r.mode == "strong" && r.ranks == 1)
        .expect("missing strong rank=1 row")
        .time_ms;
    let weak_t1 = rows
        .iter()
        .find(|r| r.mode == "weak" && r.ranks == 1)
        .expect("missing weak rank=1 row")
        .time_ms;
    let gates = scaling_gates_from_env();

    let mut score_pass = 0usize;
    let mut score_warn = 0usize;
    let mut score_fail = 0usize;

    println!("\n=== RAS Scaling Benchmark (PCG + ILU0 + overlap=1) ===");
    println!(
        "mode,ranks,mesh_n,dofs,iterations,final_residual,time_ms,strong_eff,weak_growth,owned,ghost,nnz_diag,nnz_offd,owned_cv,ghost_cv,score"
    );

    for r in &rows {
        let strong_eff = if r.mode == "strong" {
            strong_t1 / (r.ranks as f64 * r.time_ms)
        } else {
            0.0
        };
        let weak_growth = if r.mode == "weak" {
            r.time_ms / weak_t1
        } else {
            0.0
        };

        let mode_score = if r.mode == "strong" {
            score_mode_metric(strong_eff, gates.strong_eff_warn, gates.strong_eff_fail, true)
        } else {
            score_mode_metric(weak_growth, gates.weak_growth_warn, gates.weak_growth_fail, false)
        };
        let owned_score = score_mode_metric(r.owned_cv, gates.owned_cv_warn, gates.owned_cv_fail, false);
        let ghost_score = score_mode_metric(r.ghost_cv, gates.ghost_cv_warn, gates.ghost_cv_fail, false);
        let score = combine_scores(&[mode_score, owned_score, ghost_score]);

        match score {
            MaturityScore::Pass => score_pass += 1,
            MaturityScore::Warn => score_warn += 1,
            MaturityScore::Fail => score_fail += 1,
        }

        println!(
            "{},{},{},{},{},{:.3e},{:.3},{:.4},{:.4},{},{},{},{},{:.4},{:.4},{}",
            r.mode,
            r.ranks,
            r.mesh_n,
            r.dofs,
            r.iterations,
            r.final_residual,
            r.time_ms,
            strong_eff,
            weak_growth,
            r.owned,
            r.ghost,
            r.nnz_diag,
            r.nnz_offd,
            r.owned_cv,
            r.ghost_cv,
            score.as_str(),
        );
    }

    println!(
        "scoring_summary,pass={},warn={},fail={},gates,strong_eff(warn/fail)={:.3}/{:.3},weak_growth(warn/fail)={:.3}/{:.3},owned_cv(warn/fail)={:.3}/{:.3},ghost_cv(warn/fail)={:.3}/{:.3}",
        score_pass,
        score_warn,
        score_fail,
        gates.strong_eff_warn,
        gates.strong_eff_fail,
        gates.weak_growth_warn,
        gates.weak_growth_fail,
        gates.owned_cv_warn,
        gates.owned_cv_fail,
        gates.ghost_cv_warn,
        gates.ghost_cv_fail,
    );

    if let Ok(path) = env::var("RAS_SCALING_CSV") {
        let mut csv = String::from(
            "mode,ranks,mesh_n,dofs,iterations,final_residual,time_ms,strong_eff,weak_growth,owned,ghost,nnz_diag,nnz_offd,owned_cv,ghost_cv,score\n",
        );
        for r in &rows {
            let strong_eff = if r.mode == "strong" {
                strong_t1 / (r.ranks as f64 * r.time_ms)
            } else {
                0.0
            };
            let weak_growth = if r.mode == "weak" {
                r.time_ms / weak_t1
            } else {
                0.0
            };

            let mode_score = if r.mode == "strong" {
                score_mode_metric(strong_eff, gates.strong_eff_warn, gates.strong_eff_fail, true)
            } else {
                score_mode_metric(weak_growth, gates.weak_growth_warn, gates.weak_growth_fail, false)
            };
            let owned_score = score_mode_metric(r.owned_cv, gates.owned_cv_warn, gates.owned_cv_fail, false);
            let ghost_score = score_mode_metric(r.ghost_cv, gates.ghost_cv_warn, gates.ghost_cv_fail, false);
            let score = combine_scores(&[mode_score, owned_score, ghost_score]);

            csv.push_str(&format!(
                "{},{},{},{},{},{:.6e},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{}\n",
                r.mode,
                r.ranks,
                r.mesh_n,
                r.dofs,
                r.iterations,
                r.final_residual,
                r.time_ms,
                strong_eff,
                weak_growth,
                r.owned,
                r.ghost,
                r.nnz_diag,
                r.nnz_offd,
                r.owned_cv,
                r.ghost_cv,
                score.as_str(),
            ));
        }
        if let Some(parent) = Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).unwrap_or_else(|e| {
                    panic!(
                        "failed to create parent directory for RAS_SCALING_CSV {}: {}",
                        path, e
                    )
                });
            }
        }
        fs::write(&path, csv)
            .unwrap_or_else(|e| panic!("failed to write RAS_SCALING_CSV to {}: {}", path, e));
        println!("ras scaling csv written to {}", path);
    }

    for r in &rows {
        assert!(r.final_residual <= 1e-6, "residual too high in scaling row");
    }
}
