//! Example 49: built-in template driver - Fluid-Structure Interaction (FSI).
//!
//! This is a practical quasi-FSI template driver using existing fem-rs building
//! blocks:
//! - structure proxy: compliant top-wall displacement from fluid load
//! - mesh motion: boundary displacement + Laplacian interior smoothing
//! - fluid proxy: pressure solve on moving mesh
//! - conservative field transfer between nonmatching meshes
//!
//! The goal is to provide a ready-to-run built-in template entrypoint with
//! stable workflow and coupling interfaces.

use std::{f64::consts::PI, fs, io};

use fem_assembly::{
    Assembler,
    standard::DiffusionIntegrator,
    transfer_h1_p1_nonmatching_l2_projection_conservative,
};
use fem_examples::checkpoint_text::{ensure_parent_dir, format_vec_f64, parse_vec_f64};
use fem_examples::hdf5_checkpoint::{scalar_rank_field_f64, vector_rank_field_f64};
use fem_examples::template_runner::{
    maybe_write_template_kpi_csv,
    TemplateAdaptiveSummary,
    TemplateCouplingSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
    print_template_coupling_summary,
    print_template_header,
};
use fem_examples::workflow_cli::{
    assert_single_restart_source,
    push_workflow_cli_help,
    WorkflowCliOptions,
};
#[cfg(feature = "io_hdf5")]
use fem_examples::hdf5_checkpoint::{checkpoint_sidecar_path, write_scalar_checkpoint_xdmf_sidecars};
use fem_io::vtk::{DataArray, VtkWriter};
use fem_io_hdf5_parallel::{
    CheckpointBundleF64,
    IoBackend,
    ParallelIoConfig,
    read_checkpoint_fields_f64_latest,
    validate_checkpoint_layout,
    write_checkpoint_step_bundle_f64,
};
use fem_linalg::Vector;
use fem_mesh::{
    MeshMotionConfig,
    Mesh,
    all_boundary_nodes,
    apply_node_displacement,
    laplacian_smooth_2d,
};
use fem_solver::{
    BuiltinMultiphysicsTemplate,
    MultiRateConfig,
    SolverConfig,
    TemplateSyncPolicy,
    builtin_template_spec,
    run_multirate_adaptive,
    solve_gmres,
    solve_pcg_jacobi,
};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

#[derive(Clone)]
struct Args {
    n: usize,
    steps: usize,
    dt: f64,
    fast_dt: f64,
    use_subcycling: bool,
    inlet_amp: f64,
    compliance: f64,
    wall_relax: f64,
    coupling_tol: f64,
    sync_error_tol: f64,
    max_coupling: usize,
    sync_retries: usize,
    fast_dt_min: f64,
    omega: f64,
    smooth_iters: usize,
}

struct FsiTemplateResult {
    completed_steps: usize,
    steps: usize,
    converged_steps: usize,
    max_coupling_iters_used: usize,
    max_transfer_abs_int_err: f64,
    max_wall_displacement: f64,
    final_wall_displacement: f64,
    final_pressure_norm: f64,
    final_pressure_checksum: f64,
    sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
    pressure: Vec<f64>,
    final_mesh: Mesh<2>,
}

struct CliArgs {
    sim: Args,
    checkpoint: Option<String>,
    checkpoint_h5: Option<String>,
    restart: Option<String>,
    restart_h5: Option<String>,
    export_vtk_prefix: Option<String>,
}

struct FsiCheckpointState {
    args: Args,
    completed_steps: usize,
    converged_steps: usize,
    max_coupling_iters_used: usize,
    max_transfer_abs_int_err: f64,
    max_wall_displacement: f64,
    final_wall_displacement: f64,
    observed_sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
    mesh_coords: Vec<f64>,
    pressure: Vec<f64>,
}

fn main() {
    let cli = parse_args();
    assert_single_restart_source(&WorkflowCliOptions {
        checkpoint: cli.checkpoint.clone(),
        checkpoint_h5: cli.checkpoint_h5.clone(),
        restart: cli.restart.clone(),
        restart_h5: cli.restart_h5.clone(),
        export_vtk_prefix: cli.export_vtk_prefix.clone(),
    });
    let restart_state = cli
        .restart
        .as_deref()
        .map(read_fsi_checkpoint)
        .transpose()
        .unwrap_or_else(|e| panic!("failed to read restart state: {e}"))
        .or_else(|| {
            cli.restart_h5
                .as_deref()
                .map(read_fsi_hdf5_checkpoint)
                .transpose()
                .unwrap_or_else(|e| panic!("failed to read HDF5 restart state: {e}"))
        });
    let mut args = cli.sim.clone();
    if let Some(state) = restart_state.as_ref() {
        let requested_steps = args.steps.max(state.completed_steps);
        args = state.args.clone();
        args.steps = requested_steps;
    }
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::FluidStructureInteraction);

    let config_line = format!(
        "n={}, steps={}, dt={}, fast_dt={}, fast_dt_min={}, subcycling={}, inlet_amp={}, compliance={}, wall_relax={}, coupling_tol={}, sync_error_tol={}, max_coupling={}, sync_retries={}",
        args.n,
        args.steps,
        args.dt,
        args.fast_dt,
        args.fast_dt_min,
        args.use_subcycling,
        args.inlet_amp,
        args.compliance,
        args.wall_relax,
        args.coupling_tol,
        args.sync_error_tol,
        args.max_coupling,
        args.sync_retries,
    );
    print_template_header("Example 49: Built-in template driver", spec, &config_line);

    let result = if let Some(restart) = restart_state.as_ref() {
        solve_fsi_template_with_restart(&args, Some(restart))
    } else {
        solve_fsi_template(&args)
    };

    let coupling = TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.converged_steps,
        max_coupling_iters_used: result.max_coupling_iters_used,
    };
    print_template_coupling_summary(coupling);
    println!(
        "  max transfer integral error: {:.3e}",
        result.max_transfer_abs_int_err
    );
    println!("  max |wall displacement|: {:.6e}", result.max_wall_displacement);
    println!("  final wall displacement: {:.6e}", result.final_wall_displacement);
    println!("  final ||p||_2: {:.6e}", result.final_pressure_norm);
    println!("  final pressure checksum: {:.8e}", result.final_pressure_checksum);
    let adaptive = TemplateAdaptiveSummary {
        sync_retries: result.sync_retries,
        rejected_sync_steps: result.rejected_sync_steps,
        rollback_count: result.rollback_count,
    };
    print_template_adaptive_summary(adaptive);
    if let Err(e) = maybe_write_template_kpi_csv(
        spec.template.id(),
        coupling,
        adaptive,
        &[
            ("max_transfer_abs_int_err", result.max_transfer_abs_int_err),
            ("max_wall_displacement", result.max_wall_displacement),
            ("final_wall_displacement", result.final_wall_displacement),
            ("final_pressure_norm", result.final_pressure_norm),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }

    if let Some(path) = &cli.checkpoint {
        let checkpoint = FsiCheckpointState {
            args: args.clone(),
            completed_steps: result.completed_steps,
            converged_steps: result.converged_steps,
            max_coupling_iters_used: result.max_coupling_iters_used,
            max_transfer_abs_int_err: result.max_transfer_abs_int_err,
            max_wall_displacement: result.max_wall_displacement,
            final_wall_displacement: result.final_wall_displacement,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            mesh_coords: result.final_mesh.coords.clone(),
            pressure: result.pressure.clone(),
        };
        if let Err(e) = write_fsi_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write checkpoint: {e}");
        } else {
            println!("  checkpoint written: {path}");
        }
    }

    if let Some(path) = &cli.checkpoint_h5 {
        let checkpoint = FsiCheckpointState {
            args: args.clone(),
            completed_steps: result.completed_steps,
            converged_steps: result.converged_steps,
            max_coupling_iters_used: result.max_coupling_iters_used,
            max_transfer_abs_int_err: result.max_transfer_abs_int_err,
            max_wall_displacement: result.max_wall_displacement,
            final_wall_displacement: result.final_wall_displacement,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            mesh_coords: result.final_mesh.coords.clone(),
            pressure: result.pressure.clone(),
        };
        if let Err(e) = write_fsi_hdf5_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write HDF5 checkpoint: {e}");
        } else {
            println!("  HDF5 checkpoint written: {path}");
            #[cfg(feature = "io_hdf5")]
            if let Err(e) = write_fsi_hdf5_xdmf_sidecars(path, &checkpoint) {
                eprintln!("warning: failed to write checkpoint XDMF sidecars: {e}");
            }
        }
    }

    if let Some(prefix) = &cli.export_vtk_prefix {
        if let Err(e) = write_ex49_vtk_export(prefix, &result.final_mesh, &result.pressure) {
            eprintln!("warning: failed to write VTK export: {e}");
        } else {
            println!("  VTK export written: {prefix}_fsi_pressure.vtu");
        }
    }
}

fn solve_fsi_template(args: &Args) -> FsiTemplateResult {
    solve_fsi_template_with_restart(args, None)
}

fn solve_fsi_template_with_restart(
    args: &Args,
    restart: Option<&FsiCheckpointState>,
) -> FsiTemplateResult {
    if args.use_subcycling {
        solve_fsi_template_subcycling(args, restart)
    } else {
        solve_fsi_template_single_rate(args, restart)
    }
}

fn solve_fsi_template_single_rate(
    args: &Args,
    restart: Option<&FsiCheckpointState>,
) -> FsiTemplateResult {
    let ref_mesh = Mesh::<2>::unit_square_tri(args.n);
    let mut wall_disp = restart.map(|state| state.final_wall_displacement).unwrap_or(0.0_f64);
    let mut mesh = restart
        .map(|state| mesh_from_checkpoint_coords(&ref_mesh, &state.mesh_coords))
        .unwrap_or_else(|| build_deformed_mesh(&ref_mesh, wall_disp, args.omega, args.smooth_iters));
    let mut pressure = if let Some(state) = restart {
        Vector::from_vec(state.pressure.clone())
    } else {
        H1Space::new(mesh.clone(), 1).interpolate(&|_x| 0.0)
    };

    let completed_steps = restart.map(|state| state.completed_steps).unwrap_or(0);
    let mut max_wall_disp = restart.map(|state| state.max_wall_displacement).unwrap_or(0.0_f64);
    let mut max_transfer_abs_int_err = restart
        .map(|state| state.max_transfer_abs_int_err)
        .unwrap_or(0.0_f64);
    let mut converged_steps = restart.map(|state| state.converged_steps).unwrap_or(0usize);
    let mut max_coupling_iters_used = restart
        .map(|state| state.max_coupling_iters_used)
        .unwrap_or(0usize);

    if completed_steps >= args.steps {
        return FsiTemplateResult {
            completed_steps,
            converged_steps,
            steps: completed_steps,
            max_coupling_iters_used,
            max_transfer_abs_int_err,
            max_wall_displacement: max_wall_disp,
            final_wall_displacement: wall_disp,
            final_pressure_norm: l2_norm(pressure.as_slice()),
            final_pressure_checksum: checksum(pressure.as_slice()),
            sync_retries: restart.map(|state| state.observed_sync_retries).unwrap_or(0),
            rejected_sync_steps: restart.map(|state| state.rejected_sync_steps).unwrap_or(0),
            rollback_count: restart.map(|state| state.rollback_count).unwrap_or(0),
            pressure: pressure.as_slice().to_vec(),
            final_mesh: mesh,
        };
    }

    for step in completed_steps + 1..=args.steps {
        let time = step as f64 * args.dt;
        let inlet = 1.0 + args.inlet_amp * (2.0 * PI * time).sin();

        let mut step_converged = false;
        let mut step_iters = 0usize;

        for k in 0..args.max_coupling {
            let old_mesh = mesh.clone();

            mesh = ref_mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
                .into_iter()
                .filter(|&nid| (mesh.coords_of(nid)[1] - 1.0).abs() < 1.0e-12)
                .collect();

            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                let taper = (PI * p[0]).sin().powi(2);
                [0.0, wall_disp * taper]
            });

            let fixed_nodes = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(
                &mut mesh,
                &fixed_nodes,
                MeshMotionConfig {
                    omega: args.omega,
                    max_iters: args.smooth_iters,
                    tol: 1.0e-12,
                },
            );

            let src = H1Space::new(old_mesh, 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (p_transfer, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src,
                pressure.as_slice(),
                &dst,
                1.0e-12,
                4,
            )
            .expect("pressure transfer should succeed");
            max_transfer_abs_int_err =
                max_transfer_abs_int_err.max(report.absolute_integral_error_after);

            let p_solved = solve_pressure_on_mesh(&dst, inlet, &p_transfer);
            pressure = Vector::from_vec(p_solved);

            let load = top_boundary_average_pressure(&dst, pressure.as_slice());
            let target_disp = args.compliance * load;
            let new_disp = (1.0 - args.wall_relax) * wall_disp + args.wall_relax * target_disp;

            let rel = (new_disp - wall_disp).abs() / new_disp.abs().max(1.0e-12);
            wall_disp = new_disp;
            max_wall_disp = max_wall_disp.max(wall_disp.abs());

            step_iters = k + 1;
            if rel <= args.coupling_tol {
                step_converged = true;
                break;
            }
        }

        if step_converged {
            converged_steps += 1;
        }
        max_coupling_iters_used = max_coupling_iters_used.max(step_iters);
    }

    FsiTemplateResult {
        completed_steps: args.steps,
        converged_steps,
        steps: args.steps,
        max_coupling_iters_used,
        max_transfer_abs_int_err,
        max_wall_displacement: max_wall_disp,
        final_wall_displacement: wall_disp,
        final_pressure_norm: l2_norm(pressure.as_slice()),
        final_pressure_checksum: checksum(pressure.as_slice()),
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
        pressure: pressure.as_slice().to_vec(),
        final_mesh: mesh,
    }
}

fn solve_fsi_template_subcycling(
    args: &Args,
    restart: Option<&FsiCheckpointState>,
) -> FsiTemplateResult {
    #[derive(Clone)]
    struct SubcyclingState {
        mesh: Mesh<2>,
        pressure: Vector<f64>,
        wall_disp: f64,
        max_wall_disp: f64,
        max_transfer_abs_int_err: f64,
        converged_steps: usize,
    }

    let ref_mesh = Mesh::<2>::unit_square_tri(args.n);
    let init_wall_disp = restart.map(|state| state.final_wall_displacement).unwrap_or(0.0_f64);
    let init_mesh = restart
        .map(|state| mesh_from_checkpoint_coords(&ref_mesh, &state.mesh_coords))
        .unwrap_or_else(|| build_deformed_mesh(&ref_mesh, init_wall_disp, args.omega, args.smooth_iters));
    let init_pressure = if let Some(state) = restart {
        Vector::from_vec(state.pressure.clone())
    } else {
        H1Space::new(init_mesh.clone(), 1).interpolate(&|_x| 0.0)
    };
    let completed_steps = restart.map(|state| state.completed_steps).unwrap_or(0usize);
    let mut state = SubcyclingState {
        mesh: init_mesh,
        pressure: init_pressure,
        wall_disp: init_wall_disp,
        max_wall_disp: restart.map(|state| state.max_wall_displacement).unwrap_or(0.0),
        max_transfer_abs_int_err: restart
            .map(|state| state.max_transfer_abs_int_err)
            .unwrap_or(0.0),
        converged_steps: restart.map(|state| state.converged_steps).unwrap_or(0),
    };

    if completed_steps >= args.steps {
        return FsiTemplateResult {
            completed_steps,
            converged_steps: state.converged_steps,
            steps: completed_steps,
            max_coupling_iters_used: restart
                .map(|state| state.max_coupling_iters_used)
                .unwrap_or(1),
            max_transfer_abs_int_err: state.max_transfer_abs_int_err,
            max_wall_displacement: state.max_wall_disp,
            final_wall_displacement: state.wall_disp,
            final_pressure_norm: l2_norm(state.pressure.as_slice()),
            final_pressure_checksum: checksum(state.pressure.as_slice()),
            sync_retries: restart.map(|state| state.observed_sync_retries).unwrap_or(0),
            rejected_sync_steps: restart.map(|state| state.rejected_sync_steps).unwrap_or(0),
            rollback_count: restart.map(|state| state.rollback_count).unwrap_or(0),
            pressure: state.pressure.as_slice().to_vec(),
            final_mesh: state.mesh,
        };
    }

    let fast_dt = args.fast_dt.max(1.0e-12).min(args.dt);
    let cfg = MultiRateConfig {
        t_start: completed_steps as f64 * args.dt,
        t_end: args.steps as f64 * args.dt,
        fast_dt,
        slow_dt: args.dt,
    };

    let sync_policy = TemplateSyncPolicy {
        sync_error_tol: args.sync_error_tol,
        max_sync_retries: args.sync_retries,
        min_fast_dt: args.fast_dt_min.max(1.0e-12),
        retry_fast_dt_scale: 0.5,
        component_weights: vec![1.0, 1.0],
    };

    let stats = run_multirate_adaptive(
        sync_policy
            .adaptive_config(cfg)
            .expect("invalid FSI sync policy"),
        &mut state,
        |state, t_fast, dt_fast| {
            let t_next = t_fast + dt_fast;
            let inlet = 1.0 + args.inlet_amp * (2.0 * PI * t_next).sin();
            let space_fast = H1Space::new(state.mesh.clone(), 1);
            let p_solved = solve_pressure_on_mesh(&space_fast, inlet, state.pressure.as_slice());
            state.pressure = Vector::from_vec(p_solved);
        },
        |_state, _t_slow, _dt_slow| {
            // Structure update is handled on synchronization points.
        },
        |state, _t_sync| {
            let old_mesh = state.mesh.clone();

            state.mesh = ref_mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&state.mesh)
                .into_iter()
                .filter(|&nid| (state.mesh.coords_of(nid)[1] - 1.0).abs() < 1.0e-12)
                .collect();

            apply_node_displacement(&mut state.mesh, &top_nodes, |p| {
                let taper = (PI * p[0]).sin().powi(2);
                [0.0, state.wall_disp * taper]
            });

            let fixed_nodes = all_boundary_nodes(&state.mesh);
            let _ = laplacian_smooth_2d(
                &mut state.mesh,
                &fixed_nodes,
                MeshMotionConfig {
                    omega: args.omega,
                    max_iters: args.smooth_iters,
                    tol: 1.0e-12,
                },
            );

            let src = H1Space::new(old_mesh, 1);
            let dst = H1Space::new(state.mesh.clone(), 1);
            let (p_transfer, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src,
                state.pressure.as_slice(),
                &dst,
                1.0e-12,
                4,
            )
            .expect("pressure transfer should succeed");
            let sync_error = report.absolute_integral_error_after;
            state.pressure = Vector::from_vec(p_transfer);
            state.max_transfer_abs_int_err =
                state.max_transfer_abs_int_err.max(sync_error);

            let load = top_boundary_average_pressure(&dst, state.pressure.as_slice());
            let target_disp = args.compliance * load;
            let new_disp = (1.0 - args.wall_relax) * state.wall_disp + args.wall_relax * target_disp;
            let rel = (new_disp - state.wall_disp).abs() / new_disp.abs().max(1.0e-12);
            state.wall_disp = new_disp;
            state.max_wall_disp = state.max_wall_disp.max(state.wall_disp.abs());

            if rel <= args.coupling_tol {
                state.converged_steps += 1;
            }

            sync_policy.compose_error(&[sync_error, rel])
        },
    )
    .expect("adaptive subcycling scheduler failed");

    FsiTemplateResult {
        completed_steps: completed_steps + stats.sync_steps,
        converged_steps: state.converged_steps,
        steps: completed_steps + stats.sync_steps,
        max_coupling_iters_used: restart
            .map(|state| state.max_coupling_iters_used)
            .unwrap_or(1)
            .max(1),
        max_transfer_abs_int_err: state.max_transfer_abs_int_err,
        max_wall_displacement: state.max_wall_disp,
        final_wall_displacement: state.wall_disp,
        final_pressure_norm: l2_norm(state.pressure.as_slice()),
        final_pressure_checksum: checksum(state.pressure.as_slice()),
        sync_retries: restart.map(|state| state.observed_sync_retries).unwrap_or(0)
            + stats.sync_retries,
        rejected_sync_steps: restart.map(|state| state.rejected_sync_steps).unwrap_or(0)
            + stats.rejected_sync_steps,
        rollback_count: restart.map(|state| state.rollback_count).unwrap_or(0)
            + stats.rollback_count,
        pressure: state.pressure.as_slice().to_vec(),
        final_mesh: state.mesh,
    }
}

fn build_deformed_mesh(
    ref_mesh: &Mesh<2>,
    wall_disp: f64,
    omega: f64,
    smooth_iters: usize,
) -> Mesh<2> {
    let mut mesh = ref_mesh.clone();
    if wall_disp.abs() > 0.0 {
        let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
            .into_iter()
            .filter(|&nid| (mesh.coords_of(nid)[1] - 1.0).abs() < 1.0e-12)
            .collect();

        apply_node_displacement(&mut mesh, &top_nodes, |p| {
            let taper = (PI * p[0]).sin().powi(2);
            [0.0, wall_disp * taper]
        });

        let fixed_nodes = all_boundary_nodes(&mesh);
        let _ = laplacian_smooth_2d(
            &mut mesh,
            &fixed_nodes,
            MeshMotionConfig {
                omega,
                max_iters: smooth_iters,
                tol: 1.0e-12,
            },
        );
    }
    mesh
}

fn mesh_from_checkpoint_coords(ref_mesh: &Mesh<2>, coords: &[f64]) -> Mesh<2> {
    let mut mesh = ref_mesh.clone();
    mesh.coords = coords.to_vec();
    mesh
}

fn solve_pressure_on_mesh(
    space: &H1Space<Mesh<2>>,
    inlet: f64,
    initial_guess: &[f64],
) -> Vec<f64> {
    let n = space.n_dofs();
    let mut a = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        3,
    );
    let mut rhs = vec![0.0_f64; n];

    let dm = space.dof_manager();
    let left = boundary_dofs(space.mesh(), dm, &[4]);
    let right = boundary_dofs(space.mesh(), dm, &[2]);

    apply_dirichlet(&mut a, &mut rhs, &left, &vec![inlet; left.len()]);
    apply_dirichlet(&mut a, &mut rhs, &right, &vec![0.0; right.len()]);

    let mut p = if initial_guess.len() == n {
        initial_guess.to_vec()
    } else {
        vec![0.0_f64; n]
    };
    let cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };

    let _ = solve_pcg_jacobi(&a, &rhs, &mut p, &cfg)
        .or_else(|_| solve_gmres(&a, &rhs, &mut p, 60, &cfg))
        .expect("pressure solve failed");

    p
}

fn top_boundary_average_pressure(space: &H1Space<Mesh<2>>, p: &[f64]) -> f64 {
    let dm = space.dof_manager();
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for i in 0..space.n_dofs() {
        let x = dm.dof_coord(i as u32);
        if (x[1] - 1.0).abs() < 1.0e-10 {
            sum += p[i];
            cnt += 1;
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum / cnt as f64
    }
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn checksum(v: &[f64]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum()
}

fn write_ex49_vtk_export(
    prefix: &str,
    mesh: &Mesh<2>,
    pressure: &[f64],
) -> Result<(), String> {
    let path = format!("{prefix}_fsi_pressure.vtu");
    ensure_parent_dir(&path).map_err(|e| e.to_string())?;
    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("pressure", pressure.to_vec()));
    writer.write_file(&path).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_fsi_checkpoint(path: &str, state: &FsiCheckpointState) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let mesh_coords = format_vec_f64(&state.mesh_coords);
    let pressure = format_vec_f64(&state.pressure);
    let content = format!(
        "format=ex49_fsi_v1\nn={}\nsteps={}\ndt={:.17e}\nfast_dt={:.17e}\nsubcycling={}\ninlet_amp={:.17e}\ncompliance={:.17e}\nwall_relax={:.17e}\ncoupling_tol={:.17e}\nsync_error_tol={:.17e}\nmax_coupling={}\nsync_retries={}\nfast_dt_min={:.17e}\nomega={:.17e}\nsmooth_iters={}\ncompleted_steps={}\nconverged_steps={}\nmax_coupling_iters_used={}\nmax_transfer_abs_int_err={:.17e}\nmax_wall_displacement={:.17e}\nfinal_wall_displacement={:.17e}\nobserved_sync_retries={}\nrejected_sync_steps={}\nrollback_count={}\nmesh_coords={}\npressure={}\n",
        state.args.n,
        state.args.steps,
        state.args.dt,
        state.args.fast_dt,
        if state.args.use_subcycling { 1 } else { 0 },
        state.args.inlet_amp,
        state.args.compliance,
        state.args.wall_relax,
        state.args.coupling_tol,
        state.args.sync_error_tol,
        state.args.max_coupling,
        state.args.sync_retries,
        state.args.fast_dt_min,
        state.args.omega,
        state.args.smooth_iters,
        state.completed_steps,
        state.converged_steps,
        state.max_coupling_iters_used,
        state.max_transfer_abs_int_err,
        state.max_wall_displacement,
        state.final_wall_displacement,
        state.observed_sync_retries,
        state.rejected_sync_steps,
        state.rollback_count,
        mesh_coords,
        pressure,
    );
    fs::write(path, content)
}

fn write_fsi_hdf5_checkpoint(path: &str, state: &FsiCheckpointState) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(path);

    let bundle = CheckpointBundleF64 {
        mesh_meta: None,
        fields: vec![
            scalar_rank_field_f64("n", state.args.n as f64),
            scalar_rank_field_f64("steps", state.args.steps as f64),
            scalar_rank_field_f64("dt", state.args.dt),
            scalar_rank_field_f64("fast_dt", state.args.fast_dt),
            scalar_rank_field_f64("subcycling", if state.args.use_subcycling { 1.0 } else { 0.0 }),
            scalar_rank_field_f64("inlet_amp", state.args.inlet_amp),
            scalar_rank_field_f64("compliance", state.args.compliance),
            scalar_rank_field_f64("wall_relax", state.args.wall_relax),
            scalar_rank_field_f64("coupling_tol", state.args.coupling_tol),
            scalar_rank_field_f64("sync_error_tol", state.args.sync_error_tol),
            scalar_rank_field_f64("max_coupling", state.args.max_coupling as f64),
            scalar_rank_field_f64("sync_retries", state.args.sync_retries as f64),
            scalar_rank_field_f64("fast_dt_min", state.args.fast_dt_min),
            scalar_rank_field_f64("omega", state.args.omega),
            scalar_rank_field_f64("smooth_iters", state.args.smooth_iters as f64),
            scalar_rank_field_f64("completed_steps", state.completed_steps as f64),
            scalar_rank_field_f64("converged_steps", state.converged_steps as f64),
            scalar_rank_field_f64("max_coupling_iters_used", state.max_coupling_iters_used as f64),
            scalar_rank_field_f64("max_transfer_abs_int_err", state.max_transfer_abs_int_err),
            scalar_rank_field_f64("max_wall_displacement", state.max_wall_displacement),
            scalar_rank_field_f64("final_wall_displacement", state.final_wall_displacement),
            scalar_rank_field_f64("observed_sync_retries", state.observed_sync_retries as f64),
            scalar_rank_field_f64("rejected_sync_steps", state.rejected_sync_steps as f64),
            scalar_rank_field_f64("rollback_count", state.rollback_count as f64),
            vector_rank_field_f64("mesh_coords", state.mesh_coords.clone()),
            vector_rank_field_f64("pressure", state.pressure.clone()),
        ],
    };
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let step = state.completed_steps.max(1) as u64;
    write_checkpoint_step_bundle_f64(
        path,
        cfg,
        step,
        state.completed_steps.max(1) as f64,
        &bundle,
        IoBackend::Partitioned,
    )
    .map_err(|e| e.to_string())?;
    validate_checkpoint_layout(path, Some(1)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_fsi_checkpoint(path: &str) -> Result<FsiCheckpointState, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut format = None;
    let mut n = None;
    let mut steps = None;
    let mut dt = None;
    let mut fast_dt = None;
    let mut subcycling = None;
    let mut inlet_amp = None;
    let mut compliance = None;
    let mut wall_relax = None;
    let mut coupling_tol = None;
    let mut sync_error_tol = None;
    let mut max_coupling = None;
    let mut sync_retries = None;
    let mut fast_dt_min = None;
    let mut omega = None;
    let mut smooth_iters = None;
    let mut completed_steps = None;
    let mut converged_steps = None;
    let mut max_coupling_iters_used = None;
    let mut max_transfer_abs_int_err = None;
    let mut max_wall_displacement = None;
    let mut final_wall_displacement = None;
    let mut observed_sync_retries = None;
    let mut rejected_sync_steps = None;
    let mut rollback_count = None;
    let mut mesh_coords = None;
    let mut pressure = None;

    for line in content.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "format" => format = Some(value.to_string()),
            "n" => n = value.parse::<usize>().ok(),
            "steps" => steps = value.parse::<usize>().ok(),
            "dt" => dt = value.parse::<f64>().ok(),
            "fast_dt" => fast_dt = value.parse::<f64>().ok(),
            "subcycling" => subcycling = Some(value == "1"),
            "inlet_amp" => inlet_amp = value.parse::<f64>().ok(),
            "compliance" => compliance = value.parse::<f64>().ok(),
            "wall_relax" => wall_relax = value.parse::<f64>().ok(),
            "coupling_tol" => coupling_tol = value.parse::<f64>().ok(),
            "sync_error_tol" => sync_error_tol = value.parse::<f64>().ok(),
            "max_coupling" => max_coupling = value.parse::<usize>().ok(),
            "sync_retries" => sync_retries = value.parse::<usize>().ok(),
            "fast_dt_min" => fast_dt_min = value.parse::<f64>().ok(),
            "omega" => omega = value.parse::<f64>().ok(),
            "smooth_iters" => smooth_iters = value.parse::<usize>().ok(),
            "completed_steps" => completed_steps = value.parse::<usize>().ok(),
            "converged_steps" => converged_steps = value.parse::<usize>().ok(),
            "max_coupling_iters_used" => max_coupling_iters_used = value.parse::<usize>().ok(),
            "max_transfer_abs_int_err" => max_transfer_abs_int_err = value.parse::<f64>().ok(),
            "max_wall_displacement" => max_wall_displacement = value.parse::<f64>().ok(),
            "final_wall_displacement" => final_wall_displacement = value.parse::<f64>().ok(),
            "observed_sync_retries" => observed_sync_retries = value.parse::<usize>().ok(),
            "rejected_sync_steps" => rejected_sync_steps = value.parse::<usize>().ok(),
            "rollback_count" => rollback_count = value.parse::<usize>().ok(),
            "mesh_coords" => mesh_coords = Some(parse_vec_f64(value)?),
            "pressure" => pressure = Some(parse_vec_f64(value)?),
            _ => {}
        }
    }

    if format.as_deref() != Some("ex49_fsi_v1") {
        return Err("unsupported checkpoint format".into());
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        steps: steps.ok_or_else(|| "missing steps".to_string())?,
        dt: dt.ok_or_else(|| "missing dt".to_string())?,
        fast_dt: fast_dt.ok_or_else(|| "missing fast_dt".to_string())?,
        use_subcycling: subcycling.ok_or_else(|| "missing subcycling".to_string())?,
        inlet_amp: inlet_amp.ok_or_else(|| "missing inlet_amp".to_string())?,
        compliance: compliance.ok_or_else(|| "missing compliance".to_string())?,
        wall_relax: wall_relax.ok_or_else(|| "missing wall_relax".to_string())?,
        coupling_tol: coupling_tol.ok_or_else(|| "missing coupling_tol".to_string())?,
        sync_error_tol: sync_error_tol.ok_or_else(|| "missing sync_error_tol".to_string())?,
        max_coupling: max_coupling.ok_or_else(|| "missing max_coupling".to_string())?,
        sync_retries: sync_retries.ok_or_else(|| "missing sync_retries".to_string())?,
        fast_dt_min: fast_dt_min.ok_or_else(|| "missing fast_dt_min".to_string())?,
        omega: omega.ok_or_else(|| "missing omega".to_string())?,
        smooth_iters: smooth_iters.ok_or_else(|| "missing smooth_iters".to_string())?,
    };
    let mesh_coords = mesh_coords.ok_or_else(|| "missing mesh_coords".to_string())?;
    let pressure = pressure.ok_or_else(|| "missing pressure".to_string())?;
    let expected_coords = (args.n + 1) * (args.n + 1) * 2;
    if mesh_coords.len() != expected_coords {
        return Err(format!(
            "checkpoint mesh_coords length ({}) does not match expected coordinates ({expected_coords})",
            mesh_coords.len()
        ));
    }
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if pressure.len() != expected_dofs {
        return Err(format!(
            "checkpoint pressure length ({}) does not match expected dofs ({expected_dofs})",
            pressure.len()
        ));
    }

    Ok(FsiCheckpointState {
        args,
        completed_steps: completed_steps.ok_or_else(|| "missing completed_steps".to_string())?,
        converged_steps: converged_steps.ok_or_else(|| "missing converged_steps".to_string())?,
        max_coupling_iters_used: max_coupling_iters_used
            .ok_or_else(|| "missing max_coupling_iters_used".to_string())?,
        max_transfer_abs_int_err: max_transfer_abs_int_err
            .ok_or_else(|| "missing max_transfer_abs_int_err".to_string())?,
        max_wall_displacement: max_wall_displacement
            .ok_or_else(|| "missing max_wall_displacement".to_string())?,
        final_wall_displacement: final_wall_displacement
            .ok_or_else(|| "missing final_wall_displacement".to_string())?,
        observed_sync_retries: observed_sync_retries
            .ok_or_else(|| "missing observed_sync_retries".to_string())?,
        rejected_sync_steps: rejected_sync_steps
            .ok_or_else(|| "missing rejected_sync_steps".to_string())?,
        rollback_count: rollback_count.ok_or_else(|| "missing rollback_count".to_string())?,
        mesh_coords,
        pressure,
    })
}

fn read_fsi_hdf5_checkpoint(path: &str) -> Result<FsiCheckpointState, String> {
    let fields = read_checkpoint_fields_f64_latest(
        path,
        ParallelIoConfig { world_size: 1, rank: 0 },
        &[
            "n",
            "steps",
            "dt",
            "fast_dt",
            "subcycling",
            "inlet_amp",
            "compliance",
            "wall_relax",
            "coupling_tol",
            "sync_error_tol",
            "max_coupling",
            "sync_retries",
            "fast_dt_min",
            "omega",
            "smooth_iters",
            "completed_steps",
            "converged_steps",
            "max_coupling_iters_used",
            "max_transfer_abs_int_err",
            "max_wall_displacement",
            "final_wall_displacement",
            "observed_sync_retries",
            "rejected_sync_steps",
            "rollback_count",
            "mesh_coords",
            "pressure",
        ],
    )
    .map_err(|e| e.to_string())?;

    let mut n = None;
    let mut steps = None;
    let mut dt = None;
    let mut fast_dt = None;
    let mut subcycling = None;
    let mut inlet_amp = None;
    let mut compliance = None;
    let mut wall_relax = None;
    let mut coupling_tol = None;
    let mut sync_error_tol = None;
    let mut max_coupling = None;
    let mut sync_retries = None;
    let mut fast_dt_min = None;
    let mut omega = None;
    let mut smooth_iters = None;
    let mut completed_steps = None;
    let mut converged_steps = None;
    let mut max_coupling_iters_used = None;
    let mut max_transfer_abs_int_err = None;
    let mut max_wall_displacement = None;
    let mut final_wall_displacement = None;
    let mut observed_sync_retries = None;
    let mut rejected_sync_steps = None;
    let mut rollback_count = None;
    let mut mesh_coords = None;
    let mut pressure = None;

    for (name, field) in fields {
        match name.as_str() {
            "n" => n = field.values.first().map(|v| *v as usize),
            "steps" => steps = field.values.first().map(|v| *v as usize),
            "dt" => dt = field.values.first().copied(),
            "fast_dt" => fast_dt = field.values.first().copied(),
            "subcycling" => subcycling = field.values.first().map(|v| *v != 0.0),
            "inlet_amp" => inlet_amp = field.values.first().copied(),
            "compliance" => compliance = field.values.first().copied(),
            "wall_relax" => wall_relax = field.values.first().copied(),
            "coupling_tol" => coupling_tol = field.values.first().copied(),
            "sync_error_tol" => sync_error_tol = field.values.first().copied(),
            "max_coupling" => max_coupling = field.values.first().map(|v| *v as usize),
            "sync_retries" => sync_retries = field.values.first().map(|v| *v as usize),
            "fast_dt_min" => fast_dt_min = field.values.first().copied(),
            "omega" => omega = field.values.first().copied(),
            "smooth_iters" => smooth_iters = field.values.first().map(|v| *v as usize),
            "completed_steps" => completed_steps = field.values.first().map(|v| *v as usize),
            "converged_steps" => converged_steps = field.values.first().map(|v| *v as usize),
            "max_coupling_iters_used" => {
                max_coupling_iters_used = field.values.first().map(|v| *v as usize)
            }
            "max_transfer_abs_int_err" => max_transfer_abs_int_err = field.values.first().copied(),
            "max_wall_displacement" => max_wall_displacement = field.values.first().copied(),
            "final_wall_displacement" => final_wall_displacement = field.values.first().copied(),
            "observed_sync_retries" => observed_sync_retries = field.values.first().map(|v| *v as usize),
            "rejected_sync_steps" => rejected_sync_steps = field.values.first().map(|v| *v as usize),
            "rollback_count" => rollback_count = field.values.first().map(|v| *v as usize),
            "mesh_coords" => mesh_coords = Some(field.values),
            "pressure" => pressure = Some(field.values),
            _ => {}
        }
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        steps: steps.ok_or_else(|| "missing steps".to_string())?,
        dt: dt.ok_or_else(|| "missing dt".to_string())?,
        fast_dt: fast_dt.ok_or_else(|| "missing fast_dt".to_string())?,
        use_subcycling: subcycling.ok_or_else(|| "missing subcycling".to_string())?,
        inlet_amp: inlet_amp.ok_or_else(|| "missing inlet_amp".to_string())?,
        compliance: compliance.ok_or_else(|| "missing compliance".to_string())?,
        wall_relax: wall_relax.ok_or_else(|| "missing wall_relax".to_string())?,
        coupling_tol: coupling_tol.ok_or_else(|| "missing coupling_tol".to_string())?,
        sync_error_tol: sync_error_tol.ok_or_else(|| "missing sync_error_tol".to_string())?,
        max_coupling: max_coupling.ok_or_else(|| "missing max_coupling".to_string())?,
        sync_retries: sync_retries.ok_or_else(|| "missing sync_retries".to_string())?,
        fast_dt_min: fast_dt_min.ok_or_else(|| "missing fast_dt_min".to_string())?,
        omega: omega.ok_or_else(|| "missing omega".to_string())?,
        smooth_iters: smooth_iters.ok_or_else(|| "missing smooth_iters".to_string())?,
    };
    let mesh_coords = mesh_coords.ok_or_else(|| "missing mesh_coords".to_string())?;
    let pressure = pressure.ok_or_else(|| "missing pressure".to_string())?;
    let expected_coords = (args.n + 1) * (args.n + 1) * 2;
    if mesh_coords.len() != expected_coords {
        return Err(format!(
            "checkpoint mesh_coords length ({}) does not match expected coordinates ({expected_coords})",
            mesh_coords.len()
        ));
    }
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if pressure.len() != expected_dofs {
        return Err(format!(
            "checkpoint pressure length ({}) does not match expected dofs ({expected_dofs})",
            pressure.len()
        ));
    }

    Ok(FsiCheckpointState {
        args,
        completed_steps: completed_steps.ok_or_else(|| "missing completed_steps".to_string())?,
        converged_steps: converged_steps.ok_or_else(|| "missing converged_steps".to_string())?,
        max_coupling_iters_used: max_coupling_iters_used
            .ok_or_else(|| "missing max_coupling_iters_used".to_string())?,
        max_transfer_abs_int_err: max_transfer_abs_int_err
            .ok_or_else(|| "missing max_transfer_abs_int_err".to_string())?,
        max_wall_displacement: max_wall_displacement
            .ok_or_else(|| "missing max_wall_displacement".to_string())?,
        final_wall_displacement: final_wall_displacement
            .ok_or_else(|| "missing final_wall_displacement".to_string())?,
        observed_sync_retries: observed_sync_retries
            .ok_or_else(|| "missing observed_sync_retries".to_string())?,
        rejected_sync_steps: rejected_sync_steps
            .ok_or_else(|| "missing rejected_sync_steps".to_string())?,
        rollback_count: rollback_count.ok_or_else(|| "missing rollback_count".to_string())?,
        mesh_coords,
        pressure,
    })
}

#[cfg(feature = "io_hdf5")]
fn write_fsi_hdf5_xdmf_sidecars(h5_path: &str, state: &FsiCheckpointState) -> Result<(), String> {
    let step = state.completed_steps.max(1);
    write_scalar_checkpoint_xdmf_sidecars(h5_path, step as u64, step as f64, &["pressure"])
}

fn parse_args() -> CliArgs {
    let mut a = Args {
        n: 14,
        steps: 10,
        dt: 0.05,
        fast_dt: 0.01,
        use_subcycling: true,
        inlet_amp: 0.3,
        compliance: 0.02,
        wall_relax: 0.7,
        coupling_tol: 1.0e-7,
        sync_error_tol: 1.0,
        max_coupling: 12,
        sync_retries: 2,
        fast_dt_min: 1.0e-3,
        omega: 0.7,
        smooth_iters: 20,
    };
    let mut workflow = WorkflowCliOptions::default();

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex49_template_fsi");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        let mut help_options = vec![
            ("--n <int>", "Mesh resolution (default: 14)"),
            (
                "--steps <int>",
                "Number of slow synchronization steps (default: 10)",
            ),
            ("--dt <float>", "Slow-step size (default: 0.05)"),
            (
                "--fast-dt <float>",
                "Fast subcycling step size (default: 0.01)",
            ),
            ("--subcycling", "Enable multirate subcycling (default)"),
            ("--no-subcycling", "Disable subcycling and use single-rate loop"),
            ("--inlet-amp <float>", "Inlet forcing amplitude (default: 0.3)"),
            ("--compliance <float>", "Wall compliance coefficient (default: 0.02)"),
            (
                "--wall-relax <float>",
                "Wall relaxation factor in [0.1, 1.0] (default: 0.7)",
            ),
            (
                "--coupling-tol <float>",
                "Coupling convergence tolerance (default: 1e-7)",
            ),
            (
                "--sync-error-tol <float>",
                "Adaptive sync acceptance tolerance (default: 1.0)",
            ),
            (
                "--max-coupling <int>",
                "Maximum coupling iterations per slow step (default: 12)",
            ),
            (
                "--sync-retries <int>",
                "Max adaptive retry count at each sync point (default: 2)",
            ),
            (
                "--fast-dt-min <float>",
                "Minimum fast subcycling step during retries (default: 1e-3)",
            ),
            ("--omega <float>", "Mesh smoothing omega in [0.05, 0.95] (default: 0.7)"),
            (
                "--smooth-iters <int>",
                "Maximum Laplacian smoothing iterations (default: 20)",
            ),
        ];
        push_workflow_cli_help(
            &mut help_options,
            "Write final deformed-mesh pressure VTK export as <prefix>_fsi_pressure.vtu",
        );
        print_template_cli_help(bin, &help_options);
        std::process::exit(0);
    }

    let mut it = args_vec.into_iter().skip(1);
    while let Some(arg) = it.next() {
        if workflow.try_parse_arg(arg.as_str(), &mut it) {
            continue;
        }
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("14".into()).parse().unwrap_or(14),
            "--steps" => a.steps = it.next().unwrap_or("10".into()).parse().unwrap_or(10),
            "--dt" => a.dt = it.next().unwrap_or("0.05".into()).parse().unwrap_or(0.05),
            "--fast-dt" => a.fast_dt = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--subcycling" => a.use_subcycling = true,
            "--no-subcycling" => a.use_subcycling = false,
            "--inlet-amp" => {
                a.inlet_amp = it.next().unwrap_or("0.3".into()).parse().unwrap_or(0.3)
            }
            "--compliance" => {
                a.compliance = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02)
            }
            "--wall-relax" => {
                a.wall_relax = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7)
            }
            "--coupling-tol" | "--tol" => {
                a.coupling_tol = it.next().unwrap_or("1e-7".into()).parse().unwrap_or(1.0e-7)
            }
            "--sync-error-tol" => {
                a.sync_error_tol = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--max-coupling" => {
                a.max_coupling = it.next().unwrap_or("12".into()).parse().unwrap_or(12)
            }
            "--sync-retries" => {
                a.sync_retries = it.next().unwrap_or("2".into()).parse().unwrap_or(2)
            }
            "--fast-dt-min" => {
                a.fast_dt_min = it.next().unwrap_or("1e-3".into()).parse().unwrap_or(1.0e-3)
            }
            "--omega" => a.omega = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--smooth-iters" => {
                a.smooth_iters = it.next().unwrap_or("20".into()).parse().unwrap_or(20)
            }
            _ => {}
        }
    }

    a.steps = a.steps.max(1);
    a.fast_dt = a.fast_dt.max(1.0e-12);
    a.fast_dt_min = a.fast_dt_min.max(1.0e-12).min(a.fast_dt);
    a.sync_error_tol = a.sync_error_tol.max(0.0);
    a.max_coupling = a.max_coupling.max(1);
    a.wall_relax = a.wall_relax.clamp(0.1, 1.0);
    a.omega = a.omega.clamp(0.05, 0.95);
    CliArgs {
        sim: a,
        checkpoint: workflow.checkpoint,
        checkpoint_h5: workflow.checkpoint_h5,
        restart: workflow.restart,
        restart_h5: workflow.restart_h5,
        export_vtk_prefix: workflow.export_vtk_prefix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_output_path(tag: &str, ext: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "ex49_{}_{}_{}.{}",
                tag,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("valid clock")
                    .as_nanos(),
                ext
            ))
            .to_string_lossy()
            .to_string()
    }

    fn base_args() -> Args {
        Args {
            n: 8,
            steps: 4,
            dt: 0.05,
            fast_dt: 0.01,
            use_subcycling: true,
            inlet_amp: 0.2,
            compliance: 0.02,
            wall_relax: 0.7,
            coupling_tol: 1.0e-7,
            sync_error_tol: 1.0,
            max_coupling: 10,
            sync_retries: 2,
            fast_dt_min: 1.0e-3,
            omega: 0.7,
            smooth_iters: 12,
        }
    }

    #[test]
    fn ex49_fsi_template_runs_and_couples_motion() {
        let r = solve_fsi_template(&base_args());
        assert_eq!(r.steps, 4);
        assert!(r.max_coupling_iters_used <= 10);
        assert!(r.max_transfer_abs_int_err < 1.0e-9);
        assert!(r.max_wall_displacement > 0.0);
        assert!(r.final_pressure_norm > 0.0);
    }

    #[test]
    fn ex49_higher_compliance_gives_larger_wall_displacement() {
        let mut low = base_args();
        low.compliance = 0.01;
        let mut high = base_args();
        high.compliance = 0.04;

        let r_low = solve_fsi_template(&low);
        let r_high = solve_fsi_template(&high);

        assert!(r_high.max_wall_displacement > r_low.max_wall_displacement);
    }

    /// Very small compliance → near-rigid wall → negligible displacement.
    #[test]
    fn ex49_near_rigid_wall_gives_negligible_displacement() {
        let mut args = base_args();
        args.compliance = 1.0e-5;
        let r = solve_fsi_template(&args);
        // With near-zero compliance the wall barely moves.
        assert!(r.max_wall_displacement < 1.0e-3,
            "expected near-zero displacement for rigid wall: {:.4e}", r.max_wall_displacement);
    }

    /// Conservative transfer through the FSI coupling loop must not drift
    /// the fluid pressure integral.
    #[test]
    fn ex49_fluid_transfer_conserves_integral() {
        let r = solve_fsi_template(&base_args());
        assert!(r.max_transfer_abs_int_err < 1.0e-9,
            "fluid integral drifted too much: {:.3e}", r.max_transfer_abs_int_err);
    }

    /// Most coupling iterations must converge within the step budget
    /// (single-rate path; first step from cold start may not fully converge).
    #[test]
    fn ex49_all_steps_couple_to_convergence() {
        let mut args = base_args();
        args.use_subcycling = false;
        args.coupling_tol = 1.0e-5;
        let r = solve_fsi_template(&args);
        // Allow at most 1 non-converged step (cold-start transient).
        assert!(r.converged_steps >= r.steps.saturating_sub(1),
            "FSI coupling failed to converge in too many steps: {}/{}", r.converged_steps, r.steps);
    }

    /// Higher inlet amplitude drives larger wall displacement.
    #[test]
    fn ex49_higher_inlet_amplitude_drives_larger_wall_displacement() {
        let mut low = base_args();
        low.inlet_amp = 0.05;
        let mut high = base_args();
        high.inlet_amp = 0.4;

        let r_low  = solve_fsi_template(&low);
        let r_high = solve_fsi_template(&high);
        assert!(r_high.max_wall_displacement > r_low.max_wall_displacement,
            "expected higher inlet to drive larger displacement: low={:.4e} high={:.4e}",
            r_low.max_wall_displacement, r_high.max_wall_displacement);
    }

    /// Pressure checksum is identical across two runs with the same parameters (determinism).
    #[test]
    fn ex49_pressure_checksum_is_deterministic() {
        let r1 = solve_fsi_template(&base_args());
        let r2 = solve_fsi_template(&base_args());
        assert_eq!(r1.final_pressure_checksum, r2.final_pressure_checksum,
            "expected deterministic pressure checksum: r1={:.8e} r2={:.8e}",
            r1.final_pressure_checksum, r2.final_pressure_checksum);
    }

    /// Zero oscillation amplitude produces less wall displacement than the baseline (non-zero) amplitude.
    /// The inlet is `1 + amp*sin(...)`, so amp=0 still has a constant unit inlet; displacement is
    /// non-zero but must be smaller than with the full oscillating inlet.
    #[test]
    fn ex49_zero_inlet_amp_gives_less_displacement_than_baseline() {
        let mut args_base = base_args();
        let r_base = solve_fsi_template(&args_base);

        args_base.inlet_amp = 0.0;
        let r_zero = solve_fsi_template(&args_base);

        assert!(r_zero.max_wall_displacement < r_base.max_wall_displacement,
            "zero-amp displacement {:.4e} should be < baseline {:.4e}",
            r_zero.max_wall_displacement, r_base.max_wall_displacement);
    }

    #[test]
    fn ex49_text_checkpoint_roundtrip_preserves_restart_state() {
        let args = base_args();
        let partial = solve_fsi_template_with_restart(
            &Args {
                steps: 2,
                ..args.clone()
            },
            None,
        );
        let path = temp_output_path("checkpoint", "txt");
        let checkpoint = FsiCheckpointState {
            args: Args {
                steps: 2,
                ..args.clone()
            },
            completed_steps: partial.completed_steps,
            converged_steps: partial.converged_steps,
            max_coupling_iters_used: partial.max_coupling_iters_used,
            max_transfer_abs_int_err: partial.max_transfer_abs_int_err,
            max_wall_displacement: partial.max_wall_displacement,
            final_wall_displacement: partial.final_wall_displacement,
            observed_sync_retries: partial.sync_retries,
            rejected_sync_steps: partial.rejected_sync_steps,
            rollback_count: partial.rollback_count,
            mesh_coords: partial.final_mesh.coords.clone(),
            pressure: partial.pressure.clone(),
        };

        write_fsi_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_fsi_checkpoint(&path).unwrap();

        assert_eq!(restored.completed_steps, checkpoint.completed_steps);
        assert_eq!(restored.args.n, checkpoint.args.n);
        assert_eq!(restored.args.use_subcycling, checkpoint.args.use_subcycling);
        assert_eq!(restored.pressure, checkpoint.pressure);

        let resumed = solve_fsi_template_with_restart(&args, Some(&restored));
        let full = solve_fsi_template(&args);
        assert!((resumed.final_pressure_checksum - full.final_pressure_checksum).abs() < 1.0e-12);
        assert!((resumed.final_wall_displacement - full.final_wall_displacement).abs() < 1.0e-12);
        assert!((resumed.max_transfer_abs_int_err - full.max_transfer_abs_int_err).abs() < 1.0e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn ex49_vtk_export_writes_pressure_file() {
        let result = solve_fsi_template(&base_args());
        let prefix = temp_output_path("vtk", "out");
        let vtk_path = format!("{prefix}_fsi_pressure.vtu");

        write_ex49_vtk_export(&prefix, &result.final_mesh, &result.pressure).unwrap();

        let vtk = fs::read_to_string(&vtk_path).unwrap();
        assert!(vtk.contains("pressure"));

        let _ = fs::remove_file(vtk_path);
    }

    #[test]
    fn ex49_hdf5_checkpoint_roundtrip_preserves_restart_state() {
        let args = base_args();
        let partial = solve_fsi_template_with_restart(
            &Args {
                steps: 2,
                ..args.clone()
            },
            None,
        );
        let path = temp_output_path("checkpoint_h5", "h5");
        let checkpoint = FsiCheckpointState {
            args: Args {
                steps: 2,
                ..args.clone()
            },
            completed_steps: partial.completed_steps,
            converged_steps: partial.converged_steps,
            max_coupling_iters_used: partial.max_coupling_iters_used,
            max_transfer_abs_int_err: partial.max_transfer_abs_int_err,
            max_wall_displacement: partial.max_wall_displacement,
            final_wall_displacement: partial.final_wall_displacement,
            observed_sync_retries: partial.sync_retries,
            rejected_sync_steps: partial.rejected_sync_steps,
            rollback_count: partial.rollback_count,
            mesh_coords: partial.final_mesh.coords.clone(),
            pressure: partial.pressure.clone(),
        };

        write_fsi_hdf5_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_fsi_hdf5_checkpoint(&path).unwrap();

        assert_eq!(restored.completed_steps, checkpoint.completed_steps);
        assert_eq!(restored.args.n, checkpoint.args.n);
        assert_eq!(restored.args.use_subcycling, checkpoint.args.use_subcycling);
        assert_eq!(restored.mesh_coords, checkpoint.mesh_coords);
        assert_eq!(restored.pressure, checkpoint.pressure);

        let resumed = solve_fsi_template_with_restart(&args, Some(&restored));
        let full = solve_fsi_template(&args);
        assert!((resumed.final_pressure_checksum - full.final_pressure_checksum).abs() < 1.0e-12);
        assert!((resumed.final_wall_displacement - full.final_wall_displacement).abs() < 1.0e-12);
        assert!((resumed.max_transfer_abs_int_err - full.max_transfer_abs_int_err).abs() < 1.0e-12);

        let _ = fs::remove_file(path);
    }

    #[cfg(feature = "io_hdf5")]
    #[test]
    fn ex49_hdf5_checkpoint_writes_xdmf_sidecar() {
        let args = base_args();
        let result = solve_fsi_template(&args);
        let h5_path = temp_output_path("checkpoint_sidecar", "h5");
        let sidecar = checkpoint_sidecar_path(&h5_path, "pressure").unwrap();
        let checkpoint = FsiCheckpointState {
            args,
            completed_steps: result.completed_steps,
            converged_steps: result.converged_steps,
            max_coupling_iters_used: result.max_coupling_iters_used,
            max_transfer_abs_int_err: result.max_transfer_abs_int_err,
            max_wall_displacement: result.max_wall_displacement,
            final_wall_displacement: result.final_wall_displacement,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            mesh_coords: result.final_mesh.coords.clone(),
            pressure: result.pressure.clone(),
        };

        write_fsi_hdf5_checkpoint(&h5_path, &checkpoint).unwrap();
        write_fsi_hdf5_xdmf_sidecars(&h5_path, &checkpoint).unwrap();

        let xdmf = fs::read_to_string(&sidecar).unwrap();
        assert!(xdmf.contains("pressure"));

        let _ = fs::remove_file(h5_path);
        let _ = fs::remove_file(sidecar);
    }
}
