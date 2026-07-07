//! Example 52: built-in template driver - Reaction Flow Thermal.
//!
//! This template demonstrates a practical chemistry-flow-thermal coupling loop:
//! - flow proxy: pressure solve and flow metric from pressure drop / viscosity
//! - species field: diffusion + reaction-consumption source
//! - thermal field: diffusion + reaction heat-release source

use std::{
    f64::consts::PI,
    fs,
    io,
};

use fem_assembly::{
    Assembler,
    coefficient::FnCoeff,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_examples::checkpoint_text::{ensure_parent_dir, format_vec_f64, parse_vec_f64};
use fem_examples::template_runner::{
    maybe_write_template_kpi_csv,
    TemplateAdaptiveSummary,
    TemplateCouplingSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
    print_template_coupling_summary,
    print_template_header,
};
use fem_examples::hdf5_checkpoint::{scalar_rank_field_f64, vector_rank_field_f64};
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
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{
    BuiltinMultiphysicsTemplate,
    MultiRateAdaptiveConfig,
    MultiRateConfig,
    RelativeScalarTracker,
    SolverConfig,
    builtin_template_spec,
    compose_weighted_sync_error,
    run_multirate_adaptive,
    solve_gmres,
    solve_pcg_jacobi,
};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

struct ReactionFlowThermalResult {
    completed_steps: usize,
    steps: usize,
    converged_steps: usize,
    max_coupling_iters_used: usize,
    max_reaction_rate: f64,
    final_flow_metric: f64,
    final_species_norm: f64,
    final_temperature_norm: f64,
    final_species_checksum: f64,
    final_temperature_checksum: f64,
    sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
    flow_metric_tracker_prev: Option<f64>,
    rate_peak_tracker_prev: Option<f64>,
    species: Vec<f64>,
    temperature: Vec<f64>,
    mesh: Mesh<2>,
}

#[derive(Clone)]
struct Args {
    n: usize,
    steps: usize,
    dt: f64,
    fast_dt: f64,
    use_subcycling: bool,
    inlet_concentration: f64,
    flow_drive_amp: f64,
    k_species: f64,
    k_thermal: f64,
    reaction_k0: f64,
    reaction_temp_coeff: f64,
    heat_release: f64,
    viscosity0: f64,
    viscosity_temp_coeff: f64,
    viscosity_species_coeff: f64,
    relax: f64,
    coupling_tol: f64,
    sync_error_tol: f64,
    w_residual: f64,
    w_flow_metric: f64,
    w_rate_peak: f64,
    max_coupling: usize,
    sync_retries: usize,
    fast_dt_min: f64,
}

struct CliArgs {
    sim: Args,
    checkpoint: Option<String>,
    checkpoint_h5: Option<String>,
    restart: Option<String>,
    restart_h5: Option<String>,
    export_vtk_prefix: Option<String>,
}

struct ReactionFlowThermalCheckpointState {
    args: Args,
    completed_steps: usize,
    current_time: f64,
    converged_steps: usize,
    max_coupling_iters_used: usize,
    max_reaction_rate: f64,
    final_flow_metric: f64,
    observed_sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
    flow_metric_tracker_prev: Option<f64>,
    rate_peak_tracker_prev: Option<f64>,
    species: Vec<f64>,
    temperature: Vec<f64>,
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
        .map(read_reaction_flow_thermal_checkpoint)
        .transpose()
        .unwrap_or_else(|e| panic!("failed to read restart state: {e}"))
        .or_else(|| {
            cli.restart_h5
                .as_deref()
                .map(read_reaction_flow_thermal_hdf5_checkpoint)
                .transpose()
                .unwrap_or_else(|e| panic!("failed to read HDF5 restart state: {e}"))
        });
    let mut args = cli.sim.clone();
    if let Some(state) = restart_state.as_ref() {
        let requested_steps = args.steps.max(state.completed_steps);
        args = state.args.clone();
        args.steps = requested_steps;
    }
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::ReactionFlowThermal);

    let config_line = format!(
        "n={}, steps={}, dt={}, fast_dt={}, fast_dt_min={}, subcycling={}, inlet_concentration={}, flow_drive_amp={}, reaction_k0={}, heat_release={}, coupling_tol={}, sync_error_tol={}, w_residual={}, w_flow_metric={}, w_rate_peak={}, sync_retries={}",
        args.n,
        args.steps,
        args.dt,
        args.fast_dt,
        args.fast_dt_min,
        args.use_subcycling,
        args.inlet_concentration,
        args.flow_drive_amp,
        args.reaction_k0,
        args.heat_release,
        args.coupling_tol,
        args.sync_error_tol,
        args.w_residual,
        args.w_flow_metric,
        args.w_rate_peak,
        args.sync_retries,
    );
    print_template_header("Example 52: Built-in template driver", spec, &config_line);

    let result = if let Some(restart) = restart_state.as_ref() {
        solve_reaction_flow_thermal_template_with_restart(&args, Some(restart))
    } else {
        solve_reaction_flow_thermal_template(&args)
    };

    let coupling = TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.converged_steps,
        max_coupling_iters_used: result.max_coupling_iters_used,
    };
    print_template_coupling_summary(coupling);
    println!("  max reaction rate: {:.6e}", result.max_reaction_rate);
    println!("  final flow metric: {:.6e}", result.final_flow_metric);
    println!("  final ||species||_2: {:.6e}", result.final_species_norm);
    println!("  final ||temperature||_2: {:.6e}", result.final_temperature_norm);
    println!("  final species checksum: {:.8e}", result.final_species_checksum);
    println!("  final temperature checksum: {:.8e}", result.final_temperature_checksum);
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
            ("max_reaction_rate", result.max_reaction_rate),
            ("final_flow_metric", result.final_flow_metric),
            ("final_species_norm", result.final_species_norm),
            ("final_temperature_norm", result.final_temperature_norm),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }

    if let Some(path) = &cli.checkpoint {
        let checkpoint = ReactionFlowThermalCheckpointState {
            args: args.clone(),
            completed_steps: result.completed_steps,
            current_time: result.completed_steps as f64 * args.dt,
            converged_steps: result.converged_steps,
            max_coupling_iters_used: result.max_coupling_iters_used,
            max_reaction_rate: result.max_reaction_rate,
            final_flow_metric: result.final_flow_metric,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            flow_metric_tracker_prev: result.flow_metric_tracker_prev,
            rate_peak_tracker_prev: result.rate_peak_tracker_prev,
            species: result.species.clone(),
            temperature: result.temperature.clone(),
        };
        if let Err(e) = write_reaction_flow_thermal_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write checkpoint: {e}");
        } else {
            println!("  checkpoint written: {path}");
        }
    }

    if let Some(path) = &cli.checkpoint_h5 {
        let checkpoint = ReactionFlowThermalCheckpointState {
            args: args.clone(),
            completed_steps: result.completed_steps,
            current_time: result.completed_steps as f64 * args.dt,
            converged_steps: result.converged_steps,
            max_coupling_iters_used: result.max_coupling_iters_used,
            max_reaction_rate: result.max_reaction_rate,
            final_flow_metric: result.final_flow_metric,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            flow_metric_tracker_prev: result.flow_metric_tracker_prev,
            rate_peak_tracker_prev: result.rate_peak_tracker_prev,
            species: result.species.clone(),
            temperature: result.temperature.clone(),
        };
        if let Err(e) = write_reaction_flow_thermal_hdf5_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write HDF5 checkpoint: {e}");
        } else {
            println!("  HDF5 checkpoint written: {path}");
            #[cfg(feature = "io_hdf5")]
            if let Err(e) = write_reaction_flow_thermal_hdf5_xdmf_sidecars(path, &checkpoint) {
                eprintln!("warning: failed to write checkpoint XDMF sidecars: {e}");
            }
        }
    }

    if let Some(prefix) = &cli.export_vtk_prefix {
        if let Err(e) = write_ex52_vtk_export(prefix, &result.mesh, &result.species, &result.temperature)
        {
            eprintln!("warning: failed to write VTK export: {e}");
        } else {
            println!("  VTK export written: {prefix}_reaction_flow_thermal.vtu");
        }
    }
}

fn solve_reaction_flow_thermal_template(args: &Args) -> ReactionFlowThermalResult {
    solve_reaction_flow_thermal_template_with_restart(args, None)
}

fn solve_reaction_flow_thermal_template_with_restart(
    args: &Args,
    restart: Option<&ReactionFlowThermalCheckpointState>,
) -> ReactionFlowThermalResult {
    if args.use_subcycling {
        solve_reaction_flow_thermal_template_subcycling(args, restart)
    } else {
        solve_reaction_flow_thermal_template_single_rate(args, restart)
    }
}

fn solve_reaction_flow_thermal_template_single_rate(
    args: &Args,
    restart: Option<&ReactionFlowThermalCheckpointState>,
) -> ReactionFlowThermalResult {
    let mesh = Mesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    let completed_steps = restart.map(|state| state.completed_steps).unwrap_or(0);
    let mut c = restart
        .map(|state| state.species.clone())
        .unwrap_or_else(|| vec![0.0_f64; n]);
    let mut t = restart
        .map(|state| state.temperature.clone())
        .unwrap_or_else(|| vec![0.0_f64; n]);

    let mut converged_steps = restart.map(|state| state.converged_steps).unwrap_or(0);
    let mut max_coupling_iters_used = restart
        .map(|state| state.max_coupling_iters_used)
        .unwrap_or(0);
    let mut max_reaction_rate = restart.map(|state| state.max_reaction_rate).unwrap_or(0.0);
    let mut final_flow_metric = restart.map(|state| state.final_flow_metric).unwrap_or(0.0);

    if completed_steps >= args.steps {
        return ReactionFlowThermalResult {
            completed_steps,
            steps: completed_steps,
            converged_steps,
            max_coupling_iters_used,
            max_reaction_rate,
            final_flow_metric,
            final_species_norm: l2_norm(&c),
            final_temperature_norm: l2_norm(&t),
            final_species_checksum: checksum(&c),
            final_temperature_checksum: checksum(&t),
            sync_retries: restart.map(|state| state.observed_sync_retries).unwrap_or(0),
            rejected_sync_steps: restart.map(|state| state.rejected_sync_steps).unwrap_or(0),
            rollback_count: restart.map(|state| state.rollback_count).unwrap_or(0),
            flow_metric_tracker_prev: None,
            rate_peak_tracker_prev: None,
            species: c,
            temperature: t,
            mesh: space.mesh().clone(),
        };
    }

    for step in completed_steps + 1..=args.steps {
        let time = step as f64 * args.dt;
        let drive = 1.0 + args.flow_drive_amp * (2.0 * PI * time).sin();

        let mut step_converged = false;
        let mut step_iters = 0usize;

        for k in 0..args.max_coupling {
            let mean_t = t.iter().sum::<f64>() / n as f64;
            let mean_c = c.iter().sum::<f64>() / n as f64;
            let viscosity = (args.viscosity0
                * (1.0 + args.viscosity_temp_coeff * mean_t)
                * (1.0 + args.viscosity_species_coeff * mean_c))
                .max(1.0e-12);

            let p = solve_pressure_proxy(&space, drive);
            final_flow_metric = pressure_drop_metric(&space, &p) / viscosity;

            let rate_nodal: Vec<f64> = c
                .iter()
                .zip(t.iter())
                .map(|(&ci, &ti)| {
                    let r = args.reaction_k0 * ci.max(0.0) * (args.reaction_temp_coeff * ti).exp();
                    r.max(0.0)
                })
                .collect();
            let rate_max_step = rate_nodal.iter().copied().fold(0.0_f64, f64::max);
            max_reaction_rate = max_reaction_rate.max(rate_max_step);

            let c_new = solve_species(
                &space,
                &c,
                &rate_nodal,
                args.k_species,
                args.inlet_concentration,
            );

            let t_new = solve_temperature(
                &space,
                &t,
                &rate_nodal,
                args.k_thermal,
                args.heat_release,
            );

            let c_relaxed: Vec<f64> = c
                .iter()
                .zip(c_new.iter())
                .map(|(&old, &newv)| ((1.0 - args.relax) * old + args.relax * newv).max(0.0))
                .collect();
            let t_relaxed: Vec<f64> = t
                .iter()
                .zip(t_new.iter())
                .map(|(&old, &newv)| (1.0 - args.relax) * old + args.relax * newv)
                .collect();

            let rel_c = relative_change(&c, &c_relaxed);
            let rel_t = relative_change(&t, &t_relaxed);
            let rel = rel_c.max(rel_t);

            c = c_relaxed;
            t = t_relaxed;

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

    ReactionFlowThermalResult {
        completed_steps: args.steps,
        steps: args.steps,
        converged_steps,
        max_coupling_iters_used,
        max_reaction_rate,
        final_flow_metric,
        final_species_norm: l2_norm(&c),
        final_temperature_norm: l2_norm(&t),
        final_species_checksum: checksum(&c),
        final_temperature_checksum: checksum(&t),
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
        flow_metric_tracker_prev: None,
        rate_peak_tracker_prev: None,
        species: c,
        temperature: t,
        mesh: space.mesh().clone(),
    }
}

fn solve_reaction_flow_thermal_template_subcycling(
    args: &Args,
    restart: Option<&ReactionFlowThermalCheckpointState>,
) -> ReactionFlowThermalResult {
    #[derive(Clone)]
    struct SubcyclingState {
        c: Vec<f64>,
        t: Vec<f64>,
        rate_nodal: Vec<f64>,
        current_rate_peak: f64,
        rate_peak_tracker: RelativeScalarTracker,
        max_reaction_rate: f64,
        final_flow_metric: f64,
        flow_metric_tracker: RelativeScalarTracker,
        converged_steps: usize,
        last_rel: f64,
        sync_error: f64,
    }

    let mesh = Mesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();
    let completed_steps = restart.map(|state| state.completed_steps).unwrap_or(0);
    let mut state = SubcyclingState {
        c: restart
            .map(|checkpoint| checkpoint.species.clone())
            .unwrap_or_else(|| vec![0.0_f64; n]),
        t: restart
            .map(|checkpoint| checkpoint.temperature.clone())
            .unwrap_or_else(|| vec![0.0_f64; n]),
        rate_nodal: vec![0.0_f64; n],
        current_rate_peak: 0.0,
        rate_peak_tracker: seeded_scalar_tracker(
            restart.and_then(|checkpoint| checkpoint.rate_peak_tracker_prev),
        ),
        max_reaction_rate: restart
            .map(|checkpoint| checkpoint.max_reaction_rate)
            .unwrap_or(0.0),
        final_flow_metric: restart
            .map(|checkpoint| checkpoint.final_flow_metric)
            .unwrap_or(0.0),
        flow_metric_tracker: seeded_scalar_tracker(
            restart.and_then(|checkpoint| checkpoint.flow_metric_tracker_prev),
        ),
        converged_steps: restart.map(|checkpoint| checkpoint.converged_steps).unwrap_or(0),
        last_rel: 0.0,
        sync_error: 0.0,
    };

    if completed_steps >= args.steps {
        return ReactionFlowThermalResult {
            completed_steps,
            steps: completed_steps,
            converged_steps: state.converged_steps,
            max_coupling_iters_used: restart
                .map(|checkpoint| checkpoint.max_coupling_iters_used)
                .unwrap_or(1),
            max_reaction_rate: state.max_reaction_rate,
            final_flow_metric: state.final_flow_metric,
            final_species_norm: l2_norm(&state.c),
            final_temperature_norm: l2_norm(&state.t),
            final_species_checksum: checksum(&state.c),
            final_temperature_checksum: checksum(&state.t),
            sync_retries: restart.map(|checkpoint| checkpoint.observed_sync_retries).unwrap_or(0),
            rejected_sync_steps: restart.map(|checkpoint| checkpoint.rejected_sync_steps).unwrap_or(0),
            rollback_count: restart.map(|checkpoint| checkpoint.rollback_count).unwrap_or(0),
            flow_metric_tracker_prev: restart.and_then(|checkpoint| checkpoint.flow_metric_tracker_prev),
            rate_peak_tracker_prev: restart.and_then(|checkpoint| checkpoint.rate_peak_tracker_prev),
            species: state.c,
            temperature: state.t,
            mesh: space.mesh().clone(),
        };
    }

    let fast_dt = args.fast_dt.max(1.0e-12).min(args.dt);
    let cfg = MultiRateConfig {
        t_start: restart.map(|checkpoint| checkpoint.current_time).unwrap_or(0.0),
        t_end: args.steps as f64 * args.dt,
        fast_dt,
        slow_dt: args.dt,
    };

    let stats = run_multirate_adaptive(
        MultiRateAdaptiveConfig {
            base: cfg,
            sync_error_tol: args.sync_error_tol,
            max_sync_retries: args.sync_retries,
            retry_fast_dt_scale: 0.5,
            min_fast_dt: args.fast_dt_min.max(1.0e-12),
        },
        &mut state,
        |state, t_fast, dt_fast| {
            let time = t_fast + dt_fast;
            let drive = 1.0 + args.flow_drive_amp * (2.0 * PI * time).sin();

            let mean_t = state.t.iter().sum::<f64>() / n as f64;
            let mean_c = state.c.iter().sum::<f64>() / n as f64;
            let viscosity = (args.viscosity0
                * (1.0 + args.viscosity_temp_coeff * mean_t)
                * (1.0 + args.viscosity_species_coeff * mean_c))
                .max(1.0e-12);

            let p = solve_pressure_proxy(&space, drive);
            state.final_flow_metric = pressure_drop_metric(&space, &p) / viscosity;

            state.rate_nodal = state
                .c
                .iter()
                .zip(state.t.iter())
                .map(|(&ci, &ti)| {
                    let r = args.reaction_k0 * ci.max(0.0) * (args.reaction_temp_coeff * ti).exp();
                    r.max(0.0)
                })
                .collect();
            let rate_max_step = state.rate_nodal.iter().copied().fold(0.0_f64, f64::max);
            state.current_rate_peak = rate_max_step;
            state.max_reaction_rate = state.max_reaction_rate.max(rate_max_step);
        },
        |state, _t_slow, _dt_slow| {
            let c_new = solve_species(
                &space,
                &state.c,
                &state.rate_nodal,
                args.k_species,
                args.inlet_concentration,
            );
            let t_new = solve_temperature(
                &space,
                &state.t,
                &state.rate_nodal,
                args.k_thermal,
                args.heat_release,
            );

            let c_relaxed: Vec<f64> = state
                .c
                .iter()
                .zip(c_new.iter())
                .map(|(&old, &newv)| ((1.0 - args.relax) * old + args.relax * newv).max(0.0))
                .collect();
            let t_relaxed: Vec<f64> = state
                .t
                .iter()
                .zip(t_new.iter())
                .map(|(&old, &newv)| (1.0 - args.relax) * old + args.relax * newv)
                .collect();

            let rel_c = relative_change(&state.c, &c_relaxed);
            let rel_t = relative_change(&state.t, &t_relaxed);
            state.last_rel = rel_c.max(rel_t);

            state.c = c_relaxed;
            state.t = t_relaxed;
        },
        |state, _t_sync| {
            let rel_flow = state
                .flow_metric_tracker
                .observe(state.final_flow_metric, state.last_rel);
            let rel_rate = state
                .rate_peak_tracker
                .observe(state.current_rate_peak, state.last_rel);
            state.sync_error = compose_weighted_sync_error(
                &[state.last_rel, rel_flow, rel_rate],
                &[args.w_residual, args.w_flow_metric, args.w_rate_peak],
            );

            if state.last_rel <= args.coupling_tol {
                state.converged_steps += 1;
            }

            state.sync_error
        },
    )
    .expect("adaptive subcycling scheduler failed");

    ReactionFlowThermalResult {
        completed_steps: completed_steps + stats.sync_steps,
        steps: completed_steps + stats.sync_steps,
        converged_steps: state.converged_steps,
        max_coupling_iters_used: restart
            .map(|checkpoint| checkpoint.max_coupling_iters_used)
            .unwrap_or(1)
            .max(1),
        max_reaction_rate: state.max_reaction_rate,
        final_flow_metric: state.final_flow_metric,
        final_species_norm: l2_norm(&state.c),
        final_temperature_norm: l2_norm(&state.t),
        final_species_checksum: checksum(&state.c),
        final_temperature_checksum: checksum(&state.t),
        sync_retries: restart.map(|checkpoint| checkpoint.observed_sync_retries).unwrap_or(0)
            + stats.sync_retries,
        rejected_sync_steps: restart.map(|checkpoint| checkpoint.rejected_sync_steps).unwrap_or(0)
            + stats.rejected_sync_steps,
        rollback_count: restart.map(|checkpoint| checkpoint.rollback_count).unwrap_or(0)
            + stats.rollback_count,
        flow_metric_tracker_prev: Some(state.final_flow_metric),
        rate_peak_tracker_prev: Some(state.current_rate_peak),
        species: state.c,
        temperature: state.t,
        mesh: space.mesh().clone(),
    }
}

fn seeded_scalar_tracker(prev: Option<f64>) -> RelativeScalarTracker {
    let mut tracker = RelativeScalarTracker::new();
    if let Some(value) = prev {
        let _ = tracker.observe(value, 0.0);
    }
    tracker
}

fn solve_pressure_proxy(space: &H1Space<Mesh<2>>, drive: f64) -> Vec<f64> {
    let mut a = Assembler::assemble_bilinear(space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let mut rhs = vec![0.0_f64; space.n_dofs()];

    let dm = space.dof_manager();
    let left = boundary_dofs(space.mesh(), dm, &[4]);
    let right = boundary_dofs(space.mesh(), dm, &[2]);
    apply_dirichlet(&mut a, &mut rhs, &left, &vec![drive; left.len()]);
    apply_dirichlet(&mut a, &mut rhs, &right, &vec![0.0; right.len()]);

    let cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut p = vec![0.0_f64; space.n_dofs()];
    let _ = solve_pcg_jacobi(&a, &rhs, &mut p, &cfg)
        .or_else(|_| solve_gmres(&a, &rhs, &mut p, 60, &cfg))
        .expect("flow pressure solve failed");
    p
}

fn solve_species(
    space: &H1Space<Mesh<2>>,
    initial_guess: &[f64],
    rate_nodal: &[f64],
    k_species: f64,
    inlet_concentration: f64,
) -> Vec<f64> {
    let rate_coeff = FnCoeff(|x: &[f64]| sample_nodal_field(space, rate_nodal, x));
    let sink = DomainSourceIntegrator::new(|x: &[f64]| -sample_nodal_field(space, rate_nodal, x));

    let mut a = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator { kappa: k_species }, &DiffusionIntegrator { kappa: rate_coeff }],
        3,
    );
    let mut rhs = Assembler::assemble_linear(space, &[&sink], 3);

    let dm = space.dof_manager();
    let left = boundary_dofs(space.mesh(), dm, &[4]);
    let right = boundary_dofs(space.mesh(), dm, &[2]);
    apply_dirichlet(
        &mut a,
        &mut rhs,
        &left,
        &vec![inlet_concentration; left.len()],
    );
    apply_dirichlet(&mut a, &mut rhs, &right, &vec![0.0; right.len()]);

    let cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut c = if initial_guess.len() == space.n_dofs() {
        initial_guess.to_vec()
    } else {
        vec![0.0_f64; space.n_dofs()]
    };
    let _ = solve_pcg_jacobi(&a, &rhs, &mut c, &cfg)
        .or_else(|_| solve_gmres(&a, &rhs, &mut c, 60, &cfg))
        .expect("species solve failed");
    c
}

fn solve_temperature(
    space: &H1Space<Mesh<2>>,
    initial_guess: &[f64],
    rate_nodal: &[f64],
    k_thermal: f64,
    heat_release: f64,
) -> Vec<f64> {
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        heat_release * sample_nodal_field(space, rate_nodal, x)
    });

    let mut a = Assembler::assemble_bilinear(space, &[&DiffusionIntegrator { kappa: k_thermal }], 3);
    let mut rhs = Assembler::assemble_linear(space, &[&source], 3);

    let dm = space.dof_manager();
    let all = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
    apply_dirichlet(&mut a, &mut rhs, &all, &vec![0.0; all.len()]);

    let cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut t = if initial_guess.len() == space.n_dofs() {
        initial_guess.to_vec()
    } else {
        vec![0.0_f64; space.n_dofs()]
    };
    let _ = solve_pcg_jacobi(&a, &rhs, &mut t, &cfg)
        .or_else(|_| solve_gmres(&a, &rhs, &mut t, 60, &cfg))
        .expect("thermal solve failed");
    t
}

fn sample_nodal_field(space: &H1Space<Mesh<2>>, field: &[f64], x: &[f64]) -> f64 {
    let mesh = space.mesh();
    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        let a = mesh.coords_of(ns[0]);
        let b = mesh.coords_of(ns[1]);
        let c = mesh.coords_of(ns[2]);
        if let Some((l0, l1, l2)) = barycentric_2d(x, &a, &b, &c, 1.0e-12) {
            let edofs = space.element_dofs(e);
            return l0 * field[edofs[0] as usize]
                + l1 * field[edofs[1] as usize]
                + l2 * field[edofs[2] as usize];
        }
    }
    0.0
}

fn barycentric_2d(
    p: &[f64],
    a: &[f64; 2],
    b: &[f64; 2],
    c: &[f64; 2],
    tol: f64,
) -> Option<(f64, f64, f64)> {
    let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if det.abs() < 1.0e-30 {
        return None;
    }
    let l0 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det;
    let l1 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det;
    let l2 = 1.0 - l0 - l1;

    if l0 >= -tol && l1 >= -tol && l2 >= -tol {
        Some((l0, l1, l2))
    } else {
        None
    }
}

fn pressure_drop_metric(space: &H1Space<Mesh<2>>, p: &[f64]) -> f64 {
    let dm = space.dof_manager();
    let mut left_sum = 0.0_f64;
    let mut left_cnt = 0usize;
    let mut right_sum = 0.0_f64;
    let mut right_cnt = 0usize;
    for i in 0..space.n_dofs() {
        let x = dm.dof_coord(i as u32);
        if x[0].abs() < 1.0e-10 {
            left_sum += p[i];
            left_cnt += 1;
        }
        if (x[0] - 1.0).abs() < 1.0e-10 {
            right_sum += p[i];
            right_cnt += 1;
        }
    }
    let left_avg = if left_cnt == 0 { 0.0 } else { left_sum / left_cnt as f64 };
    let right_avg = if right_cnt == 0 { 0.0 } else { right_sum / right_cnt as f64 };
    (left_avg - right_avg).abs()
}

fn relative_change(a: &[f64], b: &[f64]) -> f64 {
    let mut d2 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = y - x;
        d2 += d * d;
        b2 += y * y;
    }
    d2.sqrt() / b2.sqrt().max(1.0e-14)
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

fn write_ex52_vtk_export(
    prefix: &str,
    mesh: &Mesh<2>,
    species: &[f64],
    temperature: &[f64],
) -> Result<(), String> {
    let path = format!("{prefix}_reaction_flow_thermal.vtu");
    ensure_parent_dir(&path).map_err(|e| e.to_string())?;
    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("species", species.to_vec()));
    writer.add_point_data(DataArray::scalars("temperature", temperature.to_vec()));
    writer.write_file(&path).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_reaction_flow_thermal_checkpoint(
    path: &str,
    state: &ReactionFlowThermalCheckpointState,
) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let species = format_vec_f64(&state.species);
    let temperature = format_vec_f64(&state.temperature);
    let content = format!(
        "format=ex52_reaction_flow_thermal_v1\nn={}\nsteps={}\ndt={:.17e}\nfast_dt={:.17e}\nsubcycling={}\ninlet_concentration={:.17e}\nflow_drive_amp={:.17e}\nk_species={:.17e}\nk_thermal={:.17e}\nreaction_k0={:.17e}\nreaction_temp_coeff={:.17e}\nheat_release={:.17e}\nviscosity0={:.17e}\nviscosity_temp_coeff={:.17e}\nviscosity_species_coeff={:.17e}\nrelax={:.17e}\ncoupling_tol={:.17e}\nsync_error_tol={:.17e}\nw_residual={:.17e}\nw_flow_metric={:.17e}\nw_rate_peak={:.17e}\nmax_coupling={}\nsync_retries={}\nfast_dt_min={:.17e}\ncompleted_steps={}\ncurrent_time={:.17e}\nconverged_steps={}\nmax_coupling_iters_used={}\nmax_reaction_rate={:.17e}\nfinal_flow_metric={:.17e}\nobserved_sync_retries={}\nrejected_sync_steps={}\nrollback_count={}\nflow_metric_tracker_prev={}\nrate_peak_tracker_prev={}\nspecies={}\ntemperature={}\n",
        state.args.n,
        state.args.steps,
        state.args.dt,
        state.args.fast_dt,
        if state.args.use_subcycling { 1 } else { 0 },
        state.args.inlet_concentration,
        state.args.flow_drive_amp,
        state.args.k_species,
        state.args.k_thermal,
        state.args.reaction_k0,
        state.args.reaction_temp_coeff,
        state.args.heat_release,
        state.args.viscosity0,
        state.args.viscosity_temp_coeff,
        state.args.viscosity_species_coeff,
        state.args.relax,
        state.args.coupling_tol,
        state.args.sync_error_tol,
        state.args.w_residual,
        state.args.w_flow_metric,
        state.args.w_rate_peak,
        state.args.max_coupling,
        state.args.sync_retries,
        state.args.fast_dt_min,
        state.completed_steps,
        state.current_time,
        state.converged_steps,
        state.max_coupling_iters_used,
        state.max_reaction_rate,
        state.final_flow_metric,
        state.observed_sync_retries,
        state.rejected_sync_steps,
        state.rollback_count,
        state
            .flow_metric_tracker_prev
            .map(|v| format!("{v:.17e}"))
            .unwrap_or_default(),
        state
            .rate_peak_tracker_prev
            .map(|v| format!("{v:.17e}"))
            .unwrap_or_default(),
        species,
        temperature,
    );
    fs::write(path, content)
}

fn write_reaction_flow_thermal_hdf5_checkpoint(
    path: &str,
    state: &ReactionFlowThermalCheckpointState,
) -> Result<(), String> {
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
            scalar_rank_field_f64("inlet_concentration", state.args.inlet_concentration),
            scalar_rank_field_f64("flow_drive_amp", state.args.flow_drive_amp),
            scalar_rank_field_f64("k_species", state.args.k_species),
            scalar_rank_field_f64("k_thermal", state.args.k_thermal),
            scalar_rank_field_f64("reaction_k0", state.args.reaction_k0),
            scalar_rank_field_f64("reaction_temp_coeff", state.args.reaction_temp_coeff),
            scalar_rank_field_f64("heat_release", state.args.heat_release),
            scalar_rank_field_f64("viscosity0", state.args.viscosity0),
            scalar_rank_field_f64("viscosity_temp_coeff", state.args.viscosity_temp_coeff),
            scalar_rank_field_f64("viscosity_species_coeff", state.args.viscosity_species_coeff),
            scalar_rank_field_f64("relax", state.args.relax),
            scalar_rank_field_f64("coupling_tol", state.args.coupling_tol),
            scalar_rank_field_f64("sync_error_tol", state.args.sync_error_tol),
            scalar_rank_field_f64("w_residual", state.args.w_residual),
            scalar_rank_field_f64("w_flow_metric", state.args.w_flow_metric),
            scalar_rank_field_f64("w_rate_peak", state.args.w_rate_peak),
            scalar_rank_field_f64("max_coupling", state.args.max_coupling as f64),
            scalar_rank_field_f64("sync_retries", state.args.sync_retries as f64),
            scalar_rank_field_f64("fast_dt_min", state.args.fast_dt_min),
            scalar_rank_field_f64("completed_steps", state.completed_steps as f64),
            scalar_rank_field_f64("current_time", state.current_time),
            scalar_rank_field_f64("converged_steps", state.converged_steps as f64),
            scalar_rank_field_f64("max_coupling_iters_used", state.max_coupling_iters_used as f64),
            scalar_rank_field_f64("max_reaction_rate", state.max_reaction_rate),
            scalar_rank_field_f64("final_flow_metric", state.final_flow_metric),
            scalar_rank_field_f64("observed_sync_retries", state.observed_sync_retries as f64),
            scalar_rank_field_f64("rejected_sync_steps", state.rejected_sync_steps as f64),
            scalar_rank_field_f64("rollback_count", state.rollback_count as f64),
            scalar_rank_field_f64("flow_metric_tracker_prev", state.flow_metric_tracker_prev.unwrap_or(f64::NAN)),
            scalar_rank_field_f64("rate_peak_tracker_prev", state.rate_peak_tracker_prev.unwrap_or(f64::NAN)),
            vector_rank_field_f64("species", state.species.clone()),
            vector_rank_field_f64("temperature", state.temperature.clone()),
        ],
    };
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let step = state.completed_steps.max(1) as u64;
    write_checkpoint_step_bundle_f64(path, cfg, step, state.current_time, &bundle, IoBackend::Partitioned)
        .map_err(|e| e.to_string())?;
    validate_checkpoint_layout(path, Some(1)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_reaction_flow_thermal_checkpoint(
    path: &str,
) -> Result<ReactionFlowThermalCheckpointState, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut format = None;
    let mut n = None;
    let mut steps = None;
    let mut dt = None;
    let mut fast_dt = None;
    let mut subcycling = None;
    let mut inlet_concentration = None;
    let mut flow_drive_amp = None;
    let mut k_species = None;
    let mut k_thermal = None;
    let mut reaction_k0 = None;
    let mut reaction_temp_coeff = None;
    let mut heat_release = None;
    let mut viscosity0 = None;
    let mut viscosity_temp_coeff = None;
    let mut viscosity_species_coeff = None;
    let mut relax = None;
    let mut coupling_tol = None;
    let mut sync_error_tol = None;
    let mut w_residual = None;
    let mut w_flow_metric = None;
    let mut w_rate_peak = None;
    let mut max_coupling = None;
    let mut sync_retries = None;
    let mut fast_dt_min = None;
    let mut completed_steps = None;
    let mut current_time = None;
    let mut converged_steps = None;
    let mut max_coupling_iters_used = None;
    let mut max_reaction_rate = None;
    let mut final_flow_metric = None;
    let mut observed_sync_retries = None;
    let mut rejected_sync_steps = None;
    let mut rollback_count = None;
    let mut flow_metric_tracker_prev = None;
    let mut rate_peak_tracker_prev = None;
    let mut species = None;
    let mut temperature = None;

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
            "inlet_concentration" => inlet_concentration = value.parse::<f64>().ok(),
            "flow_drive_amp" => flow_drive_amp = value.parse::<f64>().ok(),
            "k_species" => k_species = value.parse::<f64>().ok(),
            "k_thermal" => k_thermal = value.parse::<f64>().ok(),
            "reaction_k0" => reaction_k0 = value.parse::<f64>().ok(),
            "reaction_temp_coeff" => reaction_temp_coeff = value.parse::<f64>().ok(),
            "heat_release" => heat_release = value.parse::<f64>().ok(),
            "viscosity0" => viscosity0 = value.parse::<f64>().ok(),
            "viscosity_temp_coeff" => viscosity_temp_coeff = value.parse::<f64>().ok(),
            "viscosity_species_coeff" => viscosity_species_coeff = value.parse::<f64>().ok(),
            "relax" => relax = value.parse::<f64>().ok(),
            "coupling_tol" => coupling_tol = value.parse::<f64>().ok(),
            "sync_error_tol" => sync_error_tol = value.parse::<f64>().ok(),
            "w_residual" => w_residual = value.parse::<f64>().ok(),
            "w_flow_metric" => w_flow_metric = value.parse::<f64>().ok(),
            "w_rate_peak" => w_rate_peak = value.parse::<f64>().ok(),
            "max_coupling" => max_coupling = value.parse::<usize>().ok(),
            "sync_retries" => sync_retries = value.parse::<usize>().ok(),
            "fast_dt_min" => fast_dt_min = value.parse::<f64>().ok(),
            "completed_steps" => completed_steps = value.parse::<usize>().ok(),
            "current_time" => current_time = value.parse::<f64>().ok(),
            "converged_steps" => converged_steps = value.parse::<usize>().ok(),
            "max_coupling_iters_used" => max_coupling_iters_used = value.parse::<usize>().ok(),
            "max_reaction_rate" => max_reaction_rate = value.parse::<f64>().ok(),
            "final_flow_metric" => final_flow_metric = value.parse::<f64>().ok(),
            "observed_sync_retries" => observed_sync_retries = value.parse::<usize>().ok(),
            "rejected_sync_steps" => rejected_sync_steps = value.parse::<usize>().ok(),
            "rollback_count" => rollback_count = value.parse::<usize>().ok(),
            "flow_metric_tracker_prev" => {
                flow_metric_tracker_prev = if value.trim().is_empty() {
                    Some(None)
                } else {
                    Some(Some(value.parse::<f64>().map_err(|e| e.to_string())?))
                }
            }
            "rate_peak_tracker_prev" => {
                rate_peak_tracker_prev = if value.trim().is_empty() {
                    Some(None)
                } else {
                    Some(Some(value.parse::<f64>().map_err(|e| e.to_string())?))
                }
            }
            "species" => species = Some(parse_vec_f64(value)?),
            "temperature" => temperature = Some(parse_vec_f64(value)?),
            _ => {}
        }
    }

    if format.as_deref() != Some("ex52_reaction_flow_thermal_v1") {
        return Err("unsupported checkpoint format".into());
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        steps: steps.ok_or_else(|| "missing steps".to_string())?,
        dt: dt.ok_or_else(|| "missing dt".to_string())?,
        fast_dt: fast_dt.ok_or_else(|| "missing fast_dt".to_string())?,
        use_subcycling: subcycling.ok_or_else(|| "missing subcycling".to_string())?,
        inlet_concentration: inlet_concentration
            .ok_or_else(|| "missing inlet_concentration".to_string())?,
        flow_drive_amp: flow_drive_amp.ok_or_else(|| "missing flow_drive_amp".to_string())?,
        k_species: k_species.ok_or_else(|| "missing k_species".to_string())?,
        k_thermal: k_thermal.ok_or_else(|| "missing k_thermal".to_string())?,
        reaction_k0: reaction_k0.ok_or_else(|| "missing reaction_k0".to_string())?,
        reaction_temp_coeff: reaction_temp_coeff
            .ok_or_else(|| "missing reaction_temp_coeff".to_string())?,
        heat_release: heat_release.ok_or_else(|| "missing heat_release".to_string())?,
        viscosity0: viscosity0.ok_or_else(|| "missing viscosity0".to_string())?,
        viscosity_temp_coeff: viscosity_temp_coeff
            .ok_or_else(|| "missing viscosity_temp_coeff".to_string())?,
        viscosity_species_coeff: viscosity_species_coeff
            .ok_or_else(|| "missing viscosity_species_coeff".to_string())?,
        relax: relax.ok_or_else(|| "missing relax".to_string())?,
        coupling_tol: coupling_tol.ok_or_else(|| "missing coupling_tol".to_string())?,
        sync_error_tol: sync_error_tol.ok_or_else(|| "missing sync_error_tol".to_string())?,
        w_residual: w_residual.ok_or_else(|| "missing w_residual".to_string())?,
        w_flow_metric: w_flow_metric.ok_or_else(|| "missing w_flow_metric".to_string())?,
        w_rate_peak: w_rate_peak.ok_or_else(|| "missing w_rate_peak".to_string())?,
        max_coupling: max_coupling.ok_or_else(|| "missing max_coupling".to_string())?,
        sync_retries: sync_retries.ok_or_else(|| "missing sync_retries".to_string())?,
        fast_dt_min: fast_dt_min.ok_or_else(|| "missing fast_dt_min".to_string())?,
    };

    let species = species.ok_or_else(|| "missing species".to_string())?;
    let temperature = temperature.ok_or_else(|| "missing temperature".to_string())?;
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if species.len() != expected_dofs || temperature.len() != expected_dofs {
        return Err(format!(
            "checkpoint field lengths do not match expected dofs ({expected_dofs})"
        ));
    }

    Ok(ReactionFlowThermalCheckpointState {
        args,
        completed_steps: completed_steps.ok_or_else(|| "missing completed_steps".to_string())?,
        current_time: current_time.ok_or_else(|| "missing current_time".to_string())?,
        converged_steps: converged_steps.ok_or_else(|| "missing converged_steps".to_string())?,
        max_coupling_iters_used: max_coupling_iters_used
            .ok_or_else(|| "missing max_coupling_iters_used".to_string())?,
        max_reaction_rate: max_reaction_rate.ok_or_else(|| "missing max_reaction_rate".to_string())?,
        final_flow_metric: final_flow_metric.ok_or_else(|| "missing final_flow_metric".to_string())?,
        observed_sync_retries: observed_sync_retries
            .ok_or_else(|| "missing observed_sync_retries".to_string())?,
        rejected_sync_steps: rejected_sync_steps
            .ok_or_else(|| "missing rejected_sync_steps".to_string())?,
        rollback_count: rollback_count.ok_or_else(|| "missing rollback_count".to_string())?,
        flow_metric_tracker_prev: flow_metric_tracker_prev.unwrap_or(None),
        rate_peak_tracker_prev: rate_peak_tracker_prev.unwrap_or(None),
        species,
        temperature,
    })
}

fn read_reaction_flow_thermal_hdf5_checkpoint(
    path: &str,
) -> Result<ReactionFlowThermalCheckpointState, String> {
    let fields = read_checkpoint_fields_f64_latest(
        path,
        ParallelIoConfig { world_size: 1, rank: 0 },
        &[
            "n",
            "steps",
            "dt",
            "fast_dt",
            "subcycling",
            "inlet_concentration",
            "flow_drive_amp",
            "k_species",
            "k_thermal",
            "reaction_k0",
            "reaction_temp_coeff",
            "heat_release",
            "viscosity0",
            "viscosity_temp_coeff",
            "viscosity_species_coeff",
            "relax",
            "coupling_tol",
            "sync_error_tol",
            "w_residual",
            "w_flow_metric",
            "w_rate_peak",
            "max_coupling",
            "sync_retries",
            "fast_dt_min",
            "completed_steps",
            "current_time",
            "converged_steps",
            "max_coupling_iters_used",
            "max_reaction_rate",
            "final_flow_metric",
            "observed_sync_retries",
            "rejected_sync_steps",
            "rollback_count",
            "flow_metric_tracker_prev",
            "rate_peak_tracker_prev",
            "species",
            "temperature",
        ],
    )
    .map_err(|e| e.to_string())?;

    let mut n = None;
    let mut steps = None;
    let mut dt = None;
    let mut fast_dt = None;
    let mut subcycling = None;
    let mut inlet_concentration = None;
    let mut flow_drive_amp = None;
    let mut k_species = None;
    let mut k_thermal = None;
    let mut reaction_k0 = None;
    let mut reaction_temp_coeff = None;
    let mut heat_release = None;
    let mut viscosity0 = None;
    let mut viscosity_temp_coeff = None;
    let mut viscosity_species_coeff = None;
    let mut relax = None;
    let mut coupling_tol = None;
    let mut sync_error_tol = None;
    let mut w_residual = None;
    let mut w_flow_metric = None;
    let mut w_rate_peak = None;
    let mut max_coupling = None;
    let mut sync_retries = None;
    let mut fast_dt_min = None;
    let mut completed_steps = None;
    let mut current_time = None;
    let mut converged_steps = None;
    let mut max_coupling_iters_used = None;
    let mut max_reaction_rate = None;
    let mut final_flow_metric = None;
    let mut observed_sync_retries = None;
    let mut rejected_sync_steps = None;
    let mut rollback_count = None;
    let mut flow_metric_tracker_prev = None;
    let mut rate_peak_tracker_prev = None;
    let mut species = None;
    let mut temperature = None;

    for (name, field) in fields {
        match name.as_str() {
            "n" => n = field.values.first().map(|v| *v as usize),
            "steps" => steps = field.values.first().map(|v| *v as usize),
            "dt" => dt = field.values.first().copied(),
            "fast_dt" => fast_dt = field.values.first().copied(),
            "subcycling" => subcycling = field.values.first().map(|v| *v != 0.0),
            "inlet_concentration" => inlet_concentration = field.values.first().copied(),
            "flow_drive_amp" => flow_drive_amp = field.values.first().copied(),
            "k_species" => k_species = field.values.first().copied(),
            "k_thermal" => k_thermal = field.values.first().copied(),
            "reaction_k0" => reaction_k0 = field.values.first().copied(),
            "reaction_temp_coeff" => reaction_temp_coeff = field.values.first().copied(),
            "heat_release" => heat_release = field.values.first().copied(),
            "viscosity0" => viscosity0 = field.values.first().copied(),
            "viscosity_temp_coeff" => viscosity_temp_coeff = field.values.first().copied(),
            "viscosity_species_coeff" => viscosity_species_coeff = field.values.first().copied(),
            "relax" => relax = field.values.first().copied(),
            "coupling_tol" => coupling_tol = field.values.first().copied(),
            "sync_error_tol" => sync_error_tol = field.values.first().copied(),
            "w_residual" => w_residual = field.values.first().copied(),
            "w_flow_metric" => w_flow_metric = field.values.first().copied(),
            "w_rate_peak" => w_rate_peak = field.values.first().copied(),
            "max_coupling" => max_coupling = field.values.first().map(|v| *v as usize),
            "sync_retries" => sync_retries = field.values.first().map(|v| *v as usize),
            "fast_dt_min" => fast_dt_min = field.values.first().copied(),
            "completed_steps" => completed_steps = field.values.first().map(|v| *v as usize),
            "current_time" => current_time = field.values.first().copied(),
            "converged_steps" => converged_steps = field.values.first().map(|v| *v as usize),
            "max_coupling_iters_used" => {
                max_coupling_iters_used = field.values.first().map(|v| *v as usize)
            }
            "max_reaction_rate" => max_reaction_rate = field.values.first().copied(),
            "final_flow_metric" => final_flow_metric = field.values.first().copied(),
            "observed_sync_retries" => {
                observed_sync_retries = field.values.first().map(|v| *v as usize)
            }
            "rejected_sync_steps" => rejected_sync_steps = field.values.first().map(|v| *v as usize),
            "rollback_count" => rollback_count = field.values.first().map(|v| *v as usize),
            "flow_metric_tracker_prev" => {
                flow_metric_tracker_prev = field.values.first().copied().filter(|v| v.is_finite())
            }
            "rate_peak_tracker_prev" => {
                rate_peak_tracker_prev = field.values.first().copied().filter(|v| v.is_finite())
            }
            "species" => species = Some(field.values),
            "temperature" => temperature = Some(field.values),
            _ => {}
        }
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        steps: steps.ok_or_else(|| "missing steps".to_string())?,
        dt: dt.ok_or_else(|| "missing dt".to_string())?,
        fast_dt: fast_dt.ok_or_else(|| "missing fast_dt".to_string())?,
        use_subcycling: subcycling.ok_or_else(|| "missing subcycling".to_string())?,
        inlet_concentration: inlet_concentration.ok_or_else(|| "missing inlet_concentration".to_string())?,
        flow_drive_amp: flow_drive_amp.ok_or_else(|| "missing flow_drive_amp".to_string())?,
        k_species: k_species.ok_or_else(|| "missing k_species".to_string())?,
        k_thermal: k_thermal.ok_or_else(|| "missing k_thermal".to_string())?,
        reaction_k0: reaction_k0.ok_or_else(|| "missing reaction_k0".to_string())?,
        reaction_temp_coeff: reaction_temp_coeff.ok_or_else(|| "missing reaction_temp_coeff".to_string())?,
        heat_release: heat_release.ok_or_else(|| "missing heat_release".to_string())?,
        viscosity0: viscosity0.ok_or_else(|| "missing viscosity0".to_string())?,
        viscosity_temp_coeff: viscosity_temp_coeff.ok_or_else(|| "missing viscosity_temp_coeff".to_string())?,
        viscosity_species_coeff: viscosity_species_coeff
            .ok_or_else(|| "missing viscosity_species_coeff".to_string())?,
        relax: relax.ok_or_else(|| "missing relax".to_string())?,
        coupling_tol: coupling_tol.ok_or_else(|| "missing coupling_tol".to_string())?,
        sync_error_tol: sync_error_tol.ok_or_else(|| "missing sync_error_tol".to_string())?,
        w_residual: w_residual.ok_or_else(|| "missing w_residual".to_string())?,
        w_flow_metric: w_flow_metric.ok_or_else(|| "missing w_flow_metric".to_string())?,
        w_rate_peak: w_rate_peak.ok_or_else(|| "missing w_rate_peak".to_string())?,
        max_coupling: max_coupling.ok_or_else(|| "missing max_coupling".to_string())?,
        sync_retries: sync_retries.ok_or_else(|| "missing sync_retries".to_string())?,
        fast_dt_min: fast_dt_min.ok_or_else(|| "missing fast_dt_min".to_string())?,
    };
    let species = species.ok_or_else(|| "missing species".to_string())?;
    let temperature = temperature.ok_or_else(|| "missing temperature".to_string())?;
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if species.len() != expected_dofs || temperature.len() != expected_dofs {
        return Err(format!(
            "checkpoint field lengths do not match expected dofs ({expected_dofs})"
        ));
    }

    Ok(ReactionFlowThermalCheckpointState {
        args,
        completed_steps: completed_steps.ok_or_else(|| "missing completed_steps".to_string())?,
        current_time: current_time.ok_or_else(|| "missing current_time".to_string())?,
        converged_steps: converged_steps.ok_or_else(|| "missing converged_steps".to_string())?,
        max_coupling_iters_used: max_coupling_iters_used
            .ok_or_else(|| "missing max_coupling_iters_used".to_string())?,
        max_reaction_rate: max_reaction_rate.ok_or_else(|| "missing max_reaction_rate".to_string())?,
        final_flow_metric: final_flow_metric.ok_or_else(|| "missing final_flow_metric".to_string())?,
        observed_sync_retries: observed_sync_retries
            .ok_or_else(|| "missing observed_sync_retries".to_string())?,
        rejected_sync_steps: rejected_sync_steps
            .ok_or_else(|| "missing rejected_sync_steps".to_string())?,
        rollback_count: rollback_count.ok_or_else(|| "missing rollback_count".to_string())?,
        flow_metric_tracker_prev,
        rate_peak_tracker_prev,
        species,
        temperature,
    })
}

#[cfg(feature = "io_hdf5")]
fn write_reaction_flow_thermal_hdf5_xdmf_sidecars(
    h5_path: &str,
    state: &ReactionFlowThermalCheckpointState,
) -> Result<(), String> {
    let step = state.completed_steps.max(1);
    write_scalar_checkpoint_xdmf_sidecars(
        h5_path,
        step as u64,
        state.current_time,
        &["species", "temperature"],
    )
}

fn parse_args() -> CliArgs {
    let mut a = Args {
        n: 12,
        steps: 8,
        dt: 0.05,
        fast_dt: 0.01,
        use_subcycling: true,
        inlet_concentration: 1.0,
        flow_drive_amp: 0.25,
        k_species: 0.2,
        k_thermal: 1.0,
        reaction_k0: 0.8,
        reaction_temp_coeff: 0.05,
        heat_release: 5.0,
        viscosity0: 1.0,
        viscosity_temp_coeff: 0.02,
        viscosity_species_coeff: 0.01,
        relax: 0.7,
        coupling_tol: 1.0e-7,
        sync_error_tol: 1.0,
        w_residual: 1.0,
        w_flow_metric: 1.0,
        w_rate_peak: 1.0,
        max_coupling: 12,
        sync_retries: 2,
        fast_dt_min: 1.0e-3,
    };
    let mut workflow = WorkflowCliOptions::default();

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex52_template_reaction_flow_thermal");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        let mut help_options = vec![
            ("--n <int>", "Mesh resolution (default: 12)"),
            (
                "--steps <int>",
                "Number of slow synchronization steps (default: 8)",
            ),
            ("--dt <float>", "Slow-step size (default: 0.05)"),
            (
                "--fast-dt <float>",
                "Fast subcycling step size (default: 0.01)",
            ),
            ("--subcycling", "Enable multirate subcycling (default)"),
            ("--no-subcycling", "Disable subcycling and use single-rate loop"),
            (
                "--inlet-concentration <float>",
                "Inlet species concentration (default: 1.0)",
            ),
            (
                "--flow-drive-amp <float>",
                "Flow driving amplitude (default: 0.25)",
            ),
            ("--k-species <float>", "Species diffusivity (default: 0.2)"),
            ("--k-thermal <float>", "Thermal diffusivity (default: 1.0)"),
            (
                "--reaction-k0 <float>",
                "Base reaction-rate coefficient (default: 0.8)",
            ),
            (
                "--reaction-temp <float>",
                "Temperature-reaction sensitivity (default: 0.05)",
            ),
            (
                "--heat-release <float>",
                "Thermal source coefficient per consumed species (default: 5.0)",
            ),
            ("--viscosity0 <float>", "Reference viscosity (default: 1.0)"),
            (
                "--visc-temp <float>",
                "Temperature-viscosity slope (default: 0.02)",
            ),
            (
                "--visc-species <float>",
                "Species-viscosity slope (default: 0.01)",
            ),
            ("--relax <float>", "Relaxation factor in [0.1, 1.0] (default: 0.7)"),
            (
                "--coupling-tol <float>",
                "Coupling convergence tolerance (default: 1e-7)",
            ),
            (
                "--sync-error-tol <float>",
                "Adaptive sync acceptance tolerance (default: 1.0)",
            ),
            (
                "--w-residual <float>",
                "Residual term weight in sync metric (default: 1.0)",
            ),
            (
                "--w-flow-metric <float>",
                "Flow-metric weight in sync metric (default: 1.0)",
            ),
            (
                "--w-rate-peak <float>",
                "Reaction-peak weight in sync metric (default: 1.0)",
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
        ];
        push_workflow_cli_help(
            &mut help_options,
            "Write final species/temperature VTK export as <prefix>_reaction_flow_thermal.vtu",
        );
        print_template_cli_help(
            bin,
            &help_options,
        );
        std::process::exit(0);
    }

    let mut it = args_vec.into_iter().skip(1);
    while let Some(arg) = it.next() {
        if workflow.try_parse_arg(arg.as_str(), &mut it) {
            continue;
        }
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("12".into()).parse().unwrap_or(12),
            "--steps" => a.steps = it.next().unwrap_or("8".into()).parse().unwrap_or(8),
            "--dt" => a.dt = it.next().unwrap_or("0.05".into()).parse().unwrap_or(0.05),
            "--fast-dt" => a.fast_dt = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--subcycling" => a.use_subcycling = true,
            "--no-subcycling" => a.use_subcycling = false,
            "--inlet-concentration" => {
                a.inlet_concentration = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--flow-drive-amp" => {
                a.flow_drive_amp = it.next().unwrap_or("0.25".into()).parse().unwrap_or(0.25)
            }
            "--k-species" => {
                a.k_species = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2)
            }
            "--k-thermal" => {
                a.k_thermal = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--reaction-k0" => {
                a.reaction_k0 = it.next().unwrap_or("0.8".into()).parse().unwrap_or(0.8)
            }
            "--reaction-temp" => {
                a.reaction_temp_coeff = it.next().unwrap_or("0.05".into()).parse().unwrap_or(0.05)
            }
            "--heat-release" => {
                a.heat_release = it.next().unwrap_or("5.0".into()).parse().unwrap_or(5.0)
            }
            "--viscosity0" => {
                a.viscosity0 = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--visc-temp" => {
                a.viscosity_temp_coeff = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02)
            }
            "--visc-species" => {
                a.viscosity_species_coeff = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01)
            }
            "--relax" => a.relax = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--coupling-tol" => {
                a.coupling_tol = it.next().unwrap_or("1e-7".into()).parse().unwrap_or(1.0e-7)
            }
            "--sync-error-tol" => {
                a.sync_error_tol = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--w-residual" => {
                a.w_residual = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--w-flow-metric" => {
                a.w_flow_metric = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--w-rate-peak" => {
                a.w_rate_peak = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
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
            _ => {}
        }
    }

    a.steps = a.steps.max(1);
    a.fast_dt = a.fast_dt.max(1.0e-12);
    a.fast_dt_min = a.fast_dt_min.max(1.0e-12).min(a.fast_dt);
    a.sync_error_tol = a.sync_error_tol.max(0.0);
    a.w_residual = a.w_residual.max(0.0);
    a.w_flow_metric = a.w_flow_metric.max(0.0);
    a.w_rate_peak = a.w_rate_peak.max(0.0);
    a.max_coupling = a.max_coupling.max(1);
    a.relax = a.relax.clamp(0.1, 1.0);
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
                "ex52_{}_{}_{}.{}",
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
            inlet_concentration: 1.0,
            flow_drive_amp: 0.2,
            k_species: 0.2,
            k_thermal: 1.0,
            reaction_k0: 0.8,
            reaction_temp_coeff: 0.05,
            heat_release: 5.0,
            viscosity0: 1.0,
            viscosity_temp_coeff: 0.02,
            viscosity_species_coeff: 0.01,
            relax: 0.7,
            coupling_tol: 1.0e-7,
            sync_error_tol: 1.0,
            w_residual: 1.0,
            w_flow_metric: 1.0,
            w_rate_peak: 1.0,
            max_coupling: 10,
            sync_retries: 2,
            fast_dt_min: 1.0e-3,
        }
    }

    #[test]
    fn ex52_reaction_flow_thermal_template_runs_and_reacts() {
        let r = solve_reaction_flow_thermal_template(&base_args());
        assert_eq!(r.steps, 4);
        assert!(r.max_coupling_iters_used <= 10);
        assert!(r.max_reaction_rate > 0.0);
        assert!(r.final_flow_metric > 0.0);
        assert!(r.final_species_norm >= 0.0);
        assert!(r.final_temperature_norm > 0.0);
    }

    #[test]
    fn ex52_higher_inlet_concentration_increases_reaction_and_temperature() {
        let mut low = base_args();
        low.inlet_concentration = 0.5;
        let mut high = base_args();
        high.inlet_concentration = 1.5;

        let r_low = solve_reaction_flow_thermal_template(&low);
        let r_high = solve_reaction_flow_thermal_template(&high);

        assert!(r_high.max_reaction_rate > r_low.max_reaction_rate);
        assert!(r_high.final_temperature_norm > r_low.final_temperature_norm);
    }

    #[test]
    fn ex52_weighted_sync_error_path_runs() {
        let mut a = base_args();
        a.w_residual = 1.0;
        a.w_flow_metric = 0.5;
        a.w_rate_peak = 2.0;
        a.sync_error_tol = 2.0;

        let r = solve_reaction_flow_thermal_template(&a);
        assert_eq!(r.steps, 4);
        assert!(r.max_reaction_rate > 0.0);
        assert!(r.final_temperature_norm > 0.0);
    }

    #[test]
    #[should_panic(expected = "adaptive subcycling scheduler failed")]
    fn ex52_strict_weighted_sync_error_can_fail() {
        let mut a = base_args();
        a.w_rate_peak = 3.0;
        a.sync_error_tol = 1.0;
        a.sync_retries = 0;
        let _ = solve_reaction_flow_thermal_template(&a);
    }

    /// Zero reaction rate constant → no consumption, no heat release,
    /// temperature should remain near zero after zero initial condition.
    #[test]
    fn ex52_zero_reaction_rate_gives_no_temperature_rise() {
        let mut args = base_args();
        args.reaction_k0 = 0.0;
        args.heat_release = 1.0;
        let r = solve_reaction_flow_thermal_template(&args);
        assert!(r.max_reaction_rate < 1.0e-12,
            "expected zero reaction rate: {:.4e}", r.max_reaction_rate);
        assert!(r.final_temperature_norm < 1.0e-10,
            "expected near-zero temperature with no reaction: {:.4e}", r.final_temperature_norm);
    }

    /// Higher heat release per unit reaction → higher final temperature norm.
    #[test]
    fn ex52_higher_heat_release_increases_temperature() {
        let mut low = base_args();
        low.heat_release = 0.5;
        let mut high = base_args();
        high.heat_release = 2.0;

        let r_low  = solve_reaction_flow_thermal_template(&low);
        let r_high = solve_reaction_flow_thermal_template(&high);
        assert!(r_high.final_temperature_norm > r_low.final_temperature_norm,
            "higher heat release should give more temperature: low={:.4e} high={:.4e}",
            r_low.final_temperature_norm, r_high.final_temperature_norm);
    }

    /// Subcycling path should produce the same qualitative trend as single-rate:
    /// more inlet concentration → more reaction and temperature.
    #[test]
    fn ex52_subcycling_path_shows_same_inlet_trend_as_single_rate() {
        let mut single_low = base_args();
        single_low.use_subcycling = false;
        single_low.inlet_concentration = 0.5;
        let mut single_high = base_args();
        single_high.use_subcycling = false;
        single_high.inlet_concentration = 1.5;

        let mut sub_low = base_args();
        sub_low.use_subcycling = true;
        sub_low.inlet_concentration = 0.5;
        let mut sub_high = base_args();
        sub_high.use_subcycling = true;
        sub_high.inlet_concentration = 1.5;

        let rs_l = solve_reaction_flow_thermal_template(&single_low);
        let rs_h = solve_reaction_flow_thermal_template(&single_high);
        let rb_l = solve_reaction_flow_thermal_template(&sub_low);
        let rb_h = solve_reaction_flow_thermal_template(&sub_high);

        // Both paths: higher inlet → higher temperature.
        assert!(rs_h.final_temperature_norm > rs_l.final_temperature_norm,
            "single-rate: higher inlet should raise temperature");
        assert!(rb_h.final_temperature_norm > rb_l.final_temperature_norm,
            "subcycling: higher inlet should raise temperature");
    }

    /// Identical args must produce an identical temperature checksum (determinism).
    #[test]
    fn ex52_temperature_checksum_is_deterministic() {
        let r1 = solve_reaction_flow_thermal_template(&base_args());
        let r2 = solve_reaction_flow_thermal_template(&base_args());
        assert_eq!(r1.final_temperature_checksum, r2.final_temperature_checksum,
            "temperature checksum is not deterministic: run1={:.8e} run2={:.8e}",
            r1.final_temperature_checksum, r2.final_temperature_checksum);
    }

    #[test]
    fn ex52_text_checkpoint_roundtrip_preserves_restart_state() {
        let args = base_args();
        let partial = solve_reaction_flow_thermal_template_with_restart(
            &Args {
                steps: 2,
                ..args.clone()
            },
            None,
        );
        let path = temp_output_path("checkpoint", "txt");
        let checkpoint = ReactionFlowThermalCheckpointState {
            args: Args {
                steps: 2,
                ..args.clone()
            },
            completed_steps: partial.completed_steps,
            current_time: partial.completed_steps as f64 * args.dt,
            converged_steps: partial.converged_steps,
            max_coupling_iters_used: partial.max_coupling_iters_used,
            max_reaction_rate: partial.max_reaction_rate,
            final_flow_metric: partial.final_flow_metric,
            observed_sync_retries: partial.sync_retries,
            rejected_sync_steps: partial.rejected_sync_steps,
            rollback_count: partial.rollback_count,
            flow_metric_tracker_prev: partial.flow_metric_tracker_prev,
            rate_peak_tracker_prev: partial.rate_peak_tracker_prev,
            species: partial.species.clone(),
            temperature: partial.temperature.clone(),
        };

        write_reaction_flow_thermal_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_reaction_flow_thermal_checkpoint(&path).unwrap();

        assert_eq!(restored.completed_steps, checkpoint.completed_steps);
        assert_eq!(restored.args.n, checkpoint.args.n);
        assert_eq!(restored.args.use_subcycling, checkpoint.args.use_subcycling);
        assert_eq!(restored.species, checkpoint.species);
        assert_eq!(restored.temperature, checkpoint.temperature);
        assert_eq!(restored.flow_metric_tracker_prev, checkpoint.flow_metric_tracker_prev);
        assert_eq!(restored.rate_peak_tracker_prev, checkpoint.rate_peak_tracker_prev);

        let resumed = solve_reaction_flow_thermal_template_with_restart(&args, Some(&restored));
        let full = solve_reaction_flow_thermal_template(&args);
        assert!((resumed.final_species_checksum - full.final_species_checksum).abs() < 1.0e-12);
        assert!((resumed.final_temperature_checksum - full.final_temperature_checksum).abs() < 1.0e-12);
        assert!((resumed.final_flow_metric - full.final_flow_metric).abs() < 1.0e-12);
        assert!((resumed.max_reaction_rate - full.max_reaction_rate).abs() < 1.0e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn ex52_vtk_export_writes_species_and_temperature() {
        let result = solve_reaction_flow_thermal_template(&base_args());
        let prefix = temp_output_path("vtk", "out");
        let vtk_path = format!("{prefix}_reaction_flow_thermal.vtu");

        write_ex52_vtk_export(&prefix, &result.mesh, &result.species, &result.temperature).unwrap();

        let vtk = fs::read_to_string(&vtk_path).unwrap();
        assert!(vtk.contains("species"));
        assert!(vtk.contains("temperature"));

        let _ = fs::remove_file(vtk_path);
    }

    #[test]
    fn ex52_hdf5_checkpoint_roundtrip_preserves_restart_state() {
        let args = base_args();
        let partial = solve_reaction_flow_thermal_template_with_restart(
            &Args {
                steps: 2,
                ..args.clone()
            },
            None,
        );
        let path = temp_output_path("checkpoint_h5", "h5");
        let checkpoint = ReactionFlowThermalCheckpointState {
            args: Args {
                steps: 2,
                ..args.clone()
            },
            completed_steps: partial.completed_steps,
            current_time: partial.completed_steps as f64 * args.dt,
            converged_steps: partial.converged_steps,
            max_coupling_iters_used: partial.max_coupling_iters_used,
            max_reaction_rate: partial.max_reaction_rate,
            final_flow_metric: partial.final_flow_metric,
            observed_sync_retries: partial.sync_retries,
            rejected_sync_steps: partial.rejected_sync_steps,
            rollback_count: partial.rollback_count,
            flow_metric_tracker_prev: partial.flow_metric_tracker_prev,
            rate_peak_tracker_prev: partial.rate_peak_tracker_prev,
            species: partial.species.clone(),
            temperature: partial.temperature.clone(),
        };

        write_reaction_flow_thermal_hdf5_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_reaction_flow_thermal_hdf5_checkpoint(&path).unwrap();

        assert_eq!(restored.completed_steps, checkpoint.completed_steps);
        assert_eq!(restored.args.n, checkpoint.args.n);
        assert_eq!(restored.args.use_subcycling, checkpoint.args.use_subcycling);
        assert_eq!(restored.species, checkpoint.species);
        assert_eq!(restored.temperature, checkpoint.temperature);
        assert_eq!(restored.flow_metric_tracker_prev, checkpoint.flow_metric_tracker_prev);
        assert_eq!(restored.rate_peak_tracker_prev, checkpoint.rate_peak_tracker_prev);

        let resumed = solve_reaction_flow_thermal_template_with_restart(&args, Some(&restored));
        let full = solve_reaction_flow_thermal_template(&args);
        assert!((resumed.final_species_checksum - full.final_species_checksum).abs() < 1.0e-12);
        assert!((resumed.final_temperature_checksum - full.final_temperature_checksum).abs() < 1.0e-12);
        assert!((resumed.final_flow_metric - full.final_flow_metric).abs() < 1.0e-12);
        assert!((resumed.max_reaction_rate - full.max_reaction_rate).abs() < 1.0e-12);

        let _ = fs::remove_file(path);
    }

    #[cfg(feature = "io_hdf5")]
    #[test]
    fn ex52_hdf5_checkpoint_writes_xdmf_sidecars() {
        let args = base_args();
        let result = solve_reaction_flow_thermal_template(&args);
        let h5_path = temp_output_path("checkpoint_sidecar", "h5");
        let species_sidecar = checkpoint_sidecar_path(&h5_path, "species").unwrap();
        let temperature_sidecar = checkpoint_sidecar_path(&h5_path, "temperature").unwrap();
        let checkpoint = ReactionFlowThermalCheckpointState {
            args,
            completed_steps: result.completed_steps,
            current_time: result.completed_steps as f64 * 0.05,
            converged_steps: result.converged_steps,
            max_coupling_iters_used: result.max_coupling_iters_used,
            max_reaction_rate: result.max_reaction_rate,
            final_flow_metric: result.final_flow_metric,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            flow_metric_tracker_prev: result.flow_metric_tracker_prev,
            rate_peak_tracker_prev: result.rate_peak_tracker_prev,
            species: result.species.clone(),
            temperature: result.temperature.clone(),
        };

        write_reaction_flow_thermal_hdf5_checkpoint(&h5_path, &checkpoint).unwrap();
        write_reaction_flow_thermal_hdf5_xdmf_sidecars(&h5_path, &checkpoint).unwrap();

        let species_xdmf = fs::read_to_string(&species_sidecar).unwrap();
        let temperature_xdmf = fs::read_to_string(&temperature_sidecar).unwrap();
        assert!(species_xdmf.contains("species"));
        assert!(temperature_xdmf.contains("temperature"));

        let _ = fs::remove_file(h5_path);
        let _ = fs::remove_file(species_sidecar);
        let _ = fs::remove_file(temperature_sidecar);
    }
}
