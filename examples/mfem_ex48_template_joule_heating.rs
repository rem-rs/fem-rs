//! Example 48: built-in template driver - Joule Heating.
//!
//! This example demonstrates how a COMSOL-like built-in multiphysics template
//! can be executed in fem-rs using a unified template node interface.
//!
//! Coupling loop (fixed-point):
//! 1) solve electric potential: -div(sigma(T) grad(phi)) = 0
//! 2) compute Joule source: q = sigma(T) |grad(phi)|^2
//! 3) solve temperature: -div(k grad(T)) = q
//! 4) repeat until thermal state converges

use std::{fs, io};

use fem_assembly::{
    Assembler,
    coefficient::FnCoeff,
    postprocess::compute_element_gradients,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_examples::template_runner::{
    maybe_write_template_kpi_csv,
    TemplateCouplingSummary,
    TemplateAdaptiveSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
    print_template_header,
};
use fem_examples::checkpoint_text::{ensure_parent_dir, format_vec_f64, parse_vec_f64};
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
    compose_sync_error,
    run_multirate_adaptive,
    solve_gmres,
    solve_pcg_jacobi,
};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

struct JouleTemplateResult {
    completed_iterations: usize,
    converged: bool,
    iterations: usize,
    final_relative_change: f64,
    sigma_effective: f64,
    phi_norm: f64,
    temp_norm: f64,
    joule_power: f64,
    temp_checksum: f64,
    sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
    joule_power_tracker_prev: Option<f64>,
    phi: Vec<f64>,
    temperature: Vec<f64>,
    mesh: Mesh<2>,
}

#[derive(Clone)]
struct Args {
    n: usize,
    voltage: f64,
    sigma0: f64,
    sigma_beta: f64,
    kappa: f64,
    fast_dt: f64,
    fast_dt_min: f64,
    use_subcycling: bool,
    max_coupling: usize,
    tol: f64,
    sync_error_tol: f64,
    sync_retries: usize,
    relax: f64,
}

struct CliArgs {
    sim: Args,
    checkpoint: Option<String>,
    checkpoint_h5: Option<String>,
    restart: Option<String>,
    restart_h5: Option<String>,
    export_vtk_prefix: Option<String>,
}

struct JouleCheckpointState {
    args: Args,
    completed_iterations: usize,
    converged: bool,
    final_relative_change: f64,
    sigma_effective: f64,
    joule_power: f64,
    observed_sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
    joule_power_tracker_prev: Option<f64>,
    phi: Vec<f64>,
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
        .map(read_joule_checkpoint)
        .transpose()
        .unwrap_or_else(|e| panic!("failed to read restart state: {e}"))
        .or_else(|| {
            cli.restart_h5
                .as_deref()
                .map(read_joule_hdf5_checkpoint)
                .transpose()
                .unwrap_or_else(|e| panic!("failed to read HDF5 restart state: {e}"))
        });
    let mut args = cli.sim.clone();
    if let Some(state) = restart_state.as_ref() {
        let requested_iters = args.max_coupling.max(state.completed_iterations);
        args = state.args.clone();
        args.max_coupling = requested_iters;
    }
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::JouleHeating);

    let config_line = format!(
        "n={}, V={}, sigma0={}, beta={}, kappa={}, fast_dt={}, fast_dt_min={}, subcycling={}, max_coupling={}, tol={}, sync_error_tol={}, sync_retries={}, relax={}",
        args.n,
        args.voltage,
        args.sigma0,
        args.sigma_beta,
        args.kappa,
        args.fast_dt,
        args.fast_dt_min,
        args.use_subcycling,
        args.max_coupling,
        args.tol,
        args.sync_error_tol,
        args.sync_retries,
        args.relax,
    );
    print_template_header("Example 48: Built-in template driver", spec, &config_line);

    let result = if let Some(restart) = restart_state.as_ref() {
        solve_joule_template_with_restart(&args, Some(restart))
    } else {
        solve_joule_template(&args)
    };

    println!("  converged: {}", result.converged);
    println!("  coupling iterations: {}", result.iterations);
    println!("  final relative thermal change: {:.3e}", result.final_relative_change);
    println!("  effective sigma(T): {:.6e}", result.sigma_effective);
    println!("  ||phi||_2: {:.6e}", result.phi_norm);
    println!("  ||T||_2: {:.6e}", result.temp_norm);
    println!("  integrated Joule power: {:.6e}", result.joule_power);
    println!("  temperature checksum: {:.8e}", result.temp_checksum);
    let coupling = TemplateCouplingSummary {
        steps: result.iterations,
        converged_steps: if result.converged { result.iterations } else { 0 },
        max_coupling_iters_used: result.iterations,
    };
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
            ("final_relative_change", result.final_relative_change),
            ("sigma_effective", result.sigma_effective),
            ("joule_power", result.joule_power),
            ("temp_norm", result.temp_norm),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }

    if let Some(path) = &cli.checkpoint {
        let checkpoint = JouleCheckpointState {
            args: args.clone(),
            completed_iterations: result.completed_iterations,
            converged: result.converged,
            final_relative_change: result.final_relative_change,
            sigma_effective: result.sigma_effective,
            joule_power: result.joule_power,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            joule_power_tracker_prev: result.joule_power_tracker_prev,
            phi: result.phi.clone(),
            temperature: result.temperature.clone(),
        };
        if let Err(e) = write_joule_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write checkpoint: {e}");
        } else {
            println!("  checkpoint written: {path}");
        }
    }

    if let Some(path) = &cli.checkpoint_h5 {
        let checkpoint = JouleCheckpointState {
            args: args.clone(),
            completed_iterations: result.completed_iterations,
            converged: result.converged,
            final_relative_change: result.final_relative_change,
            sigma_effective: result.sigma_effective,
            joule_power: result.joule_power,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            joule_power_tracker_prev: result.joule_power_tracker_prev,
            phi: result.phi.clone(),
            temperature: result.temperature.clone(),
        };
        if let Err(e) = write_joule_hdf5_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write HDF5 checkpoint: {e}");
        } else {
            println!("  HDF5 checkpoint written: {path}");
            #[cfg(feature = "io_hdf5")]
            if let Err(e) = write_joule_hdf5_xdmf_sidecars(path, &checkpoint) {
                eprintln!("warning: failed to write checkpoint XDMF sidecars: {e}");
            }
        }
    }

    if let Some(prefix) = &cli.export_vtk_prefix {
        if let Err(e) = write_ex48_vtk_export(prefix, &result.mesh, &result.phi, &result.temperature) {
            eprintln!("warning: failed to write VTK export: {e}");
        } else {
            println!("  VTK export written: {prefix}_joule_heating.vtu");
        }
    }
}

fn solve_joule_template(args: &Args) -> JouleTemplateResult {
    solve_joule_template_with_restart(args, None)
}

fn solve_joule_template_with_restart(
    args: &Args,
    restart: Option<&JouleCheckpointState>,
) -> JouleTemplateResult {
    if args.use_subcycling {
        solve_joule_template_subcycling(args, restart)
    } else {
        solve_joule_template_single_rate(args, restart)
    }
}

fn solve_joule_template_single_rate(
    args: &Args,
    restart: Option<&JouleCheckpointState>,
) -> JouleTemplateResult {
    let mesh = Mesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let dm = space.dof_manager();
    let all_boundary = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let left_bnd = boundary_dofs(space.mesh(), dm, &[4]);
    let right_bnd = boundary_dofs(space.mesh(), dm, &[2]);

    let cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };

    let completed_iterations = restart.map(|state| state.completed_iterations).unwrap_or(0);
    let mut temp = restart
        .map(|state| state.temperature.clone())
        .unwrap_or_else(|| vec![0.0_f64; n_dofs]);
    let mut phi = restart
        .map(|state| state.phi.clone())
        .unwrap_or_else(|| vec![0.0_f64; n_dofs]);
    let mut sigma_eff = restart
        .map(|state| state.sigma_effective)
        .unwrap_or_else(|| args.sigma0.max(1.0e-12));
    let mut final_rel = restart
        .map(|state| state.final_relative_change)
        .unwrap_or(f64::INFINITY);
    let mut joule_power = restart.map(|state| state.joule_power).unwrap_or(0.0_f64);
    let mut iters_done = completed_iterations;
    let mut converged = restart.map(|state| state.converged).unwrap_or(false);

    if completed_iterations >= args.max_coupling || converged {
        return JouleTemplateResult {
            completed_iterations,
            converged,
            iterations: completed_iterations,
            final_relative_change: final_rel,
            sigma_effective: sigma_eff,
            phi_norm: l2_norm(&phi),
            temp_norm: l2_norm(&temp),
            joule_power,
            temp_checksum: checksum(&temp),
            sync_retries: restart.map(|state| state.observed_sync_retries).unwrap_or(0),
            rejected_sync_steps: restart.map(|state| state.rejected_sync_steps).unwrap_or(0),
            rollback_count: restart.map(|state| state.rollback_count).unwrap_or(0),
            joule_power_tracker_prev: None,
            phi,
            temperature: temp,
            mesh: space.mesh().clone(),
        };
    }

    for k in completed_iterations..args.max_coupling {
        let t_mean = temp.iter().sum::<f64>() / n_dofs as f64;
        sigma_eff = (args.sigma0 * (1.0 + args.sigma_beta * t_mean)).max(1.0e-12);

        // Electric solve: -div(sigma_eff grad(phi)) = 0, phi=0(left), phi=V(right).
        let sigma_coeff = FnCoeff(move |_x: &[f64]| sigma_eff);
        let mut a_phi = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa: sigma_coeff }],
            3,
        );
        let mut rhs_phi = vec![0.0_f64; n_dofs];
        apply_dirichlet(&mut a_phi, &mut rhs_phi, &left_bnd, &vec![0.0; left_bnd.len()]);
        apply_dirichlet(
            &mut a_phi,
            &mut rhs_phi,
            &right_bnd,
            &vec![args.voltage; right_bnd.len()],
        );

        let phi_res = solve_pcg_jacobi(&a_phi, &rhs_phi, &mut phi, &cfg)
            .or_else(|_| solve_gmres(&a_phi, &rhs_phi, &mut phi, 60, &cfg))
            .expect("electric solve failed");
        if !phi_res.converged {
            // Keep running but this status may indicate under-resolved setup.
            log::warn!("electric solve did not fully converge at coupling iter {}", k + 1);
        }

        // Joule source q = sigma_eff * |grad(phi)|^2 (piecewise constant per element).
        let grads = compute_element_gradients(&space, &phi);
        let q_elem: Vec<f64> = grads
            .iter()
            .map(|g| sigma_eff * (g[0] * g[0] + g[1] * g[1]))
            .collect();

        joule_power = integrate_element_scalar(&space, &q_elem);

        // Thermal solve: -div(k grad(T)) = q, T=0 on all boundaries.
        let kappa = args.kappa;
        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            sample_piecewise_constant_on_mesh(&space, &q_elem, x)
        });
        let mut rhs_t = Assembler::assemble_linear(&space, &[&source], 3);
        let mut a_t = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa }],
            3,
        );
        apply_dirichlet(&mut a_t, &mut rhs_t, &all_boundary, &vec![0.0; all_boundary.len()]);

        let mut t_new = temp.clone();
        let t_res = solve_pcg_jacobi(&a_t, &rhs_t, &mut t_new, &cfg)
            .or_else(|_| solve_gmres(&a_t, &rhs_t, &mut t_new, 60, &cfg))
            .expect("thermal solve failed");
        if !t_res.converged {
            log::warn!("thermal solve did not fully converge at coupling iter {}", k + 1);
        }

        let mut diff2 = 0.0_f64;
        let mut base2 = 0.0_f64;
        for i in 0..n_dofs {
            let relaxed = (1.0 - args.relax) * temp[i] + args.relax * t_new[i];
            let d = relaxed - temp[i];
            diff2 += d * d;
            base2 += relaxed * relaxed;
            temp[i] = relaxed;
        }
        final_rel = diff2.sqrt() / base2.sqrt().max(1.0e-14);
        iters_done = k + 1;

        if final_rel <= args.tol {
            converged = true;
            break;
        }
    }

    let phi_norm = l2_norm(&phi);
    let temp_norm = l2_norm(&temp);
    let temp_checksum = checksum(&temp);

    JouleTemplateResult {
        completed_iterations: iters_done,
        converged,
        iterations: iters_done,
        final_relative_change: final_rel,
        sigma_effective: sigma_eff,
        phi_norm,
        temp_norm,
        joule_power,
        temp_checksum,
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
        joule_power_tracker_prev: None,
        phi,
        temperature: temp,
        mesh: space.mesh().clone(),
    }
}

fn solve_joule_template_subcycling(
    args: &Args,
    restart: Option<&JouleCheckpointState>,
) -> JouleTemplateResult {
    #[derive(Clone)]
    struct SubcyclingState {
        temp: Vec<f64>,
        phi: Vec<f64>,
        sigma_eff: f64,
        final_rel: f64,
        joule_power: f64,
        joule_power_tracker: RelativeScalarTracker,
        sync_error: f64,
        iters_done: usize,
        converged: bool,
    }

    let mesh = Mesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let dm = space.dof_manager();
    let all_boundary = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let left_bnd = boundary_dofs(space.mesh(), dm, &[4]);
    let right_bnd = boundary_dofs(space.mesh(), dm, &[2]);

    let cfg_solve = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 4000,
        verbose: false,
        ..SolverConfig::default()
    };

    let completed_iterations = restart.map(|state| state.completed_iterations).unwrap_or(0);
    let mut state = SubcyclingState {
        temp: restart
            .map(|checkpoint| checkpoint.temperature.clone())
            .unwrap_or_else(|| vec![0.0_f64; n_dofs]),
        phi: restart
            .map(|checkpoint| checkpoint.phi.clone())
            .unwrap_or_else(|| vec![0.0_f64; n_dofs]),
        sigma_eff: restart
            .map(|checkpoint| checkpoint.sigma_effective)
            .unwrap_or_else(|| args.sigma0.max(1.0e-12)),
        final_rel: restart
            .map(|checkpoint| checkpoint.final_relative_change)
            .unwrap_or(f64::INFINITY),
        joule_power: restart.map(|checkpoint| checkpoint.joule_power).unwrap_or(0.0),
        joule_power_tracker: seeded_scalar_tracker(
            restart.and_then(|checkpoint| checkpoint.joule_power_tracker_prev),
        ),
        sync_error: restart
            .map(|checkpoint| checkpoint.final_relative_change)
            .unwrap_or(f64::INFINITY),
        iters_done: completed_iterations,
        converged: restart.map(|checkpoint| checkpoint.converged).unwrap_or(false),
    };

    if completed_iterations >= args.max_coupling || state.converged {
        return JouleTemplateResult {
            completed_iterations,
            converged: state.converged,
            iterations: completed_iterations,
            final_relative_change: state.final_rel,
            sigma_effective: state.sigma_eff,
            phi_norm: l2_norm(&state.phi),
            temp_norm: l2_norm(&state.temp),
            joule_power: state.joule_power,
            temp_checksum: checksum(&state.temp),
            sync_retries: restart.map(|checkpoint| checkpoint.observed_sync_retries).unwrap_or(0),
            rejected_sync_steps: restart.map(|checkpoint| checkpoint.rejected_sync_steps).unwrap_or(0),
            rollback_count: restart.map(|checkpoint| checkpoint.rollback_count).unwrap_or(0),
            joule_power_tracker_prev: restart.and_then(|checkpoint| checkpoint.joule_power_tracker_prev),
            phi: state.phi,
            temperature: state.temp,
            mesh: space.mesh().clone(),
        };
    }

    let fast_dt = args.fast_dt.max(1.0e-12).min(1.0);
    let sched_cfg = MultiRateConfig {
        t_start: completed_iterations as f64,
        t_end: args.max_coupling as f64,
        fast_dt,
        slow_dt: 1.0,
    };

    let stats = run_multirate_adaptive(
        MultiRateAdaptiveConfig {
            base: sched_cfg,
            sync_error_tol: args.sync_error_tol,
            max_sync_retries: args.sync_retries,
            retry_fast_dt_scale: 0.5,
            min_fast_dt: args.fast_dt_min.max(1.0e-12),
        },
        &mut state,
        |state, _t_fast, _dt_fast| {
            if state.converged {
                return;
            }

            let t_mean = state.temp.iter().sum::<f64>() / n_dofs as f64;
            state.sigma_eff = (args.sigma0 * (1.0 + args.sigma_beta * t_mean)).max(1.0e-12);

            let sigma_eff = state.sigma_eff;
            let sigma_coeff = FnCoeff(move |_x: &[f64]| sigma_eff);
            let mut a_phi = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: sigma_coeff }],
                3,
            );
            let mut rhs_phi = vec![0.0_f64; n_dofs];
            apply_dirichlet(&mut a_phi, &mut rhs_phi, &left_bnd, &vec![0.0; left_bnd.len()]);
            apply_dirichlet(
                &mut a_phi,
                &mut rhs_phi,
                &right_bnd,
                &vec![args.voltage; right_bnd.len()],
            );

            let phi_res = solve_pcg_jacobi(&a_phi, &rhs_phi, &mut state.phi, &cfg_solve)
                .or_else(|_| solve_gmres(&a_phi, &rhs_phi, &mut state.phi, 60, &cfg_solve))
                .expect("electric solve failed");
            if !phi_res.converged {
                log::warn!("electric solve did not fully converge during subcycling");
            }
        },
        |_state, _t_slow, _dt_slow| {
            // Thermal update is performed at synchronization points.
        },
        |state, t_sync| {
            if state.converged {
                return 0.0;
            }

            let grads = compute_element_gradients(&space, &state.phi);
            let q_elem: Vec<f64> = grads
                .iter()
                .map(|g| state.sigma_eff * (g[0] * g[0] + g[1] * g[1]))
                .collect();
            state.joule_power = integrate_element_scalar(&space, &q_elem);

            let kappa = args.kappa;
            let source = DomainSourceIntegrator::new(|x: &[f64]| {
                sample_piecewise_constant_on_mesh(&space, &q_elem, x)
            });
            let mut rhs_t = Assembler::assemble_linear(&space, &[&source], 3);
            let mut a_t = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa }],
                3,
            );
            apply_dirichlet(
                &mut a_t,
                &mut rhs_t,
                &all_boundary,
                &vec![0.0; all_boundary.len()],
            );

            let mut t_new = state.temp.clone();
            let t_res = solve_pcg_jacobi(&a_t, &rhs_t, &mut t_new, &cfg_solve)
                .or_else(|_| solve_gmres(&a_t, &rhs_t, &mut t_new, 60, &cfg_solve))
                .expect("thermal solve failed");
            if !t_res.converged {
                log::warn!("thermal solve did not fully converge during subcycling");
            }

            let mut diff2 = 0.0_f64;
            let mut base2 = 0.0_f64;
            for i in 0..n_dofs {
                let relaxed = (1.0 - args.relax) * state.temp[i] + args.relax * t_new[i];
                let d = relaxed - state.temp[i];
                diff2 += d * d;
                base2 += relaxed * relaxed;
                state.temp[i] = relaxed;
            }
            state.final_rel = diff2.sqrt() / base2.sqrt().max(1.0e-14);
            state.iters_done = t_sync.round() as usize + 1;
            let rel_power = state
                .joule_power_tracker
                .observe(state.joule_power, state.final_rel);
            state.sync_error = compose_sync_error(&[state.final_rel, rel_power]);
            if state.final_rel <= args.tol {
                state.converged = true;
            }

            state.sync_error
        },
    )
    .expect("adaptive subcycling scheduler failed");

    let phi_norm = l2_norm(&state.phi);
    let temp_norm = l2_norm(&state.temp);
    let temp_checksum = checksum(&state.temp);

    JouleTemplateResult {
        completed_iterations: if state.converged {
            state.iters_done.min(args.max_coupling)
        } else {
            args.max_coupling
        },
        converged: state.converged,
        iterations: if state.converged {
            state.iters_done.min(args.max_coupling)
        } else {
            args.max_coupling
        },
        final_relative_change: state.final_rel,
        sigma_effective: state.sigma_eff,
        phi_norm,
        temp_norm,
        joule_power: state.joule_power,
        temp_checksum,
        sync_retries: restart.map(|checkpoint| checkpoint.observed_sync_retries).unwrap_or(0)
            + stats.sync_retries,
        rejected_sync_steps: restart.map(|checkpoint| checkpoint.rejected_sync_steps).unwrap_or(0)
            + stats.rejected_sync_steps,
        rollback_count: restart.map(|checkpoint| checkpoint.rollback_count).unwrap_or(0)
            + stats.rollback_count,
        joule_power_tracker_prev: Some(state.joule_power),
        phi: state.phi,
        temperature: state.temp,
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

fn integrate_element_scalar(space: &H1Space<Mesh<2>>, elem_values: &[f64]) -> f64 {
    let mesh = space.mesh();
    let mut acc = 0.0_f64;
    for (e, &value) in mesh.elem_iter().zip(elem_values.iter()) {
        let area = tri_area(mesh, e);
        acc += value * area;
    }
    acc
}

fn tri_area(mesh: &Mesh<2>, elem: u32) -> f64 {
    let ns = mesh.elem_nodes(elem);
    let a = mesh.coords_of(ns[0]);
    let b = mesh.coords_of(ns[1]);
    let c = mesh.coords_of(ns[2]);
    let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    0.5 * det.abs()
}

fn sample_piecewise_constant_on_mesh(
    space: &H1Space<Mesh<2>>,
    elem_values: &[f64],
    x: &[f64],
) -> f64 {
    let mesh = space.mesh();
    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        let a = mesh.coords_of(ns[0]);
        let b = mesh.coords_of(ns[1]);
        let c = mesh.coords_of(ns[2]);
        if point_in_triangle_2d(x, &a, &b, &c, 1.0e-12) {
            return elem_values[e as usize];
        }
    }

    // Fallback for points numerically outside all elements.
    let mut best_e = 0usize;
    let mut best_d2 = f64::INFINITY;
    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        let a = mesh.coords_of(ns[0]);
        let b = mesh.coords_of(ns[1]);
        let c = mesh.coords_of(ns[2]);
        let xc = (a[0] + b[0] + c[0]) / 3.0;
        let yc = (a[1] + b[1] + c[1]) / 3.0;
        let d2 = (x[0] - xc).powi(2) + (x[1] - yc).powi(2);
        if d2 < best_d2 {
            best_d2 = d2;
            best_e = e as usize;
        }
    }
    elem_values[best_e]
}

fn point_in_triangle_2d(p: &[f64], a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], tol: f64) -> bool {
    let v0 = [c[0] - a[0], c[1] - a[1]];
    let v1 = [b[0] - a[0], b[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];

    let dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    let dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    let dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    let dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    let dot12 = v1[0] * v2[0] + v1[1] * v2[1];

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < 1.0e-30 {
        return false;
    }
    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;
    u >= -tol && v >= -tol && (u + v) <= 1.0 + tol
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

fn write_ex48_vtk_export(
    prefix: &str,
    mesh: &Mesh<2>,
    phi: &[f64],
    temperature: &[f64],
) -> Result<(), String> {
    let path = format!("{prefix}_joule_heating.vtu");
    ensure_parent_dir(&path).map_err(|e| e.to_string())?;
    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("electric_potential", phi.to_vec()));
    writer.add_point_data(DataArray::scalars("temperature", temperature.to_vec()));
    writer.write_file(&path).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_joule_checkpoint(path: &str, state: &JouleCheckpointState) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let phi = format_vec_f64(&state.phi);
    let temperature = format_vec_f64(&state.temperature);
    let content = format!(
        "format=ex48_joule_heating_v1\nn={}\nvoltage={:.17e}\nsigma0={:.17e}\nsigma_beta={:.17e}\nkappa={:.17e}\nfast_dt={:.17e}\nfast_dt_min={:.17e}\nsubcycling={}\nmax_coupling={}\ntol={:.17e}\nsync_error_tol={:.17e}\nsync_retries={}\nrelax={:.17e}\ncompleted_iterations={}\nconverged={}\nfinal_relative_change={:.17e}\nsigma_effective={:.17e}\njoule_power={:.17e}\nobserved_sync_retries={}\nrejected_sync_steps={}\nrollback_count={}\njoule_power_tracker_prev={}\nphi={}\ntemperature={}\n",
        state.args.n,
        state.args.voltage,
        state.args.sigma0,
        state.args.sigma_beta,
        state.args.kappa,
        state.args.fast_dt,
        state.args.fast_dt_min,
        if state.args.use_subcycling { 1 } else { 0 },
        state.args.max_coupling,
        state.args.tol,
        state.args.sync_error_tol,
        state.args.sync_retries,
        state.args.relax,
        state.completed_iterations,
        if state.converged { 1 } else { 0 },
        state.final_relative_change,
        state.sigma_effective,
        state.joule_power,
        state.observed_sync_retries,
        state.rejected_sync_steps,
        state.rollback_count,
        state
            .joule_power_tracker_prev
            .map(|v| format!("{v:.17e}"))
            .unwrap_or_default(),
        phi,
        temperature,
    );
    fs::write(path, content)
}

fn write_joule_hdf5_checkpoint(path: &str, state: &JouleCheckpointState) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(path);

    let bundle = CheckpointBundleF64 {
        mesh_meta: None,
        fields: vec![
            scalar_rank_field_f64("n", state.args.n as f64),
            scalar_rank_field_f64("voltage", state.args.voltage),
            scalar_rank_field_f64("sigma0", state.args.sigma0),
            scalar_rank_field_f64("sigma_beta", state.args.sigma_beta),
            scalar_rank_field_f64("kappa", state.args.kappa),
            scalar_rank_field_f64("fast_dt", state.args.fast_dt),
            scalar_rank_field_f64("fast_dt_min", state.args.fast_dt_min),
            scalar_rank_field_f64("subcycling", if state.args.use_subcycling { 1.0 } else { 0.0 }),
            scalar_rank_field_f64("max_coupling", state.args.max_coupling as f64),
            scalar_rank_field_f64("tol", state.args.tol),
            scalar_rank_field_f64("sync_error_tol", state.args.sync_error_tol),
            scalar_rank_field_f64("sync_retries", state.args.sync_retries as f64),
            scalar_rank_field_f64("relax", state.args.relax),
            scalar_rank_field_f64("completed_iterations", state.completed_iterations as f64),
            scalar_rank_field_f64("converged", if state.converged { 1.0 } else { 0.0 }),
            scalar_rank_field_f64("final_relative_change", state.final_relative_change),
            scalar_rank_field_f64("sigma_effective", state.sigma_effective),
            scalar_rank_field_f64("joule_power", state.joule_power),
            scalar_rank_field_f64("observed_sync_retries", state.observed_sync_retries as f64),
            scalar_rank_field_f64("rejected_sync_steps", state.rejected_sync_steps as f64),
            scalar_rank_field_f64("rollback_count", state.rollback_count as f64),
            scalar_rank_field_f64("joule_power_tracker_prev", state.joule_power_tracker_prev.unwrap_or(f64::NAN)),
            vector_rank_field_f64("electric_potential", state.phi.clone()),
            vector_rank_field_f64("temperature", state.temperature.clone()),
        ],
    };
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let step = state.completed_iterations.max(1) as u64;
    write_checkpoint_step_bundle_f64(path, cfg, step, state.completed_iterations.max(1) as f64, &bundle, IoBackend::Partitioned)
        .map_err(|e| e.to_string())?;
    validate_checkpoint_layout(path, Some(1)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_joule_checkpoint(path: &str) -> Result<JouleCheckpointState, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut format = None;
    let mut n = None;
    let mut voltage = None;
    let mut sigma0 = None;
    let mut sigma_beta = None;
    let mut kappa = None;
    let mut fast_dt = None;
    let mut fast_dt_min = None;
    let mut subcycling = None;
    let mut max_coupling = None;
    let mut tol = None;
    let mut sync_error_tol = None;
    let mut sync_retries = None;
    let mut relax = None;
    let mut completed_iterations = None;
    let mut converged = None;
    let mut final_relative_change = None;
    let mut sigma_effective = None;
    let mut joule_power = None;
    let mut observed_sync_retries = None;
    let mut rejected_sync_steps = None;
    let mut rollback_count = None;
    let mut joule_power_tracker_prev = None;
    let mut phi = None;
    let mut temperature = None;

    for line in content.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "format" => format = Some(value.to_string()),
            "n" => n = value.parse::<usize>().ok(),
            "voltage" => voltage = value.parse::<f64>().ok(),
            "sigma0" => sigma0 = value.parse::<f64>().ok(),
            "sigma_beta" => sigma_beta = value.parse::<f64>().ok(),
            "kappa" => kappa = value.parse::<f64>().ok(),
            "fast_dt" => fast_dt = value.parse::<f64>().ok(),
            "fast_dt_min" => fast_dt_min = value.parse::<f64>().ok(),
            "subcycling" => subcycling = Some(value == "1"),
            "max_coupling" => max_coupling = value.parse::<usize>().ok(),
            "tol" => tol = value.parse::<f64>().ok(),
            "sync_error_tol" => sync_error_tol = value.parse::<f64>().ok(),
            "sync_retries" => sync_retries = value.parse::<usize>().ok(),
            "relax" => relax = value.parse::<f64>().ok(),
            "completed_iterations" => completed_iterations = value.parse::<usize>().ok(),
            "converged" => converged = Some(value == "1"),
            "final_relative_change" => final_relative_change = value.parse::<f64>().ok(),
            "sigma_effective" => sigma_effective = value.parse::<f64>().ok(),
            "joule_power" => joule_power = value.parse::<f64>().ok(),
            "observed_sync_retries" => observed_sync_retries = value.parse::<usize>().ok(),
            "rejected_sync_steps" => rejected_sync_steps = value.parse::<usize>().ok(),
            "rollback_count" => rollback_count = value.parse::<usize>().ok(),
            "joule_power_tracker_prev" => {
                joule_power_tracker_prev = if value.trim().is_empty() {
                    Some(None)
                } else {
                    Some(Some(value.parse::<f64>().map_err(|e| e.to_string())?))
                }
            }
            "phi" => phi = Some(parse_vec_f64(value)?),
            "temperature" => temperature = Some(parse_vec_f64(value)?),
            _ => {}
        }
    }

    if format.as_deref() != Some("ex48_joule_heating_v1") {
        return Err("unsupported checkpoint format".into());
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        voltage: voltage.ok_or_else(|| "missing voltage".to_string())?,
        sigma0: sigma0.ok_or_else(|| "missing sigma0".to_string())?,
        sigma_beta: sigma_beta.ok_or_else(|| "missing sigma_beta".to_string())?,
        kappa: kappa.ok_or_else(|| "missing kappa".to_string())?,
        fast_dt: fast_dt.ok_or_else(|| "missing fast_dt".to_string())?,
        fast_dt_min: fast_dt_min.ok_or_else(|| "missing fast_dt_min".to_string())?,
        use_subcycling: subcycling.ok_or_else(|| "missing subcycling".to_string())?,
        max_coupling: max_coupling.ok_or_else(|| "missing max_coupling".to_string())?,
        tol: tol.ok_or_else(|| "missing tol".to_string())?,
        sync_error_tol: sync_error_tol.ok_or_else(|| "missing sync_error_tol".to_string())?,
        sync_retries: sync_retries.ok_or_else(|| "missing sync_retries".to_string())?,
        relax: relax.ok_or_else(|| "missing relax".to_string())?,
    };
    let phi = phi.ok_or_else(|| "missing phi".to_string())?;
    let temperature = temperature.ok_or_else(|| "missing temperature".to_string())?;
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if phi.len() != expected_dofs || temperature.len() != expected_dofs {
        return Err(format!(
            "checkpoint field lengths do not match expected dofs ({expected_dofs})"
        ));
    }

    Ok(JouleCheckpointState {
        args,
        completed_iterations: completed_iterations
            .ok_or_else(|| "missing completed_iterations".to_string())?,
        converged: converged.ok_or_else(|| "missing converged".to_string())?,
        final_relative_change: final_relative_change
            .ok_or_else(|| "missing final_relative_change".to_string())?,
        sigma_effective: sigma_effective.ok_or_else(|| "missing sigma_effective".to_string())?,
        joule_power: joule_power.ok_or_else(|| "missing joule_power".to_string())?,
        observed_sync_retries: observed_sync_retries
            .ok_or_else(|| "missing observed_sync_retries".to_string())?,
        rejected_sync_steps: rejected_sync_steps
            .ok_or_else(|| "missing rejected_sync_steps".to_string())?,
        rollback_count: rollback_count.ok_or_else(|| "missing rollback_count".to_string())?,
        joule_power_tracker_prev: joule_power_tracker_prev.unwrap_or(None),
        phi,
        temperature,
    })
}

fn read_joule_hdf5_checkpoint(path: &str) -> Result<JouleCheckpointState, String> {
    let fields = read_checkpoint_fields_f64_latest(
        path,
        ParallelIoConfig { world_size: 1, rank: 0 },
        &[
            "n",
            "voltage",
            "sigma0",
            "sigma_beta",
            "kappa",
            "fast_dt",
            "fast_dt_min",
            "subcycling",
            "max_coupling",
            "tol",
            "sync_error_tol",
            "sync_retries",
            "relax",
            "completed_iterations",
            "converged",
            "final_relative_change",
            "sigma_effective",
            "joule_power",
            "observed_sync_retries",
            "rejected_sync_steps",
            "rollback_count",
            "joule_power_tracker_prev",
            "electric_potential",
            "temperature",
        ],
    )
    .map_err(|e| e.to_string())?;

    let mut n = None;
    let mut voltage = None;
    let mut sigma0 = None;
    let mut sigma_beta = None;
    let mut kappa = None;
    let mut fast_dt = None;
    let mut fast_dt_min = None;
    let mut subcycling = None;
    let mut max_coupling = None;
    let mut tol = None;
    let mut sync_error_tol = None;
    let mut sync_retries = None;
    let mut relax = None;
    let mut completed_iterations = None;
    let mut converged = None;
    let mut final_relative_change = None;
    let mut sigma_effective = None;
    let mut joule_power = None;
    let mut observed_sync_retries = None;
    let mut rejected_sync_steps = None;
    let mut rollback_count = None;
    let mut joule_power_tracker_prev = None;
    let mut phi = None;
    let mut temperature = None;

    for (name, field) in fields {
        match name.as_str() {
            "n" => n = field.values.first().map(|v| *v as usize),
            "voltage" => voltage = field.values.first().copied(),
            "sigma0" => sigma0 = field.values.first().copied(),
            "sigma_beta" => sigma_beta = field.values.first().copied(),
            "kappa" => kappa = field.values.first().copied(),
            "fast_dt" => fast_dt = field.values.first().copied(),
            "fast_dt_min" => fast_dt_min = field.values.first().copied(),
            "subcycling" => subcycling = field.values.first().map(|v| *v != 0.0),
            "max_coupling" => max_coupling = field.values.first().map(|v| *v as usize),
            "tol" => tol = field.values.first().copied(),
            "sync_error_tol" => sync_error_tol = field.values.first().copied(),
            "sync_retries" => sync_retries = field.values.first().map(|v| *v as usize),
            "relax" => relax = field.values.first().copied(),
            "completed_iterations" => completed_iterations = field.values.first().map(|v| *v as usize),
            "converged" => converged = field.values.first().map(|v| *v != 0.0),
            "final_relative_change" => final_relative_change = field.values.first().copied(),
            "sigma_effective" => sigma_effective = field.values.first().copied(),
            "joule_power" => joule_power = field.values.first().copied(),
            "observed_sync_retries" => observed_sync_retries = field.values.first().map(|v| *v as usize),
            "rejected_sync_steps" => rejected_sync_steps = field.values.first().map(|v| *v as usize),
            "rollback_count" => rollback_count = field.values.first().map(|v| *v as usize),
            "joule_power_tracker_prev" => {
                joule_power_tracker_prev = field.values.first().copied().filter(|v| v.is_finite())
            }
            "electric_potential" => phi = Some(field.values),
            "temperature" => temperature = Some(field.values),
            _ => {}
        }
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        voltage: voltage.ok_or_else(|| "missing voltage".to_string())?,
        sigma0: sigma0.ok_or_else(|| "missing sigma0".to_string())?,
        sigma_beta: sigma_beta.ok_or_else(|| "missing sigma_beta".to_string())?,
        kappa: kappa.ok_or_else(|| "missing kappa".to_string())?,
        fast_dt: fast_dt.ok_or_else(|| "missing fast_dt".to_string())?,
        fast_dt_min: fast_dt_min.ok_or_else(|| "missing fast_dt_min".to_string())?,
        use_subcycling: subcycling.ok_or_else(|| "missing subcycling".to_string())?,
        max_coupling: max_coupling.ok_or_else(|| "missing max_coupling".to_string())?,
        tol: tol.ok_or_else(|| "missing tol".to_string())?,
        sync_error_tol: sync_error_tol.ok_or_else(|| "missing sync_error_tol".to_string())?,
        sync_retries: sync_retries.ok_or_else(|| "missing sync_retries".to_string())?,
        relax: relax.ok_or_else(|| "missing relax".to_string())?,
    };
    let phi = phi.ok_or_else(|| "missing electric_potential".to_string())?;
    let temperature = temperature.ok_or_else(|| "missing temperature".to_string())?;
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if phi.len() != expected_dofs || temperature.len() != expected_dofs {
        return Err(format!(
            "checkpoint field lengths do not match expected dofs ({expected_dofs})"
        ));
    }

    Ok(JouleCheckpointState {
        args,
        completed_iterations: completed_iterations
            .ok_or_else(|| "missing completed_iterations".to_string())?,
        converged: converged.ok_or_else(|| "missing converged".to_string())?,
        final_relative_change: final_relative_change
            .ok_or_else(|| "missing final_relative_change".to_string())?,
        sigma_effective: sigma_effective.ok_or_else(|| "missing sigma_effective".to_string())?,
        joule_power: joule_power.ok_or_else(|| "missing joule_power".to_string())?,
        observed_sync_retries: observed_sync_retries
            .ok_or_else(|| "missing observed_sync_retries".to_string())?,
        rejected_sync_steps: rejected_sync_steps
            .ok_or_else(|| "missing rejected_sync_steps".to_string())?,
        rollback_count: rollback_count.ok_or_else(|| "missing rollback_count".to_string())?,
        joule_power_tracker_prev,
        phi,
        temperature,
    })
}

#[cfg(feature = "io_hdf5")]
fn write_joule_hdf5_xdmf_sidecars(h5_path: &str, state: &JouleCheckpointState) -> Result<(), String> {
    let step = state.completed_iterations.max(1);
    write_scalar_checkpoint_xdmf_sidecars(
        h5_path,
        step as u64,
        step as f64,
        &["electric_potential", "temperature"],
    )
}

fn parse_args() -> CliArgs {
    let mut a = Args {
        n: 16,
        voltage: 1.0,
        sigma0: 5.0,
        sigma_beta: 0.02,
        kappa: 1.0,
        fast_dt: 0.2,
        fast_dt_min: 1.0e-3,
        use_subcycling: true,
        max_coupling: 20,
        tol: 1.0e-8,
        sync_error_tol: 1.0,
        sync_retries: 2,
        relax: 0.7,
    };
    let mut workflow = WorkflowCliOptions::default();

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex48_template_joule_heating");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        let mut help_options = vec![
            ("--n <int>", "Mesh resolution (default: 16)"),
            ("--voltage <float>", "Right-boundary voltage (default: 1.0)"),
            ("--sigma0 <float>", "Reference conductivity (default: 5.0)"),
            (
                "--sigma-beta <float>",
                "Temperature conductivity slope (default: 0.02)",
            ),
            ("--kappa <float>", "Thermal diffusivity (default: 1.0)"),
            (
                "--fast-dt <float>",
                "Fast subcycling pseudo-step size (default: 0.2)",
            ),
            (
                "--fast-dt-min <float>",
                "Minimum fast subcycling step during retries (default: 1e-3)",
            ),
            ("--subcycling", "Enable multirate subcycling (default)"),
            ("--no-subcycling", "Disable subcycling and use single-rate loop"),
            (
                "--max-coupling <int>",
                "Maximum coupling iterations (default: 20)",
            ),
            ("--tol <float>", "Coupling convergence tolerance (default: 1e-8)"),
            (
                "--sync-error-tol <float>",
                "Adaptive sync acceptance tolerance (default: 1.0)",
            ),
            (
                "--sync-retries <int>",
                "Max adaptive retry count at each sync point (default: 2)",
            ),
            ("--relax <float>", "Relaxation factor in [0.1, 1.0] (default: 0.7)"),
        ];
        push_workflow_cli_help(
            &mut help_options,
            "Write final potential/temperature VTK export as <prefix>_joule_heating.vtu",
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
            "--n" => a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16),
            "--voltage" => a.voltage = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--sigma0" => a.sigma0 = it.next().unwrap_or("5.0".into()).parse().unwrap_or(5.0),
            "--sigma-beta" => {
                a.sigma_beta = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02)
            }
            "--kappa" => a.kappa = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--fast-dt" => a.fast_dt = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            "--fast-dt-min" => {
                a.fast_dt_min = it.next().unwrap_or("1e-3".into()).parse().unwrap_or(1.0e-3)
            }
            "--subcycling" => a.use_subcycling = true,
            "--no-subcycling" => a.use_subcycling = false,
            "--max-coupling" => {
                a.max_coupling = it.next().unwrap_or("20".into()).parse().unwrap_or(20)
            }
            "--tol" => a.tol = it.next().unwrap_or("1e-8".into()).parse().unwrap_or(1.0e-8),
            "--sync-error-tol" => {
                a.sync_error_tol = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--sync-retries" => {
                a.sync_retries = it.next().unwrap_or("2".into()).parse().unwrap_or(2)
            }
            "--relax" => a.relax = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            _ => {}
        }
    }

    a.relax = a.relax.clamp(0.1, 1.0);
    a.fast_dt = a.fast_dt.max(1.0e-12);
    a.fast_dt_min = a.fast_dt_min.max(1.0e-12).min(a.fast_dt);
    a.sync_error_tol = a.sync_error_tol.max(0.0);
    a.max_coupling = a.max_coupling.max(1);
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
                "ex48_{}_{}_{}.{}",
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
            voltage: 1.0,
            sigma0: 5.0,
            sigma_beta: 0.02,
            kappa: 1.0,
            fast_dt: 0.2,
            fast_dt_min: 1.0e-3,
            use_subcycling: true,
            max_coupling: 20,
            tol: 1.0e-8,
            sync_error_tol: 1.0,
            sync_retries: 2,
            relax: 0.7,
        }
    }

    #[test]
    fn ex48_joule_template_converges_and_produces_positive_thermal_state() {
        let r = solve_joule_template(&base_args());
        assert!(r.converged, "coupling did not converge in {} iterations", r.iterations);
        assert!(r.iterations <= 20);
        assert!(r.temp_norm > 0.0);
        assert!(r.joule_power > 0.0);
        assert!(r.final_relative_change <= 1.0e-8);
    }

    #[test]
    fn ex48_higher_voltage_increases_joule_power_and_temperature() {
        let mut low = base_args();
        low.voltage = 0.5;
        let mut high = base_args();
        high.voltage = 1.5;

        let r_low = solve_joule_template(&low);
        let r_high = solve_joule_template(&high);

        assert!(r_high.joule_power > r_low.joule_power);
        assert!(r_high.temp_norm > r_low.temp_norm);
    }

    #[test]
    #[should_panic(expected = "adaptive subcycling scheduler failed")]
    fn ex48_strict_sync_error_tol_can_trigger_adaptive_failure() {
        let mut a = base_args();
        a.sync_error_tol = 0.0;
        a.sync_retries = 0;
        a.max_coupling = 4;
        let _ = solve_joule_template(&a);
    }

    /// Higher thermal conductivity kappa → faster heat diffusion →
    /// lower steady-state temperature norm (same Joule source, more cooling).
    #[test]
    fn ex48_higher_kappa_gives_lower_temperature() {
        let mut low_k = base_args();
        low_k.kappa = 0.5;
        let mut high_k = base_args();
        high_k.kappa = 5.0;

        let r_low  = solve_joule_template(&low_k);
        let r_high = solve_joule_template(&high_k);
        assert!(r_high.temp_norm < r_low.temp_norm,
            "higher kappa should give lower temperature: kappa=0.5 -> {:.4e}, kappa=5.0 -> {:.4e}",
            r_low.temp_norm, r_high.temp_norm);
    }

    /// Positive temperature coefficient sigma_beta: higher temperature increases
    /// conductivity, which increases Joule power — runaway effect.
    /// Therefore sigma_beta > 0 should yield MORE Joule power than sigma_beta = 0.
    #[test]
    fn ex48_positive_sigma_beta_increases_joule_power() {
        let mut zero_fb = base_args();
        zero_fb.sigma_beta = 0.0;
        let mut pos_fb = base_args();
        pos_fb.sigma_beta = 0.1;

        let r_zero = solve_joule_template(&zero_fb);
        let r_pos  = solve_joule_template(&pos_fb);
        assert!(r_pos.joule_power >= r_zero.joule_power,
            "positive sigma_beta should not decrease Joule power: zero={:.4e} pos={:.4e}",
            r_zero.joule_power, r_pos.joule_power);
    }

    #[test]
    fn ex48_single_rate_mode_has_no_adaptive_counters() {
        let mut a = base_args();
        a.use_subcycling = false;
        let r = solve_joule_template(&a);
        assert!(r.converged, "single-rate coupling should converge");
        assert_eq!(r.sync_retries, 0);
        assert_eq!(r.rejected_sync_steps, 0);
        assert_eq!(r.rollback_count, 0);
    }

    #[test]
    fn ex48_higher_sigma0_increases_joule_power() {
        let mut low = base_args();
        low.sigma0 = 2.0;
        let mut high = base_args();
        high.sigma0 = 8.0;

        let r_low = solve_joule_template(&low);
        let r_high = solve_joule_template(&high);
        assert!(r_low.converged && r_high.converged);
        assert!(r_high.joule_power > r_low.joule_power,
            "higher sigma0 should increase Joule power: low={:.4e} high={:.4e}",
            r_low.joule_power, r_high.joule_power);
    }

    #[test]
    fn ex48_repeated_runs_are_deterministic_for_same_inputs() {
        let a = base_args();
        let r1 = solve_joule_template(&a);
        let r2 = solve_joule_template(&a);
        assert!((r1.temp_checksum - r2.temp_checksum).abs() < 1.0e-12,
            "temperature checksum should be deterministic: r1={} r2={}",
            r1.temp_checksum,
            r2.temp_checksum);
        assert!((r1.joule_power - r2.joule_power).abs() < 1.0e-12,
            "joule power should be deterministic: r1={} r2={}",
            r1.joule_power,
            r2.joule_power);
    }

    #[test]
    fn ex48_text_checkpoint_roundtrip_preserves_restart_state() {
        let args = base_args();
        let partial = solve_joule_template_with_restart(
            &Args {
                max_coupling: 4,
                ..args.clone()
            },
            None,
        );
        let path = temp_output_path("checkpoint", "txt");
        let checkpoint = JouleCheckpointState {
            args: Args {
                max_coupling: 4,
                ..args.clone()
            },
            completed_iterations: partial.completed_iterations,
            converged: partial.converged,
            final_relative_change: partial.final_relative_change,
            sigma_effective: partial.sigma_effective,
            joule_power: partial.joule_power,
            observed_sync_retries: partial.sync_retries,
            rejected_sync_steps: partial.rejected_sync_steps,
            rollback_count: partial.rollback_count,
            joule_power_tracker_prev: partial.joule_power_tracker_prev,
            phi: partial.phi.clone(),
            temperature: partial.temperature.clone(),
        };

        write_joule_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_joule_checkpoint(&path).unwrap();

        assert_eq!(restored.completed_iterations, checkpoint.completed_iterations);
        assert_eq!(restored.args.n, checkpoint.args.n);
        assert_eq!(restored.args.use_subcycling, checkpoint.args.use_subcycling);
        assert_eq!(restored.phi, checkpoint.phi);
        assert_eq!(restored.temperature, checkpoint.temperature);

        let resumed = solve_joule_template_with_restart(&args, Some(&restored));
        let full = solve_joule_template(&args);
        assert!((resumed.temp_checksum - full.temp_checksum).abs() < 1.0e-12);
        assert!((resumed.joule_power - full.joule_power).abs() < 1.0e-12);
        assert!((resumed.sigma_effective - full.sigma_effective).abs() < 1.0e-12);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn ex48_vtk_export_writes_phi_and_temperature() {
        let result = solve_joule_template(&base_args());
        let prefix = temp_output_path("vtk", "out");
        let vtk_path = format!("{prefix}_joule_heating.vtu");

        write_ex48_vtk_export(&prefix, &result.mesh, &result.phi, &result.temperature).unwrap();

        let vtk = fs::read_to_string(&vtk_path).unwrap();
        assert!(vtk.contains("electric_potential"));
        assert!(vtk.contains("temperature"));

        let _ = fs::remove_file(vtk_path);
    }

    #[test]
    fn ex48_hdf5_checkpoint_roundtrip_preserves_restart_state() {
        let args = base_args();
        let partial = solve_joule_template_with_restart(
            &Args {
                max_coupling: 4,
                ..args.clone()
            },
            None,
        );
        let path = temp_output_path("checkpoint_h5", "h5");
        let checkpoint = JouleCheckpointState {
            args: Args {
                max_coupling: 4,
                ..args.clone()
            },
            completed_iterations: partial.completed_iterations,
            converged: partial.converged,
            final_relative_change: partial.final_relative_change,
            sigma_effective: partial.sigma_effective,
            joule_power: partial.joule_power,
            observed_sync_retries: partial.sync_retries,
            rejected_sync_steps: partial.rejected_sync_steps,
            rollback_count: partial.rollback_count,
            joule_power_tracker_prev: partial.joule_power_tracker_prev,
            phi: partial.phi.clone(),
            temperature: partial.temperature.clone(),
        };

        write_joule_hdf5_checkpoint(&path, &checkpoint).unwrap();
        let restored = read_joule_hdf5_checkpoint(&path).unwrap();

        assert_eq!(restored.completed_iterations, checkpoint.completed_iterations);
        assert_eq!(restored.args.n, checkpoint.args.n);
        assert_eq!(restored.args.use_subcycling, checkpoint.args.use_subcycling);
        assert_eq!(restored.phi, checkpoint.phi);
        assert_eq!(restored.temperature, checkpoint.temperature);

        let resumed = solve_joule_template_with_restart(&args, Some(&restored));
        let full = solve_joule_template(&args);
        assert!((resumed.temp_checksum - full.temp_checksum).abs() < 1.0e-12);
        assert!((resumed.joule_power - full.joule_power).abs() < 1.0e-12);
        assert!((resumed.sigma_effective - full.sigma_effective).abs() < 1.0e-12);

        let _ = fs::remove_file(path);
    }

    #[cfg(feature = "io_hdf5")]
    #[test]
    fn ex48_hdf5_checkpoint_writes_xdmf_sidecars() {
        let args = base_args();
        let result = solve_joule_template(&args);
        let h5_path = temp_output_path("checkpoint_sidecar", "h5");
        let phi_sidecar = checkpoint_sidecar_path(&h5_path, "electric_potential").unwrap();
        let temperature_sidecar = checkpoint_sidecar_path(&h5_path, "temperature").unwrap();
        let checkpoint = JouleCheckpointState {
            args,
            completed_iterations: result.completed_iterations,
            converged: result.converged,
            final_relative_change: result.final_relative_change,
            sigma_effective: result.sigma_effective,
            joule_power: result.joule_power,
            observed_sync_retries: result.sync_retries,
            rejected_sync_steps: result.rejected_sync_steps,
            rollback_count: result.rollback_count,
            joule_power_tracker_prev: result.joule_power_tracker_prev,
            phi: result.phi.clone(),
            temperature: result.temperature.clone(),
        };

        write_joule_hdf5_checkpoint(&h5_path, &checkpoint).unwrap();
        write_joule_hdf5_xdmf_sidecars(&h5_path, &checkpoint).unwrap();

        let phi_xdmf = fs::read_to_string(&phi_sidecar).unwrap();
        let temperature_xdmf = fs::read_to_string(&temperature_sidecar).unwrap();
        assert!(phi_xdmf.contains("electric_potential"));
        assert!(temperature_xdmf.contains("temperature"));

        let _ = fs::remove_file(h5_path);
        let _ = fs::remove_file(phi_sidecar);
        let _ = fs::remove_file(temperature_sidecar);
    }
}
