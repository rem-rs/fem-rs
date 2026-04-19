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

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::DiffusionIntegrator,
    transfer_h1_p1_nonmatching_l2_projection_conservative,
};
use fem_examples::template_runner::{
    TemplateAdaptiveSummary,
    TemplateCouplingSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
    print_template_coupling_summary,
    print_template_header,
};
use fem_linalg::Vector;
use fem_mesh::{
    MeshMotionConfig,
    SimplexMesh,
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

struct FsiTemplateResult {
    converged_steps: usize,
    steps: usize,
    max_coupling_iters_used: usize,
    max_transfer_abs_int_err: f64,
    max_wall_displacement: f64,
    final_wall_displacement: f64,
    final_pressure_norm: f64,
    final_pressure_checksum: f64,
    sync_retries: usize,
    rejected_sync_steps: usize,
    rollback_count: usize,
}

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

fn main() {
    let args = parse_args();
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

    let result = solve_fsi_template(&args);

    print_template_coupling_summary(TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.converged_steps,
        max_coupling_iters_used: result.max_coupling_iters_used,
    });
    println!(
        "  max transfer integral error: {:.3e}",
        result.max_transfer_abs_int_err
    );
    println!("  max |wall displacement|: {:.6e}", result.max_wall_displacement);
    println!("  final wall displacement: {:.6e}", result.final_wall_displacement);
    println!("  final ||p||_2: {:.6e}", result.final_pressure_norm);
    println!("  final pressure checksum: {:.8e}", result.final_pressure_checksum);
    print_template_adaptive_summary(TemplateAdaptiveSummary {
        sync_retries: result.sync_retries,
        rejected_sync_steps: result.rejected_sync_steps,
        rollback_count: result.rollback_count,
    });
}

fn solve_fsi_template(args: &Args) -> FsiTemplateResult {
    if args.use_subcycling {
        solve_fsi_template_subcycling(args)
    } else {
        solve_fsi_template_single_rate(args)
    }
}

fn solve_fsi_template_single_rate(args: &Args) -> FsiTemplateResult {
    let ref_mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let mut mesh = ref_mesh.clone();
    let mut pressure = H1Space::new(mesh.clone(), 1).interpolate(&|_x| 0.0);

    let mut wall_disp = 0.0_f64;
    let mut max_wall_disp = 0.0_f64;
    let mut max_transfer_abs_int_err = 0.0_f64;
    let mut converged_steps = 0usize;
    let mut max_coupling_iters_used = 0usize;

    for step in 1..=args.steps {
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
    }
}

fn solve_fsi_template_subcycling(args: &Args) -> FsiTemplateResult {
    #[derive(Clone)]
    struct SubcyclingState {
        mesh: SimplexMesh<2>,
        pressure: Vector<f64>,
        wall_disp: f64,
        max_wall_disp: f64,
        max_transfer_abs_int_err: f64,
        converged_steps: usize,
    }

    let ref_mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let init_mesh = ref_mesh.clone();
    let init_pressure = H1Space::new(init_mesh.clone(), 1).interpolate(&|_x| 0.0);
    let mut state = SubcyclingState {
        mesh: init_mesh,
        pressure: init_pressure,
        wall_disp: 0.0,
        max_wall_disp: 0.0,
        max_transfer_abs_int_err: 0.0,
        converged_steps: 0,
    };

    let fast_dt = args.fast_dt.max(1.0e-12).min(args.dt);
    let cfg = MultiRateConfig {
        t_start: 0.0,
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
        converged_steps: state.converged_steps,
        steps: stats.sync_steps,
        max_coupling_iters_used: 1,
        max_transfer_abs_int_err: state.max_transfer_abs_int_err,
        max_wall_displacement: state.max_wall_disp,
        final_wall_displacement: state.wall_disp,
        final_pressure_norm: l2_norm(state.pressure.as_slice()),
        final_pressure_checksum: checksum(state.pressure.as_slice()),
        sync_retries: stats.sync_retries,
        rejected_sync_steps: stats.rejected_sync_steps,
        rollback_count: stats.rollback_count,
    }
}

fn solve_pressure_on_mesh(
    space: &H1Space<SimplexMesh<2>>,
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

fn top_boundary_average_pressure(space: &H1Space<SimplexMesh<2>>, p: &[f64]) -> f64 {
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

fn parse_args() -> Args {
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

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex49_template_fsi");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_template_cli_help(
            bin,
            &[
                ("--n <int>", "Mesh resolution (default: 14)"),
                ("--steps <int>", "Number of slow synchronization steps (default: 10)"),
                ("--dt <float>", "Slow-step size (default: 0.05)"),
                (
                    "--fast-dt <float>",
                    "Fast subcycling step size (default: 0.01)",
                ),
                ("--subcycling", "Enable multirate subcycling (default)"),
                ("--no-subcycling", "Disable subcycling and use single-rate loop"),
                (
                    "--inlet-amp <float>",
                    "Inlet forcing amplitude (default: 0.3)",
                ),
                (
                    "--compliance <float>",
                    "Structure compliance coefficient (default: 0.02)",
                ),
                (
                    "--wall-relax <float>",
                    "Wall displacement relaxation in [0.1, 1.0] (default: 0.7)",
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
            ],
        );
        std::process::exit(0);
    }

    let mut it = args_vec.into_iter().skip(1);
    while let Some(arg) = it.next() {
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
            "--coupling-tol" => {
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
    a
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
