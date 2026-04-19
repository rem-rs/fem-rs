//! Example 50: built-in template driver - Acoustics-Structure.
//!
//! Provides a practical built-in template workflow for vibro-acoustic coupling:
//! - acoustic field proxy: pressure solve on H1
//! - structure proxy: interface displacement from acoustic normal load
//! - partitioned coupling iterations per time step

use std::f64::consts::PI;

use fem_assembly::{Assembler, standard::DiffusionIntegrator};
use fem_examples::template_runner::{
    TemplateAdaptiveSummary,
    TemplateCouplingSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
    print_template_coupling_summary,
    print_template_header,
};
use fem_mesh::SimplexMesh;
use fem_solver::{
    BuiltinMultiphysicsTemplate,
    MultiRateConfig,
    RelativeL2Tracker,
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

struct AcousticsStructureResult {
    steps: usize,
    converged_steps: usize,
    max_coupling_iters_used: usize,
    max_abs_interface_displacement: f64,
    final_interface_displacement: f64,
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
    fast_dt_min: f64,
    use_subcycling: bool,
    drive_amp: f64,
    structure_compliance: f64,
    relax: f64,
    coupling_tol: f64,
    sync_error_tol: f64,
    sync_retries: usize,
    max_coupling: usize,
}

fn main() {
    let args = parse_args();
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::AcousticsStructure);

    let config_line = format!(
        "n={}, steps={}, dt={}, fast_dt={}, fast_dt_min={}, subcycling={}, drive_amp={}, structure_compliance={}, relax={}, coupling_tol={}, sync_error_tol={}, sync_retries={}, max_coupling={}",
        args.n,
        args.steps,
        args.dt,
        args.fast_dt,
        args.fast_dt_min,
        args.use_subcycling,
        args.drive_amp,
        args.structure_compliance,
        args.relax,
        args.coupling_tol,
        args.sync_error_tol,
        args.sync_retries,
        args.max_coupling,
    );
    print_template_header("Example 50: Built-in template driver", spec, &config_line);

    let result = solve_acoustics_structure_template(&args);

    print_template_coupling_summary(TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.converged_steps,
        max_coupling_iters_used: result.max_coupling_iters_used,
    });
    println!(
        "  max |interface displacement|: {:.6e}",
        result.max_abs_interface_displacement
    );
    println!(
        "  final interface displacement: {:.6e}",
        result.final_interface_displacement
    );
    println!("  final ||p||_2: {:.6e}", result.final_pressure_norm);
    println!("  final pressure checksum: {:.8e}", result.final_pressure_checksum);
    print_template_adaptive_summary(TemplateAdaptiveSummary {
        sync_retries: result.sync_retries,
        rejected_sync_steps: result.rejected_sync_steps,
        rollback_count: result.rollback_count,
    });
}

fn solve_acoustics_structure_template(args: &Args) -> AcousticsStructureResult {
    if args.use_subcycling {
        solve_acoustics_structure_template_subcycling(args)
    } else {
        solve_acoustics_structure_template_single_rate(args)
    }
}

fn solve_acoustics_structure_template_single_rate(args: &Args) -> AcousticsStructureResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let mut p = vec![0.0_f64; n_dofs];
    let mut disp = 0.0_f64;
    let mut converged_steps = 0usize;
    let mut max_coupling_iters_used = 0usize;
    let mut max_abs_interface_disp = 0.0_f64;

    for step in 1..=args.steps {
        let t = step as f64 * args.dt;
        let drive = args.drive_amp * (2.0 * PI * t).sin();

        let mut step_converged = false;
        let mut step_iters = 0usize;

        for k in 0..args.max_coupling {
            p = solve_acoustic_pressure(&space, &p, drive, disp);

            let interface_load = top_boundary_average_pressure(&space, &p);
            let target_disp = args.structure_compliance * interface_load;
            let new_disp = (1.0 - args.relax) * disp + args.relax * target_disp;
            let rel = (new_disp - disp).abs() / new_disp.abs().max(1.0e-12);
            disp = new_disp;
            max_abs_interface_disp = max_abs_interface_disp.max(disp.abs());

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

    AcousticsStructureResult {
        steps: args.steps,
        converged_steps,
        max_coupling_iters_used,
        max_abs_interface_displacement: max_abs_interface_disp,
        final_interface_displacement: disp,
        final_pressure_norm: l2_norm(&p),
        final_pressure_checksum: checksum(&p),
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
    }
}

fn solve_acoustics_structure_template_subcycling(args: &Args) -> AcousticsStructureResult {
    #[derive(Clone)]
    struct SubcyclingState {
        p: Vec<f64>,
        p_l2_tracker: RelativeL2Tracker,
        disp: f64,
        max_abs_interface_disp: f64,
        converged_steps: usize,
        last_rel: f64,
    }

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();
    let mut state = SubcyclingState {
        p: vec![0.0_f64; n_dofs],
        p_l2_tracker: RelativeL2Tracker::new(),
        disp: 0.0,
        max_abs_interface_disp: 0.0,
        converged_steps: 0,
        last_rel: f64::INFINITY,
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
            .expect("invalid acoustics-structure sync policy"),
        &mut state,
        |state, t_fast, dt_fast| {
            let t_next = t_fast + dt_fast;
            let drive = args.drive_amp * (2.0 * PI * t_next).sin();
            state.p = solve_acoustic_pressure(&space, &state.p, drive, state.disp);
        },
        |_state, _t_slow, _dt_slow| {
            // Structure update occurs at sync points.
        },
        |state, _t_sync| {
            let interface_load = top_boundary_average_pressure(&space, &state.p);
            let target_disp = args.structure_compliance * interface_load;
            let new_disp = (1.0 - args.relax) * state.disp + args.relax * target_disp;
            state.last_rel = (new_disp - state.disp).abs() / new_disp.abs().max(1.0e-12);
            state.disp = new_disp;
            state.max_abs_interface_disp = state.max_abs_interface_disp.max(state.disp.abs());
            let rel_pressure = state.p_l2_tracker.observe_field(&state.p, state.last_rel);
            if state.last_rel <= args.coupling_tol {
                state.converged_steps += 1;
            }

            sync_policy.compose_error(&[state.last_rel, rel_pressure])
        },
    )
    .expect("adaptive subcycling scheduler failed");

    AcousticsStructureResult {
        steps: stats.sync_steps,
        converged_steps: state.converged_steps,
        max_coupling_iters_used: 1,
        max_abs_interface_displacement: state.max_abs_interface_disp,
        final_interface_displacement: state.disp,
        final_pressure_norm: l2_norm(&state.p),
        final_pressure_checksum: checksum(&state.p),
        sync_retries: stats.sync_retries,
        rejected_sync_steps: stats.rejected_sync_steps,
        rollback_count: stats.rollback_count,
    }
}

fn solve_acoustic_pressure(
    space: &H1Space<SimplexMesh<2>>,
    initial_guess: &[f64],
    drive: f64,
    interface_disp: f64,
) -> Vec<f64> {
    // Proxy acoustic pressure solve:
    //   -Delta p = 0
    //   p = drive on left wall
    //   p = 0 on right wall
    //   p = alpha * interface_disp on top wall (structure feedback)
    //   natural on bottom wall
    let mut a = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        3,
    );
    let mut rhs = vec![0.0_f64; space.n_dofs()];

    let dm = space.dof_manager();
    let left = boundary_dofs(space.mesh(), dm, &[4]);
    let right = boundary_dofs(space.mesh(), dm, &[2]);
    let top = boundary_dofs(space.mesh(), dm, &[1]);

    apply_dirichlet(&mut a, &mut rhs, &left, &vec![drive; left.len()]);
    apply_dirichlet(&mut a, &mut rhs, &right, &vec![0.0; right.len()]);

    let alpha = 0.5_f64;
    apply_dirichlet(
        &mut a,
        &mut rhs,
        &top,
        &vec![alpha * interface_disp; top.len()],
    );

    let mut p = if initial_guess.len() == space.n_dofs() {
        initial_guess.to_vec()
    } else {
        vec![0.0_f64; space.n_dofs()]
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
        .expect("acoustic pressure solve failed");

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
        steps: 12,
        dt: 0.05,
        fast_dt: 0.01,
        fast_dt_min: 1.0e-3,
        use_subcycling: true,
        drive_amp: 1.0,
        structure_compliance: 0.03,
        relax: 0.7,
        coupling_tol: 1.0e-7,
        sync_error_tol: 1.0,
        sync_retries: 2,
        max_coupling: 12,
    };

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex50_template_acoustics_structure");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_template_cli_help(
            bin,
            &[
                ("--n <int>", "Mesh resolution (default: 14)"),
                ("--steps <int>", "Number of slow synchronization steps (default: 12)"),
                ("--dt <float>", "Slow-step size (default: 0.05)"),
                (
                    "--fast-dt <float>",
                    "Fast subcycling step size (default: 0.01)",
                ),
                (
                    "--fast-dt-min <float>",
                    "Minimum fast subcycling step during retries (default: 1e-3)",
                ),
                ("--subcycling", "Enable multirate subcycling (default)"),
                ("--no-subcycling", "Disable subcycling and use single-rate loop"),
                ("--drive-amp <float>", "Acoustic drive amplitude (default: 1.0)"),
                (
                    "--structure-compliance <float>",
                    "Interface compliance coefficient (default: 0.03)",
                ),
                ("--relax <float>", "Displacement relaxation in [0.1, 1.0] (default: 0.7)"),
                (
                    "--coupling-tol <float>",
                    "Coupling convergence tolerance (default: 1e-7)",
                ),
                (
                    "--sync-error-tol <float>",
                    "Adaptive sync acceptance tolerance (default: 1.0)",
                ),
                (
                    "--sync-retries <int>",
                    "Max adaptive retry count at each sync point (default: 2)",
                ),
                (
                    "--max-coupling <int>",
                    "Maximum coupling iterations per slow step (default: 12)",
                ),
            ],
        );
        std::process::exit(0);
    }

    let mut it = args_vec.into_iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("14".into()).parse().unwrap_or(14),
            "--steps" => a.steps = it.next().unwrap_or("12".into()).parse().unwrap_or(12),
            "--dt" => a.dt = it.next().unwrap_or("0.05".into()).parse().unwrap_or(0.05),
            "--fast-dt" => a.fast_dt = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--fast-dt-min" => {
                a.fast_dt_min = it.next().unwrap_or("1e-3".into()).parse().unwrap_or(1.0e-3)
            }
            "--subcycling" => a.use_subcycling = true,
            "--no-subcycling" => a.use_subcycling = false,
            "--drive-amp" => {
                a.drive_amp = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--structure-compliance" => {
                a.structure_compliance =
                    it.next().unwrap_or("0.03".into()).parse().unwrap_or(0.03)
            }
            "--relax" => a.relax = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--coupling-tol" => {
                a.coupling_tol = it.next().unwrap_or("1e-7".into()).parse().unwrap_or(1.0e-7)
            }
            "--sync-error-tol" => {
                a.sync_error_tol = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--sync-retries" => {
                a.sync_retries = it.next().unwrap_or("2".into()).parse().unwrap_or(2)
            }
            "--max-coupling" => {
                a.max_coupling = it.next().unwrap_or("12".into()).parse().unwrap_or(12)
            }
            _ => {}
        }
    }

    a.steps = a.steps.max(1);
    a.fast_dt = a.fast_dt.max(1.0e-12);
    a.fast_dt_min = a.fast_dt_min.max(1.0e-12).min(a.fast_dt);
    a.sync_error_tol = a.sync_error_tol.max(0.0);
    a.max_coupling = a.max_coupling.max(1);
    a.relax = a.relax.clamp(0.1, 1.0);
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            n: 8,
            steps: 5,
            dt: 0.05,
            fast_dt: 0.01,
            fast_dt_min: 1.0e-3,
            use_subcycling: true,
            drive_amp: 1.0,
            structure_compliance: 0.03,
            relax: 0.7,
            coupling_tol: 1.0e-7,
            sync_error_tol: 1.0,
            sync_retries: 2,
            max_coupling: 10,
        }
    }

    #[test]
    fn ex50_acoustics_structure_template_runs_and_couples() {
        let r = solve_acoustics_structure_template(&base_args());
        assert_eq!(r.steps, 5);
        assert!(r.max_coupling_iters_used <= 10);
        assert!(r.max_abs_interface_displacement > 0.0);
        assert!(r.final_pressure_norm > 0.0);
    }

    #[test]
    fn ex50_higher_compliance_increases_interface_displacement() {
        let mut low = base_args();
        low.structure_compliance = 0.01;
        let mut high = base_args();
        high.structure_compliance = 0.06;

        let r_low = solve_acoustics_structure_template(&low);
        let r_high = solve_acoustics_structure_template(&high);

        assert!(
            r_high.max_abs_interface_displacement > r_low.max_abs_interface_displacement,
            "higher compliance should increase interface motion"
        );
    }

    #[test]
    #[should_panic(expected = "adaptive subcycling scheduler failed")]
    fn ex50_strict_sync_error_tol_can_trigger_adaptive_failure() {
        let mut a = base_args();
        a.sync_error_tol = 0.0;
        a.sync_retries = 0;
        a.steps = 2;
        let _ = solve_acoustics_structure_template(&a);
    }
}
