//! Example 52: built-in template driver - Reaction Flow Thermal.
//!
//! This template demonstrates a practical chemistry-flow-thermal coupling loop:
//! - flow proxy: pressure solve and flow metric from pressure drop / viscosity
//! - species field: diffusion + reaction-consumption source
//! - thermal field: diffusion + reaction heat-release source

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::FnCoeff,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_examples::template_runner::{
    TemplateAdaptiveSummary,
    TemplateCouplingSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
    print_template_coupling_summary,
    print_template_header,
};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
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
}

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

fn main() {
    let args = parse_args();
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

    let result = solve_reaction_flow_thermal_template(&args);

    print_template_coupling_summary(TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.converged_steps,
        max_coupling_iters_used: result.max_coupling_iters_used,
    });
    println!("  max reaction rate: {:.6e}", result.max_reaction_rate);
    println!("  final flow metric: {:.6e}", result.final_flow_metric);
    println!("  final ||species||_2: {:.6e}", result.final_species_norm);
    println!("  final ||temperature||_2: {:.6e}", result.final_temperature_norm);
    println!("  final species checksum: {:.8e}", result.final_species_checksum);
    println!("  final temperature checksum: {:.8e}", result.final_temperature_checksum);
    print_template_adaptive_summary(TemplateAdaptiveSummary {
        sync_retries: result.sync_retries,
        rejected_sync_steps: result.rejected_sync_steps,
        rollback_count: result.rollback_count,
    });
}

fn solve_reaction_flow_thermal_template(args: &Args) -> ReactionFlowThermalResult {
    if args.use_subcycling {
        solve_reaction_flow_thermal_template_subcycling(args)
    } else {
        solve_reaction_flow_thermal_template_single_rate(args)
    }
}

fn solve_reaction_flow_thermal_template_single_rate(args: &Args) -> ReactionFlowThermalResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    let mut c = vec![0.0_f64; n];
    let mut t = vec![0.0_f64; n];

    let mut converged_steps = 0usize;
    let mut max_coupling_iters_used = 0usize;
    let mut max_reaction_rate = 0.0_f64;
    let mut final_flow_metric = 0.0_f64;

    for step in 1..=args.steps {
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
    }
}

fn solve_reaction_flow_thermal_template_subcycling(args: &Args) -> ReactionFlowThermalResult {
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

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();
    let mut state = SubcyclingState {
        c: vec![0.0_f64; n],
        t: vec![0.0_f64; n],
        rate_nodal: vec![0.0_f64; n],
        current_rate_peak: 0.0,
        rate_peak_tracker: RelativeScalarTracker::new(),
        max_reaction_rate: 0.0,
        final_flow_metric: 0.0,
        flow_metric_tracker: RelativeScalarTracker::new(),
        converged_steps: 0,
        last_rel: f64::INFINITY,
        sync_error: f64::INFINITY,
    };

    let fast_dt = args.fast_dt.max(1.0e-12).min(args.dt);
    let cfg = MultiRateConfig {
        t_start: 0.0,
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
        steps: stats.sync_steps,
        converged_steps: state.converged_steps,
        max_coupling_iters_used: 1,
        max_reaction_rate: state.max_reaction_rate,
        final_flow_metric: state.final_flow_metric,
        final_species_norm: l2_norm(&state.c),
        final_temperature_norm: l2_norm(&state.t),
        final_species_checksum: checksum(&state.c),
        final_temperature_checksum: checksum(&state.t),
        sync_retries: stats.sync_retries,
        rejected_sync_steps: stats.rejected_sync_steps,
        rollback_count: stats.rollback_count,
    }
}

fn solve_pressure_proxy(space: &H1Space<SimplexMesh<2>>, drive: f64) -> Vec<f64> {
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
    space: &H1Space<SimplexMesh<2>>,
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
    space: &H1Space<SimplexMesh<2>>,
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
    let all = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
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

fn sample_nodal_field(space: &H1Space<SimplexMesh<2>>, field: &[f64], x: &[f64]) -> f64 {
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

fn pressure_drop_metric(space: &H1Space<SimplexMesh<2>>, p: &[f64]) -> f64 {
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

fn parse_args() -> Args {
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

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex52_template_reaction_flow_thermal");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_template_cli_help(
            bin,
            &[
                ("--n <int>", "Mesh resolution (default: 12)"),
                ("--steps <int>", "Number of slow synchronization steps (default: 8)"),
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
                    "Flow drive amplitude (default: 0.25)",
                ),
                ("--k-species <float>", "Species diffusivity (default: 0.2)"),
                ("--k-thermal <float>", "Thermal diffusivity (default: 1.0)"),
                ("--reaction-k0 <float>", "Reaction pre-factor (default: 0.8)"),
                (
                    "--reaction-temp <float>",
                    "Reaction temperature sensitivity (default: 0.05)",
                ),
                ("--heat-release <float>", "Reaction heat release (default: 5.0)"),
                ("--viscosity0 <float>", "Reference viscosity (default: 1.0)"),
                (
                    "--visc-temp <float>",
                    "Viscosity temperature coefficient (default: 0.02)",
                ),
                (
                    "--visc-species <float>",
                    "Viscosity species coefficient (default: 0.01)",
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
                    "Weight for species/temperature residual component (default: 1.0)",
                ),
                (
                    "--w-flow-metric <float>",
                    "Weight for flow-metric variation component (default: 1.0)",
                ),
                (
                    "--w-rate-peak <float>",
                    "Weight for reaction-rate-peak variation component (default: 1.0)",
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
            ],
        );
        std::process::exit(0);
    }

    let mut it = args_vec.into_iter().skip(1);
    while let Some(arg) = it.next() {
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
}
