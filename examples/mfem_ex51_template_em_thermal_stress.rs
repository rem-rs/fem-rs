//! Example 51: built-in template driver - Electromagnetic Thermal Stress.
//!
//! This template demonstrates a practical three-field coupling workflow:
//! 1) electromagnetic proxy: electric potential solve
//! 2) thermal field: Joule heating driven diffusion solve
//! 3) structural proxy: thermal-expansion displacement/stress indicators

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::FnCoeff,
    postprocess::compute_element_gradients,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_examples::template_runner::{
    maybe_write_template_kpi_csv,
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

struct EmThermalStressResult {
    steps: usize,
    converged_steps: usize,
    max_coupling_iters_used: usize,
    max_joule_power: f64,
    final_temp_norm: f64,
    final_temp_checksum: f64,
    final_mean_temp: f64,
    final_displacement_proxy: f64,
    final_stress_proxy: f64,
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
    drive_amp: f64,
    sigma0: f64,
    sigma_temp_coeff: f64,
    mech_feedback: f64,
    thermal_kappa: f64,
    alpha_thermal: f64,
    young_modulus: f64,
    relax: f64,
    coupling_tol: f64,
    sync_error_tol: f64,
    w_residual: f64,
    w_joule_source: f64,
    max_coupling: usize,
    sync_retries: usize,
    fast_dt_min: f64,
}

fn main() {
    let args = parse_args();
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::ElectromagneticThermalStress);

    let config_line = format!(
        "n={}, steps={}, dt={}, fast_dt={}, fast_dt_min={}, subcycling={}, drive_amp={}, sigma0={}, sigma_temp_coeff={}, mech_feedback={}, thermal_kappa={}, coupling_tol={}, sync_error_tol={}, w_residual={}, w_joule_source={}, sync_retries={}",
        args.n,
        args.steps,
        args.dt,
        args.fast_dt,
        args.fast_dt_min,
        args.use_subcycling,
        args.drive_amp,
        args.sigma0,
        args.sigma_temp_coeff,
        args.mech_feedback,
        args.thermal_kappa,
        args.coupling_tol,
        args.sync_error_tol,
        args.w_residual,
        args.w_joule_source,
        args.sync_retries,
    );
    print_template_header("Example 51: Built-in template driver", spec, &config_line);

    let result = solve_em_thermal_stress_template(&args);

    let coupling = TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.converged_steps,
        max_coupling_iters_used: result.max_coupling_iters_used,
    };
    print_template_coupling_summary(coupling);
    println!("  max Joule power: {:.6e}", result.max_joule_power);
    println!("  final ||T||_2: {:.6e}", result.final_temp_norm);
    println!("  final temperature checksum: {:.8e}", result.final_temp_checksum);
    println!("  final mean temperature: {:.6e}", result.final_mean_temp);
    println!(
        "  final displacement proxy: {:.6e}",
        result.final_displacement_proxy
    );
    println!("  final stress proxy: {:.6e}", result.final_stress_proxy);
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
            ("max_joule_power", result.max_joule_power),
            ("final_mean_temp", result.final_mean_temp),
            ("final_displacement_proxy", result.final_displacement_proxy),
            ("final_stress_proxy", result.final_stress_proxy),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }
}

fn solve_em_thermal_stress_template(args: &Args) -> EmThermalStressResult {
    if args.use_subcycling {
        solve_em_thermal_stress_template_subcycling(args)
    } else {
        solve_em_thermal_stress_template_single_rate(args)
    }
}

fn solve_em_thermal_stress_template_single_rate(args: &Args) -> EmThermalStressResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let mut temperature = vec![0.0_f64; n_dofs];
    let mut displacement_proxy = 0.0_f64;
    let mut stress_proxy = 0.0_f64;

    let mut converged_steps = 0usize;
    let mut max_coupling_iters_used = 0usize;
    let mut max_joule_power = 0.0_f64;

    for step in 1..=args.steps {
        let t = step as f64 * args.dt;
        let drive = 1.0 + args.drive_amp * (2.0 * PI * t).sin();

        let mut step_converged = false;
        let mut step_iters = 0usize;

        for k in 0..args.max_coupling {
            let mean_temp = temperature.iter().sum::<f64>() / n_dofs as f64;
            let sigma_eff = (args.sigma0
                * (1.0 + args.sigma_temp_coeff * mean_temp)
                * (1.0 + args.mech_feedback * displacement_proxy))
                .max(1.0e-12);

            let phi = solve_potential(&space, sigma_eff, drive);

            let grads = compute_element_gradients(&space, &phi);
            let q_elem: Vec<f64> = grads
                .iter()
                .map(|g| sigma_eff * (g[0] * g[0] + g[1] * g[1]))
                .collect();
            let joule_power = integrate_element_scalar(&space, &q_elem);
            max_joule_power = max_joule_power.max(joule_power);

            let t_new = solve_temperature(&space, &temperature, &q_elem, args.thermal_kappa);
            let t_relaxed: Vec<f64> = temperature
                .iter()
                .zip(t_new.iter())
                .map(|(&old, &newv)| (1.0 - args.relax) * old + args.relax * newv)
                .collect();

            let new_mean_temp = t_relaxed.iter().sum::<f64>() / n_dofs as f64;
            let new_disp = args.alpha_thermal * new_mean_temp;
            let new_stress = args.young_modulus * args.alpha_thermal * new_mean_temp;

            let rel_t = relative_change(&temperature, &t_relaxed);
            let rel_d = (new_disp - displacement_proxy).abs() / new_disp.abs().max(1.0e-12);
            let rel = rel_t.max(rel_d);

            temperature = t_relaxed;
            displacement_proxy = new_disp;
            stress_proxy = new_stress;

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

    let final_mean_temp = temperature.iter().sum::<f64>() / n_dofs as f64;

    EmThermalStressResult {
        steps: args.steps,
        converged_steps,
        max_coupling_iters_used,
        max_joule_power,
        final_temp_norm: l2_norm(&temperature),
        final_temp_checksum: checksum(&temperature),
        final_mean_temp,
        final_displacement_proxy: displacement_proxy,
        final_stress_proxy: stress_proxy,
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
    }
}

fn solve_em_thermal_stress_template_subcycling(args: &Args) -> EmThermalStressResult {
    #[derive(Clone)]
    struct SubcyclingState {
        temperature: Vec<f64>,
        displacement_proxy: f64,
        stress_proxy: f64,
        q_elem: Vec<f64>,
        q_l2_tracker: RelativeL2Tracker,
        max_joule_power: f64,
        converged_steps: usize,
        last_rel: f64,
        sync_error: f64,
    }

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();
    let n_elems = space.mesh().n_elements();

    let mut state = SubcyclingState {
        temperature: vec![0.0_f64; n_dofs],
        displacement_proxy: 0.0,
        stress_proxy: 0.0,
        q_elem: vec![0.0_f64; n_elems],
        q_l2_tracker: RelativeL2Tracker::new(),
        max_joule_power: 0.0,
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

    let sync_policy = TemplateSyncPolicy {
        sync_error_tol: args.sync_error_tol,
        max_sync_retries: args.sync_retries,
        min_fast_dt: args.fast_dt_min.max(1.0e-12),
        retry_fast_dt_scale: 0.5,
        component_weights: vec![args.w_residual, args.w_joule_source],
    };

    let stats = run_multirate_adaptive(
        sync_policy
            .adaptive_config(cfg)
            .expect("invalid EM-thermal-stress sync policy"),
        &mut state,
        |state, t_fast, dt_fast| {
            let t_next = t_fast + dt_fast;
            let drive = 1.0 + args.drive_amp * (2.0 * PI * t_next).sin();
            let mean_temp = state.temperature.iter().sum::<f64>() / n_dofs as f64;
            let sigma_eff = (args.sigma0
                * (1.0 + args.sigma_temp_coeff * mean_temp)
                * (1.0 + args.mech_feedback * state.displacement_proxy))
                .max(1.0e-12);

            let phi = solve_potential(&space, sigma_eff, drive);
            let grads = compute_element_gradients(&space, &phi);
            state.q_elem = grads
                .iter()
                .map(|g| sigma_eff * (g[0] * g[0] + g[1] * g[1]))
                .collect();
            let joule_power = integrate_element_scalar(&space, &state.q_elem);
            state.max_joule_power = state.max_joule_power.max(joule_power);
        },
        |state, _t_slow, _dt_slow| {
            let t_new = solve_temperature(&space, &state.temperature, &state.q_elem, args.thermal_kappa);
            let t_relaxed: Vec<f64> = state
                .temperature
                .iter()
                .zip(t_new.iter())
                .map(|(&old, &newv)| (1.0 - args.relax) * old + args.relax * newv)
                .collect();

            let new_mean_temp = t_relaxed.iter().sum::<f64>() / n_dofs as f64;
            let new_disp = args.alpha_thermal * new_mean_temp;
            let new_stress = args.young_modulus * args.alpha_thermal * new_mean_temp;

            let rel_t = relative_change(&state.temperature, &t_relaxed);
            let rel_d = (new_disp - state.displacement_proxy).abs() / new_disp.abs().max(1.0e-12);
            state.last_rel = rel_t.max(rel_d);

            state.temperature = t_relaxed;
            state.displacement_proxy = new_disp;
            state.stress_proxy = new_stress;
        },
        |state, _t_sync| {
            let rel_q = state.q_l2_tracker.observe_field(&state.q_elem, state.last_rel);
            state.sync_error = sync_policy.compose_error(&[state.last_rel, rel_q]);

            if state.last_rel <= args.coupling_tol {
                state.converged_steps += 1;
            }

            state.sync_error
        },
    )
    .expect("adaptive subcycling scheduler failed");

    let final_mean_temp = state.temperature.iter().sum::<f64>() / n_dofs as f64;
    EmThermalStressResult {
        steps: stats.sync_steps,
        converged_steps: state.converged_steps,
        max_coupling_iters_used: 1,
        max_joule_power: state.max_joule_power,
        final_temp_norm: l2_norm(&state.temperature),
        final_temp_checksum: checksum(&state.temperature),
        final_mean_temp,
        final_displacement_proxy: state.displacement_proxy,
        final_stress_proxy: state.stress_proxy,
        sync_retries: stats.sync_retries,
        rejected_sync_steps: stats.rejected_sync_steps,
        rollback_count: stats.rollback_count,
    }
}

fn solve_potential(space: &H1Space<SimplexMesh<2>>, sigma_eff: f64, drive: f64) -> Vec<f64> {
    let mut a = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator {
            kappa: FnCoeff(move |_x: &[f64]| sigma_eff),
        }],
        3,
    );
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

    let mut phi = vec![0.0_f64; space.n_dofs()];
    let _ = solve_pcg_jacobi(&a, &rhs, &mut phi, &cfg)
        .or_else(|_| solve_gmres(&a, &rhs, &mut phi, 60, &cfg))
        .expect("electric potential solve failed");
    phi
}

fn solve_temperature(
    space: &H1Space<SimplexMesh<2>>,
    initial_guess: &[f64],
    q_elem: &[f64],
    kappa: f64,
) -> Vec<f64> {
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        sample_piecewise_constant_on_mesh(space, q_elem, x)
    });
    let mut rhs = Assembler::assemble_linear(space, &[&source], 3);
    let mut a = Assembler::assemble_bilinear(space, &[&DiffusionIntegrator { kappa }], 3);

    let dm = space.dof_manager();
    let all_bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    apply_dirichlet(&mut a, &mut rhs, &all_bnd, &vec![0.0; all_bnd.len()]);

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

fn integrate_element_scalar(space: &H1Space<SimplexMesh<2>>, elem_values: &[f64]) -> f64 {
    let mesh = space.mesh();
    let mut acc = 0.0_f64;
    for (e, &value) in mesh.elem_iter().zip(elem_values.iter()) {
        acc += value * tri_area(mesh, e);
    }
    acc
}

fn tri_area(mesh: &SimplexMesh<2>, elem: u32) -> f64 {
    let ns = mesh.elem_nodes(elem);
    let a = mesh.coords_of(ns[0]);
    let b = mesh.coords_of(ns[1]);
    let c = mesh.coords_of(ns[2]);
    let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    0.5 * det.abs()
}

fn sample_piecewise_constant_on_mesh(
    space: &H1Space<SimplexMesh<2>>,
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
        drive_amp: 0.3,
        sigma0: 5.0,
        sigma_temp_coeff: 0.02,
        mech_feedback: 0.1,
        thermal_kappa: 1.0,
        alpha_thermal: 2.0e-2,
        young_modulus: 100.0,
        relax: 0.7,
        coupling_tol: 1.0e-7,
        sync_error_tol: 1.0,
        w_residual: 1.0,
        w_joule_source: 1.0,
        max_coupling: 12,
        sync_retries: 2,
        fast_dt_min: 1.0e-3,
    };

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex51_template_em_thermal_stress");
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
                ("--drive-amp <float>", "Electrical drive amplitude (default: 0.3)"),
                ("--sigma0 <float>", "Reference conductivity (default: 5.0)"),
                (
                    "--sigma-temp <float>",
                    "Conductivity temperature coefficient (default: 0.02)",
                ),
                (
                    "--mech-feedback <float>",
                    "Mechanical feedback factor (default: 0.1)",
                ),
                ("--kappa <float>", "Thermal diffusivity (default: 1.0)"),
                (
                    "--alpha-thermal <float>",
                    "Thermal expansion coefficient (default: 0.02)",
                ),
                ("--young <float>", "Young's modulus proxy (default: 100.0)"),
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
                    "Weight for EM-thermal residual sync component (default: 1.0)",
                ),
                (
                    "--w-joule-source <float>",
                    "Weight for Joule-source relative-change sync component (default: 1.0)",
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
            "--drive-amp" => {
                a.drive_amp = it.next().unwrap_or("0.3".into()).parse().unwrap_or(0.3)
            }
            "--sigma0" => a.sigma0 = it.next().unwrap_or("5.0".into()).parse().unwrap_or(5.0),
            "--sigma-temp" => {
                a.sigma_temp_coeff = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02)
            }
            "--mech-feedback" => {
                a.mech_feedback = it.next().unwrap_or("0.1".into()).parse().unwrap_or(0.1)
            }
            "--kappa" => {
                a.thermal_kappa = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
            }
            "--alpha-thermal" => {
                a.alpha_thermal = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02)
            }
            "--young" => {
                a.young_modulus = it.next().unwrap_or("100.0".into()).parse().unwrap_or(100.0)
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
            "--w-joule-source" => {
                a.w_joule_source = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0)
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
    a.w_joule_source = a.w_joule_source.max(0.0);
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
            drive_amp: 0.2,
            sigma0: 5.0,
            sigma_temp_coeff: 0.02,
            mech_feedback: 0.1,
            thermal_kappa: 1.0,
            alpha_thermal: 0.02,
            young_modulus: 100.0,
            relax: 0.7,
            coupling_tol: 1.0e-7,
            sync_error_tol: 1.0,
            w_residual: 1.0,
            w_joule_source: 1.0,
            max_coupling: 10,
            sync_retries: 2,
            fast_dt_min: 1.0e-3,
        }
    }

    #[test]
    fn ex51_em_thermal_stress_template_runs_and_heats() {
        let r = solve_em_thermal_stress_template(&base_args());
        assert_eq!(r.steps, 4);
        assert!(r.max_coupling_iters_used <= 10);
        assert!(r.max_joule_power > 0.0);
        assert!(r.final_temp_norm > 0.0);
        assert!(r.final_displacement_proxy > 0.0);
        assert!(r.final_stress_proxy > 0.0);
    }

    #[test]
    fn ex51_higher_drive_increases_heating_and_stress() {
        let mut low = base_args();
        low.drive_amp = 0.1;
        let mut high = base_args();
        high.drive_amp = 0.6;

        let r_low = solve_em_thermal_stress_template(&low);
        let r_high = solve_em_thermal_stress_template(&high);

        assert!(r_high.max_joule_power > r_low.max_joule_power);
        assert!(r_high.final_temp_norm > r_low.final_temp_norm);
        assert!(r_high.final_stress_proxy > r_low.final_stress_proxy);
    }

    /// Zero EM drive → negligible Joule heating and negligible stress.
    #[test]
    fn ex51_near_zero_conductivity_gives_negligible_joule_power() {
        // With sigma0 near zero, P ≈ σ |∇φ|² ≈ 0 regardless of drive.
        let mut args = base_args();
        args.sigma0 = 1.0e-8;
        let r = solve_em_thermal_stress_template(&args);
        assert!(r.max_joule_power < 1.0e-6,
            "expected near-zero Joule power with near-zero sigma: {:.4e}", r.max_joule_power);
        assert!(r.final_stress_proxy < 1.0e-6,
            "expected near-zero stress with near-zero sigma: {:.4e}", r.final_stress_proxy);
    }

    /// Negative temperature coefficient means higher temperature → lower conductivity
    /// → less Joule heating (stabilising feedback). Result should have lower
    /// max Joule power than the zero-feedback case.
    #[test]
    fn ex51_negative_sigma_temp_coeff_stabilises_joule_heating() {
        let mut no_feedback = base_args();
        no_feedback.sigma_temp_coeff = 0.0;
        let mut with_feedback = base_args();
        with_feedback.sigma_temp_coeff = -0.5; // σ decreases with temperature

        let r_none = solve_em_thermal_stress_template(&no_feedback);
        let r_fb   = solve_em_thermal_stress_template(&with_feedback);

        // Negative temperature coefficient damps Joule heating.
        assert!(r_fb.max_joule_power <= r_none.max_joule_power * 1.01,
            "expected stabilising feedback to not increase heating: no_fb={:.4e} fb={:.4e}",
            r_none.max_joule_power, r_fb.max_joule_power);
    }

    /// Higher conductivity (sigma0) should yield stronger Joule heating
    /// (P ∝ σ |∇φ|²) for the same drive amplitude.
    #[test]
    fn ex51_higher_conductivity_gives_more_joule_heating() {
        let mut low = base_args();
        low.sigma0 = 0.1;
        let mut high = base_args();
        high.sigma0 = 1.0;

        let r_low  = solve_em_thermal_stress_template(&low);
        let r_high = solve_em_thermal_stress_template(&high);

        assert!(r_high.max_joule_power > r_low.max_joule_power,
            "expected more heating at higher sigma: low={:.4e} high={:.4e}",
            r_low.max_joule_power, r_high.max_joule_power);
    }

    /// More time steps → more accumulated Joule heating → higher temperature.
    #[test]
    fn ex51_more_steps_accumulates_more_heat() {
        let mut few = base_args();
        few.steps = 2;
        let mut many = base_args();
        many.steps = 8;

        let r_few  = solve_em_thermal_stress_template(&few);
        let r_many = solve_em_thermal_stress_template(&many);

        assert!(r_many.final_temp_norm >= r_few.final_temp_norm,
            "expected more steps to accumulate more heat: few={:.4e} many={:.4e}",
            r_few.final_temp_norm, r_many.final_temp_norm);
    }

    /// Higher thermal diffusivity disperses heat faster → lower temperature norm.
    #[test]
    fn ex51_higher_thermal_kappa_reduces_temperature_norm() {
        let mut low_k = base_args();
        low_k.thermal_kappa = 0.1;
        let mut high_k = base_args();
        high_k.thermal_kappa = 10.0;

        let r_low  = solve_em_thermal_stress_template(&low_k);
        let r_high = solve_em_thermal_stress_template(&high_k);

        assert!(r_high.final_temp_norm < r_low.final_temp_norm,
            "expected higher kappa to reduce temperature norm: low_k={:.4e} high_k={:.4e}",
            r_low.final_temp_norm, r_high.final_temp_norm);
    }

    /// Identical args must produce an identical temperature checksum (determinism).
    #[test]
    fn ex51_deterministic_checksum_across_repeated_runs() {
        let r1 = solve_em_thermal_stress_template(&base_args());
        let r2 = solve_em_thermal_stress_template(&base_args());
        assert_eq!(r1.final_temp_checksum, r2.final_temp_checksum,
            "expected deterministic checksum: run1={:.8e} run2={:.8e}",
            r1.final_temp_checksum, r2.final_temp_checksum);
    }
}
