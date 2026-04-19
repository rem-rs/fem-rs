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

use fem_assembly::{
    Assembler,
    coefficient::FnCoeff,
    postprocess::compute_element_gradients,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_examples::template_runner::{
    TemplateAdaptiveSummary,
    print_template_adaptive_summary,
    print_template_cli_help,
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
}

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

fn main() {
    let args = parse_args();
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

    let result = solve_joule_template(&args);

    println!("  converged: {}", result.converged);
    println!("  coupling iterations: {}", result.iterations);
    println!("  final relative thermal change: {:.3e}", result.final_relative_change);
    println!("  effective sigma(T): {:.6e}", result.sigma_effective);
    println!("  ||phi||_2: {:.6e}", result.phi_norm);
    println!("  ||T||_2: {:.6e}", result.temp_norm);
    println!("  integrated Joule power: {:.6e}", result.joule_power);
    println!("  temperature checksum: {:.8e}", result.temp_checksum);
    print_template_adaptive_summary(TemplateAdaptiveSummary {
        sync_retries: result.sync_retries,
        rejected_sync_steps: result.rejected_sync_steps,
        rollback_count: result.rollback_count,
    });
}

fn solve_joule_template(args: &Args) -> JouleTemplateResult {
    if args.use_subcycling {
        solve_joule_template_subcycling(args)
    } else {
        solve_joule_template_single_rate(args)
    }
}

fn solve_joule_template_single_rate(args: &Args) -> JouleTemplateResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
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

    let mut temp = vec![0.0_f64; n_dofs];
    let mut phi = vec![0.0_f64; n_dofs];
    let mut sigma_eff = args.sigma0.max(1.0e-12);
    let mut final_rel = f64::INFINITY;
    let mut joule_power = 0.0_f64;
    let mut iters_done = 0usize;
    let mut converged = false;

    for k in 0..args.max_coupling {
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
    }
}

fn solve_joule_template_subcycling(args: &Args) -> JouleTemplateResult {
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

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
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

    let mut state = SubcyclingState {
        temp: vec![0.0_f64; n_dofs],
        phi: vec![0.0_f64; n_dofs],
        sigma_eff: args.sigma0.max(1.0e-12),
        final_rel: f64::INFINITY,
        joule_power: 0.0,
        joule_power_tracker: RelativeScalarTracker::new(),
        sync_error: f64::INFINITY,
        iters_done: 0,
        converged: false,
    };

    let fast_dt = args.fast_dt.max(1.0e-12).min(1.0);
    let sched_cfg = MultiRateConfig {
        t_start: 0.0,
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
        sync_retries: stats.sync_retries,
        rejected_sync_steps: stats.rejected_sync_steps,
        rollback_count: stats.rollback_count,
    }
}

fn integrate_element_scalar(space: &H1Space<SimplexMesh<2>>, elem_values: &[f64]) -> f64 {
    let mesh = space.mesh();
    let mut acc = 0.0_f64;
    for (e, &value) in mesh.elem_iter().zip(elem_values.iter()) {
        let area = tri_area(mesh, e);
        acc += value * area;
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

fn parse_args() -> Args {
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

    let args_vec: Vec<String> = std::env::args().collect();
    let bin = args_vec
        .first()
        .map(std::string::String::as_str)
        .unwrap_or("mfem_ex48_template_joule_heating");
    if args_vec.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_template_cli_help(
            bin,
            &[
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
            ],
        );
        std::process::exit(0);
    }

    let mut it = args_vec.into_iter().skip(1);
    while let Some(arg) = it.next() {
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
    a
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
