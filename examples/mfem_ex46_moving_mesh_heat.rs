//! Example 46: moving-mesh transient heat solve (quasi-ALE workflow).
//!
//! Per time step:
//! 1) update mesh geometry (prescribed top-wall motion + interior smoothing)
//! 2) conservatively transfer temperature old-mesh -> new-mesh
//! 3) reassemble on the new mesh and advance one implicit-Euler heat step
//!
//! Optional workflow hooks:
//! - `--checkpoint <path>` / `--restart <path>` use a lightweight text
//!   checkpoint format for split transient restart.
//! - `--checkpoint-h5 <path>` / `--restart-h5 <path>` use the shared
//!   `fem-io-hdf5-parallel` checkpoint format; when built with `--features
//!   io_hdf5`, a `temp` XDMF sidecar is also emitted.
//! - `--export-vtk-prefix <prefix>` writes the final temperature field on the
//!   deformed mesh as `<prefix>_temperature.vtu`.

use std::fs;
use std::f64::consts::PI;
use std::io;

use fem_assembly::{
    Assembler,
    coefficient::FnVectorCoeff,
    standard::{DiffusionIntegrator, MassIntegrator},
    standard::ConvectionIntegrator,
    transfer_h1_p1_nonmatching_l2_projection_conservative,
};
use fem_examples::checkpoint_text::{ensure_parent_dir, format_vec_f64, parse_vec_f64};
use fem_examples::template_runner::{
    TemplateAdaptiveSummary,
    TemplateCouplingSummary,
    maybe_write_template_kpi_csv,
    print_template_adaptive_summary,
    print_template_coupling_summary,
    print_template_header,
};
use fem_examples::hdf5_checkpoint::{scalar_rank_field_f64, vector_rank_field_f64};
use fem_examples::workflow_cli::{assert_single_restart_source, WorkflowCliOptions};
#[cfg(feature = "io_hdf5")]
use fem_examples::hdf5_checkpoint::{checkpoint_sidecar_path, write_scalar_checkpoint_xdmf_sidecars};
use fem_io_hdf5_parallel::{
    CheckpointBundleF64,
    IoBackend,
    ParallelIoConfig,
    read_checkpoint_fields_f64_latest,
    validate_checkpoint_layout,
    write_checkpoint_step_bundle_f64,
};
use fem_io::vtk::{DataArray, VtkWriter};
use fem_linalg::{CooMatrix, CsrMatrix, Vector};
use fem_mesh::{
    MeshMotionConfig,
    Mesh,
    all_boundary_nodes,
    apply_node_displacement,
    laplacian_smooth_2d,
};
use fem_solver::{
    BuiltinMultiphysicsTemplate,
    SolverConfig,
    builtin_template_spec,
    solve_pcg_jacobi,
};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

struct SolveResult {
    final_time: f64,
    completed_steps: usize,
    n_dofs: usize,
    final_l2: f64,
    final_checksum: f64,
    max_transfer_abs_int_err: f64,
    l2_history: Vec<f64>,
    prev_shift: f64,
    temperature: Vec<f64>,
    final_mesh: Mesh<2>,
}

struct TransientCheckpointState {
    completed_steps: usize,
    current_time: f64,
    dt: f64,
    prev_shift: f64,
    temperature: Vec<f64>,
}

fn main() {
    let args = parse_args();
    assert_single_restart_source(&WorkflowCliOptions {
        checkpoint: args.checkpoint.clone(),
        checkpoint_h5: args.checkpoint_h5.clone(),
        restart: args.restart.clone(),
        restart_h5: args.restart_h5.clone(),
        export_vtk_prefix: args.export_vtk_prefix.clone(),
    });
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::MovingMeshHeat);
    let config_line = format!(
        "n={}, dt={}, T={}, kappa={}, amp={}, omega={}, smooth_iters={}, mesh_advection={}",
        args.n,
        args.dt,
        args.t_end,
        args.kappa,
        args.amp,
        args.omega,
        args.smooth_iters,
        args.mesh_advection,
    );
    print_template_header("Example 46: moving-mesh transient heat", spec, &config_line);

    let restart_state = args
        .restart
        .as_deref()
        .map(read_transient_checkpoint)
        .transpose()
        .unwrap_or_else(|e| panic!("failed to read restart state: {e}"))
        .or_else(|| {
            args.restart_h5
                .as_deref()
                .map(read_transient_hdf5_checkpoint)
                .transpose()
                .unwrap_or_else(|e| panic!("failed to read HDF5 restart state: {e}"))
        });

    let result = if let Some(restart) = restart_state.as_ref() {
        solve_case_with_restart(
            args.n,
            args.dt,
            args.t_end,
            args.kappa,
            args.amp,
            args.omega,
            args.smooth_iters,
            args.mesh_advection,
            Some(restart),
        )
    } else {
        solve_case(
            args.n,
            args.dt,
            args.t_end,
            args.kappa,
            args.amp,
            args.omega,
            args.smooth_iters,
            args.mesh_advection,
        )
    };

    println!("  final time      = {:.6e}", result.final_time);
    println!("  dofs            = {}", result.n_dofs);
    println!("  recorded L2 samples = {}", result.l2_history.len());
    println!("  final ||u||_2   = {:.6e}", result.final_l2);
    println!("  final checksum  = {:.8e}", result.final_checksum);
    println!(
        "  max transfer integral error = {:.3e}",
        result.max_transfer_abs_int_err
    );
    let steps = result.l2_history.len().saturating_sub(1);
    let coupling = TemplateCouplingSummary {
        steps,
        converged_steps: steps,
        max_coupling_iters_used: 1,
    };
    print_template_coupling_summary(coupling);
    let adaptive = TemplateAdaptiveSummary {
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
    };
    print_template_adaptive_summary(adaptive);
    if let Err(e) = maybe_write_template_kpi_csv(
        spec.template.id(),
        coupling,
        adaptive,
        &[
            ("final_time", result.final_time),
            ("final_l2", result.final_l2),
            ("final_checksum", result.final_checksum),
            ("max_transfer_abs_int_err", result.max_transfer_abs_int_err),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }

    if let Some(path) = &args.checkpoint {
        let checkpoint = TransientCheckpointState {
            completed_steps: result.completed_steps,
            current_time: result.final_time,
            dt: args.dt,
            prev_shift: result.prev_shift,
            temperature: result.temperature.clone(),
        };
        if let Err(e) = write_transient_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write checkpoint: {e}");
        } else {
            println!("  checkpoint written: {path}");
        }
    }

    if let Some(path) = &args.checkpoint_h5 {
        let checkpoint = TransientCheckpointState {
            completed_steps: result.completed_steps,
            current_time: result.final_time,
            dt: args.dt,
            prev_shift: result.prev_shift,
            temperature: result.temperature.clone(),
        };
        if let Err(e) = write_transient_hdf5_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write HDF5 checkpoint: {e}");
        } else {
            println!("  HDF5 checkpoint written: {path}");
            #[cfg(feature = "io_hdf5")]
            if let Err(e) = write_transient_hdf5_xdmf_sidecars(path, &checkpoint) {
                eprintln!("warning: failed to write checkpoint XDMF sidecars: {e}");
            }
        }
    }

    if let Some(prefix) = &args.export_vtk_prefix {
        if let Err(e) = write_ex46_vtk_export(prefix, &result.final_mesh, &result.temperature) {
            eprintln!("warning: failed to write VTK export: {e}");
        } else {
            println!("  VTK export written: {prefix}_temperature.vtu");
        }
    }
}

fn solve_case(
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    mesh_advection: bool,
) -> SolveResult {
    solve_case_with_restart(
        n,
        dt,
        t_end,
        kappa,
        amp,
        omega,
        smooth_iters,
        mesh_advection,
        None,
    )
}

fn solve_case_with_restart(
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    mesh_advection: bool,
    restart: Option<&TransientCheckpointState>,
) -> SolveResult {
    let mut mesh = Mesh::<2>::unit_square_tri(n);
    let n_steps = (t_end / dt).ceil() as usize;
    let completed_steps = restart.map(|r| r.completed_steps).unwrap_or(0);
    let mut prev_shift = 0.0_f64;

    if let Some(restart) = restart {
        assert!((restart.dt - dt).abs() < 1.0e-12,
            "restart dt ({}) does not match requested dt ({dt})",
            restart.dt);
        assert_eq!(restart.temperature.len(), (n + 1) * (n + 1),
            "restart state has unexpected temperature DOF count");
        for step in 1..=restart.completed_steps {
            let _ = apply_mesh_motion_step(
                &mut mesh,
                step as f64 * dt,
                amp,
                omega,
                smooth_iters,
                &mut prev_shift,
            );
        }
        assert!((prev_shift - restart.prev_shift).abs() < 1.0e-10,
            "reconstructed mesh shift ({prev_shift}) does not match checkpoint shift ({})",
            restart.prev_shift);
    }

    let mut space = H1Space::new(mesh.clone(), 1);
    let mut u = if let Some(restart) = restart {
        Vector::from_vec(restart.temperature.clone())
    } else {
        space.interpolate(&|x| (PI * x[0]).sin() * (PI * x[1]).sin())
    };
    let mut t = restart.map(|r| r.current_time).unwrap_or(0.0);

    let solve_cfg = SolverConfig {
        rtol: 1.0e-12,
        atol: 0.0,
        max_iter: 1200,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut max_transfer_abs_int_err = 0.0_f64;
    let mut l2_history = Vec::<f64>::with_capacity(n_steps + 1);
    l2_history.push(l2_norm(u.as_slice()));

    for _local_step in 1..=n_steps {
        let old_mesh = mesh.clone();
        let next_time = t + dt;
        let delta_shift = apply_mesh_motion_step(
            &mut mesh,
            next_time,
            amp,
            omega,
            smooth_iters,
            &mut prev_shift,
        );

        let src = H1Space::new(old_mesh, 1);
        let dst = H1Space::new(mesh.clone(), 1);
        let (u_transfer, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &src,
            u.as_slice(),
            &dst,
            1.0e-12,
            4,
        )
        .expect("conservative transfer should succeed");
        max_transfer_abs_int_err =
            max_transfer_abs_int_err.max(report.absolute_integral_error_after);

        space = dst;
        let n_dofs = space.n_dofs();
        let mut u_vec = u_transfer;

        let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa }], 3);

        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let mut rhs = vec![0.0_f64; n_dofs];
        m_mat.spmv(&u_vec, &mut rhs);

        let sys = if mesh_advection {
            let shift_rate = delta_shift / dt.max(1.0e-14);
            let conv = Assembler::assemble_bilinear(
                &space,
                &[&ConvectionIntegrator {
                    velocity: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                        let taper = (PI * x[0]).sin().powi(2);
                        out[0] = shift_rate * taper;
                        out[1] = 0.0;
                    }),
                }],
                3,
            );
            add_csr_scaled3(&m_mat, &k_mat, 1.0, &conv, 1.0, dt)
        } else {
            add_csr_scaled3(&m_mat, &k_mat, 1.0, &k_mat, 0.0, dt)
        };

        let mut sys = sys;
        let vals = vec![0.0_f64; bnd.len()];
        apply_dirichlet(&mut sys, &mut rhs, &bnd, &vals);

        let mut u_new = vec![0.0_f64; n_dofs];
        let _ = solve_pcg_jacobi(&sys, &rhs, &mut u_new, &solve_cfg);
        for &d in &bnd {
            u_new[d as usize] = 0.0;
        }

        u_vec = u_new;
        u = Vector::from_vec(u_vec);

        t += dt;
        l2_history.push(l2_norm(u.as_slice()));
    }

    let final_l2 = l2_norm(u.as_slice());
    let final_checksum = checksum(u.as_slice());
    SolveResult {
        final_time: t,
        completed_steps: completed_steps + n_steps,
        n_dofs: space.n_dofs(),
        final_l2,
        final_checksum,
        max_transfer_abs_int_err,
        l2_history,
        prev_shift,
        temperature: u.as_slice().to_vec(),
        final_mesh: mesh,
    }
}

fn apply_mesh_motion_step(
    mesh: &mut Mesh<2>,
    target_time: f64,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    prev_shift: &mut f64,
) -> f64 {
    let top_nodes: Vec<u32> = all_boundary_nodes(mesh)
        .into_iter()
        .filter(|&nid| (mesh.coords_of(nid)[1] - 1.0).abs() < 1.0e-12)
        .collect();

    let target_shift = amp * (2.0 * PI * target_time).sin();
    let delta_shift = target_shift - *prev_shift;
    *prev_shift = target_shift;

    apply_node_displacement(mesh, &top_nodes, |p| {
        let taper = (PI * p[0]).sin().powi(2);
        [delta_shift * taper, 0.0]
    });

    let fixed = all_boundary_nodes(mesh);
    let _ = laplacian_smooth_2d(
        mesh,
        &fixed,
        MeshMotionConfig {
            omega,
            max_iters: smooth_iters,
            tol: 1.0e-12,
        },
    );
    delta_shift
}

fn add_csr_scaled3(
    m: &CsrMatrix<f64>,
    k: &CsrMatrix<f64>,
    k_scale: f64,
    c: &CsrMatrix<f64>,
    c_scale: f64,
    dt: f64,
) -> CsrMatrix<f64> {
    let n = m.nrows;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for ptr in m.row_ptr[i]..m.row_ptr[i + 1] {
            coo.add(i, m.col_idx[ptr] as usize, m.values[ptr]);
        }
    }
    for i in 0..n {
        for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
            coo.add(i, k.col_idx[ptr] as usize, dt * k_scale * k.values[ptr]);
        }
    }
    for i in 0..n {
        for ptr in c.row_ptr[i]..c.row_ptr[i + 1] {
            coo.add(i, c.col_idx[ptr] as usize, dt * c_scale * c.values[ptr]);
        }
    }
    coo.into_csr()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn checksum(v: &[f64]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, val)| (i as f64 + 1.0) * val)
        .sum::<f64>()
}

struct Args {
    n: usize,
    dt: f64,
    t_end: f64,
    kappa: f64,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    mesh_advection: bool,
    checkpoint: Option<String>,
    restart: Option<String>,
    checkpoint_h5: Option<String>,
    restart_h5: Option<String>,
    export_vtk_prefix: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 20,
        dt: 0.01,
        t_end: 0.2,
        kappa: 1.0,
        amp: 0.015,
        omega: 0.7,
        smooth_iters: 25,
        mesh_advection: true,
        checkpoint: None,
        restart: None,
        checkpoint_h5: None,
        restart_h5: None,
        export_vtk_prefix: None,
    };

    let mut workflow = WorkflowCliOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if workflow.try_parse_arg(arg.as_str(), &mut it) {
            continue;
        }
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--dt" => a.dt = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "--T" => a.t_end = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            "--kappa" => a.kappa = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--amp" => a.amp = it.next().unwrap_or("0.015".into()).parse().unwrap_or(0.015),
            "--omega" => a.omega = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--smooth-iters" => {
                a.smooth_iters = it.next().unwrap_or("25".into()).parse().unwrap_or(25)
            }
            "--mesh-advection" => a.mesh_advection = true,
            "--no-mesh-advection" => a.mesh_advection = false,
            _ => {}
        }
    }
    a.checkpoint = workflow.checkpoint;
    a.restart = workflow.restart;
    a.checkpoint_h5 = workflow.checkpoint_h5;
    a.restart_h5 = workflow.restart_h5;
    a.export_vtk_prefix = workflow.export_vtk_prefix;

    a.omega = a.omega.clamp(0.05, 0.95);
    a
}

fn write_ex46_vtk_export(
    prefix: &str,
    mesh: &Mesh<2>,
    temperature: &[f64],
) -> Result<(), String> {
    let path = format!("{prefix}_temperature.vtu");
    ensure_parent_dir(&path).map_err(|e| e.to_string())?;
    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("temperature", temperature.to_vec()));
    writer.write_file(&path).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_transient_checkpoint(path: &str, state: &TransientCheckpointState) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let content = format!(
        "format=ex46_moving_mesh_heat_v1\ncompleted_steps={}\ncurrent_time={:.17e}\ndt={:.17e}\nprev_shift={:.17e}\ntemperature={}\n",
        state.completed_steps,
        state.current_time,
        state.dt,
        state.prev_shift,
        format_vec_f64(&state.temperature),
    );
    fs::write(path, content)
}

fn write_transient_hdf5_checkpoint(
    path: &str,
    state: &TransientCheckpointState,
) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(path);

    let bundle = CheckpointBundleF64 {
        mesh_meta: None,
        fields: vec![
            scalar_rank_field_f64("prev_shift", state.prev_shift),
            vector_rank_field_f64("temp", state.temperature.clone()),
        ],
    };
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let step = state.completed_steps as u64;
    let time = state.current_time;
    write_checkpoint_step_bundle_f64(path, cfg, step, time, &bundle, IoBackend::Partitioned)
        .map_err(|e| e.to_string())?;
    validate_checkpoint_layout(path, Some(1)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_transient_hdf5_checkpoint(path: &str) -> Result<TransientCheckpointState, String> {
    let fields = read_checkpoint_fields_f64_latest(
        path,
        ParallelIoConfig { world_size: 1, rank: 0 },
        &["temp", "prev_shift"],
    )
    .map_err(|e| e.to_string())?;

    let mut completed_steps = None;
    let mut current_time = None;
    let mut dt = None;
    let mut prev_shift = None;
    let mut temperature = None;

    for (name, field) in fields {
        completed_steps = Some(field.step as usize);
        current_time = Some(field.time);
        dt = Some(if field.step == 0 {
            0.0
        } else {
            field.time / field.step as f64
        });
        match name.as_str() {
            "temp" => temperature = Some(field.values),
            "prev_shift" => prev_shift = field.values.first().copied(),
            _ => {}
        }
    }

    let current_time = current_time.ok_or_else(|| "missing checkpoint time".to_string())?;
    Ok(TransientCheckpointState {
        completed_steps: completed_steps.ok_or_else(|| "missing checkpoint step".to_string())?,
        current_time,
        dt: dt.ok_or_else(|| "missing checkpoint dt".to_string())?,
        prev_shift: prev_shift.ok_or_else(|| "missing prev_shift field".to_string())?,
        temperature: temperature.ok_or_else(|| "missing temp field".to_string())?,
    })
}

#[cfg(feature = "io_hdf5")]
fn write_transient_hdf5_xdmf_sidecars(
    h5_path: &str,
    state: &TransientCheckpointState,
) -> Result<(), String> {
    let step = state.completed_steps as u64;
    let time = state.current_time;
    write_scalar_checkpoint_xdmf_sidecars(h5_path, step, time, &["temp"])
}

fn read_transient_checkpoint(path: &str) -> Result<TransientCheckpointState, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut format = None;
    let mut completed_steps = None;
    let mut current_time = None;
    let mut dt = None;
    let mut prev_shift = None;
    let mut temperature = None;

    for line in content.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "format" => format = Some(value.trim().to_string()),
            "completed_steps" => {
                completed_steps = Some(value.trim().parse::<usize>().map_err(|e| e.to_string())?)
            }
            "current_time" => {
                current_time = Some(value.trim().parse::<f64>().map_err(|e| e.to_string())?)
            }
            "dt" => dt = Some(value.trim().parse::<f64>().map_err(|e| e.to_string())?),
            "prev_shift" => {
                prev_shift = Some(value.trim().parse::<f64>().map_err(|e| e.to_string())?)
            }
            "temperature" => temperature = Some(parse_checkpoint_values(value.trim())),
            _ => {}
        }
    }

    match format.as_deref() {
        Some("ex46_moving_mesh_heat_v1") => {}
        Some(other) => return Err(format!("unsupported checkpoint format: {other}")),
        None => return Err("checkpoint missing format header".into()),
    }

    Ok(TransientCheckpointState {
        completed_steps: completed_steps.ok_or_else(|| "checkpoint missing completed_steps".to_string())?,
        current_time: current_time.ok_or_else(|| "checkpoint missing current_time".to_string())?,
        dt: dt.ok_or_else(|| "checkpoint missing dt".to_string())?,
        prev_shift: prev_shift.ok_or_else(|| "checkpoint missing prev_shift".to_string())?,
        temperature: temperature.ok_or_else(|| "checkpoint missing temperature values".to_string())?,
    })
}

fn parse_checkpoint_values(value: &str) -> Vec<f64> {
    parse_vec_f64(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static KPI_ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn ex46_moving_mesh_heat_has_stable_decay_and_conservative_transfer() {
        let r = solve_case(10, 0.02, 0.2, 1.0, 0.0, 0.7, 15, true);
        assert_eq!(r.n_dofs, 121);
        assert!(r.max_transfer_abs_int_err < 1.0e-10, "transfer drift too large: {}", r.max_transfer_abs_int_err);
        assert!(r.final_l2.is_finite());
        assert!(r.final_l2 > 0.0);

        for i in 1..r.l2_history.len() {
            assert!(
                r.l2_history[i] <= r.l2_history[i - 1] + 1.0e-8,
                "L2 should be non-increasing for implicit heat step: prev={} cur={} at i={}",
                r.l2_history[i - 1],
                r.l2_history[i],
                i
            );
        }
    }

    #[test]
    fn ex46_mesh_advection_changes_solution_when_mesh_moves() {
        let no_adv = solve_case(10, 0.02, 0.2, 1.0, 0.01, 0.7, 15, false);
        let with_adv = solve_case(10, 0.02, 0.2, 1.0, 0.01, 0.7, 15, true);
        let diff = (with_adv.final_checksum - no_adv.final_checksum).abs();
        assert!(
            diff > 1.0e-8,
            "mesh-advection switch should alter solution for moving mesh, checksum diff={diff}"
        );
    }

    /// Higher diffusivity kappa should cause faster energy decay.
    #[test]
    fn ex46_higher_kappa_causes_faster_decay() {
        let low_kappa  = solve_case(10, 0.02, 0.2, 0.5, 0.0, 0.7, 15, false);
        let high_kappa = solve_case(10, 0.02, 0.2, 2.0, 0.0, 0.7, 15, false);
        assert!(
            high_kappa.final_l2 < low_kappa.final_l2,
            "higher kappa should give lower final L2: low={:.4e} high={:.4e}",
            low_kappa.final_l2, high_kappa.final_l2
        );
    }

    /// L2 norm must remain positive (no over-damping to exactly zero in finite steps).
    #[test]
    fn ex46_solution_remains_positive_and_finite() {
        let r = solve_case(8, 0.01, 0.1, 1.0, 0.005, 0.7, 10, true);
        assert!(r.final_l2 > 0.0, "solution collapsed to zero");
        assert!(r.final_l2.is_finite(), "solution blew up");
        for (i, &l2) in r.l2_history.iter().enumerate() {
            assert!(l2.is_finite(), "L2 history NaN/inf at step {i}");
        }
    }

    #[test]
    fn ex46_dof_count_matches_p1_h1_formula() {
        for &n in &[6usize, 10usize, 14usize] {
            let r = solve_case(n, 0.02, 0.2, 1.0, 0.0, 0.7, 15, false);
            assert_eq!(r.n_dofs, (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex46_zero_final_time_keeps_initial_state() {
        let r = solve_case(8, 0.01, 0.0, 1.0, 0.01, 0.7, 10, true);
        assert!((r.final_time - 0.0).abs() < 1.0e-14);
        assert_eq!(r.l2_history.len(), 1);
        let initial_l2 = r.l2_history[0];
        assert!((r.final_l2 - initial_l2).abs() < 1.0e-14,
            "zero-time integration should be no-op: initial={} final={}",
            initial_l2,
            r.final_l2);
    }

    #[test]
    fn ex46_smaller_dt_gives_consistent_solution() {
        let coarse = solve_case(10, 0.02, 0.2, 1.0, 0.0, 0.7, 15, false);
        let fine = solve_case(10, 0.01, 0.2, 1.0, 0.0, 0.7, 15, false);
        let rel = (fine.final_l2 - coarse.final_l2).abs() / fine.final_l2.max(1.0e-300);
        assert!(fine.final_l2 <= coarse.final_l2,
            "smaller dt should not increase final L2 in this implicit diffusion regime: coarse={} fine={}",
            coarse.final_l2,
            fine.final_l2);
        assert!(rel < 0.5,
            "time-step refinement drift too large: coarse={} fine={} rel={}",
            coarse.final_l2,
            fine.final_l2,
            rel);
    }

    #[test]
    fn ex46_amp_zero_makes_advection_toggle_equivalent() {
        let no_adv = solve_case(10, 0.02, 0.2, 1.0, 0.0, 0.7, 15, false);
        let with_adv = solve_case(10, 0.02, 0.2, 1.0, 0.0, 0.7, 15, true);
        assert!((with_adv.final_checksum - no_adv.final_checksum).abs() < 1.0e-10,
            "with amp=0, advection toggle should be equivalent: no_adv={} with_adv={}",
            no_adv.final_checksum,
            with_adv.final_checksum);
        assert!((with_adv.final_l2 - no_adv.final_l2).abs() < 1.0e-12);
    }

    #[test]
    fn ex46_template_kpi_csv_row_uses_moving_mesh_contract() {
        let _guard = KPI_ENV_LOCK.lock().unwrap();
        let result = solve_case(8, 0.02, 0.08, 1.0, 0.01, 0.7, 10, true);
        let temp_path = std::env::temp_dir().join(format!(
            "ex46_template_kpi_{}.csv",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        std::env::set_var("FEM_TEMPLATE_KPI_CSV", &temp_path);
        std::env::set_var("FEM_TEMPLATE_KPI_RUN_ID", "test");
        std::env::set_var("FEM_TEMPLATE_KPI_TAG", "unit");

        let steps = result.l2_history.len().saturating_sub(1);
        let coupling = TemplateCouplingSummary {
            steps,
            converged_steps: steps,
            max_coupling_iters_used: 1,
        };
        let adaptive = TemplateAdaptiveSummary {
            sync_retries: 0,
            rejected_sync_steps: 0,
            rollback_count: 0,
        };
        maybe_write_template_kpi_csv(
            builtin_template_spec(BuiltinMultiphysicsTemplate::MovingMeshHeat)
                .template
                .id(),
            coupling,
            adaptive,
            &[
                ("final_time", result.final_time),
                ("final_l2", result.final_l2),
                ("final_checksum", result.final_checksum),
                ("max_transfer_abs_int_err", result.max_transfer_abs_int_err),
            ],
        )
        .unwrap();

        let csv = fs::read_to_string(&temp_path).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains("moving_mesh_heat,test,unit"));
        assert!(lines[1].contains("final_l2="));
        assert!(lines[1].contains("max_transfer_abs_int_err="));

        std::env::remove_var("FEM_TEMPLATE_KPI_CSV");
        std::env::remove_var("FEM_TEMPLATE_KPI_RUN_ID");
        std::env::remove_var("FEM_TEMPLATE_KPI_TAG");
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn ex46_transient_checkpoint_roundtrip_restarts_consistently() {
        let full = solve_case(8, 0.02, 0.16, 1.0, 0.01, 0.7, 10, true);
        let partial = solve_case(8, 0.02, 0.08, 1.0, 0.01, 0.7, 10, true);
        let temp_path = std::env::temp_dir().join(format!(
            "ex46_checkpoint_roundtrip_{}.txt",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        let checkpoint = TransientCheckpointState {
            completed_steps: partial.completed_steps,
            current_time: partial.final_time,
            dt: 0.02,
            prev_shift: partial.prev_shift,
            temperature: partial.temperature.clone(),
        };
        write_transient_checkpoint(temp_path.to_str().unwrap(), &checkpoint).unwrap();
        let restart = read_transient_checkpoint(temp_path.to_str().unwrap()).unwrap();
        let resumed = solve_case_with_restart(8, 0.02, 0.08, 1.0, 0.01, 0.7, 10, true, Some(&restart));

        assert!((resumed.final_time - full.final_time).abs() < 1.0e-12);
        assert!((resumed.final_l2 - full.final_l2).abs() < 1.0e-10,
            "restart drift in final L2: resumed={} full={}",
            resumed.final_l2,
            full.final_l2);
        assert!((resumed.final_checksum - full.final_checksum).abs() < 1.0e-8,
            "restart drift in checksum: resumed={} full={}",
            resumed.final_checksum,
            full.final_checksum);

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn ex46_vtk_export_writes_temperature_file() {
        let result = solve_case(6, 0.02, 0.08, 1.0, 0.01, 0.7, 8, true);
        let temp_dir = std::env::temp_dir().join(format!(
            "ex46_vtk_export_{}",
            std::process::id()
        ));
        let prefix = temp_dir.join("moving_mesh_heat");
        let path = format!("{}_temperature.vtu", prefix.to_string_lossy());
        let _ = fs::remove_file(&path);

        write_ex46_vtk_export(
            prefix.to_str().unwrap(),
            &result.final_mesh,
            &result.temperature,
        )
        .unwrap();

        let vtk = fs::read_to_string(&path).unwrap();
        assert!(vtk.contains("temperature"));
        assert!(vtk.contains("UnstructuredGrid"));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn ex46_hdf5_checkpoint_roundtrip_restarts_consistently() {
        let full = solve_case(8, 0.02, 0.16, 1.0, 0.01, 0.7, 10, true);
        let partial = solve_case(8, 0.02, 0.08, 1.0, 0.01, 0.7, 10, true);
        let temp_path = std::env::temp_dir().join(format!(
            "ex46_checkpoint_roundtrip_{}.h5",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        let checkpoint = TransientCheckpointState {
            completed_steps: partial.completed_steps,
            current_time: partial.final_time,
            dt: 0.02,
            prev_shift: partial.prev_shift,
            temperature: partial.temperature.clone(),
        };
        write_transient_hdf5_checkpoint(temp_path.to_str().unwrap(), &checkpoint).unwrap();
        let restart = read_transient_hdf5_checkpoint(temp_path.to_str().unwrap()).unwrap();
        let resumed = solve_case_with_restart(8, 0.02, 0.08, 1.0, 0.01, 0.7, 10, true, Some(&restart));

        assert!((resumed.final_time - full.final_time).abs() < 1.0e-12);
        assert!((resumed.final_l2 - full.final_l2).abs() < 1.0e-10,
            "restart drift in final L2: resumed={} full={}",
            resumed.final_l2,
            full.final_l2);
        assert!((resumed.final_checksum - full.final_checksum).abs() < 1.0e-8,
            "restart drift in checksum: resumed={} full={}",
            resumed.final_checksum,
            full.final_checksum);

        let _ = fs::remove_file(&temp_path);
    }

    #[cfg(feature = "io_hdf5")]
    #[test]
    fn ex46_hdf5_checkpoint_writes_xdmf_sidecar() {
        let result = solve_case(8, 0.02, 0.08, 1.0, 0.01, 0.7, 10, true);
        let temp_path = std::env::temp_dir().join(format!(
            "ex46_sidecar_roundtrip_{}.h5",
            std::process::id()
        ));
        let h5_path = temp_path.to_string_lossy().to_string();
        let _ = fs::remove_file(&temp_path);

        let checkpoint = TransientCheckpointState {
            completed_steps: result.completed_steps,
            current_time: result.final_time,
            dt: 0.02,
            prev_shift: result.prev_shift,
            temperature: result.temperature.clone(),
        };
        write_transient_hdf5_checkpoint(&h5_path, &checkpoint).unwrap();
        write_transient_hdf5_xdmf_sidecars(&h5_path, &checkpoint).unwrap();

        let sidecar = checkpoint_sidecar_path(&h5_path, "temp").unwrap();
        let xml = fs::read_to_string(&sidecar).unwrap();
        assert!(xml.contains("temp"));
        assert!(xml.contains("checkpoint_step_"));

        let _ = fs::remove_file(&sidecar);
        let _ = fs::remove_file(&temp_path);
    }
}
