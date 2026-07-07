//! Example 45: quasi-ALE moving mesh with conservative field transfer.
//!
//! Demonstrates a lightweight dynamic mesh-update loop:
//! 1) move a boundary subset (top wall)
//! 2) smooth interior nodes (Laplacian)
//! 3) transfer field from old mesh to new mesh conservatively
//!
//! This is a quasi-ALE baseline intended as a stepping stone toward full ALE.
//!
//! Optional workflow hooks:
//! - `--checkpoint <path>` / `--restart <path>` use a lightweight text
//!   checkpoint format for split quasi-ALE restart.
//! - `--checkpoint-h5 <path>` / `--restart-h5 <path>` use the shared
//!   `fem-io-hdf5-parallel` checkpoint format; when built with `--features
//!   io_hdf5`, a `transported_scalar` XDMF sidecar is also emitted.
//! - `--checkpoint-at-step <k>` runs only through step `k` while preserving the
//!   full `--steps` trajectory count for later restart.
//! - `--export-vtk-prefix <prefix>` writes the final transported scalar field
//!   on the deformed mesh as `<prefix>_scalar.vtu`.

use std::fs;
use std::f64::consts::PI;
use std::io;

use fem_assembly::transfer_h1_p1_nonmatching_l2_projection_conservative;
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
use fem_linalg::Vector;
use fem_mesh::{
    MeshMotionConfig,
    Mesh,
    all_boundary_nodes,
    apply_node_displacement,
    laplacian_smooth_2d,
};
use fem_solver::{BuiltinMultiphysicsTemplate, builtin_template_spec};
use fem_space::{H1Space, fe_space::FESpace};

struct SolveResult {
    steps: usize,
    completed_steps: usize,
    n_dofs: usize,
    final_norm: f64,
    final_checksum: f64,
    max_abs_int_err: f64,
    prev_shift: f64,
    values: Vec<f64>,
    final_mesh: Mesh<2>,
}

struct TransientCheckpointState {
    completed_steps: usize,
    total_steps: usize,
    prev_shift: f64,
    values: Vec<f64>,
}

struct CliArgs {
    sim: Args,
    checkpoint: Option<String>,
    checkpoint_h5: Option<String>,
    checkpoint_at_step: Option<usize>,
    restart: Option<String>,
    restart_h5: Option<String>,
    export_vtk_prefix: Option<String>,
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
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::MovingMeshAle);
    let config_line = format!(
        "n={}, steps={}, amp={}, omega={}, smooth_iters={}",
        cli.sim.n, cli.sim.steps, cli.sim.amp, cli.sim.omega, cli.sim.smooth_iters
    );
    print_template_header("Example 45: quasi-ALE moving mesh", spec, &config_line);

    let restart_state = cli
        .restart
        .as_deref()
        .map(read_transient_checkpoint)
        .transpose()
        .unwrap_or_else(|e| panic!("failed to read restart state: {e}"))
        .or_else(|| {
            cli.restart_h5
                .as_deref()
                .map(read_transient_hdf5_checkpoint)
                .transpose()
                .unwrap_or_else(|e| panic!("failed to read HDF5 restart state: {e}"))
        });

    let result = if let Some(restart) = restart_state.as_ref() {
        solve_case_with_restart(&cli.sim, Some(restart), cli.checkpoint_at_step)
    } else if cli.checkpoint_at_step.is_some() {
        solve_case_with_restart(&cli.sim, None, cli.checkpoint_at_step)
    } else {
        solve_case(&cli.sim)
    };
    let coupling = TemplateCouplingSummary {
        steps: result.steps,
        converged_steps: result.steps,
        max_coupling_iters_used: 1,
    };
    let adaptive = TemplateAdaptiveSummary {
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
    };

    println!("  dofs            = {}", result.n_dofs);
    println!("  final ||u||_2 = {:.6e}", result.final_norm);
    println!("  final checksum = {:.8e}", result.final_checksum);
    println!("  max absolute integral error after correction = {:.3e}", result.max_abs_int_err);
    print_template_coupling_summary(coupling);
    print_template_adaptive_summary(adaptive);
    if let Err(e) = maybe_write_template_kpi_csv(
        spec.template.id(),
        coupling,
        adaptive,
        &[
            ("final_norm", result.final_norm),
            ("final_checksum", result.final_checksum),
            ("max_abs_int_err", result.max_abs_int_err),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }

    if let Some(path) = &cli.checkpoint {
        let checkpoint = TransientCheckpointState {
            completed_steps: result.completed_steps,
            total_steps: cli.sim.steps,
            prev_shift: result.prev_shift,
            values: result.values.clone(),
        };
        if let Err(e) = write_transient_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write checkpoint: {e}");
        } else {
            println!("  checkpoint written: {path}");
        }
    }

    if let Some(path) = &cli.checkpoint_h5 {
        let checkpoint = TransientCheckpointState {
            completed_steps: result.completed_steps,
            total_steps: cli.sim.steps,
            prev_shift: result.prev_shift,
            values: result.values.clone(),
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

    if let Some(prefix) = &cli.export_vtk_prefix {
        if let Err(e) = write_ex45_vtk_export(prefix, &result.final_mesh, &result.values) {
            eprintln!("warning: failed to write VTK export: {e}");
        } else {
            println!("  VTK export written: {prefix}_scalar.vtu");
        }
    }
}

fn solve_case(args: &Args) -> SolveResult {
    solve_case_with_restart(args, None, None)
}

fn solve_case_with_restart(
    args: &Args,
    restart: Option<&TransientCheckpointState>,
    checkpoint_at_step: Option<usize>,
) -> SolveResult {
    let mut mesh = Mesh::<2>::unit_square_tri(args.n);
    let total_steps = args.steps;
    let start_step = restart.map(|r| r.completed_steps).unwrap_or(0);
    let stop_step = checkpoint_at_step.unwrap_or(total_steps);
    assert!(stop_step <= total_steps,
        "checkpoint-at-step ({stop_step}) exceeds total steps ({total_steps})");

    let mut prev_shift = 0.0_f64;
    if let Some(restart) = restart {
        assert_eq!(restart.total_steps, total_steps,
            "restart total_steps ({}) does not match requested steps ({total_steps})",
            restart.total_steps);
        assert_eq!(restart.values.len(), (args.n + 1) * (args.n + 1),
            "restart state has unexpected DOF count");
        for step in 1..=restart.completed_steps {
            apply_mesh_motion_step(
                &mut mesh,
                step,
                total_steps,
                args.amp,
                args.omega,
                args.smooth_iters,
                &mut prev_shift,
            );
        }
        assert!((prev_shift - restart.prev_shift).abs() < 1.0e-10,
            "reconstructed mesh shift ({prev_shift}) does not match checkpoint shift ({})",
            restart.prev_shift);
    }

    let mut values = if let Some(restart) = restart {
        Vector::from_vec(restart.values.clone())
    } else {
        H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        })
    };

    let mut max_abs_int_err = 0.0_f64;
    for step in (start_step + 1)..=stop_step {
        let old_mesh = mesh.clone();
        apply_mesh_motion_step(
            &mut mesh,
            step,
            total_steps,
            args.amp,
            args.omega,
            args.smooth_iters,
            &mut prev_shift,
        );

        let src = H1Space::new(old_mesh, 1);
        let dst = H1Space::new(mesh.clone(), 1);
        let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &src,
            values.as_slice(),
            &dst,
            1.0e-12,
            4,
        )
        .expect("conservative transfer should succeed");

        max_abs_int_err = max_abs_int_err.max(report.absolute_integral_error_after);
        values = Vector::from_vec(v_new);
    }

    let final_norm = values.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
    let final_checksum = values
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, val)| (i as f64 + 1.0) * val)
        .sum::<f64>();
    SolveResult {
        steps: stop_step.saturating_sub(start_step),
        completed_steps: stop_step,
        n_dofs: H1Space::new(mesh.clone(), 1).n_dofs(),
        final_norm,
        final_checksum,
        max_abs_int_err,
        prev_shift,
        values: values.as_slice().to_vec(),
        final_mesh: mesh,
    }
}

fn apply_mesh_motion_step(
    mesh: &mut Mesh<2>,
    step: usize,
    total_steps: usize,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
    prev_shift: &mut f64,
) {
    let top_nodes: Vec<u32> = all_boundary_nodes(mesh)
        .into_iter()
        .filter(|&n| {
            let p = mesh.coords_of(n);
            (p[1] - 1.0).abs() < 1.0e-12
        })
        .collect();

    let phase = step as f64 / total_steps.max(1) as f64;
    let target_shift = amp * (2.0 * PI * phase).sin();
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
}

fn write_ex45_vtk_export(
    prefix: &str,
    mesh: &Mesh<2>,
    values: &[f64],
) -> Result<(), String> {
    let path = format!("{prefix}_scalar.vtu");
    ensure_parent_dir(&path).map_err(|e| e.to_string())?;
    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("transported_scalar", values.to_vec()));
    writer.write_file(&path).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_transient_checkpoint(path: &str, state: &TransientCheckpointState) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let values = format_vec_f64(&state.values);
    let content = format!(
        "format=ex45_moving_mesh_ale_v1\ncompleted_steps={}\ntotal_steps={}\nprev_shift={:.17e}\nvalues={}\n",
        state.completed_steps,
        state.total_steps,
        state.prev_shift,
        values,
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
            scalar_rank_field_f64("total_steps", state.total_steps as f64),
            scalar_rank_field_f64("prev_shift", state.prev_shift),
            vector_rank_field_f64("transported_scalar", state.values.clone()),
        ],
    };
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    let step = state.completed_steps as u64;
    let time = step as f64 / state.total_steps.max(1) as f64;
    write_checkpoint_step_bundle_f64(path, cfg, step, time, &bundle, IoBackend::Partitioned)
        .map_err(|e| e.to_string())?;
    validate_checkpoint_layout(path, Some(1)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_transient_hdf5_checkpoint(path: &str) -> Result<TransientCheckpointState, String> {
    let fields = read_checkpoint_fields_f64_latest(
        path,
        ParallelIoConfig { world_size: 1, rank: 0 },
        &["total_steps", "prev_shift", "transported_scalar"],
    )
    .map_err(|e| e.to_string())?;

    let mut completed_steps = None;
    let mut total_steps = None;
    let mut prev_shift = None;
    let mut values = None;

    for (name, field) in fields {
        completed_steps = Some(field.step as usize);
        match name.as_str() {
            "total_steps" => {
                total_steps = field.values.first().map(|v| *v as usize);
            }
            "prev_shift" => {
                prev_shift = field.values.first().copied();
            }
            "transported_scalar" => values = Some(field.values),
            _ => {}
        }
    }

    Ok(TransientCheckpointState {
        completed_steps: completed_steps.ok_or_else(|| "missing checkpoint step".to_string())?,
        total_steps: total_steps.ok_or_else(|| "missing total_steps field".to_string())?,
        prev_shift: prev_shift.ok_or_else(|| "missing prev_shift field".to_string())?,
        values: values.ok_or_else(|| "missing transported_scalar field".to_string())?,
    })
}

#[cfg(feature = "io_hdf5")]
fn write_transient_hdf5_xdmf_sidecars(
    h5_path: &str,
    state: &TransientCheckpointState,
) -> Result<(), String> {
    let step = state.completed_steps as u64;
    let time = step as f64 / state.total_steps.max(1) as f64;
    write_scalar_checkpoint_xdmf_sidecars(h5_path, step, time, &["transported_scalar"])
}

fn read_transient_checkpoint(path: &str) -> Result<TransientCheckpointState, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut format = None;
    let mut completed_steps = None;
    let mut total_steps = None;
    let mut prev_shift = None;
    let mut values = None;

    for line in content.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "format" => format = Some(value.trim().to_string()),
            "completed_steps" => {
                completed_steps = Some(value.trim().parse::<usize>().map_err(|e| e.to_string())?)
            }
            "total_steps" => {
                total_steps = Some(value.trim().parse::<usize>().map_err(|e| e.to_string())?)
            }
            "prev_shift" => {
                prev_shift = Some(value.trim().parse::<f64>().map_err(|e| e.to_string())?)
            }
            "values" => values = Some(parse_checkpoint_values(value.trim())),
            _ => {}
        }
    }

    match format.as_deref() {
        Some("ex45_moving_mesh_ale_v1") => {}
        Some(other) => return Err(format!("unsupported checkpoint format: {other}")),
        None => return Err("checkpoint missing format header".into()),
    }

    Ok(TransientCheckpointState {
        completed_steps: completed_steps.ok_or_else(|| "checkpoint missing completed_steps".to_string())?,
        total_steps: total_steps.ok_or_else(|| "checkpoint missing total_steps".to_string())?,
        prev_shift: prev_shift.ok_or_else(|| "checkpoint missing prev_shift".to_string())?,
        values: values.ok_or_else(|| "checkpoint missing values".to_string())?,
    })
}

fn parse_checkpoint_values(value: &str) -> Vec<f64> {
    parse_vec_f64(value).unwrap_or_default()
}

struct Args {
    n: usize,
    steps: usize,
    amp: f64,
    omega: f64,
    smooth_iters: usize,
}

fn parse_args() -> CliArgs {
    let mut sim = Args {
        n: 20,
        steps: 20,
        amp: 0.02,
        omega: 0.7,
        smooth_iters: 30,
    };
    let mut checkpoint_at_step = None;
    let mut workflow = WorkflowCliOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if workflow.try_parse_arg(arg.as_str(), &mut it) {
            continue;
        }
        match arg.as_str() {
            "--n" => sim.n = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--steps" => sim.steps = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--amp" => sim.amp = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02),
            "--omega" => sim.omega = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7),
            "--smooth-iters" => {
                sim.smooth_iters = it.next().unwrap_or("30".into()).parse().unwrap_or(30)
            }
            "--checkpoint-at-step" => {
                checkpoint_at_step = it.next().and_then(|v| v.parse::<usize>().ok())
            }
            _ => {}
        }
    }
    sim.omega = sim.omega.clamp(0.05, 0.95);
    CliArgs {
        sim,
        checkpoint: workflow.checkpoint,
        checkpoint_h5: workflow.checkpoint_h5,
        checkpoint_at_step,
        restart: workflow.restart,
        restart_h5: workflow.restart_h5,
        export_vtk_prefix: workflow.export_vtk_prefix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    static KPI_ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn ex45_conservative_transfer_keeps_integral_near_machine_precision() {
        let args = Args {
            n: 10,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 15,
        };

        let mut mesh = Mesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });

        let mut max_abs_int_err = 0.0_f64;
        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
                .into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
                .collect();

            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                let taper = (PI * p[0]).sin().powi(2);
                [shift * taper, 0.0]
            });

            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(
                &mut mesh,
                &fixed,
                MeshMotionConfig {
                    omega: args.omega,
                    max_iters: args.smooth_iters,
                    tol: 1.0e-12,
                },
            );

            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src,
                values.as_slice(),
                &dst,
                1.0e-12,
                4,
            )
            .unwrap();

            max_abs_int_err = max_abs_int_err.max(report.absolute_integral_error_after);
            values = Vector::from_vec(v_new);
        }

        assert!(max_abs_int_err < 1.0e-10, "integral drift too large: {max_abs_int_err}");
    }

    /// With zero amplitude the mesh does not move, so the conservative transfer
    /// should produce zero integral error and the field should be unchanged.
    #[test]
    fn ex45_zero_amplitude_mesh_stays_unchanged_and_transfer_is_exact() {
        let args = Args { n: 8, steps: 3, amp: 0.0, omega: 0.7, smooth_iters: 10 };

        let mut mesh = Mesh::<2>::unit_square_tri(args.n);
        let space0 = H1Space::new(mesh.clone(), 1);
        let initial_values: Vec<f64> = space0.interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        }).as_slice().to_vec();
        let initial_norm: f64 = initial_values.iter().map(|v| v*v).sum::<f64>().sqrt();
        let mut values = Vector::from_vec(initial_values.clone());

        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh).into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12).collect();
            let shift = args.amp * (2.0 * PI * t).sin(); // = 0
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                let taper = (PI * p[0]).sin().powi(2);
                [shift * taper, 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig { omega: args.omega, max_iters: args.smooth_iters, tol: 1.0e-12 });
            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src, values.as_slice(), &dst, 1.0e-12, 4,
            ).unwrap();
            assert!(report.absolute_integral_error_after < 1.0e-12,
                "step {step}: non-trivial integral error on static mesh: {}", report.absolute_integral_error_after);
            values = Vector::from_vec(v_new);
        }

        // Field norm should be preserved (same mesh, same field).
        let final_norm: f64 = values.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();
        let rel_drift = (final_norm - initial_norm).abs() / initial_norm.max(1.0e-300);
        assert!(rel_drift < 1.0e-10, "field norm drifted on static mesh: rel={rel_drift:.3e}");
    }

    /// After mesh motion with moderate amplitude, all triangle areas must remain
    /// strictly positive (no element inversion due to smoothing).
    #[test]
    fn ex45_mesh_remains_valid_no_inverted_elements_after_motion() {
        let args = Args { n: 14, steps: 6, amp: 0.015, omega: 0.7, smooth_iters: 30 };

        let mut mesh = Mesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });

        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh).into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12).collect();
            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [shift * (PI * p[0]).sin().powi(2), 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig { omega: args.omega, max_iters: args.smooth_iters, tol: 1.0e-12 });

            // Check all element areas are positive.
            for e in 0..mesh.n_elems() as u32 {
                let nodes = mesh.elem_nodes(e);
                let p0 = mesh.coords_of(nodes[0]);
                let p1 = mesh.coords_of(nodes[1]);
                let p2 = mesh.coords_of(nodes[2]);
                let area = 0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]));
                assert!(area > 0.0, "step {step}: inverted element {e}, area={area:.6e}");
            }

            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, _report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src, values.as_slice(), &dst, 1.0e-12, 4,
            ).unwrap();
            values = Vector::from_vec(v_new);
        }
    }

    /// The L2 norm of the transferred field should not blow up over many steps
    /// (no exponential growth from repeated conservative transfer).
    #[test]
    fn ex45_field_norm_is_stable_over_many_steps() {
        let args = Args { n: 10, steps: 10, amp: 0.01, omega: 0.7, smooth_iters: 20 };

        let mut mesh = Mesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let initial_norm: f64 = values.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();

        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh).into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12).collect();
            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [shift * (PI * p[0]).sin().powi(2), 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(&mut mesh, &fixed, MeshMotionConfig { omega: args.omega, max_iters: args.smooth_iters, tol: 1.0e-12 });
            let src = H1Space::new(old_mesh.clone(), 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, _) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src, values.as_slice(), &dst, 1.0e-12, 4,
            ).unwrap();
            values = Vector::from_vec(v_new);
        }

        let final_norm: f64 = values.as_slice().iter().map(|v| v*v).sum::<f64>().sqrt();
        // Norm may change slightly due to mesh deformation but must not grow unboundedly.
        assert!(final_norm < 5.0 * initial_norm,
            "field norm grew unexpectedly: initial={initial_norm:.4e} final={final_norm:.4e}");
        assert!(final_norm > 0.0, "field collapsed to zero");
    }

    #[test]
    fn ex45_dof_count_matches_p1_h1_formula_for_multiple_meshes() {
        for &n in &[6usize, 10usize, 14usize] {
            let mesh = Mesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            assert_eq!(space.n_dofs(), (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex45_top_boundary_nodes_are_detected_for_motion() {
        let mesh = Mesh::<2>::unit_square_tri(12);
        let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
            .into_iter()
            .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
            .collect();
        assert!(!top_nodes.is_empty(), "expected non-empty top boundary node set");

        let n_top = top_nodes.len();
        let expected_min = 13usize;
        assert!(n_top >= expected_min,
            "top boundary should have at least n+1 nodes, got {n_top}");
    }

    #[test]
    fn ex45_stronger_motion_still_preserves_integral_after_correction() {
        let args = Args { n: 10, steps: 5, amp: 0.02, omega: 0.7, smooth_iters: 20 };

        let mut mesh = Mesh::<2>::unit_square_tri(args.n);
        let mut values = H1Space::new(mesh.clone(), 1).interpolate(&|x| {
            (PI * x[0]).sin() * (PI * x[1]).sin()
        });

        let mut max_abs_int_err = 0.0_f64;
        for step in 1..=args.steps {
            let t = step as f64 / args.steps as f64;
            let old_mesh = mesh.clone();
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
                .into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
                .collect();

            let shift = args.amp * (2.0 * PI * t).sin();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [shift * (PI * p[0]).sin().powi(2), 0.0]
            });

            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(
                &mut mesh,
                &fixed,
                MeshMotionConfig {
                    omega: args.omega,
                    max_iters: args.smooth_iters,
                    tol: 1.0e-12,
                },
            );

            let src = H1Space::new(old_mesh, 1);
            let dst = H1Space::new(mesh.clone(), 1);
            let (v_new, _stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
                &src,
                values.as_slice(),
                &dst,
                1.0e-12,
                4,
            )
            .expect("conservative transfer should succeed");
            max_abs_int_err = max_abs_int_err.max(report.absolute_integral_error_after);
            values = Vector::from_vec(v_new);
        }

        assert!(max_abs_int_err < 5.0e-10,
            "integral correction drift too large under stronger motion: {max_abs_int_err}");
    }

    #[test]
    fn ex45_smoothing_parameter_variation_keeps_mesh_valid() {
        for &omega in &[0.3_f64, 0.7_f64, 0.9_f64] {
            let mut mesh = Mesh::<2>::unit_square_tri(10);
            let top_nodes: Vec<u32> = all_boundary_nodes(&mesh)
                .into_iter()
                .filter(|&n| (mesh.coords_of(n)[1] - 1.0).abs() < 1.0e-12)
                .collect();
            apply_node_displacement(&mut mesh, &top_nodes, |p| {
                [0.01 * (PI * p[0]).sin().powi(2), 0.0]
            });
            let fixed = all_boundary_nodes(&mesh);
            let _ = laplacian_smooth_2d(
                &mut mesh,
                &fixed,
                MeshMotionConfig {
                    omega,
                    max_iters: 25,
                    tol: 1.0e-12,
                },
            );

            for e in 0..mesh.n_elems() as u32 {
                let nodes = mesh.elem_nodes(e);
                let p0 = mesh.coords_of(nodes[0]);
                let p1 = mesh.coords_of(nodes[1]);
                let p2 = mesh.coords_of(nodes[2]);
                let area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
                assert!(area > 0.0, "omega={omega}: inverted element {e}, area={area:.6e}");
            }
        }
    }

    #[test]
    fn ex45_template_kpi_csv_row_uses_moving_mesh_ale_contract() {
        let _guard = KPI_ENV_LOCK.lock().unwrap();
        let args = Args {
            n: 8,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 10,
        };
        let result = solve_case(&args);
        let temp_path = std::env::temp_dir().join(format!(
            "ex45_template_kpi_{}.csv",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        std::env::set_var("FEM_TEMPLATE_KPI_CSV", &temp_path);
        std::env::set_var("FEM_TEMPLATE_KPI_RUN_ID", "test");
        std::env::set_var("FEM_TEMPLATE_KPI_TAG", "unit");

        let coupling = TemplateCouplingSummary {
            steps: result.steps,
            converged_steps: result.steps,
            max_coupling_iters_used: 1,
        };
        let adaptive = TemplateAdaptiveSummary {
            sync_retries: 0,
            rejected_sync_steps: 0,
            rollback_count: 0,
        };
        maybe_write_template_kpi_csv(
            builtin_template_spec(BuiltinMultiphysicsTemplate::MovingMeshAle)
                .template
                .id(),
            coupling,
            adaptive,
            &[
                ("final_norm", result.final_norm),
                ("final_checksum", result.final_checksum),
                ("max_abs_int_err", result.max_abs_int_err),
            ],
        )
        .unwrap();

        let csv = fs::read_to_string(&temp_path).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains("moving_mesh_ale,test,unit"));
        assert!(lines[1].contains("final_norm="));
        assert!(lines[1].contains("max_abs_int_err="));

        std::env::remove_var("FEM_TEMPLATE_KPI_CSV");
        std::env::remove_var("FEM_TEMPLATE_KPI_RUN_ID");
        std::env::remove_var("FEM_TEMPLATE_KPI_TAG");
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn ex45_transient_checkpoint_roundtrip_restarts_consistently() {
        let args = Args {
            n: 8,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 10,
        };
        let full = solve_case(&args);
        let partial = solve_case_with_restart(&args, None, Some(2));
        let temp_path = std::env::temp_dir().join(format!(
            "ex45_checkpoint_roundtrip_{}.txt",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        let checkpoint = TransientCheckpointState {
            completed_steps: partial.completed_steps,
            total_steps: args.steps,
            prev_shift: partial.prev_shift,
            values: partial.values.clone(),
        };
        write_transient_checkpoint(temp_path.to_str().unwrap(), &checkpoint).unwrap();
        let restart = read_transient_checkpoint(temp_path.to_str().unwrap()).unwrap();
        let resumed = solve_case_with_restart(&args, Some(&restart), None);

        assert!((resumed.final_norm - full.final_norm).abs() < 1.0e-10,
            "restart drift in final norm: resumed={} full={}",
            resumed.final_norm,
            full.final_norm);
        assert!((resumed.final_checksum - full.final_checksum).abs() < 1.0e-8,
            "restart drift in checksum: resumed={} full={}",
            resumed.final_checksum,
            full.final_checksum);

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn ex45_vtk_export_writes_scalar_file() {
        let args = Args {
            n: 6,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 8,
        };
        let result = solve_case(&args);
        let temp_dir = std::env::temp_dir().join(format!(
            "ex45_vtk_export_{}",
            std::process::id()
        ));
        let prefix = temp_dir.join("moving_mesh_ale");
        let path = format!("{}_scalar.vtu", prefix.to_string_lossy());
        let _ = fs::remove_file(&path);

        write_ex45_vtk_export(
            prefix.to_str().unwrap(),
            &result.final_mesh,
            &result.values,
        )
        .unwrap();

        let vtk = fs::read_to_string(&path).unwrap();
        assert!(vtk.contains("transported_scalar"));
        assert!(vtk.contains("UnstructuredGrid"));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn ex45_hdf5_checkpoint_roundtrip_restarts_consistently() {
        let args = Args {
            n: 8,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 10,
        };
        let full = solve_case(&args);
        let partial = solve_case_with_restart(&args, None, Some(2));
        let temp_path = std::env::temp_dir().join(format!(
            "ex45_checkpoint_roundtrip_{}.h5",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        let checkpoint = TransientCheckpointState {
            completed_steps: partial.completed_steps,
            total_steps: args.steps,
            prev_shift: partial.prev_shift,
            values: partial.values.clone(),
        };
        write_transient_hdf5_checkpoint(temp_path.to_str().unwrap(), &checkpoint).unwrap();
        let restart = read_transient_hdf5_checkpoint(temp_path.to_str().unwrap()).unwrap();
        let resumed = solve_case_with_restart(&args, Some(&restart), None);

        assert!((resumed.final_norm - full.final_norm).abs() < 1.0e-10,
            "restart drift in final norm: resumed={} full={}",
            resumed.final_norm,
            full.final_norm);
        assert!((resumed.final_checksum - full.final_checksum).abs() < 1.0e-8,
            "restart drift in checksum: resumed={} full={}",
            resumed.final_checksum,
            full.final_checksum);

        let _ = fs::remove_file(&temp_path);
    }

    #[cfg(feature = "io_hdf5")]
    #[test]
    fn ex45_hdf5_checkpoint_writes_xdmf_sidecar() {
        let args = Args {
            n: 8,
            steps: 4,
            amp: 0.01,
            omega: 0.7,
            smooth_iters: 10,
        };
        let result = solve_case(&args);
        let temp_path = std::env::temp_dir().join(format!(
            "ex45_sidecar_roundtrip_{}.h5",
            std::process::id()
        ));
        let h5_path = temp_path.to_string_lossy().to_string();
        let _ = fs::remove_file(&temp_path);

        let checkpoint = TransientCheckpointState {
            completed_steps: result.completed_steps,
            total_steps: args.steps,
            prev_shift: result.prev_shift,
            values: result.values.clone(),
        };
        write_transient_hdf5_checkpoint(&h5_path, &checkpoint).unwrap();
        write_transient_hdf5_xdmf_sidecars(&h5_path, &checkpoint).unwrap();

        let sidecar = checkpoint_sidecar_path(&h5_path, "transported_scalar").unwrap();
        let xml = fs::read_to_string(&sidecar).unwrap();
        assert!(xml.contains("transported_scalar"));
        assert!(xml.contains("checkpoint_step_"));

        let _ = fs::remove_file(&sidecar);
        let _ = fs::remove_file(&temp_path);
    }
}
