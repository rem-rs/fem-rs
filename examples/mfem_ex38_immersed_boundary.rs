//! Example 38 immersed boundary baseline (toward MFEM ex38)
//!
//! Cut-cell subtriangulation on a background Tri mesh for a circular embedded domain.
//! This version adds a Nitsche-like weak Dirichlet treatment on the immersed boundary
//! using a chord-segment approximation per cut triangle.
//!
//! ## Level-set extension
//! A `LevelSetShape` abstraction allows arbitrary immersed-boundary geometry
//! described by a signed distance / level-set function ψ(x):
//!   - `Circle`    – ψ(x) = |x − c| − r  (active: ψ < 0, i.e. inside)
//!   - `Halfspace` – ψ(x) = n · x − d    (active: ψ < 0)
//! Edge crossings are found via linear interpolation of ψ, and the outward
//! normal at the interface is ∇ψ / ‖∇ψ‖.
//!
//! Optional workflow hooks:
//! - `--checkpoint <path>` / `--restart <path>` save and reload a lightweight
//!   text bundle containing the geometry configuration, solved field, and
//!   reported metrics.
//! - `--checkpoint-h5 <path>` / `--restart-h5 <path>` use the shared
//!   `fem-io-hdf5-parallel` checkpoint format; when built with `--features
//!   io_hdf5`, an `embedded_solution` XDMF sidecar is also emitted.
//! - `--export-vtk-prefix <prefix>` writes the final embedded scalar field as
//!   `<prefix>_embedded_solution.vtu`.

use std::{
    fs,
    io,
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
use fem_io::read_msh_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{BuiltinMultiphysicsTemplate, builtin_template_spec, solve_sparse_cholesky};
use fem_space::{constraints::apply_dirichlet, fe_space::FESpace, H1Space};

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
        .map(read_embedded_checkpoint)
        .transpose()
        .unwrap_or_else(|e| panic!("failed to read restart state: {e}"))
        .or_else(|| {
            cli.restart_h5
                .as_deref()
                .map(read_embedded_hdf5_checkpoint)
                .transpose()
                .unwrap_or_else(|e| panic!("failed to read HDF5 restart state: {e}"))
        });
    let args = restart_state
        .as_ref()
        .map(|state| state.args.clone())
        .unwrap_or_else(|| cli.sim.clone());
    let spec = builtin_template_spec(BuiltinMultiphysicsTemplate::ImmersedBoundary);
    let config_line = format!(
        "n={}, shape={}, subdiv={}, alpha={}, gamma={}",
        args.n,
        level_set_name(&args),
        args.subdiv,
        args.alpha,
        args.nitsche_gamma
    );
    print_template_header("Example 38: immersed boundary baseline", spec, &config_line);
    let result = restart_state
        .as_ref()
        .map(|state| state.result.clone())
        .unwrap_or_else(|| solve_embedded_problem(&args));
    let coupling = TemplateCouplingSummary {
        steps: 1,
        converged_steps: 1,
        max_coupling_iters_used: 1,
    };
    let adaptive = TemplateAdaptiveSummary {
        sync_retries: 0,
        rejected_sync_steps: 0,
        rollback_count: 0,
    };

    if let Some(path) = &cli.restart {
        println!("  restart loaded: {path}");
    }
    println!("  Mesh: {}x{} subdivisions, P1 elements", args.n, args.n);
    print_geometry_summary(&args);
    println!("  Subtriangulation per cut cell: {}", args.subdiv);
    println!("  Nitsche gamma: {:.3}", args.nitsche_gamma);
    println!("  Active DOFs: {}", result.active_dofs);
    println!("  Embedded area (approx): {:.6e}", result.area_estimate);
    println!("  Exact area:             {:.6e}", result.area_exact);
    println!("  Relative area error:    {:.3e}", result.area_rel_error);
    println!("  Interface length (chord approx): {:.6e}", result.interface_length);
    println!("  Embedded L2 error:      {:.3e}", result.l2_error);
    println!("  Interface L2 error:     {:.3e}", result.boundary_l2_error);
    println!("  Value range on active set: [{:.6}, {:.6}]", result.min_u, result.max_u);
    print_template_coupling_summary(coupling);
    print_template_adaptive_summary(adaptive);
    if let Err(e) = maybe_write_template_kpi_csv(
        spec.template.id(),
        coupling,
        adaptive,
        &[
            ("area_rel_error", result.area_rel_error),
            ("l2_error", result.l2_error),
            ("boundary_l2_error", result.boundary_l2_error),
            ("interface_length", result.interface_length),
        ],
    ) {
        eprintln!("warning: failed to append template KPI CSV: {e}");
    }

    if let Some(path) = &cli.checkpoint {
        let checkpoint = EmbeddedCheckpointState {
            args: args.clone(),
            result: result.clone(),
        };
        if let Err(e) = write_embedded_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write checkpoint: {e}");
        } else {
            println!("  checkpoint written: {path}");
        }
    }

    if let Some(path) = &cli.checkpoint_h5 {
        let checkpoint = EmbeddedCheckpointState {
            args: args.clone(),
            result: result.clone(),
        };
        if let Err(e) = write_embedded_hdf5_checkpoint(path, &checkpoint) {
            eprintln!("warning: failed to write HDF5 checkpoint: {e}");
        } else {
            println!("  HDF5 checkpoint written: {path}");
            #[cfg(feature = "io_hdf5")]
            if let Err(e) = write_embedded_hdf5_xdmf_sidecars(path, &checkpoint) {
                eprintln!("warning: failed to write checkpoint XDMF sidecars: {e}");
            }
        }
    }

    if let Some(prefix) = &cli.export_vtk_prefix {
        if let Err(e) = write_ex38_vtk_export(prefix, &result.mesh, &result.values) {
            eprintln!("warning: failed to write VTK export: {e}");
        } else {
            println!("  VTK export written: {prefix}_embedded_solution.vtu");
        }
    }

    println!();
    println!("Note: cut-cell baseline now includes a Nitsche-like weak immersed boundary treatment.");
}

fn level_set_name(args: &Args) -> &'static str {
    match args.level_set {
        Some(LevelSetShape::Halfspace { .. }) => "halfspace",
        _ => "circle",
    }
}

fn print_geometry_summary(args: &Args) {
    match &args.level_set {
        Some(LevelSetShape::Halfspace { normal, offset }) => {
            println!(
                "  Halfspace normal: ({:.3}, {:.3}), offset = {:.3}",
                normal[0], normal[1], offset
            );
        }
        _ => {
            println!(
                "  Circle center: ({:.3}, {:.3}), radius = {:.3}",
                args.cx, args.cy, args.radius
            );
        }
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    sim: Args,
    checkpoint: Option<String>,
    checkpoint_h5: Option<String>,
    restart: Option<String>,
    restart_h5: Option<String>,
    export_vtk_prefix: Option<String>,
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    radius: f64,
    cx: f64,
    cy: f64,
    alpha: f64,
    subdiv: usize,
    nitsche_gamma: f64,
    /// Level-set shape override; if None, uses the `Circle` built from cx/cy/radius.
    level_set: Option<LevelSetShape>,
    /// Optional mesh file; if None, uses unit_square_tri(n).
    mesh_file: Option<String>,
}

#[derive(Debug, Clone)]
struct EmbeddedResult {
    active_dofs: usize,
    area_estimate: f64,
    area_exact: f64,
    area_rel_error: f64,
    interface_length: f64,
    l2_error: f64,
    boundary_l2_error: f64,
    min_u: f64,
    max_u: f64,
    values: Vec<f64>,
    mesh: SimplexMesh<2>,
}

#[derive(Debug, Clone)]
struct EmbeddedCheckpointState {
    args: Args,
    result: EmbeddedResult,
}

#[derive(Debug, Clone)]
struct Circle {
    cx: f64,
    cy: f64,
    radius: f64,
}

// ── Level-set interface geometry ─────────────────────────────────────────────

/// Generic signed-distance / level-set description of an immersed interface.
///
/// Convention: **active (interior) region = ψ(x) < 0**.
#[derive(Debug, Clone)]
enum LevelSetShape {
    /// Circular interface: ψ(x) = ‖x − c‖ − r.
    Circle(Circle),
    /// Half-space interface: ψ(x) = n · x − d  (unit outward normal `n`).
    Halfspace { normal: [f64; 2], offset: f64 },
}

impl LevelSetShape {
    /// Evaluate ψ(x).  Negative = inside (active domain).
    fn eval(&self, x: [f64; 2]) -> f64 {
        match self {
            LevelSetShape::Circle(c) => {
                let dx = x[0] - c.cx;
                let dy = x[1] - c.cy;
                (dx * dx + dy * dy).sqrt() - c.radius
            }
            LevelSetShape::Halfspace { normal, offset } => {
                normal[0] * x[0] + normal[1] * x[1] - offset
            }
        }
    }

    /// Outward unit normal at point `x` on the interface.
    fn outward_normal(&self, x: [f64; 2]) -> [f64; 2] {
        match self {
            LevelSetShape::Circle(c) => {
                let dx = x[0] - c.cx;
                let dy = x[1] - c.cy;
                let inv = 1.0 / (dx * dx + dy * dy).sqrt().max(1.0e-14);
                [dx * inv, dy * inv]
            }
            LevelSetShape::Halfspace { normal, .. } => *normal,
        }
    }

    /// True when x is in the active (interior) domain.
    fn is_active(&self, x: [f64; 2]) -> bool {
        self.eval(x) < 0.0
    }
}

/// Find the chord (midpoint, length) where the triangle (x0, x1, x2) crosses
/// the zero-level-set of `ls`, using linear interpolation of ψ along each edge.
fn triangle_level_set_chord(
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    ls: &LevelSetShape,
) -> Option<([f64; 2], f64)> {
    let mut pts = Vec::<[f64; 2]>::new();
    edge_ls_intersections(x0, x1, ls, &mut pts);
    edge_ls_intersections(x1, x2, ls, &mut pts);
    edge_ls_intersections(x2, x0, ls, &mut pts);
    dedup_points(&mut pts, 1.0e-10);
    if pts.len() < 2 {
        return None;
    }
    let p0 = pts[0];
    let p1 = pts[1];
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0e-12 {
        return None;
    }
    let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
    Some((mid, len))
}

/// Append edge–level-set intersection points to `out` using linear interpolation.
fn edge_ls_intersections(
    a: [f64; 2],
    b: [f64; 2],
    ls: &LevelSetShape,
    out: &mut Vec<[f64; 2]>,
) {
    let pa = ls.eval(a);
    let pb = ls.eval(b);
    if pa * pb < 0.0 {
        // Linear interpolation: find t ∈ (0,1) s.t. ψ(a + t(b−a)) = 0
        let t = pa / (pa - pb);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    // Exact zero at endpoint — handled by dedup
    if pa.abs() < 1.0e-14 {
        out.push(a);
    }
    if pb.abs() < 1.0e-14 {
        out.push(b);
    }
}

fn parse_args() -> CliArgs {
    let mut sim = Args {
        n: 18,
        radius: 0.30,
        cx: 0.5,
        cy: 0.5,
        alpha: 20.0,
        subdiv: 8,
        nitsche_gamma: 20.0,
        level_set: None,
        mesh_file: None,
    };
    let mut workflow = WorkflowCliOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if workflow.try_parse_arg(arg.as_str(), &mut it) {
            continue;
        }
        match arg.as_str() {
            "-m" | "--mesh" => sim.mesh_file = Some(it.next().unwrap_or("".into())),
            "--n" => sim.n = it.next().unwrap_or("18".into()).parse().unwrap_or(18),
            "--radius" => sim.radius = it.next().unwrap_or("0.30".into()).parse().unwrap_or(0.30),
            "--cx" => sim.cx = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "--cy" => sim.cy = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "--alpha" => sim.alpha = it.next().unwrap_or("20.0".into()).parse().unwrap_or(20.0),
            "--subdiv" => sim.subdiv = it.next().unwrap_or("8".into()).parse().unwrap_or(8),
            "--nitsche-gamma" => {
                sim.nitsche_gamma = it.next().unwrap_or("20.0".into()).parse().unwrap_or(20.0)
            }
            "--level-set" => {
                sim.level_set = match it.next().as_deref() {
                    Some("halfspace") => Some(LevelSetShape::Halfspace {
                        normal: [0.0, 1.0],
                        offset: 0.5,
                    }),
                    _ => None, // default: use Circle from cx/cy/radius
                };
            }
            _ => {}
        }
    }
    sim.radius = sim.radius.clamp(0.05, 0.45);
    sim.alpha = sim.alpha.max(1.0e-6);
    sim.subdiv = sim.subdiv.max(1);
    sim.nitsche_gamma = sim.nitsche_gamma.max(1.0e-6);
    CliArgs {
        sim,
        checkpoint: workflow.checkpoint,
        checkpoint_h5: workflow.checkpoint_h5,
        restart: workflow.restart,
        restart_h5: workflow.restart_h5,
        export_vtk_prefix: workflow.export_vtk_prefix,
    }
}

fn solve_embedded_problem(args: &Args) -> EmbeddedResult {
    let mesh = match args.mesh_file {
        Some(ref p) => {
            let msh = read_msh_file(p).expect("failed to read mesh file");
            msh.into_2d().expect("expected 2D mesh")
        }
        None => SimplexMesh::<2>::unit_square_tri(args.n),
    };
    let space = H1Space::new(mesh.clone(), 1);
    let ls = args.level_set.clone().unwrap_or_else(|| {
        LevelSetShape::Circle(Circle {
            cx: args.cx,
            cy: args.cy,
            radius: args.radius,
        })
    });

    let (mut mat, mut rhs, active_mask, area_estimate, interface_length) =
        assemble_embedded_system(&space, &ls, args.alpha, args.subdiv, args.nitsche_gamma);

    let inactive_dofs: Vec<u32> = active_mask
        .iter()
        .enumerate()
        .filter_map(|(i, active)| if *active { None } else { Some(i as u32) })
        .collect();
    if !inactive_dofs.is_empty() {
        apply_dirichlet(
            &mut mat,
            &mut rhs,
            &inactive_dofs,
            &vec![0.0; inactive_dofs.len()],
        );
    }

    let solution = solve_sparse_cholesky(&mat, &rhs).expect("embedded ex38 Cholesky solve failed");

    let (l2_error, boundary_l2_error, min_u, max_u) =
        embedded_solution_metrics(&space, &solution, &ls, args.subdiv);

    let area_exact = match &ls {
        LevelSetShape::Circle(c) => std::f64::consts::PI * c.radius * c.radius,
        LevelSetShape::Halfspace { normal, offset } => {
            // area of the unit square on the active side (ψ < 0)
            // For n·x = d with unit square [0,1]², the active area is
            // the integral of 1 over {x ∈ [0,1]² : n·x < d}.
            // For the canonical horizontal cut (n=[0,1], d=offset):
            let hy = offset.clamp(0.0, 1.0);
            let nx = normal[0];
            let ny = normal[1];
            // Approximate: use hy if purely vertical normal
            if nx.abs() < 1.0e-12 && ny.abs() > 0.5 {
                hy
            } else if ny.abs() < 1.0e-12 && nx.abs() > 0.5 {
                offset.clamp(0.0, 1.0)
            } else {
                // Fallback: use the area estimate itself (can't compute analytically for arbitrary n)
                area_estimate
            }
        }
    };

    let area_rel_error = ((area_estimate - area_exact) / area_exact.max(1.0e-14)).abs();
    let active_dofs = active_mask.iter().filter(|flag| **flag).count();

    EmbeddedResult {
        active_dofs,
        area_estimate,
        area_exact,
        area_rel_error,
        interface_length,
        l2_error,
        boundary_l2_error,
        min_u,
        max_u,
        values: solution,
        mesh,
    }
}

fn write_ex38_vtk_export(prefix: &str, mesh: &SimplexMesh<2>, values: &[f64]) -> Result<(), String> {
    let path = format!("{prefix}_embedded_solution.vtu");
    ensure_parent_dir(&path).map_err(|e| e.to_string())?;
    let mut writer = VtkWriter::new(mesh);
    writer.add_point_data(DataArray::scalars("embedded_solution", values.to_vec()));
    writer.write_file(&path).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_embedded_checkpoint(path: &str, state: &EmbeddedCheckpointState) -> io::Result<()> {
    ensure_parent_dir(path)?;
    let values = format_vec_f64(&state.result.values);
    let content = format!(
        "format=ex38_immersed_boundary_v1\nlevel_set={}\nn={}\nradius={:.17e}\ncx={:.17e}\ncy={:.17e}\nalpha={:.17e}\nsubdiv={}\nnitsche_gamma={:.17e}\nactive_dofs={}\narea_estimate={:.17e}\narea_exact={:.17e}\narea_rel_error={:.17e}\ninterface_length={:.17e}\nl2_error={:.17e}\nboundary_l2_error={:.17e}\nmin_u={:.17e}\nmax_u={:.17e}\nvalues={}\n",
        level_set_name(&state.args),
        state.args.n,
        state.args.radius,
        state.args.cx,
        state.args.cy,
        state.args.alpha,
        state.args.subdiv,
        state.args.nitsche_gamma,
        state.result.active_dofs,
        state.result.area_estimate,
        state.result.area_exact,
        state.result.area_rel_error,
        state.result.interface_length,
        state.result.l2_error,
        state.result.boundary_l2_error,
        state.result.min_u,
        state.result.max_u,
        values,
    );
    fs::write(path, content)
}

fn read_embedded_checkpoint(path: &str) -> Result<EmbeddedCheckpointState, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut format = None;
    let mut level_set = None;
    let mut n = None;
    let mut radius = None;
    let mut cx = None;
    let mut cy = None;
    let mut alpha = None;
    let mut subdiv = None;
    let mut nitsche_gamma = None;
    let mut active_dofs = None;
    let mut area_estimate = None;
    let mut area_exact = None;
    let mut area_rel_error = None;
    let mut interface_length = None;
    let mut l2_error = None;
    let mut boundary_l2_error = None;
    let mut min_u = None;
    let mut max_u = None;
    let mut values = None;

    for line in content.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "format" => format = Some(value.to_string()),
            "level_set" => level_set = Some(value.to_string()),
            "n" => n = value.parse::<usize>().ok(),
            "radius" => radius = value.parse::<f64>().ok(),
            "cx" => cx = value.parse::<f64>().ok(),
            "cy" => cy = value.parse::<f64>().ok(),
            "alpha" => alpha = value.parse::<f64>().ok(),
            "subdiv" => subdiv = value.parse::<usize>().ok(),
            "nitsche_gamma" => nitsche_gamma = value.parse::<f64>().ok(),
            "active_dofs" => active_dofs = value.parse::<usize>().ok(),
            "area_estimate" => area_estimate = value.parse::<f64>().ok(),
            "area_exact" => area_exact = value.parse::<f64>().ok(),
            "area_rel_error" => area_rel_error = value.parse::<f64>().ok(),
            "interface_length" => interface_length = value.parse::<f64>().ok(),
            "l2_error" => l2_error = value.parse::<f64>().ok(),
            "boundary_l2_error" => boundary_l2_error = value.parse::<f64>().ok(),
            "min_u" => min_u = value.parse::<f64>().ok(),
            "max_u" => max_u = value.parse::<f64>().ok(),
            "values" => values = Some(parse_vec_f64(value)?),
            _ => {}
        }
    }

    if format.as_deref() != Some("ex38_immersed_boundary_v1") {
        return Err("unsupported checkpoint format".into());
    }

    let args = Args {
        n: n.ok_or_else(|| "missing n".to_string())?,
        radius: radius.ok_or_else(|| "missing radius".to_string())?,
        cx: cx.ok_or_else(|| "missing cx".to_string())?,
        cy: cy.ok_or_else(|| "missing cy".to_string())?,
        alpha: alpha.ok_or_else(|| "missing alpha".to_string())?,
        subdiv: subdiv.ok_or_else(|| "missing subdiv".to_string())?,
        nitsche_gamma: nitsche_gamma.ok_or_else(|| "missing nitsche_gamma".to_string())?,
        level_set: match level_set.as_deref() {
            Some("halfspace") => Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.5,
            }),
            _ => None,
        },
        mesh_file: None,
    };
    let values = values.ok_or_else(|| "missing values".to_string())?;
    let expected_dofs = (args.n + 1) * (args.n + 1);
    if values.len() != expected_dofs {
        return Err(format!(
            "checkpoint values length ({}) does not match expected dofs ({expected_dofs})",
            values.len()
        ));
    }

    Ok(EmbeddedCheckpointState {
        result: EmbeddedResult {
            active_dofs: active_dofs.ok_or_else(|| "missing active_dofs".to_string())?,
            area_estimate: area_estimate.ok_or_else(|| "missing area_estimate".to_string())?,
            area_exact: area_exact.ok_or_else(|| "missing area_exact".to_string())?,
            area_rel_error: area_rel_error.ok_or_else(|| "missing area_rel_error".to_string())?,
            interface_length: interface_length.ok_or_else(|| "missing interface_length".to_string())?,
            l2_error: l2_error.ok_or_else(|| "missing l2_error".to_string())?,
            boundary_l2_error: boundary_l2_error.ok_or_else(|| "missing boundary_l2_error".to_string())?,
            min_u: min_u.ok_or_else(|| "missing min_u".to_string())?,
            max_u: max_u.ok_or_else(|| "missing max_u".to_string())?,
            values,
            mesh: SimplexMesh::<2>::unit_square_tri(args.n),
        },
        args,
    })
}

fn write_embedded_hdf5_checkpoint(path: &str, state: &EmbeddedCheckpointState) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(path);

    let bundle = CheckpointBundleF64 {
        mesh_meta: None,
        fields: vec![
            scalar_rank_field_f64("level_set_code", level_set_code(&state.args) as f64),
            scalar_rank_field_f64("n", state.args.n as f64),
            scalar_rank_field_f64("radius", state.args.radius),
            scalar_rank_field_f64("cx", state.args.cx),
            scalar_rank_field_f64("cy", state.args.cy),
            scalar_rank_field_f64("alpha", state.args.alpha),
            scalar_rank_field_f64("subdiv", state.args.subdiv as f64),
            scalar_rank_field_f64("nitsche_gamma", state.args.nitsche_gamma),
            scalar_rank_field_f64("active_dofs", state.result.active_dofs as f64),
            scalar_rank_field_f64("area_estimate", state.result.area_estimate),
            scalar_rank_field_f64("area_exact", state.result.area_exact),
            scalar_rank_field_f64("area_rel_error", state.result.area_rel_error),
            scalar_rank_field_f64("interface_length", state.result.interface_length),
            scalar_rank_field_f64("l2_error", state.result.l2_error),
            scalar_rank_field_f64("boundary_l2_error", state.result.boundary_l2_error),
            scalar_rank_field_f64("min_u", state.result.min_u),
            scalar_rank_field_f64("max_u", state.result.max_u),
            vector_rank_field_f64("embedded_solution", state.result.values.clone()),
        ],
    };
    let cfg = ParallelIoConfig { world_size: 1, rank: 0 };
    write_checkpoint_step_bundle_f64(path, cfg, 1, 1.0, &bundle, IoBackend::Partitioned)
        .map_err(|e| e.to_string())?;
    validate_checkpoint_layout(path, Some(1)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_embedded_hdf5_checkpoint(path: &str) -> Result<EmbeddedCheckpointState, String> {
    let fields = read_checkpoint_fields_f64_latest(
        path,
        ParallelIoConfig { world_size: 1, rank: 0 },
        &[
            "level_set_code",
            "n",
            "radius",
            "cx",
            "cy",
            "alpha",
            "subdiv",
            "nitsche_gamma",
            "active_dofs",
            "area_estimate",
            "area_exact",
            "area_rel_error",
            "interface_length",
            "l2_error",
            "boundary_l2_error",
            "min_u",
            "max_u",
            "embedded_solution",
        ],
    )
    .map_err(|e| e.to_string())?;

    let mut level_set_code_value = None;
    let mut n = None;
    let mut radius = None;
    let mut cx = None;
    let mut cy = None;
    let mut alpha = None;
    let mut subdiv = None;
    let mut nitsche_gamma = None;
    let mut active_dofs = None;
    let mut area_estimate = None;
    let mut area_exact = None;
    let mut area_rel_error = None;
    let mut interface_length = None;
    let mut l2_error = None;
    let mut boundary_l2_error = None;
    let mut min_u = None;
    let mut max_u = None;
    let mut values = None;

    for (name, field) in fields {
        match name.as_str() {
            "level_set_code" => level_set_code_value = field.values.first().copied(),
            "n" => n = field.values.first().map(|v| *v as usize),
            "radius" => radius = field.values.first().copied(),
            "cx" => cx = field.values.first().copied(),
            "cy" => cy = field.values.first().copied(),
            "alpha" => alpha = field.values.first().copied(),
            "subdiv" => subdiv = field.values.first().map(|v| *v as usize),
            "nitsche_gamma" => nitsche_gamma = field.values.first().copied(),
            "active_dofs" => active_dofs = field.values.first().map(|v| *v as usize),
            "area_estimate" => area_estimate = field.values.first().copied(),
            "area_exact" => area_exact = field.values.first().copied(),
            "area_rel_error" => area_rel_error = field.values.first().copied(),
            "interface_length" => interface_length = field.values.first().copied(),
            "l2_error" => l2_error = field.values.first().copied(),
            "boundary_l2_error" => boundary_l2_error = field.values.first().copied(),
            "min_u" => min_u = field.values.first().copied(),
            "max_u" => max_u = field.values.first().copied(),
            "embedded_solution" => values = Some(field.values),
            _ => {}
        }
    }

    let n = n.ok_or_else(|| "missing n".to_string())?;
    let args = Args {
        n,
        radius: radius.ok_or_else(|| "missing radius".to_string())?,
        cx: cx.ok_or_else(|| "missing cx".to_string())?,
        cy: cy.ok_or_else(|| "missing cy".to_string())?,
        alpha: alpha.ok_or_else(|| "missing alpha".to_string())?,
        subdiv: subdiv.ok_or_else(|| "missing subdiv".to_string())?,
        nitsche_gamma: nitsche_gamma.ok_or_else(|| "missing nitsche_gamma".to_string())?,
        level_set: decode_level_set(level_set_code_value.ok_or_else(|| "missing level_set_code".to_string())?),
        mesh_file: None,
    };
    let values = values.ok_or_else(|| "missing embedded_solution".to_string())?;
    let expected_dofs = (n + 1) * (n + 1);
    if values.len() != expected_dofs {
        return Err(format!(
            "checkpoint values length ({}) does not match expected dofs ({expected_dofs})",
            values.len()
        ));
    }

    Ok(EmbeddedCheckpointState {
        args,
        result: EmbeddedResult {
            active_dofs: active_dofs.ok_or_else(|| "missing active_dofs".to_string())?,
            area_estimate: area_estimate.ok_or_else(|| "missing area_estimate".to_string())?,
            area_exact: area_exact.ok_or_else(|| "missing area_exact".to_string())?,
            area_rel_error: area_rel_error.ok_or_else(|| "missing area_rel_error".to_string())?,
            interface_length: interface_length.ok_or_else(|| "missing interface_length".to_string())?,
            l2_error: l2_error.ok_or_else(|| "missing l2_error".to_string())?,
            boundary_l2_error: boundary_l2_error.ok_or_else(|| "missing boundary_l2_error".to_string())?,
            min_u: min_u.ok_or_else(|| "missing min_u".to_string())?,
            max_u: max_u.ok_or_else(|| "missing max_u".to_string())?,
            values,
            mesh: SimplexMesh::<2>::unit_square_tri(n),
        },
    })
}

fn level_set_code(args: &Args) -> usize {
    match args.level_set {
        Some(LevelSetShape::Halfspace { .. }) => 1,
        _ => 0,
    }
}

fn decode_level_set(code: f64) -> Option<LevelSetShape> {
    if code.round() as usize == 1 {
        Some(LevelSetShape::Halfspace {
            normal: [0.0, 1.0],
            offset: 0.5,
        })
    } else {
        None
    }
}

#[cfg(feature = "io_hdf5")]
fn write_embedded_hdf5_xdmf_sidecars(h5_path: &str, state: &EmbeddedCheckpointState) -> Result<(), String> {
    write_scalar_checkpoint_xdmf_sidecars(h5_path, 1, state.result.area_estimate, &["embedded_solution"])
}

fn assemble_embedded_system(
    space: &H1Space<SimplexMesh<2>>,
    ls: &LevelSetShape,
    alpha: f64,
    subdiv: usize,
    nitsche_gamma: f64,
) -> (CsrMatrix<f64>, Vec<f64>, Vec<bool>, f64, f64) {
    let mesh = space.mesh();
    let ndofs = space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(ndofs, ndofs);
    for i in 0..ndofs {
        coo.add(i, i, 0.0);
    }
    let mut rhs = vec![0.0_f64; ndofs];
    let mut active_dofs = vec![false; ndofs];
    let mut total_area = 0.0_f64;
    let mut total_interface_length = 0.0_f64;

    for elem in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(elem);
        let dofs = space.element_dofs(elem);

        let x0 = to_point(mesh.node_coords(nodes[0]));
        let x1 = to_point(mesh.node_coords(nodes[1]));
        let x2 = to_point(mesh.node_coords(nodes[2]));

        let grad = parent_gradients(x0, x1, x2);
        let area_parent =
            0.5 * ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();
        let h = (2.0 * area_parent).sqrt().max(1.0e-12);

        for (sub_centroid, sub_area, phi) in subdivided_triangle_samples(x0, x1, x2, subdiv) {
            if !ls.is_active(sub_centroid) {
                continue;
            }
            total_area += sub_area;
            for &dof in dofs {
                active_dofs[dof as usize] = true;
            }

            for i in 0..3 {
                rhs[dofs[i] as usize] += alpha * sub_area * phi[i];
                for j in 0..3 {
                    let stiffness = (grad[i][0] * grad[j][0] + grad[i][1] * grad[j][1]) * sub_area;
                    let mass = alpha * sub_area * phi[i] * phi[j];
                    coo.add(dofs[i] as usize, dofs[j] as usize, stiffness + mass);
                }
            }
        }

        // Nitsche weak Dirichlet on the immersed interface (level-set chord)
        // Only apply for triangles on the ACTIVE side of the interface (ψ(centroid) < 0),
        // to avoid double-counting when the interface coincides with mesh edges.
        let centroid = [
            (x0[0] + x1[0] + x2[0]) / 3.0,
            (x0[1] + x1[1] + x2[1]) / 3.0,
        ];
        if ls.eval(centroid) < 0.0 {
            if let Some((mid, seg_len)) = triangle_level_set_chord(x0, x1, x2, ls) {
                total_interface_length += seg_len;
                let phi_mid = barycentric_shape(mid, x0, x1, x2);
                let normal = ls.outward_normal(mid);
                let penalty = nitsche_gamma / h;
                let g = 1.0_f64;

                for i in 0..3 {
                    let gi = dofs[i] as usize;
                    let dni = normal[0] * grad[i][0] + normal[1] * grad[i][1];
                    rhs[gi] += seg_len * (-dni * g + penalty * phi_mid[i] * g);

                    for j in 0..3 {
                        let gj = dofs[j] as usize;
                        let dnj = normal[0] * grad[j][0] + normal[1] * grad[j][1];
                        let aij = seg_len
                            * (-dni * phi_mid[j] - dnj * phi_mid[i]
                                + penalty * phi_mid[i] * phi_mid[j]);
                        coo.add(gi, gj, aij);
                    }
                }
            }
        }
    }

    (
        coo.into_csr(),
        rhs,
        active_dofs,
        total_area,
        total_interface_length,
    )
}

fn embedded_solution_metrics(
    space: &H1Space<SimplexMesh<2>>,
    u: &[f64],
    ls: &LevelSetShape,
    subdiv: usize,
) -> (f64, f64, f64, f64) {
    let mesh = space.mesh();
    let mut err2 = 0.0_f64;
    let mut area = 0.0_f64;
    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;

    let mut bnd_err2 = 0.0_f64;
    let mut bnd_len = 0.0_f64;

    for elem in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(elem);
        let dofs = space.element_dofs(elem);

        let x0 = to_point(mesh.node_coords(nodes[0]));
        let x1 = to_point(mesh.node_coords(nodes[1]));
        let x2 = to_point(mesh.node_coords(nodes[2]));

        for (sub_centroid, sub_area, phi) in subdivided_triangle_samples(x0, x1, x2, subdiv) {
            if !ls.is_active(sub_centroid) {
                continue;
            }
            let uh = phi[0] * u[dofs[0] as usize]
                + phi[1] * u[dofs[1] as usize]
                + phi[2] * u[dofs[2] as usize];
            err2 += sub_area * (uh - 1.0) * (uh - 1.0);
            area += sub_area;
            min_u = min_u.min(uh);
            max_u = max_u.max(uh);
        }

        if let Some((mid, seg_len)) = triangle_level_set_chord(x0, x1, x2, ls) {
            let phi_mid = barycentric_shape(mid, x0, x1, x2);
            let uh_mid = phi_mid[0] * u[dofs[0] as usize]
                + phi_mid[1] * u[dofs[1] as usize]
                + phi_mid[2] * u[dofs[2] as usize];
            bnd_err2 += seg_len * (uh_mid - 1.0) * (uh_mid - 1.0);
            bnd_len += seg_len;
        }
    }

    if !min_u.is_finite() {
        min_u = 0.0;
        max_u = 0.0;
    }

    let l2 = (err2 / area.max(1.0e-14)).sqrt();
    let bnd_l2 = (bnd_err2 / bnd_len.max(1.0e-14)).sqrt();
    (l2, bnd_l2, min_u, max_u)
}

fn subdivided_triangle_samples(
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    subdiv: usize,
) -> Vec<([f64; 2], f64, [f64; 3])> {
    let mut out = Vec::new();
    let h = 1.0 / subdiv as f64;

    for i in 0..subdiv {
        for j in 0..(subdiv - i) {
            let p00 = [i as f64 * h, j as f64 * h];
            let p10 = [(i + 1) as f64 * h, j as f64 * h];
            let p01 = [i as f64 * h, (j + 1) as f64 * h];
            add_subtriangle_sample(&mut out, x0, x1, x2, p00, p10, p01);

            if i + j + 1 < subdiv {
                let p11 = [(i + 1) as f64 * h, (j + 1) as f64 * h];
                add_subtriangle_sample(&mut out, x0, x1, x2, p10, p11, p01);
            }
        }
    }

    out
}

fn add_subtriangle_sample(
    out: &mut Vec<([f64; 2], f64, [f64; 3])>,
    x0: [f64; 2],
    x1: [f64; 2],
    x2: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
) {
    let centroid_ref = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
    let centroid_phys = map_to_phys(x0, x1, x2, centroid_ref);
    let phi = [
        1.0 - centroid_ref[0] - centroid_ref[1],
        centroid_ref[0],
        centroid_ref[1],
    ];

    let pa = map_to_phys(x0, x1, x2, a);
    let pb = map_to_phys(x0, x1, x2, b);
    let pc = map_to_phys(x0, x1, x2, c);
    let area =
        0.5 * ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])).abs();

    out.push((centroid_phys, area, phi));
}

fn parent_gradients(x0: [f64; 2], x1: [f64; 2], x2: [f64; 2]) -> [[f64; 2]; 3] {
    let two_area = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0]);
    let inv_two_area = 1.0 / two_area;
    [
        [
            (x1[1] - x2[1]) * inv_two_area,
            (x2[0] - x1[0]) * inv_two_area,
        ],
        [
            (x2[1] - x0[1]) * inv_two_area,
            (x0[0] - x2[0]) * inv_two_area,
        ],
        [
            (x0[1] - x1[1]) * inv_two_area,
            (x1[0] - x0[0]) * inv_two_area,
        ],
    ]
}

fn map_to_phys(x0: [f64; 2], x1: [f64; 2], x2: [f64; 2], xi: [f64; 2]) -> [f64; 2] {
    [
        x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
        x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
    ]
}

fn barycentric_shape(x: [f64; 2], x0: [f64; 2], x1: [f64; 2], x2: [f64; 2]) -> [f64; 3] {
    let det = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0]);
    let l1 = ((x1[0] - x[0]) * (x2[1] - x[1]) - (x1[1] - x[1]) * (x2[0] - x[0])) / det;
    let l2 = ((x2[0] - x[0]) * (x0[1] - x[1]) - (x2[1] - x[1]) * (x0[0] - x[0])) / det;
    let l3 = 1.0 - l1 - l2;
    [l1, l2, l3]
}

fn dedup_points(pts: &mut Vec<[f64; 2]>, eps: f64) {
    let mut out = Vec::<[f64; 2]>::new();
    'outer: for p in pts.iter().copied() {
        for q in &out {
            let dx = p[0] - q[0];
            let dy = p[1] - q[1];
            if dx * dx + dy * dy <= eps * eps {
                continue 'outer;
            }
        }
        out.push(p);
    }
    *pts = out;
}

fn to_point(x: &[f64]) -> [f64; 2] {
    [x[0], x[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, sync::Mutex};

    static KPI_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn temp_output_path(tag: &str, ext: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "ex38_{}_{}_{}.{}",
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

    #[test]
    fn ex38_embedded_area_is_reasonable() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None, mesh_file: None,
        });
        assert!(result.area_rel_error < 3.0e-2, "area rel error = {}", result.area_rel_error);
        assert!(result.active_dofs > 0, "expected non-empty active set");
        assert!(
            result.interface_length > 1.0,
            "interface length too small: {}",
            result.interface_length
        );
    }

    #[test]
    fn ex38_embedded_solution_recovers_constant_state() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None, mesh_file: None,
        });
        assert!(result.l2_error < 6.0e-2, "embedded L2 error = {}", result.l2_error);
        assert!(
            result.boundary_l2_error < 1.0e-1,
            "embedded boundary L2 error = {}",
            result.boundary_l2_error
        );
        assert!(result.min_u > 0.75, "min_u = {}", result.min_u);
        assert!(result.max_u < 1.15, "max_u = {}", result.max_u);
    }

    #[test]
    fn ex38_finer_cut_cell_subdivision_improves_area_accuracy() {
        let coarse = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 4,
            nitsche_gamma: 20.0,
            level_set: None, mesh_file: None,
        });
        let fine = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None, mesh_file: None,
        });

        assert!(
            fine.area_rel_error < coarse.area_rel_error,
            "expected finer cut-cell subdivision to improve area accuracy: coarse={} fine={}",
            coarse.area_rel_error,
            fine.area_rel_error
        );
        assert!(fine.area_rel_error < 1.0e-3, "fine subdivision area error too large: {}", fine.area_rel_error);
    }

    #[test]
    fn ex38_nitsche_gamma_variation_preserves_constant_solution() {
        let weak = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 10.0,
            level_set: None, mesh_file: None,
        });
        let strong = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 40.0,
            level_set: None, mesh_file: None,
        });

        for (label, result) in [("weak", &weak), ("strong", &strong)] {
            assert!(result.l2_error < 1.0e-12, "{label} gamma embedded L2 error = {}", result.l2_error);
            assert!(result.boundary_l2_error < 1.0e-12, "{label} gamma boundary L2 error = {}", result.boundary_l2_error);
            assert!((result.min_u - 1.0).abs() < 1.0e-12, "{label} gamma min_u = {}", result.min_u);
            assert!((result.max_u - 1.0).abs() < 1.0e-12, "{label} gamma max_u = {}", result.max_u);
        }

        assert!(
            (weak.area_rel_error - strong.area_rel_error).abs() < 1.0e-12,
            "area estimate should be gamma-independent for fixed geometry: weak={} strong={}",
            weak.area_rel_error,
            strong.area_rel_error
        );
        assert!(
            (weak.interface_length - strong.interface_length).abs() < 1.0e-12,
            "interface length should be gamma-independent for fixed geometry: weak={} strong={}",
            weak.interface_length,
            strong.interface_length
        );
    }

    // ── Level-set halfspace tests ─────────────────────────────────────────────

    /// Horizontal cut at y = 0.5: active region = lower half [0,1] × [0, 0.5].
    /// Exact area = 0.5; the chord approximation is exact for straight interfaces.
    #[test]
    fn ex38_levelset_halfspace_area_matches_exact() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.5,
            }),
        });
        // Straight interface → area estimate should be very accurate
        assert!(
            result.area_rel_error < 1.0e-10,
            "halfspace area rel error = {}",
            result.area_rel_error
        );
        assert!(result.active_dofs > 0, "expected non-empty active set");
        // Interface is the line y=0.5 on the unit square; exact length = 1.0
        assert!(
            (result.interface_length - 1.0).abs() < 1.0e-2,
            "halfspace interface length = {}",
            result.interface_length
        );
    }

    /// The halfspace level-set should still recover u ≈ 1 in the active domain
    /// (same forcing and boundary data as the circle test).
    #[test]
    fn ex38_levelset_halfspace_recovers_constant_state() {
        let result = solve_embedded_problem(&Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.5,
            }),
        });
        assert!(
            result.l2_error < 1.5e-1,
            "halfspace embedded L2 error = {}",
            result.l2_error
        );
        assert!(result.min_u > 0.5, "halfspace min_u = {}", result.min_u);
        assert!(result.max_u < 1.2, "halfspace max_u = {}", result.max_u);
    }

    /// Varying γ for the halfspace should not affect the cut area estimate.
    #[test]
    fn ex38_levelset_halfspace_area_is_gamma_independent() {
        let args_base = Args {
            n: 14,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 6,
            nitsche_gamma: 10.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.4,
            }),
        };
        let weak   = solve_embedded_problem(&Args { nitsche_gamma: 10.0, ..args_base.clone() });
        let strong = solve_embedded_problem(&Args { nitsche_gamma: 50.0, ..args_base.clone() });

        assert!(
            (weak.area_estimate - strong.area_estimate).abs() < 1.0e-12,
            "halfspace area must be γ-independent: weak={} strong={}",
            weak.area_estimate,
            strong.area_estimate
        );
        assert!(
            (weak.interface_length - strong.interface_length).abs() < 1.0e-12,
            "halfspace interface length must be γ-independent: weak={} strong={}",
            weak.interface_length,
            strong.interface_length
        );
    }

    /// Identical circle geometry must give identical area and interface-length
    /// estimates on repeated calls (determinism).
    #[test]
    fn ex38_circle_embedded_results_are_deterministic() {
        let args = Args {
            n: 10, radius: 0.25, cx: 0.5, cy: 0.5,
            alpha: 20.0, subdiv: 6, nitsche_gamma: 20.0,
            level_set: None,
        };
        let r1 = solve_embedded_problem(&args);
        let r2 = solve_embedded_problem(&args);
        assert_eq!(r1.area_estimate, r2.area_estimate,
            "area estimate is not deterministic: {} vs {}", r1.area_estimate, r2.area_estimate);
        assert_eq!(r1.interface_length, r2.interface_length,
            "interface length is not deterministic: {} vs {}", r1.interface_length, r2.interface_length);
    }

    #[test]
    fn ex38_text_checkpoint_roundtrip_preserves_solution_and_metrics() {
        let args = Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: Some(LevelSetShape::Halfspace {
                normal: [0.0, 1.0],
                offset: 0.5,
            }),
        };
        let result = solve_embedded_problem(&args);
        let path = temp_output_path("checkpoint", "txt");
        let state = EmbeddedCheckpointState {
            args: args.clone(),
            result: result.clone(),
        };

        write_embedded_checkpoint(&path, &state).unwrap();
        let restored = read_embedded_checkpoint(&path).unwrap();

        assert_eq!(restored.args.n, args.n);
        assert_eq!(level_set_name(&restored.args), "halfspace");
        assert_eq!(restored.result.values.len(), result.values.len());
        assert_eq!(restored.result.values, result.values);
        assert!((restored.result.area_rel_error - result.area_rel_error).abs() < 1.0e-15);
        assert!((restored.result.l2_error - result.l2_error).abs() < 1.0e-15);
        assert!((restored.result.interface_length - result.interface_length).abs() < 1.0e-15);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn ex38_vtk_export_writes_embedded_solution_file() {
        let args = Args {
            n: 12,
            radius: 0.28,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 6,
            nitsche_gamma: 20.0,
            level_set: None,
        };
        let result = solve_embedded_problem(&args);
        let prefix = temp_output_path("vtk", "out");
        let vtk_path = format!("{prefix}_embedded_solution.vtu");

        write_ex38_vtk_export(&prefix, &result.mesh, &result.values).unwrap();

        let vtk = fs::read_to_string(&vtk_path).unwrap();
        assert!(vtk.contains("embedded_solution"));

        let _ = fs::remove_file(vtk_path);
    }

    #[test]
    fn ex38_hdf5_checkpoint_roundtrip_preserves_solution_and_metrics() {
        let args = Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None,
        };
        let result = solve_embedded_problem(&args);
        let path = temp_output_path("checkpoint_h5", "h5");
        let state = EmbeddedCheckpointState {
            args: args.clone(),
            result: result.clone(),
        };

        write_embedded_hdf5_checkpoint(&path, &state).unwrap();
        let restored = read_embedded_hdf5_checkpoint(&path).unwrap();

        assert_eq!(restored.args.n, args.n);
        assert_eq!(level_set_name(&restored.args), "circle");
        assert_eq!(restored.result.values, result.values);
        assert!((restored.result.area_rel_error - result.area_rel_error).abs() < 1.0e-15);
        assert!((restored.result.boundary_l2_error - result.boundary_l2_error).abs() < 1.0e-15);

        let _ = fs::remove_file(path);
    }

    #[cfg(feature = "io_hdf5")]
    #[test]
    fn ex38_hdf5_checkpoint_writes_xdmf_sidecar() {
        let args = Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None,
        };
        let result = solve_embedded_problem(&args);
        let h5_path = temp_output_path("checkpoint_sidecar", "h5");
        let state = EmbeddedCheckpointState {
            args,
            result,
        };

        write_embedded_hdf5_checkpoint(&h5_path, &state).unwrap();
        write_embedded_hdf5_xdmf_sidecars(&h5_path, &state).unwrap();

        let sidecar = checkpoint_sidecar_path(&h5_path, "embedded_solution").unwrap();
        let xdmf = fs::read_to_string(&sidecar).unwrap();
        assert!(xdmf.contains("embedded_solution"));
        assert!(xdmf.contains("CollectionType=\"Temporal\""));

        let _ = fs::remove_file(h5_path);
        let _ = fs::remove_file(sidecar);
    }

    #[test]
    fn ex38_template_kpi_csv_row_uses_immersed_boundary_contract() {
        let _guard = KPI_ENV_LOCK.lock().unwrap();
        let args = Args {
            n: 16,
            radius: 0.30,
            cx: 0.5,
            cy: 0.5,
            alpha: 20.0,
            subdiv: 8,
            nitsche_gamma: 20.0,
            level_set: None,
        };
        let result = solve_embedded_problem(&args);
        let temp_path = std::env::temp_dir().join(format!(
            "ex38_template_kpi_{}.csv",
            std::process::id()
        ));
        let _ = fs::remove_file(&temp_path);

        std::env::set_var("FEM_TEMPLATE_KPI_CSV", &temp_path);
        std::env::set_var("FEM_TEMPLATE_KPI_RUN_ID", "test");
        std::env::set_var("FEM_TEMPLATE_KPI_TAG", "unit");

        let coupling = TemplateCouplingSummary {
            steps: 1,
            converged_steps: 1,
            max_coupling_iters_used: 1,
        };
        let adaptive = TemplateAdaptiveSummary {
            sync_retries: 0,
            rejected_sync_steps: 0,
            rollback_count: 0,
        };
        maybe_write_template_kpi_csv(
            builtin_template_spec(BuiltinMultiphysicsTemplate::ImmersedBoundary)
                .template
                .id(),
            coupling,
            adaptive,
            &[
                ("area_rel_error", result.area_rel_error),
                ("l2_error", result.l2_error),
                ("boundary_l2_error", result.boundary_l2_error),
                ("interface_length", result.interface_length),
            ],
        )
        .unwrap();

        let csv = fs::read_to_string(&temp_path).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains("immersed_boundary,test,unit"));
        assert!(lines[1].contains("area_rel_error="));
        assert!(lines[1].contains("boundary_l2_error="));

        std::env::remove_var("FEM_TEMPLATE_KPI_CSV");
        std::env::remove_var("FEM_TEMPLATE_KPI_RUN_ID");
        std::env::remove_var("FEM_TEMPLATE_KPI_TAG");
        let _ = fs::remove_file(&temp_path);
    }
}

