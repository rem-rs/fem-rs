//! # Fit-Node-Position Miniapp (1:1 port of MFEM
//! `miniapps/meshing/fit-node-position.cpp`, MFEM 4.9)
//!
//! Fits a selected set of the mesh nodes (the boundary faces with attribute 2)
//! to prescribed physical positions while maintaining a valid mesh with good
//! quality, through a TMOP energy with a surface-fitting term.
//!
//! Port notes (vs C++), scope of this port:
//! - The C++ miniapp is a 1-process `ParMesh` run with purely serial logic;
//!   this port uses the serial fem-rs stack (identical math and output).
//! - Serial, full (LEGACY) assembly, quad (2D) and hex (3D) meshes, metric 2
//!   (2D) / 302 (3D), target IDEAL_SHAPE_UNIT_SIZE, fitting weight 100.
//! - The TMOP integrator uses its default integration rule
//!   `IntRules.Get(geom, 2*order + 3)` (Gauss-Legendre); the `-qo` option of
//!   the C++ miniapp only feeds the `TMOPNewtonSolver` constructor whose rule
//!   is never consumed, so `-qo` is parsed and printed but unused here too.
//! - The `square01-tri.mesh` / `cube-tet.mesh` samples of the C++ header
//!   comments need tri/tet TMOP support and are rejected (exit 3).
//! - GLVis output is not ported (`-vis` is accepted and ignored).
//! - Setting the environment variable `FNP_DUMP=<file>` additionally reports
//!   the initial/final strain energies (with the initial fitting weight) and
//!   writes the final coordinates, mirroring the reference harness used for
//!   the 1:1 comparison against the C++ run.

use fem_assembly::tmop_form::{
    count_elements_per_dof, count_wrong_orientations, curved_mesh_positions,
    linear_mesh_positions, metric_from_id_2d, metric_from_id_3d, tmop_newton_solve_surf_fit,
    SharedMinDet, SurfFitNewtonParams, SurfFitPos, TmopForm, TmopIntegrator, TmopQuadType,
    TmopTarget, TmopTargetType,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::refine_uniform;
use fem_space::constraints::boundary_dofs;
use fem_space::{DofManager, FESpace, H1Space};
use std::cell::Cell;
use std::rc::Rc;

// ─── C-style number formatting ───────────────────────────────────────────────

/// Normalize Rust's `1.8633e-1` to C's `1.8633e-01`.
fn fix_exp(s: String) -> String {
    match s.find('e') {
        Some(pos) => {
            let (m, e) = s.split_at(pos);
            let exp: i32 = e[1..].parse().unwrap();
            format!("{}e{}{:02}", m, if exp < 0 { "-" } else { "+" }, exp.abs())
        }
        None => s,
    }
}

/// C `printf("%.17e", x)`.
fn fmt_e17(x: f64) -> String {
    fix_exp(format!("{:.17e}", x))
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    rs_levels: i32,
    mesh_poly_deg: i32,
    quad_order: i32,
    visualization: bool,
    visport: i32,
}

fn print_usage(argv0: &str) {
    println!("Usage: {argv0} [options] ...");
    println!("Options:");
    println!("   -m  --mesh           Mesh file to use.");
    println!("   -rs --refine-serial  Number of times to refine the mesh uniformly in serial.");
    println!("   -o  --order          Polynomial degree of mesh finite element space.");
    println!("   -qo --quad_order     Order of the quadrature rule.");
    println!("   -vis --visualization Enable or disable GLVis visualization.");
    println!("   -no-vis              (disable GLVis visualization)");
    println!("   -p  --send-port      Socket for GLVis.");
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "square01.mesh".to_string(),
        rs_levels: 2,
        mesh_poly_deg: 2,
        quad_order: 5,
        visualization: true,
        visport: 19916,
    };
    let argv: Vec<String> = std::env::args().collect();
    let argv0 = argv.first().cloned().unwrap_or_default();
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let mut val = |s: &str| -> String {
            i += 1;
            if i >= argv.len() {
                panic!("missing value for {}", s);
            }
            argv[i].clone()
        };
        match arg {
            "-m" | "--mesh" => a.mesh_file = val(arg),
            "-rs" | "--refine-serial" => a.rs_levels = val(arg).parse().unwrap(),
            "-o" | "--order" => a.mesh_poly_deg = val(arg).parse().unwrap(),
            "-qo" | "--quad_order" => a.quad_order = val(arg).parse().unwrap(),
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-p" | "--send-port" => a.visport = val(arg).parse().unwrap(),
            "-h" | "--help" => {
                print_usage(&argv0);
                std::process::exit(0);
            }
            other => {
                println!("Unrecognized option: {other}");
                print_usage(&argv0);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    a
}

/// MFEM `OptionsParser::PrintOptions` (options in registration order, ENABLE
/// options printing the active variant without a value).
fn print_options(a: &Args) {
    println!("Options used:");
    println!("   --mesh {}", a.mesh_file);
    println!("   --refine-serial {}", a.rs_levels);
    println!("   --order {}", a.mesh_poly_deg);
    println!("   --quad_order {}", a.quad_order);
    if a.visualization {
        println!("   --visualization");
    } else {
        println!("   --no-visualization");
    }
    println!("   --send-port {}", a.visport);
}

/// Boundary attribute semantics of the miniapp: 2 = fitted (y is prescribed),
/// 1 fixes the x component, 3 the z component (3D only), 4 everything.
fn run<M: MeshTopology>(mesh: M, order: u8, dim: usize) {
    let space = H1Space::new(mesh, order);
    let mesh = space.mesh();
    let dm: &DofManager = space.dof_manager();
    let topo: &dyn MeshTopology = mesh;
    let n = dm.n_dofs;

    // Nodal positions of the starting mesh (MFEM `SetNodalFESpace` +
    // `SetNodalGridFunction`: interpolation of the file's geometry map).
    let x0 = if mesh.geom_order() > 1 {
        curved_mesh_positions(topo, &dm, order, dim)
    } else {
        linear_mesh_positions(topo, &dm, order, dim)
    };

    // Pick which nodes to fit and select the target positions.
    // (attribute 2 would have a prescribed deformation in y-direction, same x).
    let mut fit_marker = vec![false; n];
    let mut coord_target = x0.clone();
    {
        let pi = std::f64::consts::PI;
        for dof in boundary_dofs(topo, dm, &[2]) {
            let dof = dof as usize;
            let x = x0[dof];
            let y = x0[n + dof];
            let z = if dim == 3 { x0[2 * n + dof] } else { 0.0 };
            fit_marker[dof] = true;
            if y < 0.5 {
                coord_target[n + dof] = 0.1 * (4.0 * pi * x).sin() * (pi * z).cos();
            } else if x < 0.5 {
                coord_target[n + dof] = 1.0 + 0.1 * (2.0 * pi * x).sin();
            } else {
                coord_target[n + dof] = 1.0 + 0.1 * (2.0 * pi * (x + 0.5)).sin();
            }
        }
    }

    // Allow slipping along the remaining boundaries.
    // (attributes 1 and 3 would slip, while 4 is completely fixed).
    let mut ess_vdofs: Vec<usize> = Vec::new();
    for tag in [1, 3, 4] {
        let dofs = boundary_dofs(topo, dm, &[tag]);
        if dofs.is_empty() {
            continue;
        }
        if dim == 2 && tag == 3 {
            panic!(
                "Boundary attribute 3 must be used only for 3D meshes. \
Adjust the attributes (1/2/3/4 for fixed x/y/z/all components, rest for \
free nodes), or use -fix-bnd."
            );
        }
        for dof in dofs {
            let dof = dof as usize;
            match tag {
                1 => ess_vdofs.push(dof), // Fix x components.
                3 => ess_vdofs.push(2 * n + dof), // Fix z components.
                _ => {
                    for c in 0..dim {
                        // Fix all components.
                        ess_vdofs.push(c * n + dof);
                    }
                }
            }
        }
    }

    // TMOP setup. The integrator rule is MFEM's default
    // `IntRules.Get(geom, 2*order + 3)` (the `-qo` value is never consumed).
    let quad_order = 2 * order + 3;
    let min_det = SharedMinDet::new(0.0);
    let metric = if dim == 2 {
        metric_from_id_2d(2, &min_det)
    } else {
        metric_from_id_3d(302, &min_det)
    }
    .unwrap();
    let fit_weight = Rc::new(Cell::new(100.0));
    let surf_fit = SurfFitPos {
        pos: Rc::new(coord_target.clone()),
        marker: Rc::new(fit_marker.clone()),
        dof_count: Rc::new(count_elements_per_dof(topo, dm)),
        // The adaptive surface fitting multiplies this weight; keep the handle
        // to restore the initial weight for the FNP_DUMP energy report.
        coeff: fit_weight.clone(),
        // MFEM surf_fit_normal: 1.0 (this driver does not use normalization).
        normal: Cell::new(1.0),
    };

    let mut form = TmopForm::new(topo, dm, order, TmopQuadType::GaussLegendre, quad_order);
    form.set_x0(x0.clone());
    form.push_integrator(TmopIntegrator {
        metric,
        target: TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
        coeff: 1.0,
        surf_fit: Some(surf_fit),
        metric_normal: 1.0,
        limiting: None,
    });
    form.finalize_targets();
    form.ess_vdofs = ess_vdofs;

    // Nonlinear solve: TMOPNewtonSolver with MINRES as the Newton system
    // solver, adaptive surface fitting (factor 10) and termination on the
    // maximum fitting error (1e-3); print level 1.
    let mut params = SurfFitNewtonParams::new();
    params.scale_factor = 10.0;
    params.max_err_limit = 1e-3;
    let (dx, _result) = tmop_newton_solve_surf_fit(
        &form,
        params,
        /* newton_max_iter */ 200,
        /* newton_rtol    */ 1e-10,
        /* newton_atol    */ 0.0,
        /* minres_max_it  */ 100,
        /* minres_rtol    */ 1e-12,
        /* print_level    */ 1,
        &min_det,
    );

    // Optimized node positions.
    let mut x = x0.clone();
    for (xi, &di) in x.iter_mut().zip(dx.iter()) {
        *xi += di;
    }

    // Optional reference-comparison dump (mirrors the serial C++ harness).
    if let Ok(dump_path) = std::env::var("FNP_DUMP") {
        fit_weight.set(100.0); // report with the initial fitting weight
        let zeros = vec![0.0_f64; form.n_dofs()];
        println!("Initial strain energy: {}", fmt_e17(form.energy(&zeros)));
        println!("Final strain energy:   {}", fmt_e17(form.energy(&dx)));
        println!("NDofs {n}");
        let content: String = x.iter().map(|&v| format!("{}\n", fmt_e17(v))).collect();
        std::fs::write(&dump_path, content).expect("write FNP_DUMP file");
        println!("Wrote final coordinates to {dump_path}");
    }
}

fn main() {
    let a = parse_args();
    print_options(&a);

    // Initialize and refine the starting mesh.
    let file = read_mfem_file(&a.mesh_file).expect("cannot open mesh file");
    let order = if a.mesh_poly_deg <= 0 { 2 } else { a.mesh_poly_deg as u8 };

    // MFEM `Mesh::Loader` runs `CheckElementOrientation(false)` on the file's
    // geometry; it reports (never fixes) inverted elements at load time.
    {
        let wrong = if let Some(m) = file.mesh2d.as_ref() {
            count_wrong_orientations(m)
        } else if let Some(m) = file.mesh3d.as_ref() {
            count_wrong_orientations(m)
        } else {
            0
        };
        let n = file
            .mesh2d
            .as_ref()
            .map(|m| m.n_elements())
            .or_else(|| file.mesh3d.as_ref().map(|m| m.n_elements()))
            .unwrap_or(0);
        if wrong > 0 {
            println!("Elements with wrong orientation: {wrong} / {n} (NOT FIXED)");
        }
    }

    if let Some(mut m) = file.mesh2d {
        for _ in 0..a.rs_levels {
            m = refine_uniform(&m);
        }
        let dim = m.dim() as usize;
        if m.n_elements() > 0 && m.element_type(0) != ElementType::Quad4 {
            eprintln!(
                "fit-node-position (Rust port): tri/tet meshes are not supported yet \
(square01-tri.mesh / cube-tet.mesh need tri/tet TMOP support)."
            );
            std::process::exit(3);
        }
        run(m, order, dim);
    } else if file.mesh3d.is_some() {
        // 3D scope trim: the fem-rs 3D invariant-Hessian chain (ddI2b/ddI2 in
        // crates/mesh tmop) does not yet numerically match the C++ reference,
        // so the 3D Newton path cannot be verified 1:1.  Disabled explicitly.
        eprintln!(
            "fit-node-position (Rust port): 3D meshes are disabled: the fem-rs 3D \
invariant-Hessian chain does not yet numerically match the C++ reference \
(cube.mesh needs the TMOP_Metric_302 3D Hessian audit)."
        );
        std::process::exit(3);
    } else {
        panic!("mesh must be 2D or 3D");
    }
}
