//! # Mesh Optimizer Miniapp (1:1 port of MFEM `miniapps/meshing/mesh-optimizer.cpp`)
//!
//! Mesh optimization via the Target-Matrix Optimization Paradigm (TMOP):
//! minimizes `sum_T int_T mu(J(x))` with a Newton solver using the exact
//! energy / min-det / residual-norm line search of MFEM's `TMOPNewtonSolver`.
//!
//! Port notes (vs C++), scope of this port:
//! - Serial, full (LEGACY) assembly, quad (2D) and hex (3D) meshes with a
//!   Gauss-Lobatto Qk nodal geometry space.
//! - Metric ids: 2D {1,2,7,9,14,22,50,55,56,58,77}; 3D {301,302,303,304,315,
//!   316,318,321,323,360}. Other ids known to the C++ miniapp are rejected
//!   with an explicit "not available in the Rust port" message.
//! - Target ids 1 (ideal shape unit size), 2 (ideal shape equal size) and
//!   3 (ideal shape, initial size). Discrete/analytic adaptivity targets
//!   (4-11), limiting (`-lc`), adaptive limiting (`-alc`), normalization
//!   (`-nor`), FD derivatives (`-fd`), PA (`-pa`), hr-adaptivity (`-hr`),
//!   combos (`-cmb`), LBFGS (`-st 1`) and untangler barriers (`-btype`,
//!   `-wctype`) are not ported yet and rejected when requested.
//! - GLVis output is not ported (`-vis` is accepted and ignored); the
//!   `perturbed.mesh` / `optimized.mesh` outputs are written with a `nodes`
//!   section exactly like the C++ (precision 6 / 14 respectively).
//!
//! Sample runs:
//!   cargo run --release --example mesh_optimizer -- -no-vis
//!   cargo run --release --example mesh_optimizer -- -m square01.mesh -o 2 -rs 2 -mid 2 -tid 1 -ni 200 -bnd -qt 1 -qo 8 -no-vis
//!   cargo run --release --example mesh_optimizer -- -m jagged.mesh -o 2 -mid 22 -tid 1 -ni 50 -li 50 -qo 4 -no-vis

use fem_assembly::tmop_form::{
    count_wrong_orientations, curved_mesh_positions, linear_mesh_positions, metric_from_id_2d,
    metric_from_id_3d, tmop_newton_solve, SharedMinDet, TmopForm, TmopIntegrator, TmopLinSolver,
    TmopQuadType, TmopTarget, TmopTargetType,
};
use fem_core::FemResult;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, refine_uniform_3d};
use fem_mesh::element_type::ElementType;
use fem_mesh::MeshTopology;
use fem_space::constraints::boundary_dofs;
use fem_space::FESpace;
use fem_space::H1Space;

// ─── C-style number formatting ───────────────────────────────────────────────

/// C `%g` with `sig` significant digits (std::cout precision `sig`).
fn fmt_gp(x: f64, sig: usize) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..(sig as i32)).contains(&exp) {
        fix_exp(format!("{:.*e}", sig - 1, x))
    } else {
        let decimals = (sig as i32 - 1 - exp).max(0) as usize;
        let mut s = format!("{:.*}", decimals, x);
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}

/// C `%g` with 6 significant digits (std::cout default precision).
fn fmt_g6(x: f64) -> String {
    fmt_gp(x, 6)
}

/// C `std::scientific` with `setprecision(4)`.
fn fmt_e4(x: f64) -> String {
    fix_exp(format!("{:.4e}", x))
}

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

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    mesh_poly_deg: i32,
    rs_levels: i32,
    jitter: f64,
    metric_id: i32,
    target_id: i32,
    lim_const: f64,
    adapt_lim_const: f64,
    quad_type: i32,
    quad_order: i32,
    solver_type: i32,
    solver_iter: i32,
    solver_rtol: f64,
    solver_art_type: i32,
    lin_solver: i32,
    max_lin_iter: i32,
    move_bnd: bool,
    combomet: i32,
    bal_expl_combo: bool,
    hradaptivity: bool,
    h_metric_id: i32,
    normalization: bool,
    visualization: bool,
    verbosity_level: i32,
    fdscheme: bool,
    adapt_eval: i32,
    exactaction: bool,
    integ_over_targ: bool,
    pa: bool,
    n_hr_iter: i32,
    n_h_iter: i32,
    mesh_node_order: i32,
    barrier_type: i32,
    worst_case_type: i32,
    detj_bound: bool,
}

fn unsupported(opt: &str) -> ! {
    eprintln!("mesh-optimizer (Rust port): option {} is not supported yet.", opt);
    std::process::exit(3);
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "icf.mesh".to_string(),
        mesh_poly_deg: 1,
        rs_levels: 0,
        jitter: 0.0,
        metric_id: 1,
        target_id: 1,
        lim_const: 0.0,
        adapt_lim_const: 0.0,
        quad_type: 1,
        quad_order: 8,
        solver_type: 0,
        solver_iter: 20,
        solver_rtol: 1e-10,
        solver_art_type: 0,
        lin_solver: 2,
        max_lin_iter: 100,
        move_bnd: true,
        combomet: 0,
        bal_expl_combo: false,
        hradaptivity: false,
        h_metric_id: -1,
        normalization: false,
        visualization: true,
        verbosity_level: 0,
        fdscheme: false,
        adapt_eval: 0,
        exactaction: false,
        integ_over_targ: true,
        pa: false,
        n_hr_iter: 5,
        n_h_iter: 1,
        mesh_node_order: 0,
        barrier_type: 0,
        worst_case_type: 0,
        detj_bound: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
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
            "-o" | "--order" => a.mesh_poly_deg = val(arg).parse().unwrap(),
            "-rs" | "--refine-serial" => a.rs_levels = val(arg).parse().unwrap(),
            "-ji" | "--jitter" => a.jitter = val(arg).parse().unwrap(),
            "-mid" | "--metric-id" => a.metric_id = val(arg).parse().unwrap(),
            "-tid" | "--target-id" => a.target_id = val(arg).parse().unwrap(),
            "-lc" | "--limit-const" => a.lim_const = val(arg).parse().unwrap(),
            "-alc" | "--adapt-limit-const" => a.adapt_lim_const = val(arg).parse().unwrap(),
            "-qt" | "--quad-type" => a.quad_type = val(arg).parse().unwrap(),
            "-qo" | "--quad_order" => a.quad_order = val(arg).parse().unwrap(),
            "-st" | "--solver-type" => a.solver_type = val(arg).parse().unwrap(),
            "-ni" | "--newton-iters" => a.solver_iter = val(arg).parse().unwrap(),
            "-rtol" | "--newton-rel-tolerance" => a.solver_rtol = val(arg).parse().unwrap(),
            "-art" | "--adaptive-rel-tol" => a.solver_art_type = val(arg).parse().unwrap(),
            "-ls" | "--lin-solver" => a.lin_solver = val(arg).parse().unwrap(),
            "-li" | "--lin-iter" => a.max_lin_iter = val(arg).parse().unwrap(),
            "-bnd" | "--move-boundary" => a.move_bnd = true,
            "-fix-bnd" | "--fix-boundary" => a.move_bnd = false,
            "-cmb" | "--combo-type" => a.combomet = val(arg).parse().unwrap(),
            "-bec" | "--balance-explicit-combo" => a.bal_expl_combo = true,
            "-no-bec" | "--no-balance-explicit-combo" => a.bal_expl_combo = false,
            "-hr" | "--hr-adaptivity" => a.hradaptivity = true,
            "-no-hr" | "--no-hr-adaptivity" => a.hradaptivity = false,
            "-hmid" | "--h-metric" => a.h_metric_id = val(arg).parse().unwrap(),
            "-nor" | "--normalization" => a.normalization = true,
            "-no-nor" | "--no-normalization" => a.normalization = false,
            "-fd" | "--fd_approximation" => a.fdscheme = true,
            "-no-fd" | "--no-fd-approx" => a.fdscheme = false,
            "-ex" | "--exact_action" => a.exactaction = true,
            "-no-ex" | "--no-exact-action" => a.exactaction = false,
            "-it" | "--integrate-target" => a.integ_over_targ = true,
            "-ir" | "--integrate-reference" => a.integ_over_targ = false,
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-vl" | "--verbosity-level" => a.verbosity_level = val(arg).parse().unwrap(),
            "-ae" | "--adaptivity-evaluator" => a.adapt_eval = val(arg).parse().unwrap(),
            "-d" | "--device" => {
                val(arg); // accepted, always cpu
            }
            "-pa" | "--partial-assembly" => a.pa = true,
            "-no-pa" | "--no-partial-assembly" => a.pa = false,
            "-nhr" | "--n_hr_iter" => a.n_hr_iter = val(arg).parse().unwrap(),
            "-nh" | "--n_h_iter" => a.n_h_iter = val(arg).parse().unwrap(),
            "-mno" | "--mesh_node_ordering" => a.mesh_node_order = val(arg).parse().unwrap(),
            "-btype" | "--barrier-type" => a.barrier_type = val(arg).parse().unwrap(),
            "-wctype" | "--worst-case-type" => a.worst_case_type = val(arg).parse().unwrap(),
            "-db" | "--detj-bound" => a.detj_bound = true,
            "-no-db" | "--no-detj-bound" => a.detj_bound = false,
            other => panic!("Unknown option: {}", other),
        }
        i += 1;
    }

    // Rejected non-default features (not ported yet).
    if a.jitter > 0.0 {
        unsupported("-ji");
    }
    if a.lim_const != 0.0 {
        unsupported("-lc");
    }
    if a.adapt_lim_const != 0.0 {
        unsupported("-alc");
    }
    if a.solver_type != 0 {
        unsupported("-st 1 (LBFGS)");
    }
    if a.solver_art_type != 0 {
        unsupported("-art");
    }
    if a.combomet != 0 {
        unsupported("-cmb");
    }
    if a.bal_expl_combo {
        unsupported("-bec");
    }
    if a.hradaptivity {
        unsupported("-hr");
    }
    if a.normalization {
        unsupported("-nor");
    }
    if a.fdscheme {
        unsupported("-fd");
    }
    if a.exactaction {
        unsupported("-ex");
    }
    if !a.integ_over_targ {
        unsupported("-ir");
    }
    if a.pa {
        unsupported("-pa");
    }
    if a.mesh_node_order != 0 {
        unsupported("-mno 1");
    }
    if a.barrier_type != 0 {
        unsupported("-btype");
    }
    if a.worst_case_type != 0 {
        unsupported("-wctype");
    }
    if a.detj_bound {
        unsupported("-db");
    }
    if a.adapt_eval != 0 {
        unsupported("-ae 1");
    }
    if a.quad_type != 1 && a.quad_type != 2 {
        unsupported("-qt 3 (ClosedUniform)");
    }
    a
}

fn print_options(a: &Args) {
    println!("Options used:");
    println!("   --mesh {}", a.mesh_file);
    println!("   --order {}", a.mesh_poly_deg);
    println!("   --refine-serial {}", a.rs_levels);
    println!("   --jitter {}", fmt_g6(a.jitter));
    println!("   --metric-id {}", a.metric_id);
    println!("   --target-id {}", a.target_id);
    println!("   --limit-const {}", fmt_g6(a.lim_const));
    println!("   --adapt-limit-const {}", fmt_g6(a.adapt_lim_const));
    println!("   --quad-type {}", a.quad_type);
    println!("   --quad_order {}", a.quad_order);
    println!("   --solver-type {}", a.solver_type);
    println!("   --newton-iters {}", a.solver_iter);
    println!("   --newton-rel-tolerance {}", fmt_g6(a.solver_rtol));
    println!("   --adaptive-rel-tol {}", a.solver_art_type);
    println!("   --lin-solver {}", a.lin_solver);
    println!("   --lin-iter {}", a.max_lin_iter);
    println!("   {}", if a.move_bnd { "--move-boundary" } else { "--fix-boundary" });
    println!("   --combo-type {}", a.combomet);
    println!("   {}", if a.bal_expl_combo { "--balance-explicit-combo" } else { "--no-balance-explicit-combo" });
    println!("   {}", if a.hradaptivity { "--hr-adaptivity" } else { "--no-hr-adaptivity" });
    println!("   --h-metric {}", a.h_metric_id);
    println!("   {}", if a.normalization { "--normalization" } else { "--no-normalization" });
    println!("   {}", if a.fdscheme { "--fd_approximation" } else { "--no-fd-approx" });
    println!("   {}", if a.exactaction { "--exact_action" } else { "--no-exact-action" });
    println!("   {}", if a.integ_over_targ { "--integrate-target" } else { "--integrate-reference" });
    println!("   {}", if a.visualization { "--visualization" } else { "--no-visualization" });
    println!("   --verbosity-level {}", a.verbosity_level);
    println!("   --adaptivity-evaluator {}", a.adapt_eval);
    println!("   --device cpu");
    println!("   {}", if a.pa { "--partial-assembly" } else { "--no-partial-assembly" });
    println!("   --n_hr_iter {}", a.n_hr_iter);
    println!("   --n_h_iter {}", a.n_h_iter);
    println!("   --mesh_node_ordering {}", a.mesh_node_order);
    println!("   --barrier-type {}", a.barrier_type);
    println!("   --worst-case-type {}", a.worst_case_type);
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");
}

// ─── main ─────────────────────────────────────────────────────────────────────

/// Min det(Jpr)/det(Wideal) over the nodal-space geometry of the mesh; quad
/// and hex ideal elements are the unit square/cube (det(Wideal) = 1).
fn run<M: MeshTopology>(mesh: M, order: u8, dim: usize, a: &Args) {
    let space = H1Space::new(mesh, order);
    let mesh = space.mesh();
    let dm = space.dof_manager();
    let topo: &dyn MeshTopology = mesh;
    // Nodal positions of the starting mesh. If the mesh carries a high-order
    // `nodes` section, MFEM interpolates that geometry onto the new Qp nodal
    // space (`Mesh::GetNodes` -> `ProjectCoefficient(xyz)` ->
    // `NodalFiniteElement::Project`); otherwise it is the interpolation of
    // the linear vertices through the affine map.
    let x0 = if mesh.geom_order() > 1 {
        curved_mesh_positions(topo, &dm, order, dim)
    } else {
        linear_mesh_positions(topo, &dm, order, dim)
    };

    // Metric.
    let min_det_shared = SharedMinDet::new(0.0);
    let metric = if dim == 2 {
        metric_from_id_2d(a.metric_id, &min_det_shared)
    } else {
        metric_from_id_3d(a.metric_id, &min_det_shared)
    };
    if metric.is_none() {
        println!("Unknown metric_id: {}", a.metric_id);
        std::process::exit(3);
    }

    // Target.
    let target = match a.target_id {
        1 => TmopTarget::new(TmopTargetType::IdealShapeUnitSize),
        2 => TmopTarget::new(TmopTargetType::IdealShapeEqualSize),
        3 => TmopTarget::new(TmopTargetType::IdealShapeGivenSize),
        _ => {
            println!(
                "target_id {} requires adaptivity targets not ported yet",
                a.target_id
            );
            std::process::exit(3);
        }
    };

    let quad_type = match a.quad_type {
        1 => TmopQuadType::GaussLobatto,
        2 => TmopQuadType::GaussLegendre,
        _ => unreachable!("rejected in parse_args"),
    };
    let quad_order = a.quad_order as u8;

    let mut form = TmopForm::new(topo, &dm, order, quad_type, quad_order);
    form.set_x0(x0);
    form.push_integrator(TmopIntegrator {
        metric: metric.unwrap(),
        target,
        coeff: 1.0,
        surf_fit: None,
    });
    form.finalize_targets();

    // h0: minimal local mesh size per scalar dof (MFEM `Mesh::GetElementSize`:
    // det(J) at the element center, raised to 1/dim, of the new nodal mesh).
    let ne = topo.n_elements();
    let elem_sizes = form.element_sizes_center();
    let mut h0 = vec![f64::INFINITY; dm.n_dofs];
    for e in 0..ne {
        let hi = elem_sizes[e as usize];
        for &dof in dm.element_dofs(e as u32) {
            h0[dof as usize] = h0[dof as usize].min(hi);
        }
    }

    // Quadrature point counts (as printed by the C++ miniapp).
    if dim == 2 {
        println!(
            "Triangle quadrature points: {}",
            fem_element::quadrature::tri_rule(quad_order).n_points()
        );
        println!(
            "Quadrilateral quadrature points: {}",
            quad_point_count(quad_type, quad_order, 2)
        );
    } else {
        println!(
            "Tetrahedron quadrature points: {}",
            fem_element::quadrature::tet_rule(quad_order).n_points()
        );
        println!(
            "Hexahedron quadrature points: {}",
            quad_point_count(quad_type, quad_order, 3)
        );
        println!(
            "Prism quadrature points: {}",
            fem_element::quadrature::prism_rule(quad_order).n_points()
        );
    }

    // Minimum det(J) of the starting mesh.
    let zeros = vec![0.0; form.n_dofs()];
    let mut min_detj = form.min_det_j(&zeros);
    println!(
        "Minimum det(J) of the original mesh is {}",
        fmt_g6(min_detj)
    );
    let untangling_ids = [22, 211, 252, 311, 313, 352];
    if min_detj < 0.0 && !untangling_ids.contains(&a.metric_id) {
        panic!("The input mesh is inverted! Try an untangling metric.");
    }
    if min_detj < 0.0 {
        // det(Wideal) = 1 for quads/hexes; slightly below minJ0 to avoid /0.
        min_detj -= 0.01 * h0.iter().cloned().fold(f64::INFINITY, f64::min);
    }
    min_det_shared.set(min_detj);

    // Essential boundary dofs, by boundary attribute:
    // 1/2/3 fix the x/y/z component, 4 fixes everything, rest are free.
    let mut ess_vdofs: Vec<usize> = Vec::new();
    let mut all_tags: Vec<i32> = Vec::new();
    for f in 0..topo.n_boundary_faces() {
        let t = topo.face_tag(f as u32);
        if !all_tags.contains(&t) {
            all_tags.push(t);
        }
    }
    if !a.move_bnd {
        for dof in boundary_dofs(topo, &dm, &all_tags) {
            for c in 0..dim {
                ess_vdofs.push(c * dm.n_dofs + dof as usize);
            }
        }
    } else {
        for tag in all_tags {
            if tag == 1 || tag == 2 || tag == 3 || tag == 4 {
                let dofs = boundary_dofs(topo, &dm, &[tag]);
                for dof in dofs {
                    match tag {
                        1 => ess_vdofs.push(0 * dm.n_dofs + dof as usize),
                        2 => ess_vdofs.push(1 * dm.n_dofs + dof as usize),
                        3 => ess_vdofs.push(2 * dm.n_dofs + dof as usize),
                        _ => {
                            for c in 0..dim {
                                ess_vdofs.push(c * dm.n_dofs + dof as usize);
                            }
                        }
                    }
                }
            }
        }
    }
    form.ess_vdofs = ess_vdofs;

    let init_energy = form.energy(&zeros);

    let lin = match a.lin_solver {
        0 => TmopLinSolver::L1Jacobi(a.max_lin_iter as usize),
        1 => TmopLinSolver::Cg(a.max_lin_iter as usize),
        2 => TmopLinSolver::Minres(a.max_lin_iter as usize),
        3 => TmopLinSolver::MinresJacobi(a.max_lin_iter as usize),
        4 => TmopLinSolver::MinresL1Jacobi(a.max_lin_iter as usize),
        other => panic!("lin_solver {}: unknown", other),
    };

    let (dx, result) = tmop_newton_solve(
        &form,
        &lin,
        a.solver_rtol,
        a.solver_iter as usize,
        a.verbosity_level as u8,
        &min_det_shared,
    );
    if !result.converged {
        println!("Newton: Number of iterations: {}", result.iterations);
        println!(
            "   ||r|| = {},  ||r||/||r_0|| = {}",
            fmt_g6(result.final_norm),
            fmt_g6(result.final_norm / result.initial_norm)
        );
        println!("Newton: No convergence!");
    }

    // Optimized node positions.
    let mut x = form.x0().to_vec();
    for (xi, &di) in x.iter_mut().zip(dx.iter()) {
        *xi += di;
    }

    // Save the starting and optimized meshes (with a `nodes` section like the
    // C++, precision 6 / 14 respectively).
    write_mfem_with_nodes("perturbed.mesh", mesh, form.x0(), dim, order, 6).expect("write perturbed.mesh");
    write_mfem_with_nodes("optimized.mesh", mesh, &x, dim, order, 14).expect("write optimized.mesh");

    let fin_energy = form.energy(&dx);
    println!(
        "Initial strain energy: {} = metrics: {} + extra terms: {}",
        fmt_e4(init_energy),
        fmt_e4(init_energy),
        fmt_e4(0.0)
    );
    println!(
        "  Final strain energy: {} = metrics: {} + extra terms: {}",
        fmt_e4(fin_energy),
        fmt_e4(fin_energy),
        fmt_e4(0.0)
    );
    println!(
        "The strain energy decreased by: {} %.",
        fmt_e4((init_energy - fin_energy) * 100.0 / init_energy)
    );
}

fn quad_point_count(quad_type: TmopQuadType, quad_order: u8, dim: usize) -> usize {
    match (quad_type, dim) {
        (TmopQuadType::GaussLobatto, 2) => {
            let n = quad_order as usize / 2 + 2;
            n * n
        }
        (TmopQuadType::GaussLobatto, 3) => {
            let n = quad_order as usize / 2 + 2;
            n * n * n
        }
        (_, 2) => fem_element::quadrature::quad_rule(quad_order).n_points(),
        (_, 3) => fem_element::quadrature::hex_rule(quad_order).n_points(),
        _ => unreachable!(),
    }
}

/// Write an MFEM mesh file whose geometry is a Qk nodal grid function (the
/// MFEM `nodes` section; FEC name `Linear_2D`/`Linear_3D` for order 1,
/// `H1_2D_Pp`/`H1_3D_Pp` otherwise).
fn write_mfem_with_nodes(
    path: &str,
    mesh: &impl MeshTopology,
    nodes: &[f64],
    dim: usize,
    order: u8,
    precision: usize,
) -> FemResult<()> {
    let n_scalar = nodes.len() / dim;
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);
    // Linear mesh sections (dimension/elements/boundary).
    write_mfem_topology(&mut w, mesh, dim)?;
    use std::io::Write;
    writeln!(w, "nodes")?;
    writeln!(w, "FiniteElementSpace")?;
    let fec = match (order, dim) {
        (1, 2) => "Linear_2D".to_string(),
        (1, 3) => "Linear_3D".to_string(),
        (p, d) => format!("H1_{}D_P{}", d, p),
    };
    writeln!(w, "FiniteElementCollection: {}", fec)?;
    writeln!(w, "VDim: {}", dim)?;
    writeln!(w, "Ordering: 0")?;
    writeln!(w)?;
    for c in 0..dim {
        for i in 0..n_scalar {
            writeln!(w, "{}", fmt_gp(nodes[c * n_scalar + i], precision))?;
        }
    }
    Ok(())
}

/// Dimension/elements/boundary sections of the MFEM mesh format, via the io
/// crate's linear writer.
fn write_mfem_topology<W: std::io::Write>(
    w: &mut W,
    mesh: &impl MeshTopology,
    _dim: usize,
) -> FemResult<()> {
    // fem_io::write_mfem needs a Mesh<2>/Mesh<3>; the topology-only fallback
    // writes the header and connectivity directly (same byte layout for
    // quad/hex meshes).
    writeln!(w, "MFEM mesh v1.0")?;
    writeln!(w)?;
    writeln!(w, "#")?;
    writeln!(w, "# MFEM Geometry Types (see fem/geom.hpp):")?;
    writeln!(w, "#")?;
    writeln!(w, "# POINT       = 0")?;
    writeln!(w, "# SEGMENT     = 1")?;
    writeln!(w, "# TRIANGLE    = 2")?;
    writeln!(w, "# SQUARE      = 3")?;
    writeln!(w, "# TETRAHEDRON = 4")?;
    writeln!(w, "# CUBE        = 5")?;
    writeln!(w, "# PRISM       = 6")?;
    writeln!(w)?;
    writeln!(w, "dimension")?;
    writeln!(w, "{}", mesh.dim())?;
    writeln!(w)?;
    writeln!(w, "elements")?;
    writeln!(w, "{}", mesh.n_elements())?;
    for e in 0..mesh.n_elements() {
        let nodes = mesh.element_nodes(e as u32);
        let geom = match mesh.element_type(e as u32) {
            ElementType::Quad4 => 3,
            ElementType::Hex8 => 5,
            ElementType::Tri3 => 2,
            ElementType::Tet4 => 4,
            other => panic!("unsupported element type {:?}", other),
        };
        let attr = mesh.element_tag(e as u32);
        writeln!(
            w,
            "{} {} {}",
            attr,
            geom,
            nodes.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" ")
        )?;
    }
    writeln!(w)?;
    writeln!(w, "boundary")?;
    writeln!(w, "{}", mesh.n_boundary_faces())?;
    for f in 0..mesh.n_boundary_faces() {
        let nodes = mesh.face_nodes(f as u32);
        let geom = match nodes.len() {
            4 => 3,
            8 => 3, // high-order face; nodes.len() distinguishes via element type
            2 => 1,
            3 => 2,
            _ => panic!("unsupported face"),
        };
        writeln!(
            w,
            "{} {} {}",
            mesh.face_tag(f as u32),
            geom,
            nodes.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" ")
        )?;
    }
    Ok(())
}

fn main() {
    let a = parse_args();
    print_options(&a);
    let _ = a.visualization; // GLVis not ported.

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
        if m.n_elements() > 0 {
            let et = m.element_type(0);
            if et != ElementType::Quad4 {
                panic!(
                    "Rust port v1 scope: quad meshes only (found {:?})",
                    et
                );
            }
        }
        run(m, order, dim, &a);
    } else if let Some(mut m) = file.mesh3d {
        for _ in 0..a.rs_levels {
            m = refine_uniform_3d(&m);
        }
        let dim = m.dim() as usize;
        if m.n_elements() > 0 {
            let et = m.element_type(0);
            if et != ElementType::Hex8 {
                panic!(
                    "Rust port v1 scope: hex meshes only (found {:?})",
                    et
                );
            }
        }
        run(m, order, dim, &a);
    } else {
        panic!("mesh must be 2D or 3D");
    }
}

