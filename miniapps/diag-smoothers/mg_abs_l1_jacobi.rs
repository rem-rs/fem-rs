//! MG Abs-L1 Jacobi smoothers miniapp — 1:1 serial (np = 1) port of MFEM
//! `miniapps/diag-smoothers/mg-abs-l1-jacobi.cpp` (+ `ds-common.cpp`).
//!
//! Solves an H1 mass (`-i 0`) or diffusion (`-i 1`) system preconditioned by a
//! geometric multigrid whose level smoothers are the absolute-value L(1)
//! Jacobi diagonal (`diag = |A|·1`, ds-common `AbsL1GeometricMultigrid`).  The
//! hierarchy is `1 + geometric_levels` uniformly refined order-`order` levels
//! followed by `order_levels` p-refined levels of order `2^(lo+1)` (MFEM
//! `ParFiniteElementSpaceHierarchy`); the cycle is `VCYCLE, 1, 1` and the
//! coarse level runs PCG with rel. tol `sqrt(1e-10)` / 10 iterations and the
//! same AbsL1 Jacobi smoother.  The global solver is SLI (`-s 0`) or PCG
//! (`-s 1`).
//!
//! The C++ miniapp is written against `ParMesh`/`ParFiniteElementSpace`; with
//! one MPI rank its numerics equal the serial path ported here.  The verified
//! assembly surface is `-a 0` (LEGACY, the DIAG_KEEP-eliminated sparse
//! matrix): C++ `-a 1` (FULL) wraps the uneliminated matrix in a
//! `ConstrainedOperator` with DIAG_ONE semantics whose Dirichlet rows behave
//! differently, and `-a 2/3/4` dispatch to matrix-free `AbsMult` extensions
//! that fem-rs does not implement (see the run-time note below).  The Maxwell
//! integrator (`-i 2`) aborts in the C++ miniapp as well.
//!
//! Compile with: `cargo run --release --example diag_mg_abs_l1_jacobi`
//!
//! Sample runs:
//! ```text
//! cargo run --release --example diag_mg_abs_l1_jacobi -- -no-vis
//! cargo run --release --example diag_mg_abs_l1_jacobi -- -s 0 -i 0 -no-vis
//! cargo run --release --example diag_mg_abs_l1_jacobi -- -rs 2 -rp 1 -no-vis
//! cargo run --release --example diag_mg_abs_l1_jacobi -- -t 1e-5 -ni 100 -no-vis
//! cargo run --release --example diag_mg_abs_l1_jacobi -- -m data/beam-quad.mesh -a 0 -Ky 0.5 -Kz 0.5 -no-vis
//! ```

use std::fmt::Write as _;
use std::io::Write as _;

use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::kershaw::kershaw_map;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_solver::geometric_mg::{
    build_h1_hex_refined_prolongation, build_h1_p1_refined_prolongation,
    AbsL1GeometricMultigrid, MgCoarseSolverType, MgCycleType,
};
use fem_solver::{fmt_g, solve_cg_mfem, solve_sli, IterResult, SliOptions};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{build_h1_prolongation_matrix, DofManager, H1Space};

const PI: f64 = std::f64::consts::PI;

// ─── Enums (ds-common.hpp; the Maxwell mode aborts in the C++ miniapp) ──────

/// `SolverType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverType {
    /// Stationary Linear Iteration.
    Sli,
    /// Preconditioned Conjugate Gradient.
    Cg,
}

/// `IntegratorType` (`maxwell` aborts in `mg-abs-l1-jacobi.cpp` step 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegratorType {
    /// `(u, v)` — H1 mass matrix.
    Mass,
    /// `(grad u, grad v)` — diffusion operator.
    Diffusion,
}

// ─── ds-common exact solution / source ──────────────────────────────────────

/// `ds_common::diffusion_solution`.
fn diffusion_solution<const D: usize>(kappa: f64, x: &[f64]) -> f64 {
    if D == 3 {
        (kappa * x[0]).sin() * (kappa * x[1]).sin() * (kappa * x[2]).sin() + 1.0
    } else {
        (kappa * x[0]).sin() * (kappa * x[1]).sin() + 1.0
    }
}

/// `ds_common::diffusion_source`.
fn diffusion_source<const D: usize>(kappa: f64, x: &[f64]) -> f64 {
    let base = (kappa * x[0]).sin() * (kappa * x[1]).sin();
    if D == 3 {
        3.0 * kappa * kappa * base * (kappa * x[2]).sin()
    } else {
        2.0 * kappa * kappa * base
    }
}

// ─── MFEM LEGACY integration-rule orders (fem/bilininteg.cpp GetRule) ───────

/// MFEM `ElementTransformation::OrderW()` for straight elements: `p·d−1` on
/// tensor-product geometries (Qk), `(p−1)·d` on simplices (Pk).
fn order_w(space_order: u8, dim: usize, tensor: bool) -> u8 {
    if tensor {
        space_order * dim as u8 - 1
    } else {
        (space_order - 1) * dim as u8
    }
}

/// MFEM LEGACY quadrature orders:
/// - H1 mass `2p + OrderW`; H1 diffusion Qk `2p+d−1` / Pk `2p−2`;
/// - H1 DomainLF `oa·p + ob = 2p` (defaults oa=2, ob=0);
/// - `ComputeL2Error`: `2p + 3`.
struct QuadOrders {
    mass: u8,
    diffusion: u8,
    domain_lf: u8,
    l2_error: u8,
}

fn mfem_quad_orders(space_order: u8, dim: usize, tensor: bool) -> QuadOrders {
    let p = space_order;
    let ow = order_w(p, dim, tensor);
    QuadOrders {
        mass: 2 * p + ow,
        diffusion: if tensor { 2 * p + dim as u8 - 1 } else { 2 * p - 2 },
        domain_lf: 2 * p,
        l2_error: 2 * p + 3,
    }
}

fn is_tensor_geom<const D: usize>(mesh: &Mesh<D>) -> bool {
    matches!(
        mesh.element_type(0),
        fem_mesh::ElementType::Quad4
            | fem_mesh::ElementType::Hex8
            | fem_mesh::ElementType::Prism6
            | fem_mesh::ElementType::Pyramid5
    )
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Config {
    mesh_file: String,
    order: u8,
    geometric_levels: i32,
    order_levels: i32,
    solver_type: SolverType,
    integrator_type: IntegratorType,
    assembly_type_int: i32,
    refine_serial: i32,
    refine_parallel: i32,
    rel_tol: f64,
    max_iter: i32,
    eps_y: f64,
    eps_z: f64,
    freq: f64,
    use_monitor: bool,
    visualization: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let get = |flags: &[&str], default: &str| -> String {
        for f in flags {
            if let Some(i) = args.iter().position(|a| a == f) {
                return args[i + 1].clone();
            }
        }
        default.to_string()
    };
    let has = |flags: &[&str]| flags.iter().any(|f| args.iter().any(|a| a.as_str() == *f));

    let mesh_file = get(&["-m", "--mesh"], "data/ref-cube.mesh");
    let order: u8 = get(&["-o", "--order"], "1").parse().expect("bad -o");
    let geometric_levels: i32 =
        get(&["-gl", "--geometric-levels"], "1").parse().expect("bad -gl");
    let order_levels: i32 = get(&["-ol", "--order-levels"], "1").parse().expect("bad -ol");
    let solver_type = match get(&["-s", "--solver"], "1").parse::<i32>().expect("bad -s") {
        0 => SolverType::Sli,
        1 => SolverType::Cg,
        v => panic!("invalid solver type: {v}"),
    };
    let integrator_type =
        match get(&["-i", "--integrator"], "1").parse::<i32>().expect("bad -i") {
            0 => IntegratorType::Mass,
            1 => IntegratorType::Diffusion,
            2 => panic!("Maxwell integrator not supported in this miniapp!"),
            v => panic!("invalid integrator type: {v}"),
        };
    let assembly_type_int = get(&["-a", "--assembly"], "3").parse().expect("bad -a");
    let refine_serial = get(&["-rs", "--refine-serial"], "3").parse().expect("bad -rs");
    let refine_parallel = get(&["-rp", "--refine-parallel"], "0").parse().expect("bad -rp");
    let rel_tol: f64 = get(&["-t", "--tolerance"], "1e-10").parse().expect("bad -t");
    let max_iter: i32 = get(&["-ni", "--iterations"], "3000").parse().expect("bad -ni");
    let eps_y: f64 = get(&["-Ky", "--Kershaw-y"], "0").parse().expect("bad -Ky");
    let eps_z: f64 = get(&["-Kz", "--Kershaw-z"], "0").parse().expect("bad -Kz");
    let freq: f64 = get(&["-f", "--frequency"], "1").parse().expect("bad -f");
    let use_monitor = has(&["-mon", "--monitor"]);
    let visualization = has(&["-vis", "--visualization"]);

    if !(0..6).contains(&assembly_type_int) {
        panic!("invalid assembly type: {assembly_type_int}");
    }
    if geometric_levels < 0 {
        panic!("geometric_levels needs to be non-negative");
    }
    if order_levels < 0 {
        panic!("order_levels needs to be non-negative");
    }
    if !(0.0..=1.0).contains(&eps_y) {
        panic!("eps_y must be in [0,1]");
    }
    if !(0.0..=1.0).contains(&eps_z) {
        panic!("eps_z must be in [0,1]");
    }
    if assembly_type_int >= 2 {
        eprintln!(
            "note: fem-rs implements the LEGACY/FULL (assembled sparse) path; C++ assembly \
             level {assembly_type_int} uses matrix-free operators whose AbsMult \
             (AddAbsMultPA) differs from the assembled |A| row-sums, so the smoother \
             diagonal — and with it the iteration history — differs from C++ at this \
             assembly level"
        );
    }

    Config {
        mesh_file,
        order,
        geometric_levels,
        order_levels,
        solver_type,
        integrator_type,
        assembly_type_int,
        refine_serial,
        refine_parallel,
        rel_tol,
        max_iter,
        eps_y,
        eps_z,
        freq,
        use_monitor,
        visualization,
    }
}

/// Assembly-level description strings (mg-abs-l1-jacobi.cpp step 2.5).
fn assembly_description(a: i32) -> &'static str {
    match a {
        0 => "Using Legacy type of assembly level...",
        1 => "Using Full type of assembly level...",
        2 => "Using Element type of assembly level...",
        3 => "Using Partial type of assembly level...",
        4 => "Using matrix-free type of assembly level...",
        _ => "Unsupported option!",
    }
}

/// Replicates MFEM `OptionsParser::Parse()`'s "Options used:" block plus the
/// `Device::Print()` lines.
fn print_options(c: &Config) {
    let mut s = String::from("Options used:\n");
    let _ = writeln!(s, "   --mesh {}", c.mesh_file);
    let _ = writeln!(s, "   --order {}", c.order);
    let _ = writeln!(s, "   --geometric-levels {}", c.geometric_levels);
    let _ = writeln!(s, "   --order-levels {}", c.order_levels);
    let _ = writeln!(
        s,
        "   --solver {}",
        match c.solver_type {
            SolverType::Sli => 0,
            SolverType::Cg => 1,
        }
    );
    let _ = writeln!(
        s,
        "   --integrator {}",
        match c.integrator_type {
            IntegratorType::Mass => 0,
            IntegratorType::Diffusion => 1,
        }
    );
    let _ = writeln!(s, "   --assembly {}", c.assembly_type_int);
    let _ = writeln!(s, "   --refine-serial {}", c.refine_serial);
    let _ = writeln!(s, "   --refine-parallel {}", c.refine_parallel);
    let _ = writeln!(s, "   --tolerance {}", fmt_g(c.rel_tol));
    let _ = writeln!(s, "   --iterations {}", c.max_iter);
    let _ = writeln!(s, "   --Kershaw-y {}", fmt_g(c.eps_y));
    let _ = writeln!(s, "   --Kershaw-z {}", fmt_g(c.eps_z));
    let _ = writeln!(s, "   --frequency {}", fmt_g(c.freq));
    let _ = writeln!(s, "   --device cpu");
    let _ = writeln!(s, "   {}", if c.use_monitor { "--monitor" } else { "--no-monitor" });
    let _ = writeln!(
        s,
        "   {}",
        if c.visualization { "--visualization" } else { "--no-visualization" }
    );
    s.push_str("Device configuration: cpu\n");
    print!("{s}");
    println!("Memory configuration: host-std");
}

fn main() {
    let cfg = parse_args();
    print_options(&cfg);

    let kappa = cfg.freq * PI;

    let mfem = read_mfem_file(&cfg.mesh_file)
        .unwrap_or_else(|e| panic!("failed to read mesh {}: {e}", cfg.mesh_file));
    if let Some(mesh) = mfem.mesh2d {
        run(mesh, &cfg, kappa, &|m| refine_uniform(m));
    } else if let Some(mesh) = mfem.mesh3d {
        run(mesh, &cfg, kappa, &|m| refine_uniform_3d(m));
    } else {
        panic!("no mesh in {}", cfg.mesh_file);
    }
}

/// C++ steps 3–11 for a `D`-dimensional mesh (np = 1: `ParMesh(serial)` is the
/// serial mesh, and `-rp` refinements act on it like `-rs` refinements).
fn run<const D: usize>(
    mesh: Mesh<D>,
    cfg: &Config,
    kappa: f64,
    refine: &dyn Fn(&Mesh<D>) -> Mesh<D>,
) {
    // 3. Read the serial mesh; apply the serial refinements.
    let mut mesh = mesh;
    for _ in 0..(cfg.refine_serial.max(0) + cfg.refine_parallel.max(0)) {
        mesh = refine(&mesh);
    }

    // 4. Kershaw transformation (2D: eps_z is forced to 0, like MFEM).
    let eps_z = if D < 3 { 0.0 } else { cfg.eps_z };
    if cfg.eps_y != 0.0 && (D < 3 || cfg.eps_z != 0.0) {
        mesh.transform(|x| kershaw_map::<D>(x, cfg.eps_y, eps_z));
    }

    // 5–6. Finite element space hierarchy: the coarse space plus
    //      `geometric_levels` uniformly refined levels of the same order, then
    //      `order_levels` p-refined levels of order `2^(lo+1)`
    //      (`ParFiniteElementSpaceHierarchy::{AddUniformlyRefinedLevel,
    //      AddOrderRefinedLevel}`).
    let mut meshes: Vec<Mesh<D>> = vec![mesh];
    for _ in 0..cfg.geometric_levels {
        meshes.push(refine(meshes.last().unwrap()));
    }
    let mut orders: Vec<u8> = vec![cfg.order; meshes.len()];
    if cfg.order > 1 {
        println!("Warning! Polynomial order provided. Ignoring order level...");
    } else {
        for lo in 0..cfg.order_levels {
            let p = 1u8 << (lo + 1); // std::pow(2, lo + 1)
            meshes.push(meshes[meshes.len() - 1].clone());
            orders.push(p);
        }
    }
    let mut spaces: Vec<H1Space<Mesh<D>>> = Vec::with_capacity(meshes.len());
    for (m, &p) in meshes.iter().zip(orders.iter()) {
        spaces.push(H1Space::new(m.clone(), p));
    }
    let n_levels = spaces.len();

    let sys_size = spaces.last().unwrap().n_dofs();
    println!("Number of unknowns: {sys_size}");
    println!("{}", assembly_description(cfg.assembly_type_int));

    // Per-level MFEM LEGACY quadrature orders (the element geometry is shared
    // across the hierarchy).
    let tensor = is_tensor_geom(meshes.last().unwrap());
    let quad: Vec<QuadOrders> =
        spaces.iter().map(|s| mfem_quad_orders(s.order(), D, tensor)).collect();

    // 7. Essential boundary dofs per level (all boundary attributes essential;
    //    AbsL1GeometricMultigrid determines the dofs per level).
    let tags: Vec<i32> = meshes[0].unique_boundary_tags();
    let ess: Vec<Vec<u32>> = spaces
        .iter()
        .map(|s| {
            if tags.is_empty() {
                vec![]
            } else {
                boundary_dofs(s.mesh(), s.dof_manager(), &tags)
            }
        })
        .collect();

    // 8. Linear system on the finest level: a(.,.) with the level integrator
    //    and b(.) = (f, v); x carries the boundary-projected exact solution.
    let u = |x: &[f64]| diffusion_solution::<D>(kappa, x);
    let q_fin = quad.last().unwrap();
    let mass = MassIntegrator { rho: 1.0 };
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let finest_space = spaces.last().unwrap();
    let n = finest_space.n_dofs();
    let (integ, quad_bilin): (&dyn fem_assembly::BilinearIntegrator, u8) =
        match cfg.integrator_type {
            IntegratorType::Mass => (&mass, q_fin.mass),
            IntegratorType::Diffusion => (&diff, q_fin.diffusion),
        };
    let mut a: CsrMatrix<f64> =
        Assembler::assemble_bilinear(finest_space, &[integ], quad_bilin);
    let mut b = match cfg.integrator_type {
        IntegratorType::Mass => Assembler::assemble_linear(
            finest_space,
            &[&DomainSourceIntegrator::new(u)],
            q_fin.domain_lf,
        ),
        IntegratorType::Diffusion => Assembler::assemble_linear(
            finest_space,
            &[&DomainSourceIntegrator::new(|x: &[f64]| {
                diffusion_source::<D>(kappa, x)
            })],
            q_fin.domain_lf,
        ),
    };
    let mut x = {
        let dm = finest_space.dof_manager();
        let mut gf = GridFunction::new(finest_space, vec![0.0; n]);
        gf.project_bdr_coefficient(&u, &tags, dm);
        gf.dofs().to_vec()
    };

    // 9. Geometric multigrid with the AbsL1 Jacobi level smoothers
    //    (ds_common::AbsL1GeometricMultigrid::ConstructBilinearForm).  All
    //    level matrices are eliminated under DIAG_KEEP; the finest one is
    //    eliminated once — through `FormFineLinearSystem`, together with `b` —
    //    and shared by the outer solver and the finest MG level.
    let ess_fin = ess.last().unwrap().clone();
    let ess_vals: Vec<f64> = ess_fin.iter().map(|&d| x[d as usize]).collect();
    AbsL1GeometricMultigrid::form_fine_linear_system(&mut a, &mut b, &ess_fin, &ess_vals);

    let mut level_mats: Vec<CsrMatrix<f64>> = Vec::with_capacity(n_levels - 1);
    for (l, space) in spaces.iter().enumerate() {
        if l + 1 == n_levels {
            break; // the finest level shares the eliminated `a`
        }
        let (integ, qb): (&dyn fem_assembly::BilinearIntegrator, u8) =
            match cfg.integrator_type {
                IntegratorType::Mass => (&mass, quad[l].mass),
                IntegratorType::Diffusion => (&diff, quad[l].diffusion),
            };
        let mut mat = Assembler::assemble_bilinear(space, &[integ], qb);
        let mut dummy = vec![0.0f64; mat.nrows];
        for &d in &ess[l] {
            mat.apply_dirichlet_keep_diag(d as usize, 0.0, &mut dummy);
        }
        level_mats.push(mat);
    }

    let coarse_mat = if n_levels == 1 { a.clone() } else { level_mats[0].clone() };
    let mut mg = AbsL1GeometricMultigrid::new(coarse_mat, ess[0].clone());
    for l in 1..n_levels {
        let prolong = build_prolongation::<D>(
            &meshes[l - 1],
            spaces[l - 1].dof_manager(),
            &meshes[l],
            spaces[l].dof_manager(),
            orders[l - 1],
        );
        let mat = if l + 1 == n_levels { a.clone() } else { level_mats[l].clone() };
        mg.add_fine_level(mat, ess[l].clone(), prolong);
    }
    mg.set_cycle_type(MgCycleType::V, 1, 1);
    // The `-s` solver type selects the coarse-level solver as well
    // (ds-common ConstructCoarseOperatorAndSolver).
    mg.set_coarse_solver_type(match cfg.solver_type {
        SolverType::Sli => MgCoarseSolverType::Sli,
        SolverType::Cg => MgCoarseSolverType::Cg,
    });

    // DataMonitor (ds-common.cpp): CSV of `(it, res, sol)` per iteration at
    // fixed 20-digit precision.
    let mut monitor_file = if cfg.use_monitor {
        let name = format!(
            "MGABS-G{}O{}O{}I{}S{}A{}.csv",
            cfg.geometric_levels,
            cfg.order_levels,
            cfg.order,
            match cfg.integrator_type {
                IntegratorType::Mass => 0,
                IntegratorType::Diffusion => 1,
            },
            match cfg.solver_type {
                SolverType::Sli => 0,
                SolverType::Cg => 1,
            },
            cfg.assembly_type_int
        );
        println!("Saving iterations into: {name}");
        let mut f = std::fs::File::create(&name).expect("cannot create monitor file");
        writeln!(f, "it,res,sol").expect("monitor write failed");
        Some(f)
    } else {
        None
    };
    let mut monitor = move |it: i32, norm: f64, _final: bool| {
        if let Some(f) = monitor_file.as_mut() {
            let _ = writeln!(f, "{it},{norm:.20},{norm:.20}");
        }
    };
    let monitor_ref: Option<&mut dyn FnMut(i32, f64, bool)> = Some(&mut monitor);

    // Solve (the MG is the preconditioner; the MFEM iteration log is printed
    // by `solve_sli` / `solve_cg_mfem`).
    let opts = SliOptions {
        rel_tol: cfg.rel_tol,
        abs_tol: 0.0,
        max_iter: cfg.max_iter,
        print_level: 1,
    };
    let a_ref = &a;
    let apply = move |xin: &[f64], yout: &mut [f64]| a_ref.spmv(xin, yout);
    let mg_ref = &mg;
    let prec = move |r: &[f64], z: &mut [f64]| mg_ref.mult(r, z);
    let _: IterResult =
        match cfg.solver_type {
            SolverType::Sli => {
                solve_sli(n, apply, &b, &mut x, Some(prec), &opts, true, monitor_ref)
            }
            SolverType::Cg => {
                solve_cg_mfem(n, apply, &b, &mut x, Some(prec), &opts, true, monitor_ref)
            }
        };

    // 10. RecoverFineFEMSolution: for the in-place conforming serial solve the
    //     recovered FEM solution *is* `x` (`RecoverFEMSolution` copies
    //     `x = X`), so nothing to do here.

    // 11. Compute and print the L^2 norm of the error.
    let gf = GridFunction::new(finest_space, x);
    let err = gf.compute_l2_error(&u, q_fin.l2_error);
    println!("\n|| u_h - u ||_{{L^2}} = {}\n", fmt_g(err));

    if cfg.visualization {
        eprintln!("note: GLVis streaming is not reproduced by the fem-rs port");
    }
}

/// Coarse→fine prolongation between consecutive hierarchy levels.  Order-1
/// levels use the exact dyadic P1 refinement path (bitwise-equal to MFEM's
/// `RefinementOperator` on tri/quad/tet/hex meshes); higher-order nested
/// levels use fem-space's `build_h1_prolongation_matrix` (2-D nested,
/// tetrahedral nested, and same-mesh p-refinement paths) or — for nested
/// hexahedral levels, where fem-space has no locator — the fem-solver hex
/// path (nodal interpolation via Newton inversion of the trilinear map).
fn build_prolongation<const D: usize>(
    coarse_mesh: &Mesh<D>,
    coarse_dm: &DofManager,
    fine_mesh: &Mesh<D>,
    fine_dm: &DofManager,
    coarse_order: u8,
) -> CsrMatrix<f64> {
    if coarse_order == 1 {
        return build_h1_p1_refined_prolongation(
            coarse_mesh,
            &|d| {
                let c = fine_dm.dof_coord(d);
                [
                    c[0],
                    if D > 1 { c[1] } else { 0.0 },
                    if D > 2 { c[2] } else { 0.0 },
                ]
            },
            &|e| coarse_dm.element_dofs(e).to_vec(),
            fine_dm.n_dofs,
        );
    }
    let same_geom = coarse_mesh.n_elements() == fine_mesh.n_elements();
    if D == 3 && !same_geom && coarse_mesh.element_type(0) == fem_mesh::ElementType::Hex8 {
        return build_h1_hex_refined_prolongation(
            coarse_mesh,
            coarse_order,
            coarse_dm.n_dofs,
            fine_dm.n_dofs,
            &|e| coarse_dm.element_dofs(e).to_vec(),
            &|d| {
                let c = fine_dm.dof_coord(d);
                [c[0], c[1], c[2]]
            },
        );
    }
    build_h1_prolongation_matrix(coarse_mesh, coarse_dm, fine_mesh, fine_dm)
}
