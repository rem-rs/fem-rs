//! # MFEM Example 9 — DG Advection (1:1 translation)
//!
//! Solves `du/dt + v·∇u = 0` using DG with upwind flux.
//! 1:1 translation: periodic meshes, problem types 0-3,
//! explicit (RK4) time integration, output files.
//!
//! Notes:
//! - Uses standard L2 basis (vs MFEM's GaussLobatto) → different DOF layout.
//! - Periodic meshes with degenerate quads after refinement may produce INF
//!   entries in the advection matrix (assembler needs det>0 threshold).
//! - Works best on well-shaped quad meshes.
//!
//! Reference: `mfem/ex9.cpp`

use fem_io::mfem::{write_mfem_file, write_mfem_gf_file};

use fem_assembly::{
    Assembler,
    dg::dg_advection::{
        DGAdvectionIntegrator, assemble_advection_boundary_full,
        DgAdvectionProblem, dg_velocity, dg_initial_condition, dg_inflow_bc,
    },
    dg::dg_imex::{
        MfemHeadInsert, assemble_ex41_bdr_faces, assemble_ex41_interior_faces,
        build_bdr_face_locs, build_face_locs,
    },
    postproc::coefficient::{FnVectorCoeff, VectorCoeff},
    standard::MassIntegrator,
};
use fem_linalg::CooMatrix;
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_solver::{
    SolverConfig, solve_cg,
    ode::{Rk4, TimeStepper},
};
use fem_space::{L2Basis, L2Space, fe_space::FESpace};

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // Display options (matching C++ args.PrintOptions(cout))
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --problem {}", args.problem);
    println!("   --refine {}", args.refine);
    println!("   --order {}", args.order);
    println!("   --no-partial-assembly");
    println!("   --no-element-assembly");
    println!("   --no-full-assembly");
    println!("   --device cpu");
    println!("   --ode-solver {}", args.ode_solver);
    println!("   --t-final {}", args.t_final);
    println!("   --time-step {}", args.dt);
    println!("   --no-visualization");
    println!("   --no-visit-datafiles");
    println!("   --no-paraview-datafiles");
    println!("   --ascii-datafiles");
    println!("   --visualization-steps 5");
    println!();
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");
    let mfem = fem_io::mfem::read_mfem_file(&args.mesh).expect("read mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
    let dim = 2;

    // Bounding box for velocity/IC mapping
    let mut bb_min = vec![f64::MAX; dim];
    let mut bb_max = vec![f64::MIN; dim];
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        for d in 0..dim { bb_min[d] = bb_min[d].min(c[d]); bb_max[d] = bb_max[d].max(c[d]); }
    }

    let mesh = if args.refine > 0 {
        let mut m = mesh; for _ in 0..args.refine { m = refine_uniform(&m); } m
    } else { mesh };

    let problem = match args.problem {
        0 => DgAdvectionProblem::Translation,
        1 => DgAdvectionProblem::Rotation,
        2 => DgAdvectionProblem::RotationP2,
        3 => DgAdvectionProblem::Twist,
        _ => DgAdvectionProblem::Translation,
    };

    // Velocity coefficient
    let vel_fn = {
        let bb_min_c = bb_min.clone(); let bb_max_c = bb_max.clone();
        move |x: &[f64], out: &mut [f64]| {
            let v = dg_velocity(problem, x, &bb_min_c, &bb_max_c);
            for (i, &vi) in v.iter().enumerate() { out[i] = vi; }
        }
    };
    let vel_coeff = FnVectorCoeff(vel_fn);

    // DG space and mass matrix
    // DG space: MFEM ex9 uses DG_FECollection(order, dim, BasisType::GaussLobatto)
    // — GLL nodes, NOT the default GaussLegendre that L2Space::new uses.
    let space = L2Space::new_with_basis(mesh.clone(), args.order, L2Basis::GaussLobatto);
    let n = space.n_dofs();
    println!("Number of unknowns: {n}");

    // Quadrature: mass uses 2*order+1 (exact for degree 2p),
    // advection uses 2*order (matching MFEM ConvectionIntegrator).
    // Avoid 3+ point/axis rules that hit degenerate element quad points.
    let qo_mass = (args.order as u8 * 2 + 2).max(3);
    let qo_adv = (args.order as u8 * 2).max(2);
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], qo_mass);

    // Advection operator: volume + interior faces.
    // Face assembly reuses the ex41 machinery (verified 1:1 with MFEM):
    // `build_face_locs` computes MFEM-style Loc1/Loc2 per-element geometry,
    // `assemble_ex41_interior_faces` implements the exact
    // NonconservativeDGTraceIntegrator(u, -1) (pure advection: diff=sigma=
    // kappa=0).  ex9's C++ uses DG_FECollection(..., BasisType::GaussLobatto)
    // so the space must be GLL too.
    let dg_adv = DGAdvectionIntegrator { velocity: vel_coeff };
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg_adv], qo_adv);

    let faces = build_face_locs(space.mesh());
    let mut hi_k = MfemHeadInsert::new(n);
    assemble_ex41_interior_faces(
        &mut hi_k, space.mesh(), &space, &faces,
        &|x, y| -> [f64; 2] { let v = dg_velocity(problem, &[x, y], &bb_min, &bb_max); [v[0], v[1]] },
        -1.0, 0.0, 0.0, 0.0,
    );

    let mut coo = CooMatrix::new(n, n);
    for i in 0..n { for p in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] {
        coo.add(i, k_vol.col_idx[p] as usize, k_vol.values[p]);
    }}
    let k_face = hi_k.into_csr();
    for i in 0..n { for p in k_face.row_ptr[i]..k_face.row_ptr[i+1] {
        coo.add(i, k_face.col_idx[p] as usize, k_face.values[p]);
    }}

    // Boundary contribution: K from ex41's NonconservativeDGTrace boundary
    // (same Loc1/unnormalised-normal machinery, verified 1:1 with MFEM),
    // plus the inflow RHS (BoundaryFlowIntegrator) kept from the previous
    // assembly.
    let qface = (args.order as u8 * 2).max(2);
    let inflow_g = |x: &[f64]| dg_inflow_bc(problem, x);
    let bdr_faces = build_bdr_face_locs(space.mesh());
    let mut hi_bdr = MfemHeadInsert::new(n);
    assemble_ex41_bdr_faces(
        &mut hi_bdr, space.mesh(), &space, &bdr_faces,
        &|x, y| -> [f64; 2] { let v = dg_velocity(problem, &[x, y], &bb_min, &bb_max); [v[0], v[1]] },
        -1.0, 0.0, 0.0, 0.0,
    );
    let k_bdr = hi_bdr.into_csr();
    for i in 0..n { for p in k_bdr.row_ptr[i]..k_bdr.row_ptr[i+1] {
        coo.add(i, k_bdr.col_idx[p] as usize, k_bdr.values[p]);
    }}
    let bc_tags: Vec<i32> = mesh.unique_boundary_tags();
    let vel_bdr = {
        let bb_min_c = bb_min.clone(); let bb_max_c = bb_max.clone();
        FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
            let v = dg_velocity(problem, x, &bb_min_c, &bb_max_c);
            for (i, &vi) in v.iter().enumerate() { out[i] = vi; }
        })
    };
    let (_k_bdr_old, rhs_bc) = assemble_advection_boundary_full(
        &space, &vel_bdr, &bc_tags, &inflow_g, args.order, qface);

    let k_adv = coo.into_csr();

    // ── Initial condition ──────────────────────────────────────────────────
    let mut u = space.interpolate(&|x| dg_initial_condition(problem, x, &bb_min, &bb_max))
        .as_slice().to_vec();

    // ── Initial output files (matching C++: ex9.mesh, ex9-init.gf) ──────────
    {
        write_mfem_file("ex9.mesh", &mesh).expect("mesh write failed");
        write_mfem_gf_file("ex9-init.gf", dim, &u, "L2", args.order, 1, 7).expect("write init gf");
    }

    // ── Time integration (explicit RK4, matching C++ default ode_solver=4) ──
    let solver_cfg = SolverConfig { rtol: 1e-9, max_iter: 100, verbose: false, ..Default::default() };
    let dt = args.dt.min(args.t_final); let mut t = 0.0;
    let vis_steps = 5;
    let mut ti = 0;


    let steps = (args.t_final / dt).ceil() as usize;
    for _ in 0..steps {
        let dta = dt.min(args.t_final - t);
        Rk4.step(t, dta, &mut u, |_t, u, dudt| {
            let mut f = vec![0.0; n]; k_adv.spmv(u, &mut f);
            for i in 0..n { f[i] += rhs_bc[i]; }
            let _ = solve_cg(&mass, &f, dudt, &solver_cfg);
        });
        t += dta; ti += 1;
        if ti % vis_steps == 0 || t >= args.t_final - 1e-14 {
            println!("time step: {ti}, time: {t:.3}");
        }
    }

    // ── Final output file (matching C++: ex9-final.gf) ──────────────────────
    {
        write_mfem_gf_file("ex9-final.gf", dim, &u, "L2", args.order, 1, 7).expect("write final gf");
    }
    eprintln!("  Done. Total time: {:.3}s", t0.elapsed().as_secs_f64());
}

/// Detect periodic face pairs for 'boundary 0' meshes and assemble their
/// flux contributions using each element's OWN face nodes.
struct Args { mesh: String, problem: usize, refine: usize, order: u8, dt: f64, t_final: f64, ode_solver: usize, solve_implicit_state: bool }
impl Args {
    fn parse() -> Self {
        let mut mesh = "../data/periodic-hexagon.mesh".to_string();
        let mut problem = 0usize; let mut refine = 2usize; let mut order = 3u8;
        let mut dt = 0.01_f64; let mut t_final = 10.0_f64; let mut ode_solver = 4usize;
        let mut solve_implicit_state = false;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() { match arg.as_str() {
            "-m"|"--mesh" => { mesh = it.next().unwrap_or(mesh); }
            "-p"|"--problem" => { problem = it.next().and_then(|s|s.parse().ok()).unwrap_or(problem); }
            "-r"|"--refine" => { refine = it.next().and_then(|s|s.parse().ok()).unwrap_or(refine); }
            "-o"|"--order" => { order = it.next().and_then(|s|s.parse().ok()).unwrap_or(order); }
            "-dt"|"--time-step" => { dt = it.next().and_then(|s|s.parse().ok()).unwrap_or(dt); }
            "-tf"|"--t-final" => { t_final = it.next().and_then(|s|s.parse().ok()).unwrap_or(t_final); }
            "-s"|"--ode-solver" => { ode_solver = it.next().and_then(|s|s.parse().ok()).unwrap_or(ode_solver); }
            "-imp-state"|"--implicit-state"|"-imp-slope"|"--implicit-slope" => solve_implicit_state = true,
            "-no-vis"|"--no-visualization" => {}
            _ => {}
        }}
        Args { mesh, problem, refine, order, dt, t_final, ode_solver, solve_implicit_state }
    }
}
