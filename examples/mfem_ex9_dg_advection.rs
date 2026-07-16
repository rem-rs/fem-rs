//! # MFEM Example 9 — DG Advection (1:1 translation)
//!
//! Solves the time-dependent advection equation `du/dt + v·∇u = 0` using a
//! Discontinuous Galerkin (DG) discretization.
//!
//! Features:
//! - Problem types 0-3 (translation, rotation, twist)
//! - Configurable polynomial order (`-o`)
//! - Explicit (RK4) or implicit (GMRES+ILU) time integration
//! - GLVis visualization
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex9_dg_advection -- -m data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 1
//! cargo run --example mfem_ex9_dg_advection -- -m data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 1
//! cargo run --example mfem_ex9_dg_advection -- -m data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//! cargo run --example mfem_ex9_dg_advection -- -m data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 1 -o 2
//! ```
//!
//! Reference: `mfem/ex9.cpp` (DG Advection)

use fem_assembly::{
    Assembler,
    dg::dg_advection::{
        DGAdvectionIntegrator, assemble_dg_interior_faces, assemble_advection_boundary_full,
        DgAdvectionProblem, dg_velocity, dg_initial_condition, dg_inflow_bc,
    },
    interior_faces::InteriorFaceList,
    postproc::coefficient::FnVectorCoeff,
    standard::MassIntegrator,
};
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_solver::{
    SolverConfig, solve_cg,
    ode::{Rk4, ForwardEuler, TimeStepper, ImplicitTimeStepper},
};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // ── 1. Read mesh ──────────────────────────────────────────────────────────
    let mfem = fem_io::mfem::read_mfem_file(&args.mesh).expect("read mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
    let dim = 2;

    // ── 1b. Detect periodic boundaries (for 'boundary 0' periodic meshes) ─────
    let mesh = if mesh.n_boundary_faces() == 0 {
        match mesh.detect_periodic_boundary(1e-8) {
            Ok(pm) => pm,
            Err(e) => { eprintln!("  Periodic boundary detection: {e}"); mesh }
        }
    } else { mesh };

    // ── 2. Refine ─────────────────────────────────────────────────────────────
    let mesh = if args.refine > 0 {
        let mut m = mesh;
        for _ in 0..args.refine { m = refine_uniform(&m); }
        m
    } else { mesh };

    // Bounding box for velocity/IC mapping to [-1,1]^d
    let mut bb_min = vec![f64::MAX; dim];
    let mut bb_max = vec![f64::MIN; dim];
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            bb_min[d] = bb_min[d].min(c[d]);
            bb_max[d] = bb_max[d].max(c[d]);
        }
    }

    // ── 3. DG space ───────────────────────────────────────────────────────────
    let space = L2Space::new(mesh.clone(), args.order);
    let n = space.n_dofs();
    println!("Number of unknowns: {n}");

    // ── 4. Problem setup ──────────────────────────────────────────────────────
    let problem = match args.problem {
        0 => DgAdvectionProblem::Translation,
        1 => DgAdvectionProblem::Rotation,
        2 => DgAdvectionProblem::RotationP2,
        3 => DgAdvectionProblem::Twist,
        _ => DgAdvectionProblem::Translation,
    };

    // Velocity as a callable coefficient (matches MFEM velocity_function)
    let vel_coeff = FnVectorCoeff({
        let bb_min_c = bb_min.clone();
        let bb_max_c = bb_max.clone();
        move |x: &[f64], out: &mut [f64]| {
            let v = dg_velocity(problem, x, &bb_min_c, &bb_max_c);
            for (i, &vi) in v.iter().enumerate() { out[i] = vi; }
        }
    });

    // Inflow BC (all zero for MFEM ex9 problems)
    let inflow_fn = |x: &[f64]| dg_inflow_bc(problem, x);

    // ── 5. Assemble M (mass) ──────────────────────────────────────────────────
    let qo = (args.order as u8 * 2 + 2).max(3);
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], qo);

    // ── 6. Assemble K (advection: volume + interior faces + boundary) ─────────
    let dg_adv = DGAdvectionIntegrator { velocity: vel_coeff };

    // Volume term
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg_adv], qo);

    // Interior face term
    let ifl = InteriorFaceList::build(space.mesh());
    let qface = (args.order as u8 * 2).max(2);
    let mut coo = fem_linalg::CooMatrix::new(n, n);
    // Copy volume terms
    for i in 0..n {
        for p in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] {
            coo.add(i, k_vol.col_idx[p] as usize, k_vol.values[p]);
        }
    }
    assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, args.order, qface, &dg_adv);

    // Boundary term using the velocity coefficient (rebuild with owned data for the closure)
    let vel_coeff2 = FnVectorCoeff({
        let bb_min_c = bb_min.clone();
        let bb_max_c = bb_max.clone();
        move |x: &[f64], out: &mut [f64]| {
            let v = dg_velocity(problem, x, &bb_min_c, &bb_max_c);
            for (i, &vi) in v.iter().enumerate() { out[i] = vi; }
        }
    });
    let bc_tags: Vec<i32> = mesh.unique_boundary_tags();
    let (k_bdr, rhs_bc) = assemble_advection_boundary_full(
        &space, &vel_coeff2, &bc_tags, &inflow_fn, args.order, qface,
    );
    // Add boundary K
    for i in 0..n {
        for p in k_bdr.row_ptr[i]..k_bdr.row_ptr[i+1] {
            coo.add(i, k_bdr.col_idx[p] as usize, k_bdr.values[p]);
        }
    }
    let k_adv = coo.into_csr();

    // ── 7. Initial condition ──────────────────────────────────────────────────
    // Note: C++ uses DG_FECollection(BasisType::GaussLobatto) which makes
    // ProjectCoefficient equivalent to pointwise interpolation for P1.
    let mut u = space.interpolate(&|x| dg_initial_condition(problem, x, &bb_min, &bb_max))
        .as_slice().to_vec();

    // ── 8. Time integration ────────────────────────────────────────────────────
    let cfg = SolverConfig { rtol: 1e-9, max_iter: 100, verbose: false, ..Default::default() };
    let dt = args.dt.min(args.t_final);
    let mut t = 0.0;
    let mut ti = 0;
    let vis_steps = 5;

    // GLVis
    let use_vis = args.visualize;
    let mut sout: Option<std::net::TcpStream> = if use_vis {
        match std::net::TcpStream::connect("localhost:19916") {
            Ok(s) => { println!("GLVis visualization paused."); Some(s) }
            Err(_) => { println!("Unable to connect to GLVis; visualization disabled."); None }
        }
    } else { None };

    let _ = sout.as_ref().map(|s| {
        use std::io::Write;
        let mut s = s.try_clone().unwrap();
        let _ = write!(s, "solution\n{}", "pause\n");
    });

    let steps = (args.t_final / dt).ceil() as usize;
    for step in 0..steps {
        let dta = dt.min(args.t_final - t);

        match args.ode_solver {
            1 => {
                // Backward Euler (implicit)
                let mat_mass = &mass;
                let mat_k = &k_adv;
                let inner_cfg = &cfg;
                let rhs_adv = |_t: f64, u: &[f64], dudt: &mut [f64]| {
                    k_adv.spmv(u, dudt);
                    for i in 0..n { dudt[i] += rhs_bc[i]; }
                };
                let jac = |_t: f64, _u: &[f64]| {
                    let mut a = mass.clone();
                    for i in 0..n {
                        for p in a.row_ptr[i]..a.row_ptr[i+1] {
                            let col = a.col_idx[p] as usize;
                            a.values[p] += dta * k_adv.get(i, col);
                        }
                    }
                    a
                };
                use fem_solver::ode::ImplicitEuler;
                ImplicitEuler.step_implicit(t, dta, &mut u, rhs_adv, jac);
            }
            4 | _ => {
                // Explicit RK4: du/dt = M^{-1}(K·u + rhs_bc)
                Rk4.step(t, dta, &mut u, |_t: f64, u: &[f64], dudt: &mut [f64]| {
                    let mut f = vec![0.0; n];
                    k_adv.spmv(u, &mut f);
                    for i in 0..n { f[i] += rhs_bc[i]; }
                    let _ = solve_cg(&mass, &f, dudt, &cfg);
                });
            }
        }

        t += dta;
        ti = step + 1;

        if (step + 1) % vis_steps == 0 || t >= args.t_final - 1e-14 {
            println!("time step: {ti}, time: {t:.3}");
            if let Some(ref mut s) = sout {
                use std::io::Write;
                let _ = write!(s, "solution\n");
            }
        }
    }

    // ── 9. Final output ──────────────────────────────────────────────────────
    let mut mu = vec![0.0; n];
    mass.spmv(&u, &mut mu);
    let l2 = u.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>().sqrt();
    println!("L2 = {l2:.10e}");

    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh: String,
    problem: usize,
    refine: usize,
    order: u8,
    dt: f64,
    t_final: f64,
    ode_solver: usize,
    visualize: bool,
}

impl Args {
    fn parse() -> Self {
        let mut mesh = "../data/periodic-hexagon.mesh".to_string();
        let mut problem = 0usize;
        let mut refine = 2usize;
        let mut order = 3u8;
        let mut dt = 0.01_f64;
        let mut t_final = 10.0_f64;
        let mut ode_solver = 4usize;
        let mut visualize = true;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => { mesh = it.next().unwrap_or(mesh); }
                "-p" | "--problem" => { problem = it.next().and_then(|s| s.parse().ok()).unwrap_or(problem); }
                "-r" | "--refine" => { refine = it.next().and_then(|s| s.parse().ok()).unwrap_or(refine); }
                "-o" | "--order" => { order = it.next().and_then(|s| s.parse().ok()).unwrap_or(order); }
                "-dt" | "--time-step" => { dt = it.next().and_then(|s| s.parse().ok()).unwrap_or(dt); }
                "-tf" | "--t-final" => { t_final = it.next().and_then(|s| s.parse().ok()).unwrap_or(t_final); }
                "-s" | "--ode-solver" => { ode_solver = it.next().and_then(|s| s.parse().ok()).unwrap_or(ode_solver); }
                "-no-vis" | "--no-visualization" => { visualize = false; }
                _ => {}
            }
        }
        Args { mesh, problem, refine, order, dt, t_final, ode_solver, visualize }
    }
}
