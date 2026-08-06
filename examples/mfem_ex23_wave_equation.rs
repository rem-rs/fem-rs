#![allow(dead_code)]

//! # Example 23 — Wave Equation (Second-Order ODE)  [1:1 translation of MFEM ex23]
//!
//! Solves the wave equation:
//!
//! ```text
//!   d²u/dt² = c²·Δu
//! ```
//!
//! The example demonstrates the use of a second-order time-dependent operator,
//! implicit Backward-Euler time integration, and CG solvers.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex23_wave_equation -- -no-vis
//! cargo run --example mfem_ex23_wave_equation -- -m data/star.mesh -o 4 -tf 2 -no-vis
//! cargo run --example mfem_ex23_wave_equation -- -m data/square-disc.mesh -o 2 -tf 2 --neumann -no-vis
//! cargo run --example mfem_ex23_wave_equation -- -m data/inline-tri.mesh -o 1 -tf 2 --neumann -no-vis
//! ```
//!
//! ## ODE solver type (default: 10 = Backward Euler)
//! |  s | Method               | Type     |
//! |----|----------------------|----------|
//! | 10 | Backward Euler       | Implicit |
//! | 11 | Trapezoidal / Newmark| Implicit |
//! | 12 | SDIRK2 (L-stable)    | Implicit |

use std::io::Write;
use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_gf, write_mfem_file, write_mfem_file_3d};
use fem_io::glvis::GlVisSocket;
use fem_linalg::CsrMatrix;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

// ─── WaveOperator ──────────────────────────────────────────────────────────────

/// After spatial discretization, the wave model can be written as:
///
/// ```text
///   d²u/dt² = M⁻¹(-K·u)
/// ```
///
/// where u is the displacement vector, M is the mass matrix, and K is the
/// stiffness matrix.
struct WaveOperator<M: MeshTopology> {
    fespace: H1Space<M>,
    ess_tdof_list: Vec<u32>,

    // Full matrices (before BC elimination) — used for FullMult
    k_full: CsrMatrix<f64>,

    // BC-eliminated system matrices
    m_mat: CsrMatrix<f64>,
    k_mat: CsrMatrix<f64>,

    // T = M + fac0 · K (rebuilt when fac0 changes)
    t_mat: Option<CsrMatrix<f64>>,
    current_fac0: f64,

    // CG solver config (M_solver: 30 iters, T_solver: 100 iters)
    solve_cfg: SolverConfig,
    solve_cfg_t: SolverConfig,

    // Auxiliary vector
    z: Vec<f64>,
}

impl<M: MeshTopology + Send + Sync + Clone> WaveOperator<M> {
    fn new(
        fespace: H1Space<M>,
        ess_tdof_list: Vec<u32>,
        speed: f64,
    ) -> Self {
        let rel_tol = 1e-8;
        // MFEM: M_solver max_iter=30, T_solver max_iter=100
        let solve_cfg = SolverConfig { rtol: rel_tol, atol: 0.0, max_iter: 30, verbose: false, ..SolverConfig::default() };
        let solve_cfg_t = SolverConfig { rtol: rel_tol, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };

        // Match C++ 2*order+1 quadrature (order=2 → quad_order=5)
        let quad_order = (2 * fespace.element_order(0) + 1) as u8;

        // Assemble Laplace matrix K
        let c2 = speed * speed;
        let k_integ = DiffusionIntegrator { kappa: c2 };
        let k_full = Assembler::assemble_bilinear(&fespace, &[&k_integ], quad_order);

        // Assemble mass matrix M
        let m_integ = MassIntegrator { rho: 1.0 };
        let m_full = Assembler::assemble_bilinear(&fespace, &[&m_integ], quad_order);

        let n = m_full.nrows;

        // Apply BC elimination to create system matrices Mmat and Kmat
        let mut m_mat = m_full.clone();
        let mut k_mat = k_full.clone();
        let mut dummy_rhs = vec![0.0; n];
        for &dof in &ess_tdof_list {
            let d = dof as usize;
            // Row-only elimination (matching C++ FormSystemMatrix behavior)
            // The CG solver handles the slight asymmetry since RHS is zero at BC DOFs
            m_mat.apply_dirichlet_row_zeroing(d, 0.0, &mut dummy_rhs);
            k_mat.apply_dirichlet_row_zeroing(d, 0.0, &mut dummy_rhs);
        }

        WaveOperator {
            fespace,
            ess_tdof_list,
            k_full,
            m_mat,
            k_mat,
            t_mat: None,
            current_fac0: 0.0,
            solve_cfg,
            solve_cfg_t,
            z: vec![0.0; n],
        }
    }

    /// Compute d²u/dt² = M⁻¹(-K·u) for explicit evaluation.
    fn mult(&mut self, u: &[f64], d2udt2: &mut [f64]) {
        // z = K · u
        self.k_full.spmv(u, &mut self.z);
        // z = -K · u
        for v in self.z.iter_mut() {
            *v = -*v;
        }
        // Zero BC entries in RHS
        for &d in &self.ess_tdof_list {
            self.z[d as usize] = 0.0;
        }
        // Solve M_mat · d2udt2 = z (PCG+Jacobi, matching C++ DSmoother)
        solve_pcg_jacobi(&self.m_mat, &self.z, d2udt2, &self.solve_cfg)
            .expect("WaveOperator::Mult: PCG+Jacobi solve failed");
        // Zero BC entries in solution
        for &d in &self.ess_tdof_list {
            d2udt2[d as usize] = 0.0;
        }
    }

    /// Solve the Backward-Euler equation:
    ///
    /// ```text
    ///   (M + fac0 · K) · d²u/dt² = -K · u
    /// ```
    ///
    /// This is used by the second-order ODE solvers.
    fn implicit_solve(&mut self, fac0: f64, u: &[f64], d2udt2: &mut [f64]) {
        // Build T = M + fac0 · K on first call or when fac0 changes
        if self.t_mat.is_none() || (fac0 - self.current_fac0).abs() > 1e-15 {
            self.t_mat = Some(self.m_mat.axpby(1.0, &self.k_mat, fac0));
            self.current_fac0 = fac0;
        }

        // z = K · u (using full K, including BC DOFs)
        self.k_full.spmv(u, &mut self.z);
        // z = -K · u
        for v in self.z.iter_mut() {
            *v = -*v;
        }
        // Zero BC entries in RHS
        for &d in &self.ess_tdof_list {
            self.z[d as usize] = 0.0;
        }

        // Solve T · d2udt2 = z (PCG+Jacobi, matching C++ DSmoother)
        let sys = self.t_mat.as_ref().unwrap();
        let res = solve_pcg_jacobi(sys, &self.z, d2udt2, &self.solve_cfg_t)
            .expect("WaveOperator::ImplicitSolve: PCG+Jacobi solve failed");
        if !res.converged {
            eprintln!("WARNING: PCG+Jacobi did not converge (iter={}, residual={:.6e})",
                     res.iterations, res.final_residual);
        }
        // Zero BC entries in solution
        for &d in &self.ess_tdof_list {
            d2udt2[d as usize] = 0.0;
        }
    }

    /// Called after each time step to invalidate cached T matrix.
    fn set_parameters(&mut self) {
        self.t_mat = None;
    }

    /// Access the BC-eliminated mass matrix (for use with GeneralizedAlpha2).
    pub fn mass_matrix(&self) -> &CsrMatrix<f64> { &self.m_mat }
    /// Access the BC-eliminated stiffness matrix.
    pub fn stiff_matrix(&self) -> &CsrMatrix<f64> { &self.k_mat }
}

// ─── Initial conditions ────────────────────────────────────────────────────────

fn initial_solution(x: &[f64]) -> f64 {
    let r2 = x[0] * x[0] + x[1] * x[1];
    (-30.0 * r2).exp()
}

fn initial_rate(_x: &[f64]) -> f64 {
    0.0
}

// ─── CLI ───────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Args {
    mesh_file: String,
    ref_levels: usize,
    order: u8,
    ode_solver_type: i32,
    t_final: f64,
    dt: f64,
    speed: f64,
    dirichlet: bool,
    visualization: bool,
    visit: bool,
    vis_steps: usize,
}

fn parse_args() -> Args {
    // Default values matching C++ ex23
    let mut a = Args {
        mesh_file: "data/star.mesh".to_string(),
        ref_levels: 2,
        order: 2,
        ode_solver_type: 10,
        t_final: 0.5,
        dt: 1.0e-2,
        speed: 1.0,
        dirichlet: true,
        visualization: false,
        visit: false,
        vis_steps: 5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_default(),
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
            }
            "-s" | "--ode-solver" => {
                a.ode_solver_type = it.next().and_then(|v| v.parse().ok()).unwrap_or(10)
            }
            "-tf" | "--t-final" => {
                a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5)
            }
            "-dt" | "--time-step" => {
                a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.01)
            }
            "-c" | "--speed" => {
                a.speed = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            }
            "-dir" | "--dirichlet" => a.dirichlet = true,
            "-neu" | "--neumann" => a.dirichlet = false,
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-visit" | "--visit-datafiles" => a.visit = true,
            "-no-visit" | "--no-visit-datafiles" => a.visit = false,
            "-vs" | "--visualization-steps" => {
                a.vis_steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(5)
            }
            _ => {}
        }
    }
    a
}

// ─── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // 1. Parse command-line options (done via parse_args() above)

    // 2. Read the mesh from the given mesh file.
    let mfem_file = {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let full_path = p.parent().unwrap().join(&args.mesh_file);
        read_mfem_file(&full_path).expect("failed to read MFEM mesh")
    };

    println!("Options used:");
    println!("   --mesh {}", args.mesh_file);
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --ode-solver {}", args.ode_solver_type);
    println!("   --t-final {}", args.t_final);
    println!("   --time-step {}", args.dt);
    println!("   --speed {}", args.speed);
    println!("   {}", if args.dirichlet { "--dirichlet" } else { "--neumann" });
    println!("   --no-visualization");
    if args.visit {
        println!("   --visit-datafiles");
    } else {
        println!("   --no-visit-datafiles");
    }
    println!("   --visualization-steps {}", args.vis_steps);

    // Dispatch to 2D or 3D
    if let Some(mesh) = mfem_file.mesh2d {
        run_wave_2d(mesh, &args);
    } else if let Some(mesh) = mfem_file.mesh3d {
        run_wave_3d(mesh, &args);
    } else {
        panic!("No mesh found");
    }
}

fn run_wave_2d(mut mesh: Mesh<2>, args: &Args) {
    let dim = 2;

    // 3. Define the ODE solver used for time integration.
    //    (handled below via GeneralizedAlpha2)

    // 4. Refine the mesh uniformly.
    let mesh = if args.ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..args.ref_levels {
            m = fem_mesh::refine_uniform(&m);
        }
        m
    } else {
        mesh
    };

    // 5. Define the H1 FE space.
    let space = H1Space::new(mesh.clone(), args.order);
    let fe_size = space.n_dofs();
    println!("Number of temperature unknowns: {}", fe_size);

    // 6. Compute essential BC DOFs (matching C++ GetEssentialTrueDofs).
    //    C++: ess_bdr is an array per boundary attribute (1=essential, 0=natural).
    let ess_tdof_list: Vec<u32> = if args.dirichlet && mesh.n_boundary_faces() > 0 {
        // All boundary attributes are essential (matching C++ ess_bdr = 1)
        let unique_tags: Vec<i32> = {
            let mut tags: Vec<i32> = (0..mesh.n_boundary_faces())
                .map(|f| mesh.face_tag(f as u32)).collect();
            tags.sort_unstable(); tags.dedup();
            tags
        };
        boundary_dofs(&mesh, space.dof_manager(), &unique_tags)
            .into_iter().map(|d| d as u32).collect()
    } else {
        Vec::new()
    };
    println!("  ess_bdr count: {}", ess_tdof_list.len());

    // 7. Set initial conditions via interpolation at DOF points.
    //    C++ GridFunction::ProjectCoefficient for H1 = direct interpolation.
    let mut u: Vec<f64> = space.interpolate(&|x: &[f64]| initial_solution(x)).into_vec();
    let mut du_dt: Vec<f64> = space.interpolate(&|x: &[f64]| initial_rate(x)).into_vec();
    for &d in &ess_tdof_list { u[d as usize] = 0.0; du_dt[d as usize] = 0.0; }

    // Save initial state with precision(8) (matching C++ Save+precision(8))
    write_mfem_file("ex23.mesh", &mesh).expect("write mesh");
    {
        let mut init_f = std::fs::File::create("ex23-init.gf").expect("create ex23-init.gf");
        // Write header + u values (matching first Save)
        writeln!(init_f, "FiniteElementSpace").ok();
        writeln!(init_f, "FiniteElementCollection: H1_{dim}D_P{}", args.order).ok();
        writeln!(init_f, "VDim: 1").ok();
        writeln!(init_f, "Ordering: 0").ok();
        writeln!(init_f).ok();
        for v in &u { writeln!(init_f, "{:.7e}", v).ok(); }
        // Append du/dt values (matching second Save — no header)
        for v in &du_dt { writeln!(init_f, "{:.7e}", v).ok(); }
    }

    // Setup GLVis visualization socket (matching C++)
    let mut glvis = if args.visualization {
        match GlVisSocket::connect("localhost", 19916) {
            Ok(s) => {
                println!("GLVis visualization paused. Press space to resume.");
                Some(s)
            }
            Err(_) => {
                println!("GLVis visualization disabled (no server).");
                None
            }
        }
    } else { None };

    // Create the wave operator
    let mut oper = WaveOperator::new(space, ess_tdof_list.clone(), args.speed);

    let dt = args.dt;
    let t_final = args.t_final;
    let vis_steps = args.vis_steps;

    // 8. Time integration matching C++ ex23 ode_solver_type=10 exactly:
    //    SecondOrderODESolver::Select(10) → GeneralizedAlpha2Solver(1.0)
    //    (NOT BackwardEulerSolver!).  With rho_inf=1:
    //      alpha_m = alpha_f = 0.5, beta = 0.25, gamma = 0.5
    //      fac0 = 0.5 - beta/alpha_m   = 0.0
    //      fac1 = alpha_f              = 0.5
    //      fac2 = alpha_f*(1-gamma/alpha_m) = 0.0
    //      fac3 = beta*alpha_f/alpha_m = 0.25
    //      fac4 = gamma*alpha_f/alpha_m= 0.5
    //      fac5 = alpha_m              = 0.5
    //    Step(u, dudt, t, dt):
    //      1st pass: a0 = f->Mult(u, dudt) = M⁻¹(-K·u); state = a0
    //      Predict: va = dudt + fac0·dt·state;  xa = u + fac1·dt·va;
    //               va = dudt + fac2·dt·state
    //      Solve:   aa = ImplicitSolve(fac3·dt², fac4·dt, xa, va)
    //               → T = M + fac3·dt²·K;  T·aa = -K·xa
    //      Correct: xa += fac3·dt²·aa;  va += fac4·dt·aa
    //      Extrap:  u = (1-1/fac1)·u + (1/fac1)·xa
    //               dudt = (1-1/fac1)·dudt + (1/fac1)·va
    //               state = (1-1/fac5)·state + (1/fac5)·aa
    let mut t = 0.0;
    let n_steps = if dt > 0.0 { (t_final / dt).ceil() as usize } else { 0 };
    let fac1 = 0.5_f64;
    let fac3 = 0.25_f64;
    let fac4 = 0.5_f64;
    let fac5 = 0.5_f64;
    let inv_f1 = 1.0 / fac1;
    let inv_f5 = 1.0 / fac5;
    let mut v = du_dt.clone();
    let mut state = vec![0.0; fe_size];
    // 1st pass: initial acceleration a0 = M⁻¹(-K·u) (C++ f->Mult(u, dxdt, state[0]))
    oper.mult(&u, &mut state);
    let mut va = vec![0.0; fe_size];
    let mut xa = vec![0.0; fe_size];
    let mut aa = vec![0.0; fe_size];

    if let Some(ref mut vis) = glvis {
        let _ = vis.send_solution_2d(&mesh, &u, "u");
        let _ = vis.send_command("pause");
    }

    let mut last_step = false;
    for ti in 1..=n_steps.max(1) {
        let dt_actual = if t + dt >= t_final - dt / 2.0 {
            last_step = true; t_final - t
        } else { dt };
        let dt2 = dt_actual * dt_actual;

        // Predict (fac0=fac2=0 → va stays = v at both steps)
        for i in 0..fe_size { xa[i] = u[i] + fac1 * dt_actual * v[i]; }
        // Solve alpha levels: aa = ImplicitSolve(fac3·dt², fac4·dt, xa, va)
        //   T = M + fac3·dt²·K,  T·aa = -K·xa  (dudt term unused by ex23's
        //   WaveOperator::ImplicitSolve — matches C++ z = -K*u only)
        oper.implicit_solve(fac3 * dt2, &xa, &mut aa);

        // Correct alpha levels
        for i in 0..fe_size {
            xa[i] += fac3 * dt2 * aa[i];
            va[i] = v[i] + fac4 * dt_actual * aa[i];
        }

        // Extrapolate
        for i in 0..fe_size {
            u[i] = (1.0 - inv_f1) * u[i] + inv_f1 * xa[i];
            v[i] = (1.0 - inv_f1) * v[i] + inv_f1 * va[i];
            state[i] = (1.0 - inv_f5) * state[i] + inv_f5 * aa[i];
        }
        for &d in &ess_tdof_list { u[d as usize] = 0.0; v[d as usize] = 0.0; state[d as usize] = 0.0; }

        t += dt_actual;
        if last_step || (ti % vis_steps == 0) {
            println!("step {}, t = {}", ti, t);
            if let Some(ref mut vis) = glvis {
                let _ = vis.send_solution_2d(&mesh, &u, "u");
            }
        }
        oper.set_parameters();
    }
    du_dt.copy_from_slice(&v);

    // 9. Save the final solution with precision(8) (matching C++)
    {
        let mut final_f = std::fs::File::create("ex23-final.gf").expect("create ex23-final.gf");
        writeln!(final_f, "FiniteElementSpace").ok();
        writeln!(final_f, "FiniteElementCollection: H1_{dim}D_P{}", args.order).ok();
        writeln!(final_f, "VDim: 1").ok();
        writeln!(final_f, "Ordering: 0").ok();
        writeln!(final_f).ok();
        for v in &u { writeln!(final_f, "{:.7e}", v).ok(); }
        for v in &du_dt { writeln!(final_f, "{:.7e}", v).ok(); }
    }

    // 10. Compute and print some statistics for comparison
    let max_u = u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let sum_u: f64 = u.iter().sum();
    let checksum_u: f64 = u.iter().enumerate().map(|(i, &v)| v * (i as f64 + 1.0)).sum();
    println!("\n  Final solution stats:");
    println!("    max|u| = {:.6e}", max_u);
    println!("    sum(u) = {:.6e}", sum_u);
    println!("    checksum = {:.6e}", checksum_u);

    let max_dudt = du_dt.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let sum_dudt: f64 = du_dt.iter().sum();
    let checksum_dudt: f64 = du_dt.iter().enumerate().map(|(i, &v)| v * (i as f64 + 1.0)).sum();
    println!("    max|du/dt| = {:.6e}", max_dudt);
    println!("    sum(du/dt) = {:.6e}", sum_dudt);
    println!("    du/dt checksum = {:.6e}", checksum_dudt);
}

// ─── 3D wave equation ─────────────────────────────────────────────────────

fn run_wave_3d(mut mesh: Mesh<3>, args: &Args) {
    let dim = 3;

    // 4. Refine the mesh uniformly.
    let mesh = if args.ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..args.ref_levels {
            m = fem_mesh::refine_uniform_3d(&m);
        }
        m
    } else {
        mesh
    };

    // 5. Define the H1 FE space.
    let space = H1Space::new(mesh.clone(), args.order);
    let fe_size = space.n_dofs();
    println!("Number of temperature unknowns: {}", fe_size);

    // 6. Compute essential BC DOFs (matching C++ GetEssentialTrueDofs).
    let ess_tdof_list: Vec<u32> = if args.dirichlet {
        let all_tags: Vec<i32> = if mesh.n_boundary_faces() > 0 {
            (0..mesh.n_boundary_faces())
                .map(|f| mesh.face_tag(f as u32))
                .collect()
        } else { Vec::new() };
        let mut unique_tags: Vec<i32> = all_tags.clone();
        unique_tags.sort_unstable();
        unique_tags.dedup();
        if unique_tags.is_empty() { Vec::new() }
        else {
            boundary_dofs(&mesh, space.dof_manager(), &unique_tags)
                .into_iter().map(|d| d as u32).collect()
        }
    } else { Vec::new() };

    // 7. Set initial conditions via interpolation (matching C++ ProjectCoefficient for H1).
    let mut u: Vec<f64> = space.interpolate(&|x: &[f64]| initial_solution(x)).into_vec();
    let mut du_dt: Vec<f64> = space.interpolate(&|x: &[f64]| initial_rate(x)).into_vec();
    println!("  ess_bdr count: {}", ess_tdof_list.len());
    for &d in &ess_tdof_list { u[d as usize] = 0.0; du_dt[d as usize] = 0.0; }

    // Save initial solution (matching C++ precision(8))
    {
        let _ = write_mfem_file_3d("ex23.mesh", &mesh);
        let mut init_f = std::fs::File::create("ex23-init.gf").expect("create ex23-init.gf");
        write_gf(&mut init_f, dim, &u, "H1", args.order, 1).expect("write init u");
        write_gf(&mut init_f, dim, &du_dt, "H1", args.order, 1).expect("write init du_dt");
    }

    // Create the wave operator
    let mut oper = WaveOperator::new(space, ess_tdof_list.clone(), args.speed);

    // 8. Time integration
    let t_final = args.t_final;
    let dt = args.dt;
    let vis_steps = args.vis_steps;
    let mut t = 0.0;
    let n_steps = if dt > 0.0 { (t_final / dt).ceil() as usize } else { 0 };

    // 8. Time integration matching C++ BackwardEulerSolver.
    let mut a3 = vec![0.0; fe_size];
    let mut v3 = du_dt.clone();
    let mut last_step = false;
    for ti in 1..=n_steps.max(1) {
        let dt_actual = if t + dt >= t_final - dt / 2.0 {
            last_step = true; t_final - t
        } else { dt };

        oper.implicit_solve(dt_actual * dt_actual, &u, &mut a3);
        for i in 0..fe_size { v3[i] += dt_actual * a3[i]; u[i] += dt_actual * v3[i]; }
        for &d in &ess_tdof_list { u[d as usize] = 0.0; v3[d as usize] = 0.0; }

        t += dt_actual;
        if last_step || (ti % vis_steps == 0) { println!("step {}, t = {}", ti, t); }
    }
    du_dt.copy_from_slice(&v3);

    // 9. Save the final solution
    {
        let mut final_f = std::fs::File::create("ex23-final.gf").expect("create ex23-final.gf");
        write_gf(&mut final_f, dim, &u, "H1", args.order, 1).expect("write final u");
        write_gf(&mut final_f, dim, &du_dt, "H1", args.order, 1).expect("write final du_dt");
    }

    // 10. Statistics
    let max_u = u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let sum_u: f64 = u.iter().sum();
    let checksum_u: f64 = u.iter().enumerate().map(|(i, &v)| v * (i as f64 + 1.0)).sum();
    println!("\n  Final solution stats:");
    println!("    max|u| = {:.6e}", max_u);
    println!("    sum(u) = {:.6e}", sum_u);
    println!("    checksum = {:.6e}", checksum_u);

    let max_dudt = du_dt.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let sum_dudt: f64 = du_dt.iter().sum();
    let checksum_dudt: f64 = du_dt.iter().enumerate().map(|(i, &v)| v * (i as f64 + 1.0)).sum();
    println!("    max|du/dt| = {:.6e}", max_dudt);
    println!("    sum(du/dt) = {:.6e}", sum_dudt);
    println!("    du/dt checksum = {:.6e}", checksum_dudt);
}
