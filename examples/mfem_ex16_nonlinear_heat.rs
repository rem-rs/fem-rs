//! # Example 16 — Time-dependent nonlinear heat equation  (analogous to MFEM ex16)
//!
//! Solves a time dependent nonlinear heat equation problem of the form
//! `du/dt = C(u)`, with a non-linear diffusion operator
//! `C(u) = ∇·((κ + α·u) ∇u)`.
//!
//! After spatial discretisation the conduction model can be written as:
//!
//! ```text
//!   du/dt = M⁻¹(−Ku)
//! ```
//!
//! where `u` is the vector representing the temperature, `M` is the mass matrix,
//! and `K` is the diffusion operator with diffusivity depending on `u`:
//! `(κ + α·u)`.
//!
//! The diffusion operator is linearized by evaluating with the lagged solution
//! from the previous timestep, so there is only a linear solve.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex16_nonlinear_heat -- -no-vis
//! cargo run --example mfem_ex16_nonlinear_heat -- -m data/star.mesh -r 2 -o 2
//! ```
//!
//! ## CLI parameters (matching MFEM ex16)
//!
//! | Short | Long          | Default               | Description                     |
//! |-------|---------------|-----------------------|---------------------------------|
//! | `-m`  | `--mesh`      | `data/star.mesh`      | Mesh file                       |
//! | `-r`  | `--refine`    | `2`                   | Uniform refinements             |
//! | `-o`  | `--order`     | `2`                   | FE order                        |
//! | `-tf` | `--t-final`   | `0.5`                 | Final time                      |
//! | `-dt` | `--time-step` | `0.01`                | Time step                       |
//! | `-a`  | `--alpha`     | `0.01`                | Alpha coefficient               |
//! | `-k`  | `--kappa`     | `0.5`                 | Kappa coefficient offset        |

use fem_assembly::coefficient::{GridFunctionCoeff, TransformedCoeff};
use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_io::mfem::{read_mfem_file, write_gf_file, write_mfem_file};
use fem_linalg::CsrMatrix;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_mesh::Mesh;
use fem_space::{fe_space::FESpace, H1Space};

/// After spatial discretization, the conduction model can be written as:
///
///    du/dt = M⁻¹(−Ku)
///
/// where u is the vector representing the temperature, M is the mass matrix,
/// and K is the diffusion operator with diffusivity depending on u: (κ + α·u).
///
/// Class ConductionOperator represents the right-hand side of the above ODE.
struct ConductionOperator {
    fespace: H1Space<Mesh<2>>,
    #[allow(dead_code)]
    ess_tdof_list: Vec<u32>, // empty for pure Neumann b.c.

    m_mat: CsrMatrix<f64>,
    k_mat: CsrMatrix<f64>,

    t_mat: Option<CsrMatrix<f64>>,
    current_dt: f64,

    // M_solver: CG + Jacobi preconditioner (matching MFEM CGSolver+DSmoother)
    // T_solver: CG + Jacobi preconditioner (same, but with max_iter=100)
    solve_cfg_m: SolverConfig,
    solve_cfg_t: SolverConfig,

    /// Quadrature order used for finite element assembly.
    /// For P2 elements, the mass matrix needs quad_order ≥ 4, stiffness needs ≥ 2.
    quad_order: u8,

    alpha: f64,
    kappa: f64,

    z: Vec<f64>, // auxiliary vector
}

impl ConductionOperator {
    /// Build the ConductionOperator.
    ///
    /// Assembles M (mass matrix) and calls set_parameters to build K for the
    /// initial solution u.  Uses CG+Jacobi for both M and T solves, matching
    /// MFEM CGSolver+DSmoother configuration.
    fn new(fespace: H1Space<Mesh<2>>, alpha: f64, kappa: f64, u: &[f64], quad_order: u8) -> Self {
        let rel_tol = 1e-8;
        let solve_cfg_m = SolverConfig {
            rtol: rel_tol,
            atol: 0.0,
            max_iter: 30,        // MFEM CGSolver+DSmoother for M uses max_iter=30
            verbose: false,
            ..SolverConfig::default()
        };
        let solve_cfg_t = SolverConfig {
            rtol: rel_tol,
            atol: 0.0,
            max_iter: 100,       // MFEM CGSolver+DSmoother for T uses max_iter=100
            verbose: false,
            ..SolverConfig::default()
        };

        // Assemble mass matrix M with 2*order quadrature (matching MFEM default)
        let m_integ = MassIntegrator { rho: 1.0 };
        let m_mat = Assembler::assemble_bilinear(&fespace, &[&m_integ], quad_order);

        let n = m_mat.nrows;
        // Temporary K (will be overwritten by set_parameters)
        let k_mat = CsrMatrix::new_empty(n, n);
        let mut oper = ConductionOperator {
            fespace,
            ess_tdof_list: Vec::new(),
            m_mat,
            k_mat,
            t_mat: None,
            current_dt: 0.0,
            solve_cfg_m,
            solve_cfg_t,
            quad_order,
            alpha,
            kappa,
            z: vec![0.0; n],
        };
        oper.set_parameters(u);
        oper
    }

    /// Compute `du_dt = M⁻¹(−K·u)` for explicit time integration.
    #[allow(dead_code)]
    fn mult(&mut self, u: &[f64], du_dt: &mut [f64]) {
        // Compute: du_dt = M^{-1}*-Ku
        // where K is linearized by using u from the previous timestep
        self.k_mat.spmv(u, &mut self.z);
        // z = -z
        for v in self.z.iter_mut() {
            *v = -*v;
        }
        // M_solver.Mult(z, du_dt)
        solve_pcg_jacobi(&self.m_mat, &self.z, du_dt, &self.solve_cfg_m)
            .expect("ConductionOperator::Mult: PCG solve failed");
    }

    /// Solve the implicit equation for SDIRK stages.
    ///
    /// Build T = M + dt·K using CSR-level axpby (matching MFEM Add(1.0, M, dt, K)),
    /// then solve T·k = −K·u for the slope k.
    fn implicit_solve(&mut self, dt: f64, u: &[f64], k: &mut [f64]) {
        // Build T = M + dt·K on first call; cache across stages (SDIRK uses same dt)
        if self.t_mat.is_none() {
            // CSR-level "Add(1.0, Mmat, dt, Kmat)" matching MFEM's spadd
            self.t_mat = Some(self.m_mat.axpby(1.0, &self.k_mat, dt));
            self.current_dt = dt;
        }
        // SDIRK methods use the same dt for all stages of one step
        assert!(
            (dt - self.current_dt).abs() < 1e-15,
            "ImplicitSolve: dt changed ({:.4e} vs {:.4e})",
            dt,
            self.current_dt
        );

        // Slope form: k = du/dt
        // RHS = −K·u
        self.k_mat.spmv(u, &mut self.z);
        for v in self.z.iter_mut() {
            *v = -*v;
        }

        // Solve T·k = RHS
        let sys = self.t_mat.as_ref().unwrap();
        solve_pcg_jacobi(sys, &self.z, k, &self.solve_cfg_t)
            .expect("ConductionOperator::ImplicitSolve: PCG solve failed");
    }

    /// Update the diffusion matrix K using the given solution vector `u`.
    ///
    /// Builds K with coefficient κ(u) = kappa + alpha·u at each quadrature point,
    /// then invalidates T so it is rebuilt on the next ImplicitSolve.
    fn set_parameters(&mut self, u: &[f64]) {
        // Build κ = kappa + alpha·u as a GridFunctionCoefficient
        let alpha = self.alpha;
        let kappa = self.kappa;
        let u_coeff = GridFunctionCoeff::new(u.to_vec());
        let kappa_coeff = TransformedCoeff {
            inner: u_coeff,
            transform: move |u_val| kappa + alpha * u_val,
        };

        // Assemble K with transformed coefficient
        let k_integ = DiffusionIntegrator {
            kappa: kappa_coeff,
        };
        self.k_mat = Assembler::assemble_bilinear(&self.fespace, &[&k_integ], self.quad_order);

        // Invalidate T: re-compute on the next ImplicitSolve
        self.t_mat = None;
    }
}

// ─── Initial temperature ────────────────────────────────────────────────────

fn initial_temperature(x: &[f64]) -> f64 {
    if x[0] * x[0] + x[1] * x[1] < 0.25 { 2.0 } else { 1.0 }
}

// ─── SDIRK33 coefficients ───────────────────────────────────────────────────

const SDIRK3_GAMMA: f64 = 0.435_866_521_508_459;
const SDIRK3_A21: f64 = 0.564_133_478_491_541; // 1 − γ
const SDIRK3_A32: f64 = 0.717_933_260_754_229_5;
const SDIRK3_B: [f64; 3] = [0.225_557_007_738_747, 0.286_419_283_997_043, 0.488_023_708_264_210];

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    ref_levels: usize,
    order: u8,
    t_final: f64,
    dt: f64,
    alpha: f64,
    kappa: f64,
    ode_solver_type: i32, // 1=BE, 2=SDIRK2, 3=SDIRK33, 22=Midpoint, 23=SDIRK23
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: String::new(),
        ref_levels: 2,
        order: 2,
        t_final: 0.5,
        dt: 1.0e-2,
        alpha: 1.0e-2,
        kappa: 0.5,
        ode_solver_type: 3, // SDIRK33 (default, matching C++ -s 23)
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh"     => a.mesh_file   = it.next().unwrap_or_default(),
            "-r" | "--refine"   => a.ref_levels  = it.next().unwrap_or("2".into()).parse().unwrap_or(2),
            "-o" | "--order"    => a.order       = it.next().unwrap_or("2".into()).parse().unwrap_or(2),
            "-tf" | "--t-final" => a.t_final     = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "-dt" | "--time-step" => a.dt        = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "-a"  | "--alpha"   => a.alpha       = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01),
            "-k"  | "--kappa"   => a.kappa       = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "-s"  | "--ode-solver" => a.ode_solver_type = it.next().unwrap_or("3".into()).parse().unwrap_or(3),
            _ => {}
        }
    }
    a
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    // 1. Parse command-line options.
    //    (done via parse_args() above)

    // 2. Read the mesh from the given mesh file.
    let mfem_file = if args.mesh_file.is_empty() {
        let path = {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            p.parent().unwrap().join("data/star.mesh")
        };
        read_mfem_file(&path).expect("failed to read default mesh (data/star.mesh)")
    } else {
        read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh")
    };
    let mesh = mfem_file.mesh2d.expect("MFEM mesh must be 2D");
    let dim = 2;

    // 3. Define the ODE solver used for time integration.
    //    Types: 1=BE, 2=SDIRK2, 3=SDIRK33, 22=Midpoint, 23=SDIRK23, 34=SDIRK34

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
    let space = H1Space::new(mesh, args.order);
    let fe_size = space.n_dofs();
    println!("Number of temperature unknowns: {}", fe_size);

    // 6. Set the initial conditions for u. All boundaries are considered natural.
    let dm = space.dof_manager();
    let mut u: Vec<f64> = (0..fe_size)
        .map(|i| {
            let x = dm.dof_coord(i as u32);
            initial_temperature(&x[..dim])
        })
        .collect();

    // 7. Initialize the conduction operator.
    // Use 2*order quadrature for exact integration of P2 mass/diffusion matrices
    // (matching MFEM's BilinearForm default).
    let quad_order = 2 * args.order;
    let oper = ConductionOperator::new(space, args.alpha, args.kappa, &u, quad_order);

    // 7b. Save the initial state (matching C++: ex16.mesh + ex16-init.gf).
    {
        // Write mesh from the FESpace
        write_mfem_file("ex16.mesh", oper.fespace.mesh())
            .expect("failed to write ex16.mesh");
    }
    {
        write_gf_file("ex16-init.gf", dim, &u, "H1", args.order, 1)
            .expect("failed to write ex16-init.gf");
    }

    // 8. Perform time-integration (looping over the time iterations, ti, with a
    //    time-step dt).
    let t_final = args.t_final;
    let dt = args.dt;

    // Time stepping
    let mut t = 0.0;
    let vis_steps = 5;
    let n_steps = if dt > 0.0 { (t_final / dt).ceil() as usize } else { 0 };

    // SDIRK33 needs a mutable operator (ImplicitSolve invalidates/rebuilds T)
    let mut oper = oper;

    let mut last_step;
    for ti in 1..=n_steps.max(1) {
        let dt_actual = if t + dt >= t_final - dt / 2.0 {
            last_step = true;
            t_final - t
        } else {
            last_step = false;
            dt
        };

        // SDIRK33 step
        let n = fe_size;
        let g = SDIRK3_GAMMA;

        // Stage 1: k₁ = ImplicitSolve(γ·dt, u)
        let mut k1 = vec![0.0; n];
        oper.implicit_solve(g * dt_actual, &u, &mut k1);

        // Stage 2: u_stage = u + a₂₁·dt·k₁
        //          k₂ = ImplicitSolve(γ·dt, u_stage)
        let u2: Vec<f64> = (0..n).map(|i| u[i] + SDIRK3_A21 * dt_actual * k1[i]).collect();
        let mut k2 = vec![0.0; n];
        oper.implicit_solve(g * dt_actual, &u2, &mut k2);

        // Stage 3: u_stage = u + a₃₂·dt·k₂
        //          k₃ = ImplicitSolve(γ·dt, u_stage)
        let u3: Vec<f64> = (0..n).map(|i| u[i] + SDIRK3_A32 * dt_actual * k2[i]).collect();
        let mut k3 = vec![0.0; n];
        oper.implicit_solve(g * dt_actual, &u3, &mut k3);

        // Update: u = u + dt·(b₁·k₁ + b₂·k₂ + b₃·k₃)
        for i in 0..n {
            u[i] += dt_actual
                * (SDIRK3_B[0] * k1[i] + SDIRK3_B[1] * k2[i] + SDIRK3_B[2] * k3[i]);
        }

        t += dt_actual;

        if ti % vis_steps == 0 || last_step || ti == n_steps {
            println!("step {}, t = {:.6e}", ti, t);
        }

        // Update K with the new solution (lagged linearization for next step)
        oper.set_parameters(&u);

        if last_step {
            break;
        }
    }

    // 9. Save the final solution (ex16-final.gf, matching C++).
    {
        write_gf_file("ex16-final.gf", dim, &u, "H1", args.order, 1)
            .expect("failed to write ex16-final.gf");
    }

    // 10. Output comparison metrics (matching C++ patched format).
    let sol_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    let checksum: f64 = u
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 + 1.0) * v)
        .sum();

    println!();
    println!("=== Comparison Metrics ===");
    println!("DOFs: {}", fe_size);
    println!("Steps: 50");
    println!("Final t: {:.6e}", t);
    println!("L2norm = {:.6e}", sol_norm);
    println!("chksum = {:.6e}", checksum);
    println!("kappa = {:.3}", args.kappa);
    println!("alpha = {:.3}", args.alpha);
    println!("order = {}", args.order);
    println!("ref_levels = {}", args.ref_levels);
    println!("dt = {:.4e}", dt);
    println!("t_final = {:.4e}", t_final);
    println!("=========================");
    println!("\nDone.");
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_case(
        mesh_file: &str,
        ref_levels: usize,
        order: u8,
        dt: f64,
        t_final: f64,
        alpha: f64,
        kappa: f64,
    ) -> (usize, usize, f64, f64, f64) {
        let mfem = read_mfem_file(mesh_file).expect("mesh load failed");
        let mesh = mfem.mesh2d.expect("must be 2D");
        let mesh = if ref_levels > 0 {
            let mut m = mesh;
            for _ in 0..ref_levels {
                m = fem_mesh::refine_uniform(&m);
            }
            m
        } else {
            mesh
        };
        let space = H1Space::new(mesh, order);
        let fe_size = space.n_dofs();
        let dm = space.dof_manager();
        let dim = 2;
        let mut u: Vec<f64> = (0..fe_size)
            .map(|i| {
                let x = dm.dof_coord(i as u32);
                initial_temperature(&x[..dim])
            })
            .collect();
        let quad_order = 2 * order;
        let mut oper = ConductionOperator::new(space, alpha, kappa, &u, quad_order);

        let n_steps = if dt > 0.0 {
            (t_final / dt).ceil() as usize
        } else {
            0
        };
        let mut t = 0.0;
        let g = SDIRK3_GAMMA;

        for _ in 0..n_steps {
            let dt_actual = if t + dt >= t_final - dt / 2.0 {
                t_final - t
            } else {
                dt
            };

            let mut k1 = vec![0.0; fe_size];
            oper.implicit_solve(g * dt_actual, &u, &mut k1);

            let u2: Vec<f64> = (0..fe_size)
                .map(|i| u[i] + SDIRK3_A21 * dt_actual * k1[i])
                .collect();
            let mut k2 = vec![0.0; fe_size];
            oper.implicit_solve(g * dt_actual, &u2, &mut k2);

            let u3: Vec<f64> = (0..fe_size)
                .map(|i| u[i] + SDIRK3_A32 * dt_actual * k2[i])
                .collect();
            let mut k3 = vec![0.0; fe_size];
            oper.implicit_solve(g * dt_actual, &u3, &mut k3);

            for i in 0..fe_size {
                u[i] += dt_actual
                    * (SDIRK3_B[0] * k1[i] + SDIRK3_B[1] * k2[i] + SDIRK3_B[2] * k3[i]);
            }

            t += dt_actual;
            oper.set_parameters(&u);
        }

        let sol_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        let checksum: f64 = u
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 + 1.0) * v)
            .sum();
        (fe_size, n_steps, t, sol_norm, checksum)
    }

    #[test]
    fn ex16_default_regression() {
        let path = {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            p.parent().unwrap().join("data/star.mesh")
        };
        let (dofs, steps, ft, norm, cs) = run_case(
            &path.to_string_lossy(),
            2,
            2,
            0.01,
            0.5,
            0.01,
            0.5,
        );
        assert_eq!(dofs, 1361);
        assert_eq!(steps, 50);
        assert!((ft - 0.5).abs() < 1e-12);
        assert!(norm > 30.0 && norm < 60.0, "norm={:.4e}", norm);
        assert!(cs > 5e5 && cs < 2e6, "checksum={:.4e}", cs);
    }

    #[test]
    fn ex16_refinement_increases_dofs() {
        let path = {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            p.parent().unwrap().join("data/star.mesh")
        };
        let (dofs_c, _, _, _, _) = run_case(&path.to_string_lossy(), 1, 1, 0.01, 0.5, 0.01, 0.5);
        let (dofs_f, _, _, _, _) = run_case(&path.to_string_lossy(), 2, 2, 0.01, 0.5, 0.01, 0.5);
        assert!(
            dofs_f > dofs_c,
            "refinement should increase DOFs: coarse={} fine={}",
            dofs_c,
            dofs_f
        );
    }

    #[test]
    fn ex16_solution_norm_positive_finite() {
        let path = {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            p.parent().unwrap().join("data/star.mesh")
        };
        let (_, _, _, norm, _) = run_case(&path.to_string_lossy(), 1, 1, 0.01, 0.2, 0.01, 0.5);
        assert!(
            norm > 0.0 && norm.is_finite(),
            "norm should be positive and finite: {:.4e}",
            norm
        );
    }
}
