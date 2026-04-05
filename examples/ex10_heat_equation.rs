//! # Example 10 — Time-dependent heat equation  (analogous to MFEM ex10)
//!
//! Solves the time-dependent heat equation:
//!
//! ```text
//!   ∂u/∂t − κ Δu = 0    in Ω = [0,1]², t ∈ [0, T]
//!             u = 0    on ∂Ω  (Dirichlet)
//!             u = u₀    at t = 0  (initial condition)
//! ```
//!
//! The spatial semi-discretization gives the ODE:
//! ```text
//!   M du/dt + K u = 0   →   du/dt = −M⁻¹ K u = f(t,u)
//! ```
//!
//! Available time integrators (via `--method`):
//! - `euler`  : Forward Euler (explicit, conditionally stable)
//! - `rk4`    : Classical 4th-order Runge-Kutta
//! - `ie`     : Implicit Euler (unconditionally A-stable)
//! - `sdirk2` : 2-stage SDIRK (order 2, A-stable)
//! - `bdf2`   : BDF-2 (order 2, A-stable, multi-step)
//!
//! Initial condition: `u₀ = sin(πx)sin(πy)`.
//! Exact solution:    `u(x,y,t) = e^{−2π²κt} sin(πx)sin(πy)`.
//!
//! ## Usage
//! ```
//! cargo run --example ex10_heat_equation
//! cargo run --example ex10_heat_equation -- --method sdirk2 --dt 0.01 --T 0.5
//! cargo run --example ex10_heat_equation -- --method rk4 --n 16 --dt 0.001 --T 0.1
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::{
    solve_pcg_jacobi, SolverConfig,
    ForwardEuler, Rk4, ImplicitEuler, Sdirk2, Bdf2, Bdf2State,
    TimeStepper, ImplicitTimeStepper,
};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};
use fem_linalg::CsrMatrix;

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 10: Time-dependent heat equation ===");
    println!("  Method: {}, dt = {}, T = {}", args.method, args.dt, args.t_end);
    println!("  Mesh: {}×{}, κ = {}", args.n, args.n, args.kappa);

    // ─── 1. Mesh and H¹ space ─────────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n     = space.n_dofs();
    println!("  DOFs: {n}");

    // ─── 2. Assemble K (stiffness) and M (mass) ───────────────────────────────
    let mut k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: args.kappa }], 3);
    let mut m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);

    // ─── 3. Apply Dirichlet BCs (u=0 on all boundaries) ──────────────────────
    let dm   = space.dof_manager();
    let bnd  = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let vals = vec![0.0_f64; bnd.len()];
    let mut dummy_rhs = vec![0.0_f64; n];
    apply_dirichlet(&mut k_mat, &mut dummy_rhs, &bnd, &vals);
    apply_dirichlet(&mut m_mat, &mut dummy_rhs, &bnd, &vals);

    // ─── 4. Initial condition: u₀ = sin(πx)sin(πy) ───────────────────────────
    let mut u: Vec<f64> = (0..n).map(|i| {
        let x = dm.dof_coord(i as u32);
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }).collect();
    for &d in &bnd { u[d as usize] = 0.0; }

    let dt    = args.dt;
    let t_end = args.t_end;
    let n_steps = (t_end / dt).ceil() as usize;
    println!("  Steps: {n_steps}  (dt={dt:.4e})");

    let solve_cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, verbose: false, ..SolverConfig::default() };

    // ─── ODE RHS: dudt = M⁻¹(−K u) ──────────────────────────────────────────
    // Capture references to k_mat, m_mat, bnd, solve_cfg.
    let dudt = |_t: f64, u: &[f64], out: &mut [f64]| {
        let mut neg_ku = vec![0.0_f64; n];
        k_mat.spmv(u, &mut neg_ku);
        for v in neg_ku.iter_mut() { *v = -*v; }
        for &d in &bnd { neg_ku[d as usize] = 0.0; }
        out.iter_mut().for_each(|v| *v = 0.0);
        let _ = solve_pcg_jacobi(&m_mat, &neg_ku, out, &solve_cfg);
    };

    // Jacobian of f(u) = M⁻¹(−K u): J = −M⁻¹ K
    // For implicit methods, we need (I − dt γ J) = I + dt γ M⁻¹ K.
    // Equivalently, the linear system is: (M + dt γ K) v = M u_old ... (varies per method)
    // We implement the Jacobian as a CSR matrix of M⁻¹(-K) ≈ -M⁻¹K.
    // In practice, ImplicitTimeStepper builds (I - dt*J) and solves it, so
    // J(t,u) here should be ∂f/∂u = -M⁻¹K. We approximate via a finite difference
    // or just provide K/M. For simplicity, use a precomputed diagonal approximation.
    //
    // Actually, the ImplicitTimeStepper framework calls:
    //   sys = I - dt * J,  then solves sys * du = dt * f(t,u)
    // where J ≈ ∂f/∂u. For our problem f = -M⁻¹K u, so J = -M⁻¹K.
    // We can build J as a CSR by computing -M⁻¹K column by column (expensive),
    // OR we implement a custom approach.
    //
    // Simpler: for the heat equation, use direct assembly of (M + dt*K):

    let build_m_plus_alpha_k = |alpha: f64| -> CsrMatrix<f64> {
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            for ptr in m_mat.row_ptr[i]..m_mat.row_ptr[i+1] {
                let j = m_mat.col_idx[ptr] as usize;
                coo.add(i, j, m_mat.values[ptr]);
            }
        }
        for i in 0..n {
            for ptr in k_mat.row_ptr[i]..k_mat.row_ptr[i+1] {
                let j = k_mat.col_idx[ptr] as usize;
                coo.add(i, j, alpha * k_mat.values[ptr]);
            }
        }
        coo.into_csr()
    };

    // ─── 5. Time-stepping ─────────────────────────────────────────────────────
    let mut t = 0.0_f64;

    match args.method.as_str() {
        "euler" => {
            let stepper = ForwardEuler;
            for _ in 0..n_steps {
                stepper.step(t, dt, &mut u, &dudt);
                t += dt;
            }
        }
        "rk4" => {
            let stepper = Rk4;
            for _ in 0..n_steps {
                stepper.step(t, dt, &mut u, &dudt);
                t += dt;
            }
        }
        "ie" => {
            let stepper = ImplicitEuler;
            for _ in 0..n_steps {
                // J(t,u) = -M⁻¹K  →  we pass the Jacobian via a closure.
                // ImplicitTimeStepper builds (I - dt*J) and solves:
                //   (I - dt*J) du = dt * f(t,u)
                // Since J = -M⁻¹K, the system becomes:
                //   (I + dt M⁻¹K) du = dt * f  →  multiply by M:
                //   (M + dt K) du = dt * M f   (not what the trait does directly)
                // We implement a Picard Jacobian by returning a zero Jacobian
                // and handling the solve manually via the step_implicit machinery.
                // For simplicity we bypass step_implicit and use the direct IE formula:
                //   (M + dt K) u_new = M u_old
                let rhs_ie = {
                    let mut r = vec![0.0_f64; n];
                    m_mat.spmv(&u, &mut r);
                    r
                };
                let sys = build_m_plus_alpha_k(dt);
                let mut u_new = vec![0.0_f64; n];
                let _ = solve_pcg_jacobi(&sys, &rhs_ie, &mut u_new, &solve_cfg);
                for &d in &bnd { u_new[d as usize] = 0.0; }
                u.copy_from_slice(&u_new);
                t += dt;
            }
        }
        "sdirk2" => {
            // Use ImplicitTimeStepper API with Jacobian closure
            let sdirk = Sdirk2;
            for _ in 0..n_steps {
                // The Jacobian J(t,u) = ∂f/∂u where f = M⁻¹(-Ku).
                // For the SDIRK2 framework: system is (I - dt*γ*J) which is
                // not directly the CSR we can assemble easily.
                // We provide a scaled zero Jacobian (Picard) and rely on the
                // fact that our dudt already includes M^{-1}K implicitly.
                // Actually SDIRK2 builds (I - dt*γ*J) = (I + dt*γ*M⁻¹K)
                // which when multiplied by M gives (M + dt*γ*K).
                // We use a custom Jacobian that returns a scaled identity matrix.
                // The real linear solve happens through our custom PCG approach.
                //
                // For correctness, call step_implicit with a closure that
                // returns -M⁻¹K ≈ 0 (Picard) since we cannot easily invert M
                // in Jacobian form. Use a direct method instead:
                let g = 1.0 - std::f64::consts::FRAC_1_SQRT_2;

                // Stage 1: solve (M + dt*g*K) k1_hat = -K u, then k1 = M^{-1} k1_hat
                let rhs1 = {
                    let mut r = vec![0.0_f64; n];
                    k_mat.spmv(&u, &mut r);
                    for v in r.iter_mut() { *v = -*v; }
                    for &d in &bnd { r[d as usize] = 0.0; }
                    r
                };
                let sys1 = build_m_plus_alpha_k(dt * g);
                let mut k1 = vec![0.0_f64; n];
                let _ = solve_pcg_jacobi(&sys1, &rhs1, &mut k1, &solve_cfg);

                // Stage 2: u2 = u + dt*(1-g)*k1
                let u2: Vec<f64> = u.iter().zip(k1.iter()).map(|(&ui, &ki)| ui + dt * (1.0-g) * ki).collect();
                let rhs2 = {
                    let mut r = vec![0.0_f64; n];
                    k_mat.spmv(&u2, &mut r);
                    for v in r.iter_mut() { *v = -*v; }
                    for &d in &bnd { r[d as usize] = 0.0; }
                    r
                };
                let sys2 = build_m_plus_alpha_k(dt * g);
                let mut k2 = vec![0.0_f64; n];
                let _ = solve_pcg_jacobi(&sys2, &rhs2, &mut k2, &solve_cfg);

                // Update: u ← u + dt*[(1-g)*k1 + g*k2]
                for i in 0..n {
                    u[i] += dt * ((1.0-g) * k1[i] + g * k2[i]);
                }
                for &d in &bnd { u[d as usize] = 0.0; }
                t += dt;
            }
        }
        "bdf2" => {
            // BDF-2: custom 2-step scheme
            // Step 1: BDF-1 (implicit Euler) for startup
            // Step k (k≥2): (3/2 M + dt K) u_{n+1} = 2 M u_n - 1/2 M u_{n-1}
            let mut u_prev = u.clone();

            // BDF-1 startup:
            {
                let rhs_bdf1 = {
                    let mut r = vec![0.0_f64; n];
                    m_mat.spmv(&u, &mut r);
                    r
                };
                let sys = build_m_plus_alpha_k(dt);
                let mut u_new = vec![0.0_f64; n];
                let _ = solve_pcg_jacobi(&sys, &rhs_bdf1, &mut u_new, &solve_cfg);
                for &d in &bnd { u_new[d as usize] = 0.0; }
                u_prev = u.clone();
                u.copy_from_slice(&u_new);
                t += dt;
            }

            // BDF-2 main loop
            let sys_bdf2 = {
                // (3/2 M + dt K)
                let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
                for i in 0..n {
                    for ptr in m_mat.row_ptr[i]..m_mat.row_ptr[i+1] {
                        coo.add(i, m_mat.col_idx[ptr] as usize, 1.5 * m_mat.values[ptr]);
                    }
                }
                for i in 0..n {
                    for ptr in k_mat.row_ptr[i]..k_mat.row_ptr[i+1] {
                        coo.add(i, k_mat.col_idx[ptr] as usize, dt * k_mat.values[ptr]);
                    }
                }
                coo.into_csr()
            };

            for step in 1..n_steps {
                // rhs = 2 M u_n - 1/2 M u_{n-1}
                let mut rhs_bdf2 = vec![0.0_f64; n];
                let mut mu_n   = vec![0.0_f64; n];
                let mut mu_nm1 = vec![0.0_f64; n];
                m_mat.spmv(&u, &mut mu_n);
                m_mat.spmv(&u_prev, &mut mu_nm1);
                for i in 0..n {
                    rhs_bdf2[i] = 2.0 * mu_n[i] - 0.5 * mu_nm1[i];
                }

                let mut u_new = vec![0.0_f64; n];
                let _ = solve_pcg_jacobi(&sys_bdf2, &rhs_bdf2, &mut u_new, &solve_cfg);
                for &d in &bnd { u_new[d as usize] = 0.0; }
                u_prev = u.clone();
                u.copy_from_slice(&u_new);
                t += dt;
            }
        }
        other => {
            eprintln!("Unknown method '{other}'. Choose: euler, rk4, ie, sdirk2, bdf2");
            std::process::exit(1);
        }
    }

    // ─── 6. Compare to exact solution ────────────────────────────────────────
    let exact_factor = (-2.0 * PI * PI * args.kappa * t_end).exp();
    let l2 = {
        let mut err2 = 0.0_f64;
        for i in 0..n {
            let x = dm.dof_coord(i as u32);
            let u_ex = exact_factor * (PI * x[0]).sin() * (PI * x[1]).sin();
            err2 += (u[i] - u_ex).powi(2);
        }
        (err2 / n as f64).sqrt()
    };
    println!("  t = {t:.4e},  nodal RMS error vs exact = {l2:.4e}");
    println!("  Exact decay factor = {exact_factor:.6e}");
    println!("\nDone.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n:      usize,
    dt:     f64,
    t_end:  f64,
    kappa:  f64,
    method: String,
}

fn parse_args() -> Args {
    let mut a = Args { n: 8, dt: 0.01, t_end: 0.1, kappa: 1.0, method: "sdirk2".into() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"      => { a.n      = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--dt"     => { a.dt     = it.next().unwrap_or("0.01".into()).parse().unwrap_or(0.01); }
            "--T"      => { a.t_end  = it.next().unwrap_or("0.1".into()).parse().unwrap_or(0.1); }
            "--kappa"  => { a.kappa  = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--method" => { a.method = it.next().unwrap_or("sdirk2".into()); }
            _ => {}
        }
    }
    a
}
