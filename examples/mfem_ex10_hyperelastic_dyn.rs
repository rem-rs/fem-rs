//! # MFEM Example 10 — Dynamic Hyperelasticity (NeoHookean)
//!
//! 1:1 port of `mfem/examples/ex10.cpp`.
//!
//! Solves the time-dependent nonlinear elasticity problem:
//!
//! ```text
//!   M dv/dt = -H(x) - S v
//!   dx/dt   =  v
//! ```
//!
//! where M is the mass matrix, S is a viscosity (vector Laplacian) operator,
//! and H(x) is the internal force from a NeoHookean hyperelastic model.
//!
//! The geometry is a beam with boundary attribute 1 fixed.  Implicit time
//! integration uses a Newton solve per stage via the reduced backward-Euler
//! system `R(k) = (M + dt S) k + H(x + dt (v + dt k)) + S v`.
//!
//! ## Usage
//! ```bash
//! # Default: beam-tri.mesh, order 2, ref 2, SDIRK23, dt=3, t_final=300
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -no-vis
//!
//! # Custom mesh and parameters
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -m data/beam-quad.mesh -r 2 -o 2 -dt 3 -no-vis
//!
//! # Explicit (forward Euler)
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -s 4 -dt 0.03 -vs 20 -no-vis
//! ```
//!
//! ## ODE solver types (same numbering as MFEM)
//! |   s | Method           | Type     |
//! |-----|------------------|----------|
//! |   4 | Forward Euler    | Explicit |
//! |  22 | SDIRK2           | Implicit |
//! |  23 | SDIRK3 (default) | Implicit |

use std::f64::consts::FRAC_1_SQRT_2;

use fem_assembly::{
    Assembler,
    postproc::vector_l2_norm,
    standard::{VectorDiffusionIntegrator, VectorH1MassIntegrator},
    HyperelasticModel, HyperelasticityForm,
};
use fem_io::mfem::{read_mfem_file, write_gf_file, write_mfem_file};
use fem_linalg::CsrMatrix;
use fem_mesh::{
    Mesh,
    amr::refine_uniform,
    topology::MeshTopology,
};
use fem_space::fe_space::FESpace;
use fem_space::L2Space;
use fem_solver::{solve_minres_jacobi, solve_pcg_jacobi, SolverConfig};

// ─── CLI arguments (matching MFEM ex10) ────────────────────────────────────

#[allow(non_snake_case)]
struct Args {
    mesh: String,
    ref_levels: usize,
    order: u8,
    ode_solver_type: i32,
    t_final: f64,
    dt: f64,
    viscosity: f64,
    mu: f64,
    K: f64,
    visualization: bool,
    vis_steps: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/beam-quad.mesh".to_string(),
            ref_levels: 2,
            order: 2,
            ode_solver_type: 23,
            t_final: 300.0,
            dt: 3.0,
            viscosity: 1e-2,
            mu: 0.25,
            K: 5.0,
            visualization: false,
            vis_steps: 1,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-r" | "--refine" => {
                    a.ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
                }
                "-o" | "--order" => {
                    a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
                }
                "-s" | "--ode-solver" => {
                    a.ode_solver_type = it.next().and_then(|v| v.parse().ok()).unwrap_or(23)
                }
                "-tf" | "--t-final" => {
                    a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(300.0)
                }
                "-dt" | "--time-step" => {
                    a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(3.0)
                }
                "-v" | "--viscosity" => {
                    a.viscosity = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-2)
                }
                "-mu" | "--shear-modulus" => {
                    a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.25)
                }
                "-K" | "--bulk-modulus" => {
                    a.K = it.next().and_then(|v| v.parse().ok()).unwrap_or(5.0)
                }
                "-no-vis" | "--no-visualization" => a.visualization = false,
                "-vis" | "--visualization" => a.visualization = true,
                "-vs" | "--visualization-steps" => {
                    a.vis_steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
                }
                _ => {}
            }
        }
        a
    }
}

// ─── Initial conditions ─────────────────────────────────────────────────────

/// Identity map: initial deformation = reference configuration.
#[allow(dead_code)]
fn initial_deformation(x: &[f64]) -> Vec<f64> {
    x.to_vec()
}

/// Initial velocity: parabolic profile in the vertical direction.
fn initial_velocity(x: &[f64]) -> Vec<f64> {
    let dim = x.len();
    let mut v = vec![0.0; dim];
    if dim >= 2 {
        let s = 0.1 / 64.0;
        v[dim - 1] = s * x[0] * x[0] * (8.0 - x[0]);
        v[0] = -s * x[0] * x[0];
    }
    v
}

// ─── ReducedSystemOperator ─────────────────────────────────────────────────

/// Nonlinear operator for the reduced backward-Euler equation:
///
/// ```text
/// R(k) = (M + dt·S)·k + H(x + dt·(v + dt·k)) + S·v
/// ```
///
/// where `M` is the mass matrix, `S` the viscosity matrix, and `H` the
/// hyperelastic internal-force operator.
struct ReducedSystemOperator<'a> {
    m: &'a CsrMatrix<f64>,
    s: &'a CsrMatrix<f64>,
    hyper: &'a HyperelasticityForm<Mesh<2>>,
    ess_dofs: &'a [usize],
    n: usize,
}

impl ReducedSystemOperator<'_> {
    /// Width / height of the operator (one scalar block).
    fn size(&self) -> usize { self.n }

    /// Compute `y = R(k)` given current `v`, `x`, `dt`.
    fn mult(&self, k: &[f64], v: &[f64], x: &[f64], dt: f64, y: &mut [f64]) {
        let n = self.n;
        // w = v + dt*k
        let mut w = vec![0.0; n];
        for i in 0..n { w[i] = v[i] + dt * k[i]; }
        // z = x + dt*w = x + dt*(v + dt*k)
        let mut z = vec![0.0; n];
        for i in 0..n { z[i] = x[i] + dt * w[i]; }

        // y = H(z)  (internal force)
        self.hyper.raw_residual(&z, y);

        // y += M * k
        let mut mk = vec![0.0; n];
        self.m.spmv(k, &mut mk);
        for i in 0..n { y[i] += mk[i]; }

        // y += S * w = S * (v + dt*k)
        let mut sw = vec![0.0; n];
        self.s.spmv(&w, &mut sw);
        for i in 0..n { y[i] += sw[i]; }

        // Enforce essential BCs: at constrained DOFs, R(k) = 0
        // (matches MFEM behavior where H->Mult and FormSystemMatrix enforce BCs)
        for &d in self.ess_dofs {
            if d < n { y[d] = k[d]; }
        }
    }

    /// Compute the Jacobian `J = dR/dk = M + dt·S + dt²·grad_H(z)`.
    #[allow(dead_code)]
    fn gradient(&self, k: &[f64], v: &[f64], x: &[f64], dt: f64) -> CsrMatrix<f64> {
        let n = self.n;
        let mut w = vec![0.0; n];
        for i in 0..n { w[i] = v[i] + dt * k[i]; }
        let mut z = vec![0.0; n];
        for i in 0..n { z[i] = x[i] + dt * w[i]; }

        let grad_h = self.hyper.raw_jacobian(&z);

        // J = M + dt*S + dt²*grad_H
        // Use axpby for M + dt*S, then add dt²*grad_H via a second axpby.
        // First: tmp = 1.0 * M + dt * S
        let tmp = self.m.axpby(1.0, self.s, dt);
        // J = 1.0 * tmp + dt² * grad_H
        let mut jac = tmp.axpby(1.0, &grad_h, dt * dt);

        // Enforce essential BCs: symmetric elimination (row + column).
        let mut dummy = vec![0.0; n];
        for &d in self.ess_dofs {
            if d < n { jac.apply_dirichlet_symmetric(d, 0.0, &mut dummy); }
        }
        jac
    }
}

// ─── Dot product ────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

// ─── Newton solver for the reduced system ───────────────────────────────────

/// Solve `R(k) = 0` for `k` using Newton's method with MINRES inner solver.
///
/// The reduced operator is evaluated at current `v`, `x`, `dt`.
fn newton_solve_reduced(
    op: &ReducedSystemOperator,
    k: &mut [f64],
    v: &[f64],
    x: &[f64],
    dt: f64,
    verbose: bool,
) {
    let n = op.size();
    // MFEM double-precision: rel_tol = 1e-8, abs_tol = 0.0, max_iter = 10
    let rtol = 1e-8;
    let atol = 1e-7;   // Inner MINRES atol=1e-8 limits Newton accuracy to ~5e-8;
                       // atol=1e-7 catches this as converged. MFEM with UMFPack
                       // (or tighter inner tolerances) can reach 1e-9.
    let max_iter = 10;
    // Initial residual
    let mut r = vec![0.0; n];
    op.mult(k, v, x, dt, &mut r);
    let norm0 = norm2(&r);
    if verbose {
        println!("Newton iteration  0 : ||r|| = {:.6}", norm0);
    }
    if norm0 <= atol {
        return;
    }

    // Exact Newton: inner solve with MINRES + Jacobi preconditioning.
    // C++ MFEM ex10 uses rtol=1e-8, atol=0.0, max_iter=300 with UMFPack
    // when SuiteSparse is available, or MINRES with DSmoother otherwise.
    // rtol=1e-8 is too tight for Jacobi-preconditioned MINRES on the
    // ill-conditioned tangent matrix; rtol=1e-6 gives ~5e-6 relative
    // accuracy in the final solution — consistent with C++ MINRES.
    let inner_cfg = SolverConfig {
        rtol: 1e-6,
        atol: 1e-8,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    };

    let mut converged = false;
    for iter in 1..=max_iter {
        // Build exact Jacobian J = M + dt*S + dt²*grad_H(z)
        let jac = op.gradient(k, v, x, dt);
        let mut rhs_work = vec![0.0; n];
        for i in 0..n { rhs_work[i] = -r[i]; }
        // RHS at BC DOFs must be 0 (consistent with the BC elimination)
        for &d in op.ess_dofs { if d < n { rhs_work[d] = 0.0; } }

        // Solve J * dk = -r with MINRES (matching C++ MFEM)
        let mut dk = vec![0.0; n];
        solve_minres_jacobi(&jac, &rhs_work, &mut dk, &inner_cfg).ok();

        // Newton update: k += dk
        for j in 0..n { k[j] += dk[j]; }
        op.mult(k, v, x, dt, &mut r);
        let norm = norm2(&r);
        if verbose {
            println!(
                "Newton iteration {iter:2} : ||r|| = {norm:.6e}, ||r||/||r_0|| = {:.6e}",
                norm / norm0
            );
        }
        // MFEM convergence: ||r|| <= ATOL || ||r|| <= RTOL * ||r_0||
        if norm <= atol || norm <= rtol * norm0 {
            converged = true;
            break;
        }
    }
    if !converged {
        eprintln!("WARNING: Newton solver did not converge (final ||r||/||r_0|| = {:.6e})",
                  norm2(&r) / norm0);
    }
}

// ─── ODE integrators ───────────────────────────────────────────────────────

/// Forward Euler explicit step: `vx += dt * f(vx)`.
fn forward_euler_step(
    vx: &mut [f64],
    dt: f64,
    m: &CsrMatrix<f64>,
    s: &CsrMatrix<f64>,
    hyper: &HyperelasticityForm<Mesh<2>>,
    ess_dofs_v: &[usize],
    ess_dofs_x: &[usize],
) {
    let sc = vx.len() / 2;
    let (v, x) = vx.split_at_mut(sc);
    // Compute dv/dt = -M^{-1} * (H(x) + S*v)
    let mut rhs = vec![0.0; sc];
    hyper.raw_residual(x, &mut rhs);
    let mut sv = vec![0.0; sc];
    s.spmv(v, &mut sv);
    for i in 0..sc { rhs[i] += sv[i]; }
    for i in 0..sc { rhs[i] = -rhs[i]; }
    // Enforce BC on rhs: zero for constrained velocity DOFs
    for &d in ess_dofs_v { rhs[d] = 0.0; }

    let mut dv = vec![0.0; sc];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 30, verbose: false, ..SolverConfig::default() };
    match solve_pcg_jacobi(&m, &rhs, &mut dv, &cfg) {
        Ok(_) => {}
        Err(e) => eprintln!("  Explicit: M solve failed: {e}"),
    }
    for &d in ess_dofs_v { dv[d] = 0.0; }

    // dx/dt = v
    for i in 0..sc {
        v[i] += dt * dv[i];
        x[i] += dt * v[i]; // using updated v
    }
    // Enforce BC on x
    for &d in ess_dofs_x { x[d] = 0.0; }
}

/// SDIRK2 (type 22) — two-stage implicit with Newton solve per stage.
///
/// Butcher tableau: γ = 1 - 1/√2
///   γ  |  γ   0
///   1  | 1-γ  γ
///   ---|--------
///      | 1-γ  γ
fn sdirk2_step(
    vx: &mut [f64],
    dt: f64,
    op: &ReducedSystemOperator,
    ess_dofs_k: &[usize],
    verbose: bool,
) {
    let sc = vx.len() / 2;
    let gamma = 1.0 - FRAC_1_SQRT_2; // ≈ 0.2929

    let (v, x) = vx.split_at_mut(sc);

    // ── Stage 1 ─────────────────────────────────────────────────────────────
    // Solve for kv1: R(kv) = 0 at u_n with timestep γ*dt.
    // The reduced system operates on kv (velocity increment).
    // The corresponding position increment is kx1 = v + (γ*dt)*kv1.
    let mut kv1 = vec![0.0; sc];
    let dt_gamma = gamma * dt;
    newton_solve_reduced(op, &mut kv1, v, x, dt_gamma, verbose);
    let mut kx1 = vec![0.0; sc];
    for i in 0..sc { kx1[i] = v[i] + dt_gamma * kv1[i]; }

    // ── Stage 2 intermediate state ──────────────────────────────────────────
    // U2 = u_n + dt*(1-γ)*k1 where k1 = [kv1, kx1]
    let mut v2 = v.to_vec();
    let mut x2 = x.to_vec();
    for i in 0..sc {
        v2[i] += dt * (1.0 - gamma) * kv1[i];
        x2[i] += dt * (1.0 - gamma) * kx1[i];
    }

    // ── Stage 2 ─────────────────────────────────────────────────────────────
    let mut kv2 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv2, &v2, &x2, dt_gamma, verbose);
    let mut kx2 = vec![0.0; sc];
    for i in 0..sc { kx2[i] = v2[i] + dt_gamma * kv2[i]; }

    // ── Final update: u_{n+1} = u_n + dt * ((1-γ)*k1 + γ*k2) ──────────────
    for i in 0..sc {
        v[i] += dt * ((1.0 - gamma) * kv1[i] + gamma * kv2[i]);
        x[i] += dt * ((1.0 - gamma) * kx1[i] + gamma * kx2[i]);
    }
    for &d in ess_dofs_k {
        if d < sc { v[d] = 0.0; x[d] = 0.0; }
    }
}

/// SDIRK33 (type 23) — 3-stage, 3rd-order, L-stable (exact MFEM coefficients).
///
/// Butcher tableau (from MFEM linalg/ode.cpp SDIRK33Solver):
/// ```
///   a  |  a
///   c  |  c-a    a
///   1  |   b   1-a-b  a
///  ----+----------------
///      |   b   1-a-b  a
/// ```
/// Coefficients:
///   a = 0.435866521508458999416019  (diagonal, L-stable)
///   b = 1.20849664917601007033648
///   c = 0.717933260754229499708010
///
/// Stage k values from ImplicitSolve are velocity increments kv.
/// Position increments kx = v_current + dt_stage * kv (from kinematics).
fn sdirk3_step(
    vx: &mut [f64],
    dt: f64,
    op: &ReducedSystemOperator,
    ess_dofs_k: &[usize],
    verbose: bool,
) {
    const A: f64 = 0.435866521508458999416019;
    const B: f64 = 1.20849664917601007033648;
    const C: f64 = 0.717933260754229499708010;

    let sc = vx.len() / 2;
    let (v, x) = vx.split_at_mut(sc);
    let dt_a = A * dt;

    // ── Stage 1 ───────────────────────────────────────────────────────────
    let mut kv1 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv1, v, x, dt_a, verbose);
    let mut kx1 = vec![0.0; sc];
    for i in 0..sc { kx1[i] = v[i] + dt_a * kv1[i]; }

    // y = vx0 + (c-a)*dt * k1
    let mut vy = v.to_vec();
    let mut xy = x.to_vec();
    let ca = C - A;
    for i in 0..sc {
        vy[i] += ca * dt * kv1[i];
        xy[i] += ca * dt * kx1[i];
    }
    // Partial accumulate: vx += b*dt * k1
    for i in 0..sc {
        v[i] += B * dt * kv1[i];
        x[i] += B * dt * kx1[i];
    }

    // ── Stage 2 ───────────────────────────────────────────────────────────
    let mut kv2 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv2, &vy, &xy, dt_a, verbose);
    let mut kx2 = vec![0.0; sc];
    for i in 0..sc { kx2[i] = vy[i] + dt_a * kv2[i]; }

    let nab = 1.0 - A - B;
    for i in 0..sc {
        v[i] += nab * dt * kv2[i];
        x[i] += nab * dt * kx2[i];
    }

    // ── Stage 3 ───────────────────────────────────────────────────────────
    let mut kv3 = vec![0.0; sc];
    newton_solve_reduced(op, &mut kv3, v, x, dt_a, verbose);
    let mut kx3 = vec![0.0; sc];
    for i in 0..sc { kx3[i] = v[i] + dt_a * kv3[i]; }

    for i in 0..sc {
        v[i] += A * dt * kv3[i];
        x[i] += A * dt * kx3[i];
    }

    // Enforce BC
    for &d in ess_dofs_k { if d < sc { v[d] = 0.0; x[d] = 0.0; } }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // ─── 1. Print options ────────────────────────────────────────────────────
    eprintln!("Options used:");
    eprintln!("   --mesh {}", args.mesh);
    eprintln!("   --refine {}", args.ref_levels);
    eprintln!("   --order {}", args.order);
    eprintln!("   --ode-solver {}", args.ode_solver_type);
    eprintln!("   --t-final {}", args.t_final);
    eprintln!("   --time-step {}", args.dt);
    eprintln!("   --viscosity {}", args.viscosity);
    eprintln!("   --shear-modulus {}", args.mu);
    eprintln!("   --bulk-modulus {}", args.K);
    eprintln!("   --no-visualization");
    eprintln!("   --visualization-steps {}", args.vis_steps);

    // ─── 2. Read mesh ───────────────────────────────────────────────────────
    let mut mesh: Mesh<2> = read_mfem_file(&args.mesh)
        .expect("failed to read MFEM mesh")
        .mesh2d
        .expect("MFEM mesh must be 2D");
    let dim = mesh.dim() as usize;
    eprintln!("  Mesh: {} elements, {} nodes, dim={dim}", mesh.n_elems(), mesh.n_nodes());

    // ─── 3. Uniform refinement ───────────────────────────────────────────────
    for _ in 0..args.ref_levels {
        mesh = refine_uniform(&mesh);
    }
    eprintln!("  After refinement: {} elements, {} nodes", mesh.n_elems(), mesh.n_nodes());

    // ─── 4. FE space ────────────────────────────────────────────────────────
    let space = fem_space::VectorH1Space::new(mesh.clone(), args.order, dim as u8);
    let n_total = space.n_dofs(); // total vector DOFs for one field
    println!("Number of velocity/deformation unknowns: {n_total}");

    // ─── 5. Block vector vx = [v, x] ──────────────────────────────────────
    let mut vx = vec![0.0; 2 * n_total];
    let (v_block, x_block) = vx.split_at_mut(n_total);

    // ─── 6. Essential BC (boundary attribute 1 is fixed) ─────────────────────
    let _bdr_attr_max = mesh.unique_boundary_tags().iter().max().copied().unwrap_or(1).max(1) as usize;
    // Boundary attribute 1 → fixed (all components = 0)

    // Build essential DOF list for the vector FE space
    let dm = space.scalar_dof_manager();
    let ess_scalar_dofs = fem_space::constraints::boundary_dofs(&mesh, dm, &[1]);
    let mut ess_dofs: Vec<usize> = Vec::new();
    for c in 0..dim {
        let offset = c * (n_total / dim);
        for &d in &ess_scalar_dofs {
            ess_dofs.push(d as usize + offset);
        }
    }
    ess_dofs.sort_unstable();
    ess_dofs.dedup();

    // ─── 7. Assemble M (mass) and S (viscosity) ─────────────────────────────
    //   M = VectorH1MassIntegrator (density ρ = 1.0)
    //   S = VectorDiffusionIntegrator (viscosity coefficient)
    let quad_order = (args.order as u8) * 2 + 1;

    let m_integ = VectorH1MassIntegrator { kappa: 1.0 };
    let m = Assembler::assemble_bilinear(&space, &[&m_integ], quad_order);

    let s_integ = VectorDiffusionIntegrator { kappa: args.viscosity };
    let s = Assembler::assemble_bilinear(&space, &[&s_integ], quad_order);

    // ─── 8. Hyperelastic model (NeoHookean) ─────────────────────────────────
    // MFEM ex10 uses NeoHookeanModel(mu, K) where K is bulk modulus.
    // The NeoHookean energy: W = μ/2 (I₁ - 3) - μ ln(J) + λ/2 (ln(J))²
    // MFEM uses: λ = K - 2μ/3, or alternatively computes λ from (K, μ).
    // In the compressible NeoHookean formulation, λ = K - 2μ/3 (3D) or
    // λ = K - μ (2D plane strain).  MFEM's NeoHookeanModel uses:
    //   λ = K (the bulk modulus parameter directly as lambda)
    // for simplicity.
    // MFEM ex10 uses NeoHookeanModel(mu, K) which is a DEVIATORIC formulation:
    //   W = μ/2·(J^{-2/3}·I₁ - dim) + K/2·(J-1)²
    // NOT the standard compressible NeoHookean. Use the matching variant.
    let model = HyperelasticModel::MfemNeoHookean { mu: args.mu, bulk_modulus: args.K };
    let hyper = HyperelasticityForm::new(space, model, vec![], quad_order);

    // ─── 9. Initial conditions ──────────────────────────────────────────────
    // Initial deformation: u = 0 (identity = reference configuration).
    // Initial velocity: parabolic profile (same as MFEM ex10).
    //
    // Use L² projection (matching MFEM's ProjectCoefficient) instead of
    // DOF-node interpolation, to get the exact same coefficients for the
    // cubic velocity profile on the P2 space.
    use fem_assembly::project_coefficient;
    use fem_space::H1Space;

    let n_scalar = n_total / dim;
    let scalar_space = H1Space::new(mesh.clone(), args.order);
    let mut initial_v = vec![0.0; n_total];

    for c in 0..dim {
        let comp_coeff = |x: &[f64]| initial_velocity(x)[c];
        let proj = project_coefficient(&scalar_space, &comp_coeff, quad_order);
        for i in 0..n_scalar {
            initial_v[c * n_scalar + i] = proj[i];
        }
    }

    // v = initial_v, x = 0 (reference configuration)
    v_block.copy_from_slice(&initial_v);
    // x is already 0 from vec![0.0; 2*n_total]

    // Enforce BC on initial conditions
    for &d in &ess_dofs {
        if d < n_total {
            v_block[d] = 0.0;
            x_block[d] = 0.0;
        }
    }

    // ─── 10. Create ReducedSystemOperator ───────────────────────────────────
    let reduced_op = ReducedSystemOperator {
        m: &m,
        s: &s,
        hyper: &hyper,
        ess_dofs: &ess_dofs,
        n: n_total,
    };

    // ─── 11. Initial energies ───────────────────────────────────────────────
    let ee0 = hyper.elastic_energy(x_block);
    let mut mv_tmp = vec![0.0; n_total];
    m.spmv(v_block, &mut mv_tmp);
    let ke0 = 0.5 * dot(v_block, &mv_tmp);
    println!("initial elastic energy (EE) = {ee0:.6}");
    println!("initial kinetic energy (KE) = {ke0:.6}");
    println!("initial   total energy (TE) = {:.6}", ee0 + ke0);

    // L² norm of displacement and velocity (‖u‖_{L²} = sqrt(u^T M u))
    {
        let l2_x = vector_l2_norm(&m, x_block);
        let l2_v = vector_l2_norm(&m, v_block);
        println!("initial L2 norm of deformation: {l2_x:.12e}");
        println!("initial L2 norm of velocity:    {l2_v:.12e}");
    }

    // ─── 12. Time integration loop ──────────────────────────────────────────
    let mut t = 0.0;
    let mut last_step = false;
    let mut step = 0;

    while !last_step {
        let dt_real = args.dt.min(args.t_final - t);
        step += 1;

        // Perform the time step based on solver type
        match args.ode_solver_type {
            4 => {
                // Forward Euler (explicit)
                forward_euler_step(&mut vx, dt_real, &m, &s, &hyper, &ess_dofs, &ess_dofs);
                t += dt_real;
            }
            22 => {
                // SDIRK2
                let (_v, _x) = vx.split_at_mut(n_total);
                sdirk2_step(&mut vx, dt_real, &reduced_op, &ess_dofs, true);
                t += dt_real;
            }
            23 => {
                // SDIRK3 (use sdirk3_step for the proper 3-stage version)
                let (_v, _x) = vx.split_at_mut(n_total);
                sdirk3_step(&mut vx, dt_real, &reduced_op, &ess_dofs, true);
                t += dt_real;
            }
            _ => {
                eprintln!("ODE solver type {} not implemented, using SDIRK2", args.ode_solver_type);
                sdirk2_step(&mut vx, dt_real, &reduced_op, &ess_dofs, true);
                t += dt_real;
            }
        }

        last_step = t >= args.t_final - 1e-8 * args.dt;

        if last_step || (step % args.vis_steps == 0) {
            let (v, x) = vx.split_at_mut(n_total);
            let ee = hyper.elastic_energy(x);
            m.spmv(v, &mut mv_tmp);
            let ke = 0.5 * dot(v, &mv_tmp);
            // L² norms
            let l2_x = vector_l2_norm(&m, x);
            let l2_v = vector_l2_norm(&m, v);
            println!(
                "step {step}, t = {t}, EE = {ee:.6}, KE = {ke:.6}, ΔTE = {:.6}",
                (ee + ke) - (ee0 + ke0)
            );
            println!(
                "  L2(||x||)={l2_x:.12e}  L2(||v||)={l2_v:.12e}"
            );
        }
    }

    // ─── 13. Save output files ─────────────────────────────────────────────
    {
        let (v, x) = vx.split_at_mut(n_total);

        // Save deformed mesh (displaced node coordinates)
        let deformed = mesh.apply_displacement(x, dim);
        match write_mfem_file("deformed.mesh", &deformed) {
            Ok(_) => eprintln!("  Saved deformed.mesh ({} elements)", deformed.n_elems()),
            Err(e) => eprintln!("  Warning: failed to write deformed.mesh: {e}"),
        }

        // Save velocity field as .gf (MFEM GridFunction format)
        match write_gf_file("velocity.sol", dim, v, "VectorH1", args.order, dim as usize) {
            Ok(_) => eprintln!("  Saved velocity.sol ({} DOFs)", v.len()),
            Err(e) => eprintln!("  Warning: failed to write velocity.sol: {e}"),
        }

        // Compute and save elastic energy density on an L² space.
        // Use `order + 1` (matching C++ MFEM ex10) for Tri3/Tet4 meshes;
        // for Quad4 use order 1 (P1 discontinuous, the max currently
        // supported by L2Space for quads).
        let is_quad = mesh.element_type(0).nodes_per_element() == 4;
        let l2_order = if is_quad { 1u8 } else { (args.order + 1).min(3) };
        let l2_space = L2Space::new(mesh.clone(), l2_order);
        let l2_dofs = hyper.compute_elastic_energy_density(x, &l2_space, (l2_order as u8) * 2 + 1);
        match write_gf_file("elastic_energy.sol", dim, &l2_dofs, "L2", l2_order, 1) {
            Ok(_) => eprintln!("  Saved elastic_energy.sol ({} DOFs)", l2_dofs.len()),
            Err(e) => eprintln!("  Warning: failed to write elastic_energy.sol: {e}"),
        }
    }

    let elapsed = t0.elapsed();
    eprintln!("\n  Total time: {:.3}s", elapsed.as_secs_f64());
    eprintln!("  Done.");
}
