//! # Example 10 — Wave Equation (Newmark-beta)
//!
//! Solve the wave equation on the unit square [0,1]^2:
//!   d^2 u/dt^2 - c^2 Delta u = 0
//! with homogeneous Dirichlet BCs on dOmega and initial condition
//!   u(x,y,0) = sin(pi x) sin(pi y),  du/dt(x,y,0) = 0.
//!
//! Exact solution: u(x,y,t) = sin(pi x) sin(pi y) cos(c sqrt(2) pi t)
//! with c=1, the fundamental frequency is omega = pi sqrt(2).
//!
//! Uses P1 FEM for spatial discretization and Newmark-beta (beta=1/4, gamma=1/2)
//! for time integration.
//!
//! ## Usage
//! ```
//! cargo run --example ex10_wave_equation
//! cargo run --example ex10_wave_equation -- --n 32 --dt 0.0005 --T 1.0
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, SolverConfig, Newmark, NewmarkState};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 10: Wave equation (Newmark-beta) ===");
    println!("  Mesh: {}x{}, c = {}", args.n, args.n, args.c);
    println!("  dt = {:.4e}, T = {}", args.dt, args.t_end);

    // 1. Mesh and H1 space
    let mesh  = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let ndof  = space.n_dofs();
    println!("  DOFs: {ndof}");

    // 2. Assemble mass matrix M and stiffness matrix K (with c^2 = 1)
    let c2 = args.c * args.c;
    let mut k_mat = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: c2 }], 3,
    );
    let mut m_mat = Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: 1.0 }], 3,
    );

    // 3. Boundary DOFs (Dirichlet u=0 on all boundaries)
    let dm  = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let vals = vec![0.0_f64; bnd.len()];
    let mut dummy_rhs = vec![0.0_f64; ndof];
    apply_dirichlet(&mut k_mat, &mut dummy_rhs, &bnd, &vals);
    apply_dirichlet(&mut m_mat, &mut dummy_rhs, &bnd, &vals);

    // 4. Initial condition: u0 = sin(pi x) sin(pi y)
    let mut u: Vec<f64> = (0..ndof).map(|i| {
        let x = dm.dof_coord(i as u32);
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }).collect();
    for &d in &bnd { u[d as usize] = 0.0; }

    // 5. Compute initial acceleration: solve M a0 = -K u0 (with BC)
    let mut ku = vec![0.0_f64; ndof];
    k_mat.spmv(&u, &mut ku);
    let rhs_init: Vec<f64> = ku.iter().map(|&v| -v).collect();
    let mut acc0 = vec![0.0_f64; ndof];
    let solve_cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, verbose: false };
    solve_cg(&m_mat, &rhs_init, &mut acc0, &solve_cfg).expect("initial acceleration solve failed");
    for &d in &bnd { acc0[d as usize] = 0.0; }

    let mut state = NewmarkState {
        vel: vec![0.0; ndof],
        acc: acc0,
    };

    // 6. Time-stepping with Newmark-beta
    let newmark = Newmark::default(); // beta=1/4, gamma=1/2
    let force = vec![0.0_f64; ndof]; // zero forcing
    let dt    = args.dt;
    let t_end = args.t_end;
    let n_steps = (t_end / dt).round() as usize;
    println!("  Steps: {n_steps}");

    for step in 0..n_steps {
        newmark.step(&m_mat, &k_mat, &force, dt, &mut u, &mut state, &bnd);
        if (step + 1) % (n_steps / 5).max(1) == 0 {
            let t = (step + 1) as f64 * dt;
            let max_u = u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            println!("    t = {t:.4e}, max|u| = {max_u:.6e}");
        }
    }

    // 7. Compare with exact solution at T_final
    let t_final = n_steps as f64 * dt;
    let omega = PI * (2.0_f64).sqrt(); // fundamental frequency
    let exact_factor = (omega * t_final).cos();

    let mut max_err = 0.0_f64;
    let mut l2_err2 = 0.0_f64;
    for i in 0..ndof {
        let x = dm.dof_coord(i as u32);
        let u_ex = exact_factor * (PI * x[0]).sin() * (PI * x[1]).sin();
        let err = (u[i] - u_ex).abs();
        if err > max_err { max_err = err; }
        l2_err2 += err * err;
    }
    let l2_err = (l2_err2 / ndof as f64).sqrt();

    println!("\n  t_final = {t_final:.4e}");
    println!("  Exact factor cos(omega*T) = {exact_factor:.6e}");
    println!("  Max nodal error = {max_err:.4e}");
    println!("  RMS nodal error = {l2_err:.4e}");

    assert!(max_err < 0.05,
        "Wave equation error too large: max_err={max_err:.4e} (expected < 0.05)");
    println!("\nDone. Error within tolerance.");
}

// CLI argument parsing
struct Args {
    n:     usize,
    dt:    f64,
    t_end: f64,
    c:     f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 16, dt: 0.001, t_end: 0.5, c: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"  => { a.n     = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--dt" => { a.dt    = it.next().unwrap_or("0.001".into()).parse().unwrap_or(0.001); }
            "--T"  => { a.t_end = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5); }
            "--c"  => { a.c     = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            _ => {}
        }
    }
    a
}
