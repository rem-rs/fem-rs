//! # Example 10 �?Wave Equation (Newmark-beta)
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
//! cargo run --example mfem_ex10_wave_equation
//! cargo run --example mfem_ex10_wave_equation -- --n 32 --dt 0.0005 --T 1.0
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, SolverConfig, Newmark, NewmarkState};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

struct SolveResult {
    n: usize,
    dt: f64,
    requested_t_end: f64,
    final_time: f64,
    c: f64,
    n_dofs: usize,
    max_error: f64,
    rms_error: f64,
    solution_norm: f64,
    solution_checksum: f64,
    max_amplitude: f64,
    exact_factor: f64,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 10: Wave equation (Newmark-beta) ===");
    println!("  Mesh: {}x{}, c = {}", args.n, args.n, args.c);
    println!("  dt = {:.4e}, T = {}", args.dt, args.t_end);

    let result = solve_case(args.n, args.dt, args.t_end, args.c, 1.0, true);

    println!("  Confirmed mesh: {}x{}, c = {}", result.n, result.n, result.c);
    println!("  Confirmed dt = {:.4e}", result.dt);
    println!("  DOFs: {}", result.n_dofs);
    println!("\n  Requested T = {:.4e}, actual final t = {:.4e}", result.requested_t_end, result.final_time);
    println!("  Exact factor cos(omega*T) = {:.6e}", result.exact_factor);
    println!("  Max nodal error = {:.4e}", result.max_error);
    println!("  RMS nodal error = {:.4e}", result.rms_error);
    println!("  max|u(T)| = {:.6e}", result.max_amplitude);
    println!("  ||u(T)||_L2 = {:.4e}", result.solution_norm);
    println!("  checksum = {:.8e}", result.solution_checksum);

    assert!(result.max_error < 0.05,
        "Wave equation error too large: max_err={:.4e} (expected < 0.05)", result.max_error);
    println!("\nDone. Error within tolerance.");
}

fn solve_case(
    n: usize,
    dt: f64,
    t_end: f64,
    c: f64,
    initial_scale: f64,
    emit_progress: bool,
) -> SolveResult {
    // 1. Mesh and H1 space
    let mesh  = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let ndof  = space.n_dofs();
    if emit_progress {
        println!("  DOFs: {ndof}");
    }

    // 2. Assemble mass matrix M and stiffness matrix K (with c^2 = 1)
    let c2 = c * c;
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
        initial_scale * (PI * x[0]).sin() * (PI * x[1]).sin()
    }).collect();
    for &d in &bnd { u[d as usize] = 0.0; }

    // 5. Compute initial acceleration: solve M a0 = -K u0 (with BC)
    let mut ku = vec![0.0_f64; ndof];
    k_mat.spmv(&u, &mut ku);
    let rhs_init: Vec<f64> = ku.iter().map(|&v| -v).collect();
    let mut acc0 = vec![0.0_f64; ndof];
    let solve_cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, verbose: false, ..SolverConfig::default() };
    solve_cg(&m_mat, &rhs_init, &mut acc0, &solve_cfg).expect("initial acceleration solve failed");
    for &d in &bnd { acc0[d as usize] = 0.0; }

    let mut state = NewmarkState {
        vel: vec![0.0; ndof],
        acc: acc0,
    };

    // 6. Time-stepping with Newmark-beta
    let newmark = Newmark::default(); // beta=1/4, gamma=1/2
    let force = vec![0.0_f64; ndof]; // zero forcing
    let n_steps = (t_end / dt).round() as usize;
    if emit_progress {
        println!("  Steps: {n_steps}");
    }

    for step in 0..n_steps {
        newmark.step(&m_mat, &k_mat, &force, dt, &mut u, &mut state, &bnd);
        if emit_progress && (step + 1) % (n_steps / 5).max(1) == 0 {
            let t = (step + 1) as f64 * dt;
            let max_u = u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            println!("    t = {t:.4e}, max|u| = {max_u:.6e}");
        }
    }

    // 7. Compare with exact solution at T_final
    let t_final = n_steps as f64 * dt;
    let omega = PI * (2.0_f64).sqrt() * c; // fundamental frequency
    let exact_factor = (omega * t_final).cos();

    let mut max_err = 0.0_f64;
    let mut l2_err2 = 0.0_f64;
    for i in 0..ndof {
        let x = dm.dof_coord(i as u32);
        let u_ex = initial_scale * exact_factor * (PI * x[0]).sin() * (PI * x[1]).sin();
        let err = (u[i] - u_ex).abs();
        if err > max_err { max_err = err; }
        l2_err2 += err * err;
    }
    let l2_err = (l2_err2 / ndof as f64).sqrt();
    let solution_norm = u.iter().map(|value| value * value).sum::<f64>().sqrt();
    let solution_checksum = u
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let max_amplitude = u.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);

    SolveResult {
        n,
        dt,
        requested_t_end: t_end,
        final_time: t_final,
        c,
        n_dofs: ndof,
        max_error: max_err,
        rms_error: l2_err,
        solution_norm,
        solution_checksum,
        max_amplitude,
        exact_factor,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex10_wave_coarse_case_has_reasonable_accuracy() {
        let result = solve_case(16, 0.001, 0.5, 1.0, 1.0, false);
        assert_eq!(result.n_dofs, 289);
        assert!(result.max_error < 1.0e-2, "max nodal error too large: {}", result.max_error);
        assert!(result.rms_error < 5.0e-3, "RMS nodal error too large: {}", result.rms_error);
    }

    #[test]
    fn ex10_wave_spatial_refinement_improves_accuracy() {
        let coarse = solve_case(8, 0.005, 0.1, 1.0, 1.0, false);
        let fine = solve_case(16, 0.005, 0.1, 1.0, 1.0, false);
        assert!(fine.max_error < coarse.max_error,
            "refinement should reduce max error: coarse={} fine={}", coarse.max_error, fine.max_error);
        assert!(fine.rms_error < coarse.rms_error,
            "refinement should reduce RMS error: coarse={} fine={}", coarse.rms_error, fine.rms_error);
    }

    #[test]
    fn ex10_wave_short_horizon_dt_scan_remains_accurate_and_stable() {
        let dt_2e2 = solve_case(16, 0.02, 0.1, 1.0, 1.0, false);
        let dt_1e2 = solve_case(16, 0.01, 0.1, 1.0, 1.0, false);
        let dt_5e3 = solve_case(16, 0.005, 0.1, 1.0, 1.0, false);

        for result in [&dt_2e2, &dt_1e2, &dt_5e3] {
            assert!(result.max_error < 1.0e-3,
                "short-horizon max error should remain small across dt scan, got {}",
                result.max_error);
            assert!(result.rms_error < 6.0e-4,
                "short-horizon RMS error should remain small across dt scan, got {}",
                result.rms_error);
        }

        let rel_gap_21 = (dt_2e2.solution_norm - dt_1e2.solution_norm).abs()
            / dt_2e2.solution_norm.max(dt_1e2.solution_norm).max(1.0e-30);
        let rel_gap_10 = (dt_1e2.solution_norm - dt_5e3.solution_norm).abs()
            / dt_1e2.solution_norm.max(dt_5e3.solution_norm).max(1.0e-30);

        assert!(rel_gap_21 < 2.0e-4,
            "solution norm drift across dt scan is too large: dt=0.02 {} dt=0.01 {}",
            dt_2e2.solution_norm,
            dt_1e2.solution_norm);
        assert!(rel_gap_10 < 2.0e-4,
            "solution norm drift across dt scan is too large: dt=0.01 {} dt=0.005 {}",
            dt_1e2.solution_norm,
            dt_5e3.solution_norm);
    }

    #[test]
    fn ex10_wave_sign_reversed_initial_condition_flips_solution() {
        let positive = solve_case(16, 0.005, 0.1, 1.0, 1.0, false);
        let negative = solve_case(16, 0.005, 0.1, 1.0, -1.0, false);
        assert!((positive.solution_norm - negative.solution_norm).abs() < 1.0e-12);
        assert!((positive.max_amplitude - negative.max_amplitude).abs() < 1.0e-12);
        assert!((positive.solution_checksum + negative.solution_checksum).abs() < 1.0e-10,
            "solution checksum should flip sign: positive={} negative={}",
            positive.solution_checksum,
            negative.solution_checksum);
        assert!((positive.max_error - negative.max_error).abs() < 1.0e-12);
        assert!((positive.rms_error - negative.rms_error).abs() < 1.0e-12);
    }

    #[test]
    fn ex10_wave_actual_final_time_tracks_rounded_step_count() {
        let result = solve_case(16, 0.03, 0.1, 1.0, 1.0, false);
        assert!((result.requested_t_end - 0.1).abs() < 1.0e-12);
        assert!((result.final_time - 0.09).abs() < 1.0e-12, "unexpected final time: {}", result.final_time);
        let expected_factor = (PI * (2.0_f64).sqrt() * result.c * result.final_time).cos();
        assert!((result.exact_factor - expected_factor).abs() < 1.0e-14,
            "exact factor should use actual final time: got={} expected={}",
            result.exact_factor,
            expected_factor);
        assert!(result.max_error < 1.0e-3, "non-dividing-step max error too large: {}", result.max_error);
    }

    #[test]
    fn ex10_wave_zero_initial_condition_gives_trivial_solution() {
        let result = solve_case(16, 0.005, 0.1, 1.0, 0.0, false);
        assert!(result.solution_norm < 1.0e-14, "expected zero solution norm, got {}", result.solution_norm);
        assert!(result.solution_checksum.abs() < 1.0e-14,
            "expected zero checksum, got {}", result.solution_checksum);
        assert!(result.max_amplitude < 1.0e-14, "expected zero amplitude, got {}", result.max_amplitude);
        assert!(result.max_error < 1.0e-14, "expected zero max error, got {}", result.max_error);
        assert!(result.rms_error < 1.0e-14, "expected zero RMS error, got {}", result.rms_error);
    }

    #[test]
    fn ex10_wave_dof_count_matches_p1_h1_formula() {
        for &n in &[8usize, 12usize, 16usize] {
            let result = solve_case(n, 0.005, 0.1, 1.0, 1.0, false);
            assert_eq!(result.n_dofs, (n + 1) * (n + 1));
        }
    }

    #[test]
    fn ex10_wave_zero_speed_keeps_initial_state() {
        let result = solve_case(16, 0.005, 0.1, 0.0, 1.0, false);
        assert!((result.exact_factor - 1.0).abs() < 1.0e-14,
            "with c=0 exact factor should remain 1, got {}", result.exact_factor);
        assert!(result.max_error < 1.0e-12,
            "with c=0 solution should stay at initial state, max_error={}", result.max_error);
        assert!(result.rms_error < 1.0e-12,
            "with c=0 solution should stay at initial state, rms_error={}", result.rms_error);
        assert!(result.max_amplitude > 0.5,
            "with c=0 amplitude should remain nontrivial, got {}", result.max_amplitude);
    }
}

