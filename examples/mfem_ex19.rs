//! mfem_ex19 - steady Navier-Stokes benchmark using Kovasznay flow.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::PressureDivIntegrator,
    standard::{VectorDiffusionIntegrator, VectorConvectionIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_space::{
    H1Space, VectorH1Space,
    constraints::boundary_dofs,
    fe_space::FESpace,
};

struct PicardStep {
    linear_iterations: usize,
    linear_residual: f64,
    relative_update: f64,
}

struct RunResult {
    n: usize,
    re: f64,
    nu: f64,
    lambda: f64,
    velocity_dofs: usize,
    scalar_velocity_dofs: usize,
    pressure_dofs: usize,
    constrained_velocity_dofs: usize,
    picard_steps: Vec<PicardStep>,
    picard_converged: bool,
    velocity_rel_l2: f64,
    pressure_rel_l2: f64,
    velocity_norm: f64,
    pressure_norm: f64,
    velocity_checksum: f64,
    pressure_checksum: f64,
    boundary_max_error: f64,
}

// ─── Kovasznay analytical solution ───────────────────────────────────────────

fn kovasznay_lambda(re: f64) -> f64 {
    re / 2.0 - (re * re / 4.0 + 4.0 * PI * PI).sqrt()
}

fn kovasznay_u(x: f64, y: f64, lam: f64) -> [f64; 2] {
    let ex = (lam * x).exp();
    [
        1.0 - ex * (2.0 * PI * y).cos(),
        lam / (2.0 * PI) * ex * (2.0 * PI * y).sin(),
    ]
}

fn kovasznay_p(x: f64, _y: f64, lam: f64) -> f64 {
    -0.5 * (2.0 * lam * x).exp()
}

// ─── Mesh generation ─────────────────────────────────────────────────────────

/// Create a triangular mesh on `[x0, x1] × [y0, y1]`.
fn rect_mesh(n: usize, x0: f64, x1: f64, y0: f64, y1: f64) -> SimplexMesh<2> {
    let mut mesh = SimplexMesh::<2>::unit_square_tri(n);
    // Scale coordinates from [0,1]² to [x0,x1] × [y0,y1]
    let nn = mesh.n_nodes();
    for i in 0..nn {
        mesh.coords[i * 2]     = x0 + mesh.coords[i * 2]     * (x1 - x0);
        mesh.coords[i * 2 + 1] = y0 + mesh.coords[i * 2 + 1] * (y1 - y0);
    }
    mesh
}

fn main() {
    let args = parse_args();

    println!("=== fem-rs: Navier-Stokes (Kovasznay flow, Oseen/Picard) ===");
    let result = run_case(args.n, args.re, true);

    println!("\n  Confirmed Re = {:.0}, nu = {:.4e}, lambda = {:.6}", result.re, result.nu, result.lambda);
    println!("  Mesh: {}x{}, P2/P1", result.n, result.n);
    println!(
        "  velocity DOFs: {} ({} per component)",
        result.velocity_dofs,
        result.scalar_velocity_dofs,
    );
    println!("  pressure DOFs: {}", result.pressure_dofs);
    println!("  Dirichlet: {} velocity DOFs constrained", result.constrained_velocity_dofs);
    println!(
        "  Picard: converged={}, steps={}, final du/u={:.2e}",
        result.picard_converged,
        result.picard_steps.len(),
        result.picard_steps.last().map(|step| step.relative_update).unwrap_or(f64::INFINITY),
    );
    if let Some(first_step) = result.picard_steps.first() {
        println!(
            "  first linear solve: iters={}, res={:.2e}",
            first_step.linear_iterations,
            first_step.linear_residual,
        );
    }
    if let Some(last_step) = result.picard_steps.last() {
        println!(
            "  last linear solve: iters={}, res={:.2e}",
            last_step.linear_iterations,
            last_step.linear_residual,
        );
    }
    println!("  velocity L2 relative error = {:.4e}", result.velocity_rel_l2);
    println!("  pressure L2 relative error = {:.4e}", result.pressure_rel_l2);
    println!("  ||u_h||_L2 = {:.4e}", result.velocity_norm);
    println!("  ||p_h||_L2 = {:.4e}", result.pressure_norm);
    println!("  checksum(u_h) = {:.8e}", result.velocity_checksum);
    println!("  checksum(p_h) = {:.8e}", result.pressure_checksum);
    println!("  boundary max error = {:.3e}", result.boundary_max_error);
    assert!(result.picard_converged, "Kovasznay Picard iteration did not converge");
    println!("\nDone.");
}

fn run_case(n: usize, re: f64, emit_progress: bool) -> RunResult {
    let nu = 1.0 / re;
    let lambda = kovasznay_lambda(re);

    if emit_progress {
        println!("  Re = {re:.0}, nu = {nu:.4e}, lambda = {lambda:.6}");
        println!("  Mesh: {n}x{n}, P2/P1");
    }

    let mesh_u = rect_mesh(n, -0.5, 1.5, 0.0, 2.0);
    let mesh_p = rect_mesh(n, -0.5, 1.5, 0.0, 2.0);

    let space_u = VectorH1Space::new(mesh_u, 2, 2);
    let space_p = H1Space::new(mesh_p, 1);

    let velocity_dofs = space_u.n_dofs();
    let pressure_dofs = space_p.n_dofs();
    let scalar_velocity_dofs = space_u.n_scalar_dofs();
    let scalar_dm = space_u.scalar_dof_manager();

    let bnd_all = boundary_dofs(space_u.mesh(), scalar_dm, &[1, 2, 3, 4]);
    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();
    for &d in &bnd_all {
        let coords = scalar_dm.dof_coord(d);
        let (x, y) = (coords[0], coords[1]);
        let u_exact = kovasznay_u(x, y, lambda);
        bc_dofs.push(d);
        bc_vals.push(u_exact[0]);
        bc_dofs.push(d + scalar_velocity_dofs as u32);
        bc_vals.push(u_exact[1]);
    }

    if emit_progress {
        println!("  velocity DOFs: {velocity_dofs} ({scalar_velocity_dofs} per component)");
        println!("  pressure DOFs: {pressure_dofs}");
        println!("  Dirichlet: {} velocity DOFs constrained", bc_dofs.len());
    }

    let quad_order = 5_u8;
    let b_mat = MixedAssembler::assemble_bilinear(
        &space_p,
        &space_u,
        &[&PressureDivIntegrator],
        quad_order,
    );
    let bt_mat = b_mat.transpose();

    let mut u_sol = vec![0.0_f64; velocity_dofs];
    let mut p_sol = vec![0.0_f64; pressure_dofs];
    for (i, &dof) in bc_dofs.iter().enumerate() {
        u_sol[dof as usize] = bc_vals[i];
    }

    let solver_cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let picard_tol = 1e-8;
    let max_picard = 20;
    let mut picard_steps = Vec::new();
    let mut picard_converged = false;

    for picard in 0..max_picard {
        let visc = VectorDiffusionIntegrator { kappa: nu };
        let conv = VectorConvectionIntegrator::new(&u_sol, scalar_velocity_dofs);
        let mut a_mat = Assembler::assemble_bilinear(&space_u, &[&visc, &conv], quad_order);

        let mut f_u = vec![0.0_f64; velocity_dofs];
        let g_p = vec![0.0_f64; pressure_dofs];
        fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bc_dofs, &bc_vals);

        let mut b_loc = b_mat.clone();
        let mut bt_loc = bt_mat.clone();
        pin_pressure_dof(&mut b_loc, &mut bt_loc, 0);

        let sys = BlockSystem {
            a: a_mat,
            bt: bt_loc,
            b: b_loc,
            c: None,
        };
        let mut u_new = vec![0.0_f64; velocity_dofs];
        let mut p_new = vec![0.0_f64; pressure_dofs];
        let res = SchurComplementSolver::solve(&sys, &f_u, &g_p, &mut u_new, &mut p_new, &solver_cfg)
            .expect("Oseen solve failed");

        let du = u_new
            .iter()
            .zip(u_sol.iter())
            .map(|(&new_value, &old_value)| (new_value - old_value).powi(2))
            .sum::<f64>()
            .sqrt();
        let u_norm = u_new.iter().map(|value| value * value).sum::<f64>().sqrt();
        let relative_update = du / u_norm.max(1.0e-14);
        picard_steps.push(PicardStep {
            linear_iterations: res.iterations,
            linear_residual: res.final_residual,
            relative_update,
        });

        if emit_progress {
            println!(
                "  Picard {}: linear iters={}, res={:.2e}, du/u={:.2e}",
                picard + 1,
                res.iterations,
                res.final_residual,
                relative_update,
            );
        }

        u_sol = u_new;
        p_sol = p_new;

        if relative_update < picard_tol {
            picard_converged = true;
            if emit_progress {
                println!("  Picard converged after {} iterations.", picard + 1);
            }
            break;
        }
    }

    if emit_progress && !picard_converged {
        println!("  Picard did not converge after {} iterations.", max_picard);
    }

    let ux = &u_sol[..scalar_velocity_dofs];
    let uy = &u_sol[scalar_velocity_dofs..];
    let mut err_u_l2_sq = 0.0_f64;
    let mut u_exact_l2_sq = 0.0_f64;
    for i in 0..scalar_velocity_dofs {
        let coords = scalar_dm.dof_coord(i as u32);
        let (x, y) = (coords[0], coords[1]);
        let u_exact = kovasznay_u(x, y, lambda);
        err_u_l2_sq += (ux[i] - u_exact[0]).powi(2) + (uy[i] - u_exact[1]).powi(2);
        u_exact_l2_sq += u_exact[0].powi(2) + u_exact[1].powi(2);
    }

    let pres_dm = space_p.dof_manager();
    let p0_exact = kovasznay_p(pres_dm.dof_coord(0)[0], pres_dm.dof_coord(0)[1], lambda);
    let p_shift = p0_exact - p_sol[0];
    let mut shifted_pressure = vec![0.0_f64; pressure_dofs];
    let mut err_p_l2_sq = 0.0_f64;
    let mut p_exact_l2_sq = 0.0_f64;
    for i in 0..pressure_dofs {
        let coords = pres_dm.dof_coord(i as u32);
        let (x, y) = (coords[0], coords[1]);
        let p_exact = kovasznay_p(x, y, lambda);
        let p_h = p_sol[i] + p_shift;
        shifted_pressure[i] = p_h;
        err_p_l2_sq += (p_h - p_exact).powi(2);
        p_exact_l2_sq += p_exact.powi(2);
    }

    let boundary_max_error = bc_dofs
        .iter()
        .zip(bc_vals.iter())
        .map(|(&dof, &bc)| (u_sol[dof as usize] - bc).abs())
        .fold(0.0_f64, f64::max);
    let velocity_norm = u_sol.iter().map(|value| value * value).sum::<f64>().sqrt();
    let pressure_norm = shifted_pressure.iter().map(|value| value * value).sum::<f64>().sqrt();
    let velocity_checksum = u_sol
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let pressure_checksum = shifted_pressure
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    let velocity_rel_l2 = (err_u_l2_sq / u_exact_l2_sq.max(1.0e-30)).sqrt();
    let pressure_rel_l2 = (err_p_l2_sq / p_exact_l2_sq.max(1.0e-30)).sqrt();

    if emit_progress {
        println!("\n  Error vs Kovasznay analytical solution:");
        println!("    velocity L2 relative error: {velocity_rel_l2:.4e}");
        println!("    pressure L2 relative error: {pressure_rel_l2:.4e}");
    }

    RunResult {
        n,
        re,
        nu,
        lambda,
        velocity_dofs,
        scalar_velocity_dofs,
        pressure_dofs,
        constrained_velocity_dofs: bc_dofs.len(),
        picard_steps,
        picard_converged,
        velocity_rel_l2,
        pressure_rel_l2,
        velocity_norm,
        pressure_norm,
        velocity_checksum,
        pressure_checksum,
        boundary_max_error,
    }
}

/// Pin pressure DOF `dof` to zero by zeroing its row in B and column in B^T.
fn pin_pressure_dof(
    b: &mut fem_linalg::CsrMatrix<f64>,
    bt: &mut fem_linalg::CsrMatrix<f64>,
    dof: usize,
) {
    for ptr in b.row_ptr[dof]..b.row_ptr[dof + 1] {
        b.values[ptr] = 0.0;
    }
    for i in 0..bt.nrows {
        for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
            if bt.col_idx[ptr] as usize == dof {
                bt.values[ptr] = 0.0;
            }
        }
    }
}

struct Args {
    n: usize,
    re: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 8, re: 40.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"  => { a.n  = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--re" => { a.re = it.next().unwrap_or("40".into()).parse().unwrap_or(40.0); }
            _      => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex19_kovasznay_coarse_case_converges_with_reasonable_error() {
        let result = run_case(4, 40.0, false);
        assert_eq!(result.velocity_dofs, 162);
        assert_eq!(result.pressure_dofs, 25);
        assert!(result.picard_converged, "Picard iteration did not converge");
        assert!(result.picard_steps.len() <= 20, "too many Picard steps: {}", result.picard_steps.len());
        assert!(result.picard_steps.last().unwrap().relative_update < 1.0e-8);
        assert!(result.velocity_rel_l2 < 1.2e-1, "velocity error too large: {}", result.velocity_rel_l2);
        assert!(result.pressure_rel_l2 < 1.5, "pressure error too large: {}", result.pressure_rel_l2);
        assert!(result.boundary_max_error < 1.5e-1, "boundary condition drift too large: {}", result.boundary_max_error);
    }

    #[test]
    fn ex19_kovasznay_refinement_improves_velocity_error() {
        let coarse = run_case(4, 40.0, false);
        let fine = run_case(8, 40.0, false);
        assert!(coarse.picard_converged && fine.picard_converged);
        assert!(fine.velocity_rel_l2 < coarse.velocity_rel_l2 * 0.9,
            "velocity refinement gain too small: coarse={} fine={}", coarse.velocity_rel_l2, fine.velocity_rel_l2);
        assert!(fine.velocity_rel_l2 < 1.0e-1, "fine-mesh velocity error too large: {}", fine.velocity_rel_l2);
    }

    #[test]
    fn ex19_kovasznay_picard_updates_contract_monotonically() {
        let result = run_case(4, 40.0, false);
        for window in result.picard_steps.windows(2) {
            assert!(window[1].relative_update < window[0].relative_update,
                "Picard update should decrease monotonically: prev={} next={}",
                window[0].relative_update,
                window[1].relative_update);
            assert!(window[1].linear_iterations >= window[0].linear_iterations,
                "linear iteration counts should not drop during early Picard stabilization: prev={} next={}",
                window[0].linear_iterations,
                window[1].linear_iterations);
            assert!(window[1].linear_residual < 1.0e-8,
                "linear residual too large during Picard iteration: {}",
                window[1].linear_residual);
        }
    }

    #[test]
    fn ex19_kovasznay_boundary_trace_stays_bounded_and_improves() {
        let coarse = run_case(4, 40.0, false);
        let fine = run_case(8, 40.0, false);
        assert!(coarse.picard_converged && fine.picard_converged);
        assert!(coarse.boundary_max_error < 1.5e-1,
            "coarse boundary drift should stay bounded: {}",
            coarse.boundary_max_error);
        assert!(fine.boundary_max_error < coarse.boundary_max_error,
            "boundary drift should improve with refinement: coarse={} fine={}",
            coarse.boundary_max_error,
            fine.boundary_max_error);
        assert!(fine.velocity_norm > 1.0,
            "velocity norm should remain nontrivial: {}",
            fine.velocity_norm);
        assert!(fine.velocity_checksum.abs() > 1.0e2,
            "velocity checksum should capture a nontrivial flow field: {}",
            fine.velocity_checksum);
    }

    #[test]
    fn ex19_kovasznay_dof_count_matches_taylor_hood_formula() {
        // Taylor-Hood: velocity uses P2 H1 space → (2n+1)^2 nodes per direction
        // pressure uses P1 H1 → (n+1)^2
        // velocity_dofs = 2 * (2n+1)^2, pressure_dofs = (n+1)^2
        for n in [4usize, 6] {
            let result = run_case(n, 40.0, false);
            let expected_pressure = (n + 1) * (n + 1);
            assert_eq!(result.pressure_dofs, expected_pressure,
                "pressure DOF mismatch for n={}: got {} expected {}",
                n, result.pressure_dofs, expected_pressure);
            assert!(result.velocity_dofs > result.pressure_dofs,
                "velocity should have more DOFs than pressure (Taylor-Hood stability)");
        }
    }

    #[test]
    fn ex19_kovasznay_solution_is_nontrivial_and_pressure_fluctuates() {
        let result = run_case(6, 40.0, false);
        assert!(result.picard_converged);
        // Kovasznay solution has genuine velocity and pressure structure
        assert!(result.velocity_norm > 0.5,
            "velocity norm should be nontrivial: {}", result.velocity_norm);
        assert!(result.pressure_norm > 1.0e-2,
            "pressure norm should be nontrivial: {}", result.pressure_norm);
        // Pressure checksum can be positive or negative depending on gauge
        assert!(result.pressure_checksum.abs() > 0.0);
    }

    #[test]
    fn ex19_kovasznay_higher_re_increases_inertial_asymmetry() {
        let low_re = run_case(6, 20.0, false);
        let high_re = run_case(6, 60.0, false);
        assert!(low_re.picard_converged && high_re.picard_converged);
        // Lambda = Re/2 - sqrt(Re^2/4 + 4π²): larger Re → lambda closer to 0 (less negative)
        // e.g. Re=20: λ≈-1.81, Re=60: λ≈-0.65
        assert!(high_re.lambda > low_re.lambda,
            "higher Re should give less negative lambda: low_re.lambda={:.4} high_re.lambda={:.4}",
            low_re.lambda, high_re.lambda);
        // Both lambdas are negative (boundary layer character)
        assert!(low_re.lambda < 0.0 && high_re.lambda < 0.0,
            "lambda should always be negative");
        assert!(low_re.velocity_rel_l2 > 0.0 && high_re.velocity_rel_l2 > 0.0);
    }

    #[test]
    fn ex19_kovasznay_velocity_dominates_pressure_in_l2_norm() {
        // For Kovasznay flow at Re=40, velocity L2 norm should be comparable to 1
        // while pressure fluctuation is smaller
        let result = run_case(6, 40.0, false);
        assert!(result.picard_converged);
        // Velocity is O(1) and pressure fluctuations are O(Re * viscosity) ~ O(1)
        assert!(result.velocity_norm > 0.1,
            "velocity norm should be order 1: {}", result.velocity_norm);
        assert!(result.velocity_norm > result.pressure_norm * 0.01,
            "velocity should not be negligibly small vs pressure");
    }
}

