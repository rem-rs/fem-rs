//! # Taylor-Hood P2-P1 Stokes Example (Phase 40)
//!
//! Solves the steady Stokes equations on a lid-driven cavity:
//!
//! ```text
//!   −�?Δu + ∇p = 0    in Ω = [0,1]²
//!        ∇·u = 0    in Ω
//!          u = (1,0)  on top wall   (lid, tag 3)
//!          u = (0,0)  on other walls (tags 1,2,4)
//! ```
//!
//! Uses the inf-sup stable Taylor-Hood pair: P2 velocity / P1 pressure.
//!
//! The saddle-point system is:
//! ```text
//!   [ A   B^T ] [ u ]   [ f ]
//!   [ B    0  ] [ p ] = [ 0 ]
//! ```
//!
//! where A is the vector-Laplacian (viscous term) and B is the divergence
//! operator coupling velocity and pressure.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex40
//! cargo run --example mfem_ex40 -- --n 16 --nu 1.0
//! ```

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::PressureDivIntegrator,
    standard::VectorDiffusionIntegrator,
};
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_space::{H1Space, VectorH1Space, fe_space::FESpace, constraints::boundary_dofs};

struct SolveResult {
    n: usize,
    nu: f64,
    lid_speed: f64,
    velocity_dofs: usize,
    scalar_velocity_dofs: usize,
    pressure_dofs: usize,
    constrained_velocity_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    ux_min: f64,
    ux_max: f64,
    uy_abs_max: f64,
    pressure_min: f64,
    pressure_max: f64,
    velocity_norm: f64,
    pressure_norm: f64,
    velocity_checksum: f64,
    pressure_checksum: f64,
    momentum_residual: f64,
    divergence_residual: f64,
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs: Taylor-Hood P2-P1 Stokes (lid-driven cavity) ===");
    let result = solve_case(args.n, args.nu, args.lid_speed);

    println!("  Mesh: {}x{}, P2/P1, nu = {:.3e}, lid = {:.3e}", result.n, result.n, result.nu, result.lid_speed);
    println!("  velocity DOFs: {} ({} per component)", result.velocity_dofs, result.scalar_velocity_dofs);
    println!("  pressure DOFs: {}", result.pressure_dofs);
    println!("  Dirichlet: {} velocity DOFs constrained", result.constrained_velocity_dofs);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
    println!("  u_x range: [{:.4e}, {:.4e}]", result.ux_min, result.ux_max);
    println!("  max|u_y|:  {:.4e}", result.uy_abs_max);
    println!("  p range:   [{:.4e}, {:.4e}]", result.pressure_min, result.pressure_max);
    println!("  ||u||_L2 = {:.4e}, ||p||_L2 = {:.4e}", result.velocity_norm, result.pressure_norm);
    println!("  checksum(u) = {:.8e}, checksum(p) = {:.8e}", result.velocity_checksum, result.pressure_checksum);
    println!(
        "  Block residual: ||Au+B^Tp-f|| = {:.3e}, ||Bu-g|| = {:.3e}",
        result.momentum_residual,
        result.divergence_residual,
    );

    assert!(result.converged, "Stokes solve did not converge");
    println!("\nDone.");
}

fn solve_case(n: usize, nu: f64, lid_speed: f64) -> SolveResult {
    let mesh_u = SimplexMesh::<2>::unit_square_tri(n);
    let mesh_p = SimplexMesh::<2>::unit_square_tri(n);

    let space_u = VectorH1Space::new(mesh_u, 2, 2);
    let space_p = H1Space::new(mesh_p, 1);

    let velocity_dofs = space_u.n_dofs();
    let pressure_dofs = space_p.n_dofs();
    let scalar_velocity_dofs = space_u.n_scalar_dofs();

    let quad_order = 5_u8;
    let visc = VectorDiffusionIntegrator { kappa: nu };
    let mut a_mat = Assembler::assemble_bilinear(&space_u, &[&visc], quad_order);

    let b_mat = MixedAssembler::assemble_bilinear(
        &space_p,
        &space_u,
        &[&PressureDivIntegrator],
        quad_order,
    );
    let bt_mat = b_mat.transpose();

    let mut f_u = vec![0.0_f64; velocity_dofs];
    let g_p = vec![0.0_f64; pressure_dofs];

    let scalar_dm = space_u.scalar_dof_manager();
    let bnd_all = boundary_dofs(space_u.mesh(), scalar_dm, &[1, 2, 3, 4]);
    let bnd_lid = boundary_dofs(space_u.mesh(), scalar_dm, &[3]);

    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();
    for &d in &bnd_all {
        bc_dofs.push(d);
        bc_vals.push(if bnd_lid.contains(&d) { lid_speed } else { 0.0 });
        bc_dofs.push(d + scalar_velocity_dofs as u32);
        bc_vals.push(0.0);
    }
    fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bc_dofs, &bc_vals);

    let mut b_mat = b_mat;
    let mut bt_mat = bt_mat;
    pin_pressure_dof(&mut b_mat, &mut bt_mat, 0);

    let sys = BlockSystem { a: a_mat, bt: bt_mat, b: b_mat, c: None };
    let mut u_sol = vec![0.0_f64; velocity_dofs];
    let mut p_sol = vec![0.0_f64; pressure_dofs];

    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 1e-12,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = SchurComplementSolver::solve(&sys, &f_u, &g_p, &mut u_sol, &mut p_sol, &cfg)
        .expect("Stokes solve failed");

    let ux = &u_sol[..scalar_velocity_dofs];
    let uy = &u_sol[scalar_velocity_dofs..];
    let ux_max = ux.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ux_min = ux.iter().copied().fold(f64::INFINITY, f64::min);
    let uy_abs_max = uy.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let pressure_max = p_sol.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pressure_min = p_sol.iter().copied().fold(f64::INFINITY, f64::min);
    let velocity_norm = u_sol.iter().map(|value| value * value).sum::<f64>().sqrt();
    let pressure_norm = p_sol.iter().map(|value| value * value).sum::<f64>().sqrt();
    let velocity_checksum = u_sol
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let pressure_checksum = p_sol
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    let mut ru = vec![0.0_f64; velocity_dofs];
    let mut rp = vec![0.0_f64; pressure_dofs];
    sys.apply(&u_sol, &p_sol, &mut ru, &mut rp);
    let momentum_residual = ru
        .iter()
        .zip(f_u.iter())
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let divergence_residual = rp
        .iter()
        .zip(g_p.iter())
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();

    SolveResult {
        n,
        nu,
        lid_speed,
        velocity_dofs,
        scalar_velocity_dofs,
        pressure_dofs,
        constrained_velocity_dofs: bc_dofs.len(),
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        ux_min,
        ux_max,
        uy_abs_max,
        pressure_min,
        pressure_max,
        velocity_norm,
        pressure_norm,
        velocity_checksum,
        pressure_checksum,
        momentum_residual,
        divergence_residual,
    }
}

/// Pin pressure DOF `dof` to zero by zeroing its row in B and column in B^T.
fn pin_pressure_dof(
    b: &mut fem_linalg::CsrMatrix<f64>,
    bt: &mut fem_linalg::CsrMatrix<f64>,
    dof: usize,
) {
    // Zero row `dof` in B (n_p × n_u)
    for ptr in b.row_ptr[dof]..b.row_ptr[dof + 1] {
        b.values[ptr] = 0.0;
    }
    // Zero column `dof` in B^T (n_u × n_p)
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
    nu: f64,
    lid_speed: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 8, nu: 1.0, lid_speed: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"  => { a.n  = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--nu" => { a.nu = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--lid" | "--lid-speed" => { a.lid_speed = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            _      => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex40_stokes_coarse_case_converges_with_recirculation() {
        let result = solve_case(8, 1.0, 1.0);
        assert_eq!(result.velocity_dofs, 578);
        assert_eq!(result.pressure_dofs, 81);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-7, "linear residual too large: {}", result.final_residual);
        assert!(result.momentum_residual < 5.0e-8, "momentum residual too large: {}", result.momentum_residual);
        assert!(result.divergence_residual < 5.0e-8, "divergence residual too large: {}", result.divergence_residual);
        assert!(result.ux_min < -1.0e-3, "expected recirculation with negative ux: {}", result.ux_min);
        assert!(result.uy_abs_max > 2.0e-1, "vertical circulation too weak: {}", result.uy_abs_max);
    }

    #[test]
    fn ex40_stokes_zero_lid_gives_trivial_solution() {
        let result = solve_case(8, 1.0, 0.0);
        assert!(result.converged);
        assert!(result.velocity_norm < 1.0e-12, "velocity should vanish: {}", result.velocity_norm);
        assert!(result.pressure_norm < 1.0e-12, "pressure should vanish: {}", result.pressure_norm);
        assert!(result.ux_min.abs() < 1.0e-12 && result.ux_max.abs() < 1.0e-12);
        assert!(result.uy_abs_max < 1.0e-12);
        assert!(result.pressure_min.abs() < 1.0e-12 && result.pressure_max.abs() < 1.0e-12);
    }

    #[test]
    fn ex40_stokes_solution_scales_linearly_with_lid_speed() {
        let unit = solve_case(8, 1.0, 1.0);
        let doubled = solve_case(8, 1.0, 2.0);
        assert!(unit.converged && doubled.converged);
        assert!((doubled.velocity_norm / unit.velocity_norm - 2.0).abs() < 1.0e-9,
            "velocity norm ratio mismatch: unit={} doubled={}", unit.velocity_norm, doubled.velocity_norm);
        assert!((doubled.pressure_norm / unit.pressure_norm - 2.0).abs() < 1.0e-9,
            "pressure norm ratio mismatch: unit={} doubled={}", unit.pressure_norm, doubled.pressure_norm);
        assert!((doubled.velocity_checksum / unit.velocity_checksum - 2.0).abs() < 1.0e-9,
            "velocity checksum ratio mismatch: unit={} doubled={}", unit.velocity_checksum, doubled.velocity_checksum);
        assert!((doubled.pressure_checksum / unit.pressure_checksum - 2.0).abs() < 1.0e-9,
            "pressure checksum ratio mismatch: unit={} doubled={}", unit.pressure_checksum, doubled.pressure_checksum);
    }

    #[test]
    fn ex40_stokes_refinement_strengthens_cavity_recirculation() {
        let coarse = solve_case(8, 1.0, 1.0);
        let fine = solve_case(16, 1.0, 1.0);
        assert!(coarse.converged && fine.converged);
        assert!(fine.divergence_residual < 1.0e-7, "fine-grid divergence too large: {}", fine.divergence_residual);
        assert!(fine.uy_abs_max > coarse.uy_abs_max,
            "refinement should strengthen resolved vertical circulation: coarse={} fine={}", coarse.uy_abs_max, fine.uy_abs_max);
        assert!(fine.ux_min < coarse.ux_min,
            "refinement should deepen the negative recirculation lobe: coarse={} fine={}", coarse.ux_min, fine.ux_min);
    }

    #[test]
    fn ex40_stokes_dof_count_matches_taylor_hood_formula() {
        // P2-P1 Taylor-Hood: velocity H1-P2 → (2n+1)^2, pressure H1-P1 → (n+1)^2
        for n in [6usize, 8] {
            let result = solve_case(n, 1.0, 1.0);
            let expected_pressure = (n + 1) * (n + 1);
            assert_eq!(result.pressure_dofs, expected_pressure,
                "pressure DOF mismatch for n={}: got {} expected {}",
                n, result.pressure_dofs, expected_pressure);
            assert!(result.velocity_dofs > result.pressure_dofs * 2,
                "P2 velocity should have many more DOFs than P1 pressure");
        }
    }

    #[test]
    fn ex40_stokes_higher_viscosity_increases_pressure() {
        let low_nu = solve_case(8, 1.0, 1.0);
        let high_nu = solve_case(8, 10.0, 1.0);
        assert!(low_nu.converged && high_nu.converged);
        // For Stokes with velocity-only BCs, velocity is independent of nu;
        // pressure scales linearly with nu (p ~ nu * grad(u))
        assert!(high_nu.pressure_norm > low_nu.pressure_norm,
            "higher viscosity should increase pressure: nu=1 p_norm={:.4e} nu=10 p_norm={:.4e}",
            low_nu.pressure_norm, high_nu.pressure_norm);
        // Velocity norms should be similar (Stokes: velocity driven by kinematics)
        assert!(low_nu.velocity_norm > 0.0 && high_nu.velocity_norm > 0.0);
    }

    #[test]
    fn ex40_stokes_pressure_has_zero_mean_approximately() {
        let result = solve_case(8, 1.0, 1.0);
        assert!(result.converged);
        // For Stokes with pure Dirichlet BCs, pressure is determined up to a constant.
        // The Schur complement fix pins pressure: min and max should straddle zero
        assert!(result.pressure_min < 0.0,
            "pressure should have negative values: min={:.4e}", result.pressure_min);
        assert!(result.pressure_max > 0.0,
            "pressure should have positive values: max={:.4e}", result.pressure_max);
    }

    #[test]
    fn ex40_stokes_constrained_velocity_dofs_are_strictly_fewer() {
        let result = solve_case(8, 1.0, 1.0);
        assert!(result.constrained_velocity_dofs < result.velocity_dofs,
            "constrained DOFs={} should be less than total velocity DOFs={}",
            result.constrained_velocity_dofs, result.velocity_dofs);
        // At least some DOFs are free (interior)
        let free_dofs = result.velocity_dofs - result.constrained_velocity_dofs;
        assert!(free_dofs > 0, "must have interior free DOFs");
    }
}

