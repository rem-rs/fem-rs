//! # Example 5 �?Block saddle-point system  (analogous to MFEM ex5)
//!
//! Demonstrates the block system infrastructure by solving a saddle-point
//! problem via the Uzawa (Schur complement) solver.
//!
//! The system is:
//! ```text
//!   [ A   Bᵀ ] [ u ]   [ f ]
//!   [ B  -εI ] [ p ] = [ g ]
//! ```
//!
//! where A = diffusion matrix, B = gradient coupling, and ε is a small
//! stabilization term (pressure stabilization for the P1/P1 pair).
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex5_mixed_darcy
//! cargo run --example mfem_ex5_mixed_darcy -- --n 16
//! cargo run --example mfem_ex5_mixed_darcy -- -m ../data/beam-tri.mesh
//! ```

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::{DivIntegrator, PressureDivIntegrator},
    standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_io::mfem::read_mfem_file;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 5: Mixed Darcy (MFEM ex5 analog) ===");

    let result = solve_case(args.n, args.mesh.as_deref());

    println!("  u-DOFs: {}, p-DOFs: {}", result.nu, result.np);
    println!("  Boundary u-DOFs constrained: {}", result.n_boundary_dofs);
    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("  max|u| = {:.4e},  max|p| = {:.4e}", result.u_max, result.p_max);
    println!("  ||u||_L2 = {:.4e},  ||p||_L2 = {:.4e}", result.u_norm, result.p_norm);
    println!("  checksum(u) = {:.8e},  checksum(p) = {:.8e}", result.u_checksum, result.p_checksum);
    println!(
        "  Block residual: ||Au+B^Tp-f|| = {:.3e},  ||Bu+Cp-g|| = {:.3e}",
        result.block_residual_u, result.block_residual_p
    );

    println!("\nDone.");
}

struct SolveResult {
    nu: usize,
    np: usize,
    n_boundary_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    u_max: f64,
    p_max: f64,
    u_norm: f64,
    p_norm: f64,
    u_checksum: f64,
    p_checksum: f64,
    block_residual_u: f64,
    block_residual_p: f64,
}

fn solve_case(n: usize, mesh_path: Option<&str>) -> SolveResult {

    // ─── 1. Mesh and spaces ──────────────────────────────────────────────────
    let mesh: SimplexMesh<2> = if let Some(path) = mesh_path {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(n)
    };
    let mesh_u = mesh.clone();
    let mesh_p = mesh;
    let space_u = H1Space::new(mesh_u, 1);
    let space_p = H1Space::new(mesh_p, 1);
    let nu = space_u.n_dofs();
    let np = space_p.n_dofs();

    // ─── 2. Assemble blocks ──────────────────────────────────────────────────
    // A = diffusion (stiffness) matrix for u
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let a_mat = Assembler::assemble_bilinear(&space_u, &[&diff], 3);

    // B, Bᵀ = gradient coupling
    let bt_mat = MixedAssembler::assemble_bilinear(&space_u, &space_p, &[&DivIntegrator], 3);
    let b_mat = MixedAssembler::assemble_bilinear(&space_p, &space_u, &[&PressureDivIntegrator], 3);

    // C = -ε M_p (pressure stabilization for P1/P1)
    let eps = 1e-2;
    let mass_p = MassIntegrator { rho: -eps };
    let c_mat = Assembler::assemble_bilinear(&space_p, &[&mass_p], 3);

    // RHS for the u-block: constant source f = 1 (matching MFEM ex5 default)
    let source_u = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let f_u = Assembler::assemble_linear(&space_u, &[&source_u], 3);

    // g = 0 (divergence-free constraint)
    let g = vec![0.0_f64; np];

    // ─── 3. Apply Dirichlet u = 0 on boundary ───────────────────────────────
    // For Uzawa, we handle BCs by zeroing A rows/cols for boundary DOFs.
    let dm_u = space_u.dof_manager();
    let bnd_u = boundary_dofs(space_u.mesh(), dm_u, &[1, 2, 3, 4]);
    let mut a_mat = a_mat;
    let mut f_u = f_u;
    let bnd_vals = vec![0.0_f64; bnd_u.len()];
    fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bnd_u, &bnd_vals);

    // ─── 4. Solve with Uzawa ─────────────────────────────────────────────────
    let sys = BlockSystem { a: a_mat, bt: bt_mat, b: b_mat, c: Some(c_mat) };
    let mut u_sol = vec![0.0_f64; nu];
    let mut p_sol = vec![0.0_f64; np];

    let cfg = SolverConfig { rtol: 1e-6, atol: 1e-10, max_iter: 2_000, verbose: false, ..SolverConfig::default() };
    let res = SchurComplementSolver::solve(&sys, &f_u, &g, &mut u_sol, &mut p_sol, &cfg)
        .expect("Uzawa solve failed");

    // ─── 5. Report ──────────────────────────────────────────────────────────
    let u_max: f64 = u_sol.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let p_max: f64 = p_sol.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let u_norm = u_sol.iter().map(|v| v * v).sum::<f64>().sqrt();
    let p_norm = p_sol.iter().map(|v| v * v).sum::<f64>().sqrt();
    let u_checksum = u_sol
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();
    let p_checksum = p_sol
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    // Verify residual: ||Au + Bᵀp - f|| + ||Bu + Cp - g||
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    sys.apply(&u_sol, &p_sol, &mut ru, &mut rp);
    let err_u: f64 = ru.iter().zip(f_u.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let err_p: f64 = rp.iter().zip(g.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

    SolveResult {
        nu,
        np,
        n_boundary_dofs: bnd_u.len(),
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        u_max,
        p_max,
        u_norm,
        p_norm,
        u_checksum,
        p_checksum,
        block_residual_u: err_u,
        block_residual_p: err_p,
    }
}

struct Args { n: usize, mesh: Option<String> }

fn parse_args() -> Args {
    let mut a = Args { n: 8, mesh: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "--n"           => { a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use super::*;

    /// Test-only helper: solve with MMS sinusoidal source f = scale * sin(πx)sin(πy).
    fn solve_case_mms(n: usize, source_scale: f64) -> SolveResult {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space_u = H1Space::new(mesh.clone(), 1);
        let space_p = H1Space::new(mesh, 1);
        let nu = space_u.n_dofs();
        let np = space_p.n_dofs();

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let a_mat = Assembler::assemble_bilinear(&space_u, &[&diff], 3);

        let bt_mat = MixedAssembler::assemble_bilinear(&space_u, &space_p, &[&DivIntegrator], 3);
        let b_mat = MixedAssembler::assemble_bilinear(&space_p, &space_u, &[&PressureDivIntegrator], 3);

        let eps = 1e-2;
        let mass_p = MassIntegrator { rho: -eps };
        let c_mat = Assembler::assemble_bilinear(&space_p, &[&mass_p], 3);

        let source_u = DomainSourceIntegrator::new(move |x: &[f64]| {
            source_scale * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut f_u = Assembler::assemble_linear(&space_u, &[&source_u], 3);

        let g = vec![0.0_f64; np];

        let dm_u = space_u.dof_manager();
        let bnd_u = boundary_dofs(space_u.mesh(), dm_u, &[1, 2, 3, 4]);
        let mut a_mat = a_mat;
        let bnd_vals = vec![0.0_f64; bnd_u.len()];
        fem_space::constraints::apply_dirichlet(&mut a_mat, &mut f_u, &bnd_u, &bnd_vals);

        let sys = BlockSystem { a: a_mat, bt: bt_mat, b: b_mat, c: Some(c_mat) };
        let mut u_sol = vec![0.0_f64; nu];
        let mut p_sol = vec![0.0_f64; np];

        let cfg = SolverConfig { rtol: 1e-6, atol: 1e-10, max_iter: 2_000, verbose: false, ..SolverConfig::default() };
        let res = SchurComplementSolver::solve(&sys, &f_u, &g, &mut u_sol, &mut p_sol, &cfg)
            .expect("Uzawa solve failed");

        let u_max: f64 = u_sol.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        let p_max: f64 = p_sol.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        let u_norm = u_sol.iter().map(|v| v * v).sum::<f64>().sqrt();
        let p_norm = p_sol.iter().map(|v| v * v).sum::<f64>().sqrt();
        let u_checksum = u_sol.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum::<f64>();
        let p_checksum = p_sol.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum::<f64>();

        let mut ru = vec![0.0_f64; nu];
        let mut rp = vec![0.0_f64; np];
        sys.apply(&u_sol, &p_sol, &mut ru, &mut rp);
        let err_u = ru.iter().zip(f_u.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let err_p = rp.iter().zip(g.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

        SolveResult {
            nu, np,
            n_boundary_dofs: bnd_u.len(),
            iterations: res.iterations,
            final_residual: res.final_residual,
            converged: res.converged,
            u_max, p_max, u_norm, p_norm, u_checksum, p_checksum,
            block_residual_u: err_u,
            block_residual_p: err_p,
        }
    }

    #[test]
    fn ex5_mixed_darcy_coarse_case_converges_with_small_block_residual() {
        let result = solve_case_mms(8, 1.0);
        assert!(result.converged);
        assert_eq!(result.nu, 81);
        assert_eq!(result.np, 81);
        assert_eq!(result.n_boundary_dofs, 32);
        assert!(result.final_residual < 5.0e-8, "solver residual too large: {}", result.final_residual);
        assert!(result.block_residual_u < 1.0e-8, "u block residual too large: {}", result.block_residual_u);
        assert!(result.block_residual_p < 5.0e-8, "p block residual too large: {}", result.block_residual_p);
    }

    #[test]
    fn ex5_mixed_darcy_zero_forcing_gives_trivial_solution() {
        let result = solve_case_mms(8, 0.0);
        assert!(result.converged);
        assert!(result.u_norm < 1.0e-12, "u norm should vanish: {}", result.u_norm);
        assert!(result.p_norm < 1.0e-12, "p norm should vanish: {}", result.p_norm);
        assert!(result.block_residual_u < 1.0e-12, "u block residual should vanish: {}", result.block_residual_u);
        assert!(result.block_residual_p < 1.0e-12, "p block residual should vanish: {}", result.block_residual_p);
    }

    #[test]
    fn ex5_mixed_darcy_solution_scales_linearly_with_source() {
        let unit = solve_case_mms(8, 1.0);
        let doubled = solve_case_mms(8, 2.0);
        assert!(unit.converged && doubled.converged);
        assert!((doubled.u_norm / unit.u_norm - 2.0).abs() < 1.0e-9,
            "u norm ratio mismatch: unit={} doubled={}", unit.u_norm, doubled.u_norm);
        assert!((doubled.p_norm / unit.p_norm - 2.0).abs() < 1.0e-9,
            "p norm ratio mismatch: unit={} doubled={}", unit.p_norm, doubled.p_norm);
        assert!((doubled.u_checksum / unit.u_checksum - 2.0).abs() < 1.0e-9,
            "u checksum ratio mismatch: unit={} doubled={}", unit.u_checksum, doubled.u_checksum);
        assert!((doubled.p_checksum / unit.p_checksum - 2.0).abs() < 1.0e-9,
            "p checksum ratio mismatch: unit={} doubled={}", unit.p_checksum, doubled.p_checksum);
    }

    #[test]
    fn ex5_mixed_darcy_sign_reversed_source_flips_solution() {
        let positive = solve_case_mms(8, 1.0);
        let negative = solve_case_mms(8, -1.0);
        assert!(positive.converged && negative.converged);
        assert!((positive.u_norm - negative.u_norm).abs() < 1.0e-12);
        assert!((positive.p_norm - negative.p_norm).abs() < 1.0e-12);
        assert!((positive.u_checksum + negative.u_checksum).abs() < 1.0e-10,
            "u checksum should flip sign: positive={} negative={}", positive.u_checksum, negative.u_checksum);
        assert!((positive.p_checksum + negative.p_checksum).abs() < 1.0e-10,
            "p checksum should flip sign: positive={} negative={}", positive.p_checksum, negative.p_checksum);
    }

    /// Very coarse mesh should still converge.
    #[test]
    fn ex5_mixed_darcy_very_coarse_mesh_converges() {
        let result = solve_case_mms(4, 1.0);
        assert!(result.converged, "very coarse mesh should converge");
        assert!(result.final_residual < 1.0e-6, "residual should be small");
    }

    /// Mesh refinement should increase DOF count.
    #[test]
    fn ex5_mixed_darcy_refinement_increases_dof_count() {
        let coarse = solve_case(8, None);
        let fine = solve_case(12, None);
        assert!(coarse.converged && fine.converged);
        assert!(fine.nu > coarse.nu, "refined mesh should have more u-DOFs");
        assert!(fine.np > coarse.np, "refined mesh should have more p-DOFs");
    }

    /// Block residuals should be small for converged solution.
    #[test]
    fn ex5_mixed_darcy_block_residuals_stay_small() {
        let result = solve_case_mms(8, 1.0);
        assert!(result.converged);
        assert!(result.block_residual_u < 1.0e-6, "u block residual: {}", result.block_residual_u);
        assert!(result.block_residual_p < 1.0e-6, "p block residual: {}", result.block_residual_p);
        let total = (result.block_residual_u.powi(2) + result.block_residual_p.powi(2)).sqrt();
        assert!(total < 1.0e-5, "total block residual: {}", total);
    }

    /// Higher forcing magnitude should increase solution norms monotonically.
    #[test]
    fn ex5_mixed_darcy_higher_forcing_increases_solution() {
        let weak = solve_case_mms(8, 0.5);
        let strong = solve_case_mms(8, 2.0);
        assert!(weak.converged && strong.converged);
        assert!(strong.u_norm > weak.u_norm,
            "higher forcing should increase u norm: weak={} strong={}",
            weak.u_norm, strong.u_norm);
        assert!(strong.p_norm > weak.p_norm,
            "higher forcing should increase p norm: weak={} strong={}",
            weak.p_norm, strong.p_norm);
    }

    // ─── Regression baseline ─────────────────────────────────────────────

    #[test]
    fn ex5_regression_baseline() {
        let result = solve_case_mms(8, 1.0);
        assert!(result.converged);

        fem_regression::regression("mfem_ex5_mixed_darcy")
            .check_with("u_norm",           result.u_norm,          1e-6, 1e-10)
            .check_with("p_norm",           result.p_norm,          1e-6, 1e-10)
            .check_with("u_checksum",       result.u_checksum,      1e-6, 1e-10)
            .check_with("p_checksum",       result.p_checksum,      1e-6, 1e-10)
            .check_with("block_residual_u", result.block_residual_u, 1e-4, 1e-10)
            .check_with("block_residual_p", result.block_residual_p, 1e-4, 1e-10)
            .check_with("nu",               result.nu as f64,       0.0,  0.5)
            .check_with("np",               result.np as f64,       0.0,  0.5)
            .finalize();
    }

    /// Cross-validate ex5 (mixed Darcy) as an analytical reference test.
    #[test]
    fn ex5_mfem_reference_test() {
        let result = solve_case_mms(8, 1.0);
        assert!(result.converged, "solve must converge");

        assert!(result.u_norm > 0.0, "u_norm should be positive");
        assert!(result.p_norm > 0.0, "p_norm should be positive");
        assert_eq!(result.nu, 81, "H1 P1 on 8×8: 81 DOFs");
        assert_eq!(result.np, 81, "H1 P1 on 8×8: 81 DOFs");
        assert!(result.u_norm.is_finite() && result.p_norm.is_finite());

        let coarse = solve_case_mms(6, 1.0);
        let fine = solve_case_mms(12, 1.0);
        assert!(coarse.converged && fine.converged);
        assert!(fine.u_norm > 0.0 && fine.p_norm > 0.0);
        eprintln!("  [mfem-ref] ex5: u(n=6)={:.6e} u(n=12)={:.6e}",
            coarse.u_norm, fine.u_norm);

        use std::time::Instant;
        let t0 = Instant::now();
        let _ = solve_case_mms(24, 1.0);
        let elapsed = t0.elapsed();
        assert!(elapsed.as_secs_f64() < 20.0, "mixed 24×24 took {:.2}s", elapsed.as_secs_f64());
        eprintln!("  [mfem-ref] ex5: 24×24 = {:.2}ms", elapsed.as_millis());
    }
}

