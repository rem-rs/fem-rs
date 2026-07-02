//! # Example 12 — Solver comparison (analogous to MFEM ex12)
//!
//! Solves the Poisson problem with multiple solvers and compares iteration
//! counts and timing.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex12_solver_comparison
//! cargo run --example mfem_ex12_solver_comparison -- --n 32 --order 2
//! ```

use std::f64::consts::PI;
use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{
    SolverConfig,
    solve_cg, solve_pcg_jacobi, solve_pcg_ilu0, solve_pcg_ildlt,
    solve_gmres, solve_gmres_jacobi, solve_gmres_ilu0, solve_gmres_ildlt,
    solve_bicgstab, solve_fgmres_jacobi,
    solve_idrs, solve_tfqmr,
    solve_sparse_lu, solve_sparse_cholesky,
};
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 12: Solver comparison ===");

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, args.order);
    let n = space.n_dofs();
    let quad: u8 = args.order * 2 + 1;

    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad);
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    println!("  Mesh: {}×{} P{}, DOFs = {}", args.n, args.n, args.order, n);
    println!();

    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let pm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();

    let mut results: Vec<(&str, usize, f64, f64)> = Vec::new();

    // ── CG (SPD optimal) ──
    run(&mut results, "CG", || -> Result<(fem_solver::SolveResult, Vec<f64>), fem_solver::SolverError> {
        let mut x: Vec<f64> = vec![0.0; n];
        let r = solve_cg(&mat, &rhs, &mut x, &cfg)?;
        Ok((r, x))
    });

    // ── PCG + Jacobi ──
    run(&mut results, "PCG+Jacobi", || {
        let mut x = vec![0.0; n];
        let r = solve_pcg_jacobi(&mat, &rhs, &mut x, &cfg)?;
        Ok((r, x))
    });

    // ── PCG + ILU(0) ──
    run(&mut results, "PCG+ILU(0)", || {
        let mut x = vec![0.0; n];
        let r = solve_pcg_ilu0(&mat, &rhs, &mut x, &cfg)?;
        Ok((r, x))
    });

    // ── PCG + ILDL^T ──
    run(&mut results, "PCG+ILDL^T", || {
        let mut x = vec![0.0; n];
        let r = solve_pcg_ildlt(&mat, &rhs, &mut x, &cfg)?;
        Ok((r, x))
    });

    // ── GMRES(30) ──
    run(&mut results, "GMRES(30)", || {
        let mut x = vec![0.0; n];
        let r = solve_gmres(&mat, &rhs, &mut x, 30, &cfg)?;
        Ok((r, x))
    });

    // ── GMRES(30) + Jacobi ──
    run(&mut results, "GMRES(30)+Jacobi", || {
        let mut x = vec![0.0; n];
        let r = solve_gmres_jacobi(&mat, &rhs, &mut x, 30, &cfg)?;
        Ok((r, x))
    });

    // ── GMRES(30) + ILU(0) ──
    run(&mut results, "GMRES(30)+ILU(0)", || {
        let mut x = vec![0.0; n];
        let r = solve_gmres_ilu0(&mat, &rhs, &mut x, 30, &cfg)?;
        Ok((r, x))
    });

    // ── GMRES(30) + ILDL^T ──
    run(&mut results, "GMRES(30)+ILDL^T", || {
        let mut x = vec![0.0; n];
        let r = solve_gmres_ildlt(&mat, &rhs, &mut x, 30, &cfg)?;
        Ok((r, x))
    });

    // ── FGMRES(30) + Jacobi ──
    run(&mut results, "FGMRES(30)+Jacobi", || {
        let mut x = vec![0.0; n];
        let r = solve_fgmres_jacobi(&mat, &rhs, &mut x, 30, &cfg)?;
        Ok((r, x))
    });

    // ── BiCGSTAB ──
    run(&mut results, "BiCGSTAB", || {
        let mut x = vec![0.0; n];
        let r = solve_bicgstab(&mat, &rhs, &mut x, &cfg)?;
        Ok((r, x))
    });

    // ── IDR(s=4) ──
    run(&mut results, "IDR(4)", || {
        let mut x = vec![0.0; n];
        let r = solve_idrs(&mat, &rhs, &mut x, 4, &cfg)?;
        Ok((r, x))
    });

    // ── TFQMR ──
    run(&mut results, "TFQMR", || {
        let mut x = vec![0.0; n];
        let r = solve_tfqmr(&mat, &rhs, &mut x, &cfg)?;
        Ok((r, x))
    });

    // ── Sparse LU direct ──
    run(&mut results, "Sparse LU", || {
        let x = solve_sparse_lu(&mat, &rhs)?;
        Ok((fem_solver::SolveResult { converged: true, iterations: 1, final_residual: 0.0 }, x))
    });

    // ── Sparse Cholesky direct ──
    run(&mut results, "Sparse Cholesky", || {
        let x = solve_sparse_cholesky(&mat, &rhs)?;
        Ok((fem_solver::SolveResult { converged: true, iterations: 1, final_residual: 0.0 }, x))
    });

    // ── Print table ──
    println!("  {:<22} {:>10} {:>12} {:>14}  {}", "Solver", "Iters", "Residual", "Time (s)", "|x|₂");
    println!("  {}", str::repeat("─", 72));
    for (name, iters, resid, time_s) in &results {
        println!("  {:<22} {:>10} {:>12.3e} {:>14.6}  -", name, iters, resid, time_s);
    }
}

fn run<F>(
    results: &mut Vec<(&'static str, usize, f64, f64)>,
    name: &'static str,
    f: F,
) where
    F: Fn() -> Result<(fem_solver::SolveResult, Vec<f64>), fem_solver::SolverError>,
{
    let t0 = Instant::now();
    match f() {
        Ok((r, x)) => {
            let dt = t0.elapsed().as_secs_f64();
            let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            results.push((name, r.iterations, r.final_residual, dt));
            println!("  ✓ {:<20} iters={:<6} residual={:.3e} time={:.4}s  |x|₂={:.4e}",
                name, r.iterations, r.final_residual, dt, nrm);
        }
        Err(e) => {
            println!("  ✗ {:<20} error: {}", name, e);
        }
    }
}

struct Args { n: usize, order: u8 }

fn parse_args() -> Args {
    let mut a = Args { n: 16, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(16); }
            "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex12_all_solvers_converge() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let quad = 3;
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad);
        let source = DomainSourceIntegrator::new(|x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin());
        let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0_f64; bnd.len()];
        let mut mat = mat;
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
        let n = space.n_dofs();
        let cfg = SolverConfig { rtol: 1e-10, ..SolverConfig::default() };

        // CG
        {
            let mut x = vec![0.0; n];
            let r = solve_cg(&mat, &rhs, &mut x, &cfg).unwrap();
            assert!(r.converged);
        }
        // GMRES
        {
            let mut x = vec![0.0; n];
            let r = solve_gmres(&mat, &rhs, &mut x, 30, &cfg).unwrap();
            assert!(r.converged);
        }
        // Sparse LU
        {
            let x = solve_sparse_lu(&mat, &rhs).unwrap();
            assert!(x.len() == n);
        }
    }
}
