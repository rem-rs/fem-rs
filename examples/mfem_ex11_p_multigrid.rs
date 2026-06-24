//! # Example 11 — p-multigrid (analogous to MFEM ex11)
//!
//! Solves the 1-D Poisson equation `-u'' = f` on (0,1) with homogeneous
//! Dirichlet BCs using p-multigrid as a preconditioner for CG, demonstrating
//! optimal (h-independent) convergence.

use std::f64::consts::PI;

use fem_linalg::CooMatrix;
use fem_solver::{
    SolverConfig, solve_cg, solve_pcg_jacobi,
    p_multigrid::{
        PmgPrecond, build_pmg_hierarchy_1d_laplacian,
        solve_vcycle_pmg,
    },
};
use fem_element::ReferenceElement;

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 11: p-multigrid ===");

    println!("  n_elem={}, p_fine={}", args.n, args.pmax);

    // Build the p-multigrid hierarchy
    let hierarchy = build_pmg_hierarchy_1d_laplacian(args.n, args.pmax);
    let a_fine = &hierarchy.levels[0];
    let n = a_fine.nrows;

    // RHS: ∫f φ_i  for f(x) = π² sin(π x)
    let rhs = assemble_rhs_1d(args.n, args.pmax);

    // ── CG (no preconditioner) ──
    let mut u_cg = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res_cg = solve_cg(a_fine, &rhs, &mut u_cg, &cfg).unwrap_or_else(|e| {
        panic!("CG failed: {}", e)
    });

    // ── PCG + Jacobi ──
    let mut u_pc = vec![0.0; n];
    let res_pc = solve_pcg_jacobi(a_fine, &rhs, &mut u_pc, &cfg).unwrap_or_else(|e| {
        panic!("PCG+Jacobi failed: {}", e)
    });

    // ── p-multigrid V-cycle ──
    let pmg_precond = PmgPrecond { pre_sweeps: 2, post_sweeps: 2, jacobi_omega: 0.8, coarse_max_iter: 200 };
    let mut u_pmg = vec![0.0; n];
    let res_pmg = solve_vcycle_pmg(a_fine, &rhs, &mut u_pmg, &hierarchy, &pmg_precond, &cfg)
        .unwrap_or_else(|e| panic!("p-MG failed: {}", e));

    // L² error against u = sin(πx)
    let h = 1.0 / args.n as f64;
    let err_cg  = l2_error_1d(&u_cg, h);
    let err_pc  = l2_error_1d(&u_pc, h);
    let err_pmg = l2_error_1d(&u_pmg, h);

    println!();
    println!("  {:>20} {:>12} {:>14} {:>14}", "Solver", "Iterations", "Residual", "L² error");
    println!("  {}", str::repeat("─", 64));
    println!("  {:>20} {:>12} {:>14.3e} {:>14.3e}", "CG (no prec)", res_cg.iterations, res_cg.final_residual, err_cg);
    println!("  {:>20} {:>12} {:>14.3e} {:>14.3e}", "PCG + Jacobi", res_pc.iterations, res_pc.final_residual, err_pc);
    println!("  {:>20} {:>12} {:>14.3e} {:>14.3e}", "p-MG V-cycle", res_pmg.iterations, res_pmg.final_residual, err_pmg);
}

/// Assemble RHS vector ∫ f φ_i for f(x) = π² sin(π x) on P_pmax elements.
fn assemble_rhs_1d(n_elem: usize, p_max: u8) -> Vec<f64> {
    use fem_element::lagrange::SegPk;
    let p = p_max as usize;
    let re = SegPk::new(p);
    let quad = re.quadrature((p as u8 + 2) * 2);
    let h = 1.0 / n_elem as f64;
    let n_dofs_per_elem = p; // hierarchy uses p DOFs per element (not p+1) for order p
    let n_total = n_elem * n_dofs_per_elem + 1;

    let mut coo = CooMatrix::new(n_total, 1);
    let mut phi = vec![0.0; re.n_dofs()];

    for e in 0..n_elem {
        let x0 = e as f64 * h;
        let x1 = (e + 1) as f64 * h;
        let jac = h / 2.0;

        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * jac;
            let xp = 0.5 * (x0 + x1) + 0.5 * h * xi[0];
            let f = PI * PI * (PI * xp).sin();
            for i in 0..=p {
                let row = (e * p + i).min(n_total - 1);
                coo.add(row, 0, w * f * phi[i]);
            }
        }
    }
    let rhs_csr = coo.into_csr();
    let mut rhs = vec![0.0; n_total];
    for i in 0..n_total {
        rhs[i] = rhs_csr.get(i, 0);
    }
    // Enforce homogeneous Dirichlet at x=0 and x=1
    rhs[0] = 0.0;
    rhs[n_total - 1] = 0.0;
    rhs
}

/// L² error against u(x) = sin(πx)
fn l2_error_1d(uh: &[f64], h: f64) -> f64 {
    let n = uh.len() - 1;
    let mut err2 = 0.0;
    for i in 0..n {
        let ue = (PI * i as f64 * h).sin();
        let diff = uh[i] - ue;
        err2 += diff * diff * h;
    }
    err2.sqrt()
}

struct Args { n: usize, pmax: u8 }

fn parse_args() -> Args {
    let mut a = Args { n: 32, pmax: 3 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(32); }
            "--pmax" => { a.pmax = it.next().and_then(|v| v.parse().ok()).unwrap_or(3); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex11_cg_solves_poisson() {
        let n = 32;
        let pmax = 3;
        let h = build_pmg_hierarchy_1d_laplacian(n, pmax);
        let a_fine = &h.levels[0];
        let rhs = assemble_rhs_1d(n, pmax);
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };

        let mut u_cg = vec![0.0; a_fine.nrows];
        let r_cg = solve_cg(a_fine, &rhs, &mut u_cg, &cfg).unwrap();
        assert!(r_cg.converged);
    }
}
