//! IGA linear elasticity (2D plane strain beam).
//!
//! Cantilever beam discretised with NURBS, clamped left edge,
//! vertical load on right edge.  Solves K·u = f with PCG+GSSmoother.
//!
//! Usage:
//!   cargo run --example iga_linear_elasticity --release

use fem_assembly::iga::assemble_iga_elasticity_2d;
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh2D;
use fem_linalg::CsrMatrix;
use fem_solver::{solve_pcg_gssmoother, SolverConfig};

fn main() {
    let p = 2;
    let n = 9; // n×n control points
    let kv = NurbsKnotVector::uniform(p, n - p);
    let ctrl: Vec<[f64; 2]> = (0..n * n)
        .map(|idx| { let i = idx % n; let j = idx / n;
            [i as f64 * 4.0 / (n - 1) as f64, j as f64 / (n - 1) as f64]
        }).collect();
    let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n * n]);
    let n_dofs = 2 * n * n;

    let E = 1e5; let nu = 0.3;
    let lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu  = E / (2.0 * (1.0 + nu));

    let mut k = assemble_iga_elasticity_2d(&mesh, lam, mu, 4);
    let mut rhs = vec![0.0; n_dofs];

    // Dirichlet BC: left edge (i=0), fix x and y
    let mut bc_dofs = Vec::new();
    for j in 0..n {
        bc_dofs.push(2 * (j * n));         // u_x = 0
        bc_dofs.push(2 * (j * n) + 1);     // u_y = 0
    }
    // Load: downward force on right edge top half
    for j in n / 2..n {
        let dof = 2 * (j * n + (n - 1)) + 1;
        rhs[dof] -= 100.0;
    }

    println!("IGA beam: {}×{} NURBS(p={}), {} DOFs, {} BC DOFs",
             n, n, p, n_dofs, bc_dofs.len());
    // Apply BC: eliminate rows/cols for fixed DOFs
    for &dof in &bc_dofs {
        k.apply_dirichlet_symmetric(dof, 0.0, &mut rhs);
    }


    let cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    let mut u = vec![0.0; n_dofs];
    match solve_pcg_gssmoother(&k, &rhs, &mut u, &cfg) {
        Ok(res) => println!("PCG: {} iters, ||r||/||b|| = {:.3e}",
                           res.iterations, res.final_residual),
        Err(e) => eprintln!("Solver failed: {e}"),
    }

    let max_u: f64 = u.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let tip = 2 * ((n / 2) * n + (n - 1)) + 1;
    println!("Max |u| = {:.6e}, tip u_y = {:.6e}", max_u, u[tip]);
    println!("✅ IGA linear elasticity complete.");
}
