//! 2D IGA Neo-Hookean hyperelasticity.
//!
//! Solves a large-deformation hyperelasticity problem on a unit square
//! [0,1]² discretized with NURBS. Clamped on bottom (y=0), prescribed
//! displacement on top (y=1).

use fem_assembly::iga::{IgaHyperelasticity2D};
use fem_assembly::physics::nonlinear::{NewtonConfig, NewtonSolver};
use fem_assembly::physics::nonlinear_hyperelasticity::HyperelasticModel;
use fem_element::iga::NurbsKnotVector;
use fem_element::nurbs::NurbsMesh2D;

fn main() {
    let p = 2;
    let n = 8; // 8×8 control points
    let n_ctrl = n * n;
    let kv = NurbsKnotVector::uniform(p, n - p);
    let ctrl: Vec<[f64; 2]> = (0..n_ctrl)
        .map(|idx| {
            let i = idx % n;
            let j = idx / n;
            [i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]
        })
        .collect();
    let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl]);

    let mu = 10.0;
    let lam = 10.0;
    let model = HyperelasticModel::NeoHookean { mu, lambda: lam };

    let n_dofs = 2 * n_ctrl;

    // Dirichlet BC: bottom (y=0) clamped, top (y=1) prescribed uy = -0.05
    let mut dirichlet = Vec::new();
    for i in 0..n {
        let b = i;                       // bottom row
        let t = (n - 1) * n + i;         // top row
        dirichlet.push((2 * b, 0.0));
        dirichlet.push((2 * b + 1, 0.0));
        dirichlet.push((2 * t, 0.0));
        dirichlet.push((2 * t + 1, -0.05));
    }

    let form = IgaHyperelasticity2D::new(mesh, model, dirichlet, 4);
    let rhs = vec![0.0; n_dofs];
    let mut u = vec![0.0; n_dofs];

    let cfg = NewtonConfig {
        atol: 1e-6,
        rtol: 1e-6,
        max_iter: 100,
        linear_tol: 1e-8,
        line_search: false,
        ..NewtonConfig::default()
    };

    let result = NewtonSolver::new(cfg).solve(&form, &rhs, &mut u);

    match &result {
        Ok(r) => println!(
            "IGA hyperelasticity converged in {} iterations, final ||F|| = {:.3e}",
            r.iterations, r.final_residual
        ),
        Err(r) => println!(
            "IGA hyperelasticity FAILED after {} iterations, final ||F|| = {:.3e}",
            r.iterations, r.final_residual
        ),
    }

    let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("||u|| = {:.6e}", norm);
    assert!(result.is_ok(), "Newton did not converge");
    assert!(norm > 0.0 && norm < 100.0, "unexpected solution norm");
}
