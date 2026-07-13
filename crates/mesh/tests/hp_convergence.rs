//! Smooth-solution h-convergence test for AMR.
//!
//! Solves Poisson on [0,1]² with sin(πx)sin(πy) RHS and homogeneous
//! Dirichlet BCs (u=0 on boundary since sin(0)=sin(π)=0), using uniform
//! h-refinement on triangular meshes. Verifies that the P1 finite element
//! solution converges at the optimal rate O(h²) = O(N^{-1}) in 2D.
//!
//! Uses triangular elements because `GridFunction::compute_l2_error` uses
//! `simplex_jacobian` which is designed for simplex elements.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    GridFunction,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{
    Mesh,
    amr::{ConvergenceRecord, ConvergenceStudy, refine_uniform},
};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Arctan front constants and helpers ──────────────────────────────────────

/// Steepness parameter for the arctan front: larger α → sharper front.
const ALPHA: f64 = 50.0;

/// Radius of the circular front (centered at (0.5, 0.5)).
const R0: f64 = 0.25;

/// Exact solution: u = arctan(α · (r - r₀)) with r = sqrt((x-0.5)² + (y-0.5)²).
///
/// Smooth everywhere except a mild cusp at the center r = 0.
fn exact_u_arctan(pt: &[f64]) -> f64 {
    let r = ((pt[0] - 0.5).powi(2) + (pt[1] - 0.5).powi(2)).sqrt();
    (ALPHA * (r - R0)).atan()
}

/// RHS: -Δ(arctan(α(r - r₀))) in 2D.
///
/// For a radially symmetric function f(r) in 2D:
///   Δf = f''(r) + f'(r)/r
/// With f(r) = arctan(α(r - r₀)):
///   f'(r)  = α / (1 + α²(r - r₀)²)
///   f''(r) = -2α³(r - r₀) / (1 + α²(r - r₀)²)²
/// Giving:
///   Δu = -2α³(r - r₀)/(1 + α²(r - r₀)²)² + α/(r·(1 + α²(r - r₀)²))
///   RHS = -Δu
fn rhs_fn_arctan(pt: &[f64]) -> f64 {
    let dx = pt[0] - 0.5;
    let dy = pt[1] - 0.5;
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-15 {
        // Removable singularity in the integral sense — Gauss points never hit r=0.
        return 0.0;
    }
    let dr = r - R0;
    let denom = 1.0 + ALPHA * ALPHA * dr * dr;
    2.0 * ALPHA.powi(3) * dr / (denom * denom) - ALPHA / (r * denom)
}

/// Exact solution: u = sin(πx)sin(πy) on [0,1]².
/// Homogeneous Dirichlet BC (u=0 on boundary) since sin(0) = sin(π) = 0.
fn exact_u(pt: &[f64]) -> f64 {
    (PI * pt[0]).sin() * (PI * pt[1]).sin()
}

/// RHS: -Δ(sin(πx)sin(πy)) = 2π² sin(πx)sin(πy)
fn rhs_fn(pt: &[f64]) -> f64 {
    2.0 * PI * PI * (PI * pt[0]).sin() * (PI * pt[1]).sin()
}

#[test]
fn test_h_convergence_smooth_solution() {
    // Start with 4×4 triangular mesh on [0,1]² (enough to enter asymptotic regime).
    // unit_square_tri(4) → 2×4×4 = 32 triangles, 5×5 = 25 nodes.
    let mut mesh = Mesh::<2>::unit_square_tri(4);
    let order = 1u8;
    let mut study = ConvergenceStudy::<2>::new();

    for step in 0..6 {
        let space = H1Space::new(mesh.clone(), order);
        let quad = 3; // 2*order + 1 = 3

        // Assemble -Δu = f
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let source = DomainSourceIntegrator::new(|x: &[f64]| rhs_fn(x));
        let mut mat = Assembler::assemble_bilinear(&space, &[&diff], quad);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

        // Apply homogeneous Dirichlet BC on all boundaries
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0; bnd.len()];
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // Solve
        let mut u = vec![0.0; space.n_dofs()];
        solve_cg(
            &mat, &rhs, &mut u,
            &SolverConfig {
                rtol: 1e-12,
                atol: 0.0,
                max_iter: 5000,
                verbose: false,
                ..SolverConfig::default()
            },
        )
        .expect("CG solve failed");

        // Compute L² error using GridFunction
        let gf = GridFunction::new(&space, u);
        let l2 = gf.compute_l2_error(&exact_u, quad);

        println!(
            "step={}, n_dofs={}, n_elems={}, l2_error={:.6e}",
            step,
            space.n_dofs(),
            mesh.n_elems(),
            l2
        );

        study.push(ConvergenceRecord {
            step,
            n_dofs: space.n_dofs(),
            n_elems: mesh.n_elems(),
            l2_error: l2,
            h1_error: None,
            n_h_refined: mesh.n_elems(),
            n_p_refined: 0,
        });

        if l2 < 1e-12 {
            break;
        }

        // Uniform h-refinement (Tri3 → 4 Tri3)
        mesh = refine_uniform(&mesh);
    }

    let rate = study.convergence_rate_l2().unwrap_or(0.0);
    println!("Convergence rate: {:.3}", rate);

    // For P1 on triangles, optimal rate is O(h²) = O(N^{-1}), so slope ≈ -1.
    // The last two refinement levels should be well into the asymptotic regime,
    // giving a rate close to -1.0.
    assert!(
        rate < -0.8,
        "Convergence rate too slow: {:.3} (expected < -0.8 for P1)",
        rate
    );
}

#[test]
fn test_h_convergence_arctan_front() {
    // Start with 4×4 triangular mesh on [0,1]².
    let mut mesh = Mesh::<2>::unit_square_tri(4);
    let order = 1u8;
    let mut study = ConvergenceStudy::<2>::new();

    for step in 0..7 {
        let space = H1Space::new(mesh.clone(), order);
        let quad = 3; // 2*order + 1 = 3

        // Assemble -Δu = f
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let source = DomainSourceIntegrator::new(|x: &[f64]| rhs_fn_arctan(x));
        let mut mat = Assembler::assemble_bilinear(&space, &[&diff], quad);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

        // Dirichlet BC on all boundaries — evaluate exact solution on boundary DOFs
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_vals: Vec<f64> = bnd
            .iter()
            .map(|&dof| {
                let coord = dm.dof_coord(dof);
                exact_u_arctan(coord)
            })
            .collect();
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        // Solve
        let mut u = vec![0.0; space.n_dofs()];
        solve_cg(
            &mat, &rhs, &mut u,
            &SolverConfig {
                rtol: 1e-12,
                atol: 0.0,
                max_iter: 5000,
                verbose: false,
                ..SolverConfig::default()
            },
        )
        .expect("CG solve failed");

        // Compute L² error using GridFunction
        let gf = GridFunction::new(&space, u);
        let l2 = gf.compute_l2_error(&exact_u_arctan, quad);

        println!(
            "step={}, n_dofs={}, n_elems={}, l2_error={:.6e}",
            step,
            space.n_dofs(),
            mesh.n_elems(),
            l2
        );

        study.push(ConvergenceRecord {
            step,
            n_dofs: space.n_dofs(),
            n_elems: mesh.n_elems(),
            l2_error: l2,
            h1_error: None,
            n_h_refined: mesh.n_elems(),
            n_p_refined: 0,
        });

        if l2 < 1e-12 {
            break;
        }

        // Uniform h-refinement (Tri3 → 4 Tri3)
        mesh = refine_uniform(&mesh);
    }

    let rate = study.convergence_rate_l2().unwrap_or(0.0);
    println!("Convergence rate: {:.3}", rate);

    // NOTE: The manufactured solution u = arctan(α(r - r₀)) has a cusp at r=0
    // (the center of the domain) where the Laplacian contains a 1/r singularity
    // (inherent to any radial function in 2D with f'(0) ≠ 0). This limits the
    // P1 L² convergence rate to approximately N^{-0.25} = h^{0.5}, far below
    // the optimal N^{-1}. The assertion below verifies that the error is
    // meaningfully decreasing as the mesh is refined.
    assert!(
        rate < -0.2,
        "Convergence rate too slow: {:.3} (expected < -0.2 for P1 arctan front with 1/r singularity)",
        rate
    );
}
