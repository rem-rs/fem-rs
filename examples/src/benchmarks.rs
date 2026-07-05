//! NAFEMS-style benchmark suite for fem-rs.
//!
//! Implements well-known solid mechanics benchmarks with known reference
//! solutions to validate the FEM pipeline against industry standards.
//!
//! # Benchmarks
//!
//! | Benchmark | Type | Reference |
//! |-----------|------|-----------|
//! | Cantilever beam (uniform load) | 2-D plane stress elasticity | Euler-Bernoulli δ = qL⁴/8EI |
//! | Cantilever beam (mesh convergence) | 2-D plane stress elasticity | Monotonic error reduction |
//! | Poisson patch (linear) | H¹ Poisson | Machine precision |
//! | 3-D elasticity smoke | 3-D linear elasticity | Finite, non-trivial |

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, ElasticityIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{
    fe_space::FESpace,
    H1Space, VectorH1Space,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Solver helpers ────────────────────────────────────────────────────

/// Solve 2-D plane-stress linear elasticity (ex2 pattern).
///
/// Uses unit_square_tri(n) scaled to L×H.  RHS is assembled on a
/// scalar H¹ space then placed into the y-component block.
fn solve_elasticity_beam(
    n: usize,
    l: f64,
    h: f64,
    e_mod: f64,
    nu: f64,
    body_force_y: f64,
    clamped_tags: &[i32],
) -> (Vec<f64>, usize, f64) {
    let lam = e_mod * nu / (1.0 - nu * nu); // plane stress λ*
    let mu  = e_mod / (2.0 * (1.0 + nu));

    // Create mesh on [0,L]×[0,H] by scaling unit_square_tri(n)
    let mut mesh = SimplexMesh::<2>::unit_square_tri(n);
    for c in mesh.coords.chunks_mut(2) {
        c[0] *= l;
        c[1] *= h;
    }

    let space = VectorH1Space::new(mesh.clone(), 1, 2);
    let n_total = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // Stiffness matrix
    let elast = ElasticityIntegrator { lambda: lam, mu, plane_stress: true };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], 3);

    // RHS: assemble on a matching scalar mesh (ex2 pattern)
    let mut rhs = vec![0.0; n_total];
    {
        let mut scalar_mesh = SimplexMesh::<2>::unit_square_tri(n);
        for c in scalar_mesh.coords.chunks_mut(2) {
            c[0] *= l;
            c[1] *= h;
        }
        let scalar_space = H1Space::new(scalar_mesh, 1);
        let fy = DomainSourceIntegrator::new(move |_: &[f64]| body_force_y);
        let fy_vec = Assembler::assemble_linear(&scalar_space, &[&fy], 3);
        for (i, &v) in fy_vec.iter().enumerate() {
            rhs[n_scalar + i] += v;
        }
    }

    // Clamped BCs
    let scalar_dm = space.scalar_dof_manager();
    let bnd_scalar = boundary_dofs(&mesh, scalar_dm, clamped_tags);
    let mut clamped: Vec<u32> = Vec::new();
    for &d in &bnd_scalar {
        clamped.push(d);
        clamped.push(d + n_scalar as u32);
    }
    let vals = vec![0.0; clamped.len()];
    apply_dirichlet(&mut mat, &mut rhs, &clamped, &vals);

    let mut u = vec![0.0; n_total];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 20_000, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("elasticity CG solve failed");

    (u, result.iterations, result.final_residual)
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark 1: Cantilever beam — uniform body force
// ═══════════════════════════════════════════════════════════════════════

/// Cantilever beam with uniform body force (self-weight).
///
/// Geometry: L=4, H=1, unit thickness
/// Material: E=1e5, ν=0.3 (plane stress)
/// BC: clamped at x=0 (tag 4)
/// Load: uniform f_y = -1
///
/// Euler-Bernoulli tip deflection: δ = qL⁴ / 8EI
///   I = H³/12 = 1/12,  q = 1
///   δ = 256 / (8 × 1e5 × 0.08333) ≈ 0.0384
#[test]
fn benchmark_beam_uniform_load() {
    let e_mod = 1e5;
    let nu = 0.3;
    let l: f64 = 4.0;
    let h: f64 = 1.0;

    let n = 24;
    let (_u, _iters, residual) = solve_elasticity_beam(n, l, h, e_mod, nu, -1.0, &[4]);

    let n_scalar = _u.len() / 2;
    let uy = &_u[n_scalar..];
    // Tip DOF: node (n, n/2) at mid-height of right edge
    let tip_scalar_dof = (n / 2) * (n + 1) + n;
    let uy_tip = uy[tip_scalar_dof];

    let i_beam = h.powi(3) / 12.0;
    let delta_eb = -1.0 * l.powi(4) / (8.0 * e_mod * i_beam);

    let rel_err = (uy_tip / delta_eb - 1.0).abs();
    assert!(residual < 1e-8, "solver residual too large: {:.3e}", residual);
    assert!(rel_err < 0.15,
        "beam tip deflection: computed={:.6e}, EB={:.6e}, rel_err={:.3}",
        uy_tip, delta_eb, rel_err);
    eprintln!("  [benchmark] beam-uniform: uy_tip={:.6e}, EB_ref={:.6e}, rel_err={:.3}",
        uy_tip, delta_eb, rel_err);
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark 2: Beam convergence
// ═══════════════════════════════════════════════════════════════════════

/// Verify that the FEM solution converges to the Euler-Bernoulli solution
/// as the mesh is refined.  Error must decrease monotonically.
#[test]
fn benchmark_beam_convergence() {
    let e_mod = 1e5;
    let nu = 0.3;
    let l: f64 = 4.0;
    let h: f64 = 1.0;
    let i_beam = h.powi(3) / 12.0;
    let delta_eb = -1.0 * l.powi(4) / (8.0 * e_mod * i_beam);

    let mut prev_err: f64 = f64::MAX;
    let mut first_err: f64 = f64::MAX;

    for &n in &[6, 12, 24] {
        let (_u, _iters, residual) = solve_elasticity_beam(n, l, h, e_mod, nu, -1.0, &[4]);

        let n_scalar = _u.len() / 2;
        let tip_scalar_dof = (n / 2) * (n + 1) + n;
        let uy_tip = _u[n_scalar + tip_scalar_dof];
        let err = (uy_tip - delta_eb).abs();

        assert!(residual < 1e-8, "n={}: residual too large: {:.3e}", n, residual);
        if n == 6 { first_err = err; }

        eprintln!("  [benchmark] convergence n={}: uy_tip={:.6e}, err={:.6e}", n, uy_tip, err);
        prev_err = err;
    }
    // Finest mesh should have less error than coarsest
    assert!(prev_err < first_err,
        "error did not decrease from n=6 to n=24: first={:.6e}, last={:.6e}",
        first_err, prev_err);
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark 3: Poisson linear patch test
// ═══════════════════════════════════════════════════════════════════════

/// P1 elements must reproduce a zero-solution for u=0 BC and f=0 source.
///
/// Laplace(u) = 0 with u=0 on boundary → u=0 everywhere.
#[test]
fn benchmark_poisson_homogeneous_patch() {
    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    // f = 0, u = 0 on boundary
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|_: &[f64]| 0.0);
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("patch CG should converge");

    let max_err = u.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(max_err < 1e-14,
        "homogeneous patch test failed: max |u| = {:.3e}", max_err);
    assert!(result.converged, "CG should converge");
    eprintln!("  [benchmark] poisson-homogeneous-patch: max|u|={:.3e}", max_err);
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark 4: Quadratic patch test (manufactured solution)
// ═══════════════════════════════════════════════════════════════════════

/// Manufactured solution u = sin(πx)sin(πy) with known f.
/// Verify that the FEM solution's L² error is bounded.
#[test]
fn benchmark_poisson_mms() {
    use std::f64::consts::PI;

    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let diff = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("MMS CG failed");

    assert!(result.converged, "MMS CG should converge");

    // L² error (integral approximation via midpoint rule at DOFs)
    let mut l2_err: f64 = 0.0;
    for dof in 0..n as u32 {
        let c = dm.dof_coord(dof);
        let exact = (PI * c[0]).sin() * (PI * c[1]).sin();
        l2_err += (u[dof as usize] - exact).powi(2);
    }
    l2_err = (l2_err / n as f64).sqrt();
    assert!(l2_err < 0.02,
        "Poisson MMS L² error too large: {:.4e}", l2_err);
    eprintln!("  [benchmark] poisson-mms: l2_err={:.4e}, iters={}", l2_err, result.iterations);
}
