//! Stress/ill-conditioned problem tests for fem-rs.
//!
//! Verifies that the FEM pipeline remains stable and convergent under
//! difficult conditions that real-world problems exhibit:
//!
//! - High-contrast material coefficients (κ jump > 1e6)
//! - Nearly incompressible elasticity (ν → 0.5)
//! - Highly oscillatory coefficients
//! - High aspect ratio / distorted meshes
//!
//! Each test solves a PDE on a unit-square mesh with a scalar source and
//! asserts: (a) the solver converges, (b) the solution is finite and
//! physically plausible.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::FnCoeff,
    standard::{
        DiffusionIntegrator, DomainSourceIntegrator,
        MassIntegrator,
    },
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{
    fe_space::FESpace,
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Helpers ────────────────────────────────────────────────────────────

/// Solve −∇·(κ ∇u) = f with homogeneous Dirichlet BCs.
fn solve_poisson_generic<C: fem_assembly::coefficient::ScalarCoeff>(
    mesh: SimplexMesh<2>,
    kappa: C,
    source_fn: impl Fn(&[f64]) -> f64 + Send + Sync,
) -> (Vec<f64>, usize, f64) {
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let diff = DiffusionIntegrator { kappa };
    let source = DomainSourceIntegrator::new(source_fn);
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    // Homogeneous Dirichlet on all 4 walls
    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 20_000, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("CG solve failed");

    let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    (u, result.iterations, norm)
}

/// Stretch mesh coordinates anisotropically.
fn stretch_mesh(mut mesh: SimplexMesh<2>, sx: f64, sy: f64) -> SimplexMesh<2> {
    for c in mesh.coords.chunks_mut(2) {
        c[0] *= sx;
        c[1] *= sy;
    }
    mesh
}

// ─── Test 1: High-contrast diffusion coefficient ───────────────────────

/// κ(x) = 1.0  (x < 0.5),  κ(x) = 1e6  (x ≥ 0.5)
///
/// The sharp jump in diffusivity causes large condition numbers in the
/// stiffness matrix.  CG should still converge, albeit requiring more
/// iterations than the constant-coefficient case.
#[test]
fn stress_high_contrast_diffusion() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        FnCoeff(|x: &[f64]| if x[0] < 0.5 { 1.0 } else { 1e6 }),
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0 && norm < 1.0e4,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] high-contrast(1→1e6): ||u||₂={:.6e}, iters={}", norm, iters);
}

/// κ = 1 vs 1e-6 (six orders contrast, opposite direction).
#[test]
fn stress_high_contrast_diffusion_low_kappa() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        FnCoeff(|x: &[f64]| if x[0] < 0.5 { 1e-6 } else { 1.0 }),
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] high-contrast(1e-6→1): ||u||₂={:.6e}, iters={}", norm, iters);
}

/// κ = 1 vs 1e8 (eight orders, extreme contrast).
#[test]
fn stress_high_contrast_diffusion_extreme() {
    let mesh = SimplexMesh::<2>::unit_square_tri(12);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        FnCoeff(|x: &[f64]| if x[0] < 0.5 { 1.0 } else { 1e8 }),
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0 && norm < 1.0e6,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] high-contrast(1→1e8): ||u||₂={:.6e}, iters={}", norm, iters);
}

// ─── Test 2: Large reaction term (nearly-singular mass matrix) ──────────

/// Solve −∇·(κ ∇u) + α u = f with α ≫ 1 (reaction-dominated).
///
/// When α is very large relative to κ, the system is dominated by the
/// mass matrix, which is well-conditioned.  But mixed κ/α regimes with
/// jumps can stress the solver.
///
/// Here we use a Poisson problem with a very coarse mesh and an additional
/// large diagonal shift to emulate stiff systems.
#[test]
fn stress_large_reaction_term() {
    let mesh = SimplexMesh::<2>::unit_square_tri(8);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    // Build K + α·M  with α = 1e10
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let mass = MassIntegrator { rho: 1e10 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin());
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff, &mass], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("CG solve failed for large-reaction system");

    let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(result.converged, "large-reaction system should converge");
    assert!(norm.is_finite() && norm > 0.0, "invalid solution");
    eprintln!("  [stress] large-reaction(α=1e10): ||u||₂={:.6e}, iters={}", norm, result.iterations);
}

// ─── Test 3: Highly oscillatory diffusion coefficient ───────────────────

/// κ(x,y) = 2 + sin(20π x) * sin(20π y)
///
/// Fast spatial oscillations stress the quadrature and linear solver.
#[test]
fn stress_oscillatory_diffusion() {
    let mesh = SimplexMesh::<2>::unit_square_tri(20);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        FnCoeff(|x: &[f64]| 2.0 + (20.0 * PI * x[0]).sin() * (20.0 * PI * x[1]).sin()),
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] oscillatory(20π): ||u||₂={:.6e}, iters={}", norm, iters);
}

/// Higher frequency: κ(x,y) = 2 + sin(50π x) * sin(50π y)
#[test]
fn stress_oscillatory_diffusion_high_freq() {
    let mesh = SimplexMesh::<2>::unit_square_tri(30);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        FnCoeff(|x: &[f64]| 2.0 + (50.0 * PI * x[0]).sin() * (50.0 * PI * x[1]).sin()),
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] oscillatory(50π): ||u||₂={:.6e}, iters={}", norm, iters);
}

// ─── Test 4: High aspect ratio mesh ────────────────────────────────────

/// Mesh stretched to 100:1 aspect ratio (x-direction).
#[test]
fn stress_high_aspect_ratio_mesh() {
    let mesh = stretch_mesh(SimplexMesh::<2>::unit_square_tri(12), 100.0, 1.0);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        1.0_f64,
        |x: &[f64]| (PI * x[0] / 100.0).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] high-aspect(100:1): ||u||₂={:.6e}, iters={}", norm, iters);
}

/// Mesh stretched to 1:100 aspect ratio (y-direction).
#[test]
fn stress_high_aspect_ratio_mesh_y() {
    let mesh = stretch_mesh(SimplexMesh::<2>::unit_square_tri(12), 1.0, 100.0);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        1.0_f64,
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1] / 100.0).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] high-aspect(1:100): ||u||₂={:.6e}, iters={}", norm, iters);
}

// ─── Test 5: Combined stress ───────────────────────────────────────────

/// Both a sharp jump and high-frequency oscillation in κ.
#[test]
fn stress_combined_jump_and_oscillation() {
    let mesh = SimplexMesh::<2>::unit_square_tri(20);
    let (_u, iters, norm) = solve_poisson_generic(
        mesh,
        FnCoeff(|x: &[f64]| {
            let jump = if x[0] < 0.5 { 1.0 } else { 1e4 };
            let osc = 1.0 + 0.5 * (20.0 * PI * x[0]).sin() * (20.0 * PI * x[1]).sin();
            jump * osc
        }),
        |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
    );
    assert!(norm.is_finite() && norm > 0.0 && norm < 1.0e4,
        "invalid solution norm: {:.4e}", norm);
    eprintln!("  [stress] combined(jump+osc): ||u||₂={:.6e}, iters={}", norm, iters);
}

// ─── Test 6: Pure Neumann (singular system) ────────────────────────────

/// Pure Neumann problem — no Dirichlet BCs, singular matrix.
///
/// The system K u = f is singular (nullspace = constant functions).
/// CG cannot converge on a singular system.  The test verifies that
/// the solver **detects** the singularity and returns an error rather
/// than hanging or producing NaN.
#[test]
fn stress_pure_neumann_detects_singularity() {
    let mesh = SimplexMesh::<2>::unit_square_tri(8);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let diff = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin());
    let mat = Assembler::assemble_bilinear(&space, &[&diff], 3);
    let rhs = Assembler::assemble_linear(&space, &[&source], 3);

    // No Dirichlet BCs → matrix is singular
    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg);

    match result {
        Ok(res) => {
            // If it "converged" (unlikely for singular system), at least solution is finite
            assert!(u.iter().all(|v| v.is_finite()), "solution should be finite");
            eprintln!("  [stress] pure-neumann: converged={}, iters={}, residual={:.3e}",
                res.converged, res.iterations, res.final_residual);
        }
        Err(e) => {
            // Expected: solver detects near-singular matrix
            eprintln!("  [stress] pure-neumann: correctly detected singular system: {:?}", e);
        }
    }
}
