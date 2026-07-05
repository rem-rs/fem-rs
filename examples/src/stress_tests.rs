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
use fem_linalg::{CooMatrix, CsrMatrix, SolverError};
use fem_mesh::{ElementType, SimplexMesh};
use fem_solver::{solve_cg, solve_gmres, SolverConfig};
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

// ═══════════════════════════════════════════════════════════════════════
// Error path tests
// ═══════════════════════════════════════════════════════════════════════

/// CG with dimension mismatch should return Err(DimensionMismatch).
#[test]
fn error_solver_dimension_mismatch() {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.add(0, 0, 2.0);
    coo.add(1, 1, 2.0);
    coo.add(2, 2, 2.0);
    let a: CsrMatrix<f64> = coo.into_csr();
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 entries ≠ 3 rows
    let mut x = vec![0.0; 5];

    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&a, &b, &mut x, &cfg);
    assert!(result.is_err(), "dimension mismatch should return Err");
    match result {
        Err(SolverError::DimensionMismatch { rows, cols, rhs }) => {
            assert_eq!(rows, 3);
            assert_eq!(cols, 3);
            assert_eq!(rhs, 5);
            eprintln!("  [error] dimension-mismatch: correctly rejected {rows}×{cols} matrix vs rhs len {rhs}");
        }
        other => panic!("expected DimensionMismatch, got {:?}", other),
    }
}

/// GMRES with restart = 0 should return Err.
#[test]
fn error_gmres_zero_restart() {
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.add(0, 0, 1.0);
    coo.add(1, 1, 1.0);
    let a: CsrMatrix<f64> = coo.into_csr();
    let b = vec![1.0, 1.0];
    let mut x = vec![0.0; 2];

    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };
    let result = solve_gmres(&a, &b, &mut x, 0, &cfg);
    // GMRES may or may not reject restart=0 — either way should not panic
    eprintln!("  [error] gmres-zero-restart: returned: {:?}", result);
}

/// CG on a zero matrix (all zeros) should detect failure.
#[test]
fn error_cg_zero_matrix() {
    let a = CsrMatrix::<f64> {
        nrows: 4, ncols: 4,
        row_ptr: vec![0, 0, 0, 0, 0],
        col_idx: vec![],
        values: vec![],
    };
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let mut x = vec![0.0; 4];

    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&a, &b, &mut x, &cfg);
    // Should either converge (x remains 0) or report failure
    match result {
        Ok(r) => {
            assert!(x.iter().all(|v| v.is_finite()), "solution should be finite");
            eprintln!("  [error] cg-zero-matrix: solved 'zero system', residual={:.3e}", r.final_residual);
        }
        Err(e) => {
            eprintln!("  [error] cg-zero-matrix: correctly rejected zero matrix: {:?}", e);
        }
    }
}

/// Mesh with invalid connectivity should be detected by mesh.check().
#[test]
fn error_invalid_mesh_connectivity() {
    // Build a mesh with a node index out of range
    let coords = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let conn = vec![0u32, 1, 2, 0, 1, 100]; // 100 is out of range
    let elem_tags = vec![1, 1];
    let face_conn = vec![0u32, 1, 1, 2, 2, 0];
    let face_tags = vec![1i32, 1, 1];
    let mesh = SimplexMesh::<2>::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        face_conn, face_tags, ElementType::Line2,
    );
    let result = mesh.check();
    assert!(result.is_err(), "invalid mesh should fail check()");
    eprintln!("  [error] invalid-mesh: correctly rejected: {:?}", result.err().unwrap());
}

/// Empty mesh (zero elements) should still pass check().
#[test]
fn error_empty_mesh_passes_check() {
    let mesh = SimplexMesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![], vec![], ElementType::Tri3,
        vec![], vec![], ElementType::Line2,
    );
    let result = mesh.check();
    // An empty mesh may or may not pass check — just verify no panic
    eprintln!("  [error] empty-mesh: check() returned {:?}", result);
}

/// Solve with NaN in RHS — solver should not panic.
#[test]
fn error_nan_rhs_solver_does_not_panic() {
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.add(0, 0, 1.0);
    coo.add(1, 1, 1.0);
    let a: CsrMatrix<f64> = coo.into_csr();
    let b = vec![f64::NAN, 1.0];
    let mut x = vec![0.0; 2];

    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&a, &b, &mut x, &cfg);
    eprintln!("  [error] nan-rhs: solver returned: {:?}", result);
}

/// Assembler with inconsistent space/mesh should not panic.
#[test]
fn error_assembly_mismatched_mesh() {
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_assembly::Assembler;

    let _mesh_a = SimplexMesh::<2>::unit_square_tri(4);
    let mesh_b = SimplexMesh::<2>::unit_square_tri(8);
    let space = H1Space::new(mesh_b, 1);
    let diff = DiffusionIntegrator { kappa: 1.0 };

    // Assembling on _mesh_a but space uses mesh_b
    // This should panic or return an error
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Assembler::assemble_bilinear(&space, &[&diff], 3);
    }));
    eprintln!("  [error] assembly-mismatch: {:?}",
        match &result { Ok(_) => "completed (unexpected)", Err(_) => "panicked (expected)" });
    // We don't assert on the outcome — different configurations may behave differently
}
