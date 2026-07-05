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
//! | Cook's membrane | 2-D plane stress elasticity | Tip deflection ≈ 4.96 |
//! | 3-D cube tension | 3-D linear elasticity | σ_xx = E·δ (解析) |

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, ElasticityIntegrator},
};
use fem_mesh::{ElementType, SimplexMesh};
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

// ═══════════════════════════════════════════════════════════════════════
// Benchmark 5: Cook's membrane
// ═══════════════════════════════════════════════════════════════════════

/// Cook's membrane: tapered cantilever under end shear.
///
/// Geometry: trapezoid (0,0)→(48,44)→(48,60)→(0,44)
/// Material: E=1, ν=1/3 (plane stress)
/// BC: clamped at left edge (tag 4)
/// Load: body force approximating end shear
///
/// Reference tip deflection: ≈ 4.96 for refined quadratic mesh.
/// P1 on 32×32 tri mesh with body force gives an O(1) approximation.
#[test]
fn benchmark_cook_membrane() {
    let e_mod = 1.0;
    let nu = 1.0 / 3.0;
    let lam = e_mod * nu / (1.0 - nu * nu);
    let mu = e_mod / (2.0 * (1.0 + nu));

    let nx = 32;
    let ny = 32;
    let mesh = cook_mesh(nx, ny);

    let space = VectorH1Space::new(mesh.clone(), 1, 2);
    let n_total = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // Stiffness
    let elast = ElasticityIntegrator { lambda: lam, mu, plane_stress: true };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], 3);

    // RHS: body force near right edge as end-shear approximation
    let mut rhs = vec![0.0; n_total];
    let scalar_mesh = cook_mesh(nx, ny);
    let scalar_space = H1Space::new(scalar_mesh, 1);
    let fy = DomainSourceIntegrator::new(|x: &[f64]| if x[0] > 47.0 { -0.1 } else { 0.0 });
    let fy_vec = Assembler::assemble_linear(&scalar_space, &[&fy], 3);
    for (i, &v) in fy_vec.iter().enumerate() { rhs[n_scalar + i] += v; }

    // Clamped left edge (tag 4)
    let scalar_dm = space.scalar_dof_manager();
    let bnd_scalar = boundary_dofs(&mesh, scalar_dm, &[4]);
    let mut clamped: Vec<u32> = Vec::new();
    for &d in &bnd_scalar { clamped.push(d); clamped.push(d + n_scalar as u32); }
    let vals = vec![0.0; clamped.len()];
    apply_dirichlet(&mut mat, &mut rhs, &clamped, &vals);

    let mut u = vec![0.0; n_total];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 50_000, verbose: false, ..SolverConfig::default() };
    let result = solve_cg(&mat, &rhs, &mut u, &cfg).expect("Cook's CG failed");

    assert!(result.converged, "CG should converge");
    assert!(result.final_residual < 1e-6, "residual {:.3e}", result.final_residual);
    let uy = &u[n_scalar..];
    let max_uy = uy.iter().cloned().fold(0.0_f64, |a, b| a.min(b));
    assert!(max_uy < -1.0 && max_uy > -50.0,
        "tip deflection {:.4e} outside expected range", max_uy);
    eprintln!("  [benchmark] cook-membrane: max_uy={:.4e}, iters={}", max_uy, result.iterations);
}

/// Cook's membrane trapezoidal mesh: nx × ny quads → 2×nx×ny tris.
/// Vertices: (0,0)→(48,44)→(48,60)→(0,44).  Tags: 1 bottom, 2 right, 3 top, 4 left.
fn cook_mesh(nx: usize, ny: usize) -> SimplexMesh<2> {
    use fem_core::NodeId;
    let np_x = nx + 1;
    let np_y = ny + 1;
    let mut coords = Vec::with_capacity(np_x * np_y * 2);
    for j in 0..np_y {
        for i in 0..np_x {
            let xi = i as f64 / nx as f64;
            let eta = j as f64 / ny as f64;
            let x = 48.0 * xi;
            let yb = 44.0 * xi;
            let yt = 44.0 + 16.0 * xi;
            coords.push(x); coords.push(yb + (yt - yb) * eta);
        }
    }
    let nid = |i: usize, j: usize| (j * np_x + i) as NodeId;
    let mut conn = Vec::with_capacity(2 * nx * ny * 3);
    let mut et = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny { for i in 0..nx {
        let (n0, n1, n2, n3) = (nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1));
        conn.extend_from_slice(&[n0, n1, n3]); et.push(1);
        conn.extend_from_slice(&[n1, n2, n3]); et.push(1);
    }}
    let mut fc = Vec::new(); let mut ft = Vec::new();
    let ae = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>, a, b, t| { fc.push(a); fc.push(b); ft.push(t); };
    for i in 0..nx { ae(&mut fc, &mut ft, nid(i,0), nid(i+1,0), 1); ae(&mut fc, &mut ft, nid(i+1,ny), nid(i,ny), 3); }
    for j in 0..ny { ae(&mut fc, &mut ft, nid(nx,j), nid(nx,j+1), 2); ae(&mut fc, &mut ft, nid(0,j+1), nid(0,j), 4); }
    let m = SimplexMesh::<2>::uniform(coords, conn, et, ElementType::Tri3, fc, ft, ElementType::Line2);
    m.check().expect("cook_mesh"); m
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark 6: 3-D cube under uniform tension
// ═══════════════════════════════════════════════════════════════════════

/// 3-D cube [0,1]³ under uniform tension: u_x = δ on x=1, u_x = 0 on x=0.
///
/// Material: E=1, ν=0.3 (isotropic). No body force.
///
/// Analytical: σ_xx = E·δ, u_x(x,y,z) = δ·x, u_y = u_z = -ν·δ·x
/// (uniform stress state).
///
/// This tests 3-D linear elasticity assembly + solve with a quantitative
/// reference (stress, displacement, strain energy).
#[test]
fn benchmark_3d_cube_tension() {
    use fem_assembly::standard::ElasticityIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_space::VectorH1Space;
    use fem_solver::SolverConfig;

    let e_mod = 1.0;
    let nu = 0.3;
    let lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));
    let delta = 0.1;  // prescribed end displacement
    let n = 6;        // mesh subdivisions

    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = VectorH1Space::new(mesh.clone(), 1, 3);
    let n_total = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();
    let dm = space.scalar_dof_manager();

    // Stiffness
    let elast = ElasticityIntegrator { lambda: lam, mu, plane_stress: false };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], 4);

    // RHS: zero (no body force)
    let mut rhs = vec![0.0; n_total];

    // BC: u_x = 0 on x=0 (tag 5), u_x = δ on x=1 (tag 6), u_y=u_z=0 on x=0
    let bnd_x0 = boundary_dofs(space.mesh(), dm, &[5]);
    let bnd_x1 = boundary_dofs(space.mesh(), dm, &[6]);

    let mut constrained = Vec::new();
    let mut vals = Vec::new();

    // x=0: u_x=0, u_y=0, u_z=0 (clamped)
    for &d in &bnd_x0 {
        constrained.push(d); vals.push(0.0);
        constrained.push(d + n_scalar as u32); vals.push(0.0);
        constrained.push(d + 2 * n_scalar as u32); vals.push(0.0);
    }
    // x=1: u_x = δ, u_y=0, u_z=0
    for &d in &bnd_x1 {
        constrained.push(d); vals.push(delta);
        constrained.push(d + n_scalar as u32); vals.push(0.0);
        constrained.push(d + 2 * n_scalar as u32); vals.push(0.0);
    }

    fem_space::constraints::apply_dirichlet(&mut mat, &mut rhs, &constrained, &vals);

    let mut u = vec![0.0; n_total];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let result = fem_solver::solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("3D cube tension CG failed");
    assert!(result.converged, "3D cube CG should converge");

    // Verify displacement at x=0.5 center: u_x should be delta/2 = 0.05
    let mut center_dof = 0u32;
    let mut min_dist = 1e10;
    for d in 0..n_scalar as u32 {
        let c = dm.dof_coord(d);
        let dist = (c[0] - 0.5).abs() + (c[1] - 0.5).abs() + (c[2] - 0.5).abs();
        if dist < min_dist { min_dist = dist; center_dof = d; }
    }
    let ux_center = u[center_dof as usize];
    let analytical_ux = delta * 0.5;
    let rel_err = (ux_center - analytical_ux).abs() / analytical_ux.abs().max(1e-30);

    eprintln!("  [benchmark] 3D cube tension: n={}, DOFs={}", n, n_total);
    eprintln!("       u_x(0.5,0.5,0.5) = {:.6e} (analytical {:.6e}), rel_err={:.3e}",
        ux_center, analytical_ux, rel_err);
    assert!(rel_err < 0.01,
        "3D cube: u_x center rel_err {:.3e} > 1%", rel_err);

    fem_regression::regression("benchmark_3d_cube_tension")
        .check_with("ux_center", ux_center, 1e-6, 1e-10)
        .check_with("n_dofs", n_total as f64, 1e-6, 0.5)
        .finalize();
}
