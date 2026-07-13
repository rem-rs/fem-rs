//! Phase C: Matrix-Free × AMR accuracy verification
//!
//! Validates that the matrix-free operator ([`SimpleDiffusionOp`]) produces
//! the same solutions as the explicit sparse matrix on both uniform and
//! AMR (non-conforming) meshes for the Poisson equation.
//!
//! ## Test hierarchy
//!
//! | Test | Mesh | Constraints | Tolerance |
//! |------|------|-------------|----------|
//! | `test_mf_vs_matrix_uniform_q1` | Uniform Quad4 | None (conforming) | 1e-14 |
//! | `test_mf_vs_matrix_amr_q1` | AMR Quad4 | Hanging nodes | 1e-12 |
//!
//! ## Methodology
//!
//! Both matrix and MF paths solve the same problem `-Δu = f` with
//! homogeneous Dirichlet BC. The MF path wraps [`SimpleDiffusionOp`] in
//! [`solve_cg_operator`] with a projection for BCs (+ constraints on
//! AMR meshes). Solutions are compared component-wise.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    amr_mf::{SimpleDiffusionOp, AmrAwareOperator},
};
use fem_core::ElemId;
use fem_mesh::{Mesh, topology::MeshTopology, amr::refine_nonconforming_quad};
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs, apply_hanging_constraints},
};
use fem_solver::{solve_cg, solve_cg_operator, SolverConfig};

// ─── Exact solution and forcing ──────────────────────────────────────────────

/// u(x,y) = sin(πx) sin(πy) — smooth, vanishes on ∂[0,1]².
fn u_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

/// f = -Δu = 2π² sin(πx) sin(πy).
fn forcing(x: &[f64]) -> f64 {
    2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
}

// ─── Solver config (shared) ─────────────────────────────────────────────────

fn solver_cfg() -> SolverConfig {
    SolverConfig {
        rtol: 1e-12,
        max_iter: 5000,
        verbose: false,
        ..SolverConfig::default()
    }
}

// ─── Helper: max component-wise difference ──────────────────────────────────

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test C1: MF vs matrix — uniform Quad4 mesh
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn test_mf_vs_matrix_uniform_q1() {
    let n = 8;                // 8×8 quad mesh → 9×9 = 81 nodes
    let order = 1u8;
    let quad_order = 3;

    let mesh = Mesh::<2>::unit_square_quad(n);
    let space = H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();

    // ── Assemble (shared) ──
    let k_mat = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order,
    );
    let f_vec = Assembler::assemble_linear(
        &space, &[&DomainSourceIntegrator::new(forcing)], quad_order,
    );

    // Boundary DOFs (unit square: tags 1..4 on each edge)
    let bnd = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);

    // ── Matrix path ──
    let mut k_bc = k_mat.clone();
    let mut f_bc = f_vec.clone();
    apply_dirichlet(&mut k_bc, &mut f_bc, &bnd, &vec![0.0; bnd.len()]);

    let mut u_mat = vec![0.0; n_dofs];
    solve_cg(&k_bc, &f_bc, &mut u_mat, &solver_cfg())
        .expect("CG (matrix) failed on uniform mesh");

    // ── MF path ──
    let op = SimpleDiffusionOp::new(H1Space::new(mesh, order), 1.0, quad_order);

    // RHS: zero boundary DOFs
    let mut f_mf = f_vec.clone();
    for &d in &bnd { f_mf[d as usize] = 0.0; }

    let mut u_mf = vec![0.0; n_dofs];
    solve_cg_operator(
        n_dofs, n_dofs,
        |x, y| {
            y.fill(0.0);
            // Project x: zero boundary DOFs (P * x)
            let mut xp = x.to_vec();
            for &d in &bnd { xp[d as usize] = 0.0; }
            // y += K * xp
            op.element_loop(&xp, y);
            // Project y: zero boundary DOFs (P * y)
            for &d in &bnd { y[d as usize] = 0.0; }
        },
        &f_mf, &mut u_mf, &solver_cfg(),
    ).expect("CG (MF) failed on uniform mesh");

    // ── Compare ──
    let diff = max_diff(&u_mat, &u_mf);
    println!("  uniform Q1 {n}×{n}: max |u_mat − u_mf| = {:.3e}", diff);
    assert!(
        diff < 1e-14,
        "uniform Q1: max diff = {:.3e} ≥ 1e-14",
        diff,
    );
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test C2: MF vs matrix — AMR Quad4 mesh with hanging nodes
// ═════════════════════════════════════════════════════════════════════════════

/// Create a 4×4 Quad4 mesh with one quadrant refined twice to produce
/// hanging nodes at the refinement boundary.
///
/// Initial mesh: 4×4 quads = 16 elements, 25 nodes.
/// Refine element 0 (bottom-left) → 4 sub-quads, 2 hanging nodes.
/// Refine two of those sub-quads again → another level of hanging nodes.
fn make_amr_mesh() -> (Mesh<2>, Vec<fem_mesh::HangingNodeConstraint>) {
    let mesh = Mesh::<2>::unit_square_quad(4);  // 4×4

    // First refinement level: element 0 only
    let (m1, c1) = refine_nonconforming_quad(&mesh, &[0], None);
    assert!(!c1.is_empty(), "expected hanging nodes after first refine");

    // Second refinement level: sub-elements 1 and 2 of the refined block
    // (sub-elements of the original element 0 are now elements 0..3 in m1)
    let (m2, c2) = refine_nonconforming_quad(&m1, &[1, 2], None);

    // Merge constraints (c1 were resolved by the second refine; c2 are active)
    (m2, c2)
}

#[test]
fn test_mf_vs_matrix_amr_q1() {
    let order = 1u8;
    let quad_order = 3;

    let (mesh, constraints) = make_amr_mesh();
    assert!(!constraints.is_empty(), "AMR mesh must have hanging nodes");

    let n_elems = mesh.n_elems();
    println!("  AMR mesh: {} elements, {} hanging constraints",
             n_elems, constraints.len());

    let space = H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();

    // Assemble full system (unconstrained)
    let k_full = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order,
    );
    let f_full = Assembler::assemble_linear(
        &space, &[&DomainSourceIntegrator::new(forcing)], quad_order,
    );

    // Boundary DOFs
    let bnd = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);

    // ── Matrix path: apply hanging constraints, then Dirichlet BC ──
    let mut k_amr = k_full.clone();
    let mut f_amr = f_full.clone();
    apply_hanging_constraints(&mut k_amr, &mut f_amr, &constraints);
    apply_dirichlet(&mut k_amr, &mut f_amr, &bnd, &vec![0.0; bnd.len()]);

    let mut u_mat = vec![0.0; n_dofs];
    solve_cg(&k_amr, &f_amr, &mut u_mat, &solver_cfg())
        .expect("CG (matrix) failed on AMR mesh");

    // ── MF path: apply_amr with constraint + Dirichlet projection ──
    let op = SimpleDiffusionOp::new(H1Space::new(mesh, order), 1.0, quad_order);

    // RHS for MF: apply C^T (scatter of constrained entries to parents),
    // matching what apply_hanging_constraints does via expand_dof.
    // Then zero boundary DOFs for Dirichlet BC.
    let mut f_mf = f_full.clone();
    let f_orig = f_mf.clone();  // snapshot before modification
    for c in &constraints {
        if c.parent_a == c.parent_b {
            f_mf[c.parent_a] += f_orig[c.constrained];
        } else {
            f_mf[c.parent_a] += 0.5 * f_orig[c.constrained];
            f_mf[c.parent_b] += 0.5 * f_orig[c.constrained];
        }
        f_mf[c.constrained] = 0.0;
    }
    for &d in &bnd { f_mf[d as usize] = 0.0; }

    let mut u_mf = vec![0.0; n_dofs];
    solve_cg_operator(
        n_dofs, n_dofs,
        |x, y| {
            y.fill(0.0);
            // First apply Dirichlet projection on x (zero boundary)
            let mut xp = x.to_vec();
            for &d in &bnd { xp[d as usize] = 0.0; }
            // Apply constrained operator: y += C^T K C xp
            op.apply_amr(&xp, y, &constraints);
            // Project y: zero boundary DOFs
            for &d in &bnd { y[d as usize] = 0.0; }
        },
        &f_mf, &mut u_mf, &solver_cfg(),
    ).expect("CG (MF) failed on AMR mesh");

    // ── Compare ──
    let diff = max_diff(&u_mat, &u_mf);
    println!("  AMR Q1: max |u_mat − u_mf| = {:.3e}", diff);

    // AMR tolerance is relaxed because the constraint path differs slightly
    // between matrix (P^T K P via COO rebuild) and MF (gather-apply-scatter).
    // Both are algebraically equivalent, but floating-point accumulation
    // differs (element-loop assembly order vs. CSR entries), plus the
    // constraint-conditioned matrix may see different CG residuals.
    assert!(
        diff < 1e-12,
        "AMR Q1: max diff = {:.3e} ≥ 1e-12",
        diff,
    );
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test C3: MF operator spot-check — direct K·x vs element_loop
// ═════════════════════════════════════════════════════════════════════════════

/// Compare the raw operator application (no BCs) on a uniform mesh:
/// `y_mat = K·x` vs `y_mf = op.element_loop(x)` for several random vectors.
/// This validates the element-level integration independently of the solver.
#[test]
fn test_mf_operator_spot_check_uniform() {
    let n_elems = 6;
    let mesh = Mesh::<2>::unit_square_quad(n_elems);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let k = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], 3,
    );
    let op = SimpleDiffusionOp::new(space, 1.0, 3);

    // Test several random vectors
    for trial in 0..5 {
        let x: Vec<f64> = (0..n_dofs).map(|i| {
            // Deterministic pseudo-random: sin(i) pattern
            (i as f64 * 1.371 + trial as f64 * 7.319).sin()
        }).collect();

        // Matrix path
        let mut y_mat = vec![0.0; n_dofs];
        k.spmv(&x, &mut y_mat);

        // MF path
        let mut y_mf = vec![0.0; n_dofs];
        op.element_loop(&x, &mut y_mf);

        let diff = max_diff(&y_mat, &y_mf);
        println!("  trial {trial}: max |K·x − element_loop(x)| = {:.3e}", diff);
        assert!(
            diff < 1e-14,
            "operator spot-check trial {trial}: diff = {:.3e} ≥ 1e-14",
            diff,
        );
    }
}
