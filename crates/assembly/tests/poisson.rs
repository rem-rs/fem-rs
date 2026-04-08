//! Integration tests: Poisson equation -Δu = f on the unit square.
//!
//! Exact solution: u(x,y) = sin(πx) sin(πy)
//! Forcing:        f(x,y) = 2π² sin(πx) sin(πy)
//! BCs:            u = 0 on ∂Ω (Dirichlet)
//!
//! Measured convergence rates (matches theory):
//! - P1: O(h²),  constant ≈ 1.38 → n=16 gives 5.4e-3
//! - P2: O(h³),  constant ≈ 2.25 → n=16 gives 5.5e-4
//!
//! Acceptance criteria (calibrated against measured constants):
//! - P1, 16×16 mesh: L2 error < 6e-3
//! - P2, 16×16 mesh: L2 error < 8e-4

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3}};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Exact solution and forcing ──────────────────────────────────────────────

fn u_exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }

fn forcing(x: &[f64]) -> f64 { 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() }

// ─── L2 error computation ────────────────────────────────────────────────────

/// Compute ||u_h − u_exact||_{L2(Ω)} using a 5th-order quadrature rule.
fn l2_error<M: MeshTopology>(
    uh:        &[f64],
    space:     &H1Space<M>,
    ref_elem:  &dyn ReferenceElement,
) -> f64 {
    let mesh = space.mesh();
    let quad = ref_elem.quadrature(2 * ref_elem.order() as u8 + 2);
    let n_ldofs = ref_elem.n_dofs();

    let mut phi     = vec![0.0_f64; n_ldofs];
    let mut err_sq  = 0.0_f64;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs  = space.element_dofs(e);

        // Jacobian det for this element.
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;

            ref_elem.eval_basis(xi, &mut phi);

            // Discrete solution at this QP.
            let uh_q: f64 = dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| uh[d as usize] * p)
                .sum();

            // Physical coordinates.
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];

            let diff = uh_q - u_exact(&xp);
            err_sq += w * diff * diff;
        }
    }

    err_sq.sqrt()
}

// ─── Solver (dense, via nalgebra) ────────────────────────────────────────────

/// Solve `K u = f` using nalgebra's LU decomposition on a dense copy.
fn dense_solve(mat: &fem_linalg::CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = mat.nrows;
    let dense_flat = mat.to_dense();
    let a = DMatrix::from_row_slice(n, n, &dense_flat);
    let b = DVector::from_column_slice(rhs);
    let lu = a.lu();
    lu.solve(&b)
        .expect("dense_solve: system is singular")
        .as_slice()
        .to_vec()
}

// ─── Poisson solve helper ─────────────────────────────────────────────────────

fn solve_poisson<M: MeshTopology + Clone>(
    mesh:  M,
    order: u8,
) -> f64 {
    let space = H1Space::new(mesh.clone(), order);

    // Assemble K and f.
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(forcing);
    let quad_order = 2 * order + 1;

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_order);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad_order);

    // Apply homogeneous Dirichlet BC on all boundary nodes.
    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    let values = vec![0.0_f64; bdofs.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &values);

    // Solve.
    let uh = dense_solve(&mat, &rhs);

    // L2 error.
    let ref_elem: Box<dyn ReferenceElement> = match order {
        1 => Box::new(TriP1),
        2 => Box::new(TriP2),
        3 => Box::new(TriP3),
        _ => panic!("unsupported order"),
    };
    l2_error(&uh, &space, ref_elem.as_ref())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn poisson_p1_16x16_l2_error() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let err  = solve_poisson(mesh, 1);
    println!("P1 16×16 L2 error = {err:.3e}");
    assert!(err < 6e-3, "P1 L2 error too large: {err:.3e} >= 6e-3");
}

#[test]
fn poisson_p2_16x16_l2_error() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let err  = solve_poisson(mesh, 2);
    println!("P2 16×16 L2 error = {err:.3e}");
    assert!(err < 8e-4, "P2 L2 error too large: {err:.3e} >= 8e-4");
}

/// Verify P1 convergence rate ≥ 1.9 (theoretical: 2).
#[test]
fn poisson_p1_convergence_rate() {
    let err8  = solve_poisson(SimplexMesh::<2>::unit_square_tri(8),  1);
    let err16 = solve_poisson(SimplexMesh::<2>::unit_square_tri(16), 1);
    let rate = (err8 / err16).log2();
    println!("P1 convergence rate = {rate:.2}");
    assert!(rate > 1.9, "P1 convergence rate {rate:.2} < 1.9");
}

/// Verify P2 convergence rate ≥ 2.8 (theoretical: 3).
#[test]
fn poisson_p2_convergence_rate() {
    let err8  = solve_poisson(SimplexMesh::<2>::unit_square_tri(8),  2);
    let err16 = solve_poisson(SimplexMesh::<2>::unit_square_tri(16), 2);
    let rate = (err8 / err16).log2();
    println!("P2 convergence rate = {rate:.2}");
    assert!(rate > 2.8, "P2 convergence rate {rate:.2} < 2.8");
}

/// Debug: show P3 errors at multiple refinements (manual diagnostic, not CI).
#[test]
#[ignore]
fn poisson_p3_debug_rates() {
    for n in [2usize, 4, 8, 16] {
        let err = solve_poisson(SimplexMesh::<2>::unit_square_tri(n), 3);
        println!("P3 n={n}: error = {err:.6e}");
    }
    let errs: Vec<f64> = [2usize, 4, 8, 16].iter()
        .map(|&n| solve_poisson(SimplexMesh::<2>::unit_square_tri(n), 3))
        .collect();
    for i in 0..errs.len()-1 {
        let ns = [2usize, 4, 8, 16];
        println!("Rate {}→{}: {:.3}", ns[i], ns[i+1], (errs[i]/errs[i+1]).log2());
    }
}

/// Verify P3 L2 error < 8e-4 on a 16×16 mesh (theoretical: O(h⁴)).
#[test]
fn poisson_p3_16x16_l2_error() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let err  = solve_poisson(mesh, 3);
    println!("P3 16×16 L2 error = {err:.3e}");
    assert!(err < 8e-4, "P3 L2 error too large: {err:.3e} >= 8e-4");
}

/// Verify P3 convergence rate ≥ 3.5 (theoretical: 4).
/// Uses n=8→16 to avoid pre-asymptotic regime at small mesh sizes.
#[test]
fn poisson_p3_convergence_rate() {
    let err8  = solve_poisson(SimplexMesh::<2>::unit_square_tri(8),  3);
    let err16 = solve_poisson(SimplexMesh::<2>::unit_square_tri(16), 3);
    let rate = (err8 / err16).log2();
    println!("P3 convergence rate = {rate:.2}");
    assert!(rate > 3.5, "P3 convergence rate {rate:.2} < 3.5");
}

// ─── Quad4 Poisson tests ────────────────────────────────────────────────────

/// Solve Poisson on a Quad4 mesh and return the max nodal error.
fn solve_poisson_quad(n: usize) -> f64 {
    use fem_element::lagrange::QuadQ1;

    let mesh = SimplexMesh::<2>::unit_square_quad(n);
    let space = H1Space::new(mesh.clone(), 1);

    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(forcing);

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    let values = vec![0.0_f64; bdofs.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &values);

    let uh = dense_solve(&mat, &rhs);

    // Max nodal error
    let dm = space.dof_manager();
    let mut max_err = 0.0_f64;
    for i in 0..dm.n_dofs {
        let c = dm.dof_coord(i as u32);
        let err = (uh[i] - u_exact(c)).abs();
        if err > max_err { max_err = err; }
    }
    max_err
}

#[test]
fn poisson_q1_16x16_error() {
    let err = solve_poisson_quad(16);
    println!("Q1 16×16 max nodal error = {err:.3e}");
    assert!(err < 2e-2, "Q1 max error too large: {err:.3e}");
}

#[test]
fn poisson_q1_convergence_rate() {
    let err8  = solve_poisson_quad(8);
    let err16 = solve_poisson_quad(16);
    let rate = (err8 / err16).log2();
    println!("Q1 quad convergence rate = {rate:.2}");
    assert!(rate > 1.8, "Q1 convergence rate {rate:.2} < 1.8");
}

/// Patch test: a linear function u(x,y)=x is exactly represented by P1
/// elements (zero error in H1 norm for a compatible load).
#[test]
fn patch_test_linear_p1() {
    // For u(x,y) = x: -Δu = 0, so f = 0.
    // BCs: u = x on boundary.
    let mesh  = SimplexMesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh.clone(), 1);

    // Zero forcing.
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(|_x| 0.0);

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 2);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 2);

    // Dirichlet BC: u = x on the boundary.
    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    let dm     = space.dof_manager();
    let values: Vec<f64> = bdofs.iter()
        .map(|&d| dm.dof_coord(d)[0])   // x-coordinate of DOF
        .collect();
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &values);

    let uh = dense_solve(&mat, &rhs);

    // Every DOF should equal its x-coordinate (interpolation of u=x is exact for P1).
    for (i, &u) in uh.iter().enumerate() {
        let x = dm.dof_coord(i as u32)[0];
        assert!((u - x).abs() < 1e-10, "DOF {i}: uh={u:.6e} expected={x:.6e}");
    }
}

// ─── 3D TetP1 / TetP2 Poisson tests ─────────────────────────────────────────

use fem_element::lagrange::{TetP1, TetP2};

/// 3-D exact solution: u(x,y,z) = sin(πx) sin(πy) sin(πz),
/// forcing: f = 3π² u.
fn u_exact_3d(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin()
}
fn forcing_3d(x: &[f64]) -> f64 {
    3.0 * PI * PI * u_exact_3d(x)
}

/// L2 error for a 3D Poisson solution on a tet mesh.
fn l2_error_3d<M: MeshTopology>(uh: &[f64], space: &H1Space<M>, order: u8) -> f64 {
    let ref_elem: Box<dyn ReferenceElement> = match order {
        1 => Box::new(TetP1),
        2 => Box::new(TetP2),
        _ => panic!("unsupported order"),
    };
    let mesh = space.mesh();
    let quad = ref_elem.quadrature(2 * order + 2);
    let n_ldofs = ref_elem.n_dofs();
    let mut phi = vec![0.0_f64; n_ldofs];
    let mut err_sq = 0.0_f64;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs  = space.element_dofs(e);

        // Affine Jacobian for tet: columns are x1-x0, x2-x0, x3-x0
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let x3 = mesh.node_coords(nodes[3]);
        let jac = [
            [x1[0]-x0[0], x2[0]-x0[0], x3[0]-x0[0]],
            [x1[1]-x0[1], x2[1]-x0[1], x3[1]-x0[1]],
            [x1[2]-x0[2], x2[2]-x0[2], x3[2]-x0[2]],
        ];
        let det_j = (
            jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
           -jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])
           +jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])
        ).abs();

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);

            let uh_q: f64 = dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| uh[d as usize] * p).sum();

            let xp = [
                x0[0] + jac[0][0]*xi[0] + jac[0][1]*xi[1] + jac[0][2]*xi[2],
                x0[1] + jac[1][0]*xi[0] + jac[1][1]*xi[1] + jac[1][2]*xi[2],
                x0[2] + jac[2][0]*xi[0] + jac[2][1]*xi[1] + jac[2][2]*xi[2],
            ];

            let diff = uh_q - u_exact_3d(&xp);
            err_sq += w * diff * diff;
        }
    }
    err_sq.sqrt()
}

fn solve_poisson_3d(n: usize, order: u8) -> f64 {
    let mesh  = SimplexMesh::<3>::unit_cube_tet(n);
    let space = H1Space::new(mesh.clone(), order);

    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(forcing_3d);
    let quad_order = 2 * order + 1;

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_order);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad_order);

    // All boundary faces of the unit cube.
    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4, 5, 6]);
    let values = vec![0.0_f64; bdofs.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &values);

    let uh = dense_solve(&mat, &rhs);
    l2_error_3d(&uh, &space, order)
}

/// TetP1 3D Poisson: L2 error < 1e-1 on a 4×4×4 mesh.
#[test]
fn poisson_tet_p1_l2_error() {
    let err = solve_poisson_3d(4, 1);
    println!("TetP1 4×4×4 L2 error = {err:.3e}");
    assert!(err < 1e-1, "TetP1 3D L2 error too large: {err:.3e}");
}

/// TetP1 convergence rate ≥ 1.8 (theory: O(h²)).
#[test]
fn poisson_tet_p1_convergence_rate() {
    let err4 = solve_poisson_3d(4, 1);
    let err8 = solve_poisson_3d(8, 1);
    let rate = (err4 / err8).log2();
    println!("TetP1 convergence rate = {rate:.2}");
    assert!(rate > 1.8, "TetP1 convergence rate {rate:.2} < 1.8");
}

/// TetP2 3D Poisson: L2 error < 1e-2 on a 4×4×4 mesh (O(h³)).
#[test]
fn poisson_tet_p2_l2_error() {
    let err = solve_poisson_3d(4, 2);
    println!("TetP2 4×4×4 L2 error = {err:.3e}");
    assert!(err < 1e-2, "TetP2 3D L2 error too large: {err:.3e}");
}

/// TetP2 convergence rate ≥ 2.5 (theory: O(h³)).
/// NOTE: This test requires n=8 3D mesh which is slow; mark as #[ignore] for CI.
#[test]
#[ignore]
fn poisson_tet_p2_convergence_rate() {
    let err4 = solve_poisson_3d(4, 2);
    let err8 = solve_poisson_3d(8, 2);
    let rate = (err4 / err8).log2();
    println!("TetP2 convergence rate = {rate:.2}");
    assert!(rate > 2.5, "TetP2 convergence rate {rate:.2} < 2.5");
}
