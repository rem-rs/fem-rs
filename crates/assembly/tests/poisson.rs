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
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2}};
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
    let quad = ref_elem.quadrature(5);
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
