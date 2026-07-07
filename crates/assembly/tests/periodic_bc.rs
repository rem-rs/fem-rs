//! Integration tests: periodic boundary conditions.
//!
//! Tests verify that `identify_periodic_dof_pairs` + `apply_periodic` correctly
//! impose u(0,y) = u(1,y) periodicity on the unit square.
//!
//! Problem: −Δu = f on [0,1]², periodic in x, homogeneous Dirichlet in y.
//! Exact solution: u(x,y) = cos(2πx) sin(πy)
//! Forcing:        f(x,y) = (4π² + π²) cos(2πx) sin(πy)

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2}};
use fem_mesh::{topology::MeshTopology, Mesh};
use fem_mesh::amr::HangingNodeConstraint;
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, apply_hanging_constraints, boundary_dofs, identify_periodic_dof_pairs, recover_hanging_values},
};

// ─── Exact solution and forcing ──────────────────────────────────────────────

fn u_exact(x: &[f64]) -> f64 {
    (2.0 * PI * x[0]).cos() * (PI * x[1]).sin()
}

fn forcing(x: &[f64]) -> f64 {
    // -Δu = (4π² + π²) cos(2πx) sin(πy)
    5.0 * PI * PI * (2.0 * PI * x[0]).cos() * (PI * x[1]).sin()
}

// ─── L2 error (triangle mesh, P1 or P2) ─────────────────────────────────────

fn l2_error_tri<M: MeshTopology>(
    uh:       &[f64],
    space:    &H1Space<M>,
    ref_elem: &dyn ReferenceElement,
) -> f64 {
    let mesh = space.mesh();
    let quad = ref_elem.quadrature(2 * ref_elem.order() as u8 + 2);
    let n_ldofs = ref_elem.n_dofs();
    let mut phi = vec![0.0_f64; n_ldofs];
    let mut err_sq = 0.0_f64;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs  = space.element_dofs(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);

            let uh_q: f64 = dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| uh[d as usize] * p).sum();

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

// ─── Dense solver ────────────────────────────────────────────────────────────

fn dense_solve(mat: &fem_linalg::CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = mat.nrows;
    let dense_flat = mat.to_dense();
    let a = DMatrix::from_row_slice(n, n, &dense_flat);
    let b = DVector::from_column_slice(rhs);
    a.lu().solve(&b)
        .expect("dense_solve: system is singular")
        .as_slice()
        .to_vec()
}

// ─── Periodic Poisson helper ─────────────────────────────────────────────────

/// Solve −Δu = f on unit square, periodic in x (left↔right), Dirichlet in y (top/bottom).
/// Returns the L2 error against the exact solution.
fn solve_periodic_poisson(n: usize, order: u8) -> f64 {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), order);

    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(forcing);
    let quad_order = 2 * order + 1;

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_order);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad_order);

    // Tag convention for unit_square_tri:
    //   tag 1 = bottom (y=0), tag 2 = right (x=1), tag 3 = top (y=1), tag 4 = left (x=0)
    //
    // Periodic in x: slave=right (tag 2), master=left (tag 4), offset=[-1, 0]
    // (x_slave + offset = x_right + [-1,0] ≈ x_left)
    let offset = [-1.0_f64, 0.0];
    let pairs = identify_periodic_dof_pairs(&mesh, space.dof_manager(), 4, 2, &offset, 1e-10);

    // Convert pairs to HangingNodeConstraints (slave = master, unit weight).
    let periodic_constraints: Vec<HangingNodeConstraint> = pairs.iter()
        .map(|&(slave, master)| HangingNodeConstraint {
            constrained: slave as usize,
            parent_a:    master as usize,
            parent_b:    master as usize,
        })
        .collect();

    // Apply periodicity first (eliminates slave DOFs).
    apply_hanging_constraints(&mut mat, &mut rhs, &periodic_constraints);

    // Dirichlet BC on top and bottom (y=0 and y=1).
    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 3]);
    let values = vec![0.0_f64; bdofs.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &values);

    let mut uh = dense_solve(&mat, &rhs);

    // Recover slave DOF values from masters.
    recover_hanging_values(&mut uh, &periodic_constraints);

    let ref_elem: Box<dyn ReferenceElement> = match order {
        1 => Box::new(TriP1),
        2 => Box::new(TriP2),
        _ => panic!("unsupported order"),
    };
    l2_error_tri(&uh, &space, ref_elem.as_ref())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Verify that the number of periodic pairs matches the expected count.
/// For an n×n mesh with P1, each vertical boundary has n+1 nodes.
/// The right boundary has n+1 nodes, so n+1 pairs are expected
/// (corners are on both top/bottom and left/right boundaries).
#[test]
fn periodic_pairs_count_p1() {
    let n = 8_usize;
    let mesh  = Mesh::<2>::unit_square_tri(n);
    let dm    = fem_space::DofManager::new(&mesh, 1);
    let offset = [-1.0_f64, 0.0];
    // master=left (tag 4), slave=right (tag 2)
    let pairs = identify_periodic_dof_pairs(&mesh, &dm, 4, 2, &offset, 1e-10);
    // n=8: right boundary has n+1 = 9 nodes
    assert_eq!(
        pairs.len(), n + 1,
        "expected {} periodic pairs for n={n} P1, got {}",
        n + 1, pairs.len()
    );
}

/// Verify periodicity is symmetric: swapping master/slave with negated offset gives same count.
#[test]
fn periodic_pairs_symmetric() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let dm   = fem_space::DofManager::new(&mesh, 1);
    // master=left(4), slave=right(2), slave+offset -> master: right + [-1,0] = left
    let pairs_lr = identify_periodic_dof_pairs(&mesh, &dm, 4, 2, &[-1.0_f64, 0.0], 1e-10);
    // Swapped: master=right(2), slave=left(4), slave+offset -> master: left + [+1,0] = right
    let pairs_rl = identify_periodic_dof_pairs(&mesh, &dm, 2, 4, &[ 1.0_f64, 0.0], 1e-10);
    assert_eq!(pairs_lr.len(), pairs_rl.len(), "pair count should be the same regardless of swap direction");
}

/// Solve periodic Poisson with P1 elements on a 16×16 mesh.
/// Expected L2 error: O(h²) ≈ 2.5e-3 for this problem.
#[test]
fn periodic_poisson_p1_l2_error() {
    let err = solve_periodic_poisson(16, 1);
    println!("Periodic Poisson P1 16×16 L2 error = {err:.3e}");
    assert!(err < 2e-2, "P1 periodic L2 error too large: {err:.3e}");
}

/// Verify P1 convergence rate ≥ 1.9 for periodic problem.
#[test]
fn periodic_poisson_p1_convergence_rate() {
    let err8  = solve_periodic_poisson(8,  1);
    let err16 = solve_periodic_poisson(16, 1);
    let rate = (err8 / err16).log2();
    println!("Periodic P1 convergence rate = {rate:.2}");
    assert!(rate > 1.9, "P1 periodic convergence rate {rate:.2} < 1.9");
}

/// Solve periodic Poisson with P2 elements on a 16×16 mesh.
/// Expected L2 error: O(h³) ≈ 3e-4.
#[test]
fn periodic_poisson_p2_l2_error() {
    let err = solve_periodic_poisson(16, 2);
    println!("Periodic Poisson P2 16×16 L2 error = {err:.3e}");
    assert!(err < 2e-3, "P2 periodic L2 error too large: {err:.3e}");
}

/// Verify P2 convergence rate ≥ 2.8 for periodic problem.
#[test]
fn periodic_poisson_p2_convergence_rate() {
    let err8  = solve_periodic_poisson(8,  2);
    let err16 = solve_periodic_poisson(16, 2);
    let rate = (err8 / err16).log2();
    println!("Periodic P2 convergence rate = {rate:.2}");
    assert!(rate > 2.8, "P2 periodic convergence rate {rate:.2} < 2.8");
}
