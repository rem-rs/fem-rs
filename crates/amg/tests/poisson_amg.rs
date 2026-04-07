//! End-to-end integration tests: assemble → solve with AMG preconditioner.
//!
//! Assembles a 2-D Poisson problem and solves it using AMG-CG and the
//! reusable AmgSolver.  Verifies L2 accuracy and that the AMG hierarchy
//! has multiple levels (i.e. coarsening actually happened).

use std::f64::consts::PI;

use fem_amg::{AmgConfig, AmgSolver, solve_amg_cg};
use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_element::{ReferenceElement, lagrange::TriP1};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::SolverConfig;
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn u_exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
fn forcing(x: &[f64]) -> f64 { 2.0 * PI * PI * u_exact(x) }

fn build_system(n: usize) -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, H1Space<SimplexMesh<2>>) {
    let mesh  = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(forcing);
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
    (mat, rhs, space)
}

fn l2_error(uh: &[f64], space: &H1Space<SimplexMesh<2>>) -> f64 {
    let mesh = space.mesh();
    let quad = TriP1.quadrature(5);
    let mut phi = vec![0.0_f64; 3];
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
            TriP1.eval_basis(xi, &mut phi);
            let uh_q: f64 = dofs.iter().zip(phi.iter()).map(|(&d,&p)| uh[d as usize]*p).sum();
            let xp = [x0[0]+(x1[0]-x0[0])*xi[0]+(x2[0]-x0[0])*xi[1],
                      x0[1]+(x1[1]-x0[1])*xi[0]+(x2[1]-x0[1])*xi[1]];
            let diff = uh_q - u_exact(&xp);
            err_sq += w * diff * diff;
        }
    }
    err_sq.sqrt()
}

fn cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-8, atol: 1e-10, max_iter: 500, verbose: false, ..SolverConfig::default() }
}

#[test]
fn amg_cg_poisson_16x16() {
    let (mat, rhs, space) = build_system(16);
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_amg_cg(&mat, &rhs, &mut x, &AmgConfig::default(), &cfg()).unwrap();
    assert!(res.converged, "AMG-CG did not converge (iters={}, res={:.3e})", res.iterations, res.final_residual);
    let err = l2_error(&x, &space);
    println!("AMG-CG 16×16: iters={}, L2 err={:.3e}", res.iterations, err);
    assert!(err < 6e-3, "L2 error too large: {err:.3e}");
}

#[test]
fn amg_solver_reuse_poisson() {
    let (mat, rhs, space) = build_system(16);
    let solver = AmgSolver::setup(&mat, AmgConfig::default());
    assert!(solver.n_levels() >= 2, "AMG hierarchy must have at least 2 levels");
    println!("AMG levels: {}", solver.n_levels());

    // Solve twice with the same hierarchy.
    for _ in 0..2 {
        let n = mat.nrows;
        let mut x = vec![0.0_f64; n];
        let res = solver.solve(&mat, &rhs, &mut x, &cfg()).unwrap();
        assert!(res.converged);
        assert!(l2_error(&x, &space) < 6e-3);
    }
}

/// AMG should need fewer iterations than plain CG on a larger mesh (64×64).
///
/// On ~3600 free DOFs, CG needs O(√N) ≈ 60+ iterations while AMG is nearly
/// mesh-independent (< 30 iterations).
#[test]
fn amg_fewer_iters_than_cg_large() {
    use fem_solver::solve_cg;
    let (mat, rhs, _) = build_system(64);
    let n = mat.nrows;

    let mut x_cg = vec![0.0_f64; n];
    let res_cg = solve_cg(&mat, &rhs, &mut x_cg, &cfg()).unwrap();

    let mut x_amg = vec![0.0_f64; n];
    let res_amg = solve_amg_cg(&mat, &rhs, &mut x_amg, &AmgConfig::default(), &cfg()).unwrap();

    println!("n={n}: CG iters={}, AMG-CG iters={}", res_cg.iterations, res_amg.iterations);
    assert!(res_amg.iterations < res_cg.iterations,
        "AMG-CG ({}) should need fewer iters than CG ({})", res_amg.iterations, res_cg.iterations);
}
