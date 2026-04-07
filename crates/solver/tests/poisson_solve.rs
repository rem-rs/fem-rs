//! End-to-end integration tests: assemble → solve with iterative solvers.
//!
//! We assemble the Poisson system on a 16×16 P1 mesh (from fem-assembly)
//! and solve it with each solver from fem-solver, verifying L2 accuracy.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{solve_bicgstab, solve_cg, solve_gmres, solve_pcg_ilu0, solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn u_exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
fn forcing(x: &[f64]) -> f64 { 2.0 * PI * PI * u_exact(x) }

/// Assemble the Poisson system on a 16×16 P1 unit-square mesh.
/// Returns (mat, rhs, space).
fn build_poisson_system() -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, H1Space<SimplexMesh<2>>) {
    let mesh  = SimplexMesh::<2>::unit_square_tri(16);
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
    use fem_element::{ReferenceElement, lagrange::TriP1};
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
    SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() }
}

#[test]
fn poisson_cg() {
    let (mat, rhs, space) = build_poisson_system();
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_cg(&mat, &rhs, &mut x, &cfg()).unwrap();
    assert!(res.converged, "CG did not converge");
    assert!(l2_error(&x, &space) < 6e-3);
}

#[test]
fn poisson_pcg_jacobi() {
    let (mat, rhs, space) = build_poisson_system();
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_pcg_jacobi(&mat, &rhs, &mut x, &cfg()).unwrap();
    assert!(res.converged, "PCG-Jacobi did not converge");
    assert!(l2_error(&x, &space) < 6e-3);
    // Jacobi preconditioner should reduce iteration count vs plain CG
    let mut x2 = vec![0.0_f64; n];
    let res2 = solve_cg(&mat, &rhs, &mut x2, &cfg()).unwrap();
    println!("CG iters={}, PCG-Jacobi iters={}", res2.iterations, res.iterations);
}

#[test]
fn poisson_pcg_ilu0() {
    let (mat, rhs, space) = build_poisson_system();
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_pcg_ilu0(&mat, &rhs, &mut x, &cfg()).unwrap();
    assert!(res.converged, "PCG-ILU0 did not converge");
    assert!(l2_error(&x, &space) < 6e-3);
}

#[test]
fn poisson_gmres() {
    let (mat, rhs, space) = build_poisson_system();
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_gmres(&mat, &rhs, &mut x, 30, &cfg()).unwrap();
    assert!(res.converged, "GMRES did not converge");
    assert!(l2_error(&x, &space) < 6e-3);
}

#[test]
fn poisson_bicgstab() {
    let (mat, rhs, space) = build_poisson_system();
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_bicgstab(&mat, &rhs, &mut x, &cfg()).unwrap();
    assert!(res.converged, "BiCGSTAB did not converge");
    assert!(l2_error(&x, &space) < 6e-3);
}

// ── Non-conforming AMR convergence test ────────────────────────────────────

#[test]
fn poisson_nc_amr_convergence() {
    use fem_mesh::amr::{NCState, zz_estimator, dorfler_mark};
    use fem_space::constraints::{apply_hanging_constraints, recover_hanging_values};
    let mut mesh = SimplexMesh::<2>::unit_square_tri(2);
    let mut nc_state = NCState::new();
    let mut hanging_constraints = Vec::new();
    let mut errors = Vec::new();

    for level in 0..6 {
        let space = H1Space::new(mesh.clone(), 1);
        let n = space.n_dofs();

        let diffusion = DiffusionIntegrator { kappa: 1.0 };
        let source = DomainSourceIntegrator::new(forcing);
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

        apply_hanging_constraints(&mut mat, &mut rhs, &hanging_constraints);

        let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
        apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);

        let mut u = vec![0.0_f64; n];
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg()).unwrap();
        assert!(res.converged, "NC AMR level {level}: solver did not converge");

        recover_hanging_values(&mut u, &hanging_constraints);

        let err = l2_error(&u, &space);
        errors.push(err);

        if level < 5 {
            let eta = zz_estimator(&mesh, &u);
            let marked = dorfler_mark(&eta, 0.5);
            let (new_mesh, new_c, _) = nc_state.refine(&mesh, &marked);
            mesh = new_mesh;
            hanging_constraints = new_c;
        }
    }

    // Verify error decreases monotonically.
    for i in 1..errors.len() {
        assert!(errors[i] < errors[i - 1],
            "L2 error should decrease: level {} err={:.4e} >= level {} err={:.4e}",
            i, errors[i], i - 1, errors[i - 1]);
    }

    // After 5 levels of adaptive refinement starting from 2×2 mesh,
    // the error should be significantly reduced.
    assert!(errors.last().unwrap() < &0.05,
        "NC AMR should achieve < 0.05 L2 error, got {:.4e}", errors.last().unwrap());
}
