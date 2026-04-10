//! Backend-agnostic CG entrypoint tests (operator callback based).

use std::f64::consts::PI;

use fem_assembly::{Assembler, CsrLinearOperator, LinearOperator as AssemblyLinearOperator, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{solve_cg, solve_cg_operator, SolverConfig};
use fem_space::{H1Space, constraints::{apply_dirichlet, boundary_dofs}, fe_space::FESpace};

fn u_exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
fn forcing(x: &[f64]) -> f64 { 2.0 * PI * PI * u_exact(x) }

fn cfg() -> SolverConfig {
    SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        ..SolverConfig::default()
    }
}

fn build_poisson_system() -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, H1Space<SimplexMesh<2>>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(12);
    let space = H1Space::new(mesh.clone(), 1);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(forcing);

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
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
        let dofs = space.element_dofs(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x2[0] - x0[0]) * (x1[1] - x0[1])).abs();

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;
            TriP1.eval_basis(xi, &mut phi);

            let uh_q: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| uh[d as usize] * p).sum();
            let xp = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];

            let diff = uh_q - u_exact(&xp);
            err_sq += w * diff * diff;
        }
    }

    err_sq.sqrt()
}

#[test]
fn cg_operator_matches_cg_on_poisson() {
    let (mat, rhs, space) = build_poisson_system();

    let n = mat.nrows;
    let mut x_ref = vec![0.0_f64; n];
    let mut x_op = vec![0.0_f64; n];

    let res_ref = solve_cg(&mat, &rhs, &mut x_ref, &cfg()).unwrap();
    assert!(res_ref.converged, "reference solve_cg did not converge");

    let op = CsrLinearOperator::new(&mat);
    let res_op = solve_cg_operator(op.nrows(), op.ncols(), |x, y| op.apply(x, y), &rhs, &mut x_op, &cfg()).unwrap();
    assert!(res_op.converged, "solve_cg_operator did not converge");

    let max_diff = x_ref
        .iter()
        .zip(x_op.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-8, "solution mismatch between CSR-CG and operator-CG: {max_diff}");

    let err = l2_error(&x_op, &space);
    assert!(err < 1.2e-2, "operator-CG L2 error too large: {err}");
}
