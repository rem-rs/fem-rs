//! Backend-agnostic BiCGSTAB entrypoint tests (operator callback based).

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    CsrLinearOperator,
    LinearOperator as AssemblyLinearOperator,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_bicgstab, solve_bicgstab_operator, SolverConfig};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
};

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

fn build_system() -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, H1Space<SimplexMesh<2>>) {
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

#[test]
fn bicgstab_operator_matches_bicgstab() {
    let (mat, rhs, _space) = build_system();

    let n = mat.nrows;
    let mut x_ref = vec![0.0_f64; n];
    let mut x_op = vec![0.0_f64; n];

    let res_ref = solve_bicgstab(&mat, &rhs, &mut x_ref, &cfg()).unwrap();
    assert!(res_ref.converged, "reference solve_bicgstab did not converge");

    let op = CsrLinearOperator::new(&mat);
    let res_op = solve_bicgstab_operator(
        op.nrows(),
        op.ncols(),
        |x, y| op.apply(x, y),
        &rhs,
        &mut x_op,
        &cfg(),
    )
    .unwrap();
    assert!(res_op.converged, "solve_bicgstab_operator did not converge");

    let max_diff = x_ref
        .iter()
        .zip(x_op.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-8,
        "solution mismatch between CSR-BiCGSTAB and operator-BiCGSTAB: {max_diff}"
    );
}
