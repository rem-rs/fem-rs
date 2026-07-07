//! Backend-agnostic GMRES entrypoint tests (operator callback based).

use std::f64::consts::PI;

use fem_assembly::{Assembler, CsrLinearOperator, LinearOperator as AssemblyLinearOperator, standard::{ConvectionIntegrator, DiffusionIntegrator, DomainSourceIntegrator}};
use fem_assembly::postproc::coefficient::ConstantVectorCoeff;
use fem_mesh::Mesh;
use fem_solver::{solve_fgmres_ilu0, solve_fgmres_ilut, solve_gmres, solve_gmres_ilu0, solve_gmres_iluk, solve_gmres_ilut, solve_gmres_operator, solve_precond_kind, PrecondKind, SolverConfig};
use fem_space::{H1Space, constraints::{apply_dirichlet, boundary_dofs}};

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

fn build_nonsym_system() -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, H1Space<Mesh<2>>) {
    let mesh = Mesh::<2>::unit_square_tri(12);
    let space = H1Space::new(mesh.clone(), 1);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let convection = ConvectionIntegrator {
        velocity: ConstantVectorCoeff(vec![1.0, 0.25]),
    };
    let source = DomainSourceIntegrator::new(forcing);

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion, &convection], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);

    (mat, rhs, space)
}

fn assert_matches_gmres_reference<F>(label: &str, solve_with_precond: F)
where
    F: Fn(&fem_linalg::CsrMatrix<f64>, &[f64], &mut [f64]) -> fem_solver::SolveResult,
{
    let (mat, rhs, _space) = build_nonsym_system();
    let n = mat.nrows;

    let mut x_ref = vec![0.0_f64; n];
    let res_ref = solve_gmres(&mat, &rhs, &mut x_ref, 30, &cfg()).unwrap();
    assert!(res_ref.converged, "reference GMRES did not converge for {label}");

    let mut x = vec![0.0_f64; n];
    let res = solve_with_precond(&mat, &rhs, &mut x);
    assert!(res.converged, "{label} did not converge on convection-diffusion system");

    let max_diff = x_ref
        .iter()
        .zip(x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-7, "{label} solution mismatch vs GMRES reference: {max_diff}");
}

#[test]
fn gmres_operator_matches_gmres_on_convection_diffusion() {
    let (mat, rhs, _space) = build_nonsym_system();

    let n = mat.nrows;
    let mut x_ref = vec![0.0_f64; n];
    let mut x_op = vec![0.0_f64; n];

    let restart = 30;
    let res_ref = solve_gmres(&mat, &rhs, &mut x_ref, restart, &cfg()).unwrap();
    assert!(res_ref.converged, "reference solve_gmres did not converge");

    let op = CsrLinearOperator::new(&mat);
    let res_op = solve_gmres_operator(op.nrows(), op.ncols(), |x, y| op.apply(x, y), &rhs, &mut x_op, restart, &cfg()).unwrap();
    assert!(res_op.converged, "solve_gmres_operator did not converge");

    let max_diff = x_ref
        .iter()
        .zip(x_op.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-8, "solution mismatch between CSR-GMRES and operator-GMRES: {max_diff}");
}

#[test]
fn gmres_ilu0_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("GMRES+ILU0", |mat, rhs, x| {
        solve_gmres_ilu0(mat, rhs, x, 30, &cfg()).unwrap()
    });
}

#[test]
fn gmres_iluk_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("GMRES+ILU(k=1)", |mat, rhs, x| {
        solve_gmres_iluk(mat, rhs, x, 30, 1, &cfg()).unwrap()
    });
}

#[test]
fn gmres_ilut_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("GMRES+ILUT", |mat, rhs, x| {
        solve_gmres_ilut(mat, rhs, x, 30, 1e-3, 15, &cfg()).unwrap()
    });
}

#[test]
fn fgmres_ilu0_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("FGMRES+ILU0", |mat, rhs, x| {
        solve_fgmres_ilu0(mat, rhs, x, 30, &cfg()).unwrap()
    });
}

#[test]
fn fgmres_ilut_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("FGMRES+ILUT", |mat, rhs, x| {
        solve_fgmres_ilut(mat, rhs, x, 30, 1e-3, 15, &cfg()).unwrap()
    });
}

#[test]
fn precond_kind_ilu0_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("solve_precond_kind(Ilu0)", |mat, rhs, x| {
        solve_precond_kind(mat, rhs, x, 30, PrecondKind::Ilu0, &cfg()).unwrap()
    });
}

#[test]
fn precond_kind_iluk_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("solve_precond_kind(Iluk(1))", |mat, rhs, x| {
        solve_precond_kind(mat, rhs, x, 30, PrecondKind::Iluk(1), &cfg()).unwrap()
    });
}

#[test]
fn precond_kind_ilut_converges_on_convection_diffusion() {
    assert_matches_gmres_reference("solve_precond_kind(Ilut)", |mat, rhs, x| {
        solve_precond_kind(mat, rhs, x, 30, PrecondKind::Ilut { tau: 1e-3, fill: 15 }, &cfg()).unwrap()
    });
}
