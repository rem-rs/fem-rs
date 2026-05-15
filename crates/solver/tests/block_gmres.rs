use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::ConstantVectorCoeff,
    standard::{ConvectionIntegrator, DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_block_gmres, solve_gmres, BlockGmresConfig, SolverConfig};
use fem_space::{H1Space, constraints::{apply_dirichlet, boundary_dofs}};

fn forcing_sin(x: &[f64]) -> f64 {
    2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn forcing_poly(x: &[f64]) -> f64 {
    1.0 + x[0] + 0.5 * x[1]
}

fn gmres_cfg() -> SolverConfig {
    SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        ..SolverConfig::default()
    }
}

fn block_cfg() -> BlockGmresConfig {
    BlockGmresConfig {
        base: gmres_cfg(),
        restart: 30,
    }
}

fn build_nonsym_matrix_and_rhs_pair() -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(12);
    let space = H1Space::new(mesh.clone(), 1);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let convection = ConvectionIntegrator {
        velocity: ConstantVectorCoeff(vec![1.0, 0.25]),
    };

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion, &convection], 3);
    let source_1 = DomainSourceIntegrator::new(forcing_sin);
    let source_2 = DomainSourceIntegrator::new(forcing_poly);
    let mut rhs_1 = Assembler::assemble_linear(&space, &[&source_1], 3);
    let mut rhs_2 = Assembler::assemble_linear(&space, &[&source_2], 3);

    let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat, &mut rhs_1, &bdofs, &vec![0.0; bdofs.len()]);
    apply_dirichlet(&mut mat, &mut rhs_2, &bdofs, &vec![0.0; bdofs.len()]);

    (mat, rhs_1, rhs_2)
}

#[test]
fn block_gmres_matches_individual_gmres_on_convection_diffusion_rhs_pair() {
    let (mat, rhs_1, rhs_2) = build_nonsym_matrix_and_rhs_pair();
    let n = mat.nrows;

    let mut rhs_block = vec![0.0_f64; 2 * n];
    rhs_block[..n].copy_from_slice(&rhs_1);
    rhs_block[n..].copy_from_slice(&rhs_2);

    let mut x_block = vec![0.0_f64; 2 * n];
    let res_block = solve_block_gmres(&mat, &rhs_block, &mut x_block, &block_cfg()).unwrap();
    assert!(res_block.converged, "Block-GMRES did not converge on convection-diffusion RHS pair");

    let mut x_ref_1 = vec![0.0_f64; n];
    let mut x_ref_2 = vec![0.0_f64; n];
    let res_ref_1 = solve_gmres(&mat, &rhs_1, &mut x_ref_1, 30, &gmres_cfg()).unwrap();
    let res_ref_2 = solve_gmres(&mat, &rhs_2, &mut x_ref_2, 30, &gmres_cfg()).unwrap();
    assert!(res_ref_1.converged, "reference GMRES did not converge for RHS 1");
    assert!(res_ref_2.converged, "reference GMRES did not converge for RHS 2");

    let block_1 = &x_block[..n];
    let block_2 = &x_block[n..];

    let max_diff_1 = x_ref_1
        .iter()
        .zip(block_1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_diff_2 = x_ref_2
        .iter()
        .zip(block_2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(max_diff_1 < 1e-7, "Block-GMRES RHS 1 mismatch vs GMRES reference: {max_diff_1}");
    assert!(max_diff_2 < 1e-7, "Block-GMRES RHS 2 mismatch vs GMRES reference: {max_diff_2}");
}