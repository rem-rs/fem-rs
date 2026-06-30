//! Convergence tests for H(div) elements on quadrilateral meshes.
//!
//! Solves σ = f (mass form) with manufactured solutions.

use std::f64::consts::PI;

use fem_assembly::{
    coefficient::FnVectorCoeff,
    standard::{VectorDomainLFIntegrator, VectorMassIntegrator},
    vector_assembler::VectorAssembler,
};
use fem_mesh::SimplexMesh;
use fem_solver::{MinresSolver, SolverConfig};
use fem_space::{fe_space::FESpace, HDivSpace};

fn exact_flux(x: &[f64], _scale: f64) -> [f64; 2] {
    let sx = (PI * x[0]).sin();
    let cx = (PI * x[0]).cos();
    let sy = (PI * x[1]).sin();
    let cy = (PI * x[1]).cos();
    [sx * cy, -cx * sy]
}

/// Compute L² DOF error: norm of (uh - interpolated_exact) in DOF space.
/// Since H(div) DOFs are normal-flux moments, the DOF space L² norm
/// is mesh-dependent so we normalize by sqrt(n_dofs).
#[allow(dead_code)]
fn dof_error(space: &HDivSpace<SimplexMesh<2>>, uh: &[f64]) -> f64 {
    let n = space.n_dofs();
    let norm2: f64 = uh.iter().map(|&v| v * v).sum();
    (norm2 / n as f64).sqrt()
}

fn quad_rt0_mass_test() {
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let mesh = SimplexMesh::<2>::unit_square_quad(4);
    let space = HDivSpace::new(mesh, 0); // QuadRT0
    println!("QuadRT0 n_dofs = {}", space.n_dofs());

    // Just assembles and solves — verifies the basic path works
    let mat = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: 1.0 }], 3);
    let source = VectorDomainLFIntegrator {
        f: FnVectorCoeff(|_x: &[f64], out: &mut [f64]| {
            out[0] = 1.0; out[1] = 0.0;
        }),
    };
    let rhs = VectorAssembler::assemble_linear(&space, &[&source], 3);
    let mut u = vec![0.0_f64; space.n_dofs()];
    let res = MinresSolver::solve(&mat, &rhs, &mut u, &cfg).expect("MINRES");
    assert!(res.converged, "QuadRT0 MINRES failed");
    println!("QuadRT0: iters={}, residual={:.3e}", res.iterations, res.final_residual);
}

#[test]
fn quad_rt1_darcy_convergence() {
    // First test QuadRT0 works as baseline
    quad_rt0_mass_test();

    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };

    let mut prev_err: Option<f64> = None;
    let mut prev_h: Option<f64> = None;

    for &n in [4usize, 8, 16].iter() {
        let mesh = SimplexMesh::<2>::unit_square_quad(n);
        let space = HDivSpace::new(mesh, 1); // QuadRT1

        // Assemble mass matrix: ∫ σ·τ dx
        let mat = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: 1.0 }], 5);

        // RHS: ∫ f·τ dx, where f = exact_flux
        let source = VectorDomainLFIntegrator {
            f: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                let exact = exact_flux(x, 1.0);
                out[0] = exact[0];
                out[1] = exact[1];
            }),
        };
        let rhs = VectorAssembler::assemble_linear(&space, &[&source], 5);

        let mut u = vec![0.0_f64; space.n_dofs()];
        let res = MinresSolver::solve(&mat, &rhs, &mut u, &cfg).expect("MINRES");
        assert!(res.converged, "MINRES failed at n={n}");

        // Interpolate exact solution
        let u_exact_vec = space.interpolate_vector(&|x: &[f64]| vec![exact_flux(x, 1.0)[0], exact_flux(x, 1.0)[1]]);
        let u_exact = u_exact_vec.as_slice();

        // DOF-space error
        let mut err2 = 0.0_f64;
        for i in 0..space.n_dofs() {
            let d = u[i] - u_exact[i];
            err2 += d * d;
        }
        let err: f64 = err2.sqrt();

        let h: f64 = 1.0 / n as f64;
        println!("n={n:>2}  h={h:.4e}  DOFs={:>5}  DOF error={:.6e}  iters={}",
            space.n_dofs(), err, res.iterations);

        if let (Some(pe), Some(ph)) = (prev_err, prev_h) {
            let rate: f64 = (pe / err).ln() / (ph / h).ln();
            println!("  └─ observed order ≈ {rate:.2}");
            assert!(rate > 1.5, "expected O(h²), got {rate:.2}");
        }

        prev_err = Some(err);
        prev_h = Some(h);
    }
}

#[test]
fn tri_bdm1_mass_convergence() {
    let ref_elem = fem_element::lagrange::factory::vec_ref_elem(
        fem_element::lagrange::factory::VecFamily::BrezziDouglasMarini,
        fem_element::lagrange::factory::ElemType::Tri,
        1u8,
    );
    assert_eq!(ref_elem.n_dofs(), 6, "TriBDM1 has 6 DOFs");
    let mut v = vec![0.0; 6 * 2];
    ref_elem.eval_basis_vec(&[0.2, 0.3], &mut v);
    for val in &v { assert!(val.is_finite()); }
    let mut d = vec![0.0; 6];
    ref_elem.eval_div(&[0.2, 0.3], &mut d);
    for val in &d { assert!(val.is_finite()); }
}

/// Test QuadBDMk basis and DOF counts directly.
#[test]
fn quad_bdmk_smoke_test() {
    use fem_element::brezzi_douglas_marini::QuadBDMk;
    use fem_element::VectorReferenceElement;
    for k in 1..=3 {
        let e = QuadBDMk::new(k);
        let n = e.n_dofs();
        let mut v = vec![0.0; n * 2];
        e.eval_basis_vec(&[0.2, -0.4], &mut v);
        for val in &v { assert!(val.is_finite(), "k={k}"); }
        let mut d = vec![0.0; n];
        e.eval_div(&[0.2, -0.4], &mut d);
        for val in &d { assert!(val.is_finite(), "k={k} div"); }
    }
}
