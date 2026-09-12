//! D72 regression: the H¹ LOR-AMG preconditioner is full rank, so its reported
//! residual is the true one.
//!
//! The factory used to build the prolongation from a P1 space on the *same*
//! (unrefined) mesh, i.e. `P: P1 → Pk` of size `289×81` for a `P2` space on the
//! `16×16` tri mesh — rank 81.  `M⁻¹ = P·A_LO⁻¹·Pᵀ` was therefore rank
//! deficient: `A_HO`'s residual leaves `range(P)` at once and the preconditioned
//! energy `(M⁻¹r, r)` (linger's CG stopping test, matching MFEM's `CGSolver`)
//! collapses to zero while `‖A_HO x − b‖/‖b‖` is still ≈ 0.585 — PCG reported
//! "converged in 4–6 iterations" with a true residual of 0.6.
//!
//! MFEM's H¹ LOR refines the mesh instead (`lor.cpp`:
//! `Mesh::MakeRefined(mesh_ho, order)`), so the LOR P1 space has exactly as many
//! dofs as the HO space and `P` is square and orthogonal — that is
//! `fem_space::lor::LorH1`, which `build_lor_amg_h1` now uses.  These tests pin
//! both consequences: the preconditioner actually inverts (one application
//! leaves a small residual) and the reported residual equals `‖Ax−b‖/‖b‖` and
//! meets the tolerance.

use fem_assembly::lor_factory::{build_lor_amg_h1, build_lor_amg_h1_3d};
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_solver::lor::{AmgConfig, LorAmgPrecond};
use fem_solver::{solve_pcg_lor_amg, DenseVec, Preconditioner, SolverConfig};
use fem_space::{FESpace, H1Space};

fn true_rel_res(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let n = a.nrows;
    let mut ax = vec![0.0_f64; n];
    a.spmv(x, &mut ax);
    let mut num = 0.0_f64;
    for i in 0..n {
        let d = ax[i] - b[i];
        num += d * d;
    }
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    num.sqrt() / nb
}

fn cfg(rtol: f64) -> SolverConfig {
    SolverConfig {
        rtol,
        atol: 0.0,
        max_iter: 500,
        verbose: false,
        ..SolverConfig::default()
    }
}

/// Two applications of `M⁻¹` must be a solid approximate inverse: with a
/// rank-deficient `P` every image stays inside `range(P)`, so the residual
/// `b − A M⁻¹b` cannot drop below ≈ 0.7 (measured 0.68–0.78 for the old
/// rectangular `P: P1(same mesh) → Pk`), while the LOR `P` reaches ≲ 0.05.
fn assert_preconditioner_inverts<const D: usize>(a_ho: &CsrMatrix<f64>, lor: &LorAmgPrecond) {
    let n = a_ho.nrows;
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();

    // z = M⁻¹b + M⁻¹(b − A M⁻¹b)
    let mut z = DenseVec::zeros(n);
    lor.apply_precond(&DenseVec::from_vec(b.clone()), &mut z);
    let z1: Vec<f64> = z.as_slice().to_vec();
    let mut ax = vec![0.0_f64; n];
    a_ho.spmv(&z1, &mut ax);
    let r1: Vec<f64> = b.iter().zip(ax.iter()).map(|(b, a)| b - a).collect();
    let mut dz = DenseVec::zeros(n);
    lor.apply_precond(&DenseVec::from_vec(r1), &mut dz);
    let zs: Vec<f64> = z1
        .iter()
        .zip(dz.as_slice().iter())
        .map(|(a, b)| a + b)
        .collect();

    let rel = true_rel_res(a_ho, &zs, &b);
    assert!(
        rel < 0.2,
        "{D}D LOR-AMG: ‖b − A M⁻¹b‖/‖b‖ = {rel:.3e} after two preconditioner \
         applications (a rank-deficient prolongation gives ≈ 0.7), \
         ‖M⁻¹b‖ = {:.3e} vs ‖b‖ = {nb:.3e}",
        zs.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
}

fn check_2d(mesh: Mesh<2>, label: &str) -> usize {
    let space = H1Space::new(mesh, 2);
    let a_ho: CsrMatrix<f64> = Assembler::assemble_bilinear(
        &space,
        &[&MassIntegrator { rho: 1.0 }, &DiffusionIntegrator { kappa: 1.0 }],
        4,
    );
    let n = space.n_dofs();
    let b = vec![1.0_f64; n];

    let lor = build_lor_amg_h1(&space, &a_ho, None).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_preconditioner_inverts::<2>(&a_ho, &lor);

    let mut x = vec![0.0_f64; n];
    let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg(1e-10)).expect("PCG must run");
    let rel = true_rel_res(&a_ho, &x, &b);
    println!(
        "{label}: {n} dofs, {} iters, reported {:.3e}, true ‖Ax−b‖/‖b‖ = {rel:.3e}",
        res.iterations, res.final_residual
    );
    assert!(res.converged, "{label}: PCG did not converge: {res:?}");
    assert!(
        (res.final_residual - rel).abs() <= 1e-13 + 1e-3 * rel,
        "{label}: reported {:.6e} vs true {rel:.6e}",
        res.final_residual
    );
    assert!(
        rel <= 1e-10,
        "{label}: the reported convergence must be real: ‖Ax−b‖/‖b‖ = {rel:.3e}"
    );
    res.iterations
}

#[test]
fn d72_lor_amg_h1_reported_residual_is_true_tri() {
    let _ = check_2d(Mesh::<2>::unit_square_tri(2), "tri P2 n=2");
    check_2d(Mesh::<2>::unit_square_tri(4), "tri P2 n=4");
    check_2d(Mesh::<2>::unit_square_tri(8), "tri P2 n=8");
}

#[test]
fn d72_lor_amg_h1_reported_residual_is_true_quad() {
    check_2d(Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0), "quad Q2 4×4");
    check_2d(Mesh::<2>::make_cartesian_2d(8, 8, 1.0, 1.0), "quad Q2 8×8");
}

/// The LOR promise: the iteration count grows slowly under mesh refinement
/// (`plor_solvers -o 2 -rs 0..4`: 17 / 18 / 19 / 21 / 25 iterations).
#[test]
fn d72_lor_amg_h1_iterations_grow_slowly() {
    let coarse = check_2d(Mesh::<2>::unit_square_tri(2), "tri P2 n=2");
    let fine = check_2d(Mesh::<2>::unit_square_tri(8), "tri P2 n=8");
    assert!(
        fine <= coarse + 10,
        "LOR-AMG iterations should not blow up with h: {coarse} → {fine}"
    );
}

#[test]
fn d72_lor_amg_h1_3d_tet_p2_reported_residual_is_true() {
    let space = H1Space::new(Mesh::<3>::unit_cube_tet(2), 2);
    let a_ho: CsrMatrix<f64> = Assembler::assemble_bilinear(
        &space,
        &[&MassIntegrator { rho: 1.0 }, &DiffusionIntegrator { kappa: 1.0 }],
        4,
    );
    let n = space.n_dofs();
    let b = vec![1.0_f64; n];
    let lor = build_lor_amg_h1_3d(&space, &a_ho, Some(AmgConfig::default())).expect("3-D LOR-AMG");
    assert_preconditioner_inverts::<3>(&a_ho, &lor);

    let mut x = vec![0.0_f64; n];
    let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg(1e-10)).expect("PCG must run");
    let rel = true_rel_res(&a_ho, &x, &b);
    println!(
        "tet P2: {n} dofs, {} iters, reported {:.3e}, true ‖Ax−b‖/‖b‖ = {rel:.3e}",
        res.iterations, res.final_residual
    );
    assert!(res.converged, "tet: PCG did not converge: {res:?}");
    assert!((res.final_residual - rel).abs() <= 1e-13 + 1e-3 * rel);
    assert!(rel <= 1e-10, "tet: true residual {rel:.3e}");
}
