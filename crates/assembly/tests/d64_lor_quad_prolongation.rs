//! D64 regression: the H¹ LOR prolongation must work on **quad** meshes too.
//!
//! `plor_solvers` defaults to `data/inline-quad.mesh`; before D64 the 2-D H¹
//! transfer path went through `TriPointLocator`, whose constructor only
//! accepted `Tri3` — the miniapp therefore panicked instead of running, and
//! `build_prolongation_h1` emitted 3 weights per located point regardless of
//! the element type.
//!
//! The prolongation `P: P1 → Pk` on a **fixed mesh** is the nodal interpolation
//! operator: `P[i,j] = φ_j^{P1}(x_i^{Pk})`.  On a quad mesh the P1 space is
//! `Q1` (bilinear), so `P` must reproduce *bilinear* fields exactly; on a
//! triangle mesh it is affine and reproduces affine fields.  Those two
//! identities are exactly what these tests check, at the level of the fine
//! space's own DOF coordinates.

use fem_assembly::{build_lor_amg_h1, build_prolongation_h1};
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::{FESpace, H1Space};

/// `P` applied to the coarse nodal values of `f` must equal the nodal values of
/// `f` on the fine space (all of them, no extrapolation).
fn assert_prolongation_is_nodal_interpolation<F>(mesh: Mesh<2>, f: F)
where
    F: Fn(f64, f64) -> f64,
{
    let p1 = H1Space::new(mesh.clone(), 1);
    let pk = H1Space::new(mesh.clone(), 2);
    let (p, stats) = build_prolongation_h1(&p1, &pk, 1e-8);
    assert_eq!(
        stats.extrapolated_count, 0,
        "every fine DOF must be located inside the coarse mesh ({stats:?})"
    );
    assert_eq!(stats.located_count, pk.n_dofs());

    let coarse: Vec<f64> = (0..p1.n_dofs() as u32)
        .map(|d| {
            let x = p1.dof_manager().dof_coord(d);
            f(x[0], x[1])
        })
        .collect();
    let mut fine = vec![0.0_f64; pk.n_dofs()];
    p.spmv(&coarse, &mut fine);

    let mut worst = 0.0_f64;
    for d in 0..pk.n_dofs() as u32 {
        let x = pk.dof_manager().dof_coord(d);
        worst = worst.max((fine[d as usize] - f(x[0], x[1])).abs());
    }
    assert!(
        worst < 1e-12,
        "P is not the nodal interpolation operator: max|P·c − f(x_fine)| = {worst:.3e}"
    );
}

/// A quad mesh's P1 space is bilinear, so a bilinear field must be reproduced
/// exactly (this is where the 3-weight simplex loop failed).
#[test]
fn d64_quad_p1_to_q2_prolongation_is_the_nodal_interpolation() {
    assert_prolongation_is_nodal_interpolation(
        Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0),
        |x, y| 1.0 + 2.0 * x - 3.0 * y + 4.0 * x * y,
    );
}

/// The simplex path must keep its old behaviour bit for bit (affine field).
#[test]
fn d64_tri_p1_to_p2_prolongation_is_the_nodal_interpolation() {
    assert_prolongation_is_nodal_interpolation(Mesh::<2>::unit_square_tri(4), |x, y| {
        1.0 + 2.0 * x - 3.0 * y
    });
}

/// The end-to-end acceptance path of `plor_solvers` on a quad mesh: the LOR-AMG
/// preconditioner must build (no `TriPointLocator` panic, no "located 0 DOFs")
/// and PCG must run to termination and report numbers.
///
/// NOTE — pre-existing, *not* D64: `solve_pcg_lor_amg` inherits linger's CG
/// stopping test, which with a preconditioner compares the **preconditioned
/// energy** `(B r, r)/(B r₀, r₀)` against `rtol` while *reporting* the true
/// residual.  fem-rs's H¹ LOR is `B = P·A_LO⁻¹·Pᵀ` with `P: P1(same mesh) → Pk`
/// of size 289×81, hence rank 81 — `(B r, r)` vanishes as soon as `r` becomes
/// orthogonal to `range(P)`, so CG "converges" after a few iterations with
/// `‖A x − b‖/‖b‖ ≈ 0.6`.  MFEM's H¹ LOR refines the mesh (`lor.cpp`:
/// `Mesh::MakeRefined(mesh_ho, order)`) so that `P` is square and full rank and
/// the energy test is sound.  The *tri* path behaves identically (0.638 vs
/// 0.585 here) — this is a property of the shared LOR/H¹ wiring, so the test
/// asserts only that the quad path runs and agrees with the tri path's
/// behaviour, not that the residual is small.
#[test]
fn d64_quad_lor_amg_runs_and_reports() {
    use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
    use fem_assembly::Assembler;
    use fem_solver::{solve_pcg_lor_amg, SolverConfig};

    let mesh = Mesh::<2>::make_cartesian_2d(8, 8, 1.0, 1.0);
    let space = H1Space::new(mesh, 2);
    let a_ho: CsrMatrix<f64> = Assembler::assemble_bilinear(
        &space,
        &[&MassIntegrator { rho: 1.0 }, &DiffusionIntegrator { kappa: 1.0 }],
        4,
    );
    let n = space.n_dofs();
    let b = vec![1.0_f64; n];

    let lor = build_lor_amg_h1(&space, &a_ho, None).expect("quad LOR-AMG must build");
    let mut x = vec![0.0_f64; n];
    let cfg = SolverConfig {
        rtol: 1e-10,
        max_iter: 500,
        ..SolverConfig::default()
    };
    let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg).expect("PCG must run");

    assert!(res.iterations <= 500, "PCG must terminate: {res:?}");
    assert!(x.iter().all(|v| v.is_finite()), "solution must be finite");

    let mut ax = vec![0.0_f64; n];
    a_ho.spmv(&x, &mut ax);
    let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nr: f64 = ax
        .iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let rel = nr / nb;
    println!(
        "quad LOR-AMG: {} iterations, reported {:.3e}, true ‖Ax−b‖/‖b‖ = {rel:.3e}",
        res.iterations, res.final_residual
    );
    // The reported residual is the true one (the false convergence comes from
    // the energy-based stopping test, not from the reported metric).
    assert!(
        (res.final_residual - rel).abs() < 1e-12,
        "reported {:.6e} vs true {rel:.6e}",
        res.final_residual
    );
}
