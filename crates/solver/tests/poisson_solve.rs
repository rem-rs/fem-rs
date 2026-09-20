//! End-to-end integration tests: assemble → solve with iterative solvers.
//!
//! We assemble the Poisson system on a 16×16 P1 mesh (from fem-assembly)
//! and solve it with each solver from fem-solver, verifying L2 accuracy.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::{topology::MeshTopology, Mesh};
use fem_solver::{solve_bicgstab, solve_cg, solve_gmres, solve_gmres_ilu0, solve_pcg_ilu0, solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn u_exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
fn forcing(x: &[f64]) -> f64 { 2.0 * PI * PI * u_exact(x) }

/// Assemble the Poisson system on a 16×16 P1 unit-square mesh.
/// Returns (mat, rhs, space).
fn build_poisson_system() -> (fem_linalg::CsrMatrix<f64>, Vec<f64>, H1Space<Mesh<2>>) {
    let mesh  = Mesh::<2>::unit_square_tri(16);
    let space = H1Space::new(mesh.clone(), 1);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source    = DomainSourceIntegrator::new(forcing);
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);
    let bdofs  = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut mat, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
    (mat, rhs, space)
}

fn l2_error(uh: &[f64], space: &H1Space<Mesh<2>>) -> f64 {
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
fn poisson_gmres_ilu0() {
    let (mat, rhs, space) = build_poisson_system();
    let n = mat.nrows;
    let mut x = vec![0.0_f64; n];
    let res = solve_gmres_ilu0(&mat, &rhs, &mut x, 30, &cfg()).unwrap();
    assert!(res.converged, "GMRES-ILU0 did not converge");
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

// ── Non-conforming AMR convergence tests ───────────────────────────────────
//
// D404 MFEM 4.10 reference trajectory (`$HOME/work/d404/d404_ref.cpp`, copy of
// the D403 probe; Poisson MMS identical to this file, 2×2-tri unit square,
// P1 H1, MFEM `ZienkiewiczZhuEstimator` + `ThresholdRefiner` fraction 0.5,
// 5 refinement rounds), reproduced bit-identically on 2026-09-20:
//
//   mode "linf" — ThresholdRefiner default `total_norm_p=∞`
//   (threshold = 0.5·‖η‖_inf, MFEM's canonical fraction marking):
//     level 0: ne=8    ndof=9    l2=2.461762e-01
//     level 1: ne=32   ndof=25   l2=7.871430e-02
//     level 2: ne=128  ndof=81   l2=2.110699e-02
//     level 3: ne=428  ndof=241  l2=5.916769e-03
//     level 4: ne=1624 ndof=865  l2=1.585936e-03
//     level 5: ne=6032 ndof=3115 l2=4.271716e-04
//
//   mode "p2" — `SetTotalErrorNormP(2)` (threshold = 0.5·RMS(η)):
//     level 0: ne=8    ndof=9    l2=2.461762e-01
//     level 1: ne=32   ndof=25   l2=7.871430e-02
//     level 2: ne=128  ndof=81   l2=2.110699e-02
//     level 3: ne=508  ndof=285  l2=5.417776e-03
//     level 4: ne=1988 ndof=1053 l2=1.372536e-03
//     level 5: ne=7768 ndof=3997 l2=3.506350e-04
//
// D404 findings (full combo table in tmp/d404/ and tests/d404_probe.rs):
//   1. Marking rule dominates: the Dörfler(0.5) *prefix* criterion marks only
//      4–15 elements/round on the skewed NC-AMR indicator distributions
//      (top spikes soak up the 50% energy budget), stalling at ne=125 /
//      L2 3.2e-2 — 75× off MFEM.  MFEM's `total_norm_p=2` *threshold* rule
//      (η > 0.5·RMS, [`fem_assembly::postproc::error_estimate::
//      ElementIndicators::rms_mark`]) marks 8–87/round and lands in MFEM's
//      ballpark.  The earlier claim in `dorfler_mark`'s doc that Dörfler ≡
//      MFEM p2 marking held only for flat distributions (D489, doc fixed).
//   2. Estimator: the L²-projection ZZ recovery is run WITHOUT the
//      hanging-node constraints here (D490): the constrained recovery space
//      cannot fit the one-sided flux jumps at NC interfaces and inflates η
//      there (effectivity ‖η‖₂/‖e‖ true: 4.0→31.3 with constraints vs
//      4.0→12.4 without; final L2 3.24e-3 vs 4.89e-4 under RMS marking).
//      MFEM's own recovery (nodal flux averaging, no projection) imposes no
//      such constraint, so the unconstrained projection is the closer
//      analogue.  MFEM-replica cross-check: fem-rs `zz_estimator_mfem_nc` +
//      RMS marking reaches 3.72e-4 at ne=7340 vs MFEM p2 3.51e-4 at ne=7768.
//   3. Per-round ne values differ from MFEM (fem-rs refines marked triangles
//      red 1→4 with hanging-node closure; MFEM NCMesh bisects simplices), so
//      only the growth pattern and the final accuracy are comparable.

/// Estimator selector for the NC AMR driver (D404): the legacy vertex-average
/// (centroid) ZZ path is kept alongside the L²-projection ZZ so both marking
/// trajectories stay comparable.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AmrEstimator {
    /// Legacy centroid (vertex-average) ZZ — the D73a baseline path.
    CentroidZz,
    /// L²-projection ZZ ([`zz_estimator_l2_nc`], MFEM-compatible recovery),
    /// unconstrained variant + RMS threshold marking — the D404 primary path.
    L2ProjectionZz,
}

/// Shared NC AMR loop: solve → L2 error → estimate → mark → NC refine, 5
/// refinement rounds from the 2×2-triangle mesh.  The centroid path marks
/// with the D73a Dörfler(0.5) prefix; the L²-projection path marks with the
/// MFEM-style RMS threshold.  Returns the per-level `(ne, L2 error)`
/// trajectory.
fn run_nc_amr_loop(est: AmrEstimator) -> Vec<(usize, f64)> {
    use fem_assembly::postproc::error_estimate::{zz_estimator, zz_estimator_l2_nc};
    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_mesh::amr::NCState;
    use fem_space::constraints::{apply_hanging_constraints, recover_hanging_values};
    let mut mesh = Mesh::<2>::unit_square_tri(2);
    let mut nc_state = NCState::new();
    let mut hanging_constraints = Vec::new();
    let mut traj = Vec::new();

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
        println!("  level {level}: ne={} ndof={n} l2={err:.6e}", mesh.n_elements());
        traj.push((mesh.n_elements(), err));

        if level < 5 {
            let gf = GridFunction::new(&space, u.clone());
            let marked = match est {
                AmrEstimator::CentroidZz => zz_estimator(&gf).dorfler_mark(0.5),
                // D404: L²-projection ZZ recovery WITHOUT the hanging-node
                // constraints (the constrained recovery space inflates η at
                // NC interfaces, see the D490 note in the file header), and
                // MFEM `ThresholdRefiner`-style RMS threshold marking
                // (η > 0.5·RMS — the `total_norm_p=2` analogue, D489).
                AmrEstimator::L2ProjectionZz => {
                    zz_estimator_l2_nc(&gf, &[]).rms_mark(0.5)
                }
            };
            let (new_mesh, new_c, _) = nc_state.refine(&mesh, &marked, 0);
            mesh = new_mesh;
            hanging_constraints = new_c;
        }
    }
    traj
}

/// NC AMR convergence test using the L²-projection ZZ estimator (D404).
///
/// Marking is MFEM `ThresholdRefiner`-style RMS thresholding
/// (`rms_mark(0.5)`); see the file-header notes for why Dörfler(0.5)
/// prefix marking stalls the NC loop.  Cross-validated against MFEM 4.10
/// (reference trajectory in the file header): the final-level L2 error must
/// land at MFEM's order of magnitude (MFEM linf 4.272e-4, p2 3.506e-4) and
/// the ne trajectory must grow at MFEM's per-round pace.
#[test]
fn poisson_nc_amr_convergence() {
    let traj = run_nc_amr_loop(AmrEstimator::L2ProjectionZz);
    let errors: Vec<f64> = traj.iter().map(|&(_, e)| e).collect();

    // Verify error decreases monotonically.
    for i in 1..errors.len() {
        assert!(errors[i] < errors[i - 1],
            "L2 error should decrease: level {} err={:.4e} >= level {} err={:.4e}",
            i, errors[i], i - 1, errors[i - 1]);
    }

    // Threshold semantics preserved (D404): after 5 levels of adaptive
    // refinement starting from 2×2 mesh, the error should be significantly
    // reduced.
    assert!(errors.last().unwrap() < &0.05,
        "NC AMR should achieve < 0.05 L2 error, got {:.4e}", errors.last().unwrap());

    // D404 trajectory anchor (measured on this tree, L²-projection ZZ with
    // unconstrained recovery + rms_mark(0.5)); MFEM 4.10 references reach
    // 4.271716e-4 (linf, ne=6032) / 3.506350e-4 (p2, ne=7768).
    let d404 = [2.498915e-1, 7.909111e-2, 2.311638e-2, 6.987450e-3,
                1.874436e-3, 4.892040e-4];
    assert_eq!(errors.len(), d404.len(), "D404: expected 6 levels");
    for (i, (&e, &r)) in errors.iter().zip(d404.iter()).enumerate() {
        assert!((e - r).abs() <= 1e-3 * r.abs(),
            "D404 L2-ZZ RMS-marked trajectory changed at level {i}: got {e:.6e}, anchor {r:.6e}");
    }
    let ne: Vec<usize> = traj.iter().map(|&(n, _)| n).collect();
    assert_eq!(ne, vec![8, 32, 116, 440, 1676, 6452],
        "D404: ne trajectory changed (MFEM: 8→32→128→428/508→1624/1988→6032/7768)");
}

/// D73a baseline protection: the legacy centroid-ZZ NC AMR path keeps its
/// measured trajectory as a regression anchor (round 48 numbers, unchanged
/// by D404's estimator switch).
#[test]
fn poisson_nc_amr_convergence_centroid_zz() {
    let traj = run_nc_amr_loop(AmrEstimator::CentroidZz);
    let errors: Vec<f64> = traj.iter().map(|&(_, e)| e).collect();

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

    // D73a (round 48) measured trajectory anchor: ne 8→17→26→38→65→95,
    // L2 2.4989e-1→2.4597e-1→2.3384e-1→7.9090e-2→6.1045e-2→4.1594e-2.
    let ne: Vec<usize> = traj.iter().map(|&(n, _)| n).collect();
    assert_eq!(ne, vec![8, 17, 26, 38, 65, 95],
        "D73a centroid-ZZ ne trajectory changed");
    let d73a = [2.4989e-1, 2.4597e-1, 2.3384e-1, 7.9090e-2, 6.1045e-2, 4.1594e-2];
    for (i, (&e, &r)) in errors.iter().zip(d73a.iter()).enumerate() {
        assert!((e - r).abs() <= 1e-3 * r.abs(),
            "D73a centroid-ZZ L2 trajectory changed at level {i}: got {e:.6e}, anchor {r:.6e}");
    }
}
