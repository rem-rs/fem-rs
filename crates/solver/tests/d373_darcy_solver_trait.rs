//! D373: the [`DarcySolver`] trait unifies the public surface MFEM's
//! `blocksolvers::DarcySolver` base gives its three derived solvers
//! (`Mult`, `GetNumIterations`, `offsets_ = {0, size0, height}`).
//!
//! This test is the trait-object call point: all three serial solvers
//! (`BdpMinresSolver`, `BramblePasciakSolver`, `DivFreeSolver`) are built for
//! the same small saddle system, held as `Box<dyn DarcySolver>`, and driven
//! exclusively through dynamic dispatch.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::bramble_pasciak::{BpsParameters, BramblePasciakSolver};
use fem_solver::darcy_solvers::{BdpMinresSolver, DarcySolver, IterSolveParameters, SchurMode};
use fem_solver::div_free_solver::{DfsData, DfsParameters, DivFreeSolver};

/// The `saddle_data` stand-in of the crate tests: M = tridiag(-0.5, 2, -0.5),
/// B = dense patterned `(n_p × n_u)` — sized so `M − Q` stays SPD for
/// `Q = 0.5·diag(M)` (the Bramble–Pasciak requirement).
fn saddle_data(n_u: usize, n_p: usize) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let mut coo_m = CooMatrix::<f64>::new(n_u, n_u);
    for i in 0..n_u {
        coo_m.add(i, i, 2.0);
        if i > 0 {
            coo_m.add(i, i - 1, -0.5);
            coo_m.add(i - 1, i, -0.5);
        }
    }
    let mut coo_b = CooMatrix::<f64>::new(n_p, n_u);
    for r in 0..n_p {
        for c in 0..n_u {
            let v = ((r + 1) * (c + 1) % 5) as f64 * 0.1 + 0.01 * (r as f64 + c as f64);
            if v != 0.0 {
                coo_b.add(r, c, v);
            }
        }
    }
    (coo_m.into_csr(), coo_b.into_csr())
}

/// Flat saddle operator `[[M, Bᵀ], [B, 0]]` action (test reference).
fn apply_a_ref(m: &CsrMatrix<f64>, b: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    let n_u = m.nrows;
    m.spmv(&x[..n_u], &mut y[..n_u]);
    let mut t = vec![0.0_f64; n_u];
    b.transpose().spmv(&x[n_u..], &mut t);
    for (yi, ti) in y[..n_u].iter_mut().zip(&t) {
        *yi += *ti;
    }
    b.spmv(&x[..n_u], &mut y[n_u..]);
}

#[test]
fn d373_all_three_solvers_behind_dyn_darcy_solver() {
    let n_u = 6;
    let n_p = 3;
    let n = n_u + n_p;
    let (m, b) = saddle_data(n_u, n_p);

    let x_star: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64 * 0.5 - 0.2).collect();
    let mut rhs = vec![0.0_f64; n];
    apply_a_ref(&m, &b, &x_star, &mut rhs);

    let param = IterSolveParameters {
        print_level: -1,
        max_iter: 200,
        abs_tol: 1e-14,
        rel_tol: 1e-10,
    };

    let bdp = BdpMinresSolver::new(&m, &b, param.clone(), SchurMode::Dense);
    let bp = {
        let mut coo_q = CooMatrix::<f64>::new(n_u, n_u);
        for i in 0..n_u {
            coo_q.add(i, i, 0.5 * m.get(i, i));
        }
        let q = coo_q.into_csr();
        BramblePasciakSolver::new(
            &m,
            &b,
            &q,
            BpsParameters {
                iter: param,
                use_bpcg: true,
                q_scaling: 0.5,
            },
            SchurMode::Dense,
        )
    };
    let dfs = DivFreeSolver::new(
        &m,
        &b,
        &DfsData::single_level(DfsParameters {
            coarse_schur_mode: SchurMode::Dense,
            ..DfsParameters::default()
        }),
    );

    // The trait-object point: heterogeneous collection, uniform dispatch.
    let solvers: [Box<dyn DarcySolver>; 3] = [Box::new(bdp), Box::new(bp), Box::new(dfs)];
    for (k, solver) in solvers.iter().enumerate() {
        assert_eq!(solver.offsets(), [0, n_u, n], "solver {k}: offsets");
        assert_eq!(solver.size(), n, "solver {k}: height");
        let mut x = vec![0.0; solver.size()];
        solver.mult(&rhs, &mut x);
        assert!(solver.num_iterations() < 200, "solver {k}: iterations");
        let err = x
            .iter()
            .zip(&x_star)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err < 1e-6,
            "solver {k}: solution error {err:.3e} (x = {x:?})"
        );
    }
}

/// One full solve through the **inherent** methods (concrete-type method
/// resolution takes the inherent path) — `(solution, iterations, offsets,
/// converged)`.
fn inherent_bits<S: DarcySolver>(solver: &S, rhs: &[f64]) -> (Vec<f64>, usize, [usize; 3], bool) {
    let mut x = vec![0.0; rhs.len()];
    solver.mult(rhs, &mut x);
    (x, solver.num_iterations(), solver.offsets(), solver.converged())
}

#[test]
fn d373_trait_matches_inherent_results_bitwise() {
    // D580: the trait/inherent bit-consistency assertion covers ALL THREE
    // solvers (the block_solvers driver arms: BDP / BP / DFS), since the
    // driver now holds each of them as `Box<dyn DarcySolver>`.
    let (m, b) = saddle_data(6, 3);
    let rhs = vec![1.0; 9];

    let bdp = BdpMinresSolver::new(&m, &b, IterSolveParameters::default(), SchurMode::Dense);
    let bp = {
        let mut coo_q = CooMatrix::<f64>::new(6, 6);
        for i in 0..6 {
            coo_q.add(i, i, 0.5 * m.get(i, i));
        }
        let q = coo_q.into_csr();
        BramblePasciakSolver::new(
            &m,
            &b,
            &q,
            BpsParameters {
                iter: IterSolveParameters::default(),
                use_bpcg: true,
                q_scaling: 0.5,
            },
            SchurMode::Dense,
        )
    };
    let dfs = DivFreeSolver::new(
        &m,
        &b,
        &DfsData::single_level(DfsParameters {
            coarse_schur_mode: SchurMode::Dense,
            ..DfsParameters::default()
        }),
    );

    let inherent: [(Vec<f64>, usize, [usize; 3], bool); 3] = [
        inherent_bits(&bdp, &rhs),
        inherent_bits(&bp, &rhs),
        inherent_bits(&dfs, &rhs),
    ];
    let dyn_views: [&dyn DarcySolver; 3] = [&bdp, &bp, &dfs];
    for (k, dyn_view) in dyn_views.iter().enumerate() {
        let (x, iters, offsets, converged) = &inherent[k];
        // Fresh solve through the trait object (each mult restarts from a
        // zero iterate, so the second run is a bit-copy of the first).
        let mut via_trait = vec![0.0; rhs.len()];
        dyn_view.mult(&rhs, &mut via_trait);
        assert_eq!(x, &via_trait, "solver {k}: solution bits");
        assert_eq!(*iters, dyn_view.num_iterations(), "solver {k}: iteration count");
        assert_eq!(*offsets, dyn_view.offsets(), "solver {k}: offsets");
        assert_eq!(offsets[2], dyn_view.size(), "solver {k}: height");
        assert_eq!(*converged, dyn_view.converged(), "solver {k}: converged");
    }
}
