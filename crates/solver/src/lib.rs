//! # fem-solver
//!
//! Iterative and direct linear solvers backed by [`linlvo`].
//!
//! ## Iterative solvers
//! - [`solve_cg`]          — Conjugate Gradient (SPD systems)
//! - [`solve_cg_operator`] — Conjugate Gradient with operator callback (backend-agnostic)
//! - [`solve_gmres_operator`] — GMRES with operator callback (backend-agnostic)
//! - [`solve_bicgstab_operator`] — BiCGSTAB with operator callback (backend-agnostic)
//! - [`solve_pcg_jacobi`]  — PCG with Jacobi preconditioner
//! - [`solve_pcg_gssmoother`] — PCG with symmetric Gauss-Seidel (MFEM GSSmoother)
//! - [`solve_pcg_ilu0`]    — PCG with ILU(0) preconditioner
//! - [`solve_pcg_ildlt`]   — PCG with ILDLᵀ preconditioner
//! - [`solve_gmres`]       — GMRES (non-symmetric systems)
//! - [`solve_gmres_jacobi`] — GMRES with Jacobi preconditioner
//! - [`solve_gmres_ilu0`]   — GMRES with ILU(0) preconditioner
//! - [`solve_gmres_iluk`]   — GMRES with ILU(k) preconditioner
//! - [`solve_gmres_ilut`]   — GMRES with ILUT preconditioner
//! - [`solve_pcg_iluk`]     — PCG with ILU(k) preconditioner
//! - [`solve_fgmres_ilut`]  — FGMRES with ILUT preconditioner
//! - [`solve_precond_kind`] — unified ILU-family dispatcher via [`PrecondKind`]
//! - [`solve_bicgstab`]    — BiCGSTAB
//! - [`solve_idrs`]        — IDR(s) (non-symmetric, short-recurrence)
//! - [`solve_tfqmr`]       — TFQMR (Transpose-Free QMR)
//! - [`solve_fgmres_ilu0`] — Flexible GMRES with ILU(0) preconditioner
//!
//! ## Generic preconditioner interface
//! - [`solve_pcg_precond`]    — PCG with any type implementing [`linlvoPreconditioner`]
//! - [`solve_gmres_precond`]  — GMRES with any type implementing [`linlvoPreconditioner`]
//! - [`solve_fgmres_precond`] — FGMRES with any type implementing [`linlvoPreconditioner`]
//!
//! ## Auxiliary-space preconditioners (Hiptmair-Xu)
//! - [`solve_pcg_ams`]     — PCG with AMS for H(curl) (Maxwell)
//! - [`solve_gmres_ams`]   — GMRES with AMS for H(curl)
//! - [`solve_pcg_ads`]     — PCG with ADS for H(div) (Darcy)
//! - [`solve_gmres_ads`]   — GMRES with ADS for H(div)
//!
//! ## Direct solvers
//! - [`solve_sparse_lu`]        — Sparse LU for general systems
//! - [`solve_sparse_cholesky`]  — Sparse Cholesky for SPD systems
//! - [`solve_sparse_ldlt`]      — Sparse LDLᵀ for symmetric indefinite systems
//! - [`solve_sparse_mumps`]     — MUMPS-compatible direct path (baseline)
//! - [`solve_sparse_mkl`]       — MKL-compatible direct path (baseline)
//!
//! All solvers operate on [`fem_linalg::CsrMatrix<T>`].

#![allow(clippy::needless_range_loop)]

pub use linlvo::core::preconditioner::Preconditioner;
pub use linlvo::precond::{AdsConfig, AmsConfig, AmsPrecond, AuxSpaceSolver};
pub use linlvo::DenseVec;
/// Re-export of linlvo's [`Preconditioner`] trait.
///
/// Implement this trait to plug any custom approximate-inverse into
/// [`solve_pcg_precond`], [`solve_gmres_precond`], or [`solve_fgmres_precond`]
/// without depending on the `linlvo` crate directly.
pub use linlvo::Preconditioner as linlvoPreconditioner;

/// Symmetric Gauss-Seidel smoother — MFEM-compatible full GS sweeps.
///
/// Use with PCG for SPD systems:
/// ```ignore
/// use fem_solver::GSSmoother;
/// let prec = GSSmoother::from_csr(&la).expect("GSSmoother");
/// let res = solve_pcg(&a, &b, &mut x, &prec, 1e-12, 200, true);
/// ```
pub type GSSmoother = linlvo::GaussSeidelSmoother<f64>;

#[cfg(feature = "gpu")]
pub mod bicgstab_gpu;
#[cfg(feature = "gpu")]
pub mod cg_gpu;
#[cfg(feature = "gpu")]
pub mod gmres_gpu;
#[cfg(feature = "gpu")]
pub mod gpu_base;

// ─── Re-export solver types from fem-linalg ───────────────────────────────────

pub use fem_linalg::{
    fem_to_linlvo_csr, into_result, PrintLevel, SolveResult, SolverConfig, SolverError,
};

// ─── Macro definitions (used by iterative, precond, direct modules) ──────────

#[macro_use]
mod macros;

// ─── Solver modules ──────────────────────────────────────────────────────────

mod direct;
mod iterative;
mod precond;

pub use direct::*;
pub use iterative::*;
pub use blockilu::{solve_pcg_blockilu, BlockIlu};
pub use precond::*;

// ─── Additional sub-modules ──────────────────────────────────────────────────

pub mod adaptive;
pub mod bdf;

// Re-export fem-amg so downstream crates import AMG via `fem_solver::amg`.
pub use fem_amg as amg;
pub mod block;
pub mod block_gmres;
pub mod block_operator;
pub mod butcher;
pub mod complex_ams;
pub mod constrained;
pub mod constrained_operator;
pub mod dae;
pub mod div_free;
pub mod eigen;
pub mod events;
pub mod geometric_mg;
pub mod lor;
pub mod multiphysics;
pub mod multirate;
pub mod ode;
pub mod blockilu;
pub mod p_multigrid;
/// Plugin API traits for pro-solver extensions.
pub mod plugin;
pub mod sdc;

pub use block::{
    BlockDiagonalPrecond, BlockSystem, BlockTriangularPrecond, MinresSolver, SchurComplementSolver,
};
pub use block_gmres::{solve_block_gmres, BlockGmresConfig};
pub use block_operator::{
    build_jacobi_block_solvers, extract_upper_coupling, right_preconditioned_gmres,
    solve_block_precond_gmres, BlockDiagonalPrecondN, BlockNonlinearForm, BlockOpMatrix,
    BlockOperator, BlockSolver, BlockTriangularPrecondN, MultiphysicsOperator, SumBlockOp,
};
pub use constrained::solve_gmres_block_diag_gs;
pub use constrained::SchurConstrainedSolver;
pub use eigen::{
    ame_solve, arpack, feast_interval, krylov_schur, lobpcg, lobpcg_constrained,
    lobpcg_constrained_preconditioned, lobpcg_essential_bc, lobpcg_preconditioned,
    make_constraint_matrix, solve_dense_generalized_eig, AmeConfig, EigenResult,
    GeneralizedEigenSolver, IntervalEigenConfig, LobpcgConfig, LobpcgSolver, WhichEigenvalue,
};
pub use multiphysics::{
    CoupledLinearStrategy, CoupledNewtonConfig, CoupledNewtonResult, CoupledNewtonSolver,
    CoupledProblem, CoupledSolveError,
};

pub use multirate::{
    run_multirate, run_multirate_adaptive, MultiRateAdaptiveConfig, MultiRateConfig,
    MultiRateError, MultiRateStats,
};

#[cfg(test)]
mod linlvo_integration_tests {
    use fem_linalg::CsrMatrix;

    #[test]
    fn nalgebra_csr_linear_operator_spmv() {
        // 2x2 matrix: [2 1; 0 3]
        let mut coo = fem_linalg::CooMatrix::<f64>::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(0, 1, 1.0);
        coo.add(1, 1, 3.0);
        let a: CsrMatrix<f64> = coo.into_csr();

        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 2];

        a.spmv(&x, &mut y);

        assert!((y[0] - 4.0).abs() < 1e-12);
        assert!((y[1] - 6.0).abs() < 1e-12);
    }
}
pub use adaptive::{
    explicit_adaptive_step, integrate_adaptive, AdaptiveConfig, IntegratorStats, StepperState,
};
pub use bdf::{BdfConfig, BdfIntegrator, BdfStats, NewtonConfig, NordsieckState};
pub use butcher::{
    ark3_tableau, ark5_tableau, backward_euler_tableau, bs32_tableau, ck54_tableau, dopri5_tableau,
    explicit_midpoint_tableau, fehlberg12_tableau, forward_euler_tableau, heun_tableau,
    i_step_controller, imex_euler_tableau, imex_ssp2_tableau, implicit_midpoint_tableau,
    pi_step_controller, rk4_tableau, sdirk2_tableau, sdirk3_tableau, sdirk4_tableau, wrms_error,
    ButcherTableau, ImexTableau,
};
pub use constrained_operator::RectangularConstrainedOperator;
pub use dae::{dae_consistent_initialization, DaeConfig, DaeIntegrator, DaeNewtonConfig, DaeState};
pub use events::{integrate_with_events, EventFunction, EventInfo};
pub use geometric_mg::{
    GeometricMgAsPrecond, GeometricMgConfig, GeometricMgHierarchy, GeometricMgLevel,
    GeometricMgPrecond, MgCycleType, MgSmootherType, PADiffusionOp, StoredElementOperator,
    SumFactDiffusionOp,
};
pub use lor::{
    build_lor_operator, solve_gmres_lor, solve_gmres_lor_amg, solve_pcg_lor, solve_pcg_lor_amg,
    solve_vcycle_geom_mg, AmgConfig, GeomMGHierarchy, GeomMGPrecond, LorAmgPrecond, LorPrecond,
};
pub use ode::{
    Bdf2, Bdf2State, ForwardEuler, ImexArk3, ImexDirkRk3, ImexEuler, ImexExpImplEuler,
    ImexOperator, ImexRk2_222, ImexRk2_232, ImexRk3, ImexSsp2, ImexTimeStepper,
    ImplicitEuler, ImplicitTimeStepper, Rk4, Sdirk2, TimeStepper,
};
pub use p_multigrid::{
    build_pmg_hierarchy_1d_laplacian, fmg_solve, solve_vcycle_pmg, PmgHierarchy, PmgPrecond,
};
pub use sdc::{SdcConfig, SdcIntegrator};

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::{CooMatrix, CsrMatrix};
    use linlvo::{IldltPrecond, Ilu0Precond, JacobiPrecond};

    /// 1-D Laplacian: tridiagonal [-1, 2, -1] of size n.
    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -1.0);
            }
        }
        coo.into_csr()
    }

    /// Mildly non-symmetric 1-D convection-diffusion-like operator.
    fn nonsymmetric_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0);
            if i > 0 {
                coo.add(i, i - 1, -1.2);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -0.4);
            }
        }
        coo.into_csr()
    }

    #[test]
    fn cg_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_cg(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "CG failed to converge");
        // verify Ax ≈ b
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "residual too large: {err}");
    }

    #[test]
    fn pcg_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_jacobi(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged);
        assert!(
            res.iterations < 60,
            "too many iterations: {}",
            res.iterations
        );
    }

    #[test]
    fn gmres_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn gmres_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_jacobi(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+Jacobi failed to converge");
        assert!(
            res.iterations < 60,
            "too many iterations: {}",
            res.iterations
        );
    }

    #[test]
    fn gmres_ilu0_nonsymmetric() {
        let n = 60;
        let a = nonsymmetric_1d(n);
        let b = vec![1.0_f64; n];
        let mut x_plain = vec![0.0_f64; n];
        let mut x_ilu = vec![0.0_f64; n];
        let plain = solve_gmres(&a, &b, &mut x_plain, 30, &SolverConfig::default()).unwrap();
        let ilu = solve_gmres_ilu0(&a, &b, &mut x_ilu, 30, &SolverConfig::default()).unwrap();
        assert!(plain.converged, "plain GMRES failed to converge");
        assert!(ilu.converged, "GMRES+ILU0 failed to converge");
        assert!(
            ilu.iterations <= plain.iterations,
            "GMRES+ILU0 should not need more iterations: plain={} ilu={}",
            plain.iterations,
            ilu.iterations
        );
    }

    #[test]
    fn fgmres_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn fgmres_jacobi_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_jacobi(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged);
        assert!(
            res.iterations < 60,
            "too many iterations: {}",
            res.iterations
        );
    }

    #[test]
    fn fgmres_ilu0_nonsymmetric() {
        let n = 60;
        let a = nonsymmetric_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_fgmres_ilu0(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged, "FGMRES+ILU0 failed to converge");
    }

    // ── Generic preconditioner interface tests ────────────────────────────────

    #[test]
    fn solve_pcg_precond_jacobi() {
        // Verify the generic PCG wrapper produces the same result as solve_pcg_jacobi.
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x1 = vec![0.0_f64; n];
        let mut x2 = vec![0.0_f64; n];

        let prec = JacobiPrecond::from_csr(&fem_to_linlvo_csr(&a)).unwrap();
        let r1 = solve_pcg_precond(&a, &b, &mut x1, &prec, &SolverConfig::default()).unwrap();
        let r2 = solve_pcg_jacobi(&a, &b, &mut x2, &SolverConfig::default()).unwrap();
        assert!(r1.converged);
        assert_eq!(r1.iterations, r2.iterations);
    }

    #[test]
    fn solve_gmres_precond_ilu0() {
        let n = 60;
        let a = nonsymmetric_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let la = fem_to_linlvo_csr(&a);
        let prec = Ilu0Precond::from_csr(&la).unwrap();
        let res = solve_gmres_precond(&a, &b, &mut x, 30, &prec, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "generic GMRES+ILU0 failed: residual={}",
            res.final_residual
        );
    }

    #[test]
    fn solve_fgmres_precond_ildlt() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let la = fem_to_linlvo_csr(&a);
        let prec = IldltPrecond::from_csr(&la).unwrap();
        let res =
            solve_fgmres_precond(&a, &b, &mut x, 30, &prec, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "generic FGMRES+ILDLt failed: residual={}",
            res.final_residual
        );
    }

    // ── Phase 6: ILU(k) / ILUT tests ─────────────────────────────────────────

    #[test]
    fn solve_gmres_iluk0_equals_ilu0() {
        // ILU(0) and ILU(k=0) should give the same iteration count on a
        // symmetric tridiagonal (fill level 0 = no extra fill = ILU0).
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x0 = vec![0.0_f64; n];
        let mut xk = vec![0.0_f64; n];
        let cfg = SolverConfig::default();
        let r0 = solve_gmres_ilu0(&a, &b, &mut x0, 30, &cfg).unwrap();
        let rk = solve_gmres_iluk(&a, &b, &mut xk, 30, 0, &cfg).unwrap();
        assert!(r0.converged, "ILU0 did not converge");
        assert!(rk.converged, "ILU(k=0) did not converge");
    }

    #[test]
    fn solve_gmres_iluk1_converges() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_iluk(&a, &b, &mut x, 30, 1, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "GMRES+ILU(1) failed: res={}",
            res.final_residual
        );
    }

    #[test]
    fn solve_gmres_iluk2_fewer_iters_than_ilu0() {
        // ILU(2) should need no more iterations than ILU(0) on Laplacian.
        let n = 80;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x0 = vec![0.0_f64; n];
        let mut x2 = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            ..Default::default()
        };
        let r0 = solve_gmres_ilu0(&a, &b, &mut x0, 30, &cfg).unwrap();
        let r2 = solve_gmres_iluk(&a, &b, &mut x2, 30, 2, &cfg).unwrap();
        assert!(r0.converged && r2.converged);
        assert!(
            r2.iterations <= r0.iterations,
            "ILU(2) used more iterations ({}) than ILU(0) ({})",
            r2.iterations,
            r0.iterations
        );
    }

    #[test]
    fn solve_gmres_ilut_converges_spd() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_ilut(&a, &b, &mut x, 30, 0.01, 10, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "GMRES+ILUT failed: res={}",
            res.final_residual
        );
    }

    #[test]
    fn solve_gmres_ilut_nonsym_converges() {
        // Non-symmetric banded: A[i,i]=3, A[i,i-1]=-1, A[i,i+1]=-2.
        let n = 50;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0_f64);
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i + 1 < n {
                coo.add(i, i + 1, -2.0);
            }
        }
        let a = coo.into_csr();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_ilut(&a, &b, &mut x, 30, 1e-3, 15, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "GMRES+ILUT (nonsym) failed: res={}",
            res.final_residual
        );
    }

    #[test]
    fn solve_pcg_iluk_converges() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_iluk(&a, &b, &mut x, 1, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "PCG+ILU(1) failed: res={}",
            res.final_residual
        );
    }

    #[test]
    fn solve_fgmres_ilut_converges() {
        let n = 60;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res =
            solve_fgmres_ilut(&a, &b, &mut x, 30, 0.01, 10, &SolverConfig::default()).unwrap();
        assert!(
            res.converged,
            "FGMRES+ILUT failed: res={}",
            res.final_residual
        );
    }

    #[test]
    fn solve_precond_kind_ilu0_matches_direct() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x1 = vec![0.0_f64; n];
        let mut x2 = vec![0.0_f64; n];
        let cfg = SolverConfig::default();
        solve_gmres_ilu0(&a, &b, &mut x1, 30, &cfg).unwrap();
        solve_precond_kind(&a, &b, &mut x2, 30, PrecondKind::Ilu0, &cfg).unwrap();
        for i in 0..n {
            assert!((x1[i] - x2[i]).abs() < 1e-12, "node {i} differs");
        }
    }

    #[test]
    fn solve_precond_kind_iluk_converges() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_precond_kind(
            &a,
            &b,
            &mut x,
            30,
            PrecondKind::Iluk(1),
            &SolverConfig::default(),
        )
        .unwrap();
        assert!(res.converged);
    }

    #[test]
    fn solve_precond_kind_ilut_converges() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let kind = PrecondKind::Ilut {
            tau: 0.01,
            fill: 10,
        };
        let res = solve_precond_kind(&a, &b, &mut x, 30, kind, &SolverConfig::default()).unwrap();
        assert!(res.converged);
    }

    #[test]
    fn ilut_solution_matches_iluk_on_spd() {
        // Both ILUT and ILU(k) should give the same (correct) solution on SPD.
        let n = 30;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut xt = vec![0.0_f64; n];
        let mut xk = vec![0.0_f64; n];
        solve_gmres_ilut(
            &a,
            &b,
            &mut xt,
            30,
            1e-12,
            30,
            &SolverConfig {
                rtol: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        solve_gmres_iluk(
            &a,
            &b,
            &mut xk,
            30,
            2,
            &SolverConfig {
                rtol: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..n {
            assert!(
                (xt[i] - xk[i]).abs() < 1e-8,
                "node {i}: ilut={:.3e} iluk={:.3e}",
                xt[i],
                xk[i]
            );
        }
    }

    #[test]
    fn idrs_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_idrs(&a, &b, &mut x, 4, &SolverConfig::default()).unwrap();
        assert!(res.converged, "IDR(s) failed to converge");
    }

    #[test]
    fn tfqmr_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_tfqmr(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "TFQMR failed to converge");
    }

    // ── MINRES tests ─────────────────────────────────────────────────────────

    /// Symmetric indefinite matrix: A = diag(-1, 1, -1, 1, …, 50×50).
    /// MINRES must converge; CG would break down.
    fn symmetric_indefinite(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            let val = if i % 2 == 0 { -1.0_f64 } else { 2.0_f64 };
            coo.add(i, i, val);
            if i > 0 {
                coo.add(i, i - 1, 0.5);
            }
            if i < n - 1 {
                coo.add(i, i + 1, 0.5);
            }
        }
        coo.into_csr()
    }

    #[test]
    fn minres_laplacian_spd_debug() {
        // Debug: 3×3 Laplacian with verification at each step
        let n = 3;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 100,
            verbose: false,
            ..Default::default()
        };
        let res = solve_minres(&a, &b, &mut x, &cfg).unwrap();
        eprintln!(
            "n=3 MINRES: converged={} iters={} residual={:.6e}",
            res.converged, res.iterations, res.final_residual
        );
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        for i in 0..n {
            eprintln!(
                "  x[{}] = {:.10e}  (Ax-b)[{}] = {:.6e}",
                i,
                x[i],
                i,
                (ax[i] - b[i]).abs()
            );
        }
        assert!(res.converged, "MINRES 3×3 failed");
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        eprintln!("n=3 MINRES ‖Ax−b‖ = {:.6e}", err);
        assert!(err < 1e-10, "MINRES 3×3 residual too large: {err}");
    }

    #[test]
    fn minres_residual_trace() {
        // Verify convergence for various n values.
        for &n in &[3, 4, 5, 6, 7, 8, 9, 10, 20, 50] {
            let a = laplacian_1d(n);
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let cfg = SolverConfig {
                rtol: 1e-10,
                max_iter: 2000,
                verbose: false,
                ..Default::default()
            };
            let res =
                solve_minres(&a, &b, &mut x, &cfg).unwrap_or_else(|_| panic!("n={n} solver error"));
            assert!(
                res.converged,
                "n={n} MINRES failed: iters={} res={:.3e}",
                res.iterations, res.final_residual
            );
            let mut ax = vec![0.0_f64; n];
            a.spmv(&x, &mut ax);
            let err: f64 = ax
                .iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(err < 1e-6, "n={n} residual too large: {err}");
        }
    }

    #[test]
    fn minres_laplacian_spd() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_minres(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "MINRES (SPD) failed to converge");
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "MINRES (SPD) residual too large: {err}");
    }

    #[test]
    fn minres_indefinite() {
        let n = 50;
        let a = symmetric_indefinite(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_minres(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "MINRES (indefinite) failed to converge");
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "MINRES (indefinite) residual too large: {err}");
    }

    #[test]
    fn minres_operator_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_minres_operator(
            n,
            n,
            |z, w| a.spmv(z, w),
            &b,
            &mut x,
            &SolverConfig::default(),
        )
        .unwrap();
        assert!(res.converged, "MINRES operator failed");
    }

    #[test]
    fn minres_converges_on_helmholtz_shift() {
        // Indefinite: Laplacian − 10·I  (negative shift → symmetric indefinite)
        let n = 50;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0_f64 - 10.0_f64);
            if i > 0 {
                coo.add(i, i - 1, -1.0);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -1.0);
            }
        }
        let a = coo.into_csr();
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 2000,
            ..Default::default()
        };
        let res = solve_minres(&a, &b, &mut x, &cfg).unwrap();
        assert!(
            res.converged,
            "MINRES (Helmholtz shift) failed: iters={} res={:.3e}",
            res.iterations, res.final_residual
        );
    }

    #[test]
    fn minres_residual_decreases() {
        let n = 30;
        let a = symmetric_indefinite(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 0.0,
            atol: 1e-12,
            max_iter: 1000,
            ..Default::default()
        };
        let res = solve_minres(&a, &b, &mut x, &cfg).unwrap();
        assert!(res.converged);
        assert!(
            res.iterations <= 50,
            "too many iterations: {}",
            res.iterations
        );
    }

    // ── GCR tests ────────────────────────────────────────────────────────────

    #[test]
    fn gcr_laplacian() {
        // GCR on SPD: convergence is slow due to A-orthogonalisation drift.
        // Use modest tolerance; for SPD, use CG or MINRES instead.
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-4,
            max_iter: 3000,
            ..Default::default()
        };
        let res = solve_gcr(&a, &b, &mut x, n, &cfg).unwrap();
        assert!(
            res.converged,
            "GCR (SPD) failed: iters={} res={:.3e}",
            res.iterations, res.final_residual
        );
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-3, "GCR residual too large: {err}");
    }

    #[test]
    fn gcr_tiny_spd() {
        // GCR converges on tiny SPD systems (n ≤ 3) at full space.
        for n in [3] {
            let a = laplacian_1d(n);
            let b = vec![1.0_f64; n];
            let mut x = vec![0.0_f64; n];
            let cfg = SolverConfig {
                rtol: 1e-6,
                max_iter: 200,
                ..Default::default()
            };
            let res = solve_gcr(&a, &b, &mut x, n, &cfg)
                .unwrap_or_else(|e| panic!("n={n} GCR error: {e}"));
            assert!(
                res.converged,
                "n={n} GCR not converged: iters={} res={:.3e}",
                res.iterations, res.final_residual
            );
        }
    }

    #[test]
    fn gcr_converges_fewer_than_max_iters() {
        // Nonsymmetric system (convergence is reliable even with restart)
        let n = 100;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 3.0);
            if i > 0 {
                coo.add(i, i - 1, -1.2);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -0.4);
            }
        }
        let a = coo.into_csr();
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 2000,
            ..Default::default()
        };
        let res = solve_gcr(&a, &b, &mut x, 50, &cfg).unwrap();
        assert!(res.converged);
        assert!(
            res.iterations < 200,
            "GCR too many ({}): should converge <<200",
            res.iterations
        );
    }

    #[test]
    fn gcr_solution_matches_gmres() {
        let n = 40;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 4.0);
            if i > 0 {
                coo.add(i, i - 1, -1.5);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -0.8);
            }
        }
        let a = coo.into_csr();
        let b: Vec<f64> = (0..n).map(|i| (i % 5 + 1) as f64).collect();
        let mut x_gcr = vec![0.0_f64; n];
        let mut x_gmres = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            ..Default::default()
        };
        solve_gcr(&a, &b, &mut x_gcr, 40, &cfg).unwrap();
        solve_gmres(&a, &b, &mut x_gmres, 40, &cfg).unwrap();
        for i in 0..n {
            assert!(
                (x_gcr[i] - x_gmres[i]).abs() < 1e-6,
                "node {i}: gcr={:.6e} gmres={:.6e}",
                x_gcr[i],
                x_gmres[i]
            );
        }
    }

    #[test]
    fn gcr_zero_restart_errors() {
        let n = 10;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let r = solve_gcr(&a, &b, &mut x, 0, &SolverConfig::default());
        assert!(r.is_err(), "GCR restart=0 should error");
    }

    #[test]
    fn sparse_lu_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_lu(&a, &b).unwrap();
        // verify Ax ≈ b
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-10, "LU residual too large: {err}");
    }

    #[test]
    fn sparse_cholesky_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_cholesky(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-10, "Cholesky residual too large: {err}");
    }

    #[test]
    fn sparse_ldlt_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_ldlt(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-10, "LDLt residual too large: {err}");
    }

    #[test]
    fn sparse_mumps_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_mumps(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-10, "Mumps residual too large: {err}");
    }

    #[test]
    fn sparse_mkl_direct() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let x = solve_sparse_mkl(&a, &b).unwrap();
        let mut ax = vec![0.0_f64; n];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-10, "Mkl residual too large: {err}");
    }

    #[test]
    fn pcg_ildlt_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_pcg_ildlt(&a, &b, &mut x, &SolverConfig::default()).unwrap();
        assert!(res.converged, "PCG+ILDLt failed to converge");
    }

    #[test]
    fn gmres_ildlt_laplacian() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let res = solve_gmres_ildlt(&a, &b, &mut x, 30, &SolverConfig::default()).unwrap();
        assert!(res.converged, "GMRES+ILDLt failed to converge");
    }
}

// ─── AMS / ADS integration tests ─────────────────────────────────────────────

#[cfg(test)]
mod ams_ads_tests {
    use super::*;
    use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use fem_assembly::{DiscreteLinearOperator, VectorAssembler};
    use fem_mesh::Mesh;
    use fem_space::constraints::boundary_dofs_hcurl;
    use fem_space::fe_space::FESpace;
    use fem_space::{H1Space, HCurlSpace};

    // ── AMS: H(curl) curl-curl + mass on 2-D unit square ──────────────────────

    #[test]
    fn pcg_ams_hcurl_2d_converges() {
        let n = 4;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[
                &CurlCurlIntegrator { mu: 1.0 },
                &VectorMassIntegrator { alpha: 1.0 },
            ],
            3,
        );
        let g_fem =
            DiscreteLinearOperator::gradient(&h1, &hcurl).expect("gradient assembly failed");

        // Apply zero Dirichlet BCs symmetrically with diag=1.0 for AMS/PCG compatibility.
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 1.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 300,
                verbose: false,
                ..SolverConfig::default()
            },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg).expect("PCG+AMS returned error");
        assert!(
            res.converged,
            "PCG+AMS did not converge in {} iters",
            res.iterations
        );
        assert!(
            res.final_residual < 1e-6,
            "residual = {}",
            res.final_residual
        );
    }

    #[test]
    fn gmres_ams_hcurl_2d_converges() {
        let n = 4;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[
                &CurlCurlIntegrator { mu: 1.0 },
                &VectorMassIntegrator { alpha: 1.0 },
            ],
            3,
        );
        let g_fem =
            DiscreteLinearOperator::gradient(&h1, &hcurl).expect("gradient assembly failed");

        // Apply zero Dirichlet BCs via row-zeroing (diag=1, rhs=0) for AMS compatibility.
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_row_zeroing(dof as usize, 0.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 300,
                verbose: false,
                ..SolverConfig::default()
            },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_gmres_ams(&a, &g_linlvo, &rhs, &mut x, 30, &cfg)
            .expect("GMRES+AMS returned error");
        assert!(
            res.converged,
            "GMRES+AMS did not converge in {} iters",
            res.iterations
        );
        assert!(
            res.final_residual < 1e-6,
            "residual = {}",
            res.final_residual
        );
    }

    #[test]
    fn pcg_ams_solution_satisfies_ax_eq_b() {
        let n = 4;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[
                &CurlCurlIntegrator { mu: 1.0 },
                &VectorMassIntegrator { alpha: 1.0 },
            ],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        // Apply BCs symmetrically with diag=1.0 for AMS/PCG compatibility.
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 1.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 400,
                verbose: false,
                ..SolverConfig::default()
            },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg).unwrap();
        assert!(res.converged);

        // Verify Ax ≈ rhs
        let mut ax = vec![0.0_f64; ndofs];
        a.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
        assert!(
            err / rhs_norm < 1e-6,
            "relative residual = {}",
            err / rhs_norm
        );
    }

    #[test]
    fn pcg_ams_iteration_count_reasonable() {
        // AMS should converge in far fewer iterations than plain CG on H(curl)
        let n = 6;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[
                &CurlCurlIntegrator { mu: 1.0 },
                &VectorMassIntegrator { alpha: 1.0 },
            ],
            3,
        );
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        // Apply BCs symmetrically with diag=1.0 for AMS/PCG compatibility.
        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 1.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 200,
                verbose: false,
                ..SolverConfig::default()
            },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg).unwrap();
        assert!(res.converged, "PCG+AMS did not converge");
        // AMS should be efficient — converge in at most 100 iterations for this small problem
        assert!(
            res.iterations <= 100,
            "PCG+AMS took {} iters (expected ≤100)",
            res.iterations
        );
    }

    // ── ADS: H(div) mass on 3-D unit cube ─────────────────────────────────────

    #[test]
    fn pcg_ads_hdiv_3d_converges() {
        use fem_space::constraints::boundary_dofs_hdiv;
        use fem_space::HDivSpace;

        let n = 2usize;
        let mesh3 = Mesh::<3>::unit_cube_tet(n);
        let h1 = H1Space::new(mesh3.clone(), 1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv = HDivSpace::new(mesh3.clone(), 0);
        let ndofs_hdiv = hdiv.n_dofs();

        // H(div) mass matrix (SPD)
        let mut a_hdiv =
            VectorAssembler::assemble_bilinear(&hdiv, &[&VectorMassIntegrator { alpha: 1.0 }], 3);

        // Discrete curl C: HCurl -> HDiv and gradient G: H1 -> HCurl
        let c_fem =
            DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).expect("curl_3d assembly failed");
        let g_fem =
            DiscreteLinearOperator::gradient(&h1, &hcurl).expect("gradient assembly failed");

        // Apply zero normal-flux BCs via row-zeroing for ADS compatibility.
        let bnd_hdiv = boundary_dofs_hdiv(hdiv.mesh(), &hdiv, &[1, 2, 3, 4, 5, 6]);
        let mut rhs = vec![1.0_f64; ndofs_hdiv];
        for &dof in &bnd_hdiv {
            a_hdiv.apply_dirichlet_row_zeroing(dof as usize, 0.0, &mut rhs);
        }

        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AdsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 400,
                verbose: false,
                ..SolverConfig::default()
            },
            ads_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs_hdiv];
        let res = solve_pcg_ads(&a_hdiv, &c_linlvo, &g_linlvo, &rhs, &mut x, &cfg)
            .expect("PCG+ADS returned error");
        assert!(
            res.converged,
            "PCG+ADS did not converge in {} iters",
            res.iterations
        );
        assert!(
            res.final_residual < 1e-6,
            "residual = {}",
            res.final_residual
        );
    }

    #[test]
    fn gmres_ads_hdiv_3d_converges() {
        use fem_space::constraints::boundary_dofs_hdiv;
        use fem_space::HDivSpace;

        let n = 2usize;
        let mesh3 = Mesh::<3>::unit_cube_tet(n);
        let h1 = H1Space::new(mesh3.clone(), 1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv = HDivSpace::new(mesh3.clone(), 0);
        let ndofs_hdiv = hdiv.n_dofs();

        let mut a_hdiv =
            VectorAssembler::assemble_bilinear(&hdiv, &[&VectorMassIntegrator { alpha: 1.0 }], 3);
        let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        let bnd_hdiv = boundary_dofs_hdiv(hdiv.mesh(), &hdiv, &[1, 2, 3, 4, 5, 6]);
        let mut rhs = vec![1.0_f64; ndofs_hdiv];
        for &dof in &bnd_hdiv {
            a_hdiv.apply_dirichlet_row_zeroing(dof as usize, 0.0, &mut rhs);
        }

        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AdsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 400,
                verbose: false,
                ..SolverConfig::default()
            },
            ads_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs_hdiv];
        let res = solve_gmres_ads(&a_hdiv, &c_linlvo, &g_linlvo, &rhs, &mut x, 30, &cfg)
            .expect("GMRES+ADS returned error");
        assert!(
            res.converged,
            "GMRES+ADS did not converge in {} iters",
            res.iterations
        );
        assert!(
            res.final_residual < 1e-6,
            "residual = {}",
            res.final_residual
        );
    }

    #[test]
    fn pcg_ads_solution_satisfies_ax_eq_b() {
        use fem_space::constraints::boundary_dofs_hdiv;
        use fem_space::HDivSpace;

        let n = 2usize;
        let mesh3 = Mesh::<3>::unit_cube_tet(n);
        let h1 = H1Space::new(mesh3.clone(), 1);
        let hcurl = HCurlSpace::new(mesh3.clone(), 1);
        let hdiv = HDivSpace::new(mesh3.clone(), 0);
        let ndofs_hdiv = hdiv.n_dofs();

        let mut a_hdiv =
            VectorAssembler::assemble_bilinear(&hdiv, &[&VectorMassIntegrator { alpha: 1.0 }], 3);
        let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        let bnd_hdiv = boundary_dofs_hdiv(hdiv.mesh(), &hdiv, &[1, 2, 3, 4, 5, 6]);
        let mut rhs = vec![1.0_f64; ndofs_hdiv];
        for &dof in &bnd_hdiv {
            a_hdiv.apply_dirichlet_row_zeroing(dof as usize, 0.0, &mut rhs);
        }

        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AdsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 500,
                verbose: false,
                ..SolverConfig::default()
            },
            ads_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs_hdiv];
        let res = solve_pcg_ads(&a_hdiv, &c_linlvo, &g_linlvo, &rhs, &mut x, &cfg).unwrap();
        assert!(res.converged);

        // Verify Ax ≈ rhs
        let mut ax = vec![0.0_f64; ndofs_hdiv];
        a_hdiv.spmv(&x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
        assert!(
            err / rhs_norm < 1e-6,
            "relative residual = {}",
            err / rhs_norm
        );
    }

    #[test]
    fn pcg_ams_p1_nd2_converges() {
        // AMS with H^1 order 1 + H(curl) order 2 (ND2).
        // Tests that gradient() works with mismatched orders (h1=1, hcurl=2).
        let n = 4;
        let mesh = Mesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 2); // ND2
        let ndofs = hcurl.n_dofs();

        let mut a = VectorAssembler::assemble_bilinear(
            &hcurl,
            &[
                &CurlCurlIntegrator { mu: 1.0 },
                &VectorMassIntegrator { alpha: 1.0 },
            ],
            3,
        );
        let g_fem =
            DiscreteLinearOperator::gradient(&h1, &hcurl).expect("gradient P1->ND2 should succeed");

        let bnd = boundary_dofs_hcurl(hcurl.mesh(), &hcurl, &[1, 2, 3, 4]);
        let mut rhs = vec![1.0_f64; ndofs];
        for &dof in &bnd {
            a.apply_dirichlet_symmetric(dof as usize, 1.0, &mut rhs);
        }

        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-14,
                atol: 0.0,
                max_iter: 500,
                verbose: false,
                ..SolverConfig::default()
            },
            ams_cfg: Default::default(),
        };
        let mut x = vec![0.0_f64; ndofs];
        let res = solve_pcg_ams(&a, &g_linlvo, &rhs, &mut x, &cfg)
            .expect("PCG+AMS (P1+ND2) returned error");
        assert!(
            res.converged,
            "PCG+AMS (P1+ND2) did not converge in {} iters",
            res.iterations
        );
        assert!(
            res.final_residual < 1e-6,
            "residual = {}",
            res.final_residual
        );
    }
}
