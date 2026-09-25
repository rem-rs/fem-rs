//! D767 — the MFEM-faithful GMRES stopping rule vs the older
//! right-preconditioned one.
//!
//! ## The defect
//!
//! `right_preconditioned_gmres` (`crates/solver/src/block_operator.rs`) stops
//! on the **true** residual, `‖b − A x‖ ≤ rtol·‖b‖`.  MFEM's
//! `GMRESSolver::Mult` (`$MFEM/linalg/solvers.cpp:1150-1340`, 4.10) stops on
//! the **preconditioned** residual instead: `r = M(b − A x)`, `β = ‖r‖`,
//! `final_norm = max(rel_tol·β, abs_tol)`, and the Givens-rotated estimate
//! `|s(i+1)|` — an estimate of `‖M(b − A x_k)‖` — is compared against
//! `final_norm`.  On a strongly preconditioned system the two disagree by
//! orders of magnitude, so ex22 `-p 2 -o 1 -m data/inline-tri.mesh` printed
//! `No convergence!` after 1000 iterations where MFEM converges in 266 (the
//! computed solutions agree to print precision).
//!
//! `mfem_gmres` is the faithful port (see its doc comment): same criterion,
//! same `w = M·A·vᵢ` Arnoldi order, same `   Pass : ...  ||B r|| = ...`
//! console output and `GMRES: No convergence!` text.
//!
//! The two functions are pinned here on micro systems whose behaviour is
//! analytically determined, so the suite fails if anyone re-unifies the
//! criteria (the older function must keep its semantics for its existing
//! consumers: `div_free_solver`, `solve_block_precond_gmres`, ex19, d682).

use fem_linalg::CooMatrix;
use fem_solver::block_operator::{mfem_gmres, right_preconditioned_gmres, MfemGmresConfig};
use fem_solver::SolverConfig;

/// Dense diagonal matrix as a CSR matrix.
fn diag_csr(d: &[f64]) -> fem_linalg::CsrMatrix<f64> {
    let n = d.len();
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (i, &v) in d.iter().enumerate() {
        coo.add(i, i, v);
    }
    coo.into_csr()
}

/// `(Mx)ᵢ = dinvᵢ·xᵢ` — the diagonal (Jacobi/DSmoother) preconditioner.
fn diag_precond(dinv: Vec<f64>) -> impl Fn(&[f64], &mut [f64]) {
    move |r: &[f64], z: &mut [f64]| {
        for i in 0..z.len() {
            z[i] = dinv[i] * r[i];
        }
    }
}

fn cfg(max_iter: usize, kdim: usize) -> MfemGmresConfig {
    MfemGmresConfig {
        kdim,
        max_iter,
        rel_tol: 1e-12,
        abs_tol: 0.0,
        print_level: -1,
        iterative_mode: true,
    }
}

/// The discriminating case: `M = diag(1, 0)` hides the second component of the
/// residual completely, so the preconditioned residual is **exactly zero** after
/// one Arnoldi step (`M·A·v₀` is already in the span of `v₀`) while the true
/// residual keeps the whole second component.
///
/// MFEM's criterion therefore converges at iteration 1 (`‖B r‖ = 0`); the
/// true-residual criterion never sees that residual and runs to `max_iter`.
#[test]
fn mfem_criterion_stops_where_the_true_residual_would_not() {
    let a = diag_csr(&[1.0, 1.0]);
    let b = [1.0, 1.0];
    let pre = diag_precond(vec![1.0, 0.0]);

    let mut x_new = vec![0.0; 2];
    let new = mfem_gmres(&a, &b, &mut x_new, cfg(10, 50), Some(&pre)).expect("mfem_gmres");
    assert!(new.converged, "MFEM criterion must accept the preconditioned residual: {new:?}");
    assert_eq!(new.iterations, 1, "converged after one Arnoldi step: {new:?}");
    assert_eq!(new.final_residual.to_bits(), 0.0_f64.to_bits());
    // The hidden component is still untouched — proof that the two criteria see
    // different residuals on this system.
    assert_eq!(x_new[1], 0.0);

    let mut x_old = vec![0.0; 2];
    let old = right_preconditioned_gmres(
        &a,
        &b,
        &mut x_old,
        50,
        &SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 10, ..SolverConfig::default() },
        pre,
    )
    .expect("right_preconditioned_gmres");
    assert!(!old.converged, "the true residual never reaches rtol·‖b‖ here: {old:?}");
    assert_eq!(old.iterations, 10);
}

/// `M b = 0` (with the zero initial guess MFEM's ex22 uses): the preconditioned
/// initial residual is exactly zero, so MFEM converges at iteration 0 with
/// `final_norm = β = 0`; the old function (which normalises by `‖b‖ ≠ 0`)
/// iterates and never converges — the `M`-hidden part of the residual is
/// invisible to its criterion.
#[test]
fn zero_preconditioned_rhs_converges_at_iteration_zero() {
    let a = diag_csr(&[1.0, 1.0]);
    let b = [1.0, 0.0];
    let pre = diag_precond(vec![0.0, 1.0]);
    let mut x = vec![0.0, 0.0];
    let res = mfem_gmres(&a, &b, &mut x, cfg(10, 50), Some(&pre)).expect("mfem_gmres");
    assert!(res.converged && res.iterations == 0, "{res:?}");
    assert_eq!(res.final_residual.to_bits(), 0.0_f64.to_bits());
    // Nothing was solved — the residual that M hides is still there.
    assert_eq!(x, vec![0.0, 0.0]);

    let mut x_old = vec![0.0; 2];
    let old = right_preconditioned_gmres(
        &a,
        &b,
        &mut x_old,
        50,
        &SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 10, ..SolverConfig::default() },
        pre,
    )
    .expect("right_preconditioned_gmres");
    assert!(!old.converged, "‖b‖ ≠ 0 so the old criterion cannot stop: {old:?}");
}

/// `abs_tol` alone can satisfy the criterion on the *preconditioned* residual:
/// `β = 1e-8 ≤ max(1e-12·1e-8, 1e-6)` ⇒ converged at iteration 0.
#[test]
fn abs_tol_dominates_the_relative_threshold() {
    let a = diag_csr(&[1.0, 1.0]);
    let b = [1e-8, 0.0];
    let pre = diag_precond(vec![1.0, 1.0]);
    let mut x = vec![0.0; 2];
    let res = mfem_gmres(
        &a,
        &b,
        &mut x,
        MfemGmresConfig { abs_tol: 1e-6, print_level: -1, ..cfg(10, 50) },
        Some(&pre),
    )
    .expect("mfem_gmres");
    assert!(res.converged && res.iterations == 0, "{res:?}");
    assert_eq!(res.final_residual.to_bits(), 1e-8_f64.to_bits());
}

/// `iterative_mode = false` zeroes the incoming guess and solves from scratch;
/// `iterative_mode = true` with an exact guess stops at iteration 0 and leaves
/// `x` alone.  Both are MFEM `Solver::iterative_mode` semantics.
#[test]
fn iterative_mode_controls_the_initial_guess() {
    let a = diag_csr(&[2.0, 2.0]);
    let b = [2.0, 2.0];
    let pre = diag_precond(vec![1.0, 1.0]);

    let mut x = vec![5.0, 5.0];
    let res = mfem_gmres(
        &a,
        &b,
        &mut x,
        MfemGmresConfig { iterative_mode: false, ..cfg(10, 50) },
        Some(&pre),
    )
    .expect("mfem_gmres");
    assert!(res.converged, "{res:?}");
    assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14, "x = {x:?}");

    let mut x_exact = vec![1.0, 1.0];
    let res_exact =
        mfem_gmres(&a, &b, &mut x_exact, cfg(10, 50), Some(&pre)).expect("mfem_gmres");
    assert!(res_exact.converged && res_exact.iterations == 0, "{res_exact:?}");
    assert_eq!(x_exact, vec![1.0, 1.0]);
}

/// `kdim` is the restart length: a system needing more than `kdim` iterations
/// must restart and still converge (`Restarting...` is printed at level 1).
#[test]
fn kdim_restarts_and_still_converges() {
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0]);
    let b = [1.0, 1.0, 1.0, 1.0];
    let pre = diag_precond(vec![1.0, 1.0, 1.0, 1.0]);
    let mut x = vec![0.0; 4];
    let res = mfem_gmres(&a, &b, &mut x, cfg(100, 2), Some(&pre)).expect("mfem_gmres");
    assert!(res.converged, "{res:?}");
    assert!(res.iterations > 2, "must take more than one restart cycle: {res:?}");
}

/// `print_level` only changes what is written to stdout: `-1` (silent) and `1`
/// (iterations) must return the identical result.
#[test]
fn print_level_does_not_change_the_solution() {
    let a = diag_csr(&[1.0, 2.0, 3.0]);
    let b = [1.0, 2.0, 3.0];
    let pre = diag_precond(vec![1.0, 1.0, 1.0]);
    let mut x_silent = vec![0.0; 3];
    let mut x_loud = vec![0.0; 3];
    let silent = mfem_gmres(&a, &b, &mut x_silent, cfg(20, 50), Some(&pre)).expect("gmres");
    let loud = mfem_gmres(
        &a,
        &b,
        &mut x_loud,
        MfemGmresConfig { print_level: 1, ..cfg(20, 50) },
        Some(&pre),
    )
    .expect("gmres");
    assert_eq!(silent.iterations, loud.iterations);
    assert_eq!(silent.final_residual.to_bits(), loud.final_residual.to_bits());
    for i in 0..3 {
        assert_eq!(x_silent[i].to_bits(), x_loud[i].to_bits());
    }
}
