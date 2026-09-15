//! D169 — singular / degenerate CG trailer, byte-aligned with MFEM 4.10.
//!
//! Every branch of `CGSolver::Mult` (`linalg/solvers.cpp:869-1050`,
//! `MFEM_VERSION = 41000`) that emits output or changes the returned state is
//! driven here and the log is compared byte-for-byte with the C++ ground truth
//! captured from `tmp/d169/probe_cg_branches.cpp` (WSL, `$HOME/mfem410_ser`,
//! glibc): `tmp/d169/probe_stdout.txt`.
//!
//! Mechanics: each `child_*` test runs one solve through the real
//! `println!`-printing code path.  A parent test re-executes this test binary
//! with `--exact <child> --nocapture`, filters the child stdout down to the
//! MFEM log line families (`   Iteration : `, `PCG: `,
//! `Average reduction factor = `) and asserts the exact bytes.
//!
//! Known platform byte difference (the only one): the ARF of a run that broke
//! on a negative `(B r, r)` is a NaN whose sign bit is set by glibc's `pow`
//! (MFEM prints `-nan`) but clear by MSVC/Rust's (`nan`); see `fmt_g`.

use std::process::Command;

use fem_linalg::{fem_to_linlvo_csr, CsrMatrix, SolveResult, SolverError};
use fem_solver::{fmt_g, solve_pcg, GSSmoother};
use linlvo::{DenseVec, Preconditioner};

// ── fixtures ────────────────────────────────────────────────────────────────

fn diag_matrix(d: &[f64]) -> CsrMatrix<f64> {
    let n = d.len();
    CsrMatrix {
        nrows: n,
        ncols: n,
        row_ptr: (0..=n).collect(),
        col_idx: (0..n as u32).collect(),
        values: d.to_vec(),
    }
}

/// 1-D periodic Laplacian (singular: constants in the kernel).
fn periodic_laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        col_idx.push(((i + n - 1) % n) as u32);
        values.push(-1.0);
        col_idx.push(i as u32);
        values.push(2.0);
        col_idx.push(((i + 1) % n) as u32);
        values.push(-1.0);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values }
}

/// Tridiagonal (all-Dirichlet) Poisson — SPD, the case8 regression matrix.
fn tridiag_poisson(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push((i - 1) as u32);
            values.push(-1.0);
        }
        col_idx.push(i as u32);
        values.push(2.0);
        if i + 1 < n {
            col_idx.push((i + 1) as u32);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values }
}

/// `y = D y` with an arbitrary diagonal — deliberately indefinite for the
/// `PCG: The preconditioner is not positive definite.` branches; all ones is
/// the identity, which reproduces MFEM's `B == NULL` arithmetic exactly
/// (`d = r`, `(B r, r) = (r, r)`).
struct DiagPrec {
    d: Vec<f64>,
}

impl Preconditioner for DiagPrec {
    type Vector = DenseVec<f64>;

    fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
        for (i, &xi) in x.as_slice().iter().enumerate() {
            y.as_mut_slice()[i] = self.d[i] * xi;
        }
    }
}

fn identity_prec(n: usize) -> DiagPrec {
    DiagPrec { d: vec![1.0; n] }
}

/// Legacy-helper configuration `PCG(A, M, B, X, 1, 200, 1e-12, 0.0)`.
fn solve(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], prec: &DiagPrec) -> Result<SolveResult, SolverError> {
    solve_pcg(a, b, x, prec, 1e-12, 200, true)
}

fn solve_gs(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64]) -> Result<SolveResult, SolverError> {
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(a)).expect("GSSmoother");
    solve_pcg(a, b, x, &gs, 1e-12, 200, true)
}

// ── child tests: run one solve, print via the real log path ─────────────────

/// probe case1a: singular consistent system with an exactly zero initial
/// residual — the iteration-0 line `(B r, r) = 0` must print, then the solve
/// converges at iteration 0 WITHOUT any trailer (solvers.cpp:919-928).
#[test]
fn child_case1a_singular_b0() {
    let a = periodic_laplacian_1d(4);
    let b = vec![0.0; 4];
    let mut x = vec![0.0; 4];
    let res = solve_gs(&a, &b, &mut x).expect("converges at iteration 0");
    assert!(res.converged && res.iterations == 0 && res.final_residual == 0.0);
}

/// probe case1b: `b = A·v`, `x0 = v` → `r0 = b − A·v = 0` exactly (integer
/// arithmetic), with the incoming x as initial guess (iterative_mode).
#[test]
fn child_case1b_singular_consistent() {
    let a = periodic_laplacian_1d(4);
    let v = [1.0, 2.0, 3.0, 4.0];
    let mut b = vec![0.0; 4];
    a.spmv(&v, &mut b);
    let mut x = v.to_vec();
    let res = solve_gs(&a, &b, &mut x).expect("converges at iteration 0");
    assert!(res.converged && res.iterations == 0 && res.final_residual == 0.0);
}

/// probe case2: indefinite preconditioner at iteration 0 — warn + return with
/// `final_norm = nom` (the RAW value -3, no sqrt) and NO trailer
/// (solvers.cpp:904-918).
#[test]
fn child_case2_prec_indefinite_iter0() {
    let a = diag_matrix(&[1.0, 1.0]);
    let prec = DiagPrec { d: vec![1.0, -1.0] };
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0; 2];
    let res = solve(&a, &b, &mut x, &prec).expect("returns a result");
    assert!(!res.converged && res.iterations == 0);
    assert_eq!(res.final_residual, -3.0, "MFEM keeps the raw nom as final_norm");
}

/// probe case3: `(Ad, d) == 0` before the loop — warn + return,
/// `final_norm = sqrt(nom0)` (solvers.cpp:933-949).
#[test]
fn child_case3_op_den0_iter0() {
    let a = diag_matrix(&[1.0, 0.0]);
    let prec = identity_prec(2);
    let b = vec![0.0, 1.0];
    let mut x = vec![0.0; 2];
    let res = solve(&a, &b, &mut x, &prec).expect("returns a result");
    assert!(!res.converged && res.iterations == 0);
    assert_eq!(res.final_residual, 1.0);
}

/// probe case4: `(Ad, d) < 0` before the loop — warn and CONTINUE; the run
/// then converges at iteration 1 (solvers.cpp:933-949, den < 0 falls through).
#[test]
fn child_case4_op_den_neg_continues() {
    let a = diag_matrix(&[-1.0, 2.0]);
    let prec = identity_prec(2);
    let b = vec![1.0, 0.0];
    let mut x = vec![0.0; 2];
    let res = solve(&a, &b, &mut x, &prec).expect("converges");
    assert!(res.converged && res.iterations == 1 && res.final_residual == 0.0);
}

/// probe case5: indefinite preconditioner inside the loop (`betanom < 0`) —
/// warn BEFORE the iteration line (which is not printed), break into the
/// trailer with `final_iter = 1` and a NaN average reduction factor
/// (solvers.cpp:970-980, 1036-1045, 1047).
#[test]
fn child_case5_prec_indefinite_inloop() {
    let a = diag_matrix(&[1.0, 1.0]);
    let prec = DiagPrec { d: vec![1.0, -1.0] };
    let b = vec![2.0, 1.0];
    let mut x = vec![0.0; 2];
    let err = solve(&a, &b, &mut x, &prec).expect_err("not converged");
    match err {
        SolverError::ConvergenceFailed { max_iter, residual } => {
            assert_eq!(max_iter, 1);
            assert!(residual.is_nan(), "MFEM final_norm = sqrt(-1.92) = NaN");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

/// probe case6: `(Ad, d) == 0` inside the loop — warn, break with
/// `final_iter = i` AFTER `++i` (one completed pass reports 2!), full trailer
/// with `No convergence!` (solvers.cpp:1009-1024, 1032-1045).
#[test]
fn child_case6_op_den0_inloop() {
    let a = diag_matrix(&[1.0, 0.0]);
    let prec = identity_prec(2);
    let b = vec![1.0, 1.0];
    let mut x = vec![0.0; 2];
    let err = solve(&a, &b, &mut x, &prec).expect_err("not converged");
    match err {
        SolverError::ConvergenceFailed { max_iter, residual } => {
            assert_eq!(max_iter, 2, "MFEM's final_iter is the never-run pass");
            assert_eq!(residual, 2.0_f64.sqrt());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

/// probe case8: the regression grail — a NORMAL SPD run through GSSmoother
/// whose whole level-1 log (16 iterations + ARF) must stay byte-identical to
/// MFEM's (`tmp/d169/probe_case8_n32_gs.txt`); guards the ex1-style format the
/// same way the manually verified 111-iteration mfem_ex1_poisson log does.
#[test]
fn child_case8_regression_n32_gs() {
    let a = tridiag_poisson(32);
    let b = vec![1.0; 32];
    let mut x = vec![0.0; 32];
    let res = solve_gs(&a, &b, &mut x).expect("converges");
    assert!(res.converged && res.iterations == 16, "got {res:?}");
}

// ── parent tests: byte-compare each child log with the MFEM ground truth ────

/// Re-run this test binary for one child and return its MFEM log lines.
fn child_log(child: &str) -> Vec<String> {
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args(["--exact", child, "--nocapture", "--test-threads=1"])
        .output()
        .expect("spawn child test binary");
    assert!(
        out.status.success(),
        "child {child} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // With --nocapture the harness glues its `test <name> ... ` progress
    // prefix onto the first output line — strip it before filtering.
    let glue = format!("test {child} ... ");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let l = l.strip_prefix(&glue).unwrap_or(l);
            let keeps = l.starts_with("   Iteration : ")
                || l.starts_with("PCG: ")
                || l.starts_with("Average reduction factor = ");
            keeps.then(|| l.to_string())
        })
        .collect()
}

fn assert_log(child: &str, expected: &[&str]) {
    let got = child_log(child);
    assert_eq!(
        got.len(),
        expected.len(),
        "line count mismatch for {child}:\n{got:#?}"
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(g, e, "line {i} of {child}");
    }
}

#[test]
fn case1a_bytes_singular_b0() {
    assert_log(
        "child_case1a_singular_b0",
        &["   Iteration :   0  (B r, r) = 0"],
    );
}

#[test]
fn case1b_bytes_singular_consistent() {
    assert_log(
        "child_case1b_singular_consistent",
        &["   Iteration :   0  (B r, r) = 0"],
    );
}

#[test]
fn case2_bytes_prec_indefinite_iter0() {
    assert_log(
        "child_case2_prec_indefinite_iter0",
        &[
            "   Iteration :   0  (B r, r) = -3",
            "PCG: The preconditioner is not positive definite. (Br, r) = -3",
        ],
    );
}

#[test]
fn case3_bytes_op_den0_iter0() {
    assert_log(
        "child_case3_op_den0_iter0",
        &[
            "   Iteration :   0  (B r, r) = 1",
            "PCG: The operator is not positive definite. (Ad, d) = 0",
        ],
    );
}

#[test]
fn case4_bytes_op_den_neg_continues() {
    assert_log(
        "child_case4_op_den_neg_continues",
        &[
            "   Iteration :   0  (B r, r) = 1",
            "PCG: The operator is not positive definite. (Ad, d) = -1",
            "   Iteration :   1  (B r, r) = 0",
            "Average reduction factor = 0",
        ],
    );
}

#[test]
fn case5_bytes_prec_indefinite_inloop() {
    let got = child_log("child_case5_prec_indefinite_inloop");
    let expected = [
        "   Iteration :   0  (B r, r) = 3",
        "PCG: The preconditioner is not positive definite. (Br, r) = -1.92",
        "PCG: Number of iterations: 1",
        "Average reduction factor = -nan", // glibc bytes; sign of NaN is platform-dependent
        "PCG: No convergence!",
    ];
    assert_eq!(got.len(), expected.len(), "{got:#?}");
    for i in [0, 1, 2, 4] {
        assert_eq!(got[i], expected[i], "line {i}");
    }
    // The only platform-tolerant line: glibc prints "-nan", MSVC/Rust "nan".
    assert!(got[3].starts_with("Average reduction factor = "));
    assert!(got[3].ends_with("nan"), "{}", got[3]);
}

#[test]
fn case6_bytes_op_den0_inloop() {
    assert_log(
        "child_case6_op_den0_inloop",
        &[
            "   Iteration :   0  (B r, r) = 2",
            "   Iteration :   1  (B r, r) = 2",
            "PCG: The operator is not positive definite. (Ad, d) = 0",
            "PCG: Number of iterations: 2",
            "Average reduction factor = 1",
            "PCG: No convergence!",
        ],
    );
}

#[test]
fn case8_bytes_regression_n32_gs() {
    assert_log(
        "child_case8_regression_n32_gs",
        &[
            "   Iteration :   0  (B r, r) = 60.6667",
            "   Iteration :   1  (B r, r) = 338.555",
            "   Iteration :   2  (B r, r) = 187.935",
            "   Iteration :   3  (B r, r) = 97.1551",
            "   Iteration :   4  (B r, r) = 40.1509",
            "   Iteration :   5  (B r, r) = 9.9775",
            "   Iteration :   6  (B r, r) = 1.1026",
            "   Iteration :   7  (B r, r) = 0.0483259",
            "   Iteration :   8  (B r, r) = 0.000903636",
            "   Iteration :   9  (B r, r) = 9.83223e-06",
            "   Iteration :  10  (B r, r) = 1.18835e-06",
            "   Iteration :  11  (B r, r) = 5.24211e-07",
            "   Iteration :  12  (B r, r) = 2.63037e-07",
            "   Iteration :  13  (B r, r) = 1.23874e-07",
            "   Iteration :  14  (B r, r) = 3.02521e-08",
            "   Iteration :  15  (B r, r) = 7.39104e-10",
            "   Iteration :  16  (B r, r) = 2.39437e-11",
            "Average reduction factor = 0.409621",
        ],
    );
}

/// `fmt_g` must render like C `printf("%g")` for the trailer edge values
/// (signed zero, lowercase non-finites).
#[test]
fn fmt_g_matches_c_printf_edge_values() {
    assert_eq!(fmt_g(0.0), "0");
    assert_eq!(fmt_g(-0.0), "-0");
    assert_eq!(fmt_g(f64::NAN), "nan");
    assert_eq!(fmt_g(-f64::NAN), "-nan");
    assert_eq!(fmt_g(f64::INFINITY), "inf");
    assert_eq!(fmt_g(f64::NEG_INFINITY), "-inf");
    assert_eq!(fmt_g(-1.92), "-1.92");
    assert_eq!(fmt_g(2.39437e-11), "2.39437e-11");
}
