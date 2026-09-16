//! D216 — BPCG print levels byte-aligned with the C++ ground truth.
//!
//! Before D216, `solve_bpcg` derived its gates from the derived `Ord` scale
//! (`level >= PrintLevel::Iterations` / `>= PrintLevel::Summary`), so the
//! round-38 `WarningsOnly` (MFEM legacy 0) and `FirstAndLast` (legacy 3)
//! variants — which sort *after* `Iterations` — wrongly received the
//! per-iteration history and the summary trailer.  The gates are now an
//! explicit match reproducing MFEM `FromLegacyPrintLevel`
//! (`linalg/solvers.cpp:119`) as consumed by `BPCGSolver::Mult`
//! (`miniapps/solvers/bramble_pasciak.cpp:256-389`).
//!
//! C++ truth: `tmp/d216/probe_bpcg_levels.cpp` embeds `BPCGSolver::Mult`
//! verbatim from MFEM 4.10 (the miniapps headers need an MPI build, so the
//! class body is compiled against the serial `libmfem.a`), runs the 9×9
//! saddle problem below at every legacy level, and its stdout is captured in
//! `tmp/d216/d216_bpcg_cpp_stdout.txt`.  The fem-rs port is a bit-for-bit
//! translation of the same arithmetic in the same order, so every line is
//! asserted byte-for-byte, numeric tails included.
//!
//! Mechanics (round-37 D169 / round-38 D200 pattern): each `child_*` test runs
//! one solve through the real `println!`-printing code path; a parent test
//! re-executes this test binary with `--exact <child> --nocapture`, filters
//! the stdout down to the MFEM log line families and asserts the exact bytes.

use std::process::Command;

use fem_linalg::{PrintLevel, SolveResult, SolverConfig, SolverError};
use fem_solver::bpcg::solve_bpcg;

// ── fixtures (identical data to tmp/d216/probe_bpcg_levels.cpp) ─────────────

const NU: usize = 6;
const NP: usize = 3;
const N: usize = NU + NP;

fn cfg(print_level: PrintLevel, max_iter: usize) -> SolverConfig {
    SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter,
        verbose: false,
        print_level,
    }
}

/// SPD tridiagonal `M` and the mixed matrix `B`.
fn saddle_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut m = vec![vec![0.0; NU]; NU];
    for i in 0..NU {
        m[i][i] = 2.0;
        if i > 0 {
            m[i][i - 1] = -0.5;
            m[i - 1][i] = -0.5;
        }
    }
    let mut b = vec![vec![0.0; NU]; NP];
    for r in 0..NP {
        for c in 0..NU {
            b[r][c] = ((r + 1) * (c + 1) % 5) as f64 * 0.1 + 0.01 * (r as f64 + c as f64);
        }
    }
    (m, b)
}

/// Flat saddle operator `[[M, Bᵀ], [B, 0]]`.
fn apply_a(m: &[Vec<f64>], b: &[Vec<f64>], x: &[f64], y: &mut [f64]) {
    for i in 0..NU {
        let mut s = 0.0;
        for (j, &mj) in m[i].iter().enumerate() {
            s += mj * x[j];
        }
        for (k, row) in b.iter().enumerate() {
            s += row[i] * x[NU + k];
        }
        y[i] = s;
    }
    for k in 0..NP {
        let mut s = 0.0;
        for (c, &bc) in b[k].iter().enumerate() {
            s += bc * x[c];
        }
        y[NU + k] = s;
    }
}

fn dense_solve(a: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut b = rhs.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        b.swap(col, piv);
        let d = m[col][col];
        for c in col..n {
            m[col][c] /= d;
        }
        b[col] /= d;
        for r in 0..n {
            if r != col {
                let f = m[r][col];
                for c in col..n {
                    m[r][c] -= f * m[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    b
}

/// One BPCG run; `max_iter = 0` keeps the C++ probe's 200.
fn run_bpcg(print_level: PrintLevel, max_iter: usize) -> Result<SolveResult, SolverError> {
    let (m, b) = saddle_data();

    // Exact solution and rhs (x_star[i] = (i+1)/N · 0.5 − 0.2).
    let x_star: Vec<f64> = (0..N).map(|i| (i as f64 + 1.0) / N as f64 * 0.5 - 0.2).collect();
    let mut rhs = vec![0.0; N];
    apply_a(&m, &b, &x_star, &mut rhs);

    // Q = 0.5·diag(M); N = diag(invQ, 0).
    let inv_q: Vec<f64> = (0..NU).map(|i| 1.0 / (0.5 * m[i][i])).collect();
    let apply_n = |x: &[f64], y: &mut [f64]| {
        for i in 0..NU {
            y[i] = inv_q[i] * x[i];
        }
        for k in 0..NP {
            y[NU + k] = 0.0;
        }
    };

    // P = cpc·tri with cpc = diag(invQ, M1), tri = [[I, 0], [B·invQ, −I]].
    let binvq: Vec<Vec<f64>> = b
        .iter()
        .map(|row| (0..NU).map(|c| row[c] * inv_q[c]).collect())
        .collect();
    let mut s = vec![vec![0.0; NP]; NP];
    for r in 0..NP {
        for c in 0..NP {
            let mut acc = 0.0;
            for j in 0..NU {
                acc += binvq[r][j] * b[c][j];
            }
            s[r][c] = acc;
        }
    }
    let m1_cols: Vec<Vec<f64>> = (0..NP)
        .map(|c| {
            let mut e = vec![0.0; NP];
            e[c] = 1.0;
            dense_solve(&s, &e)
        })
        .collect();
    let m1_mat: Vec<Vec<f64>> = (0..NP)
        .map(|r| (0..NP).map(|c| m1_cols[c][r]).collect())
        .collect();

    let apply_p = |x: &[f64], y: &mut [f64]| {
        let mut tx = vec![0.0; N];
        tx[..NU].copy_from_slice(&x[..NU]);
        for k in 0..NP {
            let mut acc = 0.0;
            for j in 0..NU {
                acc += binvq[k][j] * x[j];
            }
            tx[NU + k] = acc - x[NU + k];
        }
        for i in 0..NU {
            y[i] = inv_q[i] * tx[i];
        }
        for r in 0..NP {
            let mut acc = 0.0;
            for (c, &m1) in m1_mat[r].iter().enumerate() {
                acc += m1 * tx[NU + c];
            }
            y[NU + r] = acc;
        }
    };

    let mut x = vec![0.0; N];
    solve_bpcg(
        N,
        |v, w| apply_a(&m, &b, v, w),
        &apply_p,
        &apply_n,
        &rhs,
        &mut x,
        &cfg(print_level, if max_iter == 0 { 200 } else { max_iter }),
    )
}

// ── child tests: one solve per print level, captured by the parents ─────────

#[test]
fn child_lm1_converged() {
    let r = run_bpcg(PrintLevel::Silent, 0);
    assert!(matches!(r, Ok(ref s) if s.converged && s.iterations == 9));
}

#[test]
fn child_l0_converged() {
    let r = run_bpcg(PrintLevel::WarningsOnly, 0);
    assert!(matches!(r, Ok(ref s) if s.converged && s.iterations == 9));
}

#[test]
fn child_l1_converged() {
    let r = run_bpcg(PrintLevel::Iterations, 0);
    assert!(matches!(r, Ok(ref s) if s.converged && s.iterations == 9));
}

#[test]
fn child_l2_converged() {
    let r = run_bpcg(PrintLevel::Summary, 0);
    assert!(matches!(r, Ok(ref s) if s.converged && s.iterations == 9));
}

#[test]
fn child_l3_converged() {
    let r = run_bpcg(PrintLevel::FirstAndLast, 0);
    assert!(matches!(r, Ok(ref s) if s.converged && s.iterations == 9));
}

#[test]
fn child_l1_maxiter2() {
    let r = run_bpcg(PrintLevel::Iterations, 2);
    assert!(matches!(r, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

#[test]
fn child_l0_maxiter2() {
    let r = run_bpcg(PrintLevel::WarningsOnly, 2);
    assert!(matches!(r, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

#[test]
fn child_l3_maxiter2() {
    let r = run_bpcg(PrintLevel::FirstAndLast, 2);
    assert!(matches!(r, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

// ── parent tests: capture child logs and compare with the C++ truth ─────────

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
                || l.starts_with("BPCG: ")
                || l.starts_with("Average reduction factor = ");
            keeps.then(|| l.to_string())
        })
        .collect()
}

fn assert_log(child: &str, expected: &[&str]) {
    let got = child_log(child);
    let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        got, expected,
        "child {child}: BPCG log mismatch against the C++ ground truth"
    );
}

// Each block below quotes tmp/d216/d216_bpcg_cpp_stdout.txt verbatim.

#[test]
fn lm1_bytes_silent() {
    // MFEM level -1: nothing at all.
    assert_log("child_lm1_converged", &[]);
}

#[test]
fn l0_bytes_converged() {
    // MFEM level 0: warnings only, and a converged run warns about nothing.
    assert_log("child_l0_converged", &[]);
}

#[test]
fn l1_bytes_converged() {
    // Level 1: full history + ARF (the count is `summary || warnings&&!conv`).
    assert_log(
        "child_l1_converged",
        &[
            "   Iteration :   0  (P r, r) = 0.393803",
            "   Iteration :   1  (Pr, r) = 0.0735976",
            "   Iteration :   2  (Pr, r) = 0.0123115",
            "   Iteration :   3  (Pr, r) = 0.00396081",
            "   Iteration :   4  (Pr, r) = 0.000590354",
            "   Iteration :   5  (Pr, r) = 2.91342e-05",
            "   Iteration :   6  (Pr, r) = 1.79293e-05",
            "   Iteration :   7  (Pr, r) = 5.81997e-07",
            "   Iteration :   8  (Pr, r) = 3.01373e-07",
            "   Iteration :   9  (Pr, r) = 4.57556e-31",
            "Average reduction factor = 0.973094",
        ],
    );
}

#[test]
fn l2_bytes_converged() {
    // Level 2: count + ARF, no history.
    assert_log(
        "child_l2_converged",
        &[
            "BPCG: Number of iterations: 9",
            "Average reduction factor = 0.973094",
        ],
    );
}

#[test]
fn l3_bytes_converged() {
    // Level 3: iter-0 with " ..." suffix, suppressed history, closing final
    // line, ARF (no count — `warnings && converged`).
    assert_log(
        "child_l3_converged",
        &[
            "   Iteration :   0  (P r, r) = 0.393803 ...",
            "   Iteration :   9  (Pr, r) = 4.57556e-31",
            "Average reduction factor = 0.973094",
        ],
    );
}

#[test]
fn l1_bytes_maxiter2() {
    // Level 1, diverged: history + count + ARF + No convergence!
    assert_log(
        "child_l1_maxiter2",
        &[
            "   Iteration :   0  (P r, r) = 0.393803",
            "   Iteration :   1  (Pr, r) = 0.0735976",
            "   Iteration :   2  (Pr, r) = 0.0123115",
            "BPCG: Number of iterations: 2",
            "Average reduction factor = 1.08635",
            "BPCG: No convergence!",
        ],
    );
}

#[test]
fn l0_bytes_maxiter2() {
    // Level 0, diverged: count + No convergence! only (no ARF, no history).
    assert_log(
        "child_l0_maxiter2",
        &[
            "BPCG: Number of iterations: 2",
            "BPCG: No convergence!",
        ],
    );
}

#[test]
fn l3_bytes_maxiter2() {
    // Level 3, diverged: first/last lines + count + ARF + No convergence!
    assert_log(
        "child_l3_maxiter2",
        &[
            "   Iteration :   0  (P r, r) = 0.393803 ...",
            "   Iteration :   2  (Pr, r) = 0.0123115",
            "BPCG: Number of iterations: 2",
            "Average reduction factor = 1.08635",
            "BPCG: No convergence!",
        ],
    );
}
