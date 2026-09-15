//! D199 + D200 — MFEM 4.10 print-level 0/3 gates (CG) and the MINRES / GMRES /
//! BiCGSTAB MFEM-style trailers, byte-aligned with the C++ ground truth.
//!
//! C++ truth: `tmp/d200/probe_d199_levels.cpp` and `tmp/d200/probe_d200_solvers.cpp`
//! (WSL, `$HOME/mfem410_ser`, `MFEM_VERSION = 41000`), captured in
//! `tmp/d200/d199_levels_cpp_stdout.txt` / `tmp/d200/d200_solvers_cpp_stdout.txt`.
//!
//! Mechanics (round-37 D169 pattern): each `child_*` test runs one solve through
//! the real `println!`-printing code path; a parent test re-executes this test
//! binary with `--exact <child> --nocapture`, filters the stdout down to the
//! MFEM log line families and asserts the exact bytes.
//!
//! Byte-exactness: CG (`solve_pcg_operator_precond`) is a bit-for-bit MFEM port,
//! so the level-0/level-3 CG lines are asserted byte-for-byte, values included.
//! MINRES/GMRES/BiCGSTAB in fem-rs are algebraically equivalent but not bitwise
//! identical to `MINRESSolver`/`GMRESSolver`/`BiCGSTABSolver` (different Givens
//! formulations: `hypot` vs `sqrt(dx²+dy²)` etc. — debt D217), so for those the
//! FIXED TEXT of every line is asserted byte-for-byte while numeric tails are
//! parsed as finite floats (the round-37 `-nan` precedent, generalized).  Values
//! that are algorithm-independent (e.g. the initial residual norms) are still
//! asserted byte-for-byte.

use std::process::Command;

use fem_linalg::SolveResult;
use fem_solver::{
    solve_bicgstab_operator, solve_gmres_operator, solve_minres_operator,
    solve_pcg_operator_precond, PrintLevel, SolverConfig, SolverError,
};

// ── fixtures ────────────────────────────────────────────────────────────────

/// cfg mirroring the C++ probe's `SetRelTol(sqrt(1e-12)), SetAbsTol(0)`.
fn cfg(print_level: PrintLevel, max_iter: usize) -> SolverConfig {
    SolverConfig {
        rtol: 1e-6, // sqrt(1e-12)
        atol: 0.0,
        max_iter,
        verbose: false,
        print_level,
    }
}

/// y = A x for the 8x8 tridiagonal Poisson matrix.
fn poisson8_apply(x: &[f64], y: &mut [f64]) {
    let n = 8;
    for i in 0..n {
        y[i] = 2.0 * x[i];
        if i > 0 {
            y[i] -= x[i - 1];
        }
        if i + 1 < n {
            y[i] -= x[i + 1];
        }
    }
}

/// y = A x for the 1-D periodic Laplacian (n = 4, singular).
fn periodic4_apply(x: &[f64], y: &mut [f64]) {
    let n = 4;
    for i in 0..n {
        y[i] = 2.0 * x[i] - x[(i + 1) % n] - x[(i + n - 1) % n];
    }
}

/// y = A x for the 4x4 non-symmetric probe matrix.
fn nonsym4_apply(x: &[f64], y: &mut [f64]) {
    y[0] = 2.0 * x[0] + x[1];
    y[1] = 3.0 * x[1] + x[2];
    y[2] = 4.0 * x[2] + x[3];
    y[3] = -x[0] + 5.0 * x[3];
}

// ── D199 child tests: CG at print levels 3 and 0 (byte-exact) ───────────────

/// C++ L3a: level 3, converged — iter-0 `" ..."` line, suppressed history,
/// closing final line, ARF (no count: `warnings && converged`).
#[test]
fn child_l3a_level3_converged() {
    let b = vec![1.0; 8];
    let mut x = vec![0.0; 8];
    let res = solve_pcg_operator_precond(
        8,
        poisson8_apply,
        &b,
        &mut x,
        |r: &[f64], z: &mut [f64]| z.copy_from_slice(r),
        &cfg(PrintLevel::FirstAndLast, 200),
    )
    .expect("converges");
    assert!(res.converged && res.iterations == 4, "got {res:?}");
}

/// C++ L3b: level 3, max_iter=2 — full trailer with `No convergence!`.
#[test]
fn child_l3b_level3_maxiter2() {
    let b = vec![1.0; 8];
    let mut x = vec![0.0; 8];
    let res = solve_pcg_operator_precond(
        8,
        poisson8_apply,
        &b,
        &mut x,
        |r: &[f64], z: &mut [f64]| z.copy_from_slice(r),
        &cfg(PrintLevel::FirstAndLast, 2),
    );
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

/// C++ L3c: level 3, converged at iteration 0 — ONLY the iter-0 `" ..."` line
/// (the early return leaves before the closing line and the ARF).
#[test]
fn child_l3c_level3_converged_iter0() {
    let b = vec![0.0; 4];
    let mut x = vec![0.0; 4];
    let res = solve_pcg_operator_precond(
        4,
        periodic4_apply,
        &b,
        &mut x,
        |r: &[f64], z: &mut [f64]| z.copy_from_slice(r),
        &cfg(PrintLevel::FirstAndLast, 200),
    )
    .expect("converges at iteration 0");
    assert!(res.converged && res.iterations == 0);
}

/// C++ L3d: level 3, indefinite preconditioner at iteration 0 — iter-0
/// `" ..."` line + warning line, no trailer.
#[test]
fn child_l3d_level3_prec_indefinite() {
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0; 2];
    let d = [1.0, -1.0];
    let res = solve_pcg_operator_precond(
        2,
        |x: &[f64], y: &mut [f64]| y.copy_from_slice(x),
        &b,
        &mut x,
        move |r: &[f64], z: &mut [f64]| {
            for i in 0..2 {
                z[i] = r[i] * d[i];
            }
        },
        &cfg(PrintLevel::FirstAndLast, 200),
    )
    .expect("returns a result");
    assert!(!res.converged && res.iterations == 0 && res.final_residual == -3.0);
}

/// C++ L0a: level 0, converged — completely silent.
#[test]
fn child_l0a_level0_converged() {
    let b = vec![1.0; 8];
    let mut x = vec![0.0; 8];
    let res = solve_pcg_operator_precond(
        8,
        poisson8_apply,
        &b,
        &mut x,
        |r: &[f64], z: &mut [f64]| z.copy_from_slice(r),
        &cfg(PrintLevel::WarningsOnly, 200),
    )
    .expect("converges");
    assert!(res.converged && res.iterations == 4, "got {res:?}");
}

/// C++ L0b: level 0, max_iter=2 — count + `PCG: No convergence!` only.
#[test]
fn child_l0b_level0_maxiter2() {
    let b = vec![1.0; 8];
    let mut x = vec![0.0; 8];
    let res = solve_pcg_operator_precond(
        8,
        poisson8_apply,
        &b,
        &mut x,
        |r: &[f64], z: &mut [f64]| z.copy_from_slice(r),
        &cfg(PrintLevel::WarningsOnly, 2),
    );
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

/// C++ L0c: level 0, indefinite preconditioner at iteration 0 — warning line
/// only (no iteration lines at level 0).
#[test]
fn child_l0c_level0_prec_indefinite() {
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0; 2];
    let d = [1.0, -1.0];
    let res = solve_pcg_operator_precond(
        2,
        |x: &[f64], y: &mut [f64]| y.copy_from_slice(x),
        &b,
        &mut x,
        move |r: &[f64], z: &mut [f64]| {
            for i in 0..2 {
                z[i] = r[i] * d[i];
            }
        },
        &cfg(PrintLevel::WarningsOnly, 200),
    )
    .expect("returns a result");
    assert!(!res.converged && res.iterations == 0 && res.final_residual == -3.0);
}

// ── D200 child tests: MINRES / GMRES / BiCGSTAB trailers ────────────────────

fn solve_minres(level: PrintLevel, max_iter: usize, b: &[f64]) -> Result<SolveResult, SolverError> {
    assert_eq!(b.len(), 8, "the Poisson fixture is 8x8");
    let mut x = vec![0.0; 8];
    solve_minres_operator(8, 8, poisson8_apply, b, &mut x, &cfg(level, max_iter))
}

/// y = D x for diag(1, -1) (the C++ Mf operator).
fn diag_indef_apply(x: &[f64], y: &mut [f64]) {
    y[0] = x[0];
    y[1] = -x[1];
}

/// C++ Mf: MINRES indefinite diag(1,-1), b = (1,1), level 1.
#[test]
fn child_mf_minres_indefinite() {
    let b = vec![1.0, 1.0];
    let mut x = vec![0.0; 2];
    let res =
        solve_minres_operator(2, 2, diag_indef_apply, &b, &mut x, &cfg(PrintLevel::Iterations, 200))
            .expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// The 8x8 tridiagonal Poisson CSR matrix (for `solve_minres_precond`).
fn poisson8_csr() -> fem_linalg::CsrMatrix<f64> {
    let n = 8;
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
    fem_linalg::CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values }
}

/// C++ Mg: preconditioned MINRES (Jacobi D⁻¹) level 1 — the
/// `solve_minres_precond` trailer path.
#[test]
fn child_mg_minres_precond_level1() {
    let a = poisson8_csr();
    let b = vec![1.0; 8];
    let mut x = vec![0.0; 8];
    let prec = |r: &[f64], z: &mut [f64]| {
        for i in 0..r.len() {
            z[i] = r[i] / 2.0; // D⁻¹, D = 2·I
        }
    };
    let res = fem_solver::solve_minres_precond(&a, &prec, &b, &mut x, &cfg(PrintLevel::Iterations, 200))
        .expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Ma: MINRES level 1, converged — iter-0, in-loop, loop_end lines.
#[test]
fn child_ma_minres_level1() {
    let b = vec![1.0; 8];
    let res = solve_minres(PrintLevel::Iterations, 200, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Mb: MINRES level 3 — iter-0 `" ..."` + loop_end line, NO count.
#[test]
fn child_mb_minres_level3() {
    let b = vec![1.0; 8];
    let res = solve_minres(PrintLevel::FirstAndLast, 200, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Mc: MINRES level 2 — the count line only.
#[test]
fn child_mc_minres_level2() {
    let b = vec![1.0; 8];
    let res = solve_minres(PrintLevel::Summary, 200, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Md: MINRES level 0, max_iter=3 — count + `MINRES: No convergence!`.
#[test]
fn child_md_minres_level0_maxiter3() {
    let b = vec![1.0; 8];
    let res = solve_minres(PrintLevel::WarningsOnly, 3, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 3, .. })));
}

/// C++ Me: MINRES b = 0, level 1 — the single loop_end line at iteration 0.
#[test]
fn child_me_minres_b0() {
    let b = vec![0.0; 8];
    let res = solve_minres(PrintLevel::Iterations, 200, &b).expect("converges at 0");
    assert!(res.converged && res.iterations == 0);
}

/// C++ Mg (Jacobi-preconditioned MINRES) has no fem-rs operator-callback twin
/// with a matching algorithm — covered by `solve_minres_precond` unit-level
/// gates instead; skipped here.

fn solve_gmres(
    level: PrintLevel,
    restart: usize,
    max_iter: usize,
    b: &[f64],
) -> Result<SolveResult, SolverError> {
    let n = b.len();
    let mut x = vec![0.0; n];
    solve_gmres_operator(n, n, nonsym4_apply, b, &mut x, restart, &cfg(level, max_iter))
}

/// C++ Ga: GMRES level 1, converged inside the first pass.
#[test]
fn child_ga_gmres_level1() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::Iterations, 50, 50, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Gb: GMRES level 3 — pass-1 `" ..."` + finish line.
#[test]
fn child_gb_gmres_level3() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::FirstAndLast, 50, 50, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Gc: GMRES level 2 — count only.
#[test]
fn child_gc_gmres_level2() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::Summary, 50, 50, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Gd: GMRES level 1, max_iter=2 — in-loop lines, NO finish line,
/// count + `GMRES: No convergence!`.
#[test]
fn child_gd_gmres_maxiter2_level1() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::Iterations, 50, 2, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

/// C++ Ge: GMRES level 3, max_iter=2 — pass-1 `" ..."`, finish line, count,
/// warning.
#[test]
fn child_ge_gmres_maxiter2_level3() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::FirstAndLast, 50, 2, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

/// C++ Gf: GMRES restart=1, max_iter=6, level 1 — `Restarting...` between
/// passes, pass index grows, no finish line, count + warning.
#[test]
fn child_gf_gmres_restart() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::Iterations, 1, 6, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 6, .. })));
}

/// C++ Gg: GMRES b = 0, level 1 — the initial-convergence finish line only.
#[test]
fn child_gg_gmres_b0() {
    let b = vec![0.0; 4];
    let res = solve_gmres(PrintLevel::Iterations, 50, 50, &b).expect("converges at 0");
    assert!(res.converged && res.iterations == 0);
}

/// C++ Gh: GMRES level 0, max_iter=2 — count + warning only.
#[test]
fn child_gh_gmres_level0() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_gmres(PrintLevel::WarningsOnly, 50, 2, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

fn solve_bicgstab(level: PrintLevel, max_iter: usize, b: &[f64]) -> Result<SolveResult, SolverError> {
    let n = b.len();
    let mut x = vec![0.0; n];
    solve_bicgstab_operator(n, n, nonsym4_apply, b, &mut x, &cfg(level, max_iter))
}

/// C++ Ba: BiCGSTAB level 1, converged — two-piece iteration lines.
#[test]
fn child_ba_bicgstab_level1() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_bicgstab(PrintLevel::Iterations, 50, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Bb: BiCGSTAB level 3 — iter-0 `" ..."` + the converging `||s||` line.
#[test]
fn child_bb_bicgstab_level3() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_bicgstab(PrintLevel::FirstAndLast, 50, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Bc: BiCGSTAB level 2 — count only.
#[test]
fn child_bc_bicgstab_level2() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_bicgstab(PrintLevel::Summary, 50, &b).expect("converges");
    assert!(res.converged, "got {res:?}");
}

/// C++ Bd: BiCGSTAB level 1, max_iter=2 — two-piece lines + count +
/// `BiCGStab: No convergence!`.
#[test]
fn child_bd_bicgstab_maxiter2_level1() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_bicgstab(PrintLevel::Iterations, 2, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
}

/// C++ Be: BiCGSTAB b = 0, level 1 — the iter-0 line only (no trailer).
#[test]
fn child_be_bicgstab_b0() {
    let b = vec![0.0; 4];
    let res = solve_bicgstab(PrintLevel::Iterations, 50, &b).expect("converges at 0");
    assert!(res.converged && res.iterations == 0);
}

/// C++ Bf: BiCGSTAB level 0, max_iter=2 — count + warning only.
#[test]
fn child_bf_bicgstab_level0() {
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let res = solve_bicgstab(PrintLevel::WarningsOnly, 2, &b);
    assert!(matches!(res, Err(SolverError::ConvergenceFailed { max_iter: 2, .. })));
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
                || l.starts_with("   Pass : ")
                || l.starts_with("MINRES: ")
                || l.starts_with("GMRES: ")
                || l.starts_with("BiCGStab: ")
                || l.starts_with("PCG: ")
                || l == "Restarting..."
                || l.starts_with("Average reduction factor = ");
            keeps.then(|| l.to_string())
        })
        .collect()
}

/// The numeric tail of `line` after the fixed `prefix` parses as a finite f64.
fn assert_g_tail(line: &str, prefix: &str) -> f64 {
    assert!(line.starts_with(prefix), "line {line:?} lacks prefix {prefix:?}");
    let tail = line[prefix.len()..].trim();
    let v: f64 = tail.parse().unwrap_or_else(|e| panic!("tail {tail:?} of {line:?}: {e}"));
    assert!(v.is_finite(), "non-finite tail in {line:?}");
    v
}

/// A two-piece BiCGSTAB iteration line `prefix<s>middle<r>`: both numeric
/// pieces parse as finite f64 (MFEM composes it from two `<<` pieces,
/// solvers.cpp:1719-1723 + :1741-1744).
fn assert_bicgstab_line(line: &str, prefix: &str, middle: &str) -> (f64, f64) {
    assert!(line.starts_with(prefix), "line {line:?} lacks prefix {prefix:?}");
    let rest = &line[prefix.len()..];
    let (s_part, r_part) = rest
        .split_once(middle)
        .unwrap_or_else(|| panic!("line {line:?} lacks middle {middle:?}"));
    let s: f64 = s_part.trim().parse().unwrap_or_else(|e| panic!("s tail of {line:?}: {e}"));
    let r: f64 = r_part.trim().parse().unwrap_or_else(|e| panic!("r tail of {line:?}: {e}"));
    assert!(s.is_finite() && r.is_finite(), "non-finite tails in {line:?}");
    (s, r)
}

// — D199 byte-exact CG comparisons (bit-for-bit port ⇒ values included) ——

#[test]
fn l3a_bytes_level3_converged() {
    assert_log(
        "child_l3a_level3_converged",
        &[
            "   Iteration :   0  (B r, r) = 8 ...",
            "   Iteration :   4  (B r, r) = 0",
            "Average reduction factor = 0",
        ],
    );
}

#[test]
fn l3b_bytes_level3_maxiter2() {
    assert_log(
        "child_l3b_level3_maxiter2",
        &[
            "   Iteration :   0  (B r, r) = 8 ...",
            "   Iteration :   2  (B r, r) = 12",
            "PCG: Number of iterations: 2",
            "Average reduction factor = 1.10668",
            "PCG: No convergence!",
        ],
    );
}

#[test]
fn l3c_bytes_level3_converged_iter0() {
    assert_log(
        "child_l3c_level3_converged_iter0",
        &["   Iteration :   0  (B r, r) = 0 ..."],
    );
}

#[test]
fn l3d_bytes_level3_prec_indefinite() {
    assert_log(
        "child_l3d_level3_prec_indefinite",
        &[
            "   Iteration :   0  (B r, r) = -3 ...",
            "PCG: The preconditioner is not positive definite. (Br, r) = -3",
        ],
    );
}

#[test]
fn l0a_bytes_level0_converged() {
    // C++ L0a prints NOTHING at level 0 on convergence.
    assert_log("child_l0a_level0_converged", &[]);
}

#[test]
fn l0b_bytes_level0_maxiter2() {
    assert_log(
        "child_l0b_level0_maxiter2",
        &["PCG: Number of iterations: 2", "PCG: No convergence!"],
    );
}

#[test]
fn l0c_bytes_level0_prec_indefinite() {
    assert_log(
        "child_l0c_level0_prec_indefinite",
        &["PCG: The preconditioner is not positive definite. (Br, r) = -3"],
    );
}

// — D200 MINRES/GMRES/BiCGSTAB: fixed text byte-exact, numeric tails parsed —
// (debt D217: the fem-rs algorithms are algebraically but not bitwise equal to
// MFEM's, unlike CG; the initial residuals ARE algorithm-independent bytes.)

#[test]
fn ma_minres_level1_structure() {
    let got = child_log("child_ma_minres_level1");
    // Byte-exact line: the iteration-0 value ‖r₀‖ = √8 (algorithm-free).
    assert_eq!(got[0], "MINRES: iteration   0: ||r||_B = 2.82843");
    let n = got.len();
    assert!(n >= 2, "{got:#?}");
    // In-loop lines carry consecutive iteration numbers 1..n-2; the loop_end
    // line repeats the converging iteration n-1 (MFEM truth: it = 4).
    for i in 1..n {
        let idx = if i == n - 1 { n - 1 } else { i };
        let v = assert_g_tail(&got[i], &format!("MINRES: iteration {:3}: ||r||_B = ", idx));
        assert!(v >= 0.0, "negative ||r||_B in {got:#?}");
    }
    // Level 1 converged: no count, no warning.
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
    assert!(!got.iter().any(|l| l.contains("No convergence!")), "{got:#?}");
}

#[test]
fn mb_minres_level3_structure() {
    let got = child_log("child_mb_minres_level3");
    assert_eq!(got.len(), 2, "{got:#?}");
    assert_eq!(got[0], "MINRES: iteration   0: ||r||_B = 2.82843 ...");
    assert!(got[1].starts_with("MINRES: iteration   4: ||r||_B = "), "{got:#?}");
    assert_g_tail(&got[1], "MINRES: iteration   4: ||r||_B = ");
    // Converged level 3: no count line.
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
}

#[test]
fn mc_minres_level2_count_only() {
    let got = child_log("child_mc_minres_level2");
    assert_eq!(got.len(), 1, "{got:#?}");
    assert!(got[0].starts_with("MINRES: Number of iterations:   "), "{got:#?}");
    assert_g_tail(&got[0], "MINRES: Number of iterations: ");
}

#[test]
fn md_minres_level0_bytes() {
    assert_log(
        "child_md_minres_level0_maxiter3",
        &["MINRES: Number of iterations:   3", "MINRES: No convergence!"],
    );
}

#[test]
fn me_minres_b0_bytes() {
    assert_log("child_me_minres_b0", &["MINRES: iteration   0: ||r||_B = 0"]);
}

#[test]
fn mf_minres_indefinite_structure() {
    let got = child_log("child_mf_minres_indefinite");
    // C++ Mf: three lines — iteration 0, in-loop 1, loop_end 2.
    assert_eq!(got.len(), 3, "{got:#?}");
    assert_eq!(got[0], "MINRES: iteration   0: ||r||_B = 1.41421");
    assert_g_tail(&got[1], "MINRES: iteration   1: ||r||_B = ");
    assert_g_tail(&got[2], "MINRES: iteration   2: ||r||_B = ");
}

#[test]
fn mg_minres_precond_level1_structure() {
    let got = child_log("child_mg_minres_precond_level1");
    // C++ Mg truth (tmp/d200/d200_solvers_cpp_stdout.txt): iteration-0 value
    // 2 = sqrt((D⁻¹r, r)) — algorithm-free — then in-loop 1..3, loop_end 4.
    assert_eq!(got.len(), 5, "{got:#?}");
    assert_eq!(got[0], "MINRES: iteration   0: ||r||_B = 2");
    for i in 1..5 {
        assert_g_tail(&got[i], &format!("MINRES: iteration {:3}: ||r||_B = ", i));
    }
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
}

#[test]
fn ga_gmres_level1_structure() {
    let got = child_log("child_ga_gmres_level1");
    // C++ Ga: pass-1 line, in-loop 1..3, finish line at 4.
    assert_eq!(got.len(), 5, "{got:#?}");
    assert_eq!(got[0], "   Pass :  1   Iteration :   0  ||B r|| = 5.47723");
    for i in 1..4 {
        assert_g_tail(&got[i], &format!("   Pass :  1   Iteration : {:3}  ||B r|| = ", i));
    }
    assert!(got[4].starts_with("   Pass :  1   Iteration :   4  ||B r|| = "), "{got:#?}");
    assert_g_tail(&got[4], "   Pass :  1   Iteration :   4  ||B r|| = ");
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
}

#[test]
fn gb_gmres_level3_structure() {
    let got = child_log("child_gb_gmres_level3");
    assert_eq!(got.len(), 2, "{got:#?}");
    assert_eq!(got[0], "   Pass :  1   Iteration :   0  ||B r|| = 5.47723 ...");
    assert!(got[1].starts_with("   Pass :  1   Iteration :   4  ||B r|| = "), "{got:#?}");
    assert_g_tail(&got[1], "   Pass :  1   Iteration :   4  ||B r|| = ");
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
}

#[test]
fn gc_gmres_level2_count_only() {
    let got = child_log("child_gc_gmres_level2");
    assert_eq!(got.len(), 1, "{got:#?}");
    assert_eq!(got[0], "GMRES: Number of iterations: 4");
}

#[test]
fn gd_gmres_maxiter2_level1_structure() {
    let got = child_log("child_gd_gmres_maxiter2_level1");
    // C++ Gd: pass-1, in-loop 1 and 2, NO finish line, count + warning.
    assert_eq!(got.len(), 5, "{got:#?}");
    assert_eq!(got[0], "   Pass :  1   Iteration :   0  ||B r|| = 5.47723");
    assert_g_tail(&got[1], "   Pass :  1   Iteration :   1  ||B r|| = ");
    assert_g_tail(&got[2], "   Pass :  1   Iteration :   2  ||B r|| = ");
    assert_eq!(got[3], "GMRES: Number of iterations: 2");
    assert_eq!(got[4], "GMRES: No convergence!");
}

#[test]
fn ge_gmres_maxiter2_level3_structure() {
    let got = child_log("child_ge_gmres_maxiter2_level3");
    // C++ Ge: pass-1 " ...", the first_and_last finish line (true residual),
    // count + warning.
    assert_eq!(got.len(), 4, "{got:#?}");
    assert_eq!(got[0], "   Pass :  1   Iteration :   0  ||B r|| = 5.47723 ...");
    assert!(got[1].starts_with("   Pass :  1   Iteration :   2  ||B r|| = "), "{got:#?}");
    assert_g_tail(&got[1], "   Pass :  1   Iteration :   2  ||B r|| = ");
    assert_eq!(got[2], "GMRES: Number of iterations: 2");
    assert_eq!(got[3], "GMRES: No convergence!");
}

#[test]
fn gf_gmres_restart_structure() {
    let got = child_log("child_gf_gmres_restart");
    // C++ Gf: pass-1, in-loop 1, then per pass p = 2..6: "Restarting..." +
    // in-loop line, and the trailer (no finish line at level 1).
    assert_eq!(got.len(), 14, "{got:#?}");
    assert_eq!(got[0], "   Pass :  1   Iteration :   0  ||B r|| = 5.47723");
    assert_g_tail(&got[1], "   Pass :  1   Iteration :   1  ||B r|| = ");
    for p in 2..=6usize {
        let restart = &got[2 * (p - 1)];
        assert_eq!(restart, "Restarting...", "pass {p}");
        let line = &got[2 * (p - 1) + 1];
        assert_g_tail(line, &format!("   Pass : {:2}   Iteration : {:3}  ||B r|| = ", p, p));
    }
    assert_eq!(got[12], "GMRES: Number of iterations: 6");
    assert_eq!(got[13], "GMRES: No convergence!");
}

#[test]
fn gg_gmres_b0_bytes() {
    assert_log("child_gg_gmres_b0", &["   Pass :  1   Iteration :   0  ||B r|| = 0"]);
}

#[test]
fn gh_gmres_level0_bytes() {
    assert_log(
        "child_gh_gmres_level0",
        &["GMRES: Number of iterations: 2", "GMRES: No convergence!"],
    );
}

#[test]
fn ba_bicgstab_level1_structure() {
    let got = child_log("child_ba_bicgstab_level1");
    // C++ Ba: iter-0, two-piece lines 1..3, converging `||s||` line at 4.
    assert_eq!(got.len(), 5, "{got:#?}");
    assert_eq!(got[0], "   Iteration :   0   ||r|| = 5.47723");
    for i in 1..4 {
        let (s, r) =
            assert_bicgstab_line(&got[i], &format!("   Iteration : {:3}   ||s|| = ", i), "   ||r|| = ");
        assert!(s > 0.0 && r >= 0.0, "{got:#?}");
    }
    assert!(got[4].starts_with("   Iteration :   4   ||s|| = "), "{got:#?}");
    assert_g_tail(&got[4], "   Iteration :   4   ||s|| = ");
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
}

#[test]
fn bb_bicgstab_level3_structure() {
    let got = child_log("child_bb_bicgstab_level3");
    assert_eq!(got.len(), 2, "{got:#?}");
    assert_eq!(got[0], "   Iteration :   0   ||r|| = 5.47723 ...");
    assert!(got[1].starts_with("   Iteration :   4   ||s|| = "), "{got:#?}");
    assert_g_tail(&got[1], "   Iteration :   4   ||s|| = ");
    assert!(!got.iter().any(|l| l.contains("Number of iterations")), "{got:#?}");
}

#[test]
fn bc_bicgstab_level2_count_only() {
    let got = child_log("child_bc_bicgstab_level2");
    assert_eq!(got.len(), 1, "{got:#?}");
    assert_eq!(got[0], "BiCGStab: Number of iterations: 4");
}

#[test]
fn bd_bicgstab_maxiter2_level1_structure() {
    let got = child_log("child_bd_bicgstab_maxiter2_level1");
    // C++ Bd: iter-0, two-piece lines 1 and 2, count + warning.
    assert_eq!(got.len(), 5, "{got:#?}");
    assert_eq!(got[0], "   Iteration :   0   ||r|| = 5.47723");
    assert_bicgstab_line(&got[1], "   Iteration :   1   ||s|| = ", "   ||r|| = ");
    assert_bicgstab_line(&got[2], "   Iteration :   2   ||s|| = ", "   ||r|| = ");
    assert_eq!(got[3], "BiCGStab: Number of iterations: 2");
    assert_eq!(got[4], "BiCGStab: No convergence!");
}

#[test]
fn be_bicgstab_b0_bytes() {
    assert_log("child_be_bicgstab_b0", &["   Iteration :   0   ||r|| = 0"]);
}

#[test]
fn bf_bicgstab_level0_bytes() {
    assert_log(
        "child_bf_bicgstab_level0",
        &["BiCGStab: Number of iterations: 2", "BiCGStab: No convergence!"],
    );
}

/// `assert_log` — the D169 helper: exact line count and exact bytes.
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
